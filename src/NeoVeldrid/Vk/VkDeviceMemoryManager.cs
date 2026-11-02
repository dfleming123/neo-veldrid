using Silk.NET.Vulkan;
using static NeoVeldrid.Vk.VulkanUtil;
using System.Collections.Generic;
using System.Diagnostics;
using System;
using System.Threading;
using VkBufferHandle = Silk.NET.Vulkan.Buffer;

namespace NeoVeldrid.Vk;

internal unsafe class VkDeviceMemoryManager : IDisposable
{
    // An allocation of a quarter of a chunk or more gets device memory of its own, freed with it. Below that it shares a
    // chunk. Chunks are small and given back when empty, so what's reserved stays close to what's used (Wattle 0060).
    private const ulong MinDedicatedAllocationSizeDynamic = 1024 * 1024 * 1;
    private const ulong MinDedicatedAllocationSizeNonDynamic = 1024 * 1024 * 2;
    // Past half the driver's limit on memory objects, the sizes before 0060: allocations under them share chunks of
    // those sizes, so a scene of many mid-sized resources can't run the device out of allocations.
    private const ulong CrowdedDedicatedAllocationSizeDynamic = 1024 * 1024 * 64;
    private const ulong CrowdedDedicatedAllocationSizeNonDynamic = 1024 * 1024 * 256;
    private readonly Device _device;
    private readonly PhysicalDevice _physicalDevice;
    private readonly ulong _bufferImageGranularity;
    private readonly Silk.NET.Vulkan.Vk _vk;
    private readonly object _lock = new object();
    private ulong _totalAllocatedBytes;
    private readonly Dictionary<uint, ChunkAllocatorSet> _allocatorsByMemoryTypeUnmapped = new Dictionary<uint, ChunkAllocatorSet>();
    private readonly Dictionary<uint, ChunkAllocatorSet> _allocatorsByMemoryType = new Dictionary<uint, ChunkAllocatorSet>();

    private readonly vkGetBufferMemoryRequirements2_t _getBufferMemoryRequirements2;
    private readonly vkGetImageMemoryRequirements2_t _getImageMemoryRequirements2;

    // Device memory objects alive, own allocations and chunks, including those waiting to be freed; and the count past
    // which allocations go back to the crowded sizes.
    private int _memoryObjects;
    private readonly int _crowdedMemoryObjects;

    // Submissions are numbered as they go to the queue, under the device's queue lock, and complete in order. Memory a
    // resource gives back is freed only once every submission made before the release has completed: a submission
    // needn't hold a reference to everything it uses (an UpdateBuffer destination, attachments, views' textures).
    private ulong _submittedSerial;
    private ulong _completedSerial;
    private readonly List<PendingRelease> _pendingReleases = new List<PendingRelease>();

    private readonly struct PendingRelease
    {
        public readonly DeviceMemory Memory;
        public readonly ulong Serial;

        public PendingRelease(DeviceMemory memory, ulong serial)
        {
            Memory = memory;
            Serial = serial;
        }
    }

    public VkDeviceMemoryManager(
        Silk.NET.Vulkan.Vk vk,
        Device device,
        PhysicalDevice physicalDevice,
        ulong bufferImageGranularity,
        uint maxMemoryAllocationCount,
        vkGetBufferMemoryRequirements2_t getBufferMemoryRequirements2,
        vkGetImageMemoryRequirements2_t getImageMemoryRequirements2)
    {
        _vk = vk;
        _device = device;
        _physicalDevice = physicalDevice;
        _bufferImageGranularity = bufferImageGranularity;
        _crowdedMemoryObjects = (int)Math.Clamp(maxMemoryAllocationCount / 2, 1u, (uint)int.MaxValue);
        _getBufferMemoryRequirements2 = getBufferMemoryRequirements2;
        _getImageMemoryRequirements2 = getImageMemoryRequirements2;
    }

    /// <summary>Numbers a submission. Called under the device's queue lock, just before it submits.</summary>
    internal ulong NextSubmissionSerial() => Interlocked.Increment(ref _submittedSerial);

    /// <summary>The number of the last submission made. Read under the device's queue lock, it's exact.</summary>
    internal ulong LastSubmissionSerial => Interlocked.Read(ref _submittedSerial);

    /// <summary>Every submission up to this number has completed: frees the memory that was waiting for them.</summary>
    internal void SubmissionsCompleted(ulong serial)
    {
        lock (_lock)
        {
            if (serial > _completedSerial)
            {
                _completedSerial = serial;
            }

            for (int i = 0; i < _pendingReleases.Count; i++)
            {
                if (_pendingReleases[i].Serial <= _completedSerial)
                {
                    FreeMemoryObject(_pendingReleases[i].Memory);
                    _pendingReleases.RemoveAt(i);
                    i--;
                }
            }
        }
    }

    /// <summary>Gives a memory object back now if nothing submitted before it may still use it, and otherwise once that's done. Under <see cref="_lock"/>.</summary>
    private void Release(DeviceMemory memory)
    {
        ulong serial = Interlocked.Read(ref _submittedSerial);
        if (serial <= _completedSerial)
        {
            FreeMemoryObject(memory);
        }
        else
        {
            _pendingReleases.Add(new PendingRelease(memory, serial));
        }
    }

    private void FreeMemoryObject(DeviceMemory memory)
    {
        _vk.FreeMemory(_device, memory, null);
        _memoryObjects--;
    }

    public VkMemoryBlock Allocate(
        PhysicalDeviceMemoryProperties memProperties,
        uint memoryTypeBits,
        MemoryPropertyFlags flags,
        bool persistentMapped,
        ulong size,
        ulong alignment)
    {
        return Allocate(
            memProperties,
            memoryTypeBits,
            flags,
            persistentMapped,
            size,
            alignment,
            false,
            default,
            default);
    }

    public VkMemoryBlock Allocate(
        PhysicalDeviceMemoryProperties memProperties,
        uint memoryTypeBits,
        MemoryPropertyFlags flags,
        bool persistentMapped,
        ulong size,
        ulong alignment,
        bool dedicated,
        Image dedicatedImage,
        VkBufferHandle dedicatedBuffer)
    {
        if (dedicated)
        {
            if (dedicatedImage.Handle != 0 && _getImageMemoryRequirements2 != null)
            {
                ImageMemoryRequirementsInfo2KHR requirementsInfo = new ImageMemoryRequirementsInfo2KHR
                {
                    SType = StructureType.ImageMemoryRequirementsInfo2
                };
                requirementsInfo.Image = dedicatedImage;
                MemoryRequirements2KHR requirements = new MemoryRequirements2KHR
                {
                    SType = StructureType.MemoryRequirements2
                };
                _getImageMemoryRequirements2(_device, &requirementsInfo, &requirements);
                size = requirements.MemoryRequirements.Size;
            }
            else if (dedicatedBuffer.Handle != 0 && _getBufferMemoryRequirements2 != null)
            {
                BufferMemoryRequirementsInfo2KHR requirementsInfo = new BufferMemoryRequirementsInfo2KHR
                {
                    SType = StructureType.BufferMemoryRequirementsInfo2
                };
                requirementsInfo.Buffer = dedicatedBuffer;
                MemoryRequirements2KHR requirements = new MemoryRequirements2KHR
                {
                    SType = StructureType.MemoryRequirements2
                };
                _getBufferMemoryRequirements2(_device, &requirementsInfo, &requirements);
                size = requirements.MemoryRequirements.Size;
            }
        }
        else
        {
            // Round up to the nearest multiple of bufferImageGranularity.
            size = ((size / _bufferImageGranularity) + 1) * _bufferImageGranularity;
        }
        _totalAllocatedBytes += size;

        lock (_lock)
        {
            if (!TryFindMemoryType(memProperties, memoryTypeBits, flags, out var memoryTypeIndex))
            {
                throw new NeoVeldridException("No suitable memory type.");
            }

            bool crowded = _memoryObjects >= _crowdedMemoryObjects;
            ulong minDedicatedAllocationSize = crowded
                ? (persistentMapped ? CrowdedDedicatedAllocationSizeDynamic : CrowdedDedicatedAllocationSizeNonDynamic)
                : (persistentMapped ? MinDedicatedAllocationSizeDynamic : MinDedicatedAllocationSizeNonDynamic);

            if (dedicated || size >= minDedicatedAllocationSize)
            {
                MemoryAllocateInfo allocateInfo = new MemoryAllocateInfo
                {
                    SType = StructureType.MemoryAllocateInfo
                };
                allocateInfo.AllocationSize = size;
                allocateInfo.MemoryTypeIndex = memoryTypeIndex;

                MemoryDedicatedAllocateInfoKHR dedicatedAI;
                if (dedicated)
                {
                    dedicatedAI = new MemoryDedicatedAllocateInfoKHR
                    {
                        SType = StructureType.MemoryDedicatedAllocateInfo
                    };
                    dedicatedAI.Buffer = dedicatedBuffer;
                    dedicatedAI.Image = dedicatedImage;
                    allocateInfo.PNext = &dedicatedAI;
                }

                DeviceMemory memory;
                Result allocationResult = _vk.AllocateMemory(_device, in allocateInfo, null, out memory);
                if (allocationResult != Result.Success)
                {
                    throw new NeoVeldridException("Unable to allocate sufficient Vulkan memory.");
                }

                void* mappedPtr = null;
                if (persistentMapped)
                {
                    Result mapResult = _vk.MapMemory(_device, memory, 0, size, 0, &mappedPtr);
                    if (mapResult != Result.Success)
                    {
                        _vk.FreeMemory(_device, memory, null);
                        throw new NeoVeldridException("Unable to map newly-allocated Vulkan memory.");
                    }
                }

                _memoryObjects++;
                return new VkMemoryBlock(memory, 0, size, memoryTypeBits, mappedPtr, true);
            }
            else
            {
                ChunkAllocatorSet allocator = GetAllocator(memoryTypeIndex, persistentMapped);
                ulong chunkSize = crowded
                    ? minDedicatedAllocationSize
                    : 4 * minDedicatedAllocationSize;
                bool result = allocator.Allocate(size, alignment, chunkSize, out VkMemoryBlock ret, out bool newChunk);
                if (newChunk)
                {
                    _memoryObjects++;
                }
                if (!result)
                {
                    throw new NeoVeldridException("Unable to allocate sufficient Vulkan memory.");
                }

                return ret;
            }
        }
    }

    public void Free(VkMemoryBlock block)
    {
        _totalAllocatedBytes -= block.Size;
        lock (_lock)
        {
            if (block.DedicatedAllocation)
            {
                Release(block.DeviceMemory);
            }
            else if (GetAllocator(block.MemoryTypeIndex, block.IsPersistentMapped).Free(block, out DeviceMemory emptied))
            {
                Release(emptied);
            }
        }
    }

    private ChunkAllocatorSet GetAllocator(uint memoryTypeIndex, bool persistentMapped)
    {
        ChunkAllocatorSet ret = null;
        if (persistentMapped)
        {
            if (!_allocatorsByMemoryType.TryGetValue(memoryTypeIndex, out ret))
            {
                ret = new ChunkAllocatorSet(_vk, _device, memoryTypeIndex, true);
                _allocatorsByMemoryType.Add(memoryTypeIndex, ret);
            }
        }
        else
        {
            if (!_allocatorsByMemoryTypeUnmapped.TryGetValue(memoryTypeIndex, out ret))
            {
                ret = new ChunkAllocatorSet(_vk, _device, memoryTypeIndex, false);
                _allocatorsByMemoryTypeUnmapped.Add(memoryTypeIndex, ret);
            }
        }

        return ret;
    }

    private class ChunkAllocatorSet : IDisposable
    {
        private readonly Silk.NET.Vulkan.Vk _vk;
        private readonly Device _device;
        private readonly uint _memoryTypeIndex;
        private readonly bool _persistentMapped;
        private readonly List<ChunkAllocator> _allocators = new List<ChunkAllocator>();

        public ChunkAllocatorSet(Silk.NET.Vulkan.Vk vk, Device device, uint memoryTypeIndex, bool persistentMapped)
        {
            _vk = vk;
            _device = device;
            _memoryTypeIndex = memoryTypeIndex;
            _persistentMapped = persistentMapped;
        }

        /// <summary>First fit across the chunks, or a new chunk of this size, or the allocation's if larger.</summary>
        public bool Allocate(ulong size, ulong alignment, ulong chunkSize, out VkMemoryBlock block, out bool newChunk)
        {
            newChunk = false;
            foreach (ChunkAllocator allocator in _allocators)
            {
                if (allocator.Allocate(size, alignment, out block))
                {
                    return true;
                }
            }

            ChunkAllocator newAllocator = new ChunkAllocator(_vk, _device, _memoryTypeIndex, _persistentMapped, Math.Max(chunkSize, size + alignment));
            _allocators.Add(newAllocator);
            newChunk = true;
            return newAllocator.Allocate(size, alignment, out block);
        }

        /// <summary>
        /// Returns a block to its chunk. A chunk left empty leaves the set at once, so nothing more is placed in it, and
        /// its memory is handed back to be freed, unless no other chunk of this set is empty: one empty chunk is kept, so
        /// short-lived staging buffers don't allocate and free a chunk on every upload.
        /// </summary>
        public bool Free(VkMemoryBlock block, out DeviceMemory emptied)
        {
            emptied = default;
            for (int i = 0; i < _allocators.Count; i++)
            {
                ChunkAllocator chunk = _allocators[i];
                if (chunk.Memory.Handle != block.DeviceMemory.Handle)
                {
                    continue;
                }

                chunk.Free(block);
                if (chunk.IsEmpty && HasEmptyChunkOtherThan(chunk))
                {
                    _allocators.RemoveAt(i);
                    emptied = chunk.Memory;
                    return true;
                }
                return false;
            }
            return false;
        }

        private bool HasEmptyChunkOtherThan(ChunkAllocator chunk)
        {
            foreach (ChunkAllocator other in _allocators)
            {
                if (other != chunk && other.IsEmpty)
                {
                    return true;
                }
            }
            return false;
        }

        public void Dispose()
        {
            foreach (ChunkAllocator allocator in _allocators)
            {
                allocator.Dispose();
            }
        }
    }

    private class ChunkAllocator : IDisposable
    {
        private readonly Silk.NET.Vulkan.Vk _vk;
        private readonly Device _device;
        private readonly uint _memoryTypeIndex;
        private readonly bool _persistentMapped;
        // Free ranges, by offset, never touching: neighbours merge when a block is freed.
        private readonly List<VkMemoryBlock> _freeBlocks = new List<VkMemoryBlock>();
        private readonly DeviceMemory _memory;
        private readonly void* _mappedPtr;

        private ulong _totalMemorySize;
        private ulong _totalAllocatedBytes = 0;
        private int _liveBlocks;

        public DeviceMemory Memory => _memory;

        /// <summary>Whether no block is allocated from this chunk.</summary>
        public bool IsEmpty => _liveBlocks == 0;

        /// <summary>A chunk of this size, which <see cref="VkDeviceMemoryManager.Allocate(PhysicalDeviceMemoryProperties, uint, MemoryPropertyFlags, bool, ulong, ulong, bool, Image, VkBufferHandle)"/> chooses.</summary>
        public ChunkAllocator(Silk.NET.Vulkan.Vk vk, Device device, uint memoryTypeIndex, bool persistentMapped, ulong size)
        {
            _vk = vk;
            _device = device;
            _memoryTypeIndex = memoryTypeIndex;
            _persistentMapped = persistentMapped;
            _totalMemorySize = size;

            MemoryAllocateInfo memoryAI = new MemoryAllocateInfo
            {
                SType = StructureType.MemoryAllocateInfo
            };
            memoryAI.AllocationSize = _totalMemorySize;
            memoryAI.MemoryTypeIndex = _memoryTypeIndex;
            // CheckResult only throws in Debug builds, so these throw themselves: a chunk must never join its set without memory.
            Result result = _vk.AllocateMemory(_device, in memoryAI, null, out _memory);
            if (result != Result.Success)
            {
                throw new NeoVeldridException("Unable to allocate sufficient Vulkan memory: " + result);
            }

            void* mappedPtr = null;
            if (persistentMapped)
            {
                result = _vk.MapMemory(_device, _memory, 0, _totalMemorySize, 0, &mappedPtr);
                if (result != Result.Success)
                {
                    _vk.FreeMemory(_device, _memory, null);
                    throw new NeoVeldridException("Unable to map newly-allocated Vulkan memory: " + result);
                }
            }
            _mappedPtr = mappedPtr;

            VkMemoryBlock initialBlock = new VkMemoryBlock(
                _memory,
                0,
                _totalMemorySize,
                _memoryTypeIndex,
                _mappedPtr,
                false);
            _freeBlocks.Add(initialBlock);
        }

        /// <summary>
        /// First fit. Padding before an aligned start stays on the free list, so a chunk whose blocks are all freed is one
        /// free range again.
        /// </summary>
        public bool Allocate(ulong size, ulong alignment, out VkMemoryBlock block)
        {
            checked
            {
                if (alignment == 0)
                {
                    alignment = 1;
                }

                for (int i = 0; i < _freeBlocks.Count; i++)
                {
                    VkMemoryBlock freeBlock = _freeBlocks[i];
                    ulong start = (freeBlock.Offset + alignment - 1) / alignment * alignment;
                    ulong padding = start - freeBlock.Offset;
                    if (padding >= freeBlock.Size || freeBlock.Size - padding < size)
                    {
                        continue;
                    }

                    ulong after = freeBlock.Size - padding - size;
                    _freeBlocks.RemoveAt(i);
                    int at = i;
                    if (padding > 0)
                    {
                        _freeBlocks.Insert(at++, new VkMemoryBlock(_memory, freeBlock.Offset, padding, _memoryTypeIndex, _mappedPtr, false));
                    }
                    if (after > 0)
                    {
                        _freeBlocks.Insert(at, new VkMemoryBlock(_memory, start + size, after, _memoryTypeIndex, _mappedPtr, false));
                    }

                    block = new VkMemoryBlock(_memory, start, size, _memoryTypeIndex, _mappedPtr, false);
#if DEBUG
                    CheckAllocatedBlock(block);
#endif
                    _totalAllocatedBytes += size;
                    _liveBlocks++;
                    return true;
                }

                block = default(VkMemoryBlock);
                return false;
            }
        }

        /// <summary>Returns a block to the free list, merged with the free ranges either side of it.</summary>
        public void Free(VkMemoryBlock block)
        {
            int i = 0;
            while (i < _freeBlocks.Count && _freeBlocks[i].Offset < block.Offset)
            {
                i++;
            }

            ulong start = block.Offset;
            ulong end = block.End;
            if (i < _freeBlocks.Count && _freeBlocks[i].Offset == end)
            {
                end = _freeBlocks[i].End;
                _freeBlocks.RemoveAt(i);
            }
            if (i > 0 && _freeBlocks[i - 1].End == start)
            {
                start = _freeBlocks[i - 1].Offset;
                _freeBlocks.RemoveAt(i - 1);
                i--;
            }
            _freeBlocks.Insert(i, new VkMemoryBlock(_memory, start, end - start, _memoryTypeIndex, _mappedPtr, false));

#if DEBUG
            RemoveAllocatedBlock(block);
#endif
            _totalAllocatedBytes -= block.Size;
            _liveBlocks--;
            Debug.Assert(_liveBlocks > 0 || (_freeBlocks.Count == 1 && _freeBlocks[0].Size == _totalMemorySize), "An empty chunk isn't one free range.");
        }

#if DEBUG
        private List<VkMemoryBlock> _allocatedBlocks = new List<VkMemoryBlock>();

        private void CheckAllocatedBlock(VkMemoryBlock block)
        {
            foreach (VkMemoryBlock oldBlock in _allocatedBlocks)
            {
                Debug.Assert(!BlocksOverlap(block, oldBlock), "Allocated blocks have overlapped.");
            }

            _allocatedBlocks.Add(block);
        }

        private bool BlocksOverlap(VkMemoryBlock first, VkMemoryBlock second)
        {
            ulong firstStart = first.Offset;
            ulong firstEnd = first.Offset + first.Size;
            ulong secondStart = second.Offset;
            ulong secondEnd = second.Offset + second.Size;

            return (firstStart <= secondStart && firstEnd > secondStart
                || firstStart >= secondStart && firstEnd <= secondEnd
                || firstStart < secondEnd && firstEnd >= secondEnd
                || firstStart <= secondStart && firstEnd >= secondEnd);
        }

        private void RemoveAllocatedBlock(VkMemoryBlock block)
        {
            Debug.Assert(_allocatedBlocks.Remove(block), "Unable to remove a supposedly allocated block.");
        }
#endif

        public void Dispose()
        {
            _vk.FreeMemory(_device, _memory, null);
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            foreach (PendingRelease pending in _pendingReleases)
            {
                FreeMemoryObject(pending.Memory);
            }
            _pendingReleases.Clear();
        }

        foreach (KeyValuePair<uint, ChunkAllocatorSet> kvp in _allocatorsByMemoryType)
        {
            kvp.Value.Dispose();
        }

        foreach (KeyValuePair<uint, ChunkAllocatorSet> kvp in _allocatorsByMemoryTypeUnmapped)
        {
            kvp.Value.Dispose();
        }
    }

    internal IntPtr Map(VkMemoryBlock memoryBlock, out Result result)
    {
        void* ret;
        result = _vk.MapMemory(_device, memoryBlock.DeviceMemory, memoryBlock.Offset, memoryBlock.Size, 0, &ret);
        return (IntPtr)ret;
    }
}

[DebuggerDisplay("[Mem:{DeviceMemory.Handle}] Off:{Offset}, Size:{Size} End:{Offset+Size}")]
internal unsafe struct VkMemoryBlock : IEquatable<VkMemoryBlock>
{
    public readonly uint MemoryTypeIndex;
    public readonly DeviceMemory DeviceMemory;
    public readonly void* BaseMappedPointer;
    public readonly bool DedicatedAllocation;

    public ulong Offset;
    public ulong Size;

    public readonly void* BlockMappedPointer => ((byte*)BaseMappedPointer) + Offset;
    public readonly bool IsPersistentMapped => BaseMappedPointer != null;
    public readonly ulong End => Offset + Size;

    public VkMemoryBlock(
        DeviceMemory memory,
        ulong offset,
        ulong size,
        uint memoryTypeIndex,
        void* mappedPtr,
        bool dedicatedAllocation)
    {
        DeviceMemory = memory;
        Offset = offset;
        Size = size;
        MemoryTypeIndex = memoryTypeIndex;
        BaseMappedPointer = mappedPtr;
        DedicatedAllocation = dedicatedAllocation;
    }

    public readonly bool Equals(VkMemoryBlock other)
    {
        return DeviceMemory.Handle.Equals(other.DeviceMemory.Handle)
            && Offset.Equals(other.Offset)
            && Size.Equals(other.Size);
    }
}
