using System.Collections.Generic;
using Silk.NET.Vulkan;

namespace NeoVeldrid.Vk;

/// <summary>Compares native handles without boxing Silk's command-buffer structs.</summary>
internal sealed class CommandBufferComparer : IEqualityComparer<CommandBuffer>
{
    public static readonly CommandBufferComparer Instance = new CommandBufferComparer();

    public bool Equals(CommandBuffer x, CommandBuffer y) => x.Handle == y.Handle;
    public int GetHashCode(CommandBuffer value) => value.Handle.GetHashCode();
}
