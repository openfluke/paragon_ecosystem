// Portal.cs — RPC-based binding for teleport_amd64_linux.so
// Exports present: Paragon_NewNetworkFloat32, Paragon_PerturbWeights, Paragon_Call

using System;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenFluke
{
    internal static class Native
    {
        private const string Lib = "teleport_amd64_linux.so";

        [DllImport(Lib, EntryPoint = "Paragon_NewNetworkFloat32", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Paragon_NewNetworkFloat32(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string layersJson,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string activsJson,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string fullyJson);

        [DllImport(Lib, EntryPoint = "Paragon_PerturbWeights", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Paragon_PerturbWeights(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string argsJson);

        [DllImport(Lib, EntryPoint = "Paragon_Call", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr Paragon_Call_Raw(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string rpcJson);

        public static string Paragon_Call(IntPtr handle, string rpcJson)
        {
            var p = Paragon_Call_Raw(handle, rpcJson);
            return p == IntPtr.Zero ? "" : Marshal.PtrToStringUTF8(p) ?? "";
        }
    }

    internal static class J
    {
        public static readonly JsonSerializerOptions Cfg = new()
        {
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            WriteIndented = false
        };
        public static string S<T>(T obj) => JsonSerializer.Serialize(obj, Cfg);
    }

    /// Minimal OOP wrapper with the method names BenchSuite expects.
    public sealed class PortalNet
    {
        private readonly IntPtr _h;
        internal PortalNet(IntPtr h) { _h = h == IntPtr.Zero ? throw new InvalidOperationException("nil handle") : h; }

        public void PerturbWeights(string argsJson)
            => Native.Paragon_PerturbWeights(_h, argsJson);

        // Call(method, args[]) encoded as ["Method", [args...]]
        private string Rpc(string method, object[] args)
            => Native.Paragon_Call(_h, J.S(new object[] { method, args }));

        public void Forward(string inputJson) => _ = Rpc("Forward", new object[] { inputJson });
        public string ExtractOutput()         => Rpc("ExtractOutput", Array.Empty<object>());
        public string InitializeOptimizedGPU()=> Rpc("InitializeOptimizedGPU", Array.Empty<object>());
        public void CleanupOptimizedGPU()     => _ = Rpc("CleanupOptimizedGPU", Array.Empty<object>());

        // Optional toggles (safe even if engine ignores)
        public void SetWebGPUNative(string _)  => _ = Rpc("SetWebGPUNative", new object[] { true });
        public void WebGPUNativeOn(string _)   => _ = Rpc("WebGPUNativeOn", Array.Empty<object>());
        public void Configure(string _)        => _ = Rpc("Configure", new object[] { new { WebGPUNative = true } });
        public void SetOptions(string _)       => _ = Rpc("SetOptions", new object[] { new { WebGPUNative = true } });
        public void SetField(string _)         => _ = Rpc("SetField", new object[] { "WebGPUNative", true });
        public void CallRawPayload(string json)=> _ = Native.Paragon_Call(_h, json); // escape hatch
    }

    public static class Portal
    {
        public static dynamic Init()
        {
            return new
            {
                NewNetworkFloat32 = (Func<string, string, string, PortalNet>)((layers, activs, fully) =>
                {
                    var h = Native.Paragon_NewNetworkFloat32(layers, activs, fully);
                    return new PortalNet(h);
                })
            };
        }
    }
}
