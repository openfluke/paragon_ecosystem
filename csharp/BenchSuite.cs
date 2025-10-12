// BenchSuite.cs — .NET 8+ bench port (no tuple deconstruction)

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace OpenFluke.Bench
{
    public record CaseShape(string Id, int[] Layers);

    public record BenchResult(
        string Id,
        string Shape,
        double EstMB,
        double CpuMs,
        double GpuMs,
        double Speedup,
        double Mae,
        double Max,
        double GpuInitMs,
        string Adapter,
        string OutCPU_Raw,
        string OutGPU_Raw
    );

    public static class Presets
    {
        public static readonly CaseShape[] MNIST_ZOO = new[]
        {
            new CaseShape("S1",  new[]{784,  64, 10}),
            new CaseShape("S2",  new[]{784, 128, 10}),
            new CaseShape("S3",  new[]{784, 256, 10}),
            new CaseShape("M1",  new[]{784, 256, 256, 10}),
            new CaseShape("M2",  new[]{784, 384, 384, 10}),
            new CaseShape("M3",  new[]{784, 512, 512, 10}),
            new CaseShape("L1",  new[]{784, 768, 768, 768, 10}),
            new CaseShape("L2",  new[]{784, 1024,1024,1024, 10}),
            new CaseShape("XL1", new[]{784, 1536,1536,1536,1536, 10}),
            new CaseShape("XL2", new[]{784, 2048,2048,2048,2048, 10}),
        };
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

    internal sealed class ForwardResult { public double Ms; public string Raw = ""; public List<double> Flat = new(); }
    internal sealed class GpuInitInfo { public double Ms; public string Adapter = "unavailable"; public bool Enabled; }
    internal sealed class DiffStatsResult { public double Mae; public double Max; public int N; }

    public sealed class BenchSuite
    {
        private readonly Func<dynamic> _initPortal;
        private dynamic? _portal;
        private string? _csvPath;

        public BenchSuite(Func<dynamic> initPortal) { _initPortal = initPortal; }
        public void EnableCsv(string path) => _csvPath = path;

        private void EnsurePortal()
        {
            if (_portal is null)
            {
                Console.WriteLine("⚙️  initPortal()…");
                _portal = _initPortal();
            }
        }

        public string ShapeStr(CaseShape spec) => string.Join(" → ", spec.Layers);

        public static double EstimateVramMB(CaseShape spec)
        {
            var L = spec.Layers;
            long paramsCount = 0;
            for (int i = 0; i < L.Length - 1; i++) paramsCount += (long)L[i] * L[i + 1];
            paramsCount += L.Skip(1).Sum(x => (long)x);
            return paramsCount * 4.0 / (1024 * 1024);
        }

        private static IReadOnlyList<object> BuildLayersFromSpec(CaseShape spec)
        {
            var outLayers = new List<object> { new { Width = 784, Height = 1 } };
            for (int i = 1; i < spec.Layers.Length; i++)
                outLayers.Add(new { Width = spec.Layers[i], Height = 1 });
            return outLayers;
        }

        private static IReadOnlyList<string> BuildActivationsFromSpec(CaseShape spec)
        {
            var acts = new List<string> { "linear" };
            for (int i = 1; i < spec.Layers.Length - 1; i++) acts.Add("relu");
            acts.Add("softmax");
            return acts;
        }

        private static IReadOnlyList<bool> BuildFully(CaseShape spec)
            => Enumerable.Repeat(true, spec.Layers.Length).ToArray();

        private static string FixedVector784(int seed = 123)
        {
            uint s = unchecked((uint)seed);
            double Next() { s = unchecked(s * 1664525u + 1013904223u); return s / 0xffffffffu; }
            var vec = Enumerable.Range(0, 784).Select(_ => Math.Round(Next(), 6)).ToArray();
            var payload = new object[] { new object[] { vec } };
            return J.S(payload);
        }

        private static List<double> FlattenOut(string json)
        {
            try
            {
                using var doc = JsonDocument.Parse(json);
                var list = new List<double>();
                void Walk(JsonElement e)
                {
                    if (e.ValueKind == JsonValueKind.Array)
                        foreach (var c in e.EnumerateArray()) Walk(c);
                    else if (e.ValueKind == JsonValueKind.Number && e.TryGetDouble(out var d))
                        list.Add(d);
                }
                Walk(doc.RootElement);
                return list;
            }
            catch { return new(); }
        }

        private static double StopwatchMs(Action act)
        {
            var sw = Stopwatch.StartNew();
            act();
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        }

        private static string SafeToString(object? o) => o is null ? "" : (o is string s ? s : J.S(o));

        private static bool TryInvoke(dynamic obj, string name, params object[] args)
        {
            try
            {
                var t = (object)obj;
                var m = t.GetType().GetMethod(name, BindingFlags.Public | BindingFlags.Instance);
                if (m is null) return false;
                _ = m.Invoke(t, args);
                return true;
            }
            catch { return false; }
        }

        private dynamic MakeNet(CaseShape spec)
        {
            var layers = BuildLayersFromSpec(spec);
            var activs = BuildActivationsFromSpec(spec);
            var fully  = BuildFully(spec);

            dynamic nn = _portal!.NewNetworkFloat32(J.S(layers), J.S(activs), J.S(fully));
            TryInvoke(nn, "PerturbWeights", J.S(new object[] { 0.1, 42 }));
            return nn;
        }

        private static ForwardResult ForwardTimedRaw(dynamic nn, string inputJson)
        {
            var fr = new ForwardResult();
            fr.Ms = StopwatchMs(() =>
            {
                nn.Forward(inputJson);
                fr.Raw = SafeToString(nn.ExtractOutput());
            });
            fr.Flat = FlattenOut(fr.Raw);
            return fr;
        }

        private static bool TrySetWebGpuNativeTrue(dynamic nn)
        {
            try
            {
                if (TryInvoke(nn, "SetWebGPUNative", J.S(new object[] { true }))) return true;
                if (TryInvoke(nn, "WebGPUNativeOn", "[]")) return true;
                if (TryInvoke(nn, "Configure", J.S(new object[] { new { WebGPUNative = true } }))) return true;
                if (TryInvoke(nn, "SetOptions", J.S(new object[] { new { WebGPUNative = true } }))) return true;
                if (TryInvoke(nn, "SetField", J.S(new object[] { "WebGPUNative", true }))) return true;
                if (TryInvoke(nn, "Call", J.S(new object[] { "SetWebGPUNative", new object[] { true } }))) return true;
            }
            catch { }
            return false;
        }

        private static GpuInitInfo InitGpuAwait(dynamic nn, string warmupInput)
        {
            bool flagged = TrySetWebGpuNativeTrue(nn);
            string adapter = "unavailable";

            double ms = StopwatchMs(() =>
            {
                try
                {
                    var t = (object)nn;
                    var m = t.GetType().GetMethod("InitializeOptimizedGPU", BindingFlags.Public | BindingFlags.Instance);
                    if (m is null) { adapter = "unavailable"; return; }
                    var resp = m.Invoke(t, Array.Empty<object>());
                    var txt = SafeToString(resp).Trim();
                    adapter = string.IsNullOrWhiteSpace(txt) ? "{}" : txt;
                }
                catch (TargetInvocationException tie) { adapter = "error:" + (tie.InnerException?.Message ?? "unknown"); }
                catch (Exception e) { adapter = "error:" + (e.Message ?? "unknown"); }
            });

            bool enabled = flagged || (adapter != "{}" && adapter != "unavailable" && !adapter.StartsWith("error:", StringComparison.OrdinalIgnoreCase));
            if (enabled)
            {
                try { nn.Forward(warmupInput); _ = SafeToString(nn.ExtractOutput()); } catch { }
            }
            return new GpuInitInfo { Ms = ms, Adapter = adapter, Enabled = enabled };
        }

        private static DiffStatsResult DiffStats(IReadOnlyList<double> a, IReadOnlyList<double> b)
        {
            int n = Math.Min(a.Count, b.Count);
            if (n == 0) return new DiffStatsResult { Mae = 0, Max = 0, N = 0 };
            double mae = 0, maxd = 0;
            for (int i = 0; i < n; i++)
            {
                var d = Math.Abs(a[i] - b[i]);
                mae += d; if (d > maxd) maxd = d;
            }
            return new DiffStatsResult { Mae = mae / n, Max = maxd, N = n };
        }

        private void WriteCsvRow(BenchResult r)
        {
            if (_csvPath is null) return;
            bool writeHeader = !File.Exists(_csvPath);
            using var fs = new FileStream(_csvPath, FileMode.Append, FileAccess.Write, FileShare.Read);
            using var sw = new StreamWriter(fs, new UTF8Encoding(false));
            if (writeHeader)
                sw.WriteLine("id,shape,estMB,cpu_ms,gpu_ms,speedup,mae,max,gpu_init_ms,adapter");

            string esc(string s) => "\"" + s.Replace("\"", "\"\"") + "\"";
            string f(double d, int k=3) => d.ToString("F"+k, CultureInfo.InvariantCulture);
            string sci(double d) => d.ToString("0.00E+00", CultureInfo.InvariantCulture);

            sw.WriteLine(string.Join(",",
                r.Id,
                esc(r.Shape),
                r.EstMB.ToString("F2", CultureInfo.InvariantCulture),
                f(r.CpuMs), f(r.GpuMs),
                r.Speedup.ToString("F2", CultureInfo.InvariantCulture),
                sci(r.Mae), sci(r.Max),
                f(r.GpuInitMs),
                esc(r.Adapter)
            ));
        }

        public BenchResult RunOne(CaseShape spec)
        {
            EnsurePortal();
            var nn = MakeNet(spec);
            string x = FixedVector784(123);

            // CPU
            TryInvoke(nn, "Forward", x);
            _ = SafeToString(nn.ExtractOutput());
            var cpu = ForwardTimedRaw(nn, x);

            // GPU
            var gpuInfo = InitGpuAwait(nn, x);
            double gpuInitMs = gpuInfo.Ms; string adapter = gpuInfo.Adapter; bool enabled = gpuInfo.Enabled;

            TryInvoke(nn, "Forward", x);
            _ = SafeToString(nn.ExtractOutput());
            var gpu = ForwardTimedRaw(nn, x);

            var dif = DiffStats(cpu.Flat, gpu.Flat);

            var result = new BenchResult(
                Id: spec.Id,
                Shape: ShapeStr(spec),
                EstMB: EstimateVramMB(spec),
                CpuMs: cpu.Ms,
                GpuMs: gpu.Ms,
                Speedup: gpu.Ms > 0 ? cpu.Ms / gpu.Ms : double.PositiveInfinity,
                Mae: dif.Mae,
                Max: dif.Max,
                GpuInitMs: gpuInitMs,
                Adapter: adapter,
                OutCPU_Raw: cpu.Raw,
                OutGPU_Raw: gpu.Raw
            );

            WriteCsvRow(result);

            Console.WriteLine($"\n=== {result.Id} ===");
            Console.WriteLine($"Shape: {result.Shape}   (~weights {result.EstMB:F2} MB)");
            Console.WriteLine($"GPU init: {result.Adapter}  in {result.GpuInitMs:F2} ms  enabled={(enabled ? "yes" : "no")}");
            Console.WriteLine($"CPU  ⏱ {result.CpuMs:F3} ms");
            Console.WriteLine($"GPU  ⏱ {result.GpuMs:F3} ms");
            Console.WriteLine($"Speedup: {result.Speedup:F2}×");
            Console.WriteLine($"Δ(CPU vs GPU)  mae={result.Mae:0.00E+00}  max={result.Max:0.00E+00}");
            Console.WriteLine($"CPU ExtractOutput: {result.OutCPU_Raw}");
            Console.WriteLine($"GPU ExtractOutput: {result.OutGPU_Raw}");

            TryInvoke(nn, "CleanupOptimizedGPU");
            return result;
        }

        public List<BenchResult> RunAll(IEnumerable<CaseShape> cases)
        {
            var outList = new List<BenchResult>();
            foreach (var c in cases) outList.Add(RunOne(c));
            return outList;
        }
    }
}
