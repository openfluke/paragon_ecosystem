// bench.ts — MNIST MLP CPU vs GPU bench (FIXED WebGPU detection)

import { initPortal } from "@openfluke/portal";
import * as fs from "fs";

type Layer = { Width: number; Height: number };
type Activ = "linear" | "relu" | "tanh" | "sigmoid" | "softmax";
type Jsonish = string;

export type CaseShape = {
  id: string;
  layers: number[];
};

export const PRESETS = {
  MNIST_ZOO: <CaseShape[]>[
    { id: "S1", layers: [784, 64, 10] },
    { id: "S2", layers: [784, 128, 10] },
    { id: "S3", layers: [784, 256, 10] },
    { id: "M1", layers: [784, 256, 256, 10] },
    { id: "M2", layers: [784, 384, 384, 10] },
    { id: "M3", layers: [784, 512, 512, 10] },
    { id: "L1", layers: [784, 768, 768, 768, 10] },
    { id: "L2", layers: [784, 1024, 1024, 1024, 10] },
    { id: "XL1", layers: [784, 1536, 1536, 1536, 1536, 10] },
    { id: "XL2", layers: [784, 2048, 2048, 2048, 2048, 10] },
  ],
};

function buildLayersFromSpec(spec: CaseShape): Layer[] {
  const out: Layer[] = [{ Width: 784, Height: 1 }];
  for (let i = 1; i < spec.layers.length; i++) {
    out.push({ Width: spec.layers[i], Height: 1 });
  }
  return out;
}

function buildActivationsFromSpec(spec: CaseShape): Activ[] {
  const acts: Activ[] = ["linear"];
  for (let i = 1; i < spec.layers.length - 1; i++) acts.push("relu");
  acts.push("softmax");
  return acts;
}

function flattenOut(jsonStr: string): number[] {
  const a = JSON.parse(jsonStr);
  return Array.isArray(a) ? a.flat(2) : [];
}

function fixedVector784(seed = 123): Jsonish {
  let s = seed >>> 0;
  const rand = () => (s = (s * 1664525 + 1013904223) >>> 0) / 0xffffffff;
  const vec = Array.from({ length: 784 }, () => Number(rand().toFixed(6)));
  return JSON.stringify([[vec]]);
}

function diffStats(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  if (!n) return { mae: 0, max: 0, n: 0 };
  let mae = 0,
    maxd = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    mae += d;
    if (d > maxd) maxd = d;
  }
  return { mae: mae / n, max: maxd, n };
}

function estVramMB(spec: CaseShape) {
  const L = spec.layers;
  let params = 0;
  for (let i = 0; i < L.length - 1; i++) {
    params += L[i] * L[i + 1];
  }
  params += L.slice(1).reduce((acc, w) => acc + w, 0);
  return (params * 4) / (1024 * 1024);
}

export type BenchResult = {
  id: string;
  shape: string;
  estMB: number;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
  mae: number;
  max: number;
  webgpuOk: boolean;
  gpuInitMs: number;
  adapter?: string;
  outCPU_raw: string;
  outGPU_raw: string;
  gpuFallbackToCpu: boolean; // NEW: explicit fallback flag
};

export class BenchSuite {
  private portal: any;
  private csvPath: string | null = null;
  private webgpuAvailable: boolean = false;

  async init() {
    if (!this.portal) {
      console.log("⚙️  initPortal()…");
      this.portal = await initPortal();

      // Check runtime WebGPU support
      this.webgpuAvailable = this.detectWebGPU();

      if (!this.webgpuAvailable) {
        console.warn("⚠️  WebGPU NOT available in this runtime!");
        console.warn("   • Bun does not support WebGPU yet");
        console.warn("   • Use Node.js with @webgpu/dawn or a browser");
        console.warn("   • All 'GPU' tests will actually run on CPU");
      } else {
        console.log("✅ WebGPU runtime support detected");
      }
    }
  }

  private detectWebGPU(): boolean {
    // Check if we're in Bun (which doesn't support WebGPU)
    if (typeof Bun !== "undefined") {
      return false;
    }

    // Check for browser WebGPU
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      return true;
    }

    // Check for Node.js @webgpu/dawn
    try {
      // This would need to be imported if using Node.js
      return false; // Conservative default
    } catch {
      return false;
    }
  }

  enableCsv(path: string) {
    this.csvPath = path;
  }

  private writeCsvRow(r: BenchResult) {
    if (!this.csvPath) return;
    const hdr =
      "id,shape,estMB,cpu_ms,gpu_ms,speedup,mae,max,webgpu_ok,gpu_init_ms,adapter,gpu_fallback\n";
    if (!fs.existsSync(this.csvPath)) fs.writeFileSync(this.csvPath, hdr);
    const row =
      [
        r.id,
        `"${r.shape}"`,
        r.estMB.toFixed(2),
        r.cpuMs.toFixed(3),
        r.gpuMs.toFixed(3),
        r.speedup.toFixed(2),
        r.mae.toExponential(2),
        r.max.toExponential(2),
        r.webgpuOk ? "true" : "false",
        r.gpuInitMs.toFixed(3),
        r.adapter ? `"${r.adapter.replace(/"/g, '""')}"` : "",
        r.gpuFallbackToCpu ? "true" : "false",
      ].join(",") + "\n";
    fs.appendFileSync(this.csvPath, row);
  }

  private makeNet(spec: CaseShape) {
    const layers = buildLayersFromSpec(spec);
    const activs = buildActivationsFromSpec(spec);
    const fully = new Array(layers.length).fill(true);

    const nn = this.portal.NewNetworkFloat32(
      JSON.stringify(layers),
      JSON.stringify(activs),
      JSON.stringify(fully)
    );
    nn.PerturbWeights(JSON.stringify([0.1, 42]));
    return nn;
  }

  private forwardTimedRaw(nn: any, input: Jsonish) {
    const t0 = performance.now();
    nn.Forward(input);
    const raw = nn.ExtractOutput();
    const ms = performance.now() - t0;
    return { ms, raw, flat: flattenOut(raw) };
  }

  private tryInitGPU(nn: any) {
    let ok = false,
      ms = 0,
      adapter: string | undefined;
    const t0 = performance.now();

    try {
      if (nn.InitializeOptimizedGPU) {
        const resp = nn.InitializeOptimizedGPU();
        ms = performance.now() - t0;

        // More robust detection
        if (resp && typeof resp === "object") {
          const respStr = JSON.stringify(resp);
          ok = respStr.length > 2 && respStr !== "{}";
          adapter = respStr;
        } else if (typeof resp === "string" && resp.trim().length > 2) {
          ok = true;
          adapter = resp;
        }
      }
    } catch (err) {
      ms = performance.now() - t0;
      ok = false;
      console.error("GPU init exception:", err);
    }

    return { ok, ms, adapter };
  }

  private cleanupGPU(nn: any) {
    try {
      if (nn.CleanupOptimizedGPU) nn.CleanupOptimizedGPU();
    } catch {}
  }

  private shapeStr(spec: CaseShape) {
    return spec.layers.join(" → ");
  }

  async runOne(spec: CaseShape): Promise<BenchResult> {
    await this.init();
    const nn = this.makeNet(spec);
    const vec784 = fixedVector784(123);

    // CPU: warmup + timed
    nn.Forward(vec784);
    nn.ExtractOutput();
    const cpu = this.forwardTimedRaw(nn, vec784);

    // GPU: init + warmup + timed
    const { ok: webgpuOk, ms: gpuInitMs, adapter } = this.tryInitGPU(nn);

    // Determine if we're actually using GPU or fell back to CPU
    const gpuFallbackToCpu = !webgpuOk || !this.webgpuAvailable;

    nn.Forward(vec784);
    nn.ExtractOutput();
    const gpu = this.forwardTimedRaw(nn, vec784);

    const { mae, max } = diffStats(cpu.flat, gpu.flat);

    const result: BenchResult = {
      id: spec.id,
      shape: this.shapeStr(spec),
      estMB: estVramMB(spec),
      cpuMs: cpu.ms,
      gpuMs: gpu.ms,
      speedup: gpu.ms > 0 ? cpu.ms / gpu.ms : Infinity,
      mae,
      max,
      webgpuOk,
      gpuInitMs,
      adapter,
      outCPU_raw: cpu.raw,
      outGPU_raw: gpu.raw,
      gpuFallbackToCpu,
    };

    this.cleanupGPU(nn);
    this.writeCsvRow(result);

    // Validate softmax sums
    const sumCPU = cpu.flat.reduce((a, b) => a + b, 0);
    const sumGPU = gpu.flat.reduce((a, b) => a + b, 0);
    if (!(isFinite(sumCPU) && Math.abs(sumCPU - 1) < 1e-3)) {
      console.warn(
        `[${spec.id}] ⚠️ CPU softmax sum=${sumCPU.toFixed(6)} (expected ≈1)`
      );
    }
    if (!(isFinite(sumGPU) && Math.abs(sumGPU - 1) < 1e-3)) {
      console.warn(
        `[${spec.id}] ⚠️ GPU softmax sum=${sumGPU.toFixed(6)} (expected ≈1)`
      );
    }

    // Enhanced output with clear fallback indication
    console.log(`\n=== ${result.id} ===`);
    console.log(
      `Shape: ${result.shape}   (~weights ${result.estMB.toFixed(2)} MB)`
    );

    if (result.gpuFallbackToCpu) {
      console.log(`GPU init: ⚠️  FALLBACK TO CPU (no WebGPU runtime support)`);
    } else if (result.webgpuOk) {
      console.log(
        `GPU init: ✅ in ${result.gpuInitMs.toFixed(2)} ms${
          result.adapter ? `  adapter=${result.adapter}` : ""
        }`
      );
    } else {
      console.log(`GPU init: ❌ failed in ${result.gpuInitMs.toFixed(2)} ms`);
    }

    console.log(`CPU  ⏱ ${result.cpuMs.toFixed(3)} ms`);
    console.log(
      `GPU  ⏱ ${result.gpuMs.toFixed(3)} ms ${
        result.gpuFallbackToCpu ? "(actually CPU)" : ""
      }`
    );
    console.log(`Speedup: ${result.speedup.toFixed(2)}×`);
    console.log(
      `Δ(CPU vs GPU)  mae=${result.mae.toExponential(
        2
      )}  max=${result.max.toExponential(2)}`
    );

    // Only show outputs for small models to avoid clutter
    if (result.estMB < 5) {
      console.log(`CPU ExtractOutput: ${result.outCPU_raw}`);
      console.log(`GPU ExtractOutput: ${result.outGPU_raw}`);
    }

    return result;
  }

  async runAll(cases: CaseShape[]): Promise<BenchResult[]> {
    const out: BenchResult[] = [];
    for (const c of cases) {
      out.push(await this.runOne(c));
    }

    // Summary
    console.log("\n" + "=".repeat(60));
    console.log("SUMMARY");
    console.log("=".repeat(60));

    const hasRealGpu = out.some((r) => !r.gpuFallbackToCpu);
    if (!hasRealGpu) {
      console.log("⚠️  NO REAL GPU ACCELERATION OCCURRED");
      console.log("   All 'GPU' times are actually CPU times");
      console.log("\n💡 To get real WebGPU acceleration:");
      console.log("   1. Use a WebGPU-capable browser (Chrome/Edge 113+)");
      console.log("   2. Use Node.js with @webgpu/dawn package");
      console.log("   3. Avoid Bun (no WebGPU support yet)");
    } else {
      console.log("✅ Real GPU acceleration detected");
      const avgSpeedup = out.reduce((s, r) => s + r.speedup, 0) / out.length;
      console.log(`   Average speedup: ${avgSpeedup.toFixed(2)}×`);
    }

    return out;
  }
}
