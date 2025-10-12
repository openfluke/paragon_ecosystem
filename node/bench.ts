// bench.ts — Node.js + Portal; CPU baseline and ASYNC GPU init (awaited); FULL ExtractOutput.
//
// - No Dawn/node-webgpu bootstrap. We just call into the native Portal API.
// - If InitializeOptimizedGPU is async (returns a Promise), we await it.
// - We also try to flip a "WebGPUNative" flag before init using several likely methods.
// - After init, we do a warmup, then time a forward on the "GPU" path.
// - We print FULL ExtractOutput for CPU and GPU, plus Δ stats, and init details.

import { initPortal } from "@openfluke/portal";
import * as fs from "fs";

type Layer = { Width: number; Height: number };
type Activ = "linear" | "relu" | "tanh" | "sigmoid" | "softmax";
type Jsonish = string;

export type CaseShape = {
  id: string; // S1..XL2
  layers: number[]; // e.g. [784, 256, 10] or [784, 256, 256, 10]
};

export const PRESETS = {
  MNIST_ZOO: [
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
  ] as CaseShape[],
};

function buildLayersFromSpec(spec: CaseShape): Layer[] {
  // Fully-connected expects flattened input: Width=784, Height=1
  const out: Layer[] = [{ Width: 784, Height: 1 }];
  for (let i = 1; i < spec.layers.length; i++)
    out.push({ Width: spec.layers[i], Height: 1 });
  return out;
}

function buildActivationsFromSpec(spec: CaseShape): Activ[] {
  const acts: Activ[] = ["linear"]; // input pseudo-layer
  for (let i = 1; i < spec.layers.length - 1; i++) acts.push("relu");
  acts.push("softmax");
  return acts;
}

function flattenOut(jsonStr: string): number[] {
  const a = JSON.parse(jsonStr);
  return Array.isArray(a) ? a.flat(2) : [];
}

function fixedVector784(seed = 123): Jsonish {
  // [[[784]]] deterministic values in [0,1)
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
  for (let i = 0; i < L.length - 1; i++) params += L[i] * L[i + 1];
  params += L.slice(1).reduce((acc, w) => acc + w, 0); // biases approx
  return (params * 4) / (1024 * 1024);
}

function isPromise<T = any>(x: any): x is Promise<T> {
  return (
    !!x &&
    (typeof x === "object" || typeof x === "function") &&
    typeof x.then === "function"
  );
}

/** Try to set a "WebGPUNative"/GPU flag to true using any likely API. */
function trySetWebGPUNativeTrue(nn: any): boolean {
  try {
    // 1) Canonical setter
    if (typeof nn.SetWebGPUNative === "function") {
      // Many Cgo/WASM shims take JSON-encoded args
      nn.SetWebGPUNative(JSON.stringify([true]));
      return true;
    }
    // 2) Alt toggles
    if (typeof nn.WebGPUNativeOn === "function") {
      nn.WebGPUNativeOn("[]");
      return true;
    }
    // 3) Bag configs
    if (typeof nn.Configure === "function") {
      nn.Configure(JSON.stringify([{ WebGPUNative: true }]));
      return true;
    }
    if (typeof nn.SetOptions === "function") {
      nn.SetOptions(JSON.stringify([{ WebGPUNative: true }]));
      return true;
    }
    // 4) Field setter
    if (typeof nn.SetField === "function") {
      nn.SetField(JSON.stringify(["WebGPUNative", true]));
      return true;
    }
    // 5) Generic Call
    if (typeof nn.Call === "function") {
      nn.Call(JSON.stringify(["SetWebGPUNative", [true]]));
      return true;
    }
  } catch {
    // ignore; we'll still attempt InitializeOptimizedGPU next
  }
  return false;
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
  gpuInitMs: number;
  adapter: string; // stringified response or marker
  outCPU_raw: string; // FULL JSON from ExtractOutput()
  outGPU_raw: string; // FULL JSON from ExtractOutput()
};

export class BenchSuite {
  private portal: any;
  private csvPath: string | null = null;

  async init() {
    if (!this.portal) {
      console.log("⚙️  initPortal()…");
      this.portal = await initPortal();
    }
  }

  enableCsv(path: string) {
    this.csvPath = path;
  }

  private writeCsvRow(r: BenchResult) {
    if (!this.csvPath) return;
    const hdr =
      "id,shape,estMB,cpu_ms,gpu_ms,speedup,mae,max,gpu_init_ms,adapter\n";
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
        r.gpuInitMs.toFixed(3),
        `"${r.adapter.replace(/"/g, '""')}"`,
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
    // Deterministic weights
    nn.PerturbWeights(JSON.stringify([0.1, 42]));
    return nn;
  }

  private forwardTimedRaw(nn: any, input: Jsonish) {
    const t0 = performance.now();
    nn.Forward(input);
    const raw = nn.ExtractOutput(); // FULL JSON string
    const ms = performance.now() - t0;
    return { ms, raw, flat: flattenOut(raw) };
  }

  /** Awaitable GPU init that mirrors the native “set flag → init → warmup” dance. */
  private async initGPUAwait(nn: any, warmupInput: Jsonish) {
    const flagged = trySetWebGPUNativeTrue(nn);

    const t0 = performance.now();
    let adapter = "unavailable";
    try {
      if (typeof nn.InitializeOptimizedGPU === "function") {
        const ret = nn.InitializeOptimizedGPU();
        const resp = isPromise(ret) ? await ret : ret;
        const txt =
          typeof resp === "string" ? resp.trim() : JSON.stringify(resp ?? {});
        adapter = txt && txt !== "" ? txt : "{}";
      } else {
        adapter = "unavailable";
      }
    } catch (e: any) {
      adapter = `error:${e?.message ?? "unknown"}`;
    }
    const ms = performance.now() - t0;

    // Optional warm-up if we *think* we toggled or got a non-empty adapter
    const enabled =
      flagged ||
      (adapter !== "{}" &&
        adapter !== "unavailable" &&
        !adapter.startsWith("error:"));
    if (enabled) {
      try {
        nn.Forward(warmupInput);
        nn.ExtractOutput();
      } catch {
        /* ignore */
      }
    }
    return { ms, adapter, enabled };
  }

  private shapeStr(spec: CaseShape) {
    return spec.layers.join(" → ");
  }

  async runOne(spec: CaseShape): Promise<BenchResult> {
    await this.init();
    const nn = this.makeNet(spec);
    const x = fixedVector784(123);

    // CPU warm-up + timed
    nn.Forward(x);
    nn.ExtractOutput();
    const cpu = this.forwardTimedRaw(nn, x);

    // GPU init (awaited) + warm-up handled inside
    const { ms: gpuInitMs, adapter, enabled } = await this.initGPUAwait(nn, x);

    // “GPU” forward timed (may still be fallback if adapter is "{}")
    nn.Forward(x);
    nn.ExtractOutput();
    const gpu = this.forwardTimedRaw(nn, x);

    const { mae, max } = diffStats(cpu.flat, gpu.flat);
    const r: BenchResult = {
      id: spec.id,
      shape: this.shapeStr(spec),
      estMB: estVramMB(spec),
      cpuMs: cpu.ms,
      gpuMs: gpu.ms,
      speedup: gpu.ms > 0 ? cpu.ms / gpu.ms : Infinity,
      mae,
      max,
      gpuInitMs,
      adapter,
      outCPU_raw: cpu.raw,
      outGPU_raw: gpu.raw,
    };

    this.writeCsvRow(r);

    console.log(`\n=== ${r.id} ===`);
    console.log(`Shape: ${r.shape}   (~weights ${r.estMB.toFixed(2)} MB)`);
    console.log(
      `GPU init: ${r.adapter}  in ${r.gpuInitMs.toFixed(2)} ms  enabled=${
        enabled ? "yes" : "no"
      }`
    );
    console.log(`CPU  ⏱ ${r.cpuMs.toFixed(3)} ms`);
    console.log(`GPU  ⏱ ${r.gpuMs.toFixed(3)} ms`);
    console.log(`Speedup: ${r.speedup.toFixed(2)}×`);
    console.log(
      `Δ(CPU vs GPU)  mae=${r.mae.toExponential(2)}  max=${r.max.toExponential(
        2
      )}`
    );
    console.log(`CPU ExtractOutput: ${r.outCPU_raw}`);
    console.log(`GPU ExtractOutput: ${r.outGPU_raw}`);

    // Cleanup if available (mirrors native)
    try {
      if (enabled && typeof nn.CleanupOptimizedGPU === "function")
        nn.CleanupOptimizedGPU();
    } catch {}

    return r;
  }

  async runAll(cases: CaseShape[]): Promise<BenchResult[]> {
    const out: BenchResult[] = [];
    for (const c of cases) out.push(await this.runOne(c));
    return out;
  }
}
