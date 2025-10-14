// utils.ts
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fetch } from "undici";
import Jimp from "jimp";

/*──────────────────────── Types ────────────────────────*/
export type Layer = { Width: number; Height: number };
export type Activ = "linear" | "relu" | "tanh" | "sigmoid" | "softmax";

export type NN = {
  // inference bridge
  Forward?: (jsonArgs: string) => string | void;
  ExtractOutput?: () => string | number[];

  // GPU hooks (optional)
  InitializeOptimizedGPU?: () => string | Promise<string> | unknown;
  CleanupOptimizedGPU?: () => void;

  // Byte-based persistence (portable & deterministic)
  MarshalJSONModel?: () => string | [string, string?]; // => [base64, err?]
  UnmarshalJSONModel?: (jsonArgs: string) => string | [string?]; // => [err?]

  // Optional deterministic nudge
  PerturbWeights?: (jsonArgs: string) => void;
};

export type PortalAPI = {
  NewNetworkFloat32?: (
    layersJson: string,
    activsJson: string,
    fullyJson: string
  ) => NN;
};

/*──────────────────────── Config ───────────────────────*/
export const IMAGES_DIR = process.env.IMAGES_DIR ?? "./images";
export const MODEL_JSON = path.resolve(
  process.env.MODEL_JSON ?? "./mnist_paragon_model.json"
);
const MNIST_DIR = "./mnist_idx";

const URLS = {
  train_images:
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
  train_labels:
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
};

/*──────────────────── File helpers ─────────────────────*/
function ensureDir(p: string) {
  fs.mkdirSync(p, { recursive: true });
}

async function downloadFile(url: string, outPath: string) {
  if (fs.existsSync(outPath)) return;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download failed: ${url} → ${r.status}`);
  const b = Buffer.from(await r.arrayBuffer());
  await fs.promises.writeFile(outPath, b);
}

function gunzipToBuffer(gzPath: string): Buffer {
  const gz = fs.readFileSync(gzPath);
  return zlib.gunzipSync(gz);
}

/*──────────────────── MNIST parse ─────────────────────*/
function readImages2D(gzPath: string): number[][][] {
  const buf = gunzipToBuffer(gzPath);
  const magic = buf.readUInt32BE(0);
  if (magic !== 2051) throw new Error(`bad magic images: ${magic}`);
  const num = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const pixels = buf.subarray(16);
  const images: number[][][] = [];
  for (let i = 0; i < num; i++) {
    const img: number[][] = [];
    const base = i * rows * cols;
    for (let r = 0; r < rows; r++) {
      const row: number[] = [];
      for (let c = 0; c < cols; c++) {
        row.push(pixels[base + r * cols + c] / 255);
      }
      img.push(row);
    }
    images.push(img);
  }
  return images;
}

function readLabels(gzPath: string): number[] {
  const buf = gunzipToBuffer(gzPath);
  const magic = buf.readUInt32BE(0);
  if (magic !== 2049) throw new Error(`bad magic labels: ${magic}`);
  const num = buf.readUInt32BE(4);
  const labs = buf.subarray(8, 8 + num);
  return Array.from(labs);
}

/*────────────── Autopopulate images/0..9.png ───────────*/
export async function autopopulateImagesFolder() {
  ensureDir(IMAGES_DIR);
  if (fs.readdirSync(IMAGES_DIR).some((f) => f.toLowerCase().endsWith(".png")))
    return;

  console.log("🧩 Populating MNIST samples into ./images ...");
  ensureDir(MNIST_DIR);
  const imgOut = path.join(MNIST_DIR, path.basename(URLS.train_images));
  const labOut = path.join(MNIST_DIR, path.basename(URLS.train_labels));
  await downloadFile(URLS.train_images, imgOut);
  await downloadFile(URLS.train_labels, labOut);

  const imgs = readImages2D(imgOut);
  const labels = readLabels(labOut);

  const seen = new Set<number>();
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    if (seen.has(label)) continue;

    const img = imgs[i];
    const flat = img
      .flat()
      .map((v) => Math.max(0, Math.min(255, Math.round(v * 255))));
    const j = await Jimp.create(28, 28, 0x000000ff);

    let idx = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const g = flat[idx++];
        j.setPixelColor(Jimp.rgbaToInt(g, g, g, 255), x, y);
      }
    }
    await j.writeAsync(path.join(IMAGES_DIR, `${label}.png`));
    seen.add(label);
    if (seen.size === 10) break;
  }

  console.log("✅ Populated ./images with digits 0–9");
}

/*────────────── Build the MNIST net ───────────────*/
function buildMNISTNet(portal: PortalAPI, deterministicSeed?: number): NN {
  if (!portal.NewNetworkFloat32) {
    throw new Error("Portal.NewNetworkFloat32 not found");
  }
  // 28×28 → 256 → 10, linear/relu/softmax, fully connected
  const layers: Layer[] = [
    { Width: 28, Height: 28 },
    { Width: 256, Height: 1 },
    { Width: 10, Height: 1 },
  ];
  const activs: Activ[] = ["linear", "relu", "softmax"];
  const fully = [true, true, true];

  const nn = portal.NewNetworkFloat32(
    JSON.stringify(layers),
    JSON.stringify(activs),
    JSON.stringify(fully)
  );

  // Optional deterministic first-time init only
  if (typeof deterministicSeed === "number") {
    try {
      nn.PerturbWeights?.(JSON.stringify([0.1, deterministicSeed]));
    } catch {}
  }
  return nn;
}

/*────────────── Persistence (byte-based) ───────────────*/
export async function saveParagonModelJSON(nn: NN, modelPath: string) {
  if (typeof nn.MarshalJSONModel !== "function")
    throw new Error("MarshalJSONModel not found on NN");

  const out = nn.MarshalJSONModel();
  const [b64, err] = typeof out === "string" ? JSON.parse(out) : out;
  if (err) throw new Error(`MarshalJSONModel error: ${err}`);

  await fs.promises.mkdir(path.dirname(modelPath), { recursive: true });
  await fs.promises.writeFile(modelPath, Buffer.from(b64, "base64"));
}

export async function loadParagonModelJSON(nn: NN, modelPath: string) {
  if (typeof nn.UnmarshalJSONModel !== "function")
    throw new Error("UnmarshalJSONModel not found on NN");

  const bytes = await fs.promises.readFile(modelPath);
  const res = nn.UnmarshalJSONModel(
    JSON.stringify([Array.from(bytes.values())])
  );
  const [err] = typeof res === "string" ? JSON.parse(res) : res;
  if (err) throw new Error(`UnmarshalJSONModel error: ${err}`);
}

/*────────────── STRICT GPU DETECTION ───────────────*/
function looksLikeGpuUnavailable(textOrObj: unknown): boolean {
  if (typeof textOrObj === "string") {
    const t = textOrObj.toLowerCase().trim();
    if (
      t.length === 0 ||
      t === "{}" ||
      t === "[]" ||
      t === "null" ||
      t.includes("not supported") ||
      t.includes("unsupported") ||
      t.includes("unavailable") ||
      t.includes("no webgpu") ||
      t.includes("error")
    )
      return true;
    try {
      return looksLikeGpuUnavailable(JSON.parse(textOrObj));
    } catch {}
    return !(
      t.includes("initialized") ||
      t.includes("webgpu ok") ||
      t.includes("gpu ok")
    );
  }
  if (Array.isArray(textOrObj)) {
    if (textOrObj.length === 0) return true;
    if (
      textOrObj.every(
        (x) => x && typeof x === "object" && Object.keys(x).length === 0
      )
    )
      return true;
    return !textOrObj.some((x) => {
      if (!x || typeof x !== "object") return false;
      const o = x as any;
      return o.ok === true || o.available === true || o.backend === "webgpu";
    });
  }
  if (textOrObj && typeof textOrObj === "object") {
    const o = textOrObj as any;
    if (Object.keys(o).length === 0) return true;
    if (o.ok === true || o.available === true || o.backend === "webgpu")
      return false;
    return true;
  }
  return true;
}

/*────────────── Boot: make or load model ───────────────*/
export async function makeOrLoadModel(
  portal: PortalAPI,
  modelPath: string
): Promise<NN> {
  if (fs.existsSync(modelPath)) {
    // Build with NO perturbation; load saved weights
    const nn = buildMNISTNet(portal, undefined);
    await loadParagonModelJSON(nn, modelPath);
    console.log(`📥 Loaded model from ${modelPath}`);
    return nn;
  }

  // First-time: build with deterministic seed, forward a probe, save, reload, verify
  const nn = buildMNISTNet(portal, 1337);

  // Probe input: use a fixed all-zeros 28x28 to deterministically check persistence
  const probe = Array.from({ length: 28 }, () =>
    Array.from({ length: 28 }, () => 0)
  );

  nn.Forward?.(JSON.stringify([probe]));
  const outAraw = nn.ExtractOutput?.();
  const outA = Array.isArray(outAraw)
    ? outAraw
    : JSON.parse(String(outAraw) || "[]");

  await saveParagonModelJSON(nn, modelPath);
  console.log(`💾 Saved initial model to ${modelPath}`);

  // Reload into a fresh instance, verify identical output
  const nn2 = buildMNISTNet(portal, /*no seed*/ undefined);
  await loadParagonModelJSON(nn2, modelPath);
  nn2.Forward?.(JSON.stringify([probe]));
  const outBraw = nn2.ExtractOutput?.();
  const outB = Array.isArray(outBraw)
    ? outBraw
    : JSON.parse(String(outBraw) || "[]");

  const same = JSON.stringify(outA) === JSON.stringify(outB);
  console.log(`🔁 Reload check → ${same ? "✅ match" : "⚠️ mismatch"}`);

  return nn;
}

/*────────────── Handle builder (CPU/GPU) ───────────────*/
export async function makeHandle(
  portal: PortalAPI,
  modelPath: string,
  useGPU: boolean
): Promise<{ handle: NN; gpuEnabled: boolean }> {
  const nn = await makeOrLoadModel(portal, modelPath);

  // Strict GPU detection
  let gpuEnabled = false;
  if (useGPU && typeof nn.InitializeOptimizedGPU === "function") {
    try {
      const resp = await nn.InitializeOptimizedGPU();
      const txt = typeof resp === "string" ? resp : JSON.stringify(resp);
      const unavailable = looksLikeGpuUnavailable(resp ?? txt);
      gpuEnabled = !unavailable;
      console.log(`🚀 GPU init: ${txt}`);
    } catch (e) {
      console.warn(`⚠️ GPU init failed: ${(e as Error).message}`);
      gpuEnabled = false;
    }
  }

  return { handle: nn, gpuEnabled };
}

/*────────────── Inference helpers ───────────────*/
export function softmax(v: number[]): number[] {
  const m = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / s);
}

function parseOutputVector(out: unknown): number[] {
  if (Array.isArray(out)) return out as number[];
  if (typeof out === "string") {
    try {
      const arr = JSON.parse(out);
      if (Array.isArray(arr) && arr.length > 0) {
        if (Array.isArray(arr[0])) return arr[0] as number[];
        if (typeof arr[0] === "number") return arr as number[];
      }
    } catch {}
  }
  return [];
}

export async function forwardProbs10(
  nn: NN,
  img28x28: number[][]
): Promise<{ pred: number; probs: number[]; latency_sec: number }> {
  const t0 = performance.now();

  // Forward pass through the Paragon network
  nn.Forward?.(JSON.stringify([img28x28]));

  // Extract the model output (already includes softmax in final layer)
  const raw = nn.ExtractOutput?.();
  const probs = parseOutputVector(raw).slice(-10); // ← no softmax here!

  // Argmax to find predicted digit
  const pred = probs.reduce(
    (best, val, i, arr) => (val > arr[best] ? i : best),
    0
  );

  // Measure latency in seconds
  const latency_sec = Math.round((performance.now() - t0) * 1e3) / 1e6;

  return { pred, probs, latency_sec };
}

export async function loadPngAs28x28(p: string): Promise<number[][]> {
  if (!fs.existsSync(p)) throw new Error(`Not found: ${p}`);
  const j = await Jimp.read(p);
  j.resize(28, 28).grayscale();
  const out: number[][] = [];
  for (let y = 0; y < 28; y++) {
    const row: number[] = [];
    for (let x = 0; x < 28; x++) {
      const { r } = Jimp.intToRGBA(j.getPixelColor(x, y));
      row.push(r / 255);
    }
    out.push(row);
  }
  return out;
}
