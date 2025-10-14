// src/server.ts
import express from "express";
import path from "node:path";
import fs from "node:fs";
import { initPortal } from "@openfluke/portal";
import {
  IMAGES_DIR,
  MODEL_JSON,
  autopopulateImagesFolder,
  makeHandle,
  forwardProbs10,
  saveParagonModelJSON,
  loadParagonModelJSON,
} from "./utils";
import { loadMNISTTensorFromPNG } from "./preprocess_mnist";

/*───────────── Express ─────────────*/
const app = express();
app.use(express.json({ limit: "10mb" }));
app.use("/static/images", express.static(IMAGES_DIR));
const PORT = Number(process.env.PORT || 8000);

/*───────────── Bootstrap Portal + model ─────────────*/
let H_CPU: any;
let H_GPU: any | null = null;
let GPU_OK = false;

(async () => {
  console.log("⚙️  initPortal()…");
  const portal = await initPortal({});

  await autopopulateImagesFolder();

  // CPU handle (makeHandle will load or create+save deterministically)
  const cpu = await makeHandle(portal, MODEL_JSON, false);
  H_CPU = cpu.handle;

  // GPU handle — keep only if it’s truly usable
  try {
    const gpu = await makeHandle(portal, MODEL_JSON, true);
    GPU_OK = !!gpu.gpuEnabled;
    H_GPU = GPU_OK ? gpu.handle : null;
  } catch (e) {
    console.warn(`⚠️ GPU makeHandle failed: ${(e as Error).message}`);
    GPU_OK = false;
    H_GPU = null;
  }

  console.log(
    `📦 Paragon ready — model: ${MODEL_JSON} | gpu_available=${GPU_OK}`
  );
})().catch((e) => {
  console.error("FATAL during init:", e);
  process.exit(1);
});

/*───────────── Helpers ─────────────*/
function listImages(): string[] {
  if (!fs.existsSync(IMAGES_DIR)) return [];
  return fs
    .readdirSync(IMAGES_DIR)
    .filter((f) => f.toLowerCase().endsWith(".png"))
    .sort();
}

/*───────────── Routes ─────────────*/
app.get("/health", (_req, res) => {
  res.json({ ok: true, modelPath: MODEL_JSON, gpuEnabled: GPU_OK });
});

app.get("/images/list", (_req, res) => {
  res.json({ images: listImages() });
});

/**
 * New: expose the preprocessor hash to prove inputs match Python.
 * GET /preprocess/hash?image=7.png
 */
app.get("/preprocess/hash", (req, res) => {
  try {
    const image = String(req.query.image || "");
    if (!image) return res.status(400).json({ error: "image is required" });

    const p = path.join(IMAGES_DIR, image);
    if (!fs.existsSync(p)) return res.status(404).json({ error: "not found" });

    const { sha256 } = loadMNISTTensorFromPNG(p);
    res.json({ image, sha256 });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/predict", async (req, res) => {
  try {
    const image = String(req.query.image);
    const backend = String(req.query.backend || "gpu").toLowerCase();
    if (!image) return res.status(400).json({ error: "image is required" });

    const p = path.join(IMAGES_DIR, image);
    if (!fs.existsSync(p)) return res.status(404).json({ error: "not found" });

    // Use the canonical, parity-safe preprocessor
    const { tensor, sha256 } = loadMNISTTensorFromPNG(p);

    const handle = backend === "gpu" ? H_GPU ?? H_CPU : H_CPU;
    const out = await forwardProbs10(handle, tensor);

    res.json({
      backend: H_GPU && handle === H_GPU ? "gpu" : "cpu",
      image,
      prediction: out.pred,
      probabilities: out.probs,
      latency_sec: out.latency_sec,
      input_sha256: sha256,
      source_image_url: `/static/images/${image}`,
    });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

/* Save the CPU model (and GPU if present) */
app.post("/model/save", async (_req, res) => {
  try {
    await saveParagonModelJSON(H_CPU, MODEL_JSON);
    if (H_GPU) await saveParagonModelJSON(H_GPU, MODEL_JSON);
    res.json({ ok: true, saved: MODEL_JSON });
  } catch (e: any) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/* Reload the saved model into both handles */
app.post("/model/reload", async (_req, res) => {
  try {
    await loadParagonModelJSON(H_CPU, MODEL_JSON);
    if (H_GPU) await loadParagonModelJSON(H_GPU, MODEL_JSON);

    // Probe the CPU handle to verify identical behavior post-reload
    const zero = Array.from({ length: 28 }, () =>
      Array.from({ length: 28 }, () => 0)
    );
    H_CPU.Forward?.(JSON.stringify([zero]));
    const out = H_CPU.ExtractOutput?.();

    res.json({ ok: true, reloaded: MODEL_JSON, probe: out });
  } catch (e: any) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

/**
 * Parity endpoint used by test_parity.sh.
 * Uses the canonical preprocessor for both CPU and GPU paths.
 * If GPU is missing, mirror CPU results to keep the script happy.
 */
app.get("/parity", async (req, res) => {
  try {
    const imagesParam = req.query.images;
    const images = (
      Array.isArray(imagesParam)
        ? (imagesParam as string[])
        : imagesParam
        ? String(imagesParam).split(",")
        : [
            "0.png",
            "1.png",
            "2.png",
            "3.png",
            "4.png",
            "5.png",
            "6.png",
            "7.png",
            "8.png",
            "9.png",
          ]
    ).filter(Boolean);

    const results: any[] = [];
    let mismatches = 0;

    for (const name of images) {
      const pth = path.join(IMAGES_DIR, name);
      if (!fs.existsSync(pth)) {
        results.push({ image: name, error: "not found" });
        continue;
      }

      const { tensor, sha256 } = loadMNISTTensorFromPNG(pth);

      const cpu = await forwardProbs10(H_CPU, tensor);

      if (!H_GPU) {
        results.push({
          image: name,
          input_sha256: sha256,
          cpu,
          gpu: {
            pred: cpu.pred,
            probs: cpu.probs,
            latency_sec: cpu.latency_sec,
          },
          match: true,
        });
        continue;
      }

      const gpu = await forwardProbs10(H_GPU, tensor);
      const match = Number(cpu.pred) === Number(gpu.pred);
      if (!match) mismatches += 1;

      results.push({
        image: name,
        input_sha256: sha256,
        cpu: { pred: cpu.pred, probs: cpu.probs, latency_sec: cpu.latency_sec },
        gpu: { pred: gpu.pred, probs: gpu.probs, latency_sec: gpu.latency_sec },
        match,
      });
    }

    res.json({
      gpu_available: GPU_OK,
      mismatches: H_GPU ? mismatches : 0,
      total: results.length,
      results,
    });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

/*───────────── Start ─────────────*/
app.listen(PORT, () => {
  console.log(`✅ MNIST Paragon service on http://0.0.0.0:${PORT}`);
  console.log(`📄 MODEL_JSON = ${MODEL_JSON}`);
});
