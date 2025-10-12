# ParagonBench — Node.js Benchmark Harness

This folder contains the **Node.js benchmark harness** for the [OpenFluke Paragon](https://github.com/openfluke/paragon) AI framework.  
It demonstrates **cross-platform neural inference** using the `@openfluke/portal` WASM/JS interop layer (built on the native `teleport` C-ABI from Go),  
measuring **CPU vs GPU performance** via WebGPU across multiple fully-connected network sizes.

⚠️ **Note:** This Node.js version currently falls back to CPU-only inference ("WebGPU not supported") due to Node's limited native WebGPU support. It requires direct C-ABI integration (e.g., via Node WebGPU extensions or native addons) for true GPU acceleration. Outputs show near-identical CPU/GPU timings as a result.

---

## 🧩 Overview

**ParagonBench** is a TypeScript console project that loads the `@openfluke/portal` module and runs a full suite of forward-pass benchmarks.

It prints detailed timing, GPU adapter info, and delta statistics (mean absolute error, max deviation)  
for each test case, verifying deterministic parity between CPU and GPU paths. Full `ExtractOutput()` JSON is also logged for debugging.

---

## 📦 Folder Layout

```
node/
├── bench.ts              # Benchmark suite (TypeScript implementation)
├── index.ts              # Entrypoint script (robust portal import)
├── package.json          # NPM project file
├── tsconfig.json         # (optional) TypeScript config
├── README.md             # You're here
└── node_modules/         # (generated) Dependencies after `npm i`
```

---

## ⚙️ Requirements

- **Node.js 18.0+** (tested on 20.x)
- **npm** (or yarn/pnpm)
- `@openfluke/portal@^0.3.0` (handles WASM/JS interop to C-ABI)

---

## 🚀 Build & Run

From the `node/` folder:

```bash
cd ~/git/paragon_ecosystem/node

# Install dependencies
npm install

# Run the benchmark suite
npm run dev
```

This uses `tsx` for direct TypeScript execution (no separate build step).

To enable CSV output (uncomment in `index.ts`):

```typescript
suite.enableCsv("bench_results.csv");
```

---

## 🧠 What It Does

Each benchmark case creates a fully-connected feedforward network with
progressively larger hidden layers (`S1` → `XL2`):

| Case | Shape (layers)       | Approx. Weights | Description          |
| ---- | -------------------- | --------------- | -------------------- |
| S1   | 784 → 64 → 10        | 0.19 MB         | Small MNIST baseline |
| S2   | 784 → 128 → 10       | 0.39 MB         | Medium layer         |
| S3   | 784 → 256 → 10       | 0.78 MB         | Medium layer         |
| M1   | 784 → 256 → 256 → 10 | 1.03 MB         | Two hidden layers    |
| M2   | 784 → 384 → 384 → 10 | 1.73 MB         | Two hidden layers    |
| M3   | 784 → 512 → 512 → 10 | 2.55 MB         | Two hidden layers    |
| L1   | 784 → 768 ×3 → 10    | 6.83 MB         | Large                |
| L2   | 784 → 1024 ×3 → 10   | 11.11 MB        | Large                |
| XL1  | 784 → 1536 ×4 → 10   | 31.68 MB        | Extreme              |
| XL2  | 784 → 2048 ×4 → 10   | 54.23 MB        | Extreme              |

The suite measures:

- CPU and GPU inference time per forward pass
- WebGPU adapter initialization time
- Mean Absolute Error (MAE) and Max Δ between CPU and GPU outputs
- Logs full JSON outputs from `ExtractOutput()`
- Optionally writes CSV summary to `bench_results.csv`

---

## 🧮 Sample Output

```
⚙️  initPortal()…
2025/10/12 16:57:04 WebGPU not supported

=== S1 ===
Shape: 784 → 64 → 10   (~weights 0.19 MB)
GPU init: [{}]  in 3.62 ms  enabled=yes
CPU  ⏱ 1.509 ms
GPU  ⏱ 0.946 ms
Speedup: 1.60×
Δ(CPU vs GPU)  mae=0.00e+0  max=0.00e+0
CPU ExtractOutput: [[4.929233358052293e-25,1.085559301397596e-23,3.716318410626954e-33,2.393077041776897e-30,0.17797063291072845,1.749549187479121e-28,1.9514735151425924e-19,1.3025796067269743e-28,0.14495758712291718,0.677071750164032]]
GPU ExtractOutput: [[4.929233358052293e-25,1.085559301397596e-23,3.716318410626954e-33,2.393077041776897e-30,0.17797063291072845,1.749549187479121e-28,1.9514735151425924e-19,1.3025796067269743e-28,0.14495758712291718,0.677071750164032]]
```

_(Subsequent cases follow similarly, with near-1.0× speedups due to CPU fallback.)_

---

## 🧰 Extending

To add custom cases, extend `PRESETS.MNIST_ZOO` in `bench.ts` or pass a new array to `suite.runAll()`.

For other platforms/environments:

- Ensure `@openfluke/portal` supports your Node version (WASM via Emscripten).
- For GPU: Integrate Node WebGPU (e.g., `@webgpu/types`) or use a native addon for direct C-ABI calls.
- Debug portal loading: `index.ts` tries multiple import paths; check console for errors.

To regenerate C-ABI from Go sources: Build `teleport` with `cgo` and `tinygo` for WASM, then publish to NPM.

---

## 🔬 Part of the OpenFluke Ecosystem

This project is part of the **[OpenFluke](https://github.com/openfluke)** ecosystem —
a modular cross-platform AI + physics stack for reproducible intelligence.

- **Paragon:** WebGPU-agnostic neural framework (Go)
- **Teleport:** Paragon’s C-ABI interface for all languages
- **Portal:** WASM/JS interop for Node.js, Ionic, Bun
- **Primecraft:** Game/physics sandbox integrating Paragon AI
- **Iso-Demo:** Deterministic reproducibility and telemetry suite

---

## 📜 License

Apache 2.0 — © 2025 Samuel Watson / OpenFluke
