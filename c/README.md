# 🧩 Paragon C BenchSuite

Cross-language benchmark harness for [OpenFluke/Paragon](https://github.com/openfluke/paragon).  
Runs the same forward passes in **C** as the `.NET BenchSuite`, comparing CPU vs GPU performance and verifying output parity.

---

## ⚙️ Overview

This C harness dynamically loads the `teleport_*.so` (or any Paragon-compatible shared library) and calls its exported functions directly through `dlopen`/`dlsym`.

It builds networks with MNIST-like shapes (`784→64→10`, `784→128→10`, etc.), runs one forward pass on both CPU and GPU, and prints detailed timing and output comparison:

```
=== S1 (784→64→10) ===
Shape: 784 → 64 → 10   (~weights 0.19 MB)
GPU init: [null]  in 39.07 ms
CPU  ⏱ 0.348 ms
GPU  ⏱ 0.251 ms
Speedup: 1.38x
Δ(CPU vs GPU)  mae=0E+00  max=0E+00
CPU ExtractOutput: [[...]]
GPU ExtractOutput: [[...]]
```

The test ensures near-bit-identical outputs across CPU and GPU, verifying reproducibility.

---

## 🧰 Requirements

- Linux or macOS with GCC
- A compatible Paragon or Teleport shared library (e.g. `teleport_amd64_linux.so`)
- WebGPU-compatible GPU driver (Mesa, Vulkan, Metal, etc.)

---

## 🔨 Build

```bash
cd ~/git/paragon_ecosystem/c
make clean && make
```

This compiles:

- `bench.c` — main benchmark logic
- `paragon.c` — dynamic loader for `.so` APIs
- `paragon.h` — function declarations

The build produces a single binary:

```
bench
```

---

## 🚀 Run

### Default (auto-detects .so)

```bash
export LD_LIBRARY_PATH=$PWD/linux_amd64:$LD_LIBRARY_PATH
./bench
```

### Explicit path

```bash
./bench linux_amd64/teleport_amd64_linux.so
```

### Quiet mode

(Suppress per-index diffs or raw vector output)

```bash
./bench linux_amd64/teleport_amd64_linux.so --quiet
```

---

## 🧮 What It Does

For each predefined shape (`S1` … `XL2`):

1. Builds a float32 feed-forward network
2. Runs a forward pass on CPU
3. Enables WebGPU backend and reruns on GPU
4. Computes:

   - `mae` (mean absolute error)
   - `max` (maximum absolute error)
   - Timing for both passes
   - Speedup ratio

5. Prints `ExtractOutput()` for both paths

When GPU and CPU agree (`mae≈0`, `max≈0`), you’ve achieved **bit-consistent cross-backend inference** — the core goal of Paragon’s deterministic reproducibility.

---

## 🧩 File Structure

```
c/
├── bench.c        # Benchmark suite
├── paragon.c      # Dynamic loader + helper functions
├── paragon.h      # API header
├── Makefile       # Simple GCC build
└── README.md      # You are here
```

---

## 🧠 Notes

- The loader tries multiple symbol names:

  - `Paragon_NewNetworkFloat32`
  - `Teleport_NewNetworkFloat32`
  - `NewNetworkFloat32`
  - `Paragon_Call` / `Teleport_Call` / `Call`

- If no explicit function is exported, it will fallback to `Call(0, "NewNetworkFloat32", args)`.
- Fully GPU-agnostic — works on AMD, NVIDIA, Intel, and Apple M-series.

---

## 📊 Output Parity Example

When correctly linked, CPU and GPU results should match perfectly:

```
CPU ExtractOutput: [[0.9985790848731995, 0.0003247287531848997, ...]]
GPU ExtractOutput: [[0.9985790848731995, 0.0003247287531848997, ...]]
Δ(CPU vs GPU)  mae=0E+00  max=0E+00
```

---

## 🧾 License

Apache-2.0 © 2025 [OpenFluke](https://github.com/openfluke)

Part of the Paragon ecosystem — “AI that runs identically everywhere.”
