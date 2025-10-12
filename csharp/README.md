# ParagonBench — C# / .NET Benchmark Harness

This folder contains the **C# benchmark harness** for the [OpenFluke Paragon](https://github.com/openfluke/paragon) AI framework.  
It demonstrates **cross-platform neural inference** using the native `teleport` C-ABI (compiled from Go),  
measuring **CPU vs GPU performance** via WebGPU across multiple fully-connected network sizes.

---

## 🧩 Overview

**ParagonBench** is a .NET console project that dynamically loads the native Paragon `teleport` shared library  
(`teleport_amd64_linux.so` on Linux) and runs a full suite of forward-pass benchmarks.

It prints detailed timing, GPU adapter info, and delta statistics (mean absolute error, max deviation)  
for each test case, verifying deterministic parity between CPU and GPU paths.

---

## 📦 Folder Layout

```
csharp/
├── BenchSuite.cs           # Benchmark suite (ported from Node.js/TypeScript)
├── Portal.cs               # C# P/Invoke bridge to teleport_amd64_linux.so
├── Program.cs              # Entrypoint for .NET console app
├── ParagonBench.csproj     # .NET project file
├── linux_amd64/            # Native C-ABI binaries + headers
│   ├── teleport_amd64_linux.so
│   └── teleport_amd64_linux.h
├── bench.csv               # (generated) Benchmark CSV output
└── README.md               # You're here
```

---

## ⚙️ Requirements

- **.NET SDK 8.0+** (works with 9.x using `RollForward=LatestMajor`)
- Linux x64 (tested on Fedora 42, Ubuntu 24.04)
- `teleport_amd64_linux.so` present in `linux_amd64/`

---

## 🚀 Build & Run

From the `csharp/` folder:

```bash
cd ~/git/paragon_ecosystem/csharp

# Make sure the native library is discoverable
export LD_LIBRARY_PATH=$PWD/linux_amd64:$LD_LIBRARY_PATH

# Build & run
dotnet run --project ParagonBench.csproj -c Release
```

If you only want to run the built binary:

```bash
cd ~/git/paragon_ecosystem/csharp/bin/Release/net8.0
export LD_LIBRARY_PATH=$PWD/../../../linux_amd64:$LD_LIBRARY_PATH
dotnet ParagonBench.dll
```

---

## 🧠 What It Does

Each benchmark case creates a fully-connected feedforward network with
progressively larger hidden layers (`S1` → `XL2`):

| Case | Shape (layers)       | Approx. Weights | Description          |
| ---- | -------------------- | --------------- | -------------------- |
| S1   | 784 → 64 → 10        | 0.19 MB         | Small MNIST baseline |
| S2   | 784 → 128 → 10       | 0.39 MB         | Medium layer         |
| M1   | 784 → 256 → 256 → 10 | 1.03 MB         | Two hidden layers    |
| L2   | 784 → 1024 ×3 → 10   | 11.11 MB        | Large                |
| XL2  | 784 → 2048 ×4 → 10   | 54.23 MB        | Extreme              |

The suite measures:

- CPU and GPU inference time per forward pass
- WebGPU adapter initialization time
- Mean Absolute Error (MAE) and Max Δ between CPU and GPU outputs
- Writes CSV summary to `bench.csv`

---

## 🧮 Sample Output

```
⚙️  initPortal()…
🚀 GPU Selected: 0x7d55 (0x8086) - Type: integrated-gpu

=== S1 ===
Shape: 784 → 64 → 10   (~weights 0.19 MB)
GPU init: {"adapter":"Intel Arc (Mesa)"}  in 0.28 ms  enabled=yes
CPU  ⏱ 2.739 ms
GPU  ⏱ 0.007 ms
Speedup: 391.34×
Δ(CPU vs GPU)  mae=0.00E+00  max=0.00E+00
```

---

## 🧰 Extending

To build for other platforms:

- macOS: `teleport_amd64_darwin.dylib`
- Windows: `teleport_amd64_windows.dll`
- ARM64: `teleport_arm64_linux.so`

Drop the matching binary in a subfolder named accordingly (`darwin_amd64/`, `windows_amd64/`, etc.)
and update `Portal.cs`’s `Lib` constant if needed.

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
