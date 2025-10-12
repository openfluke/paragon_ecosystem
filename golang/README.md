# ⚙️ Paragon-Go — Deterministic CPU vs GPU Benchmarks

Paragon-Go is the Go reference harness for benchmarking **CPU vs GPU inference** using the [Paragon](https://github.com/openfluke/paragon) framework.  
It validates **numerical determinism** and measures performance scaling across different layer widths and depths, verifying Paragon’s promise of _bit-level reproducibility_ across backends and devices.

---

## 🧩 Overview

Each case builds a small deterministic feed-forward network (MNIST-like topology), runs identical inputs through both CPU and GPU paths, and reports:

1. CPU vs GPU inference time (ms)
2. Speedup ratio (`cpu_ms / gpu_ms`)
3. Mean Absolute Error (`mae`) and maximum absolute difference (`max`)
4. Optional CSV logging for reproducibility across devices

If `mae=0.00E+00` and `max=0.00E+00`, the outputs are **bit-identical** across backends — the signature of Paragon’s reproducible compute graph.

---

## 🚀 Quick Start

### Prerequisites

- Go ≥ 1.21
- Vulkan/Metal/DX12/GL capable GPU (Mesa drivers fine)
- Environment variable for WebGPU backend

```bash
export WGPU_BACKEND=vulkan        # or gl / metal / dx12 depending on your system
```

### Install & Run

```bash
# 1. Initialize module
go mod init main

# 2. Add Paragon dependency
go get github.com/openfluke/paragon/v3@v3.1.4

# 3. Run benchmarks
go run .
```

To write CSV results:

```bash
go run . --quiet --csv bench_go.csv
```

To force a backend explicitly:

```bash
WGPU_BACKEND=vulkan go run . --quiet
```

Example output:

```
🚀 GPU Selected: 0x7d55 (0x8086) - Type: integrated-gpu
GPU init: [ok]  in 13.46 ms  enabled=yes
CPU ⏱ 8.074 ms
GPU ⏱ 2.194 ms
Speedup: 3.68×
Δ(CPU vs GPU)  mae=0.00E+00  max=0.00E+00  (n=10)
```

---

## 🧠 Benchmark Suite

Preset network shapes tested:

| ID  | Shape                                | Params (approx MB) | Notes             |
| :-- | :----------------------------------- | :----------------- | :---------------- |
| S1  | 784 → 64 → 10                        | 0.19               | Small baseline    |
| S2  | 784 → 128 → 10                       | 0.38               | Moderate width    |
| S3  | 784 → 256 → 10                       | 0.75               | Wider baseline    |
| M1  | 784 → 256 → 256 → 10                 | 1.1                | Two hidden layers |
| M2  | 784 → 384 → 384 → 10                 | 2.3                | Medium model      |
| M3  | 784 → 512 → 512 → 10                 | 3.9                | Large baseline    |
| L1  | 784 → 768 → 768 → 768 → 10           | 7.5                | Deep-medium       |
| L2  | 784 → 1024 → 1024 → 1024 → 10        | 13.4               | Deep large        |
| XL1 | 784 → 1536 → 1536 → 1536 → 1536 → 10 | 29                 | Extra-large       |
| XL2 | 784 → 2048 → 2048 → 2048 → 2048 → 10 | 53                 | Stress test       |

---

## 📊 CSV Schema

When using `--csv bench_go.csv`, each run appends rows like:

```
id,shape,estMB,cpu_ms,gpu_ms,speedup,mae,max,gpu_init_ms,adapter
```

Example:

```
L2,"784→1024→1024→1024→10",13.4,8.07,2.19,3.68,0.00E+00,0.00E+00,13.46,"0x7d55 (0x8086) integrated-gpu"
```

---

## ⚙️ Directory Structure

```
golang/
├── bench_paragon.go    # main benchmark program
├── go.mod              # module definition
├── go.sum              # dependency checksums
└── README.md           # this file
```

---

## 🔍 Tips

- Integrated GPUs show small startup latency (`gpu_init_ms`); this amortizes over multiple inferences.
- `--quiet` is useful for CI or aggregate runs.
- To confirm determinism across systems, compare `mae` and `max` — Paragon should be within 1e-8 across GPU vendors.
- Run larger shapes (≥512 units) to see real GPU scaling benefits.

---

## 🧾 License

Copyright © 2025 OpenFluke / Samuel Watson
Licensed under the **Apache License 2.0** (the “License”).

```
http://www.apache.org/licenses/LICENSE-2.0
```

Software distributed under the License is provided **“AS IS”**,
without warranties or conditions of any kind, express or implied.
See the License for specific language governing permissions and limitations under the License.

```

```
