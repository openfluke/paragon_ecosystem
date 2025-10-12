# 🧠 Paragon-Py — CPU vs GPU Microbenchmarks

Paragon-Py is the Python reference harness for benchmarking **CPU vs GPU inference** on the [Paragon AI framework](https://github.com/openfluke/paragon).  
It verifies _bit-level reproducibility_ across backends (CPU, Vulkan, Metal, DX12, WebGPU-GL) and quantifies performance scaling as model sizes grow.

---

## 🔧 Overview

Each test case builds a small fully-connected network (MNIST-style), initializes deterministic weights, and runs identical inputs through both CPU and GPU paths.

The benchmark then:

1. Measures inference latency (ms) on CPU and GPU.
2. Computes `speedup = cpu_ms / gpu_ms`.
3. Validates correctness via `mae` (mean absolute error) and `max` deviation between outputs.
4. Optionally writes results to CSV for aggregate analysis.

If `mae = 0.00E+00` and `max = 0.00E+00`, the outputs are _bit-identical_ — a hallmark of Paragon’s deterministic design.

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.9
- `pip install --upgrade paragon-py`
- Vulkan/Metal/DX12/GL capable GPU (Mesa i915 works; warnings are harmless)
- Environment variable for backend (Linux default: `vulkan`)

```bash
export WGPU_BACKEND=vulkan        # or gl / metal / dx12 depending on platform
```

### Run Benchmarks

```bash
# Verbose mode — prints vectors, softmax, per-index diffs, and timings
python3 bench_paragon.py

# Quiet mode — summary only, suitable for CSV logging
python3 bench_paragon.py --quiet --csv bench_py.csv
```

Example output:

```
🚀 GPU Selected: 0x7d55 (0x8086) - Type: integrated-gpu
GPU init: [ok]  in 0.57 ms  enabled=yes
CPU ⏱ 4.809 ms
GPU ⏱ 0.954 ms
Speedup: 5.04×
Δ(CPU vs GPU) mae=0.00E+00  max=0.00E+00
```

---

## 🧩 Benchmark Cases

Preset architectures (single-sample forward pass):

| ID  | Shape                                | Approx Params | Notes             |
| :-- | :----------------------------------- | :------------ | :---------------- |
| S1  | 784 → 64 → 10                        | 0.19 MB       | Small baseline    |
| S2  | 784 → 128 → 10                       | 0.38 MB       | Moderate width    |
| S3  | 784 → 256 → 10                       | 0.75 MB       | Wider baseline    |
| M1  | 784 → 256 → 256 → 10                 | 1.1 MB        | Two hidden layers |
| M2  | 784 → 384 → 384 → 10                 | 2.3 MB        | Medium            |
| M3  | 784 → 512 → 512 → 10                 | 3.9 MB        | Large             |
| L1  | 784 → 768 → 768 → 768 → 10           | 7.5 MB        | Deep-medium       |
| L2  | 784 → 1024 → 1024 → 1024 → 10        | 13.4 MB       | Deep large        |
| XL1 | 784 → 1536 → 1536 → 1536 → 1536 → 10 | 29 MB         | Extra-large       |
| XL2 | 784 → 2048 → 2048 → 2048 → 2048 → 10 | 53 MB         | Stress test       |

---

## 📊 CSV Schema

When `--csv` is used, each row logs:

```
id,shape,estMB,cpu_ms,gpu_ms,speedup,mae,max,gpu_init_ms,adapter
```

Example:

```
S1,"784→64→10",0.19,4.81,0.95,5.04,0.00E+00,0.00E+00,0.57,"0x7d55 (0x8086) integrated-gpu"
```

---

## ⚙️ Tips & Notes

- Integrated GPUs may show mild startup latency (`gpu_init_ms`) — it amortizes quickly in batch inference.
- Mesa `srgb clears` warnings are benign.
- Larger networks (≥ 512 hidden units) demonstrate 2–8× speedups even on entry-level GPUs.
- Deterministic equality across CPU/GPU confirms **numerical reproducibility**, the Paragon signature feature.

---

## 🧪 Directory Structure

```
python/
├── bench_paragon.py      # main benchmark script
├── utils.py              # helper math/utilities
├── bench_py.csv          # (optional) output file
├── README.md             # this file
└── .gitignore            # ignores venvs & logs
```

---

## 🧰 Example Integration

Use the benchmark module directly in Python for quick inference checks:

```python
from paragon_py import Paragon
import numpy as np

net = Paragon.new_network([784, 256, 10])
x = np.random.rand(784).astype(np.float32)
out = net.forward_cpu(x)
print(out)
```

---

## 🧾 License

Copyright © 2025 OpenFluke / Samuel Watson
Licensed under the **Apache License 2.0** (the “License”);

```
http://www.apache.org/licenses/LICENSE-2.0
```
