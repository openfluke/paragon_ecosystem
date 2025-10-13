# Paragon-Py Notebooks: Proofs of Concept and Framework Demos

## Overview

This repository contains Jupyter notebooks demonstrating the core capabilities of **Paragon-Py**, a deterministic dense neural network framework designed for cross-vendor reproducibility across CPUs and GPUs (NVIDIA, AMD, Intel, Apple, Qualcomm). The notebooks serve as proofs of concept (POCs) for key benchmarks outlined in the [Paragon Capability & Benchmark Expansion Report](benchmark_expansion_report_targets.docx), focusing initially on vision tasks like MNIST while showcasing the framework's training, inference, and evaluation pipelines.

Paragon-Py emphasizes numerical determinism (to 10⁻⁸ precision), scalability, and integration with the ISO Telemetry Harness for logging metrics such as accuracy, drift, and runtime. These notebooks provide reproducible examples to validate baseline performance in the "Green Zone" (Levels 1–4) of the capability grid, paving the way for expansion into more complex domains.

**Installation Prerequisites:**

- Python 3.10+ with NumPy and Matplotlib (via Conda or pip).
- For GPU support: Ensure WebGPU-compatible drivers (e.g., Vulkan, Metal, or DirectX 12).
- Install the framework: `pip install paragon-py` (v0.0.3 or later).

**Environment Setup:**

- Set `WGPU_BACKEND` to your preferred backend (e.g., `gl`, `vulkan`, `metal`, `dx12`).
- For WebGPU experiments, ensure Emscripten SDK is sourced if compiling to WASM.

## Notebook Sections

### 1. Quickstart + Training Demo (GPU-Accelerated)

This notebook introduces the end-to-end workflow for building, training, and evaluating a multi-layer perceptron (MLP) on a synthetic nonlinear classification task (circle-based decision boundary, akin to XOR with padding to 4D inputs).

- **Key Demonstrations:**

  - Network creation: A 3-layer MLP (4→8→8→2) with ReLU activations and trainable parameters.
  - GPU initialization and fallback to CPU.
  - Dataset generation: 512 training + 256 test samples with one-hot encoded targets.
  - Training: 50 epochs (two phases: lr=0.05 then lr=0.02) using the built-in `train` function, with shuffling for stochasticity.
  - Evaluation: Forward passes with softmax for probabilities, accuracy computation (>76% test accuracy in demo run).
  - Confidence Calibration: Bucketing predictions by max probability to assess model calibration (e.g., high-confidence bins show low error rates).
  - Inference: Single-sample predictions on edge cases.

- **Metrics Logged (via Telemetry Harness Integration):**

  - Accuracy: Train ~83%, Test ~76%.
  - Confidence Buckets: Distribution of prediction certainties, highlighting calibration gaps for future regularization.

- **Purpose:** Proves deterministic forward/backward passes on GPU, aligning with Level 2 (MNIST) benchmarks. Outputs are hashed for reproducibility.

### 2. CPU-Only Variant

A lightweight counterpart to the GPU demo, using a simplified 3-layer network (4→8→2) on a separable toy dataset (linear combination threshold for binary classification).

- **Key Demonstrations:**

  - CPU-exclusive mode for portability testing.
  - Dataset: 512 samples with random 4D features and binary targets.
  - Training: 10 epochs with lr=0.05; early stopping on negative loss (indicating convergence or numerical edge cases).
  - Inference: Raw logit extraction post-forward pass.

- **Purpose:** Validates cross-device parity (CPU vs. GPU) in the Meta/Verification domain (Level 1), with MAE drift <1e-8. Useful for debugging without hardware dependencies.

### 3. MNIST Proof of Concept (POC)

This dedicated notebook implements the flagship baseline task from the capability grid (Vision Level 2: MNIST).

- **Key Demonstrations:**

  - Data loading: 60k grayscale 28×28 images flattened to 784D inputs.
  - Model: 784→128→10 dense layers with ReLU and softmax.
  - Training: Full-dataset epochs targeting >97% accuracy.
  - Evaluation: Parity checks against reference models, including drift analysis.
  - Telemetry: JSON reports for accuracy, init time, and memory usage.

- **Purpose:** Establishes "Green Zone" feasibility, generating reports for the OpenFluke Conformance Lab. Includes visualizations of loss curves and confusion matrices.

## Framework Status

Paragon-Py is currently robust for baseline dense network tasks, with full support for multi-layer perceptrons (MLPs) using ReLU, linear, and softmax activations. Key achievements include:

- **Core Functionality:** Deterministic training and inference on GPUs and CPUs, with automatic fallback and cross-device parity validation (MAE <1e-8). Supports shapes like 784→128→10 for flattened image inputs.
- **Dataset Handling:** Modular loaders for synthetic data, tabular sets (e.g., Iris, XOR), and vision benchmarks (MNIST, Fashion-MNIST), generating one-hot targets and padded features as needed.
- **Evaluation Pipeline:** Built-in metrics for accuracy, MSE, and confidence bucketing; softmax integration for probabilistic outputs; telemetry logging for drift, runtime, and memory.
- **Reproducibility Tools:** Fixed seeding, output hashing, and early stopping to handle numerical stability.

The framework's progress is tracked against the capability grid, with completed levels marked as ✅ (fully implemented and benchmarked), 🟩 (feasible and tested), 🟨 (partial support), and ⬜ (planned). Current coverage focuses on Levels 1–3, with expansions underway.

| Domain                      | 1                      | 2                     | 3                        | 4                       | 5                       | 6                 | 7                        | 8                   | 9                    | 10                       |
| --------------------------- | ---------------------- | --------------------- | ------------------------ | ----------------------- | ----------------------- | ----------------- | ------------------------ | ------------------- | -------------------- | ------------------------ |
| **Vision / Image**          | ✅ Binary shapes       | ✅ MNIST              | ✅ Fashion-MNIST         | 🟨 CIFAR-10 (flattened) | ⬜ CIFAR-100            | ⬜ Conv surrogate | ⬜ ImageNet subset       | ⬜ Object detection | ⬜ Diffusion         | ⬜ Scene reasoning       |
| **Text / NLP**              | 🟩 Bag-of-words        | ⬜ Spam/Ham           | ⬜ TF-IDF topics         | ⬜ Word embeddings      | ⬜ Char-RNN             | ⬜ Seq2Seq        | ⬜ Transformer-lite      | ⬜ Summarization    | ⬜ GPT-scale         | ⬜ Dialogue reasoning    |
| **Tabular / Structured**    | ✅ XOR / Iris          | ✅ Titanic            | 🟨 Adult Income          | ⬜ Credit scoring       | ⬜ Regression (Housing) | ⬜ Time-series    | ⬜ Multi-task regression | ⬜ Causal graphs    | ⬜ Policy prediction | ⬜ Bayesian hybrid       |
| **Audio / Signal**          | 🟩 Sine classification | ⬜ FFT features       | ⬜ MFCC keyword          | ⬜ Spoken digits        | ⬜ ESC-10               | ⬜ Speech yes/no  | ⬜ Keyword spotting      | ⬜ ASR small-vocab  | ⬜ Full ASR          | ⬜ Music generation      |
| **Reinforcement Learning**  | 🟩 Gridworld           | ⬜ CartPole           | ⬜ MountainCar           | ⬜ LunarLander          | ⬜ Continuous control   | ⬜ Actor-Critic   | ⬜ Atari                 | ⬜ 3D biped         | ⬜ Robotics arms     | ⬜ Multi-agent           |
| **Physics / Simulation**    | 🟩 Linear motion       | 🟩 Pendulum           | ⬜ Projectile regression | ⬜ Collision prediction | ⬜ Double pendulum      | ⬜ Noisy control  | ⬜ 3D rigid bodies       | ⬜ Fluids           | ⬜ Real-time agents  | ⬜ Multi-body learning   |
| **Generative / Creativity** | 🟩 Dense autoencoder   | ⬜ Latent compression | ⬜ Variational AE        | ⬜ Toy GAN              | ⬜ Conditional GAN      | ⬜ Diffusion 2D   | ⬜ Image2Image           | ⬜ Text2Image       | ⬜ Video synthesis   | ⬜ World models          |
| **Meta / Verification**     | ✅ CPU↔GPU parity      | ✅ Device drift       | 🟨 FP16 drift            | ⬜ WebGPU vs Vulkan     | ⬜ Quantization         | ⬜ Kernel tuning  | ⬜ Compile-time optim.   | ⬜ RL-driven opt.   | ⬜ Conformance cloud | ⬜ Self-adapting runtime |

All notebooks use fixed seeds (e.g., `random.seed(7)`) for reproducibility and can run on Linux/macOS with EmSDK for WASM experiments.

## Future Aims

Looking ahead, Paragon-Py will expand to probe the limits of dense architectures in more challenging regimes, guided by the [Capability Progression Grid](example_scale.docx). Aims include:

- **Dataset and Model Extensions:** Broader loaders for NLP (e.g., IMDB reviews, AG News) and audio (e.g., ESC-10 with MFCC features); JSON-based model zoo for automated specs like CIFAR-10 (3072→512→10).
- **Advanced Metrics and Tasks:** ROC-AUC for imbalanced classification, average rewards for RL environments (e.g., CartPole targeting 200+ scores), and reconstruction error for autoencoders (<0.05 on MNIST).
- **Optimization and Portability:** FP16 quantization, kernel tuning for runtime improvements, and recurrent/sequential support (e.g., Char-RNN for text Levels 4–6).
- **Generative and Simulation Support:** VAE/GAN hooks for latent interpolation and toy physics tasks (e.g., pendulum regression with Lyapunov error tracking).
- **Architectural Hooks:** Surrogates for convolutions and attention to enable higher-level benchmarks like ImageNet subsets or Seq2Seq translation.

These developments will build a comprehensive "Capability Atlas" via telemetry-driven iteration, supporting the OpenFluke Conformance Lab v1.0 certification. Contributions are welcome for new loaders or verification tests.

## Usage Tips

- Run notebooks in Jupyter Lab for interactive plotting.
- For production: Export reports via `POST /reports/` endpoint.
- Troubleshooting: Check `model.drift` logs; fallback to CPU with `use_gpu=False`.

---

_Last Updated: October 13, 2025_  
_Project: OpenFluke / Paragon ISO Telemetry Harness_
