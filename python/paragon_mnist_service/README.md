# 🧩 Paragon MNIST Microservice

**Dual-backend FastAPI service (CPU + GPU) for testing deterministic model parity.**

This service exposes a simple REST API that hosts 10 MNIST sample images, loads a Paragon model in both CPU and GPU modes, and verifies that both produce bit-identical results.  
It’s designed to serve as a reproducibility harness for cross-language or cross-platform testing of Paragon models.

---

## 🚀 Quick Start

```bash
git clone https://github.com/openfluke/paragon_ecosystem.git
cd python/paragon_mnist_service
pip install -e .
paragon-mnist-service
```

Once running:

- API root → [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Docs → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Hosted images → [http://127.0.0.1:8000/static/images/7.png](http://127.0.0.1:8000/static/images/7.png)

---

## 🧠 Endpoints

| Route                                      | Description                                               |
| ------------------------------------------ | --------------------------------------------------------- |
| **GET `/health`**                          | Returns `{ok, gpu_available}`                             |
| **GET `/images/list`**                     | Lists hosted PNGs (0–9)                                   |
| **GET `/predict?image=7.png&backend=gpu`** | Runs inference on one image (`cpu` or `gpu`)              |
| **POST `/predict`**                        | JSON body version: `{"image":"7.png","backend":"gpu"}`    |
| **GET `/predict-raw`**                     | Returns raw logits (pre-softmax) for numeric drift checks |
| **GET `/parity`**                          | Runs all 0–9 through both backends and reports parity     |
| **Static / Images**                        | Served from `/static/images/<digit>.png`                  |

---

## ⚙️ Local Parity Check

Use the included shell harness:

```bash
chmod +x test_parity.sh
./test_parity.sh
```

This script:

- checks `/health`
- lists hosted images
- runs `/parity`
- prints a table of per-image predictions
- shows probability vectors (`cpu.probs`, `gpu.probs`)
- computes drift metrics: `max_abs_diff`, `mean_abs_diff`, `l2_diff`
- writes `parity_report.json` and `parity_summary.csv`

Example output:

```
🎯 Perfect parity across 10 images!
max_abs_diff = 0.000000000
l2_diff      = 0.000000000
```

---

## 🧩 Swapping in a Trained Model

The first run creates a placeholder `mnist_paragon_model.json`.
To use a real trained model:

```bash
export MODEL_JSON="/absolute/path/to/trained/mnist_paragon_model.json"
paragon-mnist-service
```

The same model can be shared across different Paragon microservices (Go, C#, JS, etc.) for reproducibility tests.

---

## 🧪 Comparing Drift Across Languages

1. Deploy the same model and image set in multiple services.
2. Run `test_parity.sh` (or a language-specific variant) against each.
3. Compare CSVs — identical probabilities ⇒ zero drift.

---

## 🧰 Project Layout

```
paragon_mnist_service/
│
├── __init__.py
├── server.py              # FastAPI application
├── utils.py               # image + model utilities
│
├── test_parity.sh         # parity test harness
├── mnist_paragon_model.json
├── images/                # auto-populated digits 0–9
│
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📜 License

Apache-2.0 © OpenFluke

---

**TL;DR:**
Run the service → hit `/parity` → confirm `0.000000000` drift → you’ve proven deterministic reproducibility between CPU & GPU (and soon, between languages).
Perfect for testing Paragon’s cross-platform guarantees.

```

```
