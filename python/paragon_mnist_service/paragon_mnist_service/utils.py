# paragon_mnist_service/utils.py
import os, gzip, struct, urllib.request, json, time, math
from typing import List
from PIL import Image

import paragon_py as p
from paragon_py import utils as U

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
IMAGES_DIR = os.environ.get("IMAGES_DIR", "./images")
MODEL_JSON = os.environ.get("MODEL_JSON", "./mnist_paragon_model.json")
MNIST_DIR = "./mnist_idx"

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}

# ------------------------------------------------------------
# File & download helpers
# ------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_file(url: str, out_path: str):
    if not os.path.exists(out_path):
        print(f"→ Downloading {url}")
        urllib.request.urlretrieve(url, out_path)

# ------------------------------------------------------------
# MNIST parsing & sample image creation
# ------------------------------------------------------------
def read_images_2d(gz_path: str) -> List[List[List[float]]]:
    with gzip.open(gz_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic for images: {magic}"
        buf = f.read(rows * cols * num)
        images = []
        for i in range(num):
            offs = i * rows * cols
            img = [[buf[offs + r * cols + c] / 255.0 for c in range(cols)] for r in range(rows)]
            images.append(img)
        return images

def read_labels(gz_path: str) -> List[int]:
    with gzip.open(gz_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic for labels: {magic}"
        labs = f.read(num)
        return [int(b) for b in labs]

def autopopulate_images_folder():
    """Create ./images/0.png..9.png from MNIST if not already present."""
    ensure_dir(IMAGES_DIR)
    if any(f.lower().endswith(".png") for f in os.listdir(IMAGES_DIR)):
        return

    print("🧩 Populating MNIST samples into ./images ...")
    ensure_dir(MNIST_DIR)
    for _, url in URLS.items():
        out = os.path.join(MNIST_DIR, os.path.basename(url))
        download_file(url, out)

    imgs = read_images_2d(os.path.join(MNIST_DIR, os.path.basename(URLS["train_images"])))
    labels = read_labels(os.path.join(MNIST_DIR, os.path.basename(URLS["train_labels"])))

    seen = set()
    for img, label in zip(imgs, labels):
        if label in seen:
            continue
        pil = Image.new("L", (28, 28))
        flat = [int(px * 255) for row in img for px in row]
        pil.putdata(flat)
        pil.save(os.path.join(IMAGES_DIR, f"{label}.png"))
        seen.add(label)
        if len(seen) == 10:
            break

    print("✅ Populated ./images with digits 0–9")

def load_png_as_28x28(path: str) -> List[List[float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    im = Image.open(path).convert("L").resize((28, 28))
    arr = [v / 255.0 for v in list(im.getdata())]
    return [arr[i * 28:(i + 1) * 28] for i in range(28)]

# ------------------------------------------------------------
# Paragon model helpers
# ------------------------------------------------------------
def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def call_json(handle: int, name: bytes, obj):
    raw = U.CALL(int(handle), name, json.dumps(obj).encode("utf-8"))
    return U._steal(raw or b"")

def save_checkpoint(handle: int, path: str):
    txt = call_json(handle, b"SaveJSON", [path])
    if txt and "error" in txt.lower():
        raise RuntimeError(f"SaveJSON error: {txt}")

def load_checkpoint_to_handle(h: int, path: str):
    txt = call_json(h, b"LoadJSON", [path])
    if txt and "error" in txt.lower():
        raise RuntimeError(txt)

def make_handle(path: str) -> int:
    # Minimal stub; we'll LoadJSON to replace it.
    h = p.new_network(shapes=[(1, 1)], activations=["linear"], trainable=[True], use_gpu=False)
    load_checkpoint_to_handle(h, path)
    return h

# ------------------------------------------------------------
# Automatic model creation (if missing)
# ------------------------------------------------------------
def create_default_model_if_missing():
    """
    Create a minimal Paragon MNIST model JSON if it doesn't exist.
    Always builds with use_gpu=False; GPU is enabled later via initialize_gpu().
    """
    if os.path.exists(MODEL_JSON):
        return MODEL_JSON

    print(f"⚙️ No model found at {MODEL_JSON}. Creating a minimal placeholder...")
    shapes = [(28, 28), (256, 1), (10, 1)]
    activs = ["linear", "relu", "softmax"]
    trainable = [True, True, True]

    h = p.new_network(shapes=shapes, activations=activs, trainable=trainable, use_gpu=False)

    # Try to save via Paragon's SaveJSON
    try:
        save_checkpoint(h, MODEL_JSON)
    except Exception as e:
        # Fallback to a minimal JSON stub if SaveJSON isn't available
        with open(MODEL_JSON, "w") as f:
            json.dump({
                "metadata": {
                    "created_by": "paragon_mnist_service",
                    "description": f"Auto-generated placeholder model (SaveJSON failed: {e})"
                },
                "shapes": shapes,
                "activations": activs,
                "trainable": trainable
            }, f, indent=2)

    print(f"✅ Created placeholder model at {MODEL_JSON}")
    return MODEL_JSON

# ------------------------------------------------------------
# Initialization & forward pass
# ------------------------------------------------------------
def initialize_models():
    """
    Initialize CPU and GPU handles from MODEL_JSON (creating it if missing).
    """
    create_default_model_if_missing()

    h_cpu = make_handle(MODEL_JSON)

    # Make a separate handle for GPU; enable GPU after loading.
    h_gpu = make_handle(MODEL_JSON)
    gpu_ok = False
    try:
        gpu_ok = bool(p.initialize_gpu(h_gpu))
        if gpu_ok:
            print("🚀 GPU initialized successfully.")
        else:
            print("⚠️ GPU not available; CPU-only mode.")
    except Exception as e:
        print(f"⚠️ GPU init failed: {e}")

    return h_cpu, (h_gpu if gpu_ok else None), gpu_ok

def paragon_forward_probs(handle: int, img_28x28: List[List[float]]) -> dict:
    """
    Run forward inference on one MNIST image.
    Returns: {'pred': int, 'probs': [10 floats], 'latency_sec': float}
    """
    t0 = time.time()
    p.forward(handle, img_28x28)
    logits = p.extract_output(handle)[-10:]  # last 10 are class logits
    probs = softmax(logits)
    pred = max(range(10), key=lambda i: probs[i])
    return {"pred": pred, "probs": probs, "latency_sec": round(time.time() - t0, 6)}
