from .utils import (
    ensure_dir, autopopulate_images_folder, initialize_models,
    load_png_as_28x28, paragon_forward_probs, IMAGES_DIR
)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os

# ------------------------------------------------------------
# App + CORS
# ------------------------------------------------------------
app = FastAPI(title="MNIST Paragon Dual-Backend Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Images (populate & mount)
# ------------------------------------------------------------
ensure_dir(IMAGES_DIR)
autopopulate_images_folder()

# Mount static files at /static/images so /images/* is free for API routes
app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ------------------------------------------------------------
# Models (CPU + GPU)
# ------------------------------------------------------------
H_CPU, H_GPU, GPU_OK = initialize_models()

# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str
    backend: str = "gpu"  # 'gpu' | 'cpu'

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _list_images() -> List[str]:
    return sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".png")])

def _predict_core(image_name: str, backend: str):
    path = os.path.join(IMAGES_DIR, image_name)
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"image not found: {image_name}")

    backend = (backend or "gpu").lower()
    img = load_png_as_28x28(path)

    if backend == "gpu":
        if H_GPU is None:
            raise HTTPException(503, detail="GPU backend not available")
        res = paragon_forward_probs(H_GPU, img)
    else:
        res = paragon_forward_probs(H_CPU, img)

    return JSONResponse({
        "backend": backend,
        "image": image_name,
        "prediction": int(res["pred"]),
        "probabilities": res["probs"],
        "latency_sec": res["latency_sec"],
        "source_image_url": f"/static/images/{image_name}",
    })

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "MNIST service ready", "gpu_available": GPU_OK}

@app.get("/health")
def health():
    return {"ok": True, "gpu_available": GPU_OK}

@app.get("/images/list")
def images_list():
    return {"images": _list_images()}

@app.get("/predict")
def predict(image: str = Query(...), backend: str = Query("gpu")):
    return _predict_core(image, backend)

@app.post("/predict")
def predict_post(req: PredictRequest):
    return _predict_core(req.image, req.backend)

@app.get("/parity")
def parity(images: List[str] = Query(default=["0.png","1.png","2.png","3.png","4.png","5.png","6.png","7.png","8.png","9.png"])):
    results = []
    mismatches = 0
    for name in images:
        path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(path):
            results.append({"image": name, "error": "not found"})
            continue
        img = load_png_as_28x28(path)
        cpu = paragon_forward_probs(H_CPU, img)
        if H_GPU is None:
            results.append({"image": name, "cpu": cpu, "gpu": None, "match": None})
            continue
        gpu = paragon_forward_probs(H_GPU, img)
        match = int(cpu["pred"]) == int(gpu["pred"])
        mismatches += 0 if match else 1
        results.append({"image": name, "cpu": cpu, "gpu": gpu, "match": match})
    return {
        "gpu_available": GPU_OK,
        "mismatches": mismatches,
        "total": len(results),
        "results": results
    }

@app.get("/predict-raw")
def predict_raw(image: str = Query(...), backend: str = Query("gpu")):
    path = os.path.join(IMAGES_DIR, image)
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"image not found: {image}")
    img = load_png_as_28x28(path)
    if backend.lower() == "gpu":
        if H_GPU is None:
            raise HTTPException(503, detail="GPU backend not available")
        h = H_GPU
    else:
        h = H_CPU
    # forward + raw logits (last 10)
    import paragon_py as p
    p.forward(h, img)
    logits = p.extract_output(h)[-10:]
    return {"backend": backend, "image": image, "logits": logits}


def main():
    import uvicorn
    uvicorn.run("paragon_mnist_service.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
