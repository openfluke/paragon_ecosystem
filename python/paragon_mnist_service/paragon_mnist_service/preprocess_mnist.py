from PIL import Image
import numpy as np
import hashlib

def png_to_mnist_tensor_28x28(png_path: str):
    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    a = np.asarray(img, dtype=np.float32)  # [H, W, 3]
    r, g, b = a[...,0], a[...,1], a[...,2]
    gray = 0.299*r + 0.587*g + 0.114*b     # float32 0..255

    if (W, H) != (28, 28):
        # Our own bilinear to match Node
        x = np.linspace(0.5, W - 0.5, 28, dtype=np.float32)
        y = np.linspace(0.5, H - 0.5, 28, dtype=np.float32)
        x = (x * (W / 28.0)) - 0.5
        y = (y * (H / 28.0)) - 0.5

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        x0 = np.clip(x0, 0, W - 1)
        y0 = np.clip(y0, 0, H - 1)

        wx = (x - x0).astype(np.float32)
        wy = (y - y0).astype(np.float32)

        out = np.empty((28,28), dtype=np.float32)
        for j in range(28):
            y_0, y_1, wyj = y0[j], y1[j], wy[j]
            top = gray[y_0, x0] + (gray[y_0, x1] - gray[y_0, x0]) * wx
            bot = gray[y_1, x0] + (gray[y_1, x1] - gray[y_1, x0]) * wx
            out[j,:] = top + (bot - top) * wyj
        gray = out
    else:
        gray = gray

    gray = (gray / 255.0).astype(np.float32)  # [28,28]
    sha256 = hashlib.sha256(gray.tobytes()).hexdigest()
    return gray, sha256
