import { PNG } from "pngjs";
import * as fs from "fs";
import * as crypto from "crypto";

export type Tensor2D = number[][];

function toGrayFloat(r: number, g: number, b: number): number {
  // ITU-R BT.601
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

function bilinear28x28Gray(
  src: Float32Array,
  w: number,
  h: number
): Float32Array {
  // src: grayscale float32, row-major length w*h
  const targetW = 28,
    targetH = 28;
  const dst = new Float32Array(targetW * targetH);
  const sx = w / targetW;
  const sy = h / targetH;

  for (let y = 0; y < targetH; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(h - 1, y0 + 1);
    const wy = fy - y0;

    for (let x = 0; x < targetW; x++) {
      const fx = (x + 0.5) * sx - 0.5;
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(w - 1, x0 + 1);
      const wx = fx - x0;

      const i00 = y0 * w + x0;
      const i10 = y0 * w + x1;
      const i01 = y1 * w + x0;
      const i11 = y1 * w + x1;

      const a = src[i00];
      const b = src[i10];
      const c = src[i01];
      const d = src[i11];

      // bilinear interpolate
      const top = a + (b - a) * wx;
      const bot = c + (d - c) * wx;
      dst[y * targetW + x] = top + (bot - top) * wy;
    }
  }
  return dst;
}

export function pngToMNISTTensor28x28(pngBuffer: Buffer): {
  tensor: Tensor2D;
  flatF32: Float32Array; // row-major length 784
  sha256: string;
} {
  const png = PNG.sync.read(pngBuffer); // RGBA, un-premultiplied
  const { width: W, height: H, data } = png;

  // 1) RGBA -> grayscale float32 0..255
  const gray = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const r = data[i + 0];
      const g = data[i + 1];
      const b = data[i + 2];
      // ignore alpha; MNIST digits are opaque
      gray[y * W + x] = toGrayFloat(r, g, b);
    }
  }

  // 2) Resize if needed (bilinear)
  const g28 = W === 28 && H === 28 ? gray : bilinear28x28Gray(gray, W, H);

  // 3) Normalize 0..1
  for (let i = 0; i < g28.length; i++) g28[i] = g28[i] / 255.0;

  // 4) Make 2D view for Portal input
  const tensor: Tensor2D = Array.from({ length: 28 }, (_, y) => {
    const row = new Array<number>(28);
    const base = y * 28;
    for (let x = 0; x < 28; x++) row[x] = g28[base + x];
    return row;
  });

  // 5) Hash the raw float32 bytes (for parity checks)
  const sha256 = crypto
    .createHash("sha256")
    .update(Buffer.from(g28.buffer, g28.byteOffset, g28.byteLength))
    .digest("hex");

  return { tensor, flatF32: g28, sha256 };
}

// Convenience: file path -> tensor
export function loadMNISTTensorFromPNG(path: string) {
  const buf = fs.readFileSync(path);
  return pngToMNISTTensor28x28(buf);
}
