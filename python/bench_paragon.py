#!/usr/bin/env python3
# bench_paragon.py — paragon-py benchmark, C/C#-style
import os, time, math, argparse
from typing import List, Tuple
import paragon_py as p

PRESETS = [
    ("S1",  [784,  64, 10]),
    ("S2",  [784, 128, 10]),
    ("S3",  [784, 256, 10]),
    ("M1",  [784, 256, 256, 10]),
    ("M2",  [784, 384, 384, 10]),
    ("M3",  [784, 512, 512, 10]),
    ("L1",  [784, 768, 768, 768, 10]),
    ("L2",  [784, 1024, 1024, 1024, 10]),
    ("XL1", [784, 1536, 1536, 1536, 1536, 10]),
    ("XL2", [784, 2048, 2048, 2048, 2048, 10]),
]

def shape_str(layers: List[int]) -> str:
    return " → ".join(str(x) for x in layers)

def estimate_vram_mb(layers: List[int]) -> float:
    params = 0
    for i in range(len(layers)-1):
        params += layers[i] * layers[i+1]
    params += sum(layers[1:])
    return params * 4.0 / (1024*1024)

def fixed_input_784(seed: int = 123) -> List[List[float]]:
    s = seed & 0xFFFFFFFF
    def nextu():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 0xFFFFFFFF
    vec = [round(nextu(), 6) for _ in range(784)]
    return [vec]  # [[784]]

def softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [e/s for e in ex]

def mae_max(a: List[float], b: List[float]) -> Tuple[float, float, int]:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0
    absdiffs = [abs(a[i] - b[i]) for i in range(n)]
    return sum(absdiffs)/n, max(absdiffs), n

def build_shapes(layers: List[int]) -> List[Tuple[int,int]]:
    return [(w,1) for w in layers]

def build_activations(layers: List[int]) -> List[str]:
    if len(layers) < 2:
        return ["linear"]
    acts = ["linear"]
    for _ in range(1, len(layers)-1): acts.append("relu")
    acts.append("softmax")
    return acts

def run_forward_timed(h, x_1xN) -> Tuple[float, List[float]]:
    t0 = time.perf_counter()
    p.forward(h, [x_1xN]) if isinstance(x_1xN[0], float) else p.forward(h, x_1xN)
    raw = p.extract_output(h)
    ms = (time.perf_counter() - t0) * 1000.0
    def flatten_any(z):
        out = []
        def walk(v):
            if isinstance(v, (list, tuple)):
                for c in v: walk(c)
            elif isinstance(v, (int, float)):
                out.append(float(v))
        walk(z)
        return out
    return ms, flatten_any(raw)

def csv_write_header(fp):
    fp.write("id,shape,estMB,cpu_ms,gpu_ms,speedup,mae,max,gpu_init_ms,adapter\n")

def csv_write_row(fp, rid, shape, estMB, cpu_ms, gpu_ms, speedup, mae, maxd, gpu_init_ms, adapter_txt):
    def f3(x): return f"{x:.3f}"
    def sci(x): return f"{x:.2E}"          # <-- fixed for Python
    def esc(s): return '"' + str(s).replace('"', '""') + '"'
    fp.write(",".join([
        rid,
        esc(shape),
        f"{estMB:.2f}",
        f3(cpu_ms), f3(gpu_ms),
        f"{speedup:.2f}",
        sci(mae), sci(maxd),
        f3(gpu_init_ms),
        esc(adapter_txt if adapter_txt is not None else "")
    ]) + "\n")

def run_case(case_id: str, layers: List[int], show_vectors: bool, csv_path: str | None):
    shp  = build_shapes(layers)
    acts = build_activations(layers)
    trainable = [True]*len(shp)
    x = fixed_input_784(123)  # [[784]]

    print(f"\n=== {case_id} ({shape_str(layers)}) ===")

    # One handle, GPU-capable. We'll do CPU first (pre-init), GPU second (post-init).
    h = p.new_network(shapes=shp, activations=acts, trainable=trainable, use_gpu=True)

    # (Optional) deterministically “kick” weights so they’re nontrivial and shared for both passes
    try:
        # paragon-py exposes this in recent builds; harmless no-op if missing.
        p.perturb_weights(h, amount=0.1, seed=42)
    except Exception:
        pass

    # -------- CPU pass (GPU not initialized yet) --------
    cpu_ms, cpu_vec = run_forward_timed(h, x[0])

    # -------- Initialize GPU on the SAME handle --------
    t0 = time.perf_counter()
    gpu_ok = False
    adapter_txt = None
    try:
        gpu_ok = bool(p.initialize_gpu(h))
    except Exception as e:
        adapter_txt = f"error:{e}"
    t_gpu_init = (time.perf_counter() - t0) * 1000.0
    if adapter_txt is None:
        adapter_txt = "[ok]" if gpu_ok else "[null]"
    # Warm-up
    try:
        p.forward(h, x); _ = p.extract_output(h)
    except Exception:
        pass

    # -------- GPU pass (same weights) --------
    gpu_ms, gpu_vec = run_forward_timed(h, x[0])

    mae, maxd, n = mae_max(cpu_vec, gpu_vec)
    speedup = (cpu_ms / gpu_ms) if gpu_ms > 0 else float("inf")

    print(f"GPU init: {adapter_txt}  in {t_gpu_init:.2f} ms  enabled={'yes' if gpu_ok else 'no'}")
    print(f"CPU  ⏱ {cpu_ms:.3f} ms")
    print(f"GPU  ⏱ {gpu_ms:.3f} ms")
    print(f"Speedup: {speedup:.2f}×")
    print(f"Δ(CPU vs GPU)  mae={mae:.2E}  max={maxd:.2E}  (n={n})")

    if show_vectors:
        def fmt(v): return "[" + ", ".join(f"{x:.6g}" for x in v) + "]"
        cpu_sm = softmax(cpu_vec)
        gpu_sm = softmax(gpu_vec)
        print("CPU ExtractOutput (raw):", fmt(cpu_vec[:10]), "..." if len(cpu_vec) > 10 else "")
        print("GPU ExtractOutput (raw):", fmt(gpu_vec[:10]), "..." if len(gpu_vec) > 10 else "")
        print("CPU softmax:", fmt(cpu_sm[:10]), "..." if len(cpu_sm) > 10 else "")
        print("GPU softmax:", fmt(gpu_sm[:10]), "..." if len(gpu_sm) > 10 else "")
        print("Idx |          CPU          |          GPU          | Δ")
        print("----+------------------------+------------------------+------------------")
        for i in range(min(10, n)):
            d = abs(cpu_vec[i] - gpu_vec[i])
            print(f"{i:3d} | {cpu_vec[i]:>22.14g} | {gpu_vec[i]:>22.14g} | {d:>16.6e}")

    if csv_path:
        import os
        header_needed = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as fp:
            if header_needed: csv_write_header(fp)
            csv_write_row(
                fp,
                rid=case_id,
                shape=shape_str(layers),
                estMB=estimate_vram_mb(layers),
                cpu_ms=cpu_ms,
                gpu_ms=gpu_ms,
                speedup=speedup,
                mae=mae,
                maxd=maxd,
                gpu_init_ms=t_gpu_init,
                adapter_txt=adapter_txt
            )

    try: p.cleanup_gpu(h)
    except Exception: pass


def main():
    ap = argparse.ArgumentParser(description="paragon-py benchmark (MNIST zoo)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-index vectors")
    ap.add_argument("--csv", type=str, default=None, help="write CSV to path")
    ap.add_argument("--backend", type=str, default=None, help="force WebGPU backend (vulkan|gl|metal|dx12)")
    args = ap.parse_args()

    if args.backend: os.environ["WGPU_BACKEND"] = args.backend

    print("Simple Paragon CPU vs GPU Benchmark (Python — paragon-py)")
    print("=========================================================")

    for cid, layers in PRESETS:
        run_case(cid, layers, show_vectors=not args.quiet, csv_path=args.csv)

if __name__ == "__main__":
    main()
