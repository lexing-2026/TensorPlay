"""GEMM performance comparison: TensorPlay vs torch.

Usage: python benchmark/gemm_perf.py [device]
"""
import sys
import time
import numpy as np

import torch
import tensorplay as tp

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"


def bench_torch(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_tp(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        import tensorplay._C as C
        C.synchronize() if hasattr(C, "synchronize") else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEVICE == "cuda":
        # sync via a tiny d2h copy
        _ = float(tp.zeros(1).cpu().numpy()[0]) if False else None
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def tflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def main():
    if DEVICE == "cuda":
        assert torch.cuda.is_available()

    rng = np.random.RandomState(0)

    if DEVICE == "cuda":
        shapes = [
            ("fp16 4096^3", np.float16, (4096, 4096), (4096, 4096)),
            ("fp16 2048x4096x11008", np.float16, (2048, 4096), (4096, 11008)),
            ("fp16 linear 4096x32000", np.float16, (4096, 4096), (4096, 32000)),
            ("bf16 4096^3", None, (4096, 4096), (4096, 4096)),  # special-cased below
            ("fp32 4096^3", np.float32, (4096, 4096), (4096, 4096)),
            ("fp32 512^3", np.float32, (512, 512), (512, 512)),
        ]
    else:
        shapes = [
            ("fp32 2048^3", np.float32, (2048, 2048), (2048, 2048)),
            ("fp32 512^3", np.float32, (512, 512), (512, 512)),
            ("fp64 1024^3", np.float64, (1024, 1024), (1024, 1024)),
        ]

    print(f"{'case':<28}{'torch ms':>10}{'tp ms':>10}{'t TFLOP/s':>11}{'p TFLOP/s':>11}{'ratio':>7}")
    for name, dt, sa, sb in shapes:
        m, k = sa
        k2, n = sb
        assert k == k2
        if name.startswith("bf16"):
            a = (rng.randn(m, k) * 0.1).astype(np.float32)
            b = (rng.randn(k, n) * 0.1).astype(np.float32)
            ta = tp.tensor(a).to(tp.device("cuda")).to(tp.bfloat16)
            tb = tp.tensor(b).to(tp.device("cuda")).to(tp.bfloat16)
            ha = torch.tensor(a, device="cuda").bfloat16()
            hb = torch.tensor(b, device="cuda").bfloat16()
        else:
            a = (rng.randn(m, k) * 0.1).astype(dt)
            b = (rng.randn(k, n) * 0.1).astype(dt)
            ta = tp.tensor(a).to(tp.device("cuda")) if DEVICE == "cuda" else tp.tensor(a)
            tb = tp.tensor(b).to(tp.device("cuda")) if DEVICE == "cuda" else tp.tensor(b)
            ha = torch.tensor(a, device=DEVICE)
            hb = torch.tensor(b, device=DEVICE)

        t_tp = bench_tp(lambda: ta @ tb)
        t_th = bench_torch(lambda: ha @ hb)
        print(f"{name:<28}{t_th:>10.3f}{t_tp:>10.3f}"
              f"{tflops(m, n, k, t_th):>11.2f}{tflops(m, n, k, t_tp):>11.2f}"
              f"{t_th / t_tp:>7.2f}")

    # batched matmul
    if DEVICE == "cuda":
        B, M, K, N = 12, 512, 512, 4096
        a = (rng.randn(B, M, K) * 0.1).astype(np.float16)
        b = (rng.randn(K, N) * 0.1).astype(np.float16)
        ta = tp.tensor(a).to(tp.device("cuda"))
        tb = tp.tensor(b).to(tp.device("cuda"))
        ha = torch.tensor(a, device="cuda")
        hb = torch.tensor(b, device="cuda")
        t_tp = bench_tp(lambda: ta @ tb)
        t_th = bench_torch(lambda: ha @ hb)
        fl = lambda ms: tflops(B * M, N, K, ms)
        print(f"{'fp16 bmm-bcast 12x512x512x4096':<28}{t_th:>10.3f}{t_tp:>10.3f}"
              f"{fl(t_th):>11.2f}{fl(t_tp):>11.2f}{t_th / t_tp:>7.2f}")


if __name__ == "__main__":
    main()
