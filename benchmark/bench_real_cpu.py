"""

Usage:
  taskset -c 8-15 env PYTHONPATH=. python3 benchmark/bench_real_cpu.py --dtype f32
"""
import argparse
import time

import numpy as np
import torch

import tensorplay as tp


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
]
UNARY_OPS = ["exp", "log", "sqrt", "sin", "abs", "sum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    dtype = np.float64 if args.dtype == "f64" else np.float32
    if args.threads:
        torch.set_num_threads(args.threads)
        if hasattr(tp, "set_num_threads"):
            tp.set_num_threads(args.threads)

    print(f"ref {torch.__version__} threads={torch.get_num_threads()}  "
          f"dtype={args.dtype}")
    print(f"{'op':>6} {'n':>10} {'tp_ms':>9} {'ref_ms':>9} {'speedup':>8}")
    behind = []
    for n in (4096, 1 << 16, 1 << 20, 1 << 22):
        rng = np.random.RandomState(11)
        x_np = (rng.standard_normal(n) * 0.5).astype(dtype)
        y_np = (rng.standard_normal(n) * 0.5 + 1.0).astype(dtype)
        x_th = torch.from_numpy(x_np.copy())
        y_th = torch.from_numpy(y_np.copy())
        x_tp = tp.tensor(x_np.copy())
        y_tp = tp.tensor(y_np.copy())

        for name, f in OPS:
            t_torch = timeit(lambda: f(x_th, y_th), args.reps)
            t_tp = timeit(lambda: f(x_tp, y_tp), args.reps)
            sp = t_torch / t_tp
            print(f"{name:>6} {n:>10} {t_tp*1e3:>9.3f} {t_torch*1e3:>9.3f} {sp:>7.2f}x")
            if sp < 1.0:
                behind.append((name, n, sp))
        for name in UNARY_OPS:
            if name == "sum":
                t_torch = timeit(lambda: float(x_th.sum()), args.reps)
                t_tp = timeit(lambda: x_tp.sum(), args.reps)
            else:
                t_torch = timeit(lambda: getattr(x_th, name)(), args.reps)
                t_tp = timeit(lambda: getattr(x_tp, name)(), args.reps)
            sp = t_torch / t_tp
            print(f"{name:>6} {n:>10} {t_tp*1e3:>9.3f} {t_torch*1e3:>9.3f} {sp:>7.2f}x")
            if sp < 1.0:
                behind.append((name, n, sp))

    if behind:
        print("\nBEHIND ref:")
        for name, n, sp in behind:
            print(f"  {name} @{n}: {sp:.2f}x")
    else:
        print("\nAll ops >= ref CPU")


if __name__ == "__main__":
    main()
