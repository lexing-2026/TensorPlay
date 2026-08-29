"""CPU complex elementwise benchmark.

The suite compares scalar complex reference operations with TensorPlay's
AVX2+libmvec split-lane kernels (cpu/VecComplex.h).

Usage: PYTHONPATH=. python3 benchmark/bench_complex_cpu.py [--dtype c128]
"""
import argparse
import statistics
import time

import numpy as np
import torch

import tensorplay as tp


def to_tp(x: np.ndarray) -> tp.Tensor:
    x = np.ascontiguousarray(x)
    return tp.tensor(np.stack([x.real.copy(), x.imag.copy()], -1)).view_as_complex()


def make(shape, seed, dtype):
    rng = np.random.RandomState(seed)
    re = rng.standard_normal(shape).astype(np.float64 if dtype == np.complex128 else np.float32)
    im = rng.standard_normal(shape).astype(np.float64 if dtype == np.complex128 else np.float32)
    return ((re * 0.4) + (im * 0.4j)).astype(dtype)


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)  # min is robust against concurrent-build interference


BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
]

UNARY_OPS = [
    ("exp", np.exp, lambda t: t.exp()),
    ("log", np.log, lambda t: t.log()),
    ("sqrt", np.sqrt, lambda t: t.sqrt()),
    ("sin", np.sin, lambda t: t.sin()),
    ("abs", None, lambda t: t.abs()),
    ("sum", None, lambda t: t.sum()),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0,
                    help="pin both frameworks to N intra-op threads")
    args = ap.parse_args()
    dtype = np.complex128 if args.dtype == "c128" else np.complex64
    if args.threads:
        torch.set_num_threads(args.threads)
        if hasattr(tp, "set_num_threads"):
            tp.set_num_threads(args.threads)

    print(f"ref {torch.__version__} threads={torch.get_num_threads()}  "
          f"dtype={args.dtype}")
    hdr = f"{'op':>6} {'n_complex':>10} {'tp_ms':>9} {'ref_ms':>9} {'speedup':>8}"
    print(hdr)
    slow = []
    for n in (4096, 1 << 16, 1 << 20, 1 << 22):
        x_np = make(n, 7, dtype)
        y_np = make(n, 8, dtype)
        x_th = torch.from_numpy(x_np.copy())
        y_th = torch.from_numpy(y_np.copy())
        x_tp = to_tp(x_np)
        y_tp = to_tp(y_np)

        for name, f in BINARY_OPS:
            if name == "div":
                y_np2 = np.where(y_np == 0, np.complex64(1), y_np).astype(dtype)
                y_th = torch.from_numpy(y_np2.copy())
                y_tp = to_tp(y_np2)
            t_torch = timeit(lambda: f(x_th, y_th), args.reps)
            t_tp = timeit(lambda: f(x_tp, y_tp), args.reps)
            sp = t_torch / t_tp
            print(f"{name:>6} {n:>10} {t_tp*1e3:>9.3f} {t_torch*1e3:>9.3f} {sp:>7.2f}x")
            if sp < 1.0:
                slow.append((name, n, sp))

        for name, np_op, tp_op in UNARY_OPS:
            if name == "sum":
                t_torch = timeit(lambda: float(x_th.sum().real) +
                                 float(x_th.sum().imag), args.reps)
                t_tp = timeit(lambda: x_tp.sum(), args.reps)
            elif name == "abs":
                t_torch = timeit(lambda: x_th.abs(), args.reps)
                t_tp = timeit(lambda: x_tp.abs(), args.reps)
            else:
                t_torch = timeit(lambda: getattr(x_th, name)(), args.reps)
                t_tp = timeit(lambda: tp_op(x_tp), args.reps)
            sp = t_torch / t_tp
            print(f"{name:>6} {n:>10} {t_tp*1e3:>9.3f} {t_torch*1e3:>9.3f} {sp:>7.2f}x")
            if sp < 1.0:
                slow.append((name, n, sp))

    if slow:
        print("\nBEHIND ref:")
        for name, n, sp in slow:
            print(f"  {name} @{n}: {sp:.2f}x")
    else:
        print("\nAll ops >= ref CPU")


if __name__ == "__main__":
    main()
