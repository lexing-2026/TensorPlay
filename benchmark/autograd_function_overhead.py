"""Autograd function framework-overhead benchmark.

Measures ns/iter of forward+backward through a custom Function minus the
bare-op floor, isolating each framework's apply()/graph-attach cost.

Usage:
    PYTHONPATH=/home/mega/TensorPlay python3 benchmark/autograd_function_overhead.py
"""

import argparse
import statistics
import sys
import time

import tensorplay as tp
import torch


def bench(fn, iters):
    fn()  # warmup
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter_ns() - t0) / iters)
    return best


def make_bare(mod):
    def bare(x):
        y = x * 2.0
        z = y.sum()
        z.backward()
        return z
    return bare


def make_function(mod, F):
    class Mul2(F):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0

        @staticmethod
        def backward(ctx, g):
            return g * 2.0

    if hasattr(mod, "cuda") and mod is torch:
        pass

    def via_function(x):
        out = Mul2.apply(x)
        out.sum().backward()
        return out

    return via_function


def _leaf(mod, n=8):
    x = mod.randn(n) if hasattr(mod, "randn") else mod.zeros(n)
    x.requires_grad_(True)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    results = {}
    for label, mod in (("torch", torch), ("tensorplay", tp)):
        bare = make_bare(mod)
        func = make_function(mod, mod.autograd.Function)
        b = bench(lambda: bare(_leaf(mod)), args.iters)
        f = bench(lambda: func(_leaf(mod)), args.iters)
        results[label] = (b, f, f - b)

    # keep grads from accumulating across iterations (correctness of timing)
    print(f"{'metric':<28}{'ref':>12}{'tensorplay':>14}")
    rows = [
        ("bare fwd+bwd (ns)", *[(results[k][0]) for k in ("torch", "tensorplay")]),
        ("Function fwd+bwd (ns)", *[(results[k][1]) for k in ("torch", "tensorplay")]),
        ("framework loss (ns)", *[(results[k][2]) for k in ("torch", "tensorplay")]),
    ]
    for name, tv, tpv in rows:
        print(f"{name:<28}{tv:>12,.0f}{tpv:>14,.0f}")
    ratio = results["torch"][2] / results["tensorplay"][2]
    print(f"\nframework-loss ratio ref/tensorplay: {ratio:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
