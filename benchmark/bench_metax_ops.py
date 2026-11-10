#!/usr/bin/env python3
"""Cross-library operator benchmark for the MetaX backend.

Runs the same shapes through either library selected on the command line, so
the two numbers are comparable:

    python3 benchmark/bench_metax_ops.py torch
    python3 benchmark/bench_metax_ops.py tensorplay

Every case reports milliseconds per call and the derived rate (memory
bandwidth for elementwise work, FLOPs for GEMM).  Timings use device events
around a loop of calls after a warm-up, which is what a steady-state kernel
sees; the host-side launch cost is included because both libraries pay it.
"""

from __future__ import annotations

import argparse
import sys


def load(name: str):
    if name == "torch":
        import torch

        return torch
    import tensorplay

    return tensorplay


def synchronize(lib) -> None:
    if hasattr(lib, "cuda"):
        lib.cuda.synchronize()
    else:
        lib.accelerator.synchronize()


def time_calls(lib, fn, iters: int) -> float:
    start = lib.cuda.Event(enable_timing=True)
    end = lib.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    synchronize(lib)
    return start.elapsed_time(end) / iters


def report(name: str, ms: float, detail: str) -> None:
    print(f"{name:<28} {ms:>10.3f} ms   {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", choices=["torch", "tensorplay"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    lib = load(args.library)
    dev = args.device
    n = 16 * 1024 * 1024

    def make(*shape, dtype=None):
        return lib.randn(*shape, device=dev, dtype=dtype or lib.float32)

    cases = []

    a = make(n)
    b = make(n)
    out = make(n)

    for name, fn, nbytes in [
        ("add f32 (16M)", lambda: lib.add(a, b, out=out), 3 * 4 * n),
        ("mul f32 (16M)", lambda: lib.mul(a, b, out=out), 3 * 4 * n),
        ("relu f32 (16M)", lambda: lib.nn.functional.relu(a), 2 * 4 * n),
        ("sigmoid f32 (16M)", lambda: lib.sigmoid(a, out=out), 2 * 4 * n),
        ("exp f32 (16M)", lambda: lib.exp(a, out=out), 2 * 4 * n),
        ("sum f32 (16M)", lambda: lib.sum(a), 4 * n),
        ("softmax f32 (4096x4096)", None, 0),
        ("mm f32 4096", None, 0),
        ("mm f16 4096", None, 0),
    ]:
        cases.append((name, fn, nbytes))

    sm = make(4096, 4096)
    cases[6] = ("softmax f32 (4096x4096)", lambda: lib.softmax(sm, dim=-1), 2 * 4 * 4096 * 4096)

    m = make(4096, 4096)
    cases[7] = ("mm f32 4096", lambda: lib.mm(m, m), 0)
    m16 = m.to(lib.float16)
    cases[8] = ("mm f16 4096", lambda: lib.mm(m16, m16), 0)

    for name, fn, nbytes in cases:
        for _ in range(args.warmup):
            fn()
        synchronize(lib)
        ms = time_calls(lib, fn, args.iters)
        if "mm" in name:
            flops = 2 * 4096**3
            detail = f"{flops / (ms * 1e-3) / 1e12:8.2f} TFLOP/s"
        else:
            detail = f"{nbytes / (ms * 1e-3) / 1e9:8.2f} GB/s"
        report(name, ms, detail)

    return 0


if __name__ == "__main__":
    sys.exit(main())
