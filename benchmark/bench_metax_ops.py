#!/usr/bin/env python3
"""Cross-library operator benchmark for the MetaX backend.

Runs the same shapes through either library selected on the command line, so
the two numbers are comparable:

    python3 benchmark/bench_metax_ops.py torch
    python3 benchmark/bench_metax_ops.py tensorplay

Every case reports milliseconds per call and the derived rate (memory
bandwidth for elementwise work, FLOPs for GEMM and convolution).  Timings use
device events around a loop of calls after a warm-up, which is what a
steady-state kernel sees; the host-side launch cost is included because both
libraries pay it.

On a MetaX host without a torch build the same script measures the CPU
backend (``--device cpu``), which gives a device-to-host ratio from one
reproducible entry point.

Families covered: elementwise, reduction, activation, BLAS, convolution and
attention.
"""

from __future__ import annotations

import argparse
import sys
import time


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


def time_calls(lib, fn, iters: int, device: str) -> float:
    if device == "cpu":
        # No device events on the host path: wall clock around the same loop.
        for _ in range(iters):
            fn()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - start) / iters * 1e3
    start = lib.cuda.Event(enable_timing=True)
    end = lib.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    synchronize(lib)
    return start.elapsed_time(end) / iters


def report(name: str, ms: float, detail: str) -> None:
    print(f"{name:<34} {ms:>10.3f} ms   {detail}")


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

    # (name, family, callable, bytes moved, flops performed)
    cases: list[tuple[str, str, object, int, int]] = []

    a = make(n)
    b = make(n)
    out = make(n)
    pos = a.abs() + 1.0

    # -- elementwise ------------------------------------------------------
    cases += [
        ("add f32 (16M)", "elementwise", lambda: lib.add(a, b, out=out), 3 * 4 * n, 0),
        ("mul f32 (16M)", "elementwise", lambda: lib.mul(a, b, out=out), 3 * 4 * n, 0),
        ("div f32 (16M)", "elementwise", lambda: lib.div(a, pos, out=out), 3 * 4 * n, 0),
        ("relu f32 (16M)", "elementwise", lambda: lib.relu(a, out=out), 2 * 4 * n, 0),
        ("exp f32 (16M)", "elementwise", lambda: lib.exp(a, out=out), 2 * 4 * n, 0),
        ("log f32 (16M)", "elementwise", lambda: lib.log(pos, out=out), 2 * 4 * n, 0),
    ]

    # -- reduction --------------------------------------------------------
    cases += [
        ("sum f32 (16M)", "reduction", lambda: lib.sum(a), 4 * n, 0),
        ("mean f32 (16M)", "reduction", lambda: lib.mean(a), 4 * n, 0),
        ("max f32 (16M)", "reduction", lambda: lib.max(a), 4 * n, 0),
        ("min f32 (16M)", "reduction", lambda: lib.min(a), 4 * n, 0),
    ]

    # -- activation -------------------------------------------------------
    cases += [
        ("sigmoid f32 (16M)", "activation", lambda: lib.sigmoid(a, out=out), 2 * 4 * n, 0),
        ("tanh f32 (16M)", "activation", lambda: lib.tanh(a, out=out), 2 * 4 * n, 0),
        ("gelu f32 (16M)", "activation", lambda: lib.nn.functional.gelu(a), 2 * 4 * n, 0),
        ("silu f32 (16M)", "activation", lambda: lib.nn.functional.silu(a), 2 * 4 * n, 0),
    ]

    # -- softmax ----------------------------------------------------------
    sm = make(4096, 4096)
    cases.append(
        ("softmax f32 (4096x4096)", "reduction",
         lambda: lib.softmax(sm, dim=-1), 2 * 4 * 4096 * 4096, 0))

    # -- BLAS -------------------------------------------------------------
    m = make(4096, 4096)
    m16 = m.to(lib.float16)
    cases += [
        ("mm f32 4096", "blas", lambda: lib.mm(m, m), 0, 2 * 4096**3),
        ("mm f16 4096", "blas", lambda: lib.mm(m16, m16), 0, 2 * 4096**3),
    ]
    bm1 = make(8, 1024, 1024)
    bm2 = make(8, 1024, 1024)
    cases.append(("bmm f32 8x1024x1024", "blas", lambda: lib.bmm(bm1, bm2), 0,
                  2 * 8 * 1024**3))

    # -- convolution ------------------------------------------------------
    cx = make(8, 64, 56, 56)
    cw = make(64, 64, 3, 3)
    conv_flops = 2 * 8 * 64 * 64 * 56 * 56 * 9
    cases.append(("conv2d f32 8x64x56x56 k3", "convolution",
                  lambda: lib.nn.functional.conv2d(cx, cw, padding=1), 0, conv_flops))

    # -- attention --------------------------------------------------------
    q = make(2, 8, 1024, 64)
    k = make(2, 8, 1024, 64)
    v = make(2, 8, 1024, 64)
    # qk^T and pv: 2 * (B*H*S*S*D) each.
    sdpa_flops = 4 * 2 * 8 * 1024 * 1024 * 64
    cases.append(("sdpa f32 2x8x1024x64", "attention",
                  lambda: lib.nn.functional.scaled_dot_product_attention(q, k, v),
                  0, sdpa_flops))

    for name, family, fn, nbytes, flops in cases:
        for _ in range(args.warmup):
            fn()
        synchronize(lib)
        ms = time_calls(lib, fn, args.iters, dev)
        if flops:
            detail = f"{flops / (ms * 1e-3) / 1e12:8.2f} TFLOP/s  [{family}]"
        else:
            detail = f"{nbytes / (ms * 1e-3) / 1e9:8.2f} GB/s     [{family}]"
        report(name, ms, detail)

    return 0


if __name__ == "__main__":
    sys.exit(main())
