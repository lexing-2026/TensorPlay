#!/usr/bin/env python3
"""Latency-vs-size curves for representative operators.

Each curve sweeps a problem size (square matmul, elementwise/reduction
vector length, conv resolution, softmax rows) and records, per point, the
best-of-N latency plus FLOP and streaming-byte counts. The on-site chart
turns these into throughput curves whose shape shows where an operator
transitions from cache-resident to bandwidth-bound to compute-bound.

Usage: bench_scaling_curves.py --threads 2 --json-out scaling.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tensorplay as tp
import tensorplay.functional as F


def _time(fn, reps):
    fn()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return min(samples)


def _count_flops(fn):
    try:
        from tensorplay import profiler as prof

        with prof.profile(activities=[prof.ProfilerActivity.CPU],
                          with_flops=True) as session:
            fn()
        return int(sum(event.flops for event in session.key_averages()))
    except Exception:
        return 0


def _curve_mm(sizes, reps):
    points = []
    for n in sizes:
        a = tp.tensor(np.random.default_rng(n).standard_normal((n, n)) * 0.1, dtype=tp.float32)
        b = tp.tensor(np.random.default_rng(n + 1).standard_normal((n, n)) * 0.1, dtype=tp.float32)
        fn = lambda: F.mm(a, b)
        seconds = _time(fn, reps)
        flops = _count_flops(fn)
        points.append({"n": n, "seconds": seconds, "flops": flops,
                       "bytes_moved": int(3 * n * n * 4)})
        del a, b
    return {"name": "mm", "label": "matmul (N×N)", "category": "matmul",
            "metric": "flops", "points": points}


def _curve_elementwise(sizes, reps):
    points = []
    for n in sizes:
        a = tp.tensor(np.random.default_rng(n).standard_normal((n,)) * 0.5, dtype=tp.float32)
        fn = lambda: F.relu(a)
        seconds = _time(fn, reps)
        points.append({"n": n, "seconds": seconds, "flops": 0,
                       "bytes_moved": int(2 * n * 4)})
        del a
    return {"name": "relu", "label": "relu (n)", "category": "unary",
            "metric": "bytes", "points": points}


def _curve_reduction(sizes, reps):
    points = []
    for n in sizes:
        a = tp.tensor(np.random.default_rng(n).standard_normal((n,)) * 0.5, dtype=tp.float32)
        fn = lambda: F.sum(a)
        seconds = _time(fn, reps)
        points.append({"n": n, "seconds": seconds, "flops": 0,
                       "bytes_moved": int(n * 4)})
        del a
    return {"name": "sum", "label": "sum (n)", "category": "reduction",
            "metric": "bytes", "points": points}


def _curve_conv(hw_sizes, reps):
    points = []
    for hw in hw_sizes:
        a = tp.tensor(np.random.default_rng(hw).standard_normal((8, 64, hw, hw)) * 0.1, dtype=tp.float32)
        w = tp.tensor(np.random.default_rng(999).standard_normal((64, 64, 3, 3)) * 0.1, dtype=tp.float32)
        fn = lambda: F.conv2d(a, w, padding=1)
        seconds = _time(fn, reps)
        flops = _count_flops(fn)
        points.append({"n": hw * hw, "seconds": seconds, "flops": flops,
                       "bytes_moved": int((a.numel() + w.numel() + 8 * 64 * hw * hw) * 4)})
        del a, w
    return {"name": "conv2d", "label": "conv2d 3×3 C=64", "category": "convolution",
            "metric": "flops", "points": points}


def _curve_softmax(row_sizes, reps):
    points = []
    for rows in row_sizes:
        a = tp.tensor(np.random.default_rng(rows).standard_normal((rows, 1024)) * 5, dtype=tp.float32)
        fn = lambda: F.softmax(a, dim=-1)
        seconds = _time(fn, reps)
        points.append({"n": rows * 1024, "seconds": seconds, "flops": 0,
                       "bytes_moved": int(2 * rows * 1024 * 4)})
        del a
    return {"name": "softmax", "label": "softmax (rows×1024)", "category": "normalization",
            "metric": "bytes", "points": points}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(args.threads)

    started = time.perf_counter()
    curves = [
        _curve_mm([64, 128, 256, 384, 512, 768, 1024, 1536, 2048], args.reps),
        _curve_elementwise([1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26], args.reps),
        _curve_reduction([1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26], args.reps),
        _curve_conv([14, 28, 56, 84, 112, 168, 224], args.reps),
        _curve_softmax([128, 256, 512, 1024, 2048, 4096, 8192], args.reps),
    ]
    elapsed = time.perf_counter() - started

    payload = {
        "schema_version": 1,
        "benchmark": "scaling",
        "dtype": "f32",
        "threads": args.threads or getattr(tp, "get_num_threads", lambda: 0)(),
        "curves": curves,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"scaling curves: {len(curves)} series in {elapsed:.1f}s -> {args.json_out}")


if __name__ == "__main__":
    main()
