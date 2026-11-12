#!/usr/bin/env python3
"""Where the time of a maximum pooling goes.

The sliding-window maximum is asked for in each of its spellings, next to the
average form which shares the geometry but reports no positions, and next to a
plain round trip of the same bytes.  The comparison says whether the cost is
the convolution-like window itself, the position bookkeeping, or moving the
operand between the card and the host.

    python benchmark/probe_cv_pool.py --device cuda
    python benchmark/probe_cv_pool.py --device mlu
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp  # noqa: E402
import tensorplay.nn as nn  # noqa: E402


def pick_device(name: str) -> str:
    if name == "mlu":
        import tensorplay_mlu

        tensorplay_mlu.load()
        return "mlu"
    return name


def synchronize(device: str) -> None:
    if device == "cuda":
        tp.cuda.synchronize()
    elif device == "mlu":
        tp.mlu.synchronize()


def timeit(device: str, fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    device = pick_device(args.device)
    f = tp.nn.functional

    x = tp.randn(32, 64, 112, 112, device=device)
    nbytes = int(np.prod(tuple(x.shape))) * 4
    host = np.zeros(32 * 64 * 112 * 112, dtype=np.float32)
    dev_buf = tp.randn(32, 64, 112, 112, device=device)

    rows = [
        ("max_pool2d (module path, no positions)",
         lambda: f.max_pool2d(x, 3, stride=2, padding=1)),
        ("max_pool2d(return_indices=True)",
         lambda: f.max_pool2d(x, 3, stride=2, padding=1, return_indices=True)),
        ("max_pool2d_with_indices",
         lambda: f.max_pool2d_with_indices(x, 3, stride=2, padding=1)),
        ("avg_pool2d (no positions)",
         lambda: f.avg_pool2d(x, 3, stride=2, padding=1)),
        ("relu (same-size elementwise)",
         lambda: f.relu(x)),
        ("copy card -> host",
         lambda: host.copy_(np.asarray(dev_buf.to("cpu")))),
        ("full round trip (in and out)",
         lambda: dev_buf.to("cpu").to(device)),
    ]

    print(f"device={device}  input 32x64x112x112 ({nbytes/1e6:.1f} MB)")
    for label, fn in rows:
        try:
            ms = timeit(device, fn, iters=args.iters)
            print(f"  {label:42s} {ms:10.3f} ms")
        except Exception as exc:  # a device that cannot run a spelling
            print(f"  {label:42s}   FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
