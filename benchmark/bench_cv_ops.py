#!/usr/bin/env python3
"""Per-operator timing for the convolution shapes a ResNet-18 walks through.

The whole-model figure says how far apart two devices are; this says where.
Each case is one operator at the shape the network actually uses, timed on its
own with a device synchronize around it, so a single slow operator cannot hide
behind the rest.

    python benchmark/bench_cv_ops.py --device cuda
    python benchmark/bench_cv_ops.py --device mlu --json ops_mlu.json

Only TensorPlay is imported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

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


def timeit(device: str, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def cases(device: str):
    """Every case as (label, callable, work item count)."""

    def conv(cin, cout, k, s, p, hw, n=32):
        m = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=True).to(device)
        x = tp.randn(n, cin, hw, hw, device=device)
        return m, x

    out = []

    m, x = conv(3, 64, 7, 2, 3, 224)
    out.append(("conv 3->64 7x7 s2 @224 (stem)", lambda m=m, x=x: m(x), x.numel()))

    m, x = conv(64, 64, 3, 1, 1, 56)
    out.append(("conv 64->64 3x3 s1 @56", lambda m=m, x=x: m(x), x.numel()))

    m, x = conv(64, 128, 3, 2, 1, 56)
    out.append(("conv 64->128 3x3 s2 @56", lambda m=m, x=x: m(x), x.numel()))

    m, x = conv(128, 128, 3, 1, 1, 28)
    out.append(("conv 128->128 3x3 s1 @28", lambda m=m, x=x: m(x), x.numel()))

    m, x = conv(256, 256, 3, 1, 1, 14)
    out.append(("conv 256->256 3x3 s1 @14", lambda m=m, x=x: m(x), x.numel()))

    m, x = conv(512, 512, 3, 1, 1, 7)
    out.append(("conv 512->512 3x3 s1 @7", lambda m=m, x=x: m(x), x.numel()))

    m, x = conv(1, 1, 1, 1, 0, 56)
    out.append(("conv 1->1 1x1 @56", lambda m=m, x=x: m(x), x.numel()))

    bn = nn.BatchNorm2d(64).to(device)
    bn.train()
    xb = tp.randn(32, 64, 56, 56, device=device)
    out.append(("batchnorm2d 64 (train) @56", lambda: bn(xb), xb.numel()))

    xr = tp.randn(32, 64, 56, 56, device=device)
    out.append(("relu @56", lambda: tp.nn.functional.relu(xr), xr.numel()))

    p = nn.MaxPool2d(3, stride=2, padding=1)
    xp = tp.randn(32, 64, 112, 112, device=device)
    out.append(("maxpool 3x3 s2 @112", lambda: p(xp), xp.numel()))

    lin = nn.Linear(512, 1000).to(device)
    xl = tp.randn(32, 512, device=device)
    out.append(("linear 512->1000", lambda: lin(xl), xl.numel()))

    xa = tp.randn(32, 64, 56, 56, device=device)
    xa2 = tp.randn(32, 64, 56, 56, device=device)
    out.append(("add (residual) @56", lambda: xa + xa2, xa.numel()))

    xs = tp.randn(32, 64, 56, 56, device=device)
    out.append(("sum over all", lambda: xs.sum(), xs.numel()))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    rows = []
    for label, fn, count in cases(device):
        ms = timeit(device, fn, args.warmup, args.iters) * 1e3
        rows.append({"op": label, "ms": ms, "elements": count})
        print(f"  {label:38s} {ms:10.3f} ms", flush=True)

    result = {
        "label": args.label or device,
        "device": device,
        "rows": rows,
        "other_framework_imported": "torch" in sys.modules,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({r["op"]: round(r["ms"], 3) for r in rows}, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
