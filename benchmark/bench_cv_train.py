#!/usr/bin/env python3
"""Time a convolutional training step on whichever device is asked for.

One script, one model definition, one data path: the numbers a device reports
are therefore directly comparable to another device's, since nothing about the
workload changes between them.

The inputs are drawn once from a fixed seed and moved to the device, so every
run sees byte-identical samples regardless of where it executes.  Timing is
taken around a device synchronize, after a warmup, and the median over the
timed steps is what is reported.

    python benchmark/bench_cv_train.py --device cuda --json cuda.json
    python benchmark/bench_cv_train.py --device mlu  --json mlu.json

The script builds its model and loss out of TensorPlay alone; it never imports
another tensor framework, and it reports whether one got pulled in anyway.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Use the checkout's package rather than an installed one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp  # noqa: E402
import tensorplay.nn as nn  # noqa: E402


DTYPES = {"fp32": tp.float32, "fp16": tp.float16, "bf16": tp.bfloat16}
BUILDERS = {}


def _register_builders() -> None:
    from tensorplay.vision import models as tvm

    for name in ("resnet18", "resnet34", "resnet50"):
        BUILDERS[name] = getattr(tvm, name)


def pick_device(name: str) -> str:
    """Load the backend for a device and hand back the name to allocate on."""
    if name == "mlu":
        import tensorplay_mlu

        tensorplay_mlu.load()
        if tp.mlu.device_count() == 0:
            raise RuntimeError("no MLU device is visible")
        return "mlu"
    return name


def synchronize(device: str) -> None:
    if device == "cuda":
        tp.cuda.synchronize()
    elif device == "mlu":
        tp.mlu.synchronize()


def timed(device: str, fn) -> float:
    """Seconds spent inside ``fn``, measured on the device's own clock."""
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    return time.perf_counter() - start


def make_inputs(device, cfg, dtype, batch, image_size, seed):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch, 3, image_size, image_size)).astype(np.float32)
    y_np = rng.integers(0, cfg.num_classes, size=batch).astype(np.int64)
    x = tp.from_numpy(x_np).to(device).to(dtype)
    y = tp.from_numpy(y_np).to(device)
    return x, y


def measured(device: str, fn, warmup: int, steps: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = [timed(device, fn) for _ in range(steps)]
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "steps": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = DTYPES[args.dtype]
    _register_builders()

    tp.manual_seed(args.seed)
    model = BUILDERS[args.model](num_classes=args.classes).to(device).to(dtype)
    opt = tp.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    class _Cfg:
        num_classes = args.classes

    x, y = make_inputs(device, _Cfg, dtype, args.batch_size, args.image_size, args.seed)
    params = sum(int(np.prod(tuple(p.shape))) for p in model.parameters())

    def forward():
        return model(x)

    def train_step():
        opt.zero_grad()
        loss = tp.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss

    # A first call in each mode, so the reported figures are steady-state.
    loss = train_step()
    sync_ok = True
    try:
        first_loss = float(loss.item())
    except Exception:  # pragma: no cover - only if a device cannot read back
        sync_ok = False
        first_loss = None

    fwd = measured(device, forward, args.warmup, args.steps)
    step = measured(device, train_step, args.warmup, args.steps)

    per_image = args.batch_size / step["median_s"]
    result = {
        "label": args.label or device,
        "device": device,
        "dtype": args.dtype,
        "model": args.model,
        "classes": args.classes,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "parameters": params,
        "forward_ms": fwd["median_s"] * 1e3,
        "forward_min_ms": fwd["min_s"] * 1e3,
        "train_step_ms": step["median_s"] * 1e3,
        "train_step_min_ms": step["min_s"] * 1e3,
        "train_step_all_ms": [s * 1e3 for s in step["steps"]],
        "images_per_s": per_image,
        "first_loss": first_loss,
        "readback_ok": sync_ok,
        "other_framework_imported": "torch" in sys.modules,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
