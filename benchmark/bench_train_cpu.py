#!/usr/bin/env python3
"""CPU forward and backward benchmark for representative training shapes."""

import argparse
import json
import time
from pathlib import Path

import tensorplay as tp
from tensorplay.nn import functional as F


SHAPES = (
    ("qkv-proj", 512, 4096, 4096),
    ("mlp-up", 512, 4096, 11008),
    ("mlp-down", 512, 11008, 4096),
    ("small-net", 128, 1024, 1024),
    ("decode-1", 1, 4096, 4096),
)


def _time(fn, reps):
    fn()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return min(samples)


def _measure(m, k, n, reps):
    x = tp.randn((m, k))
    weight = tp.randn((n, k))
    bias = tp.randn((n,))

    def forward():
        F.linear(x, weight, bias)

    def step():
        x_step = x.clone()
        weight_step = weight.clone()
        bias_step = bias.clone()
        x_step.requires_grad = True
        weight_step.requires_grad = True
        bias_step.requires_grad = True
        output = F.linear(x_step, weight_step, bias_step)
        output.pow(2).sum().backward()

    return _time(forward, reps), _time(step, reps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.threads < 0:
        parser.error("reps must be positive and threads must be non-negative")
    if args.threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(args.threads)

    measurements = []
    for name, batch_tokens, in_features, out_features in SHAPES:
        forward_seconds, step_seconds = _measure(
            batch_tokens, in_features, out_features, args.reps
        )
        print(
            f"{name:10} M={batch_tokens:4} K={in_features:5} N={out_features:5} "
            f"forward={forward_seconds * 1e3:9.3f} ms "
            f"step={step_seconds * 1e3:9.3f} ms"
        )
        measurements.append({
            "name": name,
            "category": "training",
            "batch_tokens": batch_tokens,
            "in_features": in_features,
            "out_features": out_features,
            "forward_seconds": forward_seconds,
            "step_seconds": step_seconds,
        })

    payload = {
        "schema_version": 1,
        "benchmark": "cpu-train-step",
        "threads": args.threads or getattr(tp, "get_num_threads", lambda: 0)(),
        "measurements": measurements,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8")
    print(f"wrote results: {args.json_out}")


if __name__ == "__main__":
    main()
