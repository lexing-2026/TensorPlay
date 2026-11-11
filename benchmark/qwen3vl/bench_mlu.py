"""Measure where a decode step on the accelerator spends its time.

Three measurements share one load:

* the ceiling — a plain half-precision product at the shapes the language
  tower uses, so the time a step could take is known;
* the step as it stands — every projection recovers its four-bit groups on
  every forward;
* the step with the values recovered once — the same step with the recovery
  taken out of the loop.

    python bench_mlu.py --model <dir> --device mlu --tokens 5
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import cache_dense_weights, load_qwen3vl  # noqa: E402

SYNC = {
    "mlu": lambda: tp.mlu.synchronize(),
    "cuda": lambda: tp.cuda.synchronize(),
    "cpu": lambda: None,
}


def pick_device(name):
    if name == "mlu":
        import tensorplay_mlu

        tensorplay_mlu.load()
        return "mlu"
    return name


def gib(value):
    return value / 1024**3


def resident_bytes(device):
    counter = getattr(getattr(tp, device, None), "memory_allocated", None)
    if counter is None:
        return None
    try:
        return int(counter())
    except Exception:  # noqa: BLE001
        return None


def gemm_ceiling(device, sync, layers=36):
    """Report the product shapes and what a plain product costs."""
    shapes = [(4096, 4096, 4096), (4096, 1024, 4096), (4096, 12288, 4096),
              (4096, 12288 * 2, 4096)]
    total = 0.0
    for rows, cols, inner in shapes:
        weight = tp.randn(rows, cols, device=device, dtype=tp.float16)
        activation = tp.randn(1, rows, device=device, dtype=tp.float16)
        for _ in range(3):
            (activation @ weight).sum()
        sync()
        start = time.time()
        for _ in range(10):
            (activation @ weight).sum()
        sync()
        per_call = (time.time() - start) / 10
        moved = weight.numel() * 2 + activation.numel() * 2
        print(
            f"  {rows}x{cols}: {per_call * 1e3:7.3f} ms  "
            f"{moved / per_call / 1e9:7.1f} GB/s  "
            f"{2 * rows * cols / per_call / 1e12:6.2f} TFLOP/s"
        )
        total += per_call
        del weight, activation
    print(f"  one of each shape: {total * 1e3:.1f} ms  "
          f"(a layer holds 7 such products, 36 layers hold 252)")


def decode(device, model, ids, steps, sync, cache=None):
    """Prefill, then time every single-token step separately."""
    if cache is None:
        cache = model.create_empty_cache()
    length = ids.shape[1]
    sync()
    start = time.time()
    with tp.no_grad():
        logits = model(ids, position_ids=_arange(0, length, device), past_key_values=cache)
        sync()
    prefill = time.time() - start

    times = []
    with tp.no_grad():
        for offset in range(steps):
            token = logits[:, -1:].argmax(dim=-1)
            positions = _arange(length + offset, length + offset + 1, device)
            sync()
            start = time.time()
            logits = model(
                token, position_ids=positions, past_key_values=cache
            )
            sync()
            times.append(time.time() - start)
    return prefill, times


def _arange(start, end, device):
    return tp.arange(start, end, device=device).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mlu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    ap.add_argument("--written-norm", action="store_true",
                    help="add a third phase with the normalizations written out")
    ap.add_argument("--static-cache", type=int, default=0,
                    help="positions to preallocate for the third phase")
    args = ap.parse_args()

    dtypes = {"float16": tp.float16, "bfloat16": tp.bfloat16}
    device = pick_device(args.device)
    sync = SYNC[device]

    print(f"device={device} dtype={args.dtype}")
    start = time.time()
    model, config = load_qwen3vl(
        args.model, dtype=dtypes[args.dtype], device=device, progress=print
    )
    print(f"load {time.time() - start:.1f}s  {config}")

    before = resident_bytes(device)
    if before is not None:
        print(f"resident after load: {gib(before):.2f} GiB")

    ids = tp.tensor(
        [[int(v) for v in args.prompt_ids.split(",")]], dtype=tp.int64, device=device
    )

    print("\nproduct ceiling")
    gemm_ceiling(device, sync)

    print(f"\nstep as it stands ({args.tokens} tokens)")
    prefill, times = decode(device, model, ids, args.tokens, sync)
    print(f"  prefill {prefill * 1e3:.1f} ms")
    print("  steps " + " ".join(f"{v * 1e3:.1f}" for v in times) + " ms")
    print(f"  median {stats.median(times) * 1e3:.1f} ms  "
          f"{1 / stats.median(times):.2f} token/s")

    print("\nrecovering the values once")
    start = time.time()
    count = cache_dense_weights(model, dtypes[args.dtype])
    sync()
    print(f"  {count} projections recovered in {time.time() - start:.1f}s")
    after = resident_bytes(device)
    if before is not None and after is not None:
        print(f"  resident now: {gib(after):.2f} GiB (+{gib(after - before):.2f} GiB)")

    prefill, times = decode(device, model, ids, args.tokens, sync)
    print(f"  prefill {prefill * 1e3:.1f} ms")
    print("  steps " + " ".join(f"{v * 1e3:.1f}" for v in times) + " ms")
    if times:
        print(f"  median {stats.median(times) * 1e3:.1f} ms  "
              f"{1 / stats.median(times):.2f} token/s")

    if args.written_norm or args.static_cache:
        from qwen3vl.model import use_written_normalization

        print("\nvalues cached, normalizations written out, cache preallocated")
        if args.written_norm:
            count = use_written_normalization(model)
            print(f"  {count} normalizations written out")
        cache = None
        if args.static_cache:
            from tensorplay.nn.cache import StaticCache

            cache = StaticCache(args.static_cache)
            print(f"  cache preallocated for {args.static_cache} positions")
        prefill, times = decode(device, model, ids, args.tokens, sync, cache=cache)
        print(f"  prefill {prefill * 1e3:.1f} ms")
        print("  steps " + " ".join(f"{v * 1e3:.1f}" for v in times) + " ms")
        if times:
            print(f"  median {stats.median(times) * 1e3:.1f} ms  "
                  f"{1 / stats.median(times):.2f} token/s")


if __name__ == "__main__":
    main()
