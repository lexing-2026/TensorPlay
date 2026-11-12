"""Split a decode step's device time across the operators it issues.

A replayed step runs the same device work as an issued one, so capturing the
step does not remove any of it.  What matters is how that time is spent: a few
wide products over the weights, or a long tail of small elementwise calls whose
cost is the fixed overhead of running a kernel at all.  The device timer
alongside each operation answers that directly.

    python probe_devtime.py --model <dir> --steps 4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import cache_dense_weights, load_qwen3vl, use_written_normalization  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    ap.add_argument("--static-cache", type=int, default=512)
    ap.add_argument("--norm", default="written")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, _ = load_qwen3vl(args.model, dtype=tp.float16, device="mlu", progress=print)
    cache_dense_weights(model, tp.float16)
    if args.norm != "module":
        use_written_normalization(model, args.norm)
    sync()

    from tensorplay.nn.cache import StaticCache

    ids = tp.tensor(
        [[int(x) for x in args.prompt_ids.split(",")]], dtype=tp.int64, device="mlu"
    )
    length = ids.shape[1]
    cache = StaticCache(args.static_cache)

    def position(offset):
        return tp.arange(length + offset, length + offset + 1, device="mlu").unsqueeze(0)

    with tp.no_grad():
        logits = model(
            ids,
            position_ids=tp.arange(0, length, device="mlu").unsqueeze(0),
            past_key_values=cache,
        )
    sync()

    tokens = []
    with tp.no_grad():
        for offset in range(2 + args.steps):
            tokens.append(logits[:, -1:].argmax(dim=-1))
            logits = model(tokens[-1], position_ids=position(offset), past_key_values=cache)
    sync()

    start = 2
    saved = list(cache._cursor)

    # Warm the same shapes once, outside the region, so nothing in the profile
    # is a first-touch cost.
    def body():
        with tp.no_grad():
            cache._cursor[:] = saved
            for offset in range(args.steps):
                model(
                    tokens[start + offset],
                    position_ids=position(start + offset),
                    past_key_values=cache,
                )

    body()
    sync()

    with tp.profiler.profile(
        activities=[tp.profiler.ProfilerActivity.PrivateUse1]
    ) as prof:
        body()
    sync()

    rows = {}
    total = 0.0
    for event in prof.events:
        device_ms = event[8]
        if device_ms < 0:
            continue
        total += device_ms
        row = rows.setdefault(event[0], [0, 0.0])
        row[0] += 1
        row[1] += device_ms

    steps = args.steps
    print(f"\none decode step: {total / steps:.3f} ms of device time over "
          f"{sum(r[0] for r in rows.values()) // steps} operations")
    print(f"  {'operation':<40} {'calls':>6} {'ms/step':>9} {'us/call':>8}")
    for name, (calls, ms) in sorted(rows.items(), key=lambda item: -item[1][1])[: args.top]:
        print(f"  {name:<40} {calls // steps:6d} {ms / steps:9.4f} "
              f"{ms / calls * 1e3:8.2f}")


if __name__ == "__main__":
    main()
