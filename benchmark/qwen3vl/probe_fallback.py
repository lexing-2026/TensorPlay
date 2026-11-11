"""Name every operator a decode step answers on the host instead of the device.

The device carries a kernel for a set of operators; anything else is copied to
the host, computed there and copied back, and the host cannot hand the answer
back until the device has drained.  A step that reaches such an operator is
therefore bounded by the host, so the first thing to know is which operators
those are and how often each is reached.

The counts come from the dispatcher itself, so nothing is guessed.

    python probe_fallback.py --model <dir> --steps 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import cache_dense_weights, load_qwen3vl  # noqa: E402


def counts():
    return {name: int(value) for name, value in tp._C._host_fallback_counts().items()}


def report(label, steps, measured):
    print(f"\n{label}: {measured} step(s)")
    total = sum(steps.values())
    if not total:
        print("  none, every operator ran on the device")
        return
    print(f"  {total} host answers in all, {total / measured:.1f} per step")
    for name, value in sorted(steps.items(), key=lambda item: -item[1]):
        print(f"    {name:40} {value:6d}  {value / measured:8.1f} per step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--static-cache", type=int, default=512)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    ap.add_argument("--norm", default="written")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, _ = load_qwen3vl(args.model, dtype=tp.float16, device="mlu", progress=print)
    cache_dense_weights(model, tp.float16)
    if args.norm != "module":
        from qwen3vl.model import use_written_normalization

        use_written_normalization(model, args.norm)
    sync()

    ids = tp.tensor([[int(x) for x in args.prompt_ids.split(",")]],
                    dtype=tp.int64, device="mlu")
    length = ids.shape[1]
    from tensorplay.nn.cache import StaticCache

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

    # Warm the step once without counting, so allocation and the first-time
    # paths are not charged to the operators.  The cache keeps advancing, so the
    # positions continue from where the warm steps left them.
    with tp.no_grad():
        for offset in range(2):
            token = logits[:, -1:].argmax(dim=-1)
            logits = model(token, position_ids=position(offset), past_key_values=cache)
    sync()

    tp._C._reset_host_fallback_counts()
    tp._C._set_host_fallback_counting(True)
    with tp.no_grad():
        for offset in range(args.steps):
            token = logits[:, -1:].argmax(dim=-1)
            logits = model(token, position_ids=position(2 + offset),
                           past_key_values=cache)
    sync()
    tp._C._set_host_fallback_counting(False)

    report("decode step", counts(), args.steps)


if __name__ == "__main__":
    main()
