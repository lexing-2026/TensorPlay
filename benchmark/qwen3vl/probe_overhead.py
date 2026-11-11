"""Split a decode step into the time spent enqueueing and the time spent running.

A step can be limited by the calls it issues from the host or by the work the
accelerator runs for them.  Both are timed separately: the host time is taken
around the forward call without synchronizing, and what remains until the
synchronization completes is the accelerator's own time.  A second pass repeats
the step with every projection returning a ready result, which keeps the call
count and the shapes but removes almost all of the reading.

    python probe_overhead.py --model <dir> [--steps 6]
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import (  # noqa: E402
    cache_dense_weights,
    load_qwen3vl,
    use_written_normalization,
)


def run(model, ids, steps, sync, label, positions=0):
    if positions:
        from tensorplay.nn.cache import StaticCache

        cache = StaticCache(positions)
    else:
        cache = model.create_empty_cache()
    length = ids.shape[1]
    with tp.no_grad():
        logits = model(ids, position_ids=tp.arange(0, length, device="mlu").unsqueeze(0),
                       past_key_values=cache)
    sync()

    host, device, total = [], [], []
    with tp.no_grad():
        for offset in range(steps):
            token = logits[:, -1:].argmax(dim=-1)
            position = tp.arange(length + offset, length + offset + 1,
                                 device="mlu").unsqueeze(0)
            sync()
            start = time.time()
            logits = model(token, position_ids=position, past_key_values=cache)
            queued = time.time()
            sync()
            done = time.time()
            host.append(queued - start)
            device.append(done - queued)
            total.append(done - start)

    print(f"\n{label}")
    print(f"  host, issuing the calls   median {stats.median(host) * 1e3:8.3f} ms")
    print(f"  accelerator               median {stats.median(device) * 1e3:8.3f} ms")
    print(f"  step                      median {stats.median(total) * 1e3:8.3f} ms")
    return stats.median(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--written-norm", action="store_true")
    ap.add_argument("--static-cache", type=int, default=512)
    ap.add_argument("--no-projection", action="store_true",
                    help="also time a pass whose projections return ready results")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, config = load_qwen3vl(args.model, dtype=tp.float16, device="mlu", progress=print)
    cache_dense_weights(model, tp.float16)
    if args.written_norm:
        print(f"wrote out {use_written_normalization(model)} normalizations")
    sync()

    ids = tp.tensor([[104198, 48, 16948, 3837, 67071]], dtype=tp.int64, device="mlu")
    run(model, ids, args.steps, sync, "step as the harness runs it",
        positions=args.static_cache)

    if args.no_projection:
        projections = []

        def replace(module):
            if hasattr(module, "_dense_cache"):
                projections.append(module)
                cached = module._dense_cache

                def forward(x, _cached=cached):
                    return tp.zeros(*x.shape[:-1], _cached.shape[0], dtype=x.dtype,
                                    device=x.device)

                module.forward = forward
            for child in module.children():
                replace(child)

        replace(model)
        print(f"\n{len(projections)} projections made to return a ready result")
        run(model, ids, args.steps, sync, "the same step without the reading",
            positions=args.static_cache)


if __name__ == "__main__":
    main()
