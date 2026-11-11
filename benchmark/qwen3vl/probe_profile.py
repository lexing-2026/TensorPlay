"""Show where the host time of a decode step is spent.

A step is issued from the host: each operator call costs the host time it takes
to leave, and the accelerator only sees work once it has been handed over.  When
the step is longer than the work it describes, the calls themselves are the
subject, and a profile of one step names them in order instead of leaving the
cost to be inferred from a handful of hand-timed operators.

    python probe_profile.py --model <dir> --steps 6 [--top 40]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import cache_dense_weights, load_qwen3vl, use_written_normalization  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    ap.add_argument("--static-cache", type=int, default=512)
    ap.add_argument("--norm", default="written")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--sort", default="tottime")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, _ = load_qwen3vl(args.model, dtype=tp.float16, device="mlu", progress=print)
    cache_dense_weights(model, tp.float16)
    if args.norm != "module":
        use_written_normalization(model, args.norm)
    sync()

    ids = tp.tensor([[int(x) for x in args.prompt_ids.split(",")]],
                    dtype=tp.int64, device="mlu")
    length = ids.shape[1]
    from tensorplay.nn.cache import StaticCache

    cache = StaticCache(args.static_cache)

    def position(offset):
        return tp.arange(length + offset, length + offset + 1,
                         device="mlu").unsqueeze(0)

    with tp.no_grad():
        logits = model(
            ids,
            position_ids=tp.arange(0, length, device="mlu").unsqueeze(0),
            past_key_values=cache,
        )
    sync()
    # Run a couple of settling steps, then collect one token per step: each
    # step's input is the token the step before it chose.
    tokens = []
    with tp.no_grad():
        for offset in range(2 + args.steps):
            tokens.append(logits[:, -1:].argmax(dim=-1))
            logits = model(tokens[-1], position_ids=position(offset),
                           past_key_values=cache)
    sync()

    start = 2
    saved = list(cache._cursor)

    def body():
        # The same steps again, from the same write positions, so the profile
        # covers exactly the work a step does.
        with tp.no_grad():
            cache._cursor[:] = saved
            for offset in range(args.steps):
                model(tokens[start + offset],
                      position_ids=position(start + offset),
                      past_key_values=cache)

    profiler = cProfile.Profile()
    profiler.enable()
    body()
    profiler.disable()
    sync()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(args.sort)
    stats.print_stats(args.top)
    print(stream.getvalue())
    print(f"({args.steps} steps profiled)")


if __name__ == "__main__":
    main()
