"""Check whether a step can be recorded and replayed, and what that saves.

A decode step issues the same operators on the same shapes every token, so
the cost of issuing them is overhead a recording can remove.  This records a
step into a device task graph and puts one replay of it next to issuing the
step again.

A recording holds device work only, so the probe first reports what a step
still reaches on the host: an operator answered there is not part of a
recording, and a recording that contained one would replay the answer it read
rather than the answer the buffers hold.

    python probe_capture.py [--model <dir>] [--repeat 20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402


def bench(label, sync, repeat, body):
    body()
    sync()
    start = time.time()
    for _ in range(repeat):
        body()
    sync()
    each = (time.time() - start) / repeat
    print(f"  {label:34} {each * 1e3:9.4f} ms")
    return each


def host_fallbacks():
    return {name: int(value) for name, value in tp._C._host_fallback_counts().items()}


def synthetic(device, sync, repeat):
    """A product and a scale, small enough to check by hand."""
    from tensorplay_mlu.graph import Graph

    print("a recorded product and scale")
    x = tp.randn(1, 4096, device=device, dtype=tp.float16)
    weight = (tp.randn(4096, 4096, device=device, dtype=tp.float16) * 0.01).contiguous()

    def body():
        with tp.no_grad():
            return (x @ weight) * 1.01 + x

    with tp.no_grad():
        for _ in range(3):
            body()
        sync()

    graph = Graph()
    recorded = graph.capture(body)
    handed, held = graph.stats()
    print(f"  captured ({handed} blocks handed out while recording, {held} now held)")

    # A replay answers with what the buffers hold, not with what they held
    # while the sequence was recorded.  Capturing does not run the sequence,
    # so the value a capture returns is only meaningful after a replay.
    with tp.no_grad():
        original = x.clone()

        x.add_(0.25)
        changed_eager = (x @ weight) * 1.01 + x
        graph.replay()
        sync()
        changed = float((recorded - changed_eager).abs().max().item())
        spread = float(
            (changed_eager - ((original @ weight) * 1.01 + original)).abs().max().item()
        )

        x.copy_(original)
        same_eager = (x @ weight) * 1.01 + x
        graph.replay()
        sync()
        same = float((recorded - same_eager).abs().max().item())

        print(f"  replayed against the same arithmetic, changed input:  "
              f"max abs {changed:.6f}")
        print(f"  replayed against the same arithmetic, original input: "
              f"max abs {same:.6f}")
        print(f"  (the two inputs alone differ by {spread:.4f}, so a replay "
              f"that ran nothing would not match)")

        eager = bench("eager", sync, repeat, body)
        replay = bench("replay", sync, repeat, graph.replay)
    print(f"  replay costs {eager / replay:.2f} of issuing the sequence")
    graph.reset()


def tower(args, device, sync, repeat):
    """The real decode step, recorded at one cache position."""
    from tensorplay_mlu.graph import Graph

    from tensorplay.nn.cache import StaticCache
    from qwen3vl.model import cache_dense_weights, load_qwen3vl, use_written_normalization

    model, config = load_qwen3vl(
        args.model, dtype=tp.float16, device=device, progress=print
    )
    cache_dense_weights(model, tp.float16)
    written = use_written_normalization(model, args.norm)
    print(f"  {written} normalizations written out")

    cache = StaticCache(args.static_cache)
    ids = tp.tensor(
        [[int(v) for v in args.prompt_ids.split(",")]], dtype=tp.int64, device=device
    )
    length = ids.shape[1]
    positions = tp.zeros(1, 1, dtype=tp.int64, device=device)

    with tp.no_grad():
        logits = model(
            ids,
            position_ids=tp.arange(0, length, device=device).unsqueeze(0),
            past_key_values=cache,
        )
    sync()
    token = logits[:, -1:].argmax(dim=-1)
    sync()

    def advance(offset):
        positions.copy_(
            tp.tensor([[length + offset]], dtype=tp.int64, device=device)
        )

    # What one step still answers on the host.  None is what a recording needs.
    def step(offset):
        advance(offset)
        with tp.no_grad():
            return model(token, position_ids=positions, past_key_values=cache)

    step(0)
    sync()
    tp._C._reset_host_fallback_counts()
    tp._C._set_host_fallback_counting(True)
    step(1)
    sync()
    tp._C._set_host_fallback_counting(False)
    fallbacks = host_fallbacks()
    total = sum(fallbacks.values())
    print(f"  one step answers {total} operator call(s) on the host")
    for name, value in sorted(fallbacks.items(), key=lambda item: -item[1])[:8]:
        print(f"    {name:40} {value}")

    eager = bench("eager step", sync, repeat, lambda: step(2))

    print("  recording one step")
    graph = Graph()
    advance(3)
    marker = list(cache._cursor)
    try:
        recorded = graph.capture(
            lambda: model(token, position_ids=positions, past_key_values=cache)
        )
    except Exception as error:  # noqa: BLE001
        print(f"  recording failed: {type(error).__name__}: {str(error)[:300]}")
        return
    handed, held = graph.stats()
    print(f"  recorded ({handed} blocks handed out while recording, {held} now held)")

    # A recording does not run what it takes in, and the step it holds writes
    # at the cache position it was recorded at.  Put the cache back there and
    # issue the same step again, to have the replay answer against something.
    cache._cursor[:] = marker
    graph.replay()
    sync()
    again = step(3)
    sync()
    difference = float((recorded - again).abs().max().item())
    print(f"  replayed step, issued again:                max abs {difference:.4f}")
    print(f"  (the recorded step writes into the slot it was recorded at, so a "
          f"replay repeats that step and not the next one)")

    # The same input update the eager step performs, so the two are measured
    # the same way: a step is only meaningful at its own position.
    def replayed(offset):
        advance(offset)
        graph.replay()

    replay_time = bench("replayed step", sync, repeat, lambda: replayed(4))
    print(f"  replay costs {eager / replay_time:.2f} of issuing the step")
    print(f"  replayed step is {stats_line(replay_time)}")
    graph.reset()
    del recorded


def stats_line(seconds):
    return f"{seconds * 1e3:.2f} ms  {1 / seconds:.2f} token/s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--norm", default="written")
    ap.add_argument("--static-cache", type=int, default=512)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    if not args.skip_synthetic:
        synthetic("mlu", sync, args.repeat)
        print()

    if args.model is None:
        return

    print(f"a decode step of the tower ({args.model})")
    tower(args, "mlu", sync, args.repeat)


if __name__ == "__main__":
    main()
