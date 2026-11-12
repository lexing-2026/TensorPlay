"""Replay one recorded decode over a whole continuation and check the tokens.

A recording of a decode step is only useful if one recording serves every
position, which needs a cache whose write does not move its address.  This runs
a continuation twice -- once issuing every step, once replaying a single
recorded step -- and compares the tokens, then puts the replayed step next to
the issued one.

    python probe_replay.py --model <dir> --tokens 16 --repeat 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import (  # noqa: E402
    GraphCache,
    cache_dense_weights,
    load_qwen3vl,
    use_written_normalization,
)

PROMPT = [104198, 48, 16948, 3837, 67071]


def eager_tokens(model, ids, steps, device):
    """The continuation when every step is issued on its own."""
    from tensorplay.nn.cache import StaticCache

    cache = StaticCache(512)
    length = ids.shape[1]
    with tp.no_grad():
        logits = model(
            ids,
            position_ids=tp.arange(0, length, device=device).unsqueeze(0),
            past_key_values=cache,
        )
    tokens = []
    with tp.no_grad():
        for step in range(steps):
            token = logits[:, -1:].argmax(dim=-1)
            tokens.append(int(token.item()))
            logits = model(
                token,
                position_ids=tp.arange(
                    length + step, length + step + 1, device=device
                ).unsqueeze(0),
                past_key_values=cache,
            )
    return tokens


def replayed_tokens(model, ids, steps, device, sync, repeat):
    """The same continuation, with one recorded step replayed every position."""
    from tensorplay_mlu.graph import Graph

    cache = GraphCache(512)
    length = ids.shape[1]
    with tp.no_grad():
        logits = model(
            ids,
            position_ids=tp.arange(0, length, device=device).unsqueeze(0),
            past_key_values=cache,
        )
    sync()

    token = logits[:, -1:].argmax(dim=-1)
    token_buffer = token.clone()
    positions = tp.zeros(1, 1, dtype=tp.int64, device=device)
    cache.advance(length)

    graph = Graph()
    with tp.no_grad():
        recorded = graph.capture(
            lambda: model(
                token_buffer,
                position_ids=positions,
                past_key_values=cache,
                mask=cache.mask,
            )
        )
    handed, held = graph.stats()
    print(f"  recorded ({handed} blocks handed out while recording, {held} now held)")

    def one_step(step):
        positions.fill_(length + step)
        cache.advance(length + step)
        graph.replay()

    # The prompt's own prediction is the first token of the continuation; each
    # replay then consumes the token the replay before it chose, so the list
    # lines up with the issued one.
    tokens = [int(token.item())]
    with tp.no_grad():
        for step in range(steps - 1):
            one_step(step)
            sync()
            chosen = recorded[:, -1:].argmax(dim=-1)
            tokens.append(int(chosen.item()))
            token_buffer.copy_(chosen)
    sync()

    # Timing: the replayed step on its own, and the whole loop a caller runs
    # (position and slot update, replay, choose the next token).
    def replay_only():
        with tp.no_grad():
            one_step(0)

    def loop_step(step):
        one_step(step)
        sync()
        chosen = recorded[:, -1:].argmax(dim=-1)
        token_buffer.copy_(chosen)

    def full_loop():
        with tp.no_grad():
            for step in range(repeat):
                loop_step(step)

    warm = 5
    for _ in range(warm):
        one_step(0)
    sync()
    start = time.time()
    for _ in range(repeat):
        replay_only()
    sync()
    replay_ms = (time.time() - start) / repeat * 1e3

    for _ in range(2):
        full_loop()
    sync()
    start = time.time()
    full_loop()
    sync()
    loop_ms = (time.time() - start) / repeat * 1e3

    graph.reset()
    return tokens, replay_ms, loop_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, _ = load_qwen3vl(args.model, dtype=tp.float16, device="mlu", progress=print)
    cache_dense_weights(model, tp.float16)
    written = use_written_normalization(model, "written")
    print(f"  {written} normalizations written out")
    sync()

    device = "mlu"
    ids = tp.tensor([PROMPT], dtype=tp.int64, device=device)

    print("\nissuing every step")
    expected = eager_tokens(model, ids, args.tokens, device)
    print(f"  {expected}")

    print("\nreplaying one recorded step")
    actual, replay_ms, loop_ms = replayed_tokens(
        model, ids, args.tokens, device, sync, args.repeat
    )
    print(f"  {actual}")

    agree = sum(a == b for a, b in zip(actual, expected))
    print(f"\ntoken agreement: {agree} / {args.tokens}")
    if agree != args.tokens:
        first = next(
            (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None
        )
        print(f"  first mismatch at {first}: replayed {actual[first]} "
              f"against issued {expected[first]}")
    print(f"replayed step  {replay_ms:8.3f} ms  {1e3 / replay_ms:8.2f} token/s")
    print(f"whole loop     {loop_ms:8.3f} ms  {1e3 / loop_ms:8.2f} token/s")


if __name__ == "__main__":
    main()
