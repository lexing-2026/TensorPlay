"""Compare two ways of writing out a decode step's attention over a cache.

One query per head attends the whole cache.  The query heads are grouped, so
either the cached keys are widened to match the query heads and a plain
product follows, or the queries are viewed as ``(groups, per-group)`` and the
product broadcasts the keys across the group.  The second avoids materializing
the widened cache.  This checks both give the same answer and what each costs
as the context grows.

    python probe_attn2.py [--repeat 200]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import tensorplay as tp

HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128


def bench(label, fn, repeat):
    for _ in range(5):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:44} {(time.time() - start) / repeat * 1e3:9.4f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    scale = HEAD_DIM**-0.5
    group = HEADS // KV_HEADS

    for length in (20, 128, 512):
        q = tp.randn(1, HEADS, 1, HEAD_DIM, device="mlu", dtype=tp.float16)
        k = tp.randn(1, KV_HEADS, length, HEAD_DIM, device="mlu", dtype=tp.float16)
        v = tp.randn(1, KV_HEADS, length, HEAD_DIM, device="mlu", dtype=tp.float16)

        def widened():
            keys = k.repeat_interleave(group, dim=1)
            values = v.repeat_interleave(group, dim=1)
            scores = (q @ keys.transpose(-1, -2)) * scale
            weights = tp.nn.functional.softmax(scores, dim=-1)
            return weights @ values

        def grouped():
            # The query heads are split into their groups, so the keys and
            # values match the queries without being copied out.
            qv = q.reshape(1, KV_HEADS, group, 1, HEAD_DIM)
            scores = (qv @ k.unsqueeze(2).transpose(-1, -2)) * scale
            weights = tp.nn.functional.softmax(scores, dim=-1)
            return (weights @ v.unsqueeze(2)).reshape(1, HEADS, 1, HEAD_DIM)

        print(f"\ncache length {length}")
        bench("widen the cache, then two products", widened, args.repeat)
        bench("split the queries into groups", grouped, args.repeat)
        bench("group split without softmax",
              lambda: (q.reshape(1, KV_HEADS, group, 1, HEAD_DIM)
                       @ k.unsqueeze(2).transpose(-1, -2)), args.repeat)

        a = widened().float().cpu().numpy()
        b = grouped().float().cpu().numpy()
        print(f"  largest difference between the two: {np.abs(a - b).max():.5f}")

        ref = np.zeros((1, HEADS, 1, HEAD_DIM), dtype=np.float32)
        qn = q.float().cpu().numpy()
        kn = k.float().cpu().numpy()
        vn = v.float().cpu().numpy()
        for h in range(HEADS):
            scores = (qn[0, h, 0] @ kn[0, h // group].T) * scale
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            ref[0, h, 0] = weights @ vn[0, h // group]
        print(f"  largest difference against a host reference: "
              f"{np.abs(a - ref).max():.5f}")


if __name__ == "__main__":
    main()
