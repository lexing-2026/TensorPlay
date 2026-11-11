"""Measure what a single operator call costs on the accelerator.

A decode step issues a few hundred small operators.  If a call costs a fixed
amount regardless of its size, the step is bound by the number of calls rather
than by the data, and the fix is to issue fewer, larger calls.  This reports
the per-call cost of the operators a step uses at the shapes a step uses.

    python probe_latency.py [--repeat 2000]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def timed(label, make, repeat, inner):
    """Run ``make()`` ``repeat`` times and report the per-call cost.

    ``inner`` is how many separate operator calls each iteration issues, so a
    fused body is comparable with the same body written out.
    """
    work = [make, make, make, make]
    for element in work:
        element()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        for element in work:
            element()
    tp.mlu.synchronize()
    per_call = (time.time() - start) / (repeat * len(work) * inner)
    print(f"  {label:34} {per_call * 1e6:9.1f} us/call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=500)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    a = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    b = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    w = tp.randn(4096, 4096, device="mlu", dtype=tp.float16)
    w2 = tp.randn(12288, 4096, device="mlu", dtype=tp.float16)
    tiny = tp.randn(1, 1, 128, device="mlu", dtype=tp.float16)
    kbig = tp.randn(1, 8, 512, 128, device="mlu", dtype=tp.float16)

    print("per-call cost")
    timed("add 1x1x4096", lambda: (a + b), args.repeat, 1)
    timed("mul 1x1x4096", lambda: (a * b), args.repeat, 1)
    timed("add 1x1x128", lambda: (tiny + tiny), args.repeat, 1)
    timed("matmul 1x4096 @ 4096x4096", lambda: (a @ w), args.repeat, 1)
    timed("matmul 1x4096 @ 12288x4096", lambda: (a @ w2), args.repeat, 1)
    timed("cat two 1x1x128", lambda: tp.cat([tiny, tiny], dim=2), args.repeat, 1)
    timed("softmax 1x8x1x512", lambda: tp.nn.functional.softmax(kbig, dim=-1), args.repeat, 1)

    # The same body two ways: a handful of calls against the same arithmetic
    # expressed in one call, to show whether calls are the unit of cost.
    def chain():
        x = a + b
        x = x * 1.01
        x = x + b
        return x * 0.5

    timed("4 elementwise calls, 4096 wide", chain, args.repeat, 4)


if __name__ == "__main__":
    main()
