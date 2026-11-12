#!/usr/bin/env python3
"""Split the cost of the positions-reporting maximum.

The operator moves its positions through the host, so its time is the card's
call, the two transfers and the rewrite.  Each is timed on its own at the
sizes the stem shape produces.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp  # noqa: E402


def sync():
    tp.mlu.synchronize()


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    out = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    return statistics.median(out) * 1e3


def main() -> int:
    import tensorplay_mlu

    tensorplay_mlu.load()
    f = tp.nn.functional
    x = tp.randn(32, 64, 112, 112, device="mlu")

    # The positions of this shape: 32 x 64 x 56 x 56 int64 entries.
    pos = tp.zeros(32, 64, 56, 56, dtype=tp.int64, device="mlu")
    host = pos.to("cpu")

    print(f"  {'plain maximum':38s} {bench(lambda: f.max_pool2d(x, 3, stride=2, padding=1)):9.3f} ms")
    print(f"  {'maximum with a batch of 16':38s} "
          f"{bench(lambda: f.max_pool2d(x[:16], 3, stride=2, padding=1)):9.3f} ms")
    print(f"  {'positions to host':38s} {bench(lambda: pos.to('cpu')):9.3f} ms")
    print(f"  {'positions back to the card':38s} {bench(lambda: host.to('mlu')):9.3f} ms")
    print(f"  {'positions round trip':38s} "
          f"{bench(lambda: pos.to('cpu').to('mlu')):9.3f} ms")
    print(f"  {'values maximum with positions':38s} "
          f"{bench(lambda: f.max_pool2d_with_indices(x, 3, stride=2, padding=1)):9.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
