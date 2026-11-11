"""Find the cheap way to get rotary tables for the positions at hand.

A decode step needs the rotation only for the position it is at, but the
table covering every position is large.  Selecting a row from it should cost
a read of that row; whether the accelerator actually does that is what this
measures, beside the cost of building the rotation for just the positions in
hand.

    python probe_rope.py [--repeat 200]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp

SIZE, HALF = 32768, 64


def bench(label, fn, repeat):
    for _ in range(5):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:44} {(time.time() - start) / repeat * 1e6:9.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    table = tp.randn(SIZE, HALF, device="mlu", dtype=tp.float32)
    inv_freq = tp.randn(HALF, device="mlu", dtype=tp.float32)
    row = tp.tensor([[20]], dtype=tp.int64, device="mlu")
    flat = row.reshape(-1)

    print("selecting one row out of 32768")
    bench("table[row]  (row is 1x1)", lambda: table[row], args.repeat)
    bench("table[flat] then reshape", lambda: table[flat].reshape(1, 1, HALF), args.repeat)
    bench("index_select(table, 0, flat)", lambda: tp.index_select(table, 0, flat), args.repeat)
    bench("embedding(row, table)", lambda: tp.nn.functional.embedding(flat, table), args.repeat)
    bench("table[20:21]", lambda: table[20:21], args.repeat)

    print("\nbuilding the rotation for the positions in hand")
    bench("outer(pos, inv_freq)", lambda: tp.outer(row.reshape(-1).float(), inv_freq),
          args.repeat)
    bench("cos of a 64-wide vector", lambda: tp.cos(flat.float() * inv_freq), args.repeat)
    bench("sin of a 64-wide vector", lambda: tp.sin(flat.float() * inv_freq), args.repeat)

    def direct():
        frequencies = tp.outer(flat.float(), inv_freq)
        return tp.cos(frequencies), tp.sin(frequencies)

    bench("outer + cos + sin (whole rotation)", direct, args.repeat)

    print("\ntable held at reduced extent")
    small = tp.randn(64, HALF, device="mlu", dtype=tp.float32)
    bench("table[20:21] of a 64-row table", lambda: small[20:21], args.repeat)


if __name__ == "__main__":
    main()
