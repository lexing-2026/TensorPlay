"""Measure what the products a decode step issues actually reach.

A decode step reads every weight once, so the time it takes is that traffic
over the achieved rate and the rate matters more than anything else in the
step.  This reports the rate for each shape the tower uses, for the products
issued back to back the way a block issues them rather than one shape
repeated, and for the weight held in the other order.

    python probe_gemm.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp

# The products a block issues: (contracted width, output width).
SHAPES = {
    "q": (4096, 4096),
    "k": (4096, 1024),
    "v": (4096, 1024),
    "o": (4096, 4096),
    "gate": (4096, 12288),
    "up": (4096, 12288),
    "down": (12288, 4096),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    weights = {
        name: tp.randn(inner, out, device="mlu", dtype=tp.float16)
        for name, (inner, out) in SHAPES.items()
    }
    activations = {
        name: tp.randn(1, inner, device="mlu", dtype=tp.float16)
        for name, (inner, out) in SHAPES.items()
    }
    tp.mlu.synchronize()

    def timed(fn, repeat):
        for _ in range(3):
            fn()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(repeat):
            fn()
        tp.mlu.synchronize()
        return (time.time() - start) / repeat

    print("one product at a time, its weight streamed")
    total = 0.0
    for name, (inner, out) in SHAPES.items():
        weight, activation = weights[name], activations[name]
        moved = weight.numel() * 2
        per_call = timed(lambda: activation @ weight, args.repeat)
        total += per_call
        print(f"  {name:5} {inner:5}x{out:<6} {per_call * 1e3:7.3f} ms  "
              f"{moved / per_call / 1e9:7.1f} GB/s")
    print(f"  a block's seven products, summed: {total * 1e3:.3f} ms  "
          f"({total * 36 * 1e3:.2f} ms for the tower)")

    def layer():
        """The block's seven products, in the order a block issues them.

        Every result is kept alive so none of them is skipped or freed early.
        """
        return [activations[name] @ weights[name] for name in SHAPES]

    per_layer = timed(layer, max(1, args.repeat // 2))
    moved = sum(weights[name].numel() * 2 for name in SHAPES)
    print(f"\n  seven products back to back: {per_layer * 1e3:7.3f} ms  "
          f"{moved / per_layer / 1e9:7.1f} GB/s")
    print(f"  ({per_layer * 36 * 1e3:.2f} ms for the tower's products)")

    print("\nthe same product at wider activation batches")
    for name in ("q", "gate", "down"):
        inner, out = SHAPES[name]
        weight = weights[name]
        moved = weight.numel() * 2
        print(f"  {name} {inner}x{out}  (weight {moved / 1e6:.0f} MB)")
        for rows_act in (1, 2, 4, 8):
            activation = tp.randn(rows_act, inner, device="mlu", dtype=tp.float16)
            per_call = timed(lambda: activation @ weight, args.repeat)
            print(f"    activation {rows_act} rows: {per_call * 1e3:7.3f} ms  "
                  f"{moved / per_call / 1e9:7.1f} GB/s")
            del activation
        weight_other = weight.t().contiguous()
        activation = tp.randn(1, inner, device="mlu", dtype=tp.float16)
        per_call = timed(lambda: activation @ weight_other.t(), args.repeat)
        print(f"    weight held the other order: {per_call * 1e3:7.3f} ms  "
              f"{moved / per_call / 1e9:7.1f} GB/s")


if __name__ == "__main__":
    main()
