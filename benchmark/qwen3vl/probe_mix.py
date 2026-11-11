"""Measure what a block's products lose when they differ from one another.

The same product repeated reaches a high rate, but a block issues seven
products of four different shapes reading seven different weights, and that
sequence runs slower than the sum of its parts.  This separates the causes:
whether the weights differ, whether the shapes differ, whether the results are
allocated afresh each time, and whether grouping the projections that share an
input into one wider product recovers the rate.

    python probe_mix.py [--repeat 30]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp

# The products a block issues, in call order: (contracted width, output width).
GROUPS = {
    "q": (4096, 4096),
    "k": (4096, 1024),
    "v": (4096, 1024),
    "o": (4096, 4096),
    "gate": (4096, 12288),
    "up": (4096, 12288),
    "down": (12288, 4096),
}

# The same work with the projections that share an input issued as one product.
FUSED = {
    "qkv": (4096, 6144),
    "o": (4096, 4096),
    "gate_up": (4096, 24576),
    "down": (12288, 4096),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    def setup(shapes):
        weights = {
            name: tp.randn(inner, out, device="mlu", dtype=tp.float16)
            for name, (inner, out) in shapes.items()
        }
        activations = {
            name: tp.randn(1, inner, device="mlu", dtype=tp.float16)
            for name, (inner, out) in shapes.items()
        }
        return weights, activations

    def timed(run, repeat):
        for _ in range(3):
            run()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(repeat):
            run()
        tp.mlu.synchronize()
        return (time.time() - start) / repeat

    results = {}

    def report(label, per_call, shapes):
        moved = sum(inner * out * 2 for inner, out in shapes.values())
        results[label] = per_call
        print(f"  {label:52} {per_call * 1e3:7.3f} ms  "
              f"{moved / per_call / 1e9:6.1f} GB/s"
              f"  ({per_call * 36 * 1e3:6.2f} ms for the tower)")

    weights, activations = setup(GROUPS)

    def block():
        return [activations[name] @ weights[name] for name in GROUPS]

    report("seven products, four shapes", timed(block, args.repeat), GROUPS)

    buffers = {
        name: tp.empty(1, out, device="mlu", dtype=tp.float16)
        for name, (inner, out) in GROUPS.items()
    }

    def block_into_buffer():
        for name in GROUPS:
            tp.mm(activations[name], weights[name], out=buffers[name])
        return buffers

    report("seven products, four shapes, written into buffers",
           timed(block_into_buffer, args.repeat), GROUPS)

    uniform = {name: (4096, 4096) for name in GROUPS}
    same_weights, same_activations = setup(uniform)
    uniform_buffers = {
        name: tp.empty(1, 4096, device="mlu", dtype=tp.float16) for name in GROUPS
    }

    def uniform_weights():
        return [same_activations[name] @ same_weights[name] for name in GROUPS]

    report("seven products, one shape, seven weights",
           timed(uniform_weights, args.repeat), uniform)

    def uniform_into_buffer():
        for name in GROUPS:
            tp.mm(same_activations[name], same_weights[name], out=uniform_buffers[name])
        return uniform_buffers

    report("seven products, one shape, written into buffers",
           timed(uniform_into_buffer, args.repeat), uniform)

    def one_weight():
        weight = same_weights["q"]
        activation = same_activations["q"]
        return [activation @ weight for _ in range(7)]

    report("seven products, one shape, one weight",
           timed(one_weight, args.repeat), uniform)

    fused_weights, fused_activations = setup(FUSED)

    def fused_block():
        return [fused_activations[name] @ fused_weights[name] for name in FUSED]

    report("four products, grouped", timed(fused_block, args.repeat), FUSED)

    print(f"\n  the plain sequence moves "
          f"{sum(i * o * 2 for i, o in GROUPS.values()) / 1e6:.0f} MB, "
          f"the grouped one {sum(i * o * 2 for i, o in FUSED.values()) / 1e6:.0f} MB")

    print("\n  one shape at a time, isolated")
    for name, (inner, out) in GROUPS.items():
        per_call = timed(
            lambda name=name: activations[name] @ weights[name], args.repeat
        )
        moved = inner * out * 2
        print(f"    {name:8} {inner:5}x{out:<6} {per_call * 1e3:7.3f} ms  "
              f"{moved / per_call / 1e9:6.1f} GB/s")

    print("\n  one grouped product at a time, isolated")
    for name, (inner, out) in FUSED.items():
        per_call = timed(
            lambda name=name: fused_activations[name] @ fused_weights[name],
            args.repeat,
        )
        moved = inner * out * 2
        print(f"    {name:8} {inner:5}x{out:<6} {per_call * 1e3:7.3f} ms  "
              f"{moved / per_call / 1e9:6.1f} GB/s")


if __name__ == "__main__":
    main()
