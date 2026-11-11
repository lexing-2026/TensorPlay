"""Find what makes a sequence of products slower than the sum of its parts.

One product repeated reaches a high rate; seven products of the shapes a block
uses, each reading its own weight, do not.  A sequence issued from the host can
be slow for two different reasons: the host takes a long time to hand each call
over, or the device takes a long time to run them once they arrive.  Both are
timed apart here, so the cause is named rather than guessed.

    python probe_seq.py [--repeat 30]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp

# The products a block issues: (contracted width, output width).
GROUPS = {
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
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    def timed(run, repeat):
        """Wall time per call, with the calls queued back to back."""
        for _ in range(3):
            run()
        sync()
        start = time.time()
        for _ in range(repeat):
            run()
        sync()
        return (time.time() - start) / repeat

    def split(run, repeat):
        """Time per call spent handing the calls over, and draining them."""
        for _ in range(3):
            run()
        sync()
        issuing = 0.0
        draining = 0.0
        for _ in range(repeat):
            mark = time.time()
            run()
            queued = time.time()
            sync()
            done = time.time()
            issuing += queued - mark
            draining += done - queued
        return issuing / repeat, draining / repeat

    def report(label, run, shapes, note=""):
        moved = sum(inner * out * 2 for inner, out in shapes.values())
        wall = timed(run, args.repeat)
        issuing, draining = split(run, args.repeat)
        print(f"  {label:44} wall {wall * 1e3:7.3f} ms  "
              f"host {issuing * 1e3:7.3f}  device {draining * 1e3:7.3f}  "
              f"({moved / wall / 1e9:6.1f} GB/s)  {note}")

    activations = {
        name: tp.randn(1, inner, device="mlu", dtype=tp.float16)
        for name, (inner, out) in GROUPS.items()
    }

    def separate():
        return {
            name: tp.randn(inner, out, device="mlu", dtype=tp.float16)
            for name, (inner, out) in GROUPS.items()
        }

    def arena():
        total = sum(inner * out for inner, out in GROUPS.values())
        flat = tp.empty(total, device="mlu", dtype=tp.float16)
        weights = {}
        offset = 0
        for name, (inner, out) in GROUPS.items():
            span = inner * out
            weights[name] = flat[offset:offset + span].view(inner, out)
            offset += span
        return weights

    weights = separate()
    arena_weights = arena()

    def sequence(table):
        return [activations[name] @ table[name] for name in GROUPS]

    print("a block's seven products")
    report("separate weights, fresh results",
           lambda: sequence(weights), GROUPS)
    report("one allocation, fresh results",
           lambda: sequence(arena_weights), GROUPS, "arena")

    print("\n  each shape on its own, issued the same number of times")
    for name, (inner, out) in GROUPS.items():
        weight = weights[name]
        activation = activations[name]
        report(f"{name} alone",
               lambda a=activation, w=weight: [a @ w], {name: (inner, out)})

    print("\n  the same product repeated, and distinct weights of the same size")
    wide = {
        f"w{index}": tp.randn(4096, 12288, device="mlu", dtype=tp.float16)
        for index in range(4)
    }
    activation = tp.randn(1, 4096, device="mlu", dtype=tp.float16)
    one = wide["w0"]
    four = {name: (4096, 12288) for name in wide}
    report("the same 4096x12288 weight four times",
           lambda: [activation @ one for _ in range(4)], four,
           "100 MB read four times")
    report("four distinct 4096x12288 weights",
           lambda: [activation @ wide[name] for name in wide], four)

    print("\n  seven products of one shape, weights together and apart")
    uniform = {name: (4096, 4096) for name in GROUPS}
    uniform_weights = {
        name: tp.randn(4096, 4096, device="mlu", dtype=tp.float16) for name in GROUPS
    }
    uniform_activations = {
        name: tp.randn(1, 4096, device="mlu", dtype=tp.float16) for name in GROUPS
    }
    flat = tp.empty(7 * 4096 * 4096, device="mlu", dtype=tp.float16)
    uniform_arena = {}
    for index, name in enumerate(GROUPS):
        start = index * 4096 * 4096
        uniform_arena[name] = flat[start:start + 4096 * 4096].view(4096, 4096)

    report("seven shapes, seven weights",
           lambda: [uniform_activations[name] @ uniform_weights[name]
                    for name in GROUPS], uniform)
    report("seven shapes, seven weights in one allocation",
           lambda: [uniform_activations[name] @ uniform_arena[name]
                    for name in GROUPS], uniform, "arena")
    report("seven shapes, one weight",
           lambda: [uniform_activations[name] @ uniform_weights["q"]
                    for name in GROUPS], uniform, "same addresses")


if __name__ == "__main__":
    main()
