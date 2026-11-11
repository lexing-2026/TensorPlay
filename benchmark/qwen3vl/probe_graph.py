"""Check whether a step's operator sequence can be captured and replayed.

A decode step issues the same operators on the same shapes every token.  If the
backend can record that sequence once and replay it, the per-operator launch
cost disappears from the step.  This checks whether capture works on this
device at all, and what a replay of a block's operator sequence costs against
issuing it eagerly.

    python probe_graph.py [--repeat 50]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def bench(label, fn, repeat):
    for _ in range(5):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:40} {(time.time() - start) / repeat * 1e3:9.4f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    print("native graph support")
    print("  _C has CUDAGraph:", hasattr(tp._C, "CUDAGraph"))

    from tensorplay.accelerator.graphs import Graph

    a = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    b = tp.randn(4096, 4096, device="mlu", dtype=tp.float16)

    def body():
        x = a @ b
        x = x * 1.01
        return x + a

    try:
        graph = Graph()
        with tp.no_grad():
            for _ in range(3):
                body()
            tp.mlu.synchronize()
            with graph:
                out = body()
        tp.mlu.synchronize()
        print("  capture: ok")
    except Exception as exc:  # noqa: BLE001
        print("  capture failed:", type(exc).__name__, str(exc)[:200])
        return

    print("\nreplay against eager")
    bench("three operators, eager", body, args.repeat)
    bench("three operators, replayed", graph.replay, args.repeat)

    wide = tp.randn(1, 1, 12288, device="mlu", dtype=tp.float16)
    w1 = tp.randn(12288, 4096, device="mlu", dtype=tp.float16)
    w2 = tp.randn(4096, 12288, device="mlu", dtype=tp.float16)
    w3 = tp.randn(4096, 12288, device="mlu", dtype=tp.float16)

    def seven():
        q = a @ b
        k = a @ b
        v = a @ b
        o = a @ b
        g = wide @ w1
        u = wide @ w3
        d = (g * u) @ w2
        return q + k + v + o + d

    try:
        graph2 = Graph()
        with tp.no_grad():
            for _ in range(3):
                seven()
            tp.mlu.synchronize()
            with graph2:
                out2 = seven()
        tp.mlu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print("  second capture failed:", type(exc).__name__, str(exc)[:200])
        return

    bench("seven products plus combining, eager", seven, args.repeat)
    bench("seven products plus combining, replayed", graph2.replay, args.repeat)


if __name__ == "__main__":
    main()
