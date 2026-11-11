"""Separate the cost of issuing an operator from the cost of running one.

A step that issues many small operators pays two different prices: the host
builds and submits each call, and the device runs each one.  If the calls are
independent the host can run ahead and only the larger of the two shows.  If
each call consumes the previous result the device latency is exposed.  This
measures a long chain of dependent calls beside the same number of
independent ones, at several widths.

    python probe_latency2.py [--length 100]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def timed(make, length, repeat):
    for _ in range(3):
        make()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        make()
    tp.mlu.synchronize()
    return (time.time() - start) / (repeat * length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    for width in (1, 128, 4096):
        a = tp.randn(1, 1, width, device="mlu", dtype=tp.float16)
        b = tp.randn(1, 1, width, device="mlu", dtype=tp.float16)

        def dependent():
            x = a
            for _ in range(args.length):
                x = x + b
            return x

        def independent():
            return [a + b for _ in range(args.length)]

        print(f"width {width}")
        print(f"  dependent adds   {timed(dependent, args.length, args.repeat) * 1e6:8.2f} us per call")
        print(f"  independent adds {timed(independent, args.length, args.repeat) * 1e6:8.2f} us per call")

    a = tp.randn(1, 128, device="mlu", dtype=tp.float16)
    w = tp.randn(128, 128, device="mlu", dtype=tp.float16)
    big = tp.randn(1, 4096, device="mlu", dtype=tp.float16)
    big_w = tp.randn(4096, 4096, device="mlu", dtype=tp.float16)

    def dependent_matmul():
        x = a
        for _ in range(args.length):
            x = x @ w
        return x

    def independent_matmul():
        return [a @ w for _ in range(args.length)]

    def dependent_big():
        x = big
        for _ in range(args.length):
            x = x @ big_w
        return x

    print("products")
    print(f"  128x128 dependent   {timed(dependent_matmul, args.length, args.repeat) * 1e6:8.2f} us per call")
    print(f"  128x128 independent {timed(independent_matmul, args.length, args.repeat) * 1e6:8.2f} us per call")
    print(f"  4096x4096 dependent {timed(dependent_big, 20, 3) * 1e6:8.2f} us per call "
          f"(20 calls per measurement)")


if __name__ == "__main__":
    main()
