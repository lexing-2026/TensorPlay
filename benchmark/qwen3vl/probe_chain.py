"""Find why a product costs more when its input is another product.

Repeating one product on a fixed input measures the product.  Chaining the
product into itself measures what a step really pays, because in a step every
product's input is the previous operator's output.  This compares the two at
several sizes and tries the plausible causes: keeping the intermediates
alive, the operand order, and the call form.

    python probe_chain.py [--length 20]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def timed(fn, length, repeat):
    for _ in range(2):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    return (time.time() - start) / (repeat * length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    print("chained against repeated, by size")
    for size in (128, 512, 1024, 2048, 4096):
        w = tp.randn(size, size, device="mlu", dtype=tp.float16)
        seed = tp.randn(1, size, device="mlu", dtype=tp.float16)

        def chained():
            x = seed
            for _ in range(args.length):
                x = x @ w
            return x

        def repeated():
            x = seed
            out = []
            for _ in range(args.length):
                out.append(x @ w)
            return out[-1]

        def kept():
            x = seed
            out = []
            for _ in range(args.length):
                x = x @ w
                out.append(x)
            return out

        per = f"{size}x{size}"
        print(f"  {per:10} chained {timed(chained, args.length, args.repeat) * 1e6:8.2f} us   "
              f"repeated {timed(repeated, args.length, args.repeat) * 1e6:8.2f} us   "
              f"chained-but-kept {timed(kept, args.length, args.repeat) * 1e6:8.2f} us")
        del w, seed

    print("\ncall forms, 4096x4096")
    w = tp.randn(4096, 4096, device="mlu", dtype=tp.float16)
    seed = tp.randn(1, 4096, device="mlu", dtype=tp.float16)

    def via_matmul():
        x = seed
        for _ in range(args.length):
            x = tp.matmul(x, w)
        return x

    def via_mm():
        x = seed
        for _ in range(args.length):
            x = tp.mm(x, w)
        return x

    def with_shape_3d():
        x = seed.reshape(1, 1, 4096)
        for _ in range(args.length):
            x = x @ w
        return x

    def with_contiguous():
        x = seed
        for _ in range(args.length):
            x = (x @ w).contiguous()
        return x

    def with_add():
        x = seed
        for _ in range(args.length):
            x = x @ w
            x = x + seed
        return x

    def half_then_wide():
        x = seed
        for _ in range(args.length):
            x = (x @ w) @ w
        return x

    print(f"  x @ w            {timed(lambda: (lambda: None)() or via_chain(seed, w, args.length), args.length, args.repeat) * 1e6:8.2f} us")
    print(f"  tensorplay.matmul {timed(via_matmul, args.length, args.repeat) * 1e6:8.2f} us")
    print(f"  tensorplay.mm     {timed(via_mm, args.length, args.repeat) * 1e6:8.2f} us")
    print(f"  three-dimensional {timed(with_shape_3d, args.length, args.repeat) * 1e6:8.2f} us")
    print(f"  contiguous after  {timed(with_contiguous, args.length, args.repeat) * 1e6:8.2f} us")
    print(f"  an add after each {timed(with_add, args.length, args.repeat) * 1e6:8.2f} us")
    print(f"  two products each {timed(half_then_wide, args.length, args.repeat) * 1e6:8.2f} us")


def via_chain(seed, w, length):
    x = seed
    for _ in range(length):
        x = x @ w
    return x


if __name__ == "__main__":
    main()
