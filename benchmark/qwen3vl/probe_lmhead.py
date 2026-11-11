"""Measure the output projection, which reads a large weight for one row.

One row is contracted against a weight whose output width is the vocabulary,
so a step reads well over a gigabyte to produce one row of results.  The rate
this reaches decides how much of the step it is, so the ways of expressing the
product and a few widths around the vocabulary are compared.

    python probe_lmhead.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp

VOCAB, HIDDEN = 151936, 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    row = tp.randn(1, HIDDEN, device="mlu", dtype=tp.float16)

    def bench(label, fn, weight):
        for _ in range(3):
            fn()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(args.repeat):
            fn()
        tp.mlu.synchronize()
        per_call = (time.time() - start) / args.repeat
        moved = weight.numel() * 2
        print(f"  {label:44} {per_call * 1e3:8.3f} ms  {moved / per_call / 1e9:6.1f} GB/s")

    weight = tp.randn(VOCAB, HIDDEN, device="mlu", dtype=tp.float16)
    print("express the same product")
    bench("x @ w.t()", lambda: row @ weight.t(), weight)
    bench("x.mm(w.t())", lambda: row.mm(weight.t()), weight)
    bench("tensorplay.linear(x, w)", lambda: tp.nn.functional.linear(row, weight), weight)
    bench("w @ x.t()", lambda: weight @ row.t(), weight)

    print("\nwidths around the vocabulary, padded and sliced back")
    for width in (VOCAB, 152064, 153600, 147456, 131072, 65536, 32768):
        if width == VOCAB:
            padded = weight
            take = None
        else:
            padded = tp.randn(width, HIDDEN, device="mlu", dtype=tp.float16)
            take = min(width, VOCAB)

        def run(padded=padded, take=take):
            out = row @ padded.t()
            return out if take is None else out[..., :take]

        bench(f"output width {width}", run, padded)
        if padded is not weight:
            del padded

    print("\nand the row reduction that follows it")
    logits = tp.randn(1, VOCAB, device="mlu", dtype=tp.float16)

    def bench_op(label, fn):
        for _ in range(5):
            fn()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(args.repeat):
            fn()
        tp.mlu.synchronize()
        print(f"  {label:44} {(time.time() - start) / args.repeat * 1e3:8.3f} ms")

    bench_op("logits.argmax(-1)", lambda: logits.argmax(dim=-1))
    bench_op("logits.max(-1).indices", lambda: logits.max(dim=-1).indices)
    bench_op("logits.softmax(-1)", lambda: logits.softmax(dim=-1))
    bench_op("logits.softmax(-1).argmax(-1)",
             lambda: logits.softmax(dim=-1).argmax(dim=-1))


if __name__ == "__main__":
    main()
