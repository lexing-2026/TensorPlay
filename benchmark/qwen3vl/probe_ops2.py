"""Compare the call forms of the operators a block issues.

The same operator can be reached as a method on the tensor, as a function in
the top-level namespace, or as a function in the functional namespace, and on
this device the three can cost very different amounts because the wrappers
around them do different work before the operator is dispatched.  This times
each form at the shapes a step uses.

    python probe_ops2.py [--repeat 500]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def bench(label, fn, repeat):
    for _ in range(10):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:48} {(time.time() - start) / repeat * 1e6:8.1f} us")


def compare(repeat, label, forms):
    """Time every form of one operator and name the cheapest."""
    print(f"\n{label}")
    usable = []
    for name, fn in forms:
        try:
            fn()
            tp.mlu.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:48} refused ({type(exc).__name__}: {str(exc)[:60]})")
            continue
        usable.append((name, fn))
    if not usable:
        return
    times = {}
    for name, fn in usable:
        for _ in range(10):
            fn()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(repeat):
            fn()
        tp.mlu.synchronize()
        times[name] = (time.time() - start) / repeat * 1e6
    best = min(times.values())
    for name, value in times.items():
        print(f"  {name:48} {value:8.1f} us   {value / best:5.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=500)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    row = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    wide = tp.randn(1, 1, 12288, device="mlu", dtype=tp.float16)
    scores = tp.randn(1, 8, 4, 512, device="mlu", dtype=tp.float16)
    index = tp.tensor([20], dtype=tp.int64, device="mlu")
    table = tp.randn(32768, 64, device="mlu", dtype=tp.float32)
    half = tp.randn(1, 1, 64, device="mlu", dtype=tp.float16)
    weight = tp.randn(4096, device="mlu", dtype=tp.float16)

    compare(args.repeat, "softmax over 512", [
        ("tp.nn.functional.softmax(x, -1)", lambda: tp.nn.functional.softmax(scores, -1)),
        ("tp.softmax(x, -1)", lambda: tp.softmax(scores, -1)),
        ("x.softmax(-1)", lambda: scores.softmax(-1)),
    ])

    compare(args.repeat, "row mean over 4096", [
        ("tp.mean(x, -1, keepdim=True)", lambda: tp.mean(row, -1, keepdim=True)),
        ("x.mean(-1, keepdim=True)", lambda: row.mean(-1, keepdim=True)),
    ])

    compare(args.repeat, "reciprocal square root", [
        ("tp.rsqrt(x)", lambda: tp.rsqrt(row)),
        ("x.rsqrt()", lambda: row.rsqrt()),
    ])

    compare(args.repeat, "sigmoid over 12288", [
        ("tp.sigmoid(x)", lambda: tp.sigmoid(wide)),
        ("x.sigmoid()", lambda: wide.sigmoid()),
    ])

    silu = tp.nn.SiLU()
    compare(args.repeat, "silu over 12288", [
        ("tp.nn.functional.silu(x)", lambda: tp.nn.functional.silu(wide)),
        ("nn.SiLU()(x)", lambda: silu(wide)),
        ("x * x.sigmoid()", lambda: wide * wide.sigmoid()),
    ])

    compare(args.repeat, "gather a row from the rotary table", [
        ("tp.index_select(table, 0, i)", lambda: tp.index_select(table, 0, index)),
        ("table.index_select(0, i)", lambda: table.index_select(0, index)),
        ("tp.nn.functional.embedding(i, table)",
         lambda: tp.nn.functional.embedding(index, table)),
        ("tp.embedding(i, table)", lambda: tp.embedding(index, table)),
    ])

    compare(args.repeat, "multiply by a row weight", [
        ("x * w", lambda: row * weight),
        ("tp.mul(x, w)", lambda: tp.mul(row, weight)),
        ("x.mul(w)", lambda: row.mul(weight)),
    ])

    compare(args.repeat, "element type round trip", [
        ("x.to(tp.float32)", lambda: row.to(tp.float32)),
        ("tp.to(x, tp.float32)", lambda: tp.to(row, tp.float32)),
        ("x.float()", lambda: row.float()),
    ])

    compare(args.repeat, "join two 64-wide rows", [
        ("tp.cat([a, b], -1)", lambda: tp.cat([half, half], -1)),
        ("tp.concatenate([a, b], -1)", lambda: tp.concatenate([half, half], -1)),
        ("tp.stack([a, b], -1)", lambda: tp.stack([half, half], -1)),
    ])

    compare(args.repeat, "an empty buffer", [
        ("tp.empty(1, 1, 4096)", lambda: tp.empty(1, 1, 4096, device="mlu",
                                                  dtype=tp.float16)),
        ("tp.empty_like(x)", lambda: tp.empty_like(row)),
    ])

    compare(args.repeat, "write a slice", [
        ("out[..., :64].copy_(y)", lambda: tp.empty(1, 1, 128, device="mlu",
                                                    dtype=tp.float16)[..., :64].copy_(half)),
        ("out[..., :64] = y", lambda: slice_write()),
    ])

    compare(args.repeat, "vector-matrix product, 4096 x 4096", [
        ("x @ w", lambda: row @ weight_big(4096)),
        ("tp.mm(x, w)", lambda: tp.mm(row, weight_big(4096))),
        ("tp.matmul(x, w)", lambda: tp.matmul(row, weight_big(4096))),
        ("x.mm(w)", lambda: row.mm(weight_big(4096))),
    ])


WEIGHTS: dict[int, object] = {}


def weight_big(size):
    if size not in WEIGHTS:
        WEIGHTS[size] = tp.randn(size, size, device="mlu", dtype=tp.float16)
    return WEIGHTS[size]


def slice_write():
    out = tp.empty(1, 1, 128, device="mlu", dtype=tp.float16)
    half = tp.randn(1, 1, 64, device="mlu", dtype=tp.float16)
    out[..., :64] = half
    return out


if __name__ == "__main__":
    main()
