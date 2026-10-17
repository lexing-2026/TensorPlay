#!/usr/bin/env python3
"""MPS operator benchmark for TensorPlay.

Runs the operators implemented by the MPS backend on the Metal device and
records per-op latency. The case inventory matches the coverage of the MPS
backend tests (test/test_mps.py) and executes entirely on MPS tensors.
Operators that are not (yet) supported on the device are probed once and
skipped, so a partially implemented backend still produces a complete,
comparable measurement set.

MPS dispatch is asynchronous, so every timed invocation synchronizes the
Metal command queue around the operation.

Usage: bench_mps_operator.py --suite short --reps 5 --json-out results.json
"""

import argparse
import faulthandler
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import tensorplay as tp
import tensorplay.functional as F
import tensorplay.nn.functional as pF

_ITEMSIZE = {"f32": 4, "f16": 2}
_VIEW_ONLY_PREFIXES = ("reshape", "split", "chunk", "diagonal", "unbind",
                       "broadcast-to", "transpose", "permute")


def _spec(kind, shape, seed=0, upper=None):
    return kind, tuple(shape), seed, upper


def _array(kind, shape, seed, upper):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)
    if kind == "positive":
        return (np.abs(rng.standard_normal(shape)) + 0.25).astype(np.float32)
    if kind == "greater_one":
        return (np.abs(rng.standard_normal(shape)) + 1.25).astype(np.float32)
    if kind == "bounded":
        return (np.tanh(rng.standard_normal(shape)) * 0.8).astype(np.float32)
    if kind == "integer":
        return rng.integers(0, upper, size=shape).astype(np.int64)
    if kind == "boolean":
        return (rng.random(shape) > 0.5).astype(np.bool_)
    if kind == "ones":
        return np.ones(shape, dtype=np.float32)
    raise ValueError(f"unknown input kind: {kind}")


def _factory(tp_dtype, operation, specs):
    def make():
        tensors = []
        for value in specs:
            kind, shape, seed, upper = value
            array = _array(kind, shape, seed, upper)
            if kind in ("integer", "boolean"):
                tensors.append(tp.tensor(array, device="mps"))
            else:
                tensors.append(tp.tensor(array, device="mps", dtype=tp_dtype))
        return lambda: operation(*tensors)

    return make


def _time(fn, reps, min_run_time, warmup):
    """Collect synchronized samples for at least the requested time and count."""
    for _ in range(warmup):
        fn()
        tp.mps.synchronize()
    samples = []
    total_time = 0.0
    while len(samples) < reps or total_time < min_run_time:
        tp.mps.synchronize()
        started = time.perf_counter()
        fn()
        tp.mps.synchronize()
        elapsed = time.perf_counter() - started
        samples.append(elapsed)
        total_time += elapsed
    return samples


def _moves_data(name):
    return not name.startswith(_VIEW_ONLY_PREFIXES)


def _estimate_bytes(specs, itemsize):
    """Streaming bytes for one invocation: read all inputs, write the
    largest-shaped output (single-pass approximation)."""
    if not specs:
        return 0
    shapes = [value[1] for value in specs]
    elements = sum(max(count, 1) for count in (int(np.prod(s)) for s in shapes))
    out = max(int(np.prod(s)) for s in shapes)
    return int((elements + out) * itemsize)


def _flops_matmul(specs):
    """Theoretical multiply-add count for a single matmul-like case."""
    for value in specs:
        shape = value[1]
        if len(shape) == 2:
            m, k = shape
            return 2 * m * k * k
        if len(shape) == 3:
            b, m, k = shape
            return 2 * b * m * k * k
    return 0


def build_cases(dtype, suite):
    tp_dtype = {"f32": tp.float32, "f16": tp.float16}[dtype]
    cases = []

    def add(name, category, operation, specs, *, long_only=False, flops=0):
        if long_only and suite == "short":
            return
        cases.append(
            (name, category, _factory(tp_dtype, operation, specs), specs, flops)
        )

    # Binary family: same-shape, broadcast, scalar, comparison and ternary.
    binary = _spec("normal", (8, 16, 32), 1)
    binary2 = _spec("normal", (8, 16, 32), 2)
    for name, operation in (
        ("add", lambda a, b: F.add(a, b)),
        ("sub", lambda a, b: F.sub(a, b)),
        ("mul", lambda a, b: F.mul(a, b)),
        ("div", lambda a, b: F.div(a, b + 1.0)),
        ("maximum", lambda a, b: F.maximum(a, b)),
        ("minimum", lambda a, b: F.minimum(a, b)),
    ):
        add(f"{name}[8x16x32]", "binary", operation, (binary, binary2))
    add("add-scalar[8x16x32]", "binary", lambda a: F.add(a, 1.25), (binary,))
    add("mul-scalar[8x16x32]", "binary", lambda a: F.mul(a, 1.25), (binary,))
    add("add-relu[8x16x32]", "binary",
        lambda a, b: pF.relu(F.add(a, b)), (binary, binary2))

    broadcast_a = _spec("normal", (32, 1, 64), 3)
    broadcast_b = _spec("normal", (1, 16, 64), 4)
    add("add-broadcast[32x1x64+1x16x64]", "binary",
        lambda a, b: F.add(a, b), (broadcast_a, broadcast_b))
    add("mul-broadcast[32x1x64*1x16x64]", "binary",
        lambda a, b: F.mul(a, b), (broadcast_a, broadcast_b))
    add("addcmul-broadcast[32x1x64]", "binary",
        lambda a, b, c: F.addcmul(a, b, c, value=0.5),
        (broadcast_a, broadcast_b, _spec("normal", (32, 16, 64), 5)))
    add("addcdiv-broadcast[32x1x64]", "binary",
        lambda a, b, c: F.addcdiv(a, b, c + 1.5, value=0.5),
        (broadcast_a, broadcast_b, _spec("positive", (32, 16, 64), 6)))

    for name, operation in (
        ("eq", lambda a, b: F.eq(a, b)),
        ("ne", lambda a, b: F.ne(a, b)),
        ("lt", lambda a, b: F.lt(a, b)),
        ("le", lambda a, b: F.le(a, b)),
        ("gt", lambda a, b: F.gt(a, b)),
        ("ge", lambda a, b: F.ge(a, b)),
    ):
        add(f"{name}[8x16x32]", "comparison", operation, (binary, binary2))

    # Unary family with domain-safe inputs.
    unary = (
        ("abs", F.abs, "normal"),
        ("neg", F.neg, "normal"),
        ("sqrt", F.sqrt, "positive"),
        ("rsqrt", F.rsqrt, "positive"),
        ("exp", F.exp, "bounded"),
        ("log", F.log, "positive"),
        ("sin", F.sin, "normal"),
        ("cos", F.cos, "normal"),
        ("tanh", F.tanh, "normal"),
        ("erf", F.erf, "normal"),
        ("floor", F.floor, "normal"),
        ("ceil", F.ceil, "normal"),
        ("round", F.round, "normal"),
        ("reciprocal", F.reciprocal, "positive"),
        ("square", F.square, "normal"),
        ("sign", F.sign, "normal"),
    )
    for offset, (name, operation, kind) in enumerate(unary):
        add(f"{name}[256k]", "unary", operation,
            (_spec(kind, (1 << 18,), 100 + offset),))
        add(f"{name}[1M]", "unary", operation,
            (_spec(kind, (1 << 20,), 200 + offset),), long_only=True)

    # Reductions and extrema.
    vector = _spec("normal", (1 << 20,), 300)
    for name, operation in (
        ("sum", lambda a: F.sum(a)),
        ("mean", lambda a: F.mean(a)),
        ("max", lambda a: F.max(a)),
        ("min", lambda a: F.min(a)),
        ("prod", lambda a: F.prod(a)),
        ("norm", lambda a: F.norm(a)),
    ):
        add(f"{name}[1M]", "reduction", operation, (vector,))
    matrix = _spec("normal", (1024, 1024), 310)
    add("sum-dim1[1024x1024]", "reduction",
        lambda a: F.sum(a, dim=1), (matrix,))
    add("mean-dim1[1024x1024]", "reduction",
        lambda a: F.mean(a, dim=1), (matrix,))
    add("max-dim1[1024x1024]", "reduction",
        lambda a: tp.max(a, dim=1), (matrix,))
    add("min-dim1[1024x1024]", "reduction",
        lambda a: tp.min(a, dim=1), (matrix,))
    add("argmax[1024x1024]", "reduction",
        lambda a: tp.argmax(a, dim=1), (matrix,))
    add("argmin[1024x1024]", "reduction",
        lambda a: tp.argmin(a, dim=1), (matrix,))
    add("logsumexp[1024x1024]", "reduction",
        lambda a: F.logsumexp(a, dim=1), (matrix,))
    bool_matrix = _spec("boolean", (1024, 1024), 320)
    add("all-dim1[1024x1024]", "reduction",
        lambda a: F.all(a, dim=1), (bool_matrix,))
    add("any-dim1[1024x1024]", "reduction",
        lambda a: F.any(a, dim=1), (bool_matrix,))

    # Matmul family (theoretical FLOP count attached).
    mm_a = _spec("normal", (256, 256), 400)
    mm_b = _spec("normal", (256, 256), 401)
    add("mm[256x256]", "matmul", lambda a, b: tp.mm(a, b),
        (mm_a, mm_b), flops=_flops_matmul([mm_a, mm_b]))
    bmm_a = _spec("normal", (16, 128, 128), 402)
    bmm_b = _spec("normal", (16, 128, 128), 403)
    add("bmm[16x128x128]", "matmul", lambda a, b: tp.bmm(a, b),
        (bmm_a, bmm_b), flops=_flops_matmul([bmm_a, bmm_b]))
    addmm_in = _spec("normal", (256, 256), 404)
    add("addmm[256x256]", "matmul",
        lambda a, b, c: tp.addmm(a, b, c),
        (addmm_in, mm_a, mm_b), flops=_flops_matmul([mm_a, mm_b]))
    mv_a = _spec("normal", (1024, 1024), 405)
    mv_v = _spec("normal", (1024,), 406)
    add("mv[1024x1024]", "matmul", lambda a, v: tp.mv(a, v),
        (mv_a, mv_v), flops=2 * 1024 * 1024)

    # Normalization, activations and convolution.
    add("softmax[1024x1024]", "normalization",
        lambda a: tp.softmax(a, dim=1), (matrix,))
    ln = _spec("normal", (64, 1024), 500)
    add("layer_norm[64x1024]", "normalization",
        lambda a: tp.layer_norm(a, [1024]), (ln,))
    bn = _spec("normal", (32, 64, 56, 56), 501)
    bn_w = _spec("ones", (64,), 502)
    add("batch_norm-eval[32x64x56x56]", "normalization",
        lambda a, w, b: tp.batch_norm(a, w, b, w, b, training=False,
                                      momentum=0.1, eps=1e-5),
        (bn, bn_w, bn_w), long_only=True)
    gn = _spec("normal", (8, 8, 64, 64), 503)
    add("group_norm[8x8x64x64]", "normalization",
        lambda a: tp.group_norm(a, 8, eps=1e-5), (gn,), long_only=True)

    conv_x = _spec("normal", (1, 64, 56, 56), 600)
    conv_w = _spec("normal", (64, 64, 3, 3), 601)
    add("conv2d[1x64x56x56]", "convolution",
        lambda a, w: tp.conv2d(a, w, None, stride=[1, 1], padding=[1, 1],
                               dilation=[1, 1], groups=1),
        (conv_x, conv_w), long_only=True)
    add("avg_pool2d[1x64x56x56]", "convolution",
        lambda a: tp.avg_pool2d(a, kernel_size=[2, 2], stride=[2, 2]),
        (conv_x,), long_only=True)
    add("max_pool2d[1x64x56x56]", "convolution",
        lambda a: tp.max_pool2d(a, kernel_size=[2, 2], stride=[2, 2]),
        (conv_x,), long_only=True)

    # Indexing, shape and embedding.
    cat_a = _spec("normal", (64, 256, 256), 700)
    add("cat[2x64x256x256]", "indexing",
        lambda a, b: tp.cat([a, b], dim=0), (cat_a, cat_a), long_only=True)
    add("stack[64x256x256]", "indexing",
        lambda a, b: tp.stack([a, b], dim=0), (cat_a, cat_a), long_only=True)
    add("transpose[64x256x256]", "indexing",
        lambda a: tp.transpose(a, 1, 2), (cat_a,), long_only=True)
    add("where[64x256x256]", "indexing",
        lambda a, b: tp.where(a > 0, a, b), (cat_a, cat_a), long_only=True)
    add("clamp[1M]", "indexing",
        lambda a: tp.clamp(a, -0.5, 0.5), (vector,))
    mask = _spec("boolean", (1 << 20,), 701)
    add("masked_fill[1M]", "indexing",
        lambda a, m: tp.masked_fill(a, m, tp.tensor(0.0, dtype=tp.float32)),
        (vector, mask))
    idx_1m = _spec("integer", (1 << 18,), 702, upper=1024)
    add("index_select[1M]", "indexing",
        lambda a, i: tp.index_select(a, 0, i),
        (_spec("normal", (1024, 1 << 18), 703), idx_1m), long_only=True)
    gather_x = _spec("normal", (1024, 1024), 704)
    gather_idx = _spec("integer", (1024, 1024), 705, upper=1024)
    add("gather[1024x1024]", "indexing",
        lambda a, i: tp.gather(a, 1, i), (gather_x, gather_idx),
        long_only=True)
    scatter_src = _spec("normal", (1024, 1024), 706)
    add("scatter[1024x1024]", "indexing",
        lambda a, i, s: tp.scatter(a, 1, i, s),
        (gather_x, gather_idx, scatter_src), long_only=True)
    emb_w = _spec("normal", (4096, 128), 707)
    emb_i = _spec("integer", (1024, 64), 708, upper=4096)
    add("embedding[4096x128]", "indexing",
        lambda w, i: tp.embedding(w, i), (emb_w, emb_i), long_only=True)

    # Triangular, scan, sort and bucketize (shader-backed MPS ops).
    tri = _spec("normal", (1024, 1024), 800)
    add("tril[1024x1024]", "triangular", lambda a: tp.tril(a), (tri,))
    add("triu[1024x1024]", "triangular", lambda a: tp.triu(a), (tri,))
    add("cumsum-dim1[1024x1024]", "scan",
        lambda a: tp.cumsum(a, dim=1), (tri,))
    add("cumprod-dim1[512x512]", "scan",
        lambda a: tp.cumprod(a, dim=1),
        (_spec("positive", (512, 512), 801),), long_only=True)
    add("sort-dim1[1024x1024]", "sort",
        lambda a: tp.sort(a, dim=1, descending=False), (tri,))
    add("argsort-dim1[1024x1024]", "sort",
        lambda a: tp.argsort(a, dim=1), (tri,))

    boundaries = _spec("integer", (1 << 16,), 802, upper=1 << 16)
    queries = _spec("integer", (1 << 20,), 803, upper=1 << 16)
    add("searchsorted[1M]", "bucketize",
        lambda b, q: tp.searchsorted(b, q, out_int32=False, right=False,
                                     side=None, sorter=None),
        (boundaries, queries))
    add("bucketize[1M]", "bucketize",
        lambda q, b: tp.bucketize(q, b, out_int32=False, right=False),
        (queries, boundaries))

    # Dropout and padding.
    add("dropout-train[1M]", "dropout",
        lambda a: tp.dropout(a, p=0.5, train=True), (vector,))
    pad_x = _spec("normal", (64, 64, 64), 804)
    add("constant_pad_nd[64x64x64]", "padding",
        lambda a: tp.constant_pad_nd(a, [2, 2, 2, 2], 0.0), (pad_x,))

    return cases


def main():
    faulthandler.enable()
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("short", "long", "all"),
                        default="short")
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--min-run-time", type=float, default=1.0,
                        help="minimum timed seconds per case")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3,
                        help="separate sampling passes over all cases")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("reps must be positive")
    if not math.isfinite(args.min_run_time) or args.min_run_time < 0:
        parser.error("min-run-time must be finite and non-negative")
    if args.warmup < 1:
        parser.error("warmup must be positive")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if not tp.mps.is_available():
        parser.error("MPS is not available on this machine")
    suite = "long" if args.suite == "all" else args.suite
    measurements = {}
    samples_by_name = {}
    skipped = set()
    cases = build_cases(args.dtype, suite)
    print(f"suite={suite} dtype={args.dtype} cases={len(cases)}")
    for round_index in range(args.rounds):
        print(f"round {round_index + 1}/{args.rounds}")
        # Rotate the starting case so each pass samples a different ordering.
        offset = round_index * len(cases) // args.rounds
        ordered_cases = cases[offset:] + cases[:offset]
        for name, category, factory, specs, flops in ordered_cases:
            if name in skipped:
                continue
            print(f"running {name}")
            fn = factory()
            # Probe support on the first pass; later failures must abort.
            if round_index == 0:
                try:
                    fn()
                    tp.mps.synchronize()
                except (NotImplementedError, RuntimeError) as exc:
                    print(f"{name:42} [skipped: {exc}]")
                    skipped.add(name)
                    del fn
                    continue
            samples = samples_by_name.setdefault(name, [])
            samples.extend(_time(fn, math.ceil(args.reps / args.rounds),
                                 args.min_run_time / args.rounds, args.warmup))
            seconds = min(samples)
            entry = {
                "name": name,
                "category": category,
                "input_shapes": [list(value[1]) for value in specs],
                "seconds": seconds,
                "sample_count": len(samples),
                "median_seconds": statistics.median(samples),
                "max_seconds": max(samples),
            }
            if flops:
                entry["flops"] = flops
            if _moves_data(name):
                entry["bytes_moved"] = _estimate_bytes(specs, _ITEMSIZE[args.dtype])
            print(f"{name:42} {seconds * 1e3:10.3f} ms "
                  f"({len(samples)} samples, median {statistics.median(samples) * 1e3:.3f} ms)")
            measurements[name] = entry
            del fn

    payload = {
        "schema_version": 1,
        "benchmark": "mps-operator",
        "suite": suite,
        "dtype": args.dtype,
        "device": "mps",
        "timing": {"statistic": "minimum", "min_reps": args.reps,
                   "min_run_time": args.min_run_time, "warmup": args.warmup,
                   "rounds": args.rounds},
        "measurements": list(measurements.values()),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8")
    print(f"wrote results: {args.json_out}")


if __name__ == "__main__":
    main()
