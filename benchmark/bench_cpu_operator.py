#!/usr/bin/env python3
"""CPU operator benchmark for TensorPlay.

The case inventory covers the standard operator benchmark families,
but the implementation is self-contained and executes TensorPlay only.  A
case is materialized immediately before it runs, so large cases do not keep a
second full suite of tensors alive.  Any unsupported operator raises and fails
the job; there is no reference-framework fallback or silent skip path.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tensorplay as tp
import tensorplay.functional as F
import tensorplay.nn.functional as pF


def _spec(kind, shape, seed=0, upper=None):
    return kind, tuple(shape), seed, upper


def _array(dtype, value):
    kind, shape, seed, upper = value
    rng = np.random.default_rng(seed)
    if kind == "normal":
        data = rng.standard_normal(shape) * 0.5
        return data.astype(dtype)
    if kind == "positive":
        data = np.abs(rng.standard_normal(shape)) + 0.25
        return data.astype(dtype)
    if kind == "greater_one":
        data = np.abs(rng.standard_normal(shape)) + 1.25
        return data.astype(dtype)
    if kind == "bounded":
        data = np.tanh(rng.standard_normal(shape)) * 0.8
        return data.astype(dtype)
    if kind == "integer":
        return rng.integers(0, upper, size=shape, dtype=np.int64)
    if kind == "offsets":
        return np.arange(0, shape[0] * upper, upper, dtype=np.int64)
    if kind == "boolean":
        return (rng.random(shape) > 0.5).astype(np.bool_)
    if kind == "ones":
        return np.ones(shape, dtype=dtype)
    if kind == "zeros":
        return np.zeros(shape, dtype=dtype)
    raise ValueError(f"unknown input kind: {kind}")


def _factory(dtype, tp_dtype, operation, specs):
    def make():
        tensors = []
        for value in specs:
            array = _array(dtype, value)
            if value[0] in ("integer", "offsets", "boolean"):
                tensors.append(tp.tensor(array))
            else:
                tensors.append(tp.tensor(array, dtype=tp_dtype))
        return lambda: operation(*tensors)

    return make


def _time(fn, reps):
    fn()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return min(samples)


def _count_flops(fn):
    """One profiled invocation; returns the op FLOP count (0 when unknown)."""
    try:
        from tensorplay import profiler as prof

        with prof.profile(activities=[prof.ProfilerActivity.CPU],
                          with_flops=True) as session:
            fn()
        return int(sum(event.flops for event in session.key_averages()))
    except Exception:
        return 0


_ITEMSIZE = {"f32": 4, "f64": 8, "bf16": 2}

# Zero-copy view operators and short-circuit reductions touch no data (or
# a data-independent subset), so a streaming-bytes estimate would be wrong.
_VIEW_ONLY_PREFIXES = (
    "reshape", "split", "chunk", "diagonal", "diag", "unbind", "broadcast-to",
)
_EARLY_EXIT_NAMES = (
    "all[1024x1024]", "any[1024x1024]",
    "all-dim1[1024x1024]", "any-dim1[1024x1024]",
)


def _moves_data(name):
    if name in _EARLY_EXIT_NAMES:
        return False
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


def build_cases(dtype, suite):
    # ``dtype`` is the benchmark's dtype name; the numpy storage type and the
    # framework tensor type differ for bfloat16, which numpy has no native
    # element type for.
    np_dtype = np.float32 if dtype in ("f32", "bf16") else np.float64
    tp_dtype = {"f32": tp.float32, "f64": tp.float64, "bf16": tp.bfloat16}[dtype]
    cases = []

    def add(name, category, operation, specs, *, long_only=False):
        if long_only and suite == "short":
            return
        cases.append(
            (name, category, _factory(np_dtype, tp_dtype, operation, specs), specs)
        )

    # Binary family: same-shape, broadcast,
    # scalar, comparison, logical and ternary arithmetic operators.
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
    add("add-scalar[8x16x32]", "binary",
        lambda a: F.add(a, 1.25), (binary,))
    add("mul-scalar[8x16x32]", "binary",
        lambda a: F.mul(a, 1.25), (binary,))
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
    boolean_a = _spec("boolean", (8, 16, 32), 7)
    boolean_b = _spec("boolean", (8, 16, 32), 8)
    for name, operation in (
        ("logical-and", lambda a, b: F.logical_and(a, b)),
        ("logical-or", lambda a, b: F.logical_or(a, b)),
        ("logical-xor", lambda a, b: F.logical_xor(a, b)),
    ):
        add(f"{name}[8x16x32]", "comparison", operation,
            (boolean_a, boolean_b))
    add("fmod[256k]", "binary",
        lambda a, b: F.fmod(a + 2.0, b + 1.5),
        (_spec("normal", (1 << 18,), 9), _spec("positive", (1 << 18,), 10)))
    add("remainder[256k]", "binary",
        lambda a, b: F.remainder(a + 2.0, b + 1.5),
        (_spec("normal", (1 << 18,), 11), _spec("positive", (1 << 18,), 12)))

    # Unary family.  Inputs are domain-safe per operator, with
    # domain-aware data generation.
    unary = (
        ("abs", F.abs, "normal"),
        ("acos", F.acos, "bounded"),
        ("acosh", F.acosh, "greater_one"),
        ("asin", F.asin, "bounded"),
        ("asinh", F.asinh, "normal"),
        ("atan", F.atan, "normal"),
        ("atanh", F.atanh, "bounded"),
        ("ceil", F.ceil, "normal"),
        ("cos", F.cos, "normal"),
        ("cosh", F.cosh, "bounded"),
        ("erf", F.erf, "normal"),
        ("erfc", F.erfc, "normal"),
        ("exp", F.exp, "bounded"),
        ("expm1", F.expm1, "bounded"),
        ("floor", F.floor, "normal"),
        ("frac", F.frac, "normal"),
        ("log", F.log, "positive"),
        ("log10", F.log10, "positive"),
        ("log1p", F.log1p, "bounded"),
        ("log2", F.log2, "positive"),
        ("neg", F.neg, "normal"),
        ("reciprocal", F.reciprocal, "positive"),
        ("round", F.round, "normal"),
        ("rsqrt", F.rsqrt, "positive"),
        ("sign", F.sign, "normal"),
        ("sin", F.sin, "normal"),
        ("sinh", F.sinh, "bounded"),
        ("sqrt", F.sqrt, "positive"),
        ("square", F.square, "normal"),
        ("tan", F.tan, "bounded"),
        ("tanh", F.tanh, "normal"),
        ("trunc", F.trunc, "normal"),
    )
    for offset, (name, operation, kind) in enumerate(unary):
        add(f"{name}[256k]", "unary", operation,
            (_spec(kind, (1 << 18,), 100 + offset),))
        add(f"{name}[1M]", "unary", operation,
            (_spec(kind, (1 << 20,), 200 + offset),), long_only=True)

    # Reductions, including dim reductions and boolean reductions.
    vector = _spec("normal", (1 << 20,), 300)
    for name, operation in (
        ("sum", lambda a: F.sum(a)),
        ("mean", lambda a: F.mean(a)),
        ("max", lambda a: F.max(a)),
        ("min", lambda a: F.min(a)),
        ("prod", lambda a: F.prod(a)),
        ("norm", lambda a: F.norm(a)),
        ("median", lambda a: F.median(a)),
    ):
        add(f"{name}[1M]", "reduction", operation, (vector,))
    matrix = _spec("normal", (1024, 1024), 301)
    for name, operation in (
        ("sum-dim0", lambda a: F.sum(a, dim=0)),
        ("sum-dim1", lambda a: F.sum(a, dim=1)),
        ("mean-dim0", lambda a: F.mean(a, dim=0)),
        ("mean-dim1", lambda a: F.mean(a, dim=1)),
        ("max-dim1", lambda a: F.max(a, dim=1)[0]),
        ("min-dim1", lambda a: F.min(a, dim=1)[0]),
        ("norm-dim1", lambda a: F.norm(a, dim=1)),
        ("argmax-dim1", lambda a: F.argmax(a, dim=1)),
        ("argmin-dim1", lambda a: F.argmin(a, dim=1)),
        ("cumsum-dim1", lambda a: F.cumsum(a, dim=1)),
    ):
        add(f"{name}[1024x1024]", "reduction", operation, (matrix,))
    add("sum-dim0[2048x2048]", "reduction",
        lambda a: F.sum(a, dim=0), (_spec("normal", (2048, 2048), 302),),
        long_only=True)
    add("sum-dim1[2048x2048]", "reduction",
        lambda a: F.sum(a, dim=1), (_spec("normal", (2048, 2048), 303),),
        long_only=True)
    boolean_matrix = _spec("boolean", (1024, 1024), 304)
    add("all[1024x1024]", "reduction", lambda a: F.all(a), (boolean_matrix,))
    add("any[1024x1024]", "reduction", lambda a: F.any(a), (boolean_matrix,))
    add("all-dim1[1024x1024]", "reduction",
        lambda a: F.all(a, dim=1), (boolean_matrix,))
    add("any-dim1[1024x1024]", "reduction",
        lambda a: F.any(a, dim=1), (boolean_matrix,))
    add("amax[1M]", "reduction", lambda a: F.amax(a), (vector,))
    add("amin[1M]", "reduction", lambda a: F.amin(a), (vector,))
    add("logsumexp[1024x1024]", "reduction",
        lambda a: F.logsumexp(a, dim=1), (matrix,))
    add("cumprod[1024x1024]", "reduction",
        lambda a: F.cumprod(a, dim=1), (matrix,))
    add("flip[1024x1024]", "reduction",
        lambda a: F.flip(a, [0]), (matrix,))
    add("roll[1024x1024]", "reduction",
        lambda a: F.roll(a, 1, 0), (matrix,))
    add("kthvalue[1024x1024]", "reduction",
        lambda a: F.kthvalue(a, 3, dim=1)[0], (matrix,))
    add("mode[1024x1024]", "reduction",
        lambda a: F.mode(a, dim=1)[0], (matrix,))

    # Matrix multiplication family used by addmm/mm/matmul/bmm tests.
    for index, (m, k, n) in enumerate(((128, 256, 128), (256, 256, 256))):
        a = _spec("normal", (m, k), 400 + index)
        b = _spec("normal", (k, n), 410 + index)
        add(f"mm[{m}x{k}x{n}]", "matmul", lambda x, y: F.mm(x, y), (a, b))
        add(f"matmul[{m}x{k}x{n}]", "matmul",
            lambda x, y: F.matmul(x, y), (a, b))
    add("mm[1024x1024x1024]", "matmul",
        lambda x, y: F.mm(x, y),
        (_spec("normal", (1024, 1024), 420),
         _spec("normal", (1024, 1024), 421)), long_only=True)
    add("mv[1024x1024]", "matmul",
        lambda x, y: F.mv(x, y),
        (_spec("normal", (1024, 1024), 422),
         _spec("normal", (1024,), 423)))
    add("dot[1M]", "matmul", lambda x, y: F.dot(x, y),
        (_spec("normal", (1 << 20,), 424),
         _spec("normal", (1 << 20,), 425)))
    add("inner[4096x128]", "matmul", lambda x, y: F.inner(x, y),
        (_spec("normal", (4096, 128), 426),
         _spec("normal", (4096, 128), 427)))
    add("outer[4096]", "matmul", lambda x, y: F.outer(x, y),
        (_spec("normal", (4096,), 428), _spec("normal", (4096,), 429)))
    a = _spec("normal", (256, 256), 430)
    b = _spec("normal", (256, 256), 431)
    c = _spec("normal", (256, 256), 432)
    add("addmm[256]", "matmul",
        lambda bias, x, y: F.addmm(bias, x, y, beta=1.0, alpha=0.5),
        (c, a, b))
    batch1 = _spec("normal", (16, 64, 64), 433)
    batch2 = _spec("normal", (16, 64, 64), 434)
    add("bmm[16x64x64]", "matmul", lambda x, y: F.bmm(x, y),
        (batch1, batch2))
    batch_bias = _spec("normal", (16, 64, 64), 435)
    add("baddbmm[16x64x64]", "matmul",
        lambda bias, x, y: F.baddbmm(bias, x, y, beta=1.0, alpha=0.5),
        (batch_bias, batch1, batch2))
    addbmm_bias = _spec("normal", (64, 64), 436)
    add("addbmm[16x64x64]", "matmul",
        lambda bias, x, y: F.addbmm(bias, x, y, beta=1.0, alpha=0.5),
        (addbmm_bias, batch1, batch2))
    add("linear[64x256x512]", "matmul",
        lambda x, weight, bias: pF.linear(x, weight, bias),
        (_spec("normal", (64, 256), 437),
         _spec("normal", (512, 256), 438),
         _spec("zeros", (512,), 439)))
    add("einsum-bmm[16x64x64]", "matmul",
        lambda x, y: F.einsum("bij,bjk->bik", x, y), (batch1, batch2))

    # Activations, softmax and normalization families.
    activation_input = _spec("normal", (1 << 18,), 500)
    activations = (
        ("relu", lambda a: pF.relu(a)),
        ("relu6", lambda a: pF.relu6(a)),
        ("gelu", lambda a: pF.gelu(a)),
        ("gelu-tanh", lambda a: pF.gelu(a, approximate="tanh")),
        ("silu", lambda a: pF.silu(a)),
        ("hardswish", lambda a: pF.hardswish(a)),
        ("hardsigmoid", lambda a: pF.hardsigmoid(a)),
        ("leaky-relu", lambda a: pF.leaky_relu(a, negative_slope=0.1)),
        ("elu", lambda a: pF.elu(a, alpha=1.0)),
        ("celu", lambda a: pF.celu(a, alpha=1.0)),
        ("selu", lambda a: pF.selu(a)),
        ("mish", lambda a: pF.mish(a)),
        ("softplus", lambda a: pF.softplus(a)),
        ("softsign", lambda a: pF.softsign(a)),
        ("tanhshrink", lambda a: pF.tanhshrink(a)),
        ("hardtanh", lambda a: pF.hardtanh(a)),
        ("rrelu", lambda a: pF.rrelu(a)),
        ("hardshrink", lambda a: F.hardshrink(a, 0.5)),
        ("softshrink", lambda a: F.softshrink(a, 0.5)),
        ("threshold", lambda a: F.threshold(a, 0.5, 0.0)),
        ("digamma", lambda a: F.digamma(a + 2.0)),
        ("erfinv", lambda a: F.erfinv(a * 0.5)),
        ("lgamma", lambda a: F.lgamma(a + 2.0)),
        ("logit", lambda a: F.logit(a * 0.5 + 0.5)),
        ("sgn", lambda a: F.sgn(a)),
        ("nan_to_num", lambda a: F.nan_to_num(a)),
    )
    for index, (name, operation) in enumerate(activations):
        add(f"{name}[256k]", "activation", operation,
            (_spec("normal", (1 << 18,), 501 + index),))
    prelu_weight = _spec("normal", (1,), 514)
    add("prelu[256k]", "activation",
        lambda a, w: pF.prelu(a, w),
        (_spec("normal", (1 << 18,), 515), prelu_weight))
    norm_input = _spec("normal", (1024, 1024), 520)
    add("softmax[1024x1024]", "normalization",
        lambda a: F.softmax(a, dim=-1), (norm_input,))
    add("log-softmax[1024x1024]", "normalization",
        lambda a: F.log_softmax(a, dim=-1), (norm_input,))
    add("softmax[256x256x256]", "normalization",
        lambda a: F.softmax(a, dim=-1),
        (_spec("normal", (256, 256, 256), 521),))
    affine = _spec("ones", (1024,), 522)
    bias = _spec("zeros", (1024,), 523)
    add("layer-norm[1024x1024]", "normalization",
        lambda a, w, b: pF.layer_norm(a, (1024,), w, b),
        (norm_input, affine, bias))
    group_affine = _spec("ones", (64,), 524)
    group_bias = _spec("zeros", (64,), 525)
    group_input = _spec("normal", (8, 64, 28, 28), 526)
    add("group-norm[8x64x28x28]", "normalization",
        lambda a, w, b: pF.group_norm(a, 8, w, b),
        (group_input, group_affine, group_bias))
    running_mean = _spec("zeros", (64,), 527)
    running_var = _spec("ones", (64,), 528)
    add("batch-norm[8x64x28x28]", "normalization",
        lambda a, rm, rv, w, b: pF.batch_norm(
            a, rm, rv, w, b, training=False),
        (group_input, running_mean, running_var, group_affine, group_bias))
    instance_input = _spec("normal", (8, 64, 28, 28), 529)
    add("instance-norm[8x64x28x28]", "normalization",
        lambda a, w, b: pF.instance_norm(a, None, None, w, b),
        (instance_input, group_affine, group_bias))
    add("normalize[1024x1024]", "normalization",
        lambda a: pF.normalize(a, p=2.0, dim=1), (norm_input,))

    # Convolution and pooling families.
    conv1d_input = _spec("normal", (8, 16, 128), 600)
    conv1d_weight = _spec("normal", (32, 16, 3), 601)
    add("conv1d-3[8x16x128]", "convolution",
        lambda a, w: pF.conv1d(a, w, None, stride=1, padding=1),
        (conv1d_input, conv1d_weight))
    conv_input = _spec("normal", (8, 64, 56, 56), 602)
    conv3 = _spec("normal", (128, 64, 3, 3), 603)
    conv1 = _spec("normal", (128, 64, 1, 1), 604)
    conv7_input = _spec("normal", (2, 3, 224, 224), 605)
    conv7 = _spec("normal", (64, 3, 7, 7), 606)
    add("conv2d-3x3[8x64x56x56]", "convolution",
        lambda a, w: pF.conv2d(a, w, None, stride=1, padding=1),
        (conv_input, conv3))
    add("conv2d-1x1[8x64x56x56]", "convolution",
        lambda a, w: pF.conv2d(a, w, None, stride=1, padding=0),
        (conv_input, conv1))
    add("conv2d-7x7[2x3x224x224]", "convolution",
        lambda a, w: pF.conv2d(a, w, None, stride=2, padding=3),
        (conv7_input, conv7))
    add("conv2d-relu[8x64x56x56]", "convolution",
        lambda a, w: pF.relu(pF.conv2d(a, w, None, stride=1, padding=1)),
        (conv_input, conv3))
    add("avg-pool1d[8x16x128]", "pool", lambda a: pF.avg_pool1d(a, 2),
        (conv1d_input,))
    add("max-pool1d[8x16x128]", "pool", lambda a: pF.max_pool1d(a, 2),
        (conv1d_input,))
    add("avg-pool2d[8x64x56x56]", "pool", lambda a: pF.avg_pool2d(a, 2),
        (conv_input,))
    add("max-pool2d[8x64x56x56]", "pool", lambda a: pF.max_pool2d(a, 2),
        (conv_input,))
    add("adaptive-avg-pool2d[8x64x56x56]", "pool",
        lambda a: pF.adaptive_avg_pool2d(a, (7, 7)), (conv_input,))
    add("adaptive-max-pool2d[8x64x56x56]", "pool",
        lambda a: pF.adaptive_max_pool2d(a, (7, 7)), (conv_input,))
    add("interpolate-nearest[8x64x56x56]", "pool",
        lambda a: pF.interpolate(a, size=(28, 28), mode="nearest"),
        (conv_input,))
    add("channel-shuffle[8x64x56x56]", "pool",
        lambda a: pF.channel_shuffle(a, 8), (conv_input,))
    conv3d_input = _spec("normal", (2, 8, 16, 16, 16), 607)
    conv3d_weight = _spec("normal", (16, 8, 3, 3, 3), 608)
    add("conv3d-3[2x8x16x16x16]", "convolution",
        lambda a, w: pF.conv3d(a, w, None, stride=1, padding=1),
        (conv3d_input, conv3d_weight), long_only=True)
    add("avg-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.avg_pool3d(a, 2), (conv3d_input,), long_only=True)
    add("adaptive-avg-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.adaptive_avg_pool3d(a, (4, 4, 4)),
        (conv3d_input,), long_only=True)
    add("max-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.max_pool3d(a, 2), (conv3d_input,), long_only=True)
    add("adaptive-max-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.adaptive_max_pool3d(a, (4, 4, 4)),
        (conv3d_input,), long_only=True)
    convt1d_input = _spec("normal", (8, 16, 128), 610)
    convt1d_weight = _spec("normal", (16, 8, 3), 611)
    add("conv-transpose1d[8x16x128]", "convolution",
        lambda a, w: pF.conv_transpose1d(a, w),
        (convt1d_input, convt1d_weight))
    convt2d_input = _spec("normal", (8, 16, 28, 28), 612)
    convt2d_weight = _spec("normal", (16, 8, 3, 3), 613)
    add("conv-transpose2d[8x16x28x28]", "convolution",
        lambda a, w: pF.conv_transpose2d(a, w),
        (convt2d_input, convt2d_weight))
    convt3d_input = _spec("normal", (2, 16, 16, 16, 16), 614)
    convt3d_weight = _spec("normal", (16, 8, 3, 3, 3), 615)
    add("conv-transpose3d[2x16x16x16x16]", "convolution",
        lambda a, w: pF.conv_transpose3d(a, w),
        (convt3d_input, convt3d_weight), long_only=True)

    # Indexing/layout family: embedding, gather, concat, views and sorting.
    table = _spec("normal", (4096, 128), 700)
    ids = _spec("integer", (1024,), 701, upper=4096)
    add("embedding[1024x4096x128]", "indexing",
        lambda w, i: F.embedding(w, i), (table, ids))
    offsets = _spec("offsets", (32,), 702, upper=8)
    bag_ids = _spec("integer", (256,), 703, upper=4096)
    add("embedding-bag[32x8x128]", "indexing",
        lambda w, i, o: pF.embedding_bag(i, w, o, mode="mean"),
        (table, bag_ids, offsets))
    index_matrix = _spec("normal", (4096, 512), 704)
    row_indices = _spec("integer", (1024,), 705, upper=4096)
    add("index-select[1024x512]", "indexing",
        lambda a, i: F.index_select(a, 0, i), (index_matrix, row_indices))
    gather_indices = _spec("integer", (1024, 512), 706, upper=4096)
    add("gather-dim0[1024x512]", "indexing",
        lambda a, i: F.gather(a, 0, i), (index_matrix, gather_indices))
    gather_dim1_indices = _spec("integer", (4096, 128), 707, upper=512)
    add("gather-dim1[4096x128]", "indexing",
        lambda a, i: F.gather(a, 1, i), (index_matrix, gather_dim1_indices))
    take_input = _spec("normal", (4096,), 720)
    take_indices = _spec("integer", (1024,), 721, upper=4096)
    add("take[1024]", "indexing",
        lambda a, i: F.take(a, i), (take_input, take_indices))
    index_add_input = _spec("normal", (128, 64), 722)
    index_add_indices = _spec("integer", (64,), 723, upper=128)
    add("index-add[128x64]", "indexing",
        lambda a, i, v: F.index_add(a, 0, i, v),
        (index_add_input, index_add_indices, _spec("normal", (64, 64), 724)))
    scatter_input = _spec("normal", (128, 64), 725)
    scatter_indices = _spec("integer", (128, 64), 726, upper=128)
    add("scatter[128x64]", "indexing",
        lambda a, i, v: F.scatter(a, 0, i, v),
        (scatter_input, scatter_indices, _spec("normal", (128, 64), 727)))
    masked_input = _spec("normal", (1 << 18,), 728)
    masked_mask = _spec("boolean", (1 << 18,), 729)
    add("masked-fill[256k]", "indexing",
        lambda a, m: F.masked_fill(a, m, 0.5), (masked_input, masked_mask))
    add("nonzero[128x64]", "indexing",
        lambda a: F.nonzero(a), (_spec("normal", (128, 64), 730),))
    add("unique[256k]", "indexing",
        lambda a: F.unique(a), (_spec("normal", (1 << 18,), 731),))
    layout = _spec("normal", (32, 64, 28, 28), 708)
    add("cat-dim0[2x32x64x28x28]", "layout",
        lambda a: F.cat([a, a], dim=0), (layout,))
    add("cat-dim1[32x128x28x28]", "layout",
        lambda a: F.cat([a, a], dim=1), (layout,))
    add("stack-dim0[2x32x64x28x28]", "layout",
        lambda a: F.stack([a, a], dim=0), (layout,))
    matrix_layout = _spec("normal", (1024, 1024), 709)
    add("transpose-copy[1024x1024]", "layout",
        lambda a: F.contiguous(F.transpose(a, 0, 1)), (matrix_layout,))
    add("permute-copy[32x28x28x64]", "layout",
        lambda a: F.contiguous(F.permute(a, [0, 2, 3, 1])), (layout,))
    add("movedim-copy[32x28x28x64]", "layout",
        lambda a: F.contiguous(F.movedim(a, 1, -1)), (layout,))
    add("reshape[1024x1024]", "layout",
        lambda a: F.reshape(a, (256, 4096)), (matrix_layout,))
    add("split[1024x1024]", "layout",
        lambda a: F.split(a, 256, dim=0), (matrix_layout,))
    add("chunk[1024x1024]", "layout",
        lambda a: F.chunk(a, 4, dim=1), (matrix_layout,))
    add("diagonal[1024x1024]", "layout",
        lambda a: F.diagonal(a), (matrix_layout,))
    add("diag[4096]", "layout",
        lambda a: F.diag(a), (_spec("normal", (4096,), 710),))
    add("unbind[32x64x28x28]", "layout",
        lambda a: F.unbind(a, dim=0), (layout,))
    add("broadcast-to[64x1024]", "layout",
        lambda a: F.broadcast_to(a, (64, 1024)),
        (_spec("normal", (1, 1024), 711),))
    add("repeat[64x1024]", "layout",
        lambda a: F.repeat(a, (64, 1)),
        (_spec("normal", (1, 1024), 712),))
    strided_layout = _spec("normal", (64, 64), 713)
    add("as-strided[64x64]", "layout",
        lambda a: F.as_strided(a, (64, 64), (1, 64)), (strided_layout,))
    triangular_layout = _spec("normal", (128, 128), 714)
    add("tril[128x128]", "layout",
        lambda a: F.tril(a), (triangular_layout,))
    add("triu[128x128]", "layout",
        lambda a: F.triu(a), (triangular_layout,))
    add("trace[128x128]", "layout",
        lambda a: F.trace(a), (triangular_layout,))
    add("flipud[128x128]", "layout",
        lambda a: F.flipud(a), (triangular_layout,))
    cdist_a = _spec("normal", (64, 16), 715)
    cdist_b = _spec("normal", (32, 16), 716)
    add("cdist[64x16+32x16]", "layout",
        lambda a, b: F.cdist(a, b), (cdist_a, cdist_b))
    add("pdist[64x16]", "layout",
        lambda a: F.pdist(a), (_spec("normal", (64, 16), 717),))

    # Ternary/select/sort family.
    misc = _spec("normal", (1 << 18,), 800)
    add("where[256k]", "misc",
        lambda a: F.where(F.gt(a, 0), a, a * 2), (misc,))
    add("masked-select[256k]", "misc",
        lambda a: F.masked_select(a, F.gt(a, 0)), (misc,))
    add("clamp[256k]", "misc", lambda a: F.clamp(a, -0.5, 0.5), (misc,))
    add("pow2[256k]", "misc", lambda a: F.pow(a, 2.0), (misc,))
    add("dropout[256k]", "misc", lambda a: pF.dropout(a, 0.1, True), (misc,))
    add("arange[1M]", "misc", lambda a: F.arange(0, 1 << 20),
        (_spec("zeros", (1,), 801),))
    sort_input = _spec("normal", (1024, 1024), 802)
    add("topk8[1024x1024]", "sorting",
        lambda a: F.topk(a, 8, dim=1)[0], (sort_input,))
    add("sort[1024x1024]", "sorting",
        lambda a: F.sort(a, dim=1)[0], (sort_input,))
    add("argsort[1024x1024]", "sorting",
        lambda a: F.argsort(a, dim=1), (sort_input,))
    long_sort = _spec("normal", (2048, 2048), 803)
    add("topk8[2048x2048]", "sorting",
        lambda a: F.topk(a, 8, dim=1)[0], (long_sort,), long_only=True)
    add("sort[2048x2048]", "sorting",
        lambda a: F.sort(a, dim=1)[0], (long_sort,), long_only=True)

    # Close the remaining reference-suite gaps: bitwise shifts, 3D
    # convolution and pooling in the short suite, fractional max pooling,
    # sigmoid and clone.  Then go beyond the reference by measuring
    # operators it does not cover at all.
    bitwise_a = _spec("integer", (8, 16, 32), 900, upper=1024)
    bitwise_b = _spec("integer", (8, 16, 32), 901, upper=8)
    add("asr[8x16x32]", "binary",
        lambda a, b: F.bitwise_right_shift(a, b), (bitwise_a, bitwise_b))
    add("lsl[8x16x32]", "binary",
        lambda a, b: F.bitwise_left_shift(a, b), (bitwise_a, bitwise_b))
    add("fmax[256k]", "binary",
        lambda a, b: F.fmax(a, b),
        (_spec("normal", (1 << 18,), 902), _spec("normal", (1 << 18,), 903)))
    add("fmin[256k]", "binary",
        lambda a, b: F.fmin(a, b),
        (_spec("normal", (1 << 18,), 904), _spec("normal", (1 << 18,), 905)))
    add("hypot[256k]", "binary",
        lambda a, b: F.hypot(a, b),
        (_spec("normal", (1 << 18,), 906), _spec("normal", (1 << 18,), 907)))
    add("copysign[256k]", "binary",
        lambda a, b: F.copysign(a, b),
        (_spec("normal", (1 << 18,), 908), _spec("normal", (1 << 18,), 909)))
    add("xlogy[256k]", "binary",
        lambda a, b: F.xlogy(a + 1.0, b + 1.0),
        (_spec("normal", (1 << 18,), 910), _spec("normal", (1 << 18,), 911)))
    add("logaddexp[256k]", "binary",
        lambda a, b: F.logaddexp(a, b),
        (_spec("normal", (1 << 18,), 912), _spec("normal", (1 << 18,), 913)))
    add("logaddexp2[256k]", "binary",
        lambda a, b: F.logaddexp2(a, b),
        (_spec("normal", (1 << 18,), 914), _spec("normal", (1 << 18,), 915)))
    add("igamma[256k]", "binary",
        lambda a, b: F.igamma(a + 2.0, b + 2.0),
        (_spec("normal", (1 << 18,), 916), _spec("normal", (1 << 18,), 917)))
    add("igammac[256k]", "binary",
        lambda a, b: F.igammac(a + 2.0, b + 2.0),
        (_spec("normal", (1 << 18,), 918), _spec("normal", (1 << 18,), 919)))
    add("zeta[256k]", "binary",
        lambda a, b: F.zeta(a + 2.0, b + 2.0),
        (_spec("normal", (1 << 18,), 920), _spec("normal", (1 << 18,), 921)))
    add("lerp[256k]", "binary",
        lambda a, b: F.lerp(a, b, 0.5),
        (_spec("normal", (1 << 18,), 922), _spec("normal", (1 << 18,), 923)))
    add("sigmoid[256k]", "activation",
        lambda a: pF.sigmoid(a), (_spec("normal", (1 << 18,), 924),))
    add("glu[256k]", "activation",
        lambda a: pF.glu(a, dim=-1), (_spec("normal", (1 << 18,), 925),))
    add("bitwise-not[256k]", "unary",
        lambda a: F.bitwise_not(a), (_spec("integer", (1 << 18,), 926, upper=1024),))
    unary_extra = (
        ("deg2rad", lambda a: F.deg2rad(a)),
        ("rad2deg", lambda a: F.rad2deg(a)),
        ("sinc", lambda a: F.sinc(a)),
        ("mvlgamma", lambda a: F.mvlgamma(a + 3.0, 2)),
        ("polygamma", lambda a: F.polygamma(1, a + 2.0)),
        ("signbit", lambda a: F.signbit(a)),
        ("isnan", lambda a: F.isnan(a)),
        ("isinf", lambda a: F.isinf(a)),
        ("isfinite", lambda a: F.isfinite(a)),
        ("logical-not", lambda a: F.logical_not(a)),
        ("real", lambda a: F.real(a)),
        ("imag", lambda a: F.imag(a)),
        ("conj", lambda a: F.conj(a)),
        ("angle", lambda a: F.angle(a)),
    )
    for index, (name, operation) in enumerate(unary_extra):
        add(f"{name}[256k]", "unary", operation,
            (_spec("normal", (1 << 18,), 930 + index),))

    clone_input = _spec("normal", (1 << 18,), 950)
    add("clone[256k]", "layout", lambda a: a.clone(), (clone_input,))
    add("flatten[1M]", "layout",
        lambda a: F.flatten(a), (_spec("normal", (32, 28, 28, 64), 951),))
    add("ravel[1M]", "layout",
        lambda a: F.ravel(a), (_spec("normal", (32, 28, 28, 64), 952),))
    add("view[1M]", "layout",
        lambda a: F.view(a, (256, 4096)), (_spec("normal", (1024, 1024), 953),))
    add("squeeze[256k]", "layout",
        lambda a: F.squeeze(a), (_spec("normal", (1, 64, 1, 28), 954),))
    add("unsqueeze[256k]", "layout",
        lambda a: F.unsqueeze(a, 0), (_spec("normal", (64, 28), 955),))
    add("narrow[256k]", "layout",
        lambda a: F.narrow(a, 1, 10, 100), (_spec("normal", (1024, 1024), 956),))
    add("tile[256k]", "layout",
        lambda a: F.tile(a, (2, 2)), (_spec("normal", (64, 64), 957),))
    add("repeat-interleave[256k]", "layout",
        lambda a: F.repeat_interleave(a, 3), (_spec("normal", (1024,), 958),))
    add("expand-as[256k]", "layout",
        lambda a, b: F.expand_as(a, b),
        (_spec("normal", (1, 1024), 959), _spec("normal", (64, 1024), 960)))
    add("hstack[256k]", "layout",
        lambda a, b: F.hstack([a, b]),
        (_spec("normal", (64, 32), 961), _spec("normal", (64, 32), 962)))
    add("vstack[256k]", "layout",
        lambda a, b: F.vstack([a, b]),
        (_spec("normal", (32, 64), 963), _spec("normal", (32, 64), 964)))
    add("kron[256k]", "layout",
        lambda a, b: F.kron(a, b),
        (_spec("normal", (16, 16), 965), _spec("normal", (16, 16), 966)))
    add("t[256k]", "layout",
        lambda a: F.t(a), (_spec("normal", (64, 32), 967),))
    add("rot90[256k]", "layout",
        lambda a: F.rot90(a, 1, [0, 1]), (_spec("normal", (32, 28, 28, 64), 968),))
    add("fliplr[256k]", "layout",
        lambda a: F.fliplr(a), (_spec("normal", (128, 128), 969),))
    add("pixel-shuffle[256k]", "layout",
        lambda a: pF.pixel_shuffle(a, 2), (_spec("normal", (8, 16, 28, 28), 970),))
    add("pixel-unshuffle[256k]", "layout",
        lambda a: pF.pixel_unshuffle(a, 2), (_spec("normal", (8, 16, 28, 28), 971),))
    add("unique-consecutive[256k]", "layout",
        lambda a: F.unique_consecutive(a), (_spec("normal", (1024,), 972),))
    add("isclose[256k]", "layout",
        lambda a, b: F.isclose(a, b),
        (_spec("normal", (1 << 18,), 973), _spec("normal", (1 << 18,), 974)))
    add("allclose[256k]", "layout",
        lambda a, b: F.allclose(a, b),
        (_spec("normal", (1 << 18,), 975), _spec("normal", (1 << 18,), 976)))
    add("msort[256k]", "sorting",
        lambda a: F.msort(a), (_spec("normal", (1024,), 977),))
    add("count-nonzero[1M]", "reduction",
        lambda a: F.count_nonzero(a), (_spec("normal", (1024, 1024), 978),))
    add("argwhere[128x64]", "indexing",
        lambda a: F.argwhere(a), (_spec("normal", (128, 64), 979),))
    add("searchsorted[256k]", "indexing",
        lambda a, b: F.searchsorted(a, b),
        (_spec("normal", (1024,), 980), _spec("normal", (256,), 981)))
    sorted_source = _spec("normal", (1024,), 982)
    add("bucketize[256k]", "indexing",
        lambda a, b: F.bucketize(a, b),
        (_spec("normal", (256,), 983), sorted_source))
    add("bincount[256k]", "indexing",
        lambda a: F.bincount(a), (_spec("integer", (1024,), 984, upper=64),))
    add("one-hot[256k]", "indexing",
        lambda a: F.one_hot(a, 10), (_spec("integer", (256,), 985, upper=10),))
    add("pairwise-distance[64x16]", "distance",
        lambda a, b: pF.pairwise_distance(a, b),
        (_spec("normal", (64, 16), 986), _spec("normal", (64, 16), 987)))
    add("cosine-similarity[64x16]", "distance",
        lambda a, b: pF.cosine_similarity(a, b, dim=1),
        (_spec("normal", (64, 16), 988), _spec("normal", (64, 16), 989)))
    add("fractional-max-pool2d[8x64x28x28]", "pool",
        lambda a: pF.fractional_max_pool2d(a, 2, output_size=(7, 7)),
        (_spec("normal", (8, 64, 28, 28), 990),))
    add("fractional-max-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.fractional_max_pool3d(a, 2, output_size=(4, 4, 4)),
        (_spec("normal", (2, 8, 16, 16, 16), 991),))
    add("softmax2d[8x16x28x28]", "normalization",
        lambda a: F.softmax(a, dim=1),
        (_spec("normal", (8, 16, 28, 28), 992),))
    add("conv3d-3[2x8x16x16x16]", "convolution",
        lambda a, w: pF.conv3d(a, w, None, stride=1, padding=1),
        (conv3d_input, conv3d_weight))
    add("conv-transpose3d[2x16x16x16x16]", "convolution",
        lambda a, w: pF.conv_transpose3d(a, w),
        (convt3d_input, convt3d_weight))
    add("max-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.max_pool3d(a, 2), (conv3d_input,))
    add("avg-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.avg_pool3d(a, 2), (conv3d_input,))
    add("adaptive-max-pool3d[2x8x16x16x16]", "pool",
        lambda a: pF.adaptive_max_pool3d(a, (4, 4, 4)), (conv3d_input,))

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("short", "long", "all"), default="short")
    parser.add_argument("--dtype", choices=("f32", "f64", "bf16"), default="f32")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--flops", action="store_true",
                        help="attach per-op FLOP counts via the native profiler")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.threads < 0:
        parser.error("reps must be positive and threads must be non-negative")
    if args.threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(args.threads)
    suite = "long" if args.suite == "all" else args.suite
    measurements = []
    cases = build_cases(args.dtype, suite)
    print(f"suite={suite} dtype={args.dtype} cases={len(cases)}")
    for name, category, factory, specs in cases:
        fn = factory()
        # A kernel may reject a dtype outright; probe once so an unsupported
        # case is reported and skipped instead of aborting the remaining
        # measurements of the pass.
        try:
            fn()
        except NotImplementedError as exc:
            print(f"{name:42} [skipped: {exc}]")
            del fn
            continue
        seconds = _time(fn, args.reps)
        entry = {
            "name": name,
            "category": category,
            "input_shapes": [list(value[1]) for value in specs],
            "seconds": seconds,
        }
        if args.flops:
            entry["flops"] = _count_flops(fn)
            if _moves_data(name):
                entry["bytes_moved"] = _estimate_bytes(specs, _ITEMSIZE[args.dtype])
        print(f"{name:42} {seconds * 1e3:10.3f} ms")
        measurements.append(entry)
        del fn

    payload = {
        "schema_version": 1,
        "benchmark": "cpu-operator",
        "suite": suite,
        "dtype": args.dtype,
        "threads": args.threads or getattr(tp, "get_num_threads", lambda: 0)(),
        "measurements": measurements,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote results: {args.json_out}")


if __name__ == "__main__":
    main()
