#!/usr/bin/env python3
"""Holding the unary activation maps to one pass into the caller's tensor.

Each of these maps answers in the operand's own width -- a float16 operand gets
a float16 answer, an integral one is promoted to float -- and every one of them
carries the same arithmetic twice over: once in the form that allocates its
result and once in the form handed a destination.  The value is stated here in
plain arithmetic, so neither spelling can agree with the other and still be
wrong, and the destination is offered in every layout the route has to judge:
packed, strided, permuted, another width, and the operand itself.
"""
import numpy as np
import tensorplay as tp
from math import erf

# The exact error function, evaluated one number at a time: the oracle states the
# value in plain arithmetic, so it must not borrow the kernel's own approximation.
erf_ = np.vectorize(erf)

NP_OF = {tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16, tp.bfloat16: np.float32}

LENGTH = 257  # a tail behind every vector width
SQRT_2_OVER_PI = 0.7978845608028654

# The parameters each map is asked about.
ALPHA = 0.7
LO, HI = -0.8, 1.3
SLOPE = 0.13
BETA, THRESHOLD = 1.7, 3.0

# name, the form that allocates, the form handed a destination, the arithmetic.
MAPS = [
    ("relu", lambda x: tp.relu(x), lambda x, out=None: tp.relu(x, out=out),
     lambda v: np.maximum(v, 0.0)),
    ("gelu", lambda x: tp.gelu(x), lambda x, out=None: tp.gelu(x, out=out),
     lambda v: 0.5 * v * (1.0 + erf_(v / np.sqrt(2.0)))),
    ("gelu[tanh]", lambda x: tp.gelu(x, "tanh"),
     lambda x, out=None: tp.gelu(x, "tanh", out=out),
     lambda v: 0.5 * v * (1.0 + np.tanh(
         SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))),
    ("silu", lambda x: tp.silu(x), lambda x, out=None: tp.silu(x, out=out),
     lambda v: v / (1.0 + np.exp(-v))),
    ("mish", lambda x: tp.mish(x), lambda x, out=None: tp.mish(x, out=out),
     lambda v: v * np.tanh(np.log1p(np.exp(v)))),
    ("elu", lambda x: tp.elu(x, ALPHA), lambda x, out=None: tp.elu(x, ALPHA, out=out),
     lambda v: np.where(v > 0.0, v, ALPHA * np.expm1(v))),
    ("hardtanh", lambda x: tp.hardtanh(x, LO, HI),
     lambda x, out=None: tp.hardtanh(x, LO, HI, out=out),
     lambda v: np.clip(v, LO, HI)),
    ("leaky_relu", lambda x: tp.leaky_relu(x, SLOPE),
     lambda x, out=None: tp.leaky_relu(x, SLOPE, out=out),
     lambda v: np.where(v >= 0.0, v, SLOPE * v)),
    ("softplus", lambda x: tp.softplus(x, BETA, THRESHOLD),
     lambda x, out=None: tp.softplus(x, BETA, THRESHOLD, out=out),
     lambda v: np.where(v * BETA > THRESHOLD, v, np.log1p(np.exp(v * BETA)) / BETA)),
    ("hardsigmoid", lambda x: tp.hardsigmoid(x),
     lambda x, out=None: tp.hardsigmoid(x, out=out),
     lambda v: np.clip(v + 3.0, 0.0, 6.0) / 6.0),
    ("hardswish", lambda x: tp.hardswish(x),
     lambda x, out=None: tp.hardswish(x, out=out),
     lambda v: v * np.clip(v + 3.0, 0.0, 6.0) / 6.0),
    ("angle", lambda x: tp.angle(x), lambda x, out=None: tp.angle(x, out=out),
     lambda v: np.where(v < 0.0, np.pi, 0.0)),
]

WIDTHS = (tp.float32, tp.float64, tp.float16, tp.bfloat16)

# The maps whose hand-over refuses a destination of a width it does not answer
# at, rather than cast into one.
STRICT = {"hardsigmoid", "hardswish"}

rng = np.random.default_rng(11)
# Spread across every branch these maps carry: zero, the hardtanh bounds, the
# -3 and 3 the hard maps clamp at, and the softplus threshold over its beta.
PLAIN = np.array(rng.standard_normal(LENGTH) * 2.2, dtype=np.float64)
PLAIN[0:3] = (0.0, LO, HI)
PLAIN[3:6] = (-3.0, 3.0, THRESHOLD / BETA)


def sample(dtype, count=LENGTH, seed=11):
    values = PLAIN if seed == 11 else np.array(
        np.random.default_rng(seed).standard_normal(count) * 2.2, dtype=np.float64)
    return tp.from_numpy(np.array(values[:count], dtype=NP_OF[dtype])).to(dtype)


def read(t):
    """The logical elements, one at a time: a narrow width reached through a
    stride is only faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def agree(got, want):
    return np.allclose(got, want, rtol=1e-3, atol=1e-4, equal_nan=True)


def test_out_fills_the_destination_it_is_given():
    for name, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            x = sample(dtype)
            want = read(alloc_fn(x))
            out = tp.zeros(LENGTH, dtype=dtype)
            result = out_fn(x, out=out)
            assert result.data_ptr() == out.data_ptr(), (name, dtype)
            assert agree(read(out), want), (name, dtype)


def test_out_answers_the_arithmetic_it_is_for():
    # Not just the same answer as our own allocating form: the value stated in
    # plain arithmetic, so a route cannot agree with a broken body.
    for name, alloc_fn, out_fn, formula in MAPS:
        want = formula(PLAIN)
        x = sample(tp.float64)
        out = tp.zeros(LENGTH, dtype=tp.float64)
        out_fn(x, out=out)
        assert agree(read(out), want), name
        assert agree(read(alloc_fn(x)), want), name


def test_out_writes_an_operand_destination_in_place():
    for name, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            want = read(alloc_fn(sample(dtype)))
            operand = sample(dtype)
            out_fn(operand, out=operand)
            assert agree(read(operand), want), (name, dtype)


def test_out_answers_through_a_destination_it_has_to_copy_into():
    # A destination the pass cannot walk is served by computing and handing the
    # value over; the answer must not depend on which of the two routes ran.
    for name, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            x = sample(dtype)
            want = read(alloc_fn(x))

            strided = tp.zeros(2 * LENGTH, dtype=dtype)[::2]
            out_fn(x, out=strided)
            assert agree(read(strided), want), (name, dtype, "strided")

            other = tp.zeros(LENGTH, dtype=tp.float64 if dtype != tp.float64
                             else tp.float32)
            if name in STRICT:
                try:
                    out_fn(x, out=other)
                except Exception:
                    continue
                raise AssertionError(
                    f"{name} cast into a width its map does not answer at")
            out_fn(x, out=other)
            assert agree(read(other), read(alloc_fn(x).to(other.dtype))), \
                (name, dtype, "another width")


def test_out_answers_into_a_destination_laid_out_as_the_operand_is():
    # The operand permuted and the destination carrying that very permutation:
    # the one case where a non-packed destination is the one the pass walks.
    for name, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            rows, cols = 13, 19  # neither factor a vector width
            operand = sample(dtype, rows * cols).reshape(rows, cols).transpose(0, 1)
            want = read(alloc_fn(operand))
            dest = tp.zeros(rows, cols, dtype=dtype).transpose(0, 1)
            out_fn(operand, out=dest)
            assert agree(read(dest), want), (name, dtype)


def test_out_answers_into_a_destination_laid_out_as_the_operand_is_not():
    # The other permutation: the operand packed, the destination stepped.  One
    # flat loop would meet the two in different orders, so the pass declines and
    # the value still has to arrive.
    for name, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            rows, cols = 13, 19
            operand = sample(dtype, rows * cols).reshape(rows, cols)
            want = read(alloc_fn(operand))
            dest = tp.zeros(cols, rows, dtype=dtype).transpose(0, 1)
            out_fn(operand, out=dest)
            assert agree(read(dest), want), (name, dtype)


def test_out_answers_a_reshaped_operand():
    for name, alloc_fn, out_fn, _ in MAPS:
        x = sample(tp.float32, 12).reshape(4, 3)
        want = read(alloc_fn(x))
        out = tp.zeros([4, 3], dtype=tp.float32)
        assert agree(read(out_fn(x, out=out)), want), name

        strided_operand = sample(tp.float32, 24)[::2]
        want = read(alloc_fn(strided_operand))
        out = tp.zeros(12, dtype=tp.float32)
        assert agree(read(out_fn(strided_operand, out=out)), want), name


def test_integral_operand_answers_at_float_width():
    # The promoting maps: an integral operand is answered as a float, so a float
    # destination of that shape is the one the pass walks.  A map whose kernel
    # declines an operand of this category is left out of the ledger.
    x = tp.from_numpy(np.arange(64, dtype=np.int64) - 32)
    for name, alloc_fn, out_fn, _ in MAPS:
        try:
            want = read(alloc_fn(x))
        except Exception:
            continue
        out = tp.zeros(64, dtype=tp.float32)
        result = out_fn(x, out=out)
        assert result.data_ptr() == out.data_ptr(), name
        assert agree(read(result), want), name
