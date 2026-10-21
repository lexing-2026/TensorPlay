#!/usr/bin/env python3
"""Holding the activation backward maps to one pass into the caller's tensor.

Each of these maps reads the gradient that arrived and the value the forward
saved at the same position, and answers at the width the gradient comes in.  The
derivative is stated here in plain arithmetic, so neither the form that allocates
its answer nor the form handed the caller's own tensor can agree with the other
and still be wrong.  The destination is offered in every layout the route has to
judge -- packed, strided, permuted as the operand is and permuted otherwise --
and as each of the two operands themselves, which is where a pass that reads one
buffer while writing another has to prove it stops.
"""
from math import erf

import numpy as np
import tensorplay as tp

NP_OF = {tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16, tp.bfloat16: np.float32}

LENGTH = 257  # a tail behind every vector width
ROWS, COLS = 13, 19  # neither factor a vector width

SQRT_2 = 1.4142135623730951
INV_SQRT_2PI = 0.3989422804014327
TANH_KAPPA = 0.044715
GELU_TANH_BETA = 0.7978845608028654  # 2 / sqrt(2 * pi)

# The parameters each map is asked about.
ALPHA, SCALE, INPUT_SCALE = 0.7, 0.9, 1.3
SLOPE = 0.13
LO, HI = -0.8, 1.3
BETA, THRESHOLD = 1.7, 3.0

erf_ = np.vectorize(erf)


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


# name, the form that allocates, the form handed a destination, the derivative as
# a function of the incoming gradient and the saved value, and whether a
# destination of a width the map does not answer at is refused rather than cast.
# The destination is the tensor autograd reads the gradient out of, so the
# spelling these maps are handed it under names it grad_input.
MAPS = [
    ("gelu[none]",
     lambda g, x: tp._C.gelu_backward(g, x, approximate="none"),
     lambda g, x, out=None: tp._C.gelu_backward(g, x, approximate="none",
                                               grad_input=out),
     lambda dy, v: dy * (0.5 * (1.0 + erf_(v / SQRT_2))
                         + v * INV_SQRT_2PI * np.exp(-v * v / 2.0)),
     True),
    ("gelu[tanh]",
     lambda g, x: tp._C.gelu_backward(g, x, approximate="tanh"),
     lambda g, x, out=None: tp._C.gelu_backward(g, x, approximate="tanh",
                                               grad_input=out),
     lambda dy, v: dy * (0.5 * (1.0 + np.tanh(GELU_TANH_BETA * (v + TANH_KAPPA * v ** 3)))
                         + 0.5 * v * (1.0 - np.tanh(
                             GELU_TANH_BETA * (v + TANH_KAPPA * v ** 3)) ** 2)
                         * GELU_TANH_BETA * (1.0 + 3.0 * TANH_KAPPA * v * v)),
     True),
    ("silu",
     lambda g, x: tp._C.silu_backward(g, x),
     lambda g, x, out=None: tp._C.silu_backward(g, x, grad_input=out),
     lambda dy, v: dy * sigmoid(v) * (1.0 + v * (1.0 - sigmoid(v))),
     True),
    ("hardsigmoid",
     lambda g, x: tp._C.hardsigmoid_backward(g, x),
     lambda g, x, out=None: tp._C.hardsigmoid_backward(g, x, grad_input=out),
     lambda dy, v: np.where(v <= -3.0, 0.0, np.where(v >= 3.0, 0.0, dy / 6.0)),
     True),
    ("hardtanh",
     lambda g, x: tp._C.hardtanh_backward(g, x, LO, HI),
     lambda g, x, out=None: tp._C.hardtanh_backward(g, x, LO, HI, grad_input=out),
     lambda dy, v: np.where(v <= LO, 0.0, np.where(v >= HI, 0.0, dy)),
     False),
    ("leaky_relu",
     lambda g, x: tp._C.leaky_relu_backward(g, x, SLOPE, False),
     lambda g, x, out=None: tp._C.leaky_relu_backward(g, x, SLOPE, False,
                                                     grad_input=out),
     lambda dy, v: np.where(v > 0.0, dy, dy * SLOPE),
     True),
    ("elu[input]",
     lambda g, x: tp._C.elu_backward(g, ALPHA, SCALE, INPUT_SCALE, False, x),
     lambda g, x, out=None: tp._C.elu_backward(g, ALPHA, SCALE, INPUT_SCALE, False,
                                               x, grad_input=out),
     lambda dy, v: np.where(v <= 0.0,
                            dy * INPUT_SCALE * ALPHA * SCALE * np.exp(v * INPUT_SCALE),
                            dy * SCALE),
     True),
    ("elu[result]",
     lambda g, x: tp._C.elu_backward(g, ALPHA, SCALE, INPUT_SCALE, True, x),
     lambda g, x, out=None: tp._C.elu_backward(g, ALPHA, SCALE, INPUT_SCALE, True,
                                               x, grad_input=out),
     lambda dy, v: np.where(v <= 0.0, dy * INPUT_SCALE * (v + ALPHA * SCALE),
                            dy * SCALE),
     True),
    ("softplus",
     lambda g, x: tp._C.softplus_backward(g, x, BETA, THRESHOLD),
     lambda g, x, out=None: tp._C.softplus_backward(g, x, BETA, THRESHOLD,
                                                   grad_input=out),
     lambda dy, v: np.where(v * BETA > THRESHOLD, dy, dy * sigmoid(v * BETA)),
     True),
]

WIDTHS = (tp.float32, tp.float64, tp.float16, tp.bfloat16)

rng = np.random.default_rng(11)
# Spread across every branch these maps carry: zero, the hardtanh bounds, the -3
# and 3 the hardsigmoid stops at, the threshold over its beta, and both signs of
# the gradient itself.  Long enough that a half-strided operand still has LENGTH
# elements.
GRAD = np.array(rng.standard_normal(2 * LENGTH) * 2.2, dtype=np.float64)
SAVED = np.array(rng.standard_normal(2 * LENGTH) * 2.2, dtype=np.float64)
GRAD[0] = 0.0
SAVED[0:6] = (0.0, LO, HI, -3.0, 3.0, THRESHOLD / BETA)


def pair(dtype, count=LENGTH):
    """The gradient and the saved value, in the same width and shape."""
    grad, value = GRAD[:count], SAVED[:count]
    return (tp.from_numpy(np.array(grad, dtype=NP_OF[dtype])).to(dtype),
            tp.from_numpy(np.array(value, dtype=NP_OF[dtype])).to(dtype))


def read(t):
    """The logical elements, one at a time: a narrow width reached through a
    stride is only faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def agree(got, want):
    return np.allclose(got, want, rtol=1e-3, atol=1e-4, equal_nan=True)


def test_grad_input_fills_the_destination_it_is_given():
    for name, alloc_fn, out_fn, _, _ in MAPS:
        for dtype in WIDTHS:
            grad, saved = pair(dtype)
            want = read(alloc_fn(grad, saved))
            out = tp.zeros(LENGTH, dtype=dtype)
            result = out_fn(grad, saved, out=out)
            assert result.data_ptr() == out.data_ptr(), (name, dtype)
            assert read(result).tolist() == want.tolist(), (name, dtype)


def test_grad_input_answers_the_derivative_it_is_for():
    # Not just the same answer as our own allocating form: the derivative stated
    # in plain arithmetic, so a route cannot agree with a broken body.
    for name, alloc_fn, out_fn, formula, _ in MAPS:
        want = formula(GRAD[:LENGTH], SAVED[:LENGTH])
        grad, saved = pair(tp.float64)
        out = tp.zeros(LENGTH, dtype=tp.float64)
        out_fn(grad, saved, out=out)
        assert agree(read(out), want), name
        assert agree(read(alloc_fn(grad, saved)), want), name


def test_grad_input_writes_an_operand_destination_in_place():
    # Either operand may be the buffer the answer goes into: the pass reads the
    # element at a position and stores at that same position, so the gradient is
    # still taken in one go.
    for name, alloc_fn, out_fn, _, _ in MAPS:
        for dtype in WIDTHS:
            grad, saved = pair(dtype)
            want = read(alloc_fn(grad, saved))

            as_grad, saved_copy = pair(dtype)
            out_fn(as_grad, saved_copy, out=as_grad)
            assert agree(read(as_grad), want), (name, dtype, "gradient")

            grad_copy, as_saved = pair(dtype)
            out_fn(grad_copy, as_saved, out=as_saved)
            assert agree(read(as_saved), want), (name, dtype, "saved value")

            # The same buffer serving as the gradient, the saved value and the
            # destination at once: the position is read twice and written once.
            dest = pair(dtype)[0]
            want_both = read(alloc_fn(pair(dtype)[0], pair(dtype)[0]))
            out_fn(dest, dest, out=dest)
            assert agree(read(dest), want_both), (name, dtype, "both operands")


def test_grad_input_answers_through_a_destination_it_has_to_copy_into():
    # A destination the pass cannot walk is served by computing and handing the
    # value over; the answer must not depend on which of the two routes ran.
    for name, alloc_fn, out_fn, _, strict in MAPS:
        for dtype in WIDTHS:
            grad, saved = pair(dtype)
            want = read(alloc_fn(grad, saved))

            strided = tp.zeros(2 * LENGTH, dtype=dtype)[::2]
            out_fn(grad, saved, out=strided)
            assert agree(read(strided), want), (name, dtype, "strided")

            other = tp.zeros(LENGTH, dtype=tp.float64 if dtype != tp.float64
                             else tp.float32)
            if strict:
                try:
                    out_fn(grad, saved, out=other)
                except Exception:
                    continue
                raise AssertionError(
                    f"{name} cast into a width its map does not answer at")
            out_fn(grad, saved, out=other)
            assert agree(read(other), read(alloc_fn(grad, saved).to(other.dtype))), \
                (name, dtype, "another width")


def test_grad_input_answers_into_a_destination_laid_out_as_the_operand_is():
    for name, alloc_fn, out_fn, _, _ in MAPS:
        for dtype in WIDTHS:
            grad, saved = pair(dtype, ROWS * COLS)
            grad = grad.reshape(ROWS, COLS).transpose(0, 1)
            saved = saved.reshape(ROWS, COLS).transpose(0, 1)
            want = read(alloc_fn(grad, saved))
            dest = tp.zeros(ROWS, COLS, dtype=dtype).transpose(0, 1)
            out_fn(grad, saved, out=dest)
            assert agree(read(dest), want), (name, dtype)


def test_grad_input_answers_into_a_destination_laid_out_as_the_operand_is_not():
    # The operands stepped and the destination packed, and the other way round:
    # one flat loop would meet the two in different orders, so the pass has to
    # step through the destination instead.
    for name, alloc_fn, out_fn, _, _ in MAPS:
        for dtype in WIDTHS:
            grad = pair(dtype, ROWS * COLS)[0].reshape(ROWS, COLS)
            saved = pair(dtype, ROWS * COLS)[1].reshape(ROWS, COLS)
            want = read(alloc_fn(grad, saved))

            stepped = tp.zeros(COLS, ROWS, dtype=dtype).transpose(0, 1)
            out_fn(grad, saved, out=stepped)
            assert agree(read(stepped), want), (name, dtype, "stepped destination")

            grad_t = grad.transpose(0, 1)
            saved_t = saved.transpose(0, 1)
            packed = tp.zeros(COLS, ROWS, dtype=dtype)
            out_fn(grad_t, saved_t, out=packed)
            assert agree(read(packed), read(alloc_fn(grad_t, saved_t))), \
                (name, dtype, "packed destination")


def test_grad_input_answers_a_strided_operand():
    for name, alloc_fn, out_fn, _, _ in MAPS:
        grad, saved = pair(tp.float32, 2 * LENGTH)
        grad, saved = grad[::2], saved[::2]
        want = read(alloc_fn(grad, saved))
        out = tp.zeros(LENGTH, dtype=tp.float32)
        assert agree(read(out_fn(grad, saved, out=out)), want), name


def test_grad_input_keeps_the_operand_at_another_width_refused():
    # The maps are defined on a gradient and a saved value of one width; a pair
    # of two widths is not the map's own, so both spellings have to answer it the
    # same way whichever way that is.
    for name, alloc_fn, out_fn, _, _ in MAPS:
        grad = pair(tp.float32)[0]
        saved = pair(tp.float64)[1]
        out = tp.zeros(LENGTH, dtype=tp.float32)
        try:
            got = read(alloc_fn(grad, saved))
        except Exception:
            refused = True
        else:
            refused = False
        if refused:
            try:
                out_fn(grad, saved, out=out)
            except Exception:
                continue
            raise AssertionError(f"{name} answered a pair its map refuses")
        assert agree(read(out_fn(grad, saved, out=out)), got), (name, "mixed widths")


def test_grad_input_answers_an_integral_pair():
    # Whole numbers are answered at their own width, so what matters is that the
    # two spellings of the map agree on the truncation the width forces.
    grad = tp.from_numpy(np.arange(LENGTH, dtype=np.int64) - LENGTH // 2)
    saved = tp.from_numpy(np.array(SAVED[:LENGTH] * 40, dtype=np.int64))
    for name, alloc_fn, out_fn, _, _ in MAPS:
        want = read(alloc_fn(grad, saved))
        out = tp.zeros(LENGTH, dtype=tp.int64)
        result = out_fn(grad, saved, out=out)
        assert result.data_ptr() == out.data_ptr(), name
        assert read(result).tolist() == want.tolist(), name


def test_grad_input_answers_an_empty_pair():
    for dtype in WIDTHS:
        empty = tp.zeros([0], dtype=dtype)
        for name, alloc_fn, out_fn, _, _ in MAPS:
            assert read(alloc_fn(empty, empty)).size == 0, name
            out = tp.zeros([0], dtype=dtype)
            result = out_fn(empty, empty, out=out)
            assert result.data_ptr() == out.data_ptr(), name
            assert read(result).size == 0, name


def test_grad_input_refuses_a_destination_of_the_wrong_shape_by_resizing_it():
    # A destination too small for the answer is grown to it, as the form that
    # allocates its own result always was.
    for name, alloc_fn, out_fn, _, _ in MAPS:
        grad, saved = pair(tp.float32)
        want = read(alloc_fn(grad, saved))
        out = tp.zeros([1], dtype=tp.float32)
        out_fn(grad, saved, out=out)
        assert agree(read(out), want), name
