#!/usr/bin/env python3
"""Holding the shrink, threshold and saved-gradient maps to one pass.

These maps keep the operand's width, so a packed destination of any width --
Half and BFloat16 included -- takes the pass that writes the answer straight
into it.  The value each one gives is stated here in plain arithmetic, which
both spellings have to answer to: the form that allocates its own tensor and the
form handed one, into whatever destination the caller holds.
"""
import numpy as np
import tensorplay as tp

NP_OF = {tp.float32: np.float32, tp.float64: np.float64, tp.float16: np.float16}

LENGTH = 257  # a tail behind every vector width

# The band every shrink map is asked about, and the limit and replacement the
# threshold map carries.
LAMBDA = 0.5
LIMIT = 0.3
REPLACEMENT = -2.0

# name, operand count, the forms that answer it, and the arithmetic it has to
# give -- a one-operand map is asked of the plain sample, a two-operand map of
# the gradient and the saved forward answer, both at float64 so the expectation
# is stated rather than recomputed.
MAPS = [
    ("hardshrink", 1, lambda x: tp.hardshrink(x, LAMBDA),
     lambda x, out=None: tp.hardshrink(x, LAMBDA, out=out),
     lambda v, _=None: np.where(np.abs(v) <= LAMBDA, 0.0, v)),
    ("softshrink", 1, lambda x: tp.softshrink(x, LAMBDA),
     lambda x, out=None: tp.softshrink(x, LAMBDA, out=out),
     lambda v, _=None: np.where(v > LAMBDA, v - LAMBDA,
                                np.where(v < -LAMBDA, v + LAMBDA, 0.0))),
    ("threshold", 1, lambda x: tp.threshold(x, LIMIT, REPLACEMENT),
     lambda x, out=None: tp.threshold(x, LIMIT, REPLACEMENT, out=out),
     lambda v, _=None: np.where(v <= LIMIT, REPLACEMENT, v)),
    ("hardshrink_backward", 2, lambda g, s: tp._C.hardshrink_backward(g, s, LAMBDA),
     lambda g, s, out=None: tp._C.hardshrink_backward(g, s, LAMBDA, grad_input=out),
     lambda grad, other: np.where(np.abs(other) <= LAMBDA, 0.0, grad)),
    ("softshrink_backward", 2, lambda g, s: tp._C.softshrink_backward(g, s, LAMBDA),
     lambda g, s, out=None: tp._C.softshrink_backward(g, s, LAMBDA, grad_input=out),
     lambda grad, other: np.where(np.abs(other) <= LAMBDA, 0.0, grad)),
    ("sigmoid_backward", 2, lambda g, o: tp._C.sigmoid_backward(g, o),
     lambda g, o, out=None: tp._C.sigmoid_backward(g, o, grad_input=out),
     lambda grad, other: grad * other * (1.0 - other)),
    ("tanh_backward", 2, lambda g, o: tp._C.tanh_backward(g, o),
     lambda g, o, out=None: tp._C.tanh_backward(g, o, grad_input=out),
     lambda grad, other: grad * (1.0 - other * other)),
    ("logit_backward", 2, lambda g, s: tp._C.logit_backward(g, s, None),
     lambda g, s, out=None: tp._C.logit_backward(g, s, None, grad_input=out),
     lambda grad, other: _logit_grad(grad, other)),
]

WIDTHS = (tp.float32, tp.float64, tp.float16)

rng = np.random.default_rng(5)
PLAIN = np.array(rng.standard_normal(LENGTH) * 1.4, dtype=np.float64)
# A saved forward answer, on a dyadic ladder so the ends of the unit interval are
# held exactly and logit's infinite branch is asked about rather than missed: a
# few values below 0 and above 1 give the NaN its turn too.
SAVED = np.array(np.arange(LENGTH, dtype=np.float64) / 64.0 - 0.5, dtype=np.float64)


def _logit_grad(grad, other):
    """The derivative of the log-odds: NaN where the operand left the unit
    interval, and an infinite gradient at its ends."""
    inside = (other >= 0.0) & (other <= 1.0)
    at_end = (other == 0.0) | (other == 1.0)
    out = np.full(other.shape, np.nan)
    safe = inside & ~at_end
    out[safe] = grad[safe] / (other[safe] * (1.0 - other[safe]))
    # At the ends the derivative is the gradient carried out to the width's
    # infinity, so its sign is the gradient's own.
    out[at_end] = grad[at_end] * np.inf
    return out


def sample(kind, dtype):
    value = PLAIN if kind == "plain" else SAVED
    if dtype not in NP_OF:
        return tp.from_numpy(np.array(value, dtype=np.float32)).to(dtype)
    return tp.from_numpy(np.array(value, dtype=NP_OF[dtype]))


def operands(arity, dtype):
    return [sample("plain", dtype)] if arity == 1 else [
        sample("plain", dtype), sample("saved", dtype)]


def expected(formula, arity):
    """The value, in plain arithmetic, at the widest width numpy holds."""
    return formula(PLAIN) if arity == 1 else formula(PLAIN, SAVED)


def read(t):
    return np.array(t.to(tp.float32).tolist(), dtype=np.float32)


def agree(got, want):
    return np.allclose(got, want, rtol=1e-3, atol=1e-4, equal_nan=True)


def test_out_fills_the_destination_it_is_given():
    for name, arity, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            ops = operands(arity, dtype)
            want = read(alloc_fn(*ops))
            out = tp.zeros(LENGTH, dtype=dtype)
            result = out_fn(*ops, out=out)
            assert result.data_ptr() == out.data_ptr(), name
            assert agree(read(out), want), (name, dtype)


def test_out_answers_the_arithmetic_it_is_for():
    # Not just the same answer as our own allocating form: the value stated in
    # plain arithmetic, so a route cannot agree with a broken body.
    for name, arity, alloc_fn, out_fn, formula in MAPS:
        want = expected(formula, arity)
        ops = operands(arity, tp.float64)
        out = tp.zeros(LENGTH, dtype=tp.float64)
        out_fn(*ops, out=out)
        assert agree(read(out), want), name
        assert agree(read(alloc_fn(*ops)), want), name


def test_out_writes_an_operand_destination_in_place():
    for name, arity, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            want = read(alloc_fn(*operands(arity, dtype)))
            if arity == 1:
                as_operand, = operands(arity, dtype)
                out_fn(as_operand, out=as_operand)
                assert agree(read(as_operand), want), (name, dtype)
                continue
            for slot in (0, 1):
                ops = operands(arity, dtype)
                out_fn(*ops, out=ops[slot])
                assert agree(read(ops[slot]), want), (name, dtype, slot)


def test_out_answers_through_a_destination_it_has_to_copy_into():
    # A strided destination is not the buffer the pass would have written; the
    # answer must not depend on which of the two routes the destination allowed.
    for name, arity, alloc_fn, out_fn, _ in MAPS:
        for dtype in WIDTHS:
            ops = operands(arity, dtype)
            want = read(alloc_fn(*ops))
            out = tp.zeros(2 * LENGTH, dtype=dtype)[::2]
            out_fn(*ops, out=out)
            assert agree(read(out), want), (name, dtype, "strided")

            narrow = tp.zeros(LENGTH, dtype=tp.float16)
            out_fn(*ops, out=narrow)
            assert agree(read(narrow), read(alloc_fn(*ops).to(tp.float16))), \
                (name, dtype, "narrower")

            wide = tp.zeros(LENGTH, dtype=tp.float64)
            out_fn(*ops, out=wide)
            assert agree(read(wide), read(alloc_fn(*ops).to(tp.float64))), \
                (name, dtype, "wider")


def test_out_broadcasts_into_the_destination():
    for name, arity, alloc_fn, out_fn, _ in MAPS:
        if arity == 1:
            left = tp.from_numpy(np.array(PLAIN[:12], dtype=np.float32).reshape(4, 3))
            want = read(alloc_fn(left))
            out = tp.zeros([4, 3], dtype=tp.float32)
            out_fn(left, out=out)
            assert agree(read(out), want), name
            continue
        left = tp.from_numpy(np.array(PLAIN[:4], dtype=np.float32).reshape(4, 1))
        # A saved strip that lies inside the unit interval: the pair has to
        # broadcast to real answers rather than to a block of NaNs.
        right = tp.from_numpy(np.array(SAVED[32:35], dtype=np.float32))
        want = read(alloc_fn(left, right))
        out = tp.zeros([4, 3], dtype=tp.float32)
        out_fn(left, right, out=out)
        assert agree(read(out), want), name


def test_out_keeps_a_destination_it_cannot_answer_into():
    # A destination whose shape the pair does not broadcast to is refused by both
    # spellings; the route never answers into a shape it was not given.
    for name, arity, alloc_fn, out_fn, _ in MAPS:
        if arity == 1:
            continue
        left = sample("plain", tp.float32)
        right = tp.from_numpy(np.array(PLAIN[:9], dtype=np.float32))
        out = tp.zeros(9, dtype=tp.float32)
        try:
            out_fn(left, right, out=out)
        except Exception:
            continue
        raise AssertionError(f"{name} answered a pair whose shapes do not meet")
