#!/usr/bin/env python3
"""Holding the softmax family to one pass into the destination it is given.

Four walks answer these maps.  The two forward ones sweep a row of a packed input
and store the answer back over the offsets they read, and the two backward ones
sum a column of a saved output before writing the gradient into it.  A destination
one of them cannot walk -- another width, another shape, one reached at a stride,
or one drawn from the very bytes still to be read -- is answered by computing the
value and handing it over, which is what makes the two routes comparable: whatever
route ran, the answer has to be the same element for element.  What each walk gives
is also stated here in plain arithmetic, so a route cannot agree with its own twin
and still be wrong.  One of these walks reads a row again after storing over it,
which is why an input serving as its own destination is answered two different ways
depending on the map: the plain one walks it, the logarithmic one may not.
"""
import numpy as np
import tensorplay as tp

NP_OF = {tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16, tp.bfloat16: np.float32}

WIDTHS = (tp.float32, tp.float64, tp.float16, tp.bfloat16)

ROWS, COLS = 7, 11  # neither factor a vector width, and their product is odd
DEEP = (3, ROWS, COLS)

rng = np.random.default_rng(13)
# Spread over the branch the exponentials care about: one whole row of equal
# values, a row whose largest element sits at each end in turn, and magnitudes
# large enough that a naive exp() would run away.
BASE = np.array(rng.standard_normal(ROWS * COLS) * 3.0, dtype=np.float64)
BASE[0:COLS] = 1.7          # an equal row: every element shares the maximum
BASE[COLS] = 640.0          # one row dominated by its first element
BASE[2 * COLS + COLS - 1] = -640.0  # a row whose maximum is its last element
GRAD_BASE = np.array(rng.standard_normal(ROWS * COLS) * 0.8, dtype=np.float64)

RTOL_OF = {tp.float32: 1e-4, tp.float64: 1e-10,
           tp.float16: 2e-3, tp.bfloat16: 2e-2}


def value(v, axis):
    """exp(x - max) / sum(exp(x - max)) along `axis`."""
    shifted = v - np.max(v, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_value(v, axis):
    """(x - max) - log(sum(exp(x - max))) along `axis`."""
    shifted = v - np.max(v, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def gradient(g, p, axis):
    """The plain map's derivative: p * (g - sum(g * p))."""
    return p * (g - np.sum(g * p, axis=axis, keepdims=True))


def log_gradient(g, p, axis):
    """The logarithmic map's derivative: g - exp(p) * sum(g)."""
    return g - np.exp(p) * np.sum(g, axis=axis, keepdims=True)


def matrix(dtype, rows=ROWS, cols=COLS, deep=False):
    """The sample, in the given width and shape."""
    count = rows * cols
    shape = (3, rows, cols) if deep else (rows, cols)
    source = np.resize(BASE[:count], count).reshape(rows, cols)
    if deep:
        source = np.stack([source, np.roll(source, 1, axis=0),
                           np.roll(source, -1, axis=1)])
        count = 3 * rows * cols
    return tp.from_numpy(np.array(source.reshape(-1)[:count],
                                  dtype=NP_OF[dtype])).to(dtype).reshape(shape)


def grad_matrix(dtype):
    source = GRAD_BASE[:ROWS * COLS].reshape(ROWS, COLS)
    return tp.from_numpy(np.array(source, dtype=NP_OF[dtype])).to(dtype)


def read(t):
    """The logical elements: a narrow width reached through a stride is only
    faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def agree(got, want, rtol):
    return np.allclose(got, want, rtol=rtol, atol=1e-6, equal_nan=True)


def stepped(dtype, rows=ROWS, cols=COLS):
    """A destination every one of these walks has to step through instead of
    writing in element order -- which is how the answer arrives by copy."""
    return tp.zeros(2 * rows, cols, dtype=dtype)[::2]


def unit(dtype, rows=ROWS, cols=COLS):
    """A packed destination of the answer's own shape and width."""
    return tp.zeros(rows, cols, dtype=dtype)


def tensor_of(values, dtype):
    """A packed tensor of the given width, holding plain arithmetic."""
    return tp.from_numpy(np.array(values, dtype=NP_OF[dtype])).to(dtype)


# The five spellings, each as (the form that allocates, the form handed a
# destination, the destination's keyword, the expectation as plain arithmetic).
def _softmax(x, dim):
    return tp._C._softmax(x, dim, False)


def _log_softmax(x, dim):
    return tp._C._log_softmax(x, dim, False)


def _softmax_backward(g, p, dim):
    return tp._C._softmax_backward_data(g, p, dim, g.dtype)


def _log_softmax_backward(g, p, dim):
    return tp._C._log_softmax_backward_data(g, p, dim, g.dtype)


# name, allocating spelling, out spelling, the keyword the destination goes under
FORWARD = [
    ("_softmax.out", lambda x, d: _softmax(x, d),
     lambda x, d, o: tp._C._softmax(x, d, False, out=o), "out"),
    ("_log_softmax.out", lambda x, d: _log_softmax(x, d),
     lambda x, d, o: tp._C._log_softmax(x, d, False, out=o), "out"),
    ("softmax.int_out", lambda x, d: tp.softmax(x, d),
     lambda x, d, o: tp.softmax(x, d, out=o), "out"),
    ("log_softmax.int_out", lambda x, d: tp.log_softmax(x, d),
     lambda x, d, o: tp.log_softmax(x, d, out=o), "out"),
]

BACKWARD = [
    ("_softmax_backward_data.out",
     lambda g, p, d: _softmax_backward(g, p, d),
     lambda g, p, d, o: tp._C._softmax_backward_data(g, p, d, g.dtype,
                                                     grad_input=o),
     "grad_input", False),
    ("_log_softmax_backward_data.out",
     lambda g, p, d: _log_softmax_backward(g, p, d),
     lambda g, p, d, o: tp._C._log_softmax_backward_data(g, p, d, g.dtype, out=o),
     "out", True),
]


# ---------------------------------------------------------------------------
# the forward walks
# ---------------------------------------------------------------------------

def test_softmax_out_fills_the_destination_it_is_given():
    for name, alloc_fn, out_fn, _ in FORWARD:
        for dtype in WIDTHS:
            for dim in (0, 1):
                x = matrix(dtype)
                wide = stepped(dtype)
                out_fn(x, dim, wide)
                want = read(wide)

                out = unit(dtype)
                result = out_fn(x, dim, out)
                assert result.data_ptr() == out.data_ptr(), (name, dtype, dim)
                assert read(result).tolist() == want.tolist(), (name, dtype, dim)
                assert agree(read(result), read(alloc_fn(x, dim)),
                             RTOL_OF[dtype]), (name, dtype, dim)


def test_softmax_out_answers_the_map_it_is_for():
    for name, _, out_fn, _ in FORWARD:
        log = "log" in name
        for dtype in WIDTHS:
            x = matrix(dtype)
            for dim in (0, 1):
                want = (log_value if log else value)(read(x), dim)
                out = unit(dtype)
                assert agree(read(out_fn(x, dim, out)), want,
                             RTOL_OF[dtype]), (name, dtype, dim)


def test_softmax_out_walks_a_destination_of_the_answer_at_another_width():
    # A wider sample asked to answer narrow, and a narrow one asked to answer
    # wide: in both the walk's width is the destination's own, so the destination
    # is written rather than met by a copy of something else.
    for name, alloc_fn, out_fn, _ in FORWARD:
        for source_dtype, dest_dtype in ((tp.float64, tp.float32),
                                         (tp.float32, tp.float64)):
            x = matrix(source_dtype)
            out = unit(dest_dtype)
            result = out_fn(x, 1, out)
            assert result.data_ptr() == out.data_ptr(), (name, dest_dtype)
            assert agree(read(result), read(alloc_fn(x, 1).to(dest_dtype)),
                         RTOL_OF[dest_dtype]), (name, dest_dtype)


def test_softmax_out_answers_whole_numbers_at_single_precision():
    for name, alloc_fn, out_fn, _ in FORWARD:
        x = tp.from_numpy(np.array(BASE[:ROWS * COLS] * 8, dtype=np.int64))
        x = x.reshape(ROWS, COLS)
        want = read(alloc_fn(x, 1))
        out = tp.zeros(ROWS, COLS, dtype=tp.float32)
        result = out_fn(x, 1, out)
        assert result.data_ptr() == out.data_ptr(), name
        assert read(result).tolist() == want.tolist(), name


def test_softmax_out_answers_a_walk_that_stepped_its_input():
    # A strided sample is walked as a packed copy of itself; the destination stays
    # the caller's own.
    for name, alloc_fn, out_fn, _ in FORWARD:
        wide = tp.zeros(2 * ROWS, COLS, dtype=tp.float32)
        x = wide[::2]
        x += matrix(tp.float32)
        want = read(alloc_fn(x, 1))
        out = unit(tp.float32)
        assert agree(read(out_fn(x, 1, out)), want, RTOL_OF[tp.float32]), name


def test_softmax_out_answers_a_destination_laid_out_as_the_input_is_not():
    # A transposed destination has the answer's shape and width but is not walked
    # in element order, so the value arrives by copy -- and has to be the same.
    for name, alloc_fn, out_fn, _ in FORWARD:
        for dim in (0, 1):
            x = matrix(tp.float32).transpose(0, 1)
            want = read(alloc_fn(x, dim))
            dest = tp.zeros(COLS, ROWS, dtype=tp.float32).transpose(0, 1)
            out_fn(x, dim, dest)
            assert agree(read(dest), want, RTOL_OF[tp.float32]), (name, dim)


def test_softmax_out_answers_a_destination_that_is_a_window_of_the_input():
    # Same block, different window: the walk would meet the same bytes at two
    # offsets, so it declines and the answer still arrives whole.
    for name, alloc_fn, out_fn, _ in FORWARD:
        for dim in (0, 1):
            block = tp.zeros(2 * ROWS, COLS, dtype=tp.float32)
            x = block[:ROWS]
            x += matrix(tp.float32)
            want = read(alloc_fn(x, dim))
            dest = block[ROWS:]
            out_fn(x, dim, dest)
            assert agree(read(dest), want, RTOL_OF[tp.float32]), (name, dim)


def test_softmax_out_answers_an_input_that_is_its_own_destination():
    # The plain walk reads an element before it stores over it, so the same tensor
    # serving as sample and destination is one it walks.  The logarithmic walk
    # reads every element of a row again after the exponentials were stored there,
    # so it has to take that answer the other way.  Either way the answer is the
    # one a fresh destination would have got.
    for name, alloc_fn, out_fn, _ in FORWARD:
        for dim in (0, 1):
            x = matrix(tp.float32)
            want = read(alloc_fn(x, dim))
            result = out_fn(x, dim, x)
            assert result.data_ptr() == x.data_ptr(), (name, dim)
            assert agree(read(result), want, RTOL_OF[tp.float32]), (name, dim)


def test_softmax_out_answers_a_strided_sample_in_a_row_that_is_not_last():
    # dim=0 of a three-dimensional sample is the streaming walk over strided rows.
    for name, alloc_fn, out_fn, _ in FORWARD:
        x = matrix(tp.float32, ROWS, COLS, deep=True)
        for dim in (0, 1, 2):
            want = read(alloc_fn(x, dim))
            out = tp.zeros(3, ROWS, COLS, dtype=tp.float32)
            assert agree(read(out_fn(x, dim, out)), want,
                         RTOL_OF[tp.float32]), (name, dim)


def test_softmax_out_refuses_the_half_to_float_conversion_it_always_refused():
    x = matrix(tp.float32)
    for call in (lambda: tp._C._softmax(x, 1, True),
                 lambda: tp._C._log_softmax(x, 1, True),
                 lambda: tp._C._softmax(x, 1, True, out=unit(tp.float32)),
                 lambda: tp._C._log_softmax(x, 1, True, out=unit(tp.float32))):
        try:
            call()
        except Exception:
            continue
        raise AssertionError("half_to_float answered where it refused")


def test_softmax_out_keeps_a_dim_out_of_range_refusing():
    for name, alloc_fn, out_fn, _ in FORWARD:
        x = matrix(tp.float32)
        out = unit(tp.float32)
        for dim in (-3, 2):
            for call in (lambda: alloc_fn(x, dim), lambda: out_fn(x, dim, out)):
                try:
                    call()
                except Exception:
                    continue
                raise AssertionError(f"{name} answered dim {dim} as before")


def test_softmax_out_answers_a_scalar_sample():
    # One normalization unit: the walk declines a destination for it, which is the
    # one place the answer still has to arrive by copy.
    for name, alloc_fn, out_fn, _ in FORWARD:
        x = matrix(tp.float32, 1, 1).view([])
        want = read(alloc_fn(x, 0))
        out = tp.ones([], dtype=tp.float32)
        result = out_fn(x, 0, out)
        assert read(result).size == 1, name
        assert agree(read(result).reshape(()), want.reshape(()),
                     RTOL_OF[tp.float32]), name


def test_softmax_out_answers_an_empty_sample():
    for name, alloc_fn, out_fn, _ in FORWARD:
        empty = tp.zeros([0, COLS], dtype=tp.float32)
        assert read(alloc_fn(empty, 1)).size == 0, name
        out = tp.zeros([0, COLS], dtype=tp.float32)
        result = out_fn(empty, 1, out)
        assert result.data_ptr() == out.data_ptr(), name
        assert read(result).size == 0, name


def test_softmax_out_grows_a_destination_too_small_for_the_answer():
    for name, alloc_fn, out_fn, _ in FORWARD:
        x = matrix(tp.float32)
        want = read(alloc_fn(x, 1))
        out = tp.zeros([1], dtype=tp.float32)
        out_fn(x, 1, out)
        assert agree(read(out), want, RTOL_OF[tp.float32]), name


# ---------------------------------------------------------------------------
# the backward walks
# ---------------------------------------------------------------------------

def test_softmax_backward_out_fills_the_destination_it_is_given():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        for dtype in WIDTHS:
            for dim in (0, 1):
                x = matrix(dtype)
                saved = (tp._C._log_softmax if log else tp._C._softmax)(
                    x, dim, False)
                g = grad_matrix(dtype)
                wide = stepped(dtype)
                out_fn(g, saved, dim, wide)
                want = read(wide)

                out = unit(dtype)
                result = out_fn(g, saved, dim, out)
                assert result.data_ptr() == out.data_ptr(), (name, dtype, dim)
                assert read(result).tolist() == want.tolist(), (name, dtype, dim)


def test_softmax_backward_out_answers_the_derivative_it_is_for():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        for dim in (0, 1):
            x = read(matrix(tp.float64))
            g = read(grad_matrix(tp.float64))
            saved = log_value(x, dim) if log else value(x, dim)
            want = log_gradient(g, saved, dim) if log else gradient(g, saved, dim)
            g_tensor, saved_tensor = tensor_of(g, tp.float64), tensor_of(saved,
                                                                        tp.float64)
            out = unit(tp.float64)
            assert agree(read(out_fn(g_tensor, saved_tensor, dim, out)), want,
                         RTOL_OF[tp.float64]), name
            assert agree(read(alloc_fn(g_tensor, saved_tensor, dim)), want,
                         RTOL_OF[tp.float64]), name


def test_softmax_backward_out_answers_an_operand_that_is_its_own_destination():
    # Either operand may be the buffer the answer goes into: the walk sums the
    # column before it stores into it, and then reads and writes that column
    # element by element.
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        for which in ("gradient", "saved"):
            x = matrix(tp.float32)
            saved = (tp._C._log_softmax if log else tp._C._softmax)(x, 1, False)
            g = grad_matrix(tp.float32)
            want = read(alloc_fn(g, saved, 1))
            dest = g if which == "gradient" else saved
            result = out_fn(g, saved, 1, dest)
            assert result.data_ptr() == dest.data_ptr(), (name, which)
            assert agree(read(result), want, RTOL_OF[tp.float32]), (name, which)


def test_softmax_backward_out_answers_a_destination_of_another_width():
    # The backward pass answers at the gradient's width unless the original input
    # was half precision; a destination the walk cannot serve is still cast.
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        for dtype, other in ((tp.float32, tp.float64), (tp.float64, tp.float32)):
            x = matrix(dtype)
            saved = (tp._C._log_softmax if log else tp._C._softmax)(x, 1, False)
            g = grad_matrix(dtype)
            want = read(alloc_fn(g, saved, 1))
            out = tp.zeros(ROWS, COLS, dtype=other)
            out_fn(g, saved, 1, out)
            assert agree(read(out), read(tp.from_numpy(
                np.array(want, dtype=NP_OF[other])).to(other)),
                RTOL_OF[other]), (name, other)


def test_softmax_backward_out_answers_a_half_precision_gradient():
    # A gradient narrower than single precision is computed wide and narrowed
    # afterwards, so the destination is filled by that second pass -- and the
    # width the answer comes in is the original input's.
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        x = matrix(tp.float16)
        saved = (tp._C._log_softmax if log else tp._C._softmax)(x, 1, False)
        g = grad_matrix(tp.float16)
        want = read(alloc_fn(g, saved, 1))
        out = unit(tp.float16)
        result = out_fn(g, saved, 1, out)
        assert result.data_ptr() == out.data_ptr(), name
        assert agree(read(result), want, RTOL_OF[tp.float16]), name


def test_softmax_backward_out_answers_a_strided_operand():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        wide = tp.zeros(2 * ROWS, COLS, dtype=tp.float32)
        g = wide[:ROWS]
        g += grad_matrix(tp.float32)
        saved = (tp._C._log_softmax if log else tp._C._softmax)(
            matrix(tp.float32), 1, False)
        want = read(alloc_fn(g, saved, 1))
        out = unit(tp.float32)
        assert agree(read(out_fn(g, saved, 1, out)), want,
                     RTOL_OF[tp.float32]), name


def test_softmax_backward_out_answers_a_destination_laid_out_as_the_operand_is_not():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        x = matrix(tp.float32)
        saved = (tp._C._log_softmax if log else tp._C._softmax)(x, 0, False)
        g = grad_matrix(tp.float32)
        want = read(alloc_fn(g, saved, 0))
        dest = tp.zeros(COLS, ROWS, dtype=tp.float32).transpose(0, 1)
        out_fn(g, saved, 0, dest)
        assert agree(read(dest), want, RTOL_OF[tp.float32]), name


def test_softmax_backward_out_answers_a_scalar_operand():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        g = tp.zeros([], dtype=tp.float32)
        g += 0.5
        saved = (tp._C._log_softmax if log else tp._C._softmax)(
            matrix(tp.float32, 1, 1).view([]), 0, False)
        want = read(alloc_fn(g, saved, 0))
        out = tp.ones([], dtype=tp.float32)
        assert agree(read(out_fn(g, saved, 0, out)).reshape(()),
                     want.reshape(()), RTOL_OF[tp.float32]), name


def test_softmax_backward_out_answers_an_empty_operand():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        empty = tp.zeros([0, COLS], dtype=tp.float32)
        assert read(alloc_fn(empty, empty, 1)).size == 0, name
        out = tp.zeros([0, COLS], dtype=tp.float32)
        result = out_fn(empty, empty, 1, out)
        assert result.data_ptr() == out.data_ptr(), name
        assert read(result).size == 0, name


def test_softmax_backward_out_grows_a_destination_too_small_for_the_answer():
    for name, alloc_fn, out_fn, _, log in BACKWARD:
        x = matrix(tp.float32)
        saved = (tp._C._log_softmax if log else tp._C._softmax)(x, 1, False)
        g = grad_matrix(tp.float32)
        want = read(alloc_fn(g, saved, 1))
        out = tp.zeros([1], dtype=tp.float32)
        out_fn(g, saved, 1, out)
        assert agree(read(out), want, RTOL_OF[tp.float32]), name
