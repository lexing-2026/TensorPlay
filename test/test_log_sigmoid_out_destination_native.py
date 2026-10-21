#!/usr/bin/env python3
"""Holding the log_sigmoid family to one pass into the destination it is given.

Three spellings answer the same map: ``log_sigmoid`` gives the value, its
``.output`` form hands over the exponential the backward reads alongside it, and
``log_sigmoid_backward`` takes a gradient and that saved exponential back to a
derivative.  The forward walk stores the value beside the exponential it saves,
and the backward walk reads three buffers, so every one of these destinations is
judged on its own -- and, for two walks that step through storage in element
order, so is every operand they read.  A destination one of them cannot walk is
still answered, by computing the value and handing it over, which is what makes
the two routes comparable: whatever route ran, the answer has to be the same
element for element.  What each map gives is also stated here in plain
arithmetic, so a route cannot agree with its own twin and still be wrong.
"""
import numpy as np
import tensorplay as tp

NP_OF = {tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16, tp.bfloat16: np.float32}

WIDTHS = (tp.float32, tp.float64, tp.float16, tp.bfloat16)

LENGTH = 257  # a tail behind every vector width
ROWS, COLS = 13, 19  # neither factor a vector width

rng = np.random.default_rng(7)
# Both branches of the map, the point they join at, and magnitudes large enough
# that the exponential the forward saves underflows to nothing.
SAMPLE = np.array(rng.standard_normal(2 * LENGTH) * 6.0, dtype=np.float64)
SAMPLE[0:5] = (0.0, -0.0, 700.0, -700.0, 1.0)
GRAD = np.array(rng.standard_normal(2 * LENGTH) * 3.0, dtype=np.float64)
GRAD[0] = 0.0


def value(v):
    """log(sigmoid(x)) = min(x, 0) - log1p(exp(-|x|))."""
    return np.minimum(v, 0.0) - np.log1p(np.exp(-np.abs(v)))


def saved(v):
    """The exponential the forward keeps and the backward reads."""
    return np.exp(-np.abs(v))


def derivative(v):
    """d/dx log(sigmoid(x)) = sigmoid(-x), split so exp() never runs away."""
    with np.errstate(over="ignore"):
        return np.where(v < 0.0, 1.0 / (1.0 + np.exp(v)),
                        np.exp(-v) / (1.0 + np.exp(-v)))


def sample(dtype, count=LENGTH):
    return tp.from_numpy(np.array(SAMPLE[:count], dtype=NP_OF[dtype])).to(dtype)


def grad(dtype, count=LENGTH):
    return tp.from_numpy(np.array(GRAD[:count], dtype=NP_OF[dtype])).to(dtype)


def read(t):
    """The logical elements, one at a time: a narrow width reached through a
    stride is only faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def agree(got, want, rtol=1e-3, atol=1e-4):
    return np.allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)


# How far a plain-arithmetic statement may sit from an answer depends on the
# width that answer was asked to come in: a narrow one keeps about three decimal
# digits of the value it holds.
RTOL_OF = {tp.float32: 1e-4, tp.float64: 1e-10,
           tp.float16: 2e-3, tp.bfloat16: 2e-2}


def stepped(dtype, count=LENGTH):
    """A destination every one of these walks has to step through instead of
    writing in element order -- which is how the answer arrives by copy."""
    return tp.zeros(2 * count, dtype=dtype)[::2]


# ---------------------------------------------------------------------------
# log_sigmoid: the value alone
# ---------------------------------------------------------------------------

def test_log_sigmoid_out_fills_the_destination_it_is_given():
    # The packed destination is written by the pass; the stepped one is served by
    # computing and handing over.  Element for element they are the same answer.
    for dtype in WIDTHS:
        x = sample(dtype)
        stepped_out = stepped(dtype)
        tp._C.log_sigmoid(x, out=stepped_out)
        want = read(stepped_out)

        out = tp.zeros(LENGTH, dtype=dtype)
        result = tp._C.log_sigmoid(x, out=out)
        assert result.data_ptr() == out.data_ptr(), dtype
        assert read(result).tolist() == want.tolist(), dtype


def test_log_sigmoid_out_answers_the_map_it_is_for():
    for dtype in WIDTHS:
        x = sample(dtype)
        want = value(read(x))
        assert agree(read(tp._C.log_sigmoid(x)), want, RTOL_OF[dtype]), dtype
        out = tp.zeros(LENGTH, dtype=dtype)
        assert agree(read(tp._C.log_sigmoid(x, out=out)), want,
                     RTOL_OF[dtype]), dtype


def test_log_sigmoid_out_answers_an_integral_input_at_single_precision():
    # Integral elements answer at single precision, and a destination of that
    # width is the answer's own: the pass writes into it instead of meeting a
    # tensor the map allocated for the same elements.
    x = tp.from_numpy(np.array(SAMPLE[:LENGTH] * 16, dtype=np.int64))
    want = read(tp._C.log_sigmoid(x))
    out = tp.zeros(LENGTH, dtype=tp.float32)
    result = tp._C.log_sigmoid(x, out=out)
    assert result.data_ptr() == out.data_ptr()
    assert read(result).tolist() == want.tolist()


def test_log_sigmoid_out_answers_a_destination_laid_out_as_the_input_is():
    for dtype in WIDTHS:
        x = sample(dtype, ROWS * COLS).reshape(ROWS, COLS).transpose(0, 1)
        want = read(tp._C.log_sigmoid(x))
        dest = tp.zeros(ROWS, COLS, dtype=dtype).transpose(0, 1)
        tp._C.log_sigmoid(x, out=dest)
        assert agree(read(dest), want), dtype


def test_log_sigmoid_out_answers_a_destination_laid_out_as_the_input_is_not():
    for dtype in WIDTHS:
        x = sample(dtype, ROWS * COLS).reshape(ROWS, COLS)
        want = read(tp._C.log_sigmoid(x))
        stepped_dest = tp.zeros(COLS, ROWS, dtype=dtype).transpose(0, 1)
        tp._C.log_sigmoid(x, out=stepped_dest)
        assert agree(read(stepped_dest), want), dtype

        xt = x.transpose(0, 1)
        want_t = read(tp._C.log_sigmoid(xt))
        packed = tp.zeros(COLS, ROWS, dtype=dtype)
        tp._C.log_sigmoid(xt, out=packed)
        assert agree(read(packed), want_t), dtype


def test_log_sigmoid_out_keeps_a_destination_of_another_width_refused():
    # The answer comes at the input's own width, so a destination of another one
    # is refused rather than cast into.
    for dtype in WIDTHS:
        x = sample(dtype)
        other = tp.zeros(LENGTH, dtype=tp.float64 if dtype != tp.float64
                         else tp.float32)
        try:
            tp._C.log_sigmoid(x, out=other)
        except Exception:
            continue
        raise AssertionError(
            f"log_sigmoid cast into a width it does not answer at for {dtype}")


# ---------------------------------------------------------------------------
# log_sigmoid_forward.output: the value and the exponential saved with it
# ---------------------------------------------------------------------------

def test_forward_output_fills_both_destinations_it_is_given():
    for dtype in WIDTHS:
        x = sample(dtype)
        stepped_out, stepped_buf = stepped(dtype), stepped(dtype)
        tp._C.log_sigmoid_forward(x, output=stepped_out, buffer=stepped_buf)
        want_value, want_saved = read(stepped_out), read(stepped_buf)

        out = tp.zeros(LENGTH, dtype=dtype)
        buf = tp.zeros(LENGTH, dtype=dtype)
        result = tp._C.log_sigmoid_forward(x, output=out, buffer=buf)
        assert result[0].data_ptr() == out.data_ptr(), dtype
        assert result[1].data_ptr() == buf.data_ptr(), dtype
        assert read(result[0]).tolist() == want_value.tolist(), dtype
        assert read(result[1]).tolist() == want_saved.tolist(), dtype


def test_forward_output_answers_the_value_and_the_exponential():
    for dtype in WIDTHS:
        x = sample(dtype)
        want_value, want_saved = value(read(x)), saved(read(x))
        computed = tp._C.log_sigmoid_forward(x)
        assert agree(read(computed[0]), want_value, RTOL_OF[dtype]), dtype
        assert agree(read(computed[1]), want_saved, RTOL_OF[dtype]), dtype
        out = tp.zeros(LENGTH, dtype=dtype)
        buf = tp.zeros(LENGTH, dtype=dtype)
        tp._C.log_sigmoid_forward(x, output=out, buffer=buf)
        assert agree(read(out), want_value, RTOL_OF[dtype]), dtype
        assert agree(read(buf), want_saved, RTOL_OF[dtype]), dtype


def test_forward_output_writes_the_input_destination_in_place():
    # The walk reads the element at a position before it stores either answer
    # there, so the operand it reads may be the buffer the value lands in.
    for dtype in WIDTHS:
        want_value = read(tp._C.log_sigmoid_forward(sample(dtype))[0])
        x = sample(dtype)
        buf = tp.zeros(LENGTH, dtype=dtype)
        tp._C.log_sigmoid_forward(x, output=x, buffer=buf)
        assert read(x).tolist() == want_value.tolist(), dtype
        assert agree(read(buf), saved(read(sample(dtype))), RTOL_OF[dtype]), dtype


def test_forward_output_answers_an_exponential_destination_in_place():
    for dtype in WIDTHS:
        want_saved = read(tp._C.log_sigmoid_forward(sample(dtype))[1])
        out = tp.zeros(LENGTH, dtype=dtype)
        buf = sample(dtype)
        tp._C.log_sigmoid_forward(sample(dtype), output=out, buffer=buf)
        assert read(buf).tolist() == want_saved.tolist(), dtype
        assert agree(read(out), value(read(sample(dtype))),
                     RTOL_OF[dtype]), dtype


def test_forward_output_answers_a_destination_it_has_to_copy_into():
    # One destination that can take the pass and one that cannot: the pass that
    # stores both answers at once is then offered to neither, and both arrive by
    # the hand-over.
    for dtype in WIDTHS:
        x = sample(dtype)
        computed = tp._C.log_sigmoid_forward(sample(dtype))
        want_value, want_saved = read(computed[0]), read(computed[1])

        for step_out, step_buf in ((True, True), (True, False), (False, True)):
            out = stepped(dtype) if step_out else tp.zeros(LENGTH, dtype=dtype)
            buf = stepped(dtype) if step_buf else tp.zeros(LENGTH, dtype=dtype)
            tp._C.log_sigmoid_forward(x, output=out, buffer=buf)
            tag = (dtype, step_out, step_buf)
            assert agree(read(out), want_value), tag
            assert agree(read(buf), want_saved), tag


def test_forward_output_answers_a_strided_operand():
    x = sample(tp.float32, 2 * LENGTH)[::2]
    want = tp._C.log_sigmoid_forward(x)
    out = tp.zeros(LENGTH, dtype=tp.float32)
    buf = tp.zeros(LENGTH, dtype=tp.float32)
    got = tp._C.log_sigmoid_forward(x, output=out, buffer=buf)
    assert read(got[0]).tolist() == read(want[0]).tolist()
    assert read(got[1]).tolist() == read(want[1]).tolist()


# ---------------------------------------------------------------------------
# log_sigmoid_backward.grad_input: the gradient through the saved exponential
# ---------------------------------------------------------------------------

def pair(dtype, count=LENGTH):
    """The gradient, the operand it was taken of, and the exponential the forward
    saved -- all at the width the answer comes in."""
    x = sample(dtype, count)
    return grad(dtype, count), x, tp._C.log_sigmoid_forward(x)[1]


def stepped_answer(dtype, g, x, buf):
    dest = stepped(dtype)
    tp._C.log_sigmoid_backward(g, x, buf, grad_input=dest)
    return read(dest)


def test_backward_grad_input_fills_the_destination_it_is_given():
    for dtype in WIDTHS:
        g, x, buf = pair(dtype)
        want = stepped_answer(dtype, g, x, buf)
        out = tp.zeros(LENGTH, dtype=dtype)
        result = tp._C.log_sigmoid_backward(g, x, buf, grad_input=out)
        assert result.data_ptr() == out.data_ptr(), dtype
        assert read(result).tolist() == want.tolist(), dtype


def test_backward_grad_input_answers_the_derivative_it_is_for():
    for dtype in WIDTHS:
        g, x, buf = pair(dtype)
        # The exponential the walk reads is the one the forward saved at the
        # operand's width, so the statement is held to that same rounding.
        expect = read(g) * derivative(read(x))
        out = tp.zeros(LENGTH, dtype=dtype)
        tp._C.log_sigmoid_backward(g, x, buf, grad_input=out)
        assert agree(read(out), expect, RTOL_OF[dtype]), dtype
        assert agree(read(tp._C.log_sigmoid_backward(g, x)), expect,
                     RTOL_OF[dtype]), dtype


def test_backward_grad_input_writes_an_operand_destination_in_place():
    # Any of the three buffers the walk reads may be the one the answer lands in:
    # the element at a position is read before it is stored there.
    for dtype in WIDTHS:
        g, x, buf = pair(dtype)
        want = stepped_answer(dtype, g, x, buf)

        as_grad, untouched_x, untouched_buf = pair(dtype)
        tp._C.log_sigmoid_backward(as_grad, untouched_x, untouched_buf,
                                   grad_input=as_grad)
        assert read(as_grad).tolist() == want.tolist(), (dtype, "gradient")

        untouched_grad, as_x, untouched_buf = pair(dtype)
        tp._C.log_sigmoid_backward(untouched_grad, as_x, untouched_buf,
                                   grad_input=as_x)
        assert read(as_x).tolist() == want.tolist(), (dtype, "operand")

        untouched_grad, untouched_x, as_buf = pair(dtype)
        tp._C.log_sigmoid_backward(untouched_grad, untouched_x, as_buf,
                                   grad_input=as_buf)
        assert read(as_buf).tolist() == want.tolist(), (dtype, "saved exponential")


def test_backward_grad_input_answers_a_destination_it_has_to_copy_into():
    for dtype in WIDTHS:
        g, x, buf = pair(dtype)
        want = read(tp._C.log_sigmoid_backward(g, x, buf,
                                               grad_input=tp.zeros(LENGTH, dtype=dtype)))
        strided = stepped(dtype)
        tp._C.log_sigmoid_backward(g, x, buf, grad_input=strided)
        assert agree(read(strided), want), (dtype, "strided")

        # A destination of another width is served by the hand-over, which casts
        # the answer into it.  The elements survive that step the way the width
        # leaves them, whichever way they were written.
        for other in (tp.float64, tp.float32):
            if other == dtype:
                continue
            dest = tp.zeros(LENGTH, dtype=other)
            tp._C.log_sigmoid_backward(g, x, buf, grad_input=dest)
            assert agree(read(dest), want,
                         max(RTOL_OF[dtype], RTOL_OF[other])), (dtype, other)


def test_backward_grad_input_answers_strided_operands():
    x_full = sample(tp.float32, 2 * LENGTH)
    g = grad(tp.float32, 2 * LENGTH)[::2]
    x = x_full[::2]
    buf = tp._C.log_sigmoid_forward(x_full)[1][::2]
    want = stepped_answer(tp.float32, g, x, buf)
    out = tp.zeros(LENGTH, dtype=tp.float32)
    tp._C.log_sigmoid_backward(g, x, buf, grad_input=out)
    assert agree(read(out), want)


def test_backward_grad_input_answers_a_pair_of_two_widths():
    # The two saved buffers are widened to the width the gradient comes in, so a
    # pair of two widths has an answer -- and both routes give the same one.
    g = grad(tp.float32)
    _, x64, buf64 = pair(tp.float64)
    want = stepped_answer(tp.float32, g, x64, buf64)
    out = tp.zeros(LENGTH, dtype=tp.float32)
    result = tp._C.log_sigmoid_backward(g, x64, buf64, grad_input=out)
    assert result.data_ptr() == out.data_ptr()
    assert read(result).tolist() == want.tolist()


def test_a_destination_of_the_wrong_shape_is_grown_to_the_answer():
    for dtype in WIDTHS:
        g, x, buf = pair(dtype)

        want = stepped_answer(dtype, g, x, buf)
        grown = tp.zeros([1], dtype=dtype)
        tp._C.log_sigmoid_backward(g, x, buf, grad_input=grown)
        assert agree(read(grown), want), (dtype, "grad_input")

        computed = tp._C.log_sigmoid_forward(sample(dtype))
        small_out = tp.zeros([1], dtype=dtype)
        small_buf = tp.zeros([1], dtype=dtype)
        tp._C.log_sigmoid_forward(sample(dtype), output=small_out, buffer=small_buf)
        assert agree(read(small_out), read(computed[0])), (dtype, "output")
        assert agree(read(small_buf), read(computed[1])), (dtype, "buffer")

        single = tp.zeros([1], dtype=dtype)
        tp._C.log_sigmoid(sample(dtype), out=single)
        assert agree(read(single), read(tp._C.log_sigmoid(sample(dtype)))), \
            (dtype, "out")


def test_an_empty_pair_answers_an_empty_destination():
    for dtype in WIDTHS:
        empty = tp.zeros([0], dtype=dtype)
        got = tp._C.log_sigmoid_forward(empty)
        assert read(got[0]).size == 0 and read(got[1]).size == 0, dtype

        out, buf = tp.zeros([0], dtype=dtype), tp.zeros([0], dtype=dtype)
        got = tp._C.log_sigmoid_forward(empty, output=out, buffer=buf)
        assert got[0].data_ptr() == out.data_ptr(), dtype
        assert got[1].data_ptr() == buf.data_ptr(), dtype
        assert read(got[0]).size == 0 and read(got[1]).size == 0, dtype

        grad_input = tp.zeros([0], dtype=dtype)
        result = tp._C.log_sigmoid_backward(empty, empty, empty,
                                            grad_input=grad_input)
        assert result.data_ptr() == grad_input.data_ptr(), dtype
        assert read(result).size == 0, dtype

        single = tp.zeros([0], dtype=dtype)
        assert read(tp._C.log_sigmoid(empty, out=single)).size == 0, dtype
