#!/usr/bin/env python3
"""Holding rrelu_with_noise to one pass into the tensors it is handed.

Two routes answer this map.  Training mode runs one serial draw over the input,
laying the answer at element i of its destination and the drawn factor at element
i of the noise, so both of those have to be packed at the input's own shape and
width before either can be walked.  Evaluation mode draws nothing at all and is the
midpoint slope applied to the input -- a map with a destination of its own.  What
either route cannot walk -- a tensor reached at a stride, a second window of the
bytes still to be read, a noise drawn over the very element the answer has to
read, a destination of another shape or width -- computes the answer and hands it
over.  Whatever route ran, the answer and the factors have to be the same element
for element, and what the draw gives is stated here in plain arithmetic too: the
answer is the input scaled by the factor it was filled with.

The factor is laid down before the answer reads its element wherever that element
is positive, so a noise taken from the input's own block is refused even when the
two are the same shape.  Answer and noise asked for one and the same tensor is
refused for the same reason: the run that hands the answer over fills the noise
first and the answer last, and one walk cannot put two stores at one element in
that order.

Every assertion reads the tensor the caller handed in, since that is what the
op contractually fills; the plain call has no destination to read, so its answer
comes from the tensor it returns.
"""
import numpy as np
import tensorplay as tp

NP_OF = {tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16, tp.bfloat16: np.float32}
RTOL_OF = {tp.float32: 1e-4, tp.float64: 1e-12,
           tp.float16: 2e-3, tp.bfloat16: 2e-2}

WIDTHS = (tp.float32, tp.float64, tp.float16, tp.bfloat16)

ROWS, COLS = 5, 7  # neither factor a vector width, and their product is odd
N = ROWS * COLS
LOWER, UPPER = 0.125, 0.333
MID = (LOWER + UPPER) / 2.0
SEED = 2026

rng = np.random.default_rng(7)
# Every branch the comparison cares about: a positive element, a negative one,
# both zeroes (a zero takes a draw, the branch being `<= 0`), a magnitude large
# enough to run away if the slope were ever applied twice, and a NaN, which fails
# `<= 0` and so keeps the slope of 1.
BASE = np.array(rng.standard_normal(N) * 2.0, dtype=np.float64)
BASE[0] = 0.0
BASE[1] = -0.0
BASE[2] = np.nan
BASE[3] = 640.0
BASE[4] = -640.0


def read(t):
    """The logical elements: a width narrower than single precision, or a stride,
    is only faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def agree(got, want, rtol):
    return np.allclose(got, want, rtol=rtol, atol=rtol, equal_nan=True)


def leaky(source):
    return np.where(source > 0, source, source * MID)


def matrix(dtype):
    return tp.from_numpy(np.array(BASE, dtype=NP_OF[dtype])).to(dtype).reshape(
        ROWS, COLS)


def tensor_of(flat, dtype):
    return tp.from_numpy(np.array(flat, dtype=NP_OF[dtype])).to(dtype)


def filled(value, dtype, shape=(ROWS, COLS)):
    return tp.full(list(shape), value, dtype=dtype)


def packed(dtype, shape=(ROWS, COLS)):
    return tp.zeros(list(shape), dtype=dtype)


def stepped(dtype):
    return tp.zeros([2 * ROWS, COLS], dtype=dtype)[::2]


def draw(x, noise, out, training=True):
    """One call, with the default generator's stream rewound to the same seed so
    that two routes are handed the same draws.  The tensor handed back is the
    destination when the caller gave one, and the answer the op allocated
    otherwise."""
    tp.manual_seed(SEED)
    if out is None:
        return tp._C.rrelu_with_noise(x, noise, LOWER, UPPER, training, None)
    tp._C.rrelu_with_noise(x, noise, LOWER, UPPER, training, None, out=out)
    return out


def drawn(x, noise, out, training=True):
    """The answer and the factors of one call, as numbers."""
    got = draw(x, noise, out, training)
    return read(got), read(noise)


# ---------------------------------------------------------------------------
# training mode: one draw, two destinations
# ---------------------------------------------------------------------------


def test_rrelu_out_fills_the_two_destinations_it_is_given():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        answer_plain, noise_plain = drawn(matrix(dtype), filled(1.0, dtype),
                                          None)
        held = packed(dtype)
        held_ptr = held.data_ptr()
        answer_held, noise_held = drawn(matrix(dtype), filled(1.0, dtype), held)
        stepped_dest = stepped(dtype)
        stepped_ptr = stepped_dest.data_ptr()
        answer_stepped, noise_stepped = drawn(matrix(dtype), filled(1.0, dtype),
                                              stepped_dest)
        assert agree(answer_held, answer_plain, rtol), dtype
        assert agree(noise_held, noise_plain, rtol), dtype
        assert agree(answer_stepped, answer_plain, rtol), dtype
        assert agree(noise_stepped, noise_plain, rtol), dtype
        # Both destinations keep their own storage: the caller's tensors hold the
        # answer and the factors, and nothing new was allocated for it.
        assert held.data_ptr() == held_ptr, dtype
        assert stepped_dest.data_ptr() == stepped_ptr, dtype


def test_rrelu_out_answers_the_draw_it_is_for():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        x = matrix(dtype)
        source = read(x).ravel()
        answer, factors = drawn(x, filled(1.0, dtype), packed(dtype))
        answer = answer.ravel()
        factors = factors.ravel()
        # The answer is the input scaled by the factor it was filled with at every
        # element: a positive one keeps a factor of 1, and so its own value.
        assert agree(answer, source * factors, rtol), dtype
        # A zero or a negative takes a draw from the interval.  The NaN element
        # fails both branches, so it keeps the slope of 1 like the positives.
        taken = source <= 0
        assert np.all((factors[taken] >= LOWER) &
                      (factors[taken] <= UPPER)), dtype
        assert np.all(factors[~taken] == 1.0), dtype
        # Both zeroes take a draw; the NaN keeps the slope of 1 and its own value.
        assert factors[0] != 1.0 and factors[1] != 1.0, dtype
        assert factors[2] == 1.0, dtype
        assert np.isnan(answer[2]), dtype


def test_rrelu_inplace_answers_the_input_it_was_handed():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        x = matrix(dtype)
        source = read(x)
        noise = filled(1.0, dtype)
        tp.manual_seed(SEED)
        got = tp._C.rrelu_with_noise_(x, noise, LOWER, UPPER, True, None)
        assert got.data_ptr() == x.data_ptr(), dtype
        answer, factors = read(x), read(noise)
        assert agree(answer, source * factors, rtol), dtype
        twin = matrix(dtype)
        want, want_factors = drawn(twin, filled(1.0, dtype), packed(dtype))
        assert agree(answer, want, rtol), dtype
        assert agree(factors, want_factors, rtol), dtype


def test_rrelu_out_answers_an_input_that_is_its_own_destination():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        x = matrix(dtype)
        source = read(x)
        noise = filled(1.0, dtype)
        answer, factors = drawn(x, noise, x)
        assert agree(answer, source * factors, rtol), dtype
        twin = matrix(dtype)
        want, want_factors = drawn(twin, filled(1.0, dtype), packed(dtype))
        assert agree(answer, want, rtol), dtype
        assert agree(factors, want_factors, rtol), dtype


def test_rrelu_out_answers_a_noise_taken_from_the_input():
    """A factor laid over the element the answer still has to read is the pairing
    the walk refuses; the handed-over answer is what the run always gave."""
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        answer, factors = drawn(matrix(dtype), matrix(dtype), packed(dtype))
        want, want_factors = drawn(matrix(dtype), matrix(dtype), None)
        assert agree(answer, want, rtol), dtype
        assert agree(factors, want_factors, rtol), dtype


def test_rrelu_out_answers_one_tensor_asked_for_as_both_destinations():
    """The noise is filled during the run and the answer handed over afterwards,
    so the tensor the caller holds ends up holding answers, not factors."""
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        one = packed(dtype)
        want, _ = drawn(matrix(dtype), filled(1.0, dtype), None)
        drawn(matrix(dtype), one, one)
        assert agree(read(one), want, rtol), dtype


def test_rrelu_out_answers_a_destination_that_is_a_window_of_the_input():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        block = tensor_of(np.concatenate([BASE, np.zeros(N)]), dtype)
        x = block[:N].reshape(ROWS, COLS)
        source = read(x)
        window = block[1:N + 1].reshape(ROWS, COLS)
        noise = filled(1.0, dtype)
        answer, factors = drawn(x, noise, window)
        assert agree(answer, source * factors, rtol), dtype


def test_rrelu_out_answers_an_input_it_has_to_lay_out_first():
    """A stepped input is drawn from a copy made for it, which leaves the
    destination outside every block the run reads."""
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        stepped_input = tensor_of(BASE, dtype).reshape(COLS, ROWS).t()
        source = read(stepped_input)
        answer, factors = drawn(stepped_input, filled(1.0, dtype),
                                packed(dtype))
        assert agree(answer, source * factors, rtol), dtype
        twin = tensor_of(BASE, dtype).reshape(COLS, ROWS).t()
        want, want_factors = drawn(twin, filled(1.0, dtype), None)
        assert agree(answer, want, rtol), dtype
        assert agree(factors, want_factors, rtol), dtype


def test_rrelu_out_answers_a_destination_laid_out_as_the_input_is_not():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        x = matrix(dtype)
        source = read(x)
        noise = filled(1.0, dtype)
        transposed = filled(0.0, dtype, (COLS, ROWS)).t()
        drawn(x, noise, transposed)
        answer, factors = read(transposed), read(noise)
        assert agree(answer, source * factors, rtol), dtype
        want, _ = drawn(matrix(dtype), filled(1.0, dtype), None)
        assert agree(answer, want, rtol), dtype


def test_rrelu_out_grows_a_destination_too_small_for_the_answer():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        small = tp.zeros([2, 3], dtype=dtype)
        x = matrix(dtype)
        noise = filled(1.0, dtype)
        got = draw(x, noise, small)
        assert tuple(got.shape) == (ROWS, COLS), dtype
        answer, factors = read(got), read(noise)
        assert agree(answer, read(x) * factors, rtol), dtype


def test_rrelu_out_answers_a_scalar_sample():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        scalar = tensor_of([-2.5], dtype).reshape([])
        source = read(scalar)
        noise = filled(1.0, dtype, [])
        held = tp.zeros([], dtype=dtype)
        answer, factors = drawn(scalar, noise, held)
        assert answer.shape == () and factors.shape == (), dtype
        assert agree(answer, source * factors, rtol), dtype
        assert bool((factors >= LOWER) & (factors <= UPPER)), dtype
        twin = tensor_of([-2.5], dtype).reshape([])
        want, want_factors = drawn(twin, filled(1.0, dtype, []), None)
        assert agree(answer, want, rtol), dtype
        assert agree(factors, want_factors, rtol), dtype


def test_rrelu_out_answers_an_empty_sample():
    for dtype in WIDTHS:
        empty = tp.zeros([0, COLS], dtype=dtype)
        noise = tp.zeros([0, COLS], dtype=dtype)
        dest = tp.zeros([0, COLS], dtype=dtype)
        got = draw(empty, noise, dest)
        assert tuple(got.shape) == (0, COLS), dtype
        assert got.numel() == 0, dtype
        assert tuple(noise.shape) == (0, COLS), dtype


def test_rrelu_out_refusals_are_the_ones_it_always_gave():
    dtype = tp.float32
    x = matrix(dtype)
    dest = packed(dtype)
    try:
        draw(x, tp.ones([COLS, ROWS], dtype=dtype), dest)
        raise AssertionError("a noise of another shape was accepted")
    except RuntimeError as err:
        assert "shape" in str(err), str(err)
    try:
        draw(x, tp.ones([2 * ROWS, COLS], dtype=dtype)[::2], dest)
        raise AssertionError("a noise reached at a stride was accepted")
    except RuntimeError as err:
        assert "contiguous" in str(err), str(err)
    try:
        draw(x, tp.ones([ROWS, COLS], dtype=dtype).to(tp.float16), dest)
        raise AssertionError("a noise of another width was accepted")
    except (TypeError, RuntimeError) as err:
        assert str(err), "the refusal said nothing"


# ---------------------------------------------------------------------------
# evaluation mode: the midpoint slope, with no draw at all
# ---------------------------------------------------------------------------


def test_rrelu_out_eval_applies_the_midpoint_slope():
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        x = matrix(dtype)
        want = leaky(read(x))
        held = packed(dtype)
        answer, noise = drawn(x, filled(0.75, dtype), held, training=False)
        assert agree(answer, want, rtol), dtype
        # Nothing is drawn, so nothing is written into the noise.
        assert np.all(noise == 0.75), dtype
        stepped_answer, _ = drawn(matrix(dtype), filled(0.75, dtype),
                                  stepped(dtype), training=False)
        assert agree(stepped_answer, want, rtol), dtype


def test_rrelu_out_eval_answers_a_destination_it_cannot_walk():
    """A destination of another width, or the input serving as its own answer, is
    answered the way the slope map answers anything it cannot walk."""
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        x = matrix(dtype)
        want = leaky(read(x))
        wide = packed(tp.float64)
        # The slope is applied at the input's width before the answer widens.
        assert agree(read(draw(x, filled(1.0, dtype), wide, training=False)),
                     want, rtol), dtype
        in_place = matrix(dtype)
        source = read(in_place)
        answer = read(draw(in_place, filled(1.0, dtype), in_place,
                           training=False))
        assert agree(answer, leaky(source), rtol), dtype


def test_rrelu_out_eval_leaves_a_destination_of_the_wrong_shape_to_grow():
    for dtype in WIDTHS:
        small = tp.zeros([3, 2], dtype=dtype)
        x = matrix(dtype)
        got = draw(x, filled(1.0, dtype), small, training=False)
        assert tuple(got.shape) == (ROWS, COLS), dtype
        assert agree(read(got), leaky(read(x)), RTOL_OF[dtype]), dtype
