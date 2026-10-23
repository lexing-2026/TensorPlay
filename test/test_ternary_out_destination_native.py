#!/usr/bin/env python3
"""Holding the ternary float maps to one pass into the tensor they answer into.

Three maps of the same shape are asked about here: lerp between two tensors with
a weight that is either a number or a tensor, addcmul, and addcdiv.  Each has two
routes.  Where every operand comes in packed at one width the map is drawn
element by element straight onto the destination.  Where they do not -- one
operand broadcast, one operand carried to another width, a width narrower than
single precision -- the answer is composed and its last step is laid on the
destination, so the elements are still stored once, where the caller asked for
them.  A destination too wide for the cache the walk works in is stored past the
cache, which lays whole blocks and leaves the elements at the edges of every
span to the step that answers one at a time.

What is refused: a destination reached at a stride, because a strided store is
the slow scalar walk while a packed answer can be drawn by the fast one and then
handed over; an operand that is a second window of the bytes the destination
covers, because the walk reads elements it has already written; a destination of
another shape, which has to grow first.  Whatever route ran, the destination has
to hold what the allocating route answers, element for element, and the plain
arithmetic is stated here too.

The plain arithmetic is allowed the magnitudes the operands carry: a composition
rounds at every step, so a difference of two large terms can lose precision in
proportion to those terms however small the answer comes out.

Every assertion reads the tensor the caller handed in, since that is what these
maps contractually fill; the allocating spelling has no destination to read, so
its answer comes from the tensor it returns.
"""
import numpy as np
import tensorplay as tp

NP_OF = {tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16, tp.bfloat16: np.float32,
         tp.int64: np.int64}
RTOL_OF = {tp.float32: 1e-4, tp.float64: 1e-12,
           tp.float16: 2e-3, tp.bfloat16: 2e-2}

WIDTHS = (tp.float32, tp.float64, tp.float16, tp.bfloat16)

ROWS, COLS = 5, 7  # neither factor a vector width, and their product is odd
N = ROWS * COLS
LERP = 0.3
VALUE = 0.5

rng = np.random.default_rng(7)
# Every branch the maps care about: a NaN, which carries through; both zeroes; a
# magnitude large enough to run away if a term were applied twice; and the rest
# ordinary.  A weight stays clear of zero, and a divisor is clear of it except at
# the one element that answers the divide-by-zero rule -- whose dividend is kept
# positive so the infinity has one sign at every width.
SELF = np.array(rng.standard_normal(N) * 2.0, dtype=np.float64)
SELF[0] = np.nan
SELF[1] = 0.0
SELF[2] = -0.0
SELF[3] = 64.0
SELF[4] = -64.0
END = np.array(rng.standard_normal(N) * 2.0, dtype=np.float64)
END[5] = 1.5
WEIGHT = np.array(0.25 + rng.random(N), dtype=np.float64)
DIVISOR = np.array(rng.standard_normal(N) * 2.0 + 4.0, dtype=np.float64)
DIVISOR[5] = 0.0
# Two ends with nothing special in them, for the weights of nothing and everything:
# a weight of everything hands over the other end only where the input is a number.
PLAIN_SELF = np.array(rng.standard_normal(N) * 2.0, dtype=np.float64)
PLAIN_END = np.array(rng.standard_normal(N) * 2.0, dtype=np.float64)


def state(plain):
    """A map applied to the flat operands, with the divide that the reference
    itself answers at one element left unstated."""

    def apply(*terms):
        with np.errstate(divide="ignore", invalid="ignore"):
            return plain(*terms)

    return apply


def read(t):
    """The logical elements: a width narrower than single precision, or a stride,
    is only faithfully enumerable as Python numbers."""
    return np.array(t.tolist(), dtype=np.float64)


def agree(got, want, rtol, *terms):
    """Equal within a bound the operands carry, a NaN matching a NaN."""
    scale = max([1.0] + [float(np.nanmax(np.abs(term))) for term in terms])
    return np.allclose(got, want, rtol=rtol, atol=rtol * scale, equal_nan=True)


def same(got, want):
    """The same elements, one infinity or one NaN standing where the other does."""
    return got.shape == want.shape and np.array_equal(got, want, equal_nan=True)


def at(dtype, flat):
    return tp.from_numpy(np.array(flat, dtype=NP_OF[dtype])).to(dtype)


def matrix(flat, dtype):
    return at(dtype, flat).reshape(ROWS, COLS)


def packed(dtype, fill=None):
    if fill is None:
        return tp.zeros([ROWS, COLS], dtype=dtype)
    return tp.full([ROWS, COLS], fill, dtype=dtype)


def operands(dtype):
    """The four operands as one width sees them, with the float64 values they
    stand for read out before any of them is written over."""
    tensors = [matrix(flat, dtype) for flat in (SELF, END, WEIGHT, DIVISOR)]
    return tensors, [read(t) for t in tensors]


ENTRIES = [
    ("lerp with a scalar weight",
     lambda s, e, w, d, out: tp._C.lerp(s, e, LERP, out=out)
     if out is not None else tp._C.lerp(s, e, LERP),
     state(lambda s, e, w, d: s + LERP * (e - s)),
     WIDTHS),
    ("lerp with a tensor weight",
     lambda s, e, w, d, out: tp._C.lerp(s, e, w, out=out)
     if out is not None else tp._C.lerp(s, e, w),
     state(lambda s, e, w, d: s + w * (e - s)),
     WIDTHS),
    ("addcmul",
     lambda s, e, w, d, out: tp._C.addcmul(s, e, w, value=VALUE, out=out)
     if out is not None else tp._C.addcmul(s, e, w, value=VALUE),
     state(lambda s, e, w, d: s + VALUE * e * w),
     WIDTHS),
    ("addcdiv",
     lambda s, e, w, d, out: tp._C.addcdiv(s, e, d, value=VALUE, out=out)
     if out is not None else tp._C.addcdiv(s, e, d, value=VALUE),
     state(lambda s, e, w, d: s + VALUE * e / d),
     WIDTHS),
]


def test_out_fills_the_destination_it_is_given():
    for name, run, formula, widths in ENTRIES:
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
            want = formula(fs, fe, fw, fd)
            answer = packed(dtype, fill=-999.0)
            got = run(s, e, w, d, answer)
            # The handle the caller passed comes back, and its bytes hold the map.
            assert got is answer, (name, dtype)
            assert agree(read(answer), want, rtol, fs, fe, fw, fd), (name, dtype)
            # The same elements the route that allocates its own answer gives.
            assert same(read(answer), read(run(s, e, w, d, None))), (name, dtype)


def test_out_writes_an_operand_that_is_also_the_destination():
    """An element is read before the store lands on it, so a tensor the map also
    reads serves as its own answer: the input, and the operand read last."""
    for name, run, formula, widths in ENTRIES:
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
            want = formula(fs, fe, fw, fd)
            as_input = matrix(SELF, dtype)
            run(as_input, e, w, d, as_input)
            assert agree(read(as_input), want, rtol, fs, fe, fw, fd), (name, dtype)

            slot = {"lerp with a tensor weight": "weight", "addcmul": "weight",
                    "addcdiv": "divisor"}.get(name)
            if slot is not None:
                flat = WEIGHT if slot == "weight" else DIVISOR
                as_operand = matrix(flat, dtype)
                if slot == "weight":
                    run(s, e, as_operand, d, as_operand)
                else:
                    run(s, e, w, as_operand, as_operand)
                assert agree(read(as_operand), want, rtol, fs, fe, fw, fd), \
                    (name, dtype)


def test_out_answers_a_second_window_of_the_bytes_it_writes():
    """The input and the destination as two stretches of one block: a walk that
    lays the answer over the destination cannot read that input afterwards, so the
    answer is computed apart and handed over -- and the input's stretch is left
    alone."""
    for name, run, formula, widths in ENTRIES:
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (_, e, w, d), (_, fe, fw, fd) = operands(dtype)
            block = tp.zeros([2 * ROWS, COLS], dtype=dtype)
            dest = block[:ROWS]
            source = block[ROWS:]
            source.copy_(matrix(SELF, dtype))
            fs = read(source)
            run(source, e, w, d, dest)
            assert agree(read(dest), formula(fs, fe, fw, fd), rtol,
                         fs, fe, fw, fd), (name, dtype)
            # The operand's own stretch still holds what it was filled with.
            assert same(read(source), fs), (name, dtype)


def test_out_leaves_a_strided_destination_the_answer_it_computes():
    """A destination reached at a stride is not walked: it is handed the answer
    the packed route would have drawn, element for element, and nothing lands
    between those elements."""
    for name, run, formula, widths in ENTRIES:
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
            wide = tp.zeros([2 * ROWS, 2 * COLS], dtype=dtype)
            stepped = wide[::2, ::2]
            assert tuple(stepped.shape) == (ROWS, COLS), (name, dtype)
            run(s, e, w, d, stepped)
            assert agree(read(stepped), formula(fs, fe, fw, fd), rtol,
                         fs, fe, fw, fd), (name, dtype)
            assert same(read(wide[1::2, 1::2]), np.zeros([ROWS, COLS])), \
                (name, dtype)


def test_out_broadcasts_into_the_destination():
    """One operand stretched over a row and one over a column: the composition
    answers at the destination's own shape, so its last step still lands on it."""
    for name, run, formula, widths in ENTRIES:
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (_, _, w, d), (_, _, fw, fd) = operands(dtype)
            col = at(dtype, SELF[:ROWS]).reshape(ROWS, 1)
            row = at(dtype, END[:COLS]).reshape(1, COLS)
            bc_col = np.broadcast_to(read(col), (ROWS, COLS))
            bc_row = np.broadcast_to(read(row), (ROWS, COLS))
            answer = packed(dtype)
            run(col, row, w, d, answer)
            assert tuple(answer.shape) == (ROWS, COLS), (name, dtype)
            assert agree(read(answer), formula(bc_col, bc_row, fw, fd), rtol,
                         bc_col, bc_row, fw, fd), (name, dtype)


def test_out_grows_a_destination_of_the_wrong_shape():
    for name, run, formula, widths in ENTRIES:
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
            small = tp.zeros([3, 2], dtype=dtype)
            run(s, e, w, d, small)
            assert tuple(small.shape) == (ROWS, COLS), (name, dtype)
            assert agree(read(small), formula(fs, fe, fw, fd), rtol,
                         fs, fe, fw, fd), (name, dtype)


def test_out_answers_a_destination_of_another_width():
    """A destination wider than the operands takes the answer; the map is worked
    out at the operands' own width first, the way the allocating route works it
    out before handing it over."""
    for name, run, formula, widths in ENTRIES:
        for dtype in (tp.float32, tp.float16, tp.bfloat16):
            rtol = RTOL_OF[dtype]
            (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
            wide = tp.zeros([ROWS, COLS], dtype=tp.float64)
            run(s, e, w, d, wide)
            assert wide.dtype == tp.float64, (name, dtype)
            assert agree(read(wide), formula(fs, fe, fw, fd), rtol,
                         fs, fe, fw, fd), (name, dtype)


def test_out_keeps_the_rules_the_allocating_form_keeps():
    """The divide-by-zero rule, a NaN carried through, and whole-number operands
    refused by the quotient form."""
    _, run, formula, _ = [entry for entry in ENTRIES
                          if entry[0] == "addcdiv"][0]
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
        answer = packed(dtype)
        run(s, e, w, d, answer)
        got = read(answer).reshape(-1)
        # A positive dividend over a divisor of zero is an infinity of that sign;
        # the NaN input carries through; the rest stay finite.
        assert np.isinf(got[5]) and got[5] > 0, dtype
        assert np.isnan(got[0]), dtype
        assert np.all(np.isfinite(got[[1, 2, 3, 4]])), dtype
        assert agree(got, formula(fs, fe, fw, fd).reshape(-1), rtol,
                     fs, fe, fw, fd), dtype

    whole = matrix([1.0 * i for i in range(1, N + 1)], tp.int64)
    # Only the spellings that reach the map answer a message here: a call carrying
    # the destination keyword loses whatever the map threw, at any entry.
    for label, call in (("allocating", lambda: tp._C.addcdiv(whole, whole, whole,
                                                             value=2)),
                        ("destination", lambda: tp._C.addcdiv(
                            whole, whole, whole, value=2,
                            out=tp.zeros([ROWS, COLS], dtype=tp.int64)))):
        try:
            call()
        except RuntimeError as err:
            assert "Integer division" in str(err), (label, str(err))
        else:
            raise AssertionError("addcdiv took whole-number operands")


def test_out_answers_lerp_at_its_endpoints():
    """A weight of nothing leaves the input and a weight of everything gives the
    other end -- at both spellings, in the destination."""
    for dtype in WIDTHS:
        rtol = RTOL_OF[dtype]
        s = matrix(PLAIN_SELF, dtype)
        e = matrix(PLAIN_END, dtype)
        fs, fe = read(s), read(e)
        for weight, want in ((0.0, fs), (1.0, fe)):
            answer = packed(dtype, fill=-999.0)
            tp._C.lerp(s, e, weight, out=answer)
            assert agree(read(answer), want, rtol, fs, fe), dtype
        for filled, want in ((tp.zeros([ROWS, COLS], dtype=dtype), fs),
                             (tp.ones([ROWS, COLS], dtype=dtype), fe)):
            answer = packed(dtype, fill=-999.0)
            tp._C.lerp(s, e, filled, out=answer)
            assert agree(read(answer), want, rtol, fs, fe), dtype


def test_inplace_forms_answer_the_input_themselves():
    spellings = {"lerp with a scalar weight": ("lerp_", LERP),
                 "lerp with a tensor weight": ("lerp_", None),
                 "addcmul": ("addcmul_", None),
                 "addcdiv": ("addcdiv_", None)}
    for name, run, formula, widths in ENTRIES:
        spelling, scalar_weight = spellings[name]
        for dtype in widths:
            rtol = RTOL_OF[dtype]
            (s, e, w, d), (fs, fe, fw, fd) = operands(dtype)
            want = formula(fs, fe, fw, fd)
            target = matrix(SELF, dtype)
            step = getattr(target, spelling)
            if spelling == "lerp_":
                step(e, scalar_weight if scalar_weight is not None else w)
            else:
                step(e, w if name == "addcmul" else d, value=VALUE)
            assert agree(read(target), want, rtol, fs, fe, fw, fd), (name, dtype)


def test_out_answers_a_destination_too_wide_to_stay_in_cache():
    """A destination too wide for the share of the last level cache the team
    holds is stored past the cache, which spares the walk fetching every line it
    is about to overwrite.  Such a store only lays whole blocks, so the elements
    ahead of the first block of every span and behind its last are the ones an
    edge can drop, and the answer has to hold them all.

    Four streams of this length are past that share on a part with a few tens of
    megabytes of last level cache, which is the width the streamed route is
    chosen at; a narrower answer takes the fetched route and has to agree with
    the same elements either way."""
    n = 2097159  # four float32 streams of eight megabytes, and odd
    rng = np.random.default_rng(11)
    base = rng.standard_normal(n).astype(np.float32)
    one = rng.standard_normal(n).astype(np.float32)
    two = np.abs(rng.standard_normal(n).astype(np.float32)) + 0.5
    weight = np.abs(rng.standard_normal(n).astype(np.float32))
    s, e, w = at(tp.float32, base), at(tp.float32, one), at(tp.float32, two)
    lw = at(tp.float32, weight)

    packed_answer = tp.zeros([n], dtype=tp.float32)
    tp._C.addcmul(s, e, w, value=VALUE, out=packed_answer)
    assert packed_answer.equal(tp._C.addcmul(s, e, w, value=VALUE))

    # The same map onto a destination that does not begin on a whole block.
    offset = tp.zeros([n + 1], dtype=tp.float32)[1:]
    tp._C.addcmul(s, e, w, value=VALUE, out=offset)
    assert offset.equal(packed_answer)

    # An operand the walk reads and writes at this width keeps the fetch: the
    # lines it holds are wanted again by the walk itself.
    own = at(tp.float32, base)
    tp._C.addcmul(own, e, w, value=VALUE, out=own)
    assert own.equal(packed_answer)

    # The tensor-weight map answers two blocks per step, so its edges are the
    # elements between the blocks of the span the thread was handed.
    interpolated = tp.zeros([n], dtype=tp.float32)
    tp._C.lerp(s, e, lw, out=interpolated)
    assert interpolated.equal(tp._C.lerp(s, e, lw))

    shifted = tp.zeros([n + 1], dtype=tp.float32)[1:]
    tp._C.lerp(s, e, lw, out=shifted)
    assert shifted.equal(interpolated)

    held = at(tp.float32, base)
    tp._C.lerp(held, e, lw, out=held)
    assert held.equal(interpolated)

    # The divided map answers four blocks per step, and its edges are held to
    # the same rule.
    divided = tp.zeros([n], dtype=tp.float32)
    tp._C.addcdiv(s, e, w, value=VALUE, out=divided)
    assert divided.equal(tp._C.addcdiv(s, e, w, value=VALUE))

    shifted_division = tp.zeros([n + 1], dtype=tp.float32)[1:]
    tp._C.addcdiv(s, e, w, value=VALUE, out=shifted_division)
    assert shifted_division.equal(divided)
