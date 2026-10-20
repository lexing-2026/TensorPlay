#!/usr/bin/env python3
"""Holding the special-function maps to one answer per destination.

These maps evaluate in Float32 or Float64 and narrow afterwards, so a caller's
tensor only takes the single pass when it already carries the answer at that
width; everywhere else the value arrives through the hand-over.  Either way the
answer must be the same one the allocating form gives, and a reduced-width step
has to land on the adjacent number the destination width can hold.
"""
import numpy as np
import tensorplay as tp

LENGTH = 37  # leaves a tail behind every vector width

# The maps whose out= pass writes the caller's tensor instead of filling a fresh
# one and copying it over.  Each entry names the map, the form that allocates its
# answer, the form that is handed one, and the widths both are asked about.
MAPS = [
    ("igamma", tp._C.igamma, tp._C.igamma,
     (tp.float32, tp.float64, tp.float16, tp.bfloat16)),
    ("igammac", tp._C.igammac, tp._C.igammac,
     (tp.float32, tp.float64, tp.float16, tp.bfloat16)),
    ("polygamma of order 0", lambda x: tp._C.polygamma(0, x),
     lambda x, out=None: tp._C.polygamma(0, x, out=out),
     (tp.float32, tp.float64, tp.float16, tp.bfloat16)),
    ("polygamma of order 3", lambda x: tp._C.polygamma(3, x),
     lambda x, out=None: tp._C.polygamma(3, x, out=out),
     (tp.float32, tp.float64, tp.float16)),
    ("nextafter", tp._C.nextafter, tp._C.nextafter,
     (tp.float32, tp.float64, tp.float16, tp.bfloat16)),
]


def values(name, dtype):
    """One set of operands, positive and away from the poles where the map has
    them, so every element is an ordinary answer rather than a boundary."""
    rng = np.random.default_rng(5)
    left = np.abs(rng.standard_normal(LENGTH)) + 0.25
    if name.startswith("polygamma"):
        return (left,)
    right = np.abs(rng.standard_normal(LENGTH)) + 0.1
    if name == "nextafter":
        right = left + 0.5
    return left, right


def tensor_of(array, dtype):
    # A reduced width is built by widening first: the values are exactly
    # representable at float32, so the narrowing step carries no rounding of
    # its own into the test.
    return tp.from_numpy(np.array(array, dtype=np.float32)).to(dtype)


def operands(name, dtype):
    return tuple(tensor_of(v, dtype) for v in values(name, dtype))


def narrow(t):
    """Reading a tensor back at float32, the widest width numpy holds exactly."""
    return t.to(tp.float32).tolist()


def test_out_fills_the_destination_it_is_given():
    for name, alloc_fn, out_fn, dtypes in MAPS:
        for dtype in dtypes:
            ops = operands(name, dtype)
            want = narrow(alloc_fn(*ops))
            out = tp.zeros(LENGTH, dtype=dtype)
            result = out_fn(*ops, out=out)
            assert result is out, name
            assert narrow(out) == want, (name, dtype)


def test_out_writes_an_operand_destination_in_place():
    # A destination that is one of the operands reads an element and stores it
    # back at the address it came from, so the pass still answers in one go.
    for name, alloc_fn, out_fn, dtypes in MAPS:
        for dtype in dtypes:
            want = narrow(alloc_fn(*operands(name, dtype)))
            if len(values(name, dtype)) == 1:
                as_operand, = operands(name, dtype)
                out_fn(as_operand, out=as_operand)
                assert narrow(as_operand) == want, (name, dtype, "operand")
                continue
            left, right = operands(name, dtype)
            out_fn(left, right, out=left)
            assert narrow(left) == want, (name, dtype, "left")
            left, right = operands(name, dtype)
            out_fn(left, right, out=right)
            assert narrow(right) == want, (name, dtype, "right")


def test_out_answers_through_a_destination_it_has_to_copy_into():
    # A strided destination is not the buffer the pass would have written, so
    # the value arrives through the hand-over; the answer must not depend on
    # which of the two routes the destination allowed.
    for name, alloc_fn, out_fn, dtypes in MAPS:
        for dtype in dtypes:
            ops = operands(name, dtype)
            want = narrow(alloc_fn(*ops))
            wide_buffer = tp.zeros(2 * LENGTH, dtype=dtype)
            out = wide_buffer[::2]
            out_fn(*ops, out=out)
            assert narrow(out) == want, (name, dtype)


def test_out_answers_at_the_width_a_narrower_destination_holds():
    # A destination carrying the answer at another width is filled with the
    # value narrowed or widened to it, exactly as the allocating form's answer
    # arrives there; the answer must not change with which route got there.
    for name, alloc_fn, out_fn, dtypes in MAPS:
        ops = operands(name, tp.float32)
        for dtype in dtypes:
            want = alloc_fn(*ops).to(dtype).to(tp.float32).tolist()
            out = tp.zeros(LENGTH, dtype=dtype)
            out_fn(*ops, out=out)
            assert out.to(tp.float32).tolist() == want, (name, dtype)


def test_out_broadcasts_into_the_destination():
    for name, alloc_fn, out_fn, dtypes in MAPS:
        if len(values(name, dtypes[0])) == 1:
            continue  # a one-operand map has nothing to broadcast against
        dtype = tp.float32
        left = tensor_of(np.arange(4, dtype=np.float32).reshape(4, 1) + 1.5, dtype)
        right = tensor_of(np.arange(3, dtype=np.float32) + 0.5, dtype)
        want = alloc_fn(left, right).tolist()
        out = tp.zeros([4, 3], dtype=dtype)
        assert out_fn(left, right, out=out).tolist() == want, name


def test_out_keeps_the_rules_the_allocating_form_keeps():
    # An order the recurrence cannot be run for is refused by both spellings.
    x = tp.tensor([1.5, 2.5], dtype=tp.float32)
    out = tp.zeros(2, dtype=tp.float32)
    for order in (-1, 1 << 31):
        for call in (lambda: tp._C.polygamma(order, x),
                     lambda: tp._C.polygamma(order, x, out=out)):
            try:
                call()
            except Exception:
                continue
            raise AssertionError(f"polygamma answered order {order}")

    # A pair whose shapes have no common shape is refused by both spellings; a
    # destination never decides the shape the answer is given in.
    left = tensor_of(np.abs(np.random.default_rng(3).standard_normal(LENGTH)) + 0.5,
                     tp.float32)
    right = tensor_of(np.abs(np.random.default_rng(4).standard_normal(8)) + 0.5,
                      tp.float32)
    short = tp.zeros(8, dtype=tp.float32)
    try:
        tp._C.igamma(left, right, out=short)
    except Exception:
        pass
    else:
        raise AssertionError("igamma answered a pair whose shapes do not broadcast")


# Every pattern of a 16-bit width, and the number the step toward each infinity
# has to land on.  The widths hold their values in a sign-and-magnitude layout,
# so stepping away from zero is one added to the magnitude and stepping toward it
# is one taken away -- the adjacency below is stated as "the next value the width
# can hold", read off the width's own ordered list.
def _grid(dtype):
    """All 65536 patterns of a 16-bit width, widened losslessly to float32."""
    bits = np.arange(1 << 16, dtype=np.uint32) << 16
    if dtype == tp.float16:
        return np.arange(1 << 16, dtype=np.uint16).view(np.float16).astype(np.float32)
    return bits.view(np.float32)


def _adjacent(grid, toward):
    """For every value in the grid, the next distinct value in the direction."""
    ordered = np.unique(grid[np.isfinite(grid)])
    finite = np.isfinite(grid)
    got = np.array(grid, copy=True)
    if toward > 0:
        index = np.searchsorted(ordered, grid, side="right")
    else:
        index = np.searchsorted(ordered, grid, side="left") - 1
    inside = finite & (index >= 0) & (index < ordered.size)
    got[inside] = ordered[index[inside]]
    got[grid == np.inf] = np.inf
    got[grid == -np.inf] = -np.inf
    return got, inside


def test_nextafter_steps_to_the_adjacent_number_the_width_holds():
    for dtype in (tp.float16, tp.bfloat16):
        grid = _grid(dtype)
        for toward in (np.inf, -np.inf):
            want, inside = _adjacent(grid, toward)
            source = tensor_of(grid, dtype)
            target = tensor_of(np.full(grid.shape, toward, dtype=np.float32), dtype)
            values_got = tp._C.nextafter(source, target).to(tp.float32).numpy()
            assert np.array_equal(values_got[inside], want[inside]), (dtype, toward)
            # A width's own infinity has no neighbour to step to; it answers itself.
            assert np.all(values_got[grid == toward] == toward), (dtype, toward)


def test_nextafter_answers_a_nan_with_a_nan():
    for dtype in (tp.float16, tp.bfloat16, tp.float32, tp.float64):
        source = tensor_of(np.array([np.nan, 1.0, np.nan]), dtype)
        target = tensor_of(np.array([1.0, np.nan, np.nan]), dtype)
        got = tp._C.nextafter(source, target).to(tp.float32).tolist()
        assert all(np.isnan(v) for v in got), dtype
        out = tp.zeros(3, dtype=dtype)
        tp._C.nextafter(source, target, out=out)
        assert all(np.isnan(v) for v in out.to(tp.float32).tolist()), dtype


def test_nextafter_steps_a_zero_away_from_zero():
    # The step out of either zero is the width's smallest subnormal, not the
    # smallest normal number, so a zero destination answers with a value whose
    # magnitude is one pattern wide.
    for dtype, tiny in ((tp.float16, 2.0 ** -24), (tp.bfloat16, 2.0 ** -133)):
        for sign in (1.0, -1.0):
            source = tensor_of(np.array([sign * 0.0]), dtype)
            target = tensor_of(np.array([np.inf if sign > 0 else -np.inf]), dtype)
            got = tp._C.nextafter(source, target).to(tp.float32).tolist()[0]
            assert got == sign * tiny, (dtype, sign, got)
