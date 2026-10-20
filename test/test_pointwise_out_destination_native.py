import numpy as np
import tensorplay as tp

NP_OF = {tp.int64: np.int64, tp.int32: np.int32,
         tp.float32: np.float32, tp.float64: np.float64,
         tp.float16: np.float16}

# The binary maps whose out= pass writes the caller's tensor instead of
# filling a fresh one and copying it over.  Each entry is the name, the form
# that allocates its answer, the form that is handed one, and the widths both
# are asked about.
MAPS = [
    ("gcd", tp._C.gcd, tp._C.gcd, (tp.int64, tp.int32)),
    ("lcm", tp._C.lcm, tp._C.lcm, (tp.int64, tp.int32)),
    ("heaviside", tp._C.heaviside, tp._C.heaviside,
     (tp.float32, tp.float64, tp.float16, tp.int64)),
    ("floor_divide", tp.floor_divide, tp._C.floor_divide,
     (tp.float32, tp.float64, tp.float16, tp.int64)),
]

LENGTH = 37


def values(name, dtype):
    """One operand pair: whole numbers for gcd and lcm, a divisor that is never
    zero and a dividend of both signs for floor_divide, and a length that leaves
    a tail behind every vector width."""
    left_rng = np.random.default_rng(5)
    right_rng = np.random.default_rng(11)
    n = LENGTH
    if name in ("gcd", "lcm"):
        return (left_rng.integers(0, 97, n), right_rng.integers(0, 97, n))
    if name == "heaviside":
        integral = dtype == tp.int64
        return (left_rng.integers(-2, 3, n) if integral else left_rng.standard_normal(n),
                right_rng.integers(-2, 3, n) if integral else right_rng.standard_normal(n))
    right = np.where(right_rng.integers(0, 4, n) == 0, 1, right_rng.integers(-7, 8, n))
    return left_rng.integers(-40, 41, n), np.abs(right) + 1


def operands(name, dtype):
    left_np, right_np = values(name, dtype)
    return (tp.from_numpy(np.array(left_np, dtype=NP_OF[dtype])),
            tp.from_numpy(np.array(right_np, dtype=NP_OF[dtype])))


def test_out_fills_the_destination_it_is_given():
    for name, alloc_fn, out_fn, dtypes in MAPS:
        for dtype in dtypes:
            left, right = operands(name, dtype)
            want = alloc_fn(left, right).tolist()
            out = tp.zeros(LENGTH, dtype=dtype)
            result = out_fn(left, right, out=out)
            assert result is out, name
            assert out.tolist() == want, (name, dtype)


def test_out_writes_an_operand_destination_in_place():
    # A destination that is one of the operands reads an element and stores it
    # back at the address it came from, so the pass still answers in one go.
    for name, alloc_fn, out_fn, dtypes in MAPS:
        for dtype in dtypes:
            left, right = operands(name, dtype)
            want = alloc_fn(left, right).tolist()

            as_left, right_copy = operands(name, dtype)
            out_fn(as_left, right_copy, out=as_left)
            assert as_left.tolist() == want, (name, dtype, "left")

            left_copy, as_right = operands(name, dtype)
            out_fn(left_copy, as_right, out=as_right)
            assert as_right.tolist() == want, (name, dtype, "right")


def test_out_answers_through_a_destination_it_has_to_copy_into():
    # A strided destination is not the buffer the pass would have written, so
    # the value arrives through the hand-over; the answer must not depend on
    # which of the two routes the destination allowed.
    for name, alloc_fn, out_fn, dtypes in MAPS:
        for dtype in dtypes:
            left, right = operands(name, dtype)
            want = alloc_fn(left, right).tolist()
            wide = tp.zeros(2 * LENGTH + 1, dtype=dtype)
            out = wide[::2]
            out_fn(left, right, out=out)
            assert out.tolist() == want, (name, dtype)


def test_out_broadcasts_into_the_destination():
    for name, alloc_fn, out_fn, dtypes in MAPS:
        dtype = dtypes[0]
        left = tp.from_numpy(np.arange(4, dtype=NP_OF[dtype]).reshape(4, 1) + 1)
        right = tp.from_numpy(np.arange(3, dtype=NP_OF[dtype]) + 5)
        want = alloc_fn(left, right).tolist()
        out = tp.zeros([4, 3], dtype=dtype)
        assert out_fn(left, right, out=out).tolist() == want, name


def test_out_keeps_the_rules_the_allocating_form_keeps():
    # gcd and lcm are defined on whole numbers, and an integral division by
    # zero is refused outright; both refusals have to survive the second
    # spelling.
    for name in ("gcd", "lcm"):
        fn = getattr(tp._C, name)
        left = tp.tensor([1.5, -2.5], dtype=tp.float32)
        right = tp.tensor([3.0, 4.0], dtype=tp.float32)
        out = tp.zeros(2, dtype=tp.float32)
        for call in (lambda: fn(left, right), lambda: fn(left, right, out=out)):
            try:
                call()
            except Exception:
                continue
            raise AssertionError(f"{name} answered a floating pair")

    some = tp.tensor([7, -7, 0], dtype=tp.int64)
    zero = tp.tensor([0, 0, 0], dtype=tp.int64)
    out = tp.zeros(3, dtype=tp.int64)
    for call in (lambda: tp.floor_divide(some, zero),
                 lambda: tp._C.floor_divide(some, zero, out=out)):
        try:
            call()
        except Exception:
            continue
        raise AssertionError("floor_divide answered an integral division by zero")

    # A float divisor of zero is an answer, not a refusal.
    f_some = tp.tensor([1.0, -1.0], dtype=tp.float32)
    f_zero = tp.tensor([0.0, 0.0], dtype=tp.float32)
    f_out = tp.zeros(2, dtype=tp.float32)
    got = tp._C.floor_divide(f_some, f_zero, out=f_out).tolist()
    assert got == tp.floor_divide(f_some, f_zero).tolist()
