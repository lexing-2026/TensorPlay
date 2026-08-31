"""Native arithmetic aliases and the rounding-mode division overloads.

References are the Python spellings of the same arithmetic: ``//`` and ``%``
carry the floor convention, ``math.fmod`` the truncating one, and integral
inputs are compared exactly rather than through a float quotient, since the
whole point of the rounded overloads is that they never leave the integers.
"""
import math

import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

# Sign-crossing pairs: every combination of dividend and divisor sign, plus a
# ratio that lands exactly on an integer (where floor and trunc part ways only
# for the negative case) and one that does not.
PAIRS = [(7, 3), (-7, 3), (7, -3), (-7, -3), (6, 3), (-6, 3), (6, -3), (-6, -3)]


def _t(values, device, dtype=tp.float32):
    return tp.tensor(list(values), dtype=dtype).to(device)


def _list(t):
    return t.cpu().tolist()


# ---------------------------------------------------------------------------
# aliases resolve to the same arithmetic as the short spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alias,short", [
    ("divide", "div"), ("multiply", "mul"), ("true_divide", "div"),
])
def test_alias_matches_short_name(alias, short, device):
    a = _t([1.5, -2.0, 7.0], device)
    b = _t([2.0, 4.0, -0.5], device)
    assert _list(getattr(tp, alias)(a, b)) == _list(getattr(tp, short)(a, b))


@pytest.mark.parametrize("device", DEVICES)
def test_subtract_matches_sub_with_alpha(device):
    a = _t([1.5, -2.0, 7.0], device)
    b = _t([2.0, 4.0, -0.5], device)
    assert _list(tp.subtract(a, b)) == _list(tp.sub(a, b))
    assert _list(tp.subtract(a, b, alpha=3)) == _list(tp.sub(a, b, alpha=3))
    assert _list(tp.subtract(a, 2.0, alpha=0.5)) == _list(tp.sub(a, 2.0, alpha=0.5))


@pytest.mark.parametrize("device", DEVICES)
def test_rsub_reverses_the_operands(device):
    a = _t([1.5, -2.0, 7.0], device)
    b = _t([2.0, 4.0, -0.5], device)
    assert _list(tp.rsub(a, b)) == _list(tp.sub(b, a))
    # alpha scales the subtrahend, so it multiplies self and not other.
    assert _list(tp.rsub(a, b, alpha=2)) == pytest.approx(
        [bv - 2 * av for av, bv in zip(_list(a), _list(b))])
    assert _list(tp.rsub(a, 10.0)) == pytest.approx([10.0 - av for av in _list(a)])


@pytest.mark.parametrize("device", DEVICES)
def test_aliases_are_native_not_python_shims(device):
    # A trimmed alias would show up as a Python-level composite; these route
    # into the extension like every other operator.
    for name in ["divide", "multiply", "subtract", "true_divide",
                 "floor_divide", "remainder", "fmod", "copysign"]:
        assert getattr(tp, name).__module__ == "tensorplay.functional"
        assert hasattr(tp.Tensor, name)


# ---------------------------------------------------------------------------
# rounding modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("x,y", PAIRS)
def test_div_rounding_matches_python_semantics(x, y, device):
    a, b = _t([x], device), _t([y], device)
    assert _list(tp.div(a, b, rounding_mode="floor"))[0] == float(x // y)
    assert _list(tp.div(a, b, rounding_mode="trunc"))[0] == math.trunc(x / y)
    assert _list(tp.divide(a, b, rounding_mode="floor"))[0] == float(x // y)
    assert _list(tp.divide(a, b, rounding_mode="trunc"))[0] == math.trunc(x / y)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("x,y", PAIRS)
def test_rounded_division_of_integers_stays_integral(x, y, device):
    a = _t([x], device, tp.int64)
    b = _t([y], device, tp.int64)
    floor_q = tp.div(a, b, rounding_mode="floor")
    trunc_q = tp.div(a, b, rounding_mode="trunc")
    assert floor_q.dtype == tp.int64
    assert trunc_q.dtype == tp.int64
    assert _list(floor_q)[0] == x // y
    assert _list(trunc_q)[0] == math.trunc(x / y)
    assert _list(tp.floor_divide(a, b))[0] == x // y
    assert tp.floor_divide(a, b).dtype == tp.int64


@pytest.mark.parametrize("device", DEVICES)
def test_rounded_division_keeps_int64_exact(device):
    # A float64 quotient loses the low bits above 2**53; the integral path
    # must not go through one.
    big = 9007199254740993
    a = _t([big], device, tp.int64)
    b = _t([7], device, tp.int64)
    assert _list(tp.floor_divide(a, b))[0] == big // 7
    assert _list(tp.remainder(a, b))[0] == big % 7


@pytest.mark.parametrize("device", DEVICES)
def test_div_without_mode_is_true_division(device):
    a = _t([7], device, tp.int64)
    b = _t([2], device, tp.int64)
    out = tp.div(a, b)
    assert out.dtype == tp.float32
    assert _list(out)[0] == 3.5
    assert _list(tp.div(a, b, rounding_mode=None))[0] == 3.5


@pytest.mark.parametrize("device", DEVICES)
def test_rounded_division_of_scalars(device):
    a = _t([7, -7], device, tp.int64)
    assert _list(tp.div(a, 2, rounding_mode="floor")) == [3, -4]
    assert _list(tp.div(a, 2, rounding_mode="trunc")) == [3, -3]
    assert tp.div(a, 2, rounding_mode="floor").dtype == tp.int64
    # A floating divisor promotes the pair, so the quotient comes back float.
    assert tp.div(a, 2.5, rounding_mode="floor").dtype == tp.float32
    assert _list(tp.div(a, 2.5, rounding_mode="floor")) == [2.0, -3.0]


@pytest.mark.parametrize("device", DEVICES)
def test_rounded_division_of_nonfinite_divisors(device):
    a = _t([1.0, -1.0, 0.0], device)
    b = _t([0.0, 0.0, 0.0], device)
    for mode in ("floor", "trunc"):
        got = _list(tp.div(a, b, rounding_mode=mode))
        assert got[0] == math.inf
        assert got[1] == -math.inf
        assert math.isnan(got[2])


@pytest.mark.parametrize("device", DEVICES)
def test_unknown_rounding_mode_is_rejected(device):
    a = _t([1.0], device)
    b = _t([2.0], device)
    with pytest.raises(RuntimeError, match="rounding_mode"):
        tp.div(a, b, rounding_mode="ceil")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float64])
def test_rounded_division_preserves_width(dtype, device):
    a = _t([7.0, -7.0], device, dtype)
    b = _t([2.0, 2.0], device, dtype)
    out = tp.div(a, b, rounding_mode="floor")
    assert out.dtype == dtype
    assert out.to(tp.float64).cpu().tolist() == [3.0, -4.0]


# ---------------------------------------------------------------------------
# floor_divide / remainder / fmod
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("x,y", PAIRS)
def test_remainder_follows_the_divisor_sign(x, y, device):
    a, b = _t([x], device), _t([y], device)
    assert _list(tp.remainder(a, b))[0] == float(x % y)
    assert _list(tp.fmod(a, b))[0] == math.fmod(x, y)


@pytest.mark.parametrize("device", DEVICES)
def test_remainder_and_fmod_promote_a_floating_divisor(device):
    a = _t([5, 7], device, tp.int64)
    for fn in (tp.remainder, tp.fmod):
        out = fn(a, 2.5)
        assert out.dtype == tp.float32
        assert _list(out) == [0.0, 2.0]
    assert tp.remainder(a, 2).dtype == tp.int64


@pytest.mark.parametrize("device", DEVICES)
def test_remainder_accepts_a_scalar_dividend(device):
    b = _t([3, -3], device, tp.int64)
    assert _list(tp.remainder(5, b)) == [5 % 3, 5 % -3]
    assert _list(tp.tensor([13, 13], dtype=tp.int64).to(device).remainder(b)) == \
        [13 % 3, 13 % -3]


@pytest.mark.parametrize("device", DEVICES)
def test_operator_forms_route_to_the_native_kernels(device):
    a = _t([7, -7], device, tp.int64)
    b = _t([3, 3], device, tp.int64)
    assert _list(a // b) == [7 // 3, -7 // 3]
    assert _list(a % b) == [7 % 3, -7 % 3]
    assert _list(13 // a) == [13 // 7, 13 // -7]
    assert _list(13 % a) == [13 % 7, 13 % -7]
    c = _t([7.0, 9.0], device)
    c //= _t([2.0], device)
    assert _list(c) == [3.0, 4.0]
    d = _t([7.0, 9.0], device)
    d %= _t([2.0], device)
    assert _list(d) == [1.0, 1.0]


# ---------------------------------------------------------------------------
# copysign
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_copysign_reads_the_sign_bit_not_the_comparison(device):
    # -0.0 is not less than zero, so a comparison-based implementation gets
    # this wrong; the sign bit is what carries.
    a = _t([1.0, 1.0, 2.0, 2.0], device)
    b = _t([-0.0, 0.0, -3.0, 3.0], device)
    assert _list(tp.copysign(a, b)) == [-1.0, 1.0, -2.0, 2.0]


@pytest.mark.parametrize("device", DEVICES)
def test_copysign_scalar_and_integral_promotion(device):
    a = _t([1.0, 2.0], device)
    assert _list(tp.copysign(a, -3.0)) == [-1.0, -2.0]
    ints = _t([1, 2], device, tp.int64)
    out = tp.copysign(ints, _t([-1, 1], device, tp.int64))
    assert out.dtype == tp.float32
    assert _list(out) == [-1.0, 2.0]


# ---------------------------------------------------------------------------
# clamp bounds given as tensors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_clamp_min_max_accept_tensor_bounds(device):
    a = _t([7.0, -7.0, 2.0], device)
    lo = _t([3.0, 3.0, -4.0], device)
    assert _list(tp.clamp_min(a, lo)) == [7.0, 3.0, 2.0]
    assert _list(tp.clamp_max(a, lo)) == [3.0, -7.0, -4.0]
    assert _list(a.clamp_min(lo)) == [7.0, 3.0, 2.0]


# ---------------------------------------------------------------------------
# promotion, broadcasting and complex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_division_keeps_complex_operands_complex(device):
    a = tp.tensor([1 + 2j, 3 - 1j]).to(device)
    b = tp.tensor([2 + 0j, 1 + 1j]).to(device)
    for fn in (tp.divide, tp.true_divide, tp.div):
        out = fn(a, b)
        assert out.dtype == a.dtype
        assert out.cpu().tolist() == pytest.approx([0.5 + 1j, 1 - 2j])


@pytest.mark.parametrize("device", DEVICES)
def test_reduced_width_scalar_division_does_not_widen(device):
    for dtype in (tp.float16, tp.bfloat16):
        a = _t([1.0, 2.0], device, dtype)
        assert tp.divide(a, 2).dtype == dtype
        assert tp.true_divide(a, 2).dtype == dtype


@pytest.mark.parametrize("device", DEVICES)
def test_aliases_broadcast_and_accept_empty_inputs(device):
    a = tp.tensor([[1.0], [2.0]]).to(device)
    b = tp.tensor([10.0, 20.0, 30.0]).to(device)
    assert tp.multiply(a, b).shape == (2, 3)
    assert tp.subtract(a, b).shape == (2, 3)
    empty = tp.tensor([], dtype=tp.float32).to(device)
    assert tp.divide(empty, empty).numel() == 0
    assert tp.remainder(empty, empty).numel() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_a_plain_number_may_be_the_left_operand(device):
    t = _t([0.0, 1.0], device)
    assert _list(tp.multiply(3.0, t)) == [0.0, 3.0]
    assert _list(tp.subtract(10.0, t)) == [10.0, 9.0]
    assert _list(tp.divide(6.0, _t([2.0, 3.0], device))) == [3.0, 2.0]
    assert _list(tp.mul(3.0, t)) == [0.0, 3.0]


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _numeric_grad(fn, values, index, eps=1e-4):
    hi = list(values); hi[index] += eps
    lo = list(values); lo[index] -= eps
    up = fn(tp.tensor(hi, dtype=tp.float64)).sum().item()
    dn = fn(tp.tensor(lo, dtype=tp.float64)).sum().item()
    return (up - dn) / (2.0 * eps)


@pytest.mark.parametrize("name,fn,values", [
    ("divide", lambda t: tp.divide(t, tp.tensor([2.0, 4.0, -0.5], dtype=tp.float64)),
     [1.5, -2.0, 7.0]),
    ("multiply", lambda t: tp.multiply(t, tp.tensor([2.0, 4.0, -0.5], dtype=tp.float64)),
     [1.5, -2.0, 7.0]),
    ("subtract", lambda t: tp.subtract(t, tp.tensor([2.0, 4.0, -0.5], dtype=tp.float64), alpha=2),
     [1.5, -2.0, 7.0]),
    ("true_divide", lambda t: tp.true_divide(t, tp.tensor([2.0, 4.0, -0.5], dtype=tp.float64)),
     [1.5, -2.0, 7.0]),
    ("remainder", lambda t: tp.remainder(t, tp.tensor([3.0, 3.0, 4.0], dtype=tp.float64)),
     [1.5, -2.0, 7.0]),
    ("fmod", lambda t: tp.fmod(t, tp.tensor([3.0, 3.0, 4.0], dtype=tp.float64)),
     [1.5, -2.0, 7.0]),
    ("divide_scalar", lambda t: tp.divide(t, 4.0), [1.5, -2.0, 7.0]),
    ("rsub", lambda t: tp.rsub(t, tp.tensor([2.0, 4.0, -0.5], dtype=tp.float64), alpha=2),
     [1.5, -2.0, 7.0]),
])
def test_gradient_matches_finite_differences(name, fn, values):
    x = tp.tensor(values, dtype=tp.float64, requires_grad=True)
    fn(x).sum().backward()
    got = x.grad.tolist()
    for i in range(len(values)):
        assert got[i] == pytest.approx(_numeric_grad(fn, values, i), rel=1e-5, abs=1e-6)


def test_divisor_gradient_counts_whole_multiples():
    # remainder steps by one whole divisor per unit change in the divisor, and
    # the count is the floored quotient (the truncated one for fmod).
    for fn, quotient in ((tp.remainder, math.floor), (tp.fmod, math.trunc)):
        b = tp.tensor([3.0, 3.0, -4.0], dtype=tp.float64, requires_grad=True)
        a = tp.tensor([7.0, -7.0, 9.0], dtype=tp.float64)
        fn(a, b).sum().backward()
        assert b.grad.tolist() == pytest.approx(
            [-quotient(av / bv) for av, bv in zip(a.tolist(), [3.0, 3.0, -4.0])])


def test_rounded_division_has_no_gradient():
    for mode in ("floor", "trunc"):
        a = tp.tensor([7.0, -7.0], dtype=tp.float64, requires_grad=True)
        b = tp.tensor([3.0, 3.0], dtype=tp.float64, requires_grad=True)
        tp.div(a, b, rounding_mode=mode).sum().backward()
        assert a.grad.tolist() == [0.0, 0.0]
        assert b.grad.tolist() == [0.0, 0.0]


def test_unrounded_mode_overload_still_carries_the_quotient_rule():
    a = tp.tensor([7.0, -7.0], dtype=tp.float64, requires_grad=True)
    b = tp.tensor([3.0, 4.0], dtype=tp.float64, requires_grad=True)
    a.div(b, rounding_mode=None).sum().backward()
    assert a.grad.tolist() == pytest.approx([1 / 3, 1 / 4])
    assert b.grad.tolist() == pytest.approx([-7 / 9, 7 / 16])


def test_copysign_gradient_takes_the_transplanted_sign():
    a = tp.tensor([1.0, -2.0, 0.0], dtype=tp.float64, requires_grad=True)
    b = tp.tensor([-1.0, 1.0, -1.0], dtype=tp.float64, requires_grad=True)
    tp.copysign(a, b).sum().backward()
    # A zero input has no sign to carry, so nothing flows back through it.
    assert a.grad.tolist() == [-1.0, -1.0, 0.0]
    assert b.grad.tolist() == [0.0, 0.0, 0.0]


def test_clamp_tensor_bound_splits_the_gradient_at_a_tie():
    a = tp.tensor([7.0, -7.0, 3.0], dtype=tp.float64, requires_grad=True)
    lo = tp.tensor([3.0, 3.0, 3.0], dtype=tp.float64, requires_grad=True)
    tp.clamp_min(a, lo).sum().backward()
    assert a.grad.tolist() == [1.0, 0.0, 0.5]
    assert lo.grad.tolist() == [0.0, 1.0, 0.5]

    c = tp.tensor([7.0, -7.0, 3.0], dtype=tp.float64, requires_grad=True)
    hi = tp.tensor([3.0, 3.0, 3.0], dtype=tp.float64, requires_grad=True)
    tp.clamp_max(c, hi).sum().backward()
    assert c.grad.tolist() == [0.0, 1.0, 0.5]
    assert hi.grad.tolist() == [1.0, 0.0, 0.5]
