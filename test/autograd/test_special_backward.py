"""Gradients of the special functions.

Each special_<name> spelling calls its de-prefixed twin, so its gradient is
the twin's.  The Bessel, zeta and incomplete-gamma derivatives are checked
against closed forms at x = 1 (reference values to double precision), the
first-order Bessel functions against their finite limits at the origin, and
everything against the numerical gradient.
"""
import math

import pytest

import tensorplay as tp
from tensorplay.autograd.gradcheck import gradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
C = tp._C


def _leaf(values, device, dtype=tp.float64):
    return tp.tensor(values, dtype=dtype, device=device, requires_grad=True)


def _grad(fn, x):
    (g,) = tp.autograd.grad(fn(x).sum(), x)
    return g


def _close(got, expected, tol=1e-9):
    expected = tp.tensor(expected, dtype=got.dtype)
    assert tp.allclose(got.detach().cpu(), expected, rtol=tol, atol=tol), got


UNARY = [
    "entr", "ndtri", "log_ndtr", "expm1", "exp2", "erf", "erfc", "erfcx",
    "erfinv", "ndtr", "i0", "i0e", "i1", "i1e", "sinc", "log1p", "bessel_j0",
    "bessel_j1", "bessel_y0", "bessel_y1", "modified_bessel_i0",
    "modified_bessel_i1", "modified_bessel_k0", "modified_bessel_k1",
]
RENAMED = {"psi": "digamma", "digamma": "digamma", "gammaln": "lgamma", "expit": "sigmoid"}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", UNARY + sorted(RENAMED))
def test_a_special_spelling_differentiates_like_its_twin(device, name):
    twin = getattr(C, RENAMED.get(name, name))
    special = getattr(C, "special_" + name)
    x = _leaf([0.3, 0.7], device)
    assert tp.allclose(_grad(special, x), _grad(twin, x))


@pytest.mark.parametrize("device", DEVICES)
def test_the_binary_and_reducing_spellings_differentiate_like_their_twins(device):
    x = _leaf([0.3, 0.7], device)
    q = _leaf([1.5, 2.5], device)
    pairs = [
        (lambda t: C.special_xlogy(t, q), lambda t: C.xlogy(t, q)),
        (lambda t: C.special_xlog1py(t, q), lambda t: C.xlog1py(t, q)),
        (lambda t: C.special_xlogy(t, 2.0), lambda t: C.xlogy(t, tp.tensor(2.0, dtype=t.dtype, device=t.device))),
        (lambda t: C.special_zeta(q + 1, t + 1), lambda t: C.zeta(q + 1, t + 1)),
        (lambda t: C.special_gammainc(q, t), lambda t: C.gammainc(q, t)),
        (lambda t: C.special_gammaincc(q, t), lambda t: C.gammaincc(q, t)),
        (lambda t: C.special_logit(t), lambda t: C.logit(t)),
        (lambda t: C.special_polygamma(1, t), lambda t: C.polygamma(1, t)),
        (lambda t: C.special_multigammaln(t + 2, 2), lambda t: C.mvlgamma(t + 2, 2)),
        (lambda t: C.special_softmax(t, 0) * q, lambda t: C.softmax(t, 0) * q),
        (lambda t: C.special_log_softmax(t, 0) * q, lambda t: C.log_softmax(t, 0) * q),
        (lambda t: C.special_logsumexp(t, [0]), lambda t: C.logsumexp(t, 0)),
    ]
    for special, twin in pairs:
        assert tp.allclose(_grad(special, x), _grad(twin, x))


@pytest.mark.parametrize("device", DEVICES)
def test_bessel_derivatives_match_their_closed_forms(device):
    x = _leaf([1.0], device)
    # I1'(1) = I0(1) - I1(1); I0e'(1) = e^-1 (I1(1) - I0(1)).
    i0_1, i1_1 = 1.2660658777520082, 0.5651591039924851
    _close(_grad(C.i1, x), [i0_1 - i1_1])
    _close(_grad(C.modified_bessel_i1, x), [i0_1 - i1_1])
    _close(_grad(C.i0e, x), [math.exp(-1.0) * (i1_1 - i0_1)])
    # I1e'(1) = e^-1 (I0(1) - 2 I1(1)).
    _close(_grad(C.i1e, x), [math.exp(-1.0) * (i0_1 - 2.0 * i1_1)])
    # K0' = -K1 and K1'(1) = -(K0(1) + K1(1)).
    k0_1, k1_1 = 0.42102443824070834, 0.6019072301972346
    _close(_grad(C.modified_bessel_k0, x), [-k1_1])
    _close(_grad(C.modified_bessel_k1, x), [-(k0_1 + k1_1)])
    # J0' = -J1, Y0' = -Y1, J1'(1) = J0(1) - J1(1), Y1'(1) = Y0(1) - Y1(1).
    j0_1, j1_1 = 0.7651976865579666, 0.44005058574493355
    y0_1, y1_1 = 0.08825696421567697, -0.7812128213002887
    _close(_grad(C.bessel_j0, x), [-j1_1])
    _close(_grad(C.bessel_y0, x), [-y1_1])
    _close(_grad(C.bessel_j1, x), [j0_1 - j1_1])
    _close(_grad(C.bessel_y1, x), [y0_1 - y1_1])


@pytest.mark.parametrize("device", DEVICES)
def test_the_first_order_functions_take_their_limit_at_the_origin(device):
    for fn in (C.i1, C.i1e, C.modified_bessel_i1, C.bessel_j1):
        _close(_grad(fn, _leaf([0.0, -0.0], device)), [0.5, 0.5])
    for dtype in (tp.float32, tp.float16):
        x = _leaf([0.0], device, dtype)
        g = _grad(C.i1, x)
        assert g.dtype == dtype and g.item() == 0.5
    g = _grad(C.bessel_y1, _leaf([0.0], device))
    assert g.item() == math.inf
    assert math.isnan(_grad(C.i1, _leaf([math.nan], device)).item())


@pytest.mark.parametrize("device", DEVICES)
def test_zeta_and_the_incomplete_gamma_differentiate_in_their_second_argument(device):
    # d/dq zeta(2, q) at q = 1 is -2 zeta(3).
    q = _leaf([1.0], device)
    s = tp.tensor([2.0], dtype=tp.float64, device=device)
    _close(_grad(lambda t: C.zeta(s, t), q), [-2.0 * 1.2020569031595942])
    # d/dx P(2, x) = x e^-x and Q = 1 - P.
    x = _leaf([1.0], device)
    a = tp.tensor([2.0], dtype=tp.float64, device=device)
    _close(_grad(lambda t: C.gammainc(a, t), x), [math.exp(-1.0)])
    _close(_grad(lambda t: C.gammaincc(a, t), x), [-math.exp(-1.0)])
    for fn in (lambda t: C.zeta(t, q.detach() + 1), lambda t: C.gammainc(t, x.detach())):
        order = _leaf([2.0], device)
        with pytest.raises(RuntimeError, match="first argument"):
            fn(order).sum().backward()


def test_the_bessel_family_passes_the_numerical_gradient_check():
    x = tp.tensor([0.4, 1.3, 2.7], dtype=tp.float64, requires_grad=True)
    signed = tp.tensor([-1.7, -0.2, 0.5, 2.1], dtype=tp.float64, requires_grad=True)
    for fn in (C.i0e, C.i1, C.i1e, C.modified_bessel_i1, C.bessel_j0, C.bessel_j1):
        assert gradcheck(fn, signed)
    for fn in (C.modified_bessel_k0, C.modified_bessel_k1, C.bessel_y0, C.bessel_y1):
        assert gradcheck(fn, x)
    s = tp.tensor([2.5, 3.0, 4.0], dtype=tp.float64)
    assert gradcheck(lambda q: C.zeta(s, q), x)
    assert gradcheck(lambda t: C.gammainc(s, t), x)
    assert gradcheck(lambda t: C.gammaincc(s, t), x)


@pytest.mark.parametrize("device", DEVICES)
def test_logsumexp_reduces_any_set_of_dimensions(device):
    tp.manual_seed(0)
    x = tp.randn(2, 3, 4, dtype=tp.float64, device=device, requires_grad=True)
    ref = tp.log(tp.exp(x).sum((0, 2)))
    got = C.special_logsumexp(x, [2, 0])
    assert tp.allclose(got, ref)
    assert tuple(C.special_logsumexp(x, [0, -1], keepdim=True).shape) == (1, 3, 1)
    assert tp.allclose(C.special_logsumexp(x, []), tp.log(tp.exp(x).sum()))
    # The gradient of a log-sum-exp is the softmax over the reduced entries.
    (g,) = tp.autograd.grad(got.sum(), x)
    assert tp.allclose(g, tp.exp(x - ref.reshape(1, 3, 1)))
    with pytest.raises(RuntimeError, match="multiple times"):
        C.special_logsumexp(x, [1, -2])


@pytest.mark.parametrize("device", DEVICES)
def test_rounding_to_decimals_applies_only_the_exact_power_of_ten(device):
    x = tp.tensor([1.789, 1234.5, -0.05], dtype=tp.float64, device=device, requires_grad=True)
    up = C.special_round(x, decimals=1)
    assert up.tolist() == [1.8, 1234.5, -0.0]
    assert C.special_round(x, decimals=-2).tolist() == [0.0, 1200.0, -0.0]
    assert tp.round(x, decimals=1).tolist() == up.tolist()
    (g,) = tp.autograd.grad(up.sum(), x)
    assert g.tolist() == [0.0, 0.0, 0.0]
    half = tp.tensor([1.789], dtype=tp.float16, device=device)
    assert C.special_round(half, decimals=1).dtype == tp.float16
    with pytest.raises(RuntimeError, match="floating-point"):
        tp.round(tp.tensor([12], device=device), decimals=-1)
