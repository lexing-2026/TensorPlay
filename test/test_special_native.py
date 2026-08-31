"""Native special-function kernels: values, dtypes, corners and gradients.

References are computed from the C library's ``erf``/``erfc``/``lgamma`` in
double precision, or from the asymptotic expansions in the regimes where those
closed forms have themselves stopped being accurate.
"""
import math

import numpy as np
import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
SQRT_2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _tensor(values, device, dtype=tp.float64):
    return tp.tensor(list(values), dtype=dtype).to(device)


def _numpy(t):
    return t.cpu().to(tp.float64).numpy()


# ---------------------------------------------------------------------------
# scalar references
# ---------------------------------------------------------------------------

def _ref_erfcx(x):
    """exp(x^2)*erfc(x), evaluated in whichever form still has digits left."""
    if x >= 0.0:
        if x <= 5.0:
            return math.exp(x * x) * math.erfc(x)
        # Asymptotic series 1/(x*sqrt(pi)) * sum (-1)^n (2n-1)!! / (2x^2)^n.
        inv = 1.0 / (2.0 * x * x)
        series = 1.0
        term = 1.0
        for n in range(1, 12):
            term *= -(2 * n - 1) * inv
            series += term
        return series / (x * math.sqrt(math.pi))
    if x < -26.7:
        return math.inf
    if x < -6.1:
        return 2.0 * math.exp(x * x)
    return 2.0 * math.exp(x * x) - math.exp(x * x) * math.erfc(-x)


def _ref_ndtr(x):
    return 0.5 * math.erfc(-x / SQRT_2)


def _ref_log_ndtr(x):
    if x > -1.0:
        # log1p keeps the digits that log(1 - tiny) would round away.
        return math.log1p(-0.5 * math.erfc(x / SQRT_2))
    if x > -10.0:
        return math.log(_ref_ndtr(x))
    # Mills-ratio expansion of log(phi(x)/-x) for the far left tail.
    inv = 1.0 / (x * x)
    series = 1.0
    term = 1.0
    for n in range(1, 10):
        term *= -(2 * n - 1) * inv
        series += term
    return -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi) + math.log(series)


def _ref_entr(x):
    if math.isnan(x):
        return x
    if x > 0.0:
        return -x * math.log(x)
    if x == 0.0:
        return 0.0
    return -math.inf


def _ref_i0(x):
    """Series sum_k ((|x|/2)^k / k!)^2, converged to double precision."""
    half = 0.5 * abs(x)
    term = 1.0
    total = 1.0
    for k in range(1, 200):
        term *= half / k
        contribution = term * term
        total += contribution
        if contribution < 1e-18 * total:
            break
    return total


def _assert_close(got, expected, rtol=1e-12, atol=0.0):
    np.testing.assert_allclose(got, np.asarray(expected, dtype=np.float64),
                               rtol=rtol, atol=atol, equal_nan=True)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

ERFCX_POINTS = [-30.0, -20.0, -8.0, -6.5, -3.0, -1.0, -0.25, 0.0,
                0.25, 1.0, 3.0, 8.0, 40.0, 100.0, 1e8]


@pytest.mark.parametrize("device", DEVICES)
def test_erfcx_matches_reference(device):
    got = _numpy(tp.erfcx(_tensor(ERFCX_POINTS, device)))
    _assert_close(got, [_ref_erfcx(v) for v in ERFCX_POINTS], rtol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_erfcx_beats_the_factored_form(device):
    """The closed form overflows on the left and cancels on the right."""
    x = _tensor([-30.0, 1e6], device)
    got = _numpy(tp.erfcx(x))
    assert math.isinf(got[0]) and got[0] > 0
    assert got[1] == pytest.approx(_ref_erfcx(1e6), rel=1e-10)
    # The factored form reaches neither: inf on the left, inf*0 on the right.
    naive = _numpy(tp.exp(x * x) * tp.erfc(x))
    assert math.isnan(naive[1]) or naive[1] == 0.0


NDTR_POINTS = [-40.0, -12.0, -5.0, -1.0, 0.0, 0.5, 1.0, 5.0, 12.0, 40.0]


@pytest.mark.parametrize("device", DEVICES)
def test_ndtr_matches_reference(device):
    got = _numpy(tp.ndtr(_tensor(NDTR_POINTS, device)))
    _assert_close(got, [_ref_ndtr(v) for v in NDTR_POINTS], rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("device", DEVICES)
def test_log_ndtr_matches_reference(device):
    points = [-40.0, -25.0, -12.0, -10.5, -3.0, -1.0, 0.0, 1.0, 5.0, 20.0]
    got = _numpy(tp.log_ndtr(_tensor(points, device)))
    _assert_close(got, [_ref_log_ndtr(v) for v in points], rtol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_log_ndtr_survives_where_ndtr_underflows(device):
    """ndtr(-40) rounds to zero, so log(ndtr(x)) would be -inf there."""
    x = _tensor([-40.0], device)
    assert _numpy(tp.ndtr(x))[0] == pytest.approx(_ref_ndtr(-40.0), rel=1e-10)
    value = _numpy(tp.log_ndtr(x))[0]
    assert math.isfinite(value)
    assert value == pytest.approx(_ref_log_ndtr(-40.0), rel=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_ndtri_is_the_inverse_of_ndtr(device):
    probabilities = [1e-12, 1e-6, 0.001, 0.025, 0.25, 0.5, 0.75, 0.975, 0.999, 1 - 1e-9]
    quantiles = _numpy(tp.ndtri(_tensor(probabilities, device)))
    _assert_close([_ref_ndtr(q) for q in quantiles], probabilities, rtol=1e-9)
    # Sign convention: the upper quantiles are positive.
    assert quantiles[-1] > 0 and quantiles[0] < 0
    assert quantiles[7] == pytest.approx(1.959963984540054, rel=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_ndtri_boundaries(device):
    got = _numpy(tp.ndtri(_tensor([0.0, 1.0, -0.5, 1.5], device)))
    assert got[0] == -math.inf
    assert got[1] == math.inf
    assert math.isnan(got[2]) and math.isnan(got[3])


ENTR_POINTS = [-2.0, -1e-30, 0.0, 1e-30, 0.25, 1.0, 2.0, 1e10]


@pytest.mark.parametrize("device", DEVICES)
def test_entr_matches_reference(device):
    got = _numpy(tp.entr(_tensor(ENTR_POINTS, device)))
    _assert_close(got, [_ref_entr(v) for v in ENTR_POINTS], rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_entr_is_negative_infinity_below_zero(device):
    """Negative inputs leave the domain; the limit from inside is -inf."""
    got = _numpy(tp.entr(_tensor([-1.0, -1e-8], device)))
    assert got[0] == -math.inf and got[1] == -math.inf


@pytest.mark.parametrize("device", DEVICES)
def test_modified_bessel_i0_matches_series(device):
    points = [-12.0, -3.5, -1.0, 0.0, 0.5, 1.0, 3.5, 12.0, 30.0]
    got = _numpy(tp.modified_bessel_i0(_tensor(points, device)))
    _assert_close(got, [_ref_i0(v) for v in points], rtol=1e-11)
    # The short spelling has to agree with the long one.
    _assert_close(_numpy(tp.i0(_tensor(points, device))), got, rtol=1e-11)


# ---------------------------------------------------------------------------
# the zero-times-singular convention
# ---------------------------------------------------------------------------

XLOG_CASES = [
    # (x, y)
    (0.0, 0.0), (0.0, -1.0), (0.0, math.inf), (0.0, -math.inf),
    (2.0, 3.0), (-1.5, 0.5), (3.0, 0.0), (1.0, -2.0),
]


def _ref_xlogy(x, y):
    if math.isnan(y):
        return math.nan
    if x == 0.0:
        return 0.0
    if y < 0.0:
        return math.nan      # log(y) has no real value; the product carries it
    if y == 0.0:
        return -math.inf if x > 0 else math.inf
    return x * math.log(y)


@pytest.mark.parametrize("device", DEVICES)
def test_xlogy_zero_shortcut_and_nan_propagation(device):
    xs = [c[0] for c in XLOG_CASES] + [0.0, 2.0]
    ys = [c[1] for c in XLOG_CASES] + [math.nan, math.nan]
    got = _numpy(tp.xlogy(_tensor(xs, device), _tensor(ys, device)))
    _assert_close(got, [_ref_xlogy(x, y) for x, y in zip(xs, ys)], rtol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_xlog1py_zero_shortcut_and_nan_propagation(device):
    xs = [0.0, 0.0, 0.0, 2.0, -1.5, 3.0, 0.0, 2.0]
    ys = [-1.0, 0.0, math.inf, 3.0, 0.5, -1.0, math.nan, math.nan]
    got = _numpy(tp.xlog1py(_tensor(xs, device), _tensor(ys, device)))
    expected = []
    for x, y in zip(xs, ys):
        if math.isnan(y):
            expected.append(math.nan)
        elif x == 0.0:
            expected.append(0.0)
        elif y == -1.0:
            expected.append(-math.inf if x > 0 else math.inf)
        else:
            expected.append(x * math.log1p(y))
    _assert_close(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype, rtol", [(tp.float32, 1e-6), (tp.float64, 1e-12)])
@pytest.mark.parametrize("name", ["erfcx", "ndtr", "ndtri", "log_ndtr", "entr",
                                  "modified_bessel_i0"])
def test_unary_dtype_is_preserved(device, dtype, rtol, name):
    points = {"ndtri": [0.1, 0.4, 0.9]}.get(name, [0.25, 1.0, 2.5])
    op = getattr(tp, name)
    out = op(_tensor(points, device, dtype))
    assert out.dtype == dtype
    _assert_close(_numpy(out), _numpy(op(_tensor(points, device))), rtol=rtol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["xlogy", "xlog1py", "zeta"])
def test_binary_reduced_width_dtype(device, name):
    """Half inputs must be widened for the math and narrowed once at the end."""
    op = getattr(tp, name)
    a = [2.0, 3.0, 4.0]
    b = [1.5, 2.0, 2.5]
    reference = _numpy(op(_tensor(a, device), _tensor(b, device)))
    for dtype in (tp.float16, tp.bfloat16):
        out = op(_tensor(a, device, dtype), _tensor(b, device, dtype))
        assert out.dtype == dtype
        _assert_close(_numpy(out), reference, rtol=3e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_integral_input_promotes_to_float32(device):
    out = tp.entr(tp.tensor([1, 2, 3], dtype=tp.int64).to(device))
    assert out.dtype == tp.float32


@pytest.mark.parametrize("device", DEVICES)
def test_empty_and_broadcast_shapes(device):
    empty = tp.entr(tp.tensor([], dtype=tp.float64).to(device))
    assert empty.numel() == 0
    x = tp.tensor([[1.0], [2.0]], dtype=tp.float64).to(device)
    y = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float64).to(device)
    assert list(tp.xlog1py(x, y).shape) == [2, 3]


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _numeric_grad(fn, values, step=1e-6):
    out = []
    for index in range(len(values)):
        up = list(values)
        down = list(values)
        up[index] += step
        down[index] -= step
        out.append((fn(up) - fn(down)) / (2 * step))
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, points", [
    ("erfcx", [-1.0, 0.5, 2.0]),
    ("ndtr", [-1.5, 0.0, 1.5]),
    ("ndtri", [0.2, 0.5, 0.8]),
    ("log_ndtr", [-2.0, 0.0, 1.0]),
    ("entr", [0.25, 1.0, 2.0]),
    ("modified_bessel_i0", [-2.0, 0.5, 3.0]),
])
def test_unary_backward(device, name, points):
    op = getattr(tp, name)
    x = _tensor(points, device)
    x.requires_grad_(True)
    op(x).sum().backward()
    analytic = _numpy(x.grad)

    def value(vs):
        return float(_numpy(op(_tensor(vs, device))).sum())

    _assert_close(analytic, _numeric_grad(value, points), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["xlogy", "xlog1py"])
def test_binary_backward(device, name):
    op = getattr(tp, name)
    xs = [1.5, -2.0, 3.0]
    ys = [0.5, 2.0, 4.0]
    x = _tensor(xs, device)
    y = _tensor(ys, device)
    x.requires_grad_(True)
    y.requires_grad_(True)
    op(x, y).sum().backward()

    def value_x(vs):
        return float(_numpy(op(_tensor(vs, device), _tensor(ys, device))).sum())

    def value_y(vs):
        return float(_numpy(op(_tensor(xs, device), _tensor(vs, device))).sum())

    _assert_close(_numpy(x.grad), _numeric_grad(value_x, xs), rtol=1e-5, atol=1e-6)
    _assert_close(_numpy(y.grad), _numeric_grad(value_y, ys), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["xlogy", "xlog1py"])
def test_zero_times_singular_gradient_is_finite(device, name):
    """Where the forward collapses to 0, the gradient must not be NaN."""
    op = getattr(tp, name)
    singular = -1.0 if name == "xlog1py" else 0.0
    x = _tensor([0.0, 0.0], device)
    y = _tensor([singular, singular - 1.0], device)
    x.requires_grad_(True)
    op(x, y).sum().backward()
    assert np.all(np.isfinite(_numpy(x.grad)))


# ---------------------------------------------------------------------------
# the tensorplay.special surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["entr", "erfcx", "log_ndtr", "ndtr", "ndtri",
                                  "xlog1py", "xlogy"])
def test_special_namespace_uses_the_native_op(name):
    assert getattr(tp.special, name) is getattr(tp, name)


def test_special_modified_bessel_i0_is_the_native_op():
    assert tp.special.modified_bessel_i0 is tp.modified_bessel_i0
