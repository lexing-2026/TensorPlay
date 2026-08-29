"""Special mathematical functions.

Functions backed by p10 pointwise kernels (erf/erfc/erfinv/exp2/expm1/log1p/
lgamma/digamma/i0/sinc/logit/exp2/...) re-export the native ops; the rest are
composed from differentiable primitives (``ndtr``/``ndtri``/``log_ndtr``/
Families that require kernels TensorPlay does not ship yet (Bessel/Airy)
raise ``NotImplementedError`` at call time instead of failing on import.
"""
import math

import tensorplay as tp
from tensorplay import (
    Tensor,
    airy_ai,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    chebyshev_polynomial_t,
    chebyshev_polynomial_u,
    chebyshev_polynomial_v,
    chebyshev_polynomial_w,
    digamma,
    erf,
    erfc,
    erfinv,
    exp,
    exp2,
    expm1,
    full_like,
    gammainc,
    gammaincc,
    hermite_polynomial_h,
    hermite_polynomial_he,
    i0,
    i0e,
    i1,
    i1e,
    laguerre_polynomial_l,
    legendre_polynomial_p,
    lgamma,
    log,
    log1p,
    logit,
    modified_bessel_i1,
    modified_bessel_k0,
    modified_bessel_k1,
    scaled_modified_bessel_k0,
    scaled_modified_bessel_k1,
    shifted_chebyshev_polynomial_t,
    shifted_chebyshev_polynomial_u,
    shifted_chebyshev_polynomial_v,
    shifted_chebyshev_polynomial_w,
    sinc,
    spherical_bessel_j0,
    sqrt,
    where,
    zeta,
)
from tensorplay._C import DType

__all__ = [
    "Tensor",
    "airy_ai",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "digamma",
    "entr",
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "exp2",
    "expit",
    "expm1",
    "gammainc",
    "gammaincc",
    "gammaln",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
    "log1p",
    "log_ndtr",
    "log_softmax",
    "logit",
    "logsumexp",
    "modified_bessel_i0",
    "modified_bessel_i1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "multigammaln",
    "ndtr",
    "ndtri",
    "polygamma",
    "psi",
    "round",
    "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "sinc",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "xlogy",
    "zeta",
]

_INV_SQRT_2 = 0.5 ** 0.5


# ---------------------------------------------------------------------------
# Native re-exports
# ---------------------------------------------------------------------------

def gammaln(input):
    """Natural log of the absolute value of the gamma function (native lgamma)."""
    return lgamma(input)


def psi(input):
    """Digamma function; alias of :func:`digamma`."""
    return digamma(input)


def round(input):
    """Rounds to nearest even integer (native ``Tensor.round``)."""
    return input.round()


def log_softmax(input, dim=None, *, dtype=None):
    from tensorplay.nn.functional import log_softmax as _ls
    return _ls(input, dim=dim if dim is not None else -1, dtype=dtype)


def softmax(input, dim=None, *, dtype=None):
    from tensorplay.nn.functional import softmax as _sm
    return _sm(input, dim=dim if dim is not None else -1, dtype=dtype)


def logsumexp(input, dim, keepdim=False, *, out=None):
    """Log-sum-exp along :attr:`dim`, computed with the max-shift trick."""
    if out is not None:
        raise NotImplementedError("logsumexp: out= is not supported")
    dims = [dim] if isinstance(dim, int) else list(dim)
    m = tp.amax(input, dim=dims, keepdim=True)
    s = (exp(input - m)).sum(dim=dims, keepdim=True)
    lse = log(s) + m
    if not keepdim:
        lse = lse.reshape([size for i, size in enumerate(lse.shape) if i not in set(dims)])
    else:
        lse = lse
    return lse


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def expit(input):
    """Numerically-stable sigmoid; identical to :func:`~tensorplay.sigmoid`."""
    return tp.sigmoid(input)


def log_ndtr(input):
    """``log`` of the standard normal CDF, stable in the left tail."""
    scaled = input * _INV_SQRT_2
    tail_mask = input < -10.0
    right = log(erfc(-scaled)) - 0.6931471805599453  # -log(2)
    t = 1.0 / (input * input)
    approx = -0.5 * input * input - 0.9189385332046727 \
        - log(t + 1.0 / (t + 2.0 / (t + 3.0 / (t + 4.0 / (t + 0.65)))))
    return where(tail_mask, approx, right)


def ndtr(input):
    """Standard normal CDF: ``Phi(x) = 0.5 * erfc(-x / sqrt(2))``."""
    return 0.5 * erfc(-input * _INV_SQRT_2)


def ndtri(input):
    """Inverse standard normal CDF: ``sqrt(2) * erfinv(2x - 1)``."""
    return -sqrt(2.0) * erfinv(2.0 * input - 1.0)


def erfcx(input):
    """Scaled complementary error function: ``exp(x^2) * erfc(x)``."""
    return exp(input * input) * erfc(input)


def entr(input):
    """Entropy of a probability element: ``-x*log(x)`` (0 at x=0, NaN below)."""
    input = input.to(_float_dtype_of(input))
    pos = input > 0
    zero = input == 0
    safe = input.clamp(min=1e-45)
    val = -(safe * log(safe))
    nan = full_like(val, float("nan"))
    return where(zero, full_like(val, 0.0), where(pos, val, nan))


def xlogy(input, other):
    """``x*log(y)`` with the convention ``0*log(anything) = 0``."""
    input, other = _promote_pair(input, other)
    res = input * log(other.clamp(min=1e-45))
    return where(input == 0, full_like(res, 0.0), res)


def xlog1py(input, other):
    """``x*log1p(y)`` with the convention ``x*log1p(-1) = 0`` when x == 0."""
    input, other = _promote_pair(input, other)
    res = input * log1p(other.clamp(min=-1.0 + 1e-12))
    return where(input == 0, full_like(res, 0.0), res)


def multigammaln(input, p):
    """Multivariate log-gamma with dimension :attr:`p` (p >= 2)."""
    p = int(p)
    if p < 2:
        raise ValueError(f"p must be >= 2, got {p}")
    res = lgamma(input)
    j = 1.0
    for i in range(1, p):
        res = res + lgamma(input - j / 2.0)
        j += 1.0
    # Constant term: p(p-1)/4 * log(pi) + sum_{j=1..p} lgamma(j/2)
    const = p * (p - 1) / 4.0 * math.log(math.pi) \
        + sum(lgamma_scalar(j / 2.0) for j in range(1, p + 1))
    return res + const


def polygamma(n, input):
    r"""Polygamma of order :attr:n: :math:`\psi^{(n)}(x)` (native kernel)."""
    from tensorplay import polygamma as _native
    order = int(n.item()) if hasattr(n, "item") else int(n)
    return _native(order, input)



def modified_bessel_i0(input):
    """Modified Bessel function of the first kind, order 0; alias of :func:`i0`."""
    return i0(input)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_dtype_of(t: Tensor) -> DType:
    dt = t.dtype
    if dt in (DType.float32, DType.float64):
        return dt
    return DType.float32


def _promote_pair(a, b):
    if not isinstance(a, Tensor):
        a = tp.tensor(float(a))
    if not isinstance(b, Tensor):
        b = tp.tensor(float(b))
    return a.to(_float_dtype_of(b)), b


def lgamma_scalar(v: float) -> float:
    return math.lgamma(v)
