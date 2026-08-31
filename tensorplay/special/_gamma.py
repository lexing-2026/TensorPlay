"""Gamma-family functions: log-gamma, poly-gamma and the incomplete gammas."""
import math

from tensorplay import digamma, gammainc, gammaincc, lgamma, zeta

from ._common import lgamma_scalar

__all__ = [
    "digamma",
    "gammainc",
    "gammaincc",
    "gammaln",
    "multigammaln",
    "polygamma",
    "psi",
    "zeta",
]


def gammaln(input):
    """Natural log of the absolute value of the gamma function (native lgamma)."""
    return lgamma(input)


def psi(input):
    """Digamma function; alias of :func:`digamma`."""
    return digamma(input)


def polygamma(n, input):
    r"""Polygamma of order :attr:`n`: :math:`\psi^{(n)}(x)` (native kernel)."""
    from tensorplay import polygamma as _native
    order = int(n.item()) if hasattr(n, "item") else int(n)
    return _native(order, input)


def multigammaln(input, p):
    """Multivariate log-gamma with dimension :attr:`p` (p >= 2)."""
    p = int(p)
    if p < 2:
        raise ValueError(f"p must be >= 2, got {p}")
    res = lgamma(input)
    j = 1.0
    for _ in range(1, p):
        res = res + lgamma(input - j / 2.0)
        j += 1.0
    # Constant term: p(p-1)/4 * log(pi) + sum_{j=1..p} lgamma(j/2)
    const = p * (p - 1) / 4.0 * math.log(math.pi) \
        + sum(lgamma_scalar(j / 2.0) for j in range(1, p + 1))
    return res + const
