"""Error function, normal-distribution and entropy-style transforms.

Every entry that has a dedicated pointwise kernel re-exports it directly. The
scalar routines behind those kernels keep their accuracy across the whole
domain, which the obvious closed forms do not: ``exp(x*x) * erfc(x)``
overflows below ``x = -26.6`` and cancels to zero for large positive ``x``,
and ``log(ndtr(x))`` has lost every significant digit by ``x = -10``.
"""
import tensorplay as tp
from tensorplay import (
    entr,
    erf,
    erfc,
    erfcx,
    erfinv,
    log_ndtr,
    logit,
    ndtr,
    ndtri,
    xlog1py,
    xlogy,
)

__all__ = [
    "entr",
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "expit",
    "log_ndtr",
    "logit",
    "ndtr",
    "ndtri",
    "xlog1py",
    "xlogy",
]


def expit(input):
    """Numerically-stable sigmoid; identical to :func:`~tensorplay.sigmoid`."""
    return tp.sigmoid(input)
