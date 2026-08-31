"""Orthogonal polynomial families evaluated by the native pointwise kernels.

Each takes ``(x, n)`` and evaluates the degree-``n`` member of the family at
``x``; ``n`` broadcasts against ``x``.
"""
from tensorplay import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_u,
    chebyshev_polynomial_v,
    chebyshev_polynomial_w,
    hermite_polynomial_h,
    hermite_polynomial_he,
    laguerre_polynomial_l,
    legendre_polynomial_p,
    shifted_chebyshev_polynomial_t,
    shifted_chebyshev_polynomial_u,
    shifted_chebyshev_polynomial_v,
    shifted_chebyshev_polynomial_w,
)

__all__ = [
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
]
