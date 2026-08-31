"""Special mathematical functions.

The package groups the families into modules so each can grow on its own:

* :mod:`~tensorplay.special._gamma` -- log-gamma, poly-gamma, incomplete gammas, zeta.
* :mod:`~tensorplay.special._error` -- erf family, normal CDF/quantile, entropy terms.
* :mod:`~tensorplay.special._bessel` -- Bessel, modified-Bessel and Airy functions.
* :mod:`~tensorplay.special._polynomials` -- Chebyshev/Hermite/Laguerre/Legendre families.
* :mod:`~tensorplay.special._elementwise` -- pointwise transforms and the
  normalized-exponential reductions.
* :mod:`~tensorplay.special._common` -- dtype promotion shared by the above.

Functions with a dedicated p10 pointwise kernel re-export the native op; the
rest are composed from differentiable primitives so they stay usable under
autograd.
"""
from tensorplay import Tensor

from ._bessel import (
    airy_ai,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    i0,
    i0e,
    i1,
    i1e,
    modified_bessel_i0,
    modified_bessel_i1,
    modified_bessel_k0,
    modified_bessel_k1,
    scaled_modified_bessel_k0,
    scaled_modified_bessel_k1,
    spherical_bessel_j0,
)
from ._elementwise import (
    exp2,
    expm1,
    log1p,
    log_softmax,
    logsumexp,
    round,
    sinc,
    softmax,
)
from ._error import (
    entr,
    erf,
    erfc,
    erfcx,
    erfinv,
    expit,
    log_ndtr,
    logit,
    ndtr,
    ndtri,
    xlog1py,
    xlogy,
)
from ._gamma import (
    digamma,
    gammainc,
    gammaincc,
    gammaln,
    multigammaln,
    polygamma,
    psi,
    zeta,
)
from ._polynomials import (
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
