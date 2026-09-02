"""Linear algebra.

Decompositions bind to the native CPU and CUDA backends; everything else is
composed from differentiable primitives, so autograd works through the whole
namespace.  The package is split so each family can grow on its own:

* :mod:`~tensorplay.linalg._decompositions` -- Cholesky/LU/LDL/QR/eigen/SVD/polar.
* :mod:`~tensorplay.linalg._solve` -- inverses, solves, least squares, pseudo-inverse.
* :mod:`~tensorplay.linalg._norms` -- vector/matrix norms, rank, condition number.
* :mod:`~tensorplay.linalg._matrix_functions` -- matrix exponential/square root/power
  and the chained products.
* :mod:`~tensorplay.linalg._common` -- result tuples and the dtype guards.

Current scope: real float32/float64.  Complex inputs raise NotImplementedError.
"""
from ._common import (
    EigResult,
    EighResult,
    LinAlgError,
    LstsqResult,
    QRResult,
    SVDResult,
    SlogdetResult,
)
from ._decompositions import (
    cholesky,
    cholesky_ex,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    householder_product,
    ldl_factor,
    ldl_factor_ex,
    lu,
    lu_factor,
    lu_factor_ex,
    polar,
    qr,
    svd,
    svdvals,
)
from ._matrix_functions import (
    cross,
    diagonal,
    matmul,
    matrix_exp,
    matrix_power,
    matrix_sqrth,
    multi_dot,
    vander,
    vecdot,
    vdot,
)
from ._norms import cond, matrix_norm, matrix_rank, norm, vector_norm
from ._solve import (
    det,
    inv,
    inv_ex,
    ldl_solve,
    lstsq,
    lu_solve,
    pinv,
    slogdet,
    solve,
    solve_ex,
    solve_triangular,
    tensorinv,
    tensorsolve,
)

__all__ = [
    "LinAlgError", "cross", "cholesky", "cholesky_ex", "inv", "solve_ex",
    "inv_ex", "det", "slogdet", "eig", "eigvals", "eigh", "eigvalsh",
    "householder_product", "ldl_factor", "ldl_factor_ex", "ldl_solve",
    "lstsq", "matrix_power", "matrix_rank", "norm", "vector_norm",
    "matrix_norm", "matmul", "diagonal", "multi_dot", "svd", "svdvals",
    "cond", "pinv", "matrix_exp", "matrix_sqrth", "solve",
    "solve_triangular", "lu_factor", "lu_factor_ex", "lu_solve", "lu",
    "tensorinv", "tensorsolve", "qr", "polar", "vander", "vecdot", "vdot",
]
