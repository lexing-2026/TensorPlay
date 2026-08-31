"""Matrix factorizations.

The kernels behind these bind to the CPU and CUDA decomposition backends;
the wrappers add the named result tuples and the mode handling.
"""
import tensorplay
from tensorplay import _C
from tensorplay._C import (
    linalg_cholesky,
    linalg_cholesky_ex,
    linalg_eig,
    linalg_eigh,
    linalg_eigvals,
    linalg_eigvalsh,
    linalg_householder_product as householder_product,
    linalg_ldl_factor as ldl_factor,
    linalg_ldl_factor_ex as ldl_factor_ex,
    linalg_lu as lu,
    linalg_lu_factor as lu_factor,
    linalg_lu_factor_ex as lu_factor_ex,
)

from ._common import EigResult, EighResult, QRResult, SVDResult

__all__ = [
    "cholesky",
    "cholesky_ex",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "householder_product",
    "ldl_factor",
    "ldl_factor_ex",
    "lu",
    "lu_factor",
    "lu_factor_ex",
    "polar",
    "qr",
    "svd",
    "svdvals",
]


def cholesky(A, *, upper=False):
    """cholesky(A, *, upper=False) -> Tensor"""
    return linalg_cholesky(A, upper=upper)


def cholesky_ex(A, *, upper=False, check_errors=False):
    """cholesky_ex(A, *, upper=False, check_errors=False) -> (Tensor, Tensor)"""
    return linalg_cholesky_ex(A, upper=upper, check_errors=check_errors)


def eigh(A, UPLO="L"):
    """eigh(A, UPLO='L') -> EighResult(eigenvalues, eigenvectors)"""
    values, vectors = linalg_eigh(A, UPLO)
    return EighResult(values, vectors)


def eigvalsh(A, UPLO="L"):
    """eigvalsh(A, UPLO='L') -> Tensor"""
    return linalg_eigvalsh(A, UPLO)


def eig(A):
    """eig(A) -> EigResult(eigenvalues, eigenvectors)"""
    values, vectors = linalg_eig(A)
    return EigResult(values, vectors)


def eigvals(A):
    """eigvals(A) -> Tensor"""
    return linalg_eigvals(A)


def svd(A, full_matrices=True, *, driver=None):
    """svd(A, full_matrices=True, *, driver=None) -> SVDResult(U, S, Vh)"""
    U, S, Vh = _C.linalg_svd(A, full_matrices, driver=driver)
    return SVDResult(U, S, Vh)


def svdvals(A, *, driver=None):
    """svdvals(A, *, driver=None) -> Tensor"""
    return _C.linalg_svdvals(A, driver=driver)


def qr(A, mode="reduced"):
    """qr(A, mode='reduced') -> QRResult(Q, R)"""
    Q, R = _C.linalg_qr(A, mode)
    if mode in ("r", "R"):
        empty = tensorplay.empty(list(A.shape[:-2]) + [A.shape[-2], 0], dtype=A.dtype)
        return QRResult(empty, R)
    return QRResult(Q, R)


def polar(A):
    """polar(A) -> (Tensor Q, Tensor R) with A = Q R"""
    U, S, Vh = svd(A, full_matrices=False)
    Q = U @ Vh
    R = Vh.transpose(-2, -1) @ (S.unsqueeze(-1) * Vh)
    return Q, R
