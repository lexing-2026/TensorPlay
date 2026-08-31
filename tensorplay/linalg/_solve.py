"""Inverses, linear solves and least squares."""
import tensorplay
from tensorplay import _C
from tensorplay._C import (
    linalg_inv_ex,
    linalg_ldl_solve as ldl_solve,
    linalg_lu_solve as lu_solve,
    linalg_solve_triangular as solve_triangular,
)

from ._common import LstsqResult, SlogdetResult, check_floating, eps_of
from ._decompositions import svd

__all__ = [
    "det",
    "inv",
    "inv_ex",
    "ldl_solve",
    "lstsq",
    "lu_solve",
    "pinv",
    "slogdet",
    "solve",
    "solve_ex",
    "solve_triangular",
    "tensorinv",
    "tensorsolve",
]


def inv(A):
    """inv(A) -> Tensor"""
    return _C.linalg_inv(A)


def inv_ex(A, *, check_errors=False):
    """inv_ex(A, *, check_errors=False) -> (Tensor, Tensor)"""
    return linalg_inv_ex(A, check_errors=check_errors)


def det(A):
    """det(A) -> Tensor"""
    return _C.linalg_det(A)


def slogdet(A):
    """slogdet(A) -> SlogdetResult(sign, logabsdet)"""
    sign, logabsdet = _C.linalg_slogdet(A)
    return SlogdetResult(sign, logabsdet)


def solve(A, B, *, left=True):
    """solve(A, B, *, left=True) -> Tensor"""
    return _C.linalg_solve(A, B, left=left)


def solve_ex(A, B, *, left=True, check_errors=False):
    """solve_ex(A, B, *, left=True, check_errors=False) -> (Tensor, Tensor)"""
    return _C.linalg_solve_ex(A, B, left=left, check_errors=check_errors)


def lstsq(A, B, rcond=None, *, driver=None):
    """lstsq(A, B, rcond=None, *, driver=None) -> LstsqResult(solution, residuals, rank, singular_values)"""
    solution, residuals, rank, coefficients = _C.linalg_lstsq(A, B, rcond, driver=driver)
    return LstsqResult(solution, residuals, rank, coefficients)


def pinv(A, *, atol=None, rtol=None, hermitian=False):
    """pinv(A, *, atol=None, rtol=None, hermitian=False) -> Tensor

    Moore-Penrose pseudo-inverse built from the SVD, with singular values at
    or below ``atol + rtol * sigma_max`` treated as zero.
    """
    check_floating(A, "pinv")
    U, S, Vh = svd(A, full_matrices=False)
    eps = eps_of(A.dtype)
    max_mn = max(A.shape[-2], A.shape[-1])
    atol_val = 0.0 if atol is None else float(atol)
    rtol_val = (eps * max_mn) if rtol is None else float(rtol)
    cutoff = atol_val + rtol_val * S.max(dim=-1, keepdim=True).values
    S_inv = tensorplay.where(S > cutoff, 1.0 / S.clamp_min(1e-300), 0.0)
    return Vh.transpose(-2, -1) @ (S_inv.unsqueeze(-1) * U.transpose(-2, -1))


def tensorinv(A, ind=2):
    """tensorinv(A, ind=2) -> Tensor

    Inverse of ``A`` seen as a square matrix over the split at ``ind``: the
    product of the leading ``ind`` dimensions must equal that of the rest.
    """
    if ind <= 0:
        raise RuntimeError("linalg.tensorinv: ind must be > 0")
    shape = list(A.shape)
    prod_front = 1
    for d in shape[:ind]:
        prod_front *= d
    prod_tail = 1
    for d in shape[ind:]:
        prod_tail *= d
    if prod_front != prod_tail:
        raise RuntimeError(
            "linalg.tensorinv: expected an equal product of dimensions on both "
            f"sides of ind={ind}, got {prod_front} and {prod_tail}")
    Ainv2 = inv(A.reshape(prod_front, prod_tail))
    return Ainv2.reshape(shape[ind:] + shape[:ind])


def tensorsolve(A, B, dims=None):
    """tensorsolve(A, B, dims=None) -> Tensor

    Solves A X = B where A and B are (tuples of) matrices interpreted as a
    single square system over the trailing dimensions.
    """
    if dims is not None:
        raise NotImplementedError(
            "linalg.tensorsolve: the dims argument is not implemented yet")
    shape_x = list(A.shape)
    front = shape_x[-1]
    tail = 1
    for d in shape_x[:-1]:
        tail *= d
    prod_b = 1
    for d in B.shape:
        prod_b *= d
    if front * front != tail or prod_b != tail:
        raise RuntimeError(
            f"linalg.tensorsolve: input tensor A of shape {tuple(shape_x)} cannot "
            f"be reshaped into a ({tail}, {tail}) square system matching B of "
            f"shape {tuple(B.shape)}")
    sol = solve(A.reshape(tail, front), B.reshape(prod_b).unsqueeze(-1))
    return sol.squeeze(-1).reshape(shape_x[:-1])
