"""Inverses, linear solves and least squares."""
import operator

import tensorplay
from tensorplay import _C
from tensorplay._C import (
    linalg_inv_ex,
    linalg_ldl_solve as ldl_solve,
    linalg_lu_solve as lu_solve,
    linalg_solve_triangular as solve_triangular,
)

from ._common import LstsqResult, SlogdetResult, as_index, check_floating, eps_of
from ._decompositions import eigh, svd

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
    if A.dim() < 2:
        raise ValueError("linalg.pinv: input must contain matrices")
    if hermitian and A.shape[-1] != A.shape[-2]:
        raise ValueError("linalg.pinv: hermitian input must be square")
    eps = eps_of(A.dtype)
    max_mn = max(A.shape[-2], A.shape[-1])
    atol_val = 0.0 if atol is None else float(atol)
    rtol_val = (eps * max_mn) if rtol is None else float(rtol)
    if hermitian:
        eig = eigh(A)
        values = eig.eigenvalues
        vectors = eig.eigenvectors
        magnitude = values.abs()
        cutoff = atol_val + rtol_val * magnitude.max(dim=-1, keepdim=True).values
        keep = magnitude > cutoff
        safe_values = tensorplay.where(
            keep, values, tensorplay.ones_like(values)
        )
        inverse_values = tensorplay.where(
            keep, 1.0 / safe_values, tensorplay.zeros_like(values)
        )
        vectors_h = _C.conj_physical(vectors).transpose(-2, -1)
        return (vectors * inverse_values.unsqueeze(-2)) @ vectors_h
    U, S, Vh = svd(A, full_matrices=False)
    cutoff = atol_val + rtol_val * S.max(dim=-1, keepdim=True).values
    S_inv = tensorplay.where(S > cutoff, 1.0 / S.clamp_min(1e-300), 0.0)
    V = _C.conj_physical(Vh).transpose(-2, -1)
    Uh = _C.conj_physical(U).transpose(-2, -1)
    return V @ (S_inv.unsqueeze(-1) * Uh)


def tensorinv(A, ind=2):
    """tensorinv(A, ind=2) -> Tensor

    Inverse of ``A`` seen as a square matrix over the split at ``ind``: the
    product of the leading ``ind`` dimensions must equal that of the rest.
    """
    ind = as_index(ind, "linalg.tensorinv ind")
    if A.dim() < 2:
        raise RuntimeError("linalg.tensorinv: input must have at least 2 dimensions")
    if ind <= 0 or ind >= A.dim():
        raise RuntimeError(
            f"linalg.tensorinv: ind must be in [1, {A.dim() - 1}], got {ind}")
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

    Solves the tensor equation ``A X = B`` after flattening the contracted
    dimensions into a square matrix.  ``dims`` identifies dimensions of ``A``
    that should be moved to the trailing side before the flattening step.
    """
    if dims is not None:
        try:
            moved_dims = [operator.index(dims)]
        except TypeError:
            try:
                moved_dims = [operator.index(d) for d in dims]
            except TypeError as exc:
                raise TypeError(
                    "linalg.tensorsolve: dims must contain integers") from exc
        ndim = A.dim()
        normalized = []
        for d in moved_dims:
            d = d + ndim if d < 0 else d
            if d < 0 or d >= ndim:
                raise IndexError(
                    f"linalg.tensorsolve: dimension {d} out of range for "
                    f"a {ndim}-D tensor")
            if d in normalized:
                raise ValueError(
                    "linalg.tensorsolve: dims must not contain duplicates")
            normalized.append(d)
        order = [d for d in range(ndim) if d not in normalized]
        A = A.permute(order + normalized)

    rank_b = B.dim()
    if rank_b > A.dim():
        raise RuntimeError(
            f"linalg.tensorsolve: B with shape {tuple(B.shape)} has more "
            f"dimensions than A with shape {tuple(A.shape)}")

    if tuple(A.shape[:rank_b]) != tuple(B.shape):
        raise RuntimeError(
            f"linalg.tensorsolve: B with shape {tuple(B.shape)} must match "
            f"the leading dimensions of A {tuple(A.shape[:rank_b])}")

    q_shape = list(A.shape[rank_b:])
    q_size = 1
    for size in q_shape:
        q_size *= size
    if q_size <= 0 or A.numel() != q_size * q_size or B.numel() != q_size:
        raise RuntimeError(
            f"linalg.tensorsolve: A with shape {tuple(A.shape)} and B with "
            f"shape {tuple(B.shape)} do not form a square tensor equation")

    matrix = A.reshape(q_size, q_size)
    rhs = B.reshape(q_size)
    return solve(matrix, rhs).reshape(q_shape)
