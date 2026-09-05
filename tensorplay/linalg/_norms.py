"""Vector and matrix norms, and the quantities derived from them."""
import tensorplay

from ._common import eps_of
from ._decompositions import eigvalsh, svdvals
from ._solve import inv

__all__ = ["cond", "matrix_norm", "matrix_rank", "norm", "vector_norm"]


def _normalize_dims(dim, ndim, expected=None):
    if ndim <= 0:
        raise ValueError(f"a reduction dimension requires a non-empty input, got {ndim}-D")
    dims = [dim] if isinstance(dim, int) else [int(d) for d in dim]
    if expected is not None and len(dims) != expected:
        raise ValueError(f"expected exactly {expected} dimensions, got {len(dims)}")
    if not dims:
        raise ValueError("at least one reduction dimension must be specified")
    result = []
    for value in dims:
        value = value + ndim if value < 0 else value
        if value < 0 or value >= ndim:
            raise ValueError(f"dimension {value} out of range for {ndim}-D input")
        if value in result:
            raise ValueError("reduction dimensions must be unique")
        result.append(value)
    return result


def _restore_matrix_axes(value, remaining, matrix_dims, ndim):
    current_positions = {dim: index for index, dim in enumerate(remaining)}
    offset = len(remaining)
    current_positions[matrix_dims[0]] = offset
    current_positions[matrix_dims[1]] = offset + 1
    order = [current_positions[dim] for dim in range(ndim)]
    return value.permute(order)


def vector_norm(x, ord=2, dim=None, keepdim=False):
    """vector_norm(x, ord=2, dim=None, keepdim=False) -> Tensor

    ``dim=None`` norms the whole tensor: the input is flattened first, and
    ``keepdim`` then restores the reduced axes as ones.
    """
    inf = float("inf")
    reduce_all = dim is None
    ndim = x.dim()
    work = x.reshape([-1]) if reduce_all else x
    axes = [0] if reduce_all else _normalize_dims(dim, ndim)
    # Reducing from the last axis toward the first keeps the remaining axis
    # numbers stable when keepdim is false.
    axes = sorted(axes, reverse=True)
    inner_keepdim = keepdim and not reduce_all
    magnitude = work.abs()
    if ord == 0:
        result = (work != 0).to(work.dtype)
        for axis in axes:
            result = result.sum(dim=axis, keepdim=inner_keepdim)
    elif ord == inf:
        result = magnitude
        for axis in axes:
            result = result.max(dim=axis, keepdim=inner_keepdim).values
    elif ord == -inf:
        result = magnitude
        for axis in axes:
            result = result.min(dim=axis, keepdim=inner_keepdim).values
    else:
        if isinstance(ord, str) or ord == 0:
            raise ValueError(f"linalg.vector_norm: invalid ord {ord!r}")
        result = magnitude.pow(ord)
        for axis in axes:
            result = result.sum(dim=axis, keepdim=inner_keepdim)
        result = result.pow(1.0 / ord)
    if reduce_all and keepdim:
        result = result.reshape([1] * ndim)
    return result


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False):
    """matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False) -> Tensor"""
    inf = float("inf")
    ndim = A.dim()
    matrix_dims = _normalize_dims(dim, ndim, expected=2)
    remaining = [axis for axis in range(ndim) if axis not in matrix_dims]
    moved = A.permute(remaining + matrix_dims).contiguous()
    batch_shape = list(moved.shape[:-2])

    def finish(value):
        if keepdim:
            value = value.reshape(batch_shape + [1, 1])
            return _restore_matrix_axes(value, remaining, matrix_dims, ndim)
        return value

    magnitude = moved.abs()
    if ord == "fro":
        return finish(magnitude.pow(2).sum(dim=[-2, -1]).sqrt())
    if ord == "nuc":
        return finish(svdvals(moved).sum(dim=-1))
    if ord == inf:
        return finish(magnitude.sum(dim=-1).max(dim=-1).values)
    if ord == -inf:
        return finish(magnitude.sum(dim=-1).min(dim=-1).values)
    if ord == 1:
        return finish(magnitude.sum(dim=-2).max(dim=-1).values)
    if ord == -1:
        return finish(magnitude.sum(dim=-2).min(dim=-1).values)
    if ord == 2 or ord == -2:
        singular_values = svdvals(moved)
        value = singular_values.max(dim=-1).values if ord == 2 \
            else singular_values.min(dim=-1).values
        return finish(value)
    raise RuntimeError(f"linalg.matrix_norm: invalid ord {ord!r}")


def norm(input, ord=None, dim=None, keepdim=False):
    """norm(input, ord=None, dim=None, keepdim=False) -> Tensor"""
    if dim is None:
        if isinstance(ord, str):
            if ord not in ("fro", "frob"):
                raise ValueError(f"linalg.norm: invalid ord {ord!r}")
            ord = 2
        return vector_norm(input, ord=2 if ord is None else ord,
                           dim=None, keepdim=keepdim)
    dims = _normalize_dims(dim, input.dim())
    if len(dims) == 1:
        if isinstance(ord, str):
            raise ValueError(
                "linalg.norm: a string ord requires two reduction dimensions")
        return vector_norm(input, ord=2 if ord is None else ord,
                           dim=dims, keepdim=keepdim)
    if len(dims) == 2:
        return matrix_norm(input, ord="fro" if ord is None else ord,
                           dim=dims, keepdim=keepdim)
    raise ValueError("linalg.norm supports one or two reduction dimensions")


def matrix_rank(A, *, atol=None, rtol=None, hermitian=False):
    """matrix_rank(A, *, atol=None, rtol=None, hermitian=False) -> Tensor"""
    if A.dim() < 2:
        raise ValueError("linalg.matrix_rank: input must contain matrices")
    S = eigvalsh(A) if hermitian else svdvals(A)
    max_S = S.max(dim=-1).values
    eps = eps_of(A.dtype)
    rtol_val = eps * max(A.shape[-2], A.shape[-1]) if rtol is None else rtol
    tol = max(atol if atol is not None else 0.0, 0.0) + rtol_val * max_S
    return (S > tol).to(S.dtype).sum(dim=-1)


def cond(A, p=None):
    """cond(A, p=None) -> Tensor"""
    if p is None:
        S = svdvals(A)
        return S.max(dim=-1).values / S.min(dim=-1).values
    if p == 2:
        S = svdvals(A)
        return S.max(dim=-1).values / S.min(dim=-1).values
    if p in ("fro", "nuc", float("inf"), -float("inf"), 1, -1):
        return matrix_norm(A, ord=p) * matrix_norm(inv(A), ord=p)
    raise RuntimeError(f"linalg.cond: p={p!r} is not supported")
