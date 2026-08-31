"""Vector and matrix norms, and the quantities derived from them."""
import tensorplay

from ._common import eps_of
from ._decompositions import eigvalsh, svdvals
from ._solve import inv

__all__ = ["cond", "matrix_norm", "matrix_rank", "norm", "vector_norm"]


def vector_norm(x, ord=2, dim=None, keepdim=False):
    """vector_norm(x, ord=2, dim=None, keepdim=False) -> Tensor

    ``dim=None`` norms the whole tensor: the input is flattened first, and
    ``keepdim`` then restores the reduced axes as ones.
    """
    inf = float("inf")
    reduce_all = dim is None
    ndim = x.dim()
    work = x.reshape([-1]) if reduce_all else x
    axis = 0 if reduce_all else dim
    # A flattened reduction rebuilds the kept axes afterwards, since the
    # working tensor no longer has them.
    inner_keepdim = keepdim and not reduce_all
    magnitude = work.abs()
    if ord == 0:
        result = (work != 0).to(work.dtype).sum(dim=axis, keepdim=inner_keepdim)
    elif ord == inf:
        result = magnitude.max(dim=axis, keepdim=inner_keepdim).values
    elif ord == -inf:
        result = magnitude.min(dim=axis, keepdim=inner_keepdim).values
    else:
        result = magnitude.pow(ord).sum(dim=axis, keepdim=inner_keepdim).pow(1.0 / ord)
    if reduce_all and keepdim:
        result = result.reshape([1] * ndim)
    return result


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False):
    """matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False) -> Tensor"""
    inf = float("inf")
    ndim = A.dim()
    d1 = dim[0] % ndim
    d2 = dim[1] % ndim

    def adj(d):
        # Index of `d` after the sibling dimension has been reduced away.
        return d - (1 if d > min(d1, d2) else 0)

    def reduce_maxmin(t, axis, which):
        axis = axis % t.dim()
        r = getattr(t, which)(dim=axis, keepdim=keepdim).values if keepdim \
            else getattr(t, which)(dim=axis).values
        return r

    if ord == "fro":
        return (A.abs() ** 2).sum(dim=list(dim), keepdim=keepdim).sqrt()
    if ord == "nuc":
        r = svdvals(A).sum(dim=-1)
        return r.unsqueeze(-1).unsqueeze(-1) if keepdim else r
    if ord == inf:  # maximum absolute row sum
        return reduce_maxmin(A.abs().sum(dim=d2), adj(d1), "max")
    if ord == -inf:
        return reduce_maxmin(A.abs().sum(dim=d2), adj(d1), "min")
    if ord == 1:  # maximum absolute column sum
        return reduce_maxmin(A.abs().sum(dim=d1), adj(d2), "max")
    if ord == -1:
        return reduce_maxmin(A.abs().sum(dim=d1), adj(d2), "min")
    if ord == 2 or ord == -2:  # spectral / smallest singular value
        s = svdvals(A)
        r = reduce_maxmin(s, -1, "max" if ord == 2 else "min")
        return r.unsqueeze(-1).unsqueeze(-1) if keepdim else r
    raise RuntimeError(f"linalg.matrix_norm: invalid ord {ord!r}")


def norm(input, ord=None, dim=None, keepdim=False):
    """norm(input, ord=None, dim=None, keepdim=False) -> Tensor"""
    if dim is None or isinstance(dim, int) or len(dim) == 1:
        d = dim if dim is not None else None
        if isinstance(dim, int):
            d = (dim,)
        o = ord if ord is not None else 2
        if isinstance(o, str):
            raise RuntimeError(
                "linalg.norm: metadata ordering got a string ord with a single dimension")
        return vector_norm(input, ord=o, dim=d, keepdim=keepdim)
    o = ord if ord is not None else "fro"
    return matrix_norm(input, ord=o, dim=dim, keepdim=keepdim)


def matrix_rank(A, *, atol=None, rtol=None, hermitian=False):
    """matrix_rank(A, *, atol=None, rtol=None, hermitian=False) -> Tensor"""
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
        return cond(A)
    if p in ("fro", "nuc", float("inf"), -float("inf"), 1, -1):
        return matrix_norm(A, ord=p) * matrix_norm(inv(A), ord=p)
    raise RuntimeError(f"linalg.cond: p={p!r} is not supported")
