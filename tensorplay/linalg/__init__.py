"""

LAPACK/cuSOLVER-backed decompositions bind to native kernels in the CPU and
CUDA backends. The public functions expose differentiable TensorPlay
primitives.

Current scope: real float32/float64. Complex inputs raise NotImplementedError.
"""

from collections import namedtuple

import tensorplay
from tensorplay import _C
from tensorplay._C import (
    linalg_cholesky,
    linalg_cholesky_ex,
    linalg_det,
    linalg_diagonal as diagonal,
    linalg_eig,
    linalg_eigvals,
    linalg_eigh,
    linalg_eigvalsh,
    linalg_cross,
    linalg_householder_product as householder_product,
    linalg_inv_ex,
    linalg_ldl_factor as ldl_factor,
    linalg_ldl_factor_ex as ldl_factor_ex,
    linalg_ldl_solve as ldl_solve,
    linalg_lu as lu,
    linalg_lu_factor as lu_factor,
    linalg_lu_factor_ex as lu_factor_ex,
    linalg_lu_solve as lu_solve,
    linalg_solve_triangular as solve_triangular,
)

__all__ = [
    "LinAlgError", "cross", "cholesky", "cholesky_ex", "inv", "solve_ex",
    "inv_ex", "det", "slogdet", "eig", "eigvals", "eigh", "eigvalsh",
    "householder_product", "ldl_factor", "ldl_factor_ex", "ldl_solve",
    "lstsq", "matrix_power", "matrix_rank", "norm", "vector_norm",
    "matrix_norm", "matmul", "diagonal", "multi_dot", "svd", "svdvals",
    "cond", "pinv", "matrix_exp", "matrix_sqrth", "solve",
    "solve_triangular", "lu_factor", "lu_factor_ex", "lu_solve", "lu",
    "tensorinv", "tensorsolve", "qr", "polar", "vander", "vecdot",
]

SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
QRResult = namedtuple("QRResult", ["Q", "R"])
LstsqResult = namedtuple("LstsqResult", ["solution", "residuals", "rank", "singular_values"])
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])


class LinAlgError(RuntimeError):
    ""

def _check_floating(A, name):
    if A.dtype not in (tensorplay.float32, tensorplay.float64):
        raise NotImplementedError(
            f"linalg.{name}: only float32/float64 tensors are implemented; "
            f"got {A.dtype}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def cholesky(A, *, upper=False):
    """cholesky(A, *, upper=False) -> Tensor"""
    return linalg_cholesky(A, upper=upper)


def cholesky_ex(A, *, upper=False, check_errors=False):
    """cholesky_ex(A, *, upper=False, check_errors=False) -> (Tensor, Tensor)"""
    return linalg_cholesky_ex(A, upper=upper, check_errors=check_errors)


def inv(A):
    """inv(A) -> Tensor"""
    return _C.linalg_inv(A)


def inv_ex(A, *, check_errors=False):
    """inv_ex(A, *, check_errors=False) -> (Tensor, Tensor)"""
    return linalg_inv_ex(A, check_errors=check_errors)


def det(A):
    """det(A) -> Tensor"""
    return linalg_det(A)


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


def lstsq(A, B, rcond=None, *, driver=None):
    """lstsq(A, B, rcond=None, *, driver=None) -> LstsqResult(solution, residuals, rank, singular_values)"""
    solution, residuals, rank, coefficients = _C.linalg_lstsq(A, B, rcond, driver=driver)
    return LstsqResult(solution, residuals, rank, coefficients)


def matrix_exp(A):
    """matrix_exp(A) -> Tensor

    Square matrix exponential via Pade approximation with scaling and
    """
    _check_floating(A, "matrix_exp")
    n = A.shape[-1]
    batch = list(A.shape[:-2])
    dtype = A.dtype
    eye = tensorplay.zeros(batch + [n, n], dtype=dtype)
    eye += tensorplay.eye(n, dtype=dtype)
    theta13 = 5.371920351148152
    b = [64764752532480000., 32382376266240000., 7771770303897600.,
         1187353796428800., 129060195264000., 10559470521600.,
         670442572800., 33522128640., 1323241920., 40840800., 960960.,
         16380., 182., 1.]

    norm = _linalg_matrix_norm_1(A)
    s = 0
    if norm > theta13:
        s = max(1, int(tensorplay.ceil(tensorplay.log2(norm / theta13)).item()))
        A_scaled = A / (2.0 ** s)
    else:
        A_scaled = A

    A2 = A_scaled @ A_scaled
    A4 = A2 @ A2
    A6 = A2 @ A4
    U = A_scaled @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
                    + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * eye)
    V = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) \
        + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * eye
    result = solve(-U + V, U + V)
    for _ in range(s):
        result = result @ result
    return result


def matrix_sqrth(A):
    """matrix_sqrth(A) -> Tensor

    Matrix square root via the Denman-Beavers fixed-point iteration
    (converges for matrices with no eigenvalues on the closed negative real axis).
    """
    _check_floating(A, "matrix_sqrth")
    n = A.shape[-1]
    batch = list(A.shape[:-2])
    dtype = A.dtype
    eye = tensorplay.eye(n, dtype=dtype)
    Y = A
    Z = eye.expand(batch + [n, n]).contiguous() * 1.0
    eps = 1e-12
    for _ in range(100):
        Y_next = 0.5 * (Y + _inv(Z))
        Z_next = 0.5 * (Z + _inv(Y))
        err = float((_abs(Y_next - Y)).max().item())
        Y, Z = Y_next, Z_next
        if err < eps * max(1.0, float(_abs(Y).max().item())):
            break
    return Y


# ---------------------------------------------------------------------------
# Composite ops over differentiable primitives (autograd works through these).
# ---------------------------------------------------------------------------

_abs = tensorplay.abs


def cross(input, other, *, dim=-1):
    """cross(input, other, *, dim=-1) -> Tensor"""
    return linalg_cross(input, other, dim=dim)


def vecdot(x, y, *, dim=-1):
    """vecdot(x, y, *, dim=-1) -> Tensor"""
    return (x * y).sum(dim=dim)


def matmul(input, other):
    """matmul(input, other) -> Tensor"""
    return input @ other


def vector_norm(x, ord=2, dim=None, keepdim=False):
    """vector_norm(x, ord=2, dim=None, keepdim=False) -> Tensor"""
    inf = float("inf")
    if ord == 0:
        result = (x != 0).to(x.dtype).sum(dim=dim, keepdim=keepdim)
    elif ord == inf:
        result = x.abs().max(dim=dim, keepdim=keepdim).values
    elif ord == -inf:
        result = x.abs().min(dim=dim, keepdim=keepdim).values
    else:
        result = x.abs().pow(ord).sum(dim=dim, keepdim=keepdim).pow(1.0 / ord)
    return result


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False):
    """matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False) -> Tensor"""
    inf = float("inf")
    ndim = A.dim()
    d1 = dim[0] % ndim
    d2 = dim[1] % ndim
    adj = lambda d: d - (1 if d > min(d1, d2) else 0)

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


def _powsum(x, ord, dim=None, keepdim=False):
    return x.abs().pow(ord).sum(dim=dim, keepdim=keepdim)


def matrix_power(A, n):
    """matrix_power(A, n) -> Tensor"""
    _check_floating(A, "matrix_power")
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise RuntimeError("linalg.matrix_power: A must be batches of square matrices")
    n = int(n)
    if n == 0:
        eye = tensorplay.eye(A.shape[-1], dtype=A.dtype)
        return eye.expand(A.shape).contiguous()
    invert = n < 0
    if invert:
        A = inv(A)
        n = -n
    result = None
    base = A
    while n > 0:
        if n & 1:
            result = base if result is None else result @ base
        n >>= 1
        if n:
            base = base @ base
    return result


def matrix_rank(A, *, atol=None, rtol=None, hermitian=False):
    """matrix_rank(A, *, atol=None, rtol=None, hermitian=False) -> Tensor"""
    S = eigvalsh(A) if hermitian else svdvals(A)
    max_S = S.max(dim=-1).values
    eps = 1.1920929e-07 if A.dtype == tensorplay.float32 else 2.220446049250313e-16
    rtol_val = eps * max(A.shape[-2], A.shape[-1]) if rtol is None else rtol
    tol = max(atol if atol is not None else 0.0, 0.0) + rtol_val * max_S
    return (S > tol).to(S.dtype).sum(dim=-1)


def multi_dot(tensors):
    """multi_dot(tensors) -> Tensor"""
    if len(tensors) < 2:
        raise RuntimeError("linalg.multi_dot: expected at least two tensors")
    shapes = [list(t.shape) for t in tensors]
    n = len(shapes)
    dims = [shapes[0][-2]] + [shapes[i][-1] for i in range(n)]
    m = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(1, n - length + 1):
            j = i + length - 1
            m[i][j] = float("inf")
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if cost < m[i][j]:
                    m[i][j] = cost
                    split[i][j] = k

    def build(i, j):
        if i == j:
            return tensors[i - 1]
        return build(i, split[i][j]) @ build(split[i][j] + 1, j)

    return build(1, n - 1)


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


def pinv(A, *, atol=None, rtol=None, hermitian=False):
    """pinv(A, *, atol=None, rtol=None, hermitian=False) -> Tensor"""
    _check_floating(A, "pinv")
    U, S, Vh = svd(A, full_matrices=False)
    if hermitian:
        pass  # real path identical
    eps = 1.1920929e-07 if A.dtype == tensorplay.float32 else 2.220446049250313e-16
    max_mn = max(A.shape[-2], A.shape[-1])
    atol_val = 0.0 if atol is None else float(atol)
    rtol_val = (eps * max_mn) if rtol is None else float(rtol)
    cutoff = atol_val + rtol_val * S.max(dim=-1, keepdim=True).values
    S_inv = tensorplay.where(S > cutoff, 1.0 / S.clamp_min(1e-300), 0.0)
    return Vh.transpose(-2, -1) @ (S_inv.unsqueeze(-1) * U.transpose(-2, -1))


def polar(A):
    """polar(A) -> (Tensor Q, Tensor R) with A = Q R"""
    U, S, Vh = svd(A, full_matrices=False)
    Q = U @ Vh
    R = Vh.transpose(-2, -1) @ (S.unsqueeze(-1) * Vh)
    return Q, R


def vander(x, N=None):
    """vander(x, N=None) -> Tensor"""
    N = x.shape[-1] if N is None else int(N)
    cols = [x.pow(N - 1 - j) for j in range(N)]
    return tensorplay.stack(cols, dim=-1)


def tensorinv(A, ind=2):
    """tensorinv(A, ind=2) -> Tensor"""
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


def _linalg_matrix_norm_1(A):
    return A.abs().sum(dim=-2).max()


def _inv(x):
    return inv(x)
