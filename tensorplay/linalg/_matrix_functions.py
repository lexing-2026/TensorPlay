"""Matrix-valued functions and the products built on top of them."""
import tensorplay
from tensorplay._C import (
    linalg_cross,
    linalg_diagonal as diagonal,
)

from ._common import check_floating
from ._solve import inv, solve

__all__ = [
    "cross",
    "diagonal",
    "matmul",
    "matrix_exp",
    "matrix_power",
    "matrix_sqrth",
    "multi_dot",
    "vander",
    "vecdot",
]

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


def vander(x, N=None):
    """vander(x, N=None) -> Tensor"""
    N = x.shape[-1] if N is None else int(N)
    cols = [x.pow(N - 1 - j) for j in range(N)]
    return tensorplay.stack(cols, dim=-1)


def _matrix_norm_1(A):
    """Maximum absolute column sum, the norm driving the exponential's scaling."""
    return A.abs().sum(dim=-2).max()


def matrix_exp(A):
    """matrix_exp(A) -> Tensor

    Square matrix exponential via the degree-13 Pade approximant with
    scaling and squaring: ``A`` is halved until its 1-norm falls under the
    approximant's accuracy threshold, then the result is squared back.
    """
    check_floating(A, "matrix_exp")
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

    norm = _matrix_norm_1(A)
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
    check_floating(A, "matrix_sqrth")
    n = A.shape[-1]
    batch = list(A.shape[:-2])
    dtype = A.dtype
    eye = tensorplay.eye(n, dtype=dtype)
    Y = A
    Z = eye.expand(batch + [n, n]).contiguous() * 1.0
    eps = 1e-12
    for _ in range(100):
        Y_next = 0.5 * (Y + inv(Z))
        Z_next = 0.5 * (Z + inv(Y))
        err = float((_abs(Y_next - Y)).max().item())
        Y, Z = Y_next, Z_next
        if err < eps * max(1.0, float(_abs(Y).max().item())):
            break
    return Y


def matrix_power(A, n):
    """matrix_power(A, n) -> Tensor"""
    check_floating(A, "matrix_power")
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


def multi_dot(tensors):
    """multi_dot(tensors) -> Tensor

    Chained matrix product evaluated in the parenthesization that minimizes
    the scalar multiplication count (matrix-chain dynamic program).
    """
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
