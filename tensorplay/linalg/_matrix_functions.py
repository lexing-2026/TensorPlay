"""Matrix-valued functions and the products built on top of them."""
import operator

import tensorplay
from tensorplay._C import (
    linalg_cross,
    linalg_diagonal as diagonal,
)

from ._common import as_index, check_floating
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
    """vecdot(x, y, *, dim=-1) -> Tensor

    Dot product along `dim` with the first argument conjugated for complex
    inputs.
    """
    dim_value = as_index(dim, "linalg.vecdot dim")
    x_dim = dim_value + x.dim() if dim_value < 0 else dim_value
    y_dim = dim_value + y.dim() if dim_value < 0 else dim_value
    if not 0 <= x_dim < x.dim() or not 0 <= y_dim < y.dim():
        raise IndexError("linalg.vecdot: dimension out of range")
    if x.shape[x_dim] != y.shape[y_dim]:
        raise RuntimeError(
            "linalg.vecdot: vector dimensions must have the same length")
    from tensorplay import functional as _F
    if x.dtype.is_complex:
        x = _F.conj_physical(x)
    return (x * y).sum(dim=dim_value)


def vdot(self, other):
    """vdot(self, other) -> Tensor

    Conjugating dot product over 1-D operands: sum(conj(self) * other).
    """
    from tensorplay import functional as _F
    if self.dim() != 1 or other.dim() != 1:
        raise RuntimeError(
            f"vdot: Expected both inputs to be 1-dimensional, but got "
            f"{self.dim()}D and {other.dim()}D tensors")
    if self.shape[0] != other.shape[0]:
        raise RuntimeError(
            f"vdot: sizes don't match, got {self.shape[0]} and {other.shape[0]}")
    a = _F.conj_physical(self) if self.dtype.is_complex else self
    return (a * other).sum()


def matmul(input, other):
    """matmul(input, other) -> Tensor"""
    return input @ other


def vander(x, N=None):
    """vander(x, N=None) -> Tensor"""
    if x.dim() != 1:
        raise ValueError(f"linalg.vander: x must be 1-dimensional, got {x.dim()}D")
    N = x.numel() if N is None else as_index(N, "linalg.vander N")
    if N < 0:
        raise ValueError(f"linalg.vander: N must be non-negative, got {N}")
    if N == 0:
        return tensorplay.empty([x.numel(), 0], dtype=x.dtype, device=x.device)
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
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("linalg.matrix_exp: input must contain square matrices")
    n = A.shape[-1]
    if n == 0:
        return A.clone()
    batch = list(A.shape[:-2])
    dtype = A.dtype
    eye = tensorplay.eye(n, dtype=dtype, device=A.device)
    if batch:
        eye = eye.expand(batch + [n, n]).contiguous()
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
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("linalg.matrix_sqrth: input must contain square matrices")
    n = A.shape[-1]
    if n == 0:
        return A.clone()
    batch = list(A.shape[:-2])
    dtype = A.dtype
    eye = tensorplay.eye(n, dtype=dtype, device=A.device)
    Y = A
    if batch:
        eye = eye.expand(batch + [n, n]).contiguous()
    Z = eye * 1.0
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
    n = as_index(n, "linalg.matrix_power exponent")
    if n == 0:
        eye = tensorplay.eye(A.shape[-1], dtype=A.dtype, device=A.device)
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
    tensors = list(tensors)
    if len(tensors) < 2:
        raise RuntimeError("linalg.multi_dot: expected at least two tensors")
    n = len(tensors)
    shapes = [tuple(t.shape) for t in tensors]
    for index, shape in enumerate(shapes):
        if len(shape) not in (1, 2):
            raise ValueError(
                f"linalg.multi_dot: tensor {index} must be 1D or 2D, got {len(shape)}D"
            )
        if len(shape) == 1 and index not in (0, n - 1):
            raise ValueError(
                "linalg.multi_dot: only the first or last tensor may be 1D"
            )

    left = [1 if len(shape) == 1 else shape[-2]
            for index, shape in enumerate(shapes)]
    right = [1 if len(shape) == 1 and index == n - 1 else shape[-1]
             for index, shape in enumerate(shapes)]
    for index in range(n - 1):
        if right[index] != left[index + 1]:
            raise ValueError(
                f"linalg.multi_dot: shapes {shapes[index]} and "
                f"{shapes[index + 1]} are incompatible"
            )

    dimensions = [left[0]] + right
    costs = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for start in range(n - length + 1):
            end = start + length - 1
            best = float("inf")
            best_split = start
            for middle in range(start, end):
                cost = (
                    costs[start][middle]
                    + costs[middle + 1][end]
                    + dimensions[start]
                    * dimensions[middle + 1]
                    * dimensions[end + 1]
                )
                if cost < best:
                    best = cost
                    best_split = middle
            costs[start][end] = best
            split[start][end] = best_split

    def build(start, end):
        if start == end:
            return tensors[start]
        middle = split[start][end]
        return build(start, middle) @ build(middle + 1, end)

    return build(0, n - 1)
