"""Result types and shared dtype guards for the linear algebra namespace."""
from collections import namedtuple
import operator

import tensorplay

__all__ = [
    "EigResult",
    "EighResult",
    "LinAlgError",
    "LstsqResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
]

SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
QRResult = namedtuple("QRResult", ["Q", "R"])
LstsqResult = namedtuple(
    "LstsqResult", ["solution", "residuals", "rank", "singular_values"]
)
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])


class LinAlgError(RuntimeError):
    """Raised when a decomposition or solve fails on a numerically invalid input."""


def check_floating(A, name):
    """Rejects dtypes the decomposition kernels do not cover."""
    if A.dtype not in (
        tensorplay.float32,
        tensorplay.float64,
        tensorplay.complex64,
        tensorplay.complex128,
    ):
        raise NotImplementedError(
            f"linalg.{name}: only float32/float64/complex64/complex128 tensors "
            "are implemented; "
            f"got {A.dtype}")


def eps_of(dtype):
    """Machine epsilon used by the rank/pseudo-inverse cutoffs."""
    return 1.1920929e-07 if dtype in (tensorplay.float32, tensorplay.complex64) \
        else 2.220446049250313e-16


def as_index(value, name):
    """Converts an integer-like argument without truncating non-integers."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}") from exc
