"""Linear-algebra helpers for internal use, plus the removed legacy faces.

The removed functions raise with the canonical migration text so callers get
actionable errors instead of an import failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorplay import Tensor


def is_sparse(A: Tensor) -> bool:
    """Check if tensor A is a sparse COO tensor.

    All other sparse storage formats (CSR, CSC, etc...) will return False.
    """
    if isinstance(A, Tensor):
        return A.layout == _layout_coo()
    raise TypeError(f"expected Tensor but got {type(A)}")


def _layout_coo():
    from tensorplay import sparse_coo
    return sparse_coo


def get_floating_dtype(A: Tensor):
    """Return the floating point dtype of tensor A.

    Integer types map to float32.
    """
    from tensorplay import bfloat16, float16, float32, float64
    dtype = A.dtype
    if dtype in (float16, float32, float64, bfloat16):
        return dtype
    return float32


def matmul(A, B: Tensor) -> Tensor:
    """Multiply two matrices.

    If A is None, return B. A can be sparse or dense. B is always
    dense.
    """
    from tensorplay import sparse
    if A is None:
        return B
    if is_sparse(A):
        return sparse.mm(A, B)
    from tensorplay import matmul as _matmul
    return _matmul(A, B)


def bform(X: Tensor, A, Y: Tensor) -> Tensor:
    """Return bilinear form of matrices: :math:`X^T A Y`."""
    return matmul(X.mT, matmul(A, Y))


def qform(A, S: Tensor) -> Tensor:
    """Return quadratic form :math:`S^T A S`."""
    return bform(S, A, S)


def basis(A: Tensor) -> Tensor:
    """Return orthogonal basis of A columns."""
    from tensorplay import linalg
    return linalg.qr(A).Q


def symeig(A: Tensor, largest=None) -> tuple[Tensor, Tensor]:
    """Return eigenpairs of A with specified ordering."""
    from tensorplay import linalg
    if largest is None:
        largest = False
    E, Z = linalg.eigh(A, UPLO="U")
    # assuming that E is ordered
    if largest:
        E = _flip(E)
        Z = _flip(Z)
    return E, Z


def _flip(t: Tensor) -> Tensor:
    from tensorplay import flip
    return flip(t, dims=(-1,))


def matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed.\n"
        "Please use the `tensorplay.linalg.matrix_rank` function instead. "
        "The parameter 'symmetric' was renamed in "
        "`tensorplay.linalg.matrix_rank()` to 'hermitian'."
    )


def solve(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`tensorplay.solve` is deprecated in favor of "
        "`tensorplay.linalg.solve`. "
        "`tensorplay.linalg.solve` has its arguments reversed and does not "
        "return the LU factorization.\n\n"
        "To get the LU factorization see `tensorplay.lu`, which can be used "
        "with `tensorplay.lu_solve` or `tensorplay.lu_unpack`.\n"
        "X = tensorplay.solve(B, A).solution "
        "should be replaced with:\n"
        "X = tensorplay.linalg.solve(A, B)"
    )


def lstsq(input: Tensor, A: Tensor, *, out=None) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`tensorplay.lstsq` is deprecated in favor of "
        "`tensorplay.linalg.lstsq`.\n"
        "`tensorplay.linalg.lstsq` has reversed arguments and does not return "
        "the QR decomposition in the returned tuple (although it returns "
        "other information about the problem).\n\n"
        "To get the QR decomposition consider using "
        "`tensorplay.linalg.qr`.\n\n"
        "The returned solution in `tensorplay.lstsq` stored the residuals of "
        "the solution in the last m - n columns of the returned value "
        "whenever m > n. In tensorplay.linalg.lstsq, "
        "the residuals are in the field 'residuals' of the returned named "
        "tuple.\n\n"
        "The unpacking of the solution, as in\n"
        "X, _ = tensorplay.lstsq(B, A).solution[:A.size(1)]\n"
        "should be replaced with:\n"
        "X = tensorplay.linalg.lstsq(A, B).solution"
    )


def _symeig(
    input,
    eigenvectors=False,
    upper=True,
    *,
    out=None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "The default behavior has changed from using the upper triangular "
        "portion of the matrix by default to using the lower triangular "
        "portion.\n\n"
        "L, _ = tensorplay.symeig(A, upper=upper) "
        "should be replaced with:\n"
        "L = tensorplay.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\n"
        "and\n\n"
        "L, V = tensorplay.symeig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L, V = tensorplay.linalg.eigh(A, UPLO='U' if upper else 'L')"
    )


def eig(
    self: Tensor,
    eigenvectors: bool = False,
    *,
    e=None,
    v=None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`tensorplay.linalg.eig` returns complex tensors of dtype `cfloat` or "
        "`cdouble` rather than real tensors mimicking complex tensors.\n\n"
        "L, _ = tensorplay.eig(A) "
        "should be replaced with:\n"
        "L_complex = tensorplay.linalg.eigvals(A)\n\n"
        "and\n\n"
        "L, V = tensorplay.eig(A, eigenvectors=True) "
        "should be replaced with:\n"
        "L_complex, V_complex = tensorplay.linalg.eig(A)"
    )
