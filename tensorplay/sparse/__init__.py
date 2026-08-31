"""Sparse COO, CSR, and semi-structured tensors.

The package keeps the sparse surface under a dedicated namespace so user
code can use the same tensor objects for sparse construction, arithmetic and
conversion.  It is split so each concern can grow on its own:

* :mod:`~tensorplay.sparse._construction` -- constructors and layout conversion.
* :mod:`~tensorplay.sparse._ops` -- arithmetic, reductions, matmul,
  normalized exponentials and the linear solver.
* :mod:`~tensorplay.sparse._invariants` -- the invariant-checking flag and the
  per-layout validators the constructors consult.
* :mod:`~tensorplay.sparse._gradcheck` -- gradient checking through sparse
  inputs and outputs.
"""
from tensorplay import Tensor

from ._construction import (
    coalesce,
    sparse_coo_tensor,
    sparse_mask,
    spdiags,
    to_dense,
    to_sparse,
    to_sparse_csr,
)
from ._gradcheck import as_sparse_gradcheck
from ._invariants import check_sparse_tensor_invariants
from ._ops import add, addmm, log_softmax, mm, mul, softmax, solve, sum
from .semi_structured import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUTLASS,
    SparseSemiStructuredTensorCUSPARSELT,
    to_sparse_semi_structured,
)

__all__ = [
    "Tensor",
    "add",
    "addmm",
    "as_sparse_gradcheck",
    "check_sparse_tensor_invariants",
    "coalesce",
    "log_softmax",
    "mm",
    "mul",
    "softmax",
    "solve",
    "sparse_coo_tensor",
    "sparse_mask",
    "spdiags",
    "sum",
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_dense",
    "to_sparse",
    "to_sparse_csr",
    "to_sparse_semi_structured",
]
