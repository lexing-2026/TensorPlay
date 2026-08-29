"""Sparse COO and CSR operations.

Provides the COO/CSR operations under a dedicated namespace so user code can
use the same tensor objects for sparse construction and conversion.
"""

import tensorplay
from tensorplay._C import sparse_coo_tensor as sparse_coo_tensor

__all__ = [
    "coalesce",
    "sparse_coo_tensor",
    "sparse_mask",
    "spdiags",
    "add",
    "mul",
    "mm",
    "sum",
    "to_dense",
    "to_sparse",
    "to_sparse_csr",
]


def coalesce(input):
    """Returns a coalesced copy of the sparse COO tensor ``input``."""
    return input.coalesce()


def sparse_mask(input, mask):
    """Returns a new sparse tensor with values of ``input`` at ``mask``'s indices."""
    return input.sparse_mask(mask)


def add(self, other):
    """Adds two sparse COO tensors with the same shape and dtype."""
    return tensorplay.sparse_add(self, other)


def mul(self, other):
    """Multiplies two sparse COO tensors elementwise on shared coordinates."""
    return tensorplay.sparse_mul(self, other)


def mm(sparse, dense):
    """Performs a matrix multiplication of a 2-D sparse COO/CSR tensor with a
"""
    return tensorplay.sparse_mm(sparse, dense)


def spdiags(diagonals, offsets, shape, layout=None):
    """Construct a sparse tensor from diagonals.

    Args:
        diagonals: matrix of shape ``(len(offsets), L)`` (or a single vector);
            row ``j`` holds the values of diagonal ``offsets[j]``, read
            starting at column ``max(offset, 0)``.
        offsets: int64 sequence of distinct diagonal offsets (0 = main,
            positive = above, negative = below).
        shape: 2-element ``(M, N)`` output size.
        layout: output layout tag — ``tensorplay.sparse_coo`` (default) or
            ``tensorplay.sparse_csr``.
    """
    if layout is not None and isinstance(layout, int):
        layout = int(layout)
    return tensorplay.spdiags(diagonals, offsets, list(shape), layout=layout)


def sum(input, dim=None, dtype=None):
    """Sum of ``input``'s values over ``dim``.

    result is a dense 0-dim tensor; reducing every sparse dim yields a dense
    tensor; a partial reduction returns a coalesced sparse COO tensor over
    the remaining dims with duplicate coordinates folded.  ``dtype`` converts
    the input first, acting as the accumulation type.
    """
    return tensorplay.sparse_sum(input, dim=dim, dtype=dtype)


def to_dense(input):
    """Materializes a sparse COO/CSR tensor into a dense tensor."""
    return tensorplay.to_dense(input)


def to_sparse(input):
    """Converts a dense tensor into a coalesced sparse COO tensor."""
    return tensorplay.to_sparse(input)


def to_sparse_csr(input):
    """Converts a 2-D dense tensor into a sparse CSR tensor."""
    return tensorplay.to_sparse_csr(input)
