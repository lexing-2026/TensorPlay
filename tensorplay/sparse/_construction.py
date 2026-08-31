"""Sparse tensor construction and layout conversion."""
import tensorplay
from tensorplay._C import sparse_coo_tensor as _sparse_coo_tensor_native

from ._invariants import _is_enabled, _validate_coo, validate

__all__ = [
    "coalesce",
    "sparse_coo_tensor",
    "sparse_mask",
    "spdiags",
    "to_dense",
    "to_sparse",
    "to_sparse_csr",
]


def sparse_coo_tensor(
    indices, values, size=None, *, is_coalesced=False, check_invariants=None
):
    """Builds a sparse COO tensor from explicit coordinates and values.

    Args:
        indices: int64 tensor of shape ``(sparse_dim, nnz)``; column ``k``
            holds the coordinate of ``values[k]``.
        values: tensor of shape ``(nnz, *dense_shape)``.
        size: full tensor shape; inferred from the coordinate maxima when omitted.
        is_coalesced (bool): declares the coordinates already sorted and
            duplicate-free, letting later ops skip a coalescing pass.  Passing
            ``True`` for coordinates that do not satisfy it yields a tensor
            whose reductions are wrong, so leave it ``False`` when unsure.
        check_invariants (bool, optional): overrides the global flag from
            :class:`check_sparse_tensor_invariants` for this call.
    """
    if size is None:
        if int(indices.shape[1]) == 0:
            raise RuntimeError(
                "cannot infer size from an empty `indices`; pass size= explicitly"
            )
        inferred = [int(indices[d].max().item()) + 1 for d in range(int(indices.shape[0]))]
        size = inferred + list(values.shape[1:])
    size = [int(s) for s in size]
    if check_invariants if check_invariants is not None else _is_enabled():
        _validate_coo(indices, values, size)
    return _sparse_coo_tensor_native(
        indices, values, size, is_coalesced=is_coalesced
    )


def spdiags(diagonals, offsets, shape, layout=None):
    """Constructs a sparse tensor from diagonals.

    Args:
        diagonals: matrix of shape ``(len(offsets), L)`` (or a single vector);
            row ``j`` holds the values of diagonal ``offsets[j]``, read
            starting at column ``max(offset, 0)``.
        offsets: int64 sequence of distinct diagonal offsets (0 = main,
            positive = above, negative = below).
        shape: 2-element ``(M, N)`` output size.
        layout: output layout tag -- ``tensorplay.sparse_coo`` (default) or
            ``tensorplay.sparse_csr``.
    """
    if layout is not None and isinstance(layout, int):
        layout = int(layout)
    return tensorplay.spdiags(diagonals, offsets, list(shape), layout=layout)


def coalesce(input):
    """Returns a coalesced copy of the sparse COO tensor ``input``.

    Coordinates are sorted and duplicates are summed, so the result has one
    value per distinct coordinate.
    """
    return input.coalesce()


def sparse_mask(input, mask):
    """Returns a new sparse tensor with values of ``input`` at ``mask``'s indices."""
    return input.sparse_mask(mask)


def to_dense(input):
    """Materializes a sparse COO/CSR tensor into a dense tensor."""
    return tensorplay.to_dense(input)


def to_sparse(input, *, check_invariants=None):
    """Converts a dense tensor into a coalesced sparse COO tensor.

    ``check_invariants`` overrides the global flag from
    :class:`check_sparse_tensor_invariants` for this conversion; when on, the
    resulting tensor is validated against the COO invariants.
    """
    out = tensorplay.to_sparse(input)
    if check_invariants if check_invariants is not None else _is_enabled():
        validate(out)
    return out


def to_sparse_csr(input, *, check_invariants=None):
    """Converts a 2-D dense tensor into a sparse CSR tensor.

    ``check_invariants`` behaves as in :func:`to_sparse`, validating the
    compressed row pointer and column indices of the result.
    """
    out = tensorplay.to_sparse_csr(input)
    if check_invariants if check_invariants is not None else _is_enabled():
        validate(out)
    return out
