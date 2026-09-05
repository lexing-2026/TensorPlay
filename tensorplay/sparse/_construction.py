"""Sparse tensor construction and layout conversion."""
import operator

import tensorplay
from tensorplay._C import sparse_coo_tensor as _sparse_coo_tensor_native

from ._invariants import _is_enabled, _validate_coo, _validate_csr, validate

__all__ = [
    "coalesce",
    "sparse_coo_tensor",
    "sparse_csr_tensor",
    "sparse_mask",
    "spdiags",
    "to_dense",
    "to_sparse",
    "to_sparse_csr",
]


def _as_index(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _normalize_size(size, name="size"):
    if isinstance(size, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        values = list(size)
    except TypeError:
        values = [size]
    result = [_as_index(value, f"{name}[{index}]") for index, value in enumerate(values)]
    if any(value < 0 for value in result):
        raise ValueError(f"{name} entries must be non-negative")
    return result


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
    if not isinstance(indices, tensorplay.Tensor) or not isinstance(
        values, tensorplay.Tensor
    ):
        raise TypeError("indices and values must be Tensor objects")
    if indices.dim() != 2:
        raise ValueError("indices must be a 2-D tensor")
    if values.dim() == 0:
        raise ValueError("values must have at least one dimension")
    if size is None:
        if _as_index(indices.shape[1], "indices.shape[1]") == 0:
            raise RuntimeError(
                "cannot infer size from an empty `indices`; pass size= explicitly"
            )
        inferred = [
            _as_index(indices[d].max().item(), f"indices[{d}].max()") + 1
            for d in range(_as_index(indices.shape[0], "indices.shape[0]"))
        ]
        size = _normalize_size(inferred + list(values.shape[1:]))
    else:
        size = _normalize_size(size)
    if check_invariants if check_invariants is not None else _is_enabled():
        _validate_coo(indices, values, size)
    return _sparse_coo_tensor_native(
        indices, values, size, is_coalesced=is_coalesced
    )


def sparse_csr_tensor(
    crow_indices,
    col_indices,
    values,
    size=None,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    check_invariants=None,
):
    """Builds a two-dimensional sparse CSR tensor from compressed buffers."""
    if not all(
        isinstance(value, tensorplay.Tensor)
        for value in (crow_indices, col_indices, values)
    ):
        raise TypeError("crow_indices, col_indices, and values must be Tensor objects")
    if size is not None:
        size = _normalize_size(size)
        if len(size) != 2:
            raise ValueError("size must contain exactly two dimensions")
    if layout is not None:
        layout = _as_index(layout, "layout")
        if layout != int(tensorplay.sparse_csr):
            raise ValueError("sparse_csr_tensor() requires the sparse_csr layout")
    if check_invariants if check_invariants is not None else _is_enabled():
        if size is None:
            raise ValueError("size is required when invariant checking is enabled")
        _validate_csr(crow_indices, col_indices, values, size)
    if size is None:
        result = tensorplay._C.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    else:
        result = tensorplay._C.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    if check_invariants if check_invariants is not None else _is_enabled():
        validate(result)
    return result


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
    shape = _normalize_size(shape)
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    if layout is not None:
        layout = _as_index(layout, "layout")
        if layout not in (int(tensorplay.sparse_coo), int(tensorplay.sparse_csr)):
            raise ValueError("layout must be sparse_coo or sparse_csr")
    return tensorplay.spdiags(diagonals, offsets, shape, layout=layout)


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


def to_sparse(input, sparse_dim=None, *, check_invariants=None):
    """Converts a dense tensor into a coalesced sparse COO tensor.

    ``check_invariants`` overrides the global flag from
    :class:`check_sparse_tensor_invariants` for this conversion; when on, the
    resulting tensor is validated against the COO invariants.
    """
    if sparse_dim is None:
        out = tensorplay.to_sparse(input)
    else:
        out = tensorplay.to_sparse(input, _as_index(sparse_dim, "sparse_dim"))
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
