"""Sparse tensor construction and layout conversion."""
import operator

import tensorplay
from tensorplay._C import sparse_coo_tensor as _sparse_coo_tensor_native

from ._invariants import _is_enabled, _validate_coo, validate

__all__ = [
    "coalesce",
    "sparse_coo_tensor",
    "sparse_compressed_tensor",
    "sparse_csr_tensor",
    "sparse_csc_tensor",
    "sparse_bsr_tensor",
    "sparse_bsc_tensor",
    "sparse_mask",
    "spdiags",
    "to_dense",
    "to_sparse",
    "to_sparse_csr",
    "to_sparse_csc",
    "to_sparse_bsr",
    "to_sparse_bsc",
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


def _sparse_compressed_tensor(
    name,
    compressed_indices,
    plain_indices,
    values,
    size,
    *,
    expected_layout,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    check_invariants=None,
):
    """Shared Python entry point for the four layout-pinned constructors."""
    if not all(
        isinstance(value, tensorplay.Tensor)
        for value in (compressed_indices, plain_indices, values)
    ):
        raise TypeError(
            "compressed indices, plain indices, and values must be Tensor objects"
        )
    if size is not None:
        size = _normalize_size(size)
    if layout is not None:
        layout = _as_index(layout, "layout")
        if layout != int(expected_layout):
            raise ValueError(f"{name}() requires its matching sparse layout")
    checking = check_invariants if check_invariants is not None else _is_enabled()
    native = getattr(tensorplay._C, name)
    if size is None:
        result = native(
            compressed_indices,
            plain_indices,
            values,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    else:
        result = native(
            compressed_indices,
            plain_indices,
            values,
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    if checking:
        validate(result)
    return result


def sparse_compressed_tensor(
    compressed_indices,
    plain_indices,
    values,
    size=None,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    check_invariants=None,
):
    """Build a CSR/CSC/BSR/BSC tensor using an explicit ``layout`` tag."""
    return _sparse_compressed_tensor(
        "sparse_compressed_tensor",
        compressed_indices,
        plain_indices,
        values,
        size,
        expected_layout=layout if layout is not None else tensorplay.sparse_csr,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        check_invariants=check_invariants,
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
    """Build a sparse CSR tensor from compressed row buffers."""
    return _sparse_compressed_tensor(
        "sparse_csr_tensor", crow_indices, col_indices, values, size,
        expected_layout=tensorplay.sparse_csr, dtype=dtype, layout=layout,
        device=device, pin_memory=pin_memory,
        check_invariants=check_invariants,
    )


def sparse_csc_tensor(
    ccol_indices,
    row_indices,
    values,
    size=None,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    check_invariants=None,
):
    """Build a sparse CSC tensor from compressed column buffers."""
    return _sparse_compressed_tensor(
        "sparse_csc_tensor", ccol_indices, row_indices, values, size,
        expected_layout=tensorplay.sparse_csc, dtype=dtype, layout=layout,
        device=device, pin_memory=pin_memory,
        check_invariants=check_invariants,
    )


def sparse_bsr_tensor(
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
    """Build a block sparse BSR tensor from compressed row buffers."""
    return _sparse_compressed_tensor(
        "sparse_bsr_tensor", crow_indices, col_indices, values, size,
        expected_layout=tensorplay.sparse_bsr, dtype=dtype, layout=layout,
        device=device, pin_memory=pin_memory,
        check_invariants=check_invariants,
    )


def sparse_bsc_tensor(
    ccol_indices,
    row_indices,
    values,
    size=None,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    check_invariants=None,
):
    """Build a block sparse BSC tensor from compressed column buffers."""
    return _sparse_compressed_tensor(
        "sparse_bsc_tensor", ccol_indices, row_indices, values, size,
        expected_layout=tensorplay.sparse_bsc, dtype=dtype, layout=layout,
        device=device, pin_memory=pin_memory,
        check_invariants=check_invariants,
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
    """Materializes a sparse COO/CSR/CSC/BSR/BSC tensor into dense form."""
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
    """Converts a dense tensor into a sparse CSR tensor.

    ``check_invariants`` behaves as in :func:`to_sparse`, validating the
    compressed row pointer and column indices of the result.
    """
    out = tensorplay.to_sparse_csr(input)
    if check_invariants if check_invariants is not None else _is_enabled():
        validate(out)
    return out


def _to_sparse_compressed(input, name, *args, check_invariants=None, **kwargs):
    out = getattr(tensorplay, name)(input, *args, **kwargs)
    if check_invariants if check_invariants is not None else _is_enabled():
        validate(out)
    return out


def to_sparse_csc(input, dense_dim=None, *, check_invariants=None):
    """Converts a dense tensor into sparse CSC form."""
    kwargs = {} if dense_dim is None else {"dense_dim": _as_index(dense_dim, "dense_dim")}
    return _to_sparse_compressed(
        input, "to_sparse_csc", check_invariants=check_invariants, **kwargs
    )


def to_sparse_bsr(input, blocksize, dense_dim=None, *, check_invariants=None):
    """Converts a dense tensor into sparse BSR form."""
    kwargs = {} if dense_dim is None else {"dense_dim": _as_index(dense_dim, "dense_dim")}
    return _to_sparse_compressed(
        input,
        "to_sparse_bsr",
        _normalize_size(blocksize, "blocksize"),
        check_invariants=check_invariants,
        **kwargs,
    )


def to_sparse_bsc(input, blocksize, dense_dim=None, *, check_invariants=None):
    """Converts a dense tensor into sparse BSC form."""
    kwargs = {} if dense_dim is None else {"dense_dim": _as_index(dense_dim, "dense_dim")}
    return _to_sparse_compressed(
        input,
        "to_sparse_bsc",
        _normalize_size(blocksize, "blocksize"),
        check_invariants=check_invariants,
        **kwargs,
    )
