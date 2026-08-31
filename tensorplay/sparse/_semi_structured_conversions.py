"""Packing helpers for semi-structured sparse matrices."""

from __future__ import annotations

from typing import Any

__all__ = [
    "_calculate_meta_reordering_scatter_offsets",
    "_compute_compressed_swizzled_bitmask",
    "_sparse_semi_structured_tile",
    "sparse_semi_structured_from_dense_cutlass",
    "sparse_semi_structured_to_dense_cutlass",
]


def _calculate_meta_reordering_scatter_offsets(
    m: int, meta_ncols: int, meta_dtype: Any, device: Any = None
) -> Any:
    import tensorplay

    if m < 0 or meta_ncols < 0:
        raise ValueError("matrix dimensions must be non-negative")
    offsets = tensorplay.arange(m * meta_ncols, dtype=tensorplay.int64)
    return offsets.reshape(m, meta_ncols)


def sparse_semi_structured_from_dense_cutlass(dense: Any) -> tuple[Any, Any]:
    from .semi_structured import SparseSemiStructuredTensorCUTLASS

    compressed = SparseSemiStructuredTensorCUTLASS.from_dense(dense)
    return compressed.packed, compressed.meta


def sparse_semi_structured_to_dense_cutlass(sparse: Any, meta_reordered: Any = None) -> Any:
    del meta_reordered
    if hasattr(sparse, "to_dense"):
        return sparse.to_dense()
    raise TypeError("expected a semi-structured sparse value")


def _sparse_semi_structured_tile(dense: Any) -> Any:
    if not hasattr(dense, "shape") or len(dense.shape) != 2:
        raise ValueError("tiling requires a two-dimensional dense matrix")
    rows, cols = tuple(int(value) for value in dense.shape)
    if rows % 16 or cols % 16:
        raise ValueError("matrix dimensions must be multiples of sixteen")
    return dense.reshape(rows // 16, 16, cols // 16, 16).transpose(1, 2)


def _compute_compressed_swizzled_bitmask(dense: Any) -> Any:
    if not hasattr(dense, "shape") or len(dense.shape) != 2:
        raise ValueError("bitmask construction requires a two-dimensional matrix")
    return dense != 0
