"""Create typed tensor views into a preallocated byte pool."""

from __future__ import annotations

from typing import Any

import tensorplay as tp


def alloc_from_pool(
    pool: tp.Tensor,
    offset_bytes: int,
    dtype: tp.dtype,
    shape: Any,
    stride: Any,
) -> tp.Tensor:
    if pool.storage_offset() != 0:
        raise AssertionError("pool tensor must have a zero storage offset")
    return tp.empty(0, dtype=dtype, device=pool.device).set_(
        pool.untyped_storage(),
        int(offset_bytes) // dtype.itemsize,
        shape,
        stride,
    )


__all__ = ["alloc_from_pool"]
