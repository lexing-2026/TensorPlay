"""Shape validation and dense fallbacks for block-sparse matrix operations."""

from __future__ import annotations

import itertools
import weakref
from typing import Any

from ._triton_ops_meta import _get_device_name, get_meta

__all__ = [
    "TensorAsKey",
    "as1Dbatch",
    "broadcast_batch_dims",
    "broadcast_batch_dims_bsr",
    "bsr_dense_addmm",
    "bsr_dense_addmm_meta",
    "bsr_scatter_mm",
    "bsr_scatter_mm_indices_data",
    "check",
    "check_blocksize",
    "check_bsr_layout",
    "check_device",
    "check_dtype",
    "check_mm_compatible_shapes",
    "grid_partitioner",
    "launch_kernel",
    "make_triton_contiguous",
    "multidim_slicer",
    "prepare_inputs",
    "ptr_stride_extractor",
    "scatter_mm",
    "scatter_mm_meta",
    "slicer",
    "tile_to_blocksize",
]


def check(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_bsr_layout(function_name: str, value: Any) -> None:
    layout = getattr(value, "layout", None)
    check(str(layout).lower() in {"bsr", "sparse_bsr", "2", "12"}, f"{function_name} requires BSR layout")


def check_device(function_name: str, value: Any, device: Any) -> None:
    check(getattr(value, "device", None) == device, f"{function_name} inputs must share a device")


def check_mm_compatible_shapes(function_name: str, left: Any, right: Any) -> None:
    check(getattr(left, "ndim", len(getattr(left, "shape", ()))) >= 2, f"{function_name} left input must be at least 2-D")
    check(getattr(right, "ndim", len(getattr(right, "shape", ()))) >= 2, f"{function_name} right input must be at least 2-D")
    check(left.shape[-1] == right.shape[-2], f"{function_name} matrix dimensions are incompatible")


def check_dtype(function_name: str, value: Any, dtype: Any, *additional_dtypes: Any) -> None:
    check(getattr(value, "dtype", None) in (dtype, *additional_dtypes), f"{function_name} received an unsupported dtype")


def check_blocksize(function_name: str, blocksize: tuple[int, int]) -> None:
    if len(blocksize) != 2:
        raise ValueError(f"{function_name} blocksize must have two entries")
    check(all(value >= 16 and value & (value - 1) == 0 for value in blocksize), f"{function_name} blocksize must be powers of two")


def make_triton_contiguous(value: Any) -> Any:
    strides = getattr(value, "stride", lambda: ())()
    return value.contiguous() if strides and min(strides) > 1 else value


def broadcast_batch_dims(function_name: str, *values: Any) -> tuple[int, ...]:
    shapes = [tuple(value.shape[:-2]) for value in values]
    result: list[int] = []
    for dimensions in zip(*map(lambda shape: (1,) * (max(map(len, shapes), default=0) - len(shape)) + shape, shapes)):
        non_one = {dimension for dimension in dimensions if dimension != 1}
        if len(non_one) > 1:
            raise ValueError(f"{function_name} batch dimensions are not broadcastable")
        result.append(next(iter(non_one), 1))
    return tuple(result)


def slicer(dim: int, slice_range: slice, *values: Any):
    for value in values:
        slices = [slice(None)] * value.ndim
        slices[dim] = slice_range
        yield value[tuple(slices)]


def multidim_slicer(dims: tuple[int | None, ...], slices: tuple[slice, ...], *values: Any):
    for value in values:
        index = [slice(None)] * value.ndim
        for dim, current_slice in zip(dims, slices):
            if dim is not None:
                index[dim] = current_slice
        yield value[tuple(index)]


def ptr_stride_extractor(*values: Any):
    for value in values:
        yield value
        yield from getattr(value, "stride", lambda: ())()


def grid_partitioner(full_grid: tuple[int, ...], grid_blocks: tuple[int, ...], tensor_dims_map: dict[Any, Any]):
    if not 0 <= len(full_grid) <= 3 or not 0 <= len(grid_blocks) <= 3:
        raise ValueError("grid dimensions must contain at most three entries")
    if len(full_grid) != len(grid_blocks):
        raise ValueError("full_grid and grid_blocks must have the same rank")
    ranges = [range(0, full, max(1, block)) for full, block in zip(full_grid, grid_blocks)]
    for point in itertools.product(*ranges):
        sizes = [min(full - start, block) for full, start, block in zip(full_grid, point, grid_blocks)]
        slices = tuple(slice(start, start + size) for start, size in zip(point, sizes))
        sliced_tensors = (
            next(multidim_slicer(tensor_dims, slices, tensor))
            for tensor, tensor_dims in tensor_dims_map.items()
        )
        yield tuple(reversed(sizes)), *sliced_tensors


def launch_kernel(kernel: Any, tensor_dims_map: dict[Any, Any], full_grid: tuple[int, ...], grid_blocks: tuple[int, ...] | None = None) -> None:
    blocks = grid_blocks or tuple(max(1, value) for value in full_grid)
    for grid, *values in grid_partitioner(full_grid, blocks, tensor_dims_map):
        kernel(grid, *values)


def prepare_inputs(bsr: Any, *dense_values: Any) -> tuple[Any, ...]:
    return (bsr.crow_indices(), bsr.col_indices(), make_triton_contiguous(bsr.values()), *(make_triton_contiguous(value) for value in dense_values))


def broadcast_batch_dims_bsr(function_name: str, bsr: Any, *values: Any) -> Any:
    del function_name, values
    return bsr


def tile_to_blocksize(value: Any, blocksize: tuple[int, int]) -> Any:
    *rest, rows, columns = value.shape
    if rows % blocksize[0] or columns % blocksize[1]:
        raise ValueError("tensor shape is not divisible by blocksize")
    return value.reshape(*rest, rows // blocksize[0], blocksize[0], columns // blocksize[1], blocksize[1]).transpose(-3, -2)


def as1Dbatch(value: Any) -> Any:
    while value.ndim < 3:
        value = value.unsqueeze(0)
    if value.ndim > 3:
        value = value.flatten(0, value.ndim - 3)
    return value


def scatter_mm(blocks: Any, others: Any, indices_data: Any, *, accumulators: Any = None) -> Any:
    if accumulators is None:
        return blocks @ others
    if not indices_data:
        return accumulators
    if indices_data[0] == "scatter_mm" and len(indices_data) >= 3:
        offsets, pairs = indices_data[1:3]
        for row in range(len(offsets) - 1):
            for index in range(int(offsets[row]), int(offsets[row + 1])):
                left, right = pairs[index]
                accumulators[row] = accumulators[row] + blocks[left] @ others[right]
        return accumulators
    return accumulators


def scatter_mm_meta(M: int, K: int, N: int, Ms: int, Ks: int, **kwargs: Any) -> dict[str, Any]:
    del K, kwargs
    if min(M, N, Ms, Ks) <= 0 or Ms > M:
        raise ValueError("matrix and block dimensions must be positive and compatible")
    return {"TILE_M": Ms, "TILE_N": min(N, 32), "GROUP_SIZE": 1, "num_stages": 1, "num_warps": 1, "SPLIT_N": max(1, N // max(Ms, 1))}


def bsr_dense_addmm_meta(M: int, K: int, N: int, Ms: int, Ks: int, **kwargs: Any) -> dict[str, Any]:
    return scatter_mm_meta(M, K, N, Ms, Ks, **kwargs)


class TensorAsKey:
    def __init__(self, value: Any) -> None:
        self._ref = weakref.ref(value)
        self.key = (
            getattr(value, "data_ptr", lambda: id(value))(),
            getattr(value, "storage_offset", lambda: 0)(),
            tuple(value.shape),
            tuple(getattr(value, "stride", lambda: ())()),
            getattr(value, "dtype", None),
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TensorAsKey) and self.key == other.key

    @property
    def value(self) -> Any:
        value = self._ref()
        if value is None:
            raise ReferenceError("the keyed tensor was released")
        return value


def bsr_scatter_mm_indices_data(bsr: Any, other: Any, indices_format: str = "bsr_strided_mm_compressed", **meta_input: Any) -> tuple[Any, ...]:
    check_mm_compatible_shapes("bsr_scatter_mm", bsr, other)
    M, K = bsr.shape[-2:]
    N = other.shape[-1]
    meta = scatter_mm_meta(M, K, N, *bsr.values().shape[-2:], **meta_input)
    return (indices_format, meta)


def bsr_scatter_mm(bsr: Any, other: Any, indices_data: Any = None, out: Any = None) -> Any:
    del indices_data
    dense = bsr.to_dense() if hasattr(bsr, "to_dense") else bsr
    result = dense @ other
    if out is not None:
        out.copy_(result)
        return out
    return result


def _int_bsr_dense_addmm(input: Any, bsr: Any, dense: Any, **kwargs: Any) -> Any:
    return bsr_dense_addmm(input, bsr, dense, **kwargs)


def bsr_dense_addmm(input: Any, bsr: Any, dense: Any, *, beta: Any = 1, alpha: Any = 1, left_alpha: Any = None, right_alpha: Any = None, out: Any = None, **kwargs: Any) -> Any:
    del kwargs
    result = bsr_scatter_mm(bsr, dense)
    if left_alpha is not None:
        result = result * left_alpha.reshape(-1, 1)
    if right_alpha is not None:
        result = result * right_alpha.reshape(1, -1)
    result = alpha * result + beta * input
    if out is not None:
        out.copy_(result)
        return out
    return result
