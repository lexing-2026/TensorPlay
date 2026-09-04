"""Shape, placement, and redistribution helpers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay

from .. import distributed_core as dist
from .._local_tensor import maybe_run_for_local_tensor
from .placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = [
    "ExplicitRedistributionContext",
    "assert_no_mixed_partial_types",
    "check_tensor_meta",
    "compute_global_tensor_info",
    "compute_global_tensor_shape",
    "compute_local_shape_and_global_offset",
    "compute_local_stride",
    "compute_local_tensor_info",
    "normalize_to_tensor_size",
    "normalize_to_torch_size",
    "try_find_mesh_from_args",
]


logger = logging.getLogger(__name__)


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def normalize_to_torch_size(shape: Any) -> tuple[int, ...]:
    if hasattr(shape, "shape") and not isinstance(shape, (tuple, list)):
        shape = shape.shape
    if isinstance(shape, int):
        return (shape,)
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]
    return tuple(int(value) for value in shape)


normalize_to_tensor_size = normalize_to_torch_size


def assert_no_mixed_partial_types(placements: Sequence[Placement]) -> None:
    reductions = {
        placement.reduce_op
        for placement in placements
        if isinstance(placement, Partial)
    }
    if len(reductions) > 1 and reductions != {"sum", "avg"}:
        raise ValueError(
            "mixed partial reductions are not supported: "
            f"{sorted(reductions)}"
        )


def _format_implicit_redistribution_msg(schema: Any) -> str:
    return (
        "implicit redistribution occurred while explicit redistribution "
        f"context was active: {schema}"
    )


class ExplicitRedistributionContext:
    _local = threading.local()

    def __init__(
        self, enable: bool = True, strict: bool = False, mode: str = "raise"
    ) -> None:
        if mode not in ("raise", "warn"):
            raise RuntimeError(f"invalid redistribution context mode {mode}")
        self._enable = bool(enable)
        self._strict = bool(strict)
        self._raise_on_redistribution = mode == "raise"

    @classmethod
    def observe_redistribution(
        cls, src_spec: Any, dst_spec: Any, redistribution_msg: Any
    ) -> None:
        instance = getattr(cls._local, "_active", None)
        if instance is None:
            return
        allowed = True
        if instance._enable:
            if instance._strict:
                allowed = False
            else:
                from ._collective_utils import redistribute_cost

                allowed = redistribute_cost(src_spec, dst_spec) <= 0
        if allowed:
            return
        if instance._raise_on_redistribution:
            raise RuntimeError(redistribution_msg)
        logger.warning("%s", redistribution_msg)

    def __enter__(self) -> "ExplicitRedistributionContext":
        self._prev = getattr(self._local, "_active", None)
        self._local._active = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._local._active = self._prev


def _strided_shard_indices(
    size: int, chunks: int, rank: int, split_factor: int
) -> list[int]:
    first_chunk = (size + split_factor - 1) // split_factor
    indices: list[int] = []
    for split_index in range(split_factor):
        start = min(split_index * first_chunk, size)
        end = min(start + first_chunk, size)
        length = end - start
        second_chunk = (length + chunks - 1) // chunks
        second_start = min(rank * second_chunk, length)
        second_end = min((rank + 1) * second_chunk, length)
        indices.extend(range(start + second_start, start + second_end))
    return indices


@maybe_run_for_local_tensor
def _get_shard_size_and_offsets(
    curr_local_size: int,
    mesh_dim_size: int,
    rank: int,
    placement: Shard | _StridedShard,
    previous_offsets: Sequence[int] | None,
    zero_global_offset: int,
    skip_offset: bool,
) -> tuple[int, list[int] | None]:
    if isinstance(placement, _StridedShard):
        offsets = _strided_shard_indices(
            int(curr_local_size),
            int(mesh_dim_size),
            int(rank),
            int(placement.split_factor),
        )
        shard_size = len(offsets)
    else:
        shard_size, offset = placement.local_shard_size_and_offset(
            int(curr_local_size), int(mesh_dim_size), int(rank)
        )
        offsets = list(range(int(offset), int(offset) + int(shard_size)))
    if skip_offset:
        return int(shard_size), None
    if int(shard_size) == 0:
        return 0, [int(zero_global_offset)]
    if previous_offsets is not None:
        offsets = [int(previous_offsets[index]) for index in offsets]
    return int(shard_size), offsets


@maybe_run_for_local_tensor
def _get_first_offset(offsets: Sequence[int]) -> int:
    return int(offsets[0])


def _compute_local_shape_and_global_offset(
    global_shape: Sequence[int],
    mesh_shape: Sequence[int],
    my_coordinate: Sequence[int] | Callable[[int], int] | None,
    placements: Sequence[Placement],
    skip_offset: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(my_coordinate, (list, tuple)):
        coordinate = my_coordinate

        def coordinate_lookup(dim: int) -> int:
            return int(coordinate[dim])

    elif my_coordinate is not None:
        coordinate_lookup = my_coordinate
    else:
        raise AssertionError("mesh coordinate is required")

    local_shape = list(global_shape)
    shard_offsets: dict[int, list[int] | None] = {}
    for mesh_dim, placement in enumerate(placements):
        if not _is_shard_like(placement):
            continue
        shard_dim = placement.dim
        if shard_dim < 0 or shard_dim >= len(local_shape):
            raise AssertionError(
                f"sharding dim {shard_dim} is outside tensor rank {len(local_shape)}"
            )
        previous_offsets = shard_offsets.get(shard_dim)
        shard_size, offsets = _get_shard_size_and_offsets(
            int(local_shape[shard_dim]),
            int(mesh_shape[mesh_dim]),
            int(coordinate_lookup(mesh_dim)),
            placement,
            previous_offsets,
            int(global_shape[shard_dim]),
            skip_offset,
        )
        local_shape[shard_dim] = shard_size
        shard_offsets[shard_dim] = offsets
    if skip_offset:
        return tuple(int(value) for value in local_shape), ()
    global_offset = [0] * len(global_shape)
    for shard_dim, offsets in shard_offsets.items():
        if offsets is None:
            raise AssertionError("shard offsets were not computed")
        global_offset[shard_dim] = _get_first_offset(offsets)
    return tuple(int(value) for value in local_shape), tuple(global_offset)


def compute_local_shape_and_global_offset(
    global_shape: Sequence[int],
    mesh: Any,
    placements: Sequence[Placement],
    skip_offset: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    normalized_shape = normalize_to_torch_size(global_shape)
    get_coordinate = getattr(mesh, "get_coordinate", None)
    if get_coordinate is None:
        coordinate = lambda dim: mesh.get_local_rank(dim)
        return _compute_local_shape_and_global_offset(
            normalized_shape, mesh.shape, coordinate, placements, skip_offset
        )
    coordinate = get_coordinate()
    if coordinate is None:
        return (0,), ()
    return _compute_local_shape_and_global_offset(
        normalized_shape, mesh.shape, coordinate, placements, skip_offset
    )


def compute_local_tensor_info(
    global_tensor: Any,
    mesh: Any,
    placements: Sequence[Placement],
) -> tuple[list[int], list[int]]:
    local_shape = [int(value) for value in global_tensor.shape]
    stride_value = global_tensor.stride
    local_stride = [
        int(value)
        for value in (stride_value() if callable(stride_value) else stride_value)
    ]
    for mesh_dim, placement in enumerate(placements):
        if _is_shard_like(placement):
            shard_dim = int(placement.dim)
            if shard_dim < 0:
                raise AssertionError(
                    f"shard dimension {shard_dim} must be normalized"
                )
            if shard_dim >= len(local_shape):
                raise AssertionError(
                    f"sharding dim {shard_dim} is outside tensor rank {len(local_shape)}"
                )
            mesh_dim_size = int(mesh.size(mesh_dim))
            global_dim_size = local_shape[shard_dim]
            if global_dim_size % mesh_dim_size != 0:
                raise AssertionError(
                    f"global dimension {global_dim_size} is not divisible by mesh size {mesh_dim_size}"
                )
            local_shape[shard_dim] = global_dim_size // mesh_dim_size
            for index in range(len(local_stride)):
                if (
                    index != shard_dim
                    and local_stride[index]
                    >= local_stride[shard_dim] * mesh_dim_size
                ):
                    local_stride[index] //= mesh_dim_size
        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} is not supported")
    return local_shape, local_stride


def compute_global_tensor_shape(
    shape: Sequence[int], mesh: Any, placements: Sequence[Placement]
) -> tuple[int, ...]:
    if len(placements) != 1:
        raise NotImplementedError(
            "compute_global_tensor_shape only supports one placement"
        )
    if len(placements) != _mesh_ndim(mesh):
        raise RuntimeError(
            "one placement is required for each mesh dimension; "
            f"got {len(placements)} placements and {_mesh_ndim(mesh)} mesh dimensions"
        )
    normalized_shape = tuple(int(value) for value in shape)
    placement = placements[0]
    if isinstance(placement, Replicate):
        return normalized_shape
    if not isinstance(placement, Shard):
        raise NotImplementedError(f"placement type {type(placement)} is not supported")
    mesh_size = int(mesh.size(0))
    if mesh_size == 1:
        return normalized_shape
    local_shape = tensorplay.tensor(
        list(normalized_shape), dtype=tensorplay.int64, device=mesh.device_type
    )
    gathered = [tensorplay.empty_like(local_shape) for _ in range(mesh_size)]
    dist.all_gather(gathered, local_shape, group=mesh.get_group(0))
    values = [
        [int(item) for item in item_tensor.tolist()] for item_tensor in gathered
    ]
    shard_dim = int(placement.dim)
    if shard_dim < 0 or shard_dim >= len(normalized_shape):
        raise AssertionError(f"sharding dim {shard_dim} is outside tensor rank")
    for item in values:
        for dim, expected in enumerate(values[0]):
            if dim != shard_dim and item[dim] != expected:
                raise RuntimeError(
                    "non-sharded dimensions must have identical sizes across ranks"
                )
    result = list(normalized_shape)
    result[shard_dim] = sum(item[shard_dim] for item in values)
    return tuple(result)


def compute_local_stride(
    global_stride: Sequence[int], local_shape: Sequence[int]
) -> tuple[int, ...]:
    permutation = sorted(
        range(len(global_stride)),
        key=lambda dim: int(global_stride[dim]),
        reverse=True,
    )
    local_stride = [0] * len(global_stride)
    running = 1
    for dim in reversed(permutation):
        local_stride[dim] = running
        running *= int(local_shape[dim])
    return tuple(local_stride)


def compute_global_tensor_info(
    local_tensor: Any,
    mesh: Any,
    placements: Sequence[Placement],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    native = tensorplay._C._DTensor_compute_global_tensor_info
    shape, stride = native(local_tensor, mesh, tuple(placements))
    return tuple(int(value) for value in shape), tuple(int(value) for value in stride)


def try_find_mesh_from_args(op_call: Any, args: Sequence[object]) -> Any:
    from ._api import DTensor
    from ._dtensor_spec import DTensorSpec

    def find(value: Any) -> Any | None:
        if isinstance(value, (DTensor, DTensorSpec)):
            return value.device_mesh
        if isinstance(value, dict):
            for item in value.values():
                result = find(item)
                if result is not None:
                    return result
        if isinstance(value, (list, tuple)):
            for item in value:
                result = find(item)
                if result is not None:
                    return result
        return None

    for arg in args:
        mesh = find(arg)
        if mesh is not None:
            return mesh
    raise ValueError(f"cannot find device mesh in arguments for operation {op_call}")


def check_tensor_meta(
    value: Any,
    *,
    shape: Sequence[int] | None = None,
    dtype: Any = None,
    check_shape_stride: bool = True,
) -> None:
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"tensor shape {tuple(value.shape)} does not match {tuple(shape)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise ValueError("tensor dtype does not match distributed metadata")
    if check_shape_stride and hasattr(value, "stride"):
        stride = value.stride() if callable(value.stride) else value.stride
        if len(stride) != value.dim():
            raise ValueError("tensor stride rank does not match tensor rank")
