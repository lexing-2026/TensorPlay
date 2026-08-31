"""Shape, placement, and redistribution validation helpers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .placement_types import Partial, Placement, Replicate, Shard, _is_shard_like

__all__ = [
    "ExplicitRedistributionContext",
    "assert_no_mixed_partial_types",
    "check_tensor_meta",
    "compute_global_tensor_info",
    "compute_local_shape_and_global_offset",
    "normalize_to_tensor_size",
    "normalize_to_torch_size",
]


def normalize_to_torch_size(shape: Any) -> tuple[int, ...]:
    if hasattr(shape, "shape") and not isinstance(shape, (tuple, list)):
        shape = shape.shape
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(value) for value in shape)


normalize_to_tensor_size = normalize_to_torch_size


def assert_no_mixed_partial_types(placements: Sequence[Placement]) -> None:
    reductions = {p.reduce_op for p in placements if isinstance(p, Partial)}
    if len(reductions) > 1:
        raise ValueError("all partial placements must use one reduction operation")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def compute_local_shape_and_global_offset(
    global_shape: Sequence[int], mesh: Any, placements: Sequence[Placement]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = list(normalize_to_torch_size(global_shape))
    offset = [0] * len(shape)
    for mesh_dim, placement in enumerate(placements):
        if not _is_shard_like(placement):
            continue
        chunks = mesh.size(mesh_dim)
        width = _ceil_div(shape[placement.dim], chunks)
        rank = mesh.get_local_rank(mesh_dim)
        offset[placement.dim] += min(rank * width, shape[placement.dim])
        shape[placement.dim] = min(width, max(0, shape[placement.dim] - rank * width))
    return tuple(shape), tuple(offset)


def compute_global_tensor_info(
    local_shape: Sequence[int],
    local_stride: Sequence[int],
    placements: Sequence[Placement],
    mesh: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = list(normalize_to_torch_size(local_shape))
    stride = tuple(int(value) for value in local_stride)
    for mesh_dim, placement in enumerate(placements):
        if _is_shard_like(placement):
            shape[placement.dim] *= mesh.size(mesh_dim)
    return tuple(shape), stride


def check_tensor_meta(value: Any, *, shape: Sequence[int] | None = None, dtype: Any = None, check_shape_stride: bool = True) -> None:
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"tensor shape {tuple(value.shape)} does not match {tuple(shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError("tensor dtype does not match the distributed metadata")
    if check_shape_stride and hasattr(value, "stride") and len(value.stride()) != value.dim():
        raise ValueError("tensor stride rank does not match tensor rank")


@dataclass(frozen=True)
class _RedistributionEvent:
    current: Any
    target: Any
    message: Any


class ExplicitRedistributionContext:
    """Collect redistribution observations for diagnostics and tests."""

    _events: list[_RedistributionEvent] = []

    @classmethod
    def observe_redistribution(cls, current: Any, target: Any, message: Any = None) -> None:
        cls._events.append(_RedistributionEvent(current, target, message))

    @classmethod
    def events(cls) -> tuple[_RedistributionEvent, ...]:
        return tuple(cls._events)

    @classmethod
    def clear(cls) -> None:
        cls._events.clear()

    @classmethod
    @contextlib.contextmanager
    def collect(cls) -> Iterator[list[_RedistributionEvent]]:
        start = len(cls._events)
        try:
            yield cls._events[start:]
        finally:
            del cls._events[start:]
