"""Shape maps for view-like tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Shard, _is_shard_like
from .utils import normalize_dim

__all__ = [
    "Broadcast",
    "ClaimedDim",
    "DimSpec",
    "Flatten",
    "InputDim",
    "NewDim",
    "Repeat",
    "Singleton",
    "Split",
    "dim_flatten",
    "dim_movedim",
    "dim_transpose",
    "propagate_shape_and_sharding",
]


class ClaimedDim(NamedTuple):
    input_dim: int
    output_dim: int


class DimSpec:
    pass


@dataclass(frozen=True)
class Singleton(DimSpec):
    pass


@dataclass(frozen=True)
class InputDim(DimSpec):
    index: int


@dataclass(frozen=True)
class Broadcast(DimSpec):
    index: int


@dataclass(frozen=True)
class NewDim(DimSpec):
    size: int


@dataclass(frozen=True)
class Repeat(DimSpec):
    size: int


@dataclass(frozen=True)
class Flatten(DimSpec):
    start: int
    end: int


@dataclass(frozen=True)
class Split(DimSpec):
    factors: tuple[int, ...]


def dim_transpose(ndim: int, dim1: int, dim2: int) -> tuple[int, ...]:
    result = list(range(ndim))
    left, right = normalize_dim(dim1, ndim), normalize_dim(dim2, ndim)
    result[left], result[right] = result[right], result[left]
    return tuple(result)


def dim_flatten(ndim: int, start_dim: int = 0, end_dim: int = -1) -> tuple[int | None, ...]:
    start, end = normalize_dim(start_dim, ndim), normalize_dim(end_dim, ndim)
    if start > end:
        raise ValueError("start_dim must not exceed end_dim")
    return tuple(range(start)) + (None,) + tuple(range(end + 1, ndim))


def dim_movedim(ndim: int, source: int | Sequence[int], destination: int | Sequence[int]) -> tuple[int, ...]:
    sources = (source,) if isinstance(source, int) else tuple(source)
    destinations = (destination,) if isinstance(destination, int) else tuple(destination)
    if len(sources) != len(destinations):
        raise ValueError("source and destination must have equal length")
    order = [index for index in range(ndim) if index not in {normalize_dim(value, ndim) for value in sources}]
    for destination_index, source_value in sorted(zip(destinations, sources), key=lambda item: normalize_dim(item[0], ndim)):
        order.insert(normalize_dim(destination_index, ndim), normalize_dim(source_value, ndim))
    return tuple(order)


def propagate_shape_and_sharding(value: DTensor, operation: Any, *args: Any, **kwargs: Any) -> DTensorSpec:
    result = operation(value.to_local(), *args, **kwargs)
    shape = tuple(result.shape) if hasattr(result, "shape") else value.shape
    return DTensorSpec(value.device_mesh, value.placements, None if shape == value.shape else None)
