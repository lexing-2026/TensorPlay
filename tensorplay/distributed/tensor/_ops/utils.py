"""Shared registration and dimension utilities for placement rules."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Partial, Replicate, Shard, _is_shard_like

__all__ = [
    "as_list",
    "generate_redistribute_costs",
    "infer_broadcast_dims_map",
    "is_tensor_dim_sharded",
    "is_tensor_evenly_shardable",
    "is_tensor_evenly_shardable_on_dim",
    "is_tensor_partial",
    "is_tensor_shardable",
    "map_placements_after_broadcast",
    "normalize_dim",
    "normalize_dims",
    "prod",
    "register_op_strategy",
    "register_prop_rule",
    "replicate_op_strategy",
]

_PROPAGATION_RULES: dict[Any, Callable[..., Any]] = {}
_STRATEGY_RULES: dict[Any, Callable[..., Any]] = {}


def _get_registration_wrapper(table: dict[Any, Callable[..., Any]], operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(function: Callable[..., Any]) -> Callable[..., Any]:
        table[operation] = function
        return function

    return register


def register_prop_rule(operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _get_registration_wrapper(_PROPAGATION_RULES, operation)


def register_op_strategy(operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _get_registration_wrapper(_STRATEGY_RULES, operation)


def replicate_op_strategy(op_schema: Any) -> Any:
    inputs = getattr(op_schema, "args", ())
    template = next((value for value in inputs if isinstance(value, DTensor)), None)
    if template is None:
        return None
    return DTensorSpec(template.device_mesh, tuple(Replicate() for _ in template.placements), None)


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


def normalize_dim(dim: int, ndim: int) -> int:
    if type(dim) is not int:
        raise TypeError("dimension must be an integer")
    result = dim + ndim if dim < 0 else dim
    if result < 0 or result >= ndim:
        raise IndexError(f"dimension {dim} is outside rank {ndim}")
    return result


def normalize_dims(dims: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(ndim))
    values = (dims,) if isinstance(dims, int) else tuple(dims)
    result = tuple(normalize_dim(value, ndim) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("dimensions must be unique")
    return result


def prod(values: Iterable[int]) -> int:
    return math.prod(values)


def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec, dim: int) -> bool:
    if dim < 0 or dim >= len(shape):
        return False
    return all(int(shape[dim]) >= spec.mesh.size(index) for index, placement in enumerate(spec.placements) if _is_shard_like(placement) and placement.dim == dim)


def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    return all(is_tensor_evenly_shardable_on_dim(shape, spec, dim) for dim in range(len(shape)))


def is_tensor_evenly_shardable_on_dim(shape: Sequence[int], spec: DTensorSpec, dim: int) -> bool:
    factor = math.prod(spec.mesh.size(index) for index, placement in enumerate(spec.placements) if _is_shard_like(placement) and placement.dim == dim)
    return factor == 0 or int(shape[dim]) % factor == 0


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    return any(_is_shard_like(placement) and placement.dim == dim for placement in spec.placements)


def is_tensor_partial(spec: DTensorSpec) -> bool:
    return any(isinstance(placement, Partial) for placement in spec.placements)


def infer_broadcast_dims_map(input_shape: Sequence[int], output_shape: Sequence[int]) -> tuple[int | None, ...]:
    if len(input_shape) > len(output_shape):
        raise ValueError("input rank cannot exceed output rank")
    padding = len(output_shape) - len(input_shape)
    result: list[int | None] = [None] * padding
    for index, (source, target) in enumerate(zip(input_shape, output_shape[padding:])):
        if source not in (1, target):
            raise ValueError("shapes are not broadcastable")
        result.append(index if source == target else None)
    return tuple(result)


def map_placements_after_broadcast(placements: Sequence[Any], dim_map: Sequence[int | None]) -> tuple[Any, ...]:
    result = []
    for placement in placements:
        if _is_shard_like(placement):
            target = dim_map[placement.dim]
            result.append(placement if target is not None else Replicate())
        else:
            result.append(placement)
    return tuple(result)


def generate_redistribute_costs(current: DTensorSpec, target: DTensorSpec) -> int:
    if len(current.placements) != len(target.placements):
        return 1 << 30
    return sum(left != right for left, right in zip(current.placements, target.placements))
