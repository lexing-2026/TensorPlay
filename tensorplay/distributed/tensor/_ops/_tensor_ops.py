"""Placement rules for indexing, concatenation, and shape creation."""

from __future__ import annotations

from typing import Any, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Replicate, Shard
from .utils import normalize_dim

__all__ = ["cat_single_dim_strategy", "create_like_single_dim_strategy", "propagate_single_input_single_dim_strategy", "stack_strategy"]


def _values(schema: Any) -> list[DTensor]:
    return [value for value in getattr(schema, "args", schema) if isinstance(value, DTensor)]


def propagate_single_input_single_dim_strategy(op_schema: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    value = next(iter(_values(op_schema)), None)
    return None if value is None else DTensorSpec(value.device_mesh, value.placements, None)


def cat_single_dim_strategy(op_schema: Any, dim: int = 0) -> Any:
    values = _values(op_schema)
    if not values:
        return None
    dim = normalize_dim(dim, values[0].ndim)
    placements = list(values[0].placements)
    if any(isinstance(value.placements[index], Shard) and value.placements[index].dim == dim for value in values for index in range(len(value.placements))):
        return DTensorSpec(values[0].device_mesh, tuple(placements), None)
    placements = [Replicate() if isinstance(placement, Shard) and placement.dim == dim else placement for placement in placements]
    return DTensorSpec(values[0].device_mesh, tuple(placements), None)


def stack_strategy(op_schema: Any, dim: int = 0) -> Any:
    values = _values(op_schema)
    if not values:
        return None
    dim = dim if dim >= 0 else dim + values[0].ndim + 1
    placements = tuple(
        Shard(placement.dim + 1 if placement.dim >= dim else placement.dim)
        if isinstance(placement, Shard)
        else placement
        for placement in values[0].placements
    )
    return DTensorSpec(values[0].device_mesh, placements, None)


def create_like_single_dim_strategy(op_schema: Any, *args: Any, **kwargs: Any) -> Any:
    return propagate_single_input_single_dim_strategy(op_schema, *args, **kwargs)
