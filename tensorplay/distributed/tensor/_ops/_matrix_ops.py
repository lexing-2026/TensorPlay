"""Placement rules for matrix products and transposes."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Partial, Replicate, Shard
from ._view_ops import dim_transpose

__all__ = ["addmm_single_dim_strategy", "bmm_single_dim_strategy", "mm_single_dim_strategy", "transpose_single_dim_strategy"]


def transpose_single_dim_strategy(value: DTensor, dim0: int = 0, dim1: int = 1) -> DTensorSpec:
    permutation = dim_transpose(value.ndim, dim0, dim1)
    placements = tuple(
        Shard(permutation.index(placement.dim)) if isinstance(placement, Shard) else placement
        for placement in value.placements
    )
    return DTensorSpec(value.device_mesh, placements, None)


def mm_single_dim_strategy(left: DTensor, right: DTensor) -> DTensorSpec:
    if left.device_mesh != right.device_mesh:
        raise ValueError("matrix operands must use the same mesh")
    placements = list(left.placements)
    for index, placement in enumerate(placements):
        if isinstance(placement, Shard) and placement.dim == left.ndim - 1:
            placements[index] = Partial("sum")
    return DTensorSpec(left.device_mesh, tuple(placements), None)


def addmm_single_dim_strategy(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    return mm_single_dim_strategy(*args[:2])


def bmm_single_dim_strategy(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    return mm_single_dim_strategy(*args[:2])
