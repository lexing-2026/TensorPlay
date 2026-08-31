"""Placement rules for reductions and normalization operations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Partial, Replicate, Shard
from .utils import normalize_dims

__all__ = ["Reduction", "common_reduction_strategy", "get_placement_from_reduction_op", "map_placements_after_reduction", "replicate_reduction_dims"]


class Reduction(str, Enum):
    SUM = "sum"
    PROD = "prod"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


def get_placement_from_reduction_op(reduction_op: str | Reduction) -> Partial:
    value = reduction_op.value if isinstance(reduction_op, Reduction) else reduction_op
    return Partial("avg" if value == "mean" else value)


def replicate_reduction_dims(placements: Sequence[Any], dims: Sequence[int]) -> tuple[Any, ...]:
    dimensions = set(dims)
    return tuple(Replicate() if isinstance(placement, Shard) and placement.dim in dimensions else placement for placement in placements)


def map_placements_after_reduction(
    spec: DTensorSpec, dims: int | Sequence[int] | None, keepdim: bool = False
) -> DTensorSpec:
    if spec.shape is None:
        return spec
    reduced = set(normalize_dims(dims, len(spec.shape)))
    placements = replicate_reduction_dims(spec.placements, tuple(reduced))
    if keepdim:
        shape = tuple(1 if index in reduced else value for index, value in enumerate(spec.shape))
    else:
        shape = tuple(value for index, value in enumerate(spec.shape) if index not in reduced)
    return DTensorSpec(spec.mesh, placements, type(spec.tensor_meta)(shape, spec.tensor_meta.stride, spec.tensor_meta.dtype) if spec.tensor_meta else None)


def common_reduction_strategy(op_schema: Any, dims: int | Sequence[int] | None = None, keepdim: bool = False) -> Any:
    value = next((item for item in getattr(op_schema, "args", ()) if isinstance(item, DTensor)), None)
    if value is None:
        return None
    return map_placements_after_reduction(DTensorSpec(value.device_mesh, value.placements), dims, keepdim)
