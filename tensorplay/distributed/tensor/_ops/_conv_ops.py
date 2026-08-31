"""Placement rules for convolution-shaped operations."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Partial, Shard

__all__ = ["convolution_backward_rules", "convolution_rules", "convolution_single_dim_strategy"]


def convolution_rules(op_schema: Any) -> DTensorSpec | None:
    value = next((item for item in getattr(op_schema, "args", ()) if isinstance(item, DTensor)), None)
    return None if value is None else DTensorSpec(value.device_mesh, value.placements, None)


def convolution_backward_rules(op_schema: Any) -> DTensorSpec | None:
    return convolution_rules(op_schema)


def convolution_single_dim_strategy(input_value: DTensor, weight: DTensor | None = None, *args: Any, **kwargs: Any) -> DTensorSpec:
    del args, kwargs
    placements = list(input_value.placements)
    if weight is not None:
        for index, placement in enumerate(placements):
            if isinstance(placement, Shard) and placement.dim == 1:
                placements[index] = Partial("sum")
    return DTensorSpec(input_value.device_mesh, tuple(placements), None)
