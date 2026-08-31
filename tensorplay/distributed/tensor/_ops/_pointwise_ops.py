"""Placement propagation for pointwise operations."""

from __future__ import annotations

from typing import Any, Callable

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Replicate
from ._common_rules import OutputSharding
from .utils import infer_broadcast_dims_map

__all__ = ["common_pointwise_single_dim_strategy", "register_inductor_prims"]


def common_pointwise_single_dim_strategy(*, partial_extra_rules: list[Any] | None = None) -> Callable[..., list[Any]]:
    def strategy(op: Any, args_schema: Any, kwargs_schema: Any) -> list[Any]:
        del op, kwargs_schema, partial_extra_rules
        values = [value for value in args_schema if isinstance(value, DTensor)]
        if not values:
            return []
        output = values[0]
        result = [OutputSharding(DTensorSpec(output.device_mesh, output.placements, None))]
        return result

    return strategy


def register_inductor_prims() -> None:
    return None
