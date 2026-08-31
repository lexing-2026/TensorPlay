"""Rules shared by elementwise and Einstein-style operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec
from ..placement_types import Replicate
from .utils import infer_broadcast_dims_map, map_placements_after_broadcast

__all__ = ["OutputSharding", "einop_rule", "pointwise_rule"]


@dataclass(frozen=True)
class OutputSharding:
    output_spec: DTensorSpec | None
    schema_suggestions: tuple[Any, ...] = ()
    failed_reason: str | None = None

def _template(args: Any) -> DTensor | None:
    if isinstance(args, DTensor):
        return args
    if isinstance(args, (tuple, list)):
        for value in args:
            found = _template(value)
            if found is not None:
                return found
    return None


def pointwise_rule(op_schema: Any, linearity: bool = False) -> OutputSharding:
    del linearity
    template = _template(getattr(op_schema, "args", op_schema))
    if template is None:
        return OutputSharding(None, failed_reason="no distributed tensor input")
    placements = template.placements
    for argument in getattr(op_schema, "args", ()):
        if isinstance(argument, DTensor) and argument.shape != template.shape:
            dim_map = infer_broadcast_dims_map(argument.shape, template.shape)
            placements = map_placements_after_broadcast(argument.placements, dim_map)
    return OutputSharding(DTensorSpec(template.device_mesh, tuple(placements), None))


def einop_rule(op_schema: Any, equation: str | None = None) -> OutputSharding:
    del equation
    return pointwise_rule(op_schema)
