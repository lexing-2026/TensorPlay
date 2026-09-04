"""Placement rules for operations with restricted distributed layouts."""

from __future__ import annotations

from .._dtensor_spec import DTensorSpec
from .._op_schema import OpSchema, OpStrategy, PlacementStrategy
from ..device_mesh import DeviceMesh
from ..placement_types import Replicate

__all__ = ["slice_backward_rules"]


def slice_backward_rules(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    del op_schema
    replicate_spec = DTensorSpec(
        mesh,
        tuple(Replicate() for _ in range(int(mesh.ndim))),
    )
    return OpStrategy([PlacementStrategy(replicate_spec)])
