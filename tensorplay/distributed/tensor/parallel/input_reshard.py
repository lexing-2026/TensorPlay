"""Activation input resharding hooks."""

from __future__ import annotations

from typing import Any

from .._api import DTensor
from ..placement_types import Replicate, Shard

__all__ = ["input_reshard"]


def _mesh_rank(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def _layout(mesh: Any, placement: Any) -> tuple[Any, ...]:
    return (placement,) + tuple(Replicate() for _ in range(_mesh_rank(mesh) - 1))


def input_reshard(module: Any, tp_device_mesh: Any, input_reshard_dim: int | None = None) -> Any:
    if input_reshard_dim is None:
        return module

    def hook(current: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        del current
        if not inputs or isinstance(inputs[0], DTensor):
            return inputs
        value = DTensor.from_local(
            inputs[0], tp_device_mesh, _layout(tp_device_mesh, Replicate()), run_check=False
        )
        return (
            value.redistribute(
                placements=_layout(tp_device_mesh, Shard(input_reshard_dim))
            ),
        ) + inputs[1:]

    module.register_forward_pre_hook(hook)
    return module
