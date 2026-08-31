"""Composite operations used by distributed tensor dispatch."""

from __future__ import annotations

from typing import Any

from ._api import DTensor

__all__ = ["local_map", "redistribute", "to_local"]


def to_local(value: Any) -> Any:
    return value.to_local() if isinstance(value, DTensor) else value


def redistribute(value: DTensor, device_mesh: Any, placements: Any) -> DTensor:
    return value.redistribute(device_mesh, placements)


def local_map(function: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(value, DTensor):
        result = function(value.to_local(), *args, **kwargs)
        return DTensor(result, value.device_mesh, value.placements, shape=value.shape)
    return function(value, *args, **kwargs)
