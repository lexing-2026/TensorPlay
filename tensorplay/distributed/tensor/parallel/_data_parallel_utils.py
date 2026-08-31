"""Flattening helpers for data-parallel handoff."""

from __future__ import annotations

from typing import Any

from .._api import DTensor

__all__ = ["_flatten_tensor", "_unflatten_tensor"]


def _flatten_tensor(value: Any) -> tuple[Any, Any]:
    if isinstance(value, DTensor):
        return value.to_local(), value
    return value, None


def _unflatten_tensor(value: Any, spec: Any, **kwargs: Any) -> Any:
    del kwargs
    if spec is None:
        return value
    return DTensor.from_local(value, spec.device_mesh, spec.placements, shape=spec.shape)
