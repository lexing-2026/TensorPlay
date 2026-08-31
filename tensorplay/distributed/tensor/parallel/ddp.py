"""Conversion hooks for composing tensor and data parallel modules."""

from __future__ import annotations

from typing import Any

from .._api import DTensor

__all__ = ["pre_dp_module_transform"]


def pre_dp_module_transform(module: Any) -> Any:
    stored: dict[tuple[Any, str], Any] = {}
    for child_name, child in module.named_modules():
        for name, value in list(child._parameters.items()):
            if isinstance(value, DTensor):
                stored[(child, name)] = value
                child._parameters[name] = value.to_local()

    def restore(current: Any, inputs: tuple[Any, ...]) -> None:
        del current, inputs
        for (child, name), value in stored.items():
            child._parameters[name] = value

    module.register_forward_pre_hook(restore)
    return module
