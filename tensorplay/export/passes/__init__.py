"""Passes that operate on exported graph programs."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from ...graph.passes.infra.pass_base import PassResult

__all__ = ["PassResult", "move_to_device_pass"]


def _device_for(value: Any, location: Any) -> Any:
    current = getattr(value, "device", None)
    if isinstance(location, Mapping):
        return location.get(str(current), current)
    return location


def _move_value(value: Any, location: Any) -> Any:
    if hasattr(value, "to") and callable(value.to):
        target = _device_for(value, location)
        return value.to(target) if target is not None else value
    if isinstance(value, tuple):
        return tuple(_move_value(item, location) for item in value)
    if isinstance(value, list):
        return [_move_value(item, location) for item in value]
    if isinstance(value, dict):
        return {key: _move_value(item, location) for key, item in value.items()}
    return value


def move_to_device_pass(exported_program: Any, location: Any) -> Any:
    """Move captured state, examples, and device metadata to a destination."""

    result = copy.deepcopy(exported_program)
    root = result.graph_module.root
    for name, value in list(result.named_parameters()) + list(result.named_buffers()):
        parent_name, _, attribute = name.rpartition(".")
        parent = root if not parent_name else result.graph_module.get_submodule(parent_name)
        setattr(parent, attribute, _move_value(value, location))
    result.example_inputs = _move_value(result.example_inputs, location)
    for node in result.graph.nodes:
        node.args = _move_value(node.args, location)
        node.kwargs = _move_value(node.kwargs, location)
        node.meta = _move_value(node.meta, location)
    result.graph_module.recompile()
    result.validate()
    return result
