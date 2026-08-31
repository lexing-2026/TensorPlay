"""Collect lowering attributes from leaf modules and attach them to nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..graph_module import GraphModule

__all__ = ["default_matching", "extract_attrs_for_lowering", "lift_lowering_attrs_to_nodes"]


def default_matching(name: str, target_version: int) -> str:
    del target_version
    return name


_ATTRIBUTE_NAMES = {
    "Linear": ("weight", "bias"),
    "Conv2d": (
        "weight", "bias", "kernel_size", "stride", "padding", "dilation", "groups",
    ),
    "BatchNorm2d": ("weight", "bias", "running_mean", "running_var", "eps"),
    "AdaptiveAvgPool2d": (),
    "MaxPool2d": ("kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"),
    "ReLU": ("inplace",),
}


def extract_attrs_for_lowering(mod: Any) -> dict[str, Any]:
    """Return stable scalar and parameter attributes exposed by a module."""

    name = type(mod).__name__
    attrs = _ATTRIBUTE_NAMES.get(name)
    if attrs is None:
        raise RuntimeError(f"module type {name!r} has no lowering attribute specification")
    result = {"name": f"{type(mod).__module__}.{name}"}
    for attr in attrs:
        if hasattr(mod, attr):
            result[attr] = getattr(mod, attr)
    return result


def lift_lowering_attrs_to_nodes(graph_module: GraphModule) -> None:
    """Attach module attributes to each module-call node recursively."""

    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        try:
            module = graph_module._get_attr(node.target)
        except AttributeError as exc:
            raise RuntimeError(f"missing module target {node.target!r}") from exc
        node.meta["attrs_for_lowering"] = extract_attrs_for_lowering(module)
