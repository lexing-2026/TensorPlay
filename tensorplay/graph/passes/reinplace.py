"""Recover safe in-place operations from functional graph operations."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Any

from .._utils import _iter_nodes
from ..graph_module import GraphModule
from ..node import Node
from .shape_prop import ShapeProp

__all__ = ["reinplace"]


class _ViewType(Enum):
    NonView = 0
    SingleOutputView = 1
    MultiOutputView = 2


_VIEW_METHODS = {
    "as_strided",
    "detach",
    "expand",
    "flatten",
    "movedim",
    "narrow",
    "permute",
    "reshape",
    "select",
    "slice",
    "squeeze",
    "t",
    "transpose",
    "unflatten",
    "unsqueeze",
    "view",
}

_FUNCTION_METHODS = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "div",
    operator.floordiv: "floordiv",
    operator.pow: "pow",
}


def _view_type(target: Any) -> _ViewType:
    name = target if isinstance(target, str) else getattr(target, "__name__", "")
    if name in _VIEW_METHODS:
        return _ViewType.SingleOutputView
    return _ViewType.NonView


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if callable(shape):
        shape = shape()
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def _dtype(value: Any) -> Any:
    dtype = getattr(value, "dtype", None)
    return dtype() if callable(dtype) else dtype


def _later_users(node: Node, position: dict[Node, int]) -> set[Node]:
    return {
        user
        for user in node.users
        if position.get(user, -1) > position[node]
    }


def _inplace_target(node: Node) -> tuple[str, str] | None:
    if node.op == "call_method" and isinstance(node.target, str):
        if node.target in _VIEW_METHODS or node.target.endswith("_"):
            return None
        return "call_method", node.target + "_"
    if node.op == "call_function" and node.target in _FUNCTION_METHODS:
        return "call_method", _FUNCTION_METHODS[node.target] + "_"
    return None


def _can_reinplace(node: Node, position: dict[Node, int]) -> bool:
    if not node.args or not isinstance(node.args[0], Node):
        return False
    source = node.args[0]
    if source.op in {"placeholder", "get_attr"}:
        return False
    if sum(1 for arg in node.args if arg is source) > 1:
        return False
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if source_value is not None and result_value is not None:
        if _shape(source_value) != _shape(result_value) or _dtype(source_value) != _dtype(result_value):
            return False
    later = _later_users(source, position)
    later.discard(node)
    for user in later:
        if user.op == "call_method" and _view_type(user.target) is not _ViewType.NonView:
            continue
        if user.op == "call_function" and user.target is operator.getitem:
            continue
        return False
    return True


def reinplace(gm: GraphModule, *sample_args: Any) -> GraphModule:
    """Rewrite safe out-of-place operations to mutate an internal value."""

    if sample_args:
        ShapeProp(gm).propagate(*sample_args)
    nodes = list(gm.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    changed = False
    for node in nodes:
        target = _inplace_target(node)
        if target is None or not _can_reinplace(node, position):
            continue
        replacement_op, replacement_target = target
        source = node.args[0]
        node.op = replacement_op
        node.target = replacement_target
        node.replace_all_uses_with(source)
        changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return gm
