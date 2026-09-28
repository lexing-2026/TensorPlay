"""Evaluate safe scalar-only operations during graph preparation."""

from __future__ import annotations

import operator
from typing import Any, Tuple

from .._utils import _iter_nodes, _map_arg
from ..node import Node
from .base import PassBase, PassResult

__all__ = ["ConstFold"]


_FOLDABLE_TARGETS = frozenset(
    {
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.neg,
        operator.pos,
        operator.abs,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.eq,
        operator.ne,
        operator.getitem,
    }
)


def _is_tensor_like(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and hasattr(value, "dtype")


def _replace_with_constant(node: Node, value: Any) -> None:
    """Replace every use of ``node`` with a literal value, then erase it."""

    for user in list(node.users):
        user.args = _map_arg(user.args, lambda v: value if v is node else v)
        user.kwargs = _map_arg(user.kwargs, lambda v: value if v is node else v)
        node.users.discard(user)
    node.erase_node()


def _iter_flat(values: Tuple[Any, ...]):
    for item in values:
        yield item


def _is_foldable(target: Any) -> bool:
    """Whether ``target`` is one of the operations worth folding.

    Compared by identity rather than looked up: a target that names an
    operation a type provides need not be hashable, and a fold keyed on a
    particular operation should not be put off by the graph also carrying
    targets of that kind.
    """

    return any(target is foldable for foldable in _FOLDABLE_TARGETS)


class ConstFold(PassBase):
    """Fold whitelisted operations whose inputs are all ordinary values."""

    def __call__(self, graph_module) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or not _is_foldable(node.target):
                continue
            if next(_iter_nodes(node.args), None) is not None:
                continue
            if next(_iter_nodes(node.kwargs), None) is not None:
                continue
            args = tuple(node.args)
            if any(_is_tensor_like(item) for item in _iter_flat(args)):
                continue
            try:
                value = node.target(*args, **node.kwargs)
            except Exception:
                continue
            _replace_with_constant(node, value)
            modified = True
        return PassResult(graph_module, modified)
