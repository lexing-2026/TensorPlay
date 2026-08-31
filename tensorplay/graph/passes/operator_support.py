"""Composable backend support predicates for graph nodes."""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._utils import _iter_nodes
from .tools_common import CALLABLE_NODE_OPS, get_node_target

TargetTypeName = str
SupportedArgumentDTypes = (
    tuple[Sequence[Sequence[Any] | None], Mapping[str, Sequence[Any] | None]]
    | None
)
SupportDict = Mapping[TargetTypeName, SupportedArgumentDTypes]

__all__ = [
    "OpSupports",
    "OperatorSupport",
    "OperatorSupportBase",
    "SupportDict",
    "any_chain",
    "chain",
    "create_op_support",
]


class OperatorSupportBase(abc.ABC):
    """Interface for deciding whether a graph node can use a backend."""

    @abc.abstractmethod
    def is_node_supported(self, submodules: Mapping[str, Any], node: Any) -> bool:
        raise NotImplementedError


def _node_dtype(node: Any) -> Any:
    metadata = node.meta.get("tensor_meta")
    if metadata is not None:
        return getattr(metadata, "dtype", None)
    value = node.meta.get("val")
    return getattr(value, "dtype", None)


def _input_nodes(value: Any):
    yield from _iter_nodes(value)


class OperatorSupport(OperatorSupportBase):
    """Match node targets and optionally constrain input dtypes."""

    _support_dict: SupportDict

    def __init__(self, support_dict: SupportDict | None = None) -> None:
        self._support_dict = dict(support_dict or {})

    def is_node_supported(self, submodules: Mapping[str, Any], node: Any) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return True
        target = get_node_target(submodules, node)
        if target not in self._support_dict:
            return False
        constraints = self._support_dict[target]
        if constraints is None:
            return True
        args_dtypes, kwargs_dtypes = constraints
        for index, allowed in enumerate(args_dtypes):
            if index >= len(node.args):
                break
            if allowed is None:
                continue
            for input_node in _input_nodes(node.args[index]):
                if _node_dtype(input_node) not in allowed:
                    return False
        for key, allowed in kwargs_dtypes.items():
            if allowed is None or key not in node.kwargs:
                continue
            for input_node in _input_nodes(node.kwargs[key]):
                if _node_dtype(input_node) not in allowed:
                    return False
        return True


def create_op_support(
    is_node_supported: Callable[[Mapping[str, Any], Any], bool]
) -> OperatorSupportBase:
    """Wrap a support predicate in the standard support interface."""

    class FunctionalOperatorSupport(OperatorSupportBase):
        def is_node_supported(self, submodules: Mapping[str, Any], node: Any) -> bool:
            return bool(is_node_supported(submodules, node))

    return FunctionalOperatorSupport()


def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Require every predicate in ``op_support`` to accept a node."""

    return create_op_support(
        lambda submodules, node: all(
            item.is_node_supported(submodules, node) for item in op_support
        )
    )


def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Accept a node when at least one predicate accepts it."""

    return create_op_support(
        lambda submodules, node: any(
            item.is_node_supported(submodules, node) for item in op_support
        )
    )


class OpSupports:
    """Factories for commonly used negative support predicates."""

    @classmethod
    def decline_if_input_dtype(cls, dtype: Any) -> OperatorSupportBase:
        def predicate(_submodules: Mapping[str, Any], node: Any) -> bool:
            return all(_node_dtype(value) != dtype for value in _input_nodes(node.args))

        return create_op_support(predicate)

    @classmethod
    def decline_if_node_in_names(cls, disallow_set: set[str]) -> OperatorSupportBase:
        return create_op_support(
            lambda _submodules, node: node.name not in disallow_set
        )
