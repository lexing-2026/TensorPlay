"""Tracing metadata, provenance records, and temporary tracing contexts."""

from __future__ import annotations

import copy
import threading
import traceback as _traceback
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Optional, ParamSpec, TypeVar

from ._compatibility import compatibility
from .graph import Graph
from .graph_module import GraphModule
from .node import Node

_P = ParamSpec("_P")
_R = TypeVar("_R")

current_meta: dict[str, Any] = {}
current_replay_node: Node | None = None
should_preserve_node_meta = False
_should_preserve_node_meta = False
_regional_name = threading.local()
_GRAPH_METADATA_REGISTRY: dict[str, dict[str, Any]] = {}

GRADIENT_ACC_SPECIAL_STACK = "Gradient accumulation node due to repeated use"


@compatibility(is_backward_compatible=False)
class NodeSourceAction(Enum):
    CREATE = "create"
    REPLACE = "replace"


@compatibility(is_backward_compatible=False)
class NodeSource:
    """Describe how a graph node was produced from earlier nodes."""

    class NodeInfo:
        def __init__(self, name: str, target: str, graph_id: int) -> None:
            self.name = name
            self.target = target
            self.graph_id = graph_id

    def __init__(
        self,
        node: Node | None,
        pass_name: str = "",
        action: NodeSourceAction | list[NodeSourceAction] | None = None,
    ) -> None:
        if action is None:
            actions: list[NodeSourceAction] = []
        elif isinstance(action, list):
            actions = action
        else:
            actions = [action]
        if any(not isinstance(item, NodeSourceAction) for item in actions):
            raise TypeError("action must contain NodeSourceAction values")
        self.pass_name = pass_name
        self.action = actions
        self.node_info = (
            self.NodeInfo(node.name, str(node.target), id(node.graph))
            if node is not None
            else None
        )
        self.from_node = (
            copy.deepcopy(node.meta.get("from_node", [])) if node is not None else []
        )
        self._dict: dict[str, Any] | None = None
        self._action_string: str | None = None

    @property
    def name(self) -> str:
        return self.node_info.name if self.node_info else ""

    @property
    def target(self) -> str:
        return self.node_info.target if self.node_info else ""

    @property
    def graph_id(self) -> int:
        return self.node_info.graph_id if self.node_info else -1

    def _get_action_string(self) -> str:
        if self._action_string is None:
            self._action_string = "+".join(item.name.lower() for item in self.action)
        return self._action_string

    def print_readable(self, indent: int = 0) -> str:
        if indent > 9:
            return ""
        result = (
            " " * indent * 4
            + f"(name={self.name}, pass_name={self.pass_name}, "
            f"action={self._get_action_string()}, graph_id={self.graph_id})\n"
        )
        return result + "".join(item.print_readable(indent + 1) for item in self.from_node)

    def __repr__(self) -> str:
        return self.print_readable()

    def to_dict(self) -> dict[str, Any]:
        if self._dict is None:
            self._dict = {
                "name": self.name,
                "target": self.target,
                "graph_id": self.graph_id,
                "pass_name": self.pass_name,
                "action": self._get_action_string(),
                "from_node": [item.to_dict() for item in self.from_node],
            }
        return self._dict

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NodeSource) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted((key, freeze(item)) for key, item in value.items()))
            if isinstance(value, list):
                return tuple(freeze(item) for item in value)
            return value

        return hash(freeze(self.to_dict()))

    @classmethod
    def _from_dict(cls, value: dict[str, Any] | None) -> Optional["NodeSource"]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("node source data must be a mapping")
        result = cls.__new__(cls)
        result.pass_name = value.get("pass_name", "")
        action_text = value.get("action", "")
        result.action = [
            NodeSourceAction[item.upper()]
            for item in str(action_text).split("+")
            if item
        ]
        if {"name", "target", "graph_id"} <= value.keys():
            result.node_info = cls.NodeInfo(
                value["name"], value["target"], value["graph_id"]
            )
        else:
            result.node_info = None
        result.from_node = [
            nested
            for item in value.get("from_node", [])
            if (nested := cls._from_dict(item)) is not None
        ]
        result._dict = None
        result._action_string = None
        return result


@compatibility(is_backward_compatible=False)
def _register_graph_metadata(module_name: str, metadata: dict[str, Any]) -> None:
    _GRAPH_METADATA_REGISTRY[module_name] = metadata


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta(enable: bool = True) -> Iterator[None]:
    global should_preserve_node_meta, current_meta
    old_enabled = should_preserve_node_meta
    old_meta = current_meta.copy()
    should_preserve_node_meta = enable
    try:
        yield
    finally:
        should_preserve_node_meta = old_enabled
        current_meta = old_meta


@contextmanager
def _preserve_node_seq_nr(preserve_seq_nr: bool = True) -> Iterator[None]:
    global _should_preserve_node_meta
    old = _should_preserve_node_meta
    _should_preserve_node_meta = preserve_seq_nr
    try:
        yield
    finally:
        _should_preserve_node_meta = old


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: list[str]) -> None:
    if should_preserve_node_meta:
        if stack:
            current_meta["stack_trace"] = "".join(stack)
        else:
            current_meta.pop("stack_trace", None)


@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr: int) -> None:
    if should_preserve_node_meta:
        current_meta.setdefault("grad_fn_seq_nr", []).append(seq_nr)
        current_meta["in_grad_fn"] = current_meta.get("in_grad_fn", 0) + 1


@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr() -> None:
    if not should_preserve_node_meta:
        return
    depth = current_meta.get("in_grad_fn", 0)
    if depth <= 0:
        raise RuntimeError("gradient sequence state is not active")
    if depth == 1:
        current_meta.pop("in_grad_fn", None)
        current_meta.pop("grad_fn_seq_nr", None)
    else:
        current_meta["in_grad_fn"] = depth - 1
        current_meta["grad_fn_seq_nr"] = current_meta["grad_fn_seq_nr"][:-1]


@compatibility(is_backward_compatible=False)
def format_stack() -> list[str]:
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    return _traceback.format_list(_traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta


def _is_preserving_node_seq_nr() -> bool:
    return _should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node: Node, pass_name: str = "") -> Iterator[None]:
    global current_meta
    if not should_preserve_node_meta or not node.meta:
        yield
        return
    old = current_meta
    current_meta = node.meta.copy()
    current_meta["from_node"] = [NodeSource(node, pass_name, NodeSourceAction.CREATE)]
    try:
        yield
    finally:
        current_meta = old


@compatibility(is_backward_compatible=False)
def get_current_meta() -> dict[str, Any]:
    return current_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def annotate(annotation_dict: dict[str, Any]) -> Iterator[None]:
    global current_meta
    old = current_meta
    current_meta = current_meta.copy()
    current_meta.setdefault("custom", {}).update(annotation_dict)
    try:
        yield
    finally:
        current_meta = old


@compatibility(is_backward_compatible=False)
def annotate_fn(annotation_dict: dict[str, Any]) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def decorator(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with annotate(annotation_dict):
                return function(*args, **kwargs)

        return wrapped

    return decorator


@contextmanager
def _set_autograd_backward(enable: bool = True) -> Iterator[None]:
    old = current_meta.get("autograd_backward")
    had_old = "autograd_backward" in current_meta
    if enable:
        _mark_autograd_backward()
    try:
        yield
    finally:
        if had_old:
            current_meta["autograd_backward"] = old
        else:
            _reset_autograd_backward()


def _mark_autograd_backward() -> None:
    current_meta["autograd_backward"] = True


def _reset_autograd_backward() -> None:
    current_meta.pop("autograd_backward", None)


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_replay_node(node: Node | None) -> Iterator[None]:
    global current_replay_node
    old = current_replay_node
    current_replay_node = node
    try:
        yield
    finally:
        current_replay_node = old


@compatibility(is_backward_compatible=False)
def get_current_replay_node() -> Node | None:
    return current_replay_node


@contextmanager
def _set_regional_inductor_subgraph_name(name: str | None) -> Iterator[None]:
    old = getattr(_regional_name, "value", None)
    _regional_name.value = name
    try:
        yield
    finally:
        _regional_name.value = old


def _get_regional_inductor_subgraph_name() -> str | None:
    return getattr(_regional_name, "value", None)


@compatibility(is_backward_compatible=False)
def get_graph_provenance_json(graph: Graph) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in graph.nodes:
        if node.op == "call_function":
            result[node.name] = [
                source.to_dict() for source in node.meta.get("from_node", [])
            ]
    return result


def _get_custom_metadata(graph_module: GraphModule) -> str:
    if not isinstance(graph_module, GraphModule):
        raise TypeError("expected a GraphModule")
    records = []
    for node in graph_module.graph.nodes:
        if node.meta.get("custom"):
            records.append((node.op, node.name, node.meta["custom"]))
    return "\n".join(str(item) for item in records)


def _get_ordered_seq_nr_groups(
    graph_modules: GraphModule | list[GraphModule],
) -> list[list[str]]:
    modules = [graph_modules] if isinstance(graph_modules, GraphModule) else graph_modules
    grouped: dict[int, list[str]] = defaultdict(list)
    for graph_module in modules:
        for node in graph_module.graph.nodes:
            sequence = node.meta.get("seq_nr")
            if node.op == "call_function" and sequence is not None:
                grouped[sequence].append(node.name)
    return [sorted(grouped[key]) for key in sorted(grouped)]


__all__ = [
    "GRADIENT_ACC_SPECIAL_STACK",
    "NodeSource",
    "NodeSourceAction",
    "annotate",
    "annotate_fn",
    "format_stack",
    "get_current_meta",
    "get_current_replay_node",
    "get_graph_provenance_json",
    "has_preserved_node_meta",
    "preserve_node_meta",
    "reset_grad_fn_seq_nr",
    "set_current_meta",
    "set_current_replay_node",
    "set_grad_fn_seq_nr",
    "set_stack_trace",
]
