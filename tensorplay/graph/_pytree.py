"""Small, dependency-free tree flattening utilities for graph arguments."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

PyTree = Any
Context = Any
_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")

FlattenFn = Callable[[Any], tuple[list[Any], Context]]
UnflattenFn = Callable[[Iterable[Any], Context], Any]
FlattenFnSpec = Callable[[PyTree, "TreeSpec"], list[Any]]
FlattenFnExactMatchSpec = Callable[[PyTree, "TreeSpec"], bool]
FlattenFuncSpec = FlattenFnSpec
FlattenFuncExactMatchSpec = FlattenFnExactMatchSpec


@dataclass(frozen=True)
class TreeSpec:
    """Description needed to rebuild one flattened tree."""

    type: Any
    context: Any = None
    children_specs: tuple["TreeSpec", ...] = ()

    @property
    def num_children(self) -> int:
        return len(self.children_specs)

    def children(self) -> tuple["TreeSpec", ...]:
        return self.children_specs

    def is_leaf(self) -> bool:
        return not self.children_specs

    def __repr__(self) -> str:
        if self.is_leaf():
            return "*"
        return (
            f"TreeSpec({self.type!r}, {self.context!r}, "
            f"{list(self.children_specs)!r})"
        )


_NODE_REGISTRY: dict[type[Any], tuple[FlattenFn, UnflattenFn]] = {}
SUPPORTED_NODES: dict[type[Any], FlattenFnSpec] = {}
SUPPORTED_NODES_EXACT_MATCH: dict[
    type[Any], FlattenFnExactMatchSpec | None
] = {}


def register_pytree_node(
    cls: type[Any], flatten_fn: FlattenFn, unflatten_fn: UnflattenFn
) -> None:
    _NODE_REGISTRY[cls] = (flatten_fn, unflatten_fn)


def _deregister_pytree_node(cls: type[Any]) -> None:
    _NODE_REGISTRY.pop(cls, None)


def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFnSpec,
    flatten_fn_exact_match_spec: FlattenFnExactMatchSpec | None = None,
) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec


def _deregister_pytree_flatten_spec(cls: type[Any]) -> None:
    SUPPORTED_NODES.pop(cls, None)
    SUPPORTED_NODES_EXACT_MATCH.pop(cls, None)


def _children_and_context(value: Any) -> tuple[type[Any], list[Any], Any] | None:
    cls = type(value)
    registered = _NODE_REGISTRY.get(cls)
    if registered is not None:
        children, context = registered[0](value)
        return cls, list(children), context
    if isinstance(value, dict):
        return dict, list(value.values()), tuple(value.keys())
    if isinstance(value, list):
        return list, list(value), None
    if isinstance(value, tuple):
        if hasattr(value, "_fields"):
            return cls, list(value), None
        return tuple, list(value), None
    return None


def tree_flatten(value: Any) -> tuple[list[Any], TreeSpec]:
    """Return leaves and a specification that preserves container shape."""

    node_info = _children_and_context(value)
    if node_info is None:
        return [value], TreeSpec(None)
    node_type, children, context = node_info
    leaves: list[Any] = []
    child_specs: list[TreeSpec] = []
    for child in children:
        child_leaves, child_spec = tree_flatten(child)
        leaves.extend(child_leaves)
        child_specs.append(child_spec)
    return leaves, TreeSpec(node_type, context, tuple(child_specs))


def _unflatten_node(spec: TreeSpec, children: list[Any]) -> Any:
    node_type = spec.type
    registered = _NODE_REGISTRY.get(node_type)
    if registered is not None:
        return registered[1](children, spec.context)
    if node_type is dict:
        return dict(zip(spec.context, children))
    if node_type is list:
        return list(children)
    if node_type is tuple:
        return tuple(children)
    if isinstance(node_type, type) and issubclass(node_type, tuple):
        if hasattr(node_type, "_make"):
            return node_type._make(children)
        return node_type(*children)
    raise TypeError(f"cannot rebuild tree node of type {node_type!r}")


def tree_unflatten(leaves: Iterable[Any], spec: TreeSpec) -> Any:
    """Rebuild a tree described by ``spec`` from its leaves."""

    iterator = iter(leaves)

    def rebuild(current: TreeSpec) -> Any:
        if current.is_leaf():
            try:
                return next(iterator)
            except StopIteration as exc:
                raise ValueError("not enough leaves for tree specification") from exc
        children = [rebuild(child) for child in current.children_specs]
        return _unflatten_node(current, children)

    result = rebuild(spec)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise ValueError("too many leaves for tree specification")


def tree_flatten_spec(value: PyTree, spec: TreeSpec) -> list[Any]:
    """Flatten ``value`` while enforcing an already-known specification."""

    if spec.is_leaf():
        return [value]
    flatten_fn = SUPPORTED_NODES.get(spec.type)
    if flatten_fn is not None:
        children = flatten_fn(value, spec)
        result: list[Any] = []
        for child, child_spec in zip(children, spec.children_specs):
            result.extend(tree_flatten_spec(child, child_spec))
        if len(children) != spec.num_children:
            raise ValueError("tree node has a different number of children")
        return result
    flat, real_spec = tree_flatten(value)
    if real_spec != spec:
        raise RuntimeError(
            f"tree specification mismatch: actual={real_spec!r}, expected={spec!r}"
        )
    return flat


def _dict_flatten_spec(d: dict[_K, _V], spec: TreeSpec) -> list[_V]:
    return [d[key] for key in spec.context]


def _list_flatten_spec(d: list[_T], spec: TreeSpec) -> list[_T]:
    return [d[index] for index in range(spec.num_children)]


def _tuple_flatten_spec(d: tuple[_T, ...], spec: TreeSpec) -> list[_T]:
    return [d[index] for index in range(spec.num_children)]


def _namedtuple_flatten_spec(d: Any, spec: TreeSpec) -> list[Any]:
    return [d[index] for index in range(spec.num_children)]


def _dict_flatten_spec_exact_match(d: dict[_K, _V], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children and tuple(d) == tuple(spec.context)


def _list_flatten_spec_exact_match(d: list[_T], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _tuple_flatten_spec_exact_match(d: tuple[_T, ...], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _namedtuple_flatten_spec_exact_match(d: Any, spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


register_pytree_flatten_spec(dict, _dict_flatten_spec, _dict_flatten_spec_exact_match)
register_pytree_flatten_spec(list, _list_flatten_spec, _list_flatten_spec_exact_match)
register_pytree_flatten_spec(
    tuple, _tuple_flatten_spec, _tuple_flatten_spec_exact_match
)
register_pytree_flatten_spec(
    namedtuple, _namedtuple_flatten_spec, _namedtuple_flatten_spec_exact_match
)


__all__ = [
    "Context",
    "FlattenFnExactMatchSpec",
    "FlattenFnSpec",
    "FlattenFuncExactMatchSpec",
    "FlattenFuncSpec",
    "PyTree",
    "SUPPORTED_NODES",
    "SUPPORTED_NODES_EXACT_MATCH",
    "TreeSpec",
    "_deregister_pytree_flatten_spec",
    "_deregister_pytree_node",
    "register_pytree_flatten_spec",
    "register_pytree_node",
    "tree_flatten",
    "tree_flatten_spec",
    "tree_unflatten",
]
