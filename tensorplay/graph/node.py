from __future__ import annotations

import builtins
import copy
import inspect
import operator
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Dict, Optional, Tuple

from ._utils import GraphCaptureError, _iter_nodes

Argument = Any
Target = Any


def _type_repr(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, type):
        if value.__module__ == "builtins":
            return value.__qualname__
        return f"{value.__module__}.{value.__qualname__}"
    origin = getattr(value, "__origin__", None)
    args = getattr(value, "__args__", ())
    if origin is not None and args:
        return f"{_type_repr(origin)}[{', '.join(_type_repr(arg) for arg in args)}]"
    return repr(value)


def _get_qualified_name(func: Callable[..., Any]) -> str:
    """Return a stable qualified name for a callable graph target."""

    name = getattr(func, "__name__", None)
    if name is None:
        return repr(func)
    if getattr(builtins, name, None) is func:
        return name
    if name == "<lambda>":
        try:
            name = inspect.getsource(func).split("=", 1)[0].strip()
        except (OSError, IOError, TypeError):
            raise RuntimeError("unable to represent an anonymous graph target") from None
    module = getattr(func, "__module__", None)
    if not module:
        module = type(func).__module__
    return f"{module}.{name}"


def _format_arg(arg: object, max_list_len: float = float("inf")) -> str:
    """Render a graph argument using the graph notation used by diagnostics."""

    custom = getattr(arg, "_custom_graph_repr_fn", None)
    if callable(custom):
        return str(custom())
    if isinstance(arg, list):
        items = ", ".join(
            _format_arg(value) for index, value in enumerate(arg) if index < max_list_len
        )
        suffix = "" if len(arg) <= max_list_len else f", ...[total_len={len(arg)}]"
        return f"[{items}{suffix}]"
    if isinstance(arg, tuple):
        items = ", ".join(
            _format_arg(value) for index, value in enumerate(arg) if index < max_list_len
        )
        suffix = "" if len(arg) <= max_list_len else f", ...[total_len={len(arg)}]"
        comma = "," if len(arg) == 1 else ""
        return f"({items}{comma}{suffix})"
    if isinstance(arg, dict):
        return "{" + ", ".join(
            f"{key}: {_format_arg(value)}" for key, value in arg.items()
        ) + "}"
    if isinstance(arg, Node):
        return "%" + str(arg)
    if isinstance(arg, slice):
        return (
            f"slice({_format_arg(arg.start)}, {_format_arg(arg.stop)}, "
            f"{_format_arg(arg.step)})"
        )
    if isinstance(arg, type) and not isinstance(arg, types.GenericAlias):
        if arg.__module__ == "builtins":
            return arg.__qualname__
        return f"{arg.__module__}.{arg.__qualname__}"
    if arg is Ellipsis:
        return "..."
    if isinstance(arg, types.FunctionType):
        return _get_qualified_name(arg)
    return repr(arg)


def _same_target(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return isinstance(result, bool) and result


def _iter_aggregate(value: Any) -> Iterable[Any]:
    if isinstance(value, Node):
        yield value
        return
    if isinstance(value, tuple | list):
        for item in value:
            yield from _iter_aggregate(item)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_aggregate(item)
        return
    if isinstance(value, slice):
        yield from _iter_aggregate(value.start)
        yield from _iter_aggregate(value.stop)
        yield from _iter_aggregate(value.step)
        return
    if isinstance(value, range):
        yield from _iter_aggregate(value.start)
        yield from _iter_aggregate(value.stop)
        yield from _iter_aggregate(value.step)


def map_arg(value: Any, fn: Callable[[Node], Any]) -> Any:
    """Apply ``fn`` to every graph node in an argument structure."""

    if not callable(fn):
        raise AssertionError("map_arg requires a callable")
    if isinstance(value, Node):
        return fn(value)
    try:
        from .proxy import Proxy

        if isinstance(value, Proxy):
            return fn(value.node)
    except ImportError:
        pass
    if isinstance(value, tuple):
        mapped = [map_arg(item, fn) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*mapped)
        try:
            return type(value)(mapped)
        except TypeError:
            return tuple(mapped)
    if isinstance(value, list):
        mapped = [map_arg(item, fn) for item in value]
        try:
            return type(value)(mapped)
        except TypeError:
            return mapped
    if isinstance(value, Mapping):
        mapped = {key: map_arg(item, fn) for key, item in value.items()}
        try:
            return type(value)(mapped)
        except TypeError:
            return mapped
    if isinstance(value, slice):
        return slice(
            map_arg(value.start, fn),
            map_arg(value.stop, fn),
            map_arg(value.step, fn),
        )
    if isinstance(value, range):
        return range(
            map_arg(value.start, fn),
            map_arg(value.stop, fn),
            map_arg(value.step, fn),
        )
    return value


def map_aggregate(value: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to every leaf while preserving the argument structure."""

    if isinstance(value, Node):
        return fn(value)
    if isinstance(value, tuple):
        mapped = [map_aggregate(item, fn) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*mapped)
        try:
            return type(value)(mapped)
        except TypeError:
            return tuple(mapped)
    if isinstance(value, list):
        mapped = [map_aggregate(item, fn) for item in value]
        try:
            return type(value)(mapped)
        except TypeError:
            return mapped
    if isinstance(value, Mapping):
        mapped = {key: map_aggregate(item, fn) for key, item in value.items()}
        try:
            return type(value)(mapped)
        except TypeError:
            return mapped
    if isinstance(value, slice):
        return slice(
            map_aggregate(value.start, fn),
            map_aggregate(value.stop, fn),
            map_aggregate(value.step, fn),
        )
    if isinstance(value, range):
        return range(
            map_aggregate(value.start, fn),
            map_aggregate(value.stop, fn),
            map_aggregate(value.step, fn),
        )
    return fn(value)


_side_effectful_targets: set[Any] = {operator.setitem, builtins.setattr, builtins.delattr}


def has_side_effect(target: Any) -> bool:
    """Return whether a callable is marked as changing observable state."""

    try:
        if target in _side_effectful_targets:
            return True
    except TypeError:
        pass
    name = getattr(target, "__name__", "")
    return bool(
        getattr(target, "_tensorplay_effectful", False)
        or (isinstance(name, str) and name.endswith("_"))
    )


class _ReplacementResult(list["Node"]):
    """List result that also retains the historical count comparison."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return len(self) == other
        return super().__eq__(other)


class _UserSet:
    """An insertion-ordered set with the mapping helpers used by graph tools."""

    __slots__ = ("_items",)

    def __init__(self, values: Iterable[Node] = ()) -> None:
        self._items: dict[Node, None] = {}
        self.update(values)

    def add(self, value: Node) -> None:
        self._items[value] = None

    def discard(self, value: Node) -> None:
        self._items.pop(value, None)

    def remove(self, value: Node) -> None:
        del self._items[value]

    def clear(self) -> None:
        self._items.clear()

    def pop(self, value: Node | None = None, default: Any = ...) -> Node:
        if value is None:
            if not self._items:
                if default is ...:
                    raise KeyError("pop from an empty user set")
                return default
            return self._items.popitem()[0]
        if default is ...:
            self._items.pop(value)
        else:
            self._items.pop(value, default)
        return value

    def popitem(self) -> tuple[Node, None]:
        return self._items.popitem()

    def copy(self) -> "_UserSet":
        return _UserSet(self._items)

    def update(self, values: Iterator[Node]) -> None:
        for value in values:
            self.add(value)

    def setdefault(self, value: Node, unused: None = None) -> None:
        self._items.setdefault(value, unused)

    def keys(self):
        return self._items.keys()

    def items(self):
        return self._items.items()

    def values(self):
        return self._items.values()

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, value: object) -> bool:
        return value in self._items

    def __getitem__(self, value: Node) -> None:
        return self._items[value]

    def __setitem__(self, value: Node, unused: None) -> None:
        self._items[value] = unused

    def __delitem__(self, value: Node) -> None:
        del self._items[value]

    def __repr__(self) -> str:
        return repr(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _UserSet):
            return self._items == other._items
        if isinstance(other, set):
            return set(self._items) == other
        if isinstance(other, dict):
            return self._items == other
        return NotImplemented

    def __bool__(self) -> bool:
        return bool(self._items)


class Node:
    """A value definition in a mutable, topologically ordered graph."""

    __slots__ = (
        "graph",
        "name",
        "op",
        "target",
        "_args",
        "_kwargs",
        "_input_nodes",
        "users",
        "meta",
        "type",
        "tag",
        "size_bytes",
        "_repr_fn",
        "_sort_key",
        "_erased",
    )

    _LEGAL_OPS = {
        "placeholder",
        "call_method",
        "call_module",
        "call_function",
        "get_attr",
        "output",
    }

    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        return_type: Any | None = None,
    ) -> None:
        if op not in self._LEGAL_OPS:
            raise GraphCaptureError(f"unsupported graph operation kind: {op!r}")
        if op == "call_function" and not callable(target):
            raise TypeError("call_function targets must be callable")
        if op != "call_function" and not isinstance(target, str):
            raise TypeError(f"{op} targets must be strings")
        # Install the structural fields before any graph bookkeeping can
        # observe this object.  Attribute hooks are for mutations of a live
        # node, not for construction.
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "_args", ())
        object.__setattr__(self, "_kwargs", {})
        object.__setattr__(self, "_input_nodes", {})
        object.__setattr__(self, "users", _UserSet())
        object.__setattr__(self, "meta", {})
        object.__setattr__(self, "type", return_type)
        object.__setattr__(self, "tag", None)
        object.__setattr__(self, "size_bytes", None)
        object.__setattr__(self, "_repr_fn", None)
        object.__setattr__(self, "_sort_key", None)
        object.__setattr__(self, "_erased", False)
        self._update_args_kwargs(tuple(args), dict(kwargs or {}))

    __hash__ = object.__hash__

    def __getstate__(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "_args": self._args,
            "_kwargs": self._kwargs,
            "_input_nodes": self._input_nodes,
            "users": self.users,
            "meta": self.meta,
            "type": self.type,
            "tag": self.tag,
            "size_bytes": self.size_bytes,
            "_repr_fn": self._repr_fn,
            "_sort_key": self._sort_key,
            "_erased": self._erased,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            if name == "users" and isinstance(value, dict):
                value = _UserSet(value)
            object.__setattr__(self, name, value)
        if not hasattr(self, "tag"):
            self.tag = None
        if not hasattr(self, "size_bytes"):
            self.size_bytes = None
        if not hasattr(self, "_erased"):
            self._erased = False

    @property
    def args(self) -> tuple[Any, ...]:
        return self._args

    @args.setter
    def args(self, value: tuple[Any, ...]) -> None:
        self._update_args_kwargs(tuple(value), self._kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value: dict[str, Any]) -> None:
        self._update_args_kwargs(self._args, dict(value))

    def _update_args_kwargs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if not isinstance(args, tuple):
            raise AssertionError(f"args must be a tuple, got {type(args)}")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")
        for input_node in self._input_nodes:
            input_node.users.discard(self)
        input_nodes: dict[Node, None] = {}
        for input_node in (*_iter_aggregate(args), *_iter_aggregate(kwargs)):
            input_nodes.setdefault(input_node)
        self._args = args
        self._kwargs = kwargs
        self._input_nodes = input_nodes
        for input_node in input_nodes:
            input_node.users.add(self)

    @property
    def all_input_nodes(self) -> list[Node]:
        return list(self._input_nodes)

    @property
    def _input_nodes_list(self) -> list[Node]:
        return self.all_input_nodes

    @property
    def next(self) -> Node | None:
        if self.graph is None:
            return None
        index = getattr(self.graph, "_index", {}).get(self)
        if index is not None:
            nodes = self.graph.nodes
            return nodes[index + 1] if index + 1 < len(nodes) else None
        try:
            return self.graph.nodes[self.graph.nodes.index(self) + 1]
        except (ValueError, IndexError):
            return None

    @property
    def prev(self) -> Node | None:
        if self.graph is None:
            return None
        index = getattr(self.graph, "_index", {}).get(self)
        if index is not None:
            return self.graph.nodes[index - 1] if index else None
        try:
            index = self.graph.nodes.index(self)
        except ValueError:
            return None
        return self.graph.nodes[index - 1] if index else None

    def prepend(self, node: Node) -> None:
        if self.graph is None or node.graph is not self.graph:
            raise GraphCaptureError("nodes must belong to the same graph")
        if node is self:
            return
        graph = self.graph
        old = tuple(graph.nodes)
        list.remove(graph.nodes, node)
        list.insert(graph.nodes, graph.nodes.index(self), node)
        graph._sync_nodes(graph.nodes, old)

    def append(self, node: Node) -> None:
        if self.graph is None or node.graph is not self.graph:
            raise GraphCaptureError("nodes must belong to the same graph")
        if node is self:
            return
        graph = self.graph
        old = tuple(graph.nodes)
        list.remove(graph.nodes, node)
        list.insert(graph.nodes, graph.nodes.index(self) + 1, node)
        graph._sync_nodes(graph.nodes, old)

    def update_arg(self, index: int, value: Any) -> None:
        args = list(self.args)
        args[index] = value
        self.args = tuple(args)

    def insert_arg(self, index: int, value: Any) -> None:
        if not 0 <= index <= len(self.args):
            raise IndexError(f"argument index out of range: {index}")
        args = list(self.args)
        args.insert(index, value)
        self.args = tuple(args)

    def update_kwarg(self, key: str, value: Any) -> None:
        kwargs = dict(self.kwargs)
        kwargs[key] = value
        self.kwargs = kwargs

    @property
    def stack_trace(self) -> str | None:
        return self.meta.get("stack_trace")

    @stack_trace.setter
    def stack_trace(self, value: str | None) -> None:
        if value is None:
            self.meta.pop("stack_trace", None)
        else:
            self.meta["stack_trace"] = value

    def replace_input_with(self, old: Node, new: Node) -> None:
        if old is new:
            return
        if self.graph is None or old.graph is not self.graph or new.graph is not self.graph:
            raise GraphCaptureError("replacement nodes must belong to the same graph")
        owner = self.graph.owning_module
        for hook in getattr(owner, "_replace_hooks", ()):
            hook(old=old, new=new.name, user=self)
        self._replace_input_with(old, new)

    def _replace_input_with(self, old: Node, new: Node) -> None:
        self._update_args_kwargs(
            map_arg(self.args, lambda value: new if value is old else value),
            map_arg(self.kwargs, lambda value: new if value is old else value),
        )

    def replace_all_uses_with(
        self,
        replace_with: Node,
        delete_user_cb: Callable[[Node], bool] | None = None,
        *,
        propagate_meta: bool = False,
    ) -> _ReplacementResult:
        if replace_with is self:
            raise GraphCaptureError("cannot replace uses of a node with itself")
        if self.graph is None or replace_with.graph is not self.graph:
            raise GraphCaptureError("replacement nodes must belong to the same graph")
        if propagate_meta:
            if replace_with.meta:
                raise AssertionError("replacement target must not already contain metadata")
            replace_with.meta.update(self.meta)
        positions = getattr(self.graph, "_index", {})
        replacement_position = positions.get(replace_with)
        result = _ReplacementResult()
        for user in list(self.users):
            if delete_user_cb is not None and not delete_user_cb(user):
                continue
            if replacement_position is not None and positions.get(user, -1) <= replacement_position:
                raise GraphCaptureError(
                    f"cannot use {replace_with.name} to replace {self.name} "
                    f"in {user.name}: it appears later in the graph"
                )
            owner = self.graph.owning_module
            for hook in getattr(owner, "_replace_hooks", ()):
                hook(old=self, new=replace_with.name, user=user)
            result.append(user)
            user._replace_input_with(self, replace_with)
        return result

    def is_impure(self, impure_random: bool = True) -> bool:
        if self.op in {"placeholder", "output"}:
            return True
        if self.meta.get("side_effect") or self.meta.get("is_impure"):
            return True
        if self.op == "call_module":
            owner = self.graph.owning_module if self.graph is not None else None
            if owner is None:
                return True
            try:
                module = owner.get_submodule(self.target)
            except (AttributeError, KeyError):
                return True
            return bool(getattr(module, "_is_impure", False))
        if self.op == "call_method":
            return isinstance(self.target, str) and self.target.endswith("_")
        if self.op == "call_function":
            if has_side_effect(self.target):
                return True
            if impure_random:
                name = getattr(self.target, "__name__", "")
                module = getattr(self.target, "__module__", "")
                if "random" in str(name).lower() or "random" in str(module).lower():
                    return True
        return False

    def normalized_arguments(
        self,
        root: Any,
        arg_types: tuple[Any, ...] | None = None,
        kwarg_types: dict[str, Any] | None = None,
        normalize_to_only_use_kwargs: bool = False,
    ) -> Any:
        from .operator_schemas import normalize_function, normalize_module

        if self.op == "call_function":
            return normalize_function(
                self.target,
                self.args,
                self.kwargs,
                arg_types,
                kwarg_types,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )
        if self.op == "call_module":
            return normalize_module(
                root,
                self.target,
                self.args,
                self.kwargs,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )
        return None

    @staticmethod
    def _pretty_print_target(target: Any) -> str:
        if isinstance(target, str):
            return target
        module = getattr(target, "__module__", None)
        name = getattr(target, "__qualname__", getattr(target, "__name__", None))
        if module and name:
            if module == "_operator":
                module = "operator"
            return f"{module}.{name}"
        return repr(target)

    def format_node(
        self,
        placeholder_names: list[str] | None = None,
        maybe_return_typename: list[str] | None = None,
        *,
        include_tensor_metadata: bool = False,
    ) -> str | None:
        if self.op == "placeholder":
            if placeholder_names is not None:
                text = str(self.target)
                if self.type is not None:
                    text += f": {_type_repr(self.type)}"
                if self.args:
                    text += f" = {self.args[0]!r}"
                placeholder_names.append(text)
                return None
            default = f"(default={self.args[0]!r})" if self.args else ""
            return (
                f"%{self.name} : [num_users={len(self.users)}] = "
                f"placeholder[target={self.target}]{default}"
            )
        if self.op == "get_attr":
            return (
                f"%{self.name} : [num_users={len(self.users)}] = "
                f"get_attr[target={self._pretty_print_target(self.target)}]"
            )
        if self.op == "output":
            if self.type is not None and maybe_return_typename is not None:
                maybe_return_typename[:] = [f" -> {_type_repr(self.type)}"]
            return f"return {self.args[0]!r}"
        metadata = ""
        if include_tensor_metadata:
            value = self.meta.get("val", self.meta.get("tensor_meta"))
            shape = getattr(value, "shape", None)
            if shape is not None:
                shape = shape() if callable(shape) else shape
                try:
                    metadata = f" shape={tuple(shape)!r}"
                except TypeError:
                    pass
        return (
            f"%{self.name}{metadata} : [num_users={len(self.users)}] = "
            f"{self.op}[target={self._pretty_print_target(self.target)}]"
            f"(args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})"
        )

    def _rename(self, new_name: str) -> None:
        if self.graph is None:
            raise GraphCaptureError("cannot rename an erased node")
        if new_name == self.name:
            return
        self.graph._live_names.discard(self.name)
        candidate = self.graph._create_unique_name(new_name)
        object.__setattr__(self, "name", candidate)
        self.graph._live_names.add(candidate)
        self.graph._graph_namespace._rename_object(self, candidate)

    def erase_node(self) -> None:
        if self.graph is None:
            raise GraphCaptureError(f"{self.name} has already been erased")
        self.graph.erase_node(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "name" and hasattr(self, "name"):
            self._rename(str(value))
            return
        graph = getattr(self, "graph", None)
        has_structural_fields = hasattr(self, "op") and hasattr(self, "target")
        graph_index = getattr(graph, "_index", None)
        mounted = False
        if graph is not None and has_structural_fields and not getattr(self, "_erased", False):
            if isinstance(graph_index, dict):
                mounted = self in graph_index
            else:
                try:
                    mounted = self in graph.nodes
                except (AttributeError, TypeError):
                    mounted = False
        tracked = (
            name in {"op", "target"}
            and graph is not None
            and hasattr(graph, "_find_nodes_lookup_table")
            and has_structural_fields
            and mounted
        )
        if tracked:
            graph._find_nodes_lookup_table.remove(self)
        object.__setattr__(self, name, value)
        if tracked:
            graph._find_nodes_lookup_table.insert(self)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (self._sort_key if self._sort_key is not None else id(self)) < (
            other._sort_key if other._sort_key is not None else id(other)
        )

    def __repr__(self) -> str:
        if self._repr_fn is not None:
            return self._repr_fn(self)
        return self.name

    __str__ = __repr__


__all__ = [
    "Argument",
    "Target",
    "Node",
    "map_arg",
    "map_aggregate",
    "has_side_effect",
]
