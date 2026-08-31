"""Deterministic serialization helpers for graph programs."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import io
import itertools
import pickle
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, NewType

from .graph import Graph
from .graph_module import GraphModule
from .node import Node, map_arg


def _ops_filter_safe(name: str) -> bool:
    """Return whether a qualified target belongs to the portable target set."""

    return name.startswith(
        (
            "builtins.",
            "operator.",
            "_operator.",
            "math.",
            "tensorplay.",
        )
    )


def _node_metadata_key_filter_safe(key: str) -> bool:
    """Keep metadata that does not contain process-local object stacks."""

    return key not in {
        "source_fn_stack",
        "nn_module_stack",
        "fwd_source_fn_stack",
        "stack_trace",
    }


@dataclasses.dataclass
class Options:
    """Policies controlling target and metadata serialization."""

    ops_filter: Callable[[str], bool] | None = _ops_filter_safe
    node_metadata_key_filter: Callable[[str], bool] | None = (
        _node_metadata_key_filter_safe
    )
    ignore_raw_node: bool = False


def _unpickle_as_none() -> None:
    return None


def _unpickle_as_weakref(referent: object) -> weakref.ref[object]:
    return weakref.ref(referent)


def _unpickle_as_dead_weakref() -> Callable[[], None]:
    return lambda: None


@contextlib.contextmanager
def patch_pytree_map_over_slice() -> Generator[None, None, None]:
    """Register slice traversal for the duration of a serialization pass."""

    from tensorplay.utils import _pytree as pytree

    if slice in pytree.SUPPORTED_NODES:
        yield
        return

    pytree.register_pytree_node(
        slice,
        lambda value: ([value.start, value.stop, value.step], None),
        lambda values, _context: slice(*values),
    )
    try:
        yield
    finally:
        pytree._deregister_pytree_node(slice)


_UnpickleStateToken = NewType("_UnpickleStateToken", object)


class _UnpickleState:
    """State shared by all custom reducers during one load operation."""

    def __init__(self, value: Any = None) -> None:
        self.value = value


class _GraphUnpickler(pickle.Unpickler):
    """Unpickler carrying the state object referenced by graph reducers."""

    def __init__(
        self,
        file: io.BufferedIOBase,
        state: _UnpickleState | None = None,
    ) -> None:
        super().__init__(file)
        self.state = state or _UnpickleState()

    def persistent_load(self, pid: object) -> object:
        if pid == "unpickle_state":
            return self.state
        raise pickle.UnpicklingError("invalid graph serialization state")


def _resolve_attribute(root: Any, name: str) -> Any:
    for part in name.split("."):
        root = getattr(root, part)
    return root


def _qualified_target(target: Any) -> tuple[str, str] | None:
    """Find an importable name for a callable target when one exists."""

    module = getattr(target, "__module__", None)
    qualified = getattr(target, "__qualname__", None)
    if qualified is None:
        qualified = getattr(target, "__name__", None)
    if isinstance(module, str) and isinstance(qualified, str):
        if "<locals>" not in qualified:
            try:
                candidate = _resolve_attribute(importlib.import_module(module), qualified)
            except (AttributeError, ImportError, ModuleNotFoundError):
                candidate = None
            if candidate is target:
                return module, qualified

    name = getattr(target, "__name__", None)
    if not isinstance(name, str):
        return None
    try:
        import tensorplay

        c_namespace = getattr(
            getattr(tensorplay, "_C", None), "_VariableFunctions", None
        )
        if c_namespace is not None and getattr(c_namespace, name, None) is not None:
            return "tensorplay._C", f"_VariableFunctions.{name}"
    except (AttributeError, ImportError):
        pass
    return None


class _TargetPickleData(ABC):
    @classmethod
    def pickle(cls, target: Any, options: Options) -> "_TargetPickleData":
        if isinstance(target, str):
            return _StringTargetPickleData(target)

        qualified = _qualified_target(target)
        name = (
            f"{qualified[0]}.{qualified[1]}"
            if qualified is not None
            else Node._pretty_print_target(target)
        )
        if options.ops_filter is not None and not options.ops_filter(name):
            raise pickle.PicklingError(f"target is not portable: {name}")
        if qualified is not None:
            return _QualifiedTargetPickleData(*qualified)
        return _RawTargetPickleData(target)

    @abstractmethod
    def unpickle(self, state: _UnpickleState) -> Any:
        raise NotImplementedError


class _StringTargetPickleData(_TargetPickleData):
    def __init__(self, target: str) -> None:
        self.target = target

    def unpickle(self, state: _UnpickleState) -> str:
        del state
        return self.target


class _QualifiedTargetPickleData(_TargetPickleData):
    def __init__(self, module: str, qualified: str) -> None:
        self.module = module
        self.qualified = qualified

    def unpickle(self, state: _UnpickleState) -> Any:
        del state
        root = importlib.import_module(self.module)
        return _resolve_attribute(root, self.qualified)


class _RawTargetPickleData(_TargetPickleData):
    def __init__(self, target: Any) -> None:
        self.target = target

    def unpickle(self, state: _UnpickleState) -> Any:
        del state
        return self.target


def _map_node_data(value: Any, mapping: dict["_NodePickleData", Node]) -> Any:
    if isinstance(value, _NodePickleData):
        return mapping[value]
    if isinstance(value, tuple):
        mapped = [_map_node_data(item, mapping) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*mapped)
        try:
            return type(value)(mapped)
        except TypeError:
            return tuple(mapped)
    if isinstance(value, list):
        mapped = [_map_node_data(item, mapping) for item in value]
        try:
            return type(value)(mapped)
        except TypeError:
            return mapped
    if isinstance(value, dict):
        mapped = {
            key: _map_node_data(item, mapping) for key, item in value.items()
        }
        try:
            return type(value)(mapped)
        except TypeError:
            return mapped
    if isinstance(value, slice):
        return slice(
            _map_node_data(value.start, mapping),
            _map_node_data(value.stop, mapping),
            _map_node_data(value.step, mapping),
        )
    if isinstance(value, range):
        return range(
            _map_node_data(value.start, mapping),
            _map_node_data(value.stop, mapping),
            _map_node_data(value.step, mapping),
        )
    return value


class _NodePickleData:
    def __init__(
        self,
        node: Node,
        mapping: dict[Node, "_NodePickleData"],
        options: Options,
    ) -> None:
        self.args = map_arg(node.args, lambda value: mapping[value])
        self.kwargs = map_arg(node.kwargs, lambda value: mapping[value])
        self.name = node.name
        self.op = node.op
        self.target = _TargetPickleData.pickle(node.target, options)
        self.type = node.type
        self.tag = node.tag
        self.size_bytes = node.size_bytes
        self.meta = {
            key: value
            for key, value in node.meta.items()
            if (
                options.node_metadata_key_filter is None
                or options.node_metadata_key_filter(key)
            )
        }

    def unpickle(
        self,
        graph: Graph,
        mapping: dict["_NodePickleData", Node],
        state: _UnpickleState,
    ) -> Node:
        args = _map_node_data(self.args, mapping)
        kwargs = _map_node_data(self.kwargs, mapping)
        target = self.target.unpickle(state)
        node = graph.create_node(
            self.op,
            target,
            args,
            kwargs,
            self.name,
            self.type,
        )
        node.meta = dict(self.meta)
        node.tag = self.tag
        node.size_bytes = self.size_bytes
        return node


class _GraphPickleData:
    def __init__(self, graph: Graph, options: Options) -> None:
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras
        self.codegen = graph._codegen
        self.codegen_hooks = tuple(graph._codegen_hooks)
        self.co_fields = dict(graph._co_fields)

        mapping: dict[Node, _NodePickleData] = {}
        for node in graph.nodes:
            mapping[node] = _NodePickleData(node, mapping, options)
        self.nodes = tuple(mapping.values())

    def unpickle(self, module: GraphModule | None, state: _UnpickleState) -> Graph:
        graph = Graph(module, self.tracer_cls, self.tracer_extras)
        mapping: dict[_NodePickleData, Node] = {}
        for node_data in self.nodes:
            mapping[node_data] = node_data.unpickle(graph, mapping, state)
        graph._codegen = self.codegen
        graph._codegen_hooks = list(self.codegen_hooks)
        graph._co_fields = dict(self.co_fields)
        return graph


def _module_state(module: GraphModule) -> tuple[type[GraphModule], Any, dict[str, Any]]:
    state = (
        module.__getstate__()
        if hasattr(module, "__getstate__")
        else module.__dict__.copy()
    )
    state = dict(state)
    root = state.pop("_root", getattr(module, "root", None))
    state.pop("_graph", None)
    state.pop("forward", None)
    state.pop("_compiled_forward", None)
    state.pop("_compiled_impl", None)
    state.pop("_python_code", None)
    return type(module), root, state


def _rebuild_graph_module(
    module_type: type[GraphModule],
    root: Any,
    graph_data: _GraphPickleData,
    state: dict[str, Any],
    unpickle_state: _UnpickleState,
) -> GraphModule:
    module = object.__new__(module_type)
    state = dict(state)
    state["_root"] = root
    setstate = getattr(module, "__setstate__", None)
    if callable(setstate):
        setstate(state)
    else:
        module.__dict__.update(state)
    graph = graph_data.unpickle(module, unpickle_state)
    object.__setattr__(module, "_graph", graph)
    graph.owning_module = module
    object.__setattr__(module, "_compiled_forward", None)
    object.__setattr__(module, "_compiled_impl", None)
    object.__setattr__(module, "_python_code", None)
    module.recompile()
    return module


class GraphPickler(pickle.Pickler):
    """Pickler that serializes graph topology without graph back-reference cycles."""

    _PASSTHROUGH_TYPES = frozenset({int, float, str, bytes, bool, type(None)})

    def __init__(self, file: io.BufferedIOBase, options: Options | None = None) -> None:
        super().__init__(file)
        self.options = options or Options()
        self._unpickle_state = _UnpickleStateToken(object())

    def reducer_override(self, obj: object) -> Any:
        if type(obj) in self._PASSTHROUGH_TYPES:
            return NotImplemented
        if isinstance(obj, GraphModule):
            module_type, root, state = _module_state(obj)
            return (
                _rebuild_graph_module,
                (
                    module_type,
                    root,
                    _GraphPickleData(obj.graph, self.options),
                    state,
                    self._unpickle_state,
                ),
            )
        if isinstance(obj, Graph):
            return (
                _rebuild_graph,
                (_GraphPickleData(obj, self.options), self._unpickle_state),
            )
        if isinstance(obj, Node):
            if self.options.ignore_raw_node:
                return _unpickle_as_none, ()
            raise AssertionError("unexpected raw graph node during serialization")
        if isinstance(obj, weakref.ReferenceType):
            referent = obj()
            if referent is None:
                return _unpickle_as_dead_weakref, ()
            return _unpickle_as_weakref, (referent,)
        return NotImplemented

    def persistent_id(self, obj: object) -> str | None:
        if obj is self._unpickle_state:
            return "unpickle_state"
        return None

    @classmethod
    def dumps(cls, obj: object, options: Options | None = None) -> bytes:
        with patch_pytree_map_over_slice(), io.BytesIO() as stream:
            cls(stream, options).dump(obj)
            return stream.getvalue()

    @staticmethod
    def loads(data: bytes, **kwargs: Any) -> object:
        state = _UnpickleState(kwargs.get("state", kwargs.get("fake_mode")))
        with patch_pytree_map_over_slice(), io.BytesIO(data) as stream:
            return _GraphUnpickler(stream, state).load()

    @classmethod
    def debug_dumps(
        cls,
        obj: object,
        options: Options | None = None,
        *,
        max_depth: int = 80,
        max_iter_items: int = 50,
        verbose: bool = True,
    ) -> str | None:
        """Return the path to the first value that cannot be serialized."""

        options = options or Options()
        pickler = cls(io.BytesIO(), options)
        visited: set[int] = set()

        def log(message: str) -> None:
            if verbose:
                print(message)

        def failure(value: Any) -> BaseException | None:
            try:
                cls.dumps(value, options)
            except BaseException as exc:
                return exc
            return None

        def walk(value: Any, path: str, depth: int) -> str | None:
            if depth > max_depth:
                log(f"{'  ' * depth}depth limit at {path} ({type(value)})")
                return f"{path} (depth_limit)"

            identity = id(value)
            if identity in visited:
                return None
            visited.add(identity)

            indent = "  " * depth
            log(f"{indent}walking: {path} ({type(value)})")
            exc = failure(value)
            if exc is None:
                log(f"{indent}serializes successfully in isolation")
                return None
            log(f"{indent}serialization failed: {type(value)} -> {exc}")

            if isinstance(value, dict):
                for key, item in value.items():
                    bad = walk(key, f"{path}.key[{key!r}]", depth + 1)
                    if bad:
                        return bad
                    bad = walk(item, f"{path}[{key!r}]", depth + 1)
                    if bad:
                        return bad
                return path

            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    bad = walk(item, f"{path}[{index}]", depth + 1)
                    if bad:
                        return bad
                return path

            if isinstance(value, (set, frozenset)):
                for index, item in enumerate(value):
                    bad = walk(item, f"{path}[{index}]", depth + 1)
                    if bad:
                        return bad
                return path

            if hasattr(value, "__iter__") and type(value).__name__.endswith("iterator"):
                try:
                    prefix = list(itertools.islice(iter(value), max_iter_items + 1))
                except Exception:
                    prefix = None
                if prefix is not None:
                    if len(prefix) > max_iter_items:
                        log(
                            f"{indent}iterator has more than {max_iter_items} items; "
                            "only the prefix is inspected"
                        )
                        prefix = prefix[:max_iter_items]
                    for index, item in enumerate(prefix):
                        bad = walk(item, f"{path}[{index}]", depth + 1)
                        if bad:
                            return bad
                    return path

            try:
                reduced = pickler.reducer_override(value)
                log(f"{indent}custom reducer -> {type(reduced)}")
            except Exception as reducer_error:
                log(f"{indent}custom reducer failed: {reducer_error}")
                return path
            if reduced is not NotImplemented:
                _, arguments = reduced
                for index, argument in enumerate(arguments):
                    bad = walk(argument, f"{path}.reduce_args[{index}]", depth + 1)
                    if bad:
                        return bad

            if dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    try:
                        item = getattr(value, field.name)
                    except Exception:
                        return f"{path}.{field.name}"
                    bad = walk(item, f"{path}.{field.name}", depth + 1)
                    if bad:
                        return bad
                return path

            getstate = getattr(value, "__getstate__", None)
            if callable(getstate):
                try:
                    state = getstate()
                except Exception:
                    return f"{path}.__getstate__()"
                bad = walk(state, f"{path}.__getstate__()", depth + 1)
                if bad:
                    return bad

            if hasattr(value, "__dict__"):
                for name, item in vars(value).items():
                    bad = walk(item, f"{path}.{name}", depth + 1)
                    if bad:
                        return bad
                return path

            slots = getattr(type(value), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            if slots:
                for name in slots:
                    if hasattr(value, name):
                        bad = walk(getattr(value, name), f"{path}.{name}", depth + 1)
                        if bad:
                            return bad
                return path

            try:
                reduced = value.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
            except Exception:
                try:
                    reduced = value.__reduce__()
                except Exception:
                    return path
            if isinstance(reduced, tuple):
                for index, item in enumerate(reduced):
                    if item is None:
                        continue
                    bad = walk(item, f"{path}.__reduce__[{index}]", depth + 1)
                    if bad:
                        return bad
            return path

        return walk(obj, "root", 0)


def _rebuild_graph(
    graph_data: _GraphPickleData,
    state: _UnpickleState,
) -> Graph:
    return graph_data.unpickle(None, state)


__all__ = [
    "GraphPickler",
    "Options",
    "_GraphUnpickler",
    "_UnpickleState",
    "_node_metadata_key_filter_safe",
    "_ops_filter_safe",
    "patch_pytree_map_over_slice",
]
