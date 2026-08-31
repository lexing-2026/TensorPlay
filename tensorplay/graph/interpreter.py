"""Node-by-node execution and symbolic graph transformation utilities."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import traceback as graph_traceback
from ._compatibility import compatibility
from ._utils import _iter_nodes, _map_arg
from .graph import Graph
from .graph_module import GraphModule
from .node import Node, map_aggregate
from .proxy import Proxy
from .tracer import Tracer

log = logging.getLogger(__name__)

__all__ = ["Interpreter", "Transformer"]


def _format_node(node: Node) -> str:
    module = getattr(node.target, "__module__", "")
    prefix = f"{module}." if module else ""
    target = getattr(node.target, "__name__", node.target)
    args = ", ".join(map(str, node.args))
    kwargs = ", ".join(f"{key}={value}" for key, value in node.kwargs.items())
    joined = ", ".join(item for item in (args, kwargs) if item)
    return f"{node.name} = {prefix}{target}({joined})"


@compatibility(is_backward_compatible=True)
class Interpreter:
    """Execute a graph one node at a time with overridable dispatch hooks."""

    def __init__(
        self,
        module: Any,
        garbage_collect_values: bool = True,
        graph: Graph | None = None,
    ) -> None:
        self.module = module
        named_modules = getattr(module, "named_modules", None)
        self.submodules = dict(named_modules()) if callable(named_modules) else {}
        self.graph = graph if graph is not None else module.graph
        self.env: dict[Node, Any] = {}
        self.name = "Interpreter"
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True
        self.args_iter: Iterator[Any] = iter(())
        self._keyword_args: dict[str, Any] = {}
        self._placeholder_defaults: dict[str, Any] = {}
        self.last_node: Node | None = None
        self.user_to_last_uses: dict[Node, list[Node]] = {}
        if garbage_collect_values:
            last_use: dict[Node, Node] = {}
            for current in reversed(self.graph.nodes):
                for value in (*_iter_nodes(current.args), *_iter_nodes(current.kwargs)):
                    if value not in last_use:
                        last_use[value] = current
                        self.user_to_last_uses.setdefault(current, []).append(value)

    @compatibility(is_backward_compatible=True)
    def run(
        self,
        *args: Any,
        initial_env: dict[Node, Any] | None = None,
        enable_io_processing: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute the graph and return the value of its output node."""

        self.env = initial_env if initial_env is not None else {}
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        self.args_iter = iter(args)
        self._keyword_args = dict(kwargs)
        self._placeholder_defaults = {
            node.name: node.meta["default"]
            for node in self.graph.placeholders
            if "default" in node.meta
        }
        for node in self.graph.nodes:
            if node in self.env:
                continue
            self.last_node = node
            try:
                value = self.run_node(node)
                self.env[node] = value
            except Exception as exc:
                if self.extra_traceback:
                    detail = f"While executing {node.format_node()}"
                    message = f"{exc.args[0] if exc.args else exc}\n\n{detail}"
                    if node.stack_trace:
                        message += f"\nOriginal traceback:\n{node.stack_trace}"
                    exc.args = (message, *exc.args[1:])
                    if isinstance(exc, KeyError):
                        raise RuntimeError(*exc.args) from exc
                raise
            if self.garbage_collect_values:
                for old_value in self.user_to_last_uses.get(node, ()):
                    self.env.pop(old_value, None)
            if node.op == "output":
                value = self.env[node]
                return self.graph.process_outputs(value) if enable_io_processing else value
        raise RuntimeError("graph has no output node")

    @compatibility(is_backward_compatible=True)
    def boxed_run(self, args_list: list[Any]) -> Any:
        """Run with a mutable positional argument list and clear that list."""

        placeholder_nodes = [node for node in self.graph.nodes if node.op == "placeholder"]
        expected = len(placeholder_nodes)
        if len(args_list) != expected:
            detail = "extra arguments" if len(args_list) > expected else "missing arguments"
            raise RuntimeError(
                f"Interpreter.boxed_run expected {expected} arguments for "
                f"placeholders but received {len(args_list)} ({detail})"
            )
        initial_env = dict(zip(placeholder_nodes, args_list))
        args_list.clear()
        return self.run(initial_env=initial_env)

    @contextmanager
    def _set_current_node(self, node: Node) -> Iterator[None]:
        with graph_traceback.set_current_meta(
            node, f"Interpreter_{self.__class__.__name__}"
        ):
            yield

    @compatibility(is_backward_compatible=True)
    def run_node(self, node: Node) -> Any:
        """Dispatch one node to its operation-specific hook."""

        log.debug("run_node %s", _format_node(node))
        with self._set_current_node(node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if not isinstance(args, tuple):
                raise AssertionError(f"Expected args to be tuple, got {type(args)}")
            if not isinstance(kwargs, dict):
                raise AssertionError(f"Expected kwargs to be dict, got {type(kwargs)}")
            return getattr(self, node.op)(node.target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        del args, kwargs
        if not isinstance(target, str):
            raise TypeError(f"placeholder target must be a string, got {type(target)!r}")
        if target.startswith("*"):
            return list(self.args_iter)
        if target in self._keyword_args:
            return self._keyword_args[target]
        try:
            return next(self.args_iter)
        except StopIteration as exc:
            if target in self._placeholder_defaults:
                return self._placeholder_defaults[target]
            raise RuntimeError(
                f"missing positional argument for parameter {target!r}"
            ) from exc

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        del args, kwargs
        if not isinstance(target, str):
            raise TypeError(f"get_attr target must be a string, got {type(target)!r}")
        return self.fetch_attr(target)

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not callable(target):
            raise TypeError(f"call_function target is not callable: {target!r}")
        return target(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not isinstance(target, str):
            raise TypeError(f"call_method target must be a string, got {type(target)!r}")
        if not args:
            raise RuntimeError("call_method node has no receiver")
        receiver, *tail = args
        return getattr(receiver, target)(*tail, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not isinstance(target, str):
            raise TypeError(f"call_module target must be a string, got {type(target)!r}")
        return self.fetch_attr(target)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        del target, kwargs
        if len(args) != 1:
            raise RuntimeError("output node must contain exactly one value")
        return args[0]

    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str) -> Any:
        """Fetch a dotted attribute path from the wrapped module."""

        if isinstance(self.module, GraphModule):
            try:
                return self.module._get_attr(target)
            except AttributeError as exc:
                raise RuntimeError(f"graph references missing attribute {target!r}") from exc
        value = self.module
        for atom in target.split("."):
            if isinstance(value, (list, tuple)) and atom.isdigit():
                value = value[int(atom)]
            elif isinstance(value, dict) and atom in value:
                value = value[atom]
            else:
                if not hasattr(value, atom):
                    raise RuntimeError(f"graph references missing attribute {target!r}")
                value = getattr(value, atom)
        return value

    @compatibility(is_backward_compatible=True)
    def fetch_args_kwargs_from_env(
        self, node: Node
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args = self.map_nodes_to_values(node.args, node)
        kwargs = self.map_nodes_to_values(node.kwargs, node)
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise AssertionError("graph arguments must remain tuple/dict containers")
        return args, kwargs

    @compatibility(is_backward_compatible=True)
    def map_nodes_to_values(self, values: Any, node: Node) -> Any:
        """Recursively replace node references with values from ``env``."""

        def load(value: Any) -> Any:
            if value not in self.env:
                raise RuntimeError(
                    f"node {node.name!r} references an unavailable value "
                    f"{value.name!r}"
                )
            return self.env[value]

        return _map_arg(values, load)


@compatibility(is_backward_compatible=True)
class Transformer(Interpreter):
    """Interpret a graph symbolically and emit a transformed graph module."""

    def __init__(self, module: GraphModule) -> None:
        super().__init__(module, garbage_collect_values=False)
        self.new_graph = Graph()
        self.new_graph.set_codegen(module.graph._codegen)
        class TransformerTracer(Tracer):
            def is_leaf_module(self, module: Any, qualified_name: str) -> bool:
                del module, qualified_name
                return True

        self.tracer = TransformerTracer()
        self.tracer.graph = self.new_graph
        self.tracer.root = module

    def placeholder(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        del kwargs
        default = args[0] if args else inspect.Parameter.empty
        return self.tracer.create_proxy(
            "placeholder", target, (), {},
        ) if default is inspect.Parameter.empty else Proxy(
            self.new_graph.placeholder(target, default_value=default), self.tracer
        )

    def get_attr(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        return self.tracer.create_proxy("get_attr", target, args, kwargs)

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        return self.tracer.create_proxy("call_function", target, args, kwargs)

    def call_method(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        return self.tracer.create_proxy("call_method", target, args, kwargs)

    def call_module(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        if not isinstance(target, str):
            raise TypeError(f"module target must be a string, got {type(target).__name__}")
        submodule = self.fetch_attr(target)
        return self.tracer.call_module(
            submodule, getattr(submodule, "forward"), args, kwargs
        )

    def run_node(self, node: Node) -> Any:
        result = super().run_node(node)
        if isinstance(result, Proxy) and node.op != "output":
            result.node.meta.update(node.meta)
            if node.type is not None:
                result.node.type = node.type
        return result

    def transform(self) -> GraphModule:
        with graph_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)

        def strip(value: Any) -> Any:
            return value.node if isinstance(value, Proxy) else value

        output = self.new_graph.output(map_aggregate(result, strip))
        old_output = self.graph.outputs[-1]
        if old_output.op != "output":
            raise AssertionError(f"Expected output node, got op={old_output.op}")
        output.meta.update(old_output.meta)
        self.new_graph.lint()
        from ._lazy_graph_module import _make_graph_module

        return _make_graph_module(self.module, self.new_graph, self.module.signature)
