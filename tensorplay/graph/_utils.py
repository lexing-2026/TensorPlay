from __future__ import annotations

import inspect
import keyword
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterable, Optional


class _LazyString:
    """String-like value whose formatting work is postponed until display."""

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], str]) -> None:
        self._factory = factory

    def __str__(self) -> str:
        return self._factory()

    def __repr__(self) -> str:
        return str(self)


def lazy_format_graph_code(
    name: str, graph_module: Any, maybe_id: int | None = None, **kwargs: Any
) -> _LazyString:
    """Return a lazily rendered description of a graph module."""

    del kwargs
    label = f"{name} {maybe_id}" if maybe_id is not None else name

    def render() -> str:
        code = getattr(graph_module, "code", "")
        return _format_graph_code(
            f"===== {label} =====\n",
            getattr(getattr(graph_module, "forward", None), "__code__", None),
            code,
        )

    return _LazyString(render)


def _format_graph_code(name: str, filename: Any, graph_str: str) -> str:
    filename_str = getattr(filename, "co_filename", filename)
    return f"TRACED GRAPH\n {name} {filename_str} {graph_str}\n"


def first_call_function_nn_module_stack(graph: Any) -> dict[str, Any] | None:
    """Return module-stack metadata from the first function node that has it."""

    for node in graph.nodes:
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            return node.meta["nn_module_stack"]
    return None


def get_node_context(node: Any, num_nodes: int = 2) -> str:
    """Return a short source-order context ending at ``node``."""

    nodes = list(node.graph.nodes)
    try:
        index = nodes.index(node)
    except ValueError:
        return str(node)
    start = max(0, index - max(1, num_nodes) + 1)
    return "\n".join(str(item) for item in nodes[start : index + 1])


class GraphCaptureError(RuntimeError):
    """Raised when Python code cannot be represented by the current graph."""


_compiling: ContextVar[bool] = ContextVar("tensorplay_graph_compiling", default=False)

_capture_disabled: ContextVar[bool] = ContextVar(
    "tensorplay_graph_capture_disabled", default=False
)

# The active tracer is exposed through a context variable so small graph
# markers can participate in capture even when they have no tensor argument.
# Keeping this state thread-local is important for nested captures and for
# concurrent compiler workers.
_active_tracer: ContextVar[Any] = ContextVar(
    "tensorplay_graph_active_tracer", default=None
)


def get_active_tracer() -> Any:
    return _active_tracer.get()


def _native_capture_state(
    entering: bool,
    *,
    compiling: bool = False,
    exporting: bool = False,
    disabled: bool = False,
) -> bool:
    """Update the thread-local state owned by the native graph runtime."""

    try:
        import tensorplay

        native = getattr(getattr(tensorplay, "_C", None), "_stax", None)
        operation = getattr(
            native,
            "capture_state_enter" if entering else "capture_state_exit",
            None,
        )
    except (AttributeError, ImportError):
        return False
    if operation is None:
        return False
    operation(compiling, exporting, disabled)
    return True


@contextmanager
def compiler_context(*, require_native: bool = False) -> Any:
    token = _compiling.set(True)
    native_entered = False
    try:
        native_entered = _native_capture_state(True, compiling=True)
        if require_native and not native_entered:
            raise GraphCaptureError(
                "TensorPlay native capture state is unavailable"
            )
        yield
    finally:
        if native_entered:
            _native_capture_state(False, compiling=True)
        _compiling.reset(token)


def _map_arg(value: Any, fn: Callable[[Any], Any]) -> Any:
    from .node import Node
    from .proxy import Proxy

    if isinstance(value, (Node, Proxy)):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_arg(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_arg(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_arg(item, fn) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(
            _map_arg(value.start, fn),
            _map_arg(value.stop, fn),
            _map_arg(value.step, fn),
        )
    return value


def _iter_proxies(value: Any) -> Iterable["Proxy"]:
    from .proxy import Proxy

    if isinstance(value, Proxy):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_proxies(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_proxies(item)
        return
    if isinstance(value, slice):
        yield from _iter_proxies(value.start)
        yield from _iter_proxies(value.stop)
        yield from _iter_proxies(value.step)


_TRACE_DEPTH = 0


def capturing() -> bool:
    return _TRACE_DEPTH > 0


def capture_call(
    target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Optional["Proxy"]:
    found = False
    for value in args:
        for _proxy in _iter_proxies(value):
            found = True
            break
        if found:
            break
    if not found and kwargs:
        for value in kwargs.values():
            for _proxy in _iter_proxies(value):
                found = True
                break
            if found:
                break
    if not found:
        return None
    proxies = list(_iter_proxies(args))
    proxies.extend(_iter_proxies(kwargs))
    if not proxies:
        return None
    if _capture_disabled.get():
        raise GraphCaptureError("graph capture is disabled for this operation")
    tracer = proxies[0].tracer
    if any(proxy.tracer is not tracer for proxy in proxies[1:]):
        raise GraphCaptureError("cannot combine proxies from different traces")
    return tracer.create_proxy("call_function", target, args, kwargs)


def _iter_nodes(value: Any) -> Iterable["Node"]:
    from .node import Node

    if isinstance(value, Node):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_nodes(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_nodes(item)
        return
    if isinstance(value, slice):
        yield from _iter_nodes(value.start)
        yield from _iter_nodes(value.stop)
        yield from _iter_nodes(value.step)


def _snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def gate_outcome(kind: str, sample: Any) -> Any:
    if kind == "iter":
        return ("iter",) + tuple(sample)
    item = sample.item() if hasattr(sample, "item") else sample
    if kind == "bool":
        return bool(item)
    if kind in ("int", "index"):
        return int(item)
    if kind == "float":
        return float(item)
    raise GraphCaptureError(f"unknown control-flow gate kind {kind!r}")


_SANITIZED_NAMES: Dict[str, str] = {}


def _sanitize_name(name: str) -> str:
    cached = _SANITIZED_NAMES.get(name)
    if cached is not None:
        return cached
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized or sanitized[0].isdigit() or keyword.iskeyword(sanitized):
        sanitized = f"_{sanitized}"
    if len(_SANITIZED_NAMES) < 8192:
        _SANITIZED_NAMES[name] = sanitized
    return sanitized


_TARGET_STRINGS: Dict[Any, str] = {}


def _target_to_str(target: Any) -> str:
    try:
        cached = _TARGET_STRINGS.get(target)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    if isinstance(target, str):
        result = _snake_case(target.split(".")[-1])
    elif callable(target):
        atom = getattr(target, "__name__", None) or type(target).__name__
        result = _snake_case(str(atom))
    else:
        result = type(target).__name__
    if len(_TARGET_STRINGS) < 8192:
        try:
            _TARGET_STRINGS[target] = result
        except TypeError:
            pass
    return result


def _format_target(target: Any) -> str:
    name = getattr(target, "__name__", None)
    if isinstance(target, str):
        return target
    if callable(target) and name:
        module = getattr(target, "__module__", "") or ""
        if module and module != "builtins":
            return f"{module}.{name}"
        return str(name)
    if name:
        return str(name)
    return repr(target)
