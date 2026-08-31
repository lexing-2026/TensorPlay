from __future__ import annotations

import builtins
import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import tensorplay as tp
from tensorplay import Tensor, nn

from ..graph import Graph
from ..proxy import Proxy
from ..tracer import Tracer

_C = TypeVar("_C", bound=Callable[..., Any])

__all__ = [
    "check_has_proxy",
    "MetaAttribute",
    "MetaDeviceAttribute",
    "MetaProxy",
    "MetaTracer",
    "embedding_override",
    "functional_relu_override",
    "gen_constructor_wrapper",
    "manual_meta_overrides",
    "nn_layernorm_override",
    "proxys_to_metas",
    "symbolic_trace",
    "torch_abs_override",
    "torch_nn_relu_override",
    "torch_relu_override",
    "torch_where_override",
]


def _shape(value: Any) -> tuple[Any, ...]:
    result = getattr(value, "shape", ())
    if callable(result):
        result = result()
    return tuple(result)


def embedding_override(module: Any, input: Any) -> Any:
    return tp.empty((*_shape(input), _shape(module.weight)[-1]), dtype=module.weight.dtype)


def nn_layernorm_override(module: Any, input: Any) -> Any:
    del module
    return input


def torch_relu_override(value: Any) -> Any:
    return value


def torch_nn_relu_override(module: Any, value: Any) -> Any:
    del module
    return value


def functional_relu_override(value: Any, inplace: bool = False) -> Any:
    if inplace:
        raise ValueError("in-place activation is not supported by metadata tracing")
    return value


def torch_where_override(condition: Any, x: Any, y: Any) -> Any:
    return condition + x + y


def torch_abs_override(value: Any, *, out: Any = None) -> Any:
    if out is not None:
        raise ValueError("out= is not supported by metadata tracing")
    return value


manual_meta_overrides: dict[Callable[..., Any] | type, Callable[..., Any]] = {
    nn.Embedding: embedding_override,
    nn.LayerNorm: nn_layernorm_override,
    tp.relu: torch_relu_override,
    nn.functional.relu: functional_relu_override,
    nn.ReLU: torch_nn_relu_override,
    tp.where: torch_where_override,
    tp.abs: torch_abs_override,
}


def _contains_proxy(value: Any) -> bool:
    if isinstance(value, Proxy):
        return True
    if isinstance(value, tuple | list):
        return any(_contains_proxy(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_proxy(item) for item in value.values())
    return False


def check_has_proxy(value: Any) -> Proxy | None:
    """Return the first proxy contained in a nested argument value."""

    if isinstance(value, Proxy):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            proxy = check_has_proxy(item)
            if proxy is not None:
                return proxy
    elif isinstance(value, dict):
        for item in value.values():
            proxy = check_has_proxy(item)
            if proxy is not None:
                return proxy
    return None


def gen_constructor_wrapper(target: _C) -> tuple[Callable[..., Any], _C]:
    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _contains_proxy((args, kwargs)):
            tracer = next(
                value.tracer
                for value in _iter_values((args, kwargs))
                if isinstance(value, Proxy)
            )
            return tracer.create_proxy("call_function", target, args, kwargs)
        return target(*args, **kwargs)

    return wrapper, target


def _iter_values(value: Any):
    if isinstance(value, Proxy):
        yield value
    elif isinstance(value, tuple | list):
        for item in value:
            yield from _iter_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_values(item)


class MetaProxy(Proxy):
    __slots__ = ("_tensor_meta",)

    def install_tensor_meta(self, tensor_meta: Any) -> None:
        self._tensor_meta = tensor_meta

    def size(self, dim: int | None = None) -> Any:
        if hasattr(self, "_tensor_meta"):
            shape = _shape(self._tensor_meta)
            return shape if dim is None else shape[dim]
        return self.tracer.create_proxy("call_method", "size", (self, dim) if dim is not None else (self,), {})

    def dim(self) -> Any:
        if hasattr(self, "_tensor_meta"):
            return len(_shape(self._tensor_meta))
        return self.tracer.create_proxy("call_method", "dim", (self,), {})

    @property
    def shape(self) -> Any:
        if hasattr(self, "_tensor_meta"):
            return _shape(self._tensor_meta)
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "shape"), {})

    @property
    def dtype(self) -> Any:
        if hasattr(self, "_tensor_meta"):
            return getattr(self._tensor_meta, "dtype", None)
        return self.tracer.create_proxy("call_function", builtins.getattr, (self, "dtype"), {})

    @property
    def device(self) -> Any:
        return MetaDeviceAttribute(self, "device")

    def __getattr__(self, name: str) -> Any:
        if name == "_tensor_meta":
            raise AttributeError(name)
        return MetaAttribute(self, name)


class MetaAttribute(MetaProxy):
    __slots__ = ("root", "attr", "_node")

    def __init__(self, root: MetaProxy, attr: str) -> None:
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Any = None

    @property
    def node(self) -> Any:  # type: ignore[override]
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tracer.create_proxy("call_method", self.attr, (self.root, *args), kwargs)


class MetaDeviceAttribute(MetaAttribute):
    pass


def proxys_to_metas(value: Any) -> Any:
    if isinstance(value, MetaDeviceAttribute):
        return "meta"
    if isinstance(value, MetaProxy):
        if not hasattr(value, "_tensor_meta"):
            raise ValueError("MetaProxy has no metadata")
        return value._tensor_meta
    if isinstance(value, tuple):
        return tuple(proxys_to_metas(item) for item in value)
    if isinstance(value, list):
        return [proxys_to_metas(item) for item in value]
    if isinstance(value, dict):
        return {key: proxys_to_metas(item) for key, item in value.items()}
    return value


class MetaTracer(Tracer):
    allow_insert_stateless_mods = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.meta_args: dict[str, Any] = {}
        self._disable_module_getattr = False
        self.orig_forward: Callable[..., Any] | None = None

    def proxy(self, node: Any) -> MetaProxy:
        result = MetaProxy(node, self)
        if node.op == "placeholder" and node.target in self.meta_args:
            result.install_tensor_meta(self.meta_args[node.target])
        return result

    def create_proxy(
        self,
        kind: str,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> MetaProxy:
        node = self.graph.create_node(kind, target, args, kwargs)
        result = MetaProxy(node, self)
        try:
            mapped_args = proxys_to_metas(args)
            mapped_kwargs = proxys_to_metas(kwargs)
            if kind == "call_function":
                target_fn = manual_meta_overrides.get(target, target)
                metadata = target_fn(*mapped_args, **mapped_kwargs)
            elif kind == "call_method":
                receiver, *tail = mapped_args
                metadata = getattr(receiver, target)(*tail, **mapped_kwargs)
            elif kind == "call_module":
                module = self.root
                for atom in str(target).split("."):
                    module = getattr(module, atom)
                target_fn = manual_meta_overrides.get(type(module), module.forward)
                metadata = target_fn(*mapped_args, **mapped_kwargs)
            elif kind == "get_attr":
                value = self.root
                for atom in str(target).split("."):
                    value = getattr(value, atom)
                metadata = value
            else:
                metadata = None
            if metadata is not None:
                result.install_tensor_meta(metadata)
        except Exception as exc:
            warnings.warn(
                f"metadata unavailable for {kind} {target!r}: {exc}",
                stacklevel=2,
            )
        return result

    def trace(
        self,
        root: Any,
        meta_args: dict[str, Any] | None = None,
        concrete_args: dict[str, Any] | None = None,
    ) -> Graph:
        if not isinstance(meta_args, dict):
            raise TypeError(f"meta_args must be a dictionary, got {type(meta_args).__name__}")
        self.meta_args = dict(meta_args)
        module = super().trace(root, sample_inputs=self.meta_args)
        return module.graph

    def getattr(
        self,
        attr: str,
        attr_val: Any,
        parameter_proxy_cache: dict[str, Proxy],
    ) -> Any:
        if self._disable_module_getattr:
            return attr_val
        cached = parameter_proxy_cache.get(attr)
        if cached is not None:
            return cached
        proxy = self.create_proxy("get_attr", attr, (), {})
        parameter_proxy_cache[attr] = proxy
        return proxy

    def call_module(
        self,
        module: Any,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        del module
        self.orig_forward = forward
        return forward(*args, **kwargs)

    def _insert_module_as_submodule(self, module: Any) -> str:
        name = type(module).__name__.lower()
        index = 0
        candidate = f"{name}_{index}"
        while hasattr(self.root, candidate):
            index += 1
            candidate = f"{name}_{index}"
        add_module = getattr(self.root, "add_module", None)
        if not callable(add_module):
            raise TypeError("root cannot register a submodule")
        add_module(candidate, module)
        return candidate

    def path_of_module(self, module: Any) -> str:
        named_modules = getattr(self.root, "named_modules", None)
        if callable(named_modules):
            for name, candidate in named_modules():
                if candidate is module:
                    return name
        if (
            self.allow_insert_stateless_mods
            and callable(getattr(module, "parameters", None))
            and callable(getattr(module, "buffers", None))
            and not list(module.parameters())
            and not list(module.buffers())
        ):
            return self._insert_module_as_submodule(module)
        raise NameError(f"module {module!r} is not registered below the trace root")


def symbolic_trace(
    root: Any,
    meta_args: dict[str, Any] | None = None,
    concrete_args: dict[str, Any] | None = None,
) -> Any:
    tracer = MetaTracer(concrete_args=concrete_args)
    meta_args = dict(meta_args or {})
    graph = tracer.trace(root, meta_args)
    from ..graph_module import GraphModule

    return GraphModule(root, graph, tracer.signature)
