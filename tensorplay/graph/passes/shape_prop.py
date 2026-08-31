"""Execute an example graph and record values useful to later passes."""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

from ..graph_module import GraphModule
from ..interpreter import Interpreter
from ..node import Node
from ..proxy import Proxy
from .base import PassBase, PassResult

__all__ = ["ShapeProp", "TensorMetadata"]


class _ShapePropInterpreter(Interpreter):
    def run_node(self, node: Node) -> Any:
        value = super().run_node(node)
        node.meta["val"] = value
        node.type = type(value)
        shape = getattr(value, "shape", None)
        if callable(shape):
            shape = shape()
        if shape is not None:
            try:
                node.meta["tensor_shape"] = tuple(int(item) for item in shape)
                node.meta["tensor_meta"] = TensorMetadata(
                    node.meta["tensor_shape"],
                    getattr(value, "dtype", None),
                    bool(getattr(value, "requires_grad", False)),
                    _stride(value),
                    getattr(value, "memory_format", None),
                    bool(getattr(value, "is_quantized", False)),
                    _qparams(value),
                )
            except (TypeError, ValueError):
                pass
        return value


class ShapeProp(PassBase):
    """Record values and tensor metadata without changing graph structure."""

    def __init__(self, module_or_inputs: GraphModule | Sequence[Any] | None = None) -> None:
        self.module = module_or_inputs if isinstance(module_or_inputs, GraphModule) else None
        self.example_inputs = (
            () if self.module is not None or module_or_inputs is None else tuple(module_or_inputs)
        )

    def propagate(self, *args: Any, **kwargs: Any) -> Any:
        if self.module is None:
            raise RuntimeError("ShapeProp.propagate requires a GraphModule")
        return _ShapePropInterpreter(self.module).run(*args, **kwargs)

    def __call__(self, graph_module) -> PassResult:
        self.module = graph_module
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and isinstance(node.target, (Node, Proxy)):
                return PassResult(graph_module, False)
        _ShapePropInterpreter(graph_module).run(*self.example_inputs)
        return PassResult(graph_module, False)


def _stride(value: Any) -> tuple[int, ...]:
    stride = getattr(value, "stride", None)
    if not callable(stride):
        return ()
    try:
        return tuple(int(item) for item in stride())
    except (TypeError, ValueError, RuntimeError):
        return ()


def _qparams(value: Any) -> dict[str, Any]:
    if not bool(getattr(value, "is_quantized", False)):
        return {}
    qscheme = getattr(value, "qscheme", lambda: None)()
    result: dict[str, Any] = {"qscheme": qscheme}
    for name, getter in (
        ("scale", "q_scale"),
        ("zero_point", "q_zero_point"),
        ("axis", "q_per_channel_axis"),
    ):
        method = getattr(value, getter, None)
        if callable(method):
            try:
                result[name] = method()
            except (TypeError, RuntimeError):
                continue
    return result


class TensorMetadata(NamedTuple):
    """Tensor properties recorded while a graph is executed."""

    shape: tuple[Any, ...]
    dtype: Any
    requires_grad: bool
    stride: tuple[int, ...] = ()
    memory_format: Any = None
    is_quantized: bool = False
    qparams: dict[str, Any] | None = None
