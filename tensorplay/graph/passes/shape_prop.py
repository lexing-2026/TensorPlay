"""Execute a graph and record values and tensor properties on its nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from ..graph_module import GraphModule
from ..interpreter import Interpreter
from ..node import Node, map_aggregate
from .base import PassBase, PassResult

__all__ = ["ShapeProp", "TensorMetadata"]


def _value_or_call(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    return value() if callable(value) else value


def _is_tensor(value: Any) -> bool:
    try:
        from tensorplay import Tensor
    except ImportError:
        return False
    return isinstance(value, Tensor)


def _is_sparse(value: Any) -> bool:
    if bool(_value_or_call(getattr(value, "is_sparse", False), False)):
        return True
    for name in ("is_sparse_csr", "is_sparse_csc", "is_sparse_bsr", "is_sparse_bsc"):
        if bool(_value_or_call(getattr(value, name, False), False)):
            return True
    return False


def _extract_tensor_metadata(
    value: Any, include_contiguity: bool = True
) -> "TensorMetadata":
    shape = _value_or_call(getattr(value, "shape", ()), ())
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    stride = ()
    if not _is_sparse(value):
        stride = tuple(_value_or_call(getattr(value, "stride", ()), ()))
    memory_format = (
        _value_or_call(getattr(value, "memory_format", None))
        if include_contiguity
        else None
    )
    is_quantized = bool(_value_or_call(getattr(value, "is_quantized", False), False))
    return TensorMetadata(
        shape,
        _value_or_call(getattr(value, "dtype", None)),
        bool(_value_or_call(getattr(value, "requires_grad", False), False)),
        stride,
        memory_format,
        is_quantized,
        _qparams(value) if is_quantized else {},
    )


def _qparams(value: Any) -> dict[str, Any]:
    qscheme = _value_or_call(getattr(value, "qscheme", None))
    if qscheme is None:
        return {}
    result: dict[str, Any] = {"qscheme": qscheme}
    for key, name in (
        ("scale", "q_scale"),
        ("zero_point", "q_zero_point"),
        ("axis", "q_per_channel_axis"),
        ("scale", "q_per_channel_scales"),
        ("zero_point", "q_per_channel_zero_points"),
    ):
        if key in result:
            continue
        method = getattr(value, name, None)
        if method is None:
            continue
        try:
            result[key] = _value_or_call(method)
        except (AttributeError, TypeError, RuntimeError):
            continue
    return result


class _ShapePropInterpreter(Interpreter):
    def run_node(self, node: Node) -> Any:
        try:
            value = super().run_node(node)
        except Exception as exc:
            raise RuntimeError(
                f"ShapeProp error for: node={node.format_node()} with meta={node.meta}"
            ) from exc

        node.meta["val"] = value
        node.meta["type"] = type(value)
        node.type = type(value)

        found_tensor = False

        def extract(item: Any) -> Any:
            nonlocal found_tensor
            if _is_tensor(item):
                found_tensor = True
                return _extract_tensor_metadata(item)
            return item

        metadata = map_aggregate(value, extract)
        if found_tensor:
            node.meta["tensor_meta"] = metadata
            if _is_tensor(value):
                node.meta["tensor_shape"] = tuple(
                    _value_or_call(getattr(value, "shape", ()), ())
                )
        return value


class ShapeProp(_ShapePropInterpreter, PassBase):
    """Run a graph while attaching concrete values and tensor metadata."""

    def __init__(
        self,
        module_or_inputs: GraphModule | Sequence[Any] | None = None,
        fake_mode: Any = None,
    ) -> None:
        self.example_inputs = (
            ()
            if module_or_inputs is None or isinstance(module_or_inputs, GraphModule)
            else tuple(module_or_inputs)
        )
        self.fake_mode = fake_mode
        self.module: GraphModule | None = None
        if isinstance(module_or_inputs, GraphModule):
            self._set_module(module_or_inputs)

    def _set_module(self, module: GraphModule) -> None:
        Interpreter.__init__(self, module)
        self.module = module

    def _propagate_inputs(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.fake_mode is None:
            return args
        converter = getattr(self.fake_mode, "from_tensor", None)
        if not callable(converter):
            raise TypeError("fake_mode must provide from_tensor")
        return tuple(converter(value) if _is_tensor(value) else value for value in args)

    def propagate(self, *args: Any, **kwargs: Any) -> Any:
        if self.module is None:
            raise RuntimeError("ShapeProp.propagate requires a GraphModule")
        return Interpreter.run(
            self,
            *self._propagate_inputs(args),
            **kwargs,
        )

    def __call__(self, graph_module: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(graph_module, GraphModule):
            self._set_module(graph_module)
            values = self.example_inputs if not args and not kwargs else args
            self.propagate(*values, **kwargs)
            return PassResult(graph_module, False)
        if self.module is None:
            raise TypeError("ShapeProp expects a GraphModule before input values")
        return self.propagate(graph_module, *args, **kwargs)


class TensorMetadata(NamedTuple):
    """Tensor properties recorded during graph execution."""

    shape: Any
    dtype: Any
    requires_grad: bool
    stride: tuple[Any, ...] = ()
    memory_format: Any = None
    is_quantized: bool = False
    qparams: dict[str, Any] | None = None
