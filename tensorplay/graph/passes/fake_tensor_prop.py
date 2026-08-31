"""Value and shape propagation for graph analysis."""

from __future__ import annotations

from typing import Any, NamedTuple

from ..interpreter import Interpreter
from ..node import Node

__all__ = ["FakeTensorProp", "TensorMetadata"]


class TensorMetadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any
    requires_grad: bool
    stride: tuple[int, ...] | None = None
    memory_format: Any = None


def _metadata(value: Any) -> TensorMetadata | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    if callable(shape):
        shape = shape()
    try:
        shape_tuple = tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None
    dtype = getattr(value, "dtype", None)
    if callable(dtype):
        dtype = dtype()
    requires_grad = getattr(value, "requires_grad", False)
    if callable(requires_grad):
        requires_grad = requires_grad()
    stride = getattr(value, "stride", None)
    if callable(stride):
        try:
            stride = tuple(int(item) for item in stride())
        except (TypeError, ValueError):
            stride = None
    return TensorMetadata(shape_tuple, dtype, bool(requires_grad), stride)


class FakeTensorProp(Interpreter):
    """Run a graph and attach tensor metadata to every computed node."""

    def run_node(self, node: Node) -> Any:
        result = super().run_node(node)
        node.meta["val"] = result
        metadata = _metadata(result)
        if metadata is not None:
            node.meta["tensor_meta"] = metadata
            node.meta["tensor_shape"] = metadata.shape
        return result
