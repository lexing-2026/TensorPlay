from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..symbolic_shapes import DimDynamic, ShapeEnv
from ...passes.shape_prop import ShapeProp

__all__ = ["infer_shape", "mksym"]


def mksym(shape_env: ShapeEnv, value: int, source: Any, dynamic_dim: DimDynamic) -> Any:
    return shape_env.create_symintnode(
        shape_env.create_symbol(value, source, dynamic_dim),
        hint=value,
        source=source,
    )


def infer_shape(
    graph_module: Any,
    input_tensors: Sequence[Any],
) -> tuple[Any, list[Any], ShapeEnv, Any] | None:
    """Annotate a graph with inferred values and symbolic input dimensions."""

    inputs = list(input_tensors)
    ShapeProp(graph_module)(*inputs)
    shape_env = ShapeEnv()
    first_symbol = None
    for index, value in enumerate(inputs):
        shape = getattr(value, "shape", ())
        if callable(shape):
            shape = shape()
        symbolic_shape = []
        for dim_index, dimension in enumerate(tuple(shape)):
            symbol = mksym(
                shape_env,
                int(dimension),
                type("Source", (), {"name": f"input{index}.size[{dim_index}]"})(),
                DimDynamic.DYNAMIC,
            )
            symbolic_shape.append(symbol)
            if first_symbol is None:
                first_symbol = symbol
        del symbolic_shape
    return graph_module, inputs, shape_env, first_symbol
