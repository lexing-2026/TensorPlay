"""Placement rules for embedding lookups."""

from __future__ import annotations

from typing import Any

from .._op_schema import OpSchema, OpStrategy
from ..placement_types import _MaskPartial, Partial, Replicate, Shard
from .utils import expand_to_full_mesh_op_strategy

__all__ = ["embedding_dense_backward_strategy", "embedding_strategy"]


def _strategy_args(schema: OpSchema) -> tuple[Any, ...]:
    values = []
    for value in schema.args_schema:
        if isinstance(value, OpStrategy):
            values.append(value)
    for value in schema.kwargs_schema.values():
        if isinstance(value, OpStrategy):
            values.append(value)
    return tuple(values)


def embedding_strategy(mesh: Any, schema: OpSchema) -> OpStrategy:
    weight, indices = _strategy_args(schema)[:2]
    weight_meta = weight.tensor_meta
    indices_meta = indices.tensor_meta
    if weight_meta is None or indices_meta is None:
        raise AssertionError("embedding strategy requires tensor metadata")
    output_dim = len(indices_meta.shape)
    rowwise_partial = _MaskPartial(
        offset_shape=tuple(int(value) for value in weight_meta.shape),
        offset_dim=0,
    )
    single_dim = [
        [Shard(output_dim), Shard(1), Replicate()],
        [rowwise_partial, Shard(0), rowwise_partial],
    ]
    for dim in range(len(indices_meta.shape)):
        single_dim.append([Shard(dim), Replicate(), Shard(dim)])
    return expand_to_full_mesh_op_strategy(
        mesh,
        schema,
        single_dim,
        output_tensor_meta=None,
        input_index=1,
    )


def embedding_dense_backward_strategy(mesh: Any, schema: OpSchema) -> OpStrategy:
    values = _strategy_args(schema)
    grad_output, indices = values[:2]
    grad_meta = grad_output.tensor_meta
    indices_meta = indices.tensor_meta
    if grad_meta is None or indices_meta is None:
        raise AssertionError("embedding backward strategy requires tensor metadata")
    input_dim = len(grad_meta.shape)
    single_dim = [[Shard(1), Shard(input_dim - 1), Replicate()]]
    for dim in range(len(indices_meta.shape)):
        single_dim.append([Partial(), Shard(dim), Shard(dim)])
    single_dim.append([Partial(), Partial(), Replicate()])
    return expand_to_full_mesh_op_strategy(
        mesh,
        schema,
        single_dim,
        output_tensor_meta=None,
        input_index=1,
    )
