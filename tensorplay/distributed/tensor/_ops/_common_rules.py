"""Propagation rules for elementwise and Einstein-style operations."""

from __future__ import annotations

import string
from typing import Any

from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OutputSharding
from .._utils import compute_local_shape_and_global_offset
from .utils import _operation_name, prod

__all__ = ["OutputSharding", "einop_rule", "pointwise_rule"]


def _replace_char_in_str(value: str, new_char: str, index: int) -> str:
    return value[:index] + new_char + value[index + 1 :]


def _schema_specs(op_schema: Any) -> tuple[DTensorSpec, ...]:
    specs = getattr(op_schema, "args_spec", None)
    if specs is not None:
        result = list(specs)
        values = getattr(op_schema, "kwargs", {})
    elif hasattr(op_schema, "args"):
        values = (
            getattr(op_schema, "args"),
            getattr(op_schema, "kwargs", {}),
        )
        result = []
    else:
        values = op_schema
        result = []

    def visit(value: Any) -> None:
        if isinstance(value, DTensorSpec):
            result.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(values)
    return tuple(result)


def _gen_reshard_suggestions(
    op_schema: OpSchema,
    input_dims: list[str],
    input_specs: tuple[DTensorSpec, ...],
    dim_to_sharding: dict[str, int],
    pending_sum: list[int],
) -> OutputSharding:
    suggested_specs: list[DTensorSpec] = []
    for input_dim, input_spec in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_specs.append(
            DTensorSpec.from_dim_map(
                mesh=input_spec.mesh,
                dim_map=dim_map,
                sums=pending_sum,
                tensor_meta=input_spec.tensor_meta,
            )
        )
    suggested_schema = OpSchema(
        op_schema.op,
        tuple(suggested_specs),
        op_schema.kwargs_schema,
        schema_info=op_schema.schema_info,
    )
    suggested_schema._inplace_rewrap_schema_suggestion(op_schema)
    return OutputSharding(
        None,
        redistribute_schema=suggested_schema,
        needs_redistribute=True,
    )


def _parse_einop_call(
    equation_or_schema: str | OpSchema,
    op_schema_or_equation: OpSchema | str | None,
) -> tuple[str, OpSchema]:
    if isinstance(equation_or_schema, str):
        if not isinstance(op_schema_or_equation, OpSchema):
            raise TypeError("einop_rule requires an operation schema")
        return equation_or_schema, op_schema_or_equation
    if not isinstance(op_schema_or_equation, str):
        raise TypeError("einop_rule requires an einsum equation")
    return op_schema_or_equation, equation_or_schema


def einop_rule(
    equation_or_schema: str | OpSchema,
    op_schema_or_equation: OpSchema | str | None = None,
    *,
    linearity: bool = False,
    enforce_sharding: dict[str, int] | None = None,
) -> OutputSharding:
    """Propagate layouts through an operation described by an einsum equation."""
    equation, op_schema = _parse_einop_call(
        equation_or_schema, op_schema_or_equation
    )
    inputs, outputs = equation.replace(" ", "").split("->", 1)
    input_dims = inputs.split(",")
    output_dims = outputs.split(",")
    if len(output_dims) != 1:
        raise ValueError("einsum propagation requires one output expression")
    input_specs = _schema_specs(op_schema)
    if len(input_specs) != len(input_dims):
        raise ValueError(
            f"einsum input count {len(input_dims)} does not match "
            f"distributed input count {len(input_specs)}"
        )
    if not input_specs:
        return OutputSharding(None, failed_reason="no distributed tensor input")

    dim_to_sharding: dict[str, int] = {}
    dim_to_size: dict[str, int] = {}
    pending_sums_counter: dict[int, int] = {}
    seen_shardings: dict[int, str] = {}
    needs_reshard = False

    def merge_sharding(dim: str, first: int, second: int) -> int:
        nonlocal needs_reshard
        if first == second:
            return first
        if first == -1 or second == -1:
            needs_reshard = True
            return first if first != -1 else second
        raise RuntimeError(
            f"{equation}: dimension {dim} is sharded in two different ways"
        )

    for input_dim, input_spec in zip(input_dims, input_specs):
        if input_spec.tensor_meta is None:
            raise ValueError("einsum propagation requires tensor metadata")
        for sum_dim in input_spec.sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = "+"
            pending_sums_counter[sum_dim] = pending_sums_counter.get(sum_dim, 0) + 1

        for index, (dim, mesh_dim) in enumerate(
            zip(input_dim, input_spec.dim_map)
        ):
            if enforce_sharding is not None and dim in enforce_sharding:
                forced = enforce_sharding[dim]
                if forced != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = forced
                dim_to_size[dim] = int(input_spec.shape[index])
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = int(input_spec.shape[index])
            else:
                dim_to_sharding[dim] = merge_sharding(
                    dim, dim_to_sharding[dim], mesh_dim
                )
                if dim_to_size[dim] != int(input_spec.shape[index]):
                    raise ValueError(f"einsum dimension {dim} has inconsistent sizes")

            merged_mesh_dim = dim_to_sharding[dim]
            if merged_mesh_dim != -1:
                previous = seen_shardings.get(merged_mesh_dim)
                if previous is not None and dim not in previous:
                    needs_reshard = True
                    seen_shardings[merged_mesh_dim] = previous + dim
                else:
                    seen_shardings[merged_mesh_dim] = dim

    if pending_sums_counter and not linearity:
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, []
        )
    if linearity:
        for count in pending_sums_counter.values():
            if count != len(input_specs):
                needs_reshard = True

    for mesh_dim, dimensions in tuple(seen_shardings.items()):
        if len(dimensions) <= 1 or dimensions == "+":
            continue
        choices = list(dimensions)
        costs: list[int] = []
        for dimension in choices:
            cost = 0
            for input_dim, input_spec in zip(input_dims, input_specs):
                if dimension not in input_dim:
                    continue
                index = input_dim.index(dimension)
                if input_spec.dim_map[index] != mesh_dim:
                    continue
                local_shape, _ = compute_local_shape_and_global_offset(
                    input_spec.shape,
                    input_spec.mesh,
                    input_spec.placements,
                )
                cost += prod(local_shape) * int(input_spec.mesh.size(mesh_dim))
            costs.append(cost)
        keep = choices[costs.index(max(costs))]
        for dimension in choices:
            if dimension != keep:
                dim_to_sharding[dimension] = -1

    pending_sums = list(pending_sums_counter)
    if needs_reshard:
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, pending_sums
        )

    for dim, mesh_dim in dim_to_sharding.items():
        if dim not in output_dims[0] and mesh_dim != -1:
            pending_sums.append(mesh_dim)

    output_dim = output_dims[0]
    output_dim_map: list[int] = []
    output_shape: list[int] = []
    for dim in output_dim:
        if dim == "1":
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            if dim not in dim_to_sharding or dim not in dim_to_size:
                raise ValueError(f"einsum output dimension {dim} is not an input dimension")
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])

    input_meta = input_specs[0].tensor_meta
    if input_meta is None:
        raise ValueError("einsum propagation requires tensor metadata")
    stride = input_meta.stride
    if len(stride) != len(output_shape):
        stride_values: list[int] = []
        for index in range(len(output_shape)):
            value = 1
            for size in output_shape[index + 1 :]:
                value *= int(size)
            stride_values.append(value)
        stride = tuple(stride_values)
    output_meta = TensorMeta(tuple(output_shape), tuple(stride), input_meta.dtype)
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_specs[0].mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=output_meta,
        )
    )


def _schema_is_inplace(op_schema: OpSchema) -> bool:
    method = getattr(op_schema, "is_inplace_op", None)
    if callable(method):
        return bool(method())
    return _operation_name(op_schema.op).endswith("_")


def _schema_is_out_variant(op_schema: OpSchema) -> bool:
    method = getattr(op_schema, "is_out_variant_op", None)
    if callable(method):
        return bool(method())
    return "out" in op_schema.kwargs_schema


def pointwise_rule(op_schema: OpSchema, linearity: bool = False) -> OutputSharding:
    """Propagate layouts for pointwise operations with broadcasting."""
    input_specs = _schema_specs(op_schema)
    if not input_specs:
        return OutputSharding(None, failed_reason="no distributed tensor input")
    max_dim = max(spec.ndim for spec in input_specs)
    if max_dim > len(string.ascii_lowercase):
        raise ValueError("pointwise propagation supports at most 26 dimensions")

    dimchars: list[str] = []
    singleton_counter = [0] * max_dim
    for input_spec in input_specs:
        start_dim = max_dim - input_spec.ndim
        dims = string.ascii_lowercase[start_dim:max_dim]
        if len(input_specs) > 1:
            for index in range(max_dim):
                if index < start_dim:
                    singleton_counter[index] += 1
                elif int(input_spec.shape[index - start_dim]) == 1:
                    singleton_counter[index] += 1
                    dims = _replace_char_in_str(dims, "1", index - start_dim)
        dimchars.append(dims)

    output_dims = string.ascii_lowercase[:max_dim]
    for index, count in enumerate(singleton_counter):
        if count == len(input_specs):
            output_dims = _replace_char_in_str(output_dims, "1", index)

    enforce_sharding: dict[str, int] = {}
    if _schema_is_inplace(op_schema):
        follow = input_specs[0]
        enforce_sharding.update(zip(output_dims, follow.dim_map))
    elif _schema_is_out_variant(op_schema):
        follow = op_schema.kwargs_schema.get("out")
        if isinstance(follow, DTensorSpec):
            enforce_sharding.update(zip(output_dims, follow.dim_map))

    return einop_rule(
        f"{','.join(dimchars)}->{output_dims}",
        op_schema,
        linearity=linearity,
        enforce_sharding=enforce_sharding,
    )
