"""Reduction handlers for layouts that need value/index reconciliation."""

from __future__ import annotations

import operator
from functools import reduce
from typing import Any

import tensorplay as tp

from .. import _functional_collectives as funcol
from .. import distributed_core as dist
from ._op_schema import OutputSharding
from ._utils import compute_local_shape_and_global_offset
from .placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = [
    "is_linear_reduction",
    "reduce_partial",
    "argminmax_handler",
    "minmax_dim_handler",
]


_ARGMINMAX_REDUCTION_OPS = {"argmax": "max", "argmin": "min"}


def _operation_name(operation: Any) -> str:
    if isinstance(operation, str):
        value = operation
    else:
        value = getattr(operation, "__name__", None)
        if value is None:
            value = getattr(operation, "name", str(operation))
    value = str(value).rsplit(".", 1)[-1]
    for suffix in ("_default", "_out", "_functional"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def _reduction_dim(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> int | None:
    if "dim" in kwargs:
        value = kwargs["dim"]
    elif len(args) > 1 and isinstance(args[1], int) and not isinstance(args[1], bool):
        value = args[1]
    else:
        return None
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("reduction dimension must be an integer or None")
    return value


def is_dim_reduction_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return _reduction_dim(args, kwargs) is not None


def is_linear_reduction(reduce_op: str) -> bool:
    return reduce_op in Partial.LINEAR_REDUCE_OPS


def reduce_partial(value: Any, placement: Partial, group: Any) -> Any:
    operation = {
        "sum": dist.ReduceOp.SUM,
        "avg": dist.ReduceOp.AVG,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "product": dist.ReduceOp.PRODUCT,
    }[placement.reduce_op]
    dist.all_reduce(value, op=operation, group=group)
    return value


def _output_spec_for_nonlinear_reduction(
    operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any], tuple_output: bool
) -> OutputSharding | None:
    del operation
    from ._api import DTensor
    from ._dtensor_spec import DTensorSpec, TensorMeta

    input_dtensor = next(
        (value for value in args if isinstance(value, DTensor)), None
    )
    if input_dtensor is None:
        return None
    dim = _reduction_dim(args, kwargs)
    ndim = input_dtensor.ndim
    reduced = set(range(ndim)) if dim is None else {
        dim if dim >= 0 else ndim + dim
    }
    if any(value < 0 or value >= ndim for value in reduced):
        raise IndexError("reduction dimension is outside tensor rank")
    keepdim = bool(kwargs.get("keepdim", args[2] if len(args) > 2 else False))
    placements: list[Placement] = []
    for placement in input_dtensor.placements:
        if isinstance(placement, Partial) or (
            _is_shard_like(placement) and placement.dim in reduced
        ):
            placements.append(Replicate())
        elif _is_shard_like(placement):
            shift = 0 if keepdim else sum(value < placement.dim for value in reduced)
            dim_value = placement.dim - shift
            if isinstance(placement, _StridedShard):
                placements.append(_StridedShard(dim_value, placement.split_factor))
            else:
                placements.append(Shard(dim_value))
        else:
            placements.append(placement)
    shape = tuple(
        1 if keepdim and index in reduced else size
        for index, size in enumerate(input_dtensor.shape)
        if keepdim or index not in reduced
    )
    stride_values = [1] * len(shape)
    running = 1
    for index in reversed(range(len(shape))):
        stride_values[index] = running
        running *= int(shape[index])
    input_spec = input_dtensor._op_dispatcher._spec_from_dtensor(input_dtensor)
    spec = DTensorSpec(
        input_dtensor.device_mesh,
        tuple(placements),
        TensorMeta(shape, tuple(stride_values), input_dtensor.dtype),
        use_strided_shard_as_shard_order=(
            input_spec.use_strided_shard_as_shard_order
        ),
    )
    if tuple_output:
        return OutputSharding((spec, spec))
    return OutputSharding(spec)


def _get_output_sharding(
    op_call: Any,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> OutputSharding:
    from ._api import DTensor

    name = _operation_name(op_call)
    tuple_output = name in {"max", "min"} and is_dim_reduction_call(args, kwargs)
    if name in _ARGMINMAX_REDUCTION_OPS or tuple_output:
        direct = _output_spec_for_nonlinear_reduction(
            op_call, args, kwargs, tuple_output
        )
        if direct is not None:
            return direct
    dispatcher = DTensor._op_dispatcher
    op_info = dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    return output_sharding


def _prep_arguments(
    op_call_repr: str,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None,
) -> tuple[Any, tuple[int, ...], Any, tuple[Placement, ...], int | None, bool]:
    del op_call_repr
    from ._api import DTensor

    input_dtensor = args[0] if args else None
    if not isinstance(input_dtensor, DTensor):
        raise NotImplementedError
    dim = _reduction_dim(args, kwargs or {})
    keepdim = bool((kwargs or {}).get("keepdim", args[2] if len(args) > 2 else False))
    device_mesh = input_dtensor.device_mesh
    placements = input_dtensor.placements
    if any(isinstance(placement, Partial) for placement in placements):
        target_placements = tuple(
            Replicate() if isinstance(placement, Partial) else placement
            for placement in placements
        )
        input_dtensor = input_dtensor.redistribute(
            device_mesh=device_mesh, placements=target_placements
        )
        placements = input_dtensor.placements
    return (
        input_dtensor.to_local(),
        tuple(int(value) for value in input_dtensor.shape),
        device_mesh,
        placements,
        dim,
        keepdim,
    )


def _get_expected_shape(
    local_tensor: Any, dim: int | None, keepdim: bool
) -> tuple[int, ...]:
    input_shape = list(local_tensor.shape)
    if dim is None:
        return tuple([1] * len(input_shape) if keepdim else [])
    dim = dim if dim >= 0 else len(input_shape) + dim
    if dim < 0 or dim >= len(input_shape):
        raise IndexError("reduction dimension is outside tensor rank")
    if keepdim:
        input_shape[dim] = 1
    else:
        input_shape.pop(dim)
    return tuple(int(value) for value in input_shape)


def _collect_shard_mesh_dims(
    op_call_repr: str,
    local_tensor: Any,
    placements: tuple[Placement, ...],
    dim: int | None,
) -> list[int]:
    shard_mesh_dims: list[int] = []
    normalized_dim = None
    if dim is not None:
        normalized_dim = dim if dim >= 0 else int(local_tensor.dim()) + dim
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, _StridedShard):
            raise NotImplementedError(f"{op_call_repr} does not support strided shards")
        if isinstance(placement, Shard) and (
            dim is None or placement.dim == normalized_dim
        ):
            shard_mesh_dims.append(mesh_dim)
    return shard_mesh_dims


def _convert_to_global_idxs(
    local_idx: Any,
    global_shape: tuple[int, ...],
    device_mesh: Any,
    placements: tuple[Placement, ...],
    dim: int | None,
) -> tuple[int, Any]:
    local_shape, global_offset = compute_local_shape_and_global_offset(
        global_shape, device_mesh, placements
    )
    if dim is None:
        gathered_idxs = tp.zeros_like(local_idx)
        remaining = local_idx
        for index in range(len(local_shape)):
            local_stride = reduce(operator.mul, local_shape[index + 1 :], 1)
            global_stride = reduce(operator.mul, global_shape[index + 1 :], 1)
            coordinate = remaining // local_stride
            remaining = remaining % local_stride
            gathered_idxs = gathered_idxs + (
                coordinate + global_offset[index]
            ) * global_stride
        return 0, gathered_idxs
    normalized_dim = dim if dim >= 0 else len(global_shape) + dim
    return normalized_dim, local_idx + global_offset[normalized_dim]


def _wait(value: Any) -> Any:
    waiter = getattr(funcol, "wait_tensor", None)
    if callable(waiter):
        return waiter(value)
    return value


def _gather_tensors(
    gather_dim: int,
    gathered_idxs: Any,
    local_redux: Any,
    device_mesh: Any,
    shard_mesh_dims: list[int],
) -> tuple[Any, Any]:
    gathered_redux = local_redux
    for mesh_dim in shard_mesh_dims:
        gathered_redux = _wait(
            funcol.all_gather_single(
                gathered_redux,
                gather_dim=gather_dim,
                group=(device_mesh, mesh_dim),
            )
        )
        gathered_idxs = _wait(
            funcol.all_gather_single(
                gathered_idxs,
                gather_dim=gather_dim,
                group=(device_mesh, mesh_dim),
            )
        )
    return gathered_redux, gathered_idxs


def _value_index(result: Any) -> tuple[Any, Any]:
    values = getattr(result, "values", None)
    indices = getattr(result, "indices", None)
    if values is not None and indices is not None:
        return values, indices
    return result[0], result[1]


def argminmax_handler(
    op_call: Any,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    name = _operation_name(op_call)
    if name not in _ARGMINMAX_REDUCTION_OPS:
        raise NotImplementedError(f"unsupported index reduction: {op_call}")
    local_tensor, global_shape, device_mesh, placements, dim, keepdim = _prep_arguments(
        str(op_call), args, kwargs
    )
    output_sharding = _get_output_sharding(op_call, args, kwargs)
    expected_shape = _get_expected_shape(local_tensor, dim, keepdim)
    shard_mesh_dims = _collect_shard_mesh_dims(
        str(op_call), local_tensor, placements, dim
    )
    value_op = tp.max if name == "argmax" else tp.min
    index_op = tp.argmax if name == "argmax" else tp.argmin
    if dim is None:
        local_redux = value_op(local_tensor).unsqueeze(0)
        local_idx = index_op(local_tensor).unsqueeze(0)
    else:
        local_redux, local_idx = _value_index(
            value_op(local_tensor, dim=dim, keepdim=True)
        )
    if not shard_mesh_dims:
        from ._api import DTensor

        return DTensor._op_dispatcher.wrap(
            local_idx.reshape(expected_shape), output_sharding.output_spec
        )
    gather_dim, gathered_idxs = _convert_to_global_idxs(
        local_idx, global_shape, device_mesh, placements, dim
    )
    gathered_redux, gathered_idxs = _gather_tensors(
        gather_dim, gathered_idxs, local_redux, device_mesh, shard_mesh_dims
    )
    select_dim = 0 if dim is None else gather_dim
    rank_winner = index_op(gathered_redux, dim=select_dim, keepdim=True)
    final_idx = tp.gather(gathered_idxs, dim=gather_dim, index=rank_winner)
    from ._api import DTensor

    return DTensor._op_dispatcher.wrap(
        final_idx.reshape(expected_shape), output_sharding.output_spec
    )


def minmax_dim_handler(
    op_call: Any,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    name = _operation_name(op_call)
    if name not in {"min", "max"} or not is_dim_reduction_call(args, kwargs):
        raise NotImplementedError(f"unsupported value/index reduction: {op_call}")
    local_tensor, global_shape, device_mesh, placements, dim, keepdim = _prep_arguments(
        str(op_call), args, kwargs
    )
    if dim is None:
        raise AssertionError("a dimension is required")
    output_sharding = _get_output_sharding(op_call, args, kwargs)
    expected_shape = _get_expected_shape(local_tensor, dim, keepdim)
    shard_mesh_dims = _collect_shard_mesh_dims(
        str(op_call), local_tensor, placements, dim
    )
    value_op = tp.max if name == "max" else tp.min
    local_redux, local_idx = _value_index(
        value_op(local_tensor, dim=dim, keepdim=True)
    )
    if not shard_mesh_dims:
        from ._api import DTensor

        return DTensor._op_dispatcher.wrap(
            (
                local_redux.reshape(expected_shape),
                local_idx.reshape(expected_shape),
            ),
            output_sharding.output_spec,
        )
    gather_dim, gathered_idxs = _convert_to_global_idxs(
        local_idx, global_shape, device_mesh, placements, dim
    )
    gathered_redux, gathered_idxs = _gather_tensors(
        gather_dim, gathered_idxs, local_redux, device_mesh, shard_mesh_dims
    )
    final_redux, rank_winner = _value_index(
        value_op(gathered_redux, dim=gather_dim, keepdim=True)
    )
    final_idx = tp.gather(gathered_idxs, dim=gather_dim, index=rank_winner)
    from ._api import DTensor

    return DTensor._op_dispatcher.wrap(
        (
            final_redux.reshape(expected_shape),
            final_idx.reshape(expected_shape),
        ),
        output_sharding.output_spec,
    )
