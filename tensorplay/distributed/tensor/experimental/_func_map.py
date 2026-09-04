"""Map distributed tensor values through a local tensor function."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay

from ..._functional_collectives import AsyncCollectiveTensor
from .._api import DTensor
from ..placement_types import Partial, Placement, Replicate, Shard

try:
    import spmd_types as spmd
except ImportError:
    spmd = None

from ....utils import _pytree as pytree

__all__ = ["local_map"]

PlacementType = Sequence[Placement] | None
InputPlacements = tuple[PlacementType, ...] | None
OutputPlacements = PlacementType | tuple[PlacementType, ...]


def _placements_to_spmd_type(
    placements: PlacementType,
    grad_placements: PlacementType,
    device_mesh: Any,
) -> dict[Any, Any]:
    if spmd is None:
        raise RuntimeError("spmd_types=True requires the spmd_types package")
    result: dict[Any, Any] = {}
    for dim_idx, placement in enumerate(placements or ()):
        axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
        grad_placement = (
            grad_placements[dim_idx] if grad_placements is not None else None
        )
        if type(placement) is Shard:
            forward_type = spmd.V
        elif type(placement) is Replicate:
            forward_type = spmd.I if type(grad_placement) is Replicate else spmd.R
        elif type(placement) is Partial:
            forward_type = spmd.P
        else:
            raise ValueError(
                "local_map(spmd_types=True) does not support placement type "
                f"{type(placement).__name__}: {placement}"
            )

        if grad_placement is not None:
            actual_grad = _dtensor_placement_if_compatible(
                forward_type.backward_type(), grad_placement
            )
            if actual_grad is None or type(actual_grad) is not type(grad_placement):
                raise ValueError(
                    "local_map(spmd_types=True) cannot represent the requested "
                    "forward and backward placements"
                )
        result[axis] = forward_type
    return result


def _annotate_spmd_types(
    flat_local_args: list[Any],
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: Any,
) -> None:
    if spmd is None:
        raise RuntimeError("spmd_types=True requires the spmd_types package")
    for index, local_arg in enumerate(flat_local_args):
        if not isinstance(local_arg, tensorplay.Tensor):
            continue
        if in_placements is None or in_placements[index] is None:
            continue
        grad_placements = (
            in_grad_placements[index] if in_grad_placements is not None else None
        )
        spmd.assert_type(
            local_arg,
            _placements_to_spmd_type(
                in_placements[index], grad_placements, device_mesh
            ),
        )


def _valid_grad_placements(placement: Placement) -> str:
    if type(placement) is Shard:
        return "Shard"
    if type(placement) is Replicate:
        return "Partial or Replicate"
    if type(placement) is Partial:
        return "Replicate"
    raise ValueError(
        "local_map(spmd_types=True) does not support placement type "
        f"{type(placement).__name__}"
    )


def _dtensor_placement_if_compatible(
    local_type: Any, placement: Placement
) -> Placement | None:
    if spmd is None:
        raise RuntimeError("spmd_types=True requires the spmd_types package")
    if local_type == spmd.V:
        if type(placement) in (Shard, Partial):
            return placement
        return None
    return spmd.spmd_type_to_dtensor_placement(local_type)


def _out_spmd_types_to_grad_placements(
    flat_out: Sequence[Any],
    out_placements_tuple: tuple[PlacementType, ...],
    device_mesh: Any,
) -> tuple[PlacementType, ...]:
    if spmd is None:
        raise RuntimeError("spmd_types=True requires the spmd_types package")
    grad_outputs: list[PlacementType] = []
    for output, spec in zip(flat_out, out_placements_tuple, strict=True):
        if spec is not None and not isinstance(output, tensorplay.Tensor):
            raise ValueError(
                f"out_placements specifies {spec} but the output is "
                f"{type(output).__name__}, not a Tensor"
            )
        if spec is None:
            if isinstance(output, tensorplay.Tensor):
                raise ValueError(
                    "out_placements cannot be None for a Tensor output"
                )
            grad_outputs.append(None)
            continue

        actual_type = spmd.get_local_type(output)
        if not actual_type:
            raise ValueError(
                "output tensor has no spmd_types annotation for the requested layout"
            )
        for dim_idx, placement in enumerate(spec or ()):
            axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
            actual = actual_type.get(axis)
            if actual is None:
                raise ValueError(
                    f"output tensor has no spmd_types annotation on mesh dimension {dim_idx}"
                )
            actual_placement = _dtensor_placement_if_compatible(actual, placement)
            if actual_placement is None or type(actual_placement) is not type(placement):
                raise ValueError(
                    f"output tensor placement mismatch on {axis}: "
                    f"requested {placement}, inferred {actual.name}"
                )

        grad_spec: list[Placement] = []
        for dim_idx, placement in enumerate(spec or ()):
            axis = spmd.MeshAxis.of(device_mesh.get_group(dim_idx))
            if actual_type[axis] == spmd.V and type(placement) is Partial:
                grad_placement = Replicate()
            else:
                grad_placement = _dtensor_placement_if_compatible(
                    actual_type[axis].backward_type(), placement
                )
            if grad_placement is None:
                raise AssertionError("a gradient placement was not inferred")
            grad_spec.append(grad_placement)
        grad_outputs.append(tuple(grad_spec))
    return tuple(grad_outputs)


def local_map(
    func: Callable[..., Any] | None = None,
    out_placements: OutputPlacements = None,
    in_placements: InputPlacements = None,
    in_grad_placements: InputPlacements = None,
    device_mesh: Any = None,
    *,
    redistribute_inputs: bool = False,
    spmd_types: bool = False,
) -> Any:
    """Wrap a local tensor function with distributed input and output layouts."""
    if func is None:
        def decorated(function: Callable[..., Any]) -> Any:
            return local_map(
                function,
                out_placements,
                in_placements,
                in_grad_placements,
                device_mesh,
                redistribute_inputs=redistribute_inputs,
                spmd_types=spmd_types,
            )

        return decorated

    return functools.partial(
        _local_map_wrapped,
        func,
        out_placements,
        in_placements,
        in_grad_placements,
        device_mesh,
        redistribute_inputs,
        spmd_types,
    )


def _local_map_wrapped(
    func: Callable[..., Any],
    out_placements: OutputPlacements,
    in_placements: InputPlacements,
    in_grad_placements: InputPlacements,
    device_mesh: Any,
    redistribute_inputs: bool,
    enable_spmd_types: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    flat_args, args_spec = pytree.tree_flatten(args)
    if in_placements is not None and len(in_placements) != len(flat_args):
        raise AssertionError(
            f"in_placements length {len(in_placements)} does not match "
            f"the number of input values {len(flat_args)}"
        )
    if in_grad_placements is not None and len(in_grad_placements) != len(flat_args):
        raise AssertionError(
            f"in_grad_placements length {len(in_grad_placements)} does not match "
            f"the number of input values {len(flat_args)}"
        )

    local_args: list[Any] = []
    seen_dtensor = False
    for index, value in enumerate(flat_args):
        if isinstance(value, DTensor):
            seen_dtensor = True
            if device_mesh is None:
                device_mesh = value.device_mesh
            if in_placements is not None:
                requested = in_placements[index]
                if requested is None:
                    raise AssertionError(
                        f"DTensor input {value} expects an input placement"
                    )
                requested = tuple(requested)
                if value.placements != requested:
                    if redistribute_inputs:
                        value = value.redistribute(placements=requested)
                    else:
                        raise ValueError(
                            f"local_map input placements {value.placements} do not "
                            f"match the requested {requested}; set "
                            "redistribute_inputs=True to allow redistribution"
                        )
            if in_grad_placements is not None:
                requested_grad = in_grad_placements[index]
                if requested_grad is None:
                    raise AssertionError(
                        f"DTensor input {value} expects a gradient placement"
                    )
                local_value = value.to_local(grad_placements=tuple(requested_grad))
            else:
                local_value = value.to_local()
            if isinstance(local_value, AsyncCollectiveTensor):
                local_value = local_value.wait()
            local_args.append(local_value)
            continue

        if (
            in_placements is not None
            and not isinstance(value, tensorplay.Tensor)
            and in_placements[index] is not None
        ):
            raise AssertionError(
                f"non-tensor input {value!r} requires a None placement"
            )
        local_args.append(value)

    local_call_args = pytree.tree_unflatten(local_args, args_spec)
    if enable_spmd_types and seen_dtensor:
        if spmd is None:
            raise RuntimeError("spmd_types=True requires the spmd_types package")
        if device_mesh is None:
            raise AssertionError("a device mesh is required for spmd_types")
        _annotate_spmd_types(
            local_args, in_placements, in_grad_placements, device_mesh
        )
        from spmd_types._checker import typecheck

        with spmd.set_current_mesh(device_mesh), typecheck(strict_mode="strict"):
            output = func(*local_call_args, **kwargs)
    else:
        output = func(*local_call_args, **kwargs)

    if not seen_dtensor:
        return output

    flat_output, output_spec = pytree.tree_flatten(output)
    out_placements_tuple = (
        out_placements if isinstance(out_placements, tuple) else (out_placements,)
    )
    if len(flat_output) != len(out_placements_tuple):
        raise AssertionError(
            "local_map requires one output placement for each output value; "
            f"received {len(out_placements_tuple)} for {len(flat_output)} outputs"
        )

    grad_outputs = None
    if enable_spmd_types:
        if device_mesh is None:
            raise AssertionError("a device mesh is required for spmd_types")
        grad_outputs = _out_spmd_types_to_grad_placements(
            flat_output, out_placements_tuple, device_mesh
        )

    distributed_output: list[Any] = []
    for index, (value, placements) in enumerate(
        zip(flat_output, out_placements_tuple, strict=True)
    ):
        if isinstance(value, tensorplay.Tensor):
            if isinstance(value, DTensor):
                raise AssertionError(
                    f"local function output must be local, got {type(value)}"
                )
            gradient_placements = (
                grad_outputs[index] if grad_outputs is not None else None
            )
            distributed_output.append(
                DTensor.from_local(
                    value,
                    device_mesh,
                    placements,
                    run_check=False,
                    grad_placements=gradient_placements,
                )
            )
        else:
            if placements is not None:
                raise AssertionError(
                    f"non-tensor output {value!r} requires a None placement"
                )
            distributed_output.append(value)
    return pytree.tree_unflatten(distributed_output, output_spec)
