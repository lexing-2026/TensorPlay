"""Apply a callable to local tensor values and rebuild distributed outputs."""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence

from .._api import DTensor
from ..placement_types import Placement

__all__ = ["local_map"]


def _flatten(value: Any) -> tuple[list[Any], Any]:
    if isinstance(value, tuple):
        leaves = []
        specs = []
        for item in value:
            current, spec = _flatten(item)
            leaves.extend(current)
            specs.append(spec)
        return leaves, (tuple, tuple(specs))
    if isinstance(value, list):
        leaves = []
        specs = []
        for item in value:
            current, spec = _flatten(item)
            leaves.extend(current)
            specs.append(spec)
        return leaves, (list, tuple(specs))
    return [value], None


def _unflatten(leaves: list[Any], spec: Any, cursor: list[int]) -> Any:
    if spec is None:
        value = leaves[cursor[0]]
        cursor[0] += 1
        return value
    kind, children = spec
    values = [_unflatten(leaves, child, cursor) for child in children]
    return kind(values) if kind is list else tuple(values)


def local_map(
    func: Callable[..., Any] | None = None,
    out_placements: Any = None,
    in_placements: Sequence[Any] | None = None,
    in_grad_placements: Any = None,
    device_mesh: Any = None,
    *,
    redistribute_inputs: bool = False,
    spmd_types: bool = False,
) -> Any:
    del in_grad_placements, spmd_types
    if func is None:
        return functools.partial(local_map, out_placements=out_placements, in_placements=in_placements, device_mesh=device_mesh, redistribute_inputs=redistribute_inputs)

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        leaves, spec = _flatten(args)
        distributed = [value for value in leaves if isinstance(value, DTensor)]
        mesh = device_mesh or (distributed[0].device_mesh if distributed else None)
        local_args = []
        for index, value in enumerate(leaves):
            if not isinstance(value, DTensor):
                local_args.append(value)
                continue
            if in_placements is not None and index < len(in_placements) and in_placements[index] is not None:
                desired = tuple(in_placements[index])
                if value.placements != desired:
                    if not redistribute_inputs:
                        raise ValueError("local_map input placement does not match the requested layout")
                    value = value.redistribute(placements=desired)
            local_args.append(value.to_local())
        result = func(*_unflatten(local_args, spec, [0]), **kwargs)
        if not distributed or out_placements is None:
            return result
        output_leaves, output_spec = _flatten(result)
        placement_values = out_placements if isinstance(out_placements, tuple) and output_spec is not None else (out_placements,)
        wrapped_output = []
        for value, placements in zip(output_leaves, placement_values):
            if placements is None:
                wrapped_output.append(value)
            else:
                wrapped_output.append(DTensor.from_local(value, mesh, placements, run_check=False))
        return _unflatten(wrapped_output, output_spec, [0])

    return wrapped
