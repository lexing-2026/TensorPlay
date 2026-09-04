"""Activation input resharding hooks."""

from __future__ import annotations

from functools import partial
from typing import Any

import tensorplay as tp
from tensorplay.autograd.graph import saved_tensors_hooks

from .._api import DTensor
from ..placement_types import Replicate, Shard, _is_shard_like

__all__ = ["input_reshard"]


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def _layout(mesh: Any, placement: Any) -> tuple[Any, ...]:
    return (placement,) + tuple(
        Replicate() for _ in range(_mesh_ndim(mesh) - 1)
    )


def input_reshard(
    module: Any,
    tp_device_mesh: Any,
    input_reshard_dim: int | None = None,
) -> Any:
    if input_reshard_dim is None:
        return module

    hook_context: Any = None

    def input_reshard_forward_pre_hook(
        current: Any, inputs: tuple[Any, ...]
    ) -> None:
        del current, inputs
        context = saved_tensors_hooks(
            partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim),
            partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim),
        )
        context.__enter__()
        nonlocal hook_context
        hook_context = context

    def input_reshard_backward_hook(
        current: Any, inputs: tuple[Any, ...], output: Any
    ) -> None:
        del current, inputs, output
        nonlocal hook_context
        if hook_context is not None:
            hook_context.__exit__(None, None, None)
            hook_context = None

    module.register_forward_pre_hook(input_reshard_forward_pre_hook)
    module.register_forward_hook(input_reshard_backward_hook)
    return module


def _pack_hook_tp(mesh: Any, input_reshard_dim: int, value: Any) -> Any:
    if isinstance(value, DTensor) and all(
        isinstance(placement, Replicate) for placement in value.placements
    ):
        return value.redistribute(
            device_mesh=mesh,
            placements=_layout(mesh, Shard(input_reshard_dim)),
        )
    if (
        not isinstance(value, DTensor)
        and isinstance(value, tp.Tensor)
        and value.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                value,
                device_mesh=mesh,
                placements=_layout(mesh, Replicate()),
                run_check=False,
            )
            .redistribute(
                device_mesh=mesh,
                placements=_layout(mesh, Shard(input_reshard_dim)),
            )
            .to_local()
        )
    return value


def _unpack_hook_tp(mesh: Any, input_reshard_dim: int, value: Any) -> Any:
    if (
        isinstance(value, DTensor)
        and len(value.placements) == 1
        and _is_shard_like(value.placements[0])
    ):
        return value.redistribute(
            device_mesh=mesh,
            placements=_layout(mesh, Replicate()),
        )
    if (
        not isinstance(value, DTensor)
        and isinstance(value, tp.Tensor)
        and value.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                value,
                device_mesh=mesh,
                placements=_layout(mesh, Shard(input_reshard_dim)),
                run_check=False,
            )
            .redistribute(
                device_mesh=mesh,
                placements=_layout(mesh, Replicate()),
            )
            .to_local()
        )
    return value
