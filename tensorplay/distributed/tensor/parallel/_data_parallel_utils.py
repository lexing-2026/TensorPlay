"""Flattening helpers for data-parallel handoff."""

from __future__ import annotations

from functools import partial
from typing import Any

from ..._functional_collectives import AsyncCollectiveTensor
from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta

__all__ = ["sync_grad_hook", "_flatten_tensor", "_unflatten_tensor"]


def sync_grad_hook(
    grad: Any, *, device_handle: Any = None, compute_stream: Any = None
) -> Any:
    if not isinstance(grad, AsyncCollectiveTensor):
        return grad
    if compute_stream is not None:
        with device_handle.stream(compute_stream):
            return grad.wait()
    return grad.wait()


def _flatten_tensor(value: Any) -> tuple[Any, DTensorSpec | None]:
    if isinstance(value, DTensor):
        local = value._local_tensor
        local.requires_grad_(bool(getattr(local, "requires_grad", False)))
        meta = TensorMeta(tuple(value.shape), tuple(value.stride()), value.dtype)
        return local, DTensorSpec(value.device_mesh, value.placements, meta)
    return value, None


def _unflatten_tensor(
    value: Any,
    spec: DTensorSpec | None,
    *,
    device_handle: Any = None,
    compute_stream: Any = None,
) -> Any:
    if spec is None:
        return value
    result = DTensor.from_local(
        value,
        spec.device_mesh,
        spec.placements,
        run_check=False,
        shape=spec.shape,
        stride=spec.stride,
    )
    if getattr(value, "requires_grad", False):
        value.register_hook(
            partial(
                sync_grad_hook,
                device_handle=device_handle,
                compute_stream=compute_stream,
            )
        )
    return result
