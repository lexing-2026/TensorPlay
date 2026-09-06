"""Gradient clipping helpers."""

import types
import warnings
from typing import Iterable, Optional, Union

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

__all__ = [
    "clip_grad_norm",
    "clip_grad_norm_",
    "clip_grad_value_",
    "clip_grads_with_norm_",
    "get_total_norm",
]

_TensorOrTensors = Union[Tensor, Iterable[Tensor]]


def _as_list(tensors, what: str, warn_empty: bool = False) -> list:
    if isinstance(tensors, Tensor):
        return [tensors]
    is_generator = isinstance(tensors, types.GeneratorType)
    listed = list(tensors)
    if warn_empty and is_generator and not listed:
        warnings.warn(
            f"`{what}` is an empty generator, no gradient clipping will occur.",
            stacklevel=3,
        )
    return listed


def _no_grad(func):
    """Wrap ``func`` so its body runs with autograd recording disabled."""
    import functools

    def _no_grad_wrapper(*args, **kwargs):
        with tp.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


@_no_grad
def get_total_norm(
    tensors: _TensorOrTensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> Tensor:
    """Norm of `tensors` taken as if they were one concatenated vector.

    Each tensor contributes its own p-norm, and those per-tensor norms are
    reduced by the same p-norm -- which is exactly the norm of the
    concatenation.  ``foreach`` selects the batched foreach kernels; the
    default ``None`` uses them where the device supports them and falls back
    to per-tensor ops elsewhere.
    """
    tensors = _as_list(tensors, "tensors")
    norm_type = float(norm_type)
    if not tensors:
        return tp.tensor(0.0)

    first_device = tensors[0].device
    grouped = _group_tensors_by_device_and_dtype([tensors])

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(tp._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on "
                f"{device.type} tensors"
            )
        else:
            norms.extend(
                [tp.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = tp.linalg.vector_norm(
        tp.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and bool(
        tp.logical_or(tp.isnan(total_norm), tp.isinf(total_norm)).item()
    ):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


@_no_grad
def clip_grads_with_norm_(
    parameters: _TensorOrTensors,
    max_norm: float,
    total_norm: Tensor,
    foreach: Optional[bool] = None,
) -> None:
    """Scale every gradient by ``min(max_norm / (total_norm + 1e-6), 1)``.

    The coefficient is clamped at one, so gradients are only ever scaled down;
    multiplying by the clamped value unconditionally keeps the decision on the
    device holding the gradients.  Gradients are modified in place.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    max_norm = float(max_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = tp.clamp(clip_coef, max=1.0)
    grouped_grads = _group_tensors_by_device_and_dtype([grads])

    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            tp._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on "
                f"{device.type} tensors"
            )
        else:
            coef = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(coef)


def clip_grad_norm_(
    parameters: _TensorOrTensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> Tensor:
    """Clip the gradient norm of an iterable of parameters, in place.

    The norm is taken over the gradients as if they were concatenated into one
    vector; the total norm is returned whether or not any clipping happened.
    """
    parameters = _as_list(parameters, "parameters", warn_empty=True)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def clip_grad_norm(
    parameters: _TensorOrTensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> Tensor:
    """Clip the gradient norm of an iterable of parameters.

    .. deprecated:: use :func:`clip_grad_norm_` instead.
    """
    warnings.warn(
        "`tensorplay.nn.utils.clip_grad_norm` is now deprecated in favor of "
        "`tensorplay.nn.utils.clip_grad_norm_`.",
        FutureWarning,
        stacklevel=2,
    )
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite,
                           foreach)


def clip_grad_value_(
    parameters: _TensorOrTensors,
    clip_value: float,
    foreach: Optional[bool] = None,
) -> None:
    """Clamp every gradient element into ``[-clip_value, clip_value]``."""
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)

    grads = [p.grad for p in parameters if p.grad is not None]
    grouped_grads = _group_tensors_by_device_and_dtype([grads])

    with tp.no_grad():
        for (device, _), ([device_grads], _) in grouped_grads.items():
            if (foreach is None and _has_foreach_support(device_grads, device)) or (
                foreach and _device_has_foreach_support(device)
            ):
                tp._foreach_clamp_min_(device_grads, -clip_value)
                tp._foreach_clamp_max_(device_grads, clip_value)
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the foreach API on "
                    f"{device.type} tensors"
                )
            else:
                for grad in device_grads:
                    grad.clamp_(min=-clip_value, max=clip_value)
