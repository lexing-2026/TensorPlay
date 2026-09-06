"""Flattening a parameter iterable into one vector and back."""

from typing import Iterable

import tensorplay as tp
from tensorplay import Tensor

__all__ = ["parameters_to_vector", "vector_to_parameters"]


def _check_device(param: Tensor, old_device: int) -> int:
    device = param.device
    if old_device is not None and device != old_device:
        raise TypeError(
            "Found two parameters on different devices, this is currently not "
            f"supported: {old_device} and {device}"
        )
    return device


def parameters_to_vector(parameters: Iterable[Tensor]) -> Tensor:
    """Concatenate the parameters into a single 1-D tensor."""
    device = None
    views = []
    for param in parameters:
        device = _check_device(param, device)
        views.append(param.reshape([-1]))
    if not views:
        raise ValueError("parameters_to_vector: expected at least one parameter")
    return tp.cat(views, 0)


def vector_to_parameters(vec: Tensor, parameters: Iterable[Tensor]) -> None:
    """Copy slices of `vec` back into the parameters, in order."""
    if not isinstance(vec, Tensor):
        raise TypeError(f"expected a Tensor, but got: {type(vec)}")
    device = None
    offset = 0
    for param in parameters:
        device = _check_device(param, device)
        count = param.numel()
        with tp.no_grad():
            param.copy_(tp.narrow(vec, 0, offset, count).reshape(list(param.shape)))
        offset += count
