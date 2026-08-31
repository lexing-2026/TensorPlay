"""
``tensorplay.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

from collections.abc import Sequence
from typing import Optional, Union

import tensorplay
from tensorplay.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge
from .grad_mode import (
    enable_grad,
    inference_mode,
    no_grad,
    set_grad_enabled,
    is_grad_enabled,
)

from .function import Function, NestedIOFunction
from .variable import Variable
from .graph import saved_tensor_hooks as saved_tensors_hooks
from . import forward_ad
from .._C._autograd import (
    backward as _backward,
    grad as _grad,
    is_anomaly_enabled,
    is_anomaly_check_nan_enabled,
    is_inference_mode_enabled,
)


__all__ = [
    "Function",
    "Variable",
    "NestedIOFunction",
    "backward",
    "grad",
    "grad_mode",
    "enable_grad",
    "is_grad_enabled",
    "inference_mode",
    "no_grad",
    "set_grad_enabled",
    # inference mode
    "is_inference_mode_enabled",
    # anomaly mode
    "detect_anomaly",
    "set_detect_anomaly",
    "is_anomaly_enabled",
    "is_anomaly_check_nan_enabled",
    # functional API for higher-order derivatives
    "jacobian",
    "hessian",
    "vjp",
    "vhp",
    "hvp",
    "jvp",
]

_OptionalTensor = Optional[tensorplay.Tensor]
_ShapeorNestedShape = Union[_size, Sequence[_size], tensorplay.Tensor]


def _as_tuple(value, length=None):
    if value is None:
        return (None,) * length if length is not None else None
    if isinstance(value, tensorplay.Tensor):
        return (value,)
    return tuple(value)


def _make_grads(outputs, grads):
    if len(outputs) != len(grads):
        raise RuntimeError(
            "The number of grad_outputs must match the number of outputs"
        )
    result = []
    for index, (output, grad_output) in enumerate(zip(outputs, grads)):
        if not isinstance(output, tensorplay.Tensor):
            raise TypeError("outputs must contain tensorplay.Tensor values")
        if grad_output is None:
            if output.requires_grad:
                if output.numel() != 1:
                    raise RuntimeError(
                        "grad can be implicitly created only for scalar outputs"
                    )
                if not output.dtype.is_floating_point:
                    raise RuntimeError(
                        "grad can be implicitly created only for real scalar outputs"
                    )
            result.append(tensorplay.ones_like(output))
            continue
        if not isinstance(grad_output, tensorplay.Tensor):
            raise TypeError(
                "gradients can be either Tensors or None, but got "
                + type(grad_output).__name__
            )
        if tuple(grad_output.shape) != tuple(output.shape):
            raise RuntimeError(
                f"Mismatch in shape: grad_output[{index}] has a shape of "
                f"{grad_output.shape} and output[{index}] has a shape of "
                f"{output.shape}."
            )
        if output.dtype.is_complex != grad_output.dtype.is_complex:
            raise RuntimeError(
                "For complex Tensors, both grad_output and output are "
                "required to have the same dtype category."
            )
        result.append(grad_output)
    return tuple(result)


def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
) -> None:
    outputs = _as_tuple(tensors)
    grad_tuple = _as_tuple(grad_tensors, len(outputs))
    grad_tuple = _make_grads(outputs, grad_tuple)
    if retain_graph is None:
        retain_graph = create_graph
    _backward(outputs, grad_tuple, retain_graph, create_graph)


def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: Optional[bool] = None,
) -> tuple[Optional[tensorplay.Tensor], ...]:
    r"""Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    Args:
        outputs (sequence of Tensor or GradientEdge): outputs of the differentiated function.
        inputs (sequence of Tensor or GradientEdge): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the vector-Jacobian product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.

    """
    outputs = _as_tuple(outputs)
    inputs = _as_tuple(inputs)

    if retain_graph is None:
        retain_graph = create_graph
    if allow_unused is None:
        allow_unused = False

    grad_outputs = _make_grads(
        outputs, _as_tuple(grad_outputs, len(outputs))
    )
    return _grad(outputs, inputs, grad_outputs, retain_graph, create_graph, allow_unused)


from .anomaly_mode import detect_anomaly, set_detect_anomaly  # noqa: E402
from . import functional as functional  # noqa: E402
from .functional import jacobian, hessian, vjp, vhp, hvp, jvp  # noqa: E402
from .gradcheck import (  # noqa: E402
    gradcheck,
    gradgradcheck,
    GradcheckError,
)

from . import profiler as profiler  # noqa: E402
from .profiler import emit_nvtx as emit_nvtx  # noqa: E402
from . import profiler_legacy as profiler_legacy  # noqa: E402
