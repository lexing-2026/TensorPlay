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

from .function import Function
from .._C._autograd import backward, grad as _grad


__all__ = [
    "Function",
    "backward",
    "grad_mode",
    "enable_grad",
    "is_grad_enabled",
    "grad",
    "inference_mode",
    "no_grad",
    "set_grad_enabled",
]

_OptionalTensor = Optional[tensorplay.Tensor]
_ShapeorNestedShape = Union[_size, Sequence[_size], tensorplay.Tensor]


def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: Optional[bool] = None,
) -> tuple[tensorplay.Tensor, ...]:
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
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
        
    if retain_graph is None:
        retain_graph = create_graph
    if allow_unused is None:
        allow_unused = False
        
    return _grad(outputs, inputs, grad_outputs, retain_graph, create_graph, allow_unused)


__all__ = ["backward", "grad", "no_grad", "enable_grad", "set_grad_enabled", "is_grad_enabled", "Function"]
