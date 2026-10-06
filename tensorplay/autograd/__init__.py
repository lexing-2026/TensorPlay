"""
``tensorplay.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import tensorplay
from tensorplay.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge
from .grad_mode import (
    enable_grad,
    inference_mode,
    no_grad,
    set_grad_enabled,
    is_grad_enabled,
    _unsafe_preserve_version_counter,
)

from .function import Function, NestedIOFunction
from .variable import Variable
from .graph import save_on_cpu, saved_tensor_hooks as saved_tensors_hooks
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
    "saved_tensors_hooks",
    "save_on_cpu",
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


def _make_grads(outputs, grads, is_grads_batched=False):
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
        grad_shape = tuple(grad_output.shape)
        if is_grads_batched:
            grad_shape = grad_shape[1:]
        if grad_shape != tuple(output.shape):
            if is_grads_batched:
                raise RuntimeError(
                    "If `is_grads_batched=True`, we interpret the first "
                    "dimension of each grad_output as the batch dimension. "
                    "The sizes of the remaining dimensions are expected to match "
                    f"the shape of corresponding output, but a mismatch was "
                    f"detected: grad_output[{index}] has a shape of "
                    f"{grad_shape} and output[{index}] has a shape of "
                    f"{tuple(output.shape)}. If you only want some tensors in "
                    "`grad_output` to be considered batched, consider using vmap."
                )
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


def _inputs_tuple(inputs, caller):
    """The tensors named by ``inputs``: one tensor, a sequence, or a dict's values."""
    if isinstance(inputs, tensorplay.Tensor):
        return (inputs,)
    if type(inputs) is dict:
        return tuple(inputs.values())
    if isinstance(inputs, Mapping):
        raise TypeError(
            f"`inputs` argument to `{caller}()` must be a dict, not "
            f"{type(inputs).__name__}. Other Mapping types are not supported."
        )
    return tuple(inputs)


def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
) -> None:
    r"""Compute the sum of gradients of given tensors with respect to graph leaves.

    The gradients are accumulated in the leaves' ``.grad``.  If any of
    ``tensors`` is non-scalar and requires gradient, ``grad_tensors`` gives
    the "vector" of the vector-Jacobian product for each of them (``None``
    for scalars and tensors that do not need one).

    Args:
        tensors (Sequence[Tensor] or Tensor): Tensors whose derivative is computed.
        grad_tensors (Sequence[Tensor or None] or Tensor, optional): The
            "vector" in the vector-Jacobian product, usually gradients w.r.t.
            each element of corresponding tensors.
        retain_graph (bool, optional): If ``False``, the graph used to compute
            the grad will be freed. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, the graph of the derivative
            is constructed, allowing higher order derivative products.
        grad_variables (Sequence[Tensor or None] or Tensor, optional): Deprecated
            spelling of ``grad_tensors``.
        inputs (Sequence[Tensor] or Tensor or dict, optional): Tensors w.r.t.
            which the gradient will be accumulated into ``.grad``.  All other
            tensors are ignored, and only the part of the graph that leads to
            these runs.  A non-leaf named here keeps its gradient as if it
            had called :meth:`~tensorplay.Tensor.retain_grad`.  A dict (e.g.
            ``dict(model.named_parameters())``) names its values.
    """
    if grad_variables is not None:
        warnings.warn(
            "`grad_variables` is deprecated. Use `grad_tensors` instead.",
            FutureWarning,
            stacklevel=2,
        )
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError(
                "`grad_tensors` and `grad_variables` (deprecated) "
                "arguments both passed to `backward()`. Please only "
                "use `grad_tensors`."
            )
    inputs_tuple = None
    if inputs is not None:
        inputs_tuple = _inputs_tuple(inputs, "backward")
        if len(inputs_tuple) == 0:
            raise RuntimeError("`inputs` argument to `backward()` cannot be empty.")
    outputs = _as_tuple(tensors)
    grad_tuple = _as_tuple(grad_tensors, len(outputs))
    grad_tuple = _make_grads(outputs, grad_tuple)
    if retain_graph is None:
        retain_graph = create_graph
    _backward(outputs, grad_tuple, retain_graph, create_graph, inputs_tuple)


def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: Optional[bool] = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
):
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
        only_inputs (bool, optional): Deprecated and ignored; gradients are
            only ever returned for ``inputs``.
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of
            each tensor in ``grad_outputs`` is a batch dimension: one
            vector-Jacobian product is computed per entry, in a single
            :func:`tensorplay.vmap` over the backward pass.  Default: ``False``.
        materialize_grads (bool, optional): If ``True``, the gradient of an
            unused input is returned as zeros instead of ``None``.  Default:
            ``False``.

    Returns a tuple with one gradient per input, or a dict keyed like
    ``inputs`` when ``inputs`` is a dict.
    """
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when materialize_grads=True, "
            "but got: allow_unused=False."
        )
    if allow_unused is None:
        allow_unused = materialize_grads
    if not only_inputs:
        warnings.warn(
            "only_inputs argument is deprecated and is ignored now "
            "(defaults to True). To accumulate gradient for other "
            "parts of the graph, please use tensorplay.autograd.backward.",
            FutureWarning,
            stacklevel=2,
        )
    named = inputs if type(inputs) is dict else None
    outputs = _as_tuple(outputs)
    inputs_tuple = _inputs_tuple(inputs, "grad")
    if len(inputs_tuple) == 0:
        raise RuntimeError("grad requires non-empty inputs.")

    if retain_graph is None:
        retain_graph = create_graph

    grad_outputs = _make_grads(
        outputs, _as_tuple(grad_outputs, len(outputs)), is_grads_batched
    )
    if is_grads_batched:
        # One vector-Jacobian product per entry of the batch dimension.  An
        # unused input has no gradient in any of them; vmap cannot return
        # None, so its slot is filled while batching and emptied after.
        unused: set[int] = set()

        def vjp(*batch):
            got = _grad(outputs, inputs_tuple, batch, retain_graph, create_graph, allow_unused)
            filled = []
            for index, (g, inp) in enumerate(zip(got, inputs_tuple)):
                if g is None:
                    unused.add(index)
                    g = tensorplay.zeros_like(inp)
                filled.append(g)
            return tuple(filled)

        result = tensorplay.vmap(vjp)(*grad_outputs)
        result = tuple(None if i in unused else r for i, r in enumerate(result))
    else:
        result = _grad(outputs, inputs_tuple, grad_outputs, retain_graph, create_graph, allow_unused)
    if materialize_grads:
        result = tuple(
            r if r is not None else tensorplay.zeros_like(inp, requires_grad=create_graph)
            for r, inp in zip(result, inputs_tuple)
        )
    if named is not None:
        return dict(zip(named.keys(), result))
    return result


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
