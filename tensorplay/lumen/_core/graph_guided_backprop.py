#!/usr/bin/env python3

"""Graph-rewrite implementations of GuidedBackprop and Deconvolution.

Instead of registering backward hooks on ReLU modules (the classic approach),
these classes trace the model into a :class:`GraphModule` once and rewrite the
ReLU nodes to call custom autograd functions that override the backward pass.
The rewritten graph is then used for every forward and backward evaluation.

Tracing is performed lazily on the first ``attribute`` call. If tracing fails
(e.g. data-dependent control flow), the implementation falls back to the
hook-based strategy from :mod:`guided_backprop_deconvnet`, so the public
behavior stays available for any model.
"""

import warnings
from typing import Any, Callable, cast, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor
from tensorplay.graph import GraphModule

from tensorplay.lumen._core.guided_backprop_deconvnet import (
    ModifiedReluGradientAttribution,
)
from tensorplay.lumen._utils.common import (
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from tensorplay.lumen._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from tensorplay.lumen._utils.graph_compile import trace_with_rewritten_relu
from tensorplay.lumen._utils.typing import (
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from tensorplay.nn import Module


class _GuidedRelu(tensorplay.autograd.Function):
    """ReLU whose backward keeps only positive activations and gradients."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return tensorplay.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return tensorplay.relu(grad_output) * (x > 0).to(grad_output.dtype)


class _DeconvRelu(tensorplay.autograd.Function):
    """ReLU whose backward applies ReLU to the incoming gradient."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        return tensorplay.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        return tensorplay.relu(grad_output)


class _GraphModifiedReluGradientAttribution(ModifiedReluGradientAttribution):
    _relu_fn: type

    def __init__(self, model: Module) -> None:
        ModifiedReluGradientAttribution.__init__(self, model)
        self._graph_module: Optional[GraphModule] = None
        self._fallback_hooks = False

    def _get_forward_fn(self) -> Callable:
        if self._graph_module is not None:
            return self._graph_module
        graph_module = trace_with_rewritten_relu(self.model, self._relu_fn)
        if graph_module is not None:
            self._graph_module = graph_module
            return graph_module
        warnings.warn(
            "Could not trace the model into a graph; falling back to "
            "registering backward hooks on ReLU modules.",
            stacklevel=2,
        )
        self._fallback_hooks = True
        self.model.apply(self._register_hooks)
        return self.model

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs_tuple)

        try:
            forward_fn = self._get_forward_fn()
            gradients = self.gradient_func(
                forward_fn, inputs_tuple, target, additional_forward_args
            )
        finally:
            if self._fallback_hooks:
                self._remove_hooks()

        undo_gradient_requirements(inputs_tuple, gradient_mask)
        return cast(
            TensorOrTupleOfTensorsGeneric, _format_output(is_inputs_tuple, gradients)
        )


class GraphGuidedBackprop(_GraphModifiedReluGradientAttribution):
    r"""GuidedBackprop computed through a rewritten graph.

    The forward model is traced once and every ReLU node is replaced with a
    custom autograd function whose backward only lets gradients of positive
    activations through. Subsequent attributions reuse the rewritten graph
    without re-tracing.

    See https://arxiv.org/abs/1412.6806 for the algorithm.
    """

    _relu_fn = _GuidedRelu

    def __init__(self, model: Module) -> None:
        _GraphModifiedReluGradientAttribution.__init__(self, model)


class GraphDeconvolution(_GraphModifiedReluGradientAttribution):
    r"""Deconvolution computed through a rewritten graph.

    The forward model is traced once and every ReLU node is replaced with a
    custom autograd function whose backward applies ReLU to the incoming
    gradient. Subsequent attributions reuse the rewritten graph without
    re-tracing.

    See https://arxiv.org/abs/1311.2901 for the algorithm.
    """

    _relu_fn = _DeconvRelu

    def __init__(self, model: Module) -> None:
        _GraphModifiedReluGradientAttribution.__init__(self, model)


__all__ = ["GraphGuidedBackprop", "GraphDeconvolution"]