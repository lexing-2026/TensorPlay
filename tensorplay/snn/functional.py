"""Functional interface for the spiking neural network primitives.

The heavy lifting lives in native fused operators (``snn_lif_step`` and
``snn_lif_step_backward``); this layer only wires them into the autograd
engine and converts Python scalars into the 0-dim tensors the operators
expect.
"""

import tensorplay as tp
import tensorplay._C as _C
from tensorplay import Tensor
from tensorplay.autograd import Function

from .surrogate import StraightThroughSurrogate

__all__ = ["heaviside", "spike", "surrogate_grad", "lif_step", "lif_seq"]


def heaviside(x: Tensor) -> Tensor:
    """Spike step: 1 where ``x >= 0``, 0 elsewhere."""
    one = tp.tensor(1, dtype=x.dtype, device=x.device)
    return tp.heaviside(x, one)


def surrogate_grad(x: Tensor, surrogate: int = 0, alpha: float = 1.0) -> Tensor:
    """Evaluate the surrogate gradient of the spike step at ``x``."""
    return _C.snn_surrogate_grad(x, surrogate=surrogate, alpha=alpha)


class _SpikeFunction(Function):
    @staticmethod
    def forward(ctx, x, surrogate):
        ctx.save_for_backward(x)
        ctx.surrogate = surrogate
        one = tp.tensor(1, dtype=x.dtype, device=x.device)
        return tp.heaviside(x, one)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * ctx.surrogate.grad(x), None


def spike(x: Tensor, surrogate=None) -> Tensor:
    """Spike with surrogate gradient.

    Forward emits the Heaviside step; backward multiplies by the surrogate
    gradient selected by ``surrogate``.
    """
    if surrogate is None:
        surrogate = StraightThroughSurrogate()
    return _SpikeFunction.apply(x, surrogate)


def _as_tensor(value, like, name):
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (int, float)):
        return tp.tensor(value, dtype=like.dtype, device=like.device)
    raise TypeError(
        f"{name} must be a Tensor or a number, got {type(value).__name__}"
    )


class _LIFStepFunction(Function):
    @staticmethod
    def forward(ctx, input, v, decay, threshold, v_reset, surrogate_id, alpha,
                detach_reset):
        spike, v_new = _C.snn_lif_step(
            input=input,
            v=v,
            decay=decay,
            threshold=threshold,
            v_reset=v_reset,
            surrogate=surrogate_id,
            alpha=alpha,
            detach_reset=detach_reset,
        )
        ctx.save_for_backward(input, v, decay, threshold, spike, v_new)
        ctx.v_reset = v_reset
        ctx.surrogate_id = surrogate_id
        ctx.alpha = alpha
        ctx.detach_reset = detach_reset
        return spike, v_new

    @staticmethod
    def backward(ctx, grad_spike, grad_v_new):
        input, v, decay, threshold, spike, v_new = ctx.saved_tensors
        grad_input, grad_v_init, grad_decay, grad_threshold = (
            _C.snn_lif_step_backward(
                grad_spike=grad_spike,
                grad_v=grad_v_new,
                input=input,
                v=v,
                decay=decay,
                threshold=threshold,
                spike=spike,
                v_new=v_new,
                v_reset=ctx.v_reset,
                surrogate=ctx.surrogate_id,
                alpha=ctx.alpha,
                detach_reset=ctx.detach_reset,
            )
        )
        return (
            grad_input,
            grad_v_init,
            grad_decay,
            grad_threshold,
            None,
            None,
            None,
            None,
        )


def lif_step(
    input: Tensor,
    v: Tensor,
    decay=1.0,
    threshold=1.0,
    v_reset=0.0,
    surrogate=None,
    detach_reset: bool = True,
):
    """Single-step leaky integrate-and-fire update.

    Computes ``h = decay * v + input``, emits a spike when ``h`` crosses
    ``threshold``, and returns ``(spike, v_new)``.  ``v_reset=None`` selects
    the soft reset (subtract the threshold); a number selects the hard reset
    to that voltage.
    """
    if surrogate is None:
        surrogate = StraightThroughSurrogate()
    decay_t = _as_tensor(decay, input, "decay")
    threshold_t = _as_tensor(threshold, input, "threshold")
    v_reset_t = None if v_reset is None else _as_tensor(v_reset, input, "v_reset")
    return _LIFStepFunction.apply(
        input,
        v,
        decay_t,
        threshold_t,
        v_reset_t,
        surrogate.id,
        surrogate.alpha,
        detach_reset,
    )


class _LIFSeqFunction(Function):
    @staticmethod
    def forward(ctx, input, v_init, decay, threshold, v_reset, surrogate_id,
                alpha, detach_reset):
        spikes, v_final = _C.snn_lif_seq(
            input=input,
            v_init=v_init,
            decay=decay,
            threshold=threshold,
            v_reset=v_reset,
            surrogate=surrogate_id,
            alpha=alpha,
            detach_reset=detach_reset,
        )
        ctx.save_for_backward(input, v_init, decay, threshold, spikes, v_final)
        ctx.v_reset = v_reset
        ctx.surrogate_id = surrogate_id
        ctx.alpha = alpha
        ctx.detach_reset = detach_reset
        return spikes, v_final

    @staticmethod
    def backward(ctx, grad_spikes, grad_v_final):
        input, v_init, decay, threshold, spikes, v_final = ctx.saved_tensors
        grad_input, grad_v_init, grad_decay, grad_threshold = (
            _C.snn_lif_seq_backward(
                grad_spikes=grad_spikes,
                grad_v_final=grad_v_final,
                input=input,
                v_init=v_init,
                decay=decay,
                threshold=threshold,
                spikes=spikes,
                v_final=v_final,
                v_reset=ctx.v_reset,
                surrogate=ctx.surrogate_id,
                alpha=ctx.alpha,
                detach_reset=ctx.detach_reset,
            )
        )
        return (
            grad_input,
            grad_v_init,
            grad_decay,
            grad_threshold,
            None,
            None,
            None,
            None,
        )


def lif_seq(
    input: Tensor,
    v_init=None,
    decay=1.0,
    threshold=1.0,
    v_reset=0.0,
    surrogate=None,
    detach_reset: bool = True,
):
    """Multi-step leaky integrate-and-fire update on ``[T, B, ...]`` input.

    Runs the single-step neuron dynamics ``T`` times inside one fused native
    kernel (no Python-level time loop) and returns ``(spikes, v_final)`` with
    shapes ``[T, B, ...]`` and ``[B, ...]``.  ``v_init`` defaults to zeros with
    the shape of ``input.shape[1:]``; pass an explicit tensor to start from a
    different membrane potential.  ``v_reset=None`` selects the soft reset.
    """
    if surrogate is None:
        surrogate = StraightThroughSurrogate()
    if input.dim() < 2:
        raise ValueError("lif_seq: input must have shape [T, B, ...]")
    if v_init is None:
        v_init = tp.zeros(
            list(input.shape[1:]), dtype=input.dtype, device=input.device
        )
    decay_t = _as_tensor(decay, input, "decay")
    threshold_t = _as_tensor(threshold, input, "threshold")
    v_reset_t = None if v_reset is None else _as_tensor(v_reset, input, "v_reset")
    return _LIFSeqFunction.apply(
        input,
        v_init,
        decay_t,
        threshold_t,
        v_reset_t,
        surrogate.id,
        surrogate.alpha,
        detach_reset,
    )