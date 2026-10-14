"""Spiking neuron modules and layers.

Neuron models share one membrane dynamic:

* :class:`IFNode` -- integrate-and-fire (``decay = 1``);
* :class:`LIFNode` -- leaky integrate-and-fire (``decay`` in ``[0, 1]``);
* :class:`ParametricLIFNode` -- LIF with a learnable per-neuron ``decay``;
* :class:`ALIFNode` -- LIF with an adaptive threshold.

Each module keeps its membrane potential in the buffer ``v``, supports both
single-step (``step_mode="s"``) and multi-step (``step_mode="m"``) inputs, and
can be reset with :meth:`reset`.  :class:`SpikeLinear` is a linear layer
meant to consume binary spike trains.
"""

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import Linear, Module, Parameter

from .functional import lif_seq, lif_step
from .surrogate import StraightThroughSurrogate

__all__ = [
    "IFNode",
    "LIFNode",
    "ParametricLIFNode",
    "ALIFNode",
    "SpikeLinear",
]


class LIFNode(Module):
    """Leaky integrate-and-fire neuron.

    Per step the membrane potential follows

    .. math::
        h     = decay * v + x
        s     = \\text{heaviside}(h - \\text{threshold})
        v_new = h - \\text{threshold} * s      \\text{ (soft reset)}
        v_new = h * (1 - s) + v_{\\text{reset}} * s   \\text{ (hard reset)}

    Args:
        threshold: spike threshold (a trainable :class:`Parameter` when
            ``learnable_threshold=True``).
        decay: membrane leak factor in ``[0, 1]``.
        v_reset: reset voltage after a spike; ``None`` selects the soft reset.
        surrogate: surrogate gradient object (see :mod:`tensorplay.snn.surrogate`).
        detach_reset: treat the spike in the voltage-update path as a
            constant during backward.
        step_mode: ``"s"`` for single-step inputs, ``"m"`` for ``[T, ...]``
            spike trains.
        learnable_threshold: register ``threshold`` as a trainable
            parameter.
        return_v_seq: in ``step_mode="m"``, also return the per-step voltage
            trajectory ``(spikes, v_seq)`` instead of only ``spikes``.
    """

    __constants__ = ["detach_reset", "step_mode", "return_v_seq"]

    def __init__(
        self,
        threshold=1.0,
        decay=0.5,
        v_reset=0.0,
        surrogate=None,
        detach_reset: bool = True,
        step_mode: str = "s",
        learnable_threshold: bool = False,
        return_v_seq: bool = False,
    ) -> None:
        super().__init__()
        if learnable_threshold:
            self.threshold = Parameter(tp.full([], threshold))
        else:
            self.threshold = threshold
        self.decay = decay
        self.v_reset = v_reset
        self.surrogate = (
            surrogate if surrogate is not None else StraightThroughSurrogate()
        )
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.return_v_seq = return_v_seq
        self.register_buffer("v", None)

    def reset(self) -> None:
        """Clear the membrane potential state."""
        self.v = None

    def detach(self) -> None:
        """Detach the membrane potential from the autograd graph."""
        if self.v is not None:
            self.v = self.v.detach()

    def _ensure_state(self, x: Tensor) -> Tensor:
        if self.v is None or self.v.shape != x.shape:
            self.v = tp.zeros_like(x)
        return self.v

    def _forward_single_step(self, x: Tensor) -> Tensor:
        v = self._ensure_state(x)
        spike, v_new = lif_step(
            x,
            v,
            decay=self.decay,
            threshold=self.threshold,
            v_reset=self.v_reset,
            surrogate=self.surrogate,
            detach_reset=self.detach_reset,
        )
        self.v = v_new
        return spike

    def _forward_multi_step(self, x: Tensor) -> Tensor:
        if x.dim() < 2:
            raise ValueError(
                "LIFNode multi-step input must have shape [T, B, ...]"
            )
        if self.v is not None and self.v.shape == x.shape[1:]:
            v_init = self.v
        else:
            v_init = tp.zeros_like(x[0])
        if self.return_v_seq:
            spikes, v_seq, v_final = lif_seq(
                x,
                v_init=v_init,
                decay=self.decay,
                threshold=self.threshold,
                v_reset=self.v_reset,
                surrogate=self.surrogate,
                detach_reset=self.detach_reset,
                return_v_seq=True,
            )
            self.v = v_final
            return spikes, v_seq
        spikes, v_final = lif_seq(
            x,
            v_init=v_init,
            decay=self.decay,
            threshold=self.threshold,
            v_reset=self.v_reset,
            surrogate=self.surrogate,
            detach_reset=self.detach_reset,
        )
        self.v = v_final
        return spikes

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return self._forward_single_step(x)
        if self.step_mode == "m":
            return self._forward_multi_step(x)
        raise ValueError(
            f"LIFNode step_mode must be 's' or 'm', got {self.step_mode!r}"
        )

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, decay={self.decay}, "
            f"v_reset={self.v_reset}, step_mode={self.step_mode}"
        )


class IFNode(LIFNode):
    """Integrate-and-fire neuron (``LIFNode`` with ``decay = 1``)."""

    def __init__(
        self,
        threshold=1.0,
        v_reset=0.0,
        surrogate=None,
        detach_reset: bool = True,
        step_mode: str = "s",
        learnable_threshold: bool = False,
        return_v_seq: bool = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            decay=1.0,
            v_reset=v_reset,
            surrogate=surrogate,
            detach_reset=detach_reset,
            step_mode=step_mode,
            learnable_threshold=learnable_threshold,
            return_v_seq=return_v_seq,
        )


class ParametricLIFNode(LIFNode):
    """LIF neuron with a learnable leak factor.

    ``decay`` is a trainable parameter initialized to ``decay_init``.  Pass a
    shape to create one decay value per feature (broadcast over the batch),
    e.g. ``decay_shape=(1, channels)``.
    """

    def __init__(
        self,
        threshold=1.0,
        decay_init=0.5,
        decay_shape=(),
        v_reset=0.0,
        surrogate=None,
        detach_reset: bool = True,
        step_mode: str = "s",
        learnable_threshold: bool = False,
        return_v_seq: bool = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            decay=decay_init,
            v_reset=v_reset,
            surrogate=surrogate,
            detach_reset=detach_reset,
            step_mode=step_mode,
            learnable_threshold=learnable_threshold,
            return_v_seq=return_v_seq,
        )
        self.decay = Parameter(tp.full(decay_shape, decay_init))

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, v_reset={self.v_reset}, "
            f"step_mode={self.step_mode}"
        )


class ALIFNode(LIFNode):
    """LIF neuron with an adaptive threshold.

    A slow adaptation variable ``a`` tracks the membrane potential and is
    added to the base threshold:

    .. math::
        h          = decay * v + x
        \\theta    = a + \\text{base\\_threshold}
        s          = \\text{heaviside}(h - \\theta)
        a_{\\text{new}} = decay_{\\text{th}} * a + v

    With ``decay_th < 1`` the threshold rises after strong activity and decays
    back to the base value, enabling homeostatic firing-rate regulation.

    Because the effective threshold changes every step, the multi-step mode
    runs the native single-step operator in a tight loop (the fused
    multi-step kernel assumes a constant threshold).
    """

    def __init__(
        self,
        base_threshold=1.0,
        decay=0.5,
        decay_th=0.9,
        v_reset=0.0,
        surrogate=None,
        detach_reset: bool = True,
        step_mode: str = "s",
        learnable_threshold: bool = False,
        return_v_seq: bool = False,
    ) -> None:
        super().__init__(
            threshold=base_threshold,
            decay=decay,
            v_reset=v_reset,
            surrogate=surrogate,
            detach_reset=detach_reset,
            step_mode=step_mode,
            learnable_threshold=learnable_threshold,
            return_v_seq=return_v_seq,
        )
        self.decay_th = decay_th
        self.register_buffer("a", None)

    def reset(self) -> None:
        """Clear the membrane potential and adaptation state."""
        super().reset()
        self.a = None

    def detach(self) -> None:
        """Detach the membrane potential and adaptation from autograd."""
        super().detach()
        if self.a is not None:
            self.a = self.a.detach()

    def _ensure_adaptation(self, x: Tensor) -> Tensor:
        if self.a is None or self.a.shape != x.shape:
            self.a = tp.zeros_like(x)
        return self.a

    def _threshold_tensor(self) -> Tensor:
        a = self._ensure_adaptation(self.v)
        return a + self.threshold

    def _forward_single_step(self, x: Tensor) -> Tensor:
        v = self._ensure_state(x)
        a = self._ensure_adaptation(x)
        threshold = a + self.threshold
        spike, v_new = lif_step(
            x,
            v,
            decay=self.decay,
            threshold=threshold,
            v_reset=self.v_reset,
            surrogate=self.surrogate,
            detach_reset=self.detach_reset,
        )
        self.a = a * self.decay_th + v
        self.v = v_new
        return spike

    def _forward_multi_step(self, x: Tensor):
        if x.dim() < 2:
            raise ValueError(
                "ALIFNode multi-step input must have shape [T, B, ...]"
            )
        v_init = (
            self.v
            if self.v is not None and self.v.shape == x.shape[1:]
            else tp.zeros_like(x[0])
        )
        a_init = (
            self.a
            if self.a is not None and self.a.shape == x.shape[1:]
            else tp.zeros_like(x[0])
        )
        v = v_init
        a = a_init
        spikes = []
        v_seq = []
        for t in range(x.shape[0]):
            threshold = a + self.threshold
            spike_t, v_new = lif_step(
                x[t],
                v,
                decay=self.decay,
                threshold=threshold,
                v_reset=self.v_reset,
                surrogate=self.surrogate,
                detach_reset=self.detach_reset,
            )
            spikes.append(spike_t)
            v_seq.append(v_new)
            a = a * self.decay_th + v
            v = v_new
        self.v = v
        self.a = a
        spikes_out = tp.stack(spikes)
        if self.return_v_seq:
            return spikes_out, tp.stack(v_seq)
        return spikes_out

    def extra_repr(self) -> str:
        return (
            f"base_threshold={self.threshold}, decay={self.decay}, "
            f"decay_th={self.decay_th}, v_reset={self.v_reset}, "
            f"step_mode={self.step_mode}"
        )


class SpikeLinear(Module):
    """Linear layer for spike-train inputs.

    A thin wrapper around :class:`tensorplay.nn.Linear` that keeps the spiking
    convention explicit: the input is expected to be a binary ``[..., in]``
    tensor (or a ``[T, ..., in]`` spike train) and the output is the weighted
    sum, which downstream neurons integrate.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def extra_repr(self) -> str:
        return f"in_features={self.linear.in_features}, out_features={self.linear.out_features}"