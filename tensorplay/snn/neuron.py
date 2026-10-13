"""Spiking neuron modules.

Three neuron models share one membrane dynamic:

* :class:`IFNode` -- integrate-and-fire (``decay = 1``);
* :class:`LIFNode` -- leaky integrate-and-fire (``decay`` in ``[0, 1]``);
* :class:`ParametricLIFNode` -- LIF with a learnable per-neuron ``decay``.

Each module keeps its membrane potential in the buffer ``v``, supports both
single-step (``step_mode="s"``) and multi-step (``step_mode="m"``) inputs, and
can be reset with :meth:`reset`.
"""

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import Module, Parameter

from .functional import lif_seq, lif_step
from .surrogate import StraightThroughSurrogate

__all__ = ["IFNode", "LIFNode", "ParametricLIFNode"]


class LIFNode(Module):
    """Leaky integrate-and-fire neuron.

    Per step the membrane potential follows

    .. math::
        h     = decay * v + x
        s     = \\text{heaviside}(h - \\text{threshold})
        v_new = h - \\text{threshold} * s      \\text{ (soft reset)}
        v_new = h * (1 - s) + v_{\\text{reset}} * s   \\text{ (hard reset)}

    Args:
        threshold: spike threshold.
        decay: membrane leak factor in ``[0, 1]``.
        v_reset: reset voltage after a spike; ``None`` selects the soft reset.
        surrogate: surrogate gradient object (see :mod:`tensorplay.snn.surrogate`).
        detach_reset: treat the spike in the voltage-update path as a
            constant during backward.
        step_mode: ``"s"`` for single-step inputs, ``"m"`` for ``[T, ...]``
            spike trains.
    """

    __constants__ = ["detach_reset", "step_mode"]

    def __init__(
        self,
        threshold=1.0,
        decay=0.5,
        v_reset=0.0,
        surrogate=None,
        detach_reset: bool = True,
        step_mode: str = "s",
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.v_reset = v_reset
        self.surrogate = (
            surrogate if surrogate is not None else StraightThroughSurrogate()
        )
        self.detach_reset = detach_reset
        self.step_mode = step_mode
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

    def forward(self, x: Tensor) -> Tensor:
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
    ) -> None:
        super().__init__(
            threshold=threshold,
            decay=1.0,
            v_reset=v_reset,
            surrogate=surrogate,
            detach_reset=detach_reset,
            step_mode=step_mode,
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
    ) -> None:
        super().__init__(
            threshold=threshold,
            decay=decay_init,
            v_reset=v_reset,
            surrogate=surrogate,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
        self.decay = Parameter(tp.full(decay_shape, decay_init))

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, v_reset={self.v_reset}, "
            f"step_mode={self.step_mode}"
        )