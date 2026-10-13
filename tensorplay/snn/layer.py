"""Layers for building deep spiking networks.

These wrappers adapt ordinary ANN layers (convolution, batch normalization)
to spike trains of shape ``[T, B, ...]``.  The time loop for convolution is a
thin Python composition over the existing native ``Conv2d``; the batch
normalization variants fold the time axis into the batch dimension, which is
the standard way to stabilize deep SNN training.
"""

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import BatchNorm1d, BatchNorm2d, Conv2d, Module

from .neuron import LIFNode

__all__ = [
    "SeqToANNContainer",
    "SpikingConv2d",
    "SpikingBatchNorm1d",
    "SpikingBatchNorm2d",
]


class SeqToANNContainer(Module):
    """Apply an ANN module to every time step of a ``[T, B, ...]`` input.

    The wrapped module must accept a ``[B, ...]`` input and return a tensor
    of the same leading batch size.  The outputs are stacked into a
    ``[T, B, ...]`` tensor.
    """

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() < 2:
            raise ValueError(
                "SeqToANNContainer input must have shape [T, B, ...]"
            )
        outs = []
        for t in range(x.shape[0]):
            outs.append(self.module(x[t]))
        return tp.stack(outs)


class SpikingConv2d(Module):
    """Spiking 2D convolution: per-time-step ``Conv2d`` followed by a neuron.

    ``forward`` takes a spike train ``[T, B, C, H, W]`` and returns the
    convolved spike train of the same layout (channels become
    ``out_channels``).  The neuron is a :class:`LIFNode` in multi-step mode by
    default; pass a custom ``neuron`` (e.g. a surrogate-equipped LIF) to
    override it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        neuron=None,
    ) -> None:
        super().__init__()
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.conv = SeqToANNContainer(conv)
        self.neuron = neuron if neuron is not None else LIFNode(step_mode="m")

    def forward(self, x: Tensor) -> Tensor:
        return self.neuron(self.conv(x))

    def reset(self) -> None:
        """Clear the membrane potential of the spiking neuron."""
        self.neuron.reset()


class _SpikingBatchNormBase(Module):
    """Shared implementation: fold ``[T, B, ...]`` into ``[T*B, ...]``."""

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1,
                 affine: bool = True) -> None:
        super().__init__()
        self.bn = self._make_bn(num_features, eps, momentum, affine)

    def _make_bn(self, num_features, eps, momentum, affine):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"{type(self).__name__} input must have shape [T, B, ...]"
            )
        T, B = x.shape[0], x.shape[1]
        flat = x.reshape(T * B, *x.shape[2:])
        y = self.bn(flat)
        return y.reshape(x.shape)


class SpikingBatchNorm1d(_SpikingBatchNormBase):
    """Batch normalization over time for 1D feature maps.

    The input ``[T, B, C]`` is reshaped to ``[T*B, C]`` so the running
    statistics accumulate across both time and batch.
    """

    def _make_bn(self, num_features, eps, momentum, affine):
        return BatchNorm1d(
            num_features, eps=eps, momentum=momentum, affine=affine
        )


class SpikingBatchNorm2d(_SpikingBatchNormBase):
    """Batch normalization over time for 2D feature maps.

    The input ``[T, B, C, H, W]`` is reshaped to ``[T*B, C, H, W]`` so the
    running statistics accumulate across both time and batch.
    """

    def _make_bn(self, num_features, eps, momentum, affine):
        return BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine
        )