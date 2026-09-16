"""Fused float building blocks.

These modules bundle an operator with the activation that immediately
follows it.  They are the fusion targets of the eager workflow and the float
counterparts used during quantization-aware training.
"""

from __future__ import annotations

from tensorplay import nn

__all__ = [
    "LinearReLU",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "BNReLU2d",
    "BNReLU3d",
]


class _FusedActivation(nn.Module):
    """Base for operator + relu bundles; subclasses set the operator class."""

    _OP_CLASS = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.op = self._OP_CLASS(*args, **kwargs)

    def forward(self, input):
        return self.op(input).relu()

    @classmethod
    def from_float(cls, mod):
        """Build the fused module from an unfused float operator module."""
        fused = cls(
            mod.in_channels if hasattr(mod, "in_channels") else mod.in_features,
            mod.out_channels if hasattr(mod, "out_channels") else mod.out_features,
            *(
                mod.kernel_size if hasattr(mod, "kernel_size") else ()
            ),
            **(
                {
                    "stride": mod.stride,
                    "padding": mod.padding,
                    "dilation": mod.dilation,
                    "groups": mod.groups,
                    "bias": mod.bias is not None,
                }
                if hasattr(mod, "stride")
                else {"bias": mod.bias is not None}
            ),
        )
        fused.op.weight = mod.weight
        if mod.bias is not None:
            fused.op.bias = mod.bias
        fused.train(mod.training)
        return fused


class LinearReLU(_FusedActivation):
    """A linear transformation followed by a rectifier."""

    _OP_CLASS = nn.Linear


class ConvReLU1d(_FusedActivation):
    """A 1d convolution followed by a rectifier."""

    _OP_CLASS = nn.Conv1d


class ConvReLU2d(_FusedActivation):
    """A 2d convolution followed by a rectifier."""

    _OP_CLASS = nn.Conv2d


class ConvReLU3d(_FusedActivation):
    """A 3d convolution followed by a rectifier."""

    _OP_CLASS = nn.Conv3d


class BNReLU2d(nn.Module):
    """A 2d batch norm followed by a rectifier."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.bn(input))


class BNReLU3d(nn.Module):
    """A 3d batch norm followed by a rectifier."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, eps, momentum)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.bn(input))
