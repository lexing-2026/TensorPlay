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


class _FusedModule(nn.Sequential):
    pass


class _FusedActivation(_FusedModule):
    """Base for operator + relu bundles; subclasses set the operator class."""

    _OP_CLASS = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args and isinstance(args[0], self._OP_CLASS):
            self.op = args[0]
        else:
            self.op = self._OP_CLASS(*args, **kwargs)

    def __getitem__(self, index):
        return (self.op, nn.ReLU())[index]

    def forward(self, input):
        return self.op(input).relu()

    @classmethod
    def from_float(cls, mod):
        """Build the fused module from an unfused float operator module."""
        fused = cls(
            mod.in_channels if hasattr(mod, "in_channels") else mod.in_features,
            mod.out_channels if hasattr(mod, "out_channels") else mod.out_features,
            *(
                (mod.kernel_size,) if hasattr(mod, "kernel_size") else ()
            ),
            **(
                {
                    "stride": mod.stride,
                    "padding": mod.padding,
                    "dilation": mod.dilation,
                    "groups": mod.groups,
                    "bias": mod.bias is not None,
                    "padding_mode": mod.padding_mode,
                    "device": mod.weight.device,
                    "dtype": mod.weight.dtype,
                }
                if hasattr(mod, "stride")
                else {"bias": mod.bias is not None, "device": mod.weight.device, "dtype": mod.weight.dtype}
            ),
        )
        fused.op.weight = mod.weight
        if mod.bias is not None:
            fused.op.bias = mod.bias
        if hasattr(mod, "qconfig"):
            fused.qconfig = mod.qconfig
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


class ConvBn1d(_FusedModule):
    pass

__all__.append("ConvBn1d")


class ConvBnReLU1d(_FusedModule):
    pass

__all__.append("ConvBnReLU1d")


class ConvBn2d(_FusedModule):
    pass

__all__.append("ConvBn2d")


class ConvBnReLU2d(_FusedModule):
    pass

__all__.append("ConvBnReLU2d")


class ConvBn3d(_FusedModule):
    pass

__all__.append("ConvBn3d")


class ConvBnReLU3d(_FusedModule):
    pass

__all__.append("ConvBnReLU3d")


class LinearBn1d(_FusedModule):
    pass

__all__.append("LinearBn1d")


class LinearLeakyReLU(_FusedModule):
    """A linear transformation followed by a leaky rectifier."""

    def __init__(self, linear, leaky_relu):
        if not (type(linear) is nn.Linear and type(leaky_relu) is nn.LeakyReLU):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(leaky_relu).__name__}"
            )
        super().__init__(linear, leaky_relu)


__all__.append("LinearLeakyReLU")


class LinearTanh(_FusedModule):
    """A linear transformation followed by a hyperbolic tangent."""

    def __init__(self, linear, tanh):
        if not (type(linear) is nn.Linear and type(tanh) is nn.Tanh):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(tanh).__name__}"
            )
        super().__init__(linear, tanh)


__all__.append("LinearTanh")


class ConvAdd2d(_FusedModule):
    """A 2d convolution with a residual add."""

    def __init__(self, conv, add):
        super().__init__(conv)
        self.add = add

    def forward(self, x1, x2):
        return self.add(self[0](x1), x2)


__all__.append("ConvAdd2d")


class ConvAddReLU2d(_FusedModule):
    """A 2d convolution with a residual add and a rectifier."""

    def __init__(self, conv, add, relu):
        super().__init__(conv)
        self.add = add
        self.relu = relu

    def forward(self, x1, x2):
        return self.relu(self.add(self[0](x1), x2))


__all__.append("ConvAddReLU2d")
