"""Normalization modules operating directly on affine quantized storage."""

import tensorplay as tp
from tensorplay import nn

__all__ = ["LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"]


def _register_qparams(module, scale, zero_point, device, dtype):
    module.register_buffer("scale", tp.tensor(float(scale), device=device, dtype=dtype))
    module.register_buffer("zero_point", tp.tensor(int(zero_point), device=device, dtype=tp.long))


class LayerNorm(nn.LayerNorm):
    """Normalize trailing dimensions and requantize with output qparams."""

    def __init__(self, normalized_shape, weight, bias, scale, zero_point,
                 eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype)
        self.weight = weight
        self.bias = bias
        _register_qparams(self, scale, zero_point, device, dtype)

    def forward(self, input):
        return tp._C.quantized_layer_norm(
            input, list(self.normalized_shape), self.weight, self.bias,
            self.eps, float(self.scale), int(self.zero_point))

    def _get_name(self):
        return "QuantizedLayerNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls.from_reference(mod, scale, zero_point)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.normalized_shape, mod.weight, mod.bias, scale, zero_point,
                   eps=mod.eps, elementwise_affine=mod.elementwise_affine,
                   device=mod.weight.device if mod.weight is not None else None)


class GroupNorm(nn.GroupNorm):
    """Normalize channel groups and retain the input quantized dtype."""

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]

    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point,
                 eps=1e-5, affine=True, device=None, dtype=None):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine,
                         device=device, dtype=dtype)
        self.weight = weight
        self.bias = bias
        _register_qparams(self, scale, zero_point, device, dtype)

    def forward(self, input):
        return tp._C.quantized_group_norm(
            input, self.num_groups, self.weight, self.bias,
            self.eps, float(self.scale), int(self.zero_point))

    def _get_name(self):
        return "QuantizedGroupNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(mod.num_groups, mod.num_channels, mod.weight, mod.bias,
                   scale, zero_point, eps=mod.eps, affine=mod.affine,
                   device=mod.weight.device if mod.weight is not None else None)


class _InstanceNorm:
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False, track_running_stats=False,
                 device=None, dtype=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats,
                         device=device, dtype=dtype)
        self.weight = weight
        self.bias = bias
        _register_qparams(self, scale, zero_point, device, dtype)

    def forward(self, input):
        return tp._C.quantized_instance_norm(
            input, self.weight, self.bias, self.eps, float(self.scale), int(self.zero_point))

    def _get_name(self):
        return "Quantized" + type(self).__name__

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls.from_reference(mod, scale, zero_point)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.num_features, mod.weight, mod.bias, scale, zero_point,
                   eps=mod.eps, momentum=mod.momentum, affine=mod.affine,
                   track_running_stats=mod.track_running_stats,
                   device=mod.weight.device if mod.weight is not None else None)


class InstanceNorm1d(_InstanceNorm, nn.InstanceNorm1d):
    """Normalize each channel's spatial elements of a batched 1D input."""


class InstanceNorm2d(_InstanceNorm, nn.InstanceNorm2d):
    """Normalize each channel's spatial elements of a batched 2D input."""


class InstanceNorm3d(_InstanceNorm, nn.InstanceNorm3d):
    """Normalize each channel's spatial elements of a batched 3D input."""
