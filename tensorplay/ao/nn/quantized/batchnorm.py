"""Quantized batch norm modules.

The float affine parameters and running statistics are kept as buffers; the
output activation qparams are declared on the module and the native kernel
computes the normalized, requantized result.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay._C import quantized_batch_norm as _quantized_batch_norm

__all__ = ["BatchNorm2d", "BatchNorm3d"]


class _BatchNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None,
                 dtype=None):
        super().__init__(num_features, eps, momentum, True, True,
                         device=device, dtype=dtype)
        self.register_buffer("scale", tensorplay.tensor(1.0))
        self.register_buffer("zero_point", tensorplay.tensor(0, dtype=tensorplay.long))

    def _resolve_output_qparams(self):
        return float(self.scale), int(self.zero_point)

    def _forward_quantized(self, input, check_dim):
        if input.dim() != check_dim:
            raise ValueError(
                f"{type(self).__name__}: expected a {check_dim}d input")
        scale, zero_point = self._resolve_output_qparams()
        return _quantized_batch_norm(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, scale, zero_point)

    @classmethod
    def _from_float(cls, mod):
        if not isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm3d)):
            raise TypeError(
                f"{cls.__name__}.from_float(): expected a float BatchNorm "
                f"module, got {type(mod).__name__}")
        output_observer = getattr(mod, "activation_post_process", None)
        if output_observer is None:
            raise ValueError(
                f"{cls.__name__}.from_float(): the float module must carry a "
                "calibrated activation_post_process")
        scale, zero_point = output_observer.calculate_qparams()
        qbn = cls(mod.num_features, mod.eps, mod.momentum)
        qbn.weight = mod.weight
        qbn.bias = mod.bias
        qbn.running_mean = mod.running_mean
        qbn.running_var = mod.running_var
        qbn.scale = tensorplay.tensor(float(scale))
        qbn.zero_point = tensorplay.tensor(int(zero_point), dtype=tensorplay.long)
        return qbn

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        qbn = cls(bn.num_features, bn.eps, bn.momentum)
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = tensorplay.tensor(float(output_scale))
        qbn.zero_point = tensorplay.tensor(int(output_zero_point),
                                           dtype=tensorplay.long)
        return qbn


class BatchNorm2d(_BatchNorm):
    """Quantized 2d batch norm over a quantized input."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None,
                 dtype=None):
        super().__init__(num_features, eps, momentum, device=device,
                         dtype=dtype)

    def _get_name(self):
        return "QuantizedBatchNorm2d"

    def forward(self, input):
        return self._forward_quantized(input, 4)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return cls._from_float(mod)


class BatchNorm3d(_BatchNorm):
    """Quantized 3d batch norm over a quantized input."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None,
                 dtype=None):
        super().__init__(num_features, eps, momentum, device=device,
                         dtype=dtype)

    def _get_name(self):
        return "QuantizedBatchNorm3d"

    def forward(self, input):
        return self._forward_quantized(input, 5)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return cls._from_float(mod)
