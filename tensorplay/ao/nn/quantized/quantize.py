"""Quantize/DeQuantize boundary modules.

``Quantize`` converts a float tensor to a native quantized tensor carrying
declared affine parameters; ``DeQuantize`` converts back through the
tensor's own quantizer.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay._C import quantize_per_tensor as _quantize_per_tensor

__all__ = ["Quantize", "DeQuantize"]


class Quantize(nn.Module):
    """Quantizes an incoming float tensor.

    Args:
        scale: scale of the output quantized tensor
        zero_point: zero point of the output quantized tensor
        dtype: quantized dtype of the output tensor (QInt8, QUInt8 or QInt32)
    """

    def __init__(self, scale, zero_point, dtype):
        super().__init__()
        self.register_buffer("scale", tensorplay.tensor([float(scale)]))
        self.register_buffer("zero_point",
                             tensorplay.tensor([int(zero_point)],
                                               dtype=tensorplay.long))
        self.dtype = dtype

    def forward(self, X):
        return _quantize_per_tensor(
            X, float(self.scale), int(self.zero_point), self.dtype)

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        observer = getattr(mod, "activation_post_process", None)
        if observer is None:
            raise ValueError(
                "Quantize.from_float(): the float module must carry a "
                "calibrated activation_post_process")
        scale, zero_point = observer.calculate_qparams()
        return Quantize(float(scale), int(zero_point), observer.dtype)

    def extra_repr(self):
        return f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}"


class DeQuantize(nn.Module):
    """Dequantizes an incoming tensor through its own affine parameters."""

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return DeQuantize()
