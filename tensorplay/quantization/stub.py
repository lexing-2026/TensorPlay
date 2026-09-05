"""Quant/DeQuant stubs for float<->quantized conversion points in a model.

QuantStub calibrates on observed activations (training/QAT) and, once
frozen, converts incoming floats to native QInt8 tensors via the
quantize_per_tensor kernel.  DeQuantStub converts a quantized tensor back to
Float32 through its own quantizer (or explicit parameters for raw code
tensors).
"""

import tensorplay
from tensorplay._C import (
    _make_per_tensor_quantized_tensor as _make_per_tensor_quantized_tensor,
    quantize_per_tensor as _quantize_per_tensor,
)
from tensorplay import nn

from .fake_quant import FakeQuantize

__all__ = ["QuantStub", "DeQuantStub"]


class QuantStub(nn.Module):
    def __init__(self, qconfig=None):
        super().__init__()
        self.fake_quant = FakeQuantize() if qconfig is None else qconfig()

    def record(self, x):
        """Calibration entry point: feeds the batch to the inner FakeQuantize
        observer without fake-quantizing (as manual calibration loops do)."""
        self.fake_quant.record(x)
        return x

    def forward(self, x):
        # Calibration / QAT path: simulated quantization keeps the graph
        # float while nudging values toward the quantized grid.
        self.fake_quant.record(x)
        scale, zero_point = self.fake_quant.calculate_qparams()
        if not self.training and self.fake_quant.frozen:
            # Inference path: produce a native quantized tensor carrying
            # its affine parameters.
            return _quantize_per_tensor(self=x, scale=scale,
                                        zero_point=zero_point,
                                        dtype=tensorplay.qint8)
        return self.fake_quant(x)

    def freeze(self):
        self.fake_quant.freeze()


class DeQuantStub(nn.Module):
    def __init__(self, scale=None, zero_point=None):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        if x.is_quantized():
            # Native path: the tensor carries its own affine parameters.
            return x.dequantize()
        if x.dtype != tensorplay.int8:
            raise TypeError(
                "DeQuantStub expects a quantized (or raw Int8 code) tensor; "
                "convert float inputs through a QuantStub first")
        scale = 1.0 if self.scale is None else float(self.scale)
        zero_point = 0 if self.zero_point is None else int(self.zero_point)
        q = _make_per_tensor_quantized_tensor(x, scale, zero_point)
        return q.dequantize()
