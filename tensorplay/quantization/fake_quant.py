"""FakeQuantize: simulated quantization with a straight-through estimator.

The forward pass maps values through the real affine Int8 grid
(dequantize(quantize(x))) using the native ``quantize_per_tensor`` /
``dequantize_per_tensor`` kernels.  The backward pass passes gradients
through where the input lies inside the representable range and blocks them
outside, matching torch's default FakeQuantize behavior.
"""

import tensorplay
from tensorplay._C import (
    dequantize_per_tensor as _dequantize_per_tensor,
    quantize_per_tensor as _quantize_per_tensor,
)
from tensorplay.autograd.function import Function
from tensorplay import nn

from .observer import QUANT_MAX, QUANT_MIN

__all__ = ["FakeQuantize", "fake_quantize_per_tensor"]


class _FakeQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        q = _quantize_per_tensor(self=x, scale=scale, zero_point=zero_point,
                                 quant_min=quant_min, quant_max=quant_max)
        y = _dequantize_per_tensor(self=q, scale=scale, zero_point=zero_point)
        # Real-domain bounds of the representable grid; gradient flows only
        # for inputs inside them (outside, quantization is saturated and a
        # straight-through would invent slope that the true function lacks).
        ctx.lo = (quant_min - zero_point) * scale
        ctx.hi = (quant_max - zero_point) * scale
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        in_range = x.clamp(min=ctx.lo, max=ctx.hi) == x
        grad_input = tensorplay.where(
            in_range, grad_output, tensorplay.zeros_like(grad_output))
        return (grad_input, None, None, None, None)


def fake_quantize_per_tensor(x, scale, zero_point,
                             quant_min=QUANT_MIN, quant_max=QUANT_MAX):
    """Applies fake quantization with fixed affine parameters."""
    return _FakeQuantizeSTE.apply(x, float(scale), int(zero_point),
                                  int(quant_min), int(quant_max))


class FakeQuantize(nn.Module):
    """Calibrating / simulating module.

    With no qparams set, the first forward pass derives scale/zero_point
    from its observer over incoming batches; call :meth:`freeze` to stop
    recalibrating.  With explicit scale/zero_point arguments it is stateless.
    """

    def __init__(self, observer=None, scale=None, zero_point=None):
        super().__init__()
        if observer is None:
            from .observer import MinMaxObserver
            observer = MinMaxObserver()
        self.observer = observer
        self.scale = scale
        self.zero_point = zero_point
        self.frozen = scale is not None

    def record(self, x):
        if not self.frozen:
            self.observer.record(x)

    def freeze(self):
        """Stops calibration and fixes the current derived qparams."""
        if self.scale is None:
            self.scale, self.zero_point = self.observer.calculate_qparams()
        self.frozen = True

    def calculate_qparams(self):
        if self.scale is not None:
            return self.scale, self.zero_point
        return self.observer.calculate_qparams()

    def forward(self, x):
        self.record(x)
        scale, zero_point = self.calculate_qparams()
        return fake_quantize_per_tensor(x, scale, zero_point)
