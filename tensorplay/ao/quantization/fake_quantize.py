"""FakeQuantize: simulated quantization with a straight-through estimator.

The forward pass maps values through the real affine Int8 grid using the
native fake-quantization kernels.  The backward pass passes gradients through
where the input lies inside the representable range and blocks them
"""

import tensorplay
from tensorplay._C import (
    fake_quantize_per_channel_affine as _fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine as _fake_quantize_per_tensor_affine,
)
from tensorplay.autograd.function import Function
from tensorplay import nn

from .observer import QUANT_MAX, QUANT_MIN

__all__ = ["FakeQuantize", "fake_quantize_per_tensor"]


class _FakeQuantizeSTE(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        y = _fake_quantize_per_tensor_affine(
            self=x, scale=scale, zero_point=zero_point,
            quant_min=quant_min, quant_max=quant_max)
        # Real-domain bounds of the representable grid; gradient flows only
        # for inputs inside them (outside, quantization is saturated and a
        # straight-through would invent slope that the true function lacks).
        ctx.lo = (quant_min - zero_point) * scale
        ctx.hi = (quant_max - zero_point) * scale
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_x, *_rest):
        # The engine delivers one gradient slot per forward argument; only
        # the first (w.r.t. ``x``) is differentiable here.
        (x,) = ctx.saved_tensors
        in_range = x.clamp(min=ctx.lo, max=ctx.hi) == x
        grad_input = tensorplay.where(
            in_range, grad_x, tensorplay.zeros_like(grad_x))
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

    def __init__(self, observer=None, scale=None, zero_point=None,
                 disable_observer=False):
        super().__init__()
        if observer is None:
            from .observer import MinMaxObserver
            observer = MinMaxObserver()
        self.observer = observer
        self.scale = scale
        self.zero_point = zero_point
        self.frozen = scale is not None
        # When True, calibration is suspended: forward keeps fake-quantizing
        # FakeQuantize.disable_observer).
        self.disable_observer = bool(disable_observer)

    def record(self, x):
        if not self.frozen and not self.disable_observer:
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

class PerChannelFakeQuantize(nn.Module):
    """Per-channel fake quantization with range-masked STE.

    ``ch_axis`` selects the quantized dimension; scale/zero_point may be
    given explicitly (tensors of length n) or derived from a
    PerChannelMinMaxObserver over incoming batches.
    """

    def __init__(self, ch_axis=0, observer=None, scales=None, zero_points=None,
                 disable_observer=False):
        super().__init__()
        if observer is None:
            from .observer import PerChannelMinMaxObserver
            observer = PerChannelMinMaxObserver(ch_axis=ch_axis)
        self.observer = observer
        self.ch_axis = ch_axis
        self.scales = scales
        self.zero_points = zero_points
        self.frozen = scales is not None
        self.disable_observer = bool(disable_observer)

    def record(self, x):
        if not self.frozen and not self.disable_observer:
            self.observer.record(x)

    def calculate_qparams(self):
        if self.scales is not None:
            return self.scales, self.zero_points
        return self.observer.calculate_qparams()

    def forward(self, x):
        self.record(x)
        scales, zero_points = self.calculate_qparams()
        return fake_quantize_per_channel(x, scales.float(),
                                         zero_points.long(),
                                         axis=self.ch_axis)


def fake_quantize_per_channel(x, scales, zero_points, axis=0,
                              quant_min=QUANT_MIN, quant_max=QUANT_MAX):
    """Applies per-channel fake quantization with fixed affine parameters.

    Gradient passes through where ``x`` lies inside its channel's
    representable real range [qmin-zp, qmax-zp]*scale, else zero.
    """
    axis = axis % x.dim()
    shape = [1] * x.dim()
    shape[axis] = x.size(axis)
    scales1 = scales.to(tensorplay.float32)
    zps1 = (zero_points.to(tensorplay.float32)
            if zero_points.dtype.is_floating_point
            else zero_points.to(tensorplay.int64))

    y = _fake_quantize_per_channel_affine(
        self=x, scale=scales1, zero_point=zps1, axis=axis,
        quant_min=quant_min, quant_max=quant_max)
    # Broadcast per-channel real-domain bounds for the STE mask.
    lo_b = ((quant_min - zps1.to(scales1.dtype)) * scales1).reshape(shape) \
        .expand(x.shape).contiguous()
    hi_b = ((quant_max - zps1.to(scales1.dtype)) * scales1).reshape(shape) \
        .expand(x.shape).contiguous()

    class _STE(Function):
        @staticmethod
        def forward(ctx, xin, lo, hi):
            ctx.x = xin
            ctx.lo = lo
            ctx.hi = hi
            return y

        @staticmethod
        def backward(ctx, grad_x, *_rest):
            # In-range iff (x-lo)*(x-hi) <= 0; clamp() only takes scalars.
            signed = (ctx.x - ctx.lo) * (ctx.x - ctx.hi)
            in_range = signed <= tensorplay.zeros_like(signed)
            return (tensorplay.where(in_range, grad_x,
                                     tensorplay.zeros_like(grad_x)),
                    None, None)

    return _STE.apply(x, lo_b, hi_b)


__all__.extend(["PerChannelFakeQuantize", "fake_quantize_per_channel"])
