"""Real quantized modules for inference after quantization.

Closed loop: activations are native QInt8 tensors carrying their affine
parameters, weights are stored as per-channel QInt8, and the fused native
kernel produces a QInt8 [M,N] output carrying the module's output affine
parameters.
"""

import tensorplay
from tensorplay import nn
from tensorplay._C import (
    quantize_per_channel as _quantize_per_channel,
    quantized_linear as _quantized_linear,
    _make_per_tensor_quantized_tensor as _make_per_tensor_quantized_tensor,
    _make_per_channel_quantized_tensor as _make_per_channel_quantized_tensor,
)

__all__ = ["QuantizedLinear"]


class QuantizedLinear(nn.Module):
    """Applies a linear transformation on a quantized input with QInt8 weights.

    out_q[m, n] = requantize( input_scale * weight_scales[n] *
                sum_k (x_q[m,k] - input_zero_point) *
                      (w_q[n,k] - weight_zero_points[n]) + bias[n] )
    under (out_scale, out_zero_point).
    """

    def __init__(self, in_features, out_features, input_scale,
                 input_zero_point, qweight, weight_scales, weight_zero_points,
                 bias=None, out_scale=1.0, out_zero_point=0):
        super().__init__()
        if qweight.dtype not in (tensorplay.qint8, tensorplay.int8):
            raise TypeError("QuantizedLinear expects QInt8 (or raw Int8) weights")
        if qweight.dtype == tensorplay.int8:
            # A raw code tensor carries no quantizer of its own; mount the
            # module's per-channel affine parameters so the fused kernel sees
            # the same per-channel QInt8 operand the from_float path builds.
            qweight = _make_per_channel_quantized_tensor(
                qweight, weight_scales.to(tensorplay.float32),
                weight_zero_points.to(tensorplay.int64), 0)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.input_scale = float(input_scale)
        self.input_zero_point = int(input_zero_point)
        self.out_scale = float(out_scale)
        self.out_zero_point = int(out_zero_point)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("weight_scales",
                             weight_scales.to(tensorplay.float32).contiguous())
        self.register_buffer("weight_zero_points",
                             weight_zero_points.to(tensorplay.int64).contiguous())
        if bias is not None:
            self.register_buffer(
                "bias", bias.to(tensorplay.float32).contiguous())
        else:
            self.bias = None

    def forward(self, x):
        if x.is_quantized():
            if x.dtype != tensorplay.qint8:
                raise TypeError(
                    "QuantizedLinear expects QInt8 activations, got "
                    f"{x.dtype}")
        elif x.dtype == tensorplay.int8:
            # Raw code tensor: wrap it with this module's activation qparams.
            x = _make_per_tensor_quantized_tensor(
                x, self.input_scale, self.input_zero_point)
        else:
            raise TypeError(
                "QuantizedLinear expects a quantized activation tensor; run "
                "it through QuantStub (or quantize_per_tensor) first")
        return _quantized_linear(
            x, self.qweight, input_scale=self.input_scale,
            input_zero_point=self.input_zero_point,
            weight_scales=self.weight_scales,
            weight_zero_points=self.weight_zero_points,
            bias=self.bias, out_scale=self.out_scale,
            out_zero_point=self.out_zero_point)

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"input_scale={self.input_scale}, "
                f"input_zero_point={self.input_zero_point}, "
                f"out_scale={self.out_scale}, "
                f"out_zero_point={self.out_zero_point}")

    @classmethod
    def from_float(cls, float_module, input_scale=None, input_zero_point=None,
                   out_scale=None, out_zero_point=None):
        """Quantizes a calibrated float Linear's weights per output channel.

        Activation ranges must come from calibration ahead of conversion
        (MinMax over the intended distributions) or be given explicitly: the
        input range through ``input_scale``/``input_zero_point`` or a
        prepared ``input_activation_post_process``, the output range through
        ``out_scale``/``out_zero_point`` or a prepared
        ``activation_post_process``.
        """
        if not isinstance(float_module, nn.Linear):
            raise TypeError("from_float(): expected a Linear module")
        from ...quantization.observer import ObserverBase
        if input_scale is None or input_zero_point is None:
            observer = getattr(float_module, "input_activation_post_process", None)
            if observer is None:
                raise ValueError(
                    "from_float(): needs explicit input scale/zero point, or a "
                    "prepared float module carrying input_activation_post_process")
            input_scale, input_zero_point = observer.calculate_qparams()
        if out_scale is None or out_zero_point is None:
            out_observer = getattr(float_module, "activation_post_process", None)
            if out_observer is None:
                raise ValueError(
                    "from_float(): needs explicit output scale/zero point, or "
                    "a prepared float module carrying activation_post_process")
            out_scale, out_zero_point = out_observer.calculate_qparams()
        input_scale = float(input_scale)
        input_zero_point = int(input_zero_point)
        weight = float_module.weight.detach()
        out_features, in_features = weight.shape
        min_vals, max_vals = tensorplay.aminmax(weight, dim=list(range(1, weight.dim())), keepdim=False)
        # Per-output-channel affine params from each row's observed range.
        scales = []
        zero_points = []
        for n in range(out_features):
            s, z = ObserverBase._calculate_qparams(float(min_vals[n]),
                                                   float(max_vals[n]))
            scales.append(s)
            zero_points.append(z)
        scales_t = tensorplay.as_tensor(scales, dtype=tensorplay.float32)
        zero_points_t = tensorplay.as_tensor(zero_points, dtype=tensorplay.int64)
        # Kernel operands must live on the weights' device.
        scales_t = scales_t.to(weight.device)
        zero_points_t = zero_points_t.to(weight.device)
        qweight = _quantize_per_channel(
            self=weight, scales=scales_t, zero_points=zero_points_t, axis=0,
            dtype=tensorplay.qint8)
        bias = None
        if float_module.bias is not None:
            bias = float_module.bias.detach().to(tensorplay.float32)
        return cls(in_features, out_features, input_scale, input_zero_point,
                   qweight, scales_t, zero_points_t, bias=bias,
                   out_scale=float(out_scale),
                   out_zero_point=int(out_zero_point))
