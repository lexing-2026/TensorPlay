"""Linear inference with reusable integer weight storage and fused output transforms."""

import tensorplay as tp
from tensorplay import nn, _C
from .packed import PackedWeight, QuantizedState, quantize_module_weight, conversion_source

__all__ = ['QuantizedLinear']


class QuantizedLinear(QuantizedState, nn.Module):
    _FUSED_RELU = False

    def __init__(self, in_features, out_features, input_scale=1., input_zero_point=0,
                 qweight=None, weight_scales=None, weight_zero_points=None,
                 bias=None, out_scale=1., out_zero_point=0):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.input_scale, self.input_zero_point = float(input_scale), int(input_zero_point)
        self.out_scale, self.out_zero_point = float(out_scale), int(out_zero_point)
        if qweight is None:
            qweight = _C.quantize_per_tensor(tp.zeros(out_features, in_features), 1., 0, tp.qint8)
        elif qweight.dtype == tp.int8:
            qweight = _C._make_per_channel_quantized_tensor(qweight, weight_scales.float(), weight_zero_points.long(), 0)
        self._packed_params = PackedWeight(qweight, bias)

    @property
    def qweight(self):
        return self._packed_params.weight

    @property
    def bias(self):
        return self._packed_params.bias

    @property
    def weight_scales(self):
        return self._packed_params.scales

    @property
    def weight_zero_points(self):
        return self._packed_params.zero_points

    @property
    def scale(self):
        return self.out_scale

    @scale.setter
    def scale(self, value):
        self.out_scale = float(value)

    @property
    def zero_point(self):
        return self.out_zero_point

    @zero_point.setter
    def zero_point(self, value):
        self.out_zero_point = int(value)

    def weight(self):
        return self.qweight

    def set_weight_bias(self, weight, bias):
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError('Linear weight shape does not match its declared dimensions')
        self._packed_params.set_weight_bias(weight, bias)

    def forward(self, x):
        if x.dtype == tp.int8:
            x = _C._make_per_tensor_quantized_tensor(x, self.input_scale, self.input_zero_point)
        return self._packed_params.run(x, self.out_scale, self.out_zero_point, relu=self._FUSED_RELU)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, scale={self.out_scale}, zero_point={self.out_zero_point}'

    @classmethod
    def from_float(cls, float_module, input_scale=None, input_zero_point=None,
                   out_scale=None, out_zero_point=None, use_precomputed_fake_quant=False):
        source = conversion_source(float_module)
        if not isinstance(source, nn.Linear):
            raise TypeError('Linear conversion requires a floating linear module')
        if input_scale is None or input_zero_point is None:
            observer = getattr(float_module, 'input_activation_post_process', None)
            if observer is None:
                raise ValueError('Linear conversion requires input calibration or explicit parameters')
            input_scale, input_zero_point = observer.calculate_qparams()
        if out_scale is None or out_zero_point is None:
            observer = getattr(float_module, 'activation_post_process', None)
            if observer is None:
                raise ValueError('Linear conversion requires output calibration or explicit parameters')
            out_scale, out_zero_point = observer.calculate_qparams()
        owner = float_module if hasattr(float_module, 'weight_fake_quant') else source
        qweight = quantize_module_weight(owner, use_precomputed_fake_quant, weight=source.weight)
        return cls(source.in_features, source.out_features, input_scale, input_zero_point,
                   qweight, bias=source.bias.detach() if source.bias is not None else None,
                   out_scale=out_scale, out_zero_point=out_zero_point)
