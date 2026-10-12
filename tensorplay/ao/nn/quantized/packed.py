"""Canonical integer weights and reusable native execution storage."""

import tensorplay as tp
from tensorplay import nn, _C


def quantize_module_weight(module, use_precomputed_fake_quant=False, weight=None):
    from ...quantization.fake_quantize import FakeQuantizeBase
    from ...quantization.observer import default_per_channel_weight_observer
    weight = module.weight.detach() if weight is None else weight.detach()
    observer = getattr(module, 'weight_fake_quant', None)
    if observer is None:
        observer = getattr(module, 'weight_observer', None)
    if observer is None:
        qconfig = getattr(module, 'qconfig', None)
        observer = qconfig.weight() if qconfig is not None else default_per_channel_weight_observer()
    observer.to(weight.device)
    if not use_precomputed_fake_quant:
        observer(weight)
    if isinstance(observer, FakeQuantizeBase):
        scale, zp = observer.scale, observer.zero_point
        config = observer.activation_post_process
    else:
        scale, zp = observer.calculate_qparams()
        config = observer
    if config.dtype not in (tp.qint8, tp.int8):
        raise ValueError('Linear and convolution weights require a signed 8-bit grid')
    channel = config.qscheme in (tp.per_channel_affine, tp.per_channel_symmetric)
    if channel:
        if config.ch_axis != 0:
            raise ValueError('Weights must be quantized along the output channel axis')
        quantized = _C.quantize_per_channel(weight.float(), scale.float(), zp.long(), 0, tp.qint8)
    else:
        quantized = _C.quantize_per_tensor(weight.float(), float(scale), int(zp), tp.qint8)
    if config.quant_min > -128 or config.quant_max < 127:
        codes = quantized.int_repr().clamp(config.quant_min, config.quant_max)
        quantized = (_C._make_per_channel_quantized_tensor(codes, scale.float(), zp.long(), 0)
                     if channel else _C._make_per_tensor_quantized_tensor(codes, float(scale), int(zp)))
    return quantized


def conversion_source(module):
    source = getattr(module, 'op', module)
    if hasattr(source, 'bn') and hasattr(source, 'weight_fake_quant'):
        folded = source.to_float()
        source = getattr(folded, 'op', folded)
    return source


class PackedWeight(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.register_buffer('packed', None, persistent=False)
        self.register_buffer('offsets', None, persistent=False)
        self.register_buffer('scales', None, persistent=False)
        self.register_buffer('zero_points', None, persistent=False)
        self.set_weight_bias(weight, bias)

    def set_weight_bias(self, weight, bias):
        if weight.dtype != tp.qint8:
            raise TypeError('Packed weights require signed 8-bit quantized storage')
        if weight.qscheme() == tp.per_tensor_affine:
            scales = tp.full((weight.size(0),), weight.q_scale(), dtype=tp.float32, device=weight.device)
            zero_points = tp.full((weight.size(0),), weight.q_zero_point(), dtype=tp.int64, device=weight.device)
        elif weight.qscheme() == tp.per_channel_affine and weight.q_per_channel_axis() == 0:
            scales = weight.q_per_channel_scales().float().contiguous()
            zero_points = weight.q_per_channel_zero_points().long().contiguous()
        else:
            raise ValueError('Packed weights require tensor or output-channel affine quantization')
        if scales.numel() and (not bool(tp.isfinite(scales).all()) or float(scales.min()) <= 0):
            raise ValueError('Packed weight scales must be finite and positive')
        if zero_points.numel() and (int(zero_points.min()) < -128 or int(zero_points.max()) > 127):
            raise ValueError('Packed weight zero points exceed the signed code range')
        if bias is not None and (bias.dim() != 1 or bias.numel() != weight.size(0)):
            raise ValueError('Bias must contain one value per output channel')
        self.weight = weight.contiguous()
        self.bias = bias.to(device=weight.device, dtype=tp.float32).contiguous() if bias is not None else None
        self.scales, self.zero_points = scales, zero_points
        self.packed, self.offsets = _C._quantized_weight_prepack(self.weight)

    def _weight_bias(self):
        return self.weight, self.bias

    def __getstate__(self):
        state = super().__getstate__()
        state['_buffers'] = state['_buffers'].copy()
        for name in ('packed', 'offsets', 'scales', 'zero_points'):
            state['_buffers'][name] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_weight_bias(self.weight, self.bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.set_weight_bias(self.weight, self.bias)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'weight'
        if key in state_dict:
            weight = state_dict[key].to(self.weight.device)
            bias = state_dict.get(prefix + 'bias')
            self.set_weight_bias(weight, bias)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def run(self, input, out_scale=1., out_zero_point=0, stride=(), padding=(),
            dilation=(), groups=1, relu=False, output_float=False):
        return _C._quantized_packed_run(input, self.packed, self.offsets, self.scales,
                                       self.zero_points, self.bias, list(self.weight.shape),
                                       float(out_scale), int(out_zero_point), list(stride),
                                       list(padding), list(dilation), groups, relu, output_float)


class QuantizedState:
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name in ('input_scale', 'input_zero_point', 'out_scale', 'out_zero_point'):
            value = getattr(self, name, None)
            if value is not None:
                destination[prefix + name] = tp.tensor(value, dtype=tp.int64 if 'zero_point' in name else tp.float64)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ('input_scale', 'input_zero_point', 'out_scale', 'out_zero_point'):
            key = prefix + name
            if key in state_dict:
                value = state_dict.pop(key)
                setattr(self, name, int(value) if 'zero_point' in name else float(value))
            elif strict and getattr(self, name, None) is not None:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
