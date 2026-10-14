"""Integer convolution modules with reusable packed weights."""

import tensorplay as tp
from tensorplay import nn, _C
from .packed import PackedWeight, QuantizedState, quantize_module_weight, conversion_source

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


def _pair(value, rank):
    result = tuple(value) if isinstance(value, (tuple, list)) else (value,) * rank
    if len(result) != rank:
        raise ValueError(f'Expected {rank} convolution parameters')
    return tuple(int(v) for v in result)


class _ConvNd(QuantizedState, nn.Module):
    _RANK = None
    _FUSED_RELU = False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_scale=1.,
                 weight_zero_point=0, input_scale=None, input_zero_point=None,
                 out_scale=1., out_zero_point=0, padding_mode='zeros'):
        super().__init__()
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError('Channels must be divisible by positive groups')
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size = _pair(kernel_size, self._RANK)
        self.stride, self.padding = _pair(stride, self._RANK), _pair(padding, self._RANK)
        self.dilation, self.groups = _pair(dilation, self._RANK), int(groups)
        if padding_mode not in ('zeros', 'reflect') or (self._RANK == 3 and padding_mode == 'reflect'):
            raise ValueError('Unsupported integer convolution padding mode')
        self.padding_mode = padding_mode
        self._explicit_padding = None
        self.input_scale, self.input_zero_point = input_scale, input_zero_point
        self.out_scale, self.out_zero_point = float(out_scale), int(out_zero_point)
        shape = (out_channels, in_channels // groups) + self.kernel_size
        weight = _C.quantize_per_tensor(tp.zeros(shape), float(weight_scale), int(weight_zero_point), tp.qint8)
        self._packed_params = PackedWeight(weight, tp.zeros(out_channels) if bias else None)

    @property
    def weight(self):
        return self._packed_params.weight

    @weight.setter
    def weight(self, value):
        self.set_weight_bias(value, self.bias)

    @property
    def bias(self):
        return self._packed_params.bias

    @bias.setter
    def bias(self, value):
        self.set_weight_bias(self.weight, value)

    @property
    def weight_scale(self):
        return self.weight.q_scale()

    @property
    def weight_zero_point(self):
        return self.weight.q_zero_point()

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

    def set_weight_bias(self, weight, bias):
        expected = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        if tuple(weight.shape) != expected:
            raise ValueError('Convolution weight shape does not match its declared dimensions')
        self._packed_params.set_weight_bias(weight, bias)

    def forward(self, x):
        if x.dtype == tp.int8:
            if self.input_scale is None:
                raise TypeError('Raw codes require explicit input quantization parameters')
            x = _C._make_per_tensor_quantized_tensor(x, float(self.input_scale), int(self.input_zero_point))
        padding = self.padding
        explicit = self._explicit_padding
        if self.padding_mode != 'zeros' and explicit is None:
            explicit = tuple(value for p in reversed(padding) for value in (p, p))
        if explicit is not None:
            mode = 'constant' if self.padding_mode == 'zeros' else self.padding_mode
            codes = nn.functional.pad(x.int_repr(), explicit, mode=mode,
                                      value=x.q_zero_point() if mode == 'constant' else None)
            x = _C._make_per_tensor_quantized_tensor(codes, x.q_scale(), x.q_zero_point())
            padding = (0,) * self._RANK
        return self._packed_params.run(x, self.out_scale, self.out_zero_point,
                                       self.stride, padding, self.dilation,
                                       self.groups, self._FUSED_RELU)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        source = conversion_source(mod)
        if not isinstance(source, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            raise TypeError('Convolution conversion requires a floating convolution module')
        padding = source.padding
        explicit = source._reversed_padding_repeated_twice
        if len(padding) == 2 * cls._RANK:
            explicit = tuple(value for d in range(cls._RANK - 1, -1, -1)
                             for value in padding[2 * d:2 * d + 2])
            padding = (0,) * cls._RANK
        output = getattr(mod, 'activation_post_process', None)
        input_observer = getattr(mod, 'input_activation_post_process', None)
        out_scale, out_zp = output.calculate_qparams() if output is not None else (1., 0)
        input_scale, input_zp = input_observer.calculate_qparams() if input_observer is not None else (None, None)
        result = cls(source.in_channels, source.out_channels, source.kernel_size,
                     source.stride, padding, source.dilation, source.groups,
                     bias=source.bias is not None, input_scale=None if input_scale is None else float(input_scale),
                     input_zero_point=None if input_zp is None else int(input_zp),
                     out_scale=out_scale, out_zero_point=out_zp, padding_mode=source.padding_mode)
        result._explicit_padding = explicit
        owner = mod if hasattr(mod, 'weight_fake_quant') else source
        weight = quantize_module_weight(owner, use_precomputed_fake_quant, weight=source.weight)
        result.set_weight_bias(weight, source.bias.detach() if source.bias is not None else None)
        return result

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        """Build a native quantized convolution from a reference module."""
        result = cls(
            ref_qconv.in_channels,
            ref_qconv.out_channels,
            ref_qconv.kernel_size,
            ref_qconv.stride,
            ref_qconv.padding,
            ref_qconv.dilation,
            ref_qconv.groups,
            ref_qconv.bias is not None,
            padding_mode=ref_qconv.padding_mode,
        )
        result.set_weight_bias(ref_qconv.get_quantized_weight(), ref_qconv.bias)
        result.out_scale = float(output_scale)
        result.out_zero_point = int(output_zero_point)
        return result

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        """Build a quantized convolution directly from a float module."""
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        if weight_post_process.dtype != tp.qint8:
            raise AssertionError(
                f"Weight observer must have a dtype of qint8, got {weight_post_process.dtype}"
            )
        from .packed import quantize_module_weight
        qweight = quantize_module_weight(mod, use_precomputed_fake_quant=False)
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            padding_mode=mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if activation_post_process is None or activation_post_process.dtype == tp.float32:
            return qconv
        act_scale, act_zp = activation_post_process.calculate_qparams()
        qconv.out_scale = float(act_scale)
        qconv.out_zero_point = int(act_zp)
        return qconv

    def extra_repr(self):
        return f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}, scale={self.out_scale}, zero_point={self.out_zero_point}'


class Conv1d(_ConvNd):
    _RANK = 1


class Conv2d(_ConvNd):
    _RANK = 2


class Conv3d(_ConvNd):
    _RANK = 3


class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 out_scale=1., out_zero_point=0):
        nn.Module.__init__(self)
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError('Channels must be divisible by positive groups')
        if padding_mode != 'zeros':
            raise ValueError('Transposed convolution supports zero padding')
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size = _pair(kernel_size, self._RANK)
        self.stride, self.padding = _pair(stride, self._RANK), _pair(padding, self._RANK)
        self.dilation, self.output_padding = _pair(dilation, self._RANK), _pair(output_padding, self._RANK)
        self.groups, self.padding_mode = int(groups), padding_mode
        self.input_scale = self.input_zero_point = None
        self.out_scale, self.out_zero_point = float(out_scale), int(out_zero_point)
        self.register_buffer('_weight', _C.quantize_per_tensor(
            tp.zeros((in_channels, out_channels // groups) + self.kernel_size), 1., 0, tp.qint8))
        self.register_buffer('_bias', tp.zeros(out_channels) if bias else None)

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias

    def set_weight_bias(self, weight, bias):
        expected = (self.in_channels, self.out_channels // self.groups) + self.kernel_size
        if tuple(weight.shape) != expected or weight.dtype != tp.qint8 or weight.qscheme() != tp.per_tensor_affine:
            raise ValueError('Transposed weights require the declared shape and a signed tensor grid')
        if bias is not None and (bias.dim() != 1 or bias.numel() != self.out_channels):
            raise ValueError('Bias must contain one value per output channel')
        self._weight = weight.contiguous()
        self._bias = None if bias is None else bias.detach().float().to(weight.device).contiguous()

    def forward(self, x):
        return _C.quantized_convolution(x, self.weight, self.bias, self.out_scale, self.out_zero_point,
            self.stride, self.padding, self.dilation, self.groups, True, self.output_padding)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        observer = getattr(mod, 'activation_post_process', None)
        scale, zp = observer.calculate_qparams() if observer is not None else (1., 0)
        result = cls(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride,
                     mod.padding, mod.output_padding, mod.groups, mod.bias is not None,
                     mod.dilation, mod.padding_mode, float(scale), int(zp))
        result.set_weight_bias(quantize_module_weight(mod, use_precomputed_fake_quant), mod.bias)
        return result


class ConvTranspose1d(_ConvTransposeNd):
    _RANK = 1


class ConvTranspose2d(_ConvTransposeNd):
    _RANK = 2


class ConvTranspose3d(_ConvTransposeNd):
    _RANK = 3


__all__ += ['ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']
