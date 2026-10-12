"""Dynamic integer convolutions with floating inputs and outputs."""
import tensorplay as tp
from tensorplay import _C, nn
from ... import conv as static


class _DynamicConvolution:
    def __init__(self, *args, device=None, dtype=None, reduce_range=True, **kwargs):
        if dtype not in (None, tp.qint8, tp.float32):
            raise ValueError('Dynamic convolution requires signed byte weights')
        super().__init__(*args, **kwargs)
        self.reduce_range = bool(reduce_range)
        if device is not None:
            self.to(device)

    def forward(self, input, reduce_range=None):
        if input.dtype != tp.float32 or input.dim() != self._RANK + 2:
            raise ValueError('Dynamic convolution requires a floating batch with the declared spatial rank')
        padding = self.padding
        explicit = getattr(self, '_explicit_padding', None)
        if self.padding_mode != 'zeros' and explicit is None:
            explicit = tuple(value for p in reversed(padding) for value in (p,p))
        if explicit is not None:
            input = nn.functional.pad(input, explicit,
                mode='constant' if self.padding_mode == 'zeros' else self.padding_mode)
            padding = (0,) * self._RANK
        reduced = self.reduce_range if reduce_range is None else bool(reduce_range)
        quantized = (_C.quantize_per_tensor_dynamic(input,tp.quint8,reduced) if input.numel()
                     else _C.quantize_per_tensor(input,.1,0,tp.quint8))
        scale,zp = quantized.q_scale(),quantized.q_zero_point()
        if isinstance(self, static._ConvTransposeNd):
            result = _C.quantized_convolution(quantized,self.weight,self.bias,scale,zp,
                self.stride,padding,self.dilation,self.groups,True,self.output_padding)
        else:
            result = self._packed_params.run(quantized,scale,zp,self.stride,padding,
                                            self.dilation,self.groups)
        return result.dequantize()


class Conv1d(_DynamicConvolution, static.Conv1d):
    pass


class Conv2d(_DynamicConvolution, static.Conv2d):
    pass


class Conv3d(_DynamicConvolution, static.Conv3d):
    pass


class ConvTranspose1d(_DynamicConvolution, static.ConvTranspose1d):
    pass


class ConvTranspose2d(_DynamicConvolution, static.ConvTranspose2d):
    pass


class ConvTranspose3d(_DynamicConvolution, static.ConvTranspose3d):
    pass


__all__ = ['Conv1d','Conv2d','Conv3d','ConvTranspose1d','ConvTranspose2d','ConvTranspose3d']
