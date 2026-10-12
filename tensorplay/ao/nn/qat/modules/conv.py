"""Convolution modules with trainable floating weights on a simulated integer grid."""

from tensorplay import nn
from tensorplay.nn import functional as F
from .linear import _weight_fake_quant

__all__ = ['Conv1d', 'Conv2d', 'Conv3d']


class _ConvNd:
    _BASE = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 qconfig=None, device=None, dtype=None):
        self._BASE.__init__(self, in_channels, out_channels, kernel_size,
                            stride, padding, dilation, groups, bias, padding_mode,
                            device=device, dtype=dtype)
        self.qconfig, self.weight_fake_quant = _weight_fake_quant(qconfig, self.weight.device)

    def _conv_forward(self, input, weight, bias):
        return _conv_forward(self, input, weight, bias)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        source = getattr(mod, 'op', mod)
        if not isinstance(source, cls._BASE):
            raise TypeError(f'Expected a {cls._BASE.__name__} module')
        result = cls(source.in_channels, source.out_channels, source.kernel_size,
                     stride=source.stride, padding=source.padding, dilation=source.dilation,
                     groups=source.groups, bias=source.bias is not None,
                     padding_mode=source.padding_mode, qconfig=getattr(mod, 'qconfig', None),
                     device=source.weight.device, dtype=source.weight.dtype)
        result.weight, result.bias = source.weight, source.bias
        result._reversed_padding_repeated_twice = source._reversed_padding_repeated_twice
        result.train(mod.training)
        return result

    def to_float(self):
        result = self._BASE(self.in_channels, self.out_channels, self.kernel_size,
                            self.stride, self.padding, self.dilation, self.groups,
                            self.bias is not None, self.padding_mode,
                            device=self.weight.device, dtype=self.weight.dtype)
        result.weight = nn.Parameter(self.weight.detach())
        result.bias = nn.Parameter(self.bias.detach()) if self.bias is not None else None
        result._reversed_padding_repeated_twice = self._reversed_padding_repeated_twice
        result.train(self.training)
        return result


class Conv1d(_ConvNd, nn.Conv1d):
    _BASE = nn.Conv1d


class Conv2d(_ConvNd, nn.Conv2d):
    _BASE = nn.Conv2d


class Conv3d(_ConvNd, nn.Conv3d):
    _BASE = nn.Conv3d


def _conv_forward(module, input, weight, bias):
    if module.padding_mode != 'zeros':
        input = F.pad(input, list(module._reversed_padding_repeated_twice), mode=module.padding_mode)
    elif module._reversed_padding_repeated_twice is not None:
        input = F.pad(input, list(module._reversed_padding_repeated_twice))
    operation = (F.conv1d, F.conv2d, F.conv3d)[weight.dim() - 3]
    return operation(input, weight, bias, module.stride, module.padding, module.dilation, module.groups)
