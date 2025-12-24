import math
from typing import Optional

from ..parameter import Parameter
from .. import functional as F
from .. import init
from .module import Module
from .utils import _single, _pair, _triple
import tensorplay as tp


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DepthwiseConv2d",
]


class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[tp.Tensor], 'weight': tp.Tensor}

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = None
        
        if isinstance(self.padding[0], str):
            if self.transposed:
                raise ValueError("String padding not supported for Transposed Conv")
            
            if self.padding[0] == 'valid':
                self.padding = tuple(0 for _ in self.padding)
            elif self.padding[0] == 'same':
                if any(s != 1 for s in self.stride):
                    raise ValueError("padding='same' is not supported for strided convolutions")
                
                total_padding = []
                is_asymmetric = False
                for d in range(len(self.kernel_size)):
                    k = self.kernel_size[d]
                    dil = self.dilation[d]
                    total_pad = dil * (k - 1)
                    left_pad = total_pad // 2
                    right_pad = total_pad - left_pad
                    total_padding.append((left_pad, right_pad))
                    if left_pad != right_pad:
                        is_asymmetric = True
                
                if is_asymmetric:
                    if len(self.kernel_size) == 2:
                        # Optimized path for Conv2d: pass asymmetric padding directly to kernel
                        # total_padding is [(top, bottom), (left, right)]
                        # We want (top, bottom, left, right)
                        flat_padding = []
                        for pads in total_padding:
                            flat_padding.extend(pads)
                        self.padding = tuple(flat_padding)
                    else:
                        # F.pad expects (last_dim_left, last_dim_right, 2nd_last_left, ...)
                        pad_arg = []
                        for pads in reversed(total_padding):
                            pad_arg.extend(pads)
                        self._reversed_padding_repeated_twice = tuple(pad_arg)
                        self.padding = tuple(0 for _ in self.padding)
                else:
                    self.padding = tuple(p[0] for p in total_padding)
            else:
                raise ValueError("Invalid padding string: {}. Only 'valid' and 'same' are supported.".format(self.padding[0]))

        if transposed:
            self.weight = Parameter(tp.empty((in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(tp.empty((out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
            
        if bias:
            self.bias = Parameter(tp.empty((out_channels,), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def _output_padding(self, input, output_size, stride, padding, kernel_size, dilation):
        if output_size is None:
            return self.output_padding

        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        min_sizes = []
        max_sizes = []
        for d in range(k):
            dim_size = ((input.size(d + 2) - 1) * stride[d] -
                        2 * padding[d] +
                        dilation[d] * (kernel_size[d] - 1) + 1)
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)

        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError(
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})".format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        res = []
        for d in range(k):
            res.append(output_size[d] - min_sizes[d])

        return tuple(res)

class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)
        
    def forward(self, input):
        if self._reversed_padding_repeated_twice is not None:
            input = F.pad(input, self._reversed_padding_repeated_twice)
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        
    def forward(self, input):
        if self._reversed_padding_repeated_twice is not None:
            input = F.pad(input, self._reversed_padding_repeated_twice)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, **factory_kwargs)
        
    def forward(self, input):
        if self._reversed_padding_repeated_twice is not None:
            input = F.pad(input, self._reversed_padding_repeated_twice)
        return F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ConvTranspose2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)
        
    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')
        
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

class DepthwiseConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        if out_channels % in_channels != 0:
            raise ValueError("out_channels must be divisible by in_channels for DepthwiseConv2d")
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups=in_channels, bias=bias, padding_mode=padding_mode, **factory_kwargs)

class ConvTranspose3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)
        
    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        return F.conv_transpose3d(input, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
