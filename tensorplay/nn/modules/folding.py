from .module import Module
from .. import functional as F
from .utils import _pair


__all__ = ['Fold', 'Unfold']


class Fold(Module):
    r"""Combines an array of sliding local blocks into a large containing
    tensor (torch ``torch.nn.Fold``).

    Examples::

        >>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
        >>> input = tp.randn(1, 3 * 2 * 2, 12)
        >>> output = fold(input)
    """

    def __init__(self, output_size, kernel_size, dilation=1,
                 padding=0, stride=1):
        super().__init__()
        self.output_size = _pair(output_size)
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)

    def forward(self, input):
        return F.fold(input, self.output_size, self.kernel_size,
                      self.dilation, self.padding, self.stride)

    def extra_repr(self):
        return ('output_size={output_size}, kernel_size={kernel_size}'
                ', dilation={dilation}, padding={padding}, stride={stride}'
                ).format(**self.__dict__)


class Unfold(Module):
    r"""Extracts sliding local blocks from a batched input tensor
    (torch ``torch.nn.Unfold``).

    Examples::

        >>> unfold = nn.Unfold(kernel_size=(2, 3))
        >>> input = tp.randn(2, 5, 3, 4)
        >>> output = unfold(input)  # (2, 5 * 2 * 3, 2 * 2)
    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)

    def forward(self, input):
        return F.unfold(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)

    def extra_repr(self):
        return ('kernel_size={kernel_size}, dilation={dilation}'
                ', padding={padding}, stride={stride}'
                ).format(**self.__dict__)
