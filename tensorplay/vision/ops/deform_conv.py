"""Deformable convolution with learnable sampling offsets (DCNv2 style)."""
from typing import Optional, Tuple, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once, _make_ntuple


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Convolve ``input`` with ``weight`` where each kernel tap samples the
    input at a position shifted by the per-position values in ``offset``.

    ``offset`` has shape ``(N, 2 * offset_groups * kh * kw, out_h, out_w)``;
    ``mask``, when given, has shape ``(N, offset_groups * kh * kw, out_h,
    out_w)`` and scales each tap. Weight groups and offset groups are derived
    from the channel counts (``groups = in_channels // weight.shape[1]``,
    ``offset_groups = offset.shape[1] // (2 * kh * kw)``).
    """
    _log_api_usage_once(deform_conv2d)
    out_channels = weight.shape[0]
    groups = input.shape[1] // weight.shape[1] if weight.shape[1] > 0 else 0
    if input.shape[1] % weight.shape[1] != 0:
        raise ValueError(
            f"Expected input channels to be a multiple of weight channels, "
            f"got input channels {input.shape[1]} and weight channels {weight.shape[1]}"
        )
    if out_channels % groups != 0:
        raise ValueError(
            f"Expected output channels to be a multiple of groups, got {out_channels} and {groups}"
        )
    kh, kw = weight.shape[-2:]
    if offset.shape[1] % (2 * kh * kw) != 0 or offset.shape[1] == 0:
        raise ValueError(
            f"Expected offset channels to be a multiple of 2 * kh * kw, got {offset.shape[1]}"
        )
    use_mask = mask is not None
    if use_mask and mask.shape[1] % (kh * kw) != 0:
        raise ValueError(
            f"Expected mask channels to be a multiple of kh * kw, got {mask.shape[1]}"
        )
    if mask is None:
        mask = tensorplay.zeros((input.shape[0], 1), device=str(input.device))
    if bias is None:
        bias = tensorplay.zeros((out_channels,), device=str(input.device))
    stride = _make_ntuple(stride, 2)
    padding = _make_ntuple(padding, 2)
    dilation = _make_ntuple(dilation, 2)
    offset_groups = offset.shape[1] // (2 * kh * kw)
    return tensorplay.functional.deform_conv2d(
        input, weight, offset, mask, bias, stride, padding, dilation,
        groups, offset_groups, use_mask,
    )


class DeformConv2d(tensorplay.nn.Module):
    """Deformable 2-D convolution module.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: spatial size of the kernel
        stride, padding, dilation: convolution geometry, int or pair
        groups: number of blocked connections from input to output channels
        bias: add a learnable bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        kernel_size = _make_ntuple(kernel_size, 2)
        stride = _make_ntuple(stride, 2)
        padding = _make_ntuple(padding, 2)
        dilation = _make_ntuple(dilation, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = tensorplay.nn.Parameter(
            tensorplay.empty(
                (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
            )
        )
        if bias:
            self.bias = tensorplay.nn.Parameter(tensorplay.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        tensorplay.nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            tensorplay.nn.init.constant_(self.bias, 0)

    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return deform_conv2d(
            input, offset, self.weight, self.bias, self.stride,
            self.padding, self.dilation, mask,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}"
        s += f", dilation={self.dilation}"
        s += f", groups={self.groups}"
        s += f", bias={self.bias is not None}"
        return s
