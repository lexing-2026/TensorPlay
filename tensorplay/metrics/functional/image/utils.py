"""Shared helpers for image metrics."""

from collections.abc import Sequence
from typing import Any, Union

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import functional as F  # noqa: N812


def _gaussian(kernel_size: int, sigma: float, dtype: Any, device: Any) -> Tensor:
    """Compute a 1D gaussian kernel.

    Args:
        kernel_size: size of the gaussian kernel
        sigma: standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    """
    dist = tp.arange((1 - kernel_size) / 2, (1 + kernel_size) / 2, 1, dtype=dtype, device=device)
    gauss = tp.exp(-tp.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)


def _gaussian_kernel_2d(
    channel: int,
    kernel_size: Sequence[int],
    sigma: Sequence[float],
    dtype: Any,
    device: Any,
) -> Tensor:
    """Compute a 2D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w)
        sigma: standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    """
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = tp.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])


def _uniform_weight_bias_conv2d(inputs: Tensor, window_size: int) -> tuple[Tensor, Tensor]:
    """Construct uniform weight and bias for a 2d convolution.

    Args:
        inputs: input image
        window_size: size of convolutional kernel

    Return:
        The weight and bias for 2d convolution

    """
    kernel_weight = tp.ones(1, 1, window_size, window_size, dtype=inputs.dtype, device=inputs.device)
    kernel_weight /= window_size**2
    kernel_bias = tp.zeros(1, dtype=inputs.dtype, device=inputs.device)
    return kernel_weight, kernel_bias


def _single_dimension_pad(inputs: Tensor, dim: int, pad: int, outer_pad: int = 0) -> Tensor:
    """Apply single-dimension reflection padding.

    Args:
        inputs: input image
        dim: a dimension the image should be padded over
        pad: number of pads
        outer_pad: number of outer pads

    Return:
        Image padded over a single dimension

    """
    _max = inputs.shape[dim]
    x = tp.index_select(inputs, dim, tp.arange(pad - 1, -1, -1).to(inputs.device))
    y = tp.index_select(inputs, dim, tp.arange(_max - 1, _max - pad - outer_pad, -1).to(inputs.device))
    return tp.cat((x, inputs, y), dim)


def _reflection_pad_2d(inputs: Tensor, pad: int, outer_pad: int = 0) -> Tensor:
    """Apply reflection padding to the input image.

    Args:
        inputs: input image
        pad: number of pads
        outer_pad: number of outer pads

    Return:
        Padded image

    """
    for dim in [2, 3]:
        inputs = _single_dimension_pad(inputs, dim, pad, outer_pad)
    return inputs


def _uniform_filter(inputs: Tensor, window_size: int) -> Tensor:
    """Apply a uniform filter with a window of a given size over the input image.

    Args:
        inputs: input image
        window_size: sliding window used for rmse calculation

    Return:
        Image transformed with the uniform input

    """
    inputs = _reflection_pad_2d(inputs, window_size // 2, window_size % 2)
    kernel_weight, kernel_bias = _uniform_weight_bias_conv2d(inputs, window_size)
    # Iterate over channels
    return tp.cat(
        [
            F.conv2d(inputs[:, channel].unsqueeze(1), kernel_weight, kernel_bias, padding=0)
            for channel in range(inputs.shape[1])
        ],
        dim=1,
    )


def _gaussian_kernel_3d(
    channel: int, kernel_size: Sequence[int], sigma: Sequence[float], dtype: Any, device: Any
) -> Tensor:
    """Compute a 3D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w, d)
        sigma: standard deviation of the gaussian kernel
        dtype: data type of the output tensor
        device: device of the output tensor

    """
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    gaussian_kernel_z = _gaussian(kernel_size[2], sigma[2], dtype, device)
    kernel_xy = tp.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)
    kernel = tp.mul(
        kernel_xy.unsqueeze(-1).repeat(1, 1, kernel_size[2]),
        gaussian_kernel_z.expand(kernel_size[0], kernel_size[1], kernel_size[2]),
    )
    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1], kernel_size[2])


def _reflection_pad_3d(inputs: Tensor, pad_h: int, pad_w: int, pad_d: int) -> Tensor:
    """Apply reflective padding to a 3d input.

    Args:
        inputs: tensor to pad, should be a 3D tensor of shape ``[N, C, H, W, D]``
        pad_w: amount of padding in the height dimension
        pad_h: amount of padding in the width dimension
        pad_d: amount of padding in the depth dimension

    Returns:
        padded input tensor

    """
    return F.pad(inputs, (pad_h, pad_h, pad_w, pad_w, pad_d, pad_d), mode="reflect")