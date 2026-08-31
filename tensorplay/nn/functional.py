"""Functional interface."""

import math
import os
import warnings
from typing import Optional

import tensorplay
import tensorplay._C as _C
from tensorplay._C import _add_docstr, DType
from tensorplay import Tensor
from tensorplay.graph import capture_call as _capture_call

def threshold(
    input: Tensor,
    threshold: float,
    value: float,
    inplace: bool = False,
) -> Tensor:
    r"""Apply a threshold to each element of the input Tensor.

    See :class:`~tensorplay.nn.Threshold` for more details.
    """
    captured = _capture_call(
        globals()["threshold"], (input, threshold, value), {"inplace": inplace}
    )
    if captured is not None:
        return captured
    if inplace:
        result = _C.threshold_(input, threshold, value)
    else:
        result = _C.threshold(input, threshold, value)
    return result


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~tensorplay.nn.SiLU` for more details.
    """
    captured = _capture_call(silu, (input,), {"inplace": inplace})
    if captured is not None:
        return captured
    if inplace:
        return tensorplay._C.silu_(input)
    return tensorplay._C.silu(input)


gelu = _add_docstr(
    tensorplay._C.gelu,
    r"""
gelu(input, approximate = 'none') -> Tensor

When the approximate argument is 'none', it applies element-wise the function
:math:`\text{GELU}(x) = x * \Phi(x)`

where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

When the approximate argument is 'tanh', Gelu is estimated with

.. math::
    \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
""",
)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Weight: :math:`(H_\text{out}, H_\text{in})` where
          :math:`H_\text{out} = \text{out\_features}`.
        - Bias: :math:`(H_\text{out})`
        - Output: :math:`(*, H_\text{out})`

    See :class:`~tensorplay.nn.Linear` for more details.
    """
    captured = _capture_call(linear, (input, weight, bias), {})
    if captured is not None:
        return captured

    # _matmul_impl checks this again later, but the native flatten path does
    # not work on scalar inputs, so try to catch this here already
    input_dim = input.dim()
    weight_dim = weight.dim()
    if input_dim == 0 or weight_dim == 0:
        raise RuntimeError(
            "both arguments to linear need to be at least 1D, but they are "
            f"{input_dim}D and {weight_dim}D"
        )

    # Native dispatch: CPU runs linear_kernel (single seeded-GEMM addmm with
    # raw as_strided weight.t(), bias folded into the epilogue / seed);
    # other backends fall through to the recordable matmul/add composite.
    return tensorplay.linear(input, weight, bias)


def bilinear(input1, input2, weight, bias=None):
    if list(input1.shape)[:-1] != list(input2.shape)[:-1]:
        raise ValueError("input1 and input2 must have the same batch dimensions")
    
    out_features, in1_features, in2_features = weight.shape
    
    # w: (Out, H1, H2) -> (Out, H2, H1)
    # TensorPlay permute expects a sequence
    w = weight.permute([0, 2, 1])
    
    # w: (Out * H2, H1)
    w = w.reshape(-1, in1_features)
    
    # input1: (*, H1)
    # input1 @ w.T: (*, H1) @ (H1, Out * H2) -> (*, Out * H2)
    temp = input1.matmul(w.t())
    
    # temp: (*, Out * H2)
    # Reshape to (*, Out, H2)
    new_shape = list(input1.shape)[:-1] + [out_features, in2_features]
    temp = temp.view(new_shape)
    
    # input2: (*, H2)
    # unsqueeze to (*, H2, 1)
    input2_expanded = input2.unsqueeze(-1)
    
    # temp: (*, Out, H2)
    # result: (*, Out, 1)
    output = temp.matmul(input2_expanded)
    
    # squeeze
    output = output.squeeze(-1)
    
    if bias is not None:
        output = output + bias
        
    return output

def relu(input, inplace=False):
    captured = _capture_call(
        relu,
        (input,),
        {"inplace": True} if inplace else {},
    )
    if captured is not None:
        return captured
    if inplace:
        return _C.relu_(input)
    return _C.relu(input)

def softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    if dtype is None:
        dtype = tensorplay.undefined
    return input.softmax(dim, dtype)

def log_softmax(input, dim=None, dtype=None):
    captured = _capture_call(log_softmax, (input, dim, dtype), {})
    if captured is not None:
        return captured
    if dim is None:
        dim = -1
    if dtype is None:
        dtype = tensorplay.undefined
    return _C.log_softmax(input, dim, dtype)

def prelu(input, weight):
    captured = _capture_call(prelu, (input, weight), {})
    if captured is not None:
        return captured
    # PReLU(x) = max(0, x) + weight * min(0, x)
    #          = relu(x) - weight * relu(-x)
    
    if weight.numel() != 1:
        if input.dim() < 2:
             raise ValueError("Input must have at least 2 dimensions when num_parameters > 1")
        
        # Check if num_parameters matches channel dim (dim 1)
        if input.size(1) != weight.numel():
            raise ValueError(f"num_parameters {weight.numel()} does not match input channel size {input.size(1)}")
        
        # Reshape weight for broadcasting
        # We want (1, C, 1, ...)
        view_shape = [1] * input.dim()
        view_shape[1] = weight.numel()
        weight = weight.view(view_shape)

    return _C.prelu(input, weight)

def flatten(input, start_dim=0, end_dim=-1):
    return input.flatten(start_dim, end_dim)

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    captured = _capture_call(embedding, (input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse), {})
    if captured is not None:
        return captured
    if max_norm is not None:
        raise NotImplementedError('embedding: max_norm is not supported')
    if padding_idx is None:
        padding_idx = -1
    else:
        if padding_idx < -weight.size(0) or padding_idx >= weight.size(0):
            raise AssertionError('Padding_idx must be within num_embeddings')
        if padding_idx < 0:
            padding_idx += weight.size(0)
    return _C.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)

# Add more functionals as needed

def dropout(input, p=0.5, training=True, inplace=False):
    captured = _capture_call(dropout, (input, p, training, inplace), {})
    if captured is not None:
        return captured
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    if not training or p == 0:
        return input

    if inplace:
        # dropout_ mutates self and records the mask for backward).
        mask = (_C.rand(input.shape, device=input.device) > p).to(input.dtype)
        return input.mul_(mask).mul_(1.0 / (1.0 - p))

    if p == 1:
        # zeroes everything in this case.
        return _C.zeros_like(input)

    out, _mask = _C.native_dropout(input, p)
    return out

def dropout2d(input, p=0.5, training=True, inplace=False):
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    if not training or p == 0:
        return input
        
    # Input must be at least 2D (N, C, ...)
    if input.dim() < 2:
         raise ValueError("Feature dropout requires at least 2 dimensions")
         
    shape = list(input.shape)
    shape[2:] = [1] * (input.dim() - 2)
    
    mask = (_C.rand(shape, device=input.device) > p).to(input.dtype)
    scale = 1.0 / (1.0 - p)
    
    if inplace:
        return input.mul_(mask).mul_(scale)
    else:
        return input * mask * scale

def dropout3d(input, p=0.5, training=True, inplace=False):
    return dropout2d(input, p, training, inplace)

def alpha_dropout(input, p=0.5, training=True, inplace=False):
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    if not training or p == 0:
        return input

    # Native fused forward plus generated backward through the saved mask.
    if p == 1:
        result = _C.zeros_like(input)
    else:
        result, _mask = _C.native_alpha_dropout(input, p)
    if inplace:
        return input.copy_(result)
    return result


def feature_dropout(input, p=0.5, training=False, inplace=False):
    r"""Randomly zeroes entire channels (dim 1), port of
"""
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    if input.dim() < 2:
        raise RuntimeError(
            f"Feature dropout requires at least 2 dimensions in the input, "
            f"but got {input.dim()}")
    if not training or p == 0 or input.numel() == 0:
        return input

    result, _mask = _C.native_feature_dropout(input, p)
    if inplace:
        return input.copy_(result)
    return result


def dropout_(input, p=0.5, training=True):
    r"""In-place version of :func:`dropout`."""
    return dropout(input, p=p, training=training, inplace=True)


def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    r"""In-place version of :func:`rrelu`."""
    return rrelu(input, lower=lower, upper=upper, training=training,
                 inplace=True)


def feature_dropout_(input, p=0.5, training=True):
    r"""In-place version of :func:`feature_dropout`."""
    return feature_dropout(input, p=p, training=training, inplace=True)

# Pooling helpers
def _pair(x):
    if isinstance(x, (int, float)):
        return (x, x)
    return tuple(x)

def _single(x):
    if isinstance(x, (int, float)):
        return (x,)
    return tuple(x)

def _triple(x):
    if isinstance(x, (int, float)):
        return (x, x, x)
    return tuple(x)

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    r"""Applies a 1D convolution over an input signal composed of several input planes.

    See :class:`~tensorplay.nn.Conv1d` for details and output shape.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
        weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
        bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
        stride: the stride of the convolving kernel. Can be a single number or
          a one-element tuple `(sW,)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a single number or a one-element tuple `(padW,)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a one-element tuple `(dW,)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
          the number of groups. Default: 1

    Examples::

        >>> inputs = tp.randn(33, 16, 30)
        >>> filters = tp.randn(20, 16, 5)
        >>> F.conv1d(inputs, filters)
    """
    captured = _capture_call(conv1d, (input, weight, bias, stride, padding, dilation, groups), {})
    if captured is not None:
        return captured
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    if bias is None:
        bias = Tensor()
    return _C.conv1d(input, weight, bias, stride, padding, dilation, groups)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    r"""Applies a 2D convolution over an input image composed of several input planes.

    See :class:`~tensorplay.nn.Conv2d` for details and output shape.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
        bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, both :math:`\text{in\_channels}` and :math:`\text{out\_channels}`
          should be divisible by the number of groups. Default: 1

    Examples::

        >>> # With square kernels and equal stride
        >>> filters = tp.randn(8, 4, 3, 3)
        >>> inputs = tp.randn(1, 4, 5, 5)
        >>> F.conv2d(inputs, filters, padding=1)
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    captured = _capture_call(
        conv2d,
        (input, weight, bias, stride, padding, dilation, groups),
        {},
    )
    if captured is not None:
        return captured
    if bias is None:
        bias = Tensor()
    return _C.conv2d(input, weight, bias, stride, padding, dilation, groups)

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    r"""Applies a 3D convolution over an input image composed of several input planes.

    See :class:`~tensorplay.nn.Conv3d` for details and output shape.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iD, iH , iW)`
        weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kD, kH , kW)`
        bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a single number or a tuple `(padD, padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dD, dH, dW)`. Default: 1
        groups: split input into groups, both :math:`\text{in\_channels}` and :math:`\text{out\_channels}`
          should be divisible by the number of groups. Default: 1

    Examples::

        >>> # With square kernels and equal stride
        >>> filters = tp.randn(8, 4, 3, 3, 3)
        >>> inputs = tp.randn(1, 4, 5, 5, 5)
        >>> F.conv3d(inputs, filters, padding=1)
    """
    captured = _capture_call(conv3d, (input, weight, bias, stride, padding, dilation, groups), {})
    if captured is not None:
        return captured
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    if bias is None:
        bias = Tensor()
    return _C.conv3d(input, weight, bias, stride, padding, dilation, groups)

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    captured = _capture_call(conv_transpose2d, (input, weight, bias, stride, padding, output_padding, groups, dilation), {})
    if captured is not None:
        return captured
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)
    if bias is None:
        bias = Tensor()
    return _C.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    stride = _triple(stride)
    padding = _triple(padding)
    output_padding = _triple(output_padding)
    dilation = _triple(dilation)
    if bias is None:
        bias = Tensor()
    return _C.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)

def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    stride = _single(stride)
    padding = _single(padding)
    output_padding = _single(output_padding)
    dilation = _single(dilation)
    if bias is None:
        bias = Tensor()
    return _C.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    r"""
    """
    if input.dim() not in (3, 4):
        raise ValueError(
            f"unfold: expected 3D (unbatched) or 4D input, got {input.dim()}D")
    return _C.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))

def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    r"""Combine an array of sliding local blocks into a tensor containing
    """
    if input.dim() not in (2, 3):
        raise ValueError(
            f"fold: expected 2D (unbatched) or 3D input, got {input.dim()}D")
    return _C.col2im(input, _pair(output_size), _pair(kernel_size), _pair(dilation),
                     _pair(padding), _pair(stride))

def conv_tbc(input, weight, bias=None, pad=0):
    r"""Applies a 1D convolution over an input of shape (T, B, C) along the
    ``(kernel_width, in_channels, out_channels)``; the math is a standard
    cross-channel conv1d after permuting to (B, C, T).
    """
    if input.dim() != 3:
        raise ValueError("conv_tbc: input must have 3 dimensions (T, B, C)")
    if weight.dim() != 3:
        raise ValueError(
            "Weight tensor must have 3 dims: kernel_width, in_channels, out_channels.")
    if weight.size(1) != input.size(2):
        raise ValueError(
            f"Input dim 2 (input channels) is not == dim 1 in the weight tensor")
    x = input.permute(1, 2, 0)              # (B, C_in, T)
    w = weight.permute(2, 1, 0)             # (C_out, C_in, k)
    if bias is None:
        bias = Tensor()
    out = _C.conv1d(x, w.contiguous(), bias, (1,), (pad,), (1,), 1)
    return out.permute(2, 0, 1)             # (T, B, C_out)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if return_indices:
        return max_pool2d_with_indices(
            input, kernel_size, stride=stride, padding=padding,
            dilation=dilation, ceil_mode=ceil_mode)
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    captured = _capture_call(
        max_pool2d,
        (input, kernel_size, stride, padding, dilation, ceil_mode, return_indices),
        {},
    )
    if captured is not None:
        return captured
    # native kernel assumes contiguous layout; normalize views (no-op when
    return _C.max_pool2d(input.contiguous(), kernel_size, stride, padding,
                         dilation, ceil_mode)

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    captured = _capture_call(avg_pool2d, (input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override), {})
    if captured is not None:
        return captured
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    padding = _pair(padding)
    return _C.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def adaptive_avg_pool2d(input, output_size):
    # output_size can be int or (int, int) or (None, int) etc.
    output_size = _pair(output_size)
    captured = _capture_call(adaptive_avg_pool2d, (input, output_size), {})
    if captured is not None:
        return captured
    return _C.adaptive_avg_pool2d(input, output_size)

def adaptive_max_pool2d(input, output_size):
    captured = _capture_call(adaptive_max_pool2d, (input, output_size), {})
    if captured is not None:
        return captured
    output_size = list(_pair(output_size))
    # Route through the (values, indices) op so autograd saves indices and the
    return _C.adaptive_max_pool2d_with_indices(input, output_size)[0]

# Normalization functions

def batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    captured = _capture_call(
        batch_norm,
        (input, running_mean, running_var, weight, bias, training, momentum, eps),
        {},
    )
    if captured is not None:
        return captured
    return _C.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    captured = _capture_call(layer_norm, (input, normalized_shape, weight, bias, eps), {})
    if captured is not None:
        return captured
    normalized_shape = _single(normalized_shape)
    return _C.layer_norm(input, normalized_shape, weight, bias, eps)

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    captured = _capture_call(group_norm, (input, num_groups, weight, bias, eps), {})
    if captured is not None:
        return captured
    return _C.group_norm(input, num_groups, weight, bias, eps)

def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    captured = _capture_call(instance_norm, (input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps), {})
    if captured is not None:
        return captured
    return _C.instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps)

def pad(input, pad, mode='constant', value=0):
    r"""Pads tensor.  ``pad`` values are described starting from the last
    Non-constant modes support the last 3 dimensions of a 3D/4D/5D input
    """
    captured = _capture_call(globals()["pad"], (input, pad, mode, value), {})
    if captured is not None:
        return captured
    if mode == 'constant':
        return _C.constant_pad_nd(input, list(pad), value)
    ndim = input.dim()
    if ndim == 3:
        if len(pad) != 2:
            raise ValueError("3D tensors expect 2 values for padding")
    elif ndim == 4:
        if len(pad) != 4:
            raise ValueError("4D tensors expect 4 values for padding")
    elif ndim == 5:
        if len(pad) != 6:
            raise ValueError("5D tensors expect 6 values for padding")
    else:
        raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant "
                                  "padding are supported for now")
    pad = list(pad)
    if mode == 'reflect':
        return _C.reflection_pad_nd(input, pad)
    if mode == 'replicate':
        return _C.replication_pad_nd(input, pad)
    if mode == 'circular':
        return _C.circular_pad_nd(input, pad)
    raise ValueError(f"Padding mode '{mode}' not supported")

# Loss functions
def mse_loss(input, target, reduction='mean'):
    captured = _capture_call(mse_loss, (input, target, reduction), {})
    if captured is not None:
        return captured
    if not (target.size() == input.size()):
        print(f"Warning: Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.")
    
    reduction_enum = 1
    if reduction == 'none': reduction_enum = 0
    elif reduction == 'mean': reduction_enum = 1
    elif reduction == 'sum': reduction_enum = 2
    else: raise ValueError(f"{reduction} is not a valid value for reduction")

    return _C.mse_loss(input, target, reduction_enum)

def _nll_loss_red_enum(reduction):
    if reduction == 'none': return 0
    elif reduction == 'mean': return 1
    elif reduction == 'sum': return 2
    raise ValueError(f"{reduction} is not a valid value for reduction")

def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='mean'):
    r"""The negative log likelihood loss.

    ``input`` with a scalar ``target``, 2D ``(N, C)``, and N-d
    ``target``.

    See :class:`~tensorplay.nn.NLLLoss` for details.
    """
    captured = _capture_call(nll_loss, (input, target, weight, size_average, ignore_index, reduce, reduction), {})
    if captured is not None:
        return captured
    if size_average is not None or reduce is not None:
        if size_average is None: size_average = True
        if reduce is None: reduce = True
        if not reduce: reduction = 'none'
        elif size_average: reduction = 'mean'
        else: reduction = 'sum'

    reduction_enum = _nll_loss_red_enum(reduction)

    if input.dim() <= 2:
        # nll_loss returns (output, total_weight)
        output, _ = _C.nll_loss(input, target, weight, reduction_enum, ignore_index)
        return output

    if input.dim() == 4:
        # input with (N, H, W) target; autograd flows through
        # nll_loss2d_backward.
        t = target if target.dtype == DType.int64 else target.to(DType.int64)
        output, _ = _C.nll_loss2d(input, t, weight, reduction_enum, ignore_index)
        return output

    # every spatial position acts as its own batch row, so move classes last,
    # flatten to (-1, C), run the 2-D kernel and restore the target shape.
    if tuple(target.size())[1:] != tuple(input.size())[2:]:
        expected = tuple(input.size()[:1] + input.size()[2:])
        raise ValueError(f"Expected target size {expected}, got {tuple(target.size())}")
    n = input.size(0)
    c = input.size(1)
    # (N, C, d_1, ..., d_k) -> (N, d_1, ..., d_k, C) -> (N * prod(d_i), C);
    # row order matches target's contiguous flattening.
    x = input.permute([0] + list(range(2, input.dim())) + [1]).contiguous().reshape(-1, c)
    t = target.contiguous().reshape(-1)
    if t.dtype != DType.int64:
        t = t.to(DType.int64)
    output, _ = _C.nll_loss(x, t, weight, reduction_enum, ignore_index)
    if reduction == 'none':
        output = output.reshape(target.size())
    return output

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean', label_smoothing=0.0):
    r"""Compute the cross entropy loss between input logits and target.

    the class-probability path, positive ``label_smoothing`` blends the NLL
    with a smoothed uniform term, and otherwise this is
    ``nll_loss(log_softmax(input), target)`` with N-d support.

    See :class:`~tensorplay.nn.CrossEntropyLoss` for details.
    """
    captured = _capture_call(cross_entropy, (input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing), {})
    if captured is not None:
        return captured
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)

    # Hot path for the common training-loop case ((N, C) logits, int64
    # class targets, no class weights / label smoothing): go straight to
    # the native ops, skipping the legacy-reduction and N-d shape handling.
    if (weight is None and label_smoothing == 0.0 and input.dim() == 2
            and target.dtype == tensorplay.int64):
        red = 1 if reduction == 'mean' else (
            2 if reduction == 'sum' else (0 if reduction == 'none' else -1))
        if red >= 0:
            output, _ = _C.nll_loss(
                _C.log_softmax(input, 1, tensorplay.undefined),
                target, None, red, ignore_index)
            return output

    class_dim = 0 if input.dim() == 1 else 1
    n_classes = input.size(class_dim)
    if weight is not None and (weight.dim() != 1 or weight.numel() != n_classes):
        raise ValueError(
            f"cross_entropy: weight tensor should be defined either for all "
            f"{n_classes} classes or no classes but got weight tensor of "
            f"shape: {tuple(weight.size())}")

    if tuple(target.size()) == tuple(input.size()):
        # Soft targets when input and target shapes are the same
        # (port of cross_entropy_loss_prob_target, LossNLL.cpp:529).
        if ignore_index >= 0:
            raise ValueError("ignore_index is not supported for floating point target")
        if label_smoothing > 1.0:
            raise ValueError(f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}")
        input_ = log_softmax(input, class_dim)
        if label_smoothing > 0.0:
            target = target * (1 - label_smoothing) + label_smoothing / n_classes
        if weight is not None:
            w_shape = [1] * input.dim()
            w_shape[class_dim] = n_classes
            loss = -(input_ * target * weight.view(w_shape)).sum(class_dim)
        else:
            loss = -(input_ * target).sum(class_dim)
        if reduction == "none":
            return loss
        total = loss.sum()
        if reduction == "sum":
            return total
        if input.numel() == 0:
            return tensorplay.full([], float("nan"), dtype=total.dtype, device=total.device)
        return total / (input.numel() / n_classes)

    if label_smoothing > 0.0:
        # Port of cross_entropy_loss_label_smoothing (LossNLL.cpp:577).
        if label_smoothing > 1.0:
            raise ValueError(f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}")
        input_ = log_softmax(input, class_dim)
        nllloss = nll_loss(input_, target, weight, None, ignore_index, None, reduction)
        if weight is not None:
            w_shape = [1] * input_.dim()
            w_shape[class_dim] = n_classes
            smooth_loss = -(input_ * weight.view(w_shape)).sum(class_dim)
        else:
            smooth_loss = -input_.sum(class_dim)
        ignore_mask = target.eq(ignore_index)
        smooth_loss = smooth_loss.masked_fill(ignore_mask, 0.0)
        if reduction == "mean":
            if weight is not None:
                filtered_target = target.masked_fill(ignore_mask, 0)
                tgt_weights = weight.index_select(0, filtered_target.reshape(-1))
                weight_sum = tgt_weights.masked_fill_(ignore_mask.reshape(-1), 0).sum()
                ret = smooth_loss.sum() / weight_sum
            else:
                true_mask = tensorplay.logical_not(ignore_mask)
                ret = smooth_loss.sum() / true_mask.to(smooth_loss.dtype).sum()
        elif reduction == "sum":
            ret = smooth_loss.sum()
        elif reduction == "none":
            ret = smooth_loss
        else:
            raise ValueError(f"{reduction} is not valid")
        return (1 - label_smoothing) * nllloss + ret * (label_smoothing / n_classes)

    return nll_loss(log_softmax(input, class_dim), target, weight, None, ignore_index, None, reduction)


# -----------------------------------------------------------------------------
# Activation / misc functions.  These are thin wrappers over the native
# dispatcher ops declared in the native schema; the element-wise
# -----------------------------------------------------------------------------

def gelu(input: Tensor, approximate: str = 'none') -> Tensor:
    r"""gelu(input, approximate='none') -> Tensor

    When `approximate` is 'none', applies
    :math:`\text{GELU}(x) = x * \Phi(x)`; 'tanh' uses the tanh estimation.
    """
    captured = _capture_call(gelu, (input,), {"approximate": approximate})
    if captured is not None:
        return captured
    return tensorplay._C.gelu(self=input, approximate=approximate)


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    r"""relu6(input, inplace=False) -> Tensor

    """
    out = tensorplay.relu6(input)
    return input.copy_(out) if inplace else out


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""hardswish(input, inplace=False) -> Tensor"""
    out = tensorplay.hardswish(input)
    return input.copy_(out) if inplace else out


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""hardsigmoid(input, inplace=False) -> Tensor"""
    out = tensorplay.hardsigmoid(input)
    return input.copy_(out) if inplace else out


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    r"""leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor"""
    out = tensorplay.leaky_relu(input, negative_slope)
    return input.copy_(out) if inplace else out


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    r"""softplus(input, beta=1, threshold=20) -> Tensor"""
    return tensorplay.softplus(input, beta, threshold)


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""elu(input, alpha=1, inplace=False) -> Tensor"""
    out = tensorplay.elu(input, alpha)
    return input.copy_(out) if inplace else out


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    r"""mish(input, inplace=False) -> Tensor"""
    out = tensorplay.mish(input)
    return input.copy_(out) if inplace else out


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""selu(input, inplace=False) -> Tensor"""
    out = tensorplay.selu(input)
    return input.copy_(out) if inplace else out


def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    r"""celu(input, alpha=1, inplace=False) -> Tensor"""
    out = tensorplay.celu(input, alpha)
    return input.copy_(out) if inplace else out


def glu(input: Tensor, dim: int = -1) -> Tensor:
    r"""glu(input, dim=-1) -> Tensor

    Gated Linear Unit: :math:`a * \sigma(b)` where the input is split in half
    along ``dim``.
    """
    return tensorplay.glu(input, dim)


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    r"""normalize(input, p=2, dim=1, eps=1e-12) -> Tensor

    Performs :math:`L_p` normalization over the specified dimension —
    """
    denom = input.norm([dim], p, True).clamp_min(eps)
    return input / denom


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    r"""one_hot(tensor, num_classes=-1) -> Tensor"""
    return tensorplay.one_hot(tensor, num_classes)


def interpolate(
    input: Tensor,
    size=None,
    scale_factor=None,
    mode: str = 'nearest',
    align_corners=None,
    recompute_scale_factor=None,
    antialias: bool = False,
) -> Tensor:
    r"""interpolate(input, size=None, scale_factor=None, mode='nearest',
    align_corners=None) -> Tensor

    Routes to the native ``upsample_*`` ops exactly like
    """
    captured = _capture_call(interpolate, (input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias), {})
    if captured is not None:
        return captured
    if antialias:
        raise NotImplementedError("interpolate: antialias=True is not supported")
    if size is None and scale_factor is None:
        raise ValueError("need to define size or scale_factor")
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")

    ndim = input.dim()
    # mode branches can operate on a concrete size.
    if size is None and scale_factor is not None:
        spatial = ndim - 2
        if isinstance(scale_factor, (int, float)):
            scale_list = [scale_factor] * spatial
        else:
            scale_list = list(scale_factor)
        import math
        size = [int(math.floor(float(input.size(2 + i)) * s))
                for i, s in enumerate(scale_list)]
    if mode in ('nearest', 'nearest-exact'):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with interpolating modes")
        if ndim == 3:
            size_ = [size] * 1 if isinstance(size, int) else list(size)
            return tensorplay.upsample_nearest1d(input, size_)
        elif ndim == 4:
            size_ = [size] * 2 if isinstance(size, int) else list(size)
            return tensorplay.upsample_nearest2d(input, size_)
        elif ndim == 5:
            size_ = [size] * 3 if isinstance(size, int) else list(size)
            return tensorplay.upsample_nearest3d(input, size_)
        raise ValueError(f"Expected 3D, 4D or 5D input, got {ndim}D")

    if align_corners is None:
        align_corners = False

    import math
    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            spatial = ndim - 2
            scale_factor = [scale_factor] * spatial
        if recompute_scale_factor:
            size = [int(math.floor(float(input.shape[2 + i]) * f)) for i, f in enumerate(scale_factor)]

    if mode == 'linear':
        if ndim != 3:
            raise ValueError("linear interpolation expects 3D input")
        return tensorplay.upsample_linear1d(input, list(size), align_corners)
    elif mode == 'bilinear':
        if ndim != 4:
            raise ValueError("bilinear interpolation expects 4D input")
        return tensorplay.upsample_bilinear2d(input, list(size), align_corners)
    elif mode == 'bicubic':
        if ndim != 4:
            raise ValueError("bicubic interpolation expects 4D input")
        return tensorplay.upsample_bicubic2d(input, list(size), align_corners)
    elif mode == 'trilinear':
        if ndim != 5:
            raise ValueError("trilinear interpolation expects 5D input")
        return tensorplay.upsample_trilinear3d(input, list(size), align_corners)
    raise NotImplementedError(f"interpolate: mode '{mode}' is not supported")


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor = None,
    in_proj_bias: Tensor = None,
    bias_k=None,
    bias_v=None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Tensor = None,
    out_proj_bias: Tensor = None,
    training: bool = True,
    key_padding_mask=None,
    need_weights: bool = True,
    attn_mask=None,
    use_separate_proj_weight: bool = False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
):
    r"""

    Follows the structure of
    _scaled_dot_product_attention paths composed from dispatched primitives.
    """
    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight:
        q = tensorplay.linear(query, q_proj_weight)
        k = tensorplay.linear(key, k_proj_weight)
        v = tensorplay.linear(value, v_proj_weight)
    else:
        qkv = tensorplay.linear(
            tensorplay.cat([query, key, value], dim=0),
            in_proj_weight,
            in_proj_bias,
        )
        q, k, v = qkv[:tgt_len], qkv[tgt_len:tgt_len + src_len], qkv[tgt_len + src_len:]

    q = q * scaling
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len_b = src_len
    if add_zero_attn:
        zero = tensorplay.zeros(k.shape[0], 1, k.shape[2], dtype=k.dtype, device=k.device)
        k = tensorplay.cat([k, zero], dim=1)
        v = tensorplay.cat([v, zero], dim=1)
        src_len_b += 1

    q4 = q.reshape(bsz, num_heads, tgt_len, head_dim)
    k4 = k.reshape(bsz, num_heads, src_len_b, head_dim)
    v4 = v.reshape(bsz, num_heads, src_len_b, head_dim)
    attn_output_weights = q4 @ k4.transpose(-2, -1)

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dtype == tensorplay.bool:
            zeros = tensorplay.zeros_like(attn_mask, dtype=q.dtype)
            ones = tensorplay.ones_like(zeros)
            attn_mask = tensorplay.where(attn_mask, zeros - ones, zeros)
        attn_output_weights = attn_output_weights + attn_mask.to(q.dtype).reshape(1, 1, *attn_mask.shape[-2:])

    if key_padding_mask is not None:
        if key_padding_mask.dtype == tensorplay.bool:
            zeros = tensorplay.zeros_like(key_padding_mask, dtype=q.dtype)
            ones = tensorplay.ones_like(zeros)
            key_padding_mask = tensorplay.where(key_padding_mask, zeros - ones, zeros)
        kp = key_padding_mask.transpose(0, 1).reshape(1, bsz, 1, src_len).to(q.dtype)
        attn_output_weights = attn_output_weights + kp

    attn_output_weights = tensorplay.softmax(attn_output_weights, dim=-1)

    if dropout_p > 0.0 and training:
        keep = (tensorplay.rand(attn_output_weights.shape) >= dropout_p).to(attn_output_weights.dtype)
        attn_output_weights = attn_output_weights * keep / (1.0 - dropout_p)

    attn_output = (attn_output_weights @ v4).reshape(bsz * num_heads, tgt_len, head_dim)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = tensorplay.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        aw = attn_output_weights.view(bsz, num_heads, tgt_len, src_len_b)
        if average_attn_weights:
            aw = aw.mean(dim=1)
        return attn_output, aw
    return attn_output, None


# -----------------------------------------------------------------------------
# dispatcher ops available here, following the formulas in
# -----------------------------------------------------------------------------


def _get_reduction_enum(reduction: str) -> int:
    if reduction == 'none':
        return 0
    elif reduction == 'mean':
        return 1
    elif reduction == 'sum':
        return 2
    raise ValueError(f"{reduction} is not valid")


def _legacy_get_string(size_average, reduce):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        return 'mean'
    elif reduce:
        return 'sum'
    else:
        return 'none'


def logsigmoid(input: Tensor) -> Tensor:
    r"""logsigmoid(input) -> Tensor

    Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`

    See :class:`~tensorplay.nn.LogSigmoid` for more details.
    """
    # autograd flows through log_sigmoid_backward.
    return _C.log_sigmoid(input)


def softmin(input: Tensor, dim: Optional[int] = None, dtype=None) -> Tensor:
    r"""Apply a softmin function.

    Note that :math:`\text{Softmin}(x) = \text{Softmax}(-x)`.

    See :class:`~tensorplay.nn.Softmin` for more details.
    """
    if dim is None:
        dim = input.dim() - 1 if input.dim() in (0, 1, 3) else 1
    return softmax(-input, dim=dim, dtype=dtype)


def softsign(input: Tensor) -> Tensor:
    r"""softsign(input) -> Tensor

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    See :class:`~tensorplay.nn.Softsign` for more details.
    """
    return input / (input.abs() + 1)


def tanhshrink(input: Tensor) -> Tensor:
    r"""tanhshrink(input) -> Tensor

    Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`

    See :class:`~tensorplay.nn.Tanhshrink` for more details.
    """
    return input - tensorplay.tanh(input)


def hardtanh(
    input: Tensor, min_val: float = -1.0, max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    r"""hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False) -> Tensor"""
    result = tensorplay.hardtanh(input, min_val=min_val, max_val=max_val)
    if inplace:
        return input.copy_(result)
    return result


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    r"""hardshrink(input, lambd=0.5) -> Tensor

    Applies the hard shrinkage function element-wise.

    See :class:`~tensorplay.nn.Hardshrink` for more details.
    """
    return tensorplay.hardshrink(input, lambd)


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    r"""softshrink(input, lambd=0.5) -> Tensor

    Applies the soft shrinkage function element-wise.

    See :class:`~tensorplay.nn.Softshrink` for more details.
    """
    return tensorplay.softshrink(input, lambd)


def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

    Randomized leaky ReLU.

    See :class:`~tensorplay.nn.RReLU` for more details.
    """
    if training:
        noise = lower + (upper - lower) * _C.rand(input.shape, device=input.device)
        result = _C.rrelu_with_noise(input, noise, lower, upper, True)
    else:
        # Eval ignores noise (leaky slope (lower+upper)/2); pass input itself
        # to avoid an allocation.
        result = _C.rrelu_with_noise(input, input, lower, upper, False)
    if inplace:
        return input.copy_(result)
    return result


# -----------------------------------------------------------------------------
# Pooling helpers
# -----------------------------------------------------------------------------


def avg_pool1d(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override=None,
) -> Tensor:
    r"""avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
    count_include_pad=True, divisor_override=None) -> Tensor

    Applies a 1D average pooling over an input signal composed of several
    input planes. Input shape ``(N, C, L)`` or unbatched ``(C, L)``.
    """
    captured = _capture_call(avg_pool1d, (input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override), {})
    if captured is not None:
        return captured
    unbatched = input.dim() == 2
    x = input.unsqueeze(0) if unbatched else input
    k = _single(kernel_size)[0]
    s = k if stride is None else _single(stride)[0]
    p = _single(padding)[0]
    out = avg_pool2d(
        x.unsqueeze(3), (k, 1), (s, 1), (p, 0), ceil_mode,
        count_include_pad, divisor_override,
    ).squeeze(3)
    return out.squeeze(0) if unbatched else out


def max_pool1d(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    return_indices: bool = False,
):
    r"""max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
    ceil_mode=False, return_indices=False) -> Tensor

    Applies a 1D max pooling over an input signal composed of several input
    planes. Input shape ``(N, C, L)`` or unbatched ``(C, L)``.
    """
    captured = _capture_call(max_pool1d, (input, kernel_size, stride, padding, dilation, ceil_mode, return_indices), {})
    if captured is not None:
        return captured
    if return_indices:
        return max_pool1d_with_indices(
            input, kernel_size, stride=stride, padding=padding,
            dilation=dilation, ceil_mode=ceil_mode)
    unbatched = input.dim() == 2
    x = input.unsqueeze(0) if unbatched else input
    k = _single(kernel_size)[0]
    s = k if stride is None else _single(stride)[0]
    p = _single(padding)[0]
    d = _single(dilation)[0]
    out = max_pool2d(
        x.unsqueeze(3), (k, 1), (s, 1), (p, 0), (d, 1), ceil_mode,
    ).squeeze(3)
    return out.squeeze(0) if unbatched else out


def adaptive_avg_pool1d(input: Tensor, output_size) -> Tensor:
    r"""adaptive_avg_pool1d(input, output_size) -> Tensor"""
    unbatched = input.dim() == 2
    x = input.unsqueeze(0) if unbatched else input
    out = adaptive_avg_pool2d(x.unsqueeze(3), (output_size, 1)).squeeze(3)
    return out.squeeze(0) if unbatched else out


def adaptive_max_pool1d(input: Tensor, output_size):
    r"""adaptive_max_pool1d(input, output_size) -> Tensor"""
    unbatched = input.dim() == 2
    x = input.unsqueeze(0) if unbatched else input
    out = adaptive_max_pool2d(x.unsqueeze(3), (output_size, 1)).squeeze(3)
    return out.squeeze(0) if unbatched else out


def lp_pool2d(
    input: Tensor,
    norm_type,
    kernel_size,
    stride=None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Apply a 2D power-average pooling over an input signal.

    If the sum of all inputs to the power of `p` is zero, the gradient is set
    to zero as well.

    See :class:`~tensorplay.nn.LPPool2d` for details.
    """
    kw, kh = _pair(kernel_size)
    if isinstance(norm_type, (int, float)):
        if norm_type == 0:
            raise ValueError(f"norm_type must be a non-zero value, but got {norm_type}")
        if norm_type == float("inf"):
            return max_pool2d(input.abs(), kernel_size, stride, 0, 1, ceil_mode)
        if norm_type == -float("inf"):
            return -max_pool2d(-input.abs(), kernel_size, stride, 0, 1, ceil_mode)

    if stride is not None:
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (tensorplay.sign(out) * relu(tensorplay.abs(out))).mul(kw * kh).pow(1.0 / norm_type)


def lp_pool1d(
    input: Tensor,
    norm_type,
    kernel_size,
    stride=None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Apply a 1D power-average pooling over an input signal.

    See :class:`~tensorplay.nn.LPPool1d` for details.
    """
    k = _single(kernel_size)[0]
    s = None if stride is None else _single(stride)[0]

    if isinstance(norm_type, (int, float)):
        if norm_type == 0:
            raise ValueError(f"norm_type must be a non-zero value, but got {norm_type}")
        if norm_type == float("inf"):
            return max_pool1d(input.abs(), kernel_size, stride, 0, 1, ceil_mode)
        if norm_type == -float("inf"):
            return -max_pool1d(-input.abs(), kernel_size, stride, 0, 1, ceil_mode)

    if stride is not None:
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (tensorplay.sign(out) * relu(tensorplay.abs(out))).mul(k).pow(1.0 / norm_type)


def local_response_norm(
    input: Tensor,
    size: int,
    alpha: float = 1e-4,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor:
    r"""Apply local response normalization over an input signal.

    The input signal is composed of several input planes, where channels
    occupy the second dimension. Normalization is applied across channels.

    See :class:`~tensorplay.nn.LocalResponseNorm` for details.
    """
    captured = _capture_call(local_response_norm, (input, size, alpha, beta, k), {})
    if captured is not None:
        return captured
    dim = input.dim()
    if dim < 3:
        raise ValueError(
            f"Expected 3D or higher dimensionality input (got {dim} dimensions)"
        )
    if input.numel() == 0:
        return input

    # Windowed sum of squares along the channel axis (dim 1), equivalent to
    div = input.mul(input)
    pad_left = size // 2
    pad_right = (size - 1) // 2

    def _zero_channels(n: int, ref: Tensor) -> Tensor:
        shape = list(ref.shape)
        shape[1] = n
        return tensorplay.zeros(shape, dtype=ref.dtype, device=ref.device)

    # The leading zero channel beyond ``pad_left`` makes the prefix sums
    # 1-indexed, so the window covering output channel ``i`` -- padded
    # channels ``[i, i + size)`` -- is exactly ``cs[i + size] - cs[i]``.
    parts = [_zero_channels(pad_left + 1, div), div]
    if pad_right:
        parts.append(_zero_channels(pad_right, div))
    padded = tensorplay.cat(parts, dim=1)

    cs = padded.cumsum(1)
    c = input.size(1)
    hi = tensorplay.narrow(cs, 1, size, c)
    lo = tensorplay.narrow(cs, 1, 0, c)
    window_sum = hi - lo

    div = window_sum.mul(alpha / size).add(k).pow(beta)
    return input / div


def dropout1d(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly zero out entire channels (a channel is a 1D feature map).

    See :class:`~tensorplay.nn.Dropout1d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    inp_dim = input.dim()
    if inp_dim not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {inp_dim}D input. "
            "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
            "spatial dimension, a channel dimension, and an optional batch dimension "
            "(i.e. 2D or 3D inputs)."
        )
    if not training or p == 0:
        return input

    is_batched = inp_dim == 3
    x = input if is_batched else input.unsqueeze(0)

    mask_shape = list(x.shape)
    mask_shape[2:] = [1] * (x.dim() - 2)
    mask = (_C.rand(mask_shape, device=x.device) > p).to(x.dtype)
    scale = 1.0 / (1.0 - p)

    if inplace:
        result = x.mul_(mask).mul_(scale)
    else:
        result = x * mask * scale

    if not is_batched:
        result = result.squeeze(0)
    return result


def feature_alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    r"""Randomly masks out entire channels, setting activations to the
    negative saturation value of the SELU activation function.

    See :class:`~tensorplay.nn.FeatureAlphaDropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    if input.dim() < 2:
        raise RuntimeError(
            f"feature_alpha_dropout: Expected input to have at least 2 dimensions, "
            f"but got {input.dim()}"
        )

    if not training or p == 0 or input.numel() == 0:
        return input

    alpha_c = 1.7580993408473766
    a = 1.0 / math.sqrt((alpha_c * alpha_c * p + 1) * (1 - p))
    b_coeff = alpha_c * a

    mask_shape = list(input.shape)
    mask_shape[2:] = [1] * (input.dim() - 2)
    noise = (_C.rand(mask_shape, device=input.device) > p).to(input.dtype)
    b = (noise - 1).mul(b_coeff).add(b_coeff * p)
    noise = noise.mul(a)

    result = input * noise + b
    if inplace:
        return input.copy_(result)
    return result


# -----------------------------------------------------------------------------
# Distance helpers
# -----------------------------------------------------------------------------


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    r"""Returns cosine similarity between x1 and x2, computed along dim."""
    denom = x1.norm([dim]).mul(x2.norm([dim]))
    denom = tensorplay.clamp(denom, min=eps)
    return (x1 * x2).sum(dim) / denom


# -----------------------------------------------------------------------------
# Mask helpers used by Transformer modules.
# -----------------------------------------------------------------------------


def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type=None,
    other_name: str = "",
    target_type=None,
    check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = _mask_dtype in (
            DType.float16, DType.bfloat16, DType.float32, DType.float64,
        )
        if _mask_dtype != DType.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead.",
                    stacklevel=2,
                )
        if not _mask_is_float:
            mask = tensorplay.zeros_like(mask, dtype=target_type).masked_fill_(
                mask, float("-inf")
            )
    return mask


def _none_or_dtype(input: Optional[Tensor]):
    if input is None:
        return None
    elif isinstance(input, Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or Tensor")


# -----------------------------------------------------------------------------
# live in the native schema and backend kernels).
# -----------------------------------------------------------------------------


def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the multilabel soft margin loss.

    See :class:`~tensorplay.nn.MultiLabelSoftMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)

    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    class_dim = input.dim() - 1
    C = input.size(class_dim)
    loss = loss.sum(class_dim) / C  # only return N loss values

    if reduction == "none":
        ret = loss
    elif reduction == "mean":
        ret = loss.mean()
    elif reduction == "sum":
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Gaussian negative log likelihood loss.

    See :class:`~tensorplay.nn.GaussianNLLLoss` for details.
    """
    # Entries of var must be non-negative
    if isinstance(var, float):
        if var < 0:
            raise ValueError("var has negative entry/entries")
        var = tensorplay.ones_like(input) * var
    elif (var < 0).to(input.dtype).sum().item() > 0:
        raise ValueError("var has negative entry/entries")

    # Check var size
    if tuple(var.size()) != tuple(input.size()):
        # If var is one dimension short of input, but the sizes match otherwise,
        # then this is a homoscedastic case.
        if tuple(input.size())[:-1] == tuple(var.size()):
            var = tensorplay.unsqueeze(var, -1)
        elif (
            input.dim() == var.dim()
            and sum(y for x, y in zip(input.size(), var.size()) if x != y) == 1
        ):  # Heteroscedastic case
            pass
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(reduction + " is not valid")

    # Clamp for stability
    var = var.clone()
    with tensorplay.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (tensorplay.log(var) + (input - target) ** 2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function=None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the triplet margin loss using a custom distance function.

    See :class:`~tensorplay.nn.TripletMarginWithDistanceLoss` for details.
    """
    # Check validity of reduction mode
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    # Check validity of margin
    if margin <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")

    # Check dimensions
    a_dim = anchor.dim()
    p_dim = positive.dim()
    n_dim = negative.dim()
    if not (a_dim == p_dim and p_dim == n_dim):
        raise RuntimeError(
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        )

    # Calculate loss
    if distance_function is None:
        distance_function = tensorplay.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = tensorplay.minimum(dist_neg, dist_swap)
    loss = tensorplay.clamp(margin + dist_pos - dist_neg, 0.0)

    # Apply reduction
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:  # reduction == "none"
        return loss


# =============================================================================
# 2.15.0a0 @ 893b6406).  Ops that are native in this repo call the dispatcher
# directly; the rest are composed from dispatched primitives following the
# same math without new kernels.
# =============================================================================


def _broadcast_shapes(*shapes):
    out = []
    for dims in zip(*(reversed(s) for s in shapes)):
        d = 1
        for x in dims:
            if x != 1:
                if d != 1 and x != d:
                    raise ValueError("Shape mismatch: objects cannot be broadcast to a single shape")
                d = x
        out.append(d)
    return tuple(reversed(out))


def _apply_reduction(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"{reduction} is not a valid value for reduction")


def _band(t, lo, hi):
    """t >= lo AND t < hi, elementwise (bool)."""
    return tensorplay.logical_and(t >= lo, t < hi)


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    r"""one_hot(tensor, num_classes=-1) -> LongTensor

    Returns long tensor shaped ``tensor.shape + (num_classes,)`` with a 1 at
    """
    captured = _capture_call(one_hot, (tensor, num_classes), {})
    if captured is not None:
        return captured
    if num_classes < 0:
        if tensor.numel() == 0:
            raise RuntimeError("Cannot infer num classes from empty tensor")
        num_classes = int(tensor.max().item()) + 1
    t64 = tensor.to(DType.int64)
    rng = tensorplay.arange(num_classes, dtype=DType.int64, device=tensor.device)
    return t64.unsqueeze(-1).eq(rng).to(DType.int64)


def sigmoid(input):
    r"""sigmoid(input) -> Tensor

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`
    """
    return tensorplay.sigmoid(input)


def tanh(input):
    r"""tanh(input) -> Tensor

    Applies element-wise :math:`\text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`
    """
    return tensorplay.tanh(input)


def rms_norm(
    input: Tensor,
    normalized_shape,
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None,
) -> Tensor:
    r"""Apply Root Mean Square Layer Normalization.

    Dispatches to the native fused kernel (single dispatch, CPU vectorized
    rows / CUDA block-per-row); falls back to the composite below under
    CompositeImplicitAutograd rms_norm."""
    return _rms_norm_impl(input, normalized_shape, weight, eps)


def _rms_norm_composite(
    input: Tensor,
    normalized_shape,
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None,
) -> Tensor:
    r"""
    rms_norm composite (fp32 compute for reduced dtypes)."""
    shape = list(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else [int(normalized_shape)]
    ndim = len(shape)
    dims = list(range(input.dim() - ndim, input.dim()))
    if eps is None:
        eps = 1e-5
    compute_dtype = DType.float32 if input.dtype in (DType.float16, DType.bfloat16) else input.dtype
    x = input.to(compute_dtype)
    denom = x.pow(2).mean(dims, keepdim=True) + eps
    inv = tensorplay.rsqrt(denom)
    out = x * inv
    if weight is not None:
        w_shape = [1] * (input.dim() - ndim) + list(shape)
        out = out * weight.to(compute_dtype).view(w_shape)
    return out.to(input.dtype)


def _rms_norm_impl(
    input: Tensor,
    normalized_shape,
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None,
) -> Tensor:
    """Native fused kernel (CPU vectorized rows / CUDA block-per-row).

    Falls back to the composite above under autograd: the native forward has
    flows through its inner ops), so training graphs must keep composing."""
    needs_grad = tensorplay.is_grad_enabled() and (
        input.requires_grad
        or (weight is not None and getattr(weight, "requires_grad", False))
    )
    if needs_grad:
        return _rms_norm_composite(input, normalized_shape, weight, eps)
    return tensorplay._C.rms_norm(
        input,
        list(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else [int(normalized_shape)],
        weight,
        float(eps) if eps is not None else None,
    )


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> Tensor:
    r"""Sample from the Gumbel-Softmax distribution and optionally discretize.

    straight-through when ``hard=True``).
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.", stacklevel=2)
    u = tensorplay.empty_like(logits)
    u.exponential_()
    gumbels = -u.log()
    gumbels = (logits + gumbels) / tau
    y_soft = softmax(gumbels, dim=dim)
    if hard:
        idx = _C.argmax(y_soft, dim=dim, keepdim=True)
        ar = tensorplay.arange(y_soft.size(dim), dtype=DType.int64, device=logits.device)
        view = [1] * logits.dim()
        view[dim] = -1
        y_hard = idx.eq(ar.view(view)).to(logits.dtype)
        return y_hard.detach() - y_soft.detach() + y_soft
    return y_soft


# -----------------------------------------------------------------------------
# Loss functions (F-alignment)
# -----------------------------------------------------------------------------


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the L1 loss, with optional weighting.

    Function that takes the mean element-wise absolute value difference.
    See :class:`~tensorplay.nn.L1Loss` for details.
    """
    captured = _capture_call(l1_loss, (input, target, size_average, reduce, reduction, weight), {})
    if captured is not None:
        return captured
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    if target.size() != input.size():
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )
    if weight is not None:
        if weight.size() != input.size():
            raise ValueError("Weights and input must have the same size.")
        absolute_errors = (input - target).abs()
        weighted = absolute_errors * weight
        if reduction == "none":
            return weighted
        if reduction == "sum":
            return weighted.sum()
        return weighted.sum() / weight.sum()

    expanded_input, expanded_target = _expand_pair(input, target)
    return _C.tp_l1_loss(expanded_input, expanded_target, _get_reduction_enum(reduction))


def _expand_pair(input: Tensor, target: Tensor):
    """Broadcast input/target to a common shape when they differ."""
    if tuple(input.size()) == tuple(target.size()):
        return input, target
    shape = _broadcast_shapes(tuple(input.size()), tuple(target.size()))
    return input.expand(shape), target.expand(shape)


_REDUCTION_STRINGS = {0: "none", 1: "mean", 2: "sum"}


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    r"""Compute the Smooth L1 loss.

    Function uses a squared term if the absolute element-wise error falls
    below beta and an L1 term otherwise.
    See :class:`~tensorplay.nn.SmoothL1Loss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    if target.size() != input.size():
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )
    expanded_input, expanded_target = _expand_pair(input, target)
    enum_ = _get_reduction_enum(reduction)
    if beta == 0.0:
        return _C.tp_l1_loss(expanded_input, expanded_target, enum_)
    return _C.smooth_l1_loss(expanded_input, expanded_target, enum_, beta)


def huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    delta: float = 1.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the Huber loss, with optional weighting.

    Function uses a squared term if the absolute error falls below delta and
    a delta-scaled L1 term otherwise.
    See :class:`~tensorplay.nn.HuberLoss` for details.
    """
    if target.size() != input.size():
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )
    expanded_input, expanded_target = _expand_pair(input, target)
    enum_ = _get_reduction_enum(reduction)
    if weight is None:
        return _C.huber_loss(expanded_input, expanded_target, enum_, delta)

    if weight.size() != input.size():
        raise ValueError("Weights and input must have the same size.")
    unweighted = _C.huber_loss(expanded_input, expanded_target, 0, delta)
    weighted = unweighted * weight
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.mean()


def kl_div(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    r"""Compute the KL Divergence loss.

    ``input`` holds log-probabilities; see :class:`~tensorplay.nn.KLDivLoss`.
    Note that :attr:`reduction='mean'` divides by the number of elements and
    does not return the true KL divergence value — use ``'batchmean'``.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _get_reduction_enum(_legacy_get_string(size_average, reduce))
    else:
        if reduction == "mean":
            warnings.warn(
                "reduction: 'mean' divides the total loss by both the batch size and the support size."
                "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
                "'mean' will be changed to behave the same as 'batchmean' in the next major release.",
                stacklevel=2,
            )
        if reduction == "batchmean":
            reduction_enum = 2  # sum
        else:
            reduction_enum = _get_reduction_enum(reduction)

    reduced = _C.tp_kl_div(input, target, reduction_enum, log_target)
    if reduction == "batchmean" and input.dim() != 0:
        reduced = reduced / input.size(0)
    return reduced


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute Binary Cross Entropy between the target and input probabilities.

    See :class:`~tensorplay.nn.BCELoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _get_reduction_enum(_legacy_get_string(size_average, reduce))
    else:
        reduction_enum = _get_reduction_enum(reduction)
    if target.size() != input.size():
        raise ValueError(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}) is deprecated. "
            "Please ensure they have the same size."
        )
    if weight is not None:
        new_size = _broadcast_shapes(tuple(target.size()), tuple(weight.size()))
        weight = weight.expand(new_size)
    return _C.binary_cross_entropy(input, target, weight, reduction_enum)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute Binary Cross Entropy between target and input logits.

    optionally rescaled by ``weight``, then reduced.
    See :class:`~tensorplay.nn.BCEWithLogitsLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction_enum = _get_reduction_enum(_legacy_get_string(size_average, reduce))
    else:
        reduction_enum = _get_reduction_enum(reduction)
    if tuple(target.size()) != tuple(input.size()):
        raise ValueError(
            f"Target size ({target.size()}) must be the same as input size ({input.size()})"
        )

    log_sigmoid_input = logsigmoid(input)
    if pos_weight is not None:
        log_weight = (pos_weight - 1).mul(target).add(1)
        log_sigmoid_input = log_sigmoid_input.mul(log_weight)

    loss = (1 - target).mul(input).sub(log_sigmoid_input)
    if weight is not None:
        loss = loss.mul(weight)
    return _apply_reduction(loss, _REDUCTION_STRINGS[reduction_enum])


def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average=None,
    eps: float = 1e-8,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the Poisson negative log likelihood loss.

    See :class:`~tensorplay.nn.PoissonNLLLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return _C.tp_poisson_nll_loss(
        input, target, log_input, full, eps, _get_reduction_enum(reduction)
    )


def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the soft margin loss.

    See :class:`~tensorplay.nn.SoftMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    return _C.tp_soft_margin_loss(input, target, _get_reduction_enum(reduction))


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the cosine embedding loss.

    See :class:`~tensorplay.nn.CosineEmbeddingLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    return _C.tp_cosine_embedding_loss(
        input1, input2, target, margin, _get_reduction_enum(reduction)
    )


def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the margin ranking loss.

    See :class:`~tensorplay.nn.MarginRankingLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    return _C.tp_margin_ranking_loss(
        input1, input2, target, margin, _get_reduction_enum(reduction)
    )


def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the hinge embedding loss.

    See :class:`~tensorplay.nn.HingeEmbeddingLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    return _C.tp_hinge_embedding_loss(
        input, target, margin, _get_reduction_enum(reduction)
    )


def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the multi margin loss, with optional weighting.

    ``sum_d max(0, margin - x_y + x_d)^p * w_y / C`` over non-target classes.
    See :class:`~tensorplay.nn.MultiMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    if p != 1 and p != 2:
        raise ValueError("only p == 1 and p == 2 supported")
    if weight is not None and weight.dim() != 1:
        raise ValueError("weight must be one-dimensional")
    return _C.multi_margin_loss(input, target.to(DType.int64), p, margin, weight,
                                _get_reduction_enum(reduction))


def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the multilabel margin loss.

    for each positive label ``y`` (targets are active until the first
    ``-1``), add ``max(0, 1 - x[y] + x[d])`` over non-target labels ``d``;
    divide by C.
    See :class:`~tensorplay.nn.MultiLabelMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    return _C.multilabel_margin_loss(input, target.to(DType.int64),
                                     _get_reduction_enum(reduction))



def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the triplet loss between given input tensors and a margin
    greater than 0.

    See :class:`~tensorplay.nn.TripletMarginLoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _legacy_get_string(size_average, reduce)
    if margin <= 0:
        raise ValueError(f"margin must be greater than 0, got {margin}")
    dist_pos = pairwise_distance(anchor, positive, p, eps)
    dist_neg = pairwise_distance(anchor, negative, p, eps)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" (Balntas et al.).
    if swap:
        dist_swap = pairwise_distance(positive, negative, p, eps)
        dist_neg = tensorplay.minimum(dist_neg, dist_swap)
    loss = tensorplay.clamp(margin + dist_pos - dist_neg, min=0.0)
    return _apply_reduction(loss, reduction)


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths,
    target_lengths,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    r"""Compute the Connectionist Temporal Classification loss.

    (alpha recurrence over the blank-extended target sequence); autograd flows
    through ``log_probs`` via the composed primitives.

    Args:
        log_probs: :math:`(T, N, C)` or :math:`(T, C)` log-softmax outputs.
        targets: :math:`(N, S)` or concatenated :math:`(\sum S_n,)`.
        input_lengths / target_lengths: :math:`(N,)` or scalars.
        blank: index of the blank label. Default: 0.
        reduction: ``'none' | 'mean' | 'sum'``.
        zero_infinity: zero out infinite losses (targets too long for T).
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(1)
        unbatched = True
    else:
        unbatched = False
    if log_probs.dim() != 3:
        raise ValueError(f"ctc_loss: log_probs must be 2D or 3D, got {log_probs.dim()}D")
    T, N, C = log_probs.shape

    def _as_len_tensor(x):
        if x is None:
            return None
        if isinstance(x, Tensor):
            return x.to(DType.int64).reshape(-1)
        if isinstance(x, int):
            return tensorplay.full([N], x, dtype=DType.int64, device=log_probs.device)
        return tensorplay.tensor(list(x), dtype=DType.int64, device=log_probs.device)

    in_lens = _as_len_tensor(input_lengths)
    tgt_lens = _as_len_tensor(target_lengths)

    if targets.dim() == 1:
        rows = []
        off = 0
        for n in range(N):
            s = int(tgt_lens[n].item())
            rows.append(tensorplay.narrow(targets.to(DType.int64), 0, off, s))
            off += s
        S = max((int(r.numel()) for r in rows), default=0)
        padded = tensorplay.zeros(N, S, dtype=DType.int64, device=log_probs.device)
        for n in range(N):
            k = rows[n].numel()
            if k:
                padded[n, :k] = rows[n]
        targets2d = padded
    elif targets.dim() == 2:
        targets2d = targets.to(DType.int64)
        S = targets2d.size(1)
    else:
        raise ValueError(f"ctc_loss: targets must be 1D or 2D, got {targets.dim()}D")

    # Native dispatch: _ctc_loss + reduction compose on the dispatcher, and
    # the _ctc_loss derivative formula (derivatives.yaml) drives the backward.
    return _ctc_loss_impl(log_probs, targets2d, in_lens, tgt_lens, blank,
                          reduction, zero_infinity, unbatched)


def _ctc_zero_inf_mask(nll):
    # impossible alignments carry +inf (or NaN from inf - inf) raw NLL
    return tensorplay.logical_or(nll.ge(float("inf")), nll.ne(nll))


def _ctc_loss_impl(log_probs, targets2d, in_lens, tgt_lens, blank, reduction,
                   zero_infinity, unbatched):
    # NB: the FASTCALL binding layer for underscore ops requires kwargs.
    nll, _ = _C._ctc_loss(log_probs=log_probs, targets=targets2d,
                          input_lengths=in_lens, target_lengths=tgt_lens,
                          blank=blank, zero_infinity=zero_infinity)
    if zero_infinity:
        nll = tensorplay.where(_ctc_zero_inf_mask(nll),
                               tensorplay.zeros_like(nll), nll)
    if reduction == "none":
        return nll.squeeze(0) if unbatched else nll
    if reduction == "sum":
        return nll.sum()
    # 'mean': divide each by its target length, then average
    denom = tgt_lens.to(nll.dtype).clamp(min=1)
    return (nll / denom).mean()



def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    r"""Rearranges elements in a tensor of shape ``(*, C x r^2, H, W)`` to a
    tensor of shape ``(*, C, H x r, W x r)``.

    input[n, c*r^2 + i*r + j, h, w]``.
    """
    captured = _capture_call(pixel_shuffle, (input, upscale_factor), {})
    if captured is not None:
        return captured
    r = int(upscale_factor)
    if input.dim() != 4:
        raise ValueError(f"pixel_shuffle expects 4D input, got {input.dim()}D")
    N, C, H, W = input.shape
    if C % (r * r) != 0:
        raise ValueError(
            f"pixel_shuffle expects input channel to be divisible by square of upscale_factor, "
            f"but got C={C} and upscale_factor={r}")
    Cc = C // (r * r)
    x = input.reshape(N, Cc, r, r, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.reshape(N, Cc, H * r, W * r)


def pixel_unshuffle(input: Tensor, downscale_factor: int) -> Tensor:
    r"""Reverses the :func:`pixel_shuffle` transformation: ``(*, C, H x r,
    W x r) -> (*, C x r^2, H, W)``."""
    captured = _capture_call(pixel_unshuffle, (input, downscale_factor), {})
    if captured is not None:
        return captured
    r = int(downscale_factor)
    if input.dim() != 4:
        raise ValueError(f"pixel_unshuffle expects 4D input, got {input.dim()}D")
    N, C, H, W = input.shape
    if H % r != 0 or W % r != 0:
        raise ValueError(
            f"pixel_unshuffle expects input height and width divisible by downscale_factor, "
            f"but got H={H}, W={W} and downscale_factor={r}")
    x = input.reshape(N, C, H // r, r, W // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.reshape(N, C * r * r, H // r, W // r)


def channel_shuffle(input: Tensor, groups: int) -> Tensor:
    r"""Divide the channels in a tensor into ``g`` groups and rearrange them
    as in ShuffleNet: ``(*, C, H, W) -> (*, C, H, W)`` with channels
    interleaved across groups."""
    g = int(groups)
    if input.dim() != 4:
        raise ValueError(f"channel_shuffle expects 4D input, got {input.dim()}D")
    N, C, H, W = input.shape
    if C % g != 0:
        raise ValueError(f"channel_shuffle expects channel count divisible by groups, got C={C}, groups={g}")
    x = input.reshape(N, g, C // g, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(N, C, H, W)


def native_channel_shuffle(input: Tensor, groups: int) -> Tensor:
    return channel_shuffle(input, groups)


GRID_SAMPLE_INTERPOLATION_MODES = ('bilinear', 'nearest', 'bicubic')
GRID_SAMPLE_PADDING_MODES = ('zeros', 'border', 'reflection')


def _linspace_from_neg_one(num_steps, align_corners, dtype, device):
    """Port of AffineGridGenerator.cpp linspace_from_neg_one."""
    if num_steps <= 1:
        return tensorplay.zeros([num_steps], dtype=dtype, device=device)
    rng = tensorplay.linspace(-1, 1, num_steps, dtype=dtype, device=device)
    if not align_corners:
        rng = rng * (num_steps - 1) / num_steps
    return rng


def affine_grid(theta: Tensor, size, align_corners=None) -> Tensor:
    r"""Generate 2D or 3D flow field (sampling grid), given a batch of affine
    matrices :attr:`theta`.

    """
    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of affine_grid for details.",
            stacklevel=2)
        align_corners = False
    if len(size) == 4:
        n, _, h, w = size
        if theta.dim() != 3 or theta.size(0) != n or theta.size(1) != 2 or theta.size(2) != 3:
            raise ValueError(f"affine_grid: expected theta of shape ({n}, 2, 3), got {tuple(theta.shape)}")
        dtype, dev = theta.dtype, theta.device
        xs = _linspace_from_neg_one(w, align_corners, dtype, dev).view([1, 1, w]).expand(n, h, w)
        ys = _linspace_from_neg_one(h, align_corners, dtype, dev).view([1, h, 1]).expand(n, h, w)
        ones = tensorplay.ones([n, h, w], dtype=dtype, device=dev)
        flat = tensorplay.stack([xs, ys, ones], dim=-1).reshape(n, h * w, 3)
        grid = flat.bmm(theta.transpose(1, 2))
        return grid.reshape(n, h, w, 2)
    elif len(size) == 5:
        n, _, d, h, w = size
        if theta.dim() != 3 or theta.size(0) != n or theta.size(1) != 3 or theta.size(2) != 4:
            raise ValueError(f"affine_grid: expected theta of shape ({n}, 3, 4), got {tuple(theta.shape)}")
        dtype, dev = theta.dtype, theta.device
        xs = _linspace_from_neg_one(w, align_corners, dtype, dev).view([1, 1, 1, w]).expand(n, d, h, w)
        ys = _linspace_from_neg_one(h, align_corners, dtype, dev).view([1, 1, h, 1]).expand(n, d, h, w)
        zs = _linspace_from_neg_one(d, align_corners, dtype, dev).view([1, d, 1, 1]).expand(n, d, h, w)
        ones = tensorplay.ones([n, d, h, w], dtype=dtype, device=dev)
        flat = tensorplay.stack([xs, ys, zs, ones], dim=-1).reshape(n, d * h * w, 4)
        grid = flat.bmm(theta.transpose(1, 2))
        return grid.reshape(n, d, h, w, 3)
    raise ValueError("affine_grid: size must be length 4 or 5")


def _gs_unnormalize(coord, size, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (size - 1)
    return ((coord + 1) * size - 1) / 2


def _gs_adjust(coord, size, padding_mode, align_corners=False):
    if isinstance(padding_mode, str):
        padding_mode = GRID_SAMPLE_PADDING_MODES.index(padding_mode)
    if padding_mode == 1:  # border
        return coord.clamp(0, size - 1)
    if padding_mode == 2:  # reflection
        # reflect_coordinates over [twice_low/2, twice_high/2], then border clip.
        if align_corners:
            twice_low, twice_high = 0, 2 * (size - 1)
        else:
            twice_low, twice_high = -1, 2 * size - 1
        min_ = twice_low / 2
        span = (twice_high - twice_low) / 2
        if twice_low == twice_high:
            return tensorplay.zeros_like(coord)
        c = coord - min_
        c = tensorplay.where(c < 0, -c, c)
        flips = tensorplay.floor(c / span)
        extra = c - flips * span  # == fmod(c, span); sign-safe since c >= 0
        even = (tensorplay.floor(flips / 2) * 2) == flips
        out = tensorplay.where(even, extra + min_, span - extra + min_)
        return out.clamp(0, size - 1)
    return coord


def _grid_sample_gather(input, xs, ys, in_bounds):
    """Gather input planes at integer pixel coords.

    input: (N, C, H, W); xs/ys: (N, Ho, Wo) int64; in_bounds: bool or None.
    Returns (N, C, Ho, Wo) values zeroed outside bounds when requested.
    """
    N, C, H, W = input.shape
    dev = input.device
    i64 = DType.int64
    xs_c = xs.clamp(0, W - 1)
    ys_c = ys.clamp(0, H - 1)
    pos = ys_c * W + xs_c                                  # (N, Ho, Wo)
    if in_bounds is not None:
        pos = tensorplay.where(in_bounds, pos, pos * 0)
    # keep every operand rank-4: our broadcast kernel requires equal ranks.
    bidx = tensorplay.arange(N, dtype=i64, device=dev).view([N, 1, 1, 1]) * (C * H * W)
    cidx = tensorplay.arange(C, dtype=i64, device=dev).view([1, C, 1, 1]) * (H * W)
    gid = bidx + cidx + pos.unsqueeze(1)                   # (N, C, Ho, Wo)
    # NB: use reshape, not view - our current view() binding does not record
    # autograd, silently detaching the graph (reshape does).
    vals = tensorplay.embedding(input.contiguous().reshape(-1), gid.reshape(-1)).reshape(
        [N, C, xs.shape[-2], xs.shape[-1]])
    if in_bounds is not None:
        # expand (not broadcast): our engine lacks the sum-to-shape reduction
        # propagate wrong-shaped grads to non-leaf operands mid-graph.
        vals = vals * in_bounds.unsqueeze(1).expand(
            [N, C, xs.shape[-2], xs.shape[-1]]).to(vals.dtype)
    return vals


def _grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners):
    if isinstance(padding_mode, str):
        padding_mode = GRID_SAMPLE_PADDING_MODES.index(padding_mode)
    N, C, H_in, W_in = input.shape
    H_out, W_out = grid.shape[1], grid.shape[2]
    x = _gs_unnormalize(grid[..., 0], W_in, align_corners)
    y = _gs_unnormalize(grid[..., 1], H_in, align_corners)
    # padding-adjusted individually instead (GridSamplerKernel.cpp
    # Bicubic::get_value_bounded -> compute_coordinates).
    if padding_mode != 0 and interpolation_mode != 2:
        x = _gs_adjust(x, W_in, padding_mode, align_corners)
        y = _gs_adjust(y, H_in, padding_mode, align_corners)

    if interpolation_mode == 1:  # nearest
        xi = tensorplay.round(x).to(DType.int64)
        yi = tensorplay.round(y).to(DType.int64)
        ib = tensorplay.logical_and(
            tensorplay.logical_and(xi >= 0, xi < W_in),
            tensorplay.logical_and(yi >= 0, yi < H_in)) if padding_mode == 0 else None
        vals = _grid_sample_gather(input, xi, yi, ib)
        if x.requires_grad:
            # composite has no such automatic path (embedding backward emits
            # gradients for weights only), so bridge an exact-zero term.
            vals = vals + (x * 0).sum()
        return vals

    x0f = tensorplay.floor(x)
    y0f = tensorplay.floor(y)
    x0 = x0f.to(DType.int64)
    y0 = y0f.to(DType.int64)
    wx = x - x0f
    wy = y - y0f

    def corner_mask(cx, cy):
        return tensorplay.logical_and(
            tensorplay.logical_and(cx >= 0, cx < W_in),
            tensorplay.logical_and(cy >= 0, cy < H_in))

    if interpolation_mode == 0:  # bilinear
        corners = [
            (x0, y0, (1 - wx) * (1 - wy)),
            (x0 + 1, y0, wx * (1 - wy)),
            (x0, y0 + 1, (1 - wx) * wy),
            (x0 + 1, y0 + 1, wx * wy),
        ]
        out = None
        for cx, cy, wgt in corners:
            ib = corner_mask(cx, cy) if padding_mode == 0 else None
            v = _grid_sample_gather(input, cx, cy, ib)
            wgt = wgt.unsqueeze(1).expand([N, C, H_out, W_out]).to(v.dtype)
            term = v * wgt
            out = term if out is None else out + term
        return out

    # bicubic (alpha = -0.75, port of get_cubic_upsample_coefficients)
    def _c1(t_, a=-0.75):
        return ((a + 2) * t_ - (a + 3)) * t_ * t_ + 1

    def _c2(t_, a=-0.75):
        return ((a * t_ - 5 * a) * t_ + 8 * a) * t_ - 4 * a

    tx = x - x0f
    ty = y - y0f
    wxs = [_c2(tx + 1), _c1(tx), _c1(1 - tx), _c2(2 - tx)]
    wys = [_c2(ty + 1), _c1(ty), _c1(1 - ty), _c2(2 - ty)]
    out = None
    for j in range(4):          # rows (y offset j-1)
        for k in range(4):      # cols (x offset k-1)
            cxf = x0f + (k - 1)
            cyf = y0f + (j - 1)
            if padding_mode == 2:
                # reflection: adjust each tap index (border is handled by the
                # clamp inside _grid_sample_gather; zeros masks per tap).
                cxf = _gs_adjust(cxf, W_in, 2, align_corners)
                cyf = _gs_adjust(cyf, H_in, 2, align_corners)
            cx = cxf.to(DType.int64)
            cy = cyf.to(DType.int64)
            ib = corner_mask(cx, cy) if padding_mode == 0 else None
            v = _grid_sample_gather(input, cx, cy, ib)
            wgt = (wxs[k] * wys[j]).unsqueeze(1).expand([N, C, H_out, W_out]).to(v.dtype)
            term = v * wgt
            out = term if out is None else out + term
    return out


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners=None,
) -> Tensor:
    r"""Compute grid sample.

    Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from
    :attr:`grid`. Currently, only spatial (4-D) and volumetric (5-D)
    :attr:`input` are supported.

    Args:
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        mode (str): ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
        padding_mode (str): ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): extrema treatment, default ``False``.

    Dispatches to the native grid_sampler_2d / grid_sampler_3d kernels
    both :attr:`input` and :attr:`grid`.
    """
    if mode not in GRID_SAMPLE_INTERPOLATION_MODES:
        raise ValueError(
            f"nn.functional.grid_sample(): expected mode to be 'bilinear', 'nearest' or 'bicubic', but got: '{mode}'")
    if padding_mode not in GRID_SAMPLE_PADDING_MODES:
        raise ValueError(
            "nn.functional.grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            f"but got: '{padding_mode}'")
    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details.",
            stacklevel=2)
        align_corners = False
    mode_enum = GRID_SAMPLE_INTERPOLATION_MODES.index(mode)
    pad_enum = GRID_SAMPLE_PADDING_MODES.index(padding_mode)

    if input.dim() == 4:
        return _C.grid_sampler_2d(input, grid, mode_enum, pad_enum, align_corners)
    if input.dim() == 5:
        if mode_enum == 2:
            raise ValueError("nn.functional.grid_sample(): bicubic only supports 4D input")
        return _C.grid_sampler_3d(input, grid, mode_enum, pad_enum, align_corners)
    raise ValueError(f"nn.functional.grid_sample(): expected 4D or 5D input, got {input.dim()}D")


def _grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners):
    if isinstance(padding_mode, str):
        padding_mode = GRID_SAMPLE_PADDING_MODES.index(padding_mode)
    N, C, D_in, H_in, W_in = input.shape
    D_out, H_out, W_out = grid.shape[1], grid.shape[2], grid.shape[3]
    x = _gs_unnormalize(grid[..., 0], W_in, align_corners)
    y = _gs_unnormalize(grid[..., 1], H_in, align_corners)
    z = _gs_unnormalize(grid[..., 2], D_in, align_corners)
    if padding_mode != 0:
        x = _gs_adjust(x, W_in, padding_mode, align_corners)
        y = _gs_adjust(y, H_in, padding_mode, align_corners)
        z = _gs_adjust(z, D_in, padding_mode, align_corners)

    def gather3(xi, yi, zi, ib):
        xi_c = xi.clamp(0, W_in - 1)
        yi_c = yi.clamp(0, H_in - 1)
        zi_c = zi.clamp(0, D_in - 1)
        pos = (zi_c * H_in + yi_c) * W_in + xi_c
        if ib is not None:
            pos = tensorplay.where(ib, pos, pos * 0)
        i64 = DType.int64
        dev = input.device
        bidx = tensorplay.arange(N, dtype=i64, device=dev).view([N, 1, 1, 1, 1]) * (C * D_in * H_in * W_in)
        cidx = tensorplay.arange(C, dtype=i64, device=dev).view([1, C, 1, 1, 1]) * (D_in * H_in * W_in)
        gid = bidx + cidx + pos.unsqueeze(1)
        vals = tensorplay.embedding(input.contiguous().reshape(-1), gid.reshape(-1)).reshape(
            [N, C, D_out, H_out, W_out])
        if ib is not None:
            vals = vals * ib.unsqueeze(1).expand([N, C, D_out, H_out, W_out]).to(vals.dtype)
        return vals

    def bounds3(cx, cy, cz):
        m = tensorplay.logical_and(cx >= 0, cx < W_in)
        m = tensorplay.logical_and(m, tensorplay.logical_and(cy >= 0, cy < H_in))
        m = tensorplay.logical_and(m, tensorplay.logical_and(cz >= 0, cz < D_in))
        return m

    xf = tensorplay.floor(x)
    yf = tensorplay.floor(y)
    zf = tensorplay.floor(z)
    x0, y0, z0 = xf.to(DType.int64), yf.to(DType.int64), zf.to(DType.int64)
    wx, wy, wz = x - xf, y - yf, z - zf

    if interpolation_mode == 1:  # nearest
        xi = tensorplay.round(x).to(DType.int64)
        yi = tensorplay.round(y).to(DType.int64)
        zi = tensorplay.round(z).to(DType.int64)
        ib = bounds3(xi, yi, zi) if padding_mode == 0 else None
        vals = gather3(xi, yi, zi, ib)
        if x.requires_grad:
            # d/dgrid; bridge one so grid.grad stays defined.
            vals = vals + (x * 0).sum()
        return vals

    out = None
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                cx, cy, cz = x0 + dx, y0 + dy, z0 + dz
                ib = bounds3(cx, cy, cz) if padding_mode == 0 else None
                v = gather3(cx, cy, cz, ib)
                wgt = (
                    ((wx if dx else 1 - wx) * (wy if dy else 1 - wy) * (wz if dz else 1 - wz))
                    .unsqueeze(1).expand([N, C, D_out, H_out, W_out]).to(v.dtype))
                term = v * wgt
                out = term if out is None else out + term
    return out


# -----------------------------------------------------------------------------
# Distances, embedding bags, SDPA and remaining public surface (F-alignment).
# -----------------------------------------------------------------------------


def _is_float_dtype(dt):
    return dt in (DType.float16, DType.bfloat16, DType.float32, DType.float64)


def _vector_norm(vec, p, keepdim=False):
    """Norm over the last dim of a broadcast difference, matching
"""
    dim = vec.dim() - 1
    if p == float("inf"):
        return _C.max(vec.abs(), dim=dim, keepdim=keepdim)[0]
    if p == -float("inf"):
        return -_C.max((-vec.abs()), dim=dim, keepdim=keepdim)[0]
    if p == 0:
        return vec.ne(0).to(DType.float32).sum(dim=[dim], keepdim=keepdim)
    return _C.norm(vec, [dim], float(p), keepdim)


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> Tensor:
    r"""Computes the pairwise distance between input vectors.

    dimension.
    """
    return _vector_norm(x1 - x2 + eps, p, keepdim)


def pdist(input: Tensor, p: float = 2.0) -> Tensor:
    r"""Computes the pairwise distance between rows of :attr:`input`.

    Returns the flattened upper triangle of the ``N x N`` distance matrix —
    """
    if input.dim() != 2:
        raise RuntimeError(f"pdist expects a 2D input, got {input.dim()}D")
    n = input.size(0)
    diff = input.unsqueeze(1) - input.unsqueeze(0)      # (N, N, D)
    dist = _vector_norm(diff, p, keepdim=False)         # (N, N)
    ri = tensorplay.arange(n, dtype=DType.int64, device=input.device)
    mask = ri.view(-1, 1) < ri.view(1, -1)
    return tensorplay.masked_select(dist, mask)


def _no_grad_embedding_renorm_(weight: Tensor, input, max_norm: float, norm_type: float) -> Tensor:
    """Renorm referenced embedding rows in-place under no_grad (port of
"""
    with tensorplay.no_grad():
        if input.numel() == 0 or weight.numel() == 0:
            return weight
        idx = input.to(DType.int64).reshape(-1)
        e = weight.size(0)
        # referenced-row mask via index_add histogram
        ref = tensorplay.index_add(
            tensorplay.zeros([e], dtype=DType.float32, device=weight.device),
            0, idx, tensorplay.ones(idx.numel(), dtype=DType.float32, device=weight.device),
        )
        norms = _C.norm(weight.contiguous(), [1], float(norm_type), False)  # (E,)
        bad = tensorplay.logical_and(ref > 0, norms > max_norm)
        scale = tensorplay.where(bad, max_norm / (norms + 1e-7), tensorplay.ones_like(norms))
        weight.mul_(scale.view(-1, 1))
    return weight


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets=None,
    max_norm=None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights=None,
    include_last_offset: bool = False,
    padding_idx=None,
) -> Tensor:
    r"""Compute sums, means or maxes of ``bags`` of embeddings.

    1-D inputs with :attr:`offsets` (incl. ``include_last_offset``), fixed
    length 2-D inputs, ``per_sample_weights`` (sum mode), ``padding_idx``
    exclusion and ``max_norm`` renormalization.
    See :class:`tensorplay.nn.EmbeddingBag` for details.
    """
    # Backward compatibility with the old (weight, input) argument order.
    if weight.dtype == DType.int64 and _is_float_dtype(input.dtype):
        warnings.warn(
            "Argument order of nn.functional.embedding_bag was changed. "
            "Usage `embedding_bag(weight, input, ...)` is deprecated, "
            "and should now be `embedding_bag(input, weight, ...)`.",
            stacklevel=2,
        )
        weight, input = input, weight

    if per_sample_weights is not None and tuple(input.shape) != tuple(per_sample_weights.shape):
        raise ValueError(
            f"embedding_bag: If per_sample_weights ({per_sample_weights.shape}) is not None, "
            f"then it must have the same shape as the input ({input.shape})"
        )
    if weight.dim() != 2:
        raise ValueError(f"weight has to be a 2D Tensor, but got Tensor of dimension {weight.dim()}")

    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2
        if scale_grad_by_freq:
            raise ValueError("max mode does not support scaling the gradient by the frequency")
        if sparse:
            raise ValueError("max mode does not support sparse weights")
    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    if per_sample_weights is not None and mode != "sum":
        raise NotImplementedError(
            "embedding_bag: per_sample_weights was not None. "
            "per_sample_weights is only supported for mode='sum' "
            f"(got mode='{mode}').")

    if padding_idx is not None:
        padding_idx = int(padding_idx)
        if padding_idx >= weight.size(0) or padding_idx < -weight.size(0):
            raise ValueError(
                f"padding_idx must be within the number of embeddings ({weight.size(0)}), "
                f"got {padding_idx}")
        if padding_idx < 0:
            padding_idx += weight.size(0)
    else:
        padding_idx = -1

    i64 = DType.int64
    dev = input.device
    numel = input.numel()
    flat = input.to(i64).reshape(-1)

    include_last_offset = bool(include_last_offset)
    if input.dim() == 2:
        if offsets is not None:
            raise ValueError(
                "if input is 2D, then offsets has to be None"
                ", as input is treated is a mini-batch of"
                " fixed length sequences.")
        seq_len = int(input.size(1))
        # Fixed-length bags: one offset per row.  A zero-width row still owns a
        # bag, which arange cannot express with a zero step.
        if seq_len == 0:
            offs = tensorplay.zeros([input.size(0)], dtype=i64, device=dev)
        else:
            offs = tensorplay.arange(0, numel, seq_len, dtype=i64, device=dev)
        include_last_offset = False
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
        offs = offsets.to(i64)
    else:
        raise ValueError(
            f"input has to be 1D or 2D Tensor, but got Tensor of dimension {input.dim()}")

    if per_sample_weights is not None:
        per_sample_weights = per_sample_weights.reshape(-1).to(weight.dtype)

    return _C._embedding_bag(
        weight, flat, offs, scale_grad_by_freq, mode_enum, sparse,
        per_sample_weights, include_last_offset, padding_idx)[0]


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    backend: Optional[str] = None,
) -> Tensor:
    r"""scaled_dot_product_attention(query, key, value, attn_mask=None,
    dropout_p=0.0, is_causal=False, scale=None, backend=None) -> Tensor

    Computes scaled dot product attention on query, key and value. Routes to
    reference:

    .. math::
        \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{E}}) V

    Args:
        backend (str, optional): ``'flash'`` | ``'mem_efficient'`` |
            ``'math'``, or ``None`` to pick automatically. ``'flash'``
            selects the fused flash-attention kernel (impl=1), ``'math'``
            forces the composed reference path; ``'mem_efficient'``,
            this build.
    """
    if attn_mask is not None and is_causal:
        raise AssertionError("Explicit attn_mask should not be set when is_causal=True")
    if backend not in (None, "math", "flash", "mem_efficient"):
        raise ValueError(
            f"scaled_dot_product_attention: unknown backend '{backend}'; "
            "expected 'flash', 'mem_efficient', 'math' or None")
    if backend == "mem_efficient":
        raise NotImplementedError(
            "scaled_dot_product_attention: the mem_efficient backend requires "
            "native memory-efficient attention kernels, which are not yet "
            "available in this build.")

    if backend == "flash":
        if dropout_p != 0.0:
            raise NotImplementedError(
                "scaled_dot_product_attention: the flash backend does not support dropout")
        return tensorplay.scaled_dot_product_attention(query, key, value, is_causal=is_causal, impl=1)

    # Auto-routing: only the plain case hits the fused native kernel; every
    # other combination (and an explicit 'math' request) uses the reference
    # composition below.
    if backend is None and scale is None and attn_mask is None and dropout_p == 0.0:
        return tensorplay.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query * scale_factor @ key.transpose(-2, -1)

    if is_causal:
        ri = tensorplay.arange(L, dtype=DType.int64, device=query.device)
        ci = tensorplay.arange(S, dtype=DType.int64, device=query.device)
        causal_mask = ri.view(L, 1) < ci.view(1, S)
        fill = tensorplay.full([], float("-inf"), dtype=query.dtype, device=query.device)
        attn_weight = tensorplay.where(causal_mask, fill, attn_weight)
    if attn_mask is not None:
        if attn_mask.dtype == DType.bool:
            fill = tensorplay.full([], float("-inf"), dtype=query.dtype, device=query.device)
            keep = tensorplay.where(attn_mask, attn_weight, fill)
            attn_weight = keep
        else:
            attn_weight = attn_weight + attn_mask.to(query.dtype)

    out = softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        keep = (_C.rand(out.shape, device=out.device) >= dropout_p).to(out.dtype) / (1.0 - dropout_p)
        out = out * keep
    return out @ value


def linear_cross_entropy(
    input: Tensor,
    linear_weight: Tensor,
    target: Tensor,
    *,
    linear_bias=None,
    weight=None,
    reduction: str = "mean",
    ignore_index=None,
    label_smoothing: float = 0.0,
    options=None,
) -> Tensor:
    r"""Compute cross entropy between ``input``, transformed linearly, and
    target.

    Equivalent to ``cross_entropy(linear(input, linear_weight), target,
    **kwargs)`` (reference path; chunked/fused options are ignored).
    """
    if options is not None:
        warnings.warn(
            "linear_cross_entropy: ``options`` ignored; reference path used.",
            stacklevel=2,
        )
    logits = linear(input, linear_weight, linear_bias)
    ig = -100 if ignore_index is None else ignore_index
    return cross_entropy(logits, target, weight=weight, ignore_index=ig,
                         reduction=reduction, label_smoothing=label_smoothing)


def grouped_mm(*args, **kwargs):
    raise NotImplementedError(
        "grouped_mm requires native grouped GEMM kernels, which are not yet "
        "available in this build.")


def scaled_grouped_mm(*args, **kwargs):
    raise NotImplementedError(
        "scaled_grouped_mm requires native FP8 grouped GEMM kernels, which "
        "are not yet available in this build.")


def scaled_mm(*args, **kwargs):
    raise NotImplementedError(
        "scaled_mm requires native FP8 GEMM kernels, which are not yet "
        "available in this build.")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _get_softmax_dim(name: str, ndim: int, stacklevel: int = 3) -> int:
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads):
    if query.dim() == 3:
        bsz, tgt_len, embed_dim_to_check = query.shape
        assert query.shape == (bsz, tgt_len, embed_dim_to_check)
        assert key.shape == value.shape
        bsz, src_len, _ = key.shape
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len)
            assert key_padding_mask.dtype == DType.bool
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                assert attn_mask.shape == correct_2d_size
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                assert attn_mask.shape == correct_3d_size
    elif query.dim() == 2:
        assert key.dim() == 2
        assert key.shape == value.shape
        src_len, _ = key.shape
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (src_len,)
            assert key_padding_mask.dtype == DType.bool
        if attn_mask is not None:
            correct_2d_size = (src_len, src_len)
            if attn_mask.dim() == 2:
                assert attn_mask.shape == correct_2d_size
    else:
        raise AssertionError(
            f"query has to be 2d or 3d, but got {query.dim()}d")


def _in_projection(
    q,
    k,
    v,
    w_q=None,
    w_k=None,
    w_v=None,
    b_q=None,
    b_k=None,
    b_v=None,
):
    """
    projections with shape constraints ensuring embedding uniformity."""
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    if w_q.shape != (Eq, Eq):
        raise AssertionError(f"expecting query weights shape of {(Eq, Eq)}, but got {tuple(w_q.shape)}")
    if w_k.shape != (Eq, Ek):
        raise AssertionError(f"expecting key weights shape of {(Eq, Ek)}, but got {tuple(w_k.shape)}")
    if w_v.shape != (Eq, Ev):
        raise AssertionError(f"expecting value weights shape of {(Eq, Ev)}, but got {tuple(w_v.shape)}")
    if b_q is not None and b_q.shape != (Eq,):
        raise AssertionError(f"expecting query bias shape of {(Eq,)}, but got {tuple(b_q.shape)}")
    if b_k is not None and b_k.shape != (Eq,):
        raise AssertionError(f"expecting key bias shape of {(Eq,)}, but got {tuple(b_k.shape)}")
    if b_v is not None and b_v.shape != (Eq,):
        raise AssertionError(f"expecting value bias shape of {(Eq,)}, but got {tuple(b_v.shape)}")
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(q, k, v, w, b=None):
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention: one packed projection. Reshape (not chunk) so an
            # unflatten(-1, (3, E)) validation.
            proj = linear(q, w, b)
            p = proj.reshape(tuple(proj.shape[:-1]) + (3, E))
            return tuple(p.select(-2, i).contiguous() for i in range(3))
        # encoder-decoder attention: packed kv + separate q (split is exact).
        if w.size(0) != 3 * E:
            raise RuntimeError(
                f"in_projection_packed: expected packed weight with 3*E={3 * E} "
                f"rows, got {w.size(0)}")
        w_q = w.narrow(0, 0, E)
        w_kv = w.narrow(0, E, E * 2)
        if b is None:
            b_q = b_kv = None
        else:
            b_q = b.narrow(0, 0, E)
            b_kv = b.narrow(0, E, E * 2)
        q_proj = linear(q, w_q, b_q)
        kv_proj = linear(k, w_kv, b_kv)
        p_k, p_v = tensorplay.chunk(kv_proj, 2, dim=-1)
        return q_proj, p_k.contiguous(), p_v.contiguous()
    # (3, -1), which requires the row count to be divisible by three).
    if w.size(0) % 3 != 0:
        raise RuntimeError(
            f"in_projection_packed: packed weight rows ({w.size(0)}) must be "
            f"divisible by 3")
    w_chunks = tensorplay.chunk(w, 3, dim=0)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_chunks = tensorplay.chunk(b, 3, dim=0)
        b_q, b_k, b_v = b_chunks[0], b_chunks[1], b_chunks[2]
    return (
        linear(q, w_chunks[0], b_q),
        linear(k, w_chunks[1], b_k),
        linear(v, w_chunks[2], b_v),
    )


# -----------------------------------------------------------------------------
# Pooling family (F-alignment).  Native kernels exist for 2-D pooling only;
# 3-D pools decompose into 2-D + 1-D stages (window placement factors per
# Index-returning variants reuse the native/composed values and recover
# -----------------------------------------------------------------------------


def _pool_out_size(in_size, k, s, p, d, ceil_mode):
    eff_k = d * (k - 1) + 1
    if ceil_mode:
        out = int(math.ceil((in_size + 2 * p - eff_k) / s)) + 1
        if (out - 1) * s >= in_size + p:
            out -= 1
    else:
        out = int(math.floor((in_size + 2 * p - eff_k) / s)) + 1
    return max(out, 0)


def _max_pool2d_indices(x4, kernel_size, stride, padding, dilation, oH=None, oW=None):
    """Per-plane linear indices of a 2-D max pool over ``(N, C, H, W)``.

    ``row * W + col``; first occurrence wins ties.
    """
    with tensorplay.no_grad():
        N, C, H, W = x4.shape
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation
        dev = x4.device
        i64 = DType.int64
        if oH is None:
            oH = _pool_out_size(H, kh, sh, ph, dh, False)
        if oW is None:
            oW = _pool_out_size(W, kw, sw, pw, dw, False)

        def _rng(n):
            return tensorplay.arange(n, dtype=i64, device=dev)

        Rm = (_rng(oH) * sh).view(-1, 1, 1, 1) + (_rng(kh) * dh).view(1, 1, -1, 1) - ph
        Cm = (_rng(oW) * sw).view(1, -1, 1, 1) + (_rng(kw) * dw).view(1, 1, 1, -1) - pw
        gR = (Rm + Cm * 0).reshape(1, oH * oW, kh * kw)
        gC = (Cm + Rm * 0).reshape(1, oH * oW, kh * kw)
        valid = tensorplay.logical_and(_band(gR, 0, H), _band(gC, 0, W))

        pos = gR.clamp(0, H - 1) * W + gC.clamp(0, W - 1)
        pos = tensorplay.where(valid, pos, pos * 0)

        P = N * C
        M, K = oH * oW, kh * kw
        base = (_rng(P) * (H * W)).view(P, 1)
        gid = base + pos.reshape(1, M * K)
        vals = tensorplay.embedding(x4.contiguous().reshape(-1), gid).view(P, M, K)
        vals = tensorplay.where(valid, vals, tensorplay.full_like(vals, float("-inf")))
        # argmax over the kernel axis (last of (P, M, K)): one-hot select of
        # the winning offset per output position.
        am = _C.argmax(vals, 2, False)

        kar = _rng(K).view(1, 1, K)
        sel = am.unsqueeze(-1).eq(kar)
        pick_r = tensorplay.where(sel, gR, gR * 0).sum(-1)
        pick_c = tensorplay.where(sel, gC, gC * 0).sum(-1)
        ok = tensorplay.logical_and(_band(pick_r, 0, H), _band(pick_c, 0, W))
        idx = tensorplay.where(ok, pick_r * W + pick_c, pick_r * 0)
        return idx.view(N, C, oH, oW)


def max_pool2d_with_indices(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    return_indices: bool = True,
):
    r"""Applies a 2D max pooling over an input composed of several input
    planes, returning ``(output, indices)``.

    See :class:`~tensorplay.nn.MaxPool2d` for details.
    """
    kernel_size = _pair(kernel_size)
    stride = kernel_size if stride is None else _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    # int64 indices into each (n, c) input plane; autograd flows through
    # max_pool2d_with_indices_backward.
    return _C.max_pool2d_with_indices(input, list(kernel_size), list(stride),
                                      list(padding), list(dilation), ceil_mode)


def max_pool1d_with_indices(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    return_indices: bool = True,
):
    r"""Applies a 1D max pooling over an input signal, returning
    ``(output, indices)``.

    See :class:`~tensorplay.nn.MaxPool1d` for details.
    """
    unbatched = input.dim() == 2
    x = input.unsqueeze(0) if unbatched else input
    k = _single(kernel_size)[0]
    s = k if stride is None else _single(stride)[0]
    p = _single(padding)[0]
    d = _single(dilation)[0]
    values, indices = max_pool2d_with_indices(
        x.unsqueeze(3), (k, 1), (s, 1), (p, 0), (d, 1), ceil_mode
    )
    values = values.squeeze(3)
    indices = indices.squeeze(3)
    if unbatched:
        return values.squeeze(0), indices.squeeze(0)
    return values, indices


def max_pool3d(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    r"""max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
    ceil_mode=False, return_indices=False) -> Tensor

    Applies a 3D max pooling over an input signal composed of several input
    planes. Input shape ``(N, C, D, H, W)`` or unbatched ``(C, D, H, W)``.
    """
    captured = _capture_call(max_pool3d, (input, kernel_size, stride, padding, dilation, ceil_mode, return_indices), {})
    if captured is not None:
        return captured
    if return_indices:
        return max_pool3d_with_indices(
            input, kernel_size, stride=stride, padding=padding,
            dilation=dilation, ceil_mode=ceil_mode)
    kd, kh, kw = _triple(kernel_size)
    sd, sh, sw = _triple(stride) if stride is not None else (kd, kh, kw)
    pd_, ph, pw = _triple(padding)
    dd, dh, dw = _triple(dilation)
    # max_pool3d_backward.
    return _C.max_pool3d(input, [kd, kh, kw], [sd, sh, sw], [pd_, ph, pw],
                         [dd, dh, dw], ceil_mode)


def max_pool3d_with_indices(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode: bool = False,
    return_indices: bool = True,
):
    r"""Applies a 3D max pooling over an input signal, returning
    ``(output, indices)``.

    See :class:`~tensorplay.nn.MaxPool3d` for details.
    """
    kd, kh, kw = _triple(kernel_size)
    sd, sh, sw = _triple(stride) if stride is not None else (kd, kh, kw)
    pd_, ph, pw = _triple(padding)
    dd, dh, dw = _triple(dilation)
    # Native kernel returns values plus int64 indices into each (n, c) input
    # (D, H, W) volume; autograd flows through max_pool3d_with_indices_backward.
    return _C.max_pool3d_with_indices(input, [kd, kh, kw], [sd, sh, sw],
                                      [pd_, ph, pw], [dd, dh, dw], ceil_mode)


def avg_pool3d(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override=None,
) -> Tensor:
    r"""avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
    count_include_pad=True, divisor_override=None) -> Tensor

    Applies a 3D average pooling over an input signal composed of several
    input planes. Input shape ``(N, C, D, H, W)`` or unbatched ``(C, D, H, W)``.
    """
    captured = _capture_call(avg_pool3d, (input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override), {})
    if captured is not None:
        return captured
    unbatched = input.dim() == 4
    x = input.unsqueeze(0) if unbatched else input
    kd, kh, kw = _triple(kernel_size)
    if stride is None:
        sd, sh, sw = kd, kh, kw
    else:
        sd, sh, sw = _triple(stride)
    pd_, ph, pw = _triple(padding)
    output = _C.avg_pool3d(x, (kd, kh, kw), (sd, sh, sw), (pd_, ph, pw),
                            ceil_mode, count_include_pad, divisor_override)
    return output.squeeze(0) if unbatched else output


def adaptive_avg_pool3d(input: Tensor, output_size) -> Tensor:
    r"""Apply a 3D adaptive average pooling over an input signal."""
    unbatched = input.dim() == 4
    x = input.unsqueeze(0) if unbatched else input
    od, oh, ow = _triple(output_size)
    return _C.adaptive_avg_pool3d(x, (od, oh, ow)).squeeze(0) if unbatched \
        else _C.adaptive_avg_pool3d(x, (od, oh, ow))


def lp_pool3d(
    input: Tensor,
    norm_type,
    kernel_size,
    stride=None,
    ceil_mode: bool = False,
) -> Tensor:
    r"""Apply a 3D power-average pooling over an input signal.

    See :class:`~tensorplay.nn.LPPool3d` for details.
    """
    kd, kw, kh = _triple(kernel_size)
    if isinstance(norm_type, (int, float)):
        if norm_type == 0:
            raise ValueError(f"norm_type must be a non-zero value, but got {norm_type}")
        if norm_type == float("inf"):
            return max_pool3d(input.abs(), kernel_size, stride, 0, 1, ceil_mode)
        if norm_type == -float("inf"):
            return -max_pool3d((-input.abs()), kernel_size, stride, 0, 1, ceil_mode)

    if stride is not None:
        out = avg_pool3d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool3d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return (tensorplay.sign(out) * relu(tensorplay.abs(out))).mul(kd * kw * kh).pow(1.0 / norm_type)


def _adaptive_window_bounds(in_size, out_size):
    starts = [i * in_size // out_size for i in range(out_size)]
    ends = [-((-(i + 1) * in_size) // out_size) for i in range(out_size)]
    return starts, ends


def _adaptive_max_pool2d_wi(x4, oH, oW):
    """Values + indices for adaptive 2-D max pooling (loop over cells)."""
    with tensorplay.no_grad():
        N, C, H, W = x4.shape
        hs_list, he_list = _adaptive_window_bounds(H, oH)
        ws_list, we_list = _adaptive_window_bounds(W, oW)
        vals, idxs = [], []
        for i in range(oH):
            for j in range(oW):
                win = x4[:, :, hs_list[i]:he_list[i], ws_list[j]:we_list[j]].contiguous()
                a = he_list[i] - hs_list[i]
                b = we_list[j] - ws_list[j]
                t = win.reshape(N, C, a, b)
                cv, ci = _C.topk(t, 1, -1, True, True, 0)
                cv = cv.view(N, C, a, 1)
                ci = ci.view(N, C, a, 1)
                # topk kernel is last-dim only: swap the window axis to last.
                rv, ri = _C.topk(cv.transpose(3, 2), 1, -1, True, True, 0)
                ri = ri.view(N, C)
                sel = ri.view(N, C, 1, 1).eq(
                    tensorplay.arange(a, dtype=DType.int64,
                                      device=x4.device).view(1, 1, -1, 1))
                col = tensorplay.where(sel, ci, ci * 0).sum(2).view(N, C)
                vals.append(rv.view(N, C))
                idxs.append((hs_list[i] + ri) * W + (ws_list[j] + col))
        v = tensorplay.stack(vals, dim=2).reshape(N, C, oH, oW)
        ix = tensorplay.stack(idxs, dim=2).reshape(N, C, oH, oW)
        return v, ix


def adaptive_max_pool2d_with_indices(input: Tensor, output_size, return_indices: bool = True):
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes, returning ``(output, indices)``.

    See :class:`~tensorplay.nn.AdaptiveMaxPool2d` for details.
    """
    output_size = list(_pair(output_size))
    unbatched = input.dim() == 3
    x = input.unsqueeze(0) if unbatched else input
    values, indices = _C.adaptive_max_pool2d_with_indices(x, output_size)
    if unbatched:
        return values.squeeze(0), indices.squeeze(0)
    return values, indices


def adaptive_max_pool1d_with_indices(input: Tensor, output_size, return_indices: bool = True):
    r"""Applies a 1D adaptive max pooling over an input signal, returning
    ``(output, indices)``.

    See :class:`~tensorplay.nn.AdaptiveMaxPool1d` for details.
    """
    unbatched = input.dim() == 2
    x = input.unsqueeze(0) if unbatched else input
    values, indices = adaptive_max_pool2d_with_indices(x.unsqueeze(3), (output_size, 1))
    values = values.squeeze(3)
    indices = indices.squeeze(3)
    if unbatched:
        return values.squeeze(0), indices.squeeze(0)
    return values, indices


def _adaptive_max_values_3d(x5, od, oh, ow):
    N, C, D, H, W = x5.shape
    hs, he = _adaptive_window_bounds(D, od)
    vs, ixs = [], []
    for d in range(od):
        dsz = he[d] - hs[d]
        sl = x5[:, :, hs[d]:he[d], :, :].reshape(N * C * dsz, 1, H, W)
        pv = _C.adaptive_max_pool2d(sl, [oh, ow]).reshape(N, C, dsz, oh, ow)
        _, pi = _adaptive_max_pool2d_wi(sl.reshape(N * C * dsz, H, W).unsqueeze(1), oh, ow)
        pi = pi.reshape(N, C, dsz, oh, ow)
        # max is associative: reduce the depth window after pooling (H, W).
        vs.append(_C.max(pv, dim=2)[0])
        # argmax kernel mishandles non-last dims: bring dsz to the end.
        zt = pv.transpose(2, 4).contiguous().reshape(N * C * oh * ow, dsz)
        _, zt_idx = _C.topk(zt, 1, -1, True, True, 0)
        # rows are (n, c, w_out, h_out) after the transpose: undo it.
        z = zt_idx.reshape(N, C, ow, oh).transpose(2, 3)
        sel = z.unsqueeze(2).eq(tensorplay.arange(dsz, dtype=DType.int64,
                                                 device=x5.device).view(1, 1, -1, 1, 1))
        win_idx = (pi * sel).sum(2) + (hs[d] + z) * (H * W)
        ixs.append(win_idx)
    return tensorplay.stack(vs, dim=2), tensorplay.stack(ixs, dim=2)


def adaptive_max_pool3d(input: Tensor, output_size, return_indices: bool = False):
    r"""adaptive_max_pool3d(input, output_size, return_indices=False)

    Applies a 3D adaptive max pooling over an input signal composed of several
    input planes. Input shape ``(N, C, D, H, W)`` or unbatched ``(C, D, H, W)``.

    See :class:`~tensorplay.nn.AdaptiveMaxPool3d` for details.
    """
    od, oh, ow = _triple(output_size)
    if return_indices:
        return adaptive_max_pool3d_with_indices(input, (od, oh, ow))
    # through adaptive_max_pool3d_backward.
    return _C.adaptive_max_pool3d(input, [od, oh, ow])


def adaptive_max_pool3d_with_indices(input: Tensor, output_size, return_indices: bool = True):
    r"""Applies a 3D adaptive max pooling over an input signal, returning
    ``(output, indices)``.

    See :class:`~tensorplay.nn.AdaptiveMaxPool3d` for details.
    """
    od, oh, ow = _triple(output_size)
    unbatched = input.dim() == 4
    x = input.unsqueeze(0) if unbatched else input
    N, C, D, H, W = x.shape
    with tensorplay.no_grad():
        _values, indices = _adaptive_max_values_3d(x, od, oh, ow)
    # Values come from the native kernel so autograd flows through
    # adaptive_max_pool3d_backward; indices stay a no-grad int64 tensor.
    values = _C.adaptive_max_pool3d(x, [od, oh, ow])
    if unbatched:
        return values.squeeze(0), indices.squeeze(0)
    return values, indices


# -----------------------------------------------------------------------------
# generate_intervals + FractionalMaxPool{2d,3d}.cpp window scan, vectorized
# through an embedding gather so autograd flows to the input).
# -----------------------------------------------------------------------------


def _frac_generate_intervals(sample, in_size, out_size, pool_size):
    """

    ``sample`` is a ``(P,)`` float tensor; returns ``(P, out_size)`` int64
    window start positions.
    """
    dev = sample.device
    last = in_size - pool_size
    P = sample.numel()
    if out_size <= 1:
        return tensorplay.full([P, max(out_size, 0)], last, dtype=DType.int64, device=dev)
    alpha = float(in_size - pool_size) / float(out_size - 1)
    i = tensorplay.arange(out_size - 1, dtype=sample.dtype, device=dev)
    # static_cast<int> truncates toward zero; operands are non-negative here
    seq = ((i.unsqueeze(0) + sample.unsqueeze(1)) * alpha).to(DType.int64)
    seq = seq - (sample * alpha).to(DType.int64).unsqueeze(1)
    tail = tensorplay.full([P, 1], last, dtype=DType.int64, device=dev)
    return tensorplay.cat([seq, tail], dim=1)


def _frac_pool_check(input, _random_samples, ndim_spatial):
    if _random_samples.dim() != 3:
        raise ValueError(f"Expect _random_samples to have 3 dimensions, got {_random_samples.dim()}")
    nbatch = 1 if input.dim() == ndim_spatial + 1 else input.size(0)
    channels = input.size(0) if input.dim() == ndim_spatial + 1 else input.size(1)
    if _random_samples.size(0) < nbatch:
        raise ValueError("Expect _random_samples.size(0) no less then input batch size.")
    if _random_samples.size(1) != channels:
        raise ValueError("Expect _random_samples.size(1) equals to input channel size.")
    if _random_samples.size(2) != ndim_spatial:
        raise ValueError(f"Expect _random_samples.size(2) equals to {ndim_spatial}; got {_random_samples.size(2)}.")


def _frac_windowed_max(x_planes, pos, plane_size):
    """Max + argmax of windows gathered per-plane through an embedding lookup.

    x_planes: ``(P, V)``; pos: ``(P, M, K)`` int64 within-plane positions.
    ``val > maxVal || isnan(val)`` scan.
    """
    P = x_planes.size(0)
    M, K = pos.shape[1], pos.shape[2]
    base = (tensorplay.arange(P, dtype=DType.int64, device=x_planes.device) * plane_size).view(P, 1)
    gid = base + pos.reshape(P, M * K)
    vals = tensorplay.embedding(x_planes.contiguous().reshape(-1), gid).view(P, M, K)
    sub = tensorplay.where(vals.ne(vals), tensorplay.full_like(vals, float("inf")), vals)
    am = _C.argmax(sub, 2, False)  # window-max index along K
    kar = tensorplay.arange(K, dtype=DType.int64, device=x_planes.device).view(1, 1, K)
    sel = am.unsqueeze(-1).eq(kar)
    picked = tensorplay.where(sel, vals, vals * 0).sum(-1)
    idx = tensorplay.where(sel, pos, pos * 0).sum(-1)
    return picked, idx


def fractional_max_pool2d_with_indices(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = True,
    _random_samples=None,
):
    r"""Applies 2D fractional max pooling over an input signal composed of
    several input planes, returning ``(output, indices)``.

    The max-pooling operation is applied in :math:`kH \times kW` regions by a
    stochastic step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window, ``k`` or ``(kH, kW)``
        output_size: target output size ``oH x oW``
        output_ratio: alternative to output_size, in range (0, 1)
        return_indices: return pooling indices as well
        _random_samples: optional ``(B, C, 2)`` random starts override

    See :class:`~tensorplay.nn.FractionalMaxPool2d` for details.
    """
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool2d requires specifying either an output_size or an output_ratio")
    if output_size is None:
        _output_ratio = _pair(output_ratio)
        output_size = [int(input.size(-2) * _output_ratio[0]), int(input.size(-1) * _output_ratio[1])]
    output_size = list(_pair(output_size))
    kh, kw = _pair(kernel_size)
    oH, oW = int(output_size[0]), int(output_size[1])

    unbatched = input.dim() == 3
    x = input.unsqueeze(0) if unbatched else input
    B, C, H, W = x.shape
    if oH < 1 or oW < 1 or kh < 1 or kw < 1:
        raise ValueError(
            f"fractional_max_pool2d: kernel_size ({kh}, {kw}) and output_size ({oH}, {oW}) must be positive")
    if oH + kh - 1 > H or oW + kw - 1 > W:
        raise ValueError(
            f"fractional_max_pool2d: output_size ({oH}, {oW}) too large relative to "
            f"input ({H}, {W}) and kernel ({kh}, {kw})")

    if _random_samples is None:
        _random_samples = tensorplay.rand(B, C, 2, dtype=input.dtype, device=input.device)
    _frac_pool_check(x, _random_samples, 2)

    # intervals derive from _random_samples, indices are flat in-plane offsets.
    values, indices = _C.fractional_max_pool2d(x, [kh, kw], [oH, oW], _random_samples)
    if unbatched:
        return values.squeeze(0), indices.squeeze(0)
    return values, indices


def _fractional_max_pool2d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
) -> Tensor:
    return fractional_max_pool2d_with_indices(
        input, kernel_size, output_size=output_size, output_ratio=output_ratio,
        return_indices=return_indices, _random_samples=_random_samples)[0]


def fractional_max_pool2d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
):
    r"""Applies 2D fractional max pooling over an input signal.

    If :attr:`return_indices` is ``True``, returns ``(output, indices)``;
    otherwise just the output.
    """
    if return_indices:
        return fractional_max_pool2d_with_indices(
            input, kernel_size, output_size=output_size, output_ratio=output_ratio,
            return_indices=True, _random_samples=_random_samples)
    return _fractional_max_pool2d(
        input, kernel_size, output_size=output_size, output_ratio=output_ratio,
        _random_samples=_random_samples)


def fractional_max_pool3d_with_indices(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = True,
    _random_samples=None,
):
    r"""Applies 3D fractional max pooling over an input signal composed of
    several input planes, returning ``(output, indices)``.

    Each plane consumes three random samples ordered ``(T, H, W)``, matching
    """
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool3d requires specifying either an output_size or an output_ratio")
    if output_size is None:
        _output_ratio = _triple(output_ratio)
        output_size = [
            int(input.size(-3) * _output_ratio[0]),
            int(input.size(-2) * _output_ratio[1]),
            int(input.size(-1) * _output_ratio[2]),
        ]
    output_size = list(_triple(output_size))
    kt, kh, kw = _triple(kernel_size)
    oT, oH, oW = int(output_size[0]), int(output_size[1]), int(output_size[2])

    unbatched = input.dim() == 4
    x = input.unsqueeze(0) if unbatched else input
    B, C, T, H, W = x.shape
    if min(oT, oH, oW) < 1 or min(kt, kh, kw) < 1:
        raise ValueError("fractional_max_pool3d: kernel_size and output_size must be positive")
    if oT + kt - 1 > T or oH + kh - 1 > H or oW + kw - 1 > W:
        raise ValueError(
            f"fractional_max_pool3d: output_size ({oT}, {oH}, {oW}) too large relative to "
            f"input ({T}, {H}, {W}) and kernel ({kt}, {kh}, {kw})")

    if _random_samples is None:
        _random_samples = tensorplay.rand(B, C, 3, dtype=input.dtype, device=input.device)
    _frac_pool_check(x, _random_samples, 3)

    # samples ordered (T, H, W), indices flat in-plane offsets.
    values, indices = _C.fractional_max_pool3d(x, [kt, kh, kw], [oT, oH, oW], _random_samples)
    if unbatched:
        return values.squeeze(0), indices.squeeze(0)
    return values, indices


def _fractional_max_pool3d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
) -> Tensor:
    return fractional_max_pool3d_with_indices(
        input, kernel_size, output_size=output_size, output_ratio=output_ratio,
        return_indices=return_indices, _random_samples=_random_samples)[0]


def fractional_max_pool3d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
):
    r"""Applies 3D fractional max pooling over an input signal.

    If :attr:`return_indices` is ``True``, returns ``(output, indices)``;
    otherwise just the output.
    """
    if return_indices:
        return fractional_max_pool3d_with_indices(
            input, kernel_size, output_size=output_size, output_ratio=output_ratio,
            return_indices=True, _random_samples=_random_samples)
    return _fractional_max_pool3d(
        input, kernel_size, output_size=output_size, output_ratio=output_ratio,
        _random_samples=_random_samples)


# -----------------------------------------------------------------------------
# Max unpooling (partial inverse of max pooling).  Composed as a scatter of
# the pooled values into a zero canvas via the differentiable index_add op.
# -----------------------------------------------------------------------------


def _unpool_output_size(
    input: Tensor,
    kernel_size,
    stride,
    padding,
    output_size,
):
    input_size = input.size()
    n = len(kernel_size)
    default_size = [
        (input_size[-n + d] - 1) * stride[d] + kernel_size[d] - 2 * padding[d]
        for d in range(n)
    ]
    if output_size is None:
        ret = default_size
    else:
        if len(output_size) == n + 2:
            output_size = output_size[2:]
        if len(output_size) != n:
            raise ValueError(
                "output_size should be a sequence containing "
                f"{n} or {n + 2} elements, but it has a length of '{len(output_size)}'"
            )
        ret = list(output_size)
        for d in range(n):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not (min_size < ret[d] < max_size):
                raise ValueError(
                    f'invalid output_size "{output_size}" (dim {d} must be between {min_size} and {max_size})'
                )
    for d in range(n):
        if ret[d] < 0:
            raise ValueError(
                "max_unpooling: output_size must contain non-negative spatial "
                f"dimensions, but got output_size[{d}]={ret[d]}"
            )
    return ret


def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool1d`.

    See :class:`~tensorplay.nn.MaxUnpool1d` for details.
    """
    kernel_size = _single(kernel_size)
    _stride = _single(stride) if stride is not None else kernel_size
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return _C.max_unpool2d(
        input.unsqueeze(-1), indices.unsqueeze(-1), list(output_size) + [1]
    ).squeeze(-1)


def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool2d`.

    See :class:`~tensorplay.nn.MaxUnpool2d` for details.
    """
    kernel_size = _pair(kernel_size)
    _stride = _pair(stride) if stride is not None else kernel_size
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    # pooled values into a zero canvas at the flat in-plane int64 indices.
    return _C.max_unpool2d(input, indices, list(output_size))


def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool3d`.

    See :class:`~tensorplay.nn.MaxUnpool3d` for details.
    """
    kernel_size = _triple(kernel_size)
    _stride = _triple(stride) if stride is not None else kernel_size
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return _C.max_unpool3d(input, indices, list(output_size), list(_stride), list(padding))
