"""Utilities for filling tensors with values drawn from common
initialization distributions."""

import math
import warnings

import tensorplay as tp

__all__ = [
    "calculate_gain",
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "dirac_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
]

_FLOATING_DTYPES = (
    tp.float16,
    tp.bfloat16,
    tp.float32,
    tp.float64,
)


def _cast_scalar_like(value, dtype):
    """Round a python number to the precision of ``dtype`` (floating
    dtypes only); returns the original value otherwise."""
    if dtype in _FLOATING_DTYPES:
        return tp.tensor(value, dtype=dtype).item()
    return value


def _uniform_fill_(tensor, a, b, generator=None):
    """Fill ``tensor`` in-place from U(a, b) without tracking gradients.

    ``generator`` routes the draw through the given generator stream;
    without one the default global stream is used.
    """
    with tp.no_grad():
        return tensor.uniform_(a, b, generator=generator)


def _normal_fill_(tensor, mean, std, generator=None):
    with tp.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity.

    Supported names: linear/conv{1,2,3}d/conv_transpose{1,2,3}d and
    sigmoid map to 1, tanh to 5/3, relu to sqrt(2), selu to 3/4, and
    leaky_relu to sqrt(2 / (1 + negative_slope**2)) where the slope is
    taken from ``param`` (default 0.01).
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
                  'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def uniform_(tensor, a=0.0, b=1.0, generator=None):
    """Fill ``tensor`` in-place with samples from U(a, b)."""
    return _uniform_fill_(tensor, a, b, generator)


def normal_(tensor, mean=0.0, std=1.0, generator=None):
    """Fill ``tensor`` in-place with samples from N(mean, std^2)."""
    return _normal_fill_(tensor, mean, std, generator)


def constant_(tensor, val):
    """Fill ``tensor`` in-place with the scalar ``val``."""
    with tp.no_grad():
        return tensor.fill_(val)


def ones_(tensor):
    """Fill ``tensor`` in-place with the scalar value 1."""
    with tp.no_grad():
        return tensor.fill_(1)


def zeros_(tensor):
    """Fill ``tensor`` in-place with the scalar value 0."""
    with tp.no_grad():
        return tensor.fill_(0)


def eye_(tensor):
    """Fill the 2-D ``tensor`` in-place with the identity matrix.

    As many inputs as possible are preserved through a Linear layer.
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    with tp.no_grad():
        tensor.zero_()
        n = min(tensor.shape[0], tensor.shape[1])
        idx = tp.arange(n, device=tensor.device)
        tensor[idx, idx] = 1
    return tensor


def dirac_(tensor, groups=1):
    """Fill the {3, 4, 5}-D ``tensor`` in-place with Dirac delta kernels.

    Preserves identity of inputs in convolutional layers; with
    ``groups > 1`` each group of output channels preserves identity
    independently.
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.shape
    if sizes[0] % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    with tp.no_grad():
        tensor.zero_()
        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:  # Temporal convolution
                    tensor[g * out_chans_per_grp + d, d, sizes[2] // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[g * out_chans_per_grp + d, d,
                           sizes[2] // 2, sizes[3] // 2] = 1
                else:  # Volumetric convolution
                    tensor[g * out_chans_per_grp + d, d,
                           sizes[2] // 2, sizes[3] // 2, sizes[4] // 2] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = math.prod(tensor.shape[2:]) if tensor.dim() > 2 else 1

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def xavier_uniform_(tensor, gain=1.0, generator=None):
    """Fill ``tensor`` in-place with U(-a, a) where
    a = gain * sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (float(fan_in + fan_out)))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _uniform_fill_(tensor, -a, a, generator)


def xavier_normal_(tensor, gain=1.0, generator=None):
    """Fill ``tensor`` in-place with N(0, std^2) where
    std = gain * sqrt(2 / (fan_in + fan_out))."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (float(fan_in + fan_out)))
    return _normal_fill_(tensor, 0., std, generator)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu',
                     generator=None):
    """Fill ``tensor`` in-place with U(-bound, bound) where
    bound = gain * sqrt(3 / fan)."""
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensor")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _uniform_fill_(tensor, -bound, bound, generator)


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu',
                    generator=None):
    """Fill ``tensor`` in-place with N(0, std^2) where std = gain / sqrt(fan)."""
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensor")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _normal_fill_(tensor, 0, std, generator)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    r"""Fills the input Tensor with values drawn from a truncated normal
    distribution.

    Method is based on the rejection-sampling scheme in
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf —
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Compare against bounds rounded to the tensor's own precision so the
    # rejection mask is consistent with representable values.
    lo = _cast_scalar_like(a, tensor.dtype)
    hi = _cast_scalar_like(b, tensor.dtype)

    with tp.no_grad():
        p = norm_cdf((b - mean) / std) - norm_cdf((a - mean) / std)

        if p > 0.3:
            result = tensor.normal_(mean, std, generator=generator)
            while True:
                mask = (result < lo) | (result > hi)
                if not bool(mask.any().item()):
                    break
                rejected = tp.empty_like(result).normal_(mean, std,
                                                         generator=generator)
                result = tp.where(mask, rejected, result)
            if tensor is not result:
                tensor.copy_(result)
        else:
            mode = max(a, min(mean, b))
            log_peak = -0.5 * ((mode - mean) / std) ** 2

            candidates = tp.empty_like(tensor)
            accept_buf = tp.empty_like(tensor)

            # First iteration: sample directly into tensor.
            _uniform_fill_(tensor, a, b, generator)
            candidates.copy_(tensor)
            candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
            pending = _uniform_fill_(accept_buf, 0.0, 1.0, generator).log_() > candidates
            if not bool(pending.any().item()):
                pass
            else:
                result = tensor
                while True:
                    _uniform_fill_(candidates, a, b, generator)
                    result = tp.where(pending, candidates, result)
                    candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
                    redraw = _uniform_fill_(accept_buf, 0.0, 1.0, generator).log_() > candidates
                    pending = tp.where(pending, redraw, pending)
                    if not bool(pending.any().item()):
                        break
                tensor.copy_(result)

        return tensor


def orthogonal_(tensor, gain=1, generator=None):
    """Fill ``tensor`` in-place with a (semi) orthogonal matrix.

    The tensor must have at least 2 dimensions; trailing dimensions are
    flattened.  Rows (or columns, when narrower) are orthonormalized
    with a QR factorization of a standard-normal sample, and Q is
    rescaled by the diagonal signs of R so its distribution is uniform
    over the orthogonal group.
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    if tensor.numel() == 0:
        return tensor

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tp.empty((rows, cols), dtype=tensor.dtype,
                         device=tensor.device).normal_(0, 1, generator=generator)

    swapped = rows < cols
    work = flattened.t() if swapped else flattened

    q, r = tp.linalg.qr(work)

    dim = min(r.shape[0], r.shape[1])
    ph = tp.diagonal(r).sign()
    q = q * ph

    if swapped:
        q = q.t()

    with tp.no_grad():
        tensor.copy_(q.reshape(tensor.shape))
        tensor.mul_(gain)
    return tensor


def sparse_(tensor, sparsity, std=0.01, generator=None):
    """Fill the 2-D ``tensor`` in-place as a sparse matrix.

    Each column gets ``sparsity * rows`` zeroed entries (a random row
    subset per column); the remaining entries come from N(0, std^2).
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = math.ceil(sparsity * rows)

    with tp.no_grad():
        tensor.normal_(0, std, generator=generator)
        for col_idx in range(cols):
            if generator is None:
                row_indices = tp.randperm(rows)
            else:
                row_indices = tp._C.randperm(rows, generator=generator)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor
