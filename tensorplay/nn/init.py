import math
import warnings
import tensorplay as tp

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
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
        return 3.0 / 4  # Value from SNN paper
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def uniform_(tensor, a=0.0, b=1.0):
    with tp.no_grad():
        return tensor.uniform_(a, b)

def normal_(tensor, mean=0.0, std=1.0):
    with tp.no_grad():
        return tensor.normal_(mean, std)

def constant_(tensor, val):
    with tp.no_grad():
        return tensor.fill_(val)

def ones_(tensor):
    with tp.no_grad():
        return tensor.fill_(1)

def zeros_(tensor):
    with tp.no_grad():
        return tensor.fill_(0)

def eye_(tensor):
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
        
    with tp.no_grad():
        # Implementation via copy since eye_ might not be native
        # tp.eye creates a new tensor, we copy it to tensor
        rows, cols = tensor.shape
        tensor.copy_(tp.eye(rows, cols, dtype=tensor.dtype, device=tensor.device))
        return tensor

def dirac_(tensor, groups=1):
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")
    
    sizes = tensor.shape
    min_dim = min(sizes[0], sizes[1])
    with tp.no_grad():
        tensor.zero_()
        for g in range(groups):
            for i in range(min_dim // groups):
                d = i + g * (min_dim // groups)
                if dimensions == 3:  # Temporal convolution
                    tensor[d, d, sizes[2] // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[d, d, sizes[2] // 2, sizes[3] // 2] = 1
                else:  # Volumetric convolution
                    tensor[d, d, sizes[2] // 2, sizes[3] // 2, sizes[4] // 2] = 1
    return tensor

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is available in Python 3.8+
        # receptive_field_size = math.prod(tensor.shape[2:])
        receptive_field_size = 1
        for s in list(tensor.shape)[2:]:
            receptive_field_size *= s
            
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (float(fan_in + fan_out)))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return uniform_(tensor, -a, a)

def xavier_normal_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (float(fan_in + fan_out)))
    return normal_(tensor, 0., std)

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensor")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with tp.no_grad():
        return tensor.uniform_(-bound, bound)

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensor")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with tp.no_grad():
        return tensor.normal_(0, std)

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated normal
    distribution.

    Method is based on the rejection-sampling scheme in
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf —
    a direct port of ``torch.nn.init._no_grad_trunc_normal_`` (torch/nn/init.py),
    which torchvision transformer models use for positional embeddings.
    """
    import math
    import warnings

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with tp.no_grad():
        p = norm_cdf((b - mean) / std) - norm_cdf((a - mean) / std)

        if p > 0.3:
            lo = float(a)
            hi = float(b)
            result = tensor.normal_(mean, std)
            while True:
                mask = (result < lo) | (result > hi)
                if not bool(mask.any().item()):
                    break
                rejected = tensor.empty_like(result).normal_(mean, std)
                result = tp.where(mask, rejected, result)
            if tensor is not result:
                tensor.copy_(result)
        else:
            mode = max(a, min(mean, b))
            log_peak = -0.5 * ((mode - mean) / std) ** 2

            candidates = tensor.empty_like(tensor)
            accept_buf = tensor.empty_like(tensor)

            # First iteration: sample directly into tensor.
            tensor.uniform_(a, b)
            candidates.copy_(tensor)
            candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
            pending = accept_buf.uniform_().log_() > candidates
            if not bool(pending.any().item()):
                pass
            else:
                result = tensor
                while True:
                    candidates.uniform_(a, b)
                    result = tp.where(pending, candidates, result)
                    candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
                    new_pending = accept_buf.uniform_().log_() > candidates
                    pending = tp.where(pending, new_pending, pending)
                    if not bool(pending.any().item()):
                        break
                tensor.copy_(result)

        return tensor

def _qr_reduced(a):
    # torch.nn.init.orthogonal_ relies on ATen's linalg_qr.  TensorPlay has no
    # native QR yet, so this pure-tensor Householder reduction provides the
    # same reduced-QR contract (Q: m x n with orthonormal columns, R: n x n).
    m, n = a.shape
    q = tp.eye(m, m, dtype=a.dtype, device=a.device)
    r = a.clone()
    for k in range(min(n, m - 1)):
        x = r[k:, k]
        norm_x = float(x.norm())
        if norm_x == 0.0:
            continue
        v = x.clone()
        v[0] = v[0] + (norm_x if float(x[0]) >= 0 else -norm_x)
        v_norm = float(v.norm())
        if v_norm == 0.0:
            continue
        v = v / v_norm
        col = v.reshape(-1, 1)
        row = v.reshape(1, -1)
        r[k:, k:] = r[k:, k:] - 2.0 * tp.matmul(col, tp.matmul(row, r[k:, k:]))
        q[:, k:] = q[:, k:] - 2.0 * tp.matmul(tp.matmul(q[:, k:], col), row)
    return q[:, :n], r[:n, :]

def orthogonal_(tensor, gain=1, generator=None):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tp.empty((rows, cols), dtype=tensor.dtype, device=tensor.device).normal_(0, 1)

    swapped = rows < cols
    if swapped:
        flattened = flattened.t().clone()

    q, r = _qr_reduced(flattened)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    dim = min(r.shape[0], r.shape[1])
    ph = tp.tensor([float(r[i, i]) for i in range(dim)]).sign().to(r.dtype, r.device)
    q = q * ph

    if swapped:
        q = q.t()

    with tp.no_grad():
        tensor.copy_(q.reshape(tensor.shape))
        tensor.mul_(gain)
    return tensor

def sparse_(tensor, sparsity, std=0.01, generator=None):
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = math.ceil(sparsity * rows)

    with tp.no_grad():
        tensor.normal_(0, std)
        for col_idx in range(cols):
            row_indices = tp.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor
