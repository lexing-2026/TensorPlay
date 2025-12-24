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
