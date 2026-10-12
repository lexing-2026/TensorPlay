"""Calibration statistics and uniform quantization parameter selection."""
import math
import warnings
from collections import OrderedDict
from functools import partial
import tensorplay
from tensorplay import nn

QUANT_MIN = -128
QUANT_MAX = 127

def is_per_tensor(qscheme):
    return qscheme == tensorplay.per_tensor_affine or qscheme == tensorplay.per_tensor_symmetric

def is_per_channel(qscheme):
    return qscheme in [tensorplay.per_channel_affine, tensorplay.per_channel_affine_float_qparams, tensorplay.per_channel_symmetric]

def check_min_max_valid(min_val, max_val):
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn('must run observer before calling calculate_qparams. ' + 'Returning default values.', stacklevel=2)
        return False
    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float('inf') and max_val == float('-inf'):
            warnings.warn('must run observer before calling calculate_qparams. ' + 'Returning default values.', stacklevel=2)
            return False
        if min_val > max_val:
            raise AssertionError(f'min {min_val} should be less than max {max_val}')
    elif tensorplay.any(min_val > max_val):
        raise AssertionError(f'min {min_val} should be less than max {max_val}')
    return True

def calculate_qmin_qmax(quant_min, quant_max, has_customized_qrange, dtype, reduce_range):
    if has_customized_qrange:
        if dtype in [tensorplay.qint32, tensorplay.int32]:
            initial_quant_min, initial_quant_max = (0, 2 ** 32 - 1)
        else:
            initial_quant_min, initial_quant_max = (0, 255)
        custom_quant_min, custom_quant_max = (quant_min, quant_max)
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (custom_quant_min, custom_quant_max)
        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype in [tensorplay.qint8, tensorplay.int8]:
            if not 0 < qrange_len <= 256:
                raise AssertionError('quantization range should be positive and not exceed the maximum bit range (=256).')
        elif dtype in [tensorplay.qint32, tensorplay.int32]:
            if not 0 < qrange_len <= 2 ** 32:
                raise AssertionError('quantization range should be positive and not exceed the maximum bit range (=4294967296).')
        if reduce_range:
            quant_min, quant_max = (quant_min // 2, quant_max // 2)
    elif dtype in [tensorplay.qint8, tensorplay.int8]:
        if reduce_range:
            quant_min, quant_max = (-64, 63)
        else:
            quant_min, quant_max = (-128, 127)
    elif dtype in [tensorplay.quint8, tensorplay.uint8]:
        if reduce_range:
            quant_min, quant_max = (0, 127)
        else:
            quant_min, quant_max = (0, 255)
    elif dtype in [tensorplay.qint32, tensorplay.int32]:
        quant_min, quant_max = (-1 * 2 ** 31, 2 ** 31 - 1)
    elif dtype == tensorplay.uint16:
        quant_min, quant_max = (0, 2 ** 16 - 1)
    elif dtype == tensorplay.int16:
        quant_min, quant_max = (-2 ** 15, 2 ** 15 - 1)
    else:
        quant_min, quant_max = (0, 15)
    return (quant_min, quant_max)

def validate_qmin_qmax(quant_min, quant_max):
    if not quant_min <= 0 <= quant_max:
        raise AssertionError('Used-specified quantization range must include 0.')
    if quant_min >= quant_max:
        raise AssertionError('qmin must be strictly less than qmax for user-specified quantization range.')

class _PartialWrapper:

    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, arg_name: self.callable_args[arg_name]()}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result

def _with_args(cls_or_self, **kwargs):
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

def _with_callable_args(cls_or_self, **kwargs):
    r = _PartialWrapper(partial(cls_or_self))
    return r.with_callable_args(**kwargs)

class ObserverBase(nn.Module):

    def __init__(self, dtype, is_dynamic=False):
        super().__init__()
        self.dtype = dtype
        self.is_dynamic = is_dynamic

    def forward(self, x):
        pass

    def calculate_qparams(self, **kwargs):
        pass
    with_args = classmethod(_with_args)
    with_callable_args = classmethod(_with_callable_args)

class UniformQuantizationObserverBase(ObserverBase):
    _version = 3

    def __init__(self, dtype=tensorplay.quint8, qscheme=tensorplay.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=tensorplay.finfo(tensorplay.float32).eps, is_dynamic=False, **kwargs):
        factory_kwargs = tensorplay.nn.factory_kwargs(factory_kwargs)
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn('Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of quantization.', stacklevel=2)
        self.reduce_range = reduce_range
        self.register_buffer('eps', tensorplay.tensor([eps], **factory_kwargs))
        if self.qscheme not in (tensorplay.per_tensor_affine, tensorplay.per_tensor_symmetric, tensorplay.per_channel_affine, tensorplay.per_channel_symmetric, tensorplay.per_channel_affine_float_qparams):
            raise AssertionError('Default Observer only works for per_tensor_affine, per_tensor_symmetric, per_channel_affine, per_channel_symmetric and per_channel_float_qparams quantization scheme')
        _ALLOWED_DTYPES = (tensorplay.qint8, tensorplay.quint8, tensorplay.quint4x2, tensorplay.qint32, tensorplay.int8, tensorplay.uint8, tensorplay.int16, tensorplay.int32, tensorplay.float8_e5m2, tensorplay.float8_e4m3fn, tensorplay.uint16)
        if self.dtype not in _ALLOWED_DTYPES:
            raise AssertionError(f'Default Observer only works for {_ALLOWED_DTYPES} data type')
        self.has_customized_qrange = quant_min is not None and quant_max is not None
        if self.has_customized_qrange:
            validate_qmin_qmax(quant_min, quant_max)
        self.quant_min, self.quant_max = calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            eps = tensorplay.tensor([tensorplay.finfo(tensorplay.float32).eps])
            state_dict[prefix + 'eps'] = eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _validate_qmin_qmax(self, quant_min, quant_max):
        if not quant_min <= 0 <= quant_max:
            raise AssertionError('Used-specified quantization range must include 0.')
        if quant_min >= quant_max:
            raise AssertionError('qmin must be strictly less than qmax for user-specified quantization range.')

    def _derive_qparams(self, min_val, max_val):
        if not check_min_max_valid(min_val, max_val):
            return (tensorplay.tensor([1.0], device=min_val.device.type), tensorplay.tensor([0], device=min_val.device.type))
        quant_min, quant_max = (self.quant_min, self.quant_max)
        min_val_neg = tensorplay.minimum(min_val, tensorplay.zeros_like(min_val))
        max_val_pos = tensorplay.maximum(max_val, tensorplay.zeros_like(max_val))
        device = min_val_neg.device
        scale = tensorplay.ones(min_val_neg.size(), dtype=tensorplay.float32, device=device)
        zero_point = tensorplay.zeros(min_val_neg.size(), dtype=tensorplay.int64, device=device)
        if self.qscheme == tensorplay.per_tensor_symmetric or self.qscheme == tensorplay.per_channel_symmetric:
            max_val_pos = tensorplay.maximum(-min_val_neg, max_val_pos)
            scale = _qparam_div(max_val_pos, float(quant_max - quant_min) / 2)
            scale = tensorplay.maximum(scale, self.eps)
            if self.dtype in [tensorplay.quint8, tensorplay.uint8]:
                if self.has_customized_qrange:
                    zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
            elif self.dtype == tensorplay.uint16:
                zero_point = zero_point.new_full(zero_point.size(), 2 ** 15)
        elif self.qscheme == tensorplay.per_channel_affine_float_qparams:
            scale = _qparam_div(max_val - min_val, float(quant_max - quant_min))
            scale = tensorplay.where(scale > self.eps, scale, tensorplay.ones_like(scale))
            zero_point = -1 * min_val / scale
        else:
            scale = _qparam_div(max_val_pos - min_val_neg, float(quant_max - quant_min))
            scale = tensorplay.maximum(scale, self.eps)
            zero_point = quant_min - tensorplay.round(min_val_neg / scale).to(tensorplay.int)
            zero_point = tensorplay.clamp(zero_point, quant_min, quant_max)
        if scale.dim() == 0:
            scale = scale.reshape(1)
        if zero_point.dim() == 0:
            zero_point = zero_point.reshape(1)
        return (scale, zero_point)

    def reset_min_max_vals(self):
        raise NotImplementedError('Cannot reset min/max values in the given observer.')

class MinMaxObserver(UniformQuantizationObserverBase):

    def __init__(self, dtype=tensorplay.quint8, qscheme=tensorplay.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=tensorplay.finfo(tensorplay.float32).eps, is_dynamic=False, **kwargs):
        if not is_per_tensor(qscheme):
            raise NotImplementedError("MinMaxObserver's qscheme only support per_tensor_symmetric                     and per_tensor_affine.")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs, eps=eps, is_dynamic=is_dynamic, **kwargs)
        factory_kwargs = tensorplay.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val', tensorplay.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val', tensorplay.tensor(float('-inf'), **factory_kwargs))
        if self.qscheme == tensorplay.per_tensor_symmetric and self.reduce_range and (self.dtype == tensorplay.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric                                        quantization for quint8')

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = tensorplay.aminmax(x)
        min_val = tensorplay.minimum(min_val_cur, self.min_val)
        max_val = tensorplay.maximum(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def calculate_qparams(self):
        return self._derive_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'

    def reset_min_max_vals(self):
        self.min_val.fill_(float('inf'))
        self.max_val.fill_(float('-inf'))

class MovingAverageMinMaxObserver(MinMaxObserver):

    def __init__(self, averaging_constant=0.01, dtype=tensorplay.quint8, qscheme=tensorplay.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, eps=tensorplay.finfo(tensorplay.float32).eps, is_dynamic=False, **kwargs):
        if not is_per_tensor(qscheme):
            raise NotImplementedError(f"MovingAverageMinMaxObserver's qscheme only support                 per_tensor_symmetric and per_tensor_affine.                 but got: {qscheme}")
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(f"MovingAverageMinMaxObserver doesn't support dynamic quantization for averaging constant of {self.averaging_constant}")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, eps=eps, is_dynamic=is_dynamic, **kwargs)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float('inf') and max_val == float('-inf'):
            min_val, max_val = tensorplay.aminmax(x)
        else:
            min_val_cur, max_val_cur = tensorplay.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

class PerChannelMinMaxObserver(UniformQuantizationObserverBase):

    def __init__(self, ch_axis=0, dtype=tensorplay.quint8, qscheme=tensorplay.per_channel_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=tensorplay.finfo(tensorplay.float32).eps, is_dynamic=False, **kwargs):
        if not is_per_channel(qscheme):
            raise NotImplementedError("PerChannelMinMaxObserver's qscheme only support                     per_channel_symmetric, per_channel_affine and per_channel_affine_float_qparams.")
        if is_dynamic:
            raise NotImplementedError("PerChannelMinMaxObserver doesn't support dynamic quantization")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs, eps=eps, is_dynamic=is_dynamic, **kwargs)
        factory_kwargs = tensorplay.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer('min_val', tensorplay.tensor([], **factory_kwargs))
        self.register_buffer('max_val', tensorplay.tensor([], **factory_kwargs))
        if self.qscheme == tensorplay.per_channel_symmetric and self.reduce_range and (self.dtype == tensorplay.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric quantization for quint8')

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = y.to(self.min_val.dtype)
        y = tensorplay.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = tensorplay.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = tensorplay.aminmax(y, dim=1)
            min_val = tensorplay.minimum(min_val_cur, min_val)
            max_val = tensorplay.maximum(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def calculate_qparams(self):
        return self._derive_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ('min_val', 'max_val'):
            old_name = name.replace('_val', '_vals')
            if prefix + old_name in state_dict and prefix + name not in state_dict:
                state_dict[prefix + name] = state_dict.pop(prefix + old_name)
            if prefix + name in state_dict:
                getattr(self, name).resize_(state_dict[prefix + name].shape)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def reset_min_max_vals(self):
        self.min_val.resize_(0)
        self.max_val.resize_(0)

class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):

    def __init__(self, averaging_constant=0.01, ch_axis=0, dtype=tensorplay.quint8, qscheme=tensorplay.per_channel_affine, reduce_range=False, quant_min=None, quant_max=None, eps=tensorplay.finfo(tensorplay.float32).eps, is_dynamic=False, **kwargs):
        if not is_per_channel(qscheme):
            raise NotImplementedError("MovingAveragePerChannelMinMaxObserver's qscheme only support                     per_channel_symmetric, per_channel_affine and per_channel_affine_float_qparams.")
        if is_dynamic:
            raise NotImplementedError("MovingAveragePerChannelMinMaxObserver doesn't support dynamic quantization")
        super().__init__(ch_axis=ch_axis, dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, eps=eps, is_dynamic=is_dynamic, **kwargs)
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = tensorplay.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = tensorplay.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = tensorplay.aminmax(y, dim=1)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

class HistogramObserver(UniformQuantizationObserverBase):

    def __init__(self, bins=2048, dtype=tensorplay.quint8, qscheme=tensorplay.per_tensor_affine, reduce_range=False, quant_min=None, quant_max=None, factory_kwargs=None, eps=tensorplay.finfo(tensorplay.float32).eps, is_dynamic=False, **kwargs):
        if not is_per_tensor(qscheme):
            raise NotImplementedError("HistogramObserver's qscheme only support per_tensor_symmetric                     and per_tensor_affine.")
        if is_dynamic:
            raise NotImplementedError("HistogramObserver doesn't support dynamic quantization")
        super().__init__(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, quant_min=quant_min, quant_max=quant_max, factory_kwargs=factory_kwargs, eps=eps, is_dynamic=is_dynamic, **kwargs)
        factory_kwargs = tensorplay.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer('histogram', tensorplay.zeros(self.bins, **factory_kwargs))
        self.register_buffer('min_val', tensorplay.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val', tensorplay.tensor(float('-inf'), **factory_kwargs))
        self.dst_nbins = 2 ** (32 if self.dtype in (tensorplay.qint32, tensorplay.int32) else 8 if self.dtype in (tensorplay.qint8, tensorplay.quint8) else tensorplay.iinfo(self.dtype).bits)
        self.upsample_rate = 16

    def _get_norm(self, delta_begin, delta_end, density):
        norm = (delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin, next_end_bin):
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins
        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0
        src_bin = tensorplay.arange(self.bins, device=self.histogram.device)
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width
        dst_bin_of_begin = tensorplay.clamp(tensorplay.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1)
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width
        dst_bin_of_end = tensorplay.clamp(tensorplay.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1)
        density = self.histogram / bin_width
        norm = tensorplay.zeros(self.bins, device=self.histogram.device)
        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin, tensorplay.ones(self.bins, device=self.histogram.device) * delta_end, density)
        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(-dst_bin_width / 2, dst_bin_width / 2, density)
        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2
        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(delta_begin, delta_end, density)
        return norm.sum().item()

    def _non_linear_param_search(self):
        if self.histogram.size()[0] != self.bins:
            raise AssertionError('bins mismatch')
        bin_width = (self.max_val - self.min_val) / self.bins
        total = tensorplay.sum(self.histogram).item()
        cSum = tensorplay.cumsum(self.histogram, dim=0)
        stepsize = 1e-05
        alpha = 0.0
        beta = 1.0
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float('inf')
        while alpha < beta:
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1
            next_start_bin = start_bin
            next_end_bin = end_bin
            if l - start_bin > end_bin - r:
                next_start_bin = l
                alpha = next_alpha
            else:
                next_end_bin = r
                beta = next_beta
            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)
            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin
        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return (new_min, new_max)

    def _upscale_histogram(self, histogram, orig_min, orig_max, update_min, update_max):
        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
        mid_points_histogram = tensorplay.linspace(float(orig_min), float(orig_max), self.bins * self.upsample_rate + 1, device=orig_min.device)[:-1].to(histogram.device) + 0.5 * bin_size
        boundaries_new_histogram = tensorplay.linspace(float(update_min), float(update_max), self.bins + 1, device=update_min.device).to(histogram.device)
        bucket_assignments = tensorplay.bucketize(mid_points_histogram, boundaries_new_histogram, right=True) - 1
        bucket_assignments[bucket_assignments >= self.bins] = self.bins - 1
        bucket_assignments[bucket_assignments < 0] = 0
        update_histogram = tensorplay.bincount(bucket_assignments, weights=histogram, minlength=self.bins)
        return update_histogram

    def _combine_histograms(self, orig_hist, orig_min, orig_max, update_hist, update_min, update_max):
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist
        if orig_min == orig_max:
            bin_value = tensorplay.sum(orig_hist)
            transformed_orig_hist = tensorplay.histc(orig_min, bins=self.bins, min=float(update_min), max=float(update_max)) * bin_value
            return transformed_orig_hist + update_hist
        if update_min > orig_min:
            raise AssertionError('update_min must be <= orig_min')
        if update_max < orig_max:
            raise AssertionError('update_max must be >= orig_max')
        transformed_orig_hist = self._upscale_histogram(orig_hist, orig_min, orig_max, update_min, update_max)
        return update_hist + transformed_orig_hist

    def reset_histogram(self, x, min_val, max_val):
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)
        if min_val.numel() != 1 or max_val.numel() != 1:
            raise AssertionError('histogram min/max values must be scalar.')
        new_histogram = tensorplay.histc(x, self.bins, min=float(min_val), max=float(max_val))
        self.histogram.resize_(new_histogram.shape)
        self.histogram.copy_(new_histogram)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x_min, x_max = tensorplay.aminmax(x)
        if x_min == -tensorplay.inf or x_max == tensorplay.inf:
            warnings.warn('inf detected in input tensor, ignoring input', stacklevel=2)
            x = x[x.abs() != tensorplay.inf]
            if x.numel() == 0:
                return x_orig
            x_min, x_max = tensorplay.aminmax(x)
        current_min = self.min_val
        current_max = self.max_val
        is_uninitialized = self.min_val == float('inf') or self.max_val == float('-inf')
        if is_uninitialized:
            self.reset_histogram(x, x_min, x_max)
        else:
            update_min, update_max = (x_min, x_max)
            new_min = tensorplay.minimum(current_min, update_min)
            new_max = tensorplay.maximum(current_max, update_max)
            new_min, new_max = (new_min.detach(), new_max.detach())
            update_histogram = tensorplay.histc(x, self.bins, min=float(new_min), max=float(new_max)).to(self.histogram.device)
            if new_min == current_min and new_max == current_max:
                combined_histogram = self.histogram + update_histogram
                self.histogram.resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
            else:
                combined_histogram = self._combine_histograms(self.histogram, current_min, current_max, update_histogram, new_min, new_max)
                self.histogram.resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.resize_(new_min.shape)
                self.min_val.copy_(new_min)
                self.max_val.resize_(new_max.shape)
                self.max_val.copy_(new_max)
        return x_orig

    def calculate_qparams(self):
        is_uninitialized = self.min_val == float('inf') and self.max_val == float('-inf')
        if is_uninitialized:
            warnings.warn('must run observer before calling calculate_qparams.                                    Returning default scale and zero point ', stacklevel=2)
            return (tensorplay.tensor([1.0], device=self.min_val.device.type), tensorplay.tensor([0], device=self.min_val.device.type))
        if self.bins != len(self.histogram):
            raise AssertionError('The number of bins in histogram should be equal to the number of bins supplied while making this observer')
        new_min, new_max = self._non_linear_param_search()
        return self._derive_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 3:
            min_val_name, max_val_name = (prefix + 'min_val', prefix + 'max_val')
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == tensorplay.Size([0]):
                    state_dict[min_val_name] = tensorplay.tensor(float('inf'))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == tensorplay.Size([0]):
                    state_dict[max_val_name] = tensorplay.tensor(float('-inf'))
        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'

class FixedQParamsObserver(ObserverBase):

    def __init__(self, scale, zero_point, dtype=tensorplay.quint8, qscheme=tensorplay.per_tensor_affine, quant_min=0, quant_max=255, is_dynamic=False, **kwargs):
        if is_dynamic:
            raise NotImplementedError("FixedQParamsObserver doesn't support dynamic quantization")
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer('scale', tensorplay.tensor([scale], dtype=tensorplay.float))
        self.register_buffer('zero_point', tensorplay.tensor([zero_point], dtype=tensorplay.int))
        self.dtype = dtype
        self.qscheme = qscheme

    def forward(self, X):
        return X

    def calculate_qparams(self):
        return (self.scale, self.zero_point)

class PlaceholderObserver(ObserverBase):

    def __init__(self, dtype=tensorplay.float32, custom_op_name='', compute_dtype=None, quant_min=None, quant_max=None, qscheme=None, eps=None, is_dynamic=False):
        super().__init__(dtype=dtype, is_dynamic=is_dynamic)
        if qscheme is None:
            qscheme = tensorplay.per_tensor_affine
        if eps is None:
            eps = tensorplay.finfo(tensorplay.float32).eps
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.custom_op = custom_op_name
        if compute_dtype:
            is_dynamic = True
            warnings.warn('Please use `is_dynamic` instead of `compute_dtype`.                     `compute_dtype` will be deprecated in a future release                     of quantization.', stacklevel=2)

    def forward(self, x):
        return x

    def extra_repr(self):
        return f'dtype={self.dtype}, is_dynamic={self.is_dynamic}'

    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for PlaceholderObserver')

class RecordingObserver(ObserverBase):
    __annotations__ = {'tensor_val': list[tensorplay.Tensor | None]}

    def __init__(self, dtype=tensorplay.quint8):
        super().__init__(dtype=dtype, is_dynamic=False)
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(x.clone())
        return x

    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for RecordingObserver')

    def get_tensor_value(self):
        return self.tensor_val

class NoopObserver(ObserverBase):

    def __init__(self, dtype=tensorplay.float16, custom_op_name=''):
        super().__init__(dtype=dtype, is_dynamic=False)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for NoopObserver')

class ReuseInputObserver(ObserverBase):

    def __init__(self):
        super().__init__(tensorplay.quint8, is_dynamic=False)

    def forward(self, x):
        return x

    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for ReuseInputObserver')


def with_args(**kwargs):
    return lambda cls: _with_args(cls, **kwargs)


def _record(self, x):
    return self(x)


def _observation_state(self):
    return self.state_dict()


def _load_observation_state(self, state):
    return self.load_state_dict(state)


def _legacy_qparams(min_val, max_val, quant_min=QUANT_MIN, quant_max=QUANT_MAX):
    lo, hi = min(float(min_val), 0.), max(float(max_val), 0.)
    scale = max((hi - lo) / (quant_max - quant_min), 1.1920928955078125e-7)
    return scale, max(quant_min, min(quant_max, quant_min - round(lo / scale)))


ObserverBase.record = _record
ObserverBase.observation_state = _observation_state
ObserverBase.load_observation_state = _load_observation_state
ObserverBase._calculate_qparams = staticmethod(_legacy_qparams)
MinMaxObserver.reset = MinMaxObserver.reset_min_max_vals
PerChannelMinMaxObserver.reset = PerChannelMinMaxObserver.reset_min_max_vals


def _reset_histogram(self):
    self.histogram.zero_()
    self.min_val.fill_(float('inf'))
    self.max_val.fill_(float('-inf'))


HistogramObserver.reset = _reset_histogram

default_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127)
default_weight_observer = MinMaxObserver.with_args(dtype=tensorplay.qint8, qscheme=tensorplay.per_tensor_symmetric)
default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(dtype=tensorplay.qint8, qscheme=tensorplay.per_channel_symmetric)
default_histogram_observer = HistogramObserver.with_args(quant_min=0, quant_max=127)
default_dynamic_quant_observer = PlaceholderObserver.with_args(dtype=tensorplay.quint8, quant_min=0, quant_max=255, is_dynamic=True)
default_float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=tensorplay.quint8, qscheme=tensorplay.per_channel_affine_float_qparams)
default_float_qparams_observer_4bit = PerChannelMinMaxObserver.with_args(dtype=tensorplay.quint4x2, qscheme=tensorplay.per_channel_affine_float_qparams,ch_axis=0)
default_fixed_qparams_range_neg1to1_observer = FixedQParamsObserver.with_args(scale=2./256, zero_point=128)
default_fixed_qparams_range_0to1_observer = FixedQParamsObserver.with_args(scale=1./256, zero_point=0)
default_placeholder_observer = PlaceholderObserver


def get_observer_state_dict(model):
    from .fake_quantize import FakeQuantizeBase
    state = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (ObserverBase, FakeQuantizeBase)):
            module._save_to_state_dict(state, name + '.' if name else '', False)
    return state


def load_observer_state_dict(model, obs_dict):
    expected = get_observer_state_dict(model)
    if set(expected) != set(obs_dict):
        raise RuntimeError('Observer state keys do not match the model')
    state = model.state_dict()
    state.update(obs_dict)
    model.load_state_dict(state)


def _qparam_div(value, denominator):
    if value.device.type == 'cuda':
        return value * (1.0 / denominator)
    return value / denominator
