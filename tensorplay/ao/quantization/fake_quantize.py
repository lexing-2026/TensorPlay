"""Stateful simulated quantization with native rounding masks."""

import tensorplay as tp
from tensorplay import nn
from tensorplay.autograd.function import Function
from tensorplay import _C
from .observer import (
    ObserverBase, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
    FixedQParamsObserver, HistogramObserver, _with_args, is_per_channel, QUANT_MIN, QUANT_MAX,
    default_fixed_qparams_range_neg1to1_observer,
    default_fixed_qparams_range_0to1_observer,
)


class _TensorFakeQuant(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        output, mask = _C.fake_quantize_per_tensor_affine_cachemask(
            x, float(scale), int(zero_point), quant_min, quant_max)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad, *_rest):
        return (_C.fake_quantize_per_tensor_affine_cachemask_backward(
            grad, ctx.saved_tensors[0]), None, None, None, None)


class _ChannelFakeQuant(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, axis, quant_min, quant_max):
        output, mask = _C.fake_quantize_per_channel_affine_cachemask(
            x, scale, zero_point, axis, quant_min, quant_max)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad, *_rest):
        return (_C.fake_quantize_per_channel_affine_cachemask_backward(
            grad, ctx.saved_tensors[0]), None, None, None, None, None)


def fake_quantize_per_tensor(x, scale, zero_point, quant_min=QUANT_MIN, quant_max=QUANT_MAX):
    return _TensorFakeQuant.apply(x, scale, zero_point, quant_min, quant_max)


def fake_quantize_per_channel(x, scales, zero_points, axis=0, quant_min=QUANT_MIN, quant_max=QUANT_MAX):
    zp = zero_points.float() if zero_points.dtype.is_floating_point else zero_points.to(tp.int32)
    return _ChannelFakeQuant.apply(x, scales.float(), zp, axis, quant_min, quant_max)


class FakeQuantizeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('fake_quant_enabled', tp.tensor([1], dtype=tp.uint8))
        self.register_buffer('observer_enabled', tp.tensor([1], dtype=tp.uint8))

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled.fill_(int(enabled))

    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    def enable_observer(self, enabled=True):
        self.observer_enabled.fill_(int(enabled))

    def disable_observer(self):
        self.enable_observer(False)

    with_args = classmethod(_with_args)


class FakeQuantize(FakeQuantizeBase):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None,
                 quant_max=None, is_dynamic=False, scale=None, zero_point=None,
                 disable_observer=False, **observer_kwargs):
        super().__init__()
        if observer is None:
            observer = MovingAverageMinMaxObserver
        if quant_min is not None and quant_max is not None:
            observer_kwargs.update(quant_min=quant_min, quant_max=quant_max)
        observer_kwargs['is_dynamic'] = is_dynamic
        self.activation_post_process = (observer if isinstance(observer, ObserverBase)
                                        else observer(**observer_kwargs))
        obs = self.activation_post_process
        self.dtype, self.qscheme = obs.dtype, obs.qscheme
        self.quant_min, self.quant_max = obs.quant_min, obs.quant_max
        self.is_dynamic = obs.is_dynamic
        self.ch_axis = getattr(obs, 'ch_axis', -1)
        self.is_per_channel = is_per_channel(self.qscheme)
        storage_dtype = {tp.qint8: tp.int8, tp.quint8: tp.uint8,
                         tp.qint32: tp.int32}.get(self.dtype, self.dtype)
        limits = tp.iinfo(storage_dtype)
        if self.quant_min < limits.min or self.quant_max > limits.max:
            raise ValueError('Quantization range exceeds the dtype bounds')
        zp_dtype = tp.float32 if self.qscheme == tp.per_channel_affine_float_qparams else tp.int32
        device = next(obs.buffers(), tp.tensor(0)).device
        self.register_buffer('scale', tp.ones(1, dtype=tp.float32, device=device))
        self.register_buffer('zero_point', tp.zeros(1, dtype=zp_dtype, device=device))
        if scale is not None:
            if zero_point is None:
                raise ValueError('Explicit scale requires zero_point')
            self._update_qparams(tp.as_tensor(scale, device=device).reshape(-1),
                                 tp.as_tensor(zero_point, device=device).reshape(-1))
        if disable_observer or scale is not None:
            self.disable_observer()

    @property
    def observer(self):
        return self.activation_post_process

    @property
    def frozen(self):
        return not bool(self.observer_enabled[0])

    @property
    def scales(self):
        return self.scale

    @property
    def zero_points(self):
        return self.zero_point

    def _update_qparams(self, scale, zero_point):
        self.scale.resize_(scale.shape).copy_(scale.to(self.scale.device))
        self.zero_point.resize_(zero_point.shape).copy_(zero_point.to(self.zero_point.device))

    def record(self, x):
        # Boolean assignment is retained for callers using the earlier flag API.
        disabled = self.__dict__.get('disable_observer', False) is True
        if bool(self.observer_enabled[0]) and not disabled:
            self.activation_post_process(x.detach())
            self._update_qparams(*self.calculate_qparams())
        return x

    def freeze(self):
        if bool(self.observer_enabled[0]):
            self._update_qparams(*self.calculate_qparams())
        self.enable_observer(False)

    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, x):
        self.record(x)
        if not bool(self.fake_quant_enabled[0]):
            return x
        if self.is_per_channel:
            return fake_quantize_per_channel(x, self.scale, self.zero_point,
                                             self.ch_axis, self.quant_min, self.quant_max)
        return fake_quantize_per_tensor(x, self.scale, self.zero_point,
                                        self.quant_min, self.quant_max)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name in ('scale', 'zero_point'):
            if prefix + name in state_dict:
                getattr(self, name).resize_(state_dict[prefix + name].shape)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class PerChannelFakeQuantize(FakeQuantize):
    def __init__(self, ch_axis=0, observer=None, scales=None, zero_points=None, **kwargs):
        if observer is None:
            observer = MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=tp.qint8, qscheme=tp.per_channel_symmetric, ch_axis=ch_axis)
        super().__init__(observer=observer, scale=scales, zero_point=zero_points, **kwargs)


class FixedQParamsFakeQuantize(FakeQuantize):
    def __init__(self, observer):
        super().__init__(observer=observer)
        if type(self.activation_post_process) is not FixedQParamsObserver:
            raise ValueError('A fixed parameter observer is required')
        self._update_qparams(*self.activation_post_process.calculate_qparams())

    def calculate_qparams(self):
        return self.scale, self.zero_point


class _FusedFakeQuant(Function):
    @staticmethod
    def forward(ctx, x, observer_on, fake_quant_on, running_min, running_max,
                scale, zero_point, averaging_const, quant_min, quant_max,
                ch_axis, per_channel, symmetric):
        output, mask = _C._fused_moving_avg_obs_fq_helper(
            x, observer_on, fake_quant_on, running_min, running_max, scale,
            zero_point, averaging_const, quant_min, quant_max, ch_axis,
            per_channel, symmetric)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad, *_rest):
        return (_C.fake_quantize_per_tensor_affine_cachemask_backward(
            grad, ctx.saved_tensors[0]),) + (None,) * 12


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **kwargs):
        super().__init__(observer, quant_min, quant_max, **kwargs)
        if not isinstance(self.observer, (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)):
            raise ValueError('A moving average observer is required')
        self.fake_quant_enabled = self.fake_quant_enabled.long()
        self.observer_enabled = self.observer_enabled.long()
        self.is_symmetric_quant = self.qscheme in (tp.per_tensor_symmetric, tp.per_channel_symmetric)

    def forward(self, x):
        return _FusedFakeQuant.apply(
            x, self.observer_enabled, self.fake_quant_enabled,
            self.observer.min_val, self.observer.max_val, self.scale, self.zero_point,
            self.observer.averaging_constant, self.quant_min, self.quant_max,
            self.ch_axis, self.is_per_channel, self.is_symmetric_quant)


def enable_fake_quant(module):
    if isinstance(module, FakeQuantizeBase):
        module.enable_fake_quant()


def disable_fake_quant(module):
    if isinstance(module, FakeQuantizeBase):
        module.disable_fake_quant()


def enable_observer(module):
    if isinstance(module, FakeQuantizeBase):
        module.enable_observer()


def disable_observer(module):
    if isinstance(module, FakeQuantizeBase):
        module.disable_observer()


default_fake_quant = FakeQuantize.with_args(quant_min=0, quant_max=255, reduce_range=True)
default_weight_fake_quant = FakeQuantize.with_args(dtype=tp.qint8, qscheme=tp.per_tensor_symmetric, quant_min=-128, quant_max=127)
default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, dtype=tp.qint8, qscheme=tp.per_channel_symmetric, quant_min=-128, quant_max=127, ch_axis=0)
default_dynamic_fake_quant = FakeQuantize.with_args(dtype=tp.quint8, quant_min=0, quant_max=255, is_dynamic=True, averaging_constant=1)
default_fixed_qparams_range_neg1to1_fake_quant = FixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_neg1to1_observer)
default_fixed_qparams_range_0to1_fake_quant = FixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_0to1_observer)
default_symmetric_fixed_qparams_fake_quant = default_fixed_qparams_range_neg1to1_fake_quant
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant
default_embedding_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, dtype=tp.quint8, qscheme=tp.per_channel_affine_float_qparams, ch_axis=0, averaging_constant=1)
default_embedding_fake_quant_4bit = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, dtype=tp.quint4x2, qscheme=tp.per_channel_affine_float_qparams, ch_axis=0, averaging_constant=1)
default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver, quant_min=0, quant_max=255, dtype=tp.quint8, qscheme=tp.per_tensor_affine, reduce_range=True)

__all__ = [name for name in globals() if name.startswith('default_')] + [
    'FakeQuantizeBase', 'FakeQuantize', 'PerChannelFakeQuantize',
    'FixedQParamsFakeQuantize', 'FusedMovingAvgObsFakeQuantize',
    'fake_quantize_per_tensor', 'fake_quantize_per_channel',
    'enable_fake_quant', 'disable_fake_quant', 'enable_observer', 'disable_observer',
]

default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(quant_min=0, quant_max=255, reduce_range=True)
default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(dtype=tp.qint8, qscheme=tp.per_tensor_symmetric, quant_min=-128, quant_max=127)
default_fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, dtype=tp.qint8, qscheme=tp.per_channel_symmetric, quant_min=-128, quant_max=127, ch_axis=0)
fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-127,
    quant_max=127,
    dtype=tp.qint8,
    qscheme=tp.per_tensor_symmetric,
    eps=2**-12,
)
fused_per_channel_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    quant_min=-127,
    quant_max=127,
    dtype=tp.qint8,
    qscheme=tp.per_channel_symmetric,
    eps=2**-12,
)
__all__ += ['default_fused_act_fake_quant', 'default_fused_wt_fake_quant', 'default_fused_per_channel_wt_fake_quant', 'fused_wt_fake_quant_range_neg_127_to_127', 'fused_per_channel_wt_fake_quant_range_neg_127_to_127']
