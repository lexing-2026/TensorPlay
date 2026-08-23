"""Observers for post-training quantization calibration.

An Observer records activation/weight ranges during calibration passes and
derives affine Int8 quantization parameters (scale, zero_point) from them,
mirroring the essentials of ``torch.ao.quantization.observer``.

Usage:
    obs = MinMaxObserver()
    for batch in calibration_data:
        obs(batch)          # or obs.record(batch)
    scale, zero_point = obs.calculate_qparams()
"""

import math

import tensorplay
from tensorplay import nn

__all__ = [
    "ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
]

# Int8 affine range used across the quantization stack.
QUANT_MIN = -128
QUANT_MAX = 127


def _as_float_tensor(value):
    if isinstance(value, tensorplay.Tensor):
        return value.to(tensorplay.float32)
    return tensorplay.as_tensor(float(value), dtype=tensorplay.float32)


class ObserverBase(nn.Module):
    """Base class: fixed Int8 range, dtype bookkeeping, qparam derivation."""

    def __init__(self, dtype=tensorplay.int8):
        super().__init__()
        self.dtype = dtype
        self.quant_min = QUANT_MIN
        self.quant_max = QUANT_MAX

    @staticmethod
    def _calculate_qparams(min_val, max_val, quant_min=QUANT_MIN,
                           quant_max=QUANT_MAX):
        """Affine parameters from observed range (torch's _calculate_qparams)."""
        min_val = float(min_val)
        max_val = float(max_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)
        # Include zero in the range so that 0 maps exactly to zero_point.
        span = max_val - min_val
        if span == 0.0:
            scale = 1.0
        else:
            scale = span / float(quant_max - quant_min)
        # Guard against a degenerate scale (empty/degenerate range).
        if not math.isfinite(scale) or scale == 0.0:
            scale = 1.0
        zero_point = int(round(quant_min - min_val / scale))
        zero_point = max(quant_min, min(quant_max, zero_point))
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    """Tracks the running min/max of observed tensors; per-tensor params."""

    def __init__(self, dtype=tensorplay.int8):
        super().__init__(dtype=dtype)
        self.min_val = None
        self.max_val = None

    def record(self, x):
        with tensorplay.no_grad():
            current_min = float(x.min().item())
            current_max = float(x.max().item())
        if self.min_val is None:
            self.min_val = current_min
            self.max_val = current_max
        else:
            self.min_val = min(self.min_val, current_min)
            self.max_val = max(self.max_val, current_max)
        return x

    def reset(self):
        self.min_val = None
        self.max_val = None

    __call__ = record

    def calculate_qparams(self):
        if self.min_val is None:
            raise RuntimeError(
                "MinMaxObserver has not observed any tensors; run "
                "calibration data through it before calculate_qparams()")
        return self._calculate_qparams(self.min_val, self.max_val,
                                       self.quant_min, self.quant_max)


class MovingAverageMinMaxObserver(ObserverBase):
    """Exponential moving average of min/max, as used for QAT-style
    calibration on streamed data."""

    def __init__(self, averaging_constant=0.01, dtype=tensorplay.int8):
        super().__init__(dtype=dtype)
        if not 0.0 < averaging_constant <= 1.0:
            raise ValueError("averaging_constant must be in (0, 1]")
        self.averaging_constant = averaging_constant
        self.min_val = None
        self.max_val = None

    def record(self, x):
        with tensorplay.no_grad():
            current_min = float(x.min().item())
            current_max = float(x.max().item())
        if self.min_val is None:
            self.min_val = current_min
            self.max_val = current_max
        else:
            c = self.averaging_constant
            self.min_val = (1 - c) * self.min_val + c * current_min
            self.max_val = (1 - c) * self.max_val + c * current_max
        return x

    def reset(self):
        self.min_val = None
        self.max_val = None

    __call__ = record

    def calculate_qparams(self):
        if self.min_val is None:
            raise RuntimeError(
                "MovingAverageMinMaxObserver has not observed any tensors")
        return self._calculate_qparams(self.min_val, self.max_val,
                                       self.quant_min, self.quant_max)


class PerChannelMinMaxObserver(ObserverBase):
    """Running per-channel min/max along ``ch_axis``; returns per-channel
    scale/zero_point tensors suitable for quantize_per_channel."""

    def __init__(self, ch_axis=0, dtype=tensorplay.int8):
        super().__init__(dtype=dtype)
        self.ch_axis = ch_axis
        self.min_val = None  # list of floats, one per channel
        self.max_val = None

    def record(self, x):
        axis = self.ch_axis % x.dim()
        mins = []
        maxs = []
        with tensorplay.no_grad():
            for c in range(x.size(axis)):
                # select() drops the channel dim; min()/max() reduce the rest.
                channel = x.select(axis, c)
                mins.append(float(channel.min().item()))
                maxs.append(float(channel.max().item()))
        if self.min_val is None:
            self.min_val = mins
            self.max_val = maxs
        else:
            self.min_val = [min(a, b) for a, b in zip(self.min_val, mins)]
            self.max_val = [max(a, b) for a, b in zip(self.max_val, maxs)]
        return x

    def reset(self):
        self.min_val = None
        self.max_val = None

    __call__ = record

    def calculate_qparams(self):
        if self.min_val is None:
            raise RuntimeError("PerChannelMinMaxObserver has not observed tensors")
        scales = []
        zero_points = []
        for lo, hi in zip(self.min_val, self.max_val):
            s, z = self._calculate_qparams(lo, hi, self.quant_min, self.quant_max)
            scales.append(s)
            zero_points.append(z)
        return (
            tensorplay.as_tensor(scales, dtype=tensorplay.float32),
            tensorplay.as_tensor(zero_points, dtype=tensorplay.int64),
        )
