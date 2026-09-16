"""Observers for post-training quantization calibration.

An Observer records activation/weight ranges during calibration passes and
derives affine Int8 quantization parameters (scale, zero_point) from them,

Usage:
    obs = MinMaxObserver()
    for batch in calibration_data:
        obs(batch)          # or obs.record(batch)
    scale, zero_point = obs.calculate_qparams()
"""

import math
from collections import OrderedDict

import tensorplay
from tensorplay import nn

__all__ = [
    "ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "HistogramObserver",
    "FixedQParamsObserver",
    "PlaceholderObserver",
    "with_args",
    "default_observer",
    "default_weight_observer",
    "default_dynamic_quant_observer",
    "get_observer_state_dict",
    "load_observer_state_dict",
]

# Int8 affine range used across the quantization stack.
QUANT_MIN = -128
QUANT_MAX = 127


class _PartialWrapper:
    """
    observer classes be specialized with constructor arguments while staying
    callable as ``observer_cls(**kwargs)``."""

    def __init__(self, impl, **kwargs):
        self.impl = impl
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        return self.impl(*args, **merged)

    def with_args(self, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        return _PartialWrapper(self.impl, **merged)

    def __repr__(self):
        return f"{self.impl.__name__}({', '.join(f'{k}={v!r}' for k, v in self.kwargs.items())})"


def with_args(**kwargs):
    """Decorator form: ``@with_args(quant_min=0)`` specializes an observer."""
    def decorator(cls_or_fn):
        return _PartialWrapper(cls_or_fn, **kwargs)
    return decorator


def _as_float_tensor(value):
    if isinstance(value, tensorplay.Tensor):
        return value.to(tensorplay.float32)
    return tensorplay.as_tensor(float(value), dtype=tensorplay.float32)


class ObserverBase(nn.Module):
    """Base class: fixed Int8 range, dtype bookkeeping, qparam derivation."""

    def __init__(self, dtype=tensorplay.int8, quant_min=QUANT_MIN,
                 quant_max=QUANT_MAX, eps=None):
        super().__init__()
        self.dtype = dtype
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = 1.1920928955078125e-07 if eps is None else eps

    @classmethod
    def with_args(cls, **kwargs):
        return _PartialWrapper(cls, **kwargs)

    def observation_state(self):
        """Serializable calibration state (tensors / numbers / None).

        Observers keep their statistics as plain attributes rather than
        registered buffers, so ``state_dict()`` cannot see them; this pair
        is what get/load_observer_state_dict persist.
        """
        return {"min_val": self.min_val, "max_val": self.max_val}

    def load_observation_state(self, state):
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]

    @staticmethod
    def _calculate_qparams(min_val, max_val, quant_min=QUANT_MIN,
                           quant_max=QUANT_MAX):
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

    def __init__(self, dtype=tensorplay.int8, quant_min=QUANT_MIN,
                 quant_max=QUANT_MAX, eps=None):
        super().__init__(dtype=dtype, quant_min=quant_min,
                         quant_max=quant_max, eps=eps)
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

    def __init__(self, averaging_constant=0.01, dtype=tensorplay.int8,
                 quant_min=QUANT_MIN, quant_max=QUANT_MAX, eps=None):
        super().__init__(dtype=dtype, quant_min=quant_min,
                         quant_max=quant_max, eps=eps)
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

    def __init__(self, ch_axis=0, dtype=tensorplay.int8, quant_min=QUANT_MIN,
                 quant_max=QUANT_MAX, eps=None):
        super().__init__(dtype=dtype, quant_min=quant_min,
                         quant_max=quant_max, eps=eps)
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

    def observation_state(self):
        return {
            "min_val": None if self.min_val is None
            else tensorplay.as_tensor(self.min_val, dtype=tensorplay.float32),
            "max_val": None if self.max_val is None
            else tensorplay.as_tensor(self.max_val, dtype=tensorplay.float32),
        }

    def load_observation_state(self, state):
        min_val = state["min_val"]
        max_val = state["max_val"]
        self.min_val = None if min_val is None else [float(v) for v in min_val]
        self.max_val = None if max_val is None else [float(v) for v in max_val]


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """Exponential moving average of per-channel min/max values."""

    def __init__(self, averaging_constant=0.01, ch_axis=0,
                 dtype=tensorplay.int8, quant_min=QUANT_MIN,
                 quant_max=QUANT_MAX, eps=None):
        super().__init__(ch_axis=ch_axis, dtype=dtype, quant_min=quant_min,
                         quant_max=quant_max, eps=eps)
        if not 0.0 < averaging_constant <= 1.0:
            raise ValueError("averaging_constant must be in (0, 1]")
        self.averaging_constant = averaging_constant

    def record(self, x):
        axis = self.ch_axis % x.dim()
        mins = []
        maxs = []
        with tensorplay.no_grad():
            for c in range(x.size(axis)):
                channel = x.select(axis, c)
                mins.append(float(channel.min().item()))
                maxs.append(float(channel.max().item()))
        if self.min_val is None:
            self.min_val = mins
            self.max_val = maxs
        else:
            a = self.averaging_constant
            self.min_val = [o + a * (n - o)
                            for o, n in zip(self.min_val, mins)]
            self.max_val = [o + a * (n - o)
                            for o, n in zip(self.max_val, maxs)]
        return x

    __call__ = record


def _histc(x, bins, lo, hi):
    """

    Values outside the range are clamped into the edge bins; a degenerate
    range is widened by an epsilon so every value lands in one bucket.
    """
    width = max(hi - lo, 1e-12)
    bin_width = width / float(bins)
    idx = ((x - lo) / bin_width).floor().clamp(0, bins - 1)
    idx = idx.to(tensorplay.int64)
    return tensorplay.zeros(bins).scatter_add_(
        0, idx, tensorplay.ones_like(idx.to(tensorplay.float32)))


class HistogramObserver(ObserverBase):
    """Running-histogram observer.

    Records a running histogram of incoming values together with the global
    min/max; ``calculate_qparams`` narrows the range with the L2-quantization-
    error search from caffe2's NormMinimization before deriving affine
    parameters, which filters outliers instead of trusting raw extremes.
    """

    def __init__(self, bins=2048, dtype=tensorplay.int8,
                 quant_min=QUANT_MIN, quant_max=QUANT_MAX, eps=None):
        super().__init__(dtype=dtype, quant_min=quant_min,
                         quant_max=quant_max, eps=eps)
        self.bins = int(bins)
        self.histogram = None   # tp tensor of shape [bins]
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.dst_nbins = 256
        self.upsample_rate = 16

    def _get_norm(self, delta_begin, delta_end, density):
        # norm = density * integral_{begin,end} x^2 dx over uniform mass.
        return density * (delta_end ** 3 - delta_begin ** 3) / 3.0

    def reset_histogram(self, x, min_val, max_val):
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.histogram = _histc(x, self.bins, self.min_val, self.max_val)

    def _upscale_histogram(self, histogram, orig_min, orig_max,
                           update_min, update_max):
        rate = self.upsample_rate
        bins = self.bins
        # repeat_interleave equivalent via an index gather: element j lands
        # at positions [j*rate, (j+1)*rate).
        gather = tensorplay.as_tensor(
            [j for j in range(bins) for _ in range(rate)],
            dtype=tensorplay.int64)
        upscaled = histogram.index_select(0, gather) / float(rate)
        fine_bins = bins * rate
        fine_bin_size = (orig_max - orig_min) / fine_bins
        mids = tensorplay.linspace(orig_min, orig_max, fine_bins + 1)[:-1] \
            + 0.5 * fine_bin_size
        boundaries = tensorplay.linspace(update_min, update_max, bins + 1)
        buckets = tensorplay.bucketize(mids, boundaries, right=True)
        buckets = (buckets - 1).clamp(0, bins - 1).to(tensorplay.int64)
        return tensorplay.zeros(bins).scatter_add_(0, buckets, upscaled)

    def _combine_histograms(self, orig_hist, orig_min, orig_max,
                            update_hist, update_min, update_max):
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist
        if orig_min == orig_max:
            total = float(orig_hist.sum())
            transformed = _histc(tensorplay.as_tensor(
                [orig_min]), self.bins, update_min, update_max) * total
            return transformed + update_hist
        if update_min > orig_min or update_max < orig_max:
            raise RuntimeError("HistogramObserver: new range must enclose "
                               "the old range")
        transformed = self._upscale_histogram(
            orig_hist, orig_min, orig_max, update_min, update_max)
        return update_hist + transformed

    def record(self, x):
        if x.numel() == 0:
            return x
        with tensorplay.no_grad():
            x_min = float(x.min().item())
            x_max = float(x.max().item())
            # uselessness while real inputs get clamped at saturation.
            if x_min == -float("inf") or x_max == float("inf"):
                mask = x.abs() != float("inf")
                x = x[mask]
                if x.numel() == 0:
                    return x
                x_min = float(x.min().item())
                x_max = float(x.max().item())

            if self.histogram is None:
                self.reset_histogram(x, x_min, x_max)
                return x

            new_min = min(self.min_val, x_min)
            new_max = max(self.max_val, x_max)
            update_hist = _histc(x, self.bins, new_min, new_max)
            combined = self._combine_histograms(
                self.histogram, self.min_val, self.max_val,
                update_hist, new_min, new_max)
            self.histogram = combined
            self.min_val = new_min
            self.max_val = new_max
        return x

    __call__ = record

    def _compute_quantization_error(self, next_start_bin, next_end_bin):
        bin_width = (self.max_val - self.min_val) / float(self.bins)
        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) \
            / float(self.dst_nbins)
        if dst_bin_width == 0.0:
            return 0.0
        hist_list = self.histogram.tolist()
        norm = 0.0
        mid_norm = self._get_norm(-dst_bin_width / 2.0, dst_bin_width / 2.0,
                                  1.0)
        for j in range(self.bins):
            src_begin = (j - next_start_bin) * bin_width
            src_end = src_begin + bin_width
            density = hist_list[j] / bin_width
            dst_of_begin = min(max(int(src_begin // dst_bin_width),
                                   0), self.dst_nbins - 1)
            begin_center = (dst_of_begin + 0.5) * dst_bin_width
            dst_of_end = min(max(int(src_end // dst_bin_width), 0),
                             self.dst_nbins - 1)
            end_center = dst_of_end * dst_bin_width + dst_bin_width / 2.0
            norm += self._get_norm(src_begin - begin_center,
                                   dst_bin_width / 2.0, density)
            norm += (dst_of_end - dst_of_begin - 1) * mid_norm * density
            norm += self._get_norm(-dst_bin_width / 2.0,
                                   src_end - end_center, density)
        return norm

    def _non_linear_param_search(self):
        """Approximate L2 error minimization over (start_bin, end_bin).

        Follows NormMinimization::NonlinearQuantizationParamsSearch: shrink
        quantile bounds stepwise and keep moving whichever side buys more
        error reduction, stopping once the error starts growing.
        """
        if self.histogram.size(0) != self.bins:
            raise RuntimeError("HistogramObserver: bins mismatch")
        bin_width = (self.max_val - self.min_val) / float(self.bins)

        hist_list = self.histogram.tolist()
        total = sum(hist_list)
        csum = []
        acc = 0.0
        for h in hist_list:
            acc += h
            csum.append(acc)

        stepsize = 1e-5
        alpha, beta = 0.0, 1.0
        start_bin, end_bin = 0, self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize
            left = start_bin
            right = end_bin
            while left < end_bin and csum[left] < next_alpha * total:
                left += 1
            while right > start_bin and csum[right] > next_beta * total:
                right -= 1

            next_start, next_end = start_bin, end_bin
            if (left - start_bin) > (end_bin - right):
                next_start = left
                alpha = next_alpha
            else:
                next_end = right
                beta = next_beta

            if next_start == start_bin and next_end == end_bin:
                continue

            saved_min, saved_max = self.min_val, self.max_val
            norm = self._compute_quantization_error(next_start, next_end)
            self.min_val, self.max_val = saved_min, saved_max

            if norm > norm_min:
                break
            norm_min = norm
            start_bin, end_bin = next_start, next_end

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def calculate_qparams(self):
        if self.histogram is None:
            raise RuntimeError(
                "HistogramObserver has not observed any tensors")
        new_min, new_max = self._non_linear_param_search()
        return self._calculate_qparams(new_min, new_max,
                                       self.quant_min, self.quant_max)

    def observation_state(self):
        return {"histogram": self.histogram,
                "min_val": self.min_val, "max_val": self.max_val}

    def load_observation_state(self, state):
        self.histogram = state["histogram"]
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]

    def reset(self):
        self.histogram = None
        self.min_val = float("inf")
        self.max_val = float("-inf")


class FixedQParamsObserver(ObserverBase):
    """Reports fixed scale/zero_point without observing data; used when the
    quantization parameters are dictated by construction (sigmoid/tanh style
"""

    def __init__(self, scale, zero_point, dtype=tensorplay.int8,
                 quant_min=QUANT_MIN, quant_max=QUANT_MAX):
        super().__init__(dtype=dtype, quant_min=quant_min,
                         quant_max=quant_max)
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def record(self, x):
        return x

    __call__ = record

    def calculate_qparams(self):
        return self.scale, self.zero_point

    def observation_state(self):
        return {"scale": self.scale, "zero_point": self.zero_point}

    def load_observation_state(self, state):
        self.scale = float(state["scale"])
        self.zero_point = int(state["zero_point"])


class PlaceholderObserver(ObserverBase):
    """No-op observer that only carries configuration, e.g. for float16
    "quantization" or dynamic-quantization markers that need no ranges."""

    def __init__(self, dtype=tensorplay.float32, custom_op_name="",
                 quant_min=None, quant_max=None, eps=None):
        super().__init__(dtype=dtype,
                         quant_min=QUANT_MIN if quant_min is None else quant_min,
                         quant_max=QUANT_MAX if quant_max is None else quant_max,
                         eps=eps)
        self.custom_op = custom_op_name

    def record(self, x):
        return x

    __call__ = record

    def calculate_qparams(self):
        raise Exception(
            "calculate_qparams should not be called for PlaceholderObserver")


# Int8 loop (activations unsigned-style range, weights full signed range).
default_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127)
default_weight_observer = MinMaxObserver.with_args(dtype=tensorplay.int8,
                                                   quant_min=-128,
                                                   quant_max=127)
default_dynamic_quant_observer = PlaceholderObserver.with_args(
    dtype=tensorplay.float32)


def _iter_observers(model):
    """Yields (path, observer) for every ObserverBase reachable from the
    module tree — either as a submodule itself or mounted on an attribute
    (e.g. ``QuantStub.fake_quant.observer``)."""
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            yield name, module
            continue
        obs = getattr(module, "observer", None)
        if isinstance(obs, ObserverBase):
            yield f"{name}.observer", obs


def get_observer_state_dict(model):
    """Collects the calibration state of every observer under ``model``,
    keyed by module path — the observer counterpart of ``state_dict()``."""
    od = OrderedDict()
    for name, observer in _iter_observers(model):
        for key, value in observer.observation_state().items():
            od[f"{name}.{key}"] = value
    return od


def load_observer_state_dict(model, obs_dict):
    """Loads observer stats produced by :func:`get_observer_state_dict` back
    into the matching observers."""
    expected = get_observer_state_dict(model)
    missing = sorted(set(expected) - set(obs_dict))
    unexpected = sorted(set(obs_dict) - set(expected))
    for key in missing:
        raise Exception(f"Missing keys for observer {key} in state_dict")
    for key in unexpected:
        raise Exception(f"Unexpected keys for observer {key} in state_dict")

    by_observer = {}
    for key, value in obs_dict.items():
        path, _, param = key.rpartition(".")
        by_observer.setdefault(path, {})[param] = value
    for name, observer in _iter_observers(model):
        if name in by_observer:
            observer.load_observation_state(by_observer[name])
