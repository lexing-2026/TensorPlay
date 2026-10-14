"""Functional interface for the peak signal-to-noise ratio metric."""

from typing import Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities import rank_zero_warn, reduce


def _psnr_compute(
    sum_squared_error: Tensor,
    num_obs: Tensor,
    data_range: Tensor,
    base: float = 10.0,
    reduction: str = "elementwise_mean",
) -> Tensor:
    """Compute the peak signal-to-noise ratio.

    Args:
        sum_squared_error: sum of square of errors over all observations
        num_obs: number of predictions or observations
        data_range: the range of the data. If ``None``, it is determined from the data (max - min).
            ``data_range`` must be given when ``dim`` is not ``None``.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    """
    psnr_base_e = 2 * tp.log(data_range) - tp.log(sum_squared_error / num_obs)
    psnr_vals = psnr_base_e * (10 / tp.log(tp.tensor(base)))
    return reduce(psnr_vals, reduction=reduction)


def _psnr_update(
    preds: Tensor,
    target: Tensor,
    dim: Optional[Union[int, tuple[int, ...]]] = None,
) -> tuple[Tensor, Tensor]:
    """Update and return variables required to compute the peak signal-to-noise ratio.

    Args:
        preds: predicted tensor
        target: ground truth tensor
        dim: dimensions to reduce PSNR scores over, provided as either an integer or a list of integers.
            Default is ``None`` meaning scores will be reduced across all dimensions.

    """
    if not preds.is_floating_point():
        preds = preds.to(tp.float32)
    if not target.is_floating_point():
        target = target.to(tp.float32)

    if dim is None:
        sum_squared_error = tp.sum(tp.pow(preds - target, 2))
        num_obs = tp.tensor(target.numel(), device=target.device)
        return sum_squared_error, num_obs

    diff = preds - target
    sum_squared_error = tp.sum(diff * diff, dim=dim)

    dim_list = [dim] if isinstance(dim, int) else list(dim)
    if not dim_list:
        num_obs = tp.tensor(target.numel(), device=target.device)
    else:
        num_obs = tp.tensor(target.size(), device=target.device)[dim_list].prod()
        num_obs = num_obs.expand_as(sum_squared_error)

    return sum_squared_error, num_obs


def peak_signal_noise_ratio(
    preds: Tensor,
    target: Tensor,
    data_range: Union[float, tuple[float, float]],
    base: float = 10.0,
    reduction: str = "elementwise_mean",
    dim: Optional[Union[int, tuple[int, ...]]] = None,
) -> Tensor:
    """Compute the peak signal-to-noise ratio (PSNR).

    The metric measures the ratio between the maximum possible power of a signal and the power of
    corrupting noise, expressed in a logarithmic scale.

    Args:
        preds: estimated signal
        target: ground truth signal
        data_range: the range of the data. If a tuple is provided then the range is calculated as the
            difference and input is clamped between the values.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        dim: dimensions to reduce PSNR scores over, provided as either an integer or a list of integers.
            Default is ``None`` meaning scores will be reduced across all dimensions.

    Return:
        Tensor with PSNR score

    """
    if dim is None and reduction != "elementwise_mean":
        rank_zero_warn(f"The `reduction={reduction}` will not have any effect when `dim` is None.")

    if isinstance(data_range, tuple):
        preds = tp.clamp(preds, min=data_range[0], max=data_range[1])
        target = tp.clamp(target, min=data_range[0], max=data_range[1])
        data_range_val = tp.tensor(data_range[1] - data_range[0])
    else:
        data_range_val = tp.tensor(float(data_range))

    sum_squared_error, num_obs = _psnr_update(preds, target, dim=dim)
    return _psnr_compute(sum_squared_error, num_obs, data_range_val, base=base, reduction=reduction)