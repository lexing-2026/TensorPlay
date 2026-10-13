"""Signal-to-noise ratio metrics."""

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from tensorplay.metrics.utilities.checks import _check_same_shape

__all__ = ["signal_noise_ratio", "scale_invariant_signal_noise_ratio"]


def signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """Calculate the signal-to-noise ratio (SNR) for audio signals.

    .. math::
        \\text{SNR} = \\frac{P_{signal}}{P_{noise}}

    where :math:`P` denotes the power of each signal. A high SNR value means
    the audio is clear relative to the background noise.

    Args:
        preds: float tensor with shape ``(..., time)``.
        target: float tensor with shape ``(..., time)``.
        zero_mean: whether to subtract the mean of each signal first.

    Returns:
        float tensor with shape ``(...,)`` of SNR values per sample.
    """
    _check_same_shape(preds, target)
    eps = tp.finfo(preds.dtype).eps

    if zero_mean:
        target = target - tp.mean(target, dim=-1, keepdim=True)
        preds = preds - tp.mean(preds, dim=-1, keepdim=True)

    noise = target - preds

    snr_value = (tp.sum(target**2, dim=-1) + eps) / (tp.sum(noise**2, dim=-1) + eps)
    return 10 * tp.log10(snr_value)


def scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    """Calculate the scale-invariant signal-to-noise ratio (SI-SNR).

    Args:
        preds: float tensor with shape ``(..., time)``.
        target: float tensor with shape ``(..., time)``.

    Returns:
        float tensor with shape ``(...,)`` of SI-SNR values per sample.
    """
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=True)
