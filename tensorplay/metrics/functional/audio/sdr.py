"""Signal-to-distortion ratio metrics."""

import math
from typing import Optional

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.utilities.checks import _check_same_shape

__all__ = [
    "signal_distortion_ratio",
    "scale_invariant_signal_distortion_ratio",
    "source_aggregated_signal_distortion_ratio",
]


def _symmetric_toeplitz(vector: Tensor) -> Tensor:
    """Construct a symmetric Toeplitz matrix from one vector.

    Args:
        vector: shape ``[..., L]``

    Returns:
        a symmetric Toeplitz matrix of shape ``[..., L, L]``
    """
    vec_exp = tp.cat([tp.flip(vector, dims=(-1,)), vector[..., 1:]], dim=-1)
    v_len = vector.shape[-1]
    return tp.as_strided(
        vec_exp, size=(*vec_exp.shape[:-1], v_len, v_len), stride=(*vec_exp.stride()[:-1], 1, 1)
    ).flip(dims=(-1,))


def _compute_autocorr_crosscorr(target: Tensor, preds: Tensor, corr_len: int) -> tuple[Tensor, Tensor]:
    r"""Compute the auto-correlation of ``target`` and cross-correlation of ``target``/``preds``.

    The computation uses the fast Fourier transform. Solving the Toeplitz
    system built from these correlations gives the coordinates of ``preds`` in
    the column space of ``corr_len`` shifts of ``target``.

    Args:
        target: the reference signal of shape ``[..., time]``.
        preds: the estimated signal of shape ``[..., time]``.
        corr_len: length of the auto/cross correlations.

    Returns:
        auto-correlation of ``target`` with shape ``[..., corr_len]`` and
        cross-correlation of ``target``/``preds`` with shape ``[..., corr_len]``.
    """
    # the valid length for the signal after convolution
    n_fft = 2 ** math.ceil(math.log2(preds.shape[-1] + target.shape[-1] - 1))

    # computes the auto correlation of `target`
    t_fft = tp.fft.rfft(target, n=n_fft, dim=-1)
    power = t_fft.real**2 + t_fft.imag**2
    r_0 = tp.fft.irfft(tp.complex(power, tp.zeros_like(power)), n=n_fft)[..., :corr_len]

    # computes the cross-correlation of `target` and `preds`
    p_fft = tp.fft.rfft(preds, n=n_fft, dim=-1)
    b = tp.fft.irfft(t_fft.conj() * p_fft, n=n_fft, dim=-1)[..., :corr_len]

    return r_0, b


def signal_distortion_ratio(
    preds: Tensor,
    target: Tensor,
    use_cg_iter: Optional[int] = None,
    filter_length: int = 512,
    zero_mean: bool = False,
    load_diag: Optional[float] = None,
) -> Tensor:
    """Calculate the signal-to-distortion ratio (SDR).

    Args:
        preds: float tensor with shape ``(..., time)``.
        target: float tensor with shape ``(..., time)``.
        use_cg_iter: if provided, conjugate gradient descent is used to solve
            for the distortion filter coefficients. Requires ``fast-bss-eval``.
        filter_length: length of the distortion filter.
        zero_mean: whether to subtract the mean of all signals first.
        load_diag: if provided, this small value is added to the diagonal
            coefficients of the system matrix when solving for the filter
            coefficients, which can help stabilize the metric.

    Returns:
        float tensor with shape ``(...,)`` of SDR values per sample.
    """
    _check_same_shape(preds, target)

    # use double precision
    preds_dtype = preds.dtype
    preds = preds.double()
    target = target.double()

    if zero_mean:
        preds = preds - preds.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

    # normalize along time-axis to make preds and target have unit norm
    target = target / tp.clamp(tp.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
    preds = preds / tp.clamp(tp.linalg.norm(preds, dim=-1, keepdim=True), min=1e-6)

    # solve for the optimal filter
    # compute auto-correlation and cross-correlation
    r_0, b = _compute_autocorr_crosscorr(target, preds, corr_len=filter_length)

    if load_diag is not None:
        # the diagonal factor of the Toeplitz matrix is the first coefficient of r_0
        r_0[..., 0] += load_diag

    # regular matrix solver
    r = _symmetric_toeplitz(r_0)  # the auto-correlation of the L shifts of `target`
    sol = tp.linalg.solve(r, b)

    # compute the coherence
    coh = tp.einsum("...l,...l->...", b, sol)

    # transform to decibels
    ratio = coh / (1 - coh)
    val = 10.0 * tp.log10(ratio)

    if preds_dtype == tp.float64:
        return val
    return val.float()


def scale_invariant_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """Calculate the scale-invariant signal-to-distortion ratio (SI-SDR).

    Args:
        preds: float tensor with shape ``(..., time)``.
        target: float tensor with shape ``(..., time)``.
        zero_mean: whether to subtract the mean of each signal first.

    Returns:
        float tensor with shape ``(...,)`` of SI-SDR values per sample.
    """
    _check_same_shape(preds, target)
    eps = tp.finfo(preds.dtype).eps

    if zero_mean:
        target = target - tp.mean(target, dim=-1, keepdim=True)
        preds = preds - tp.mean(preds, dim=-1, keepdim=True)

    alpha = (tp.sum(preds * target, dim=-1, keepdim=True) + eps) / (tp.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (tp.sum(target_scaled**2, dim=-1) + eps) / (tp.sum(noise**2, dim=-1) + eps)
    return 10 * tp.log10(val)


def source_aggregated_signal_distortion_ratio(
    preds: Tensor,
    target: Tensor,
    scale_invariant: bool = True,
    zero_mean: bool = False,
) -> Tensor:
    """Calculate the source-aggregated signal-to-distortion ratio (SA-SDR).

    Args:
        preds: float tensor with shape ``(..., spk, time)``.
        target: float tensor with shape ``(..., spk, time)``.
        scale_invariant: whether to scale the targets of different speakers
            with the same factor.
        zero_mean: whether to subtract the mean of each signal first.

    Returns:
        float tensor with shape ``(...)`` of SA-SDR values.
    """
    _check_same_shape(preds, target)
    eps = tp.finfo(preds.dtype).eps

    if zero_mean:
        target = target - tp.mean(target, dim=-1, keepdim=True)
        preds = preds - tp.mean(preds, dim=-1, keepdim=True)

    if scale_invariant:
        alpha = (tp.sum(preds * target, dim=(-2, -1), keepdim=True) + eps) / (
            tp.sum(target**2, dim=(-2, -1), keepdim=True) + eps
        )
        target_scaled = alpha * target
    else:
        alpha = (tp.sum(preds * target, dim=-1, keepdim=True) + eps) / (
            tp.sum(target**2, dim=-1, keepdim=True) + eps
        )
        target_scaled = alpha * target

    noise = target_scaled - preds

    val = (tp.sum(target_scaled**2, dim=(-2, -1)) + eps) / (tp.sum(noise**2, dim=(-2, -1)) + eps)
    return 10 * tp.log10(val)