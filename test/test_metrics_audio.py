"""Tests for audio metrics (SNR, SI-SNR, SDR, SI-SDR, PIT)."""

import pytest

import tensorplay as tp
from tensorplay.metrics import (
    PermutationInvariantTraining,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from tensorplay.metrics.functional.audio import (
    permutation_invariant_training,
    pit_permutate,
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)


def test_signal_noise_ratio():
    target = tp.tensor([3.0, -0.5, 2.0, 7.0])
    preds = tp.tensor([2.5, 0.0, 2.0, 8.0])
    assert signal_noise_ratio(preds, target).item() == pytest.approx(16.1805, abs=1e-3)


def test_scale_invariant_signal_noise_ratio():
    target = tp.tensor([3.0, -0.5, 2.0, 7.0])
    preds = tp.tensor([2.5, 0.0, 2.0, 8.0])
    assert scale_invariant_signal_noise_ratio(preds, target).item() == pytest.approx(15.0918, abs=1e-3)


def test_scale_invariant_signal_distortion_ratio():
    target = tp.tensor([3.0, -0.5, 2.0, 7.0])
    preds = tp.tensor([2.5, 0.0, 2.0, 8.0])
    assert scale_invariant_signal_distortion_ratio(preds, target).item() == pytest.approx(18.4030, abs=1e-3)


def test_signal_distortion_ratio():
    preds = tp.randn(2048)
    target = tp.randn(2048)
    value = signal_distortion_ratio(preds, target, filter_length=64)
    assert value.ndim == 0
    assert value.item() != float("inf")


def test_signal_noise_ratio_class():
    metric = SignalNoiseRatio()
    preds = tp.randn(64)
    target = tp.randn(64)
    value = metric(preds, target)
    assert value.ndim == 0
    metric.reset()
    assert metric.total.item() == 0


def test_scale_invariant_signal_noise_ratio_class():
    metric = ScaleInvariantSignalNoiseRatio()
    preds = tp.randn(64)
    target = tp.randn(64)
    assert metric(preds, target).ndim == 0


def test_signal_distortion_ratio_class():
    metric = SignalDistortionRatio(filter_length=64)
    preds = tp.randn(2048)
    target = tp.randn(2048)
    assert metric(preds, target).ndim == 0


def test_scale_invariant_signal_distortion_ratio_class():
    metric = ScaleInvariantSignalDistortionRatio()
    preds = tp.randn(64)
    target = tp.randn(64)
    assert metric(preds, target).ndim == 0


def test_permutation_invariant_training_speaker_wise():
    preds = tp.tensor([[[-0.0579, 0.3560, -0.9604], [-0.1719, 0.3205, 0.2951]]])
    target = tp.tensor([[[1.0958, -0.1648, 0.5228], [-0.4100, 1.1942, -0.5103]]])
    best_metric, best_perm = permutation_invariant_training(
        preds, target, scale_invariant_signal_distortion_ratio, mode="speaker-wise", eval_func="max"
    )
    assert best_metric.shape == (1,)
    assert best_perm.shape == (1, 2)
    assert best_perm.tolist() == [[0, 1]]


def test_permutation_invariant_training_permutation_wise():
    preds = tp.randn(2, 2, 16)
    target = tp.randn(2, 2, 16)
    best_metric, best_perm = permutation_invariant_training(
        preds, target, scale_invariant_signal_distortion_ratio, mode="permutation-wise", eval_func="max"
    )
    assert best_metric.shape == (2,)
    assert best_perm.shape == (2, 2)


def test_pit_permutate():
    preds = tp.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    perm = tp.tensor([[1, 0]])
    out = pit_permutate(preds, perm)
    assert out.tolist() == [[[3.0, 4.0], [1.0, 2.0]]]


def test_pit_invalid_args():
    preds = tp.randn(2, 2, 8)
    target = tp.randn(2, 2, 8)
    with pytest.raises(ValueError, match="eval_func"):
        permutation_invariant_training(preds, target, signal_noise_ratio, eval_func="median")
    with pytest.raises(ValueError, match="mode"):
        permutation_invariant_training(preds, target, signal_noise_ratio, mode="bad")
    with pytest.raises(ValueError, match="Inputs must be of shape"):
        permutation_invariant_training(tp.randn(2), tp.randn(2), signal_noise_ratio)


def test_pit_class():
    metric = PermutationInvariantTraining(scale_invariant_signal_distortion_ratio, eval_func="max")
    preds = tp.randn(2, 2, 32)
    target = tp.randn(2, 2, 32)
    assert metric(preds, target).ndim == 0


def test_pit_permutation_wise_class():
    metric = PermutationInvariantTraining(
        scale_invariant_signal_distortion_ratio, mode="permutation-wise", eval_func="max"
    )
    preds = tp.randn(2, 2, 32)
    target = tp.randn(2, 2, 32)
    assert metric(preds, target).ndim == 0