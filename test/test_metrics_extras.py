"""Tests for the extended metrics: regression extras, classification extras and wrappers."""

import pytest

import tensorplay as tp
import tensorplay.metrics as metrics
from tensorplay.metrics import MetricTracker, Running
from tensorplay.metrics.aggregation import SumMetric
from tensorplay.metrics.collections import MetricCollection
from tensorplay.metrics.functional import classification as fcls
from tensorplay.metrics.functional import regression as freg


# ---------------------------------------------------------------------------
# Regression functional
# ---------------------------------------------------------------------------


def test_mean_absolute_percentage_error():
    preds = tp.tensor([0.9, 15.0, 1.2e6])
    target = tp.tensor([1.0, 10.0, 1e6])
    out = freg.mean_absolute_percentage_error(preds, target)
    assert out.item() == pytest.approx(0.2667, abs=1e-4)


def test_weighted_mean_absolute_percentage_error():
    preds = tp.tensor([0.9, 15.0, 1.2e6])
    target = tp.tensor([1.0, 10.0, 1e6])
    out = freg.weighted_mean_absolute_percentage_error(preds, target)
    expected = (tp.abs(preds - target).sum() / tp.abs(target).sum()).item()
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_cosine_similarity():
    preds = tp.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]])
    target = tp.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    out = freg.cosine_similarity(preds, target, "none")
    assert out.tolist() == pytest.approx([1.0, -1.0], abs=1e-5)
    assert freg.cosine_similarity(preds, target, "mean").item() == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence():
    p = tp.tensor([[0.36, 0.48, 0.16]])
    q = tp.tensor([[1 / 3, 1 / 3, 1 / 3]])
    assert freg.kl_divergence(p, q).item() == pytest.approx(0.0853, abs=1e-4)
    assert freg.kl_divergence(p, q, reduction="sum").item() == pytest.approx(0.0853, abs=1e-4)
    assert freg.kl_divergence(p, q, reduction=None).shape == (1,)


def test_jensen_shannon_divergence():
    p = tp.tensor([[0.36, 0.48, 0.16]])
    q = tp.tensor([[1 / 3, 1 / 3, 1 / 3]])
    assert freg.jensen_shannon_divergence(p, q).item() == pytest.approx(0.0225, abs=1e-4)


def test_log_cosh_error():
    preds = tp.tensor([3.0, 5.0, 2.5, 7.0])
    target = tp.tensor([2.5, 5.0, 4.0, 8.0])
    assert freg.log_cosh_error(preds, target).item() == pytest.approx(0.3523, abs=1e-4)

    preds2 = tp.tensor([[3.0, 5.0, 1.2], [-2.1, 2.5, 7.0]])
    target2 = tp.tensor([[2.5, 5.0, 1.3], [0.3, 4.0, 8.0]])
    assert freg.log_cosh_error(preds2, target2).tolist() == pytest.approx([0.9176, 0.4277, 0.2194], abs=1e-4)


def test_minkowski_distance():
    preds = tp.tensor([1.0, 2.8, 3.5, 4.5])
    target = tp.tensor([6.1, 2.11, 3.1, 5.6])
    assert freg.minkowski_distance(preds, target, p=3).item() == pytest.approx(5.1220, abs=1e-4)


def test_concordance_corrcoef():
    preds = tp.tensor([2.5, 0.0, 2.0, 8.0])
    target = tp.tensor([3.0, -0.5, 2.0, 7.0])
    assert freg.concordance_corrcoef(preds, target).item() == pytest.approx(0.9777, abs=1e-4)

    preds2 = tp.tensor([[2.5, 0.0], [2.0, 8.0]])
    target2 = tp.tensor([[3.0, -0.5], [2.0, 7.0]])
    assert freg.concordance_corrcoef(preds2, target2).tolist() == pytest.approx([0.7273, 0.9887], abs=1e-4)


# ---------------------------------------------------------------------------
# Regression classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_cls,kwargs,preds,target,expected",
    [
        (metrics.regression.MeanAbsolutePercentageError, {}, [0.9, 15.0, 1.2e6], [1.0, 10.0, 1e6], 0.2667),
        (metrics.regression.WeightedMeanAbsolutePercentageError, {}, [0.9, 15.0, 1.2e6], [1.0, 10.0, 1e6], 0.2000),
        (metrics.regression.LogCoshError, {}, [3.0, 5.0, 2.5, 7.0], [2.5, 5.0, 4.0, 8.0], 0.3523),
        (metrics.regression.MinkowskiDistance, {"p": 3}, [1.0, 2.8, 3.5, 4.5], [6.1, 2.11, 3.1, 5.6], 5.1220),
        (metrics.regression.ConcordanceCorrCoef, {}, [2.5, 0.0, 2.0, 8.0], [3.0, -0.5, 2.0, 7.0], 0.9777),
    ],
)
def test_regression_class_metrics(metric_cls, kwargs, preds, target, expected):
    metric = metric_cls(**kwargs)
    metric(tp.tensor(preds), tp.tensor(target))
    assert metric.compute().item() == pytest.approx(expected, abs=1e-4)


def test_cosine_similarity_class():
    metric = metrics.regression.CosineSimilarity(reduction="none")
    preds = tp.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]])
    target = tp.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    metric.update(preds, target)
    assert metric.compute().tolist() == pytest.approx([1.0, -1.0], abs=1e-5)


def test_kl_divergence_class():
    metric = metrics.regression.KLDivergence()
    p = tp.tensor([[0.36, 0.48, 0.16]])
    q = tp.tensor([[1 / 3, 1 / 3, 1 / 3]])
    metric(p, q)
    assert metric.compute().item() == pytest.approx(0.0853, abs=1e-4)


def test_js_divergence_class():
    metric = metrics.regression.JensenShannonDivergence()
    p = tp.tensor([[0.36, 0.48, 0.16]])
    q = tp.tensor([[1 / 3, 1 / 3, 1 / 3]])
    metric(p, q)
    assert metric.compute().item() == pytest.approx(0.0225, abs=1e-4)


# ---------------------------------------------------------------------------
# Classification functional
# ---------------------------------------------------------------------------


def test_binary_jaccard_index():
    preds = tp.tensor([0, 1, 0, 0])
    target = tp.tensor([1, 1, 0, 0])
    assert fcls.binary_jaccard_index(preds, target).item() == pytest.approx(0.5)


def test_multiclass_jaccard_index():
    preds = tp.tensor([2, 1, 0, 1])
    target = tp.tensor([2, 1, 0, 0])
    assert fcls.multiclass_jaccard_index(preds, target, 3).item() == pytest.approx(0.6667, abs=1e-4)
    assert fcls.multiclass_jaccard_index(preds, target, 3, average=None).tolist() == pytest.approx(
        [0.5, 0.5, 1.0], abs=1e-6
    )
    assert fcls.multiclass_jaccard_index(preds, target, 3, average="micro").item() == pytest.approx(0.6, abs=1e-6)
    assert fcls.multiclass_jaccard_index(preds, target, 3, average="weighted").item() == pytest.approx(0.625, abs=1e-6)


def test_multilabel_jaccard_index():
    preds = tp.tensor([[0, 0, 1], [1, 0, 1]])
    target = tp.tensor([[0, 1, 0], [1, 0, 1]])
    assert fcls.multilabel_jaccard_index(preds, target, 3).item() == pytest.approx(0.5)
    assert fcls.multilabel_jaccard_index(preds, target, 3, average=None).tolist() == pytest.approx(
        [1.0, 0.0, 0.5], abs=1e-6
    )


def test_binary_hinge_loss():
    preds = tp.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
    target = tp.tensor([0, 0, 1, 1, 1])
    assert fcls.binary_hinge_loss(preds, target).item() == pytest.approx(0.6900, abs=1e-4)
    assert fcls.binary_hinge_loss(preds, target, squared=True).item() == pytest.approx(0.6905, abs=1e-4)


def test_multiclass_hinge_loss():
    preds = tp.tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
    target = tp.tensor([0, 1, 2, 0])
    assert fcls.multiclass_hinge_loss(preds, target, 3).item() == pytest.approx(0.9125, abs=1e-4)
    assert fcls.multiclass_hinge_loss(preds, target, 3, squared=True).item() == pytest.approx(1.1131, abs=1e-4)
    assert fcls.multiclass_hinge_loss(preds, target, 3, multiclass_mode="one-vs-all").tolist() == pytest.approx(
        [0.8750, 1.1250, 1.1000], abs=1e-4
    )


def test_binary_calibration_error():
    preds = tp.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
    target = tp.tensor([0, 0, 1, 1, 1])
    assert fcls.binary_calibration_error(preds, target, n_bins=2, norm="l1").item() == pytest.approx(0.2900, abs=1e-4)
    assert fcls.binary_calibration_error(preds, target, n_bins=2, norm="l2").item() == pytest.approx(0.2918, abs=1e-4)
    assert fcls.binary_calibration_error(preds, target, n_bins=2, norm="max").item() == pytest.approx(0.3167, abs=1e-4)


def test_multiclass_calibration_error():
    preds = tp.tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
    target = tp.tensor([0, 1, 2, 0])
    assert fcls.multiclass_calibration_error(preds, target, 3, n_bins=3, norm="l1").item() == pytest.approx(
        0.2000, abs=1e-4
    )
    assert fcls.multiclass_calibration_error(preds, target, 3, n_bins=3, norm="l2").item() == pytest.approx(
        0.2082, abs=1e-4
    )
    assert fcls.multiclass_calibration_error(preds, target, 3, n_bins=3, norm="max").item() == pytest.approx(
        0.2333, abs=1e-4
    )


def test_binary_specificity():
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    assert fcls.binary_specificity(preds, target).item() == pytest.approx(0.6667, abs=1e-4)


def test_multiclass_specificity():
    preds = tp.tensor([2, 1, 0, 1])
    target = tp.tensor([2, 1, 0, 0])
    assert fcls.multiclass_specificity(preds, target, 3).item() == pytest.approx(0.8889, abs=1e-4)
    assert fcls.multiclass_specificity(preds, target, 3, average=None).tolist() == pytest.approx(
        [1.0, 0.6667, 1.0], abs=1e-4
    )
    assert fcls.multiclass_specificity(preds, target, 3, average="micro").item() == pytest.approx(0.875, abs=1e-6)
    assert fcls.multiclass_specificity(preds, target, 3, average="weighted").item() == pytest.approx(
        0.9167, abs=1e-4
    )


def test_multilabel_specificity():
    preds = tp.tensor([[0, 0, 1], [1, 0, 1]])
    target = tp.tensor([[0, 1, 0], [1, 0, 1]])
    assert fcls.multilabel_specificity(preds, target, 3).item() == pytest.approx(0.6667, abs=1e-4)
    assert fcls.multilabel_specificity(preds, target, 3, average=None).tolist() == pytest.approx(
        [1.0, 1.0, 0.0], abs=1e-6
    )
    assert fcls.multilabel_specificity(preds, target, 3, average="micro").item() == pytest.approx(
        2 / 3, abs=1e-6
    )


def test_specificity_samplewise():
    preds = tp.tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.78], [0.45, 0.37]]])
    target = tp.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
    out = fcls.binary_specificity(preds, target, multidim_average="samplewise")
    assert out.tolist() == pytest.approx([0.0, 0.3333], abs=1e-4)


# ---------------------------------------------------------------------------
# Classification classes
# ---------------------------------------------------------------------------


def test_jaccard_class():
    metric = metrics.classification.BinaryJaccardIndex()
    metric(tp.tensor([0, 1, 0, 0]), tp.tensor([1, 1, 0, 0]))
    assert metric.compute().item() == pytest.approx(0.5)

    metric = metrics.classification.MulticlassJaccardIndex(num_classes=3)
    metric(tp.tensor([2, 1, 0, 1]), tp.tensor([2, 1, 0, 0]))
    assert metric.compute().item() == pytest.approx(0.6667, abs=1e-4)


def test_hinge_class():
    metric = metrics.classification.BinaryHingeLoss()
    metric(tp.tensor([0.25, 0.25, 0.55, 0.75, 0.75]), tp.tensor([0, 0, 1, 1, 1]))
    assert metric.compute().item() == pytest.approx(0.6900, abs=1e-4)

    metric = metrics.classification.MulticlassHingeLoss(num_classes=3)
    preds = tp.tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
    metric(preds, tp.tensor([0, 1, 2, 0]))
    assert metric.compute().item() == pytest.approx(0.9125, abs=1e-4)


def test_calibration_error_class():
    metric = metrics.classification.BinaryCalibrationError(n_bins=2)
    metric(tp.tensor([0.25, 0.25, 0.55, 0.75, 0.75]), tp.tensor([0, 0, 1, 1, 1]))
    assert metric.compute().item() == pytest.approx(0.2900, abs=1e-4)

    metric = metrics.classification.MulticlassCalibrationError(num_classes=3, n_bins=3)
    preds = tp.tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
    metric(preds, tp.tensor([0, 1, 2, 0]))
    assert metric.compute().item() == pytest.approx(0.2000, abs=1e-4)


def test_specificity_class():
    metric = metrics.classification.MulticlassSpecificity(num_classes=3)
    metric(tp.tensor([2, 1, 0, 1]), tp.tensor([2, 1, 0, 0]))
    assert metric.compute().item() == pytest.approx(0.8889, abs=1e-4)

    metric = metrics.classification.MultilabelSpecificity(num_labels=3)
    metric(tp.tensor([[0, 0, 1], [1, 0, 1]]), tp.tensor([[0, 1, 0], [1, 0, 1]]))
    assert metric.compute().item() == pytest.approx(0.6667, abs=1e-4)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


def test_metric_tracker_single():
    tracker = MetricTracker(metrics.classification.MulticlassAccuracy(num_classes=10, average="micro"), maximize=True)
    for _ in range(3):
        tracker.increment()
        for _ in range(3):
            tracker.update(tp.randint(0, 10, (100,)), tp.randint(0, 10, (100,)))
        tracker.compute()
    assert tracker.n_steps == 3
    all_vals = tracker.compute_all()
    assert all_vals.shape == (3,)
    best, step = tracker.best_metric(return_step=True)
    assert 0 <= best <= 1
    assert step in (0, 1, 2)


def test_metric_tracker_collection():
    tracker = MetricTracker(
        MetricCollection([metrics.regression.MeanSquaredError(), metrics.regression.ExplainedVariance()]),
        maximize=[False, True],
    )
    for _ in range(3):
        tracker.increment()
        for _ in range(3):
            tracker.update(tp.randn(100), tp.randn(100))
        tracker.compute()
    res = tracker.compute_all()
    assert set(res.keys()) == {"MeanSquaredError", "ExplainedVariance"}
    best, steps = tracker.best_metric(return_step=True)
    assert set(best.keys()) == {"MeanSquaredError", "ExplainedVariance"}
    assert set(steps.keys()) == {"MeanSquaredError", "ExplainedVariance"}


def test_metric_tracker_requires_increment():
    tracker = MetricTracker(metrics.regression.MeanSquaredError())
    with pytest.raises(ValueError, match="increment"):
        tracker.update(tp.tensor([1.0]), tp.tensor([2.0]))


def test_running():
    metric = Running(SumMetric(), window=3)
    expected = [0.0, 1.0, 3.0, 6.0, 9.0, 12.0]
    for i in range(6):
        metric(tp.tensor([i]))
        assert metric.compute().item() == pytest.approx(expected[i])


def test_running_requires_full_state_update_false():
    with pytest.raises(ValueError, match="full_state_update"):
        Running(metrics.regression.PearsonCorrCoef(), window=3)


# ---------------------------------------------------------------------------
# Operator-level regressions (unique flat on ND, sum on 0-d)
# ---------------------------------------------------------------------------


def test_unique_flattens_nd():
    x = tp.tensor([[1, 2, 1], [3, 2, 0]])
    assert tp.unique(x).tolist() == [0, 1, 2, 3]


def test_sum_0d_with_dim():
    x = tp.tensor(5.0)
    assert x.sum(dim=0).item() == pytest.approx(5.0)
    assert x.sum(dim=-1).item() == pytest.approx(5.0)
    assert x.sum(dim=0, keepdim=True).shape == (1,)
    assert tp.mean(x, dim=0).item() == pytest.approx(5.0)


def test_where_single_arg():
    x = tp.tensor([0, 1, 0, 2])
    idx = tp.where(x)
    assert isinstance(idx, tuple)
    assert len(idx) == 1
    assert idx[0].tolist() == [1, 3]
    # 2D input: one index tensor per dimension
    y = tp.tensor([[0, 1], [2, 0]])
    idx2 = tp.where(y)
    assert len(idx2) == 2
    assert idx2[0].tolist() == [0, 1]
    assert idx2[1].tolist() == [1, 0]