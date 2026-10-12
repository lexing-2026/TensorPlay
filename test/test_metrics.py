"""Tests for ``tensorplay.metrics``.

Expected values are hand-computed / taken from the reference metric library's
doctests so the tests are self-contained (no external metric package needed).
"""
import pytest

import tensorplay as tp
import tensorplay.metrics as metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_sum_metric():
    m = metrics.SumMetric()
    m.update(1)
    m.update(tp.tensor([2, 3]))
    assert m.compute().item() == pytest.approx(6.0)


def test_mean_metric():
    m = metrics.MeanMetric()
    m.update(1)
    m.update(tp.tensor([2, 3]))
    assert m.compute().item() == pytest.approx(2.0)


def test_max_min_metric():
    mx = metrics.MaxMetric()
    mx.update(1)
    mx.update(tp.tensor([2, 3]))
    assert mx.compute().item() == pytest.approx(3.0)

    mn = metrics.MinMetric()
    mn.update(1)
    mn.update(tp.tensor([2, 3]))
    assert mn.compute().item() == pytest.approx(1.0)


def test_cat_metric():
    m = metrics.CatMetric()
    m.update(1)
    m.update(tp.tensor([2, 3]))
    assert m.compute().tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Classification functional
# ---------------------------------------------------------------------------
def test_binary_accuracy_int():
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    out = metrics.functional.classification.binary_accuracy(preds, target)
    assert out.item() == pytest.approx(0.6667, abs=1e-4)


def test_binary_accuracy_float():
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    preds = tp.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
    out = metrics.functional.classification.binary_accuracy(preds, target)
    assert out.item() == pytest.approx(0.6667, abs=1e-4)


def test_multiclass_accuracy():
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([2, 1, 0, 1])
    out = metrics.functional.classification.multiclass_accuracy(preds, target, num_classes=3)
    assert out.item() == pytest.approx(0.8333, abs=1e-4)
    out_per_class = metrics.functional.classification.multiclass_accuracy(preds, target, num_classes=3, average=None)
    assert out_per_class.tolist() == pytest.approx([0.5, 1.0, 1.0], abs=1e-4)


def test_multiclass_accuracy_float():
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
    out = metrics.functional.classification.multiclass_accuracy(preds, target, num_classes=3)
    assert out.item() == pytest.approx(0.8333, abs=1e-4)


def test_multilabel_accuracy():
    target = tp.tensor([[0, 1, 0], [1, 0, 1]])
    preds = tp.tensor([[0, 0, 1], [1, 0, 1]])
    out = metrics.functional.classification.multilabel_accuracy(preds, target, num_labels=3)
    assert out.item() == pytest.approx(0.6667, abs=1e-4)


def test_binary_f1():
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    out = metrics.functional.classification.binary_f1_score(preds, target)
    assert out.item() == pytest.approx(0.6667, abs=1e-4)


def test_multiclass_f1():
    target = tp.tensor([0, 1, 2, 0, 1, 2])
    preds = tp.tensor([0, 2, 1, 0, 0, 1])
    out = metrics.functional.classification.multiclass_f1_score(preds, target, num_classes=3)
    assert out.item() == pytest.approx(0.3333, abs=1e-4)


def test_binary_stat_scores():
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    out = metrics.functional.classification.binary_stat_scores(preds, target)
    assert out.tolist() == [2, 1, 2, 1, 3]


def test_multiclass_stat_scores():
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([2, 1, 0, 1])
    out = metrics.functional.classification.multiclass_stat_scores(preds, target, num_classes=3, average="micro")
    assert out.tolist() == [3, 1, 7, 1, 4]
    out_none = metrics.functional.classification.multiclass_stat_scores(preds, target, num_classes=3, average=None)
    assert out_none.tolist() == [[1, 0, 2, 1, 2], [1, 1, 2, 0, 1], [1, 0, 3, 0, 1]]


def test_binary_confusion_matrix():
    target = tp.tensor([1, 1, 0, 0])
    preds = tp.tensor([0, 1, 0, 0])
    out = metrics.functional.classification.binary_confusion_matrix(preds, target)
    assert out.tolist() == [[2, 0], [1, 1]]


def test_multiclass_confusion_matrix():
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([2, 1, 0, 1])
    out = metrics.functional.classification.multiclass_confusion_matrix(preds, target, num_classes=3)
    assert out.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_multilabel_confusion_matrix():
    target = tp.tensor([[0, 1, 0], [1, 0, 1]])
    preds = tp.tensor([[0, 0, 1], [1, 0, 1]])
    out = metrics.functional.classification.multilabel_confusion_matrix(preds, target, num_labels=3)
    assert out.tolist() == [[[1, 0], [0, 1]], [[1, 0], [1, 0]], [[0, 1], [0, 1]]]


def test_accuracy_wrapper():
    target = tp.tensor([0, 1, 2, 3])
    preds = tp.tensor([0, 2, 1, 3])
    out = metrics.functional.classification.accuracy(preds, target, task="multiclass", num_classes=4)
    assert out.item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Classification class-based
# ---------------------------------------------------------------------------
def test_binary_accuracy_class():
    metric = metrics.classification.BinaryAccuracy()
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.6667, abs=1e-4)


def test_multiclass_accuracy_class():
    metric = metrics.classification.MulticlassAccuracy(num_classes=3)
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([2, 1, 0, 1])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.8333, abs=1e-4)


def test_accuracy_class_accumulates():
    metric = metrics.classification.Accuracy(task="multiclass", num_classes=3)
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([2, 1, 0, 1])
    metric.update(preds, target)
    metric.update(preds, target)
    assert metric.compute().item() == pytest.approx(0.8333, abs=1e-4)
    metric.reset()
    assert metric.compute().item() == pytest.approx(0.0, abs=1e-4)


def test_f1_class():
    metric = metrics.classification.F1Score(task="multiclass", num_classes=3)
    target = tp.tensor([0, 1, 2, 0, 1, 2])
    preds = tp.tensor([0, 2, 1, 0, 0, 1])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.3333, abs=1e-4)


def test_confusion_matrix_class():
    metric = metrics.classification.ConfusionMatrix(task="multiclass", num_classes=3)
    target = tp.tensor([2, 1, 0, 0])
    preds = tp.tensor([2, 1, 0, 1])
    out = metric(preds, target)
    assert out.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_stat_scores_class():
    metric = metrics.classification.StatScores(task="binary")
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    out = metric(preds, target)
    assert out.tolist() == [2, 1, 2, 1, 3]


# ---------------------------------------------------------------------------
# Regression functional
# ---------------------------------------------------------------------------
def test_mean_squared_error():
    preds = tp.tensor([0.0, 1, 2, 3])
    target = tp.tensor([0.0, 1, 2, 2])
    out = metrics.functional.regression.mean_squared_error(preds, target)
    assert out.item() == pytest.approx(0.25)
    out_rmse = metrics.functional.regression.mean_squared_error(preds, target, squared=False)
    assert out_rmse.item() == pytest.approx(0.5)


def test_mean_absolute_error():
    preds = tp.tensor([0.0, 1, 2, 3])
    target = tp.tensor([0.0, 1, 2, 2])
    out = metrics.functional.regression.mean_absolute_error(preds, target)
    assert out.item() == pytest.approx(0.25)


def test_r2_score():
    target = tp.tensor([[0.5, 1], [-1, 1], [7, -6]])
    preds = tp.tensor([[0, 2], [-1, 2], [8, -5]])
    out = metrics.functional.regression.r2_score(preds, target, multioutput="raw_values")
    assert out.tolist() == pytest.approx([0.9654, 0.9082], abs=1e-4)


def test_pearson_corrcoef():
    target = tp.tensor([3, -0.5, 2, 7])
    preds = tp.tensor([2.5, 0.0, 2, 8])
    out = metrics.functional.regression.pearson_corrcoef(preds, target)
    assert out.item() == pytest.approx(0.9849, abs=1e-4)


def test_spearman_corrcoef():
    target = tp.tensor([3, -0.5, 2, 7])
    preds = tp.tensor([2.5, 0.0, 2, 8])
    out = metrics.functional.regression.spearman_corrcoef(preds, target)
    assert out.item() == pytest.approx(1.0, abs=1e-4)


def test_explained_variance():
    target = tp.tensor([[0.5, 1], [-1, 1], [7, -6]])
    preds = tp.tensor([[0, 2], [-1, 2], [8, -5]])
    out = metrics.functional.regression.explained_variance(preds, target, multioutput="raw_values")
    assert out.tolist() == pytest.approx([0.9677, 1.0], abs=1e-4)


def test_symmetric_mape():
    target = tp.tensor([1, 10, 1e6])
    preds = tp.tensor([0.9, 15, 1.2e6])
    out = metrics.functional.regression.symmetric_mean_absolute_percentage_error(preds, target)
    assert out.item() == pytest.approx(0.2290, abs=1e-4)


# ---------------------------------------------------------------------------
# Regression class-based
# ---------------------------------------------------------------------------
def test_mse_class():
    metric = metrics.regression.MeanSquaredError()
    preds = tp.tensor([0.0, 1, 2, 3])
    target = tp.tensor([0.0, 1, 2, 2])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.25)


def test_mae_class():
    metric = metrics.regression.MeanAbsoluteError()
    preds = tp.tensor([0.0, 1, 2, 3])
    target = tp.tensor([0.0, 1, 2, 2])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.25)


def test_r2_class():
    metric = metrics.regression.R2Score()
    target = tp.tensor([[0.5, 1], [-1, 1], [7, -6]])
    preds = tp.tensor([[0, 2], [-1, 2], [8, -5]])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.9368, abs=1e-4)


def test_pearson_class():
    metric = metrics.regression.PearsonCorrCoef()
    target = tp.tensor([3, -0.5, 2, 7])
    preds = tp.tensor([2.5, 0.0, 2, 8])
    out = metric(preds, target)
    assert out.item() == pytest.approx(0.9849, abs=1e-4)


def test_spearman_class():
    metric = metrics.regression.SpearmanCorrCoef()
    target = tp.tensor([3, -0.5, 2, 7])
    preds = tp.tensor([2.5, 0.0, 2, 8])
    out = metric(preds, target)
    assert out.item() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# MetricCollection
# ---------------------------------------------------------------------------
def test_metric_collection_dict():
    target = tp.tensor([0, 1, 2, 0, 1, 2])
    preds = tp.tensor([0, 2, 1, 0, 0, 1])
    collection = metrics.MetricCollection(
        {
            "f1": metrics.classification.MulticlassF1Score(num_classes=3, average="macro"),
            "acc": metrics.classification.MulticlassAccuracy(num_classes=3, average="macro"),
        }
    )
    out = collection(preds, target)
    assert set(out.keys()) == {"acc", "f1"}
    assert out["f1"].item() == pytest.approx(0.3333, abs=1e-4)
    assert out["acc"].item() == pytest.approx(0.3333, abs=1e-4)


def test_metric_collection_sequence():
    collection = metrics.MetricCollection(
        [
            metrics.regression.MeanSquaredError(),
            metrics.regression.MeanAbsoluteError(),
        ]
    )
    preds = tp.tensor([0.0, 1, 2, 3])
    target = tp.tensor([0.0, 1, 2, 2])
    out = collection(preds, target)
    assert set(out.keys()) == {"MeanAbsoluteError", "MeanSquaredError"}
    assert out["MeanSquaredError"].item() == pytest.approx(0.25)
    assert out["MeanAbsoluteError"].item() == pytest.approx(0.25)