"""Tests for the classification curve/agreement metrics in ``tensorplay.metrics``."""

import pytest

import tensorplay as tp
import tensorplay.metrics as metrics

F = metrics.functional.classification


def test_binary_auroc():
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    assert F.binary_auroc(preds, target).item() == pytest.approx(0.5, abs=1e-4)
    assert F.binary_auroc(preds, target, max_fpr=0.5).item() == pytest.approx(0.5556, abs=1e-4)
    assert F.binary_auroc(preds, target, thresholds=5).item() == pytest.approx(0.4167, abs=1e-4)


def test_multiclass_auroc():
    preds = tp.tensor(
        [[0.75, 0.05, 0.05, 0.05, 0.05],
         [0.05, 0.75, 0.05, 0.05, 0.05],
         [0.05, 0.05, 0.75, 0.05, 0.05],
         [0.05, 0.05, 0.05, 0.75, 0.05]]
    )
    target = tp.tensor([0, 1, 3, 2])
    assert F.multiclass_auroc(preds, target, num_classes=5).item() == pytest.approx(0.5333, abs=1e-4)
    assert F.multiclass_auroc(preds, target, num_classes=5, average=None).tolist() == pytest.approx(
        [1.0, 1.0, 1 / 3, 1 / 3, 0.0], abs=1e-4
    )
    assert F.multiclass_auroc(preds, target, num_classes=5, thresholds=5).item() == pytest.approx(0.5333, abs=1e-4)


def test_multilabel_auroc():
    preds = tp.tensor(
        [[0.75, 0.05, 0.35],
         [0.45, 0.75, 0.05],
         [0.05, 0.55, 0.75],
         [0.05, 0.65, 0.05]]
    )
    target = tp.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
    assert F.multilabel_auroc(preds, target, num_labels=3, average="macro").item() == pytest.approx(0.6528, abs=1e-4)
    assert F.multilabel_auroc(preds, target, num_labels=3, average="micro").item() == pytest.approx(0.6571, abs=1e-4)
    assert F.multilabel_auroc(preds, target, num_labels=3, average=None).tolist() == pytest.approx(
        [0.625, 0.5, 0.8333], abs=1e-4
    )


def test_binary_average_precision():
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    assert F.binary_average_precision(preds, target).item() == pytest.approx(0.7556, abs=1e-4)
    assert F.binary_average_precision(preds, target, thresholds=5).item() == pytest.approx(0.5667, abs=1e-4)


def test_multiclass_average_precision():
    preds = tp.tensor(
        [[0.75, 0.05, 0.05, 0.05, 0.05],
         [0.05, 0.75, 0.05, 0.05, 0.05],
         [0.05, 0.05, 0.75, 0.05, 0.05],
         [0.05, 0.05, 0.05, 0.75, 0.05]]
    )
    target = tp.tensor([0, 1, 3, 2])
    assert F.multiclass_average_precision(preds, target, num_classes=5).item() == pytest.approx(0.625, abs=1e-4)
    res = F.multiclass_average_precision(preds, target, num_classes=5, average=None)
    assert res[:4].tolist() == pytest.approx([1.0, 1.0, 0.25, 0.25], abs=1e-4)
    assert tp.isnan(res[4]).item()


def test_multilabel_average_precision():
    preds = tp.tensor(
        [[0.75, 0.05, 0.35],
         [0.45, 0.75, 0.05],
         [0.05, 0.55, 0.75],
         [0.05, 0.65, 0.05]]
    )
    target = tp.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
    assert F.multilabel_average_precision(preds, target, num_labels=3).item() == pytest.approx(0.75, abs=1e-4)
    assert F.multilabel_average_precision(preds, target, num_labels=3, average=None).tolist() == pytest.approx(
        [0.75, 0.5833, 0.9167], abs=1e-4
    )


def test_binary_precision_recall_curve():
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    precision, recall, thresholds = F.binary_precision_recall_curve(preds, target)
    assert precision.tolist() == pytest.approx([0.6, 0.5, 0.6667, 0.5, 1.0, 1.0], abs=1e-4)
    assert recall.tolist() == pytest.approx([1.0, 0.6667, 0.6667, 0.3333, 0.3333, 0.0], abs=1e-4)
    assert thresholds.tolist() == pytest.approx([0.08, 0.13, 0.19, 0.26, 0.34], abs=1e-4)


def test_multiclass_precision_recall_curve():
    preds = tp.tensor(
        [[0.75, 0.05, 0.05, 0.05, 0.05],
         [0.05, 0.75, 0.05, 0.05, 0.05],
         [0.05, 0.05, 0.75, 0.05, 0.05],
         [0.05, 0.05, 0.05, 0.75, 0.05]]
    )
    target = tp.tensor([0, 1, 3, 2])
    precision, recall, thresholds = F.multiclass_precision_recall_curve(preds, target, num_classes=5)
    assert precision[0].tolist() == pytest.approx([0.25, 1.0, 1.0], abs=1e-4)
    assert recall[0].tolist() == pytest.approx([1.0, 1.0, 0.0], abs=1e-4)


def test_binary_roc():
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    fpr, tpr, thresholds = F.binary_roc(preds, target)
    assert fpr.tolist() == pytest.approx([0.0, 0.0, 0.5, 0.5, 1.0, 1.0], abs=1e-4)
    assert tpr.tolist() == pytest.approx([0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0], abs=1e-4)
    assert len(thresholds) == 6


def test_binary_cohen_kappa():
    preds = tp.tensor([0, 1, 0, 0])
    target = tp.tensor([1, 1, 0, 0])
    assert F.binary_cohen_kappa(preds, target).item() == pytest.approx(0.5, abs=1e-4)
    # float preds are thresholded at 0.5
    preds = tp.tensor([0.35, 0.85, 0.48, 0.01])
    assert F.binary_cohen_kappa(preds, target).item() == pytest.approx(0.5, abs=1e-4)


def test_multiclass_cohen_kappa():
    preds = tp.tensor([2, 1, 0, 1])
    target = tp.tensor([2, 1, 0, 0])
    assert F.multiclass_cohen_kappa(preds, target, num_classes=3).item() == pytest.approx(0.6364, abs=1e-4)
    assert F.multiclass_cohen_kappa(preds, target, num_classes=3, weights="linear").item() == pytest.approx(0.7143, abs=1e-4)
    assert F.multiclass_cohen_kappa(preds, target, num_classes=3, weights="quadratic").item() == pytest.approx(0.8, abs=1e-4)


def test_binary_matthews_corrcoef():
    preds = tp.tensor([0, 1, 0, 0])
    target = tp.tensor([1, 1, 0, 0])
    assert F.binary_matthews_corrcoef(preds, target).item() == pytest.approx(0.5774, abs=1e-4)


def test_multiclass_matthews_corrcoef():
    preds = tp.tensor([2, 1, 0, 1])
    target = tp.tensor([2, 1, 0, 0])
    assert F.multiclass_matthews_corrcoef(preds, target, num_classes=3).item() == pytest.approx(0.7, abs=1e-4)


def test_multilabel_matthews_corrcoef():
    preds = tp.tensor([[0, 0, 1], [1, 0, 1]])
    target = tp.tensor([[0, 1, 0], [1, 0, 1]])
    assert F.multilabel_matthews_corrcoef(preds, target, num_labels=3).item() == pytest.approx(1 / 3, abs=1e-4)


def test_binary_hamming_distance():
    preds = tp.tensor([0, 0, 1, 1, 0, 1])
    target = tp.tensor([0, 1, 0, 1, 0, 1])
    assert F.binary_hamming_distance(preds, target).item() == pytest.approx(1 / 3, abs=1e-4)


def test_multiclass_hamming_distance():
    preds = tp.tensor([2, 1, 0, 1])
    target = tp.tensor([2, 1, 0, 0])
    assert F.multiclass_hamming_distance(preds, target, num_classes=3).item() == pytest.approx(1 / 6, abs=1e-4)
    assert F.multiclass_hamming_distance(preds, target, num_classes=3, average=None).tolist() == pytest.approx(
        [0.5, 0.0, 0.0], abs=1e-4
    )


def test_multilabel_hamming_distance():
    preds = tp.tensor([[0, 0, 1], [1, 0, 1]])
    target = tp.tensor([[0, 1, 0], [1, 0, 1]])
    assert F.multilabel_hamming_distance(preds, target, num_labels=3).item() == pytest.approx(1 / 3, abs=1e-4)
    assert F.multilabel_hamming_distance(preds, target, num_labels=3, average=None).tolist() == pytest.approx(
        [0.0, 0.5, 0.5], abs=1e-4
    )


# ---------------------------------------------------------------------------
# Class-based versions
# ---------------------------------------------------------------------------


def test_binary_auroc_class():
    metric = metrics.classification.BinaryAUROC()
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    assert metric(preds, target).item() == pytest.approx(0.5, abs=1e-4)
    assert metric.compute().item() == pytest.approx(0.5, abs=1e-4)
    metric.reset()
    metric.update(preds, target)
    assert metric.compute().item() == pytest.approx(0.5, abs=1e-4)


def test_multiclass_auroc_class_accumulates():
    metric = metrics.classification.MulticlassAUROC(num_classes=3)
    preds = tp.tensor([[0.75, 0.05, 0.05], [0.05, 0.75, 0.05], [0.05, 0.05, 0.75]])
    target = tp.tensor([0, 1, 2])
    metric.update(preds, target)
    metric.update(preds, target)
    assert metric.compute().item() == pytest.approx(1.0, abs=1e-4)


def test_auroc_wrapper():
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    assert F.auroc(preds, target, task="binary").item() == pytest.approx(0.5, abs=1e-4)


def test_metric_collection_curves():
    collection = metrics.MetricCollection(
        {
            "auroc": metrics.classification.BinaryAUROC(),
            "ap": metrics.classification.BinaryAveragePrecision(),
            "hd": metrics.classification.BinaryHammingDistance(),
        }
    )
    preds = tp.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = tp.tensor([0, 0, 1, 1, 1])
    out = collection(preds, target)
    assert set(out.keys()) == {"auroc", "ap", "hd"}
    assert out["auroc"].item() == pytest.approx(0.5, abs=1e-4)
    assert out["ap"].item() == pytest.approx(0.7556, abs=1e-4)
    assert out["hd"].item() == pytest.approx(0.6, abs=1e-4)
    out = collection.compute()
    assert out["auroc"].item() == pytest.approx(0.5, abs=1e-4)