"""Tests for image (PSNR/SSIM) and text (BLEU/ROUGE) metrics."""

import pytest

import tensorplay as tp
import tensorplay.metrics as metrics
from tensorplay.metrics.functional import image as fimg
from tensorplay.metrics.functional import text as ftext


# ---------------------------------------------------------------------------
# Image functional
# ---------------------------------------------------------------------------


def test_peak_signal_noise_ratio():
    pred = tp.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = tp.tensor([[3.0, 2.0], [1.0, 0.0]])
    assert fimg.peak_signal_noise_ratio(pred, target, data_range=3.0).item() == pytest.approx(2.5527, abs=1e-4)
    assert fimg.peak_signal_noise_ratio(pred, target, data_range=3.0, reduction="sum").item() == pytest.approx(
        2.5527, abs=1e-4
    )
    out = fimg.peak_signal_noise_ratio(pred, target, data_range=3.0, reduction="none", dim=0)
    assert out.tolist() == pytest.approx([2.5527, 2.5527], abs=1e-4)


def test_structural_similarity_index_measure():
    preds = tp.rand([2, 3, 16, 16])
    targets = preds * 0.75
    out = fimg.structural_similarity_index_measure(preds, targets)
    assert 0.0 <= out.item() <= 1.0
    # uniform kernel and per-sample reduction
    out_uniform = fimg.structural_similarity_index_measure(preds, targets, gaussian_kernel=False)
    assert 0.0 <= out_uniform.item() <= 1.0
    per_sample = fimg.structural_similarity_index_measure(preds, targets, reduction="none")
    assert per_sample.shape == (2,)
    # full image return
    sim, full = fimg.structural_similarity_index_measure(preds, targets, return_full_image=True)
    assert full.shape == (2, 3, 16, 16)
    assert sim.ndim == 0


def test_multiscale_structural_similarity_index_measure():
    preds = tp.rand([3, 3, 256, 256])
    targets = preds * 0.75
    out = fimg.multiscale_structural_similarity_index_measure(preds, targets, data_range=1.0)
    assert 0.0 <= out.item() <= 1.0
    per_sample = fimg.multiscale_structural_similarity_index_measure(preds, targets, reduction="none")
    assert per_sample.shape == (3,)


# ---------------------------------------------------------------------------
# Image classes
# ---------------------------------------------------------------------------


def test_psnr_class():
    metric = metrics.PeakSignalNoiseRatio(data_range=3.0)
    pred = tp.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = tp.tensor([[3.0, 2.0], [1.0, 0.0]])
    metric(pred, target)
    assert metric.compute().item() == pytest.approx(2.5527, abs=1e-4)


def test_ssim_class():
    metric = metrics.StructuralSimilarityIndexMeasure()
    preds = tp.rand([2, 3, 16, 16])
    metric.update(preds, preds * 0.75)
    assert 0.0 <= metric.compute().item() <= 1.0

    metric = metrics.MultiScaleStructuralSimilarityIndexMeasure()
    preds = tp.rand([2, 3, 256, 256])
    metric.update(preds, preds * 0.75)
    assert 0.0 <= metric.compute().item() <= 1.0


# ---------------------------------------------------------------------------
# Text functional
# ---------------------------------------------------------------------------


def test_bleu_score():
    preds = ["the cat is on the mat"]
    target = [["there is a cat on the mat", "a cat is on the mat"]]
    assert ftext.bleu_score(preds, target).item() == pytest.approx(0.7598, abs=1e-4)
    assert ftext.bleu_score(preds, target, n_gram=2).item() == pytest.approx(0.8165, abs=1e-4)
    assert ftext.bleu_score(preds, target, smooth=True).item() == pytest.approx(0.8034, abs=1e-4)
    assert ftext.bleu_score(preds, target, n_gram=2, weights=[0.5, 0.5]).item() == pytest.approx(0.8165, abs=1e-4)
    with pytest.raises(ValueError, match="weights"):
        ftext.bleu_score(preds, target, n_gram=2, weights=[0.5])


def test_rouge_score():
    preds = "My name is John"
    target = "Is your name John"
    out = ftext.rouge_score(preds, target, rouge_keys=("rouge1", "rouge2", "rougeL"))
    assert out["rouge1_fmeasure"].item() == pytest.approx(0.75, abs=1e-4)
    assert out["rouge1_precision"].item() == pytest.approx(0.75, abs=1e-4)
    assert out["rouge1_recall"].item() == pytest.approx(0.75, abs=1e-4)
    assert out["rouge2_fmeasure"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["rougeL_fmeasure"].item() == pytest.approx(0.5, abs=1e-4)

    out = ftext.rouge_score(
        ["a quick brown fox jumps over the lazy dog"],
        [["a fast brown fox leaps over the sleepy dog", "the quick brown fox jumps over the lazy dog"]],
        accumulate="avg",
        rouge_keys=("rouge1", "rougeL"),
    )
    assert out["rouge1_fmeasure"].item() == pytest.approx(0.7778, abs=1e-4)
    with pytest.raises(ValueError, match="rouge key"):
        ftext.rouge_score(preds, target, rouge_keys=("rouge10",))


# ---------------------------------------------------------------------------
# Text classes
# ---------------------------------------------------------------------------


def test_bleu_class():
    metric = metrics.BLEUScore()
    preds = ["the cat is on the mat"]
    target = [["there is a cat on the mat", "a cat is on the mat"]]
    metric(preds, target)
    assert metric.compute().item() == pytest.approx(0.7598, abs=1e-4)


def test_rouge_class():
    metric = metrics.ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"))
    metric(["My name is John"], [["Is your name John"]])
    out = metric.compute()
    assert out["rouge1_fmeasure"].item() == pytest.approx(0.75, abs=1e-4)
    assert out["rougeL_fmeasure"].item() == pytest.approx(0.5, abs=1e-4)


def test_wer_cer():
    preds = ["this is the prediction", "there is a value"]
    target = ["this is the reference", "there is another value"]
    assert ftext.word_error_rate(preds, target).item() == pytest.approx(0.25, abs=1e-6)
    assert ftext.char_error_rate(preds, target).item() == pytest.approx(0.3256, abs=1e-4)

    metric = metrics.WordErrorRate()
    metric(preds, target)
    assert metric.compute().item() == pytest.approx(0.25, abs=1e-6)

    metric = metrics.CharErrorRate()
    metric(preds, target)
    assert metric.compute().item() == pytest.approx(0.3256, abs=1e-4)


def test_edit_distance():
    assert ftext.edit_distance(["rain"], ["shine"]).item() == pytest.approx(3.0)
    assert ftext.edit_distance(["rain"], ["shine"], substitution_cost=2).item() == pytest.approx(5.0)
    out = ftext.edit_distance(["rain", "lnaguaeg"], ["shine", "language"], reduction=None)
    assert out.tolist() == [3, 4]
    metric = metrics.EditDistance()
    metric(["rain"], ["shine"])
    assert metric.compute().item() == pytest.approx(3.0)


def test_total_variation():
    img = tp.rand([5, 3, 28, 28])
    out = fimg.total_variation(img)
    assert out.item() > 0
    assert fimg.total_variation(img, reduction="mean").item() > 0
    assert fimg.total_variation(img, reduction="none").shape == (5,)
    metric = metrics.TotalVariation()
    metric(img)
    assert metric.compute().item() > 0


def test_minmax_metric():
    metric = metrics.MinMaxMetric(metrics.MeanSquaredError())
    for i in range(3):
        metric(tp.tensor([float(i)]), tp.tensor([float(i + 1)]))
        out = metric.compute()
        assert set(out.keys()) == {"raw", "max", "min"}
    assert out["min"].item() <= out["raw"].item() <= out["max"].item()


def test_classwise_wrapper():
    base = metrics.MulticlassAccuracy(num_classes=3, average=None)
    wrapped = metrics.ClasswiseWrapper(base, labels=["cat", "dog", "fish"])
    preds = tp.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    target = tp.tensor([0, 1, 2])
    wrapped(preds, target)
    out = wrapped.compute()
    assert set(out.keys()) == {"multiclassaccuracy_cat", "multiclassaccuracy_dog", "multiclassaccuracy_fish"}
    assert all(v.item() == pytest.approx(1.0) for v in out.values())


def test_multioutput_wrapper():
    base = metrics.SpearmanCorrCoef()
    wrapped = metrics.MultioutputWrapper(base, num_outputs=2)
    preds = tp.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 4.0]])
    target = tp.tensor([[2.0, 1.0], [1.0, 2.0], [3.0, 4.0], [4.0, 3.0]])
    wrapped(preds, target)
    out = wrapped.compute()
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------


def test_image_text_top_level_exports():
    for name in [
        "PeakSignalNoiseRatio",
        "StructuralSimilarityIndexMeasure",
        "MultiScaleStructuralSimilarityIndexMeasure",
        "BLEUScore",
        "ROUGEScore",
    ]:
        assert hasattr(metrics, name), f"missing top-level export {name}"
    for name in [
        "peak_signal_noise_ratio",
        "structural_similarity_index_measure",
        "multiscale_structural_similarity_index_measure",
        "bleu_score",
        "rouge_score",
    ]:
        assert hasattr(metrics.functional, name), f"missing functional export {name}"