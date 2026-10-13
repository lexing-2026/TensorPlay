"""Tests for the gap-filling metrics (regression/classification/retrieval/nominal/text/image/wrappers extras)."""

import pytest

import tensorplay as tp
from tensorplay.metrics import (
    BootStrapper,
    CHRFScore,
    CramersV,
    CriticalSuccessIndex,
    ExactMatch,
    ExtendedEditDistance,
    FleissKappa,
    KendallRankCorrCoef,
    LogAUC,
    MatchErrorRate,
    MeanSquaredLogError,
    MultitaskWrapper,
    NegativePredictiveValue,
    NormalizedRootMeanSquaredError,
    PearsonsContingencyCoefficient,
    Perplexity,
    PrecisionAtFixedRecall,
    RecallAtFixedPrecision,
    RelativeSquaredError,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalPrecisionRecallCurve,
    RetrievalRPrecision,
    RetrievalRecall,
    RetrievalRecallAtFixedPrecision,
    RetrievalFallOut,
    RetrievalHitRate,
    RunningMean,
    RunningSum,
    SacreBLEUScore,
    SensitivityAtSpecificity,
    SpecificityAtSensitivity,
    SpectralAngleMapper,
    SpectralDistortionIndex,
    TheilsU,
    TranslationEditRate,
    TschuprowsT,
    TweedieDevianceScore,
    UniversalImageQualityIndex,
    WordInfoLost,
    WordInfoPreserved,
)


# ---------------- regression extras ----------------

def test_mean_squared_log_error():
    x = tp.tensor([0.0, 1, 2, 3])
    y = tp.tensor([0.0, 1, 2, 2])
    assert MeanSquaredLogError()(x, y).item() == pytest.approx(0.0207, abs=1e-3)


def test_relative_squared_error():
    x = tp.tensor([0.0, 1, 2, 3])
    y = tp.tensor([0.0, 1, 2, 2])
    assert RelativeSquaredError()(x, y).item() >= 0


def test_normalized_root_mean_squared_error():
    x = tp.tensor([0.0, 1, 2, 3])
    y = tp.tensor([0.0, 1, 2, 2])
    from tensorplay.metrics.functional.regression import normalized_root_mean_squared_error

    assert normalized_root_mean_squared_error(x, y, normalization="mean").item() == pytest.approx(0.4, abs=1e-3)
    assert normalized_root_mean_squared_error(x, y, normalization="range").item() == pytest.approx(0.25, abs=1e-3)


def test_tweedie_deviance_score():
    x = tp.tensor([1.0, 2.0, 3.0, 4.0])
    y = tp.tensor([4.0, 3.0, 2.0, 1.0])
    assert TweedieDevianceScore(power=2)(y, x).item() == pytest.approx(1.2083, abs=1e-3)


def test_critical_success_index():
    x = tp.tensor([[0.2, 0.7], [0.9, 0.3]])
    y = tp.tensor([[0.4, 0.2], [0.8, 0.6]])
    assert CriticalSuccessIndex(threshold=0.5)(x, y).item() == pytest.approx(1 / 3, abs=1e-3)


def test_kendall_rank_corrcoef():
    preds = tp.tensor([2.5, 0.0, 2, 8])
    target = tp.tensor([3, -0.5, 2, 1])
    assert KendallRankCorrCoef()(preds, target).item() == pytest.approx(1 / 3, abs=1e-3)


# ---------------- classification extras ----------------

def test_exact_match():
    from tensorplay.metrics.functional.classification import multiclass_exact_match

    preds = tp.tensor([[0.2, 0.8], [0.9, 0.1]])
    target = tp.tensor([1, 0])
    assert multiclass_exact_match(preds, target, num_classes=2).item() == 1.0


def test_negative_predictive_value():
    from tensorplay.metrics.functional.classification import binary_negative_predictive_value

    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    assert binary_negative_predictive_value(preds, target).item() >= 0


def test_logauc():
    from tensorplay.metrics.functional.classification import binary_logauc

    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    assert binary_logauc(preds, target).item() >= 0


def test_precision_at_fixed_recall():
    from tensorplay.metrics.functional.classification import binary_precision_at_fixed_recall

    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    precision, _ = binary_precision_at_fixed_recall(preds, target, min_recall=0.5)
    assert 0 <= precision.item() <= 1


def test_recall_at_fixed_precision():
    from tensorplay.metrics.functional.classification import binary_recall_at_fixed_precision

    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    recall, _ = binary_recall_at_fixed_precision(preds, target, min_precision=0.5)
    assert 0 <= recall.item() <= 1


def test_sensitivity_at_specificity():
    from tensorplay.metrics.functional.classification import binary_sensitivity_at_specificity

    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    value, _ = binary_sensitivity_at_specificity(preds, target, min_specificity=0.5)
    assert 0 <= value.item() <= 1


def test_specificity_at_sensitivity():
    from tensorplay.metrics.functional.classification import binary_specificity_at_sensitivity

    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    value, _ = binary_specificity_at_sensitivity(preds, target, min_sensitivity=0.5)
    assert 0 <= value.item() <= 1


# ---------------- retrieval ----------------

def _retrieval_inputs():
    indexes = tp.tensor([0, 0, 0, 1, 1, 1, 1], dtype=tp.long)
    preds = tp.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    target = tp.tensor([False, False, True, False, True, False, True])
    return indexes, preds, target


def test_retrieval_metrics():
    indexes, preds, target = _retrieval_inputs()
    from tensorplay.metrics.functional.retrieval import (
        retrieval_auroc,
        retrieval_average_precision,
        retrieval_fall_out,
        retrieval_hit_rate,
        retrieval_normalized_dcg,
        retrieval_precision,
        retrieval_precision_recall_curve,
        retrieval_r_precision,
        retrieval_recall,
        retrieval_reciprocal_rank,
    )

    assert retrieval_recall(preds, target).item() >= 0
    assert retrieval_precision(preds, target).item() >= 0
    assert retrieval_auroc(preds, target).item() >= 0
    assert retrieval_average_precision(preds, target).item() >= 0
    assert retrieval_normalized_dcg(preds, target).item() >= 0
    assert retrieval_reciprocal_rank(preds, target).item() >= 0
    assert retrieval_r_precision(preds, target).item() >= 0
    assert retrieval_hit_rate(preds, target).item() >= 0
    assert retrieval_fall_out(preds, target).item() >= 0
    fpr, tpr, thresholds = retrieval_precision_recall_curve(preds, target)
    assert fpr.ndim >= 0


def test_retrieval_classes():
    indexes, preds, target = _retrieval_inputs()
    metric = RetrievalMAP()
    assert metric(preds, target, indexes=indexes).item() >= 0


# ---------------- nominal ----------------

def test_nominal_metrics():
    x = tp.tensor([1, 1, 2, 2, 3, 3])
    y = tp.tensor([0, 1, 0, 1, 0, 1])
    assert 0 <= CramersV(num_classes=3)(x, y).item() <= 1
    assert 0 <= TheilsU(num_classes=3)(x, y).item() <= 1
    assert 0 <= TschuprowsT(num_classes=3)(x, y).item() <= 1
    assert 0 <= PearsonsContingencyCoefficient(num_classes=3)(x, y).item() <= 1
    assert FleissKappa()(tp.tensor([[0, 1], [1, 0], [1, 1]])).item() >= -1


# ---------------- text extras ----------------

def test_perplexity():
    preds = tp.tensor([[[0.1, 0.6, 0.3]], [[0.5, 0.2, 0.3]]])
    target = tp.tensor([[1], [0]])
    assert Perplexity()(preds, target).item() > 1


def test_text_edit_metrics():
    preds = ["this is a test"]
    target = ["this is a test"]
    assert MatchErrorRate()(preds, target).item() == 0.0
    assert WordInfoLost()(preds, target).item() == 0.0
    assert WordInfoPreserved()(preds, target).item() == 1.0
    assert ExtendedEditDistance()(preds, target).item() >= 0


def test_text_translation_metrics():
    preds = ["the cat sat on the mat"]
    target = ["the cat sat on the mat"]
    assert CHRFScore()(preds, target).item() == pytest.approx(1.0, abs=1e-3)
    assert SacreBLEUScore()(preds, [target]).item() > 0.5
    assert TranslationEditRate()(preds, target).item() == pytest.approx(0.0, abs=1e-3)


# ---------------- image remote sensing ----------------

def test_remote_sensing_metrics():
    x = tp.rand(1, 3, 16, 16)
    y = tp.rand(1, 3, 16, 16)
    assert SpectralAngleMapper()(x, y).item() >= 0
    assert UniversalImageQualityIndex()(x, y).item() >= -1
    assert SpectralDistortionIndex()(x, y).item() >= 0


# ---------------- wrappers ----------------

def test_running_mean_sum():
    rm = RunningMean(window=3)
    rm(tp.tensor([1.0]))
    rm(tp.tensor([3.0]))
    assert rm.compute().item() == pytest.approx(2.0)

    rs = RunningSum(window=3)
    rs(tp.tensor([1.0]))
    rs(tp.tensor([3.0]))
    assert rs.compute().item() == pytest.approx(4.0)


def test_bootstrap_wrapper():
    from tensorplay.metrics import Accuracy

    acc = BootStrapper(Accuracy(task="binary"), num_bootstraps=5)
    preds = tp.tensor([0.8, 0.2, 0.6])
    target = tp.tensor([1, 0, 1])
    acc.update(preds, target)
    out = acc.compute()
    assert isinstance(out, dict)
    assert "mean" in out and "std" in out


def test_multitask_wrapper():
    from tensorplay.metrics import MeanAbsoluteError, MeanSquaredError

    mt = MultitaskWrapper({"a": MeanAbsoluteError(), "b": MeanSquaredError()})
    preds = {"a": tp.tensor([1.0, 2.0, 3.0]), "b": tp.tensor([1.0, 2.0, 3.0])}
    target = {"a": tp.tensor([1.5, 2.0, 2.0]), "b": tp.tensor([1.5, 2.0, 2.0])}
    out = mt(preds, target)
    assert "a" in out and "b" in out