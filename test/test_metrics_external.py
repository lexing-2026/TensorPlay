"""Tests for external-dependency metrics (lazy loading / ModuleNotFoundError behaviour).

These metrics mirror the upstream API. Metrics whose third-party dependency is
missing (pesq / pystoi / pycocotools) raise a clear ModuleNotFoundError.
Metrics that rely on transformers are importable and only load the model at
call time, so they are not instantiated here to avoid model downloads.
"""

import pytest

import tensorplay as tp
from tensorplay.metrics import (
    BERTScore,
    CLIPImageQualityAssessment,
    CLIPScore,
    InfoLM,
    LipVertexError,
    ModifiedPanopticQuality,
    PanopticQuality,
    PerceptualEvaluationSpeechQuality,
    SQuAD,
    ShortTimeObjectiveIntelligibility,
)


def test_external_metrics_are_importable():
    assert BERTScore is not None
    assert CLIPScore is not None
    assert CLIPImageQualityAssessment is not None
    assert InfoLM is not None
    assert LipVertexError is not None
    assert PanopticQuality is not None
    assert ModifiedPanopticQuality is not None
    assert SQuAD is not None
    assert PerceptualEvaluationSpeechQuality is not None
    assert ShortTimeObjectiveIntelligibility is not None


def test_pesq_raises_without_dependency():
    with pytest.raises(ModuleNotFoundError, match="pesq"):
        PerceptualEvaluationSpeechQuality(fs=8000, mode="wb")


def test_functional_pesq_raises_without_dependency():
    from tensorplay.metrics.functional.audio import perceptual_evaluation_speech_quality

    with pytest.raises(ModuleNotFoundError, match="pesq"):
        perceptual_evaluation_speech_quality(tp.randn(16000), tp.randn(16000), 8000, "wb")


def test_stoi_raises_without_dependency():
    with pytest.raises(ModuleNotFoundError, match="pystoi"):
        ShortTimeObjectiveIntelligibility(fs=16000)


def test_panoptic_quality_works():
    pq = PanopticQuality(things=[6], stuffs=[7])
    preds = tp.tensor(
        [[[[6, 0], [6, 0], [7, 0], [7, 0]], [[6, 0], [6, 0], [7, 0], [7, 0]], [[7, 0], [7, 0], [6, 1], [6, 1]], [[7, 0], [7, 0], [6, 1], [6, 1]]]]
    )
    target = preds.clone()
    pq.update(preds, target)
    assert pq.compute().item() >= 0