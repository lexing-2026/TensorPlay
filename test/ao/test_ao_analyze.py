"""Tests for the beyond-upstream model quantization analysis module."""

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.ao import quantization as q
from tensorplay.ao.quantization.analyze import (
    analyze_model,
    model_quantization_summary,
    print_model_analysis,
    quantization_ready,
)


def _model():
    return nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Conv2d(3, 4, 3),
        nn.Dropout(0.5),
    ).eval()


def test_analyze_model_basic():
    report = analyze_model(_model())
    linear_entry = report["0"]
    assert linear_entry["module_class"] == "Linear"
    assert linear_entry["static_quantizable"] is True
    assert linear_entry["quantized_module"].__name__ == "QuantizedLinear"
    assert linear_entry["dynamic_quantizable"] is True
    assert linear_entry["fusion_candidate"] is True
    assert linear_entry["has_observer"] is False

    relu_entry = report["1"]
    assert relu_entry["static_quantizable"] is False
    assert relu_entry["fusion_candidate"] is True

    conv_entry = report["2"]
    assert conv_entry["quantized_module"].__name__ == "Conv2d"


def test_analyze_model_with_qconfig_mapping():
    qconfig = q.QConfig(q.MinMaxObserver, q.default_weight_observer)
    mapping = q.QConfigMapping().set_object_type(nn.Linear, qconfig)
    report = analyze_model(_model(), mapping)
    # the resolved qconfig is visible in the report
    assert report["0"]["qconfig"] is qconfig
    assert report["0"]["dtype"] is not None
    # modules without a matching qconfig fall back to the global (None)
    assert report["2"]["qconfig"] is None


def test_model_quantization_summary():
    summary = model_quantization_summary(_model())
    assert summary["total_modules"] == 4
    assert summary["leaf_modules"] == 4
    assert summary["static_quantizable"] == 3  # Linear, Conv2d, Dropout
    assert summary["fusion_candidates"] == 3  # Linear, ReLU, Conv2d
    assert summary["with_observer"] == 0


def test_quantization_ready():
    ok, blockers = quantization_ready(_model())
    assert ok is True
    assert blockers == []

    class WithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(self.emb(x))

    ok, blockers = quantization_ready(WithEmbedding())
    assert ok is True


def test_analyze_model_prepared_model():
    model = _model()
    # assign qconfigs first, matching the eager workflow contract
    model[0].qconfig = q.default_qconfig
    model[2].qconfig = q.default_qconfig
    q.prepare(model, inplace=True)
    report = analyze_model(model)
    assert report["0"]["has_observer"] is True
    assert report["2"]["has_observer"] is True
    # a module without qconfig stays untouched
    assert report["3"]["has_observer"] is False


def test_print_model_analysis(capsys):
    print_model_analysis(_model())
    captured = capsys.readouterr()
    assert "model quantization analysis" in captured.out
    assert "total_modules: 4" in captured.out
    assert "Linear" in captured.out


def test_public_export():
    assert q.analyze_model is analyze_model
    assert q.print_model_analysis is print_model_analysis
    assert q.model_quantization_summary is model_quantization_summary
    assert q.quantization_ready is quantization_ready