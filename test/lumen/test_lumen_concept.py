"""Tests for the concept-based interpretability module (CAV / TCAV).

The tests cover what is testable without sklearn installed: Concept / CAV
construction, DataIterator helpers, string utilities, AV importability, and
the friendly error raised by TCAV when sklearn is absent (this environment has
no sklearn).
"""

import pytest

import tensorplay as tensorplay
from tensorplay import nn
from tensorplay.utils.data import DataLoader, TensorDataset

from tensorplay.lumen.concept import CAV, Concept, ConceptInterpreter, TCAV
from tensorplay.lumen.concept._utils.common import concepts_to_str
from tensorplay.lumen.concept._utils.data_iterator import (
    dataset_to_dataloader,
)
from tensorplay.lumen._utils.av import AV


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        return self.linear(x)


def _make_concepts() -> list:
    dataset = TensorDataset(tensorplay.randn(8, 4), tensorplay.zeros(8, dtype=tensorplay.int64))
    dl = dataset_to_dataloader(dataset, batch_size=4)
    return [Concept(0, "striped", dl), Concept(1, "random", dl)]


def test_concept_construction() -> None:
    c = Concept(0, "striped", None)
    assert c.identifier == "striped-0"
    assert repr(c) == "Concept(0, 'striped')"


def test_concepts_to_str() -> None:
    concepts = _make_concepts()
    assert concepts_to_str(concepts) == "0-1"


def test_cav_construction(tmp_path) -> None:
    concepts = _make_concepts()
    cav = CAV(concepts, layer="linear", save_path=str(tmp_path))
    assert cav.layer == "linear"
    assert cav.model_id == "default_model_id"
    assert len(cav.concepts) == 2


def test_concept_interpreter_base() -> None:
    model = TinyModel()
    interpreter = ConceptInterpreter(model)
    assert interpreter.model is model


def test_dataset_to_dataloader() -> None:
    dataset = TensorDataset(tensorplay.randn(6, 3))
    dl = dataset_to_dataloader(dataset, batch_size=2)
    batches = list(dl)
    assert len(batches) == 3
    assert batches[0][0].shape == (2, 3)


def test_tcav_construct(tmp_path) -> None:
    model = TinyModel()
    tcav = TCAV(model, layers=["linear"], save_path=str(tmp_path))
    assert tcav.model is model
    assert tcav.layers == ["linear"]


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    _sklearn_available(), reason="sklearn installed; friendly error not exercised"
)
def test_tcav_sklearn_missing_error(tmp_path) -> None:
    model = TinyModel()
    tcav = TCAV(model, layers=["linear"], save_path=str(tmp_path))
    ds = TensorDataset(tensorplay.randn(8, 4))
    dl = dataset_to_dataloader(ds, batch_size=4)
    striped = Concept(0, "striped", dl)
    random = Concept(1, "random", dl)
    with pytest.raises(ValueError, match="sklearn"):
        tcav.interpret(
            inputs=tensorplay.randn(2, 4),
            experimental_sets=[[striped], [random]],
        )


def test_av_importable() -> None:
    assert AV is not None