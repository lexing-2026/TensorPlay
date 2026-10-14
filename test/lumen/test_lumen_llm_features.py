"""Tests for SAE feature learning, logit lens, and attribution graphs.

All tests run natively on the shared tensorplay transformer; no third-party
runtime or pretrained checkpoint is involved.
"""

import numpy as np
import pytest

import tensorplay as tensorplay
from tensorplay.utils.data import DataLoader, TensorDataset

from tensorplay.lumen.llm import (
    AttributionGraph,
    LogitLens,
    SparseAutoencoder,
    collect_hidden_states,
    feature_frequency,
    reconstruction_error,
    top_activating_indices,
    train_sae,
)

from native_model import NativeGPT, NativeTokenizer, VOCAB, ids_from_text


@pytest.fixture(scope="module")
def model() -> NativeGPT:
    return NativeGPT(vocab=len(VOCAB))


@pytest.fixture(scope="module")
def tokenizer() -> NativeTokenizer:
    return NativeTokenizer()


def test_sae_training_reduces_loss() -> None:
    rng = np.random.default_rng(0)
    data = tensorplay.tensor(rng.normal(size=(64, 8)).astype("float32"))
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=16)
    sae = SparseAutoencoder(input_dim=8, hidden_dim=16)
    history = train_sae(sae, dl, n_epochs=5, lr=1e-2, l1_coef=1e-3)
    assert len(history) == 5
    assert history[-1] < history[0]


def test_sae_feature_utilities() -> None:
    sae = SparseAutoencoder(input_dim=8, hidden_dim=16)
    x = tensorplay.randn(32, 8)
    recon, features = sae(x)
    assert recon.shape == (32, 8)
    assert features.shape == (32, 16)
    freq = feature_frequency(features)
    assert freq.shape == (16,)
    assert ((freq >= 0) & (freq <= 1)).all()
    top = top_activating_indices(features, k=3)
    assert len(top) == 16
    assert all(len(v) == 3 for v in top.values())
    err = reconstruction_error(sae, x)
    assert isinstance(err, float)


def test_collect_hidden_states(model: NativeGPT) -> None:
    seqs = [
        ids_from_text(NativeTokenizer(), "The capital of France is"),
        ids_from_text(NativeTokenizer(), "The capital of England is"),
    ]
    hs = collect_hidden_states(model, layer=1, input_ids_list=seqs, batch_size=2)
    # 2 sequences * 5 tokens * hidden_dim 16
    assert hs.shape == (10, 16)


def test_logit_lens(model: NativeGPT, tokenizer: NativeTokenizer) -> None:
    ids = ids_from_text(tokenizer, "The capital of France is")
    lens = LogitLens(model, tokenizer)
    result = lens.lens(ids, position=-1, topk=3)
    assert result["layers"] == [0, 1]
    assert len(result["predictions"][0]) == 3
    # probabilities are in [0, 1] and sorted descending
    probs0 = [p for _, p in result["predictions"][0]]
    assert all(0.0 <= p <= 1.0 for p in probs0)
    assert probs0 == sorted(probs0, reverse=True)
    assert len(result["tokens"]) == 5


def test_logit_lens_target_probability(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    ids = ids_from_text(tokenizer, "The capital of France is")
    lens = LogitLens(model, tokenizer)
    target_id = tokenizer.tok2id["Paris"]
    probs = lens.target_probability(ids, target_id)
    assert list(probs.keys()) == [0, 1]
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_logit_lens_visualize(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    import matplotlib.pyplot as plt

    ids = ids_from_text(tokenizer, "The capital of France is")
    lens = LogitLens(model, tokenizer)
    result = lens.lens(ids, position=-1, topk=3)
    fig, ax = lens.visualize(result, use_pyplot=False)
    plt.close(fig)
    assert ax is not None


def test_attribution_graph(model: NativeGPT, tokenizer: NativeTokenizer) -> None:
    ids = ids_from_text(tokenizer, "The capital of France is")
    ag = AttributionGraph(model)
    graph = ag.build(ids, target_position=-1, topk=5)
    assert len(graph.nodes) > 0
    assert graph.layers == [0, 1]
    # all kept nodes have positive attribution
    assert all(n.attribution > 0 for n in graph.nodes)
    # edges only go from layer 0 to layer 1
    for edge in graph.edges:
        assert graph.nodes[edge.source].layer == 0
        assert graph.nodes[edge.target].layer == 1
        assert edge.weight > 0


def test_attribution_graph_visualize(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    import matplotlib.pyplot as plt

    ids = ids_from_text(tokenizer, "The capital of France is")
    ag = AttributionGraph(model)
    graph = ag.build(ids, target_position=-1, topk=4)
    fig, ax = ag.visualize(graph, use_pyplot=False)
    plt.close(fig)
    assert ax is not None