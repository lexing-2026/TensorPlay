"""Tests for cross-layer transcoders and feature circuits.

All tests run natively on the shared tensorplay transformer.
"""

import numpy as np
import pytest

import tensorplay as tensorplay
from tensorplay.utils.data import DataLoader, TensorDataset

from tensorplay.lumen.llm import (
    CrossLayerTranscoder,
    FeatureCircuit,
    collect_component_updates,
    top_attention_heads,
    top_mlp_features,
    train_clt,
)

from native_model import NativeGPT, NativeTokenizer, ids_from_text


@pytest.fixture(scope="module")
def model() -> NativeGPT:
    return NativeGPT(vocab=len(NativeTokenizer().id2tok))


@pytest.fixture(scope="module")
def tokenizer() -> NativeTokenizer:
    return NativeTokenizer()


def test_clt_training_reduces_loss() -> None:
    rng = np.random.default_rng(0)
    residual = tensorplay.tensor(rng.normal(size=(64, 8)).astype("float32"))
    delta = tensorplay.tensor(rng.normal(size=(64, 8)).astype("float32"))
    ds = TensorDataset(residual, delta)
    dl = DataLoader(ds, batch_size=16)
    clt = CrossLayerTranscoder(input_dim=8, hidden_dim=16)
    history = train_clt(clt, dl, n_epochs=5, lr=1e-2, l1_coef=1e-3)
    assert len(history) == 5
    assert history[-1] < history[0]


def test_collect_component_updates(model: NativeGPT) -> None:
    tok = NativeTokenizer()
    seqs = [
        ids_from_text(tok, "The capital of France is"),
        ids_from_text(tok, "The capital of England is"),
    ]
    inputs, deltas = collect_component_updates(
        model, layer=0, input_ids_list=seqs, batch_size=2
    )
    assert inputs.shape == (10, 16)
    assert deltas.shape == (10, 16)


def test_top_mlp_features(model: NativeGPT) -> None:
    tok = NativeTokenizer()
    ids = ids_from_text(tok, "The capital of France is")
    nodes = top_mlp_features(model, ids, target_position=-1, topk=3)
    assert nodes
    for layer, kind, index in nodes:
        assert kind == "mlp"
        assert layer in (0, 1)
        assert isinstance(index, int)


def test_top_attention_heads(model: NativeGPT) -> None:
    tok = NativeTokenizer()
    ids = ids_from_text(tok, "The capital of France is")
    heads = top_attention_heads(model, ids, target_position=-1, topk=2)
    assert heads
    for layer, kind, index in heads:
        assert kind == "attn"
        assert layer in (0, 1)
        assert index in (0, 1, 2, 3)


def test_feature_circuit_build(model: NativeGPT, tokenizer: NativeTokenizer) -> None:
    ids = ids_from_text(tokenizer, "The capital of France is")
    ids2d = ids.unsqueeze(0)
    with tensorplay.no_grad():
        clean_logits = model(input_ids=ids2d).logits
    target_id = int(clean_logits[0, -1].argmax())

    def target_fn(out) -> float:
        return float(out.logits[0, -1, target_id])

    mlp_nodes = top_mlp_features(model, ids, target_position=-1, topk=4)
    attn_nodes = top_attention_heads(model, ids, target_position=-1, topk=2)
    circuit = FeatureCircuit(model, target_fn)
    data = circuit.build(ids, nodes=mlp_nodes + attn_nodes, topk=6, target=target_id)
    assert len(data.nodes) > 0
    assert all(n.direct_effect > 0 for n in data.nodes)
    for edge in data.edges:
        assert edge.weight > 0
        assert 0 <= edge.source < len(data.nodes)
        assert 0 <= edge.target < len(data.nodes)


def test_feature_circuit_visualize(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    import matplotlib.pyplot as plt

    ids = ids_from_text(tokenizer, "The capital of France is")
    ids2d = ids.unsqueeze(0)
    with tensorplay.no_grad():
        clean_logits = model(input_ids=ids2d).logits
    target_id = int(clean_logits[0, -1].argmax())
    circuit = FeatureCircuit(
        model, lambda out: float(out.logits[0, -1, target_id])
    )
    nodes = top_mlp_features(model, ids, target_position=-1, topk=3)
    data = circuit.build(ids, nodes=nodes, topk=4, target=target_id)
    fig, ax = circuit.visualize(data, use_pyplot=False)
    plt.close(fig)
    assert ax is not None