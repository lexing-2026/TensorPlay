"""Tests for the LLM research helpers (attention maps + activation patching).

The tests use small synthetic models so they run offline without downloading
any pretrained checkpoint.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tensorplay.lumen.llm import (
    ActivationPatching,
    extract_attention,
    visualize_attention_heads,
)


class SyntheticTransformer(nn.Module):
    """A tiny causal-LM-like module: embedding -> residual MLP blocks -> head.

    Layer outputs are plain tensors (not tuples), which exercises the
    non-tuple branch of the patching hooks.
    """

    def __init__(self, vocab: int = 20, hidden: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> Any:
        h = self.embed(input_ids)
        for layer in self.transformer.h:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


class FakeOutput:
    def __init__(self, attentions: Tuple[torch.Tensor, ...]) -> None:
        self.attentions = attentions


class FakeAttentionModel:
    def __init__(self, layers: int = 2, heads: int = 2, seq: int = 5) -> None:
        self._attentions = tuple(
            torch.rand(1, heads, seq, seq) for _ in range(layers)
        )
        self._param = nn.Parameter(torch.zeros(1))

    def parameters(self) -> Any:
        return iter([self._param])

    def eval(self) -> "FakeAttentionModel":
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> FakeOutput:
        return FakeOutput(self._attentions)


class FakeTokenizer:
    def __init__(self, vocab: int = 20, seq: int = 5) -> None:
        self.vocab = vocab
        self.seq = seq

    def __call__(
        self,
        text,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        ids = torch.randint(0, self.vocab, (len(text), self.seq))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return ["t%d" % i for i in ids]


def test_visualize_attention_heads() -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    attn = rng.dirichlet(np.ones(6), size=(3, 6))
    fig, axes = visualize_attention_heads(
        attn,
        tokens=[f"w{i}" for i in range(6)],
        heads=[0, 2],
        use_pyplot=False,
    )
    plt.close(fig)
    assert len(np.atleast_1d(axes)) == 2


def test_extract_attention_synthetic() -> None:
    model = FakeAttentionModel(layers=2, heads=3, seq=4)
    tokenizer = FakeTokenizer(seq=4)
    result = extract_attention(model, tokenizer, "hello")
    assert len(result["attention"]) == 2
    assert result["attention"][0].shape == (1, 3, 4, 4)
    assert len(result["tokens"]) == 4
    assert result["layers"] == [0, 1]
    assert result["heads"] == [0, 1, 2]


def test_hidden_states_shapes() -> None:
    model = SyntheticTransformer()
    patcher = ActivationPatching(model)
    states = patcher.hidden_states(torch.tensor([[1, 2, 3, 4]]))
    assert sorted(states) == [0, 1, 2]
    assert states[0].shape == (1, 4, 8)


def test_intervene_reproduces_clean() -> None:
    model = SyntheticTransformer()
    ids = torch.tensor([[1, 2, 3, 4]])
    patcher = ActivationPatching(model)
    states = patcher.hidden_states(ids)
    # replacing the whole layer-0 hidden state with the clean one must
    # reproduce the clean logits exactly.
    out = patcher.intervene(ids, layer=0, replacement=states[0])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    assert torch.allclose(out.logits, expected, atol=1e-6)


def test_ablate_changes_logits() -> None:
    model = SyntheticTransformer()
    ids = torch.tensor([[1, 2, 3, 4]])
    patcher = ActivationPatching(model)
    out = patcher.ablate(ids, layer=1, positions=[2])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    assert not torch.allclose(out.logits, expected, atol=1e-6)


def test_causal_trace_recovery() -> None:
    model = SyntheticTransformer()
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupted = torch.tensor([[5, 6, 7, 8]])
    patcher = ActivationPatching(model)
    trace = patcher.causal_trace(clean, corrupted, target_position=-1)
    scores = trace["scores"]
    assert scores.shape == (3, 4)
    assert trace["layers"] == [0, 1, 2]
    assert trace["positions"] == [0, 1, 2, 3]
    # restoring the clean hidden state at the final layer and the target
    # position fully recovers the clean logit (score ~ 1).
    assert abs(scores[2, 3] - 1.0) < 1e-5