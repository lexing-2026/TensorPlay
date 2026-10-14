"""Tests for the LLM research helpers (attention maps + activation patching).

All tests run natively on a small transformer built with tensorplay primitives;
no third-party runtime or pretrained checkpoint is involved.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

import tensorplay as tensorplay
from tensorplay import nn

from tensorplay.lumen.llm import (
    ActivationPatching,
    extract_attention,
    visualize_attention_heads,
)


def _causal_mask(seq: int) -> tensorplay.Tensor:
    return tensorplay.zeros(seq, seq, dtype=tensorplay.float32).masked_fill(
        tensorplay.triu(tensorplay.ones(seq, seq, dtype=tensorplay.bool), diagonal=1),
        float("-inf"),
    )


class Block(nn.Module):
    def __init__(self, d: int = 16, heads: int = 4, ff: int = 32) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(
        self, x: tensorplay.Tensor, mask: tensorplay.Tensor
    ) -> Tuple[tensorplay.Tensor, tensorplay.Tensor]:
        h = self.ln1(x)
        o, w = self.attn(
            h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False
        )
        x = x + o
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x, w


class NativeGPT(nn.Module):
    def __init__(
        self, vocab: int = 16, d: int = 16, heads: int = 4, layers: int = 2, ff: int = 32
    ) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, d)
        self.transformer.wpe = nn.Embedding(16, d)
        self.transformer.h = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.transformer.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(
        self, input_ids: tensorplay.Tensor, output_attentions: bool = False
    ) -> Any:
        seq = input_ids.shape[1]
        pos = tensorplay.arange(0, seq, dtype=tensorplay.int64).unsqueeze(0)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        mask = _causal_mask(seq)
        attentions = []
        for blk in self.transformer.h:
            x, w = blk(x, mask)
            attentions.append(w)
        x = self.transformer.ln_f(x)
        return SimpleNamespace(
            logits=self.lm_head(x),
            attentions=tuple(attentions) if output_attentions else None,
        )

    def generate(
        self, input_ids: tensorplay.Tensor, max_new_tokens: int = 5, **kwargs: Any
    ) -> tensorplay.Tensor:
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            next_id = self.forward(ids).logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = tensorplay.cat([ids, next_id], dim=1)
        return ids


VOCAB = ["<pad>", "<unk>", "The", "capital", "of", "France", "England", "is", "Paris", "London", "<eos>"]


class NativeTokenizer:
    def __init__(self) -> None:
        self.id2tok = {i: t for i, t in enumerate(VOCAB)}
        self.tok2id = {t: i for i, t in enumerate(VOCAB)}

    def __call__(
        self,
        text,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Any:
        if isinstance(text, str):
            words = text.split()
            ids = [self.tok2id.get(w, 1) for w in words]
            res: Dict[str, Any] = {"input_ids": ids}
            if return_offsets_mapping:
                offsets, pos = [], 0
                for w in words:
                    start = text.index(w, pos)
                    offsets.append((start, start + len(w)))
                    pos = start + len(w)
                res["offset_mapping"] = offsets
            return res
        return [
            self.__call__(t, return_offsets_mapping, add_special_tokens, **kwargs)
            for t in text
        ]

    def encode(
        self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False
    ) -> Any:
        ids = [self.tok2id.get(w, 1) for w in text.split()]
        if return_tensors == "pt":
            return tensorplay.tensor([ids], dtype=tensorplay.int64)
        return ids

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self.id2tok[int(i)] for i in ids]

    def convert_tokens_to_ids(self, toks: Sequence[str]) -> List[int]:
        return [self.tok2id[t] for t in toks]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.convert_ids_to_tokens(ids))


@pytest.fixture(scope="module")
def model() -> NativeGPT:
    return NativeGPT()


@pytest.fixture(scope="module")
def tokenizer() -> NativeTokenizer:
    return NativeTokenizer()


def test_visualize_attention_heads() -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    attn = rng.dirichlet(np.ones(6), size=(3, 6))
    fig, axes = visualize_attention_heads(
        attn, tokens=[f"w{i}" for i in range(6)], heads=[0, 2], use_pyplot=False
    )
    plt.close(fig)
    assert len(np.atleast_1d(axes)) == 2


def test_extract_attention_native(model: NativeGPT, tokenizer: NativeTokenizer) -> None:
    result = extract_attention(
        model, tokenizer, "The capital of France is", layers=[0, 1]
    )
    assert len(result["attention"]) == 2
    assert result["attention"][0].shape == (1, 4, 5, 5)
    assert result["tokens"] == ["The", "capital", "of", "France", "is"]
    assert result["layers"] == [0, 1]


def test_hidden_states_shapes(model: NativeGPT) -> None:
    patcher = ActivationPatching(model)
    states = patcher.hidden_states(tensorplay.tensor([[2, 3, 4, 5]]))
    assert sorted(states) == [0, 1]
    assert states[0].shape == (1, 4, 16)


def test_intervene_reproduces_clean(model: NativeGPT) -> None:
    ids = tensorplay.tensor([[2, 3, 4, 5]])
    patcher = ActivationPatching(model)
    states = patcher.hidden_states(ids)
    out = patcher.intervene(ids, layer=0, replacement=states[0])
    with tensorplay.no_grad():
        expected = model(input_ids=ids).logits
    assert tensorplay.allclose(out.logits, expected, atol=1e-6)


def test_ablate_changes_logits(model: NativeGPT) -> None:
    ids = tensorplay.tensor([[2, 3, 4, 5]])
    patcher = ActivationPatching(model)
    out = patcher.ablate(ids, layer=1, positions=[2])
    with tensorplay.no_grad():
        expected = model(input_ids=ids).logits
    assert not tensorplay.allclose(out.logits, expected, atol=1e-6)


def test_causal_trace_recovery(model: NativeGPT, tokenizer: NativeTokenizer) -> None:
    clean = tensorplay.tensor(
        [tokenizer.tok2id[w] for w in "The capital of France is".split()]
    )
    corrupted = tensorplay.tensor(
        [tokenizer.tok2id[w] for w in "The capital of England is".split()]
    )
    patcher = ActivationPatching(model)
    trace = patcher.causal_trace(clean, corrupted, target_position=-1)
    assert trace["scores"].shape == (2, 5)
    assert trace["layers"] == [0, 1]
    assert trace["positions"] == [0, 1, 2, 3, 4]
    # restoring the clean hidden state at the final layer and the target
    # position fully recovers the clean logit.
    assert abs(trace["scores"][1, 4] - 1.0) < 1e-5


def test_llm_gradient_attribution_native(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    from tensorplay.lumen import LayerIntegratedGradients, LLMGradientAttribution, TextTokenInput

    ig = LayerIntegratedGradients(model, model.transformer.wte)
    llm_attr = LLMGradientAttribution(ig, tokenizer)
    result = llm_attr.attribute(
        TextTokenInput("The capital of France is", tokenizer),
        target="Paris",
        n_steps=5,
    )
    assert result.input_tokens == ["The", "capital", "of", "France", "is"]
    assert result.output_tokens == ["Paris"]
    assert result.seq_attr.shape == (5,)
    assert tensorplay.isfinite(result.seq_attr).all()
    assert result.token_attr.shape == (1, 5)

def test_visualize_attention_multi_layer(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    import matplotlib.pyplot as plt

    from tensorplay.lumen.llm import (
        extract_attention,
        visualize_attention_multi_layer,
    )

    result = extract_attention(model, tokenizer, "The capital of France is")
    fig, axes = visualize_attention_multi_layer(
        result,
        query_token="France",
        title="All layers",
        use_pyplot=False,
    )
    plt.close(fig)
    # 2 layers x 4 heads grid
    assert np.asarray(axes).size == 8


def test_token_attention_and_visualize(
    model: NativeGPT, tokenizer: NativeTokenizer
) -> None:
    import matplotlib.pyplot as plt

    from tensorplay.lumen.llm import (
        extract_attention,
        token_attention,
        visualize_token_attention,
    )

    result = extract_attention(model, tokenizer, "The capital of France is")
    ta = token_attention(result, query_token="France")
    assert ta["matrix"].shape == (2, 4, 5)
    assert ta["query_position"] == 3
    assert ta["query_token"] == "France"
    # attention from France sums to ~1 per layer/head
    assert abs(ta["matrix"].sum(axis=-1) - 1.0).max() < 1e-4

    fig, ax = visualize_token_attention(
        ta, title="What France attends to", use_pyplot=False
    )
    plt.close(fig)
    assert ax is not None
