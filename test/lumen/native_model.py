"""Shared native transformer + tokenizer used by lumen LLM tests.

Kept in the test tree (not the library) so the tools can be exercised on a
real tensorplay model without any pretrained checkpoint or third-party runtime.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tensorplay as tensorplay
from tensorplay import nn

VOCAB = ["<pad>", "<unk>", "The", "capital", "of", "France", "England", "is", "Paris", "London", "<eos>"]


def causal_mask(seq: int) -> tensorplay.Tensor:
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
        mask = causal_mask(seq)
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

    @property
    def pad_token_id(self) -> int:
        return 0


def ids_from_text(tokenizer: NativeTokenizer, text: str) -> tensorplay.Tensor:
    return tensorplay.tensor(
        [tokenizer.tok2id[w] for w in text.split()], dtype=tensorplay.int64
    )