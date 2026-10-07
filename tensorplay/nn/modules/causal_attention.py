# mypy: allow-untyped-defs
"""Causal attention with grouped query heads.

``GroupedQueryAttention`` projects the input into ``num_heads`` query heads
while sharing ``num_kv_heads`` key/value heads across them, which shrinks the
KV cache by ``num_heads / num_kv_heads`` without widening the output space.
Setting ``num_kv_heads == num_heads`` gives full multi-head attention and
``num_kv_heads == 1`` gives multi-query attention.

Optional per-head RMS normalization is applied to the query and key vectors
*before* the rotary transform: the normalization sees each position's own head
vector, so the rotary rotation of later positions does not mix into it.
"""

from __future__ import annotations

from typing import Optional

import tensorplay
from tensorplay import Tensor

from .linear import Linear
from .module import Module
from .normalization import RMSNorm
from .rotary import RotaryEmbedding, apply_rotary_emb

__all__ = ["GroupedQueryAttention"]


class GroupedQueryAttention(Module):
    """Causal self-attention with shared key/value head groups.

    Args:
        embed_dim: model width of the input and output.
        num_heads: number of query heads.
        num_kv_heads: number of key/value heads; must divide ``num_heads``.
            Defaults to ``num_heads``.
        head_dim: per-head width. Defaults to ``embed_dim // num_heads``; when
            overridden, the input and output projections still use
            ``embed_dim`` and the attention works on ``num_heads * head_dim``.
        dropout: attention dropout, applied only in training mode.
        bias: whether the projections carry a bias.
        qk_norm: apply RMS normalization to each head's query/key vector.
        rms_norm_eps: epsilon for the optional query/key normalization.
        rope_base: base of the rotary frequency progression.
        rope_interleaved: rotary pair layout; see :class:`RotaryEmbedding`.
        layer_idx: cache slot this layer reads and writes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        rope_interleaved: bool = False,
        layer_idx: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        if head_dim is None:
            head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5

        self.q_proj = Linear(embed_dim, num_heads * head_dim, bias=bias, **factory_kwargs)
        self.k_proj = Linear(embed_dim, num_kv_heads * head_dim, bias=bias, **factory_kwargs)
        self.v_proj = Linear(embed_dim, num_kv_heads * head_dim, bias=bias, **factory_kwargs)
        self.o_proj = Linear(num_heads * head_dim, embed_dim, bias=bias, **factory_kwargs)

        self.qk_norm = qk_norm
        if qk_norm:
            # Normalization spans one head vector only, so it stays valid under
            # any head grouping and after the cache concatenation.
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, **factory_kwargs)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, **factory_kwargs)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope = RotaryEmbedding(
            head_dim, base=rope_base, interleaved=rope_interleaved, device=device
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
        past_key_values=None,
    ) -> Tensor:
        """Attend over the sequence.

        Args:
            hidden_states: ``[batch, seq, embed_dim]``.
            position_ids: ``[batch, seq]`` absolute positions; defaults to
                continuing after the cached prefix.
            past_key_values: optional cache; its ``update`` receives the
                projected key/value heads for this layer.

        Returns:
            ``[batch, seq, embed_dim]``.
        """
        bsz, seq_len, _ = hidden_states.shape
        offset = 0
        if past_key_values is not None:
            offset = past_key_values.get_seq_length(self.layer_idx)
        if position_ids is None:
            position_ids = (
                tensorplay.arange(offset, offset + seq_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, seq_len)
            )

        q = (
            self.q_proj(hidden_states)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rope(hidden_states, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin, self.rope.interleaved)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # Top-left causal alignment only holds when the query covers the whole
        # cached prefix; a decode step (one query against a longer cache) must
        # attend to every cached position unmasked.
        attn = tensorplay.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=q.shape[2] == k.shape[2],
            scale=self.scaling,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        attn = attn.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"qk_norm={self.qk_norm}, layer_idx={self.layer_idx}"
        )
