# mypy: allow-untyped-defs
"""Multi-head latent attention.

The layer factorizes the per-head key/value projections into two stages:
a shared down-projection to a low-rank latent, and per-head up-projections
applied after the cache. Only the latent and a single shared RoPE key are
stored per layer, so the KV cache holds ``kv_lora_rank + qk_rope_head_dim``
numbers per position regardless of ``num_heads`` — for typical head counts
an order of magnitude less than full multi-head attention.

Queries optionally pass through their own low-rank bottleneck (query LoRA)
before the per-head projection.
"""

from __future__ import annotations

from typing import Optional

import tensorplay
from tensorplay import Tensor

from .linear import Linear
from .module import Module
from .normalization import RMSNorm
from .rotary import RotaryEmbedding, apply_rotary_emb

__all__ = ["MultiheadLatentAttention"]


class MultiheadLatentAttention(Module):
    """Causal attention with a compressed key/value latent cache.

    Args:
        embed_dim: model width of the input and output.
        num_heads: number of query heads.
        kv_lora_rank: width of the compressed key/value latent.
        qk_nope_head_dim: per-head key/query width that carries content only.
        qk_rope_head_dim: per-head query/key width carrying position; a single
            shared key is rotated with it and cached next to the latent.
        v_head_dim: value width per head; defaults to ``qk_nope_head_dim``.
        q_lora_rank: query low-rank bottleneck width; ``None`` projects
            queries directly.
        dropout: attention dropout, applied only in training mode.
        bias: whether the query and latent-input projections carry a bias.
        rope_base: base of the rotary frequency progression.
        rope_interleaved: rotary pair layout; see :class:`RotaryEmbedding`.
        layer_idx: cache slot this layer reads and writes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
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
        if qk_rope_head_dim % 2:
            raise ValueError(
                f"qk_rope_head_dim must be even for rotary embedding, got {qk_rope_head_dim}"
            )
        if v_head_dim is None:
            v_head_dim = qk_nope_head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.dropout = dropout
        self.layer_idx = layer_idx
        self.scaling = self.qk_head_dim**-0.5

        if q_lora_rank is None:
            self.q_proj = Linear(embed_dim, num_heads * self.qk_head_dim, bias=bias, **factory_kwargs)
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = Linear(embed_dim, q_lora_rank, bias=bias, **factory_kwargs)
            self.q_a_layernorm = RMSNorm(q_lora_rank, **factory_kwargs)
            self.q_b_proj = Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False, **factory_kwargs)

        self.kv_a_proj_with_mqa = Linear(
            embed_dim, kv_lora_rank + qk_rope_head_dim, bias=bias, **factory_kwargs
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, **factory_kwargs)
        self.kv_b_proj = Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False, **factory_kwargs
        )
        self.o_proj = Linear(num_heads * v_head_dim, embed_dim, bias=bias, **factory_kwargs)

        self.rope = RotaryEmbedding(
            qk_rope_head_dim, base=rope_base, interleaved=rope_interleaved, device=device
        )

    def _expand_latents(self, kv_latent: Tensor, k_rot: Tensor) -> tuple[Tensor, Tensor]:
        """Up-project the cached latents into per-head keys and values."""
        bsz, _, seq_len, _ = kv_latent.shape
        kv = self.kv_b_proj(kv_latent).view(
            bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = tensorplay.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)
        return tensorplay.cat([k_nope, k_rot], dim=-1), v

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Causal attention over the expanded heads.

        The fused path applies only when value heads share the query/key
        width. Otherwise the attention is composed explicitly with absolute
        positions, so a decode row (query offset ``kv_len - q_len``) is
        unmasked while a full prefill gets the triangular pattern.
        """
        q_len, kv_len = q.shape[2], k.shape[2]
        if v.shape[-1] == q.shape[-1]:
            return tensorplay.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=q_len == kv_len,
                scale=self.scaling,
            )
        query_pos = tensorplay.arange(kv_len - q_len, kv_len, device=q.device)
        key_pos = tensorplay.arange(kv_len, device=q.device)
        allowed = (key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        scores = tensorplay.matmul(q, k.transpose(-1, -2)) * self.scaling
        scores = tensorplay.where(
            allowed,
            scores,
            tensorplay.full((), float("-inf"), dtype=scores.dtype),
        )
        attn = tensorplay.nn.functional.softmax(scores, dim=-1)
        return tensorplay.matmul(attn, v)

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
            past_key_values: optional cache. The cache pair is the *compressed*
                latent ``[batch, 1, seq, kv_lora_rank]`` and the shared RoPE key
                ``[batch, 1, seq, qk_rope_head_dim]``; up-projection happens
                after the cache read.

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

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = tensorplay.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rot = tensorplay.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        # Both latents stay single-head 4-D so cache layers treat them like
        # ordinary [batch, heads, seq, dim] pairs.
        kv_latent = kv_latent.view(bsz, 1, seq_len, self.kv_lora_rank)
        k_rot = k_rot.view(bsz, 1, seq_len, self.qk_rope_head_dim)

        cos, sin = self.rope(hidden_states, position_ids)
        q_rot, k_rot = apply_rotary_emb(q_rot, k_rot, cos, sin, self.rope.interleaved)

        if past_key_values is not None:
            kv_latent, k_rot = past_key_values.update(kv_latent, k_rot, self.layer_idx)

        q = tensorplay.cat([q_pass, q_rot], dim=-1)
        k, v = self._expand_latents(kv_latent, k_rot)

        attn = self._attend(q, k, v)
        attn = attn.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"kv_lora_rank={self.kv_lora_rank}, "
            f"qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, "
            f"v_head_dim={self.v_head_dim}, q_lora_rank={self.q_lora_rank}, "
            f"layer_idx={self.layer_idx}"
        )
