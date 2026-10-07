# mypy: allow-untyped-defs
"""Causal encoder-decoder scaffold.

Asymmetric two-stack architecture in the style of flash encoder-decoder
models: a causal encoder digests the context, and a causal decoder generates
against it, fusing the encoded memory through cross attention after every
decoder block. Both stacks are built from :class:`TransformerBlock`, so the
mixers (attention, linear attention, compressed attention) are swappable.

This is the structural scaffold; per-layer recipe choices (which mixer per
layer, sparse routing between the stacks) are caller-supplied via
``mixer_factory``.
"""

from __future__ import annotations

from typing import Callable, Optional

import tensorplay
from tensorplay import Tensor

from .causal_attention import GroupedQueryAttention
from .container import ModuleList
from .linear import Linear
from .module import Module
from .modern_block import TransformerBlock
from .normalization import RMSNorm

__all__ = ["CrossAttention", "CausalEncoderDecoder"]


class CrossAttention(Module):
    """Decoder-to-encoder attention: queries from the decoder stream, keys and
    values from the encoded memory. No causal mask — the encoder outputs are
    already causal representations."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"embed_dim ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if head_dim is None:
            head_dim = hidden_size // num_heads
        if kv_dim is None:
            kv_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=bias, **factory_kwargs)
        self.k_proj = Linear(kv_dim, num_heads * head_dim, bias=bias, **factory_kwargs)
        self.v_proj = Linear(kv_dim, num_heads * head_dim, bias=bias, **factory_kwargs)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        """``x``: ``[batch, seq, hidden_size]``; ``memory``:
        ``[batch, enc_seq, kv_dim]``."""
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        enc_len = memory.shape[1]
        k = self.k_proj(memory).view(bsz, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(bsz, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = tensorplay.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scaling
        )
        attn = attn.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn)


class CausalEncoderDecoder(Module):
    """Causal encoder stack + causal decoder stack + per-layer cross attention.

    Args:
        hidden_size: model width throughout.
        num_heads: query heads of the default mixers and cross attention.
        n_encoder_layers / n_decoder_layers: stack depths.
        mixer_factory: optional ``mixer_factory(hidden_size, layer_idx) -> Module``
            with the TransformerBlock mixer signature; defaults to full
            multi-head attention.
        num_kv_heads: key/value heads of the default mixers.
        mlp_intermediate_size: feed-forward width shared by both stacks.
        rms_norm_eps: epsilon of the block pre-norms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        mixer_factory: Optional[Callable[[int, int], Module]] = None,
        num_kv_heads: Optional[int] = None,
        mlp_intermediate_size: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if mixer_factory is None:
            def mixer_factory(hidden_size, layer_idx):
                return GroupedQueryAttention(
                    hidden_size, num_heads, num_kv_heads=num_kv_heads,
                    layer_idx=layer_idx, **factory_kwargs,
                )

        self.hidden_size = hidden_size
        self.encoder = ModuleList(
            TransformerBlock(
                hidden_size,
                mixer_factory(hidden_size, i),
                mlp_intermediate_size=mlp_intermediate_size,
                rms_norm_eps=rms_norm_eps,
                **factory_kwargs,
            )
            for i in range(n_encoder_layers)
        )
        self.decoder = ModuleList(
            TransformerBlock(
                hidden_size,
                mixer_factory(hidden_size, i),
                mlp_intermediate_size=mlp_intermediate_size,
                rms_norm_eps=rms_norm_eps,
                **factory_kwargs,
            )
            for i in range(n_decoder_layers)
        )
        self.cross_attention = ModuleList(
            CrossAttention(hidden_size, num_heads, **factory_kwargs)
            for _ in range(n_decoder_layers)
        )
        self.memory_proj = Linear(hidden_size, hidden_size, bias=False, **factory_kwargs)
        self.encoder_final_norm = RMSNorm(hidden_size, eps=rms_norm_eps, **factory_kwargs)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Encode-then-decode.

        Args:
            hidden_states: ``[batch, seq, hidden_size]`` source tokens.

        Returns:
            ``[batch, seq, hidden_size]`` decoder states.
        """
        memory = hidden_states
        for block in self.encoder:
            memory = block(memory)
        memory = self.memory_proj(self.encoder_final_norm(memory))

        decoder_states = memory
        for block, cross_attn in zip(self.decoder, self.cross_attention, strict=True):
            decoder_states = block(decoder_states)
            decoder_states = decoder_states + cross_attn(decoder_states, memory)
        return decoder_states
