# mypy: allow-untyped-defs
"""Compressed Sparse Attention.

Long-context mixer that keeps two KV tiers:

* a sliding window of recent tokens attended with full precision (a single
  shared KV head, ``k == v``);
* one compressed entry per ``compress_rate`` source tokens, produced by a
  softmax-gated window reduction and injected as extra key/value slots.

A *Lightning indexer* — a scaled-down copy of the same window compressor plus
a small scoring head ``sum_h w_{t,h} * ReLU(q_{t,h} . K^IComp_s)`` — picks the
at most ``index_topk`` compressed entries each query may see. Everything else
is masked to ``-inf`` through an additive block bias, on top of the sliding
window's causality+distance mask.

Queries and keys use partial rotary embeddings; the attention output's rope
slice is rotated back with ``-sin`` at the query position, which makes each
KV entry's contribution depend only on its distance to the query.

When a KV cache is passed, its sliding part holds the recent window and a
``csa_states`` dict attached to the cache object carries the compressor's
tail buffer, overlap slices and already-compressed entries across calls.
"""

from __future__ import annotations

from typing import Optional

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import Tensor

from .linear import Linear
from .module import Module
from .normalization import RMSNorm
from ..parameter import Parameter
from .rotary import RotaryEmbedding, apply_rotary_emb

__all__ = ["CompressedSparseAttention"]


class _WindowCompressor(Module):
    """Softmax-gated window reduction shared by the KV compressor and indexer.

    Every ``compress_rate`` source tokens form one window. Each token
    contributes two series stored in one ``2 * head_dim`` projection: ``Cb``
    (``[..., head_dim:]``) votes for *its own* window's entry, ``Ca``
    (``[..., :head_dim]``) for the *next* window's entry. A window's entry is
    the softmax-gated convex combination of ``2 * compress_rate`` slots — its
    own ``Cb`` votes plus the previous window's ``Ca`` votes — normalized and
    rotated to the window's start position. The first window of the very
    first call has an empty ``Ca`` half (zero keys, ``-inf`` gates) which
    softmax turns into weight 0.
    """

    def __init__(
        self, hidden_size: int, head_dim: int, compress_rate: int, rms_norm_eps: float = 1e-6,
        device=None, dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.head_dim = head_dim
        self.compress_rate = compress_rate
        self.kv_proj = Linear(hidden_size, 2 * head_dim, bias=False, **factory_kwargs)
        self.gate_proj = Linear(hidden_size, 2 * head_dim, bias=False, **factory_kwargs)
        self.position_bias = Parameter(
            tensorplay.zeros(compress_rate, 2 * head_dim, **factory_kwargs)
        )
        self.kv_norm = RMSNorm(head_dim, eps=rms_norm_eps, **factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        rope: RotaryEmbedding,
        seq_start: int,
        state: dict,
    ) -> tuple[Optional[Tensor], dict]:
        """Compress complete windows of the new tokens.

        Args:
            hidden_states: ``[batch, seq, hidden_size]`` new tokens only.
            rope: rotary used to rotate each window's entry to its start.
            seq_start: absolute position of ``hidden_states[0]``.
            state: mutable dict carrying ``tail_kv``/``tail_gate``/``tail_start``,
                ``overlap_kv``/``overlap_gate`` and post-rope ``entries``.

        Returns:
            ``(entries, state)`` — ``entries`` is ``[batch, n, head_dim]``
            (``None`` before the first complete window), ``state`` is updated
            in place.
        """
        batch, seq_len, _ = hidden_states.shape
        ratio = self.compress_rate
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        tail_kv = state.get("tail_kv")
        if tail_kv is not None:
            full_kv = tensorplay.cat([tail_kv, kv], dim=1)
            full_gate = tensorplay.cat([state["tail_gate"], gate], dim=1)
            base = state["tail_start"]
        else:
            full_kv, full_gate, base = kv, gate, seq_start

        total = full_kv.shape[1]
        n_windows = total // ratio
        usable = n_windows * ratio
        chunk_kv = full_kv[:, :usable]
        chunk_gate = full_gate[:, :usable]
        new_state = {
            "tail_kv": full_kv[:, usable:],
            "tail_gate": full_gate[:, usable:],
            "tail_start": base + usable,
        }
        entries = state.get("entries")
        if n_windows == 0:
            # No window completed in this call: carry the accumulated entries,
            # their end positions and the overlap slices forward untouched.
            for carried in ("entries", "entry_end", "overlap_kv", "overlap_gate"):
                if carried in state:
                    new_state[carried] = state[carried]
            return entries, new_state

        chunk_kv = chunk_kv.view(batch, n_windows, ratio, 2 * self.head_dim)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, 2 * self.head_dim) + self.position_bias

        window_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        window_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
        window_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
        window_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
        if n_windows > 1:
            window_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            window_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        overlap_kv = state.get("overlap_kv")
        if overlap_kv is not None:
            window_kv[:, 0, :ratio] = overlap_kv
            window_gate[:, 0, :ratio] = state["overlap_gate"]

        # Softmax in float32: bf16 logits can collapse near-tied gate pairs.
        gated = (window_kv.float() * window_gate.float().softmax(dim=2)).sum(dim=2)
        compressed = self.kv_norm(gated.to(chunk_kv.dtype))

        positions = (
            tensorplay.arange(n_windows, device=compressed.device) * ratio + base
        ).unsqueeze(0).expand(batch, -1)
        cos, sin = rope(compressed, positions)
        compressed, _ = apply_rotary_emb(
            compressed.unsqueeze(1), compressed.unsqueeze(1), cos, sin
        )
        compressed = compressed.squeeze(1)

        entries = compressed if entries is None else tensorplay.cat([entries, compressed], dim=1)
        # Window start positions are batch-uniform (absolute positions are
        # shared across rows), so track their ends as a single [n] tensor.
        entry_end = positions[0] + ratio - 1
        prior_end = state.get("entry_end")
        if prior_end is not None:
            entry_end = tensorplay.cat([prior_end, entry_end])
        new_state["overlap_kv"] = chunk_kv[:, -1, :, : self.head_dim]
        new_state["overlap_gate"] = chunk_gate[:, -1, :, : self.head_dim]
        new_state["entries"] = entries
        new_state["entry_end"] = entry_end
        return entries, new_state


class _LightningIndexer(Module):
    """Top-``index_topk`` selection over compressed entries per query."""

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        compress_rate: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rms_norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.softmax_scale = index_head_dim**-0.5
        self.weights_scaling = index_n_heads**-0.5
        self.compressor = _WindowCompressor(
            hidden_size, index_head_dim, compress_rate, rms_norm_eps, **factory_kwargs
        )
        self.q_b_proj = Linear(
            q_lora_rank, index_n_heads * index_head_dim, bias=False, **factory_kwargs
        )
        self.weights_proj = Linear(hidden_size, index_n_heads, bias=False, **factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        q_residual: Tensor,
        position_ids: Tensor,
        rope: RotaryEmbedding,
        seq_start: int,
        state: dict,
    ) -> tuple[dict, Optional[Tensor]]:
        """Score compressed entries for every query and keep the top ones.

        Returns ``(state, topk_indices)``; ``topk_indices`` is
        ``[batch, seq, k]`` or ``None`` while no entry exists yet.
        """
        batch, seq_len, _ = hidden_states.shape
        entries, state = self.compressor(hidden_states, rope, seq_start, state)
        if entries is None or entries.shape[1] == 0:
            return state, None

        q = self.q_b_proj(q_residual).view(
            batch, seq_len, self.index_n_heads, self.index_head_dim
        ).transpose(1, 2)
        cos_q, sin_q = rope(hidden_states, position_ids)
        q, _ = apply_rotary_emb(q, q, cos_q, sin_q)
        q = q.transpose(1, 2)  # [batch, seq, heads, index_head_dim]

        scores = tensorplay.matmul(q.float(), entries.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [batch, seq, n]

        # An entry is visible to a query only once its last source token lies
        # in that query's past.
        entry_end = state["entry_end"]
        future = entry_end.view(1, 1, -1) > position_ids.unsqueeze(-1)
        scores = scores.masked_fill(future, float("-inf"))
        k = min(self.index_topk, entries.shape[1])
        return state, tensorplay.topk(scores, k, dim=-1).indices


class CompressedSparseAttention(Module):
    """Sliding-window attention augmented with compressed sparse entries.

    Args:
        hidden_size: model width of the input and output.
        num_heads: number of query heads (the single shared KV head is
            broadcast to all of them).
        head_dim: per-head width.
        q_lora_rank: query low-rank bottleneck width.
        compress_rate: source tokens per compressed entry.
        window_size: sliding window width in tokens.
        rope_head_dim: rotary span per head; the tail dims stay unrotated.
            Defaults to ``head_dim``.
        index_n_heads / index_head_dim: indexer scoring head shapes.
        index_topk: compressed entries a query may attend to.
        rope_base: base of the rotary frequency progression.
        rms_norm_eps: epsilon of the normalization layers.
        layer_idx: cache slot this layer reads and writes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        q_lora_rank: int,
        compress_rate: int = 4,
        window_size: int = 128,
        rope_head_dim: Optional[int] = None,
        index_n_heads: int = 4,
        index_head_dim: int = 32,
        index_topk: int = 16,
        rope_base: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"embed_dim ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sliding_window = window_size
        self.compress_rate = compress_rate
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5

        self.q_a_proj = Linear(hidden_size, q_lora_rank, bias=False, **factory_kwargs)
        self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps, **factory_kwargs)
        self.q_b_proj = Linear(q_lora_rank, num_heads * head_dim, bias=False, **factory_kwargs)
        self.kv_proj = Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.kv_norm = RMSNorm(head_dim, eps=rms_norm_eps, **factory_kwargs)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False, **factory_kwargs)
        self.sinks = Parameter(tensorplay.zeros(num_heads, **factory_kwargs))

        self.rope = RotaryEmbedding(
            rope_head_dim if rope_head_dim is not None else head_dim,
            base=rope_base,
            device=device,
        )
        self.compressor = _WindowCompressor(
            hidden_size, head_dim, compress_rate, rms_norm_eps, **factory_kwargs
        )
        self.indexer = _LightningIndexer(
            hidden_size, q_lora_rank, compress_rate, index_n_heads, index_head_dim,
            index_topk, rms_norm_eps, **factory_kwargs,
        )

    def _layer_state(self, past_key_values) -> dict:
        states = getattr(past_key_values, "csa_states", None)
        if states is None:
            states = {}
            past_key_values.csa_states = states
        return states.setdefault(self.layer_idx, {"compressor": {}, "indexer": {}})

    def _block_bias(
        self, topk_indices: Tensor, entry_end: Tensor, position_ids: Tensor, num_entries: int
    ) -> Tensor:
        """``[batch, 1, seq, num_entries]`` additive bias: 0 where a query may
        read an entry (indexer-selected *and* fully in its past), ``-inf``
        elsewhere."""
        causal = entry_end.view(1, 1, -1) <= position_ids.unsqueeze(-1)
        selected = (
            (topk_indices.unsqueeze(-1) == tensorplay.arange(num_entries).view(1, 1, 1, -1))
            & (topk_indices >= 0).unsqueeze(-1)
        ).any(dim=2)
        bias = tensorplay.where(selected & causal, 0.0, float("-inf"))
        return bias.unsqueeze(1)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
        past_key_values=None,
    ) -> Tensor:
        """Attend over the sliding window plus the compressed entries.

        Args:
            hidden_states: ``[batch, seq, hidden_size]``.
            position_ids: ``[batch, seq]`` absolute positions; defaults to
                continuing after the cached window.
            past_key_values: optional :class:`~tensorplay.nn.cache.SlidingWindowCache`;
                compressor state piggybacks on it as ``csa_states``.

        Returns:
            ``[batch, seq, hidden_size]``.
        """
        batch, seq_len, _ = hidden_states.shape
        offset = 0
        if past_key_values is not None:
            offset = past_key_values.get_seq_length(self.layer_idx)
        if position_ids is None:
            position_ids = (
                tensorplay.arange(offset, offset + seq_len, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch, seq_len)
            )

        cos_q, sin_q = self.rope(hidden_states, position_ids)
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q, _ = apply_rotary_emb(q, q, cos_q, sin_q)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(
            batch, seq_len, 1, self.head_dim
        ).transpose(1, 2)
        kv, _ = apply_rotary_emb(kv, kv, cos_q, sin_q)

        if past_key_values is not None:
            kv, _ = past_key_values.update(kv, kv, self.layer_idx)
            kv_positions = past_key_values.positions[self.layer_idx]
        else:
            kv_positions = position_ids

        seq_start = int(position_ids[:, 0].max().item()) if seq_len else 0
        layer_state = self._layer_state(past_key_values) if past_key_values is not None else {
            "compressor": {}, "indexer": {}
        }
        entries, comp_state = self.compressor(
            hidden_states, self.rope, seq_start, layer_state["compressor"]
        )
        layer_state["compressor"] = comp_state
        index_state, topk_indices = self.indexer(
            hidden_states, q_residual, position_ids, self.rope, seq_start, layer_state["indexer"]
        )
        layer_state["indexer"] = index_state

        num_entries = 0 if entries is None else entries.shape[1]
        key = value = kv
        query_pos = position_ids.unsqueeze(-1)
        key_pos = kv_positions.unsqueeze(1)
        causal = key_pos <= query_pos
        within_window = (query_pos - key_pos) < self.sliding_window
        bias = tensorplay.where(causal & within_window, 0.0, float("-inf")).unsqueeze(1)
        if num_entries > 0:
            key = tensorplay.cat([kv, entries.unsqueeze(1)], dim=2)
            value = key
            entry_bias = self._block_bias(
                topk_indices, comp_state["entry_end"], position_ids, num_entries
            )
            bias = tensorplay.cat([bias, entry_bias.to(bias.dtype)], dim=-1)

        scores = tensorplay.matmul(q, key.transpose(-1, -2)) * self.scaling + bias
        # Attention sinks add one per-head logit with no value; the softmax
        # renormalizes without changing what is read.
        sink = self.sinks.to(scores.dtype).view(1, self.num_heads, 1, 1).expand(
            batch, self.num_heads, seq_len, 1
        )
        weights = tensorplay.cat([scores, sink], dim=-1).float().softmax(dim=-1)[..., :-1]
        attn = tensorplay.matmul(weights.to(q.dtype), value)

        # Undo the rope on the output's rope slice with the conjugate rotation
        # at the query position, so each entry's contribution depends only on
        # its distance to the query.
        attn, _ = apply_rotary_emb(attn, attn, cos_q, -sin_q)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, compress_rate={self.compress_rate}, "
            f"window_size={self.sliding_window}, layer_idx={self.layer_idx}"
        )
