# mypy: allow-untyped-defs
"""Fine-grained gated delta rule linear attention.

A variant of the gated delta rule mixer (see
:mod:`tensorplay.nn.modules.gated_delta`) where the decay gate is resolved per
*channel* of the key vector instead of one scalar per head: the forget gate is
produced by a low-rank projection and lives on ``[batch, seq, heads, key_dim]``.
Per-channel decay lets different key coordinates hold associations with
different time constants; the delta-rule write itself is unchanged.

The chunked path applies the decay differences inside the similarity scores —
``(q_i * k_j)`` weighted by ``exp(g_i - g_j)`` per channel — and folds each
chunk's updates into a unit lower triangular system solved by forward
substitution, exactly mirroring the head-scalar variant's structure.
"""

from __future__ import annotations

from typing import Optional

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import Tensor

from .gated_delta import GatedRMSNorm, _causal_conv1d, _lower_recurrence, l2norm
from .linear import Linear
from .module import Module
from ..parameter import Parameter

__all__ = ["KimiDeltaAttention", "chunk_kimi_delta_rule", "recurrent_kimi_delta_rule"]


def _stable_softplus(x: Tensor) -> Tensor:
    """``log(1 + exp(x))`` with an upper bound short-circuit to avoid overflow."""
    return tensorplay.where(x > 20.0, x, tensorplay.log(1.0 + tensorplay.exp(x)))


def recurrent_kimi_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Token-by-token fine-grained gated delta rule.

    Args:
        query: ``[batch, seq, heads, key_dim]``.
        key: ``[batch, seq, heads, key_dim]``.
        value: ``[batch, seq, heads, value_dim]``.
        g: log-space decay ``[batch, seq, heads, key_dim]`` (per channel).
        beta: write strength ``[batch, seq, heads]``.
        initial_state: ``[batch, heads, key_dim, value_dim]``.
        output_final_state: return the new recurrent state.
        use_qk_l2norm_in_kernel: L2-normalize query/key in float32 first.

    Returns:
        Output ``[batch, seq, heads, value_dim]`` and (optionally) the new
        recurrent state in float32.
    """
    initial_dtype = query.dtype
    batch_size, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, g, beta = [x.to(tensorplay.float32) for x in (query, key, value, g, beta)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * key.shape[-1] ** -0.5

    if initial_state is None:
        state = tensorplay.zeros(
            (batch_size, num_heads, key_dim, value_dim), dtype=value.dtype, device=value.device
        )
    else:
        state = initial_state.to(value.dtype)

    outputs = []
    for i in range(seq_len):
        state = state * g[:, i].exp()[..., None]
        kv_mem = (state * key[:, i][..., None]).sum(dim=-2)
        delta = (value[:, i] - kv_mem) * beta[:, i][..., None]
        state = state + key[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[:, i].unsqueeze(-1)).sum(dim=-2).unsqueeze(1))

    output = tensorplay.cat(outputs, dim=1).to(initial_dtype)
    return output, state if output_final_state else None


def chunk_kimi_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Chunked fine-grained gated delta rule; same contract as the recurrent
    path with an additional ``chunk_size`` argument."""
    initial_dtype = query.dtype
    batch_size, seq_len, num_heads, key_dim = key.shape
    value_dim = value.shape[-1]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(tensorplay.float32) for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * key.shape[-1] ** -0.5

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    total_len = seq_len + pad
    num_chunks = total_len // chunk_size
    query = F.pad(query, (0, 0, 0, pad)) * 1.0
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    decay = F.pad(decay, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, decay, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, value, decay, k_beta, v_beta)
    ]

    diag_mask = tensorplay.ones(chunk_size, chunk_size, dtype=tensorplay.bool, device=query.device).triu(0)
    strict_upper = tensorplay.ones(chunk_size, chunk_size, dtype=tensorplay.bool, device=query.device).triu(1)

    # Cumulative per-channel decay inside each chunk; differences give the
    # decay accumulated between two positions on each key coordinate.
    decay = decay.cumsum(dim=-2)
    decay_mask = (decay.unsqueeze(-2) - decay.unsqueeze(-3)).masked_fill(
        strict_upper[..., None], float("-inf")
    ).exp()

    # The chunk's UT system: entry (i, j) sums the decayed key match over
    # channels, so the substitution below folds beta-scaled delta updates.
    # Negated input: the recurrence here starts from +L, unlike the head-scalar
    # variant which starts from -L.
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(diag_mask, 0)
    attn = _lower_recurrence(-attn)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * decay.exp())

    if initial_state is None:
        state = tensorplay.zeros(
            (batch_size, num_heads, key_dim, value_dim), dtype=value.dtype, device=value.device
        )
    else:
        state = initial_state.to(value.dtype)

    chunk_outputs = []
    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], decay[:, :, i]
        inter_chunk = (q_i * g_i.exp()) @ state
        intra_chunk = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1)
        intra_chunk = intra_chunk.masked_fill(strict_upper, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        chunk_outputs.append(inter_chunk + intra_chunk @ v_new)
        state = state * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new

    output = tensorplay.cat(chunk_outputs, dim=2).reshape(
        batch_size, num_heads, total_len, value_dim
    )
    output = output[:, :, :seq_len].transpose(1, 2).to(initial_dtype)
    return output, state if output_final_state else None


class KimiDeltaAttention(Module):
    """Fine-grained gated delta rule token mixer.

    Parallel projections produce query/key/value (each passing through a
    shared short causal convolution), a per-head write strength, a low-rank
    per-channel forget gate, and a low-rank output gate. The forget gate maps
    through ``-exp(A_log) * softplus(f(hidden) + dt_bias)`` so it is always a
    non-positive log-decay.

    Args:
        hidden_size: model width of the input and output.
        num_heads: number of heads.
        head_dim: per-head key/value width.
        conv_kernel_size: width of the depthwise causal convolution.
        rms_norm_eps: epsilon of the output gating norm.
        chunk_size: chunk width of the chunked scan path.
        layer_idx: cache slot this layer reads and writes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        chunk_size: int = 64,
        layer_idx: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_dim = head_dim * num_heads
        self.conv_kernel_size = conv_kernel_size
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.activation = "silu"

        self.q_proj = Linear(hidden_size, self.qkv_dim, bias=False, **factory_kwargs)
        self.k_proj = Linear(hidden_size, self.qkv_dim, bias=False, **factory_kwargs)
        self.v_proj = Linear(hidden_size, self.qkv_dim, bias=False, **factory_kwargs)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d_weight = Parameter(
            tensorplay.zeros(self.conv_dim, conv_kernel_size, **factory_kwargs)
        )
        self.conv1d_bias = None

        # Fine-grained forget gate: rank-``head_dim`` bottleneck.
        self.f_a_proj = Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.f_b_proj = Linear(head_dim, self.qkv_dim, bias=False, **factory_kwargs)
        self.dt_bias = Parameter(tensorplay.zeros(self.qkv_dim, **factory_kwargs))
        decay_scale = tensorplay.rand(1, 1, num_heads, 1, device=device) * 15.0 + 1.0
        self.A_log = Parameter(tensorplay.log(decay_scale).to(dtype=dtype))

        self.b_proj = Linear(hidden_size, num_heads, bias=False, **factory_kwargs)

        self.g_a_proj = Linear(hidden_size, head_dim, bias=False, **factory_kwargs)
        self.g_b_proj = Linear(head_dim, self.qkv_dim, bias=False, **factory_kwargs)
        self.o_norm = GatedRMSNorm(head_dim, eps=rms_norm_eps, activation="sigmoid", **factory_kwargs)
        self.o_proj = Linear(self.qkv_dim, hidden_size, bias=False, **factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values=None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Mix the sequence with the fine-grained gated delta rule.

        Args:
            hidden_states: ``[batch, seq, hidden_size]``.
            past_key_values: optional :class:`~tensorplay.nn.cache.LinearStateCache`
                holding this layer's conv window and recurrent matrix.

        Returns:
            ``[batch, seq, hidden_size]``.
        """
        bsz, seq_len, _ = hidden_states.shape
        mixed = tensorplay.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)],
            dim=-1,
        ).transpose(1, 2)

        has_state = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if past_key_values is not None:
            mixed = past_key_values.update_conv_state(self.layer_idx, mixed, self.conv_kernel_size)
        mixed = _causal_conv1d(mixed, self.conv1d_weight, self.conv1d_bias, self.activation)
        mixed = mixed[..., -seq_len:] if past_key_values is not None else mixed[..., :seq_len]

        mixed = mixed.transpose(1, 2)
        query, key, value = tensorplay.split(mixed, [self.qkv_dim] * 3, dim=-1)
        hidden_shape = (bsz, seq_len, self.num_heads, self.head_dim)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # Per-channel log decay in (-inf, 0].
        forget = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget.to(tensorplay.float32) + self.dt_bias).view(hidden_shape)
        g = -self.A_log.to(tensorplay.float32).exp() * _stable_softplus(g)
        beta = tensorplay.sigmoid(self.b_proj(hidden_states))

        initial_state = past_key_values.recurrent_states.get(self.layer_idx) if has_state else None
        if has_state and seq_len == 1:
            core_attn_out, new_state = recurrent_kimi_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=initial_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, new_state = chunk_kimi_delta_rule(
                query, key, value, g=g, beta=beta,
                chunk_size=self.chunk_size, initial_state=initial_state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
        if past_key_values is not None:
            past_key_values.update_recurrent_state(self.layer_idx, new_state)

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        core_attn_out = self.o_norm(core_attn_out, gate)
        return self.o_proj(core_attn_out.reshape(bsz, seq_len, -1))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, layer_idx={self.layer_idx}"
        )
