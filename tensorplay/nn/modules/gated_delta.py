# mypy: allow-untyped-defs
"""Gated delta rule linear attention.

A linear-attention mixer whose recurrent state is one ``[key_dim, value_dim]``
matrix per head. Each token first decays the state multiplicatively, then
applies a *delta rule* update: the state's prediction of the current value is
subtracted from the value before the outer-product write, so the state keeps a
constantly refreshed key/value association instead of only accumulating.

Two execution paths share one semantics:

* :func:`recurrent_gated_delta_rule` walks the sequence token by token; it is
  the exact reference used for single-step decoding.
* :func:`chunk_gated_delta_rule` splits the sequence into fixed-size chunks
  and folds each chunk's updates into a unit lower triangular system that is
  inverted once per chunk, replacing the token loop with batched matmuls.

The decay ``g`` lives in log space (entries ``<= 0``) and is produced by the
mixer from a slow-rate projection; here it is applied per value head. A
fine-grained per-channel variant lives in
:mod:`tensorplay.nn.modules.kimi_delta`.
"""

from __future__ import annotations

import math
from typing import Optional

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import Tensor

from .module import Module
from .linear import Linear
from ..parameter import Parameter

__all__ = [
    "GatedDeltaNet",
    "GatedRMSNorm",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
]


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Unit-normalize ``x`` along ``dim``; zero vectors stay near zero."""
    return x * tensorplay.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _lower_recurrence(lower: Tensor) -> Tensor:
    """Invert the unit lower triangular matrix that shares ``lower``'s strictly
    lower triangle (its diagonal is treated as 1), by forward substitution.

    Each row starts from ``-lower[i, :i]`` and absorbs the rows above it, so
    the result is ``I + N`` where ``N`` folds the series
    ``-L + L L - L L L + ...`` for ``L = lower.tril(-1)``.
    """
    n = lower.shape[-1]
    result = -lower.tril(-1)
    for i in range(1, n):
        row = result[..., i, :i].clone()
        sub = result[..., :i, :i].clone()
        result[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return result + tensorplay.eye(n, dtype=lower.dtype, device=lower.device)


def chunk_gated_delta_rule(
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
    """Chunked gated delta rule.

    Args:
        query: ``[batch, seq, heads, key_dim]``.
        key: ``[batch, seq, heads, key_dim]``.
        value: ``[batch, seq, heads, value_dim]``.
        g: log-space decay ``[batch, seq, heads]``; entries ``<= 0``.
        beta: write strength ``[batch, seq, heads]``.
        chunk_size: sequence chunk width.
        initial_state: recurrent state ``[batch, heads, key_dim, value_dim]``.
        output_final_state: return the new recurrent state alongside the output.
        use_qk_l2norm_in_kernel: L2-normalize query/key in float32 first.

    Returns:
        Output ``[batch, seq, heads, value_dim]`` and (optionally) the new
        recurrent state in float32.
    """
    initial_dtype = query.dtype
    batch_size, seq_len, _, key_dim = key.shape
    num_heads, value_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(tensorplay.float32) for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * key.shape[-1] ** -0.5

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    total_len = seq_len + pad
    num_chunks = total_len // chunk_size

    # The beta-scaled vectors: beta plays the role of the delta rule's learning
    # rate, 0 keeps the state untouched and 1 fully overwrites the old value.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], num_chunks, chunk_size)

    strict_upper = tensorplay.ones(chunk_size, chunk_size, dtype=tensorplay.bool, device=query.device).triu(1)

    # cum_decay[..., t] is the log of the decay product from the chunk start to
    # position t; pairwise differences give the decay accumulated between two
    # positions. Future pairs (j > i) are masked before the exp.
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strict_upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Solving the unit lower triangular system once per chunk condenses a
    # whole chunk of delta-rule updates into two matmuls.
    ut_inverse = _lower_recurrence(ut_system)
    new_values = ut_inverse @ v_beta
    k_cumdecay = ut_inverse @ decayed_k_beta

    if initial_state is None:
        state = tensorplay.zeros(
            (batch_size, num_heads, key_dim, value_dim),
            dtype=new_values.dtype,
            device=new_values.device,
        )
    else:
        state = initial_state.to(new_values.dtype)

    # Decays are folded into the tensors once instead of inside the scan.
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    chunk_outputs = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        inter_chunk = query[:, :, i] @ state
        chunk_outputs.append(inter_chunk + intra_chunk_attn[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    output = tensorplay.cat(chunk_outputs, dim=2).reshape(
        batch_size, num_heads, total_len, value_dim
    )
    output = output[:, :, :seq_len].transpose(1, 2).to(initial_dtype)
    return output, state if output_final_state else None


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Token-by-token gated delta rule; same contract as the chunked path."""
    initial_dtype = query.dtype
    batch_size, seq_len, _, key_dim = key.shape
    num_heads, value_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(tensorplay.float32) for x in (query, key, value, beta, g)
    ]
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
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        state = state * decay[:, :, i].exp()[..., None, None]
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(2))

    output = tensorplay.cat(outputs, dim=2).transpose(1, 2).to(initial_dtype)
    return output, state if output_final_state else None


class GatedRMSNorm(Module):
    """RMS normalization over the last dimension followed by a gate.

    The gate is transformed by ``activation`` (``'silu'`` or ``'sigmoid'``)
    and multiplies the normalized vector elementwise. Computation runs in
    float32 and is cast back at the end.
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, activation: str = "silu", device=None, dtype=None
    ) -> None:
        super().__init__()
        if activation not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported gate activation {activation!r}")
        self.weight = Parameter(tensorplay.ones(dim, device=device, dtype=dtype))
        self.eps = eps
        self.activation = activation

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(tensorplay.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * tensorplay.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        gate = gate.to(tensorplay.float32)
        gate = F.silu(gate) if self.activation == "silu" else tensorplay.sigmoid(gate)
        return (x * gate).to(input_dtype)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, activation={self.activation!r}"


def _causal_conv1d(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], activation: Optional[str] = None
) -> Tensor:
    """Depthwise causal convolution over the trailing (time) dimension.

    ``x`` is ``[batch, channels, seq]`` and ``weight`` is
    ``[channels, kernel_size]`` (one filter per channel). The input is
    zero-padded on the left by ``kernel_size - 1`` so every output position
    only reads present and past inputs.
    """
    kernel_size = weight.shape[-1]
    channels, seq_len = x.shape[-2], x.shape[-1]
    padded = F.pad(x, (kernel_size - 1, 0))
    out = tensorplay.zeros_like(x)
    for tap in range(kernel_size):
        out = out + weight[:, tap].reshape(-1, 1) * padded[..., tap : tap + seq_len]
    if bias is not None:
        out = out + bias.reshape(-1, 1)
    if activation is not None:
        out = F.silu(out) if activation == "silu" else F.gelu(out)
    return out.to(x.dtype)


class GatedDeltaNet(Module):
    """Gated delta rule token mixer.

    The input is projected to per-head query/key/value plus an output gate
    (``in_proj_qkvz``) and to the beta / slow-rate gate inputs
    (``in_proj_ba``). Query, key and value pass through a short depthwise
    causal convolution; the decay is ``g = -exp(A_log) * softplus(a + dt_bias)``
    per value head, and the delta-rule scan runs in float32. The scan output
    is normalized and gated, then projected back to the model width.

    Args:
        hidden_size: model width of the input and output.
        num_v_heads: number of value heads (the recurrent state's heads).
        num_k_heads: number of key/query heads; must divide ``num_v_heads``,
            shared across value heads by replication.
        head_k_dim: key/query head width.
        head_v_dim: value head width; defaults to ``head_k_dim``.
        conv_kernel_size: width of the depthwise causal convolution.
        rms_norm_eps: epsilon of the output gating norm.
        chunk_size: chunk width of the chunked scan path.
        layer_idx: cache slot this layer reads and writes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        num_k_heads: int,
        head_k_dim: int,
        head_v_dim: Optional[int] = None,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        chunk_size: int = 64,
        layer_idx: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_v_heads % num_k_heads:
            raise ValueError(
                f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads})"
            )
        if head_v_dim is None:
            head_v_dim = head_k_dim
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.activation = "silu"

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkvz = Linear(hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False, **factory_kwargs)
        self.in_proj_ba = Linear(hidden_size, num_v_heads * 2, bias=False, **factory_kwargs)

        self.conv1d_weight = Parameter(
            tensorplay.zeros(self.conv_dim, conv_kernel_size, **factory_kwargs)
        )
        self.conv1d_bias = None

        self.dt_bias = Parameter(tensorplay.ones(num_v_heads, **factory_kwargs))
        # Lower bound kept away from 0 so log(A) never becomes -inf.
        decay_scale = tensorplay.rand(num_v_heads, device=device) * 15.99 + 0.01
        self.A_log = Parameter(tensorplay.log(decay_scale).to(dtype=dtype))

        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation="silu", **factory_kwargs)
        self.out_proj = Linear(self.value_dim, hidden_size, bias=False, **factory_kwargs)

    def _fix_qkvz_ordering(self, mixed_qkvz: Tensor, mixed_ba: Tensor):
        """Split the fused projections into per-head q/k/v/z and b/a.

        The fused layout interleaves ``[q, k, value, z]`` per key-head group so
        that value/z blocks stay contiguous per group; this undoes that.
        """
        bsz, seq_len = mixed_qkvz.shape[:2]
        ratio = self.num_v_heads // self.num_k_heads
        mixed_qkvz = mixed_qkvz.view(
            bsz, seq_len, self.num_k_heads, 2 * self.head_k_dim + 2 * ratio * self.head_v_dim
        )
        mixed_ba = mixed_ba.view(bsz, seq_len, self.num_k_heads, 2 * ratio)
        query, key, value, z = tensorplay.split(
            mixed_qkvz,
            [self.head_k_dim, self.head_k_dim, ratio * self.head_v_dim, ratio * self.head_v_dim],
            dim=3,
        )
        b, a = tensorplay.split(mixed_ba, [ratio, ratio], dim=3)
        value = value.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        b = b.reshape(bsz, seq_len, self.num_v_heads)
        a = a.reshape(bsz, seq_len, self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: Tensor,
        past_key_values=None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Mix the sequence with the gated delta rule.

        Args:
            hidden_states: ``[batch, seq, hidden_size]``.
            past_key_values: optional :class:`~tensorplay.nn.cache.LinearStateCache`
                holding this layer's conv window and recurrent matrix.

        Returns:
            ``[batch, seq, hidden_size]``.
        """
        bsz, seq_len, _ = hidden_states.shape
        query, key, value, z, b, a = self._fix_qkvz_ordering(
            self.in_proj_qkvz(hidden_states), self.in_proj_ba(hidden_states)
        )

        mixed = tensorplay.cat(
            [
                query.reshape(bsz, seq_len, -1),
                key.reshape(bsz, seq_len, -1),
                value.reshape(bsz, seq_len, -1),
            ],
            dim=-1,
        ).transpose(1, 2)

        has_state = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if past_key_values is not None:
            mixed = past_key_values.update_conv_state(self.layer_idx, mixed, self.conv_kernel_size)
        mixed = _causal_conv1d(mixed, self.conv1d_weight, self.conv1d_bias, self.activation)
        mixed = mixed[..., -seq_len:] if past_key_values is not None else mixed[..., :seq_len]

        mixed = mixed.transpose(1, 2)
        query, key, value = tensorplay.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.to(tensorplay.float32).exp() * F.softplus(a.to(tensorplay.float32) + self.dt_bias)
        ratio = self.num_v_heads // self.num_k_heads
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        initial_state = past_key_values.recurrent_states.get(self.layer_idx) if has_state else None
        if has_state and seq_len == 1:
            core_attn_out, new_state = recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=initial_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, new_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                chunk_size=self.chunk_size, initial_state=initial_state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
        if past_key_values is not None:
            past_key_values.update_recurrent_state(self.layer_idx, new_state)

        core_attn_out = self.norm(core_attn_out, z)
        return self.out_proj(core_attn_out.reshape(bsz, seq_len, -1))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_v_heads={self.num_v_heads}, "
            f"num_k_heads={self.num_k_heads}, head_k_dim={self.head_k_dim}, "
            f"head_v_dim={self.head_v_dim}, layer_idx={self.layer_idx}"
        )
