"""Shared helpers for per-head attention analysis.

These utilities reconstruct per-head attention outputs from the internals of a
``tensorplay.nn.MultiheadAttention`` module (the input projections and the
attention weights). They are used by feature circuits and attribution graphs to
intervene on individual attention heads.
"""

from typing import Any

import tensorplay as tensorplay


def compute_head_outputs(
    attn_module: Any, h_attn_input: tensorplay.Tensor, attn_weights: tensorplay.Tensor
) -> tensorplay.Tensor:
    """Reconstruct per-head attention outputs.

    Args:
        attn_module: A ``tensorplay.nn.MultiheadAttention`` module.
        h_attn_input: The attention input tensor, shape ``(batch, seq, dim)``.
        attn_weights: Attention weights with shape
            ``(batch, heads, seq, seq)`` (from ``need_weights=True`` with
            ``average_attn_weights=False``).

    Returns:
        Per-head outputs with shape ``(batch, heads, seq, head_dim)``.
    """
    batch, seq, dim = h_attn_input.shape
    n_heads = attn_weights.shape[1]
    head_dim = dim // n_heads
    in_proj_weight = attn_module.in_proj_weight
    in_proj_bias = attn_module.in_proj_bias
    # value projection: rows [2*dim : 3*dim] of the fused projection
    values = h_attn_input @ in_proj_weight[2 * dim :].t() + in_proj_bias[2 * dim :]
    values = (
        values.reshape(batch, seq, n_heads, head_dim).transpose(1, 2)
    )  # (batch, heads, seq, head_dim)
    return tensorplay.einsum(
        "bhst,bhtd->bhsd", attn_weights, values
    )


def head_output_contribution(
    attn_module: Any, head_outputs: tensorplay.Tensor, head_index: int
) -> tensorplay.Tensor:
    """Project one head's outputs back into the residual stream.

    Args:
        attn_module: A ``tensorplay.nn.MultiheadAttention`` module.
        head_outputs: Per-head outputs from :func:`compute_head_outputs`.
        head_index: Head index to project.

    Returns:
        The contribution of ``head_index`` to the full attention output, shape
        ``(batch, seq, dim)``.
    """
    head_dim = head_outputs.shape[-1]
    out_proj_weight = attn_module.out_proj.weight
    slice_weight = out_proj_weight[
        :, head_index * head_dim : (head_index + 1) * head_dim
    ]
    return head_outputs[:, head_index] @ slice_weight.t()