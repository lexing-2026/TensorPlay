"""CPU behavior tests for the decoder activation and RoPE primitives."""

import numpy as np
import pytest
import torch

import tensorplay as tp


def _rope_reference(x, cos, sin, position_offset):
    c = cos[position_offset:position_offset + x.shape[-2]]
    s = sin[position_offset:position_offset + x.shape[-2]]
    c = c.reshape((1,) * (x.ndim - 2) + c.shape)
    s = s.reshape((1,) * (x.ndim - 2) + s.shape)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return np.stack((even * c - odd * s, even * s + odd * c), axis=-1).reshape(
        x.shape
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_swiglu_fused_activation_cpu(dtype):
    rng = np.random.default_rng(17)
    gate = (rng.standard_normal((3, 11)) * 2).astype(dtype)
    up = rng.standard_normal((3, 11)).astype(dtype)
    expected = gate / (1 + np.exp(-gate)) * up

    gate_tp = tp.from_numpy(gate)
    up_tp = tp.from_numpy(up)
    np.testing.assert_allclose(
        tp.silu_mul(gate_tp, up_tp).numpy(), expected, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        tp.fused_swiglu(gate_tp, up_tp).numpy(), expected, rtol=2e-6, atol=2e-7
    )

    packed = np.concatenate((gate, up), axis=-1)
    np.testing.assert_allclose(
        tp.silu_and_mul(tp.from_numpy(packed)).numpy(),
        expected,
        rtol=2e-6,
        atol=2e-7,
    )


def test_rotary_embedding_and_fused_rope_cpu():
    rng = np.random.default_rng(23)
    batch, q_heads, kv_heads, tokens, head_dim = 2, 5, 3, 4, 8
    position_offset = 3
    q = rng.standard_normal((batch, q_heads, tokens, head_dim)).astype(np.float32)
    k = rng.standard_normal((batch, kv_heads, tokens, head_dim)).astype(np.float32)
    angles = rng.standard_normal((tokens + position_offset + 2, head_dim // 2)).astype(
        np.float32
    )
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)

    expected_q = _rope_reference(q, cos, sin, position_offset)
    expected_k = _rope_reference(k, cos, sin, position_offset)
    q_tp = tp.from_numpy(q)
    k_tp = tp.from_numpy(k)
    cos_tp = tp.from_numpy(cos)
    sin_tp = tp.from_numpy(sin)

    np.testing.assert_allclose(
        tp.rotary_embedding(q_tp, cos_tp, sin_tp, position_offset).numpy(),
        expected_q,
        rtol=2e-6,
        atol=2e-6,
    )
    out_q, out_k = tp.fused_rope(q_tp, k_tp, cos_tp, sin_tp, position_offset)
    np.testing.assert_allclose(out_q.numpy(), expected_q, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(out_k.numpy(), expected_k, rtol=2e-6, atol=2e-6)


def test_fused_decoder_ops_reject_bad_rope_table_cpu():
    x = tp.randn(1, 2, 3, 8)
    cos = tp.randn(2, 4)
    sin = tp.randn(2, 4)
    with pytest.raises(Exception):
        tp.rotary_embedding(x, cos, sin, position_offset=1)
