"""Transformers kernels: SDPA fast path + MoE grouped GEMM parity vs torch."""

import numpy as np
import pytest
import torch

import tensorplay as tp


def _ref_sdpa(q, k, v, causal):
    t = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
        is_causal=causal)
    return t.numpy()


@pytest.mark.parametrize("B,H,T,S,D", [
    (1, 2, 8, 8, 8), (2, 3, 16, 16, 32), (1, 2, 5, 9, 16),
    (1, 4, 64, 2048, 64),  # decode-like cross length
])
@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_parity_f32(B, H, T, S, D, causal):
    q = np.random.randn(B, H, T, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)
    out = tp.scaled_dot_product_attention(
        tp.from_numpy(q), tp.from_numpy(k), tp.from_numpy(v),
        is_causal=causal).numpy()
    ref = _ref_sdpa(q, k, v, causal)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_sdpa_prefill_large_causal():
    B, H, T, D = 1, 2, 512, 128
    q = np.random.randn(B, H, T, D).astype(np.float32)
    k = np.random.randn(B, H, T, D).astype(np.float32)
    v = np.random.randn(B, H, T, D).astype(np.float32)
    out = tp.scaled_dot_product_attention(
        tp.from_numpy(q), tp.from_numpy(k), tp.from_numpy(v),
        is_causal=True).numpy()
    ref = _ref_sdpa(q, k, v, True)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_sdpa_gemm_cuda_parity(dtype):
    """The native GEMM-backed path remains numerically equivalent to Torch."""
    rng = np.random.default_rng(91)
    q = rng.standard_normal((1, 2, 33, 16)).astype(dtype)
    k = rng.standard_normal((1, 2, 33, 16)).astype(dtype)
    v = rng.standard_normal((1, 2, 33, 16)).astype(dtype)
    q_tp = tp.from_numpy(q).to("cuda")
    k_tp = tp.from_numpy(k).to("cuda")
    v_tp = tp.from_numpy(v).to("cuda")
    out = tp.scaled_dot_product_attention(
        q_tp, k_tp, v_tp, is_causal=True, impl=2
    ).cpu().numpy()
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda(),
        torch.from_numpy(v).cuda(), is_causal=True
    ).cpu().numpy()
    np.testing.assert_allclose(
        out.astype(np.float32), ref.astype(np.float32),
        rtol=2e-3 if dtype is np.float16 else 1e-4,
        atol=2e-3 if dtype is np.float16 else 1e-5,
    )


def test_sdpa_requires_4d():
    q = tp.randn([2, 8]); k = tp.randn([2, 8, 8]); v = tp.randn([2, 8, 8])
    with pytest.raises(Exception):
        tp.scaled_dot_product_attention(q, k, v)


def _grouped_ref(a, b, offs):
    parts = []
    prev = 0
    for g in range(b.shape[0]):
        e = int(offs[g])
        if e > prev:
            parts.append(a[prev:e] @ b[g])
        prev = e
    return np.concatenate(parts, 0) if parts else np.zeros((a.shape[0], b.shape[2]), np.float32)


@pytest.mark.parametrize("M,K,N,G", [(64, 128, 128, 8), (33, 96, 80, 5), (4, 64, 64, 4)])
def test_grouped_mm_parity(M, K, N, G):
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(G, K, N).astype(np.float32)
    cuts = sorted(np.random.choice(range(1, M), G - 1, replace=False)) if G > 1 else []
    offs = np.array(cuts + [M], dtype=np.int32)
    out = tp.grouped_mm(tp.from_numpy(a), tp.from_numpy(b), tp.from_numpy(offs)).numpy()
    ref = _grouped_ref(a, b, offs)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_grouped_mm_empty_group():
    a = np.random.randn(16, 8).astype(np.float32)
    b = np.random.randn(3, 8, 8).astype(np.float32)
    offs = np.array([0, 10, 16], dtype=np.int32)  # first group empty
    out = tp.grouped_mm(tp.from_numpy(a), tp.from_numpy(b), tp.from_numpy(offs)).numpy()
    ref = _grouped_ref(a, b, offs)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_grouped_mm_int64_offsets():
    M, K, N, G = 16, 8, 8, 2
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(G, K, N).astype(np.float32)
    offs = np.array([8, M], dtype=np.int64)
    out = tp.grouped_mm(tp.from_numpy(a), tp.from_numpy(b), tp.from_numpy(offs)).numpy()
    ref = _grouped_ref(a, b, offs.astype(np.int32))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_grouped_mm_autograd_composite():
    A = tp.randn([16, 8], requires_grad=True)
    B = tp.randn([2, 8, 8], requires_grad=True)
    out = tp.grouped_mm(A, B, tp.tensor([10, 16], dtype=tp.int32))
    out.sum().backward()
    assert A.grad is not None and A.grad.shape == (16, 8)
    assert B.grad is not None and B.grad.shape == (2, 8, 8)


def test_grouped_mm_rejects_bad_offsets():
    a = tp.randn([16, 8]); b = tp.randn([2, 8, 8])
    with pytest.raises(Exception):
        tp.grouped_mm(a, b, tp.tensor([20, 16], dtype=tp.int32))  # end > M
    with pytest.raises(Exception):
        tp.grouped_mm(a, b, tp.tensor([16, 10], dtype=tp.int32))  # decreasing
