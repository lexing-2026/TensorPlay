import unittest

import numpy as np
import pytest

import tensorplay


def _vulkan_available():
    try:
        return tensorplay.is_vulkan_available()
    except NotImplementedError:
        # Stale build without the availability kernel registered.
        return False


pytestmark = pytest.mark.skipif(
    not _vulkan_available(), reason="Vulkan is not available"
)


class TestVulkanAvailability(unittest.TestCase):
    def test_is_vulkan_available(self):
        self.assertTrue(tensorplay.is_vulkan_available())

    def test_device_type(self):
        self.assertEqual(tensorplay.Device("vulkan").type, "vulkan")
        self.assertTrue(tensorplay.Device("vulkan").is_vulkan())
        self.assertFalse(tensorplay.Device("cpu").is_vulkan())


class TestVulkanTensorCreation(unittest.TestCase):
    def test_zeros(self):
        x = tensorplay.zeros((2, 3), device="vulkan")
        self.assertEqual(x.shape, (2, 3))
        self.assertTrue(x.device.is_vulkan())

    def test_ones(self):
        x = tensorplay.ones((2, 3), device="vulkan")
        self.assertEqual(x.shape, (2, 3))
        self.assertTrue(x.device.is_vulkan())

    def test_full(self):
        x = tensorplay.full((2, 3), 2.5, device="vulkan")
        self.assertEqual(x.shape, (2, 3))

    def test_empty(self):
        x = tensorplay.empty((4, 5), device="vulkan")
        self.assertEqual(x.shape, (4, 5))
        self.assertTrue(x.device.is_vulkan())


class TestVulkanRoundtrip(unittest.TestCase):
    def test_cpu_to_vulkan_to_cpu(self):
        x = tensorplay.arange(6, dtype=tensorplay.float32).reshape(2, 3)
        v = x.to("vulkan")
        self.assertTrue(v.device.is_vulkan())
        back = v.cpu()
        np.testing.assert_allclose(back.numpy(), x.numpy())

    def test_zeros_roundtrip(self):
        x = tensorplay.zeros((2, 3), device="vulkan")
        back = x.cpu()
        np.testing.assert_allclose(back.numpy(), np.zeros((2, 3), dtype=np.float32))

    def test_ones_roundtrip(self):
        x = tensorplay.ones((2, 3), device="vulkan")
        back = x.cpu()
        np.testing.assert_allclose(back.numpy(), np.ones((2, 3), dtype=np.float32))

    def test_full_roundtrip(self):
        x = tensorplay.full((2, 3), 2.5, device="vulkan")
        back = x.cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 2.5, dtype=np.float32)
        )


class TestVulkanOps(unittest.TestCase):
    def test_add_tensor(self):
        a = tensorplay.ones((2, 3), device="vulkan")
        b = tensorplay.full((2, 3), 2.0, device="vulkan")
        back = (a + b).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )

    def test_sub_tensor(self):
        a = tensorplay.full((2, 3), 5.0, device="vulkan")
        b = tensorplay.ones((2, 3), device="vulkan")
        back = (a - b).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 4.0, dtype=np.float32)
        )

    def test_mul_tensor(self):
        a = tensorplay.full((2, 3), 3.0, device="vulkan")
        b = tensorplay.full((2, 3), 2.0, device="vulkan")
        back = (a * b).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 6.0, dtype=np.float32)
        )

    def test_div_tensor(self):
        a = tensorplay.full((2, 3), 6.0, device="vulkan")
        b = tensorplay.full((2, 3), 2.0, device="vulkan")
        back = (a / b).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )

    def test_add_scalar(self):
        a = tensorplay.ones((2, 3), device="vulkan")
        back = (a + 2.0).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )

    def test_mul_scalar(self):
        a = tensorplay.full((2, 3), 3.0, device="vulkan")
        back = (a * 2.0).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 6.0, dtype=np.float32)
        )

    def test_unary_exp(self):
        a = tensorplay.ones((2, 3), device="vulkan")
        back = a.exp().cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), np.e, dtype=np.float32), rtol=1e-5
        )

    def test_unary_sqrt(self):
        a = tensorplay.full((2, 3), 4.0, device="vulkan")
        back = a.sqrt().cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 2.0, dtype=np.float32)
        )

    def test_unary_abs_neg(self):
        a = tensorplay.full((2, 3), -3.0, device="vulkan")
        back = a.abs().cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )
        back = (-a).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )

    def test_fill(self):
        a = tensorplay.zeros((2, 3), device="vulkan")
        a.fill_(1.5)
        back = a.cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 1.5, dtype=np.float32)
        )

    def test_inplace_add(self):
        a = tensorplay.ones((2, 3), device="vulkan")
        b = tensorplay.ones((2, 3), device="vulkan")
        a += b
        back = a.cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 2.0, dtype=np.float32)
        )

    def test_inplace_add_scalar(self):
        a = tensorplay.ones((2, 3), device="vulkan")
        a += 2.0
        back = a.cpu()
        np.testing.assert_allclose(
            back.numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )

    def test_clamp(self):
        a = tensorplay.arange(6, dtype=tensorplay.float32).to("vulkan")
        back = a.clamp(1.5, 4.0).cpu()
        np.testing.assert_allclose(
            back.numpy(), np.array([1.5, 1.5, 2.0, 3.0, 4.0, 4.0], dtype=np.float32)
        )

    def test_copy(self):
        a = tensorplay.ones((2, 3), device="vulkan")
        b = tensorplay.zeros((2, 3), device="vulkan")
        b.copy_(a)
        back = b.cpu()
        np.testing.assert_allclose(
            back.numpy(), np.ones((2, 3), dtype=np.float32)
        )


# Reference for vulkan scaled_dot_product_attention: the attention math
# composite semantics with top-left aligned causal masking, group-query
# expansion over contiguous head blocks, additive attention masks, and a
# fully masked row normalizing to zeros.
def _sdpa_reference(q, k, v, mask=None, causal=False, scale=None):
    q = q.astype(np.float64)
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    hq = q.shape[1]
    hkv = k.shape[1]
    if hkv != hq:
        g = hq // hkv
        k = np.repeat(k, g, axis=1)
        v = np.repeat(v, g, axis=1)
    s = np.einsum("bhqd,bhkd->bhqk", q, k)
    s = s * (1.0 / np.sqrt(q.shape[-1]) if scale is None else scale)
    if causal:
        s = s + np.triu(np.ones((q.shape[2], k.shape[2])), 1) * (-2.0 ** 31)
    if mask is not None:
        if mask.dtype == np.bool_:
            m = np.where(mask, 0.0, -np.inf)
        else:
            m = mask.astype(np.float64)
        s = s + m
    s = s.astype(np.float32).astype(np.float64)
    row = np.where(np.isneginf(s.max(-1, keepdims=True)), 0.0, s.max(-1, keepdims=True))
    e = np.exp(s - row)
    denom = e.sum(-1, keepdims=True)
    p = np.divide(e, denom, out=np.zeros_like(e), where=(denom != 0))
    return np.einsum("bhqk,bhkd->bhqd", p, v)


def _sdpa_check(
    self,
    B,
    Hq,
    Hkv,
    Sq,
    Skv,
    D,
    dtype=tensorplay.float32,
    causal=False,
    mask=None,
    scale=None,
    gqa=False,
    *_,
):
    tensorplay.manual_seed(1234)
    q = tensorplay.randn((B, Hq, Sq, D), device="cpu")
    k = tensorplay.randn((B, Hkv, Skv, D), device="cpu")
    v = tensorplay.randn((B, Hkv, Skv, D), device="cpu")
    if dtype == tensorplay.float16:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    got = tensorplay.nn.functional.scaled_dot_product_attention(
        q.to("vulkan"),
        k.to("vulkan"),
        v.to("vulkan"),
        attn_mask=(mask.to("vulkan") if mask is not None else None),
        is_causal=causal,
        scale=scale,
        enable_gqa=gqa).cpu().numpy()
    want = _sdpa_reference(
        q.numpy(), k.numpy(), v.numpy(),
        mask=(mask.numpy() if mask is not None else None),
        causal=causal, scale=scale)
    if dtype == tensorplay.float16:
        atol = 2.0e-2
    else:
        atol = 4.0e-3
    np.testing.assert_allclose(
        got.astype(np.float64), want, rtol=1.0e-3, atol=atol)


class TestVulkanScaledDotProductAttention(unittest.TestCase):
    def test_plain_mha(self):
        _sdpa_check(self, B=1, Hq=8, Hkv=8, Sq=16, Skv=16, D=32, causal=False)
        _sdpa_check(self, B=2, Hq=4, Hkv=4, Sq=9, Skv=13, D=5, causal=False)
        _sdpa_check(self, B=2, Hq=6, Hkv=6, Sq=2, Skv=5, D=48, causal=False)

    def test_causal_mha(self):
        _sdpa_check(self, B=1, Hq=12, Hkv=12, Sq=64, Skv=64, D=64, causal=True)
        _sdpa_check(self, B=2, Hq=4, Hkv=4, Sq=9, Skv=13, D=5, causal=True)
        _sdpa_check(self, B=1, Hq=3, Hkv=3, Sq=4, Skv=4, D=3, causal=True)

    def test_decode_without_shifted_causal(self):
        # Causal masking aligns top-left: a single query position attends
        # the first key only.
        _sdpa_check(self, B=1, Hq=4, Hkv=4, Sq=1, Skv=1, D=8, causal=True)
        _sdpa_check(self, B=1, Hq=4, Hkv=4, Sq=1, Skv=7, D=16, causal=True)
        _sdpa_check(self, B=1, Hq=4, Hkv=4, Sq=1, Skv=7, D=16, causal=False)

    def test_gqa(self):
        _sdpa_check(self, B=1, Hq=8, Hkv=2, Sq=1, Skv=31, D=64, gqa=True)
        _sdpa_check(
            self, B=1, Hq=12, Hkv=4, Sq=64, Skv=64, D=32, causal=True,
            gqa=True)
        _sdpa_check(
            self, B=2, Hq=6, Hkv=3, Sq=5, Skv=5, D=16, causal=True, gqa=True)
        _sdpa_check(
            self, B=1, Hq=4, Hkv=2, Sq=3, Skv=9, D=8, gqa=True)

    def test_bool_mask(self):
        B, H, Sq, Skv, D = 2, 4, 6, 10, 8
        shared = np.tril(np.ones((Sq, Skv)), k=Skv - Sq).astype(bool)
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(shared))
        per_head = np.random.default_rng(7).random((H, Sq, Skv)) > 0.4
        per_head[:, 0, :] = True  # keep every head's rows visible
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(per_head))
        full = np.random.default_rng(8).random((B, H, Sq, Skv)) > 0.4
        full[:, :, 0, :] = True
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(full))

    def test_float_mask(self):
        B, H, Sq, Skv, D = 2, 4, 6, 10, 8
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(
                np.random.default_rng(9).standard_normal(
                    (Sq, Skv)).astype(np.float32)))
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(
                np.random.default_rng(10).standard_normal(
                    (1, 1, Sq, Skv)).astype(np.float32)))
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(
                np.random.default_rng(11).standard_normal(
                    (B, 1, Sq, Skv)).astype(np.float32)))
        _sdpa_check(
            self, B, H, H, Sq, Skv, D,
            mask=tensorplay.tensor(
                np.random.default_rng(12).standard_normal(
                    (H, Sq, Skv)).astype(np.float32)))

    def test_scaled(self):
        _sdpa_check(
            self, B=2, Hq=4, Hkv=4, Sq=8, Skv=8, D=16, scale=0.37,
            causal=True)

    def test_float16(self):
        _sdpa_check(
            self, B=1, Hq=8, Hkv=2, Sq=1, Skv=64, D=64,
            dtype=tensorplay.float16, gqa=True)
        _sdpa_check(
            self, B=1, Hq=8, Hkv=8, Sq=128, Skv=128, D=64,
            dtype=tensorplay.float16, causal=True)
        _sdpa_check(
            self, B=2, Hq=4, Hkv=1, Sq=16, Skv=16, D=32,
            dtype=tensorplay.float16)

    def test_single_kv_head_broadcast(self):
        _sdpa_check(self, B=2, Hq=4, Hkv=1, Sq=16, Skv=16, D=32, causal=True)
        _sdpa_check(
            self, B=2, Hq=4, Hkv=1, Sq=16, Skv=16, D=32, causal=False)

    def test_sparse_split_memory_scan(self):
        # Short queries over long key ranges take the scan + combine route.
        _sdpa_check(
            self, B=1, Hq=4, Hkv=4, Sq=1, Skv=600, D=64, causal=False)
        _sdpa_check(
            self, B=1, Hq=4, Hkv=4, Sq=1, Skv=600, D=64, causal=True)
        _sdpa_check(
            self, B=1, Hq=8, Hkv=2, Sq=3, Skv=513, D=16, gqa=True)
        _sdpa_check(
            self, B=2, Hq=4, Hkv=4, Sq=8, Skv=600, D=8, causal=False)
        _sdpa_check(
            self, B=1, Hq=4, Hkv=1, Sq=2, Skv=600, D=32)
        _sdpa_check(
            self, B=1, Hq=8, Hkv=8, Sq=4, Skv=517, D=16,
            dtype=tensorplay.float16)

    def test_split_memory_scan_with_masks(self):
        # Every mask mode reaching the scan + combine route.
        _sdpa_check(
            self, B=1, Hq=4, Hkv=4, Sq=1, Skv=300, D=16,
            mask=tensorplay.tensor(np.tril(np.ones((1, 300))).astype(bool)))
        _sdpa_check(
            self, B=2, Hq=6, Hkv=6, Sq=2, Skv=300, D=8,
            mask=tensorplay.tensor(
                np.random.default_rng(14).random((6, 2, 300)) > 0.3))
        _sdpa_check(
            self, B=1, Hq=3, Hkv=3, Sq=2, Skv=300, D=8,
            mask=tensorplay.tensor(
                np.random.default_rng(15).random((1, 3, 2, 300)) > 0.3))
        _sdpa_check(
            self, B=1, Hq=8, Hkv=2, Sq=4, Skv=300, D=16, gqa=True,
            mask=tensorplay.tensor(
                np.random.default_rng(16).random((1, 1, 4, 300)) > 0.3))

    def test_fully_masked_rows_emit_zeros(self):
        B, H, Sq, Skv, D = 1, 4, 6, 10, 8
        m = np.ones((B, H, Sq, Skv), dtype=bool)
        m[0, :, 0, :] = False  # every entry of row zero masked
        got = tensorplay.nn.functional.scaled_dot_product_attention(
            tensorplay.randn((B, H, Sq, D), device="cpu").to("vulkan"),
            tensorplay.randn((B, H, Skv, D), device="cpu").to("vulkan"),
            tensorplay.randn((B, H, Skv, D), device="cpu").to("vulkan"),
            attn_mask=tensorplay.tensor(m).to("vulkan")).cpu().numpy()
        # No entry survives the mask, so the whole row stays zero in every
        # output component instead of dividing zero by zero.
        np.testing.assert_array_equal(got[:, :, 0, :], 0.0)
        self.assertFalse(np.isnan(got).any())


def _rope_tables(head_dim, rows):
    half = head_dim // 2
    positions = np.arange(rows)
    inv = 1.0 / (half ** (2.0 * np.arange(half) / head_dim))
    cos = np.cos(positions[:, None] * inv[None, :]).astype(np.float32)
    sin = np.sin(positions[:, None] * inv[None, :]).astype(np.float32)
    return cos, sin


def _rope_reference(x, cos, sin, position_offset):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    seq = x.shape[-2]
    c = cos[position_offset:position_offset + seq]
    s = sin[position_offset:position_offset + seq]
    out = np.empty_like(x)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out


class TestVulkanLlmKernels(unittest.TestCase):
    def test_rotary_embedding(self):
        tensorplay.manual_seed(1234)
        for B, S, D, table_len, off, dtype, tol in (
            (1, 64, 64, 64, 0, tensorplay.float32, 3.0e-4),
            (2, 9, 32, 512, 0, tensorplay.float32, 3.0e-4),
            (1, 1, 64, 600, 512, tensorplay.float32, 3.0e-4),
            (1, 1, 128, 600, 300, tensorplay.float16, 3.0e-2),
        ):
            x = tensorplay.randn((B, S, D), device="cpu").to(dtype)
            cos, sin = _rope_tables(D, table_len)
            rotated = tensorplay.functional.rotary_embedding(
                x.to("vulkan"),
                tensorplay.tensor(cos, dtype=dtype).to("vulkan"),
                tensorplay.tensor(sin, dtype=dtype).to("vulkan"),
                position_offset=off)
            # The rotation is elementwise, so the element type is carried
            # through: a half-precision decode step must not widen itself.
            self.assertEqual(rotated.dtype, dtype)
            got = rotated.cpu().numpy()
            want = _rope_reference(x.numpy().astype(np.float32), cos, sin, off)
            np.testing.assert_allclose(got.astype(np.float32), want, rtol=0, atol=tol)

    def test_glu_splits_and_dtypes(self):
        # Gated linear unit over texture storage: first_half *
        # sigmoid(second_half), along whichever axis is halved.  Half
        # precision must stay half precision end to end.
        import tensorplay as tp
        tensorplay.manual_seed(1234)
        for shape, dim, dtype, tol in (
            ((3, 2, 32), -1, tensorplay.float32, 1.0e-5),
            ((3, 2, 32), -1, tensorplay.float16, 3.0e-2),
            ((2, 8, 16), 1, tensorplay.float32, 1.0e-5),
            ((2, 8, 16), 1, tensorplay.float16, 3.0e-2),
            ((4, 64), 0, tensorplay.float32, 1.0e-5),
            ((4, 64), 0, tensorplay.float16, 3.0e-2),
            # Halving the channel axis crosses texel lanes inside one texel.
            ((2, 8, 3, 5), 1, tensorplay.float32, 1.0e-5),
            ((2, 8, 3, 5), 1, tensorplay.float16, 3.0e-2),
        ):
            x = (tensorplay.randn(shape, device="cpu") * 1.5).to(dtype)
            got = tp.glu(x.to("vulkan"), dim)
            self.assertEqual(got.dtype, dtype)
            xn = x.numpy().astype(np.float64)
            n = xn.shape[dim]
            a = np.take(xn, np.arange(n // 2), axis=dim)
            b = np.take(xn, np.arange(n // 2, n), axis=dim)
            want = a / (1.0 + np.exp(-b))
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), want, rtol=0, atol=tol)

    def test_silu_and_mul(self):
        tensorplay.manual_seed(1234)
        for B, S, W, dtype, tol in (
            (1, 3, 8, tensorplay.float32, 1.0e-5),
            (1, 1, 512, tensorplay.float16, 3.0e-2),
            (2, 5, 4096, tensorplay.float32, 1.0e-5),
        ):
            x = (tensorplay.randn((B, S, 2 * W), device="cpu") * 1.5).to(dtype)
            got = tensorplay.functional.silu_and_mul(x.to("vulkan")).cpu().numpy()
            a = x.numpy()[..., :W].astype(np.float64)
            b = x.numpy()[..., W:].astype(np.float64)
            want = ((a / (1.0 + np.exp(-a))) * b).astype(np.float32)
            np.testing.assert_allclose(got.astype(np.float64), want, rtol=0, atol=tol)

    def test_rms_norm(self):
        tensorplay.manual_seed(1234)
        for B, S, H, eps, dtype, tol in (
            (1, 64, 1024, 1.0e-5, tensorplay.float32, 1.0e-5),
            (1, 1, 1024, 1.0e-5, tensorplay.float16, 3.0e-2),
            (2, 3, 64, 1.0e-1, tensorplay.float32, 1.0e-5),
        ):
            x = (tensorplay.randn((B, S, H), device="cpu") * 2.0).to(dtype)
            w = (tensorplay.randn((H,), device="cpu") * 0.7 + 1.0).to(dtype)
            got = tensorplay.nn.functional.rms_norm(
                x.to("vulkan"), [H], w.to("vulkan"), eps).cpu().numpy()
            xn = x.numpy().astype(np.float64)
            want = (xn / np.sqrt((xn ** 2).mean(-1, keepdims=True) + eps)) * \
                w.numpy().astype(np.float64)
            np.testing.assert_allclose(
                got.astype(np.float64), want, rtol=0, atol=tol)


class TestVulkanIndexCopy(unittest.TestCase):
    def test_index_copy_rows_inplace(self):
        tensorplay.manual_seed(1234)
        for B, C, S, D, idx, dtype in (
            (1, 2, 64, 64, [2, 5], tensorplay.float32),
            (1, 1, 4, 8, [0, 0], tensorplay.float32),
            (2, 4, 16, 4, [8, 9], tensorplay.float16),
            (1, 1, 64, 8, [63], tensorplay.float32),
        ):
            buf = tensorplay.randn((B, C, S, D), device="cpu").to(dtype).to("vulkan")
            before = buf.cpu().numpy()
            src = tensorplay.randn((B, C, len(idx), D), device="cpu").to(dtype).to("vulkan")
            positions = tensorplay.tensor(idx, dtype=tensorplay.int64)
            tensorplay.index_copy_(buf, 2, positions, src)
            got = buf.cpu().numpy()
            want = before.copy()
            want[:, :, idx, :] = src.cpu().numpy().astype(got.dtype)
            np.testing.assert_array_equal(got, want)

    def test_index_copy_rejects_out_of_range(self):
        buf = tensorplay.randn((1, 2, 8, 4), device="cpu").to("vulkan")
        src = tensorplay.randn((1, 2, 1, 4), device="cpu").to("vulkan")
        with self.assertRaises(RuntimeError):
            tensorplay.index_copy_(
                buf, 2, tensorplay.tensor([8], dtype=tensorplay.int64), src)

    def test_index_copy_out_of_place(self):
        buf = tensorplay.randn((1, 2, 8, 4), device="cpu").to("vulkan")
        snapshot = buf.cpu().numpy()
        src = tensorplay.randn((1, 2, 2, 4), device="cpu").to("vulkan")
        out = tensorplay.index_copy(
            buf, 2, tensorplay.tensor([1, 3], dtype=tensorplay.int64), src)
        got = out.cpu().numpy()
        want = snapshot.copy()
        want[:, :, [1, 3], :] = src.cpu().numpy()
        np.testing.assert_array_equal(got, want)
        # The functional form leaves the destination untouched.
        np.testing.assert_array_equal(buf.cpu().numpy(), snapshot)

    def test_static_cache_update_writes_rows(self):
        # A slice-then-copy append materializes into an orphaned copy on
        # vulkan, so the cache routes its rows through the in-place scatter;
        # this pins both the write-through and the untouched tail.
        from tensorplay.nn.cache import StaticCache
        tensorplay.manual_seed(1234)
        B, C, D = 1, 2, 8
        cache = StaticCache(16)
        chunks = []
        for _ in range(5):
            k = tensorplay.randn((B, C, 3, D), device="cpu").to("vulkan")
            v = tensorplay.randn((B, C, 3, D), device="cpu").to("vulkan")
            cache.update(k, v, 0)
            chunks.append((k, v))
        keys = np.concatenate([k.cpu().numpy() for k, _ in chunks], axis=2)
        values = np.concatenate([v.cpu().numpy() for _, v in chunks], axis=2)
        self.assertEqual(cache.get_seq_length(), 15)
        np.testing.assert_array_equal(
            cache.key_cache[0].cpu().numpy()[:, :, :15, :], keys)
        np.testing.assert_array_equal(
            cache.value_cache[0].cpu().numpy()[:, :, :15, :], values)
        self.assertFalse(cache.key_cache[0].cpu().numpy()[:, :, 15:, :].any())


class TestVulkanDecodeMatmul(unittest.TestCase):
    @staticmethod
    def _relerr(got, want):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        denom = max(float(np.abs(want).max()), 1.0)
        return float(np.abs(got - want).max()) / denom

    def test_decode_matvec_and_matmat(self):
        # Narrow-row products run the two-pass split reduction; pin its
        # numerics across dtypes, non-multiple-of-four tails, and the
        # tiled-path fallback for wider outputs.
        import tensorplay as tp
        tensorplay.manual_seed(1234)
        cases = []
        for K, N in ((1024, 2048), (4096, 1024), (517, 303)):
            cases.append((1, K, N, tensorplay.float32, 2.0e-5))
            cases.append((4, K, N, tensorplay.float16, 2.0e-3))
        cases.append((8, 1024, 1024, tensorplay.float32, 2.0e-5))
        for M, K, N, dtype, tol in cases:
            a = tensorplay.randn((M, K), device="cpu").to(dtype)
            b = tensorplay.randn((K, N), device="cpu").to(dtype)
            got = (a.to("vulkan") @ b.to("vulkan")).cpu().numpy()
            want = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
            self.assertLessEqual(self._relerr(got, want), tol,
                                 f"mm M={M} K={K} N={N} {dtype}")

    def test_decode_addmm_bias_broadcast(self):
        import tensorplay as tp
        tensorplay.manual_seed(1234)
        K, N = 1024, 1030
        for M, dtype, tol in (
            (1, tensorplay.float32, 2.0e-5),
            (4, tensorplay.float16, 2.0e-3),
        ):
            a = tensorplay.randn((M, K), device="cpu").to(dtype)
            b = tensorplay.randn((K, N), device="cpu").to(dtype)
            bias_vec = tensorplay.randn((N,), device="cpu").to(dtype)
            got = tensorplay.addmm(
                bias_vec.to("vulkan"), a.to("vulkan"), b.to("vulkan")
            ).cpu().numpy()
            want = (a.numpy().astype(np.float32) @
                    b.numpy().astype(np.float32) +
                    bias_vec.numpy().astype(np.float32))
            self.assertLessEqual(self._relerr(got, want), tol,
                                 f"addmm 1d M={M} {dtype}")
            bias_mat = tensorplay.randn((M, N), device="cpu").to(dtype)
            got = tensorplay.addmm(
                bias_mat.to("vulkan"), a.to("vulkan"), b.to("vulkan")).cpu().numpy()
            want = (a.numpy().astype(np.float32) @
                    b.numpy().astype(np.float32) +
                    bias_mat.numpy().astype(np.float32))
            self.assertLessEqual(self._relerr(got, want), tol,
                                 f"addmm 2d M={M} {dtype}")


class TestVulkanRmsNorm(unittest.TestCase):
    @staticmethod
    def _relerr(got, want):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        denom = max(float(np.abs(want).max()), 1.0)
        return float(np.abs(got - want).max()) / denom

    @staticmethod
    def _rms_ref(x, w, eps, axes=-1):
        # Root-mean-square normalization computed in fp32 over the given
        # trailing axes: y = x * rsqrt(mean(x^2) + eps) * w.
        x = np.asarray(x, dtype=np.float32)
        sq = (x.astype(np.float32) ** 2).mean(axis=axes, keepdims=True)
        rstd = 1.0 / np.sqrt(sq + eps)
        out = x * rstd
        if w is not None:
            out = out * np.asarray(w, dtype=np.float32)
        return out

    def test_single_dispatch_trailing_axis(self):
        # The fused single-dispatch kernel covers normalization over exactly
        # the trailing (texture width) axis across dtypes and ranks.
        import tensorplay as tp
        tensorplay.manual_seed(1234)
        cases = []
        for shape in ((1024,), (8, 1024), (1, 1, 1024), (2, 3, 256),
                      (1, 1, 1000)):
            cases.append((shape, tensorplay.float32, 2.0e-5))
            cases.append((shape, tensorplay.float16, 3.0e-3))
        for shape, dtype, tol in cases:
            x = tensorplay.randn(shape, device="cpu").to(dtype)
            for use_w in (False, True):
                w = (
                    tensorplay.randn((shape[-1],), device="cpu").to(dtype)
                    if use_w else None
                )
                got = tp._C.rms_norm(
                    x.to("vulkan"),
                    [shape[-1]],
                    w.to("vulkan") if w is not None else None,
                    1e-6,
                ).cpu().numpy()
                want = self._rms_ref(
                    x.numpy(),
                    w.numpy() if w is not None else None,
                    1e-6,
                )
                self.assertLessEqual(
                    self._relerr(got, want),
                    tol,
                    f"rms {shape} {dtype} w={int(use_w)}",
                )

    def test_composite_fallback_multi_axis(self):
        # Normalized shapes covering more than the trailing axis keep the
        # multi-kernel composite path; its numerics must stay intact.
        import tensorplay as tp
        tensorplay.manual_seed(1234)
        for dtype, tol in ((tensorplay.float32, 2.0e-5),
                           (tensorplay.float16, 3.0e-3)):
            x = tensorplay.randn((4, 8, 16), device="cpu").to(dtype)
            got = tp._C.rms_norm(
                x.to("vulkan"), [8, 16], None, 1e-6
            ).cpu().numpy()
            want = self._rms_ref(x.numpy(), None, 1e-6, axes=(1, 2))
            self.assertLessEqual(
                self._relerr(got, want), tol,
                f"rms multi-axis {dtype}")


class TestVulkanPackedParameterCache(unittest.TestCase):
    # Matmul relayouts its operands into packed textures and reuses them per
    # parameter identity, so these cases pin the cache contract: an in-place
    # update must be picked up, one tensor used in two roles must keep two
    # packs, and a working set far beyond a handful of entries must stay
    # numerically stable across repeated passes.

    @staticmethod
    def _relerr(got, want):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        denom = max(float(np.abs(want).max()), 1.0)
        return float(np.abs(got - want).max()) / denom

    def test_inplace_weight_update_rereads_packed_operand(self):
        import tensorplay as tp
        tensorplay.manual_seed(21)
        x = tensorplay.randn((16, 48), device="cpu").to(tensorplay.float32)
        w = tensorplay.randn((48, 24), device="cpu").to(tensorplay.float32)
        xv, wv = x.to("vulkan"), w.to("vulkan")

        first = (xv @ wv).cpu().numpy()
        self.assertLessEqual(
            self._relerr(first, x.numpy() @ w.numpy()), 2.0e-5)

        wv.mul_(2.0)
        w.mul_(2.0)
        second = (xv @ wv).cpu().numpy()
        self.assertLessEqual(
            self._relerr(second, x.numpy() @ w.numpy()), 2.0e-5,
            "in-place weight update was not picked up by the packed operand")
        self.assertGreater(
            self._relerr(second, first), 0.4,
            "reused pack survived the version bump")

        # A fresh tensor that happens to carry the same values must agree with
        # the updated one, so the identity is the storage, not the size.
        wv2 = w.to("vulkan")
        self.assertLessEqual(
            self._relerr((xv @ wv2).cpu().numpy(), x.numpy() @ w.numpy()),
            2.0e-5)

    def test_one_weight_in_two_roles_keeps_two_packs(self):
        # The left operand streams along K in the width-packed layout and the
        # right operand in the height-packed one; the same tensor playing both
        # parts must not have one pack served for the other.
        tensorplay.manual_seed(22)
        s_cpu = tensorplay.randn((32, 32), device="cpu").to(tensorplay.float32)
        a_cpu = tensorplay.randn((16, 32), device="cpu").to(tensorplay.float32)
        b_cpu = tensorplay.randn((32, 16), device="cpu").to(tensorplay.float32)

        sv, av, bv = s_cpu.to("vulkan"), a_cpu.to("vulkan"), b_cpu.to("vulkan")
        cases = (
            ("as_rhs", lambda: (av @ sv).cpu().numpy(),
             lambda: a_cpu.numpy() @ s_cpu.numpy()),
            ("as_lhs", lambda: (sv @ bv).cpu().numpy(),
             lambda: s_cpu.numpy() @ b_cpu.numpy()),
            ("both", lambda: (sv @ sv).cpu().numpy(),
             lambda: s_cpu.numpy() @ s_cpu.numpy()),
        )
        for label, run, want in cases:
            self.assertLessEqual(
                self._relerr(run(), want()), 2.0e-5,
                f"packed operand role collision: {label}")

        sv.mul_(-1.0)
        s_cpu.mul_(-1.0)
        for label, run, want in cases:
            self.assertLessEqual(
                self._relerr(run(), want()), 2.0e-5,
                f"packed operand stale after update: {label}")

    def test_working_set_beyond_a_few_entries(self):
        # A decoder references one weight per projection per layer, so the
        # resident set is far larger than the handful of tensors the early
        # convolution workloads needed; repeated passes over it must agree.
        tensorplay.manual_seed(23)
        count = 300
        x_cpu = tensorplay.randn((8, 64), device="cpu").to(tensorplay.float32)
        weights_cpu = [
            tensorplay.randn((64, 64), device="cpu").to(tensorplay.float32)
            for _ in range(count)
        ]
        xv = x_cpu.to("vulkan")
        wv = [w.to("vulkan") for w in weights_cpu]

        first = [(xv @ w).cpu().numpy() for w in wv]
        second = [(xv @ w).cpu().numpy() for w in wv]
        for index in (0, count // 2, count - 1):
            want = x_cpu.numpy() @ weights_cpu[index].numpy()
            self.assertLessEqual(
                self._relerr(first[index], want), 2.0e-5,
                f"pass 1 weight {index}")
            self.assertLessEqual(
                self._relerr(second[index], want), 2.0e-5,
                f"pass 2 weight {index}")
        self.assertTrue(
            np.array_equal(first[0], second[0]),
            "repeated pass over a cached operand changed the result")


class TestVulkanLinearPayload(unittest.TestCase):
    # A row wider than the device's image-axis limit is held in a storage
    # buffer instead of a texture.  That is the shape of a large-vocabulary
    # logit vector, so the decode-time elementwise ops must have a linear form.
    WIDTH = 200000

    def test_clamp_bounds_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(7)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 8.0).numpy()
        x = tensorplay.from_numpy(x_cpu.astype(np.float32)).to("vulkan")
        for lo, hi in ((-1.5, 2.5), (0.0, None), (None, 1.0), (-1e9, 1e9)):
            want = np.clip(x_cpu, lo, hi)
            got = F.clamp(x, min=lo, max=hi).cpu().numpy()
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)

    def test_clamp_inplace_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(11)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 8.0).numpy()
        y = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        out = F.clamp_(y, min=-1.0, max=1.0)
        np.testing.assert_allclose(
            out.cpu().numpy(), np.clip(x_cpu, -1.0, 1.0), rtol=0, atol=1e-6)
        # In place means the same payload: a second read-back must show the
        # clip already applied, not the original values.
        np.testing.assert_allclose(
            y.cpu().numpy(), np.clip(x_cpu, -1.0, 1.0), rtol=0, atol=1e-6)

    def test_clamp_int32_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(13)
        x_cpu = np.floor(tp.randn((self.WIDTH,), device="cpu")
                          * 300.0).numpy().astype(np.int32)
        x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        for lo, hi in ((-100, 100), (0, None), (None, -500)):
            want = np.clip(x_cpu, lo, hi)
            got = F.clamp(x, min=lo, max=hi).cpu().numpy()
            np.testing.assert_array_equal(got, want)

    def test_inplace_pointwise_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(17)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        sig = 1.0 / (1.0 + np.exp(-x_cpu))
        cases = [
            ("abs_", (), np.abs(x_cpu)),
            ("neg_", (), -x_cpu),
            ("sigmoid_", (), sig),
            ("silu_", (), x_cpu * sig),
            ("relu6_", (), np.clip(x_cpu, 0.0, 6.0)),
            ("hardsigmoid_", (), np.clip(x_cpu / 6.0 + 0.5, 0.0, 1.0)),
            ("hardswish_", (), x_cpu * np.clip(x_cpu / 6.0 + 0.5, 0.0, 1.0)),
            ("mish_", (), x_cpu * np.tanh(np.log1p(np.exp(x_cpu)))),
            ("exp_", (), np.exp(x_cpu)),
            ("threshold_", (0.0, -1.0), np.where(x_cpu > 0.0, x_cpu, -1.0)),
            ("leaky_relu_", (0.1,), np.where(x_cpu > 0.0, x_cpu, 0.1 * x_cpu)),
            ("mul_", (2.0,), 2.0 * x_cpu),
            ("add_", (1.5,), x_cpu + 1.5),
        ]
        for name, args, want in cases:
            y = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            out = getattr(F, name)(y, *args)
            got = out.cpu().numpy()
            self.assertIs(out, y, f"{name} did not return the same tensor")
            np.testing.assert_allclose(
                got, want, rtol=1e-4, atol=1e-4, err_msg=f"{name} left the payload untouched or wrong")

    def test_gelu_inplace_over_linear_payload(self):
        from math import erf, sqrt
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(19)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        # The default form is the exact one: 0.5 * x * (1 + erf(x / sqrt(2))).
        want = 0.5 * x_cpu * (
            1.0 + np.vectorize(erf)(x_cpu / sqrt(2.0))).astype(np.float32)
        y = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        out = F.gelu_(y)
        self.assertIs(out, y)
        np.testing.assert_allclose(
            out.cpu().numpy(), want, rtol=1e-4, atol=1e-4)

    def test_factory_fill_over_linear_payload(self):
        import tensorplay as tp
        # A payload too wide for image storage is cleared and offset by the
        # linear fill kernels; an uninitialized allocation would not read as
        # the requested constant.
        np.testing.assert_array_equal(
            tensorplay.zeros((self.WIDTH,), device="vulkan").cpu().numpy(),
            np.zeros(self.WIDTH, dtype=np.float32))
        np.testing.assert_array_equal(
            tensorplay.ones((self.WIDTH,), device="vulkan").cpu().numpy(),
            np.ones(self.WIDTH, dtype=np.float32))
        np.testing.assert_array_equal(
            tensorplay.full((self.WIDTH,), 3.25, device="vulkan").cpu().numpy(),
            np.full(self.WIDTH, 3.25, dtype=np.float32))
        np.testing.assert_array_equal(
            tensorplay.full(
                (self.WIDTH,), -7, dtype=tp.int32, device="vulkan").cpu().numpy(),
            np.full(self.WIDTH, -7, dtype=np.int32))


    def test_compare_scalar_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(23)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        cases = (
            ("gt", lambda: x > 0.5, x_cpu > 0.5),
            ("ge", lambda: x >= -1.25, x_cpu >= -1.25),
            ("lt", lambda: x < 0.0, x_cpu < 0.0),
            ("le", lambda: x <= 2.0, x_cpu <= 2.0),
            ("eq", lambda: x == 0.0, x_cpu == 0.0),
            ("ne", lambda: x != 0.0, x_cpu != 0.0),
        )
        for name, apply_op, want in cases:
            got = apply_op()
            self.assertEqual(got.dtype, tensorplay.bool)
            np.testing.assert_array_equal(
                got.cpu().numpy(), want, err_msg=name)

    def test_compare_tensor_and_select_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        tensorplay.manual_seed(29)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        y_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        y = tensorplay.from_numpy(y_cpu.copy()).to("vulkan")

        mask = x < y
        self.assertEqual(mask.dtype, tensorplay.bool)
        np.testing.assert_array_equal(mask.cpu().numpy(), x_cpu < y_cpu)

        selected = F.where(mask, x, y).cpu().numpy()
        np.testing.assert_array_equal(selected, np.where(x_cpu < y_cpu, x_cpu, y_cpu))

        filled = F.masked_fill(x, mask, -1000.0).cpu().numpy()
        np.testing.assert_array_equal(
            filled, np.where(x_cpu < y_cpu, -1000.0, x_cpu))

    def test_compare_int32_over_linear_payload(self):
        import tensorplay as tp
        tensorplay.manual_seed(31)
        x_cpu = np.floor(tp.randn((self.WIDTH,), device="cpu")
                          * 100.0).numpy().astype(np.int32)
        x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        np.testing.assert_array_equal((x >= 0).cpu().numpy(), x_cpu >= 0)
        np.testing.assert_array_equal((x != 7).cpu().numpy(), x_cpu != 7)

    def test_extremum_over_linear_payload(self):
        import tensorplay as tp
        tensorplay.manual_seed(41)
        # A whole-payload extremum has no row to spread work over, so it
        # reduces fixed tiles and then the tile results.  Pick a length that
        # leaves a partial tile so the tail clamp is covered too.
        width = self.WIDTH + 37
        x_cpu = (tp.randn((width,), device="cpu") * 3.0).numpy()
        x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")

        np.testing.assert_array_equal(
            x.max().cpu().numpy(), np.max(x_cpu), err_msg="max over tiles")
        np.testing.assert_array_equal(
            x.min().cpu().numpy(), np.min(x_cpu), err_msg="min over tiles")
        np.testing.assert_array_equal(
            tp.amax(x).cpu().numpy(), np.max(x_cpu), err_msg="amax global")
        np.testing.assert_array_equal(
            tp.amin(x).cpu().numpy(), np.min(x_cpu), err_msg="amin global")

        # Few rows, long axis: one pass with one invocation per row.
        rows_cpu = (tp.randn((3, 80000), device="cpu") * 3.0).numpy()
        rows_v = tensorplay.from_numpy(rows_cpu.copy()).to("vulkan")
        np.testing.assert_array_equal(
            tp.amax(rows_v, dim=[1]).cpu().numpy(),
            rows_cpu.max(axis=1),
            err_msg="amax along the long axis")
        np.testing.assert_array_equal(
            tp.amin(rows_v, dim=[1]).cpu().numpy(),
            rows_cpu.min(axis=1),
            err_msg="amin along the long axis")

        # Many rows, short axis: the walk strides by the trailing extent.
        np.testing.assert_array_equal(
            tp.amax(rows_v, dim=[0]).cpu().numpy(),
            rows_cpu.max(axis=0),
            err_msg="amax across rows")
        np.testing.assert_array_equal(
            tp.amin(rows_v, dim=[0]).cpu().numpy(),
            rows_cpu.min(axis=0),
            err_msg="amin across rows")
        np.testing.assert_array_equal(
            tp.amax(rows_v, dim=[0], keepdim=True).cpu().numpy(),
            rows_cpu.max(axis=0, keepdims=True),
            err_msg="amax keepdim")

    def test_tiled_sum_over_linear_payload(self):
        import tensorplay as tp
        tensorplay.manual_seed(43)
        # A whole-payload sum has no row to spread over either, so it tiles the
        # payload and adds the tile results.  The order changes, so compare
        # with a tolerance rather than bit for bit.
        width = self.WIDTH + 37
        x_cpu = tp.randn((width,), device="cpu").numpy()
        x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")

        np.testing.assert_allclose(
            float(x.sum().cpu().numpy()),
            float(x_cpu.sum()),
            rtol=1e-3,
            atol=1e-3,
            err_msg="sum over tiles")
        np.testing.assert_allclose(
            float(x.mean().cpu().numpy()),
            float(x_cpu.mean()),
            rtol=1e-3,
            atol=1e-3,
            err_msg="mean over tiles")

        # Many-row reduces keep the single pass, including variance, whose
        # raw moments stay inside one row walk.
        rows_cpu = tp.randn((4, 50000), device="cpu").numpy()
        rows_v = tensorplay.from_numpy(rows_cpu.copy()).to("vulkan")
        np.testing.assert_allclose(
            rows_v.sum(dim=[1]).cpu().numpy(),
            rows_cpu.sum(axis=1),
            rtol=1e-3,
            atol=1e-3,
            err_msg="row sums")
        np.testing.assert_allclose(
            tp.var(rows_v, dim=[1]).cpu().numpy(),
            rows_cpu.var(axis=1),
            rtol=1e-3,
            atol=1e-3,
            err_msg="row variances")


    def test_unaligned_widths_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # Linear kernels move four lanes per word, so a width that is not a
        # multiple of four leaves a partial last word.  Writing that word whole
        # must not disturb the lanes before it, and the read-back length has to
        # stop at the logical end.
        tensorplay.manual_seed(37)
        for width in (self.WIDTH - 3, self.WIDTH - 1, self.WIDTH + 1,
                      self.WIDTH + 6):
            x_cpu = (tp.randn((width,), device="cpu") * 2.0).numpy()
            y_cpu = (tp.randn((width,), device="cpu") * 2.0).numpy()
            x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            y = tensorplay.from_numpy(y_cpu.copy()).to("vulkan")

            inplace = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            F.clamp_(inplace, -1.0, 1.0)
            np.testing.assert_array_equal(
                inplace.cpu().numpy(),
                np.clip(x_cpu, -1.0, 1.0),
                err_msg=f"clamp_ at width {width}")

            z = tp.from_numpy(x_cpu.copy()).to("vulkan")
            F.abs_(z)
            np.testing.assert_array_equal(
                z.cpu().numpy(), np.abs(x_cpu), err_msg=f"abs_ at width {width}")

            mask = x < y
            self.assertEqual(tuple(mask.shape), (width,))
            np.testing.assert_array_equal(
                mask.cpu().numpy(), x_cpu < y_cpu, err_msg="lt at odd width")
            np.testing.assert_array_equal(
                F.where(mask, x, y).cpu().numpy(),
                np.where(x_cpu < y_cpu, x_cpu, y_cpu),
                err_msg="where at odd width")
            np.testing.assert_array_equal(
                F.masked_fill(x, mask, -7.0).cpu().numpy(),
                np.where(x_cpu < y_cpu, -7.0, x_cpu),
                err_msg="masked_fill at odd width")


if __name__ == "__main__":
    unittest.main()
