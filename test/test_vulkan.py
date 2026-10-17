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


if __name__ == "__main__":
    unittest.main()
