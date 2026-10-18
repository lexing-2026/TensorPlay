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

    def test_rotary_embedding_over_linear_storage(self):
        import tensorplay as tp
        # A token axis longer than the image axis limit has no texture form at
        # all, so the rotation consumes the row-major payload directly.  The
        # cosine/sine tables keep their own storage: one position still fits an
        # image, so a long activation with a short table is the mixed case.
        tensorplay.manual_seed(211)
        for B, S, D, table_len, off in (
                (3, 20000, 64, 20000, 0),
                (2, 20003, 32, 20008, 5),
                (1, 16400, 64, 1, 0),
                (2, 8, 16400, 32, 0)):
            x_cpu = (tensorplay.randn((B, S, D), device="cpu") * 0.3).numpy()
            cos_cpu, sin_cpu = _rope_tables(D, table_len)
            rotated = tp.functional.rotary_embedding(
                tensorplay.from_numpy(x_cpu.copy()).to("vulkan"),
                tensorplay.from_numpy(cos_cpu.copy()).to("vulkan"),
                tensorplay.from_numpy(sin_cpu.copy()).to("vulkan"),
                position_offset=off)
            self.assertTrue(rotated.device.is_vulkan())
            np.testing.assert_allclose(
                rotated.cpu().numpy(),
                _rope_reference(x_cpu, cos_cpu, sin_cpu, off),
                rtol=1e-5, atol=1e-5,
                err_msg=f"rope over linear storage at B={B} S={S} D={D}")

        # A decode step reads one short activation from a cached table that has
        # outgrown the image limit, so the payload keeps its texture form while
        # the tables are read as flat arrays.
        for S, D, table_len, off, dtype, tol in (
                (1, 64, 20000, 19999, tensorplay.float32, 3.0e-4),
                (9, 32, 20000, 512, tensorplay.float32, 3.0e-4),
                (1, 128, 16500, 400, tensorplay.float16, 3.0e-2)):
            x_cpu = (tensorplay.randn((2, S, D), device="cpu")
                     * 0.3).to(dtype).numpy()
            cos_cpu, sin_cpu = _rope_tables(D, table_len)
            rotated = tp.functional.rotary_embedding(
                tensorplay.from_numpy(x_cpu.copy()).to("vulkan"),
                tensorplay.from_numpy(cos_cpu.copy()).to("vulkan"),
                tensorplay.from_numpy(sin_cpu.copy()).to("vulkan"),
                position_offset=off)
            self.assertEqual(rotated.dtype, dtype)
            np.testing.assert_allclose(
                rotated.cpu().numpy().astype(np.float32),
                _rope_reference(x_cpu.astype(np.float32), cos_cpu, sin_cpu, off),
                rtol=0, atol=tol,
                err_msg=f"rope over a linear table at S={S} D={D}")

    def test_half_precision_rotary_embedding_over_linear_payload(self):
        import tensorplay as tp
        # Decoding keeps the activations and the cached tables in half
        # precision, and a rotated pair is exactly the two lanes of one 32-bit
        # word, so an invocation owns and writes that word whole.  A linear
        # table is read as words of its own element type, which pins its width
        # to the payload's.
        tensorplay.manual_seed(4207)
        for B, S, D, table_len, off, flat in (
                (2, 20000, 64, 20000, 0, False),
                (3, 20003, 32, 20010, 5, False),
                (1, 16400, 64, 1, 0, False),
                (2, 8, 16400, 32, 0, False),
                (1, 16390, 6, 16390, 0, False),
                (1, 16385, 128, 1, 0, True)):
            x_cpu = (tensorplay.randn((B, S, D), device="cpu")
                     * 0.3).to(tensorplay.float16).numpy()
            cos_cpu, sin_cpu = _rope_tables(D, table_len)
            if flat:
                cos_cpu, sin_cpu = cos_cpu[0], sin_cpu[0]
            cos_cpu = cos_cpu.astype(np.float16)
            sin_cpu = sin_cpu.astype(np.float16)
            rotated = tp.functional.rotary_embedding(
                tensorplay.from_numpy(x_cpu.copy()).to("vulkan"),
                tensorplay.from_numpy(
                    np.ascontiguousarray(cos_cpu)).to("vulkan"),
                tensorplay.from_numpy(
                    np.ascontiguousarray(sin_cpu)).to("vulkan"),
                position_offset=off)
            self.assertEqual(rotated.dtype, tensorplay.float16)
            self.assertEqual(tuple(rotated.shape), x_cpu.shape)
            # The rotation is exact once the tables carry the payload width, so
            # compare the bit patterns instead of a tolerance.
            want = _rope_reference(
                x_cpu.astype(np.float32),
                cos_cpu.astype(np.float32),
                sin_cpu.astype(np.float32),
                off).astype(np.float16)
            np.testing.assert_array_equal(
                rotated.cpu().numpy().view(np.uint16),
                want.view(np.uint16),
                err_msg=f"half rope at B={B} S={S} D={D} off={off}")

        # A float32 table would be read as pairs of halves over linear storage.
        with self.assertRaisesRegex(RuntimeError, "payload dtype"):
            tp.functional.rotary_embedding(
                tensorplay.from_numpy(
                    (np.random.randn(1, 16400, 32) * 0.3)
                    .astype(np.float16)).to("vulkan"),
                tensorplay.from_numpy(np.zeros((20000, 16), np.float32)
                                      ).to("vulkan"),
                tensorplay.from_numpy(np.zeros((20000, 16), np.float32)
                                      ).to("vulkan"))

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

    def test_decode_product_over_linear_weight(self):
        import tensorplay as tp
        # A weight matrix wider than the image axis limit cannot be relayed
        # into a texture at all, so a decode-shaped product reads it as a
        # linear payload.  The reduction length is deliberately not a multiple
        # of four to cover the lanes a packed texel leaves over.
        tensorplay.manual_seed(47)
        for m, k, n in ((1, 520, 20000), (3, 1000, 17000), (2, 521, 20003)):
            a_cpu = (tp.randn((m, k), device="cpu") * 0.5).numpy()
            b_cpu = (tp.randn((k, n), device="cpu") * 0.02).numpy()
            a = tensorplay.from_numpy(a_cpu.copy()).to("vulkan")
            b = tensorplay.from_numpy(b_cpu.copy()).to("vulkan")

            out = tp.mm(a, b)
            self.assertTrue(out.device.is_vulkan())
            np.testing.assert_allclose(
                out.cpu().numpy(), a_cpu @ b_cpu,
                rtol=1e-3, atol=1e-3, err_msg=f"mm m={m} k={k} n={n}")

            bias_cpu = (tp.randn((n,), device="cpu") * 0.25).numpy()
            bias = tensorplay.from_numpy(bias_cpu.copy()).to("vulkan")
            shifted = tp.addmm(bias, a, b)
            np.testing.assert_allclose(
                shifted.cpu().numpy(), a_cpu @ b_cpu + bias_cpu,
                rtol=1e-3, atol=1e-3, err_msg=f"addmm m={m} k={k} n={n}")

    def test_decode_product_with_linear_activation(self):
        import tensorplay as tp
        # A reduction axis longer than four times the image axis limit leaves
        # the activation in a buffer too, and a narrow result still fits an
        # image, so the fold pass writes texels while the reads stay linear.
        tensorplay.manual_seed(53)
        for n in (200, 197):
            m, k = 1, 70000
            a_cpu = (tp.randn((m, k), device="cpu") * 0.5).numpy()
            b_cpu = (tp.randn((k, n), device="cpu") * 0.005).numpy()
            a = tensorplay.from_numpy(a_cpu.copy()).to("vulkan")
            b = tensorplay.from_numpy(b_cpu.copy()).to("vulkan")

            out = tp.mm(a, b)
            np.testing.assert_allclose(
                out.cpu().numpy(), a_cpu @ b_cpu,
                rtol=1e-3, atol=1e-3, err_msg=f"linear operands at n={n}")

            bias_cpu = (tp.randn((n,), device="cpu") * 0.25).numpy()
            bias = tensorplay.from_numpy(bias_cpu.copy()).to("vulkan")
            shifted = tp.addmm(bias, a, b)
            np.testing.assert_allclose(
                shifted.cpu().numpy(), a_cpu @ b_cpu + bias_cpu,
                rtol=1e-3, atol=1e-3,
                err_msg=f"addmm with linear operands at n={n}")

    def test_decode_product_over_layer_layout_weight(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A linear layer keeps its weights in (out_features, in_features) order,
        # so the reduction axis runs along memory and every output column
        # streams a row of its own.  Reduction lengths that divide by four let a
        # row open on a vector boundary and are walked four words per fetch;
        # lengths that do not stay element-wise.  The narrow shapes keep the
        # result inside an image while the reads stay linear.
        tensorplay.manual_seed(59)
        for m, k, n in ((1, 512, 20000), (2, 516, 17000), (1, 517, 20003),
                        (3, 1001, 19000), (1, 70000, 200), (1, 66003, 197)):
            x_cpu = (tp.randn((m, k), device="cpu") * 0.5).numpy()
            w_cpu = (tp.randn((n, k), device="cpu") * 0.02).numpy()
            bias_cpu = (tp.randn((n,), device="cpu") * 0.25).numpy()
            x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            weight = tensorplay.from_numpy(w_cpu.copy()).to("vulkan")
            bias = tensorplay.from_numpy(bias_cpu.copy()).to("vulkan")

            expect = x_cpu @ w_cpu.T
            out = F.linear(x, weight)
            self.assertTrue(out.device.is_vulkan())
            np.testing.assert_allclose(
                out.cpu().numpy(), expect, rtol=1e-3, atol=1e-3,
                err_msg=f"linear m={m} k={k} n={n}")
            np.testing.assert_allclose(
                F.linear(x, weight, bias).cpu().numpy(), expect + bias_cpu,
                rtol=1e-3, atol=1e-3,
                err_msg=f"linear with bias m={m} k={k} n={n}")

    def test_tiled_product_over_linear_weight(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # Past a handful of rows the decode path would re-read the weight once
        # per row, so a product with a linear-stored operand stages 32x32 blocks
        # of both operands through shared memory instead.  Row, reduction and
        # output counts are all chosen off the tile edges to cover the staged
        # zeros outside the extents, and the wide-input shapes keep the
        # activation in linear storage while the result still fits an image.
        tensorplay.manual_seed(61)
        for m, k, n in ((37, 640, 20003), (64, 1000, 17000), (40, 20003, 100)):
            a_cpu = (tp.randn((m, k), device="cpu") * 0.5).numpy()
            b_cpu = (tp.randn((k, n), device="cpu") * 0.02).numpy()
            bias_cpu = (tp.randn((n,), device="cpu") * 0.25).numpy()
            a = tensorplay.from_numpy(a_cpu.copy()).to("vulkan")
            b = tensorplay.from_numpy(b_cpu.copy()).to("vulkan")
            bias = tensorplay.from_numpy(bias_cpu.copy()).to("vulkan")

            expect = a_cpu @ b_cpu
            out = tp.mm(a, b)
            self.assertTrue(out.device.is_vulkan())
            np.testing.assert_allclose(
                out.cpu().numpy(), expect, rtol=1e-3, atol=1e-3,
                err_msg=f"tiled mm m={m} k={k} n={n}")
            np.testing.assert_allclose(
                tp.addmm(bias, a, b).cpu().numpy(), expect + bias_cpu,
                rtol=1e-3, atol=1e-3,
                err_msg=f"tiled addmm m={m} k={k} n={n}")

        for m, k, n in ((33, 517, 20003), (8, 2048, 16391)):
            x_cpu = (tp.randn((m, k), device="cpu") * 0.5).numpy()
            w_cpu = (tp.randn((n, k), device="cpu") * 0.02).numpy()
            bias_cpu = (tp.randn((n,), device="cpu") * 0.25).numpy()
            x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            weight = tensorplay.from_numpy(w_cpu.copy()).to("vulkan")
            bias = tensorplay.from_numpy(bias_cpu.copy()).to("vulkan")

            expect = x_cpu @ w_cpu.T
            np.testing.assert_allclose(
                F.linear(x, weight, bias).cpu().numpy(), expect + bias_cpu,
                rtol=1e-3, atol=1e-3,
                err_msg=f"tiled linear m={m} k={k} n={n}")

    def test_fmod_exact_multiples_over_linear_payload(self):
        import tensorplay.functional as F
        # The remainder is formed as x - y*trunc(x/y), so a quotient that lands
        # one unit below its true integer makes the dividend come back whole
        # instead of a zero remainder.  Exact multiples pin that boundary, and
        # one guarded step has to clear it for every lane.
        vals = np.array(
            [-2.544922, 1.262695, 4.300781, -2.027344, 7.0, 3.0, -6.0, 5.0],
            dtype=np.float32)
        divs = np.array(
            [2.544922, 1.262695, 2.150391, 1.013672, 2.0, 3.0, 2.0, 2.0],
            dtype=np.float32)
        for width in (self.WIDTH, vals.size):
            for dtype in (np.float32, np.float16):
                x = np.resize(vals, width).astype(dtype)
                b = np.resize(divs, width).astype(dtype)
                got = F.fmod(
                    tensorplay.from_numpy(x.copy()).to("vulkan"),
                    tensorplay.from_numpy(b.copy()).to("vulkan"),
                ).cpu().numpy().astype(np.float32)
                want = np.fmod(
                    x.astype(np.float64), b.astype(np.float64)).astype(dtype)
                np.testing.assert_allclose(
                    got, want.astype(np.float32), rtol=1e-3, atol=1e-3,
                    err_msg=(f"fmod over {np.dtype(dtype).name} payload of "
                             f"width {width} returned the dividend instead of "
                             f"the remainder"))

    def test_rms_norm_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.nn.functional as F
        # A row or a batch axis past the image axis limit has no texture form, so
        # the normalization consumes the row-major payload directly.  One
        # workgroup reduces one row through shared memory, which is why the row
        # count, not the lane count, is what a single dispatch axis has to
        # cover: the last case needs more rows than one axis holds.  The weight
        # keeps its own storage, so both the short image-stored vector and the
        # wide linear one appear here.
        tensorplay.manual_seed(71)
        for shape, span, with_weight in (
                ((20000, 512), 512, True),
                ((20000, 512), 512, False),
                ((3, 17000), 17000, True),
                ((40000, 2), 2, False),
                ((2, 3, 9000, 128), 128, True),
                ((70000, 128), 128, True)):
            x_cpu = (tp.randn(shape, device="cpu") * 0.7).numpy()
            w_cpu = ((tp.randn((span,), device="cpu") + 1.0) * 0.5).numpy() \
                if with_weight else None
            x = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            w = tensorplay.from_numpy(w_cpu.copy()).to("vulkan") if with_weight \
                else None
            got = F.rms_norm(x, [span], w, 1.0e-5)
            self.assertTrue(got.device.is_vulkan())
            xf = x_cpu.astype(np.float64)
            want = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1.0e-5)
            if with_weight:
                want = want * w_cpu.astype(np.float64)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), want,
                rtol=1e-5, atol=1e-5,
                err_msg=f"rms_norm over linear payload at shape {shape} "
                        f"span {span} weight {with_weight}")

    def test_half_precision_pointwise_over_linear_payload(self):
        from math import erf
        import tensorplay as tp
        import tensorplay.functional as F
        # Device runtimes keep weights and activations in half precision, so a
        # payload past the image axis limit arrives as a half buffer.  The
        # linear kernels pack two lanes into one 32-bit word, low lane first,
        # and compute in single precision with a single narrowing at the store;
        # the expectation below is that arithmetic rounded through half.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        def upload(a):
            return tensorplay.from_numpy(
                np.asarray(a, dtype=np.float16)).to("vulkan")

        tensorplay.manual_seed(51)
        x_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        y_cpu = (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        pos_cpu = np.abs(x_cpu) + 0.5
        safe_cpu = np.abs(y_cpu) + 0.75
        x16, y16 = narrowed(x_cpu), narrowed(y_cpu)
        pos16, safe16 = narrowed(pos_cpu), narrowed(safe_cpu)
        sig = 1.0 / (1.0 + np.exp(-x16))

        # A divisor away from zero: near-zero denominators make the single
        # precision quotient of the kernel differ from the wider reference.
        unary = [
            ("abs", F.abs, np.abs(x16), x_cpu),
            ("neg", F.neg, -x16, x_cpu),
            ("floor", F.floor, np.floor(x16), x_cpu),
            ("exp", F.exp, np.exp(x16), x_cpu),
            ("tanh", F.tanh, np.tanh(x16), x_cpu),
            ("sigmoid", F.sigmoid, sig, x_cpu),
            ("silu", F.silu, x16 * sig, x_cpu),
            ("relu6", F.relu6, np.clip(x16, 0.0, 6.0), x_cpu),
            ("hardsigmoid", F.hardsigmoid,
             np.clip(x16 / 6.0 + 0.5, 0.0, 1.0), x_cpu),
            ("hardswish", F.hardswish,
             x16 * np.clip(x16 / 6.0 + 0.5, 0.0, 1.0), x_cpu),
            ("sqrt", F.sqrt, np.sqrt(pos16), pos_cpu),
            ("rsqrt", F.rsqrt, 1.0 / np.sqrt(pos16), pos_cpu),
            ("reciprocal", F.reciprocal, 1.0 / pos16, pos_cpu),
            ("gelu", F.gelu, 0.5 * x16 * (
                1.0 + np.vectorize(erf)(x16 / np.sqrt(2.0))), x_cpu),
        ]
        for name, fn, want, src in unary:
            got = fn(upload(src))
            self.assertEqual(got.dtype, tensorplay.float16,
                             f"{name} widened a half linear payload")
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32), narrowed(want),
                rtol=2e-3, atol=2e-3,
                err_msg=f"{name} on a half linear payload")

        for name, want in (("abs_", np.abs(x16)), ("exp_", np.exp(x16)),
                           ("sigmoid_", sig), ("floor_", np.floor(x16)),
                           ("relu_", np.clip(x16, 0.0, None))):
            v = upload(x_cpu)
            out = getattr(F, name)(v)
            self.assertIs(out, v, f"{name} did not return the same tensor")
            np.testing.assert_allclose(
                out.cpu().numpy().astype(np.float32), narrowed(want),
                rtol=2e-3, atol=2e-3,
                err_msg=f"{name} left the half payload untouched or wrong")

        for lo, hi in ((-1.5, 2.5), (0.0, None), (None, 1.0)):
            got = F.clamp(upload(x_cpu), min=lo, max=hi).cpu().numpy()
            np.testing.assert_allclose(
                got.astype(np.float32), narrowed(np.clip(x16, lo, hi)),
                rtol=0, atol=1e-6, err_msg=f"clamp({lo}, {hi}) on a half payload")

        parametric = [
            ("leaky_relu", lambda v: F.leaky_relu(v, 0.1),
             np.where(x16 > 0, x16, 0.1 * x16)),
            ("threshold", lambda v: F.threshold(v, 0.0, -1.0),
             np.where(x16 > 0.0, x16, -1.0)),
            ("hardshrink", lambda v: F.hardshrink(v, 0.5),
             np.where(np.abs(x16) > 0.5, x16, 0.0)),
            # An integral exponent keeps the kernel on its sign-preserving
            # branch of the power formula.
            ("pow_scalar", lambda v: F.pow(v, 3.0),
             np.sign(x16) * np.power(np.abs(x16), 3.0)),
        ]
        for name, fn, want in parametric:
            got = fn(upload(x_cpu))
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32), narrowed(want),
                rtol=5e-3, atol=5e-3, err_msg=f"{name} on a half linear payload")

        binary = [
            ("add", F.add, x16 + y16, y_cpu),
            ("sub", F.sub, x16 - y16, y_cpu),
            ("mul", F.mul, x16 * y16, y_cpu),
            ("div", F.div, x16 / safe16, safe_cpu),
            ("maximum", F.maximum, np.maximum(x16, safe16), safe_cpu),
            ("minimum", F.minimum, np.minimum(x16, safe16), safe_cpu),
        ]
        for name, fn, want, other in binary:
            got = fn(upload(x_cpu), upload(other))
            self.assertEqual(got.dtype, tensorplay.float16,
                             f"{name} widened a half linear payload")
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32), narrowed(want),
                rtol=2e-3, atol=2e-3,
                err_msg=f"{name} over two half linear payloads")

        v = upload(x_cpu)
        np.testing.assert_allclose(
            F.add(v, upload(y_cpu), alpha=0.5).cpu().numpy().astype(np.float32),
            narrowed(x16 + 0.5 * y16), rtol=2e-3, atol=2e-3,
            err_msg="scaled add over half linear payloads")
        v = upload(x_cpu)
        np.testing.assert_allclose(
            F.rsub(v, 1.0).cpu().numpy().astype(np.float32),
            narrowed(1.0 - x16), rtol=2e-3, atol=2e-3,
            err_msg="scalar reverse subtract over a half linear payload")

        # An odd length leaves one padding lane in the final word; the lane
        # before it has to survive, and the length still bounds every read.
        for width in (self.WIDTH - 1, self.WIDTH + 1, self.WIDTH + 3):
            xs = narrowed((tp.randn((width,), device="cpu") * 3.0).numpy())
            ys = narrowed((tp.randn((width,), device="cpu") * 3.0).numpy())
            np.testing.assert_allclose(
                F.exp(upload(xs)).cpu().numpy().astype(np.float32),
                narrowed(np.exp(xs)), rtol=2e-3, atol=2e-3,
                err_msg=f"exp over a half payload of odd length {width}")
            np.testing.assert_allclose(
                F.add(upload(xs), upload(ys)).cpu().numpy().astype(np.float32),
                narrowed(xs + ys), rtol=2e-3, atol=2e-3,
                err_msg=f"add over a half payload of odd length {width}")

        # Short payloads stay in a texture, where the half twin of every pointwise
        # kernel already applied; check the same values to keep both forms honest.
        short = narrowed((tp.randn((64,), device="cpu") * 3.0).numpy())
        np.testing.assert_allclose(
            F.exp(upload(short)).cpu().numpy().astype(np.float32),
            narrowed(np.exp(short)), rtol=2e-3, atol=2e-3,
            err_msg="exp over a half texture payload")
        np.testing.assert_allclose(
            F.add(upload(short), upload(short)).cpu().numpy().astype(np.float32),
            narrowed(short + short), rtol=2e-3, atol=2e-3,
            err_msg="add over two half texture payloads")

    def test_half_precision_rms_norm_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.nn.functional as F
        # A half payload keeps the row stride in words, so the fused
        # normalization owns one word per invocation: the two lanes it holds are
        # written together and never by a neighbour.  The moment accumulates in
        # single precision and only the store narrows.  An odd axis would put two
        # rows inside one word, so it stays rejected rather than silently
        # clobbering a lane.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        tensorplay.manual_seed(83)
        for shape, span, with_weight in (
                ((20000, 512), 512, True),
                ((20000, 512), 512, False),
                ((3, 17000), 17000, True),
                ((40000, 2), 2, False),
                ((2, 3, 9000, 128), 128, True),
                ((1000, 4096), 4096, True)):
            x_cpu = (tp.randn(shape, device="cpu") * 0.8).numpy()
            w_cpu = ((tp.randn((span,), device="cpu") + 1.2) * 0.5).numpy() \
                if with_weight else None
            x16 = narrowed(x_cpu)
            w16 = narrowed(w_cpu) if with_weight else None
            x = tensorplay.from_numpy(x16.astype(np.float16)).to("vulkan")
            w = tensorplay.from_numpy(w16.astype(np.float16)).to("vulkan") \
                if with_weight else None
            got = F.rms_norm(x, [span], w, 1.0e-5)
            self.assertEqual(got.dtype, tensorplay.float16,
                             f"rms_norm widened a half payload at shape {shape}")
            xf = x16.astype(np.float64)
            want = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1.0e-5)
            if with_weight:
                want = want * w16.astype(np.float64)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32), narrowed(want),
                rtol=2e-3, atol=2e-3,
                err_msg=f"rms_norm over a half linear payload at shape {shape} "
                        f"span {span} weight {with_weight}")

        odd = tensorplay.from_numpy(narrowed(
            tp.randn((20000, 33), device="cpu").numpy()).astype(
                np.float16)).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "even normalized axis"):
            F.rms_norm(odd, [33], None, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
