import math
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


class TestVulkanDeviceIdentity(unittest.TestCase):
    # A payload names the slot it sits in, so the device a tensor reports and
    # the device a request resolves to compare equal however the tensor got
    # there.  Two-operand paths check that pair first: a device that differs
    # only in whether it spells the slot makes every mixed call refuse a pair
    # that shares the hardware.

    def test_every_creation_path_reports_the_same_device(self):
        host = tensorplay.arange(6, dtype=tensorplay.float32).reshape(2, 3)
        uploaded = host.to("vulkan")
        made = [
            ("requested slot", tensorplay.zeros((1,), device="vulkan:0").device),
            ("bare name", tensorplay.zeros((1,), device="vulkan").device),
            ("zeros", tensorplay.zeros((2, 3), device="vulkan").device),
            ("ones", tensorplay.ones((2, 3), device="vulkan").device),
            ("full", tensorplay.full((2, 3), 1.0, device="vulkan").device),
            ("empty", tensorplay.empty((2, 3), device="vulkan").device),
            ("arange", tensorplay.arange(6, device="vulkan").device),
            ("upload", uploaded.device),
            ("host array", tensorplay.tensor(
                host.cpu().numpy(), device="vulkan").device),
            ("sum result", (uploaded + uploaded).device),
            ("reduction result", uploaded.sum().device),
            ("reshape result", uploaded.reshape(3, 2).device),
        ]
        for name, device in made:
            self.assertEqual(
                device, uploaded.device,
                f"{name} reported {device} rather than {uploaded.device}")

    def test_a_device_built_payload_pairs_with_an_uploaded_one(self):
        import tensorplay.functional as F
        host = np.arange(6, dtype=np.float32).reshape(2, 3)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        uploaded = tensorplay.from_numpy(host.copy()).to("vulkan")
        for name, got, want in (
                ("add", F.add(built, uploaded), 2.0 * host),
                ("mul", F.mul(built, uploaded), host * host),
                ("sub", F.sub(uploaded, built), np.zeros_like(host)),
                ("div", F.div(uploaded, built + 1.0), host / (host + 1.0))):
            np.testing.assert_allclose(
                got.cpu().numpy(), want, rtol=1e-6, atol=1e-6,
                err_msg=f"{name} over a device-built and an uploaded payload")

    def test_device_built_tensors_survive_a_round_trip(self):
        host = np.arange(12, dtype=np.float32).reshape(3, 4)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        np.testing.assert_allclose(
            built.cpu().numpy(), host, rtol=0, atol=1e-6,
            err_msg="a device-built payload did not read back")

    def test_a_device_payload_prints_the_device_it_sits_on(self):
        # The summary line names the backend, so a Vulkan payload does not read
        # as if it lived on the other accelerator.
        text = repr(tensorplay.zeros((2, 2), device="vulkan"))
        self.assertIn("device='vulkan:0'", text)
        self.assertNotIn("cuda", text)


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


# Reference for one window of query rows only.  A key range wide enough that no
# two-product chain can hold its scores needs a reference that stays as narrow
# as the kernel: one float64 row per (query row, key) pair over the whole range
# would ask for more memory than the comparison is checking.
def _sdpa_window_reference(
    q, k, v, causal=False, scale=None, rows=(0, 8)):
    q = q.astype(np.float64)
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    hq = q.shape[1]
    hkv = k.shape[1]
    if hkv != hq:
        g = hq // hkv
        k = np.repeat(k, g, axis=1)
        v = np.repeat(v, g, axis=1)
    r0, r1 = rows
    s = np.einsum("bhqd,bhkd->bhqk", q[:, :, r0:r1], k)
    s = s * (1.0 / np.sqrt(q.shape[-1]) if scale is None else scale)
    if causal:
        keys = np.arange(k.shape[2])[None, None, None, :]
        queries = np.arange(r0, r1)[None, None, :, None]
        s = np.where(keys <= queries, s, -2.0 ** 31)
    s = s.astype(np.float32).astype(np.float64)
    row = np.where(
        np.isneginf(s.max(-1, keepdims=True)), 0.0, s.max(-1, keepdims=True))
    e = np.exp(s - row)
    denom = e.sum(-1, keepdims=True)
    p = np.divide(e, denom, out=np.zeros_like(e), where=(denom != 0))
    return np.einsum("bhqk,bhkd->bhqd", p, v)


def _sdpa_window_check(
    self,
    B,
    Hq,
    Hkv,
    Sq,
    Skv,
    D,
    rows,
    causal=False,
    gqa=False,
    *_,
):
    tensorplay.manual_seed(1234)
    q = tensorplay.randn((B, Hq, Sq, D), device="cpu")
    k = tensorplay.randn((B, Hkv, Skv, D), device="cpu")
    v = tensorplay.randn((B, Hkv, Skv, D), device="cpu")
    got = tensorplay.nn.functional.scaled_dot_product_attention(
        q.to("vulkan"),
        k.to("vulkan"),
        v.to("vulkan"),
        is_causal=causal,
        enable_gqa=gqa).cpu().numpy()
    want = _sdpa_window_reference(
        q.numpy(), k.numpy(), v.numpy(), causal=causal, rows=rows)
    np.testing.assert_allclose(
        got[:, :, rows[0]:rows[1]].astype(np.float64),
        want, rtol=1.0e-3, atol=4.0e-3)


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

    def test_sequences_past_the_image_axis_limit(self):
        # A sequence longer than the device's 3-D image axis leaves the score
        # matrix with no texture at all, so attention composes over linear
        # storage: the two products are the batched kernels and the row
        # reduction walks a score matrix in memory.
        LONG = 20000
        _sdpa_check(self, B=1, Hq=2, Hkv=2, Sq=4, Skv=LONG, D=8)
        _sdpa_check(self, B=1, Hq=2, Hkv=2, Sq=4, Skv=LONG, D=8, causal=True)
        _sdpa_check(
            self, B=1, Hq=4, Hkv=2, Sq=4, Skv=LONG, D=8, gqa=True)
        _sdpa_check(
            self, B=1, Hq=2, Hkv=2, Sq=4, Skv=LONG, D=8, scale=0.125)
        _sdpa_check(
            self, B=1, Hq=2, Hkv=2, Sq=4, Skv=LONG, D=8,
            dtype=tensorplay.float16, causal=True)
        # The query range is the long axis instead: the same route, with the
        # score rows spread past the limit.
        _sdpa_check(self, B=1, Hq=1, Hkv=1, Sq=LONG, Skv=8, D=8, causal=True)

    def test_linear_storage_masks(self):
        LONG = 20000
        rng = np.random.default_rng(21)
        _sdpa_check(
            self, B=1, Hq=2, Hkv=2, Sq=4, Skv=LONG, D=8,
            mask=tensorplay.tensor(rng.random((4, LONG)) > 0.02))
        _sdpa_check(
            self, B=1, Hq=2, Hkv=2, Sq=4, Skv=LONG, D=8,
            mask=tensorplay.tensor(
                (rng.random((1, 2, 4, LONG)).astype(np.float32) * 0.5)))
        # Grouped heads share a mask through the merged head and row axes.
        _sdpa_check(
            self, B=1, Hq=4, Hkv=2, Sq=4, Skv=LONG, D=8, gqa=True,
            mask=tensorplay.tensor(rng.random((4, LONG)) > 0.02))
        # A rejected row and a surviving one share the same key range.
        dead = np.ones((4, LONG), dtype=bool)
        dead[0] = False
        _sdpa_check(
            self, B=1, Hq=1, Hkv=1, Sq=4, Skv=LONG, D=8,
            mask=tensorplay.tensor(dead))

    def test_long_prefill_single_walk(self):
        # Above four million score lanes the score matrix is never written
        # down: one workgroup owns eight query rows and rescales its running
        # maximum, exponential sum and value accumulator as it reads the keys.
        _sdpa_check(self, B=1, Hq=8, Hkv=8, Sq=1024, Skv=600, D=16)
        _sdpa_check(
            self, B=1, Hq=8, Hkv=8, Sq=1024, Skv=1024, D=32, causal=True)
        _sdpa_check(
            self, B=1, Hq=16, Hkv=4, Sq=512, Skv=600, D=8, causal=True,
            gqa=True)
        # A query range that is not a whole number of row tiles, an odd head
        # dimension, and a half payload the walk widens on the way in.
        _sdpa_check(self, B=1, Hq=16, Hkv=16, Sq=511, Skv=601, D=5)
        _sdpa_check(
            self, B=1, Hq=16, Hkv=16, Sq=512, Skv=600, D=8,
            dtype=tensorplay.float16, causal=True)

    def test_long_prefill_masks(self):
        Sq, Skv = 512, 601
        rng = np.random.default_rng(31)
        shared = rng.random((Sq, Skv)) > 0.3
        shared[0] = True  # keep the first row visible
        _sdpa_check(
            self, B=1, Hq=16, Hkv=16, Sq=Sq, Skv=Skv, D=8,
            mask=tensorplay.tensor(shared))
        per_head = rng.random((16, Sq, Skv)) > 0.35
        per_head[:, 0, :] = True
        _sdpa_check(
            self, B=1, Hq=16, Hkv=16, Sq=Sq, Skv=Skv, D=8,
            mask=tensorplay.tensor(per_head))
        full = rng.random((2, 16, Sq, Skv)) > 0.35
        full[:, :, 0, :] = True
        _sdpa_check(
            self, B=2, Hq=16, Hkv=16, Sq=Sq, Skv=Skv, D=8,
            mask=tensorplay.tensor(full))
        _sdpa_check(
            self, B=1, Hq=16, Hkv=4, Sq=Sq, Skv=Skv, D=8, gqa=True,
            mask=tensorplay.tensor(
                rng.random((Sq, Skv)).astype(np.float32) - 0.5))
        # A row the mask rejects entirely normalizes to zeros here too.
        killed = np.ones((Sq, Skv), dtype=bool)
        killed[7] = False
        killed[300] = False
        _sdpa_check(
            self, B=1, Hq=16, Hkv=16, Sq=Sq, Skv=Skv, D=8,
            mask=tensorplay.tensor(killed))

    def test_wide_window_without_a_score_matrix(self):
        # The tail rows carry the widest softmax window a causal prefill has.
        _sdpa_window_check(
            self, B=1, Hq=8, Hkv=8, Sq=2048, Skv=2048, D=64,
            rows=(2040, 2048), causal=True)
        # Thirty-two heads over an eight thousand key range is 2^31 score
        # lanes: an element offset in a linear payload cannot even number them,
        # yet the walk needs only the operands and their running sums.
        _sdpa_window_check(
            self, B=1, Hq=32, Hkv=32, Sq=8192, Skv=8192, D=8,
            rows=(0, 8), causal=True)
        _sdpa_window_check(
            self, B=1, Hq=32, Hkv=1, Sq=8192, Skv=8192, D=8,
            rows=(4096, 4100), causal=True, gqa=True)


def _sdpa_backward_reference(
    q, k, v, go, mask=None, causal=False, scale=None, group=1):
    """The gradient of the attention that `_sdpa_reference` spells out.

    The scores are taken again, so a fully masked query row is a row of zeros;
    the probabilities then split the incoming gradient over the keys a row read.
    A group above one hands each key and value to the query heads it serves, so
    the gradients of those heads are added back onto the one key head they share.
    """
    q = q.astype(np.float64)
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    go = go.astype(np.float64)
    kv_heads = k.shape[1]
    k = np.repeat(k, group, axis=1)
    v = np.repeat(v, group, axis=1)
    s = 1.0 / np.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q * s) @ k.transpose(0, 1, 3, 2)
    if causal:
        keys = np.arange(k.shape[2])[None, None, None, :]
        queries = np.arange(q.shape[-2])[None, None, :, None]
        scores = np.where(keys <= queries, scores, -np.inf)
    if mask is not None:
        if mask.dtype == np.bool_:
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = scores + mask.astype(np.float64)
    row = np.where(
        np.isneginf(scores.max(-1, keepdims=True)), 0.0,
        scores.max(-1, keepdims=True))
    e = np.exp(scores - row)
    denom = e.sum(-1, keepdims=True)
    probs = np.divide(e, denom, out=np.zeros_like(e), where=(denom != 0))
    d_probs = go @ v.transpose(0, 1, 3, 2)
    d_value = probs.transpose(0, 1, 3, 2) @ go
    d_scores = probs * (
        d_probs - (d_probs * probs).sum(-1, keepdims=True))
    d_query = (d_scores @ k) * s
    d_key = d_scores.transpose(0, 1, 3, 2) @ (q * s)
    if group > 1:
        d_key = d_key.reshape(
            d_key.shape[0], kv_heads, group, d_key.shape[-2],
            d_key.shape[-1]).sum(2)
        d_value = d_value.reshape(
            d_value.shape[0], kv_heads, group, d_value.shape[-2],
            d_value.shape[-1]).sum(2)
    return d_query, d_key, d_value


def _sdpa_backward_check(
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
):
    """Every gradient of one attention call, against the same arithmetic.

    The draws come from one numpy stream, so the host reference and the device
    see the very same query, key, value and incoming gradient.
    """
    rng = np.random.default_rng(B * 977 + Hq * 43 + Sq * 7 + Skv + D)
    np_dtype = np.float16 if dtype == tensorplay.float16 else np.float32
    q_np = (rng.standard_normal((B, Hq, Sq, D)) * 1.5).astype(np_dtype)
    k_np = (rng.standard_normal((B, Hkv, Skv, D)) * 1.5).astype(np_dtype)
    v_np = (rng.standard_normal((B, Hkv, Skv, D)) * 1.5).astype(np_dtype)
    go_np = (rng.standard_normal((B, Hq, Sq, D)) * 1.5).astype(np_dtype)
    mask_np = None if mask is None else mask.numpy()
    group = Hq // Hkv if (gqa or Hkv == 1) else 1

    want = _sdpa_backward_reference(
        q_np, k_np, v_np, go_np,
        mask=mask_np, causal=causal, scale=scale, group=group)

    q = tensorplay.tensor(q_np, dtype=dtype).to("vulkan")
    k = tensorplay.tensor(k_np, dtype=dtype).to("vulkan")
    v = tensorplay.tensor(v_np, dtype=dtype).to("vulkan")
    go = tensorplay.tensor(go_np, dtype=dtype).to("vulkan")
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = tensorplay.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=(mask.to("vulkan") if mask is not None else None),
        is_causal=causal,
        scale=scale,
        enable_gqa=gqa)
    out.backward(go)
    got = [q.grad, k.grad, v.grad]
    atol = 2.0e-2 if dtype == tensorplay.float16 else 4.0e-3
    for name, g, w in zip(("grad_query", "grad_key", "grad_value"), got, want):
        self.assertIsNotNone(g, msg=name)
        np.testing.assert_allclose(
            g.cpu().numpy().astype(np.float64), w,
            rtol=1.0e-2, atol=atol, err_msg=name)


class TestVulkanSdpaBackward(unittest.TestCase):
    # The gradient is the arithmetic the attention was made of, recomputed and
    # differentiated: no fused backward exists on the device yet, so whatever a
    # call can express -- a causal mask, an explicit one, a scale, a context
    # longer than the query -- is differentiable for the same reason.

    def test_plain_self_attention(self):
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=8, Skv=8, D=16)
        _sdpa_backward_check(self, B=2, Hq=3, Hkv=3, Sq=5, Skv=5, D=8)

    def test_causal(self):
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=9, Skv=9, D=16,
                             causal=True)
        _sdpa_backward_check(self, B=2, Hq=2, Hkv=2, Sq=3, Skv=11, D=8,
                             causal=True)

    def test_a_decode_row_grad_its_whole_context(self):
        # One query row against a long context is the shape a generated token
        # arrives in, and the gradient has to reach every key it read.
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=1, Skv=64, D=32)
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=1, Skv=64, D=32,
                             causal=True)

    def test_bool_mask(self):
        Sq, Skv = 5, 9
        keep = np.tril(np.ones((Sq, Skv)), k=Skv - Sq).astype(bool)
        _sdpa_backward_check(
            self, B=2, Hq=3, Hkv=3, Sq=Sq, Skv=Skv, D=8,
            mask=tensorplay.tensor(keep))
        rng = np.random.default_rng(11)
        per_head = rng.random((2, 1, Sq, Skv)) > 0.3
        per_head[:, :, 0, :] = True
        _sdpa_backward_check(
            self, B=2, Hq=3, Hkv=3, Sq=Sq, Skv=Skv, D=8,
            mask=tensorplay.tensor(per_head))

    def test_a_fully_masked_row_grads_nothing(self):
        # A row that reads no key answers with zeros in the forward, so its
        # gradient has to be zeros too rather than a number over nothing.
        keep = np.ones((2, 1, 4, 6), dtype=bool)
        keep[:, :, 1, :] = False
        _sdpa_backward_check(
            self, B=2, Hq=2, Hkv=2, Sq=4, Skv=6, D=8,
            mask=tensorplay.tensor(keep))

    def test_float_mask(self):
        rng = np.random.default_rng(13)
        _sdpa_backward_check(
            self, B=2, Hq=2, Hkv=2, Sq=4, Skv=7, D=8,
            mask=tensorplay.tensor(
                (rng.standard_normal((2, 1, 4, 7)) * 2).astype(np.float32)))
        _sdpa_backward_check(
            self, B=1, Hq=3, Hkv=3, Sq=5, Skv=5, D=16,
            mask=tensorplay.tensor(
                (rng.standard_normal((5, 5)) * 2).astype(np.float32)))

    def test_a_given_scale(self):
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=8, Skv=8, D=32,
                             scale=0.125, causal=True)

    def test_half_inputs(self):
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=8, Skv=8, D=32,
                             dtype=tensorplay.float16)
        _sdpa_backward_check(self, B=2, Hq=2, Hkv=2, Sq=6, Skv=6, D=16,
                             dtype=tensorplay.float16, causal=True)

    def test_grouped_query_heads(self):
        # Every key and value is read by a whole block of query heads, so its
        # gradient is the sum of theirs -- the same mapping the forward takes.
        _sdpa_backward_check(self, B=1, Hq=8, Hkv=2, Sq=6, Skv=6, D=16,
                             gqa=True)
        _sdpa_backward_check(self, B=2, Hq=8, Hkv=4, Sq=7, Skv=7, D=32,
                             gqa=True, causal=True)
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=4, Sq=5, Skv=5, D=8,
                             gqa=True)

    def test_a_grouped_context_longer_than_the_queries(self):
        Sq, Skv = 5, 9
        rng = np.random.default_rng(23)
        _sdpa_backward_check(
            self, B=1, Hq=6, Hkv=3, Sq=Sq, Skv=Skv, D=8, gqa=True,
            mask=tensorplay.tensor(rng.random((Sq, Skv)) > 0.3))
        _sdpa_backward_check(
            self, B=1, Hq=4, Hkv=2, Sq=4, Skv=4, D=8, gqa=True, scale=0.25)
        _sdpa_backward_check(
            self, B=1, Hq=8, Hkv=2, Sq=6, Skv=6, D=32,
            dtype=tensorplay.float16, gqa=True)

    def test_one_key_head_serving_every_query_head(self):
        # A single key and value broadcasts across the query heads whether or
        # not the flag was raised, and its gradient gathers all of them back.
        _sdpa_backward_check(self, B=1, Hq=4, Hkv=1, Sq=5, Skv=5, D=8)
        _sdpa_backward_check(self, B=2, Hq=8, Hkv=1, Sq=4, Skv=11, D=16,
                             causal=True, gqa=True)

    def test_a_decode_row_grads_a_grouped_context(self):
        # The shape a generated token arrives in, read through the group: the
        # causal alignment keeps the newest key the only one the row can see.
        _sdpa_backward_check(self, B=1, Hq=8, Hkv=2, Sq=1, Skv=32, D=32,
                             causal=True, gqa=True)

    def test_a_request_the_device_cannot_answer_refuses(self):
        import tensorplay.nn.functional as Fn
        rng = np.random.default_rng(17)
        q = tensorplay.tensor(
            (rng.standard_normal((1, 4, 4, 8))).astype(np.float32),
            ).to("vulkan")
        k = tensorplay.tensor(
            (rng.standard_normal((1, 2, 4, 8))).astype(np.float32),
            ).to("vulkan")
        v = tensorplay.tensor(
            (rng.standard_normal((1, 2, 4, 8))).astype(np.float32),
            ).to("vulkan")
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        # A count that does not divide leaves no whole group to broadcast to.
        qn = tensorplay.tensor(
            (rng.standard_normal((1, 6, 4, 8))).astype(np.float32),
            ).to("vulkan")
        kn = tensorplay.tensor(
            (rng.standard_normal((1, 4, 4, 8))).astype(np.float32),
            ).to("vulkan")
        vn = tensorplay.tensor(
            (rng.standard_normal((1, 4, 4, 8))).astype(np.float32),
            ).to("vulkan")
        qn.requires_grad_(True)
        kn.requires_grad_(True)
        vn.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "multiple of"):
            out = Fn.scaled_dot_product_attention(
                qn, kn, vn, enable_gqa=True)
            out.backward(tensorplay.ones_like(out))
        # Above one key head the flag is the only route that groups.
        with self.assertRaisesRegex(RuntimeError, "broadcastable through"):
            Fn.scaled_dot_product_attention(q, k, v)
        # A proportion to drop needs the mask the drop was decided by.
        qs = q.detach().requires_grad_(True)
        ks = k.detach().requires_grad_(True)
        vs = v.detach().requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "does not support dropout"):
            Fn.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.2)

    def test_the_gradient_of_a_long_context_reaches_every_head(self):
        # A context wide enough that its scores have no image form is the
        # linear route, where the walk over the keys is serial per row tile.
        _sdpa_backward_check(self, B=1, Hq=2, Hkv=2, Sq=4, Skv=4096, D=8)


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

    def test_a_short_cache_scatters_the_rows_its_shape_offers(self):
        # A cache of two or three axes has the same texel geometry as the four-axis
        # one: the axis above the last is the texture's height, so its rows are
        # lines of whole texels.  Appending a decode step into a [slots, features]
        # page is the shape a runtime actually holds.
        tensorplay.manual_seed(4242)
        cases = (
            ((9, 64), 0, [0, 8, 3], tensorplay.float32),
            ((7, 24), 0, [6, 0, 1], tensorplay.float16),
            ((6, 7), 0, [5, 1, 2], tensorplay.float32),
            ((2, 5, 8), 1, [4, 0, 2], tensorplay.float16),
            ((4, 6, 5), 1, [0, 5, 2], tensorplay.float32),
            ((1, 1, 4, 8), 2, [0, 3], tensorplay.float32),
        )
        for shape, axis, idx, dtype in cases:
            buf = tensorplay.randn(shape, device="cpu").to(dtype).to("vulkan")
            before = buf.cpu().numpy()
            rows = list(shape)
            rows[axis] = len(idx)
            src = tensorplay.randn(rows, device="cpu").to(dtype).to("vulkan")
            moved = src.cpu().numpy().astype(before.dtype)
            want = before.copy()
            picked = [slice(None)] * len(shape)
            picked[axis] = idx
            write = [slice(None)] * len(shape)
            write[axis] = range(len(idx))
            want[tuple(picked)] = moved[tuple(write)]
            np.testing.assert_array_equal(
                tensorplay.index_copy_(
                    buf, axis, tensorplay.tensor(
                        idx, dtype=tensorplay.int64, device="vulkan"),
                    src).cpu().numpy(),
                want,
                err_msg=f"rows of a {shape} cache along axis {axis}")
            # The functional form writes the same rows into the payload the
            # in-place call left behind, so it answers the same table again and
            # changes nothing the scatter did not name.
            np.testing.assert_array_equal(
                tensorplay.index_copy(
                    buf, axis,
                    tensorplay.tensor(
                        idx, dtype=tensorplay.int64, device="vulkan"),
                    src).cpu().numpy(),
                want,
                err_msg=f"out-of-place rows of a {shape} cache")

    def test_a_cache_without_an_image_form_scatters_any_axis(self):
        # Past the image axis limit a payload is plain row-major memory, so the
        # row an invocation moves is a run of consecutive elements and any axis can
        # name it - including the innermost one, whose rows an image packs into the
        # lanes of a texel and cannot scatter whole.
        tensorplay.manual_seed(99)
        cases = (
            ((20000,), 0, [0, 19999, 7]),
            ((40000, 8), 0, [0, 39999, 7]),
            ((40000, 8), 1, [0, 2, 7]),
            ((6, 40000, 4), 1, [0, 39999, 7, 100]),
            ((6, 40000, 4), 0, [5, 0, 1]),
        )
        for shape, axis, idx in cases:
            buf = tensorplay.randn(shape, device="cpu").to("vulkan")
            before = buf.cpu().numpy()
            rows = list(shape)
            rows[axis] = len(idx)
            src = tensorplay.randn(rows, device="cpu").to("vulkan")
            moved = src.cpu().numpy()
            want = before.copy()
            picked = [slice(None)] * len(shape)
            picked[axis] = idx
            write = [slice(None)] * len(shape)
            write[axis] = range(len(idx))
            want[tuple(picked)] = moved[tuple(write)]
            np.testing.assert_array_equal(
                tensorplay.index_copy_(
                    buf, axis,
                    tensorplay.tensor(
                        idx, dtype=tensorplay.int64, device="vulkan"),
                    src).cpu().numpy(),
                want,
                err_msg=f"a {shape} linear cache along axis {axis}")
    def test_an_image_row_that_straddles_its_packed_axis_stays_refused(self):
        # A texel holds four lanes of the axis a payload packs, so writing one of
        # its rows means rewriting lanes the scatter did not name.  Neither the
        # innermost axis nor a rank-1 payload's only axis is the texture's height.
        buf = tensorplay.randn((9, 64), device="cpu").to("vulkan")
        src = tensorplay.randn((9, 2), device="cpu").to("vulkan")
        codes = tensorplay.tensor([0, 2], dtype=tensorplay.int64,
                                  device="vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "an image payload scatters the axis above"):
            tensorplay.index_copy_(buf, 1, codes, src)
        flat = tensorplay.randn((12,), device="cpu").to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "an image payload scatters the axis above"):
            tensorplay.index_copy_(
                flat, 0,
                tensorplay.tensor([1, 5], dtype=tensorplay.int64,
                                  device="vulkan"),
                tensorplay.randn((2,), device="cpu").to("vulkan"))

    def test_index_copy_writes_nothing_for_a_row_it_cannot_name(self):
        # Nothing asserts on the device, and folding a list onto the host to
        # check its range would cost a queue sync per call.  Both scatter
        # shaders settle it themselves: a code naming no row of the destination
        # writes nothing, the way a lookup returns zeros for such a code.  The
        # second case pushes the destination past the image axis limit, so the
        # row it skips is addressed in linear storage instead of a texel.
        tensorplay.manual_seed(1234)
        for S in (8, 20000):
            buf = tensorplay.randn((1, 2, S, 4), device="cpu").to("vulkan")
            snapshot = buf.cpu().numpy()
            src = tensorplay.randn((1, 2, 3, 4), device="cpu").to("vulkan")
            want = snapshot.copy()
            want[:, :, 3, :] = src.cpu().numpy()[:, :, 0, :]
            codes = [3, S, -1]
            for on_device in (False, True):
                buf.copy_(tensorplay.from_numpy(snapshot))
                positions = tensorplay.tensor(codes, dtype=tensorplay.int64)
                if on_device:
                    positions = positions.to("vulkan")
                tensorplay.index_copy_(buf, 2, positions, src)
                np.testing.assert_array_equal(
                    buf.cpu().numpy(), want,
                    err_msg=f"an unaddressable row wrote anyway at S={S}, "
                    f"index on device={on_device}")
            # The destination keeps every row the scatter did not name.
            np.testing.assert_array_equal(buf.cpu().numpy()[:, :, 4:, :],
                                          snapshot[:, :, 4:, :])

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
        # 66558 and 66559 need a second look: their word counts (16640) are an
        # exact multiple of the workgroup size, so the launch carries no padded
        # invocation that could pick up a leftover lane.
        for width in (self.WIDTH - 3, self.WIDTH - 1, self.WIDTH + 1,
                      self.WIDTH + 6, 66558, 66559):
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

    def test_index_payload_over_linear_storage(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A gather over a table too tall for a texture reads its index list
        # from a linear payload, where one element is one int in row order
        # rather than one lane of a texel.  The lengths below also leave a
        # partial last word, so the leftover lanes must be reached too.
        rng = np.random.default_rng(5)
        table = rng.standard_normal(70000, dtype=np.float32)
        for count in (66558, 66559, 66560):
            sel = rng.integers(0, table.size, count).astype(np.int32)
            out = F.index_select(
                tp.from_numpy(table.copy()).to("vulkan"), 0,
                tp.from_numpy(sel.copy()).to("vulkan")).cpu().numpy()
            np.testing.assert_array_equal(
                out, table[sel], err_msg=f"index_select at {count}")
            id_sel = np.arange(count, dtype=np.int32)
            out = F.index_select(
                tp.from_numpy(table.copy()).to("vulkan"), 0,
                tp.from_numpy(id_sel.copy()).to("vulkan")).cpu().numpy()
            np.testing.assert_array_equal(
                out, table[:count], err_msg=f"identity select at {count}")

    def test_gather_over_texture_table_with_long_index_list(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # An index list carries one index per texel, so a list longer than the
        # device's image-axis limit has no texture form even when the table it
        # selects rows from does.  The gather then copies the texture into
        # dense row-major storage and runs the linear shader, rather than
        # refusing the pair of shapes.  The lengths below straddle that limit
        # so both routes are checked against one table.
        rng = np.random.default_rng(23)
        table = rng.standard_normal((1000, 8), dtype=np.float32)
        vk_table = tp.from_numpy(table.copy()).to("vulkan")
        for count in (512, 16383, 16384, 16385, 20000):
            sel = rng.integers(0, table.shape[0], count).astype(np.int32)
            vk_sel = tp.from_numpy(sel.copy()).to("vulkan")
            np.testing.assert_array_equal(
                F.index_select(vk_table, 0, vk_sel).cpu().numpy(),
                table[sel], err_msg=f"index_select at {count} indices")
            np.testing.assert_array_equal(
                F.embedding(vk_table, vk_sel).cpu().numpy(),
                table[sel], err_msg=f"embedding at {count} indices")

    def test_gather_writes_every_slot_of_a_wide_row(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A gathered row spans more than one RGBA texel once the table is
        # wider than four columns, so the texture route must be launched over
        # (slot inside the row, index) instead of over the flat element count
        # of the output it allocates -- otherwise only the first lookup is
        # written and the rest of the output stays zero.  The widths below
        # include one that is not a multiple of four.
        rng = np.random.default_rng(29)
        for rows, width in ((1000, 8), (17, 5), (64, 64)):
            table = rng.standard_normal((rows, width), dtype=np.float32)
            sel = rng.integers(0, rows, 40).astype(np.int32)
            vk_table = tp.from_numpy(table.copy()).to("vulkan")
            vk_sel = tp.from_numpy(sel.copy()).to("vulkan")
            np.testing.assert_array_equal(
                F.index_select(vk_table, 0, vk_sel).cpu().numpy(),
                table[sel], err_msg=f"index_select over {rows}x{width}")
            np.testing.assert_array_equal(
                F.embedding(vk_table, vk_sel).cpu().numpy(),
                table[sel], err_msg=f"embedding over {rows}x{width}")

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

    def test_half_precision_layers_down_a_streamed_embedding(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A table taller than the image axis limit has no texel form, so the rows
        # a lookup reads out of it -- and every half product that a linear weight
        # writes into linear storage -- used to reach the next layer as a payload
        # its texture-only kernel cannot read.  Both producers now hand back an
        # image as soon as the shape still has one, a half payload crossing that
        # boundary two lanes to a word.  The last layer takes a weight small
        # enough to keep its texels, so it only runs if the repack happened.
        rng = np.random.default_rng(83)
        rows, features, narrow, tail = 20000, 64, 33, 16

        def half(scale, *shape):
            return (rng.standard_normal(shape) * scale).astype(np.float16)

        table_cpu = half(0.4, rows, features)
        w1_cpu = half(0.05, rows, features)
        w2_cpu = half(0.05, narrow, rows)
        w3_cpu = half(0.05, tail, narrow)
        ids_cpu = np.array([5, 19999, 0, 7777], dtype=np.int64)

        table = tensorplay.from_numpy(table_cpu.copy()).to("vulkan")
        ids = tensorplay.from_numpy(ids_cpu.copy()).to("vulkan")
        w1 = tensorplay.from_numpy(w1_cpu.copy()).to("vulkan")
        w2 = tensorplay.from_numpy(w2_cpu.copy()).to("vulkan")
        w3 = tensorplay.from_numpy(w3_cpu.copy()).to("vulkan")

        x = tp.embedding(table, ids)
        self.assertEqual(x.dtype, tp.float16)
        y = F.linear(x, w1)
        z = F.linear(y, w2)
        out = F.linear(z, w3)

        expect = (
            (
                table_cpu[ids_cpu].astype(np.float32)
                @ w1_cpu.T.astype(np.float32)
            )
            @ w2_cpu.T.astype(np.float32)
        ) @ w3_cpu.T.astype(np.float32)
        self.assertEqual(tuple(out.shape), (4, tail))
        np.testing.assert_allclose(
            out.cpu().numpy(), expect, rtol=2e-2, atol=2e-3)

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

    def test_half_precision_reductions_over_linear_payload(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A reduced half result stores two rows in one 32-bit word, so the row
        # grid becomes a word grid: the invocation that owns a word writes both
        # lanes at once and no neighbour can read-modify-write it.  An odd row
        # count puts the tail alone in its word, and an odd reduction length
        # leaves a partial last word in the input, so both are exercised here.
        # Every accumulator is held in single precision and only the store
        # narrows, which is what keeps a long-axis variance finite.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        def upload(a):
            return tensorplay.from_numpy(np.asarray(a, dtype=np.float16)
                                         ).to("vulkan")

        def softmax_ref(a, axis):
            m = a.max(axis=axis, keepdims=True)
            e = np.exp(a - m)
            return e / e.sum(axis=axis, keepdims=True)

        def log_softmax_ref(a, axis):
            m = a.max(axis=axis, keepdims=True)
            return a - m - np.log(np.exp(a - m).sum(axis=axis, keepdims=True))

        tensorplay.manual_seed(97)
        flat_cpu = narrowed((tp.randn((self.WIDTH,), device="cpu")
                             * 3.0).numpy())
        flat = upload(flat_cpu)
        np.testing.assert_array_equal(
            F.argmax(flat).cpu().numpy(), np.argmax(flat_cpu),
            err_msg="argmax over an odd-length half payload")
        np.testing.assert_array_equal(
            F.argmin(flat).cpu().numpy(), np.argmin(flat_cpu),
            err_msg="argmin over an odd-length half payload")
        np.testing.assert_allclose(
            flat.max().cpu().numpy().astype(np.float32), flat_cpu.max(),
            atol=1e-3, err_msg="maximum over an odd-length half payload")
        np.testing.assert_allclose(
            flat.mean().cpu().numpy().astype(np.float32), flat_cpu.mean(),
            rtol=1e-3, atol=1e-3, err_msg="mean over a half payload")
        np.testing.assert_allclose(
            F.softmax(flat, 0).cpu().numpy().astype(np.float32),
            narrowed(softmax_ref(flat_cpu, 0)),
            rtol=2e-3, atol=2e-3, err_msg="softmax over a half payload")
        np.testing.assert_allclose(
            F.log_softmax(flat, 0).cpu().numpy().astype(np.float32),
            narrowed(log_softmax_ref(flat_cpu, 0)),
            rtol=2e-3, atol=2e-3, err_msg="log_softmax over a half payload")

        # An axis long enough that its raw sums leave the finite range of the
        # format: the walk must keep the moments wider than the payload.
        rows_cpu = narrowed((tp.randn((3, 80000), device="cpu")
                             * 2.0).numpy())
        rows = upload(rows_cpu)
        np.testing.assert_array_equal(
            F.argmax(rows, dim=1).cpu().numpy(), rows_cpu.argmax(axis=1),
            err_msg="row argmax over a half linear payload")
        np.testing.assert_array_equal(
            F.argmin(rows, dim=0).cpu().numpy(), rows_cpu.argmin(axis=0),
            err_msg="column argmin over an odd half output row count")
        self.assertEqual(F.argmax(rows, dim=1).dtype, tensorplay.int32,
                         "half argmax changed its index dtype")
        np.testing.assert_allclose(
            rows.sum(dim=[1]).cpu().numpy().astype(np.float32),
            narrowed(rows_cpu.sum(axis=1)),
            rtol=1e-3, atol=1e-2, err_msg="long row sums over a half payload")
        np.testing.assert_allclose(
            F.var(rows, dim=[1]).cpu().numpy().astype(np.float32),
            narrowed(rows_cpu.var(axis=1, ddof=1)),
            rtol=2e-3, atol=2e-3,
            err_msg="variance over a long half axis")
        np.testing.assert_allclose(
            F.std(rows, dim=[1]).cpu().numpy().astype(np.float32),
            narrowed(rows_cpu.std(axis=1, ddof=1)),
            rtol=5e-3, atol=5e-3, err_msg="std over a long half axis")
        np.testing.assert_allclose(
            F.var(rows, dim=[0, 1]).cpu().numpy().astype(np.float32),
            narrowed(rows_cpu.var()), rtol=5e-3, atol=5e-3,
            err_msg="variance over several half axes")
        self.assertEqual(F.var(rows, dim=[1]).dtype, tensorplay.float16,
                         "half variance widened to single precision")
        self.assertEqual(F.softmax(rows, 1).dtype, tensorplay.float16,
                         "half softmax widened to single precision")
        got_max = F.max(rows, dim=1)
        np.testing.assert_allclose(
            got_max[0].cpu().numpy().astype(np.float32),
            narrowed(rows_cpu.max(axis=1)), atol=1e-3,
            err_msg="row maximum values over a half linear payload")
        np.testing.assert_array_equal(
            got_max[1].cpu().numpy(), rows_cpu.argmax(axis=1),
            err_msg="row maximum indices over a half linear payload")

        # Walking along the strided axis keeps the same word rule for the
        # output, whose length here is odd.
        np.testing.assert_allclose(
            rows.amax(dim=[0]).cpu().numpy().astype(np.float32),
            narrowed(rows_cpu.max(axis=0)), atol=1e-3,
            err_msg="column amax over a half linear payload")
        np.testing.assert_allclose(
            F.log_softmax(rows, 0).cpu().numpy().astype(np.float32),
            narrowed(log_softmax_ref(rows_cpu, 0)),
            rtol=2e-3, atol=2e-3,
            err_msg="log_softmax along the strided half axis")

        # A short payload stays in a texture, where the reductions reach the
        # half twins of the image kernels instead.
        small_cpu = narrowed((tp.randn((640, 128), device="cpu")
                              * 1.5).numpy())
        small = upload(small_cpu)
        np.testing.assert_array_equal(
            F.argmax(small, dim=1).cpu().numpy(), small_cpu.argmax(axis=1),
            err_msg="argmax over a half texture payload")
        np.testing.assert_allclose(
            F.var(small, dim=[1]).cpu().numpy().astype(np.float32),
            narrowed(small_cpu.var(axis=1, ddof=1)),
            rtol=2e-3, atol=2e-3, err_msg="variance over a half texture axis")
        np.testing.assert_allclose(
            F.softmax(small, 1).cpu().numpy().astype(np.float32),
            narrowed(softmax_ref(small_cpu, 1)),
            rtol=2e-3, atol=2e-3, err_msg="softmax over a half texture axis")

        # The single-precision kernels must be untouched on the same shapes.
        flat32 = tensorplay.from_numpy(
            (tp.randn((self.WIDTH,), device="cpu") * 3.0).numpy()
        ).to("vulkan")
        np.testing.assert_array_equal(
            F.argmax(flat32).cpu().numpy(),
            flat32.cpu().numpy().argmax(),
            err_msg="argmax over a single-precision linear payload")
        rows32 = tensorplay.from_numpy(
            (tp.randn((3, 80000), device="cpu") * 2.0).numpy()
        ).to("vulkan")
        np.testing.assert_allclose(
            F.var(rows32, dim=[1]).cpu().numpy(),
            rows32.cpu().numpy().var(axis=1, ddof=1),
            rtol=1e-4, atol=1e-4,
            err_msg="variance over a single-precision long axis")

    def test_half_precision_decode_product_over_linear_weight(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A weight row past the image axis limit reaches the decode product as
        # a linear payload of half lanes, so the split pass unpacks pairs and
        # keeps its accumulators in single precision.  When the result is too
        # wide for an image, the fold pass owns one word and writes both of the
        # elements it carries, which is what lets rows start at an odd offset
        # without disturbing a neighbour's lane.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        def upload(a):
            return tensorplay.from_numpy(np.asarray(a, dtype=np.float16)
                                         ).to("vulkan")

        tensorplay.manual_seed(113)
        for M, N, K in ((1, 8, 20000), (4, 130, 20000), (3, 7, 20010),
                        (1, 64, 20003), (1, 4096, 20000)):
            x_cpu = narrowed((tp.randn((M, K), device="cpu") * 0.3).numpy())
            w_cpu = narrowed((tp.randn((N, K), device="cpu") * 0.3).numpy())
            b_cpu = narrowed((tp.randn((N,), device="cpu") * 0.2).numpy())
            want = narrowed(x_cpu @ w_cpu.T)
            x, w = upload(x_cpu), upload(w_cpu)
            got = F.linear(x, w)
            self.assertEqual(got.dtype, tensorplay.float16,
                             f"linear widened a half product at {M}x{N}x{K}")
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32), want,
                rtol=2e-3, atol=2e-3,
                err_msg=f"linear over a half linear weight at {M}x{N}x{K}")
            np.testing.assert_allclose(
                F.linear(x, w, upload(b_cpu)).cpu().numpy().astype(
                    np.float32),
                narrowed(want + b_cpu), rtol=2e-3, atol=2e-3,
                err_msg=f"addend folded into a half product at {M}x{N}x{K}")

        # Results this wide have no image form, so the fold writes words.
        for M, N, K in ((1, 20003, 64), (4, 20001, 128), (2, 32000, 64)):
            x_cpu = narrowed((tp.randn((M, K), device="cpu") * 0.3).numpy())
            w_cpu = narrowed((tp.randn((N, K), device="cpu") * 0.3).numpy())
            b_cpu = narrowed((tp.randn((N,), device="cpu") * 0.2).numpy())
            got = F.linear(upload(x_cpu), upload(w_cpu), upload(b_cpu))
            self.assertTrue(got.device.is_vulkan())
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32),
                narrowed(narrowed(x_cpu @ w_cpu.T) + b_cpu),
                rtol=2e-3, atol=2e-3,
                err_msg=f"half result in linear storage at {M}x{N}x{K}")

        # A reduction length that leaves a ragged last word, and an activation
        # long enough to stream from a payload of its own.
        for K in (20001, 20005):
            x_cpu = narrowed((tp.randn((2, K), device="cpu") * 0.3).numpy())
            w_cpu = narrowed((tp.randn((17, K), device="cpu") * 0.3).numpy())
            np.testing.assert_allclose(
                F.linear(upload(x_cpu), upload(w_cpu)).cpu().numpy().astype(
                    np.float32),
                narrowed(x_cpu @ w_cpu.T), rtol=2e-3, atol=2e-3,
                err_msg=f"half product over a ragged reduction of {K}")

        # Weights that keep an image form are restaged by the packing kernels,
        # which now pick the format of the payload they are handed.
        for M, N, K in ((1, 256, 64), (4, 128, 96)):
            x_cpu = narrowed((tp.randn((M, K), device="cpu") * 0.4).numpy())
            w_cpu = narrowed((tp.randn((N, K), device="cpu") * 0.4).numpy())
            np.testing.assert_allclose(
                F.linear(upload(x_cpu), upload(w_cpu)).cpu().numpy().astype(
                    np.float32),
                narrowed(x_cpu @ w_cpu.T), rtol=2e-3, atol=2e-3,
                err_msg=f"half product over a packed weight at {M}x{N}x{K}")

        batch1 = narrowed(
            (tp.randn((3, 32, 24), device="cpu") * 0.4).numpy())
        batch2 = narrowed(
            (tp.randn((3, 24, 16), device="cpu") * 0.4).numpy())
        np.testing.assert_allclose(
            F.bmm(upload(batch1), upload(batch2)).cpu().numpy().astype(
                np.float32),
            narrowed(np.stack([a @ c for a, c in zip(batch1, batch2)])),
            rtol=2e-3, atol=2e-3, err_msg="half bmm over packed planes")

        with self.assertRaisesRegex(RuntimeError, "one dtype"):
            F.linear(
                upload(narrowed(
                    (tp.randn((1, 64), device="cpu") * 0.4).numpy())),
                tensorplay.from_numpy(
                    (tp.randn((32, 64), device="cpu") * 0.4).numpy()
                ).to("vulkan"))

        # Single precision keeps its own builds on the same shapes.
        x32 = (tp.randn((1, 20000), device="cpu") * 0.3).numpy()
        w32 = (tp.randn((8, 20000), device="cpu") * 0.3).numpy()
        np.testing.assert_allclose(
            F.linear(tensorplay.from_numpy(x32).to("vulkan"),
                     tensorplay.from_numpy(w32).to("vulkan")).cpu().numpy(),
            x32 @ w32.T, rtol=1e-4, atol=1e-4,
            err_msg="single-precision product over a linear weight")

    def test_half_precision_tiled_product_over_linear_storage(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # Past a handful of rows the same weight words get re-read once per row,
        # so the staged tiled product takes over: unpacking happens while the
        # lanes go through shared memory, which holds floats, and the 32x32
        # accumulation never narrows between terms.  A result too wide for an
        # image stores word pairs, and an odd width would make two threads
        # claim one word, so it is refused instead of raced.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        def upload(a):
            return tensorplay.from_numpy(np.asarray(a, dtype=np.float16)
                                         ).to("vulkan")

        tensorplay.manual_seed(131)
        # An activation of its own in linear storage, image result.
        for M, N, K in ((8, 64, 20000), (5, 33, 20005), (40, 7, 20000),
                        (32, 128, 20000)):
            x_cpu = narrowed((tp.randn((M, K), device="cpu") * 0.2).numpy())
            w_cpu = narrowed((tp.randn((N, K), device="cpu") * 0.2).numpy())
            got = F.linear(upload(x_cpu), upload(w_cpu))
            self.assertEqual(got.dtype, tensorplay.float16,
                             f"tiled product widened a half payload at "
                             f"{M}x{N}x{K}")
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32),
                narrowed(x_cpu @ w_cpu.T), rtol=3e-3, atol=3e-3,
                err_msg=f"tiled product over a half linear activation at "
                        f"{M}x{N}x{K}")

        # A result in linear storage, with and without an addend of the same
        # payload form.
        for M, N, K in ((8, 20000, 64), (33, 20000, 64), (5, 20004, 96)):
            x_cpu = narrowed((tp.randn((M, K), device="cpu") * 0.3).numpy())
            w_cpu = narrowed((tp.randn((N, K), device="cpu") * 0.3).numpy())
            b_cpu = narrowed((tp.randn((N,), device="cpu") * 0.2).numpy())
            want = narrowed(x_cpu @ w_cpu.T)
            np.testing.assert_allclose(
                F.linear(upload(x_cpu), upload(w_cpu)).cpu().numpy().astype(
                    np.float32),
                want, rtol=3e-3, atol=3e-3,
                err_msg=f"half result in linear storage at {M}x{N}x{K}")
            np.testing.assert_allclose(
                F.linear(upload(x_cpu), upload(w_cpu),
                         upload(b_cpu)).cpu().numpy().astype(np.float32),
                narrowed(want + b_cpu), rtol=3e-3, atol=3e-3,
                err_msg=f"half linear result with a half addend at "
                        f"{M}x{N}x{K}")

        # A (K x N) weight walked along its column step instead of its row.
        a_cpu = narrowed((tp.randn((9, 20000), device="cpu") * 0.2).numpy())
        b_cpu = narrowed((tp.randn((20000, 48), device="cpu") * 0.2).numpy())
        np.testing.assert_allclose(
            (upload(a_cpu) @ upload(b_cpu)).cpu().numpy().astype(np.float32),
            narrowed(a_cpu @ b_cpu), rtol=3e-3, atol=3e-3,
            err_msg="tiled half product over a reduction-major weight")

        odd = tensorplay.from_numpy(narrowed(
            (tp.randn((8, 64), device="cpu") * 0.3).numpy()).astype(
                np.float16)).to("vulkan")
        wide = tensorplay.from_numpy(narrowed(
            (tp.randn((20001, 64), device="cpu") * 0.3).numpy()).astype(
                np.float16)).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "even result width"):
            F.linear(odd, wide)

    def test_bmm_over_linear_storage(self):
        import tensorplay as tp
        import tensorplay.functional as F
        # A batched product with an axis past the image limit has no texture
        # for its result, and one of its operands may have lost its texture
        # too.  One dispatch still covers every matrix: the batch rides the
        # workgroup z axis and reaches the next matrix through a stride.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        def softmax_rows(a):
            shifted = a - a.max(axis=-1, keepdims=True)
            e = np.exp(shifted)
            return e / e.sum(axis=-1, keepdims=True)

        def upload(a, dtype):
            return tensorplay.from_numpy(np.asarray(a, dtype=dtype)
                                         ).to("vulkan")

        tensorplay.manual_seed(151)
        # (batch, rows past the limit, k) x (batch, k, cols): only the left
        # operand needs a linear payload, and (batch, rows, cols past the
        # limit) x (batch, k, cols): only the right one.
        cases = (
            ((4, 20000, 16), (4, 16, 32), "batch rows past the limit"),
            ((3, 32, 20000), (3, 20000, 8), "reduction past the limit"),
            ((2, 48, 20005), (2, 20005, 12), "ragged reduction"),
        )
        for lhs_shape, rhs_shape, tag in cases:
            # Half-representable values in single precision keep the reference
            # product exact for both payload formats.
            a_cpu = narrowed((tp.randn(lhs_shape, device="cpu") * 0.2).numpy())
            b_cpu = narrowed((tp.randn(rhs_shape, device="cpu") * 0.2).numpy())
            want = np.matmul(a_cpu, b_cpu)
            for dtype, tol in ((np.float32, 2e-3), (np.float16, 6e-3)):
                got = F.bmm(upload(a_cpu, dtype), upload(b_cpu, dtype))
                self.assertEqual(
                    got.shape,
                    (lhs_shape[0], lhs_shape[1], rhs_shape[2]),
                    f"{tag} in {np.dtype(dtype).name}")
                np.testing.assert_allclose(
                    got.cpu().numpy().astype(np.float32), want,
                    rtol=tol, atol=tol,
                    err_msg=f"batched product with {tag} in "
                            f"{np.dtype(dtype).name}")

        # A result wide enough for no image at all still writes every column.
        a_cpu = narrowed((tp.randn((2, 64, 8), device="cpu") * 0.2).numpy())
        b_cpu = narrowed((tp.randn((2, 8, 20000), device="cpu") * 0.2).numpy())
        got = F.bmm(upload(a_cpu, np.float16), upload(b_cpu, np.float16))
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float32),
            narrowed(np.matmul(a_cpu, b_cpu)), rtol=6e-3, atol=6e-3,
            err_msg="batched product with batch columns past the limit")

        # Downstream ops read the same linear payload the product wrote, so a
        # long-context attention chain continues past the image limit.
        follow = F.softmax(got, -1)
        np.testing.assert_allclose(
            follow.cpu().numpy().astype(np.float32),
            softmax_rows(np.matmul(a_cpu, b_cpu)),
            rtol=6e-3, atol=6e-3,
            err_msg="softmax over a batched product in linear storage")

        # A half result packs two lanes per word, so two matrices may not
        # claim one word: an odd result width is refused rather than raced.
        odd_lhs = upload(narrowed(
            (tp.randn((3, 20000, 8), device="cpu") * 0.2).numpy()), np.float16)
        odd_rhs = upload(narrowed(
            (tp.randn((3, 8, 1001), device="cpu") * 0.2).numpy()), np.float16)
        with self.assertRaisesRegex(RuntimeError, "even result width"):
            F.bmm(odd_lhs, odd_rhs)

    def test_matmul_batch_shape_fold(self):
        import tensorplay as tp
        # The leading axes of both operands broadcast to one batch shape, fold
        # into the single batch axis a batched dispatch needs, and are restored
        # on the way out.
        def narrowed(a):
            return np.asarray(a, dtype=np.float16).astype(np.float32)

        def upload(a, dtype):
            return tensorplay.from_numpy(np.asarray(a, dtype=dtype)
                                         ).to("vulkan")

        tensorplay.manual_seed(163)
        cases = (
            ((6, 40, 8), (6, 8, 12), "equal batches"),
            ((2, 4, 60, 8), (2, 4, 8, 32), "four-dimensional operands"),
            ((1, 8, 48, 16), (1, 8, 16, 24), "attention-shaped pair"),
            ((4, 30, 6), (2, 4, 6, 9), "leading axis only on the right"),
            ((2, 4, 30, 6), (4, 6, 9), "leading axis only on the left"),
            ((1, 40, 8), (4, 8, 12), "single batch on the left"),
            ((4, 40, 8), (1, 8, 12), "single batch on the right"),
            ((3, 1, 20, 5), (3, 4, 5, 7), "inner singleton batch"),
            ((1, 5, 20, 5), (3, 5, 5, 7), "outer singleton batch"),
            ((4, 6), (2, 3, 6, 9), "a lone matrix on the left"),
            ((2, 3, 24, 6), (6, 5), "a lone matrix on the right"),
            ((1, 1, 12, 5), (1, 1, 5, 8), "all singleton batches"),
            ((0, 6, 4), (0, 4, 5), "an empty batch"),
            ((2, 0, 6, 4), (2, 0, 4, 5), "an empty inner batch"),
        )
        for lhs_shape, rhs_shape, tag in cases:
            a_cpu = (tp.randn(lhs_shape, device="cpu") * 0.2).numpy()
            b_cpu = (tp.randn(rhs_shape, device="cpu") * 0.2).numpy()
            want = np.matmul(a_cpu, b_cpu)
            got = (upload(a_cpu, np.float32)
                   @ upload(b_cpu, np.float32)).cpu().numpy()
            self.assertEqual(got.shape, want.shape, tag)
            np.testing.assert_allclose(
                got, want, rtol=2e-3, atol=2e-3, err_msg=tag)

        a_cpu = narrowed((tp.randn((2, 4, 50, 8), device="cpu") * 0.2).numpy())
        b_cpu = narrowed((tp.randn((2, 4, 8, 32), device="cpu") * 0.2).numpy())
        got = (upload(a_cpu, np.float16)
               @ upload(b_cpu, np.float16)).cpu().numpy().astype(np.float32)
        np.testing.assert_allclose(
            got, narrowed(np.matmul(a_cpu, b_cpu)), rtol=6e-3, atol=6e-3,
            err_msg="four-dimensional half product")

        # A batch axis neither operand can broadcast stays a loud error.
        with self.assertRaisesRegex(RuntimeError, "must match the size"):
            (upload((tp.randn((3, 20, 4), device="cpu").numpy()), np.float32)
             @ upload((tp.randn((2, 4, 6), device="cpu").numpy()), np.float32))


class TestVulkanDtypeConversion(unittest.TestCase):
    # Single precision is the hub of the conversion table: each payload type
    # the backend stores has a kernel to and from it, over a texture as well as
    # over a linear payload.  A byte-packed payload keeps one byte per element,
    # so a length that is not a multiple of four leaves lanes in the last word
    # that no read-back reaches.
    LENGTHS = (128, 200003)

    @staticmethod
    def _bits(array):
        # A Bool read-back may arrive as bytes or as a numpy bool and a half
        # arrives as the pattern it is stored in, so the lanes are compared in
        # the reading that names the bits rather than the one that interprets
        # them.
        if array.dtype == np.dtype(np.float16):
            return array.view(np.uint16)
        if array.dtype == np.dtype(np.bool_):
            return array.astype(np.int8)
        return array

    def _payload_cases(self, count):
        import tensorplay as tp
        rng = np.random.default_rng(31)
        return [
            ("bool", np.bool_, tp.bool, rng.random(count) > 0.35),
            ("int8", np.int8, tp.int8,
             rng.integers(-128, 128, count).astype(np.int8)),
            ("uint8", np.uint8, tp.uint8,
             rng.integers(0, 256, count).astype(np.uint8)),
            ("int32", np.int32, tp.int32,
             rng.integers(-100000, 100000, count).astype(np.int32)),
            ("float16", np.float16, tp.float16,
             (rng.standard_normal(count) * 3.0).astype(np.float16)),
        ]

    def test_narrow_payloads_convert_to_single_precision(self):
        import tensorplay as tp
        for count in self.LENGTHS:
            for name, _, _, values in self._payload_cases(count):
                device = tensorplay.from_numpy(values.copy()).to("vulkan")
                np.testing.assert_array_equal(
                    device.to(tp.float32).cpu().numpy(),
                    values.astype(np.float32),
                    err_msg=f"{name} -> float32 at {count} elements")

    def test_single_precision_converts_to_narrow_payloads(self):
        import tensorplay as tp
        rng = np.random.default_rng(37)
        for count in self.LENGTHS:
            x_cpu = (rng.standard_normal(count) * 5.0).astype(np.float32)
            device = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
            for name, np_dtype, dtype, _ in self._payload_cases(count):
                if name == "float16":
                    continue
                got = device.to(dtype).cpu().numpy()
                want = x_cpu.astype(np_dtype)
                # A Bool read-back may arrive as bytes or as a numpy bool; the
                # lanes are what matters, so compare them as integers.
                np.testing.assert_array_equal(
                    np.asarray(got).astype(np.int8 if name == "bool" else got.dtype),
                    np.asarray(want).astype(np.int8 if name == "bool" else want.dtype),
                    err_msg=f"float32 -> {name} at {count} elements")

    def test_byte_conversion_wraps_out_of_range_values(self):
        import tensorplay as tp
        # A value beyond the byte range truncates toward zero, then wraps into
        # the 256 byte patterns, which is what a host cast of the same value
        # gives.
        x_cpu = np.array(
            [300.7, -300.7, 255.5, -1.2, 127.9, -128.9, 2.5, -2.5, 0.0],
            dtype=np.float32)
        device = tensorplay.from_numpy(x_cpu.copy()).to("vulkan")
        np.testing.assert_array_equal(
            device.to(tp.int8).cpu().numpy(), x_cpu.astype(np.int8),
            err_msg="float32 -> int8 wrap")
        np.testing.assert_array_equal(
            device.to(tp.uint8).cpu().numpy(), x_cpu.astype(np.uint8),
            err_msg="float32 -> uint8 wrap")

    def test_single_precision_to_half_rounds_to_nearest_even(self):
        import tensorplay as tp
        # The value a half lane holds is decided here, not by the store into a
        # 16-bit image: those bits are rebuilt so the number handed to the
        # image already has an exact half form.  The samples below cover a tie,
        # the subnormal range and the overflow that becomes an infinity.
        x_cpu = np.array(
            [1.00048828125, 1.0009765625, -0.0001234, 5.96e-08, 2.0 ** -25,
             3.0 * 2.0 ** -24, 65504.0, 65520.0, -70000.0, 0.0],
            dtype=np.float32)
        for count in (128, 200003):
            values = np.resize(x_cpu, count).astype(np.float32)
            device = tensorplay.from_numpy(values.copy()).to("vulkan")
            np.testing.assert_array_equal(
                device.to(tp.float16).cpu().numpy(), values.astype(np.float16),
                err_msg=f"float32 -> float16 rounding at {count} elements")

    def test_the_narrow_float_and_the_signed_word_convert_directly(self):
        import tensorplay as tp
        # A select between a half plane and a signed-word plane is decided in
        # the half width, so the pair needs its own kernel in both directions.
        # Values beyond the largest finite half become infinities, exactly as a
        # host cast of the same code does.
        rng = np.random.default_rng(41)
        codes = np.concatenate([
            rng.integers(-4096, 4096, 64),
            np.array([65504, 65505, 65520, 65536, -65504, -65505, -70000,
                      2 ** 31 - 1, -(2 ** 31), 0, 1, -1], dtype=np.int64),
        ]).astype(np.int32)
        for count in self.LENGTHS:
            values = np.resize(codes, count).astype(np.int32)
            device = tensorplay.from_numpy(values.copy()).to("vulkan")
            np.testing.assert_array_equal(
                device.to(tp.float16).cpu().numpy(), values.astype(np.float16),
                err_msg=f"int32 -> float16 at {count} elements")
            # The codes a half lane holds exactly round-trip to the value they
            # came from; above 2048 the spacing is wider than one, so the
            # nearest half is a different number, and an infinity has no
            # integer form on either side.
            exact = np.abs(values) <= 2048
            back = device.to(tp.float16).to(tp.int32).cpu().numpy()
            np.testing.assert_array_equal(
                back[exact], values[exact],
                err_msg=f"int32 -> float16 -> int32 at {count} elements")

    def test_half_to_signed_word_truncates_toward_zero(self):
        import tensorplay as tp
        # Every half is an integer multiple of a power of two that single
        # precision holds, so widening first keeps the magnitude exact and only
        # the truncation decides the code.  Infinities and a not-a-number have
        # no integer reading on either side, so they stay out of the samples.
        values = np.array(
            [0.0, -0.0, 1.5, -1.5, 2.5, -2.5, 3.999, -3.999, 0.999,
             2047.9, -2048.1, 65504.0, -65504.0, 1.0 / 2048, -1.0 / 2048],
            dtype=np.float16)
        for count in (16, 200016):
            lane = np.resize(values, count).astype(np.float16)
            device = tensorplay.from_numpy(lane.copy()).to("vulkan")
            np.testing.assert_array_equal(
                device.to(tp.int32).cpu().numpy(), lane.astype(np.int32),
                err_msg=f"float16 -> int32 truncation at {count} elements")

    def test_conversion_rejects_pairs_without_a_kernel(self):
        import tensorplay as tp
        # A wide code carries the low bits single precision cannot hold, and a
        # signed word carries more than a byte's worth of them, so a pair that
        # needs those bits has no hop to answer it: it has to be refused
        # instead of answering with a rounded value.
        word = tensorplay.from_numpy(
            np.arange(64, dtype=np.int32)).to("vulkan")
        wide = tensorplay.from_numpy(
            np.arange(64, dtype=np.int64)).to("vulkan")
        for target in (tp.int8, tp.uint8):
            with self.assertRaisesRegex(
                    RuntimeError, "Unsupported Vulkan dtype conversion"):
                word.to(target)
        for target in (tp.int8, tp.uint8, tp.float16):
            with self.assertRaisesRegex(
                    RuntimeError, "Unsupported Vulkan dtype conversion"):
                wide.to(target)

    def test_a_pair_without_a_kernel_hops_through_single_precision(self):
        import tensorplay as tp
        # A byte-packed lane - a mask, a signed byte, a quantized code - and a
        # half lane hold a value single precision represents exactly, so the
        # pair that shares no kernel still has an answer: widen to the hub in
        # the storage the payload already sits in, then narrow.  The long row
        # lives in a buffer and the rank-4 block in an image, and the hop has to
        # answer in both.
        cases = [
            (tp.bool, np.bool_, np.array([True, False, False, True])),
            (tp.int8, np.int8,
             np.array([0, -1, 127, -128, 5, -5, 100, -100], dtype=np.int8)),
            (tp.uint8, np.uint8,
             np.array([0, 1, 255, 128, 5, 7, 64, 200], dtype=np.uint8)),
            (tp.float16, np.float16,
             np.array([0.0, -0.0, 1.5, -2.75, 255.5, -129.0, 100.25, -0.5],
                      dtype=np.float16)),
        ]
        targets = [(tp.bool, np.bool_), (tp.int8, np.int8), (tp.uint8, np.uint8),
                   (tp.int32, np.int32), (tp.int64, np.int64),
                   (tp.float16, np.float16), (tp.float32, np.float32)]
        for dtype, np_dtype, values in cases:
            for count in self.LENGTHS:
                lane = np.resize(values, count).astype(np_dtype)
                device = tensorplay.from_numpy(lane.copy()).to("vulkan")
                for target, target_np in targets:
                    np.testing.assert_array_equal(
                        self._bits(device.to(target).cpu().numpy()),
                        self._bits(lane.astype(target_np)),
                        err_msg=f"{np_dtype.__name__} -> {target_np.__name__} "
                                f"over {count} elements")
            block = np.resize(values, 12).astype(np_dtype).reshape(1, 2, 2, 3)
            device = tensorplay.from_numpy(block.copy()).to("vulkan")
            for target, target_np in targets:
                np.testing.assert_array_equal(
                    self._bits(device.to(target).cpu().numpy()),
                    self._bits(block.astype(target_np)),
                    err_msg=f"{np_dtype.__name__} -> {target_np.__name__} "
                            f"in an image")
            # A hop is still one conversion: the answer keeps the shape the
            # payload had and nothing else about it changes.
            self.assertEqual(tuple(device.to(tp.float32).shape), (1, 2, 2, 3))

    def test_the_wide_code_hops_for_the_zero_test_only(self):
        import tensorplay as tp
        # Rounding never moves a nonzero onto zero, so a wide code asked whether
        # each lane is zero reaches the answer through single precision even
        # where it holds bits the hub cannot name.
        values = np.array(
            [0, -1, 1, 255, -(2 ** 31), 2 ** 31, 2 ** 62, -(2 ** 62),
             2 ** 24 + 1, 2 ** 24], dtype=np.int64)
        for count in self.LENGTHS:
            lane = np.resize(values, count).astype(np.int64)
            device = tensorplay.from_numpy(lane.copy()).to("vulkan")
            np.testing.assert_array_equal(
                self._bits(device.to(tp.bool).cpu().numpy()),
                self._bits(lane.astype(np.bool_)),
                err_msg=f"int64 -> bool over {count} elements")
        block = np.resize(values, 12).astype(np.int64).reshape(1, 2, 2, 3)
        device = tensorplay.from_numpy(block.copy()).to("vulkan")
        np.testing.assert_array_equal(
            self._bits(device.to(tp.bool).cpu().numpy()),
            self._bits(block.astype(np.bool_)),
            err_msg="int64 -> bool in an image")
        # The same code answered exactly by its own kernel keeps its reading.
        np.testing.assert_array_equal(
            device.to(tp.int32).cpu().numpy(), block.astype(np.int32),
            err_msg="int64 -> int32 in an image")

    def test_a_hop_answers_a_view_as_the_payload_it_shows(self):
        import tensorplay as tp
        # A view is read before it converts, so the hop answers the lanes the
        # view shows rather than the storage behind it.
        base = np.array(
            [0.0, -0.0, 1.5, -2.75, 100.5, -100.5, 7.0, -8.5, 127.0, -128.0],
            dtype=np.float16)
        block = np.resize(base, 24).reshape(4, 6)
        device = tensorplay.from_numpy(block.copy()).to("vulkan")
        shown = device[:, 1::2]
        for target, np_dtype in ((tp.int8, np.int8), (tp.int64, np.int64),
                                 (tp.uint8, np.uint8)):
            np.testing.assert_array_equal(
                self._bits(shown.to(target).cpu().numpy()),
                self._bits(block[:, 1::2].astype(np_dtype)),
                err_msg=f"a strided half view read as {np_dtype.__name__}")


class TestVulkanHalfComparison(unittest.TestCase):
    # Comparisons read a half payload through single precision, which maps each
    # lane exactly, so ties, signed zeros, infinities and not-a-number decide
    # the same way the host reference does.  Both storages need a case: a row
    # wider than the image-axis limit lives in a buffer, where two lanes share
    # one word and the Bool result packs four lanes per word.
    LENGTHS = (4096, 200003)
    OPS = ("eq", "ne", "lt", "le", "gt", "ge")

    def _half_values(self, count, seed):
        rng = np.random.default_rng(seed)
        values = (rng.standard_normal(count) * 4.0).astype(np.float16)
        specials = np.array(
            [np.inf, -np.inf, np.nan, 0.0, -0.0, 65504.0, -65504.0,
             2.0 ** -24, 1.0009765625, 1.0], dtype=np.float16)
        values[:specials.size] = specials
        return values

    def test_half_tensors_compare_elementwise(self):
        import tensorplay.functional as F
        for count in self.LENGTHS:
            a_cpu = self._half_values(count, 5)
            b_cpu = self._half_values(count, 6)
            a = tensorplay.from_numpy(a_cpu.copy()).to("vulkan")
            b = tensorplay.from_numpy(b_cpu.copy()).to("vulkan")
            host_a = tensorplay.from_numpy(a_cpu.copy())
            host_b = tensorplay.from_numpy(b_cpu.copy())
            for name in self.OPS:
                op = getattr(F, name)
                got = op(a, b).cpu().numpy()
                want = op(host_a, host_b).cpu().numpy()
                np.testing.assert_array_equal(
                    got.astype(np.int8), want.astype(np.int8),
                    err_msg=f"{name} over half lanes at {count} elements")

    def test_half_against_scalar_rounds_the_threshold(self):
        import tensorplay.functional as F
        # The scalar enters the element domain before the decision, so a value
        # that falls between two representable halves is worth whichever one it
        # rounds to: 1.00048828125 sits below the midpoint above 1.0 and so
        # matches a lane holding exactly 1.0.
        scalars = (0.0, -0.0, 1.0, 1.00048828125, 1.0009765625, -3.5,
                   65504.0, 70000.0, 2, 1)
        for count in self.LENGTHS:
            values = self._half_values(count, 9)
            device = tensorplay.from_numpy(values.copy()).to("vulkan")
            host = tensorplay.from_numpy(values.copy())
            for name in self.OPS:
                op = getattr(F, name)
                for scalar in scalars:
                    np.testing.assert_array_equal(
                        op(device, scalar).cpu().numpy().astype(np.int8),
                        op(host, scalar).cpu().numpy().astype(np.int8),
                        err_msg=f"{name} against {scalar!r} at {count} elements")

    def test_half_comparison_rejects_operand_without_a_kernel(self):
        import tensorplay.functional as F
        # Byte lanes have no linear comparison form: a wide payload of them is
        # refused instead of being read with the wrong element width.
        device = tensorplay.from_numpy(
            np.arange(200003, dtype=np.int8)).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "over buffer storage"):
            F.gt(device, 3)


class TestVulkanHalfWhere(unittest.TestCase):
    # Over a linear payload a half word carries two lanes while the Bool
    # condition carries four bytes, so the two condition bytes a result word
    # needs sit at bit 0 or bit 16 of their word.  Structured conditions, not
    # random ones, tell those byte slots apart.
    LENGTHS = (4096, 200003)

    def _gradient(self, count):
        idx = np.arange(count)
        return ((idx % 512).astype(np.float16),
                (512 + idx % 512).astype(np.float16))

    def _patterns(self, count):
        idx = np.arange(count)
        return [
            ("every lane", np.ones(count, dtype=bool)),
            ("no lane", np.zeros(count, dtype=bool)),
            ("alternating lanes", idx % 2 == 0),
            ("lane pairs", (idx % 4) < 2),
            ("every fourth lane", idx % 4 == 0),
            ("upper half of each word", (idx % 4) >= 2),
        ]

    def test_half_where_selects_lanes_in_order(self):
        import tensorplay.functional as F
        for count in self.LENGTHS:
            a_np, b_np = self._gradient(count)
            a = tensorplay.from_numpy(a_np.copy()).to("vulkan")
            b = tensorplay.from_numpy(b_np.copy()).to("vulkan")
            for name, condition in self._patterns(count):
                device = tensorplay.from_numpy(condition.copy()).to("vulkan")
                np.testing.assert_array_equal(
                    F.where(device, a, b).cpu().numpy().view(np.uint16),
                    np.where(condition, a_np, b_np).view(np.uint16),
                    err_msg=f"half where selecting {name} at {count} elements")

    def test_half_where_keeps_rejected_lane_out_of_result(self):
        import tensorplay.functional as F
        # Selection is per lane and stays in the 16-bit fields: an infinity or a
        # not-a-number on the rejected side cannot reach the result through
        # arithmetic recombination.
        count = 200003
        a_np = np.full(count, np.inf, dtype=np.float16)
        a_np[1::7] = np.nan
        _, b_np = self._gradient(count)
        a = tensorplay.from_numpy(a_np.copy()).to("vulkan")
        b = tensorplay.from_numpy(b_np.copy()).to("vulkan")
        for name, condition in self._patterns(count):
            device = tensorplay.from_numpy(condition.copy()).to("vulkan")
            np.testing.assert_array_equal(
                F.where(device, a, b).cpu().numpy().view(np.uint16),
                np.where(condition, a_np, b_np).view(np.uint16),
                err_msg=f"half where with {name} of an infinite branch")

    def test_half_where_with_scalar_branch(self):
        import tensorplay.functional as F
        for count in self.LENGTHS:
            a_np, b_np = self._gradient(count)
            _, condition = self._patterns(count)[3]
            a = tensorplay.from_numpy(a_np.copy()).to("vulkan")
            b = tensorplay.from_numpy(b_np.copy()).to("vulkan")
            device = tensorplay.from_numpy(condition.copy()).to("vulkan")
            for scalar in (0.0, -2.5, 65504.0, 70000.0, 3):
                np.testing.assert_array_equal(
                    F.where(device, scalar, b).cpu().numpy().view(np.uint16),
                    np.where(condition, np.float16(scalar), b_np).view(np.uint16),
                    err_msg=f"half scalar branch {scalar!r} at {count}")
                np.testing.assert_array_equal(
                    F.where(device, a, scalar).cpu().numpy().view(np.uint16),
                    np.where(condition, a_np, np.float16(scalar)).view(np.uint16),
                    err_msg=f"half scalar branch against {scalar!r} at {count}")


class TestVulkanHalfGather(unittest.TestCase):
    # A half gather moves 16-bit lanes: one result word owns two elements, and a
    # row width that is not a multiple of two leaves a word shared between two
    # rows.  Whole-number payloads keep the bit-for-bit comparison meaningful.

    @staticmethod
    def _table(rows, cols):
        r = np.arange(rows, dtype=np.int64)[:, None]
        c = np.arange(cols, dtype=np.int64)[None, :]
        return ((r * 31 + c * 7) % 2048 - 1024).astype(np.float16)

    def _check_pair(self, table_np, sel, vk_table, vk_sel, err_msg):
        import tensorplay.functional as F
        want = table_np[sel].view(np.uint16)
        for name, out in (
                ("embedding", F.embedding(vk_table, vk_sel)),
                ("index_select", F.index_select(vk_table, 0, vk_sel))):
            got = out.cpu().numpy()
            np.testing.assert_array_equal(
                got.view(np.uint16), want, err_msg=f"{name} {err_msg}")

    def test_half_gather_over_texel_table(self):
        # A table whose flat form fits an image axis is gathered through the
        # texel route, which reads half lanes widened by the format conversion.
        rng = np.random.default_rng(3)
        for cols in (32, 37):
            table_np = self._table(200, cols)
            vk_table = tensorplay.from_numpy(table_np.copy()).to("vulkan")
            for count in (129, 512):
                sel = rng.integers(0, 200, count).astype(np.int32)
                self._check_pair(
                    table_np, sel, vk_table,
                    tensorplay.from_numpy(sel.copy()).to("vulkan"),
                    f"over a 200x{cols} texel table at {count} indices")

    def test_half_gather_straddling_the_image_axis_limit(self):
        # The index list holds one index per texel, so a list beyond the axis
        # limit has no image form even when the table does; the gather then
        # copies the half payload into dense linear storage instead of refusing
        # the shape pair.  The lengths below cross the limit from both sides.
        rng = np.random.default_rng(4)
        table_np = self._table(1000, 8)
        vk_table = tensorplay.from_numpy(table_np.copy()).to("vulkan")
        for count in (512, 16383, 16384, 16385, 20000):
            sel = rng.integers(0, 1000, count).astype(np.int32)
            self._check_pair(
                table_np, sel, vk_table,
                tensorplay.from_numpy(sel.copy()).to("vulkan"),
                f"over a texel table at {count} indices")

    def test_half_gather_over_linear_table(self):
        # A vocabulary past the axis limit lives in linear storage already, so
        # the gather reads it where it lies; the widths below leave half rows
        # straddling word bounds.
        rng = np.random.default_rng(6)
        for cols in (64, 33):
            table_np = self._table(20000, cols)
            vk_table = tensorplay.from_numpy(table_np.copy()).to("vulkan")
            count = 512 if cols == 64 else 4096
            sel = rng.integers(0, 20000, count).astype(np.int32)
            self._check_pair(
                table_np, sel, vk_table,
                tensorplay.from_numpy(sel.copy()).to("vulkan"),
                f"over a 20000x{cols} linear table at {count} indices")

    def test_half_gather_of_a_flat_linear_payload(self):
        import tensorplay.functional as F
        # A long 1-d payload is stored linearly and gathered with one element per
        # logical position, so the row stride of the selection is one.
        rng = np.random.default_rng(7)
        table_np = (np.arange(70000) % 2048).astype(np.float16)
        vk_table = tensorplay.from_numpy(table_np.copy()).to("vulkan")
        for count in (66558, 66559, 66560):
            sel = rng.integers(0, table_np.size, count).astype(np.int32)
            got = F.index_select(
                vk_table, 0,
                tensorplay.from_numpy(sel.copy()).to("vulkan")).cpu().numpy()
            np.testing.assert_array_equal(
                got.view(np.uint16), table_np[sel].view(np.uint16),
                err_msg=f"flat half gather at {count} indices")

    def test_a_counting_table_this_long_is_read_in_words(self):
        # A payload this flat has no image form, so the codes travel through
        # linear storage - both for a list of rows and for a per-position pick.
        import tensorplay.functional as F
        table = np.arange(200003, dtype=np.int32)
        device = tensorplay.from_numpy(table.copy()).to("vulkan")
        sel = np.arange(0, 200003, 25000, dtype=np.int32)
        codes = tensorplay.from_numpy(sel.copy()).to("vulkan")
        got = F.index_select(device, 0, codes)
        np.testing.assert_array_equal(
            got.cpu().numpy(), table[sel], err_msg="long int32 table")
        self.assertEqual(got.dtype, tensorplay.int32)
        picked = tensorplay.gather(device, 0, codes)
        np.testing.assert_array_equal(
            picked.cpu().numpy(), table[sel], err_msg="long int32 gather")
        self.assertEqual(picked.dtype, tensorplay.int32)


class TestVulkanHalfCat(unittest.TestCase):
    # Concatenating half payloads shares a destination word between two lanes,
    # so a join at an odd element count leaves one word owned by two passes.
    # Whole numbers keep a lane taken from the wrong side of a join visible.

    @staticmethod
    def _flat(count):
        return (np.arange(count, dtype=np.float32) % 2048 - 1024).astype(
            np.float16)

    @staticmethod
    def _grid(rows, cols):
        return (np.arange(rows * cols, dtype=np.float32) % 2048 - 1024).astype(
            np.float16).reshape(rows, cols)

    def _cat(self, parts, dim):
        import tensorplay.functional as F
        device = [tensorplay.from_numpy(p.copy()).to("vulkan") for p in parts]
        return F.cat(device, dim).cpu().numpy()

    def test_half_cat_over_linear_storage(self):
        # A 1-d join longer than the image axis limit lands in linear storage,
        # and the odd totals leave the last word half filled.
        for lengths in ((20000, 19999), (10000, 10000, 10001)):
            parts = [self._flat(n) for n in lengths]
            np.testing.assert_array_equal(
                self._cat(parts, 0).view(np.uint16),
                np.concatenate(parts).view(np.uint16),
                err_msg=f"half 1-d cat of {lengths}")

    def test_half_cat_with_a_join_inside_a_word(self):
        # 1001 rows of 37 columns ends at element 37037, so the word holding
        # elements 37036 and 37037 is written by two different passes.
        for lengths in ((1001, 999), (1001, 999, 1000)):
            parts = [self._grid(n, 37) for n in lengths]
            np.testing.assert_array_equal(
                self._cat(parts, 0).view(np.uint16),
                np.concatenate(parts, 0).view(np.uint16),
                err_msg=f"half row cat of {lengths} rows x 37")

    def test_half_cat_along_the_width_axis(self):
        # Joining on the width axis puts the split inside every row, and a row
        # of 17 halfs shifts which word the next row's join lands in.
        a, b = self._grid(20000, 8), self._grid(20000, 9)
        np.testing.assert_array_equal(
            self._cat([a, b], 1).view(np.uint16),
            np.concatenate([a, b], 1).view(np.uint16),
            err_msg="half cat along the width axis")

    def test_half_cat_over_texel_storage(self):
        for dim, shapes in ((0, ((64, 33), (32, 33))),
                            (1, ((64, 33), (64, 34)))):
            parts = [self._grid(*shape) for shape in shapes]
            np.testing.assert_array_equal(
                self._cat(parts, dim).view(np.uint16),
                np.concatenate(parts, dim).view(np.uint16),
                err_msg=f"half cat of {shapes} along dim {dim}")

    def test_half_glu_over_both_storages(self):
        import tensorplay.functional as F
        # The gate is a sigmoid, so the check is on values rather than bits; the
        # split itself still has to land on the right half of every word.
        for rows, cols, dim, label in (
                (20000, 34, 0, "linear"),
                (64, 68, 1, "texel")):
            half_rows, half_cols = (rows, cols // 2) if dim else (rows // 2, cols)
            parts = [self._grid(half_rows, half_cols) for _ in range(2)]
            joined = np.concatenate(parts, dim)
            device = F.glu(
                tensorplay.from_numpy(joined.copy()).to("vulkan"),
                dim).cpu().numpy()
            span = joined.shape[dim] // 2
            first = np.take(joined, np.arange(0, span), axis=dim).astype(
                np.float32)
            second = np.take(joined, np.arange(span, 2 * span), axis=dim).astype(
                np.float32)
            # A gate of -1024 saturates: the sigmoid is zero, which numpy
            # reports as an overflow in the exponential.
            with np.errstate(over="ignore"):
                want = second / (1.0 + np.exp(-first))
            self.assertEqual(device.shape, want.shape, msg=label)
            np.testing.assert_allclose(
                device.astype(np.float32), want, rtol=2e-3, atol=1e-3,
                err_msg=f"half glu over {label} storage")


class TestVulkanHalfLinearViews(unittest.TestCase):
    # A strided view of a half payload has to gather 16-bit fields: a word
    # holding two elements is read whole and split, and an odd start or step
    # puts the two lanes a view needs in different words.

    @staticmethod
    def _flat(count):
        return (np.arange(count, dtype=np.float32) % 2048 - 1024).astype(
            np.float16)

    @staticmethod
    def _grid(rows, cols):
        return (np.arange(rows * cols, dtype=np.float32) % 2048 - 1024).astype(
            np.float16).reshape(rows, cols)

    def test_half_slices_of_a_linear_payload(self):
        # 70000 elements is past the image axis limit, so the source lies in
        # linear storage and each slice gathers from there.
        host = self._flat(70000)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        for start, stop, step in ((0, 20001, 1), (1, 20001, 1), (3, 16385, 2),
                                  (0, 70000, 3), (7, 7 + 16383, 4)):
            np.testing.assert_array_equal(
                device[start:stop:step].cpu().numpy().view(np.uint16),
                host[start:stop:step].view(np.uint16),
                err_msg=f"half slice [{start}:{stop}:{step}]")

    def test_half_slice_into_a_texel_output(self):
        # A short slice of a linear payload has an image form, so the gather
        # writes four half lanes per texel instead of one word per invocation.
        host = self._flat(70000)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_array_equal(
            device[10:4011].cpu().numpy().view(np.uint16),
            host[10:4011].view(np.uint16),
            err_msg="half slice gathered into an image")

    def test_half_views_of_a_linear_grid(self):
        # A 20000 x 37 table is linear through its height axis; the row slice
        # stays contiguous while the column slice reads every other field of
        # words shared with the neighbouring column.
        host = self._grid(20000, 37)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_array_equal(
            device[1:17001].cpu().numpy().view(np.uint16),
            host[1:17001].view(np.uint16),
            err_msg="half row slice of a linear grid")
        np.testing.assert_array_equal(
            device[:, 3:34:2].cpu().numpy().view(np.uint16),
            host[:, 3:34:2].view(np.uint16),
            err_msg="half strided column slice of a linear grid")
        np.testing.assert_array_equal(
            device[9999].cpu().numpy().view(np.uint16),
            host[9999].view(np.uint16),
            err_msg="half row selection of a linear grid")

    def test_half_transpose_of_a_linear_grid(self):
        host = self._grid(20000, 37)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_array_equal(
            device.t().cpu().numpy().view(np.uint16),
            host.T.view(np.uint16),
            err_msg="half transpose of a linear grid")

    def test_half_reshape_across_storages(self):
        host = self._flat(70000)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_array_equal(
            device[:4004].reshape(28, 143).cpu().numpy().view(np.uint16),
            host[:4004].reshape(28, 143).view(np.uint16),
            err_msg="half reshape of a linear payload into an image")
        grid = self._grid(20000, 37)
        vk_grid = tensorplay.from_numpy(grid.copy()).to("vulkan")
        np.testing.assert_array_equal(
            vk_grid[:, 0].reshape(grid.shape[0]).cpu().numpy().view(np.uint16),
            grid[:, 0].view(np.uint16),
            err_msg="half strided column re-shaped flat")

    def test_half_expand_from_texel_into_linear_storage(self):
        # A broadcast whose extents pass the axis limit is materialized out of
        # an image into a linear half payload; the re-read of one position per
        # lane is what a zero stride means.
        host = self._grid(1, 4)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_array_equal(
            device.expand(20000, 4).cpu().numpy().view(np.uint16),
            np.repeat(host, 20000, axis=0).view(np.uint16),
            err_msg="half broadcast into linear storage")


class TestVulkanHalfFlip(unittest.TestCase):
    # A flip only moves lanes, so every field of a half payload has to land
    # where the host reference puts it.  Over linear storage the two fields of
    # a word name neighbouring positions, which a flipped axis can separate,
    # so the variant unwinds the element index before it picks the word.

    @staticmethod
    def _flat(count):
        return (np.arange(count, dtype=np.float32) % 97).astype(np.float16)

    @staticmethod
    def _grid(sizes):
        count = int(np.prod(sizes))
        return (np.arange(count, dtype=np.float32)
                % 97).astype(np.float16).reshape(sizes)

    def test_half_flip_of_an_image_payload(self):
        import tensorplay.functional as F
        host = self._grid((2, 3, 4, 5))
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        for dims in ([0], [1], [2], [3], [-1], [0, 3], [1, 2],
                     [0, 1, 2, 3]):
            np.testing.assert_array_equal(
                F.flip(device, dims).cpu().numpy().view(np.uint16),
                np.flip(host, axis=dims).view(np.uint16),
                err_msg=f"half image flip at dims={dims}")

    def test_half_flip_of_a_linear_payload(self):
        import tensorplay.functional as F
        # 70001 lanes leave a half word holding one live field, so the tail
        # word of the destination is written from a single lane.
        host = self._flat(70001)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        for dims in ([0], [-1]):
            np.testing.assert_array_equal(
                F.flip(device, dims).cpu().numpy().view(np.uint16),
                np.flip(host, axis=dims).view(np.uint16),
                err_msg=f"half linear flip of a flat payload at dims={dims}")
        for sizes in ((40000, 3), (40001, 3)):
            grid = self._grid(sizes)
            vk_grid = tensorplay.from_numpy(grid.copy()).to("vulkan")
            for dims in ([0], [1], [0, 1]):
                np.testing.assert_array_equal(
                    F.flip(vk_grid, dims).cpu().numpy().view(np.uint16),
                    np.flip(grid, axis=dims).view(np.uint16),
                    err_msg=f"half linear flip of {sizes} at dims={dims}")

    def test_half_flip_keeps_the_special_lanes(self):
        import tensorplay.functional as F
        # Infinities, both zeros and the smallest subnormal all survive the
        # unpack/pack round trip exactly, so a reorder keeps their bits.
        host = np.array(
            [np.inf, -np.inf, 0.0, -0.0, 2.0 ** -24, 65504.0, -65504.0,
             1.0, -1.0, 2.0 ** -14], dtype=np.float16)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        for dims in ([0], [-1]):
            np.testing.assert_array_equal(
                F.flip(device, dims).cpu().numpy().view(np.uint16),
                np.flip(host, axis=dims).view(np.uint16),
                err_msg="half flip lost a special lane")
        wide = np.tile(host, 7000).astype(np.float16)
        vk_wide = tensorplay.from_numpy(wide.copy()).to("vulkan")
        np.testing.assert_array_equal(
            F.flip(vk_wide, [-1]).cpu().numpy().view(np.uint16),
            np.flip(wide, axis=-1).view(np.uint16),
            err_msg="half flip of a linear payload lost a special lane")


class TestVulkanHalfLerp(unittest.TestCase):
    # The mix runs in single precision and narrows on the way in, so a lane
    # may sit one half unit of least precision from the host value.  A payload
    # too wide for an image has no texel to name, and the texel-only variants
    # refuse it instead of reading a buffer as one.

    def _operands(self, count):
        base = np.linspace(-3.0, 3.0, count)
        end = np.linspace(5.0, -5.0, count)
        weight = (np.arange(count, dtype=np.float32) % 17) / 16.0
        # The device reads half lanes, so the host mix starts from the same
        # rounded operands.
        cast = lambda a: np.asarray(a, dtype=np.float16).astype(np.float32)
        return cast(base), cast(end), cast(weight)

    def test_half_lerp_with_a_scalar_weight(self):
        import tensorplay.functional as F
        shapes = (4, 5, 6)
        base, end, _ = self._operands(4 * 5 * 6)
        device = tensorplay.from_numpy(
            base.reshape(shapes).astype(np.float16)).to("vulkan")
        vk_end = tensorplay.from_numpy(
            end.reshape(shapes).astype(np.float16)).to("vulkan")
        for weight in (0.0, 0.25, 1.0, -0.5):
            want = (base + weight * (end - base)).astype(
                np.float16).astype(np.float32).reshape(shapes)
            np.testing.assert_allclose(
                F.lerp(device, vk_end, weight).cpu().numpy().astype(np.float32),
                want, rtol=1e-2, atol=1e-2, err_msg=f"half lerp at weight={weight}")

    def test_half_lerp_with_a_tensor_weight(self):
        import tensorplay.functional as F
        shapes = (4, 5, 6)
        base, end, weight = self._operands(4 * 5 * 6)
        start = tensorplay.from_numpy(
            base.reshape(shapes).astype(np.float16)).to("vulkan")
        vk_end = tensorplay.from_numpy(
            end.reshape(shapes).astype(np.float16)).to("vulkan")
        vk_weight = tensorplay.from_numpy(
            weight.reshape(shapes).astype(np.float16)).to("vulkan")
        want = (base + weight * (end - base)).astype(
            np.float16).astype(np.float32).reshape(shapes)
        np.testing.assert_allclose(
            F.lerp(start, vk_end, vk_weight).cpu().numpy().astype(np.float32),
            want, rtol=1e-2, atol=1e-2, err_msg="half lerp with a tensor weight")
        out = F.lerp_(start, vk_end, vk_weight)
        self.assertIs(out, start, "half lerp_ did not return the same tensor")
        np.testing.assert_allclose(
            out.cpu().numpy().astype(np.float32), want,
            rtol=1e-2, atol=1e-2, err_msg="half lerp_ left the payload wrong")

    def test_lerp_refuses_a_linear_payload(self):
        import tensorplay.functional as F
        for dtype in (np.float32, np.float16):
            base = tensorplay.from_numpy(
                np.linspace(-1.0, 1.0, 70001).astype(dtype)).to("vulkan")
            end = tensorplay.from_numpy(
                np.linspace(1.0, -1.0, 70001).astype(dtype)).to("vulkan")
            weight = tensorplay.from_numpy(
                np.zeros(70001, dtype=dtype)).to("vulkan")
            with self.assertRaisesRegex(
                    RuntimeError, "requires texture storage"):
                F.lerp(base, end, weight)


class TestVulkanCumsumStorages(unittest.TestCase):
    # A scan keeps its running total in single precision and rounds once when
    # the lane is stored, so a host reference that does the same matches bit for
    # bit in either element type and over either storage.  Channels are the
    # packed axis of an image: a prefix there has to carry the whole of every
    # earlier texel and the lower lanes of its own, which a per-lane carry gets
    # wrong.

    @staticmethod
    def _flat(count):
        # A period of nine values that sum to zero keeps every prefix small.
        return (np.arange(count, dtype=np.float32) % 9 - 4).astype(np.float16)

    @staticmethod
    def _grid(sizes):
        count = int(np.prod(sizes))
        return (np.arange(count, dtype=np.float32)
                % 9 - 4).astype(np.float16).reshape(sizes)

    def test_half_cumsum_of_an_image_payload(self):
        import tensorplay.functional as F
        host = self._grid((3, 5, 4, 2))
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        for dim in range(4):
            np.testing.assert_array_equal(
                F.cumsum(device, dim).cpu().numpy().view(np.uint16),
                np.cumsum(
                    host.astype(np.float32), axis=dim).astype(
                        np.float16).view(np.uint16),
                err_msg=f"half image scan at dim={dim}")

    def test_cumsum_channel_axis_carries_the_whole_texel(self):
        import tensorplay.functional as F
        # Five channels need two texels, so the second texel depends on the
        # total of the first one rather than on a per-lane carry.
        host = (np.arange(2 * 5 * 3 * 4, dtype=np.float32)
                % 9 - 4).astype(np.float32).reshape(2, 5, 3, 4)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_array_equal(
            F.cumsum(device, 1).cpu().numpy(),
            np.cumsum(host, axis=1),
            err_msg="float scan along the packed channel axis")

    def test_half_cumsum_of_a_linear_payload(self):
        import tensorplay.functional as F
        host = self._flat(70001)
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        want = np.cumsum(host.astype(np.float32), axis=0).astype(
            np.float16).view(np.uint16)
        np.testing.assert_array_equal(
            F.cumsum(device, 0).cpu().numpy().view(np.uint16), want,
            err_msg="half flat scan over a payload with a tail word")
        for sizes in ((40000, 3), (40001, 2)):
            grid = self._grid(sizes)
            vk_grid = tensorplay.from_numpy(grid.copy()).to("vulkan")
            for dim in range(2):
                np.testing.assert_array_equal(
                    F.cumsum(vk_grid, dim).cpu().numpy().view(np.uint16),
                    np.cumsum(
                        grid.astype(np.float32), axis=dim).astype(
                            np.float16).view(np.uint16),
                    err_msg=f"half linear scan of {sizes} at dim={dim}")
        # A row-count axis beyond the image limit while the width still
        # fits one texel: the scan walks 20000 rows of 37 columns.
        wide = (np.arange(20000 * 37, dtype=np.float32)
                % 9 - 4).astype(np.float16).reshape(20000, 37)
        vk_wide = tensorplay.from_numpy(wide.copy()).to("vulkan")
        np.testing.assert_array_equal(
            F.cumsum(vk_wide, 1).cpu().numpy().view(np.uint16),
            np.cumsum(wide.astype(np.float32), axis=1).astype(
                np.float16).view(np.uint16),
            err_msg="half scan along the width of a linear grid")

    def test_cumsum_widens_to_a_requested_type(self):
        import tensorplay.functional as F
        host = self._grid((40000, 3))
        device = tensorplay.from_numpy(host.copy()).to("vulkan")
        # The walk runs in the width the result is asked for, so a float result
        # carries a float total rather than the payload's own rounded one.
        want = np.cumsum(host.astype(np.float32), axis=0)
        np.testing.assert_array_equal(
            F.cumsum(device, 0, dtype=tensorplay.float32).cpu().numpy(), want,
            err_msg="half payload scanned into a float result")


class TestVulkanCumsumOfCounts(unittest.TestCase):
    # A running total of counts needs more range than one count, so a counted
    # payload answers with the wide code and the walk carries across the pair of
    # words that holds one element.  A narrower answer is the same total cut to
    # the width named, which is also what a walk that wrapped inside that width
    # at every step gives.

    @staticmethod
    def _device(host):
        return tensorplay.from_numpy(np.asarray(host).copy()).to("vulkan")

    @staticmethod
    def _back(payload):
        return np.asarray(payload.cpu())

    def test_a_count_answers_with_the_wide_code(self):
        import tensorplay.functional as F
        host = np.arange(24, dtype=np.int32).reshape(4, 6) - 8
        device = self._device(host)
        for dim in (0, 1):
            got = F.cumsum(device, dim)
            self.assertEqual(got.dtype, tensorplay.int64)
            np.testing.assert_array_equal(
                self._back(got), np.cumsum(host.astype(np.int64), axis=dim),
                err_msg=f"int32 scan at dim={dim}")
        np.testing.assert_array_equal(
            self._back(F.cumsum(device, -1)),
            np.cumsum(host.astype(np.int64), axis=1))
        # The source still holds its own counts.
        np.testing.assert_array_equal(self._back(device), host)

    def test_a_wide_payload_keeps_its_width(self):
        import tensorplay.functional as F
        host = (np.arange(60, dtype=np.int64) * 1000 - 20000).reshape(4, 3, 5)
        device = self._device(host)
        for dim in range(3):
            got = F.cumsum(device, dim)
            self.assertEqual(got.dtype, tensorplay.int64)
            np.testing.assert_array_equal(
                self._back(got), np.cumsum(host, axis=dim),
                err_msg=f"int64 scan at dim={dim}")
        grid = (np.arange(2 * 3 * 4 * 5, dtype=np.int32)
                % 13 - 6).reshape(2, 3, 4, 5)
        for dim in range(4):
            np.testing.assert_array_equal(
                self._back(F.cumsum(self._device(grid), dim)),
                np.cumsum(grid.astype(np.int64), axis=dim),
                err_msg=f"rank-4 counts at dim={dim}")

    def test_the_total_reaches_past_one_word(self):
        import tensorplay.functional as F
        crossing = np.array([3_000_000_000, 3_000_000_000, 3_000_000_000,
                             -4_000_000_000, 1], dtype=np.int64)
        np.testing.assert_array_equal(
            self._back(F.cumsum(self._device(crossing), 0)),
            np.cumsum(crossing),
            err_msg="a sum that leaves one word still counts on")
        mixed = np.array([2**40, -(2**40), 2**33, -(2**32), 5, 2**50, -(2**50)],
                         dtype=np.int64)
        np.testing.assert_array_equal(
            self._back(F.cumsum(self._device(mixed), 0)), np.cumsum(mixed),
            err_msg="a borrow out of the high word comes back")
        past = np.array([np.iinfo(np.int64).max - 3, 2, 2, 2], dtype=np.int64)
        with np.errstate(over="ignore"):
            want = np.cumsum(past)
        np.testing.assert_array_equal(
            self._back(F.cumsum(self._device(past), 0)), want,
            err_msg="a total past the signed range lands where the codes land")

    def test_a_narrower_answer_wraps_where_a_short_walk_would(self):
        import tensorplay.functional as F
        host = np.array([2_000_000_000, 2_000_000_000, 2_000_000_000,
                         -2_000_000_000, 1], dtype=np.int32)
        got = F.cumsum(self._device(host), 0, dtype=tensorplay.int32)
        self.assertEqual(got.dtype, tensorplay.int32)
        # Cut once at the end...
        with np.errstate(over="ignore"):
            narrowed = np.cumsum(host.astype(np.int64)).astype(np.int32)
        # ...and the same answer as a walk that wraps inside one word at every
        # step, which is what a counted result promises.
        walked = np.zeros(host.shape, dtype=np.int32)
        total = 0
        for at, value in enumerate(host):
            total = (total + int(value)) % (1 << 32)
            if total >= 1 << 31:
                total -= 1 << 32
            walked[at] = total
        np.testing.assert_array_equal(self._back(got), narrowed)
        np.testing.assert_array_equal(self._back(got), walked)

    def test_a_mask_counts_its_own_positions(self):
        import tensorplay.functional as F
        mask = np.arange(40, dtype=np.int32) % 3 == 0
        got = F.cumsum(self._device(mask), 0)
        self.assertEqual(got.dtype, tensorplay.int64)
        drawn = self._back(got)
        np.testing.assert_array_equal(drawn, np.cumsum(mask.astype(np.int64)))
        # The last prefix is the number of positions, which is what a scan of a
        # mask is asked for.
        self.assertEqual(int(drawn[-1]), int(mask.sum()))
        narrow = F.cumsum(self._device(mask), 0, dtype=tensorplay.int32)
        self.assertEqual(narrow.dtype, tensorplay.int32)
        np.testing.assert_array_equal(self._back(narrow), drawn.astype(np.int32))

    def test_a_narrow_count_scans_in_the_wide_code(self):
        import tensorplay.functional as F
        signed = ((np.arange(24) % 9) - 4).astype(np.int8).reshape(2, 3, 4)
        unsigned = ((np.arange(24) * 7) % 200).astype(np.uint8).reshape(2, 3, 4)
        for host in (signed, unsigned):
            for dim in range(3):
                got = F.cumsum(self._device(host), dim)
                self.assertEqual(got.dtype, tensorplay.int64)
                np.testing.assert_array_equal(
                    self._back(got),
                    np.cumsum(host.astype(np.int64), axis=dim),
                    err_msg=f"{host.dtype} scan at dim={dim}")

    def test_a_count_can_be_asked_for_a_measured_answer(self):
        import tensorplay.functional as F
        counts = np.arange(12, dtype=np.int32).reshape(3, 4)
        drawn = F.cumsum(self._device(counts), 1, dtype=tensorplay.float32)
        self.assertEqual(drawn.dtype, tensorplay.float32)
        np.testing.assert_array_equal(
            self._back(drawn), np.cumsum(counts.astype(np.float32), axis=1))
        wide = F.cumsum(self._device(counts), 0, dtype=tensorplay.int64)
        np.testing.assert_array_equal(
            self._back(wide), np.cumsum(counts.astype(np.int64), axis=0))
        # The reverse: a measured payload answered with counts.
        floats = (np.arange(12, dtype=np.float32) / 2).reshape(3, 4)
        counted = F.cumsum(self._device(floats), 0, dtype=tensorplay.int64)
        np.testing.assert_array_equal(
            self._back(counted), np.cumsum(floats.astype(np.int64), axis=0))

    def test_a_view_is_laid_out_before_it_is_walked(self):
        import tensorplay.functional as F
        rows = np.arange(24, dtype=np.int64).reshape(4, 6)
        columns = self._device(rows)[:, 1:6:2]
        np.testing.assert_array_equal(
            self._back(F.cumsum(columns, 0)),
            np.cumsum(rows[:, 1:6:2], axis=0),
            err_msg="strided counts")
        grid = np.arange(20, dtype=np.float32).reshape(4, 5)
        np.testing.assert_array_equal(
            self._back(F.cumsum(self._device(grid)[1:4], 1)),
            np.cumsum(grid[1:4], axis=1),
            err_msg="a cropped row block of a float payload")

    def test_an_axis_beyond_the_image_limit_walks_its_rows(self):
        import tensorplay.functional as F
        long = (np.arange(50_001, dtype=np.int64) % 7) - 3
        np.testing.assert_array_equal(
            self._back(F.cumsum(self._device(long), 0)), np.cumsum(long),
            err_msg="a scan row longer than an image axis")
        grid = (np.arange(40_000 * 3, dtype=np.int32) % 11).reshape(40_000, 3)
        for dim in (0, 1):
            np.testing.assert_array_equal(
                self._back(F.cumsum(self._device(grid), dim)),
                np.cumsum(grid.astype(np.int64), axis=dim),
                err_msg=f"wide grid counts at dim={dim}")

    def test_an_empty_count_payload_still_answers_empty(self):
        import tensorplay.functional as F
        for sizes, dtype in (((0, 3), np.int32), ((0,), np.int64),
                             ((0, 2), np.bool_)):
            host = np.zeros(sizes, dtype=dtype)
            got = F.cumsum(self._device(host), 0)
            self.assertEqual(tuple(got.shape), sizes)
            self.assertEqual(got.dtype, tensorplay.int64)
            self.assertEqual(self._back(got).shape, sizes)
        # An axis of zero length with a real axis beside it.
        host = np.zeros((3, 0), dtype=np.int32)
        got = F.cumsum(self._device(host), 1)
        self.assertEqual(tuple(got.shape), (3, 0))

    def test_the_in_place_entry_keeps_the_payload_width(self):
        import tensorplay.functional as F
        host = np.arange(12, dtype=np.int32).reshape(3, 4)
        device = self._device(host)
        F.cumsum_(device, 1)
        self.assertEqual(device.dtype, tensorplay.int32)
        np.testing.assert_array_equal(
            self._back(device),
            np.cumsum(host.astype(np.int64), axis=1).astype(np.int32))
        wide_host = (np.arange(12, dtype=np.int64) * 3000).reshape(3, 4)
        wide = self._device(wide_host)
        F.cumsum_(wide, 0)
        self.assertEqual(wide.dtype, tensorplay.int64)
        np.testing.assert_array_equal(
            self._back(wide), np.cumsum(wide_host, axis=0))

    def test_a_payload_or_result_outside_the_surface_refuses(self):
        import tensorplay.functional as F
        with self.assertRaisesRegex(RuntimeError, "supported Vulkan dtype"):
            F.cumsum(self._device(np.arange(6, dtype=np.float64)), 0)
        with self.assertRaisesRegex(RuntimeError, "supported Vulkan dtype"):
            F.cumsum(self._device(np.arange(6, dtype=np.int16)), 0)
        with self.assertRaisesRegex(RuntimeError, "and Int64 results"):
            F.cumsum(self._device(np.arange(6, dtype=np.int32)), 0,
                     dtype=tensorplay.uint8)
        counts = self._device(np.arange(24, dtype=np.int32).reshape(4, 6))
        for dim in (2, -3):
            with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
                F.cumsum(counts, dim)
        with self.assertRaisesRegex(RuntimeError, "ndim <= 4"):
            F.cumsum(self._device(np.zeros((1, 1, 1, 1, 2), np.int32)), 0)


class TestVulkanDeviceBuiltPayload(unittest.TestCase):
    # Constructing straight on the device fills a row-major buffer, and the
    # shape still fits an image, so the storage a result gets is not decided by
    # the shape alone.  Two failures hide behind that: a kernel that writes the
    # buffer of an image-stored result leaves the image holding whatever the
    # allocator recycled into it, which reads back as plausible but wrong data
    # whenever an earlier dispatch of the same geometry happens to agree; and a
    # kernel that picks its variant from the result's class never reaches its
    # dense build, because an image exists for these shapes.  Every case below
    # therefore runs on a payload small enough to have a texture form.

    @staticmethod
    def _grid(sizes, modulus=29, scale=4.0):
        count = int(np.prod(sizes))
        return ((np.arange(count, dtype=np.float32) % modulus - modulus // 2)
                / scale).reshape(sizes)

    def test_masked_fill_over_a_device_built_payload(self):
        import tensorplay.functional as F
        host = self._grid((2, 3, 4, 5))
        mask = host > 0.0
        want = np.where(mask, 0.5, host)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        for name, condition in (
                ("built", tensorplay.tensor(mask, device="vulkan")),
                ("uploaded", tensorplay.tensor(mask).to("vulkan"))):
            got = F.masked_fill(built, condition, 0.5).cpu().numpy()
            np.testing.assert_array_equal(
                got, want, err_msg=f"masked_fill with a {name} mask")
        # An odd element count leaves the last condition word partly past the
        # logical end; the tail lane still has to select.
        tail = self._grid((37,))
        tail_mask = tail > 0.0
        got = F.masked_fill(
            tensorplay.tensor(tail, device="vulkan"),
            tensorplay.tensor(tail_mask, device="vulkan"),
            -1.0).cpu().numpy()
        np.testing.assert_array_equal(
            got, np.where(tail_mask, -1.0, tail), err_msg="masked_fill tail")

    def test_where_over_a_device_built_payload(self):
        import tensorplay.functional as F
        host = self._grid((2, 3, 4, 5))
        condition = host > 0.0
        other = (np.arange(host.size, dtype=np.float32) % 5 + 1).reshape(
            host.shape)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        want = np.where(condition, host, other)
        for name, cond, rhs in (
                ("built both",
                 tensorplay.tensor(condition, device="vulkan"),
                 tensorplay.tensor(other, device="vulkan")),
                ("uploaded mask", tensorplay.tensor(condition).to("vulkan"),
                 tensorplay.tensor(other, device="vulkan")),
                ("uploaded branch", tensorplay.tensor(condition,
                                                     device="vulkan"),
                 tensorplay.tensor(other).to("vulkan"))):
            got = F.where(cond, built, rhs).cpu().numpy()
            np.testing.assert_array_equal(
                got, want, err_msg=f"where with operands {name}")
        # A half payload packs two lanes per word, so one invocation has to own
        # the whole destination word of the selection.
        half = host.astype(np.float16)
        half_other = other.astype(np.float16)
        half_want = np.where(condition, half, half_other)
        got = F.where(
            tensorplay.tensor(condition, device="vulkan"),
            tensorplay.tensor(half, device="vulkan"),
            tensorplay.tensor(half_other, device="vulkan")).cpu().numpy()
        np.testing.assert_array_equal(
            got.view(np.uint16), half_want.view(np.uint16),
            err_msg="half where over a device-built payload")

    def test_rms_norm_over_a_device_built_payload(self):
        import tensorplay.nn.functional as Fn
        host = self._grid((2, 3, 4, 8), modulus=23, scale=3.0)
        weight = ((np.arange(8, dtype=np.float32) % 5 + 1) / 3.0)
        xf = host.astype(np.float64)
        want = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1.0e-5)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        for name, scaled in (
                ("plain", False),
                ("built weight", True),
                ("uploaded weight", True)):
            w = None
            if scaled:
                w = tensorplay.tensor(
                    weight, device="vulkan") if name == "built weight" \
                    else tensorplay.tensor(weight).to("vulkan")
            expected = want * weight if scaled else want
            got = Fn.rms_norm(built, [8], w, 1.0e-5).cpu().numpy()
            np.testing.assert_allclose(
                got.astype(np.float64), expected, rtol=1e-5, atol=1e-5,
                err_msg=f"rms_norm with a {name}")
        # Half lanes round once when stored, so the reference narrows the same
        # way and the bit patterns compare.
        half = host.astype(np.float16)
        half_weight = weight.astype(np.float16)
        hf = half.astype(np.float32)
        half_want = (hf / np.sqrt(
            (hf ** 2).mean(-1, keepdims=True) + 1.0e-5)
            * half_weight).astype(np.float16)
        got = Fn.rms_norm(
            tensorplay.tensor(half, device="vulkan"), [8],
            tensorplay.tensor(half_weight, device="vulkan"),
            1.0e-5).cpu().numpy()
        np.testing.assert_array_equal(
            got.view(np.uint16), half_want.view(np.uint16),
            err_msg="half rms_norm over a device-built payload")

    def test_rotary_embedding_over_a_device_built_payload(self):
        import tensorplay.functional as F
        for (B, S, D, table_len, off) in (
                (2, 6, 8, 12, 0), (1, 4, 16, 6, 2), (3, 5, 4, 16, 1)):
            host = self._grid((B, S, D), modulus=23, scale=3.0)
            cos, sin = _rope_tables(D, table_len)
            want = _rope_reference(host, cos, sin, off)
            built = tensorplay.tensor(host.copy(), device="vulkan")
            for name, maker in (
                    ("built", lambda a: tensorplay.tensor(a, device="vulkan")),
                    ("uploaded", lambda a: tensorplay.tensor(a).to("vulkan"))):
                got = F.rotary_embedding(
                    built, maker(cos), maker(sin), off).cpu().numpy()
                np.testing.assert_allclose(
                    got, want, rtol=1e-5, atol=1e-5,
                    err_msg=f"rope over a device-built payload with {name} "
                            f"tables at B={B} S={S} D={D} off={off}")

    def test_index_search_over_a_device_built_payload(self):
        import tensorplay.functional as F
        host = self._grid((2, 3, 4, 5), modulus=17, scale=2.0)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        for dim in range(4):
            # An axis shorter than the 256 lanes of a workgroup leaves idle
            # lanes in the search; they must lose, whichever bound they carry.
            np.testing.assert_array_equal(
                F.argmin(built, dim).cpu().numpy(),
                np.argmin(host, axis=dim),
                err_msg=f"argmin over a device-built payload at dim {dim}")
            np.testing.assert_array_equal(
                F.argmax(built, dim).cpu().numpy(),
                np.argmax(host, axis=dim),
                err_msg=f"argmax over a device-built payload at dim {dim}")
        # The first occurrence wins a tie, and a strided search walks several
        # positions per lane before the shared reduction meets them.
        ties = np.full((2, 300), 1.0, dtype=np.float32)
        ties[0, 7] = -2.0
        ties[1, 150] = 3.0
        tie_built = tensorplay.tensor(ties, device="vulkan")
        np.testing.assert_array_equal(
            F.argmin(tie_built, 1).cpu().numpy(), np.argmin(ties, axis=1),
            err_msg="argmin first occurrence")
        np.testing.assert_array_equal(
            F.argmax(tie_built, 1).cpu().numpy(), np.argmax(ties, axis=1),
            err_msg="argmax first occurrence")
        # A flat payload longer than the image axis limit keeps the same rule
        # over one long axis.
        long_axis = (np.arange(70001, dtype=np.float32) % 300 - 150)
        long_built = tensorplay.tensor(long_axis, device="vulkan")
        self.assertEqual(
            int(F.argmin(long_built, 0).cpu().numpy()),
            int(np.argmin(long_axis)))
        self.assertEqual(
            int(F.argmax(long_built, 0).cpu().numpy()),
            int(np.argmax(long_axis)))

    def test_softmax_over_a_device_built_payload(self):
        import tensorplay.functional as F
        host = self._grid((2, 3, 4, 5))
        built = tensorplay.tensor(host.copy(), device="vulkan")
        half_built = tensorplay.tensor(host.astype(np.float16), device="vulkan")
        for dim in range(4):
            row_max = host.max(axis=dim, keepdims=True)
            ex = np.exp(host - row_max)
            total = ex.sum(axis=dim, keepdims=True)
            np.testing.assert_allclose(
                F.softmax(built, dim).cpu().numpy(), ex / total,
                rtol=1e-5, atol=1e-6, err_msg="softmax along axis %d" % dim)
            np.testing.assert_allclose(
                F.softmax(half_built, dim).cpu().numpy().astype(np.float32),
                (ex / total).astype(np.float16).astype(np.float32),
                rtol=2e-3, atol=2e-3,
                err_msg="half softmax along axis %d" % dim)
        row_max = host.max(axis=-1, keepdims=True)
        ex = np.exp(host - row_max)
        np.testing.assert_allclose(
            F.log_softmax(built, -1).cpu().numpy(),
            (host - row_max) - np.log(ex.sum(axis=-1, keepdims=True)),
            rtol=1e-5, atol=1e-6, err_msg="log_softmax")
        channel_max = host.max(axis=1, keepdims=True)
        channel_ex = np.exp(host - channel_max)
        np.testing.assert_allclose(
            F.log_softmax(built, 1).cpu().numpy(),
            (host - channel_max) - np.log(channel_ex.sum(axis=1, keepdims=True)),
            rtol=1e-5, atol=1e-6, err_msg="log_softmax along the channel axis")

    def test_activations_over_a_device_built_payload(self):
        from math import erf
        import tensorplay.functional as F
        host = self._grid((2, 3, 4, 5))
        built = tensorplay.tensor(host.copy(), device="vulkan")
        sigmoid = 1.0 / (1.0 + np.exp(-host))
        cases = [
            ("silu", F.silu, host * sigmoid),
            ("mish", F.mish, host * np.tanh(np.log1p(np.exp(host)))),
            ("relu6", F.relu6, np.clip(host, 0.0, 6.0)),
            ("hardsigmoid", F.hardsigmoid, np.clip(host / 6.0 + 0.5, 0.0, 1.0)),
            ("hardswish", F.hardswish, host * np.clip(host / 6.0 + 0.5, 0.0, 1.0)),
            ("gelu", F.gelu, 0.5 * host * (
                1.0 + np.vectorize(erf)(host / np.sqrt(2.0)))),
            ("gelu_tanh", lambda v: F.gelu(v, "tanh"), 0.5 * host * (
                1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                              (host + 0.044715 * host ** 3)))),
            ("leaky_relu", lambda v: F.leaky_relu(v, 0.1),
             np.where(host > 0.0, host, 0.1 * host)),
            ("threshold", lambda v: F.threshold(v, 0.0, 6.0),
             np.where(host > 0.0, host, 6.0)),
            ("hardshrink", lambda v: F.hardshrink(v, 0.1),
             np.where(np.abs(host) > 0.1, host, 0.0)),
        ]
        for name, op, want in cases:
            np.testing.assert_allclose(
                op(built).cpu().numpy(), want, rtol=1e-5, atol=1e-6,
                err_msg="%s over a device-built payload" % name)
        # A half payload keeps its pointwise formula exact: the same rounding
        # on both sides lands on the same half.
        half_built = tensorplay.tensor(host.astype(np.float16), device="vulkan")
        np.testing.assert_array_equal(
            F.silu(half_built).cpu().numpy().view(np.uint16),
            (host * sigmoid).astype(np.float16).view(np.uint16),
            err_msg="half silu over a device-built payload")
        # In-place keeps the payload dense and updates it where it lies.
        inplace = tensorplay.tensor(host.copy(), device="vulkan")
        F.silu_(inplace)
        np.testing.assert_allclose(
            inplace.cpu().numpy(), host * sigmoid, rtol=1e-5, atol=1e-6,
            err_msg="silu_ in place")

    def test_cat_over_a_device_built_payload(self):
        host = self._grid((2, 3, 4, 5))
        built = tensorplay.tensor(host.copy(), device="vulkan")
        for dim in range(4):
            second_sizes = list(host.shape)
            second_sizes[dim] += 2
            second = self._grid(tuple(second_sizes), modulus=13, scale=8.0)
            np.testing.assert_allclose(
                tensorplay.cat([built, tensorplay.tensor(second.copy(),
                                                          device="vulkan")],
                               dim).cpu().numpy(),
                np.concatenate([host, second], axis=dim),
                err_msg="cat along axis %d" % dim)
        # One image operand alongside a dense one: the dense class wins, and
        # the texel plane is pulled into words on the way through.
        uploaded = tensorplay.from_numpy(host.copy()).to("vulkan")
        np.testing.assert_allclose(
            tensorplay.cat([built, uploaded], -1).cpu().numpy(),
            np.concatenate([host, host], axis=-1),
            err_msg="cat of a dense payload with an uploaded one")
        np.testing.assert_allclose(
            tensorplay.cat([uploaded, built], 0).cpu().numpy(),
            np.concatenate([host, host], axis=0),
            err_msg="cat with the uploaded payload first")
        half = host.astype(np.float16)
        half_built = tensorplay.tensor(half, device="vulkan")
        np.testing.assert_array_equal(
            tensorplay.cat([half_built, half_built], 1).cpu().numpy()
            .view(np.uint16),
            np.concatenate([half, half], axis=1).view(np.uint16),
            err_msg="half cat over device-built payloads")

    def test_batch_norm_over_a_device_built_payload(self):
        host = self._grid((2, 3, 4, 5))
        channels = host.shape[1]
        shape = (1, channels, 1, 1)
        weight = ((np.arange(channels, dtype=np.float32) + 1) / 4).reshape(shape)
        bias = ((np.arange(channels, dtype=np.float32) - channels // 2)
                / 8).reshape(shape)
        mean = ((np.arange(channels, dtype=np.float32) % 5 - 2)
                / 4).reshape(shape)
        var = ((np.arange(channels, dtype=np.float32) % 7 + 1) / 2).reshape(shape)
        built = tensorplay.tensor(host.copy(), device="vulkan")
        want = (host - mean) / np.sqrt(var + 1e-5)

        def as_device(array):
            if array is None:
                return None
            return (tensorplay.tensor(array.copy(), device="vulkan")
                    if use_built else
                    tensorplay.from_numpy(array.copy()).to("vulkan"))

        for use_built in (True, False):
            for with_weight, with_bias, expected in (
                    (True, True, want * weight + bias),
                    (True, False, want * weight),
                    (False, True, want + bias),
                    (False, False, want)):
                out = tensorplay.batch_norm(
                    built,
                    as_device(weight if with_weight else None),
                    as_device(bias if with_bias else None),
                    as_device(mean), as_device(var),
                    False, 0.1, 1e-5)
                np.testing.assert_allclose(
                    out.cpu().numpy(), expected, rtol=1e-5, atol=1e-6,
                    err_msg="batch_norm built=%s weight=%s bias=%s" % (
                        use_built, with_weight, with_bias))

    @staticmethod
    def _uses_dense_storage(payload):
        # Only single precision has both a texture and a linear form of this
        # kernel, so a lerp that reports the missing image marks a payload that
        # keeps the elements in words.
        import tensorplay.functional as F
        try:
            F.lerp(payload, payload, 0.5)
            return False
        except RuntimeError:
            return True

    def test_clone_keeps_the_storage_of_its_model(self):
        host = self._grid((2, 3, 4, 5))
        for name, source, want_dense in (
                ("built",
                 tensorplay.tensor(host.copy(), device="vulkan"), True),
                ("uploaded", tensorplay.from_numpy(host.copy()).to("vulkan"),
                 False)):
            out = source.clone()
            self.assertIs(
                self._uses_dense_storage(out), want_dense,
                f"clone of a {name} payload changed its storage class")
            np.testing.assert_array_equal(
                out.cpu().numpy(), host,
                err_msg=f"clone of a {name} payload lost its values")

        # An integer payload rounds nowhere, so a copy that lands in the wrong
        # binding shows up as wrong numbers rather than as a rounding error.
        ints = ((np.arange(40, dtype=np.int32) % 17) - 8).reshape(2, 5, 4)
        built = tensorplay.tensor(ints.copy(), device="vulkan")
        np.testing.assert_array_equal(
            built.clone().cpu().numpy(), ints,
            err_msg="clone of an integer payload lost its values")

    def test_conversion_from_a_device_built_payload(self):
        host = self._grid((2, 3, 4, 5))
        half = host.astype(np.float16)
        ints = ((np.arange(40, dtype=np.int32) % 17) - 8).reshape(2, 5, 4)
        for name, source, half_source, int_source in (
                ("built",
                 tensorplay.tensor(host.copy(), device="vulkan"),
                 tensorplay.tensor(half.copy(), device="vulkan"),
                 tensorplay.tensor(ints.copy(), device="vulkan")),
                ("uploaded", tensorplay.from_numpy(host.copy()).to("vulkan"),
                 tensorplay.from_numpy(half.copy()).to("vulkan"),
                 tensorplay.from_numpy(ints.copy()).to("vulkan"))):
            np.testing.assert_array_equal(
                source.to(tensorplay.float16).cpu().numpy(), half,
                err_msg=f"single to half over a {name} payload")
            np.testing.assert_array_equal(
                half_source.to(tensorplay.float32).cpu().numpy(),
                half.astype(np.float32),
                err_msg=f"half to single over a {name} payload")
            np.testing.assert_array_equal(
                int_source.to(tensorplay.float32).cpu().numpy(),
                ints.astype(np.float32),
                err_msg=f"integer to single over a {name} payload")

    def test_copy_between_storage_classes(self):
        host = self._grid((2, 3, 4, 5))
        built = tensorplay.tensor(host.copy(), device="vulkan")
        uploaded = tensorplay.from_numpy(host.copy()).to("vulkan")
        for name, destination, source, want_dense in (
                ("dense destination", built, uploaded, True),
                ("image destination", uploaded, built, False)):
            out = tensorplay.empty_like(destination)
            self.assertIs(
                self._uses_dense_storage(out), want_dense,
                f"empty_like did not follow the {name}")
            out.copy_(source)
            np.testing.assert_array_equal(
                out.cpu().numpy(), host,
                err_msg=f"copy_ into the {name} lost its values")

    def test_copy_between_storages_needs_matching_shapes(self):
        # The crossing builds its carrier from the source and the transfer that
        # follows runs over the destination's extents, so payloads that only
        # share an element count have to be refused rather than read past the
        # payload just built.
        host = self._grid((2, 3, 4, 5))
        flat = tensorplay.tensor(host.reshape(-1).copy(), device="vulkan")
        cube = tensorplay.from_numpy(host.copy()).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "equal shapes"):
            flat.copy_(cube)


class TestVulkanWidePayload(unittest.TestCase):
    # A texel carries at most four 32-bit channels, so an element wider than one
    # word has no image form on any device and keeps the plain linear layout
    # whatever its shape.  That is what lets a 64-bit index vector reach the
    # device instead of being narrowed to 32 bits on the host, which costs a
    # queue sync per call.  An op that moves a payload keeps to the pair of words
    # one element is; an op that folds lanes into a total still refuses, because
    # no 32-bit accumulator can hold the answer.  A range resolves to the widest
    # code that both routes read.

    @staticmethod
    def _weight(sizes=(16, 6)):
        count = int(np.prod(sizes))
        return ((np.arange(count, dtype=np.float32) % 13) / 4).reshape(sizes)

    def test_upload_and_readback_keep_every_bit(self):
        # Codes that differ only in the word above them are the case worth
        # pinning: a route that folded a pair of words into one, or narrowed on
        # the way in, still reads back as a plausible sequence.
        values = np.array(
            [0, -1, 1, 2**31 - 1, -(2**31), 2**32, 2**53, -(2**53)],
            dtype=np.int64)
        for name, payload in (
                ("uploaded",
                 tensorplay.from_numpy(values.copy()).to("vulkan")),
                ("built", tensorplay.tensor(values, device="vulkan"))):
            self.assertEqual(payload.dtype, tensorplay.int64, name)
            np.testing.assert_array_equal(
                payload.cpu().numpy(), values,
                err_msg=f"{name} 64-bit payload changed its codes")

    def test_factories_emit_wide_sequences(self):
        cases = (
            ("zeros", tensorplay.zeros, np.zeros(9, dtype=np.int64)),
            ("ones", tensorplay.ones, np.ones(9, dtype=np.int64)),
            ("full", lambda shape, **kw: tensorplay.full(shape, 7, **kw),
             np.full(9, 7, dtype=np.int64)),
        )
        for name, make, want in cases:
            out = make((9,), dtype=tensorplay.int64, device="vulkan")
            self.assertEqual(out.dtype, tensorplay.int64, name)
            np.testing.assert_array_equal(
                out.cpu().numpy(), want, err_msg=f"{name} of a wide dtype")
        # An odd count leaves the last 16-byte group of the staging transfer
        # partly past the logical end, so the tail element is the one a
        # group-at-a-time move drops.
        tail = tensorplay.full((17,), -3, dtype=tensorplay.int64,
                               device="vulkan")
        np.testing.assert_array_equal(
            tail.cpu().numpy(), np.full(17, -3, dtype=np.int64),
            err_msg="wide fill lost its tail")

    def test_empty_of_a_wide_dtype_receives_a_copy(self):
        values = np.arange(23, dtype=np.int64) * 4_000_000_000
        out = tensorplay.empty((23,), dtype=tensorplay.int64,
                               device="vulkan")
        out.copy_(tensorplay.tensor(values.copy(), device="vulkan"))
        np.testing.assert_array_equal(
            out.cpu().numpy(), values, err_msg="empty_ + copy_ of a wide dtype")

    def test_index_vector_drives_the_row_lookups(self):
        # The whole point of carrying the codes: a row lookup reads its indices
        # on the device.  Both weight storages have to accept them, because the
        # index payload is linear either way.
        ids = np.array([0, 7, 3, 11, 2], dtype=np.int64)
        host = self._weight()
        want = host[ids]
        indices = tensorplay.from_numpy(ids.copy()).to("vulkan")
        for name, weight in (
                ("uploaded", tensorplay.from_numpy(host.copy()).to("vulkan")),
                ("built", tensorplay.tensor(host.copy(), device="vulkan"))):
            np.testing.assert_allclose(
                tensorplay.index_select(weight, 0, indices).cpu().numpy(), want,
                rtol=0, atol=1e-6, err_msg=f"index_select over a {name} weight")
            np.testing.assert_allclose(
                tensorplay.embedding(weight, indices).cpu().numpy(), want,
                rtol=0, atol=1e-6, err_msg=f"embedding over a {name} weight")

    def test_a_range_resolves_to_the_widest_readable_code(self):
        # Nothing downstream of a range reads a pair of words per element yet,
        # so an index vector built on the device arrives as Int32.  Pinning the
        # resolution keeps a future wide gather honest about what it must also
        # switch over.
        for kwargs in ({"dtype": tensorplay.int64}, {}):
            got = tensorplay.arange(0, 40, 2, device="vulkan", **kwargs)
            self.assertEqual(got.dtype, tensorplay.int32)
            np.testing.assert_array_equal(
                got.cpu().numpy(), np.arange(0, 40, 2, dtype=np.int32))

    def test_wide_payloads_convert_on_the_device(self):
        # A wide code is narrowed a pair of words at a time, so the element the
        # destination names decides the route: an image destination cannot hold
        # one, so the codes are converted while both payloads are still linear
        # and only then carried across as the destination's own type.
        codes = np.array(
            [0, 1, -1, 5, 2**31 - 1, -(2**31), 2**32, -(2**32), 2**40, -12345],
            dtype=np.int64)
        payload = tensorplay.from_numpy(codes.copy()).to("vulkan")
        cases = (
            ("narrowed", payload.to(tensorplay.int32),
             codes.astype(np.int32).astype(np.int64)),
            ("single precision", payload.to(tensorplay.float32),
             codes.astype(np.float32).astype(np.int64)),
        )
        for name, got, want in cases:
            np.testing.assert_array_equal(
                got.cpu().numpy().astype(np.int64), want,
                err_msg=f"{name} cast of 64-bit codes")
        # Int32 codes widen by repeating the sign bit into the word above them.
        widened = tensorplay.tensor(
            codes.astype(np.int32).copy(), device="vulkan").to(
                tensorplay.int64)
        np.testing.assert_array_equal(
            widened.cpu().numpy(), codes.astype(np.int32).astype(np.int64),
            err_msg="widened codes lost their sign")
        # An odd element count ends a wide payload on a 16-byte boundary that a
        # four-element group does not cover: the tail codes still convert.
        tail = np.array([7, -8, 9], dtype=np.int64)
        np.testing.assert_array_equal(
            tensorplay.from_numpy(tail.copy()).to("vulkan").to(
                tensorplay.float32).cpu().numpy(),
            tail.astype(np.float32), err_msg="wide conversion lost its tail")

    @staticmethod
    def _released_as_texels(payload):
        # Whether a payload has an image or a buffer to bind decides which kernels
        # can take it: a texel kernel refuses a dense operand outright.
        import tensorplay.functional as F
        try:
            F.lerp(payload, payload, 0.5)
        except RuntimeError:
            return False
        return True

    def test_a_device_index_list_stays_on_the_device(self):
        # The index list is built in whichever class the lookup already uses, so
        # a table with an image form keeps its texel route and a table without one
        # keeps its flat route: neither drags the other's payload across the
        # boundary because the codes arrived 64-bit wide.
        rows, feats = 40000, 64
        host_table = (
            (np.arange(rows, dtype=np.float32) % 9).reshape(rows, 1)
            * np.ones((1, feats), np.float32)).astype(np.float32)
        # Every code is a row of the short table too, so one reference serves
        # both classes.
        ids = np.array([0, 17, 255, 3, 128], dtype=np.int64)
        lists = (
            ("64-bit", tensorplay.from_numpy(ids.copy()).to("vulkan")),
            ("32-bit",
             tensorplay.from_numpy(ids.astype(np.int32)).to("vulkan")),
            ("host", tensorplay.from_numpy(ids.copy())),
        )
        for name, used in (("texel", 256), ("dense", rows)):
            table = tensorplay.from_numpy(
                host_table[:used].copy()).to("vulkan")
            self.assertIs(
                self._released_as_texels(table), name == "texel",
                f"the {name} fixture is not the class it names")
            want = host_table[:used][ids]
            classes = []
            for lname, index in lists:
                out = tensorplay.embedding(table, index)
                np.testing.assert_allclose(
                    out.cpu().numpy(), want, rtol=0, atol=1e-6,
                    err_msg=f"embedding with a {lname} index list")
                classes.append(self._released_as_texels(out))
            self.assertEqual(
                classes, [classes[0]] * 3,
                f"an index list routed the {name} lookup differently")
            # Whichever route read the table, the rows come back in an image: a
            # consumer picks its route from what it was handed, and a lookup
            # result is usually narrow enough that a texture-only kernel is the
            # only one that can read it at all.
            self.assertEqual(
                classes, [True] * 3,
                f"a {name} table released its rows outside an image")

    def test_wide_payloads_stop_at_the_walk_that_folds_a_lane_into_a_word(self):
        values = np.array([0, 5, 2, 7, 1], dtype=np.int64)
        payload = tensorplay.from_numpy(values.copy()).to("vulkan")
        # A total of wide lanes has no single word to accumulate in, so it walks
        # the pair of words each lane is carried in and answers whole; a fold
        # that read one word per element would have added up the low halves.
        total = payload.sum()
        self.assertEqual(str(total.dtype), "tensorplay.int64")
        self.assertEqual(int(total.cpu().numpy()), int(values.sum()),
                         "a total of wide lanes arrived truncated")
        # A half payload packs two lanes into one word, so a wide code would need
        # three of them and the two storages share no word boundary.  The hop
        # through single precision cannot answer either: it drops the bits above
        # the low word the half narrows, so the pair names itself in the refusal.
        with self.assertRaisesRegex(
                RuntimeError, "Unsupported Vulkan dtype conversion from Int64"):
            payload.to(tensorplay.float16)

    def test_views_of_a_wide_payload_move_whole_codes(self):
        # Every one of these reads a code out of linear storage and writes it
        # somewhere else in the same layout.  Before a wide element moved as the
        # pair of words it is, a stride that skipped one skipped only its low
        # word, and a code with nothing above the truncation looked correct.
        base = (np.arange(30, dtype=np.int64) << 41) - 7
        shape = (2, 3, 5)
        payload = tensorplay.from_numpy(base.reshape(shape).copy()).to(
            "vulkan")
        cases = (
            ("slice", payload[1], base.reshape(shape)[1]),
            ("narrow", tensorplay.narrow(payload, 1, 1, 2),
             base.reshape(shape)[:, 1:3, :]),
            ("select", tensorplay.select(payload, 0, 0),
             base.reshape(shape)[0]),
            ("permute", tensorplay.permute(payload, (2, 0, 1)),
             np.transpose(base.reshape(shape), (2, 0, 1))),
            ("transpose", tensorplay.transpose(payload, 0, 2),
             np.transpose(base.reshape(shape), (2, 1, 0))),
            ("view", payload.view(3, 10), base.reshape(3, 10)),
            ("flatten", payload.view(-1), base),
        )
        for name, got, want in cases:
            np.testing.assert_array_equal(
                got.cpu().numpy(), want, err_msg=f"{name} of a wide payload")
            self.assertEqual(got.dtype, tensorplay.int64, name)

    def test_a_strided_and_broadcast_view_of_a_wide_payload_keeps_its_codes(
            self):
        # Neither of these reads contiguously: the first names an element by
        # offset and stride, the second names one element from several positions.
        # Both answer from the identity gather, so a code has to arrive whole at
        # every position that repeats it.
        base = np.arange(30, dtype=np.int64) * 4_000_000_000 - 3
        payload = tensorplay.from_numpy(base.copy()).to("vulkan").view(2, 3, 5)
        np.testing.assert_array_equal(
            payload.as_strided((2, 2, 2), (5, 15, 1), 1).cpu().numpy(),
            np.array([[[base[1 + 5 * a + 15 * b + c] for c in range(2)]
                       for b in range(2)] for a in range(2)]),
            err_msg="strided view of a wide payload")
        row = tensorplay.from_numpy(base[:5].copy()).to("vulkan")
        np.testing.assert_array_equal(
            row.expand(4, 5).cpu().numpy(),
            np.broadcast_to(base[:5], (4, 5)),
            err_msg="broadcast view of a wide payload")

    def test_narrow_codes_drive_a_lookup_of_a_wide_index_payload(self):
        # The reason the codes stay 64-bit: a decode step takes the codes it still
        # has to read off the end of its index list, which is a stride that skips a
        # whole code.  A lookup whose indices are such a view reads each code from
        # the pair of words it is.
        host = self._weight((16, 6))
        ids = np.arange(24, dtype=np.int64) % 16
        indices = tensorplay.from_numpy(ids.copy()).to("vulkan")
        weight = tensorplay.from_numpy(host.copy()).to("vulkan")
        cases = (
            ("tail", tensorplay.narrow(indices, 0, 20, 4), ids[20:]),
            ("head", indices[:3], ids[:3]),
            ("every second", indices[::2], ids[::2]),
            ("single", tensorplay.select(indices, 0, 7).view(1), ids[7:8]),
        )
        for name, used, want in cases:
            picked = tensorplay.index_select(weight, 0, used)
            np.testing.assert_allclose(
                picked.cpu().numpy(), host[want], rtol=0, atol=1e-6,
                err_msg=f"index_select with a {name} wide index payload")
            np.testing.assert_allclose(
                tensorplay.embedding(weight, used).cpu().numpy(), host[want],
                rtol=0, atol=1e-6,
                err_msg=f"embedding with a {name} wide index payload")

    def test_a_wide_index_payload_gathers_whole_codes(self):
        # The gathered axis of the index has to name the axis it replaces, so a
        # table of six rows is picked at six positions per row: each code is read
        # from the pair of words it is, and the row it names is a Float32 one.
        host = self._weight((6, 8))
        ids = (np.arange(18, dtype=np.int64) % 6).reshape(3, 6)
        indices = tensorplay.from_numpy(ids.copy()).to("vulkan")
        table = tensorplay.from_numpy(host.copy()).to("vulkan")
        coords = np.indices(ids.shape)
        coords[1] = ids
        np.testing.assert_allclose(
            tensorplay.gather(table, 1, indices).cpu().numpy(),
            host[tuple(coords)], rtol=0, atol=1e-6,
            err_msg="gather with a wide index payload")
        coords = np.indices(ids.shape)
        coords[0] = ids
        np.testing.assert_allclose(
            tensorplay.gather(table, 0, indices).cpu().numpy(),
            host[tuple(coords)], rtol=0, atol=1e-6,
            err_msg="gather along the other axis with a wide index payload")

    def test_a_single_precision_payload_widens_to_the_whole_it_holds(self):
        # Truncation is toward zero, and a magnitude too large for 64 bits keeps
        # the endpoint of its own sign rather than wrapping into the other.  A
        # float has twenty-four significant bits, so splitting the magnitude at
        # two to the thirty-second is exact: no rounding is hidden in the pair.
        values = np.array(
            [0.0, 1.5, -2.25, 2**31, -(2**31), 2**32, -(2**32), 2**53,
             4.5e18, -4.5e18],
            dtype=np.float32)
        got = tensorplay.tensor(values.copy(), device="vulkan").to(
            tensorplay.int64)
        want = np.array([int(float(v)) for v in values], dtype=np.int64)
        want = np.where(values >= 2**63, 2**63 - 1, want)
        want = np.where(values < -2**63, -2**63, want)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want, err_msg="widened single precision codes")
        # A value past the endpoint of its sign, and a value with no whole part at
        # all, are the two answers no arithmetic of the magnitude can give.
        edges = np.array(
            [float("inf"), -float("inf"), float("nan"), 9.3e18, -9.3e18],
            dtype=np.float32)
        np.testing.assert_array_equal(
            tensorplay.tensor(edges.copy(), device="vulkan").to(
                tensorplay.int64).cpu().numpy(),
            np.array([2**63 - 1, -(2**63), 0, 2**63 - 1, -(2**63)],
                     dtype=np.int64),
            err_msg="widened codes past the endpoint")

    def test_a_wide_payload_widens_or_narrows_its_tail_whole(self):
        # Four lanes share the two 16-byte groups a group-at-a-time move covers,
        # so a payload of any other length ends inside a group that the source or
        # the destination allocation still holds.  The codes in that group are the
        # ones a loop that assumes a multiple of four drops or writes over.
        for count in (1, 3, 5, 7, 9, 14, 17):
            wide = (np.arange(count, dtype=np.int64) << 41) - 1
            # Codes a 32-bit word, and a single-precision value, both hold whole.
            exact = np.arange(count, dtype=np.int32) * 100_003 - 500_000
            device_wide = tensorplay.from_numpy(wide.copy()).to("vulkan")
            np.testing.assert_array_equal(
                device_wide.to(tensorplay.int32).cpu().numpy(),
                wide.astype(np.int32),
                err_msg=f"narrowed wide payload of {count} codes")
            device_exact = tensorplay.from_numpy(exact.copy()).to("vulkan")
            np.testing.assert_array_equal(
                device_exact.to(tensorplay.int64).cpu().numpy(),
                exact.astype(np.int64),
                err_msg=f"widened payload of {count} codes")
            np.testing.assert_array_equal(
                device_exact.to(tensorplay.float32).to(
                    tensorplay.int64).cpu().numpy(),
                exact.astype(np.int64),
                err_msg=f"single precision payload of {count} codes")


class TestVulkanGather(unittest.TestCase):
    # A gather reads one code per output element, so the output shape is the index
    # shape and the source keeps its own.  Which route runs depends on where the
    # gathered axis falls in the texel layout: the two innermost axes are the width
    # and the height, along which a code moves a fetch, while the channel and batch
    # axes live in the lanes and the depth planes and gather over linear storage.

    @staticmethod
    def _reference(table_np, dim, codes_np):
        # Every output coordinate is the index's own except the gathered one,
        # which the code at that position names.
        coords = np.indices(codes_np.shape)
        coords[dim] = codes_np
        return table_np[tuple(coords)]

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _table(sizes, dtype=np.float32):
        r = np.arange(int(np.prod(sizes)), dtype=np.float32).reshape(sizes)
        return (r % 97 - 48).astype(dtype)

    def test_gather_follows_the_code_at_each_position(self):
        # The shapes a decode step gathers through: a per-row pick out of a logits
        # table, a pick of whole rows, and a pick out of a flat payload.
        rng = np.random.default_rng(11)
        cases = [
            (24, 40, 1, 6),            # [rows, vocabulary] -> [rows, 6]
            (24, 40, 0, 7),            # gathered along the height axis
            (129, 5, 1, 3),            # a row width that is not a multiple of four
            (60, 1, 1, 17),            # rank 1, gathered along the only axis
        ]
        for rows, cols, dim, picks in cases:
            table_np = self._table((rows, cols) if cols > 1 else (rows,))
            if cols == 1:
                dim = 0
            shape = list(table_np.shape)
            shape[dim] = picks
            codes = rng.integers(0, table_np.shape[dim], shape).astype(np.int64)
            want = self._reference(table_np, dim, codes)
            got = tensorplay.gather(
                self._device(table_np), dim, self._device(codes)).cpu().numpy()
            np.testing.assert_allclose(
                got, want, rtol=0, atol=1e-6,
                err_msg=f"gather of a {table_np.shape} table along axis {dim}")

    def test_a_lane_axis_gathers_over_linear_storage(self):
        # The channel and batch axes are the lanes and the depth planes of a
        # texel, so a code there would have to move a value between lanes of
        # different texels: those axes cross into a dense payload and back.
        rng = np.random.default_rng(12)
        table_np = self._table((3, 5, 4, 6))
        for dim in (0, 1, 2, 3):
            shape = list(table_np.shape)
            shape[dim] = 2
            codes = rng.integers(0, table_np.shape[dim], shape).astype(np.int32)
            want = self._reference(table_np, dim, codes)
            got = tensorplay.gather(
                self._device(table_np), dim, self._device(codes)).cpu().numpy()
            np.testing.assert_allclose(
                got, want, rtol=0, atol=1e-6,
                err_msg=f"gather of a 4-d table along axis {dim}")

    def test_an_index_narrower_than_the_source_names_its_own_planes(self):
        # The depth axis stacks each payload's own channel groups, so a plane of an
        # output with fewer channels than the source sits at a different depth in
        # the source: reusing the output's plane number would read another channel
        # of an earlier batch and still return plausible values.
        rng = np.random.default_rng(13)
        table_np = self._table((2, 5, 3, 4))
        for channels, dim in ((2, 3), (1, 2), (5, 3), (2, 2), (3, 3)):
            shape = [2, channels, 3, 4]
            shape[dim] = 2
            codes = rng.integers(
                0, table_np.shape[dim], shape).astype(np.int32)
            want = self._reference(table_np, dim, codes)
            got = tensorplay.gather(
                self._device(table_np), dim, self._device(codes)).cpu().numpy()
            self.assertEqual(tuple(got.shape), tuple(shape))
            np.testing.assert_allclose(
                got, want, rtol=0, atol=1e-6,
                err_msg=f"index with {channels} channels gathered along axis "
                        f"{dim}")

    def test_half_gather_keeps_every_bit(self):
        # A half payload packs two lanes into a word, so an odd element count
        # leaves the last word shared with nothing: it must still land whole.
        rng = np.random.default_rng(14)
        table_np = self._table((3, 5), dtype=np.float16)
        codes = rng.integers(0, 5, (3, 3)).astype(np.int64)
        want = self._reference(table_np, 1, codes)
        got = tensorplay.gather(
            self._device(table_np), 1, self._device(codes)).cpu().numpy()
        np.testing.assert_array_equal(
            got.view(np.uint16), want.view(np.uint16),
            err_msg="texel gather of a half payload")
        # A 3-d table gathered along its batch axis crosses into linear storage
        # with an odd element count.
        table_np = self._table((2, 3, 5), dtype=np.float16)
        codes = rng.integers(0, 2, (1, 3, 5)).astype(np.int64)
        want = self._reference(table_np, 0, codes)
        got = tensorplay.gather(
            self._device(table_np), 0, self._device(codes)).cpu().numpy()
        np.testing.assert_array_equal(
            got.view(np.uint16), want.view(np.uint16),
            err_msg="linear gather of an odd half payload")

    def test_an_out_of_range_code_contributes_a_zero(self):
        # The answer the other row lookups of this backend already give: a code
        # naming no row writes zeros rather than reading outside the payload, so a
        # bad pick never has to be caught on the host.
        def expected(table_np, dim, codes):
            valid = (codes >= 0) & (codes < table_np.shape[dim])
            ref = self._reference(table_np, dim, np.where(valid, codes, 0))
            return np.where(valid, ref, 0)

        table_np = self._table((4, 6))
        codes = np.array(
            [[0, -1, 3, 6], [1, 2, -7, 4], [5, 5, 9, 0], [-1, -1, 2, 7]],
            dtype=np.int32)
        got = tensorplay.gather(
            self._device(table_np), 1, self._device(codes)).cpu().numpy()
        np.testing.assert_array_equal(
            got, expected(table_np, 1, codes),
            err_msg="texel gather of an out-of-range code")

        rng = np.random.default_rng(15)
        table_np = self._table((2, 5, 3, 4))
        codes = rng.integers(-6, 9, (2, 3, 3, 4)).astype(np.int32)
        got = tensorplay.gather(
            self._device(table_np), 1, self._device(codes)).cpu().numpy()
        np.testing.assert_array_equal(
            got, expected(table_np, 1, codes),
            err_msg="linear gather of an out-of-range code")

    def test_gather_stops_at_an_index_it_cannot_address(self):
        table = self._device(self._table((4, 6)))
        with self.assertRaisesRegex(RuntimeError, "same rank"):
            tensorplay.gather(table, 1, self._device(np.zeros((3,), np.int64)))
        # Along the gathered axis the index may be any width - that width is the
        # output.  Along another axis it may only be narrower.
        self.assertEqual(
            tuple(tensorplay.gather(
                table, 1, self._device(np.zeros((4, 9), np.int64))).shape),
            (4, 9))
        with self.assertRaisesRegex(RuntimeError, "wider than the source"):
            tensorplay.gather(
                table, 1, self._device(np.zeros((5, 3), np.int64)))
        with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
            tensorplay.gather(table, 2, self._device(np.zeros((4, 6), np.int64)))
        # An index with no codes has no output to write; the result keeps the
        # index shape and stays empty.
        empty = tensorplay.gather(
            table, 1, self._device(np.zeros((4, 0), np.int64)))
        self.assertEqual(tuple(empty.shape), (4, 0))


class TestVulkanFencedReadback(unittest.TestCase):
    # The transfer that carries a payload to the host waits on a fence of its
    # own.  A queue starts a batch only after the one before it has finished, so
    # that fence also covers every dispatch recorded into the same command batch
    # beforehand, and it is the only evidence a readback needs to hand the
    # command and descriptor pools back for reuse.

    def test_a_compute_step_and_a_readback_share_the_fence(self):
        import tensorplay.functional as F
        # Each round records a dispatch that no one submits on its own, then a
        # readback that ends and hands over the batch.  The payload is freshly
        # built every round, so the retired one waits in the clear list while
        # the next transfer recycles the pools it was recorded against.
        rounds = 120
        acc = tensorplay.zeros((37,), device="vulkan")
        step = tensorplay.ones((37,), device="vulkan")
        for i in range(1, rounds + 1):
            acc = F.add(acc, step)
            np.testing.assert_array_equal(
                acc.cpu().numpy(), np.full((37,), float(i), dtype=np.float32),
                err_msg=f"image readback at round {i} saw a stale payload")

    def test_an_upload_and_a_readback_share_the_fence(self):
        # A wide element has no image form on any device, so this round trips
        # the other fenced transfer: one that streams a linear payload into a
        # host-visible buffer with a copy command.
        rounds = 120
        shape = (41,)
        acc = tensorplay.zeros(shape, dtype=tensorplay.int64, device="vulkan")
        for i in range(1, rounds + 1):
            acc.copy_(tensorplay.from_numpy(np.full(shape, i, dtype=np.int64)))
            np.testing.assert_array_equal(
                acc.cpu().numpy(), np.full(shape, i, dtype=np.int64),
                err_msg=f"linear readback at round {i} saw a stale payload")


class TestVulkanMaskLogic(unittest.TestCase):
    # A mask is combined before it reaches attention: a padding row ands a
    # causal triangle, an inverse padding mask negates a row, and a numeric one
    # enters the connectives through the test against zero.  A Bool payload
    # carries one byte per element and counts any nonzero byte as true, so the
    # connectives are arithmetic over reduced lanes.  Both storages hold the
    # same decision: a texel reads four lanes of one plane, a linear word packs
    # four bytes and one invocation owns the whole word it writes.

    SHAPES = (
        ((2, 3, 5, 7), (2, 3, 5, 7)),
        ((4, 6, 8), (6, 8)),
        ((2, 3, 4, 5), (1, 1, 4, 5)),
        ((4, 8), (8,)),
        ((3, 1, 6), (3, 5, 6)),
        ((2, 4, 3, 5), (2, 1, 3, 5)),
        ((1, 1, 1), (2, 3, 5, 7)),
    )
    LENGTHS = (100000, 100001, 100003)

    def _bools(self, shape, seed):
        rng = np.random.default_rng(seed)
        return tensorplay.tensor(rng.random(shape) > 0.5)

    def _compare(self, name, host_a, host_b=None):
        import tensorplay.functional as F
        op = getattr(F, name)
        if host_b is None:
            got = op(host_a.to("vulkan")).cpu().numpy()
            want = op(host_a).numpy()
        else:
            got = op(host_a.to("vulkan"), host_b.to("vulkan")).cpu().numpy()
            want = op(host_a, host_b).numpy()
        self.assertEqual(got.dtype, np.bool_, f"{name} lost the Bool result")
        np.testing.assert_array_equal(
            got.astype(np.int8), want.astype(np.int8),
            err_msg=f"{name} over {tuple(want.shape)}")

    def test_bitwise_connectives_broadcast_over_planes(self):
        for index, (shape_a, shape_b) in enumerate(self.SHAPES):
            a = self._bools(shape_a, index)
            b = self._bools(shape_b, 100 + index)
            for name in ("bitwise_and", "bitwise_or", "bitwise_xor"):
                self._compare(name, a, b)
            self._compare("bitwise_not", a)

    def test_logical_connectives_broadcast_over_planes(self):
        for index, (shape_a, shape_b) in enumerate(self.SHAPES):
            a = self._bools(shape_a, index)
            b = self._bools(shape_b, 100 + index)
            for name in ("logical_and", "logical_or", "logical_xor"):
                self._compare(name, a, b)
            self._compare("logical_not", a)

    def test_operators_match_the_functions(self):
        # The dunder forms reach the same kernels, so an inverted mask written
        # as `~mask` behaves as the function call does.
        a = self._bools((3, 4, 5), 3)
        b = self._bools((3, 4, 5), 4)
        va, vb = a.to("vulkan"), b.to("vulkan")
        for got, want in ((va & vb, a & b), (va | vb, a | b),
                          (va ^ vb, a ^ b), (~va, ~a)):
            np.testing.assert_array_equal(
                got.cpu().numpy().astype(np.int8),
                want.numpy().astype(np.int8))

    def test_linear_payloads_pack_four_lanes_per_word(self):
        # A row longer than the image axis limit lives in a buffer.  The last
        # word of a length that is not a multiple of four is only partly real,
        # and its neighbour plane may be a texture that had to be copied in, so
        # every length carries a broadcast case as well.
        for length in self.LENGTHS:
            a = self._bools((length,), length)
            b = self._bools((length,), length + 1)
            one = self._bools((1,), length + 2)
            for name in ("bitwise_and", "bitwise_or", "bitwise_xor",
                         "logical_and", "logical_xor"):
                self._compare(name, a, b)
            self._compare("bitwise_not", a)
            self._compare("logical_not", a)
            self._compare("bitwise_and", a, one)
            self._compare("bitwise_or", one, b)
            self._compare("logical_xor", a, one)

    def test_numeric_operands_enter_through_the_test_against_zero(self):
        # A logical connective decides by nonzero in the operand's own domain,
        # so a float, half, int or byte plane is first reduced to Bool.  A lane
        # that is not a number counts as true, and two of them cancel in an
        # exclusive or.
        rng = np.random.default_rng(21)
        cases = {
            tensorplay.float32: (rng.standard_normal((2, 3, 4)) * 2.0).astype(
                np.float32),
            tensorplay.float16: (rng.standard_normal((2, 3, 4)) * 2.0).astype(
                np.float16),
            tensorplay.int32: rng.integers(-3, 4, (2, 3, 4)).astype(np.int32),
            tensorplay.uint8: rng.integers(0, 4, (2, 3, 4)).astype(np.uint8),
        }
        for dtype, values in cases.items():
            a = tensorplay.tensor(values, dtype=dtype)
            b = tensorplay.tensor(values[::-1].copy(), dtype=dtype)
            for name in ("logical_and", "logical_or", "logical_xor"):
                self._compare(name, a, b)
            self._compare("logical_not", a)
        lanes = [np.nan, -0.0, 0.0, 1.0, np.inf, np.nan]
        specials = tensorplay.tensor(lanes, dtype=tensorplay.float32)
        reversed_specials = tensorplay.tensor(
            lanes[::-1], dtype=tensorplay.float32)
        for name in ("logical_and", "logical_or", "logical_xor"):
            self._compare(name, specials, reversed_specials)
        self._compare("logical_not", specials)

    def test_a_combined_mask_reaches_attention(self):
        import tensorplay.functional as F
        import tensorplay.nn.functional as NF
        tensorplay.manual_seed(3210)
        B, H, Sq, Skv, D = 2, 4, 6, 10, 8
        q = tensorplay.randn((B, H, Sq, D))
        k = tensorplay.randn((B, H, Skv, D))
        v = tensorplay.randn((B, H, Skv, D))
        keep = np.random.default_rng(7).random((B, 1, 1, Skv)) > 0.2
        # Keep the first key of every batch visible so no query row loses every
        # candidate; a row with nothing to attend to is a separate question.
        keep[:, :, :, 0] = True
        padding = tensorplay.tensor(keep)
        causal = tensorplay.tensor(np.tril(np.ones((Sq, Skv), dtype=bool)))
        # A padding row ands a causal triangle, and the whole mask is then
        # negated once to exercise the inverse form a decoder uses.
        host_mask = F.bitwise_and(padding, causal)
        device_mask = F.bitwise_and(padding.to("vulkan"), causal.to("vulkan"))
        np.testing.assert_array_equal(
            device_mask.cpu().numpy(), host_mask.numpy())
        got = NF.scaled_dot_product_attention(
            q.to("vulkan"), k.to("vulkan"), v.to("vulkan"),
            attn_mask=device_mask).cpu().numpy()
        want = _sdpa_reference(
            q.numpy(), k.numpy(), v.numpy(), mask=host_mask.numpy())
        np.testing.assert_allclose(
            got.astype(np.float64), want, rtol=1.0e-3, atol=4.0e-3)
        inverted = F.logical_not(host_mask)
        device_inverted = F.logical_not(device_mask)
        np.testing.assert_array_equal(
            device_inverted.cpu().numpy(), inverted.numpy())

    def test_bitwise_needs_a_bool_payload(self):
        import tensorplay.functional as F
        # An integral plane wants the bitwise operation of its own width, which
        # no kernel carries: the request is refused instead of being decided by
        # the nonzero test that spells the connectives for a Bool.
        ints = tensorplay.tensor([1, 2, 3], dtype=tensorplay.int32).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "connects Bool payloads"):
            F.bitwise_and(ints, ints)
        with self.assertRaisesRegex(RuntimeError, "connects Bool payloads"):
            F.bitwise_not(ints)


class TestVulkanComparisonBroadcast(unittest.TestCase):
    # Two payloads merge into one broadcast shape, and a plane narrower than the
    # answer is repeated inside the kernel instead of being materialized first.
    # A texel operand is addressed through its own geometry, a linear one
    # through row-major strides, so both storages need the arithmetic: a
    # vocabulary row lives in buffers, where a singleton costs one element
    # rather than one texel.

    OPERATORS = (
        ("gt", lambda a, b: a > b),
        ("ge", lambda a, b: a >= b),
        ("lt", lambda a, b: a < b),
        ("le", lambda a, b: a <= b),
        ("eq", lambda a, b: a == b),
        ("ne", lambda a, b: a != b),
    )
    SHAPES = (
        ((3, 4), (1, 4)),          # one row repeats
        ((3, 4), (3, 1)),          # one channel repeats
        ((4,), (3, 4)),            # rank left-padded
        ((2, 3, 4, 5), (2, 1, 1, 5)),  # a mask across heads
        ((2, 3, 4, 5), (1, 1, 4, 5)),
        ((2, 3, 4, 5), (2, 3, 1, 5)),
    )

    def _check(self, host_a, host_b, label):
        for name, apply_op in self.OPERATORS:
            got = apply_op(host_a.to("vulkan"), host_b.to("vulkan"))
            want = apply_op(host_a, host_b)
            self.assertEqual(got.dtype, tensorplay.bool, msg=f"{name} {label}")
            np.testing.assert_array_equal(
                got.cpu().numpy(), want.cpu().numpy(),
                err_msg=f"{name} {label}")

    def _pair(self, shape_a, shape_b, seed, kind="float", scale=6.0):
        rng = np.random.default_rng(seed)

        def make(shape):
            if kind == "half":
                values = (rng.standard_normal(shape) * scale).astype(np.float16)
                dtype = tensorplay.float16
            elif kind == "int32":
                values = rng.integers(-999, 1000, shape).astype(np.int32)
                dtype = tensorplay.int32
            elif kind == "int64":
                values = rng.integers(
                    -(2 ** 62), 2 ** 62, shape).astype(np.int64)
                dtype = tensorplay.int64
            else:
                values = (rng.standard_normal(shape) * scale).astype(np.float32)
                dtype = tensorplay.float32
            return tensorplay.tensor(values, dtype=dtype)

        return make(shape_a), make(shape_b)

    def test_texel_planes_compare_across_the_broadcast_shape(self):
        for i, (shape_a, shape_b) in enumerate(self.SHAPES):
            a, b = self._pair(shape_a, shape_b, 41 + i)
            self._check(a, b, f"{shape_a} against {shape_b}")
            # Swapping the pair swaps which plane repeats.
            self._check(b, a, f"{shape_b} against {shape_a} swapped")

    def test_halves_compare_across_the_broadcast_shape(self):
        for i, (shape_a, shape_b) in enumerate(self.SHAPES):
            a, b = self._pair(shape_a, shape_b, 61 + i, kind="half", scale=3.0)
            self._check(a, b, f"half {shape_a} against {shape_b}")

    def test_linear_planes_compare_across_a_broadcast(self):
        # A row past the image-axis limit has no texel form, so the answer is
        # packed one byte per element into buffers; a length that is not a
        # multiple of four leaves a partial result word.
        for count in (100000, 100001, 100003):
            rng = np.random.default_rng(71 + count % 7)
            row = tensorplay.tensor(
                (rng.standard_normal((count,)) * 5).astype(np.float32))
            one = tensorplay.tensor(np.array([0.25], dtype=np.float32))
            self._check(row, one, f"float row {count} against one element")
            self._check(one, row, f"one element against float row {count}")
            ints = tensorplay.tensor(
                rng.integers(-999, 1000, (count,)).astype(np.int32),
                dtype=tensorplay.int32)
            one_int = tensorplay.tensor(
                np.array([7], dtype=np.int32), dtype=tensorplay.int32)
            self._check(ints, one_int, f"int row {count} against one element")

    def test_a_mask_row_of_a_vocabulary_compares_broadcast(self):
        # The shape a logits mask arrives in: one threshold per batch row
        # against a whole vocabulary.
        rng = np.random.default_rng(83)
        B, V = 4, 60000
        logits = tensorplay.tensor(
            (rng.standard_normal((B, V)) * 20).astype(np.float32))
        threshold = tensorplay.tensor(
            (rng.standard_normal((B, 1)) * 20).astype(np.float32))
        self._check(logits, threshold, f"({B}, {V}) against ({B}, 1)")
        ints = tensorplay.tensor(
            rng.integers(-100000, 100000, (B, V)).astype(np.int32),
            dtype=tensorplay.int32)
        codes = tensorplay.tensor(
            rng.integers(-3, 4, (B, 1)).astype(np.int32),
            dtype=tensorplay.int32)
        self._check(ints, codes, "int32 vocabulary against per-row codes")

    def test_counts_wider_than_a_word_compare_elementwise(self):
        # An Int64 payload carries one element per pair of words, so the
        # decision has to be taken word against word, with the sign living in
        # the high word.
        values = np.array(
            [0, -1, 1, -(2 ** 63), 2 ** 63 - 1, -(2 ** 62), 2 ** 62, 5, -5,
             2 ** 31, -(2 ** 31) - 1, 7, -7, 3],
            dtype=np.int64)
        other = np.array(
            [0, 0, 0, 2 ** 63 - 1, -(2 ** 63), 2 ** 62 - 1, -(2 ** 62), -5, 5,
             2 ** 31 - 1, -(2 ** 31), 8, -6, 3],
            dtype=np.int64)
        a = tensorplay.tensor(values.copy(), dtype=tensorplay.int64)
        b = tensorplay.tensor(other.copy(), dtype=tensorplay.int64)
        self._check(a, b, "int64 equal shapes")
        # A tail that does not fill a result word.
        self._check(a.narrow(0, 0, 9), b.narrow(0, 0, 9), "int64 odd length")

    def test_counts_wider_than_a_word_compare_broadcast(self):
        rng = np.random.default_rng(89)
        rows, cols = 14, 3
        wide = tensorplay.tensor(
            rng.integers(-(2 ** 60), 2 ** 60, (rows, cols),
                         dtype=np.int64),
            dtype=tensorplay.int64)
        per_row = tensorplay.tensor(
            rng.integers(-(2 ** 60), 2 ** 60, (rows, 1), dtype=np.int64),
            dtype=tensorplay.int64)
        self._check(wide, per_row, "int64 column repeat")
        self._check(per_row, wide, "int64 column repeat swapped")
        one = tensorplay.tensor(
            np.array([-(2 ** 63)], dtype=np.int64), dtype=tensorplay.int64)
        self._check(wide, one, "int64 against one element")

    def test_a_wider_count_meets_a_number_in_every_order(self):
        # The number a count is measured against travels as the pair of words
        # such a count is carried in, so a fixed side answers the same six orders
        # the two-payload walk does - and a request that names the extremes of
        # the width is the tightest one it can make.
        values = np.array(
            [0, 1, -1, 2, 5, -7, 2 ** 31, -(2 ** 31) - 1, 2 ** 63 - 1,
             -(2 ** 63), (2 ** 40) + 7],
            dtype=np.int64)
        counts = tensorplay.tensor(values.copy(), dtype=tensorplay.int64)
        for threshold in (2, -1, 0, 5, 2 ** 31, (2 ** 63) - 1, -(2 ** 63)):
            for name, want in (
                    ("eq", values == threshold),
                    ("ne", values != threshold),
                    ("lt", values < threshold),
                    ("le", values <= threshold),
                    ("gt", values > threshold),
                    ("ge", values >= threshold)):
                got = getattr(tensorplay, name)(
                    counts.to("vulkan"), threshold).cpu().numpy()
                np.testing.assert_array_equal(
                    got, want,
                    err_msg=f"int64 {name} against {threshold}")

    def test_a_wider_count_turned_into_a_mask_keeps_its_own_width(self):
        # The mask a vocabulary keeps is the same walk with the count side first:
        # a reversed request compares the number against the pair, not the pair
        # read as a word.
        values = np.array(
            [0, 1, -1, 5, -(1 << 32), (1 << 40) + 7, -(2 ** 63), 2 ** 63 - 1],
            dtype=np.int64)
        counts = tensorplay.tensor(values.copy(), dtype=tensorplay.int64)
        np.testing.assert_array_equal(
            (counts.to("vulkan") > 5).cpu().numpy(), values > 5,
            err_msg="int64 kept above a number")
        np.testing.assert_array_equal(
            (counts.to("vulkan") >= (1 << 40)).cpu().numpy(),
            values >= (1 << 40), err_msg="int64 kept past a word")
        np.testing.assert_array_equal(
            (5 > counts.to("vulkan")).cpu().numpy(), 5 > values,
            err_msg="a number kept above an int64")
        # A whole number written with a fraction in it is still whole.
        np.testing.assert_array_equal(
            (counts.to("vulkan") > 2.0).cpu().numpy(), values > 2,
            err_msg="int64 against a whole number of a different family")
        # The answer packs one byte per element, four lanes to a word, so a
        # length that fills no whole number of words still needs every lane.
        for length in (1, 2, 3, 5, 17, 65, 257):
            rng = np.random.default_rng(length)
            host = rng.integers(-(2 ** 62), 2 ** 62, size=length)
            np.testing.assert_array_equal(
                tensorplay.gt(
                    tensorplay.tensor(host.copy(), dtype=tensorplay.int64)
                    .to("vulkan"), 0).cpu().numpy(),
                host > 0, err_msg=f"{length} counts kept above none")

    def test_a_wider_count_and_a_fraction_cannot_share_a_kernel(self):
        counts = tensorplay.tensor(
            np.array([1, 2, 3], dtype=np.int64),
            dtype=tensorplay.int64).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "against a whole number"):
            counts > 2.5

    def test_a_causal_mask_keeps_its_strict_triangle(self):
        # The additive mask of attention is one comparison of two ranges: a row
        # keeps every key up to and including its own position.
        Sq, Skv = 5, 7
        rows = np.arange(Sq, dtype=np.int64).reshape(Sq, 1)
        cols = np.arange(Skv, dtype=np.int64).reshape(1, Skv)
        a = tensorplay.tensor(rows.copy(), dtype=tensorplay.int64)
        b = tensorplay.tensor(cols.copy(), dtype=tensorplay.int64)
        got = (a >= b).cpu().numpy()
        np.testing.assert_array_equal(got, rows >= cols, err_msg="causal keep")


class TestVulkanWhereBranches(unittest.TestCase):
    # A select reads both branches in one element width.  Pairing a single
    # precision plane with a signed-word plane therefore cannot read the word as
    # a float: the bits survive and the number does not, so the pair is first
    # promoted to the type that holds both.

    def _mixed(self, host_a, host_b, condition):
        import tensorplay.functional as F
        want = F.where(condition, host_a, host_b)
        got = F.where(
            condition.to("vulkan"), host_a.to("vulkan"), host_b.to("vulkan"))
        return got.cpu(), want

    def test_a_mixed_branch_pair_selects_in_the_wider_kind(self):
        rng = np.random.default_rng(23)
        condition = tensorplay.tensor(rng.random((2, 3, 4, 5)) > 0.5)
        floats = tensorplay.tensor(
            (rng.standard_normal((2, 3, 4, 5)) * 8).astype(np.float32))
        halves = tensorplay.tensor(
            (rng.standard_normal((2, 3, 4, 5)) * 8).astype(np.float16))
        ints = tensorplay.tensor(
            rng.integers(-9, 10, (2, 3, 4, 5)).astype(np.int32))
        for host_a, host_b in [(floats, ints), (ints, floats),
                               (halves, ints), (ints, halves),
                               (floats, halves), (halves, floats)]:
            for a, b in [(host_a, host_b), (host_b, host_a)]:
                got, want = self._mixed(a, b, condition)
                self.assertEqual(
                    got.dtype, want.dtype,
                    msg=f"{a.dtype} against {b.dtype} changed the result kind")
                np.testing.assert_array_equal(
                    got.numpy(), want.numpy(),
                    err_msg=f"{a.dtype} against {b.dtype}")

    def test_a_rejected_branch_cannot_poison_the_result(self):
        # The lane that loses the selection must not reach the arithmetic at
        # all: a recombination multiplies it by zero, and zero times an
        # infinity or a not-a-number is a not-a-number that lands on the lane
        # that won.
        import tensorplay.functional as F
        condition = tensorplay.tensor([True, False, True, False])
        floats = tensorplay.tensor(
            [1.0, np.nan, np.inf, -np.inf], dtype=tensorplay.float32)
        ints = tensorplay.tensor([7, 7, 7, 7], dtype=tensorplay.int32)
        got = F.where(condition.to("vulkan"), floats.to("vulkan"),
                      ints.to("vulkan")).cpu().numpy()
        np.testing.assert_array_equal(got, np.array([1.0, 7.0, np.inf, 7.0]))
        got = F.where(condition.to("vulkan"), ints.to("vulkan"),
                      floats.to("vulkan")).cpu().numpy()
        np.testing.assert_array_equal(got, np.array([7.0, np.nan, 7.0, -np.inf]))

    def test_branches_outside_the_kernel_vocabulary_are_refused(self):
        import tensorplay.functional as F
        # A Bool pair has no float or word width to promote to, and the select
        # would read byte lanes as one of those: refuse instead.
        condition = tensorplay.tensor([True, False]).to("vulkan")
        flags = tensorplay.tensor([True, False]).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError,
                "selects between Float32, Float16, Int32 and Int64"):
            F.where(condition, flags, flags)


class TestVulkanWhereBroadcast(unittest.TestCase):
    # The condition and the two branches merge into one output shape, and a
    # plane narrower than it is repeated inside the kernel instead of being
    # materialized at the output shape.  Both storages need the arithmetic: a
    # row beyond the image-axis limit lives in buffers, where a singleton costs
    # one element rather than one texel.

    SHAPES = (
        ((2, 3, 4, 5), (2, 1, 1, 5), (2, 3, 4, 5)),  # condition across heads
        ((2, 3, 4, 5), (2, 3, 4, 5), (1, 1, 4, 5)),  # one branch is shared
        ((2, 3, 4, 5), (4, 5), (2, 3, 4, 5)),  # rank left-padded
        ((2, 3, 4, 5), (2, 3, 1, 5), (2, 3, 4, 5)),  # one channel repeats
        ((3, 5, 6), (3, 5, 6), (1, 1, 1)),
        ((2, 4, 3, 5), (1, 1, 1, 1), (2, 4, 3, 5)),
    )
    LENGTHS = (100000, 100001, 100003)

    def _check(self, condition, host_a, host_b, label):
        import tensorplay.functional as F
        want = F.where(condition, host_a, host_b)
        got = F.where(
            condition.to("vulkan"), host_a.to("vulkan"), host_b.to("vulkan"))
        self.assertEqual(got.dtype, want.dtype, msg=label)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want.cpu().numpy(), err_msg=label)

    def _planes(self, out_shape, cond_shape, branch_shape, seed):
        rng = np.random.default_rng(seed)
        return (
            tensorplay.tensor(rng.random(cond_shape) > 0.4),
            tensorplay.tensor(
                (rng.standard_normal(branch_shape) * 7).astype(np.float32)),
            tensorplay.tensor(
                (rng.standard_normal(out_shape) * 7).astype(np.float32)))

    def test_texel_planes_broadcast_to_the_output_shape(self):
        for i, (out, cond, branch) in enumerate(self.SHAPES):
            condition, host_a, host_b = self._planes(out, cond, branch, 11 + i)
            self._check(condition, host_a, host_b, f"{out} {cond} {branch}")
            # Swapping the branches swaps which plane repeats.
            self._check(condition, host_b, host_a, f"{out} swapped")

    def test_a_zero_dim_branch_repeats(self):
        import tensorplay as tp
        rng = np.random.default_rng(19)
        condition = tensorplay.tensor(rng.random((2, 3, 4, 5)) > 0.5)
        dense = tensorplay.tensor(
            (rng.standard_normal((2, 3, 4, 5)) * 7).astype(np.float32))
        self._check(condition, tensorplay.tensor(3.25, dtype=tp.float32), dense,
                    "0-d self branch")
        self._check(condition, dense, tensorplay.tensor(-1.5, dtype=tp.float32),
                    "0-d other branch")

    def test_a_scalar_branch_keeps_the_other_branch_kind(self):
        import tensorplay as tp
        import tensorplay.functional as F
        rng = np.random.default_rng(29)
        condition = tensorplay.tensor(rng.random((3, 4, 6)) > 0.5)
        halves = tensorplay.tensor(
            (rng.standard_normal((3, 4, 6)) * 4).astype(np.float16),
            dtype=tp.float16)
        # The folded branch is one element that the selection repeats, so the
        # pair stays in the half width and the scalar follows it.
        want = F.where(condition, halves, 7)
        got = F.where(
            condition.to("vulkan"), halves.to("vulkan"), 7)
        self.assertEqual(got.dtype, want.dtype)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want.cpu().numpy(), err_msg="scalar other")
        want = F.where(condition, -3, halves)
        got = F.where(
            condition.to("vulkan"), -3, halves.to("vulkan"))
        self.assertEqual(got.dtype, want.dtype)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want.cpu().numpy(), err_msg="scalar self")

    def test_linear_planes_broadcast_across_every_vocabulary(self):
        import tensorplay as tp
        rng = np.random.default_rng(23)
        for count in self.LENGTHS:
            condition = tensorplay.tensor(rng.random((count,)) > 0.4)
            floats = tensorplay.tensor(
                (rng.standard_normal((count,)) * 7).astype(np.float32))
            halves = tensorplay.tensor(
                (rng.standard_normal((count,)) * 4).astype(np.float16),
                dtype=tp.float16)
            ints = tensorplay.tensor(
                rng.integers(-9, 10, (count,)).astype(np.int32),
                dtype=tp.int32)
            # A one-element plane of each kind, in either branch position, so
            # both the shared word layout and the singleton read get used.
            single_floats = tensorplay.tensor(
                np.array([2.5], dtype=np.float32))
            single_halves = tensorplay.tensor(
                np.array([-3.5], dtype=np.float16), dtype=tp.float16)
            single_ints = tensorplay.tensor(
                np.array([7], dtype=np.int32), dtype=tp.int32)
            self._check(
                condition, floats, single_floats, f"float tail at {count}")
            self._check(
                condition, single_halves, halves, f"half self at {count}")
            self._check(
                condition, halves, single_ints, f"half against int at {count}")
            self._check(
                condition, ints, single_ints, f"int at {count}")
            # A condition of one element decides the whole row.
            self._check(
                tensorplay.tensor(np.array([True])), floats,
                tensorplay.tensor(
                    (rng.standard_normal((count,)) * 7).astype(np.float32)),
                f"single condition at {count}")

    def test_a_padding_mask_selects_a_whole_attention_row(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(31)
        B, H, Sq, Skv = 2, 4, 5, 9
        keep = rng.random((B, 1, 1, Skv)) > 0.35
        keep[:, :, :, 0] = True
        scores = (rng.standard_normal((B, H, Sq, Skv)) * 4).astype(np.float32)
        masked = np.where(keep, scores, -1e4)
        condition = tensorplay.tensor(keep)
        host_scores = tensorplay.tensor(scores)
        bias = tensorplay.tensor(
            np.full((B, 1, 1, Skv), -1.0e4, dtype=np.float32))
        self._check(condition, host_scores, bias, "mask across heads")
        want = F.where(condition, host_scores, bias)
        np.testing.assert_array_equal(
            masked, want.cpu().numpy(), err_msg="reference agrees with host")

    def test_shapes_that_do_not_broadcast_are_refused(self):
        import tensorplay.functional as F
        condition = tensorplay.tensor(
            np.ones((2, 3, 4, 5), dtype=bool)).to("vulkan")
        a = tensorplay.tensor(np.zeros((2, 3, 4, 5), dtype=np.float32))
        b = tensorplay.tensor(np.zeros((3, 4), dtype=np.float32)).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "must match the size of tensor"):
            F.where(condition, a.to("vulkan"), b)


class TestVulkanMaskedFillWidths(unittest.TestCase):
    # A masked lane takes the fill and every other lane keeps the payload, which
    # is the same choice a selection makes between two branches.  One walk
    # answers for both, so the fill covers the widths the selection covers and a
    # mask narrower than the payload broadcasts inside the kernel instead of
    # being read as though it already carried the payload's shape.

    LENGTHS = (100000, 100001, 100003)

    @staticmethod
    def _payload(dtype, shape, seed):
        rng = np.random.default_rng(seed)
        if dtype is tensorplay.float32:
            return (rng.standard_normal(shape) * 8).astype(np.float32)
        if dtype is tensorplay.float16:
            return (rng.standard_normal(shape) * 8).astype(np.float16)
        if dtype is tensorplay.int32:
            return rng.integers(-1000, 1000, shape).astype(np.int32)
        return rng.integers(-10**12, 10**12, shape).astype(np.int64)

    def _check(self, host, mask, fill, label):
        import tensorplay.functional as F
        want = F.masked_fill(
            tensorplay.tensor(host), tensorplay.tensor(mask), fill)
        got = F.masked_fill(
            tensorplay.tensor(host).to("vulkan"),
            tensorplay.tensor(mask).to("vulkan"), fill)
        self.assertEqual(got.dtype, want.dtype, msg=label)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want.cpu().numpy(), err_msg=label)

    def test_a_fill_reaches_every_payload_width(self):
        rng = np.random.default_rng(7)
        widths = (
            (tensorplay.float32, -1.75),
            (tensorplay.float16, -2.5),
            (tensorplay.int32, -7),
            # A count past the range of a word still lands where the mask says.
            (tensorplay.int64, -99999999999),
        )
        for i, (dtype, fill) in enumerate(widths):
            host = self._payload(dtype, (4, 6), 41 + i)
            mask = rng.random((4, 6)) > 0.45
            self._check(host, mask, fill, f"{dtype} filled")

    def test_a_narrower_mask_broadcasts_over_the_payload(self):
        # Each of these once read only as many mask entries as the payload held
        # lanes of its own texel, filling a few elements and leaving the rest.
        cases = (
            ((2, 4), (2, 1)),  # a column decides whole rows
            ((2, 4), (4,)),  # a row pattern repeats down the rows
            ((2, 3, 4, 5), (1, 1, 4, 1)),  # a per-position gate across heads
            ((2, 3, 4, 5), (2, 3, 1, 5)),
            ((3, 5, 6), (1, 1, 1)),
        )
        rng = np.random.default_rng(13)
        for out, cond in cases:
            host = (rng.standard_normal(out) * 8).astype(np.float32)
            mask = rng.random(cond) > 0.4
            self._check(host, mask, -9.0, f"{out} under a {cond} mask")
        # A padding mask over a half payload, the shape an attention score
        # plane arrives in.
        B, H, Sq, Skv = 2, 4, 5, 9
        keep = rng.random((B, 1, 1, Skv)) > 0.35
        scores = (rng.standard_normal((B, H, Sq, Skv)) * 4).astype(np.float16)
        self._check(scores, keep, np.float16(-1e4), "half padding mask")
        # The payload may be the plane that has to repeat instead.
        self._check(
            (rng.standard_normal((1, 4)) * 8).astype(np.float32),
            rng.random((3, 4)) > 0.4, -9.0, "a payload that grows to the mask")

    def test_a_replaced_lane_leaks_nothing_of_its_own(self):
        import tensorplay.functional as F
        host = np.array([np.inf, 1.0, np.nan, -np.inf, 2.0, 3.0], np.float32)
        mask = np.array([True, False, True, True, False, True])
        want = F.masked_fill(
            tensorplay.tensor(host), tensorplay.tensor(mask), -2.0)
        got = F.masked_fill(
            tensorplay.tensor(host).to("vulkan"),
            tensorplay.tensor(mask).to("vulkan"), -2.0)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want.cpu().numpy(), err_msg="lanes replaced")
        # A lane the mask leaves alone keeps its own infinity.
        keep = np.array([np.inf, np.nan, 5.0], np.float32)
        untouched = np.array([False, True, False])
        self._check(keep, untouched, 0.0, "an untouched infinity")

    def test_a_linear_payload_fills_in_its_own_storage(self):
        rng = np.random.default_rng(17)
        widths = (
            (tensorplay.float32, -1.5),
            (tensorplay.float16, -2.5),
            (tensorplay.int32, -7),
            (tensorplay.int64, -99999999999),
        )
        for count in self.LENGTHS:
            for i, (dtype, fill) in enumerate(widths):
                host = self._payload(dtype, (count,), 61 + i)
                mask = rng.random(count) > 0.5
                self._check(host, mask, fill, f"{dtype} row at {count}")
                # One condition element decides the whole row.
                self._check(
                    host, np.array([True]), fill, f"{dtype} single mask")

    def test_a_tail_that_fills_no_whole_word_still_selects(self):
        for count in (1, 2, 3, 5, 17, 65, 257):
            halves = (np.arange(count) % 7).astype(np.float16)
            mask = np.arange(count) % 3 == 1
            self._check(halves, mask, np.float16(-3.0), f"half at {count}")
            counts = (np.arange(count) * 1000000007).astype(np.int64)
            self._check(counts, mask, -5, f"count at {count}")

    def test_an_in_place_fill_keeps_the_width_it_was_given(self):
        rng = np.random.default_rng(19)
        fills = {
            tensorplay.float32: -2.5,
            tensorplay.float16: -2.5,
            tensorplay.int32: -7,
            tensorplay.int64: -10**11,
        }
        for i, dtype in enumerate((
                tensorplay.float32, tensorplay.float16, tensorplay.int32,
                tensorplay.int64)):
            host = self._payload(dtype, (3, 5), 71 + i)
            mask = rng.random((3, 5)) > 0.4
            fill = fills[dtype]
            want = tensorplay.tensor(host.copy())
            want.masked_fill_(tensorplay.tensor(mask), fill)
            got = tensorplay.tensor(host.copy()).to("vulkan")
            got.masked_fill_(tensorplay.tensor(mask).to("vulkan"), fill)
            self.assertEqual(got.dtype, want.dtype, msg=f"in place {dtype}")
            np.testing.assert_array_equal(
                got.cpu().numpy(), want.cpu().numpy(),
                err_msg=f"in place {dtype}")

    def test_a_value_of_another_width_is_read_into_the_payload(self):
        import tensorplay.functional as F
        halves = (np.arange(10) % 5).astype(np.float16)
        mask = np.arange(10) % 4 == 1
        value = tensorplay.tensor(np.float32(1.75))
        want = F.masked_fill(
            tensorplay.tensor(halves), tensorplay.tensor(mask), value)
        got = F.masked_fill(
            tensorplay.tensor(halves).to("vulkan"),
            tensorplay.tensor(mask).to("vulkan"), value.to("vulkan"))
        self.assertEqual(got.dtype, tensorplay.float16)
        np.testing.assert_array_equal(
            got.cpu().numpy(), want.cpu().numpy(), err_msg="0-dim value")

    def test_a_payload_that_is_only_truth_is_refused(self):
        import tensorplay.functional as F
        flags = tensorplay.tensor([True, False, True]).to("vulkan")
        condition = tensorplay.tensor([True, False, True]).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError,
                "supports Float32, Float16, Int32 and Int64 tensors only"):
            F.masked_fill(flags, condition, False)
        # A payload built straight on the device can outgrow the ranks the walks
        # cover; the fill names its own refusal rather than filling lanes it
        # cannot address.
        deep = tensorplay.tensor(
            np.zeros((2, 2, 2, 2, 2), np.float32), device="vulkan")
        with self.assertRaisesRegex(RuntimeError, "supports up to 4d tensors"):
            F.masked_fill(
                deep, tensorplay.tensor([True], device="vulkan"), -1.0)


class TestVulkanScalarVocabulary(unittest.TestCase):
    # Element-wise arithmetic names the width it computes in after the operands
    # it was given: a count beside a whole number stays a count, a count beside
    # a fraction moves to a width that has fractions, and a pair written in two
    # codes is read in the one that holds both.  A number on the left of the
    # operator has to reach the slot of the formula it belongs to, since a texel
    # and a word disagree about what the same bits mean.

    LENGTHS = (100001, 200003)

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.array(values)).to("vulkan")

    def test_a_whole_number_beside_a_count_stays_a_count(self):
        import tensorplay as tp
        counts = np.array([-5, -3, -2, 1, 2, 4, 5, 7, 8], dtype=np.int32)
        cases = (
            ("add", lambda t, n: t + n, lambda n: counts + n),
            ("sub", lambda t, n: t - n, lambda n: counts - n),
            ("mul", lambda t, n: t * n, lambda n: counts * n),
            ("floordiv", lambda t, n: t // n,
             lambda n: np.floor_divide(counts, n)),
            ("mod", lambda t, n: t % n, lambda n: np.mod(counts, n)),
            ("pow", lambda t, n: t ** n, lambda n: np.power(counts, n)),
        )
        for name, on_device, on_numbers in cases:
            for number in ((2, -3) if name != "pow" else (2, 3)):
                got = on_device(self._device(counts), number)
                self.assertEqual(got.dtype, tp.int32, msg=f"{name} by {number}")
                np.testing.assert_array_equal(
                    got.cpu().numpy(), on_numbers(number).astype(np.int32),
                    err_msg=f"{name} by {number}")

    def test_a_payload_beside_a_number_takes_the_width_the_pair_needs(self):
        import tensorplay as tp
        counts = np.array([-5, -3, -2, 1, 2, 4, 5, 7, 8], dtype=np.int32)
        reals = counts.astype(np.float32)
        cases = (
            ("add", lambda t: t + 2.5, reals + 2.5),
            ("mul", lambda t: t * 2.5, reals * 2.5),
            # A quotient that is a whole number has to land on it: the divisor
            # is an operand, so no reciprocal error sits at the boundary the
            # floor is deciding.
            ("floordiv", lambda t: t // 2.5, np.floor(reals / 2.5)),
            ("mod", lambda t: t % 2.5, np.mod(reals, 2.5)),
        )
        for name, on_device, want in cases:
            got = on_device(self._device(counts))
            self.assertEqual(got.dtype, tp.float32, msg=name)
            np.testing.assert_allclose(
                got.cpu().numpy(), want, rtol=1e-3, atol=1e-3, err_msg=name)

        truth = np.array([True, False, True, True])
        for name, on_device, want in (
                ("add", lambda t: t + 2, truth.astype(np.float32) + 2),
                ("floordiv", lambda t: t // 2, np.floor_divide(truth, 2)),
                ("pow", lambda t: t ** 2, np.power(truth, 2))):
            got = on_device(self._device(truth))
            self.assertEqual(got.dtype, tp.float32, msg=f"truth {name}")
            np.testing.assert_allclose(
                got.cpu().numpy(), want, rtol=1e-3, atol=1e-3,
                err_msg=f"truth {name}")

    def test_a_reflected_number_reaches_the_slot_its_formula_needs(self):
        import tensorplay as tp
        counts = np.array([-5, -3, -2, 1, 2, 4, 5, 7, 8], dtype=np.int32)
        reals = counts.astype(np.float32)
        cases = (
            ("rsub", lambda t: 5 - t, 5 - counts, tp.int32),
            ("rfloordiv", lambda t: 10 // t, np.floor_divide(10, counts),
             tp.int32),
            ("rmod", lambda t: 10 % t, np.mod(10, counts), tp.int32),
            ("rdiv", lambda t: 6 / t, 6 / reals, tp.float32),
        )
        for name, on_device, want, dtype in cases:
            got = on_device(self._device(counts))
            self.assertEqual(got.dtype, dtype, msg=name)
            np.testing.assert_allclose(
                got.cpu().numpy(), want, rtol=1e-3, atol=1e-3, err_msg=name)

        for name, on_device, want in (
                ("rfloordiv", lambda t: 10.0 // t, np.floor(10.0 / reals)),
                ("rmod", lambda t: 10.0 % t, np.mod(10.0, reals)),
                ("rpow", lambda t: 2.0 ** t, np.power(2.0, reals))):
            got = on_device(self._device(reals))
            self.assertEqual(got.dtype, tp.float32, msg=f"real {name}")
            np.testing.assert_allclose(
                got.cpu().numpy(), want, rtol=2e-2, atol=2e-2,
                err_msg=f"real {name}")

    def test_the_remainder_takes_the_sign_of_its_divisor(self):
        # The difference is taken against the floored quotient, so a lane keeps
        # the sign of the number it was divided by rather than its own.
        counts = np.array([-7, -3, -1, 1, 2, 4, 8], dtype=np.int32)
        for number in (2, -2, 3, -3, 5):
            got = self._device(counts) % number
            np.testing.assert_array_equal(
                got.cpu().numpy(), np.mod(counts, number),
                err_msg=f"count mod {number}")
            written = self._device(counts)
            written.remainder_(number)
            np.testing.assert_array_equal(
                written.cpu().numpy(), np.mod(counts, number),
                err_msg=f"count mod {number} written back")

        divisors = np.array([2, -3, 4, -5, 2, 3, -4], dtype=np.int32)
        got = self._device(counts) % self._device(divisors)
        np.testing.assert_array_equal(
            got.cpu().numpy(), np.mod(counts, divisors),
            err_msg="count mod count")

        rng = np.random.default_rng(19)
        for count in self.LENGTHS:
            row = rng.integers(-9, 10, count).astype(np.int32)
            step = rng.integers(1, 6, count).astype(np.int32)
            if count % 2:
                step = -step
            got = self._device(row) % self._device(step)
            np.testing.assert_array_equal(
                got.cpu().numpy(), np.mod(row, step),
                err_msg=f"linear remainder at {count} elements")

    def test_a_negative_base_keeps_its_sign_through_a_whole_exponent(self):
        # A power has no value defined for a negative base in the shader's own
        # arithmetic, so the magnitude is raised and an even exponent puts the
        # sign back.
        reals = np.array([-3.0, -2.0, 4.0, -5.0], dtype=np.float32)
        for exponent in (2, 3, 4, 5):
            np.testing.assert_allclose(
                (self._device(reals) ** exponent).cpu().numpy(),
                np.power(reals, exponent),
                rtol=1e-3, atol=1e-3, err_msg=f"real base to {exponent}")
        counts = np.array([-3, -2, 4, -5], dtype=np.int32)
        for exponent in (2, 3):
            got = self._device(counts) ** exponent
            np.testing.assert_array_equal(
                got.cpu().numpy(), np.power(counts, exponent),
                err_msg=f"count to {exponent}")
        exponents = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        # The base has to be parenthesized: a minus written beside a power binds
        # looser than the operator, so `-2.0 ** t` negates a positive base
        # instead of naming a negative one.
        np.testing.assert_allclose(
            ((-2.0) ** self._device(exponents)).cpu().numpy(),
            np.power(-2.0, exponents),
            rtol=1e-3, atol=1e-3, err_msg="negative base reflected")
        # Past the image edge the same exponent payload rides the linear writer,
        # where the sign rule is spelled against a whole uniform instead of a
        # vector lane.
        wide = (np.arange(self.LENGTHS[0], dtype=np.float32) % 6) + 2.0
        np.testing.assert_allclose(
            ((-2.0) ** self._device(wide)).cpu().numpy(),
            np.power(-2.0, wide),
            rtol=1e-2, atol=1e-2, err_msg="negative base reflected, linear")

    def test_operands_of_two_widths_are_read_in_the_one_that_holds_both(self):
        import tensorplay as tp
        kinds = (tp.float32, tp.float16, tp.int32, tp.bool)
        rng = np.random.default_rng(41)
        for count in (24,) + self.LENGTHS:
            planes = {
                tp.float32: (rng.standard_normal(count) * 6).astype(np.float32),
                tp.float16: (rng.standard_normal(count) * 6).astype(np.float16),
                tp.int32: rng.integers(-9, 10, count).astype(np.int32),
                tp.bool: rng.random(count) > 0.4,
            }
            for first in kinds:
                for second in kinds:
                    if first == second:
                        continue
                    host_a = tensorplay.from_numpy(planes[first].copy())
                    host_b = tensorplay.from_numpy(planes[second].copy())
                    for op in ("add", "multiply"):
                        want = getattr(host_a, op)(host_b)
                        got = getattr(self._device(planes[first]), op)(
                            self._device(planes[second]))
                        label = f"{first} {op} {second} at {count}"
                        self.assertEqual(got.dtype, want.dtype, msg=label)
                        np.testing.assert_allclose(
                            got.cpu().numpy().astype(np.float64),
                            want.cpu().numpy().astype(np.float64),
                            rtol=1e-3, atol=1e-3, err_msg=label)

    def test_a_pair_that_no_kernel_can_write_is_named(self):
        counts = self._device(np.array([1, 2], dtype=np.int32))
        with self.assertRaisesRegex(
                RuntimeError, "wider than its destination"):
            counts.add_(
                self._device(np.array([1.0, 2.0], dtype=np.float32)))
        with self.assertRaisesRegex(RuntimeError, "fractional operand"):
            counts.add_(2.5)
        # A negative whole power of a count names a fraction, and no word holds
        # it: the answer would be a rounded quotient rather than a refusal.
        with self.assertRaisesRegex(RuntimeError, "negative exponent"):
            counts ** -1
        with self.assertRaisesRegex(
                RuntimeError, "tensors only"):
            self._device(np.array([1, 2], dtype=np.int8)) + 2


class TestVulkanUnaryWidth(unittest.TestCase):
    # The width a unary answer lands in belongs to the formula, not to the
    # payload: a series or a ratio has fractions in it, so a count is read in
    # single precision and the result names that width, while rounding, sign and
    # magnitude work keep the count a count.  Written back the destination keeps
    # the width it was allocated in, so a fractional formula has no place there
    # and is named rather than rounded into the words.

    LENGTHS = (100001, 200003)

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.array(values)).to("vulkan")

    def test_a_fractional_formula_reads_a_count_in_single_precision(self):
        import tensorplay as tp
        counts = np.array([-3, -1, 0, 1, 2, 5], dtype=np.int32)
        for name in ("exp", "sqrt", "sigmoid", "sin", "cos", "tanh",
                     "reciprocal"):
            host = getattr(tensorplay.from_numpy(counts.copy()), name)()
            got = getattr(self._device(counts), name)()
            label = f"count {name}"
            self.assertEqual(got.dtype, host.dtype, msg=label)
            self.assertEqual(got.dtype, tp.float32, msg=label)
            np.testing.assert_allclose(
                got.cpu().numpy(), host.cpu().numpy(),
                rtol=1e-3, atol=1e-3, err_msg=label)

        # Past the image edge the same payload rides the linear writer, where
        # the width is named by the element type of the words rather than by a
        # texel format.
        rng = np.random.default_rng(7)
        for count in self.LENGTHS:
            row = rng.integers(-4, 6, count).astype(np.int32)
            for name in ("exp", "sigmoid", "cos", "reciprocal"):
                host = getattr(tensorplay.from_numpy(row.copy()), name)()
                got = getattr(self._device(row), name)()
                self.assertEqual(got.dtype, tp.float32,
                                 msg=f"linear {name} at {count}")
                np.testing.assert_allclose(
                    got.cpu().numpy(), host.cpu().numpy(),
                    rtol=1e-3, atol=1e-3,
                    err_msg=f"linear {name} at {count}")

        # A float payload keeps its own width: nothing is read in for it.
        halves = np.array([-3.0, -1.0, 0.5, 1.0, 2.0, 5.0], dtype=np.float16)
        for name in ("exp", "sin", "sigmoid"):
            got = getattr(self._device(halves), name)()
            self.assertEqual(got.dtype, tp.float16, msg=f"half {name}")

    def test_an_integral_formula_keeps_its_count(self):
        counts = np.array([-3, -1, 0, 1, 2, 5], dtype=np.int32)
        # The sign of a count has no writer that keeps the count in place yet,
        # so only its out-of-place form is checked.
        written_back = ("abs", "neg", "floor", "ceil", "trunc", "round",
                        "square", "relu")
        for name in ("abs", "neg", "floor", "ceil", "trunc", "round", "sign",
                     "square", "relu"):
            host = getattr(tensorplay.from_numpy(counts.copy()), name)()
            got = getattr(self._device(counts), name)()
            self.assertEqual(got.dtype, host.dtype, msg=name)
            np.testing.assert_array_equal(
                got.cpu().numpy(), host.cpu().numpy(), err_msg=name)

            if name not in written_back:
                continue
            written = self._device(counts)
            getattr(written, name + "_")()
            np.testing.assert_array_equal(
                written.cpu().numpy(), host.cpu().numpy(),
                err_msg=f"{name} written back")

    def test_a_fractional_formula_is_refused_where_it_cannot_write(self):
        import tensorplay as tp
        counts = np.array([-3, -1, 0, 1, 2, 5], dtype=np.int32)
        for name in ("sin_", "cos_", "exp_", "sqrt_", "tanh_", "sigmoid_",
                     "reciprocal_"):
            with self.assertRaisesRegex(
                    RuntimeError, "no kernel for a signed word"):
                getattr(self._device(counts), name)()
        # Written into a float the same formulas need no such step.
        reals = np.array([-3.0, -1.0, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        for name in ("sin_", "exp_", "sigmoid_", "reciprocal_"):
            written = self._device(reals)
            getattr(written, name)()
            self.assertEqual(written.dtype, tp.float32, msg=name)


class TestVulkanPowerLanes(unittest.TestCase):
    # A texel and a word each carry several exponents, so the three decisions a
    # power makes -- whether the exponent is whole, which sign a negative base
    # takes, and how a fraction reads back into a count -- belong to one lane at
    # a time.  Taken for the whole word they answer for exponents the lane never
    # had: a signed word rounded the half a negative power names up into a whole
    # count, and a single precision word lost the sign of the lanes that shared
    # it with a fractional exponent.

    LENGTHS = (100001, 200003)

    # Whole exponents of both parities beside fractional ones that have no
    # value at all over a negative base.
    EXPONENTS = np.array([2.0, 2.5, 4.0, 4.5, 3.0, 0.5, 6.0, 1.5],
                         dtype=np.float32)
    SIGNED = np.array([4.0, np.nan, 16.0, np.nan, -8.0, np.nan, 64.0, np.nan],
                      dtype=np.float32)

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.array(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.array(values))

    def _check_signed(self, got, want, label):
        self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)),
                        msg=f"{label}: which lanes have no value")
        np.testing.assert_allclose(
            np.nan_to_num(got, nan=0.0), np.nan_to_num(want, nan=0.0),
            rtol=1e-3, atol=1e-3, err_msg=label)

    def test_a_base_reflected_into_mixed_exponents_keeps_each_lane(self):
        for dtype in (np.float32, np.float16):
            exponents = self.EXPONENTS.astype(dtype)
            signed = self.SIGNED.astype(dtype)
            got = ((-2.0) ** self._device(exponents)).cpu().numpy()
            self._check_signed(
                got, signed, f"reflected base, {np.dtype(dtype).name}")
            self.assertEqual(got.dtype, dtype, msg="reflected width")

        # One exponent per word on the linear path, and a whole row of texels
        # past the image edge on the packed one.
        for count in self.LENGTHS:
            exponents = np.resize(self.EXPONENTS, count).astype(np.float32)
            signed = np.resize(self.SIGNED, count).astype(np.float32)
            got = ((-2.0) ** self._device(exponents)).cpu().numpy()
            self._check_signed(got, signed, f"reflected base at {count}")

    def test_a_pair_of_payloads_keeps_each_lanes_value(self):
        bases = np.full(self.EXPONENTS.shape, -2.0, dtype=np.float32)
        got = tensorplay.pow(self._device(bases),
                             self._device(self.EXPONENTS)).cpu().numpy()
        self._check_signed(got, self.SIGNED, "base and exponent as payloads")

        for count in self.LENGTHS:
            exponents = np.resize(self.EXPONENTS, count).astype(np.float32)
            bases = np.full(count, -2.0, dtype=np.float32)
            got = tensorplay.pow(self._device(bases),
                                 self._device(exponents)).cpu().numpy()
            self._check_signed(got, np.resize(self.SIGNED, count),
                               f"payload pair at {count}")

    def test_a_fraction_of_a_count_is_read_the_way_the_word_holds_it(self):
        # Half of a count: the answer has no fractions to keep, and a half
        # rounded up would name a count no division of it gives.
        np.testing.assert_array_equal(
            (2 ** self._device(np.array([-1], dtype=np.int32))).cpu().numpy(),
            np.array([0], dtype=np.int32),
            err_msg="a half read back into a count")

        exponents = np.array([-1, 2, -1, 2, -2, 3, -3, 0], dtype=np.int32)
        for base in (2, -2, 3):
            want = (base ** self._host(exponents)).cpu().numpy()
            np.testing.assert_array_equal(
                (base ** self._device(exponents)).cpu().numpy(), want,
                err_msg=f"count exponents under {base}")

        # A whole power that reaches past a word stays on the number it names.
        wide = np.array([30, 29, -1, 0], dtype=np.int32)
        np.testing.assert_array_equal(
            (2 ** self._device(wide)).cpu().numpy(),
            (2 ** self._host(wide)).cpu().numpy(),
            err_msg="count exponents at the top of a word")

        for count in self.LENGTHS:
            row = np.resize(exponents, count).astype(np.int32)
            np.testing.assert_array_equal(
                (2 ** self._device(row)).cpu().numpy(),
                (2 ** self._host(row)).cpu().numpy(),
                err_msg=f"count exponents at {count}")

    # Counts for both operands, and the numbers a power names where the
    # hardware answer runs out: a base of nothing, an exponent of nothing, and
    # the two bases that are their own reciprocals against an exponent that has
    # no end.

    COUNT_BASES = np.array([1, -1, 2, -2, 3, -3, 10, -7, 5, -6],
                           dtype=np.int32)
    COUNT_POWERS = np.array([0, 0, 2, -1, 3, -2, 7, 3, 4, 5], dtype=np.int32)

    def test_a_pair_of_counts_stays_whole_the_whole_way_up(self):
        # An estimate of the answer in single precision is already half a count
        # away from it by the time the exponent reaches the top of a word, so
        # the exponent is spent over the word itself.
        bases = np.array([2, 2, 3, 3, 5, 10, 10, -6, 7], dtype=np.int32)
        powers = np.array([30, 15, 19, 12, 13, 7, 6, 5, 4], dtype=np.int32)
        counts = (bases.size,) + self.LENGTHS
        for count in counts:
            row_b = np.resize(bases, count).astype(np.int32)
            row_e = np.resize(powers, count).astype(np.int32)
            np.testing.assert_array_equal(
                tensorplay.pow(self._device(row_b),
                               self._device(row_e)).cpu().numpy(),
                tensorplay.pow(self._host(row_b),
                               self._host(row_e)).cpu().numpy(),
                err_msg=f"count pair at {count}")

        # One count raised to a whole row of them, and the same pair written
        # back over the base.
        np.testing.assert_array_equal(
            tensorplay.pow(self._device(self.COUNT_BASES),
                           3).cpu().numpy(),
            tensorplay.pow(self._host(self.COUNT_BASES), 3).cpu().numpy(),
            err_msg="count base with a count uniform")

        written = self._device(self.COUNT_BASES.copy())
        written.pow_(self._device(self.COUNT_POWERS))
        np.testing.assert_array_equal(
            written.cpu().numpy(),
            tensorplay.pow(self._host(self.COUNT_BASES),
                           self._host(self.COUNT_POWERS)).cpu().numpy(),
            err_msg="count pair written back")

    def test_a_fraction_of_a_count_keeps_only_its_own_reciprocal(self):
        # Below nothing the answer is a fraction, and only the two bases that
        # are their own reciprocals have a value to keep: the one word stays
        # itself, the negative one turns with whether the exponent is odd or
        # even, and everything wider -- a nothing base included -- is nothing.
        bases = np.array([1, -1, -1, -1, 2, -2, 10, 0, 0, 1], dtype=np.int32)
        powers = np.array([-3, -1, -2, -3, -1, -3, -2, -1, -3, -4],
                          dtype=np.int32)
        want = np.array([1, -1, 1, -1, 0, 0, 0, 0, 0, 1], dtype=np.int32)
        for count in (bases.size,) + self.LENGTHS:
            row_b = np.resize(bases, count).astype(np.int32)
            row_e = np.resize(powers, count).astype(np.int32)
            np.testing.assert_array_equal(
                tensorplay.pow(self._device(row_b),
                               self._device(row_e)).cpu().numpy(),
                np.resize(want, count).astype(np.int32),
                err_msg=f"count fractions at {count}")

    def test_a_power_names_what_its_edge_of_a_word_means(self):
        bases = np.array([0.0, 0.0, 0.0, np.nan, np.inf, 1.0, 1.0, -1.0, -1.0,
                          2.0, 0.5, -2.0], dtype=np.float32)
        powers = np.array([0.0, 2.0, -1.0, 0.0, 0.0, np.inf, -np.inf, np.inf,
                           -np.inf, np.inf, np.inf, 0.0], dtype=np.float32)
        for dtype in (np.float32, np.float16):
            for count in (bases.size,) + self.LENGTHS:
                row_b = np.resize(bases, count).astype(dtype)
                row_e = np.resize(powers, count).astype(dtype)
                np.testing.assert_allclose(
                    tensorplay.pow(self._device(row_b),
                                   self._device(row_e)).cpu().numpy(),
                    tensorplay.pow(self._host(row_b),
                                   self._host(row_e)).cpu().numpy(),
                    rtol=1e-3, atol=1e-3, err_msg=f"power corners at {count}")

        # The same corners once the base and once the exponent is the number
        # carried as a uniform, which reaches the reflected and the linear
        # twins of both orders.
        for dtype in (np.float32, np.float16):
            row_e = self._device(np.resize(powers, 100001).astype(dtype))
            np.testing.assert_allclose(
                tensorplay.pow(0.0, row_e).cpu().numpy(),
                tensorplay.pow(0.0, self._host(np.resize(powers, 100001)
                                               .astype(dtype))).cpu().numpy(),
                rtol=1e-3, atol=1e-3, err_msg="a nothing base as a uniform")
            row_b = self._device(np.resize(bases, 100001).astype(dtype))
            np.testing.assert_allclose(
                tensorplay.pow(row_b, 0.0).cpu().numpy(),
                tensorplay.pow(self._host(np.resize(bases, 100001)
                                          .astype(dtype)), 0.0).cpu().numpy(),
                rtol=1e-3, atol=1e-3, err_msg="a nothing exponent as a uniform")

        written = self._device(bases.copy())
        written.pow_(self._device(powers))
        np.testing.assert_allclose(
            written.cpu().numpy(),
            tensorplay.pow(self._host(bases), self._host(powers)).cpu().numpy(),
            rtol=1e-3, atol=1e-3, err_msg="corners written back")


class TestVulkanScatterValue(unittest.TestCase):
    # Writing one number wherever a list of codes says so is a gather read
    # backwards: the walk belongs to the list, while the stores belong to the
    # destination.  A texel or a word carries four lanes and one store replaces
    # all of them, so a store made for the lane a code names would drop the
    # lanes that code never mentioned.  The first pass marks the destinations
    # the list names and the second reads a whole group at a time, keeping the
    # unmarked lanes as they were.

    LENGTHS = (100001, 100003, 200003)

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _host(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload))

    @staticmethod
    def _fill(payload, dim, codes, value):
        # Each coordinate of the list picks a row along the scattered axis of
        # the same coordinates in the destination.  A list may be narrower than
        # the destination anywhere but along that axis, so the walk belongs to
        # the list rather than to the destination.
        out = payload.copy()
        axis = dim if dim >= 0 else dim + payload.ndim
        for at in np.ndindex(codes.shape):
            destination = list(at)
            destination[axis] = int(codes[at])
            out[tuple(destination)] = value
        return out

    def _check_written(self, base_np, codes, dim, value, dtype=np.float32):
        payload = base_np.astype(dtype)
        want = self._fill(payload, dim, codes, value).astype(np.float32)
        got = self._device(payload).scatter_(
            dim, self._device(codes), value).cpu().numpy().astype(np.float32)
        np.testing.assert_allclose(
            got, want, rtol=1e-3, atol=1e-3,
            err_msg=f"written into a {payload.shape} destination along axis {dim}")
        # The copy on top of the destination is what the out-of-place call has to
        # return, so both orders of the same write are covered.
        kept = self._device(payload)
        held = tensorplay.scatter(kept, dim, self._device(codes), value)
        np.testing.assert_allclose(
            held.cpu().numpy().astype(np.float32), want,
            rtol=1e-3, atol=1e-3, err_msg="out-of-place write")
        np.testing.assert_allclose(
            kept.cpu().numpy().astype(np.float32), payload.astype(np.float32),
            rtol=1e-3, atol=1e-3, err_msg="the destination left alone")

    def test_a_code_names_one_lane_of_the_destination(self):
        # The last axis is the width of a texel, so the lanes of a group are
        # neighbours in the list of codes: a store that answered for the whole
        # group would spread one code over its neighbours.
        rng = np.random.default_rng(31)
        base = np.zeros((2, 6), np.float32)
        self._check_written(base, np.array([[0, 2, 4], [1, 3, 5]], np.int64),
                            1, -1.0)
        self._check_written(base, np.array([[0, 2, 4], [1, 3, 5]], np.int32),
                            1, 7.5)
        # A code naming the same lane twice is still that lane; two codes of a
        # group naming different lanes have to keep both.
        self._check_written(np.zeros((1, 8), np.float32),
                            np.array([[1, 1, 2, 2]], np.int64), 1, 5.0)
        self._check_written(np.ones((4, 7), np.float32),
                            np.array([[0, 6], [1, 5], [2, 4], [3, 0]], np.int32),
                            1, 6.0)
        # A width that is not a multiple of four ends in a partly used group.
        self._check_written(
            np.zeros((3, 5), np.float32),
            rng.integers(0, 5, (3, 4)).astype(np.int64), 1, 0.25)
        # A negative axis counts from the back, as the destination is walked.
        self._check_written(
            np.zeros((3, 4), np.float32),
            np.array([[0, 1], [2, 3], [0, 2]], np.int64), -1, 2.0)

    def test_a_code_moves_a_whole_inner_span(self):
        # Along an axis other than the last a code stands for a row of the
        # destination, so every element the span reaches is named while the
        # spans it skipped keep their values.
        rng = np.random.default_rng(32)
        self._check_written(
            np.zeros((2, 5, 3), np.float32),
            np.array([[[0, 1], [2, 3]], [[1, 1], [0, 2]]], np.int64), 1, 9.0)
        self._check_written(
            np.zeros((2, 3, 4, 5), np.float32),
            rng.integers(0, 3, (1, 3, 4, 5)).astype(np.int64), 1, 4.0)
        # A list narrower than the destination along the scattered axis leaves
        # the rows nobody names, but it still has to reach the right rows.
        self._check_written(
            np.zeros((2, 4, 6), np.float32),
            np.array([[[0, 2], [1, 3]], [[3, 1], [0, 2]]], np.int64), 1, -3.0)
        for axis in (0, 1, 2):
            self._check_written(
                rng.normal(size=(3, 5, 4)).astype(np.float32),
                rng.integers(0, (3, 5, 4)[axis], (3, 5, 4)).astype(np.int64),
                axis, 1.5)

    def test_a_half_plane_keeps_its_neighbour(self):
        # Two half values share a word of linear storage and four share a texel,
        # so a write that answered for the group would rewrite the half beside
        # the one asked for.
        rng = np.random.default_rng(33)
        self._check_written(
            np.zeros((2, 6), np.float32),
            np.array([[0, 2, 4], [1, 3, 5]], np.int64), 1, -1.5,
            dtype=np.float16)
        self._check_written(
            rng.normal(size=(3, 5, 4)).astype(np.float32),
            rng.integers(0, 4, (3, 5, 4)).astype(np.int64), 2, 0.75,
            dtype=np.float16)

    def test_a_long_destination_writes_through_linear_storage(self):
        # Past the reach of an image the destination rides words of four lanes,
        # so the group a store replaces is a word rather than a texel and the
        # marking pass counts elements of the flat payload.
        rng = np.random.default_rng(34)
        for dtype in (np.float32, np.float16):
            for width in self.LENGTHS:
                base = np.zeros((2, width), np.float32)
                codes = np.sort(
                    rng.integers(0, width, (2, 5)), axis=1).astype(np.int64)
                self._check_written(
                    base, codes, 1, 3.25, dtype=dtype)
        # The last lane of a ragged width sits in a partly used word.
        self._check_written(
            np.ones((1, 100003), np.float32),
            np.array([[0, 100001, 100002]], np.int64), 1, -0.5)

    def test_a_row_of_a_wide_destination_writes_every_group_it_spans(self):
        # A decode step bans tokens by naming a row of the logits, which is the
        # whole width of the destination: the write has to reach the last texel
        # and the last word of a width that is not a multiple of four.
        rng = np.random.default_rng(35)
        for width in self.LENGTHS:
            base = np.ones((1, width), np.float32) * 0.5
            codes = np.sort(rng.integers(0, width, (1, 17)),
                            axis=1).astype(np.int64)
            self._check_written(base, codes, 1, -1e4)

    def test_a_code_outside_the_rows_names_nothing(self):
        # Reaching past the end of an axis is refused on the host but left
        # untouched here, the same policy the row lookups already follow.
        base = np.zeros((1, 8), np.float32)
        codes = np.array([[0, 2, 9, -1]], np.int64)
        want = base.copy()
        want[0, 0] = want[0, 2] = 5.0
        got = self._device(base.copy()).scatter_(
            1, self._device(codes), 5.0).cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6,
                                   err_msg="a code outside the rows")

    def test_two_writes_over_one_destination_do_not_stay_marked(self):
        # The marks are a temporary of the destination's size, so a second write
        # that reuses the storage of the first has to clear it: a leftover mark
        # would keep writing where the earlier list said so.
        base = np.zeros((3, 8), np.float32)
        first = np.array([[0, 1], [2, 3], [4, 5]], np.int64)
        second = np.array([[7], [6], [0]], np.int64)
        payload = self._device(base.copy())
        payload.scatter_(1, self._device(first), 1.0)
        payload.scatter_(1, self._device(second), 2.0)
        want = self._fill(self._fill(base, 1, first, 1.0), 1, second, 2.0)
        np.testing.assert_allclose(
            payload.cpu().numpy(), want, rtol=0, atol=1e-6,
            err_msg="two writes over one destination")

    def test_a_destination_without_lanes_is_left_alone(self):
        empty = self._device(np.zeros((0, 4), np.float32))
        self.assertEqual(
            tuple(empty.scatter_(
                1, self._device(np.zeros((0, 4), np.int64)), 3.0
            ).shape), (0, 4))
        kept = self._device(np.zeros((2, 3), np.float32))
        np.testing.assert_allclose(
            kept.scatter_(1, self._device(np.zeros((2, 0), np.int64)),
                         3.0).cpu().numpy(),
            np.zeros((2, 3), np.float32), rtol=0, atol=0,
            err_msg="an empty list writes nothing")

    def test_an_impossible_shape_is_refused(self):
        payload = self._device(np.zeros((2, 4), np.float32))
        with self.assertRaisesRegex(RuntimeError, "same rank as the destination"):
            payload.scatter_(1, self._device(np.zeros((4,), np.int64)), 1.0)
        with self.assertRaisesRegex(RuntimeError, "wider than the destination"):
            payload.scatter_(1, self._device(np.zeros((3, 2), np.int64)), 1.0)
        with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
            payload.scatter_(2, self._device(np.zeros((2, 4), np.int64)), 1.0)
        with self.assertRaisesRegex(RuntimeError, "Int32 or Int64"):
            payload.scatter_(1, self._device(np.zeros((2, 2), np.float32)), 1.0)


class TestVulkanScatterSrc(unittest.TestCase):
    # Moving a payload differs from writing a constant in one respect: the entry
    # that names a destination carries its own number, read at the entry's own
    # position in the list.  The mark pass of a constant write already records
    # that position, so the same two passes run with the payload bound beside
    # the marks and the apply pass takes a value instead of a uniform.

    LENGTHS = (100001, 100003)

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _values(count):
        # Numbers that differ from each other and from zero, so a lane that kept
        # its own value or took a neighbour's cannot pass for the one asked for.
        return (np.arange(count, dtype=np.float32) % 23) - 11.0

    @staticmethod
    def _permutation(shape, axis, order):
        # Codes that name each destination at most once along the scattered axis:
        # two entries naming one destination leave the winner to whichever store
        # lands last, which is what a scattered write of a payload answers to.
        rng = np.random.default_rng(order)
        out = np.empty(shape, np.int64)
        for at in np.ndindex(
                *[size for index, size in enumerate(shape) if index != axis]):
            view = list(at)
            view.insert(axis, slice(None))
            out[tuple(view)] = rng.permutation(shape[axis])
        return out

    def _move(self, payload, dim, codes, values):
        out = payload.copy()
        axis = dim if dim >= 0 else dim + payload.ndim
        held = np.ascontiguousarray(
            np.broadcast_to(np.asarray(values), codes.shape)).reshape(-1)
        position = 0
        for at in np.ndindex(codes.shape):
            destination = list(at)
            destination[axis] = int(codes[at])
            out[tuple(destination)] = held[position]
            position += 1
        return out

    def _check_moved(self, base_np, codes, dim, values, dtype=np.float32):
        payload = base_np.astype(dtype)
        want = self._move(payload, dim, codes,
                          np.asarray(values).astype(dtype))
        kept = self._device(payload)
        got = kept.scatter_(
            dim, self._device(codes), self._device(values)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float32), want.astype(np.float32), rtol=1e-3,
            atol=1e-3,
            err_msg=f"moved into a {payload.shape} destination along axis {dim}")
        # Out-of-place leaves the destination alone and answers with the copy.
        held = self._device(payload)
        moved = tensorplay.scatter(
            held, dim, self._device(codes), self._device(values))
        np.testing.assert_allclose(
            moved.cpu().numpy().astype(np.float32), want.astype(np.float32),
            rtol=1e-3, atol=1e-3, err_msg="moved out-of-place")
        np.testing.assert_allclose(
            held.cpu().numpy().astype(np.float32), payload.astype(np.float32),
            rtol=1e-3, atol=1e-3, err_msg="the destination left alone")

    def test_each_entry_carries_its_own_number(self):
        # Along the width a code names one lane of a four-lane group, so the
        # value of the entry has to reach that lane and no other.
        self._check_moved(
            np.zeros((2, 6), np.float32),
            np.array([[0, 2, 4], [1, 3, 5]], np.int64), 1,
            self._values(6).reshape(2, 3))
        self._check_moved(
            np.ones((4, 7), np.float32),
            np.array([[0, 6], [1, 5], [2, 4], [3, 0]], np.int32), 1,
            self._values(8).reshape(4, 2))
        # A negative axis counts from the back.
        self._check_moved(
            np.ones((3, 4), np.float32),
            np.array([[0, 1], [2, 3], [0, 2]], np.int64), -1,
            self._values(6).reshape(3, 2))
        # A width that is not a multiple of four ends in a partly used group.
        self._check_moved(
            np.zeros((3, 5), np.float32),
            self._permutation((3, 4), 1, 41), 1, self._values(12).reshape(3, 4))

    def test_an_entry_moves_a_whole_inner_span(self):
        # An axis other than the last has a code stand for a row of the
        # destination, so the entries of one span carry consecutive values while
        # the spans nobody names keep theirs.
        self._check_moved(
            np.zeros((2, 5, 3), np.float32),
            np.array([[[0, 1], [2, 3]], [[1, 0], [3, 2]]], np.int64), 1,
            self._values(8).reshape(2, 2, 2))
        # A list narrower than the destination along an outer axis still has to
        # reach the rows of the plane it covers.
        self._check_moved(
            np.zeros((2, 3, 4, 5), np.float32),
            self._permutation((1, 3, 4, 5), 1, 42), 1,
            self._values(60).reshape(1, 3, 4, 5))
        for axis in (0, 1, 2):
            self._check_moved(
                np.zeros((3, 5, 4), np.float32),
                self._permutation((3, 5, 4), axis, 43 + axis), axis,
                self._values(60).reshape(3, 5, 4))

    def test_a_half_payload_moves_between_halves(self):
        # Two halves share a word and four share a texel, so a value that changed
        # width on its way in or out of the group would show up as a neighbour
        # rewritten rather than a lane missed.
        self._check_moved(
            np.zeros((2, 6), np.float32),
            np.array([[0, 2, 4], [1, 3, 5]], np.int64), 1,
            self._values(6).reshape(2, 3), dtype=np.float16)
        self._check_moved(
            np.zeros((3, 5, 4), np.float32),
            self._permutation((3, 5, 4), 2, 44), 2,
            self._values(60).reshape(3, 5, 4), dtype=np.float16)

    def test_a_long_destination_moves_through_linear_storage(self):
        # Past the reach of an image both passes count elements of the flat
        # payload, and a ragged count ends in a partly used word.
        for width in self.LENGTHS:
            codes = np.array(
                [[0, 1, width - 3, width - 2, width - 1],
                 [3, 4, 5, 6, 7]], np.int64)
            self._check_moved(
                np.zeros((2, width), np.float32), codes, 1,
                self._values(10).reshape(2, 5))
            self._check_moved(
                np.zeros((2, width), np.float32), codes, 1,
                self._values(10).reshape(2, 5), dtype=np.float16)
        self._check_moved(
            np.ones((1, 100001), np.float32) * 0.5,
            np.array([[0, 50000, 100000]], np.int64), 1, self._values(3))

    def test_the_payload_may_reach_the_list_from_further_off(self):
        # A single number repeats over every entry, a number per row repeats over
        # the entries of that row, an axis the payload leaves out stands for the
        # outer ones, and a payload of the narrower width is read as the width the
        # destination carries.
        codes = np.array([[0, 2, 4], [1, 3, 5]], np.int64)
        base = np.zeros((2, 6), np.float32)
        self._check_moved(base, codes, 1, np.array([[7.0]], np.float32))
        self._check_moved(base, codes, 1, np.array([[3.0], [5.0]], np.float32))
        self._check_moved(base, codes, 1, self._values(3))
        self._check_moved(
            base, codes, 1, self._values(6).reshape(2, 3).astype(np.float16))

    def test_a_source_taken_from_the_plane_being_written_keeps_its_values(self):
        # Reading rows out of the very destination the stores rewrite would bind
        # one allocation in both directions at once, so the pass could answer for
        # a lane out of the bytes it had just changed.
        base = np.arange(24, dtype=np.float32).reshape(4, 6)
        codes = np.array(
            [[5, 4, 3], [0, 1, 2], [2, 3, 4], [5, 0, 1]], np.int64)
        payload = self._device(base.copy())
        got = payload.scatter_(
            1, self._device(codes), payload[:, :3]).cpu().numpy()
        np.testing.assert_allclose(
            got, self._move(base, 1, codes, base[:, :3]), rtol=0, atol=1e-6,
            err_msg="a source read out of the plane being written")

    def test_a_payload_that_cannot_cover_the_list_is_refused(self):
        payload = self._device(np.zeros((2, 4), np.float32))
        with self.assertRaisesRegex(RuntimeError, "more axes than the list"):
            payload.scatter_(
                1, self._device(np.zeros((2, 3), np.int64)),
                self._device(np.zeros((2, 3, 1), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "reach the list at dimension"):
            payload.scatter_(
                1, self._device(np.zeros((2, 3), np.int64)),
                self._device(np.zeros((2, 2), np.float32)))


class TestVulkanWhereScalars(unittest.TestCase):
    # With both branches numbers, the condition is the only payload and its own
    # shape answers for the result.  A mask built this way is what a causal or a
    # padded attention window turns into the additive form, so the fraction the
    # lane takes has to arrive at the plane the logits are read in.

    LENGTHS = (100001, 100003)

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _mask(shape, order):
        return np.random.default_rng(order).random(shape) > 0.5

    def test_a_fraction_is_taken_where_the_truth_is(self):
        for shape in ((3, 8), (7,), (2, 3, 4, 5)):
            condition = self._mask(shape, 51)
            want = np.where(condition, 2.5, -1.5)
            got = tensorplay.where(self._device(condition), 2.5, -1.5)
            self.assertEqual(str(got.dtype), "tensorplay.float32")
            np.testing.assert_array_equal(
                got.cpu().numpy(), want.astype(np.float32),
                err_msg=f"a fraction pair over {shape}")

    def test_a_rejected_lane_carries_the_number_that_bans_it(self):
        # The shape a causal mask turns into: one lane takes zero, the other a
        # number no softmax can raise above the rest.
        condition = self._mask((2, 4, 6, 6), 52)
        got = tensorplay.where(self._device(condition), 0.0, -float("inf"))
        want = np.where(condition, 0.0, -np.inf).astype(np.float32)
        read = got.cpu().numpy()
        self.assertTrue(np.array_equal(np.isinf(read), np.isinf(want)),
                        msg="which lanes were banned")
        self.assertTrue(np.array_equal(np.isneginf(read), np.isneginf(want)),
                        msg="which side of infinity")
        np.testing.assert_array_equal(
            np.nan_to_num(read, neginf=-1e30),
            np.nan_to_num(want, neginf=-1e30), err_msg="a banned lane")

    def test_a_count_pair_is_carried_as_an_integer_plane(self):
        condition = self._mask((4, 9), 53)
        got = tensorplay.where(self._device(condition), 7, 3)
        self.assertEqual(str(got.dtype), "tensorplay.int32")
        np.testing.assert_array_equal(
            got.cpu().numpy(), np.where(condition, 7, 3).astype(np.int32),
            err_msg="a count pair")

    def test_a_mixed_pair_is_carried_as_the_wider_of_the_two(self):
        # A count beside a fraction has no integer answer, so the plane the
        # numbers land in is the one that can hold both.
        condition = self._mask((5, 6), 54)
        got = tensorplay.where(self._device(condition), 1, 2.0)
        self.assertEqual(str(got.dtype), "tensorplay.float32")
        np.testing.assert_array_equal(
            got.cpu().numpy(),
            np.where(condition, 1, 2.0).astype(np.float32),
            err_msg="a count and a fraction")

    def test_a_long_condition_selects_through_linear_storage(self):
        # Past the reach of an image the lanes live in words of four, so a lane
        # has to take its own number out of the pair rather than the group's.
        for width in self.LENGTHS:
            condition = self._mask((2, width), 55)
            got = tensorplay.where(self._device(condition), 4.5, -6.5)
            np.testing.assert_array_equal(
                got.cpu().numpy(),
                np.where(condition, 4.5, -6.5).astype(np.float32),
                err_msg=f"a fraction pair over {width} lanes")

    def test_a_single_lane_answers_for_itself(self):
        for condition in (np.array([True]), np.array([False])):
            got = tensorplay.where(self._device(condition), 1.0, 0.0)
            self.assertEqual(float(got.cpu().numpy()[0]), 1.0 if condition[0] else 0.0)


class TestVulkanIntegralSum(unittest.TestCase):
    # A plane that counts rather than measures has no reduced type to carry its
    # own total: the answer to a sum over a mask is the number of lanes it
    # settles on, and it has to arrive as a whole number wider than the lane.
    # Every step of a decode asks for one of these (how much of the window is
    # real, how many lanes a check banned), so a round trip that loses a lane's
    # number on the way in is a bug the model would answer with.

    LENGTHS = (100001, 100003)

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _truth(shape, order):
        return np.random.default_rng(order).random(shape) > 0.4

    @staticmethod
    def _held(shape, order):
        rng = np.random.default_rng(order)
        return rng.integers(-1000, 1000, shape).astype(np.int32)

    def test_a_truth_plane_totals_the_lanes_it_sets(self):
        truth = self._truth((3, 8), 61)
        got = self._device(truth).sum()
        self.assertEqual(str(got.dtype), "tensorplay.int64")
        self.assertEqual(int(got.cpu().numpy()), int(truth.sum()),
                         "a mask's total is its set lanes")

    def test_a_counted_plane_carries_its_own_total(self):
        held = self._held((4, 5), 62)
        got = self._device(held).sum()
        self.assertEqual(str(got.dtype), "tensorplay.int64")
        self.assertEqual(int(got.cpu().numpy()), int(held.sum(dtype=np.int64)),
                         "a plane of whole numbers")

    def test_a_row_of_truth_counts_alone(self):
        truth = self._truth((2, 3, 4), 63)
        got = self._device(truth).sum(1)
        self.assertEqual(tuple(got.shape), (2, 4))
        np.testing.assert_array_equal(
            got.cpu().numpy(), truth.sum(axis=1, dtype=np.int64),
            err_msg="a middle axis of a truth plane")

    def test_a_truth_plane_keeps_the_axis_it_collapsed(self):
        truth = self._truth((3, 6), 64)
        got = self._device(truth).sum(-1, keepdim=True)
        self.assertEqual(tuple(got.shape), (3, 1))
        np.testing.assert_array_equal(
            got.cpu().numpy(),
            truth.sum(axis=-1, dtype=np.int64, keepdims=True),
            err_msg="a kept axis")

    def test_a_long_truth_plane_is_totalled_through_linear_storage(self):
        # Past the reach of an image the lanes live in words, so the total of a
        # whole window has to be built without the walk that reads a texture.
        for width in self.LENGTHS:
            truth = self._truth((2, width), 65)
            got = self._device(truth).sum(-1)
            np.testing.assert_array_equal(
                got.cpu().numpy(), truth.sum(axis=-1, dtype=np.int64),
                err_msg=f"a row of {width} lanes")

    def test_a_lane_past_the_reach_of_a_fraction_keeps_its_number(self):
        # Whole numbers past what single precision names exactly have to arrive
        # whole, which is why the walk reads the plane as its own width.
        held = np.array(
            [2**24 + 1, 2**24 + 3, -(2**24 + 5), 2**24], np.int32)
        got = self._device(held).sum()
        self.assertEqual(int(got.cpu().numpy()), int(held.sum(dtype=np.int64)),
                         "a total no fraction could hold")

    def test_a_wide_lane_counted_across_axes_keeps_its_number(self):
        held = (np.arange(6 * 7, dtype=np.int32).reshape(6, 7) * 4_000_000)
        got = self._device(held).sum(1)
        np.testing.assert_array_equal(
            got.cpu().numpy(), held.sum(axis=1, dtype=np.int64),
            err_msg="rows of numbers past a fraction's reach")

    def test_a_fraction_is_named_by_the_caller(self):
        truth = self._truth((5, 7), 66)
        got = self._device(truth).sum(dtype=tensorplay.float32)
        self.assertEqual(str(got.dtype), "tensorplay.float32")
        self.assertAlmostEqual(
            float(got.cpu().numpy()), float(truth.sum()), places=4,
            msg="a mask's total asked for as a fraction")

    def test_a_count_of_a_truth_plane_answers_the_same_as_its_total(self):
        truth = self._truth((4, 9, 2), 67)
        counted = self._device(truth).count_nonzero(-1)
        np.testing.assert_array_equal(
            counted.cpu().numpy(),
            truth.sum(axis=-1, dtype=np.int64),
            err_msg="a count over the last axis of a mask")

    def test_a_plane_without_lanes_totals_to_nothing(self):
        for payload in (np.zeros((0, 4), np.bool_), np.zeros((0, 4), np.int32)):
            got = self._device(payload).sum()
            self.assertEqual(str(got.dtype), "tensorplay.int64")
            self.assertEqual(int(got.cpu().numpy()), 0,
                             "nothing adds to nothing")

    def test_a_width_the_backend_cannot_name_is_refused(self):
        truth = self._truth((2, 3), 68)
        with self.assertRaisesRegex(RuntimeError, "Int64, Float32"):
            self._device(truth).sum(dtype=tensorplay.int16)

    def test_a_plane_of_wide_lanes_totals_in_the_pairs_it_holds(self):
        # A lane that is already the wide word has no texel to read and no
        # single-precision register to fold into, so its total walks the pair of
        # words the payload is carried in and answers whole.
        held = np.array([2 ** 40, 2 ** 40, -(2 ** 40), 2 ** 32], dtype=np.int64)
        got = self._device(held).sum()
        self.assertEqual(str(got.dtype), "tensorplay.int64")
        self.assertEqual(int(got.cpu().numpy()), int(held.sum(dtype=np.int64)),
                         "a total of wide lanes lost a word on the way in")
        total = int(held.sum(dtype=np.int64))
        narrowed = self._device(held).sum(dtype=tensorplay.int32)
        self.assertEqual(str(narrowed.dtype), "tensorplay.int32")
        low = total & 0xFFFFFFFF
        self.assertEqual(
            int(narrowed.cpu().numpy()),
            low - (1 << 32) if low >= (1 << 31) else low,
            "a wide total narrowed to the wrong word")


class TestVulkanCountingFolds(unittest.TestCase):
    # How many lanes of a plane stand, and whether a span stands at all: a count
    # and the two truth folds answer every check a decode makes about a mask, a
    # window of real tokens or a lane a ban settled on.  A plane whose shape has no
    # image form lives in words instead of texels, and a lane carried in a width the
    # walk does not read has to reach the answer through the width it does.

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _host(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload))

    @staticmethod
    def _truth(shape, order):
        return np.random.default_rng(order).random(shape) > 0.4

    @staticmethod
    def _measured(shape, order):
        # A row long enough that no image axis holds it, so the payload answers
        # from linear storage.
        rng = np.random.default_rng(order)
        return (rng.standard_normal(shape) * 8).astype(np.float32)

    def _compare(self, got, wanted, message):
        self.assertEqual(str(got.dtype), str(wanted.dtype), message)
        self.assertEqual(tuple(got.shape), tuple(wanted.shape), message)
        np.testing.assert_array_equal(
            got.cpu().numpy(), wanted.cpu().numpy(), err_msg=message)

    def test_a_count_names_the_lanes_of_every_width(self):
        planes = (
            np.array([[0.0, 1.5, -2.0, 0.0], [3.0, 0.0, 0.0, 4.5]], np.float32),
            np.array([[0.0, 1.5, -2.0, 0.0], [3.0, 0.0, 0.0, 4.5]], np.float16),
            np.array([[0, 3, -1, 0], [2, 0, 0, 5]], np.int32),
            np.array([[0, 3, -1, 0], [2, 0, 0, 5]], np.int64),
            np.array([[0, 3, -1, 0], [2, 0, 0, 5]], np.int8),
            np.array([[True, False, True, True], [False, False, True, False]]),
        )
        for plane in planes:
            self._compare(
                self._device(plane).count_nonzero(),
                self._host(plane).count_nonzero(),
                f"a whole plane counted, {plane.dtype}")
            for axis in (0, 1, -1):
                self._compare(
                    self._device(plane).count_nonzero(axis),
                    self._host(plane).count_nonzero(axis),
                    f"a plane counted over axis {axis}, {plane.dtype}")

    def test_the_truth_folds_answer_for_every_element_type(self):
        planes = (
            (np.random.default_rng(71).standard_normal((3, 5)) * 4).astype(np.float32),
            (np.random.default_rng(72).standard_normal((3, 5)) * 4).astype(np.float16),
            np.random.default_rng(73).integers(-4, 5, (3, 5)).astype(np.int32),
            np.random.default_rng(74).integers(-4, 5, (3, 5)).astype(np.int64),
            np.random.default_rng(75).integers(-4, 5, (3, 5)).astype(np.int8),
            self._truth((3, 5), 76),
        )
        for plane in planes:
            for fold in ("all", "any"):
                self._compare(
                    getattr(self._device(plane), fold)(),
                    getattr(self._host(plane), fold)(),
                    f"a whole plane through {fold}, {plane.dtype}")
                for axis, keepdim in ((0, False), (1, True), (-1, False)):
                    self._compare(
                        getattr(self._device(plane), fold)(axis, keepdim),
                        getattr(self._host(plane), fold)(axis, keepdim),
                        f"a plane through {fold} over axis {axis}, {plane.dtype}")

    def test_a_row_past_the_reach_of_an_image_counts_from_linear_storage(self):
        for width in (100001, 30000):
            truth = self._truth((2, width), 77)
            self._compare(
                self._device(truth).count_nonzero(-1),
                self._host(truth).count_nonzero(-1),
                f"a row of {width} truth lanes")
            self._compare(
                self._device(truth).all(-1),
                self._host(truth).all(-1),
                f"a fold over {width} truth lanes")
            measured = self._measured((1, 1, 1, width), 78)
            self._compare(
                self._device(measured).count_nonzero(-1),
                self._host(measured).count_nonzero(-1),
                f"a measured row of {width} lanes")
            self._compare(
                self._device(measured).any(0),
                self._host(measured).any(0),
                f"a measured plane through any, {width} lanes")

    def test_a_signed_word_past_the_reach_of_a_fraction_keeps_its_total(self):
        # Linear storage used to add a plane of whole numbers through single
        # precision, which rounded each lane on the way in and narrowed the
        # running total to the end of a signed word.
        held = (np.arange(30000, dtype=np.int32) * 4096).reshape(1, 1, 1, 30000)
        got = self._device(held).sum(-1)
        self.assertEqual(str(got.dtype), "tensorplay.int64")
        self.assertEqual(
            int(got.cpu().numpy()[0, 0, 0]),
            int(held.sum(dtype=np.int64)),
            "a total no fraction could hold")
        counted = self._device(held).count_nonzero()
        self.assertEqual(
            int(counted.cpu().numpy()), int(np.count_nonzero(held)))

    def test_a_span_with_no_lane_answers_the_fold_without_a_lane(self):
        for plane in (np.zeros((0, 3), np.float32),
                      np.zeros((0, 3), np.bool_),
                      np.zeros((2, 0, 3), np.float32)):
            axis = 0 if plane.shape[0] == 0 else 1
            self._compare(
                self._device(plane).count_nonzero(axis),
                self._host(plane).count_nonzero(axis),
                f"a count over a span of no lanes, {plane.shape}")
            for fold in ("all", "any"):
                self._compare(
                    getattr(self._device(plane), fold)(axis),
                    getattr(self._host(plane), fold)(axis),
                    f"{fold} over a span of no lanes, {plane.shape}")

    def test_a_lane_that_is_not_a_number_stands(self):
        plane = np.array(
            [[np.nan, 0.0, -0.0, np.inf], [-np.inf, 2.5, 0.0, np.nan]],
            np.float32)
        self._compare(
            self._device(plane).count_nonzero(),
            self._host(plane).count_nonzero(),
            "a plane of special lanes counted")
        for fold in ("all", "any"):
            self._compare(
                getattr(self._device(plane), fold)(),
                getattr(self._host(plane), fold)(),
                f"a plane of special lanes through {fold}")
        wide = self._measured((1, 1, 1, 30000), 79)
        wide[0, 0, 0, 3] = np.nan
        wide[0, 0, 0, 7] = 0.0
        self._compare(
            self._device(wide).count_nonzero(-1),
            self._host(wide).count_nonzero(-1),
            "a lane that is not a number in a row without an image form")


class TestVulkanIntegerExtrema(unittest.TestCase):
    # The largest row code a window holds, the end of a position walk, the
    # widest count a page table names: an extremum of a counted plane answers
    # in the width the plane itself names.  Read through single precision a lane
    # past 2**24 arrives as a different number than it left, and a wide lane has
    # no single word that carries it, so these folds walk the signed word and
    # the pair of words the payload actually lives in.

    INT32_LOW = -2147483647 - 1
    INT32_HIGH = 2147483647
    INT64_LOW = -9223372036854775807 - 1
    INT64_HIGH = 9223372036854775807

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _host(answer):
        return answer.cpu().numpy()

    def test_the_ends_of_a_signed_word_survive_the_fold(self):
        # Both ends of the range in one row: a fold that borrowed a wider
        # measurement would have to name a number the word cannot hold.
        held = np.array([7, -3, self.INT32_LOW, 0, self.INT32_HIGH, 2],
                        np.int32)
        folded = self._device(held)
        self.assertEqual(str(folded.max().dtype), "tensorplay.int32")
        self.assertEqual(int(self._host(folded.max())), self.INT32_HIGH)
        self.assertEqual(int(self._host(folded.min())), self.INT32_LOW)

    def test_a_signed_word_past_the_reach_of_a_fraction_keeps_its_number(self):
        held = np.array(
            [2**24 + 1, 2**24 + 3, -(2**24 + 5), 2**24, 2**24 + 2], np.int32)
        folded = self._device(held)
        self.assertEqual(int(self._host(folded.max())), 2**24 + 3,
                         "the largest lane, not the largest fraction")
        self.assertEqual(int(self._host(folded.min())), -(2**24 + 5))

    def test_the_ends_of_a_wide_lane_survive_the_fold(self):
        held = np.array(
            [self.INT64_LOW, 0, 12345, self.INT64_HIGH, -(2**62)], np.int64)
        folded = self._device(held)
        self.assertEqual(str(folded.max().dtype), "tensorplay.int64")
        self.assertEqual(int(self._host(folded.max())), self.INT64_HIGH)
        self.assertEqual(int(self._host(folded.min())), self.INT64_LOW)

    def test_a_wide_lane_that_differs_only_in_its_low_word_is_the_larger(self):
        # Two lanes sharing a high word are ordered by their low words, so a
        # fold that stopped at the top word would call these the same number.
        held = np.array(
            [2**32, 2**32 + 1, 2**32 - 1, -(2**32), -(2**32) + 1], np.int64)
        folded = self._device(held)
        self.assertEqual(int(self._host(folded.max())), 2**32 + 1)
        self.assertEqual(int(self._host(folded.min())), -(2**32))

    def test_a_fold_of_lanes_around_zero_orders_the_sign_right(self):
        # A biased high word puts a lane that borrows from the sign above every
        # lane that does not, and the walk starts at the row's own first element
        # rather than at a number a count could fall outside of.
        for held in (np.array([-1, -(2**32), 0, 1], np.int64),
                     np.array([-(2**32), -1, 1, 0], np.int64)):
            folded = self._device(held)
            np.testing.assert_array_equal(
                self._host(folded.max()), np.max(held),
                err_msg="the largest of a handful of counts")
            np.testing.assert_array_equal(
                self._host(folded.min()), np.min(held),
                err_msg="the smallest of a handful of counts")

    def test_a_counted_row_extremes_the_axis_it_names(self):
        rng = np.random.default_rng(91)
        held = rng.integers(-10_000, 10_000, (3, 5)).astype(np.int32)
        rows = self._device(held)
        np.testing.assert_array_equal(
            self._host(rows.amax(dim=[1])), np.max(held, axis=1),
            err_msg="the widest lane of each row")
        np.testing.assert_array_equal(
            self._host(rows.amin(dim=[0], keepdim=True)),
            np.min(held, axis=0, keepdims=True),
            err_msg="a kept axis of a counted plane")

    def test_a_fold_across_two_axes_of_a_counted_plane(self):
        rng = np.random.default_rng(92)
        held = rng.integers(-5_000_000, 5_000_000, (2, 3, 4)).astype(np.int32)
        folded = self._device(held)
        np.testing.assert_array_equal(
            self._host(folded.amax(dim=[0, 2])),
            np.max(held, axis=(0, 2)),
            err_msg="a span of two axes")

    def test_a_fold_over_a_channel_axis_drops_the_lanes_that_pad_the_texture(self):
        # Five channels fill two texels, so three of the eight lanes a walk
        # touches name no element.  A padded lane has to fold the identity of
        # the direction, or the answer comes from a lane that is not there.
        rng = np.random.default_rng(93)
        held = rng.integers(-2**30, 2**30, (2, 5, 3, 4)).astype(np.int32)
        folded = self._device(held)
        self.assertEqual(tuple(folded.amax(dim=[1], keepdim=True).shape),
                         (2, 1, 3, 4))
        np.testing.assert_array_equal(
            self._host(folded.amax(dim=[1])),
            np.max(held, axis=1),
            err_msg="the widest channel of a counted plane")
        np.testing.assert_array_equal(
            self._host(folded.amin(dim=[1])),
            np.min(held, axis=1),
            err_msg="the narrowest channel of a counted plane")

    def test_a_wide_fold_beyond_an_image_walks_the_words_it_lives_in(self):
        rng = np.random.default_rng(94)
        held = rng.integers(-2**60, 2**60, (2, 100_003)).astype(np.int64)
        folded = self._device(held)
        np.testing.assert_array_equal(
            self._host(folded.amax(dim=[1])), np.max(held, axis=1),
            err_msg="a row of wide lanes in linear storage")
        self.assertEqual(int(self._host(folded.max())), int(np.max(held)),
                         "the widest lane of a whole plane")

    def test_a_fold_of_one_long_row_walks_every_tile(self):
        # A whole-plane fold of a single row has no row to spread over, so it
        # folds fixed tiles and then the tile results.  The extremes sit on the
        # far side of a tile boundary and on the last lane, where the walk stops
        # short of the payload's end.
        length = 5000
        for place, value in ((257, np.int64(2**60)),
                             (4099, np.int64(-(2**60) - 1)),
                             (length - 1, np.int64(2**60 - 1)),
                             (256, np.int64(-(2**60) - 2))):
            held = np.zeros(length, np.int64)
            held[place] = value
            folded = self._device(held)
            self.assertEqual(
                int(self._host(folded.max())), int(np.max(held)),
                f"the widest lane with an extreme at {place}")
            self.assertEqual(
                int(self._host(folded.min())), int(np.min(held)),
                f"the narrowest lane with an extreme at {place}")

    def test_a_position_walk_answers_with_the_end_of_its_window(self):
        # The number a rotary walk reads to size its table: the last position of
        # a window of counted lanes, which is a whole number, not a fraction of
        # one that rounds at 2**24.  A walk may name its positions in either
        # width, so both folds are checked.
        for payload in (
                np.arange(0, 4096, dtype=np.int32),
                np.arange(0, 4096, dtype=np.int64)):
            folded = self._device(payload)
            self.assertEqual(
                int(self._host(folded.max())), 4095,
                f"the last position of a {payload.dtype} window")
            self.assertEqual(int(self._host(folded.min())), 0)

    def test_a_counted_plane_folds_the_way_a_host_fold_does(self):
        rng = np.random.default_rng(95)
        held = rng.integers(-10_000, 10_000, (8, 37)).astype(np.int32)
        folded = self._device(held)
        np.testing.assert_array_equal(
            self._host(folded.max()), np.max(held))
        np.testing.assert_array_equal(
            self._host(folded.min()), np.min(held))
        np.testing.assert_array_equal(
            self._host(folded.amin(dim=[])), np.min(held),
            err_msg="a whole plane folded through an empty axis list")

    def test_a_plane_without_lanes_has_no_fold(self):
        # A fold that would answer the whole of a plane has no lane to start
        # from, so it asks for the axis argument that would give it one instead
        # of answering a number no lane was read for.
        for payload in (np.zeros((0, 4), np.int32), np.zeros((0,), np.int64)):
            with self.subTest(shape=payload.shape):
                with self.assertRaisesRegex(
                        RuntimeError, "Expected reduction dim to be specified"):
                    self._device(payload).max()
                with self.assertRaisesRegex(
                        IndexError, "to have non-zero size"):
                    self._device(payload).max(0)

    def test_a_fold_that_names_a_position_is_refused_loudly(self):
        # The value of a counted plane is in hand here; the position that holds
        # it has no counted walk yet, so the pair is refused rather than
        # answered with a position read from the wrong width.
        held = self._device(np.arange(6, dtype=np.int64).reshape(2, 3))
        with self.assertRaisesRegex(RuntimeError, "index reductions"):
            held.max(0)

    def test_a_product_of_a_wide_plane_folds_the_pairs_it_holds(self):
        # A product of wide lanes has no single word to accumulate in either, so
        # it walks the pairs the payload is carried in.  The fold starts at the
        # identity it names, and a count past both words keeps the low bits a
        # count wrapped at its own width lands on.
        held = np.array([2, 3, 4, 5], dtype=np.int64)
        got = self._device(held).prod()
        self.assertEqual(str(got.dtype), "tensorplay.int64")
        self.assertEqual(int(got.cpu().numpy()), int(np.prod(held)),
                         "a small product of wide lanes")
        carried = np.array([(1 << 32) - 1, (1 << 32) - 1], dtype=np.int64)
        with np.errstate(over="ignore"):
            want = int(np.prod(carried, dtype=np.int64))
        got = self._device(carried).prod()
        self.assertEqual(int(got.cpu().numpy()), want,
                         "a product past both words lost its low bits")
        grid = np.arange(2, 8, dtype=np.int64).reshape(2, 3)
        folded = self._device(grid).prod(0)
        self.assertEqual(tuple(folded.shape), (3,))
        np.testing.assert_array_equal(
            np.asarray(folded.cpu()), np.prod(grid, axis=0, dtype=np.int64),
            err_msg="an axis product of wide lanes")


class TestVulkanAdvancedIndexing(unittest.TestCase):
    # A list of row codes is the indexing a decode loop does on its own: read
    # the rows a page table names, or write a step back into the slots that
    # table hands out.  The payload is read as planes x rows x width, so the
    # axis the codes name may be any axis of any rank, and a code that names no
    # row answers zero and writes nothing.  Range is settled inside the walk,
    # because a host that asked for it would pay a queue sync for every step.

    SHAPES = ((5,), (5, 4), (2, 5, 3), (2, 3, 5, 2))

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @classmethod
    def _table(cls, shape, dtype=np.float32):
        return (np.arange(int(np.prod(shape)), dtype=np.float32)
                .reshape(shape) % 17).astype(dtype)

    @staticmethod
    def _codes(values, dtype=np.int64):
        return TestVulkanAdvancedIndexing._device(
            np.array(values, dtype=dtype))

    @staticmethod
    def _read(payload, axis, codes):
        return tensorplay.index(payload, [None] * axis + [codes])

    def _gather_matches(self, np_dtype, tensorplay_dtype, shapes, axes):
        for shape in shapes:
            table = self._table(shape, np_dtype)
            payload = self._device(table)
            self.assertEqual(payload.dtype, tensorplay_dtype)
            for axis in axes(shape):
                size = shape[axis]
                codes = [0, size - 1, size // 2, 1, size - 1]
                want = np.take(table, codes, axis=axis)
                got = self._read(payload, axis, self._codes(codes))
                self.assertEqual(tuple(got.shape), want.shape)
                np.testing.assert_array_equal(
                    got.cpu().numpy(), want,
                    err_msg=f"gather of {shape} axis {axis}")

    def test_a_code_list_names_a_row_of_every_axis(self):
        self._gather_matches(np.float32, tensorplay.float32,
                             self.SHAPES, lambda shape: range(len(shape)))

    def test_a_half_payload_gathers_the_rows_it_is_asked_for(self):
        self._gather_matches(np.float16, tensorplay.float16,
                             self.SHAPES, lambda shape: range(len(shape)))

    def test_a_table_past_the_reach_of_a_texture_gathers_through_words(self):
        # A payload with no image form is walked in dense row-major storage,
        # where the planes above the gathered axis and the row the codes name
        # are arithmetic on an element count rather than a texel coordinate.
        self._gather_matches(np.float32, tensorplay.float32,
                             ((20000, 3), (2, 20000, 3)),
                             lambda shape: range(len(shape)))
        self._gather_matches(np.float16, tensorplay.float16, ((20000, 3),),
                             lambda shape: range(len(shape)))

    def test_a_rank_one_payload_gathers_from_either_storage(self):
        for count in (6, 20000):
            table = np.arange(count, dtype=np.float32)
            codes = [count - 1, 0, count // 3]
            np.testing.assert_array_equal(
                self._read(self._device(table), 0, self._codes(codes))
                .cpu().numpy(),
                table[codes], err_msg=f"a flat payload of {count} lanes")

    def test_a_wide_code_list_answers_the_narrow_one(self):
        table = self._table((9, 5))
        payload = self._device(table)
        for axis, codes in ((0, [0, 8, 3, 1]), (1, [0, 4, 2, 1])):
            wide = np.take(table, codes, axis=axis)
            narrow = self._codes(codes, np.int32)
            np.testing.assert_array_equal(
                self._read(payload, axis, narrow).cpu().numpy(), wide,
                err_msg=f"int32 codes over axis {axis}")

    def test_a_code_list_keeps_the_shape_it_arrived_in(self):
        table = self._table((2, 3, 4))
        payload = self._device(table)
        grid = [[0, 1], [1, 0], [0, 0]]
        for axis in (0, 1, 2):
            codes = self._device(np.array(grid, dtype=np.int64))
            # A grid of codes names a grid of rows: the axis it walks is
            # replaced by the grid's own shape.
            want = np.take(table, grid, axis=axis)
            got = tensorplay.index(payload, [None] * axis + [codes])
            np.testing.assert_array_equal(
                got.cpu().numpy(), want, err_msg=f"2-d codes over axis {axis}")
        lone = table[1]
        np.testing.assert_array_equal(
            self._read(payload, 0, self._codes(1)).cpu().numpy(), lone,
            err_msg="a single code drops the axis it walked")

    def test_an_empty_code_list_answers_no_rows(self):
        payload = self._device(self._table((4, 6)))
        got = self._read(payload, 0, self._codes(np.zeros(0, dtype=np.int64)))
        self.assertEqual(tuple(got.shape), (0, 6))

    def test_a_code_outside_the_rows_reads_nothing(self):
        # Both storages settle the range in the walk: a code past the end names
        # no row and answers zeros, while a code below zero counts back from the
        # end of the axis it names, which is how the host reads it.  A host that
        # asked for either bound would pay a queue sync for every step.
        table = self._table((5, 4))
        small = self._device(table)
        np.testing.assert_array_equal(
            self._read(small, 0, self._codes([0, 7, -2])).cpu().numpy(),
            np.array([[0, 1, 2, 3], [0, 0, 0, 0], table[3]], np.float32),
            err_msg="an image payload")
        big_table = self._table((20000, 3))
        big = self._device(big_table)
        np.testing.assert_array_equal(
            self._read(big, 0, self._codes([0, 20000, -1])).cpu().numpy(),
            np.array([[0, 1, 2], [0, 0, 0], big_table[19999]], np.float32),
            err_msg="a linear payload")

    def _write_matches(self, shape, axis, np_dtype=np.float32):
        table = np.zeros(shape, dtype=np_dtype)
        payload = self._device(table)
        size = shape[axis]
        # Two codes naming one row race for it, so a write is only answered
        # once its codes name different rows.
        codes = [0, size - 1] if size < 3 else [0, size - 1, size // 2]
        src_shape = list(shape)
        src_shape[axis] = len(codes)
        rows = (np.arange(int(np.prod(src_shape)), dtype=np_dtype)
                + np_dtype(1)).reshape(src_shape)
        want = table.copy()
        for slot, code in enumerate(codes):
            dst = [slice(None)] * len(shape)
            dst[axis] = code
            src = [slice(None)] * len(shape)
            src[axis] = slot
            want[tuple(dst)] = rows[tuple(src)]
        tensorplay.index_put_(payload, [None] * axis + [self._codes(codes)],
                              self._device(rows), False)
        np.testing.assert_array_equal(
            payload.cpu().numpy(), want,
            err_msg=f"write over {shape} axis {axis}")

    def test_a_write_lands_on_every_row_the_codes_name(self):
        # The axis an image payload can scatter is the one its texture puts on
        # its own height; a table with no image form scatters any axis.
        self._write_matches((5, 4), 0)
        self._write_matches((2, 5, 3), 1)
        self._write_matches((9, 64), 0, np.float16)
        for axis in (0, 1, 2):
            self._write_matches((2, 20000, 3), axis)

    def test_a_narrow_source_is_read_once_per_row_it_lands_in(self):
        table = np.zeros((4, 3), dtype=np.float32)
        payload = self._device(table)
        tensorplay.index_put_(payload, [self._codes([1, 3])],
                              self._device(np.array([[5.0], [7.0]], np.float32)),
                              False)
        want = table.copy()
        want[1] = 5.0
        want[3] = 7.0
        np.testing.assert_array_equal(payload.cpu().numpy(), want,
                                      err_msg="a column broadcast over a row")

    def test_a_write_that_names_nothing_changes_nothing(self):
        payload = self._device(np.ones((4, 3), dtype=np.float32))
        tensorplay.index_put_(payload, [self._codes(np.zeros(0, np.int64))],
                              self._device(np.zeros((0, 3), np.float32)), False)
        np.testing.assert_array_equal(payload.cpu().numpy(),
                                      np.ones((4, 3), np.float32),
                                      err_msg="an empty code list")
        tensorplay.index_put_(payload, [self._codes([9, -1])],
                              self._device(np.zeros((2, 3), np.float32)), False)
        np.testing.assert_array_equal(payload.cpu().numpy(),
                                      np.ones((4, 3), np.float32),
                                      err_msg="codes that name no row")

    def test_a_list_left_on_the_host_is_taken_along(self):
        table = self._table((5, 4))
        payload = self._device(table)
        codes = tensorplay.tensor([0, 2, 1], dtype=tensorplay.int64)
        self.assertEqual(codes.device.type, "cpu")
        np.testing.assert_array_equal(
            self._read(payload, 0, codes).cpu().numpy(),
            np.take(table, [0, 2, 1], axis=0), err_msg="codes on the host")
        written = tensorplay.tensor([1, 3], dtype=tensorplay.int64)
        tensorplay.index_put_(payload, [written], self._device(table[[0, 4]]),
                              False)
        want = table.copy()
        want[1] = table[0]
        want[3] = table[4]
        np.testing.assert_array_equal(payload.cpu().numpy(), want,
                                      err_msg="a write with host codes")
        lone = self._device(np.zeros((3, 2), dtype=np.float32))
        tensorplay.index_put_(lone, [self._codes([0, 2])],
                              tensorplay.tensor(np.float32(3.0)), False)
        np.testing.assert_array_equal(
            lone.cpu().numpy(),
            np.array([[3, 3], [0, 0], [3, 3]], np.float32),
            err_msg="a scalar that stayed on the host")

    def test_a_subscript_answers_the_way_the_walk_does(self):
        table = self._table((6, 4))
        payload = self._device(table)
        np.testing.assert_array_equal(
            payload[self._codes([5, 0])].cpu().numpy(),
            np.take(table, [5, 0], axis=0), err_msg="a subscript read")
        t = self._device(np.zeros((3, 2), dtype=np.float32))
        t[self._codes([0, 2])] = self._device(np.full((2, 2), -4.0, np.float32))
        np.testing.assert_array_equal(
            t.cpu().numpy(), np.array([[-4, -4], [0, 0], [-4, -4]], np.float32),
            err_msg="a subscript write")
        t2 = self._device(np.zeros((3, 2), dtype=np.float32))
        t2[self._codes([0, 2]), ...] = self._device(
            np.full((2, 2), -3.0, np.float32))
        np.testing.assert_array_equal(
            t2.cpu().numpy(), np.array([[-3, -3], [0, 0], [-3, -3]], np.float32),
            err_msg="a subscript write that spells its whole axis")

    def test_an_index_of_nothing_but_numbers_is_answered_off_it(self):
        payload = self._device(self._table((5, 4)))
        # A mask now compacts to the rows it selects and several lists fold into
        # one address per answer, so the only tuple a read refuses here is one
        # whose index holds neither counts nor lanes.
        with self.assertRaisesRegex((RuntimeError, IndexError), "long, int, byte or bool"):
            tensorplay.index(payload, [self._device(np.ones(2, np.float32))])


class TestVulkanIndexWalk(unittest.TestCase):
    # A tuple that reaches past one row of one axis is read by turning every
    # coordinate it names into an address of the payload flattened to a line, so
    # a single lookup gathers what several axes would have taken together.  A
    # mask is first compacted to the positions of the lanes it selects, several
    # lists broadcast against each other into one block, and lists that sit apart
    # hand that block to the front of the answer, which is the order a walk over
    # the whole payload leaves it in.  Nothing here needs a shader of its own:
    # the compaction, the counting and the lookup are answered elsewhere in this
    # backend, and the addresses are counts wider than a word because a
    # payload's own reach is.

    @staticmethod
    def _table(shape, dtype=np.float32):
        return (np.arange(int(np.prod(shape)), dtype=np.float32)
                .reshape(shape) % 17).astype(dtype)

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @staticmethod
    def _codes(values, dtype=np.int64):
        return TestVulkanIndexWalk._device(np.array(values, dtype=dtype))

    def _read_matches(self, table, keys, label):
        # The tuple is read on both lanes and against the arithmetic the answer
        # has to match, so a walk that lands an address off by one stride of its
        # own axis shows up rather than looking plausible in isolation.
        want = table[tuple(slice(None) if key is None else key
                           for key in keys)]
        for device in ("vulkan", "cpu"):
            payload = tensorplay.from_numpy(np.ascontiguousarray(table)).to(device)
            indices = [None if key is None else
                       tensorplay.from_numpy(np.asarray(key)).to(device)
                       for key in keys]
            got = tensorplay.index(payload, indices)
            self.assertEqual(tuple(got.shape), want.shape,
                             msg=f"{label} on {device}")
            np.testing.assert_array_equal(
                got.cpu().numpy(), want, err_msg=f"{label} on {device}")

    def test_two_lists_fold_into_one_address_per_answer(self):
        table = self._table((3, 4, 5))
        self._read_matches(table, [np.array([0, 1]), np.array([1, 2])],
                           "two lists of one length")
        self._read_matches(
            table,
            [np.array([0, 2, 1]), np.array([3, 0, 1]), np.array([1, 2, 4])],
            "three lists")

    def test_lists_broadcast_against_each_other(self):
        table = self._table((3, 4, 5))
        self._read_matches(table,
                           [np.array([[0], [1], [2]]), np.array([1, 2, 3])],
                           "a grid of codes")
        self._read_matches(table,
                           [np.array([0, 2]), np.array([[3], [0], [1]])],
                           "lists of one length held in different shapes")
        self._read_matches(table, [np.array([0, 1, 2]), np.zeros((2, 3), np.int64)],
                           "a list broadcast over a grid")
    def test_an_axis_left_whole_keeps_its_own_lanes(self):
        table = self._table((3, 4, 5))
        self._read_matches(table, [None, np.array([0, 2])], "a middle list")
        self._read_matches(table, [None, np.array([1])], "one row of a middle axis")
        self._read_matches(table, [np.array([0, 2]), None],
                           "a list that names the first axis")

    def test_a_mask_compacts_the_lanes_it_selects(self):
        table = self._table((3, 4, 5))
        self._read_matches(table, [table % 3 == 0], "a mask over the whole payload")
        self._read_matches(table, [np.array([True, False, True])],
                           "a mask over one axis")
        self._read_matches(table, [None, np.array([True, False, True, False])],
                           "a mask over a middle axis")
        self._read_matches(table, [table[:, :, 0] % 2 == 0],
                           "a mask that covers two axes")

    def test_lists_that_sit_apart_move_their_block_first(self):
        table = self._table((3, 4, 5))
        self._read_matches(table,
                           [np.array([0, 1]), None, np.array([0, 1])],
                           "codes at both ends of a tuple")
        self._read_matches(table,
                           [np.array([0, 2]), None, np.array([True, False, True, False, False])],
                           "a list and a mask at both ends")
        wide = self._table((2, 3, 4, 5))
        self._read_matches(wide,
                           [np.array([0, 1]), None, np.array([1, 3]), None],
                           "a rank four payload with the block in the middle")

    def test_a_negative_code_counts_back_from_the_axis_it_names(self):
        table = self._table((3, 4, 5))
        self._read_matches(table, [np.array([-1, 0])],
                           "a lone list with a code below zero")
        self._read_matches(table, [np.array([-1, 0]), np.array([-2, 3])],
                           "two lists with codes below zero")
        self._read_matches(table, [None, np.array([-1, -3]), np.array([-1])],
                           "codes below zero across an axis left whole")

    def test_a_half_payload_is_walked_the_same_way(self):
        table = self._table((3, 4, 5), np.float16)
        self._read_matches(table, [np.array([0, 2]), np.array([1, 3])],
                           "a half payload under two lists")
        self._read_matches(table, [table % 2 == 0],
                           "a half payload under a mask")

    def test_a_counting_payload_is_walked_the_same_way(self):
        table = np.arange(60, dtype=np.int64).reshape(3, 4, 5)
        self._read_matches(table, [np.array([0, 2]), np.array([1, 3])],
                           "a payload of counts under two lists")
        self._read_matches(table, [table % 7 == 0],
                           "a payload of counts under a mask")

    def test_a_tuple_that_names_nothing_answers_no_lanes(self):
        table = self._table((3, 4, 5))
        self._read_matches(table,
                           [np.zeros(0, np.int64), np.zeros(0, np.int64)],
                           "two empty lists")
        self._read_matches(table, [np.zeros((3, 4, 5), bool)],
                           "a mask with no lane selected")
        self._read_matches(table, [None, np.zeros(0, np.int64)],
                           "an empty list over a middle axis")

    def test_the_walk_carries_a_list_left_on_the_host(self):
        table = self._table((3, 4, 5))
        payload = self._device(table)
        codes = tensorplay.tensor([0, 2], dtype=tensorplay.int64)
        grid = tensorplay.tensor([[1], [3]], dtype=tensorplay.int64)
        self.assertEqual(codes.device.type, "cpu")
        got = tensorplay.index(payload, [codes, grid])
        np.testing.assert_array_equal(
            got.cpu().numpy(), table[[0, 2], [[1], [3]]],
            err_msg="lists that stayed on the host")

    def test_a_subscript_walks_the_payload_the_same_way(self):
        table = self._table((3, 4, 5))
        payload = self._device(table)
        np.testing.assert_array_equal(
            payload[self._codes([0, 2]), self._codes([1, 3])].cpu().numpy(),
            table[[0, 2], [1, 3]], err_msg="a subscript of two lists")
        mask = table[:, :, 0] % 2 == 0
        np.testing.assert_array_equal(
            payload[self._device(mask)].cpu().numpy(), table[mask],
            err_msg="a subscript under a mask")

    def test_a_block_wider_than_the_axes_it_names_keeps_its_place(self):
        # A list of more than one dimension names one axis but leaves a block of
        # several, so the addresses carry the rank of the answer rather than
        # that of the payload.  An axis held whole then has to stand after the
        # whole block, not where its own axis sat.
        table = self._table((3, 4, 5))
        self._read_matches(table, [None, np.array([[0], [1]])],
                           "a two dimensional list over a middle axis")
        self._read_matches(table, [np.array([[0], [1]]), None],
                           "a two dimensional list with the axes after it whole")
        self._read_matches(table, [None, None, np.array([[0, 1], [2, 3]])],
                           "a two dimensional list at the end of a tuple")
        self._read_matches(table, [np.array([[0], [1]]), None, np.array([0, 1])],
                           "a wide block moved in front of an axis left whole")
        wide = self._table((2, 3, 4, 5))
        self._read_matches(wide, [np.array([[0], [1]]), None, None,
                                  np.array([0, 1])],
                           "a wide block at both ends of a rank four payload")

    def test_a_tuple_the_walk_cannot_follow_is_answered_off_it(self):
        payload = self._device(self._table((3, 4, 5)))
        with self.assertRaisesRegex((RuntimeError, IndexError), "long, int, byte or bool"):
            tensorplay.index(payload, [self._device(np.ones(2, np.float32))])
        with self.assertRaisesRegex((RuntimeError, IndexError), "shape of the mask"):
            tensorplay.index(payload, [self._device(np.array([True, False]))])
        with self.assertRaisesRegex((RuntimeError, IndexError), "shape mismatch"):
            tensorplay.index(payload,
                             [self._codes([0, 1]), self._codes([0, 1, 2])])
        with self.assertRaisesRegex((RuntimeError, IndexError), "too many indices"):
            tensorplay.index(payload, [self._codes([0]), self._codes([0]),
                                       self._codes([0]), self._codes([0])])

    def _mask_write_matches(self, table, keys, value, accumulate, label,
                            host_number=False):
        # The lanes a mask opens are numbered by the order the payload holds
        # them, so the answer is the payload read back with those lanes written
        # over - which is the arithmetic the host does here.  A lone number is
        # laid in every lane the mask opens; a payload arrives lane by lane.
        if not isinstance(keys, list):
            keys = [keys]
        selected = tuple(slice(None) if key is None else key for key in keys)
        want = table.copy()
        picked = want[selected]
        number = np.asarray(value)
        spread = np.full(picked.shape, number, want.dtype) \
            if number.ndim == 0 else np.asarray(value, want.dtype)
        want[selected] = (picked + spread) if accumulate else spread
        for device in ("vulkan", "cpu"):
            # The transfer holds the array it is given rather than a copy of it,
            # so a write starts from a payload with memory of its own.
            payload = tensorplay.from_numpy(
                np.ascontiguousarray(table).copy()).to(device)
            indices = [None if key is None else
                       tensorplay.from_numpy(np.asarray(key)).to(device)
                       for key in keys]
            source = tensorplay.from_numpy(np.ascontiguousarray(number))
            if not (host_number and number.ndim == 0):
                source = source.to(device)
            tensorplay.index_put_(payload, indices, source, accumulate)
            np.testing.assert_array_equal(
                payload.cpu().numpy(), want, err_msg=f"{label} on {device}")

    def test_a_lone_number_lands_on_every_lane_a_mask_opens(self):
        table = self._table((3, 4, 5))
        self._mask_write_matches(table, table % 3 == 0, np.float32(9.0), False,
                                 "a number over a whole mask")
        self._mask_write_matches(table, table % 3 == 0, np.float32(9.0), False,
                                 "a number that stayed on the host",
                                 host_number=True)
        self._mask_write_matches(table, np.array([True, False, True]),
                                 np.float32(-2.0), False,
                                 "a number over a mask of one axis")
        self._mask_write_matches(table, [None, np.array([True, False, True, False])],
                                 np.float32(3.0), False,
                                 "a number over a mask of a middle axis")
        self._mask_write_matches(table, table[:, :, 0] % 2 == 0,
                                 np.float32(4.0), True,
                                 "a number added under a mask of two axes")
        self._mask_write_matches(table, table % 5 == 0, np.float32(0.5), True,
                                 "a number added over a whole mask")

    def test_a_lone_number_writes_a_payload_of_any_width(self):
        for dtype, held in ((np.float32, np.float32(7.0)),
                            (np.float16, np.float16(7.0)),
                            (np.int64, np.int64(7))):
            table = self._table((3, 4, 5), dtype)
            mask = table % (7 if np.issubdtype(dtype, np.integer) else 3) == 0
            self._mask_write_matches(table, mask, held, False,
                                     f"a number over a {np.dtype(dtype).name} payload")

    def test_several_numbers_spread_in_the_order_the_mask_walks(self):
        table = self._table((3, 4, 5))
        whole = table % 3 == 0
        count = int(np.count_nonzero(whole))
        self._mask_write_matches(
            table, whole, (np.arange(count, dtype=np.float32) + 100.0), False,
            "a payload spread over the lanes a whole mask opens")
        self._mask_write_matches(
            table, whole, (np.arange(count, dtype=np.float32) + 100.0), True,
            "a payload added over the lanes a whole mask opens")
        # A mask that covers two axes opens a whole line of the payload per lane
        # it selects, so the numbers arrive one line at a time.
        lines = table[:, :, 0] % 2 == 0
        rows = int(np.count_nonzero(lines))
        self._mask_write_matches(
            table, lines,
            np.arange(rows * 5, dtype=np.float32).reshape(rows, 5), False,
            "a payload spread over the lines a mask of two axes opens")
        # A mask of one axis opens a plane, so the numbers arrive plane by plane.
        planes = np.array([True, False, True])
        self._mask_write_matches(
            table, planes,
            np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5), False,
            "a payload spread over the planes a mask of one axis opens")

    def test_a_mask_that_opens_no_lane_writes_nothing(self):
        table = self._table((4, 3))
        closed = np.zeros((4, 3), bool)
        for device in ("vulkan", "cpu"):
            payload = tensorplay.from_numpy(table.copy()).to(device)
            tensorplay.index_put_(
                payload, [tensorplay.from_numpy(closed).to(device)],
                tensorplay.from_numpy(np.zeros(0, np.float32)).to(device), False)
            np.testing.assert_array_equal(
                payload.cpu().numpy(), table,
                err_msg=f"a mask that opens no lane on {device}")
            tensorplay.index_put_(
                payload, [tensorplay.from_numpy(closed).to(device)],
                tensorplay.from_numpy(np.float32(5.0)).to(device), True)
            np.testing.assert_array_equal(
                payload.cpu().numpy(), table,
                err_msg=f"a number over no lane on {device}")

    def test_a_subscript_writes_through_a_mask(self):
        table = self._table((4, 3))
        payload = self._device(table)
        mask = self._device(table[:, 0] > 1)
        payload[mask] = self._device(np.float32(-1.0))
        want = table.copy()
        want[table[:, 0] > 1] = -1.0
        np.testing.assert_array_equal(
            payload.cpu().numpy(), want, err_msg="a subscript write under a mask")

    def test_a_write_of_points_the_walk_names_stays_refused(self):
        # The addresses a walk composes name single lanes, and the payload is
        # held in a form a write through it narrowed to a line does not reach,
        # so the scatter keeps to whole rows and this stays refused rather than
        # answered by carrying the payload through two copies.
        payload = self._device(np.zeros((3, 4, 5), np.float32))
        with self.assertRaisesRegex(RuntimeError, "several lists"):
            tensorplay.index_put_(
                payload, [self._codes([0, 1]), self._codes([0, 1])],
                self._device(np.zeros((2, 5), np.float32)), False)
        with self.assertRaisesRegex(RuntimeError, "several lists"):
            tensorplay.index_put_(
                payload, [self._device(np.zeros((3, 4, 5), bool)),
                          self._codes([0, 1])],
                self._device(np.zeros((2, 5), np.float32)), False)

    def test_a_mask_that_does_not_line_up_is_answered_off_the_write(self):
        payload = self._device(np.zeros((3, 4, 5), np.float32))
        with self.assertRaisesRegex(RuntimeError, "shape of the mask"):
            tensorplay.index_put_(
                payload, [self._device(np.zeros(4, bool))],
                self._device(np.float32(1.0)), False)
        with self.assertRaisesRegex(RuntimeError, "dtypes match"):
            tensorplay.index_put_(
                payload, [self._device(np.zeros((3, 4, 5), bool))],
                self._device(np.zeros(4, np.int64)), False)


class TestVulkanMaskCompaction(unittest.TestCase):
    # A mask cannot be answered in one pass: the length of the result is only
    # known once the whole payload has been read, and the device has to be told
    # it before it writes the coordinates.  The walk is therefore two passes over
    # one packed payload - a total per workgroup that comes to the host, and a
    # second pass that writes each workgroup's lanes at the offset the totals add
    # up to - so both the boundary a workgroup ends at and the boundary of a
    # packed word have to be pinned, not just a payload that fits inside one.

    SHAPES = ((7,), (3, 4), (2, 3, 5), (2, 2, 3, 2))

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    def _nz_matches(self, arr, name):
        arr = np.asarray(arr)
        payload = self._device(arr)
        got = tensorplay.nonzero(payload)
        self.assertEqual(got.dtype, tensorplay.int64, name)
        np.testing.assert_array_equal(
            got.cpu().numpy(), np.argwhere(arr != 0), err_msg=name)

    def test_the_positions_a_mask_names_are_the_row_major_ones(self):
        for shape in self.SHAPES:
            for np_dtype in (np.float32, np.float16, np.int32, np.int8):
                rng = np.random.default_rng(abs(hash((shape, np_dtype))) % 2**32)
                arr = rng.integers(-2, 3, size=shape).astype(np_dtype)
                self._nz_matches(arr, f"nonzero of a {shape} {np_dtype.__name__}")
            flat = np.arange(int(np.prod(shape))).reshape(shape)
            self._nz_matches(flat % 3 == 0, f"nonzero of a {shape} truth plane")

    def test_a_lane_at_the_edge_of_a_word_a_group_and_a_walk_is_named_exactly(self):
        # One lane set is the smallest answer there is, so a position that comes
        # back shifted names a mapping between the packed payload and the walk
        # that read it wrong: byte 3 and 4 straddle a packed word, 1023 and 1024
        # straddle a workgroup, and 65535 straddles the boundary where a walk
        # gives every lane several words.
        size = 70000
        for at in (0, 1, 3, 4, 5, 1023, 1024, 1027, 4095, 4096, 4097,
                   65535, 65536, 69999):
            arr = np.zeros(size, dtype=np.float32)
            arr[at] = 1.0
            self._nz_matches(arr, f"a lane set at {at} of {size}")
        arr = np.zeros(size, dtype=np.float32)
        edges = [0, 3, 4, 1024, 65535, 69999]
        arr[edges] = 1.0
        self._nz_matches(arr, "every edge of one payload at once")

    def test_a_mask_made_on_the_device_compacts_like_an_uploaded_one(self):
        # Above a texture's reach a comparison answers in dense storage, and the
        # payload it writes belongs to the comparison rather than to the caller.
        # A walk that hands that memory back before its second pass would read
        # whatever the next allocation wrote over it.
        for size in (20000, 70000):
            arr = (np.arange(size, dtype=np.float32) % 7 == 0).astype(np.float32)
            self._nz_matches(arr, f"a float payload of {size} tested on the device")
            truth = (np.arange(size) % 11 == 0).astype(np.bool_)
            self._nz_matches(truth, f"a truth plane of {size} uploaded as it stands")

    def test_no_lanes_and_all_lanes_answer_without_a_walk(self):
        self._nz_matches(np.zeros((4, 5), np.float32), "a payload that selects none")
        self._nz_matches(np.ones((6,), np.float32), "a payload that selects all")
        self._nz_matches(np.zeros((0, 3), np.float32), "a payload with no lanes")
        for value in (0.0, 2.5):
            self._nz_matches(np.array(value, dtype=np.float32),
                             f"a laneless payload of {value}")

    def _select_matches(self, payload, mask, name):
        payload, mask = np.asarray(payload), np.asarray(mask)
        got = tensorplay.masked_select(self._device(payload), self._device(mask))
        want = payload[np.broadcast_to(mask, np.broadcast_shapes(payload.shape,
                                                                mask.shape)) != 0]
        np.testing.assert_array_equal(got.cpu().numpy(), want, err_msg=name)

    def test_a_select_pulls_the_lanes_a_mask_holds(self):
        table = (np.arange(12, dtype=np.float32) % 5).reshape(3, 4)
        truth = np.arange(12).reshape(3, 4) % 3 == 0
        self._select_matches(table, truth, "a select over one shape")
        self._select_matches(table, (np.arange(12).reshape(3, 4) % 4).astype(np.uint8),
                             "a select over a byte mask")
        self._select_matches(table.astype(np.float16), table > 1,
                             "a select of a half payload")
        self._select_matches(table, np.array([True, False, True, True]),
                             "a select under a row broadcast")
        self._select_matches(table, np.array(True), "a select under a scalar mask")
        self._select_matches(table, np.zeros((3, 4), np.bool_),
                             "a select that holds nothing")
        self._select_matches(table, np.ones((3, 4), np.bool_),
                             "a select that holds every lane")

    def test_a_select_past_one_workgroup_keeps_the_order_of_the_mask(self):
        payload = np.arange(70000, dtype=np.float32) % 9
        self._select_matches(payload, payload > 2,
                             "a long select that walks several words a lane")

    def test_a_mask_the_device_cannot_count_is_answered_off_it(self):
        # A count is one byte a lane here, so a payload wider than a word has no
        # walk tested on the device, and neither has a gather of a payload the
        # select path cannot read.
        with self.assertRaisesRegex(RuntimeError, "not tested on the device"):
            tensorplay.nonzero(self._device(np.arange(6, dtype=np.int64)))
        with self.assertRaisesRegex(RuntimeError, "not tested on the device"):
            tensorplay.nonzero(self._device((np.arange(6) % 2 == 0).astype(np.int64)))
        # A payload of five axes has no coordinate the walk can spell out, and
        # the refusal comes before the walk: which layer names it is not part of
        # the contract, that it is refused loudly is.
        with self.assertRaisesRegex(RuntimeError, "4"):
            tensorplay.nonzero(self._device(np.ones((2, 2, 2, 2, 2), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16 payloads"):
            tensorplay.masked_select(self._device(np.arange(6, dtype=np.int32)),
                                     self._device(np.array([True] * 6)))


class TestVulkanMaskSpread(unittest.TestCase):
    # The opposite walk of a compaction: instead of pulling the lanes a mask
    # holds into a result of a length only the device knows, the mask pushes the
    # leading elements of a source into a destination whose shape it cannot
    # change.  The rank a lane gets from the count pass is the position in the
    # source its selection takes, so the answer is one fixed sequence however
    # the work is spread - and an element the mask leaves alone has to arrive
    # untouched, which pins the copy the walk fuses with the spread.

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        # A list of numbers arrives as a double plane; the payload this walk
        # spreads is a single-precision one, so the written values arrive whole.
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @staticmethod
    def _host(tensor):
        return tensor.cpu().numpy()

    def _spread_matches(self, payload, mask, source, name):
        payload, mask, source = (np.asarray(a) for a in (payload, mask, source))
        got = self._spread(
            self._device(payload), self._device(mask), self._device(source))
        want = payload.copy().reshape(-1)
        wanted_mask = np.broadcast_to(mask, payload.shape).reshape(-1) != 0
        take = int(np.count_nonzero(wanted_mask))
        want[wanted_mask] = source.reshape(-1)[:take]
        np.testing.assert_array_equal(
            self._host(got).reshape(-1), want, err_msg=name)

    def _spread(self, destination, mask, source):
        return tensorplay.masked_scatter(destination, mask, source)

    def test_the_leading_source_elements_spread_in_the_order_the_mask_reads(self):
        self._spread_matches(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[True, False, True], [False, True, False]],
            [10.0, 20.0, 30.0],
            "a spread over a plane")
        destination = self._device([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        answer = self._spread(
            destination, self._device([[True, False, True], [False, True, False]]),
            self._device([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(
            self._host(answer), [[10.0, 2.0, 20.0], [4.0, 30.0, 6.0]])
        np.testing.assert_array_equal(
            self._host(destination), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            err_msg="an out-of-place spread leaves the destination alone")

    def test_the_write_lands_in_the_destination_it_names(self):
        destination = self._device([1.0, 2.0, 3.0, 4.0])
        answer = destination.masked_scatter_(
            self._device([False, True, True, False]), self._device([7.0, 8.0]))
        np.testing.assert_array_equal(
            self._host(answer), [1.0, 7.0, 8.0, 4.0])
        self.assertIs(answer, destination)

    def test_a_source_longer_than_the_selection_keeps_its_tail_and_a_short_one_is_named(self):
        self._spread_matches(
            [0.0, 0.0, 0.0], [True, False, True], [1.0, 2.0, 3.0, 4.0],
            "a source with more elements than the mask takes")
        with self.assertRaisesRegex(RuntimeError, "fewer elements"):
            tensorplay.masked_scatter(
                self._device([1.0, 2.0]), self._device([True, True]),
                self._device([1.0]))

    def test_a_mask_that_holds_nothing_or_everything_answers_with_what_the_destination_holds(self):
        self._spread_matches([1.0, 2.0], [False, False], [9.0],
                             "a spread that selects none")
        self._spread_matches([1.0, 2.0], [True, True], [8.0, 7.0],
                             "a spread that selects every lane")
        self._spread_matches(np.zeros((2, 3), np.float32),
                             np.zeros((2, 3), np.bool_), np.zeros(0),
                             "a source with no elements and a mask that needs none")
        empty = tensorplay.masked_scatter(
            self._device(np.zeros((0, 3), np.float32)),
            self._device(np.zeros((0, 3), np.bool_)),
            self._device([1.0]))
        self.assertEqual(tuple(self._host(empty).shape), (0, 3))

    def test_a_mask_spreads_over_the_rows_it_broadcasts_to(self):
        table = (np.arange(12, dtype=np.float32)).reshape(3, 4)
        self._spread_matches(table, [False, True, False, True],
                             np.arange(6, dtype=np.float32),
                             "a row mask repeated over three rows")
        self._spread_matches(table, np.array(True), np.arange(12, dtype=np.float32),
                             "a scalar mask that names every lane")
        self._spread_matches(table, np.zeros((3, 1), np.bool_),
                             np.arange(4, dtype=np.float32),
                             "a mask that selects nothing by column")

    def test_a_spread_over_a_payload_of_several_planes_follows_the_row_major_order(self):
        table = (np.arange(2 * 4 * 3, dtype=np.float32)).reshape(2, 4, 3)
        truth = np.zeros((2, 4, 3), np.bool_)
        truth[0, 1, 2] = True
        truth[1, 0, 0] = True
        truth[1, 3, 2] = True
        answer = self._host(self._spread(
            self._device(table), self._device(truth), self._device([500.0, 600.0, 700.0])))
        # The lane the payload holds first is the one the mask reads first, so the
        # selection in the second plane takes the second source element and every
        # lane beside it arrives as the destination held it.
        want = table.copy()
        want[0, 1, 2] = 500.0
        want[1, 0, 0] = 600.0
        want[1, 3, 2] = 700.0
        np.testing.assert_array_equal(answer, want)

    def test_a_lane_at_the_edge_of_a_word_a_group_and_a_walk_takes_its_own_element(self):
        size = 70000
        for at in (0, 1, 3, 4, 5, 1023, 1024, 1027, 4095, 4096, 4097,
                   65535, 65536, size - 1):
            truth = np.zeros(size, np.bool_)
            truth[at] = True
            self._spread_matches(
                np.zeros(size, np.float32), truth, [float(at) + 1.0],
                f"the single lane set at {at} of {size}")

    def test_a_sparse_and_a_dense_mask_over_a_long_payload_keep_the_source_in_line(self):
        size = 70000
        positions = np.arange(0, size, 7)
        truth = np.zeros(size, np.bool_)
        truth[positions] = True
        source = (np.arange(positions.size, dtype=np.float32) + 100.0)
        answer = self._host(self._spread(
            self._device(np.zeros(size, np.float32)), self._device(truth),
            self._device(source)))
        np.testing.assert_array_equal(answer[positions], source)
        self.assertEqual(float(np.count_nonzero(answer)), float(positions.size),
                         "a spread writes only the lanes a mask names")
        dense = (np.arange(100000) % 3) == 0
        self._spread_matches(np.zeros(100000, np.float32), dense,
                             np.arange(int(dense.sum()), dtype=np.float32),
                             "a mask that names a third of a long payload")

    def test_a_spread_undoes_a_compaction_of_the_same_mask(self):
        truth = (np.arange(24) % 2) == 1
        # Only a payload that holds zero where the mask does not name it comes
        # back whole: the spread carries the lanes it is given, not the ones it
        # was never told about.
        payload = np.where(truth, np.arange(1, 25, dtype=np.float32), np.float32(0.0))
        picked = tensorplay.masked_select(self._device(payload), self._device(truth))
        answer = self._spread(self._device(np.zeros(24, np.float32)),
                              self._device(truth), picked)
        np.testing.assert_array_equal(self._host(answer), payload)

    def test_a_mask_made_on_the_device_spreads_like_an_uploaded_one(self):
        # A comparison of a payload beyond a texture's reach answers in dense
        # storage, and the truth plane it writes belongs to the comparison rather
        # than to the caller.  The spread reads that plane after the count pass
        # has come back, so memory handed back between the two passes would be
        # answered with whatever the next allocation wrote over it.
        for size in (20000, 70000):
            values = np.arange(size, dtype=np.float32) % 7
            payload = self._device(values)
            truth = payload > 3
            named = (values > 3)
            take = int(np.count_nonzero(named))
            answer = self._host(tensorplay.masked_scatter(
                payload, truth, self._device(
                    np.arange(take, dtype=np.float32) + 100.0)))
            want = values.copy()
            want[named] = np.arange(take, dtype=np.float32) + 100.0
            np.testing.assert_array_equal(
                answer, want, err_msg=f"a mask the device made of {size} lanes")

    def test_a_destination_of_either_storage_holds_the_answer(self):
        # A payload a texture reaches and one only linear memory holds both have
        # to answer, since the walk reads the flat order of whichever it got.
        small = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
        truth = (np.arange(42).reshape(6, 7) % 5) == 0
        self._spread_matches(
            small, truth, np.arange(int(truth.sum()), dtype=np.float32),
            "a spread over a texture-sized payload")
        large = np.arange(40000, dtype=np.float32) % 8
        wide = (np.arange(40000) % 3) == 0
        self._spread_matches(
            large, wide, np.arange(int(wide.sum()), dtype=np.float32),
            "a spread over a payload beyond a texture")

    def test_a_destination_the_walk_cannot_answer_is_refused_loudly(self):
        truth = np.array([True, False])
        for dtype, needle in ((np.int64, "Float32"), (np.int32, "Float32"),
                              (np.float16, "Float32")):
            with self.assertRaisesRegex(RuntimeError, needle):
                tensorplay.masked_scatter(
                    self._device(np.arange(2, dtype=dtype)), self._device(truth),
                    self._device(np.array([1.0, 2.0], np.float32)))
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            tensorplay.masked_scatter(
                self._device([1.0]), self._device([True]),
                self._device(np.array([1], np.int64)))
        with self.assertRaisesRegex(RuntimeError, "broadcast"):
            tensorplay.masked_scatter(
                self._device([1.0, 2.0]),
                self._device(np.ones((2, 2), np.bool_)),
                self._device([1.0]))
        with self.assertRaisesRegex(RuntimeError, "4"):
            tensorplay.masked_scatter(
                self._device(np.ones((2, 2, 2, 2, 2), np.float32)),
                self._device(np.ones((2, 2, 2, 2, 2), np.bool_)),
                self._device([1.0]))
        # A source the host still holds has no flat order for the walk to read
        # on the device, so the refusal says so rather than copying it aside.
        with self.assertRaisesRegex(RuntimeError, "source"):
            tensorplay.masked_scatter(
                self._device([1.0]), self._device([True]),
                tensorplay.tensor([1.0]))


class TestVulkanForeachGroup(unittest.TestCase):
    # A group walk hands one entry of the arithmetic table to every tensor a list
    # names, so the shape and the width a tensor holds are the shape and the width
    # its answer keeps.  What the walk adds is a contract and nothing else: two
    # lists have to pair up by position, and the count a whole group shares has to
    # name a single lane.  The arithmetic itself is answered by the kernels a lone
    # tensor reaches, so every walk here is held against the host lane instead of
    # trusted on its own - and an answer that is a fresh group has to leave the
    # group it walked untouched, which is what pins the walk as out-of-place.

    @staticmethod
    def _table(shape, dtype=np.float32, start=0.0):
        return (((np.arange(int(np.prod(shape)), dtype=np.float32) % 13) / 4.0
                 + start)).astype(dtype).reshape(shape)

    @staticmethod
    def _group(tables, device):
        # The walk is handed a copy of each lane: a host tensor shares the
        # buffer it was read from, so an in-place answer would otherwise rewrite
        # the table the next device is about to be built from.
        return [tensorplay.from_numpy(np.array(table, copy=True))
                .to(device)
                for table in tables]

    @staticmethod
    def _read(group):
        return [held.cpu().numpy() for held in group]

    def _unscale(self, group, device, inv_scale, found=None):
        if found is None:
            found = tensorplay.tensor(
                0.0, dtype=tensorplay.float32, device=device)
        scale = tensorplay.tensor(
            inv_scale, dtype=tensorplay.float32, device=device)
        self.assertIsNone(
            tensorplay._amp_foreach_non_finite_check_and_unscale_(
                group, found, scale))
        return found

    def _answers(self, held, want, label, device, rtol=1e-5, atol=1e-6):
        self.assertEqual(len(held), len(want), msg=f"{label} on {device}")
        for index, (answer, truth) in enumerate(zip(held, want)):
            self.assertEqual(tuple(answer.shape), truth.shape,
                             msg=f"{label} entry {index} on {device}")
            self.assertEqual(answer.dtype, truth.dtype,
                             msg=f"{label} entry {index} on {device}")
            if np.issubdtype(truth.dtype, np.integer):
                np.testing.assert_array_equal(
                    answer, truth, err_msg=f"{label} entry {index} on {device}")
            else:
                np.testing.assert_allclose(
                    answer, truth, rtol=rtol, atol=atol,
                    err_msg=f"{label} entry {index} on {device}")

    def _read_matches(self, tables, want, answer, label, rtol=1e-5, atol=1e-6):
        # The walk is answered twice per case: once as the group it has to hand
        # back, once as the group it was handed, which a walk that wrote through
        # its own reads would show up on immediately.
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            self._answers(answer(group, device), want, label, device, rtol, atol)
            self._answers(self._read(group), [np.ascontiguousarray(table)
                                              for table in tables],
                          label + " leaves the group alone", device, rtol, atol)

    def _write_matches(self, tables, want, write, label, rtol=1e-5, atol=1e-6):
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            self.assertIsNone(write(group, device), msg=f"{label} on {device}")
            self._answers(self._read(group), want, label, device, rtol, atol)

    def test_a_count_reaches_every_tensor_of_the_group(self):
        tables = [self._table((3,)), self._table((2, 4)), self._table((5,))]
        count = 1.5
        for op, sign in ((tensorplay._foreach_add, lambda t: t + count),
                         (tensorplay._foreach_sub, lambda t: t - count),
                         (tensorplay._foreach_mul, lambda t: t * count),
                         (tensorplay._foreach_div, lambda t: t / count)):
            want = [sign(np.ascontiguousarray(table)) for table in tables]
            self._read_matches(
                tables, want,
                lambda group, device, op=op: self._read(op(group, count)),
                f"{op.__name__} by one count")

    def test_a_weight_scales_a_partner_before_it_lands(self):
        tables = [self._table((4,)), self._table((3, 2))]
        partners = [self._table((4,), start=1.0), self._table((3, 2), start=2.0)]
        for op, sign in ((tensorplay._foreach_add, lambda t, p: t + 2.0 * p),
                         (tensorplay._foreach_sub, lambda t, p: t - 2.0 * p)):
            want = [sign(np.ascontiguousarray(table), np.ascontiguousarray(partner))
                    for table, partner in zip(tables, partners)]
            self._read_matches(
                tables, want,
                lambda group, device, op=op: self._read(
                    op(group, self._group(partners, device), alpha=2.0)),
                f"{op.__name__} of a weighted group")

    def test_partners_pair_with_tensors_by_position(self):
        tables = [self._table((3,)), self._table((2, 4)), self._table((5,))]
        partners = [self._table((3,), start=1.0), self._table((2, 4), start=3.0),
                    self._table((5,), start=5.0)]
        for op, sign in ((tensorplay._foreach_add, lambda t, p: t + p),
                         (tensorplay._foreach_sub, lambda t, p: t - p),
                         (tensorplay._foreach_mul, lambda t, p: t * p),
                         (tensorplay._foreach_div, lambda t, p: t / p)):
            want = [sign(np.ascontiguousarray(table), np.ascontiguousarray(partner))
                    for table, partner in zip(tables, partners)]
            self._read_matches(
                tables, want,
                lambda group, device, op=op: self._read(
                    op(group, self._group(partners, device))),
                f"{op.__name__} of a matching group")

    def test_each_tensor_takes_the_count_it_is_named(self):
        tables = [self._table((6,)), self._table((2, 3)), self._table((4,))]
        counts = [1.5, -2.0, 0.25]
        for op, sign in ((tensorplay._foreach_add, lambda t, c: t + c),
                         (tensorplay._foreach_sub, lambda t, c: t - c),
                         (tensorplay._foreach_mul, lambda t, c: t * c),
                         (tensorplay._foreach_div, lambda t, c: t / c)):
            want = [sign(np.ascontiguousarray(table), count)
                    for table, count in zip(tables, counts)]
            self._read_matches(
                tables, want,
                lambda group, device, op=op: self._read(op(group, counts)),
                f"{op.__name__} by a list of counts")

    def test_one_count_spreads_over_the_group_from_wherever_it_is_held(self):
        tables = [self._table((3,)), self._table((2, 4))]
        for host_held in (False, True):
            # A count the host still holds crosses as the one lane it is; a
            # count already on the walk's own device is read where it stands.
            want = [np.ascontiguousarray(table) + 0.5 * 2.0 for table in tables]
            self._read_matches(
                tables, want,
                lambda group, device, host_held=host_held: self._read(
                    tensorplay._foreach_add(
                        group,
                        tensorplay.tensor(np.float32(2.0)) if host_held else
                        tensorplay.tensor(np.float32(2.0)).to(device),
                        alpha=0.5)),
                "a shared count " + ("held by the host" if host_held
                                    else "held by the device"))

    def test_the_walk_writes_where_the_tensor_stands(self):
        tables = [self._table((7,)), self._table((2, 3))]
        partners = [self._table((7,), start=1.0), self._table((2, 3), start=2.0)]
        want = [np.ascontiguousarray(table) * np.ascontiguousarray(partner)
                for table, partner in zip(tables, partners)]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_mul_(
                group, self._group(partners, device)),
            "a product written into the group")
        want = [np.ascontiguousarray(table) - 1.5 for table in tables]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_sub_(group, 1.5),
            "a count written into the group")
        want = [np.ascontiguousarray(table) + count for table, count
                in zip(tables, [0.5, -0.25])]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_add_(group, [0.5, -0.25]),
            "a list of counts written into the group")
        want = [np.ascontiguousarray(table) + 2.0 * np.ascontiguousarray(partner)
                for table, partner in zip(tables, partners)]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_add_(
                group, self._group(partners, device), alpha=2.0),
            "a weighted group written into the group")
        want = [np.ascontiguousarray(table) / 2.0 for table in tables]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_div_(
                group, tensorplay.tensor(np.float32(2.0)).to(device)),
            "a shared count written into the group")

    def test_a_group_keeps_the_width_it_holds(self):
        for dtype, count in ((np.float16, 2.0), (np.int32, 3), (np.int64, 3)):
            tables = [self._table((5,), dtype), self._table((2, 3), dtype)]
            for op, sign in ((tensorplay._foreach_add, lambda t, c: t + c),
                             (tensorplay._foreach_sub, lambda t, c: t - c),
                             (tensorplay._foreach_mul, lambda t, c: t * c)):
                want = [sign(np.ascontiguousarray(table), count)
                        for table in tables]
                want = [held.astype(np.asarray(table).dtype)
                        for held, table in zip(want, tables)]
                self._read_matches(
                    tables, want,
                    lambda group, device, op=op: self._read(op(group, count)),
                    f"{op.__name__} over a {np.dtype(dtype).name} group",
                    rtol=1e-3, atol=1e-3)

    def test_a_group_of_many_tensors_keeps_each_answer_apart(self):
        tables = [self._table((count,), start=index)
                  for index, count in enumerate([1, 2, 3, 5, 8, 13, 21, 34])]
        want = [np.ascontiguousarray(table) * 0.5 for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(tensorplay._foreach_mul(group, 0.5)),
            "a group of eight tensors")

    def test_lists_that_do_not_pair_are_refused(self):
        tables = [self._table((3,)), self._table((2, 4)), self._table((5,))]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            with self.assertRaisesRegex(
                    ValueError, "tensor list arguments must have the same length"):
                tensorplay._foreach_add(group, self._group(tables[:1], device))
            with self.assertRaisesRegex(
                    ValueError, "tensor and scalar lists must have the same length"):
                tensorplay._foreach_mul(group, [1.5, 2.0])

    def test_a_count_wider_than_one_lane_is_refused(self):
        # The count a whole group shares has nowhere to pair up with, so a
        # payload of several lanes is refused rather than broadcast by position.
        tables = [self._table((3,)), self._table((2, 4))]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            with self.assertRaisesRegex(ValueError, "expected to be 0 dim"):
                tensorplay._foreach_add(group, tensorplay.tensor([1.0, 2.0]).to(device))
            with self.assertRaisesRegex(ValueError, "expected to be 0 dim"):
                tensorplay._foreach_sub_(
                    group, tensorplay.tensor([1.0, 2.0, 3.0]).to(device))

    def test_a_group_of_any_size_is_answered_entry_by_entry(self):
        # One tensor is the smallest group a walk can be handed, and it has to
        # reach the same entry of the table the walk uses for a group of many.
        tables = [self._table((2, 3))]
        want = [np.ascontiguousarray(tables[0]) - 0.75]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_sub(group, 0.75)),
            "a group of one tensor")

    FORMULAS = {
        "sqrt": np.sqrt,
        "rsqrt": lambda held: 1.0 / np.sqrt(held),
        "neg": lambda held: -held,
        "abs": np.abs,
        "reciprocal": lambda held: 1.0 / held,
        "sign": np.sign,
        "ceil": np.ceil,
        "floor": np.floor,
        "trunc": np.trunc,
        "frac": lambda held: held - np.trunc(held),
        "exp": np.exp,
        "expm1": np.expm1,
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "log1p": np.log1p,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "sigmoid": lambda held: 1.0 / (1.0 + np.exp(-held)),
    }

    def _formula_tables(self, name):
        # An angle walk only answers where an angle is defined, so those two
        # formulas are handed a group narrow enough to stay inside it.
        if name in ("asin", "acos"):
            return [self._table((5,)) * 0.2, self._table((2, 3)) * 0.2]
        return [self._table((5,), start=1.0), self._table((2, 3), start=1.0)]

    def test_a_formula_spreads_over_the_group(self):
        for name in list(self.FORMULAS) + ["asin", "acos", "atan"]:
            tables = self._formula_tables(name)
            sign = (self.FORMULAS[name] if name in self.FORMULAS
                    else {"asin": np.arcsin, "acos": np.arccos,
                          "atan": np.arctan}[name])
            want = [sign(np.ascontiguousarray(table)) for table in tables]
            self._read_matches(
                tables, want,
                lambda group, device, name=name: self._read(
                    getattr(tensorplay, "_foreach_" + name)(group)),
                f"_foreach_{name}")

    def test_a_formula_writes_where_the_tensor_stands(self):
        # The in-place walk answers with the same formula, and the tensor it
        # writes keeps its own shape rather than the shape of a fresh read.
        for name in ("sqrt", "rsqrt", "neg", "abs", "reciprocal", "sign",
                     "floor", "trunc", "exp", "tanh", "sigmoid"):
            tables = self._formula_tables(name)
            sign = self.FORMULAS[name]
            want = [sign(np.ascontiguousarray(table)) for table in tables]
            self._write_matches(
                tables, want,
                lambda group, device, name=name: getattr(
                    tensorplay, "_foreach_" + name + "_")(group),
                f"_foreach_{name}_")

    def test_a_tie_in_a_rounding_formula_lands_the_same_way_on_both_lanes(self):
        # A lane exactly between two counts is where a rounding rule shows up,
        # and both walks have to keep the same count for it.
        tables = [self._table((13,)) + 0.5, self._table((2, 6)) + 0.5]
        want = [np.round(np.ascontiguousarray(table)) for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(tensorplay._foreach_round(group)),
            "_foreach_round of a tied group", rtol=0.0, atol=0.0)
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_round_(group),
            "_foreach_round_ of a tied group", rtol=0.0, atol=0.0)

    def test_a_count_group_finds_the_reciprocal_in_a_wider_width(self):
        # A count has no negative power to fold into itself, so the reciprocal
        # of a counting group is kept in the width that can hold it.
        tables = [np.array([2, 4, 8], dtype=np.int32),
                  np.array([1, 5], dtype=np.int64)]
        want = [(1.0 / table.astype(np.float32)).astype(np.float32)
                for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_reciprocal(group)),
            "_foreach_reciprocal of a counting group")

    def test_a_formula_keeps_the_half_width_it_holds(self):
        tables = [self._table((6,), dtype=np.float16, start=1.0),
                  self._table((2, 2), dtype=np.float16, start=2.0)]
        for name in ("neg", "abs", "floor", "sign", "sqrt", "exp"):
            want = [self.FORMULAS[name](
                        np.ascontiguousarray(table).astype(np.float32)
                    ).astype(np.float16)
                    for table in tables]
            self._read_matches(
                tables, want,
                lambda group, device, name=name: self._read(
                    getattr(tensorplay, "_foreach_" + name)(group)),
                f"_foreach_{name} of a half group", rtol=1e-3, atol=1e-3)

    BOUNDS = (("clamp_min", np.maximum), ("clamp_max", np.minimum),
              ("maximum", np.maximum), ("minimum", np.minimum))

    def test_a_bound_holds_the_group_up(self):
        # A bound that keeps a lane off the floor and one that keeps it off the
        # ceiling are the same walk seen from either end, and a partner bound
        # wins whichever of the two lanes is wider.
        tables = [self._table((5,)), self._table((2, 3)), self._table((4,))]
        partners = [self._table((5,), start=0.5), self._table((2, 3), start=1.0),
                    self._table((4,), start=1.5)]
        counts = [0.5, 2.0, 1.25]
        for name, fold in self.BOUNDS:
            call = getattr(tensorplay, "_foreach_" + name)
            self._read_matches(
                tables,
                [fold(np.ascontiguousarray(table), 1.0) for table in tables],
                lambda group, device, call=call: self._read(call(group, 1.0)),
                f"_foreach_{name} of one bound")
            self._read_matches(
                tables,
                [fold(np.ascontiguousarray(table), np.ascontiguousarray(partner))
                 for table, partner in zip(tables, partners)],
                lambda group, device, call=call: self._read(
                    call(group, self._group(partners, device))),
                f"_foreach_{name} of a partner bound")
            self._read_matches(
                tables,
                [fold(np.ascontiguousarray(table), count)
                 for table, count in zip(tables, counts)],
                lambda group, device, call=call: self._read(call(group, counts)),
                f"_foreach_{name} of a bound per tensor")

    def test_a_bound_writes_where_the_tensor_stands(self):
        tables = [self._table((5,)), self._table((2, 3)), self._table((4,))]
        partners = [self._table((5,), start=0.5), self._table((2, 3), start=1.0),
                    self._table((4,), start=1.5)]
        counts = [0.5, 2.0, 1.25]
        for name, fold in self.BOUNDS:
            write = getattr(tensorplay, "_foreach_" + name + "_")
            self._write_matches(
                tables,
                [fold(np.ascontiguousarray(table), 1.0) for table in tables],
                lambda group, device, write=write: write(group, 1.0),
                f"_foreach_{name}_ of one bound")
            self._write_matches(
                tables,
                [fold(np.ascontiguousarray(table), np.ascontiguousarray(partner))
                 for table, partner in zip(tables, partners)],
                lambda group, device, write=write: write(
                    group, self._group(partners, device)),
                f"_foreach_{name}_ of a partner bound")
            self._write_matches(
                tables,
                [fold(np.ascontiguousarray(table), count)
                 for table, count in zip(tables, counts)],
                lambda group, device, write=write: write(group, counts),
                f"_foreach_{name}_ of a bound per tensor")

    def test_a_group_raised_to_a_count_keeps_its_shape(self):
        tables = [self._table((5,), start=1.0), self._table((2, 3), start=1.0),
                  self._table((4,), start=2.0)]
        powers = [np.full((5,), 2.0, dtype=np.float32),
                  np.full((2, 3), 3.0, dtype=np.float32),
                  np.full((4,), 1.5, dtype=np.float32)]
        counts = [2.0, 3.0, 1.5]
        self._read_matches(
            tables, [np.power(np.ascontiguousarray(table), 2.0)
                     for table in tables],
            lambda group, device: self._read(tensorplay._foreach_pow(group, 2.0)),
            "_foreach_pow by one count")
        self._read_matches(
            tables, [np.power(np.ascontiguousarray(table), count)
                     for table, count in zip(tables, counts)],
            lambda group, device: self._read(
                tensorplay._foreach_pow(group, counts)),
            "_foreach_pow by a count per tensor")
        self._read_matches(
            tables, [np.power(np.ascontiguousarray(table), np.ascontiguousarray(power))
                     for table, power in zip(tables, powers)],
            lambda group, device: self._read(
                tensorplay._foreach_pow(group, self._group(powers, device))),
            "_foreach_pow by a payload of exponents")
        self._read_matches(
            tables, [np.power(np.float32(2.0), np.ascontiguousarray(table))
                     for table in tables],
            lambda group, device: self._read(
                tensorplay._foreach_pow(2.0, group)),
            "_foreach_pow of a shared base")

    def test_a_power_writes_where_the_tensor_stands(self):
        tables = [self._table((5,), start=1.0), self._table((2, 3), start=1.0),
                  self._table((4,), start=2.0)]
        powers = [np.full((5,), 2.0, dtype=np.float32),
                  np.full((2, 3), 3.0, dtype=np.float32),
                  np.full((4,), 1.5, dtype=np.float32)]
        counts = [2.0, 3.0, 1.5]
        self._write_matches(
            tables, [np.power(np.ascontiguousarray(table), 3.0)
                     for table in tables],
            lambda group, device: tensorplay._foreach_pow_(group, 3.0),
            "_foreach_pow_ by one count")
        self._write_matches(
            tables, [np.power(np.ascontiguousarray(table), count)
                     for table, count in zip(tables, counts)],
            lambda group, device: tensorplay._foreach_pow_(group, counts),
            "_foreach_pow_ by a count per tensor")
        self._write_matches(
            tables, [np.power(np.ascontiguousarray(table), np.ascontiguousarray(power))
                     for table, power in zip(tables, powers)],
            lambda group, device: tensorplay._foreach_pow_(
                group, self._group(powers, device)),
            "_foreach_pow_ by a payload of exponents")

    def test_two_partners_fold_into_the_group_weighed_by_one_count(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        left = [self._table((4,), start=1.0), self._table((3, 2), start=2.0),
                self._table((5,), start=3.0)]
        right = [self._table((4,), start=4.0), self._table((3, 2), start=5.0),
                 self._table((5,), start=6.0)]
        count = 2.0
        for op, sign in (
                (tensorplay._foreach_addcmul,
                 lambda held, one, other: held + count * (one * other)),
                (tensorplay._foreach_addcdiv,
                 lambda held, one, other: held + count * (one / other))):
            want = [sign(np.ascontiguousarray(table), np.ascontiguousarray(one),
                         np.ascontiguousarray(other))
                    for table, one, other in zip(tables, left, right)]
            self._read_matches(
                tables, want,
                lambda group, device, op=op: self._read(
                    op(group, self._group(left, device), self._group(right, device),
                       count)),
                f"{op.__name__} weighed by one count")

    def test_each_tensor_is_weighed_by_the_count_it_is_named(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        left = [self._table((4,), start=1.0), self._table((3, 2), start=2.0),
                self._table((5,), start=3.0)]
        right = [self._table((4,), start=4.0), self._table((3, 2), start=5.0),
                 self._table((5,), start=6.0)]
        counts = [0.5, -2.0, 3.0]
        for op, sign in (
                (tensorplay._foreach_addcmul,
                 lambda held, one, other, weight: held + weight * (one * other)),
                (tensorplay._foreach_addcdiv,
                 lambda held, one, other, weight: held + weight * (one / other))):
            for label, weights in (
                    ("a list of counts", counts),
                    ("a run the host holds",
                     tensorplay.from_numpy(np.array(counts, dtype=np.float32)))):
                want = [sign(np.ascontiguousarray(table), np.ascontiguousarray(one),
                             np.ascontiguousarray(other), weight)
                        for table, one, other, weight
                        in zip(tables, left, right, counts)]
                self._read_matches(
                    tables, want,
                    lambda group, device, op=op, weights=weights: self._read(
                        op(group, self._group(left, device),
                           self._group(right, device), weights)),
                    f"{op.__name__} weighed by {label}")

    def test_a_ternary_fold_writes_where_the_tensor_stands(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        left = [self._table((4,), start=1.0), self._table((3, 2), start=2.0),
                self._table((5,), start=3.0)]
        right = [self._table((4,), start=4.0), self._table((3, 2), start=5.0),
                 self._table((5,), start=6.0)]
        counts = [0.5, -2.0, 3.0]
        folded = [
            (tensorplay._foreach_addcmul_,
             lambda held, one, other, weight: held + weight * (one * other)),
            (tensorplay._foreach_addcdiv_,
             lambda held, one, other, weight: held + weight * (one / other)),
        ]
        for write, sign in folded:
            want = [sign(np.ascontiguousarray(table), np.ascontiguousarray(one),
                         np.ascontiguousarray(other), 1.5)
                    for table, one, other in zip(tables, left, right)]
            self._write_matches(
                tables, want,
                lambda group, device, write=write: write(
                    group, self._group(left, device), self._group(right, device),
                    1.5),
                f"{write.__name__} weighed by one count")
            want = [sign(np.ascontiguousarray(table), np.ascontiguousarray(one),
                         np.ascontiguousarray(other), weight)
                    for table, one, other, weight
                    in zip(tables, left, right, counts)]
            self._write_matches(
                tables, want,
                lambda group, device, write=write: write(
                    group, self._group(left, device), self._group(right, device),
                    counts),
                f"{write.__name__} weighed by a list of counts")
            self._write_matches(
                tables, want,
                lambda group, device, write=write: write(
                    group, self._group(left, device), self._group(right, device),
                    tensorplay.from_numpy(np.array(counts, dtype=np.float32))),
                f"{write.__name__} weighed by a run the host holds")

    def test_a_group_slides_toward_a_partner(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        ends = [self._table((4,), start=2.0), self._table((3, 2), start=4.0),
                self._table((5,), start=6.0)]
        runs = [np.full((4,), 0.25, dtype=np.float32),
                np.full((3, 2), 0.75, dtype=np.float32),
                np.full((5,), -0.5, dtype=np.float32)]
        counts = [0.25, 0.75, -0.5]

        def slide(held, stop, weight):
            return held + weight * (stop - held)

        want = [slide(np.ascontiguousarray(table), np.ascontiguousarray(stop), 0.4)
                for table, stop in zip(tables, ends)]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_lerp(group, self._group(ends, device), 0.4)),
            "_foreach_lerp slid by one count")
        want = [slide(np.ascontiguousarray(table), np.ascontiguousarray(stop), weight)
                for table, stop, weight in zip(tables, ends, counts)]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_lerp(
                    group, self._group(ends, device), counts)),
            "_foreach_lerp slid by a count per tensor")
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_lerp(
                    group, self._group(ends, device),
                    self._group(runs, device))),
            "_foreach_lerp slid by a payload of counts")

    def test_a_slide_writes_where_the_tensor_stands(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        ends = [self._table((4,), start=2.0), self._table((3, 2), start=4.0),
                self._table((5,), start=6.0)]
        runs = [np.full((4,), 0.25, dtype=np.float32),
                np.full((3, 2), 0.75, dtype=np.float32),
                np.full((5,), -0.5, dtype=np.float32)]
        counts = [0.25, 0.75, -0.5]

        def slide(held, stop, weight):
            return held + weight * (stop - held)

        want = [slide(np.ascontiguousarray(table), np.ascontiguousarray(stop), 0.4)
                for table, stop in zip(tables, ends)]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_lerp_(
                group, self._group(ends, device), 0.4),
            "_foreach_lerp_ slid by one count")
        want = [slide(np.ascontiguousarray(table), np.ascontiguousarray(stop), weight)
                for table, stop, weight in zip(tables, ends, counts)]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_lerp_(
                group, self._group(ends, device), counts),
            "_foreach_lerp_ slid by a count per tensor")
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_lerp_(
                group, self._group(ends, device), self._group(runs, device)),
            "_foreach_lerp_ slid by a payload of counts")

    def test_a_ternary_group_refuses_lists_that_do_not_pair(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            short = self._group(tables[:2], device)
            with self.assertRaisesRegex(
                    ValueError, "ternary tensor lists must have the same length"):
                tensorplay._foreach_addcmul(group, short, short, 1.0)
            with self.assertRaisesRegex(
                    ValueError,
                    "ternary tensor/scalar lists must have the same length"):
                tensorplay._foreach_addcmul(group, group, group, [1.0, 2.0])
            with self.assertRaisesRegex(
                    ValueError, "lerp lists must have the same length"):
                tensorplay._foreach_lerp(group, short, short)
            with self.assertRaisesRegex(
                    ValueError, "tensor/scalar lists must have the same length"):
                tensorplay._foreach_lerp(group, short, [1.0, 2.0, 3.0])
            pair = group[:2]
            with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected packed scalar Tensor to be of dimension 1"):
                tensorplay._foreach_addcdiv(
                    pair, pair, pair, tensorplay.tensor(np.float32(2.0)))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected length of scalars to match input of length 2 "
                    "but got 3 instead"):
                tensorplay._foreach_addcdiv(
                    pair, pair, pair,
                    tensorplay.from_numpy(np.array([1.0, 2.0, 3.0],
                                                   dtype=np.float32)))
        with self.assertRaisesRegex(
                RuntimeError, "Expected scalars to be on CPU, got"):
            group = self._group(tables, "vulkan")
            tensorplay._foreach_addcmul(
                group, group, group,
                tensorplay.from_numpy(np.array([1.0, 2.0, 3.0],
                                               dtype=np.float32)).to("vulkan"))

    def test_a_group_moves_into_the_shape_it_was_asked_to_answer(self):
        destinations = [self._table((4,)), self._table((3, 2), start=1.0),
                        self._table((5,), start=2.0)]
        partners = [self._table((4,), start=5.0), self._table((3, 2), start=7.0),
                    self._table((5,), start=9.0)]
        for label, tables, moved in (
                ("measured", destinations, partners),
                ("counted", [table.astype(np.int32) for table in destinations],
                 [table.astype(np.int32) for table in partners])):
            want = [np.ascontiguousarray(one.astype(np.asarray(two).dtype))
                    for one, two in zip(moved, tables)]
            self._read_matches(
                tables, want,
                lambda group, device, moved=moved: self._read(
                    tensorplay._foreach_copy(group, self._group(moved, device))),
                f"_foreach_copy of a {label} group")

    def test_a_move_answers_in_the_width_the_destination_holds(self):
        # A fresh copy keeps the width the group was asked to answer in and takes
        # its values from the partner, so a count written into a measurement stays
        # a measurement.  The host lane hands back the partner as it stands, which
        # keeps the partner's width instead, so this rule is pinned here alone.
        destinations = [self._table((4,)), self._table((3, 2), start=1.0)]
        partners = [table.astype(np.int32) for table in destinations]
        want = [np.ascontiguousarray(one.astype(np.float32)) for one in partners]
        group = self._group(destinations, "vulkan")
        held = tensorplay._foreach_copy(
            group, self._group(partners, "vulkan"))
        self._answers(self._read(held), want,
                      "_foreach_copy keeps the destination width", "vulkan")
        self._answers(self._read(group), [np.ascontiguousarray(table)
                                          for table in destinations],
                      "_foreach_copy leaves the group alone", "vulkan")

    def test_a_group_written_into_keeps_its_own_lanes(self):
        destinations = [self._table((4,)), self._table((3, 2), start=1.0)]
        partners = [self._table((4,), start=5.0), self._table((3, 2), start=7.0)]
        want = [np.ascontiguousarray(one) for one in partners]
        self._write_matches(
            destinations, want,
            lambda group, device: tensorplay._foreach_copy_(
                group, self._group(partners, device)),
            "_foreach_copy_ of a like group")

        counted = [table.astype(np.int32) for table in destinations]
        want = [np.ascontiguousarray(table.astype(np.int32))
                for table in destinations]
        self._write_matches(
            counted, want,
            lambda group, device: tensorplay._foreach_copy_(
                group, self._group(destinations, device)),
            "_foreach_copy_ of a measurement into a count")

        want = [np.ascontiguousarray(table.astype(np.int32).astype(np.float32))
                for table in partners]
        self._write_matches(
            destinations, want,
            lambda group, device: tensorplay._foreach_copy_(
                group, self._group(
                    [table.astype(np.int32) for table in partners], device)),
            "_foreach_copy_ of a count into a measurement")

    def test_a_group_handed_back_as_it_stands_is_copied_out_of_itself(self):
        tables = [self._table((4,)), self._table((3, 2), start=1.0),
                  self._table((5,), dtype=np.int32, start=2.0)]
        want = [np.ascontiguousarray(table) for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(tensorplay._foreach_clone(group)),
            "_foreach_clone of a group")

    def test_a_group_set_to_nothing_answers_zeroes(self):
        # The walk that sets a whole group to nothing is pinned in the form the
        # frontend reaches.  The fresh-answer registration is kept beside it so a
        # group asked on either lane answers the same key, but it has no wrapper
        # of its own to call here.
        tables = [self._table((4,)), self._table((3, 2), start=1.0),
                  self._table((5,), dtype=np.int32, start=2.0)]
        want = [np.zeros_like(np.ascontiguousarray(table)) for table in tables]
        self._write_matches(
            tables, want,
            lambda group, device: tensorplay._foreach_zero_(group),
            "_foreach_zero_ of a group")

    def test_a_moving_group_refuses_a_partner_list_that_runs_short(self):
        tables = [self._table((4,)), self._table((3, 2)), self._table((5,))]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            short = self._group(tables[:2], device)
            with self.assertRaisesRegex(
                    ValueError, "tensor list arguments must have the same length"):
                tensorplay._foreach_copy(group, short)
            with self.assertRaisesRegex(
                    ValueError, "tensor list arguments must have the same length"):
                tensorplay._foreach_copy_(group, short)

    def test_a_group_folded_down_answers_the_largest_lane_it_holds(self):
        # One lane answers each tensor of the group, in the width the tensor itself
        # is held in: a measurement folds to a measurement and a count to a count.
        # A fold answers a single lane, so the answer holds no axis at all.
        tables = [self._table((4,)), self._table((3, 2), start=1.0),
                  self._table((5,), dtype=np.int32, start=2.0),
                  self._table((6,), dtype=np.int64, start=3.0)]
        want = [np.array(table.max()) for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(tensorplay._foreach_max(group)),
            "_foreach_max of a group")

    def test_a_fold_down_names_a_lane_that_is_not_a_number(self):
        # A lane that is not a number wins the fold, on either lane, so the answer
        # is a fold that holds one rather than the largest lane it can compare.
        tables = [self._table((4,)), self._table((3, 2), start=1.0)]
        tables[0][1] = np.float32(np.nan)
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            held = self._read(tensorplay._foreach_max(group))
            self.assertTrue(bool(np.isnan(held[0])),
                            msg=f"_foreach_max on {device}")
            self.assertEqual(float(held[1]), float(tables[1].max()),
                             msg=f"_foreach_max on {device}")

    def test_a_fold_down_refuses_a_tensor_holding_no_lane(self):
        # A fold over nothing has no starting value, so a group handed to it
        # refuses before an answer of no lane can be read back.
        for device in ("vulkan", "cpu"):
            empty = [tensorplay.tensor([], dtype=tensorplay.float32).to(device)]
            with self.assertRaisesRegex(
                    RuntimeError, "Expected reduction dim to be specified"):
                tensorplay._foreach_max(empty)

    def test_a_group_folded_down_answers_the_length_of_a_tensor(self):
        # A length is totalled in the width a measurement folds in, and the power
        # names how the total is taken: a sum of magnitudes, a root of a sum of
        # powers, the largest magnitude, or the count of lanes holding a number.
        tables = [self._table((4,)), self._table((3, 2), start=1.0),
                  self._table((5,), start=2.0)]
        for power in (1, 2, 3):
            want = [np.array(float(np.sum(np.abs(table) ** power))
                             ** (1.0 / power)).astype(np.float32)
                    for table in tables]
            self._read_matches(
                tables, want,
                lambda group, device, power=power: self._read(
                    tensorplay._foreach_norm(group, power)),
                f"_foreach_norm of a group at power {power}")

        want = [np.array(np.abs(table).max()) for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_norm(group, float("inf"))),
            "_foreach_norm of a group at the infinity power")

        want = [np.array(float(np.count_nonzero(table))).astype(np.float32)
                for table in tables]
        self._read_matches(
            tables, want,
            lambda group, device: self._read(
                tensorplay._foreach_norm(group, 0)),
            "_foreach_norm of a group at the zero power")

    def test_a_fold_down_refuses_a_count_group_handed_no_width(self):
        # The walk rides the fold a lone tensor reaches, and that fold answers only
        # a payload that measures, so a group of counts is refused until it is told
        # to hold in a measuring width.  The host lane folds a count without asking
        # for a width, which is why this refusal is pinned here alone.
        counted = [self._table((4,), dtype=np.int32),
                   self._table((3, 2), dtype=np.int32, start=1.0)]
        group = self._group(counted, "vulkan")
        with self.assertRaisesRegex(
                TypeError, "expects a floating or complex input"):
            tensorplay._foreach_norm(group, 2.0)

    def test_a_fold_down_holds_a_group_in_a_width_it_was_told_to_answer_in(self):
        # A group of counts asked to fold in a measuring width is widened before the
        # walk starts, so the length is totalled in the wider word rather than the
        # narrow one it counts in.
        tables = [self._table((4,)), self._table((3, 2), start=1.0)]
        counted = [table.astype(np.int32) for table in tables]
        want = [np.array(
                    float(np.sum(np.abs(count.astype(np.float32)) ** 2)) ** 0.5)
                .astype(np.float32) for count in counted]
        self._read_matches(
            counted, want,
            lambda group, device: self._read(
                tensorplay._foreach_norm(group, 2.0, dtype=tensorplay.float32)),
            "_foreach_norm of a counted group held in a measuring width")

    def test_a_group_folded_down_answers_the_sum_a_length_is_built_from(self):
        tables = [self._table((4,)), self._table((3, 2), start=1.0),
                  self._table((5,), start=2.0)]
        for power in (1, 2, 3):
            want = [np.array(float(np.sum(np.abs(table) ** power)))
                    .astype(np.float32) for table in tables]
            self._read_matches(
                tables, want,
                lambda group, device, power=power: self._read(
                    tensorplay._foreach_powsum(group, power)),
                f"_foreach_powsum of a group at power {power}")

        counted = [table.astype(np.int32) for table in tables]
        want = [np.array(float(np.sum(np.abs(count.astype(np.float32)) ** 2)))
                .astype(np.float32) for count in counted]
        self._read_matches(
            counted, want,
            lambda group, device: self._read(
                tensorplay._foreach_powsum(group, 2, dtype=tensorplay.float32)),
            "_foreach_powsum of a counted group held in a measuring width")

    def test_a_group_of_products_answers_each_product_it_names(self):
        # Every tensor is the left hand of a product of its own, so one tensor may
        # pair under shapes another tensor never sees; the answer is the product
        # the pair of shapes names.
        lefts = [self._table((3, 4)), self._table((2, 2), start=1.0),
                 self._table((5, 3), start=2.0)]
        rights = [self._table((4, 2), start=5.0), self._table((2, 5), start=7.0),
                  self._table((3, 1), start=9.0)]
        want = [np.ascontiguousarray(one @ two) for one, two in zip(lefts, rights)]
        self._read_matches(
            lefts, want,
            lambda group, device: self._read(
                tensorplay._foreach_mm(group, self._group(rights, device))),
            "_foreach_mm of a group")

    def test_a_group_of_products_refuses_a_partner_list_that_runs_short(self):
        lefts = [self._table((3, 4)), self._table((2, 2)), self._table((5, 3))]
        rights = [self._table((4, 2)), self._table((2, 5))]
        for device in ("vulkan", "cpu"):
            with self.assertRaisesRegex(
                    ValueError, "tensor list arguments must have the same length"):
                tensorplay._foreach_mm(
                    self._group(lefts, device), self._group(rights, device))

    def test_a_clean_group_is_divided_by_the_scale_it_was_given(self):
        # The scale is held as a single lane, and every lane of every tensor is
        # divided by it; a scale of one leaves the group where it was.
        tables = [self._table((3,)), self._table((2, 4), start=1.0)]
        for inv_scale in (0.25, 1.0):
            want = [np.ascontiguousarray(table * np.float32(inv_scale))
                    for table in tables]
            for device in ("vulkan", "cpu"):
                group = self._group(tables, device)
                found = self._unscale(group, device, inv_scale)
                self.assertEqual(float(found.cpu().numpy()), 0.0, msg=device)
                self._answers(
                    self._read(group), want,
                    f"amp unscale by {inv_scale}", device)

    def test_a_group_with_a_lane_that_is_not_a_number_raises_the_flag(self):
        tables = [np.array([2.0, np.inf], dtype=np.float32),
                  np.array([4.0, -np.inf], dtype=np.float32),
                  np.array([np.nan, 1.0], dtype=np.float32)]
        divided = [np.ascontiguousarray(table * np.float32(0.5))
                   for table in tables]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            found = self._unscale(group, device, 0.5)
            self.assertEqual(float(found.cpu().numpy()), 1.0, msg=device)
            if device == "vulkan":
                # The device lane divides every group it is handed, whether or
                # not the scan raised the flag; the host lane leaves a flagged
                # group alone.
                np.testing.assert_allclose(
                    self._read(group), divided, rtol=1e-5, atol=1e-6,
                    equal_nan=True)

    def test_a_group_is_still_divided_after_the_flag_was_raised(self):
        # A flag that is already up is not a reason to skip the division: the
        # flag is only ever raised, never lowered, by this walk.  The host lane
        # returns before touching the group in this case.
        tables = [self._table((4,))]
        want = [np.ascontiguousarray(table * np.float32(0.5))
                for table in tables]
        device = "vulkan"
        group = self._group(tables, device)
        found = tensorplay.tensor(1.0, dtype=tensorplay.float32, device=device)
        found = self._unscale(group, device, 0.5, found)
        self.assertEqual(float(found.cpu().numpy()), 1.0)
        self._answers(self._read(group), want,
                      "amp unscale after the flag was raised", device)

    def test_a_group_that_names_no_tensor_leaves_the_flag_alone(self):
        for device in ("vulkan", "cpu"):
            found = self._unscale([], device, 0.5)
            self.assertEqual(float(found.cpu().numpy()), 0.0, msg=device)

    def test_a_half_group_is_divided_in_a_wider_width(self):
        tables = [self._table((4,), dtype=np.float16),
                  self._table((3,), dtype=np.float16, start=1.0)]
        want = [np.ascontiguousarray(
            (table.astype(np.float32) * np.float32(0.25)).astype(np.float16))
            for table in tables]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            found = self._unscale(group, device, 0.25)
            self.assertEqual(float(found.cpu().numpy()), 0.0, msg=device)
            self._answers(self._read(group), want,
                          "amp unscale of a half group", device)

    def test_a_flag_or_scale_that_is_not_one_float_lane_is_refused(self):
        # The flag is written as one float lane and the scale is read as one, so
        # a wider flag or a counted one is refused on the device lane.  The host
        # lane reads the first word of whatever it is handed.
        device = "vulkan"
        group = self._group([self._table((2,))], device)
        scale = tensorplay.tensor(0.5, dtype=tensorplay.float32, device=device)
        wide = tensorplay.tensor(
            [0.0, 0.0], dtype=tensorplay.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "1-element tensors"):
            tensorplay._amp_foreach_non_finite_check_and_unscale_(
                group, wide, scale)
        counted = tensorplay.tensor([0], dtype=tensorplay.int32, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be float32"):
            tensorplay._amp_foreach_non_finite_check_and_unscale_(
                group, counted, scale)

    def test_a_counted_group_is_refused(self):
        tables = [np.array([1, 2], dtype=np.int32)]
        for device in ("vulkan", "cpu"):
            group = self._group(tables, device)
            found = tensorplay.tensor(
                0.0, dtype=tensorplay.float32, device=device)
            scale = tensorplay.tensor(
                0.5, dtype=tensorplay.float32, device=device)
            with self.assertRaisesRegex(
                    NotImplementedError, "unsupported dtype"):
                tensorplay._amp_foreach_non_finite_check_and_unscale_(
                    group, found, scale)
            self.assertEqual(float(found.cpu().numpy()), 0.0, msg=device)


class TestVulkanAmpScale(unittest.TestCase):
    # A step ends by turning the flag the gradient scan raised into a new loss
    # scale.  The update watches two one-lane tensors - the scale and the count
    # of clean steps since it last changed - and a raised flag, a clean count
    # and the step that reaches the growth interval each take a different
    # branch.  The device lane keeps the count and the flag resident; the host
    # lane is the cross-check for the branches the two lanes share.

    @staticmethod
    def _scale(device, value):
        return tensorplay.tensor(
            value, dtype=tensorplay.float32, device=device)

    @staticmethod
    def _tracker(device, value):
        return tensorplay.tensor(
            value, dtype=tensorplay.int32, device=device)

    @staticmethod
    def _flag(device, value):
        return tensorplay.tensor(
            value, dtype=tensorplay.float32, device=device)

    @staticmethod
    def _read(held):
        return held.cpu().numpy().item()

    def test_a_step_that_found_a_bad_lane_backs_the_scale_off(self):
        for device in ("vulkan", "cpu"):
            scale = self._scale(device, 128.0)
            tracker = self._tracker(device, 7)
            found = self._flag(device, 1.0)
            tensorplay._amp_update_scale_(
                scale, tracker, found, 2.0, 0.5, 100)
            self.assertEqual(self._read(scale), 64.0, msg=device)
            self.assertEqual(self._read(tracker), 0, msg=device)
            # The flag is read, never written, by the update.
            self.assertEqual(self._read(found), 1.0, msg=device)

    def test_a_clean_step_counts_toward_the_growth_interval(self):
        for device in ("vulkan", "cpu"):
            scale = self._scale(device, 128.0)
            tracker = self._tracker(device, 0)
            found = self._flag(device, 0.0)
            for counted in (1, 2):
                tensorplay._amp_update_scale_(
                    scale, tracker, found, 2.0, 0.5, 3)
                self.assertEqual(self._read(tracker), counted, msg=device)
                self.assertEqual(self._read(scale), 128.0, msg=device)
            tensorplay._amp_update_scale_(
                scale, tracker, found, 2.0, 0.5, 3)
            self.assertEqual(self._read(tracker), 0, msg=device)
            self.assertEqual(self._read(scale), 256.0, msg=device)

    def test_a_growth_that_would_overflow_keeps_the_previous_scale(self):
        # A growth past the largest finite single-precision value keeps the
        # previous scale and still clears the count.  The host lane has no bound
        # check and stores an infinite scale instead.
        device = "vulkan"
        kept = float(np.float32(3.0e38))
        scale = tensorplay.tensor(
            [3.0e38], dtype=tensorplay.float32, device=device)
        tracker = tensorplay.tensor([0], dtype=tensorplay.int32, device=device)
        found = tensorplay.tensor(
            [0.0], dtype=tensorplay.float32, device=device)
        tensorplay._amp_update_scale_(scale, tracker, found, 2.0, 0.5, 1)
        self.assertEqual(self._read(scale), kept)
        self.assertTrue(np.isfinite(self._read(scale)))
        self.assertEqual(self._read(tracker), 0)

    def test_a_flag_lane_names_a_raised_flag_by_not_being_zero(self):
        # The flag is a truth: any lane that is not zero, including a negative
        # one or a not-a-number, backs the scale off.  The host lane asks
        # whether the flag is positive.
        device = "vulkan"
        for flag in (-2.0, np.nan):
            scale = self._scale(device, 128.0)
            tracker = self._tracker(device, 5)
            found = self._flag(device, flag)
            tensorplay._amp_update_scale_(
                scale, tracker, found, 2.0, 0.5, 100)
            self.assertEqual(self._read(scale), 64.0, msg=flag)
            self.assertEqual(self._read(tracker), 0, msg=flag)

    def test_a_scale_or_count_that_is_not_one_lane_is_refused(self):
        device = "vulkan"
        scale = self._scale(device, 128.0)
        tracker = self._tracker(device, 0)
        found = self._flag(device, 0.0)
        wide = tensorplay.tensor(
            [128.0, 128.0], dtype=tensorplay.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "1-element tensors"):
            tensorplay._amp_update_scale_(wide, tracker, found, 2.0, 0.5, 3)
        counted = tensorplay.tensor(
            128, dtype=tensorplay.int32, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be float32"):
            tensorplay._amp_update_scale_(
                counted, tracker, found, 2.0, 0.5, 3)
        wide_tracker = tensorplay.tensor(
            [0, 0], dtype=tensorplay.int32, device=device)
        with self.assertRaisesRegex(RuntimeError, "1-element tensors"):
            tensorplay._amp_update_scale_(
                scale, wide_tracker, found, 2.0, 0.5, 3)
        half_tracker = tensorplay.tensor(
            0, dtype=tensorplay.float16, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be int32"):
            tensorplay._amp_update_scale_(
                scale, half_tracker, found, 2.0, 0.5, 3)

    def test_a_scaler_unscales_and_updates_on_the_device(self):
        # The whole step the two amp entries exist for: a scaled backward pass
        # leaves a multiple of the scale in the gradient, the unscale divides it
        # back and flags nothing, and the update counts the clean step.  The
        # gradient is a power of two multiple of the truth, so the division is
        # exact.
        from tensorplay.amp import GradScaler
        from tensorplay.optim import SGD

        device = "vulkan"
        param = tensorplay.tensor(
            [1.0, -2.0], dtype=tensorplay.float32, device=device,
            requires_grad=True)
        optimizer = SGD([param], lr=0.25)
        scaler = GradScaler(device, init_scale=8.0, growth_interval=2)
        scaler.scale((param * param).sum()).backward()
        np.testing.assert_allclose(
            param.grad.cpu().numpy(), [16.0, -32.0], rtol=1e-6, atol=1e-6)
        scaler.unscale_(optimizer)
        np.testing.assert_allclose(
            param.grad.cpu().numpy(), [2.0, -4.0], rtol=1e-6, atol=1e-6)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 8.0)


class TestVulkanRowAccumulation(unittest.TestCase):
    # A list of row codes may name one row several times, and every row it names
    # has to reach the destination.  Nothing on this device locks a row while it
    # is being totalled, so the walk runs from the destination instead: the group
    # that owns a row reads the whole list and adds the rows it names in the
    # order the list states them.  The answer is therefore the same every time,
    # not merely close, and it is the answer the host walk gives.  Range is
    # settled inside the walk, as it is for the reads and writes beside it.

    SHAPES = ((5,), (5, 4), (2, 5, 3), (2, 3, 5, 2))

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @staticmethod
    def _codes(values, dtype=np.int64):
        return TestVulkanRowAccumulation._device(np.array(values, dtype=dtype))

    @staticmethod
    def _operands(shape, dim, codes, dtype):
        # Small whole numbers keep a sum exact in either element type, so a
        # wrong lane shows up as a wrong number rather than as rounding.
        axis = dim if dim >= 0 else dim + len(shape)
        seed = [len(shape), dim + 4, len(codes), *shape]
        rng = np.random.default_rng(seed)
        base = rng.integers(-3, 4, size=shape).astype(dtype)
        source_shape = list(shape)
        source_shape[axis] = len(codes)
        added = rng.integers(-3, 4, size=source_shape).astype(dtype)
        return base, added

    @classmethod
    def _want(cls, base, dim, codes, added):
        axis = dim if dim >= 0 else dim + base.ndim
        want = base.astype(np.float64).copy()
        for slot, code in enumerate(codes):
            take = [slice(None)] * base.ndim
            take[axis] = code
            give = [slice(None)] * base.ndim
            give[axis] = slot
            want[tuple(take)] += added[tuple(give)]
        return want

    def _sum_matches(self, shape, dim, codes, dtype=np.float32):
        base, added = self._operands(shape, dim, codes, dtype)
        want = self._want(base, dim, codes, added)
        name = (f"shape={shape} dim={dim} codes={codes} "
                f"dtype={np.dtype(dtype).name}")
        got = self._device(base).index_add_(
            dim, self._codes(codes), self._device(added)).cpu().numpy()
        np.testing.assert_array_equal(got, want, err_msg=f"device: {name}")
        host = tensorplay.index_add(
            tensorplay.from_numpy(base.copy()),
            dim,
            tensorplay.from_numpy(np.array(codes, np.int64)),
            tensorplay.from_numpy(added.copy()),
        ).cpu().numpy()
        np.testing.assert_array_equal(host, want, err_msg=f"host: {name}")

    def test_the_rows_a_list_names_total_in_the_order_it_states(self):
        for shape in self.SHAPES:
            for dim in range(len(shape)):
                size = shape[dim]
                codes = [0, size - 1, size // 2, 1, size - 1]
                for dtype in (np.float32, np.float16):
                    self._sum_matches(shape, dim, codes, dtype)

    def test_an_axis_the_texels_do_not_offer_accumulates_the_same_way(self):
        # The row axis of a texture is the one just above the last, so every
        # other axis - and a payload of one axis - travels through linear
        # storage and back.  The walk is the same, so the answer had better be.
        for shape, dim in (((5, 4), 1), ((2, 5, 3), 2), ((2, 3, 5, 2), 0),
                           ((2, 3, 5, 2), 3), ((5,), 0)):
            size = shape[dim]
            for dtype in (np.float32, np.float16):
                self._sum_matches(shape, dim, [size - 1, 0, 1, size - 1], dtype)

    def test_a_row_that_several_codes_name_is_added_to_that_many_times(self):
        for dtype in (np.float32, np.float16):
            self._sum_matches((6, 4), 0, [2] * 8, dtype)
            self._sum_matches((3, 7), 1, [3, 3, 0, 3, 6, 6], dtype)

    def test_the_answer_to_a_repeated_row_is_the_same_every_time(self):
        # A fold that raced would settle differently on two runs; the whole
        # point of walking from the destination is that it cannot.
        base, added = self._operands((6, 4), 0, [2, 2, 2, 2], np.float32)
        codes = self._codes([2, 2, 2, 2])
        source = self._device(added)
        first = self._device(base).index_add_(0, codes, source).cpu().numpy()
        second = self._device(base).index_add_(0, codes, source).cpu().numpy()
        np.testing.assert_array_equal(
            first, second, err_msg="two runs of the same fold")
        np.testing.assert_array_equal(
            first, self._want(base, 0, [2, 2, 2, 2], added),
            err_msg="a row named four times")

    def test_a_row_of_a_width_that_is_not_a_group_wide_keeps_its_lanes(self):
        # Five features is one whole group of four and a leftover, and a half
        # payload pairs its lanes into words, so the last lane of a row can sit
        # in the same word as the first of the next.  Each lane must still land
        # in its own row.
        for dtype in (np.float32, np.float16):
            self._sum_matches((4, 5), 0, [0, 2, 0, 3, 1], dtype)
            self._sum_matches((4, 5), 1, [0, 1, 1, 3], dtype)
            self._sum_matches((13,), 0, [0, 12, 5, 5, 6], dtype)
            self._sum_matches((3, 5, 7), 1, [0, 2, 1, 1], dtype)

    def test_a_payload_too_long_for_a_texel_walk_accumulates_flat(self):
        for dtype in (np.float32, np.float16):
            self._sum_matches((200000,), 0,
                              [0, 1, 199999, 100000, 100000, 3], dtype)

    def test_the_out_of_place_form_leaves_its_canvas_alone(self):
        codes = [0, 3, 3, 1]
        base, added = self._operands((5, 4), 0, codes, np.float32)
        canvas = self._device(base)
        out = tensorplay.index_add(canvas, 0, self._codes(codes),
                                   self._device(added))
        np.testing.assert_array_equal(
            out.cpu().numpy(), self._want(base, 0, codes, added),
            err_msg="the out-of-place form")
        np.testing.assert_array_equal(
            canvas.cpu().numpy(), base, err_msg="a write left its input alone")
        self.assertEqual(out.dtype, tensorplay.float32)

    def test_the_forms_that_fold_nothing_answer_with_what_they_hold(self):
        base, _ = self._operands((6, 4), 0, [], np.float32)
        held = self._device(base).index_add_(
            0, self._codes([]), self._device(np.zeros((0, 4), np.float32)))
        np.testing.assert_array_equal(
            held.cpu().numpy(), base, err_msg="a list of no codes")
        self._sum_matches((6, 4), 1, [3], np.float32)
        self._sum_matches((6, 4), 1, [3], np.float16)

    def test_a_code_beyond_the_rows_the_destination_offers_changes_nothing(self):
        base = np.arange(24, dtype=np.float32).reshape(6, 4)
        added = np.full((3, 4), 5.0, np.float32)
        got = self._device(base).index_add_(
            0, self._codes([0, -1, 6]), self._device(added)).cpu().numpy()
        np.testing.assert_array_equal(got[0], base[0] + 5.0,
                                      err_msg="the row an in-range code names")
        np.testing.assert_array_equal(got[1:], base[1:],
                                      err_msg="rows no in-range code names")

    def test_a_list_of_counts_in_either_width_accumulates_alike(self):
        base, added = self._operands((6, 4), 0, [0, 5, 5, 2], np.float32)
        want = self._want(base, 0, [0, 5, 5, 2], added)
        for width in (np.int32, np.int64):
            got = self._device(base).index_add_(
                0, self._codes([0, 5, 5, 2], width),
                self._device(added)).cpu().numpy()
            np.testing.assert_array_equal(
                got, want, err_msg=f"a list of {np.dtype(width).name} codes")

    def test_a_subscript_write_can_add_the_rows_it_names(self):
        # Two of these codes name row zero, which a plain write leaves unsettled
        # and an adding one folds into that row in the order the list states.
        base = np.zeros((5, 4), np.float32)
        added = np.array([[1.0, 1.0, 1.0, 1.0],
                          [2.0, 2.0, 2.0, 2.0],
                          [4.0, 4.0, 4.0, 4.0]], np.float32)
        codes = [0, 2, 0]
        got = tensorplay.index_put_(
            self._device(base), [self._codes(codes)], self._device(added),
            True).cpu().numpy()
        want = self._want(base, 0, codes, added)
        np.testing.assert_array_equal(
            got, want, err_msg="a subscript write that adds")

    def test_the_expert_accumulation_a_mixture_of_experts_step_does(self):
        # One canvas of tokens, and one dispatch per expert adding the rows that
        # expert was routed to.  Whole tokens repeat across experts, so the
        # folds are the point of the exercise.
        tokens, hidden, experts = 256, 1024, 4
        rng = np.random.default_rng([tokens, hidden, experts])
        canvas = np.zeros((tokens, hidden), np.float32)
        routed = rng.integers(-3, 4, size=(experts * 32, hidden)).astype(
            np.float32)
        assignment = rng.integers(0, tokens, size=(experts, 32)).astype(np.int64)
        payload = self._device(canvas)
        for expert in range(experts):
            codes = assignment[expert]
            source = routed[expert * 32:(expert + 1) * 32]
            payload = payload.index_add_(0, self._codes(codes.tolist()),
                                         self._device(source))
        want = canvas.astype(np.float64)
        for expert in range(experts):
            for slot, code in enumerate(assignment[expert]):
                want[code] += routed[expert * 32 + slot]
        np.testing.assert_array_equal(
            payload.cpu().numpy(), want, err_msg="four experts into one canvas")

    def test_a_walk_the_destination_and_the_list_both_make_long_is_answered_off_it(
            self):
        length = 300000
        self._sum_matches((length,), 0, [0, 1, length - 1], np.float32)
        canvas = self._device(np.zeros((length,), np.float32))
        codes = self._codes(list(range(length)))
        source = self._device(np.ones((length,), np.float32))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            canvas.index_add_(0, codes, source)

    def test_operands_that_do_not_line_up_are_answered_off_it(self):
        payload = self._device(np.zeros((5, 4), np.float32))
        with self.assertRaisesRegex(RuntimeError, "same rank"):
            tensorplay.index_add(payload, 0, self._codes([0, 1]),
                                 self._device(np.ones(2, np.float32)))
        with self.assertRaisesRegex(RuntimeError, "match the index length"):
            tensorplay.index_add(payload, 0, self._codes([0, 1]),
                                 self._device(np.ones((3, 4), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "off-axis sizes"):
            tensorplay.index_add(payload, 0, self._codes([0, 1]),
                                 self._device(np.ones((2, 6), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.index_add(self._device(np.zeros((5, 4), np.int32)),
                                 0, self._codes([0, 1]),
                                 self._device(np.ones((2, 4), np.int32)))
        with self.assertRaisesRegex(RuntimeError, "dtype must match"):
            tensorplay.index_add(payload, 0, self._codes([0, 1]),
                                 self._device(np.ones((2, 4), np.float16)))
        with self.assertRaisesRegex(RuntimeError, "Int64 or Int32"):
            tensorplay.index_add(payload, 0, self._codes([0.0, 1.0], np.float32),
                                 self._device(np.ones((2, 4), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
            tensorplay.index_add(payload, 2, self._codes([0, 1]),
                                 self._device(np.ones((2, 4), np.float32)))


class TestVulkanRowFill(unittest.TestCase):
    # A fill hands one number to every row a list of codes names.  Nothing races
    # for a row here, because the number is the same wherever it lands: the walk
    # records which rows are named and then rewrites a whole texel or word at a
    # time, so a code the list repeats costs nothing beyond the first and the
    # rows it left alone hold what they held before.

    SHAPES = ((5,), (5, 4), (2, 5, 3), (2, 3, 5, 2))

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @staticmethod
    def _codes(values, dtype=np.int64):
        return TestVulkanRowFill._device(np.array(values, dtype=dtype))

    @staticmethod
    def _canvas(shape, dtype=np.float32, seed=0):
        # Small whole numbers keep either element type exact, so a lane written
        # in the wrong place shows up as a wrong number rather than as rounding.
        rng = np.random.default_rng([len(shape), seed, *shape])
        return rng.integers(-3, 4, size=shape).astype(dtype)

    @classmethod
    def _want(cls, base, dim, codes, value):
        axis = dim if dim >= 0 else dim + base.ndim
        rows = base.shape[axis]
        out = base.astype(np.float64)
        for code in codes:
            slot = int(code)
            if slot < 0:
                # A code given from the end names the row that many from it.
                slot += rows
            if slot < 0 or slot >= rows:
                continue
            take = [slice(None)] * out.ndim
            take[axis] = slot
            out[tuple(take)] = value
        return out

    def _filled(self, shape, dim, codes, value, dtype=np.float32):
        base = self._canvas(shape, dtype)
        name = (f"shape={shape} dim={dim} codes={codes} value={value} "
                f"dtype={np.dtype(dtype).name}")
        got = self._device(base).index_fill_(
            dim, self._codes(codes), value).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, dim, codes, value),
                                      err_msg=f"device: {name}")
        return base, got

    def test_every_row_a_list_names_takes_the_number_and_no_other_does(self):
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                for dtype in (np.float32, np.float16):
                    self._filled(shape, dim, [1, 3], 7.0, dtype)

    def test_a_row_named_on_the_host_walk_takes_the_same_number(self):
        # Only where every code fits the rows it names: the host walk answers a
        # code beyond them with an error rather than by writing nothing.
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                base = self._canvas(shape)
                codes = list(range(0, base.shape[dim if dim >= 0 else dim + len(shape)]))
                if not codes:
                    continue
                host = tensorplay.index_fill(
                    tensorplay.from_numpy(base.copy()),
                    dim,
                    tensorplay.from_numpy(np.array(codes, np.int64)),
                    7.0).cpu().numpy()
                np.testing.assert_array_equal(
                    host, self._want(base, dim, codes, 7.0),
                    err_msg=f"host: shape={shape} dim={dim}")

    def test_an_axis_the_texels_do_not_offer_is_filled_the_same_way(self):
        # The axes above the one the texture carries as its rows reach the linear
        # walk, which fills a dense copy and carries it back.
        for shape, dim in (((2, 3, 5, 2), 0), ((2, 3, 5, 2), 1),
                           ((2, 5, 3), 0), ((5, 4), 1), ((5,), 0)):
            self._filled(shape, dim, [0, 2], -3.0)

    def test_a_code_given_from_the_end_names_the_row_that_many_from_it(self):
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                self._filled(shape, dim, [-1, -2], 9.0)

    def test_a_row_that_several_codes_name_holds_the_number_they_all_give(self):
        base, once = self._filled((4, 5), 0, [2], 5.0)
        _, repeated = self._filled((4, 5), 0, [2, 2, 2], 5.0)
        np.testing.assert_array_equal(once, repeated)

    def test_the_number_of_the_last_request_is_the_one_that_stands(self):
        # Each request records its own rows on a fresh mark, so a second fill
        # cannot find a row still named by the first.
        base = self._canvas((6, 4))
        canvas = self._device(base)
        canvas.index_fill_(0, self._codes([0, 1, 2]), 1.0)
        canvas.index_fill_(0, self._codes([3, 4]), 2.0)
        got = canvas.cpu().numpy()
        np.testing.assert_array_equal(
            got, self._want(self._want(base, 0, [0, 1, 2], 1.0), 0, [3, 4], 2.0))

    def test_a_row_of_a_width_that_is_not_a_group_keeps_its_lanes(self):
        # A row that does not fill its last word shares it with the row below, so
        # the lanes of an unnamed row have to come back as they were.
        for shape, dim in (((4, 5), 0), ((13,), 0), ((3, 5, 7), 1),
                           ((2, 3, 5), 0), ((5, 3), 1)):
            for dtype in (np.float32, np.float16):
                self._filled(shape, dim, [1], 6.0, dtype)

    def test_a_half_canvas_keeps_the_rounding_its_storage_asks(self):
        base = self._canvas((4, 6), np.float16)
        got = self._device(base).index_fill_(
            0, self._codes([1, 3]), 0.1).cpu().numpy()
        named = np.float16(0.1)
        want = base.astype(np.float16).copy()
        want[[1, 3]] = named
        np.testing.assert_array_equal(got, want)

    def test_the_number_may_arrive_in_a_payload_of_one(self):
        base = self._canvas((5, 4))
        got = self._device(base).index_fill_(
            1, self._codes([0, 2]),
            self._device(np.array(7.0, np.float32))).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 1, [0, 2], 7.0))
        with self.assertRaisesRegex(RuntimeError, "single number"):
            self._device(base).index_fill_(
                1, self._codes([0]), self._device(np.array([7.0, 8.0], np.float32)))

    def test_the_out_of_place_form_leaves_its_canvas_alone(self):
        base = self._canvas((2, 5, 3))
        canvas = self._device(base)
        answered = canvas.index_fill(0, self._codes([1]), 7.0).cpu().numpy()
        np.testing.assert_array_equal(answered, self._want(base, 0, [1], 7.0))
        np.testing.assert_array_equal(canvas.cpu().numpy(), base)

    def test_the_forms_that_name_nothing_answer_with_what_they_hold(self):
        base = self._canvas((3, 4))
        np.testing.assert_array_equal(
            self._device(base).index_fill_(0, self._codes([]), 7.0).cpu().numpy(),
            base)
        np.testing.assert_array_equal(
            self._device(np.zeros((0, 4), np.float32)).index_fill_(
                0, self._codes([1]), 7.0).cpu().numpy().shape, (0, 4))
        # A single code that spells out no axis at all names one row.
        got = self._device(base).index_fill_(
            0, self._device(np.array(2, np.int64)), 7.0).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, [2], 7.0))

    def test_a_code_beyond_the_rows_the_canvas_offers_changes_nothing(self):
        base = self._canvas((4, 5))
        for codes in ([9], [-9], [0, 9, -9]):
            got = self._device(base).index_fill_(
                0, self._codes(codes), 7.0).cpu().numpy()
            np.testing.assert_array_equal(
                got, self._want(base, 0, codes, 7.0), err_msg=f"codes={codes}")

    def test_a_list_of_counts_in_either_width_fills_alike(self):
        base = self._canvas((2, 3, 5))
        for dtype in (np.int32, np.int64):
            got = self._device(base).index_fill_(
                1, self._codes([0, 2], dtype), 7.0).cpu().numpy()
            np.testing.assert_array_equal(
                got, self._want(base, 1, [0, 2], 7.0), err_msg=f"codes={dtype}")

    def test_a_payload_too_long_for_a_texel_walk_is_filled_flat(self):
        base = np.arange(200000, dtype=np.float32)
        codes = [0, 7, 99999, 199999]
        got = self._device(base).index_fill_(0, self._codes(codes), -1.0).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, codes, -1.0))

    def test_the_rows_a_decoder_blanks_are_written_in_one_request(self):
        # The padding an autoregressive step drops is a stride of rows of the
        # token canvas, and every one of them takes the same number.
        base = self._canvas((2048, 1024), np.float16, seed=3)
        codes = list(range(0, 2048, 4))
        got = self._device(base).index_fill_(
            0, self._codes(codes), 0.0).cpu().numpy()
        want = self._want(base, 0, codes, 0.0)
        np.testing.assert_array_equal(got, want)
        self.assertTrue((got[codes] == 0).all())
        self.assertTrue((got[np.setdiff1d(np.arange(2048), codes)]
                        == base[np.setdiff1d(np.arange(2048), codes)]).all())

    def test_operands_that_do_not_line_up_are_answered_off_it(self):
        payload = self._device(self._canvas((3, 4)))
        with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
            payload.index_fill(5, self._codes([0]), 1.0)
        with self.assertRaisesRegex(RuntimeError, "vector or a single code"):
            payload.index_fill(0, self._codes(np.arange(6).reshape(2, 3)), 1.0)
        with self.assertRaisesRegex(RuntimeError, "Int64 or Int32"):
            payload.index_fill(0, self._codes([0.0, 1.0], np.float32), 1.0)
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            self._device(np.zeros((3, 4), np.int32)).index_fill(
                0, self._codes([0]), 1.0)


class TestVulkanScatterAdd(unittest.TestCase):
    # An accumulating scatter hands every entry of a list to the element it names
    # and grows that element by the entry's value.  Several entries can name one
    # element, so nothing may be dropped on the way: the request records which
    # element each entry names and then walks the destination, adding the entries
    # that reach a word in the order the list states them.  One invocation owns
    # one word, so the total of a repeated element does not depend on who won a
    # race, and the same call answers the same number twice over.

    SHAPES = ((5,), (5, 4), (2, 5, 3), (2, 3, 5, 2))

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @staticmethod
    def _codes(values, dtype=np.int64):
        return TestVulkanScatterAdd._device(np.asarray(values, dtype=dtype))

    @staticmethod
    def _canvas(shape, dtype=np.float32, seed=0):
        # Small whole numbers keep either element type exact, so an entry added
        # to the wrong element shows up as a wrong number rather than as
        # rounding.
        rng = np.random.default_rng([len(shape), seed, *shape])
        return rng.integers(-3, 4, size=shape).astype(dtype)

    @classmethod
    def _spread(cls, src, list_shape, dtype):
        # The axes the list carries that the payload does not spell out stand at
        # the front of it, each with one value to reach the entries below.
        held = np.asarray(src, dtype=dtype)
        lead = len(list_shape) - held.ndim
        return np.broadcast_to(
            held.reshape([1] * lead + list(held.shape)), list_shape)

    @classmethod
    def _want(cls, base, dim, codes, src):
        axis = dim if dim >= 0 else dim + base.ndim
        rows = base.shape[axis]
        codes = np.asarray(codes)
        # A half total is kept at full width and rounded on the way out.
        acc = base.astype(
            np.float32 if base.dtype == np.float16 else base.dtype)
        values = cls._spread(src, codes.shape, base.dtype)
        for flat in range(codes.size):
            code = int(codes.flat[flat])
            if code < 0 or code >= rows:
                continue
            coord = list(np.unravel_index(flat, codes.shape))
            slot = list(coord)
            slot[axis] = code
            acc[tuple(slot)] = acc[tuple(slot)] + values[tuple(coord)]
        return acc.astype(base.dtype)

    def _scattered(self, shape, dim, codes, src, dtype=np.float32, label=""):
        base = self._canvas(shape, dtype)
        list_shape = np.asarray(codes).shape
        values = np.asarray(src, dtype=dtype)
        if values.shape != list_shape:
            values = np.broadcast_to(values, list_shape).copy()
        name = (f"{label} shape={shape} dim={dim} codes_shape={list_shape} "
                f"dtype={np.dtype(dtype).name}")
        got = self._device(base).scatter_add_(
            dim, self._codes(codes), self._device(values)).cpu().numpy()
        np.testing.assert_array_equal(
            got, self._want(base, dim, codes, src), err_msg=name)
        return base, got

    def test_every_element_a_list_names_holds_its_values_added_on(self):
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                axis = dim if dim >= 0 else dim + len(shape)
                codes = (np.arange(int(np.prod(shape)))
                         % shape[axis]).reshape(shape)
                for dtype in (np.float32, np.float16):
                    self._scattered(shape, dim, codes,
                                    np.ones(shape, dtype), dtype)

    def test_a_table_that_names_each_element_once_moves_the_values_it_carries(
            self):
        base = self._canvas((6, 5), np.float32, seed=1)
        codes = np.tile(np.arange(6).reshape(6, 1), (1, 5))
        src = np.arange(30, dtype=np.float32).reshape(6, 5)
        got = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, codes, src))

    def test_an_element_several_entries_name_holds_the_sum_they_all_give(self):
        # A rewrite would let the last entry to reach an element win; a total
        # keeps every one of them, each at the coordinates its own entry carries.
        base = np.zeros((4, 5), np.float32)
        codes = np.array([[1, 1, 1, 3, 0], [1, 2, 1, 0, 0]], np.int64)
        src = np.array([[1.0, 2.0, 4.0, 5.0, 7.0],
                        [8.0, 0.0, 16.0, 0.0, 0.0]], np.float32)
        got = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, codes, src))
        self.assertEqual(got[1, 0], 1.0 + 8.0)
        self.assertEqual(got[1, 2], 4.0 + 16.0)
        self.assertEqual(got[1, 1], 2.0)
        self.assertEqual(got[3, 3], 5.0)
        self.assertEqual(got[0, 4], 7.0)
        self.assertEqual(got[2, 1], 0.0)

    def test_the_entries_are_added_in_the_order_the_list_states_them(self):
        # A total whose terms cancel only comes out as it does here when the
        # middle term is kept before the last one is added: read in another
        # order, the wide terms would meet each other first and the small one
        # would be lost to the rounding.
        base = np.zeros((4,), np.float32)
        codes = np.array([1, 1, 1], np.int64)
        src = np.array([1e30, 1e23, -1e30], np.float32)
        got = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, codes, src))
        self.assertNotEqual(got[1], np.float32(1e23))

    def test_a_walk_of_the_list_on_the_host_answers_the_same_number(self):
        # Only where every code fits the rows it names: the host walk answers a
        # code beyond them with an error rather than by adding nothing.
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                base = self._canvas(shape, np.float32, seed=2)
                axis = dim if dim >= 0 else dim + len(shape)
                codes = (np.arange(base.size) % base.shape[axis]).reshape(shape)
                src = self._canvas(shape, np.float32, seed=5)
                host = tensorplay.scatter_add(
                    tensorplay.from_numpy(base.copy()), dim,
                    tensorplay.from_numpy(codes),
                    tensorplay.from_numpy(src)).cpu().numpy()
                np.testing.assert_array_equal(
                    host, self._want(base, dim, codes, src),
                    err_msg=f"host: shape={shape} dim={dim}")

    def test_a_code_outside_the_rows_it_names_adds_nothing(self):
        # The last row is never named here, so it stands as it did on a canvas
        # nothing reached.
        base = np.zeros((5, 4), np.float32)
        codes = np.array([[9, -1, 0, 0], [-9, 7, 1, 1], [2, 2, -3, 0],
                          [0, 1, 1, 2], [3, 3, 3, -4]], np.int64)
        src = np.arange(20, dtype=np.float32).reshape(5, 4)
        got = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, codes, src))
        self.assertTrue((got[4] == 0.0).all())

    def test_a_code_given_from_the_end_names_no_element(self):
        # The list of a scatter counts positions, not rows from the end, so a
        # code below zero names nothing - which is what the rewrites above answer
        # the same list with.  The second list differs only in that its codes that
        # name nothing are given from the other end.
        base = np.zeros((4, 3), np.float32)
        src = np.arange(12, dtype=np.float32).reshape(4, 3)
        from_the_end = np.array([[-1, 0, 1], [2, -2, 3], [0, 1, -4], [3, 2, 1]],
                                np.int64)
        beyond = np.array([[4, 0, 1], [2, 5, 3], [0, 1, 7], [3, 2, 1]], np.int64)
        first = self._device(base).scatter_add_(
            0, self._codes(from_the_end), self._device(src)).cpu().numpy()
        second = self._device(base).scatter_add_(
            0, self._codes(beyond), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first, self._want(base, 0, from_the_end, src))

    def test_a_payload_that_reaches_the_list_through_fewer_axes_is_read_again(
            self):
        for shape, dim, src_shape in (((2, 5, 3), 1, (5, 3)),
                                      ((2, 3, 5, 2), 2, (5, 2)),
                                      ((5, 4), 0, (4,))):
            codes = np.zeros(shape, np.int64)
            self._scattered(shape, dim, codes,
                            np.arange(np.prod(src_shape),
                                      dtype=np.float32).reshape(src_shape))

    def test_one_number_reaching_the_whole_list_is_added_to_every_element_it_names(
            self):
        self._scattered((3, 4), 1, np.zeros((3, 4), np.int64),
                        np.array([2.5], np.float32))
        self._scattered((3, 4), 1, np.zeros((2, 4), np.int64),
                        np.array([-1.5], np.float32))

    def test_a_word_only_partly_named_gives_its_other_lanes_back_as_they_were(
            self):
        # A row that does not fill its last word shares it with the elements
        # below, so an entry that names one lane must hand the rest of the word
        # back untouched.  One entry names an element; every other code is given
        # from an end that names nothing.
        for shape, dim in (((4, 5), 0), ((13,), 0), ((3, 5, 7), 1),
                           ((5, 3), 1), ((2, 3, 5), 0)):
            for dtype in (np.float32, np.float16):
                axis = dim if dim >= 0 else dim + len(shape)
                codes = np.full(shape, -1, np.int64)
                codes.flat[0] = 1 % shape[axis]
                self._scattered(shape, dim, codes,
                                np.full(shape, 3.0, dtype), dtype)

    def test_a_half_canvas_rounds_its_total_once(self):
        # Every entry of the list names the first element of its row, so each row
        # answers one total of the six values it was handed - kept at full width
        # across the additions and rounded on the way out.
        base = np.zeros((4, 6), np.float16)
        codes = np.zeros((4, 6), np.int64)
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], np.float16)
        payload = np.broadcast_to(values, (4, 6)).copy()
        got = self._device(base).scatter_add_(
            1, self._codes(codes), self._device(payload)).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 1, codes, payload))
        self.assertEqual(got.dtype, np.float16)
        total = np.float16(np.float32(values).sum(dtype=np.float32))
        self.assertTrue((got[:, 0] == total).all(), msg=f"{got[:, 0]} != {total}")
        self.assertTrue((got[:, 1:] == 0.0).all())

    def test_a_list_of_either_width_names_the_same_elements(self):
        base = self._canvas((6, 4))
        codes = (np.arange(24) % 6).reshape(6, 4)
        src = np.ones((6, 4), np.float32)
        for dtype in (np.int32, np.int64):
            got = self._device(base).scatter_add_(
                0, self._codes(codes, dtype), self._device(src)).cpu().numpy()
            np.testing.assert_array_equal(
                got, self._want(base, 0, codes, src),
                err_msg=np.dtype(dtype).name)

    def test_a_request_with_nothing_in_it_changes_nothing(self):
        base = self._canvas((5, 4))
        kept = self._device(base)
        got = kept.scatter_add_(
            0, self._codes(np.zeros((2, 0), np.int64)),
            self._device(np.zeros((2, 0), np.float32))).cpu().numpy()
        np.testing.assert_array_equal(got, base)
        empty = self._device(np.zeros((0, 3), np.float32))
        answer = empty.scatter_add_(
            0, self._codes(np.zeros((0, 3), np.int64)),
            self._device(np.zeros((0, 3), np.float32)))
        self.assertEqual(tuple(answer.cpu().numpy().shape), (0, 3))

    def test_the_out_of_place_answer_leaves_the_canvas_alone(self):
        base = self._canvas((6, 5), np.float32, seed=4)
        codes = (np.arange(30) % 6).reshape(6, 5)
        src = np.ones((6, 5), np.float32)
        kept = self._device(base)
        moved = tensorplay.scatter_add(kept, 0, self._codes(codes),
                                       self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(moved, self._want(base, 0, codes, src))
        np.testing.assert_array_equal(kept.cpu().numpy(), base)

    def test_two_calls_answer_the_same_number(self):
        base = self._canvas((8, 7), np.float32, seed=6)
        rng = np.random.default_rng(11)
        codes = rng.integers(0, 8, size=(8, 7))
        src = rng.integers(-2, 3, size=(8, 7)).astype(np.float32)
        first = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        second = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, self._want(base, 0, codes, src))

    def test_a_routing_shaped_list_totals_a_decoder_canvas(self):
        # Four experts handing their rows to one canvas, as a mixture of experts
        # routes them: every token receives the rows its own list names.
        tokens, hidden, routed = 512, 256, 4
        rng = np.random.default_rng(13)
        base = np.zeros((tokens, hidden), np.float16)
        codes = rng.integers(0, tokens, size=(tokens, routed))
        src = rng.integers(-2, 3, size=(tokens, routed)).astype(np.float16)
        got = self._device(base).scatter_add_(
            0, self._codes(codes), self._device(src)).cpu().numpy()
        np.testing.assert_array_equal(got, self._want(base, 0, codes, src))

    def test_a_walk_the_destination_and_the_list_both_make_long_is_answered_off_it(
            self):
        length = 4000000
        rng = np.random.default_rng(17)
        # A list just inside the budget answers instead of refusing.
        held = self._device(np.zeros((length,), np.float32))
        held.scatter_add_(0, self._codes(rng.integers(0, 7, size=(60000,))),
                          self._device(np.ones((60000,), np.float32)))
        self.assertEqual(held.cpu().numpy().sum(), 60000.0)
        canvas = self._device(np.zeros((length,), np.float32))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            canvas.scatter_add_(0, self._codes(rng.integers(0, 7, size=(200000,))),
                                self._device(np.ones((200000,), np.float32)))

    def test_operands_that_do_not_line_up_are_answered_off_it(self):
        payload = self._device(self._canvas((5, 4)))
        ones = np.ones((5, 4), np.float32)
        with self.assertRaisesRegex(RuntimeError, "same rank"):
            payload.scatter_add_(0, self._codes(np.zeros((5,), np.int64)),
                                 self._device(np.ones((5,), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "wider than the destination"):
            payload.scatter_add_(1, self._codes(np.zeros((6, 4), np.int64)),
                                 self._device(np.zeros((6, 4), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
            payload.scatter_add_(2, self._codes(np.zeros((5, 4), np.int64)),
                                 self._device(ones))
        with self.assertRaisesRegex(RuntimeError, "Int32 or Int64"):
            payload.scatter_add_(0, self._codes([[0.0] * 4] * 5, np.float32),
                                 self._device(ones))
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            self._device(np.zeros((5, 4), np.int32)).scatter_add_(
                0, self._codes(np.zeros((5, 4), np.int64)),
                self._device(np.zeros((5, 4), np.int32)))
        with self.assertRaisesRegex(RuntimeError, "more axes than the list"):
            payload.scatter_add_(0, self._codes(np.zeros((5, 4), np.int64)),
                                 self._device(np.zeros((2, 5, 4), np.float32)))
        with self.assertRaisesRegex(RuntimeError, "must reach the list"):
            payload.scatter_add_(0, self._codes(np.zeros((5, 4), np.int64)),
                                 self._device(np.zeros((3, 4), np.float32)))


class TestVulkanMatvec(unittest.TestCase):
    """A matrix against a vector: one reduction over the shared axis."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _pair(m, k, seed, dtype):
        rng = np.random.default_rng(seed)
        mat = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((k)) * 0.5).astype(np.float32)
        if dtype is np.float16:
            mat = mat.astype(np.float16)
            vec = vec.astype(np.float16)
        return mat, vec

    def _answered(self, mat, vec, label, rtol=None, atol=None):
        """Reduce on the device and compare with the same product in full
        precision, which is what the walk answers up to its own summation."""
        dtype = np.asarray(mat).dtype
        got = tensorplay.mv(self._device(mat), self._device(vec)).cpu().numpy()
        want = np.matmul(mat.astype(np.float32).astype(np.float64),
                         vec.astype(np.float32).astype(np.float64))
        if rtol is None:
            # A half payload holds about ten bits of mantissa, and the device
            # keeps its running total at full width, so the gap to the model is
            # the input rounding plus the split reduction's summation order.
            rtol, atol = (6e-3, 6e-3) if dtype == np.float16 else (1e-4, 1e-4)
        np.testing.assert_allclose(
            got.astype(np.float64), want, rtol=rtol, atol=atol,
            err_msg=f"matrix against vector at {label}")
        want_dtype = np.float16 if dtype == np.float16 else np.float32
        self.assertEqual(got.dtype, want_dtype)
        return got

    def test_every_row_of_the_matrix_is_answered(self):
        for m, k, seed in [(5, 4, 1), (16, 16, 2), (1, 7, 3), (40, 40, 4),
                           (3, 101, 5), (257, 64, 6), (700, 3, 7)]:
            for dtype in (np.float32, np.float16):
                mat, vec = self._pair(m, k, seed, dtype)
                self._answered(mat, vec, f"{m}x{k} {dtype.__name__}")

    def test_a_row_reduced_to_a_number_keeps_the_operand_type(self):
        mat, vec = self._pair(9, 12, 8, np.float16)
        got = tensorplay.mv(self._device(mat), self._device(vec))
        self.assertEqual(got.shape, (9,))
        self.assertTrue(got.device.is_vulkan())
        self.assertEqual(got.dtype, tensorplay.float16)

    def test_a_long_reduction_streams_from_a_weight_with_no_texture_form(
            self):
        # A square this wide passes the device's image limit, so the reduction
        # reads the layer straight out of its linear payload.
        for k, seed in [(1024, 9), (2048, 10), (4096, 11)]:
            for dtype in (np.float32, np.float16):
                mat, vec = self._pair(k, k, seed, dtype)
                self._answered(mat, vec, f"{k}x{k} {dtype.__name__}")

    def test_a_single_row_projection_answers_what_the_layer_answers(self):
        import tensorplay.functional as F

        for k, seed in [(64, 12), (512, 13), (4096, 14)]:
            for dtype in (np.float32, np.float16):
                mat, vec = self._pair(k, k, seed, dtype)
                by_vector = tensorplay.mv(
                    self._device(mat), self._device(vec)).cpu().numpy()
                by_layer = F.linear(
                    self._device(vec), self._device(mat)).cpu().numpy()
                # Both walk the same layer with the same single row, so the
                # split reduction lands on the same answer both times.
                np.testing.assert_array_equal(by_vector, by_layer)
                self._answered(mat, vec, f"layer at {k} {dtype.__name__}")

    def test_a_matrix_held_as_its_own_transpose_is_read_without_a_copy(self):
        # The dense twin is (k, rows): the matrix the caller names is the view,
        # and the packing gather folds the swap on the way into the reduction.
        rng = np.random.default_rng(15)
        dense = (rng.standard_normal((6, 9)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((6)) * 0.5).astype(np.float32)
        held = self._device(dense)
        got = held.transpose(0, 1).mv(self._device(vec)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(dense.T.astype(np.float64), vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="transposed view as the matrix")

        big = (rng.standard_normal((4096, 2048)) * 0.4).astype(np.float32)
        long_vec = (rng.standard_normal((4096)) * 0.4).astype(np.float32)
        got = self._device(big).transpose(0, 1).mv(
            self._device(long_vec)).cpu().numpy()
        want = np.matmul(big.T.astype(np.float64), long_vec.astype(np.float64))
        np.testing.assert_allclose(got.astype(np.float64), want, rtol=1e-4,
                                   atol=1e-3,
                                   err_msg="transposed view past the image limit")

    def test_a_stride_pattern_the_gather_cannot_read_answers_the_same(self):
        rng = np.random.default_rng(16)
        wide = (rng.standard_normal((8, 20)) * 0.5).astype(np.float32)
        stepped = np.ascontiguousarray(wide[:, ::2])
        vec = (rng.standard_normal((10)) * 0.5).astype(np.float32)
        got = self._device(wide)[:, ::2].mv(self._device(vec)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(stepped.astype(np.float64), vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="matrix held a column at a time")

        dense = (rng.standard_normal((4, 6)) * 0.5).astype(np.float32)
        tail = dense[1:3]
        row_vec = (rng.standard_normal((6)) * 0.5).astype(np.float32)
        got = self._device(dense)[1:3].mv(self._device(row_vec)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(tail.astype(np.float64), row_vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="matrix past a storage offset")

    def test_a_reduction_over_no_entries_answers_zero(self):
        for dtype in (np.float32, np.float16):
            got = tensorplay.mv(self._device(np.zeros((4, 0), dtype)),
                                self._device(np.zeros((0), dtype))).cpu().numpy()
            self.assertEqual(got.shape, (4,))
            np.testing.assert_array_equal(got, np.zeros((4), dtype))
        # A matrix with no rows has nothing to answer for.
        got = tensorplay.mv(self._device(np.zeros((0, 3), np.float32)),
                            self._device(np.zeros((3), np.float32))).cpu().numpy()
        self.assertEqual(got.shape, (0,))

    def test_a_product_that_names_a_vector_operand_keeps_the_far_axis(self):
        rng = np.random.default_rng(17)
        mat = (rng.standard_normal((9, 5)) * 0.5).astype(np.float32)
        other = (rng.standard_normal((5, 7)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((5)) * 0.5).astype(np.float32)

        by_product = tensorplay.matmul(self._device(mat),
                                       self._device(vec)).cpu().numpy()
        np.testing.assert_array_equal(
            by_product,
            tensorplay.mv(self._device(mat), self._device(vec)).cpu().numpy())

        row = tensorplay.matmul(self._device(vec), self._device(other)).cpu().numpy()
        self.assertEqual(row.shape, (7,))
        np.testing.assert_allclose(
            row.astype(np.float64),
            np.matmul(vec.astype(np.float64), other.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="vector against a matrix")

        half_row = tensorplay.matmul(
            self._device(vec.astype(np.float16)),
            self._device(other.astype(np.float16))).cpu().numpy()
        np.testing.assert_allclose(
            half_row.astype(np.float64),
            np.matmul(vec.astype(np.float64), other.astype(np.float64)),
            rtol=6e-3, atol=6e-3, err_msg="half vector against a matrix")
        self.assertEqual(half_row.dtype, np.float16)

        # With no entries to reduce, a shared row answers zero across the far
        # axis.
        empty = tensorplay.matmul(self._device(np.zeros((0), np.float32)),
                                  self._device(np.zeros((0, 4), np.float32))
                                  ).cpu().numpy()
        np.testing.assert_array_equal(empty, np.zeros((4), np.float32))

    def test_a_batch_of_matrices_against_a_shared_vector_folds_the_batch(
            self):
        rng = np.random.default_rng(18)
        stacked = (rng.standard_normal((2, 3, 4)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((4)) * 0.5).astype(np.float32)
        got = tensorplay.matmul(self._device(stacked),
                                self._device(vec)).cpu().numpy()
        self.assertEqual(got.shape, (2, 3))
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(stacked.astype(np.float64), vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="batch against a vector")

        deeper = (rng.standard_normal((4, 2, 3, 5)) * 0.5).astype(np.float32)
        long_vec = (rng.standard_normal((5)) * 0.5).astype(np.float32)
        got = tensorplay.matmul(self._device(deeper),
                                self._device(long_vec)).cpu().numpy()
        self.assertEqual(got.shape, (4, 2, 3))
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(deeper.astype(np.float64), long_vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="deeper batch against a vector")

    def test_a_shared_vector_against_a_batch_of_matrices_drops_the_row_axis(
            self):
        rng = np.random.default_rng(19)
        vec = (rng.standard_normal((5)) * 0.5).astype(np.float32)
        planes = (rng.standard_normal((4, 5, 3)) * 0.5).astype(np.float32)
        got = tensorplay.matmul(self._device(vec),
                                self._device(planes)).cpu().numpy()
        self.assertEqual(got.shape, (4, 3))
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(vec.astype(np.float64), planes.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="a vector across a batch")

        # A lone plane of the batch answers the same as the matrix itself.
        tail = tensorplay.matmul(self._device(vec),
                                 self._device(planes[1:3])).cpu().numpy()
        self.assertEqual(tail.shape, (2, 3))
        np.testing.assert_allclose(
            tail.astype(np.float64),
            np.matmul(vec.astype(np.float64), planes[1:3].astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="a vector across a slice of planes")

    def test_a_weight_reduced_twice_answers_the_same(self):
        mat, vec = self._pair(640, 256, 20, np.float32)
        other = (np.random.default_rng(21).standard_normal((256))
                 * 0.5).astype(np.float32)
        held_mat = self._device(mat)
        first = tensorplay.mv(held_mat, self._device(vec)).cpu().numpy()
        # The packed operand for the weight stays cached across calls; a second
        # weight interleaved must not disturb it.
        tensorplay.mv(held_mat, self._device(other))
        again = tensorplay.mv(held_mat, self._device(vec)).cpu().numpy()
        np.testing.assert_array_equal(first, again)
        np.testing.assert_allclose(
            again.astype(np.float64),
            np.matmul(mat.astype(np.float64), vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="cached weight reduced again")

    def test_operands_that_do_not_line_up_are_answered_off_it(self):
        mat, vec = self._pair(4, 5, 22, np.float32)
        with self.assertRaisesRegex(RuntimeError, "expected a matrix"):
            tensorplay.mv(self._device(mat[0]), self._device(vec))
        with self.assertRaisesRegex(RuntimeError, "expected a matrix"):
            tensorplay.mv(self._device(mat), self._device(vec.reshape(5, 1)))
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            tensorplay.mv(self._device(mat), self._device(vec[:3]))
        with self.assertRaisesRegex(RuntimeError, "same element type"):
            tensorplay.mv(self._device(mat), self._device(
                vec.astype(np.float16)))
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.mv(self._device(np.zeros((4, 5), np.int32)),
                          self._device(np.zeros((5), np.int32)))
        with self.assertRaisesRegex(RuntimeError, "matrix dimensions do not match"):
            tensorplay.matmul(self._device(vec), self._device(mat))
        with self.assertRaisesRegex(RuntimeError, "same element type"):
            tensorplay.matmul(self._device(vec),
                              self._device(mat.astype(np.float16).reshape(5, 4)))


class TestVulkanVectorProducts(unittest.TestCase):
    """The products whose operands are vectors: reduce, rank one, contract."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _tolerance(dtype):
        return (6e-3, 6e-3) if dtype == np.float16 else (1e-4, 1e-4)

    def test_a_pair_of_vectors_reduces_to_one_number(self):
        rng = np.random.default_rng(31)
        for length, seed in [(4, 32), (64, 33), (1000, 34), (4096, 35)]:
            for dtype in (np.float32, np.float16):
                r = np.random.default_rng(seed)
                a = (r.standard_normal(length) * 0.5).astype(dtype)
                b = (r.standard_normal(length) * 0.5).astype(dtype)
                got = tensorplay.dot(self._device(a), self._device(b))
                self.assertEqual(got.shape, ())
                self.assertEqual(got.dtype, tensorplay.float16
                                 if dtype == np.float16 else tensorplay.float32)
                rtol, atol = self._tolerance(dtype)
                np.testing.assert_allclose(
                    np.asarray(got.cpu().numpy(), np.float64),
                    np.matmul(a.astype(np.float64), b.astype(np.float64)),
                    rtol=rtol, atol=atol,
                    err_msg=f"pair reduced at {length} {dtype.__name__}")

    def test_a_pair_with_no_entries_answers_zero(self):
        for dtype in (np.float32, np.float16):
            empty = self._device(np.zeros(0, dtype))
            got = tensorplay.dot(empty, empty).cpu().numpy()
            self.assertEqual(got.shape, ())
            self.assertEqual(float(got), 0.0)

    def test_a_column_and_a_row_of_different_types_answer_one_product(self):
        # Every entry is one pair of readings, so the operands need no shared
        # element type: the answer carries the type their product promotes to.
        rng = np.random.default_rng(36)
        col = (rng.standard_normal(6) * 0.5).astype(np.float32)
        row = (rng.standard_normal(5) * 0.5).astype(np.float32)
        for dtype_a, dtype_b in [
                (np.float32, np.float16),
                (np.float16, np.float32),
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float32, np.int32),
                (np.int32, np.float32),
                (np.int32, np.float16),
                (np.int32, np.int32),
                (np.bool_, np.float16),
                (np.bool_, np.int32)]:
            a, b = col.astype(dtype_a), row.astype(dtype_b)
            on_device = (self._device(a), self._device(b))
            got = tensorplay.outer(*on_device)
            self.assertEqual(
                got.dtype, tensorplay.result_type(*on_device),
                f"rank one typed by {dtype_a.__name__} against "
                f"{dtype_b.__name__}")
            want = np.outer(a.astype(np.float64), b.astype(np.float64))
            rtol, atol = self._tolerance(
                np.float16 if got.dtype == tensorplay.float16 else np.float32)
            on_host = tensorplay.outer(
                tensorplay.from_numpy(a), tensorplay.from_numpy(b))
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), want,
                rtol=rtol, atol=atol,
                err_msg=f"rank one {dtype_a.__name__}x{dtype_b.__name__}")
            self.assertEqual(on_host.dtype, got.dtype)
            np.testing.assert_allclose(
                on_host.cpu().numpy().astype(np.float64), want,
                rtol=rtol, atol=atol, err_msg="the host read a wider type")

        # A gradient of this product is a matrix-vector walk, which reads both
        # operands as one type, so a mixed pair answers the value alone.
        for device in ("cpu", "vulkan"):
            on_device = tensorplay.from_numpy(col).to(device)
            other = tensorplay.from_numpy(row.astype(np.float16)).to(device)
            on_device.requires_grad_(True)
            with self.assertRaisesRegex(
                    RuntimeError, "dtype",
                    msg=f"{device} gradient of a mixed rank one"):
                tensorplay.outer(on_device, other).sum().backward()

    def test_a_column_against_a_row_answers_every_pair(self):
        rng = np.random.default_rng(36)
        for m, n in [(5, 7), (1, 4), (4, 1), (3, 13)]:
            for dtype in (np.float32, np.float16):
                col = (rng.standard_normal(m) * 0.5).astype(dtype)
                row = (rng.standard_normal(n) * 0.5).astype(dtype)
                got = tensorplay.outer(self._device(col),
                                       self._device(row)).cpu().numpy()
                self.assertEqual(got.shape, (m, n))
                rtol, atol = self._tolerance(dtype)
                np.testing.assert_allclose(
                    got.astype(np.float64),
                    np.outer(col.astype(np.float64), row.astype(np.float64)),
                    rtol=rtol, atol=atol,
                    err_msg=f"rank one at {m}x{n} {dtype.__name__}")
        # An axis with no entries leaves no pair to answer.
        for m, n in [(0, 7), (7, 0)]:
            got = tensorplay.outer(self._device(np.zeros(m, np.float32)),
                                   self._device(np.zeros(n, np.float32))
                                   ).cpu().numpy()
            self.assertEqual(got.shape, (m, n))

    def test_a_contracted_axis_folds_the_axes_ahead_of_it(self):
        rng = np.random.default_rng(37)
        planes = (rng.standard_normal((2, 3, 4)) * 0.5).astype(np.float32)
        rows = (rng.standard_normal((5, 4)) * 0.5).astype(np.float32)
        vector = (rng.standard_normal((4)) * 0.5).astype(np.float32)

        got = tensorplay.inner(self._device(planes),
                               self._device(rows)).cpu().numpy()
        self.assertEqual(got.shape, (2, 3, 5))
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.tensordot(planes.astype(np.float64), rows.astype(np.float64),
                         axes=([2], [1])),
            rtol=1e-4, atol=1e-4, err_msg="three axes against two")

        got = tensorplay.inner(self._device(rows[0]),
                               self._device(rows)).cpu().numpy()
        self.assertEqual(got.shape, (5,))
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.tensordot(rows[0].astype(np.float64), rows.astype(np.float64),
                         axes=([0], [1])),
            rtol=1e-4, atol=1e-4, err_msg="a lone row across a batch")

        got = tensorplay.inner(self._device(rows),
                               self._device(rows)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.tensordot(rows.astype(np.float64), rows.astype(np.float64),
                         axes=([1], [1])),
            rtol=1e-4, atol=1e-4, err_msg="a batch contracted with itself")

        # A pair of vectors answers the number the reduction already answers,
        # and a pair of numbers simply multiplies.
        got = tensorplay.inner(self._device(vector),
                               self._device(rows[0])).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.dot(vector.astype(np.float64), rows[0].astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="pair of vectors")
        got = tensorplay.inner(self._device(np.float32(1.5)),
                               self._device(np.float32(2.5))).cpu().numpy()
        self.assertEqual(got.shape, ())
        self.assertAlmostEqual(float(got), 3.75, places=5)

        half_planes = np.asarray(planes, np.float16)
        half_rows = np.asarray(rows, np.float16)
        got = tensorplay.inner(self._device(half_planes),
                               self._device(half_rows)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float32).astype(np.float64),
            np.tensordot(half_planes.astype(np.float32),
                         half_rows.astype(np.float32), axes=([2], [1])),
            rtol=6e-3, atol=6e-3, err_msg="half axes contracted")

    def test_a_seeded_reduction_carries_the_addend_and_the_weights(self):
        rng = np.random.default_rng(38)
        mat = (rng.standard_normal((6, 4)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((4)) * 0.5).astype(np.float32)
        seed = (rng.standard_normal((6)) * 0.5).astype(np.float32)
        reduced = np.matmul(mat.astype(np.float64), vec.astype(np.float64))

        cases = [
            ("both weights at one", {}, seed + reduced),
            ("a doubled seed", {"beta": 2.0}, 2.0 * seed + reduced),
            ("a tripled reduction", {"alpha": 3.0}, seed + 3.0 * reduced),
            ("both weights", {"beta": 2.0, "alpha": 3.0},
             2.0 * seed + 3.0 * reduced),
            ("no reduction", {"alpha": 0.0}, seed),
            ("no seed", {"beta": 0.0}, reduced),
        ]
        for label, weights, want in cases:
            got = tensorplay.addmv(self._device(seed), self._device(mat),
                                   self._device(vec), **weights).cpu().numpy()
            np.testing.assert_allclose(
                got.astype(np.float64), want, rtol=1e-4, atol=1e-4,
                err_msg=f"seeded reduction with {label}")

        # One number seeds every row.
        got = tensorplay.addmv(self._device(np.float32(1.25)),
                               self._device(mat), self._device(vec),
                               beta=1.0, alpha=0.0).cpu().numpy()
        np.testing.assert_allclose(got, np.full(6, 1.25), rtol=0, atol=1e-6,
                                   err_msg="one number seeds every row")

        # A reduction over no entries leaves the seed standing alone.
        got = tensorplay.addmv(self._device(seed[:4]),
                               self._device(np.zeros((4, 0), np.float32)),
                               self._device(np.zeros(0, np.float32))
                               ).cpu().numpy()
        np.testing.assert_allclose(got.astype(np.float64), seed[:4],
                                   rtol=0, atol=1e-6,
                                   err_msg="reduction over no entries")

        half_mat, half_vec = np.asarray(mat, np.float16), np.asarray(vec, np.float16)
        half_seed = np.asarray(seed, np.float16)
        got = tensorplay.addmv(self._device(half_seed), self._device(half_mat),
                               self._device(half_vec)).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float32).astype(np.float64),
            half_seed.astype(np.float32) + np.matmul(
                half_mat.astype(np.float32), half_vec.astype(np.float32)),
            rtol=6e-3, atol=6e-3, err_msg="half seeded reduction")

    def test_an_indeterminate_addend_does_not_cross_a_zero_weight(self):
        rng = np.random.default_rng(39)
        mat = (rng.standard_normal((5, 4)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((4)) * 0.5).astype(np.float32)
        poisoned = (rng.standard_normal((5)) * 0.5).astype(np.float32)
        poisoned[2] = np.nan
        got = tensorplay.addmv(self._device(poisoned), self._device(mat),
                               self._device(vec), beta=0.0).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64),
            np.matmul(mat.astype(np.float64), vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4,
            err_msg="a weightless addend still carried across")

    def test_the_in_place_seed_keeps_its_own_rows(self):
        rng = np.random.default_rng(40)
        mat = (rng.standard_normal((7, 5)) * 0.5).astype(np.float32)
        vec = (rng.standard_normal((5)) * 0.5).astype(np.float32)
        seed = (rng.standard_normal((7)) * 0.5).astype(np.float32)
        held = self._device(seed)
        held.addmv_(self._device(mat), self._device(vec), beta=2.0, alpha=1.5)
        self.assertTrue(held.device.is_vulkan())
        np.testing.assert_allclose(
            held.cpu().numpy().astype(np.float64),
            2.0 * seed.astype(np.float64) + 1.5 * np.matmul(
                mat.astype(np.float64), vec.astype(np.float64)),
            rtol=1e-4, atol=1e-4, err_msg="in place seeded reduction")

    def test_operands_that_do_not_line_up_are_answered_off_it(self):
        f32 = np.float32
        with self.assertRaisesRegex(RuntimeError, "expected two vectors"):
            tensorplay.dot(self._device(np.zeros((2, 3), f32)),
                           self._device(np.zeros(3, f32)))
        with self.assertRaisesRegex(RuntimeError, "entries"):
            tensorplay.dot(self._device(np.zeros(3, f32)),
                           self._device(np.zeros(4, f32)))
        with self.assertRaisesRegex(RuntimeError, "same element type"):
            tensorplay.dot(self._device(np.zeros(3, f32)),
                           self._device(np.zeros(3, np.float16)))
        with self.assertRaisesRegex(RuntimeError, "expected two vectors"):
            tensorplay.outer(self._device(np.zeros((3, 1), f32)),
                             self._device(np.zeros(3, f32)))
        # A rank-one walk over a pair of counts is a product of the payload, so
        # it answers; the total of such a pair has no walk over a wide code and
        # the product path names the widths it reads.
        with self.assertRaisesRegex(
                RuntimeError, "Float32 and Float16"):
            tensorplay.dot(self._device(np.zeros(3, np.int64)),
                           self._device(np.zeros(3, np.int64)))
        with self.assertRaisesRegex(RuntimeError, "last axis"):
            tensorplay.inner(self._device(np.zeros((3, 4), f32)),
                             self._device(np.zeros((3, 5), f32)))
        with self.assertRaisesRegex(RuntimeError, "two axes"):
            tensorplay.addmv(self._device(np.zeros(3, f32)),
                             self._device(np.zeros((3, 3, 1), f32)),
                             self._device(np.zeros(3, f32)))
        with self.assertRaisesRegex(RuntimeError, "one axis"):
            tensorplay.addmv(self._device(np.zeros(3, f32)),
                             self._device(np.zeros((3, 3), f32)),
                             self._device(np.zeros((3, 1), f32)))
        with self.assertRaisesRegex(RuntimeError, "vector or a number"):
            tensorplay.addmv(self._device(np.zeros((3, 1), f32)),
                             self._device(np.zeros((3, 3), f32)),
                             self._device(np.zeros(3, f32)))
        with self.assertRaisesRegex(RuntimeError, "cannot reach"):
            tensorplay.addmv(self._device(np.zeros(2, f32)),
                             self._device(np.zeros((3, 3), f32)),
                             self._device(np.zeros(3, f32)))
        with self.assertRaisesRegex(RuntimeError, "reduction axis"):
            tensorplay.addmv(self._device(np.zeros(3, f32)),
                             self._device(np.zeros((3, 4), f32)),
                             self._device(np.zeros(3, f32)))
        with self.assertRaisesRegex(RuntimeError, "same element type"):
            tensorplay.addmv(self._device(np.zeros(3, f32)),
                             self._device(np.zeros((3, 4), f32)),
                             self._device(np.zeros(4, np.float16)))
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.addmv(self._device(np.zeros(3, np.int32)),
                             self._device(np.zeros((3, 4), np.int32)),
                             self._device(np.zeros(4, np.int32)))


class TestVulkanBatchSummedProduct(unittest.TestCase):
    """A stack of products folded onto one plane: the sum the batches answer
    before the addend weighs in, the shapes the addend may reach that plane
    from, and the corners where nothing contracts at all."""

    STACKS = ((1, 4, 5, 3), (3, 4, 5, 3), (5, 7, 3, 11), (2, 16, 16, 16),
              (8, 13, 9, 5))

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.asarray(values))

    @staticmethod
    def _stack(shape, seed):
        rng = np.random.default_rng(seed)
        batches, rows, reach, cols = shape
        return (
            (rng.standard_normal((batches, rows, reach)) * 0.5).astype(np.float32),
            (rng.standard_normal((batches, reach, cols)) * 0.5).astype(np.float32),
        )

    @staticmethod
    def _folded(lhs, rhs):
        # The cross-batch contraction the stack answers, held wide so the
        # tolerances below measure the walk rather than the arithmetic.
        return np.einsum(
            "bij,bjk->ik",
            np.asarray(lhs).astype(np.float64), np.asarray(rhs).astype(np.float64))

    def test_a_stack_folds_onto_the_plane(self):
        for index, shape in enumerate(self.STACKS):
            lhs, rhs = self._stack(shape, 120 + index)
            seed = (np.random.default_rng(130 + index).standard_normal(
                (shape[1], shape[3])) * 0.5).astype(np.float32)
            folded = self._folded(lhs, rhs)
            for label, weights, want in (
                    ("both weights at one", {}, seed + folded),
                    ("a doubled seed", {"beta": 2.0}, 2.0 * seed + folded),
                    ("a tripled contraction", {"alpha": 3.0}, seed + 3.0 * folded),
                    ("both weights", {"beta": -1.5, "alpha": 0.25},
                     -1.5 * seed + 0.25 * folded),
                    ("no contraction", {"alpha": 0.0}, seed),
                    ("no seed", {"beta": 0.0}, folded),
            ):
                with self.subTest(shape=shape, weights=label):
                    got = tensorplay.addbmm(
                        self._device(seed), self._device(lhs), self._device(rhs),
                        **weights).cpu().numpy()
                    self.assertEqual(tuple(got.shape), (shape[1], shape[3]))
                    np.testing.assert_allclose(
                        got.astype(np.float64), want, rtol=1e-4, atol=1e-4,
                        err_msg=f"stack folded with {label}")

    def test_the_addend_reaches_the_plane_its_own_way(self):
        lhs, rhs = self._stack((4, 3, 5, 6), 140)
        folded = self._folded(lhs, rhs)
        for label, seed in (
                ("the plane itself", np.full((3, 6), 0.5, np.float32)),
                ("one row of it", np.arange(6, dtype=np.float32)),
                ("one column of it", np.arange(3, dtype=np.float32)[:, None]),
                ("a corner of it", np.full((1, 1), 0.25, np.float32)),
                ("a single number", np.float32(1.5)),
        ):
            with self.subTest(addend=label):
                got = tensorplay.addbmm(
                    self._device(seed), self._device(lhs),
                    self._device(rhs)).cpu().numpy()
                np.testing.assert_allclose(
                    got.astype(np.float64), folded + np.broadcast_to(
                        np.asarray(seed).astype(np.float64), (3, 6)),
                    rtol=1e-4, atol=1e-4, err_msg=f"addend as {label}")

    def test_an_indeterminate_addend_does_not_cross_a_zero_weight(self):
        lhs, rhs = self._stack((3, 5, 4, 6), 150)
        poisoned = (np.random.default_rng(151).standard_normal(
            (5, 6)) * 0.5).astype(np.float32)
        poisoned[2, 1] = np.nan
        poisoned[4] = np.inf
        got = tensorplay.addbmm(self._device(poisoned), self._device(lhs),
                                self._device(rhs), beta=0.0).cpu().numpy()
        np.testing.assert_allclose(
            got.astype(np.float64), self._folded(lhs, rhs),
            rtol=1e-4, atol=1e-4,
            err_msg="a weightless addend still carried across")

    def test_a_walk_that_contracts_nothing_answers_the_seed_its_own_weight(self):
        # A stack with no batch, and a stack whose reduction axis holds no
        # entry, both leave the addend standing alone — scaled by its weight,
        # and nothing where the weight is nothing.
        plane = np.full((3, 5), 3.0, np.float32)
        for label, lhs, rhs in (
                ("no batch", np.zeros((0, 3, 4), np.float32),
                 np.zeros((0, 4, 5), np.float32)),
                ("no entry to contract", np.zeros((2, 3, 0), np.float32),
                 np.zeros((2, 0, 5), np.float32)),
        ):
            for beta, want in ((2.0, 6.0), (1.0, 3.0), (0.0, 0.0)):
                with self.subTest(walk=label, beta=beta):
                    got = tensorplay.addbmm(
                        self._device(plane), self._device(lhs),
                        self._device(rhs), beta=beta, alpha=1.0).cpu().numpy()
                    self.assertEqual(got.shape, (3, 5))
                    np.testing.assert_allclose(got, np.full((3, 5), want),
                                               rtol=0, atol=1e-6,
                                               err_msg=f"{label} at weight {beta}")
            with self.subTest(walk=label, kind="an indeterminate seed"):
                got = tensorplay.addbmm(
                    self._device(np.full((3, 5), np.nan, np.float32)),
                    self._device(lhs), self._device(rhs), beta=0.0,
                    alpha=1.0).cpu().numpy()
                np.testing.assert_allclose(got, np.zeros((3, 5)), rtol=0,
                                           atol=0, err_msg=f"{label} unweighted")
            with self.subTest(walk=label, kind="a row of a seed"):
                got = tensorplay.addbmm(
                    self._device(np.arange(5, dtype=np.float32)),
                    self._device(lhs), self._device(rhs), beta=2.0,
                    alpha=1.0).cpu().numpy()
                self.assertEqual(got.shape, (3, 5))
                np.testing.assert_allclose(
                    got, 2.0 * np.broadcast_to(np.arange(5, dtype=np.float32),
                                               (3, 5)), rtol=0, atol=1e-6,
                    err_msg=f"{label} seeded by one row")

    def test_the_device_walk_answers_the_host_walk(self):
        for index, shape in enumerate(self.STACKS[:3]):
            lhs, rhs = self._stack(shape, 160 + index)
            seed = (np.random.default_rng(170 + index).standard_normal(
                (shape[1], shape[3])) * 0.5).astype(np.float32)
            for weights in ({}, {"beta": 0.0}, {"beta": 2.0, "alpha": -0.5}):
                with self.subTest(shape=shape, weights=weights):
                    np.testing.assert_allclose(
                        tensorplay.addbmm(self._device(seed), self._device(lhs),
                                          self._device(rhs),
                                          **weights).cpu().numpy(),
                        tensorplay.addbmm(self._host(seed), self._host(lhs),
                                          self._host(rhs),
                                          **weights).cpu().numpy(),
                        rtol=1e-4, atol=1e-4)

    def test_a_narrow_stack_keeps_its_own_width(self):
        lhs, rhs = self._stack((4, 32, 48, 16), 180)
        narrow_lhs, narrow_rhs = np.asarray(lhs, np.float16), np.asarray(rhs, np.float16)
        narrow_seed = np.asarray(np.random.default_rng(181).standard_normal(
            (32, 16)) * 0.5, np.float16)
        answered = tensorplay.addbmm(self._device(narrow_seed),
                                     self._device(narrow_lhs),
                                     self._device(narrow_rhs))
        self.assertEqual(answered.dtype, tensorplay.float16)
        wide = self._folded(narrow_lhs, narrow_rhs) + narrow_seed.astype(np.float64)
        np.testing.assert_allclose(
            answered.cpu().numpy().astype(np.float32).astype(np.float64), wide,
            rtol=2e-2, atol=2e-2, err_msg="a narrow stack folded wide")

    def test_the_in_place_seed_keeps_its_own_plane(self):
        lhs, rhs = self._stack((3, 5, 4, 6), 190)
        seed = (np.random.default_rng(191).standard_normal((5, 6))
                * 0.5).astype(np.float32)
        held = self._device(seed)
        held.addbmm_(self._device(lhs), self._device(rhs), beta=2.0, alpha=1.5)
        self.assertTrue(held.device.is_vulkan())
        np.testing.assert_allclose(
            held.cpu().numpy().astype(np.float64),
            2.0 * seed.astype(np.float64) + 1.5 * self._folded(lhs, rhs),
            rtol=1e-4, atol=1e-4, err_msg="in place folded stack")

    def test_the_gradient_reaches_every_batch_of_the_stack(self):
        lhs, rhs = self._stack((3, 5, 4, 6), 200)
        seed = (np.random.default_rng(201).standard_normal((5, 6))
                * 0.5).astype(np.float32)
        held = self._device(lhs).requires_grad_(True)
        total = tensorplay.addbmm(self._device(seed), held, self._device(rhs),
                                  beta=1.0, alpha=2.0).sum()
        gradient = tensorplay.autograd.grad(total, held)[0]
        self.assertTrue(gradient.device.is_vulkan())
        want = np.stack([
            2.0 * np.ones((5, 6), np.float64) @ matrix.astype(np.float64).T
            for matrix in rhs])
        np.testing.assert_allclose(
            gradient.cpu().numpy().astype(np.float64), want,
            rtol=1e-4, atol=1e-4, err_msg="gradient spread over the stack")

    def test_a_walk_out_of_shape_or_kind_refuses(self):
        lhs, rhs = self._stack((2, 3, 4, 5), 210)
        plane = np.ones((3, 5), np.float32)
        f32 = np.float32
        with self.subTest(kind="a stack that is not three axes"):
            with self.assertRaisesRegex(RuntimeError, "batch1 must be 3d"):
                tensorplay.addbmm(self._device(plane),
                                  self._device(np.zeros((2, 3, 4, 1), f32)),
                                  self._device(rhs))
        with self.subTest(kind="a stack that is not three axes on the far side"):
            with self.assertRaisesRegex(RuntimeError, "batch2 must be 3d"):
                tensorplay.addbmm(self._device(plane), self._device(lhs),
                                  self._device(np.zeros((2, 4, 5, 1), f32)))
        with self.subTest(kind="stacks of different counts"):
            with self.assertRaisesRegex(
                    RuntimeError, "batch dimensions do not match"):
                tensorplay.addbmm(self._device(plane), self._device(lhs),
                                  self._device(np.zeros((3, 4, 5), f32)))
        with self.subTest(kind="matrices that do not meet"):
            with self.assertRaisesRegex(
                    RuntimeError, "matrix dimensions do not match"):
                tensorplay.addbmm(self._device(plane), self._device(lhs),
                                  self._device(np.zeros((2, 6, 5), f32)))
        with self.subTest(kind="an addend off the plane"):
            with self.assertRaisesRegex(RuntimeError, "cannot reach"):
                tensorplay.addbmm(self._device(np.ones((6, 5), f32)),
                                  self._device(lhs), self._device(rhs))
        with self.subTest(kind="an addend stacked above the plane"):
            with self.assertRaisesRegex(RuntimeError, "must sit on the plane"):
                tensorplay.addbmm(self._device(np.ones((2, 3, 5), f32)),
                                  self._device(lhs), self._device(rhs))
        with self.subTest(kind="a stack of counts"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                tensorplay.addbmm(
                    self._device(np.ones((3, 5), np.int64)),
                    self._device(np.ones((2, 3, 4), np.int64)),
                    self._device(np.ones((2, 4, 5), np.int64)))
        with self.subTest(kind="a stack read at two widths"):
            with self.assertRaisesRegex(RuntimeError, "same element type"):
                tensorplay.addbmm(self._device(plane), self._device(lhs),
                                  self._device(np.zeros((2, 4, 5), np.float16)))
        with self.subTest(kind="an addend read at another width"):
            with self.assertRaisesRegex(RuntimeError, "the addend must be read"):
                tensorplay.addbmm(self._device(np.ones((3, 5), np.float16)),
                                  self._device(lhs), self._device(rhs))


class TestVulkanTrace(unittest.TestCase):
    """The diagonal of a matrix folded into one lane: the width a count totals
    into, the wider fold a narrow measured lane is read through before it is
    rounded back, and the gradient that lands on the diagonal and nowhere
    else."""

    SHAPES = ((3, 3), (5, 7), (8, 5), (1, 6), (6, 1), (12, 12), (0, 4), (7, 0))

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.asarray(values))

    @staticmethod
    def _matrix(shape, seed, dtype):
        rng = np.random.default_rng(seed)
        if np.issubdtype(dtype, np.integer):
            return rng.integers(-30, 31, size=shape).astype(dtype)
        return (rng.standard_normal(shape) * 12.0).astype(dtype)

    @staticmethod
    def _truth(values):
        return values.diagonal().astype(np.float64).sum()

    def test_the_diagonal_folds_into_one_lane(self):
        for index, shape in enumerate(self.SHAPES):
            for dtype in (np.float32, np.float16):
                values = self._matrix(shape, 200 + index, dtype)
                payload = self._device(values)
                answer = tensorplay.trace(payload)
                with self.subTest(dtype=np.dtype(dtype).name, shape=shape):
                    self.assertEqual(tuple(answer.shape), ())
                    self.assertEqual(answer.dtype, payload.dtype)
                    got = float(answer.cpu().numpy())
                    if dtype is np.float32:
                        self.assertAlmostEqual(
                            got, float(self._truth(values)), delta=1e-3)
                    else:
                        # The total is held in a width that can carry the whole
                        # diagonal, so one rounding covers the entire walk.
                        self.assertEqual(
                            got, float(np.float16(self._truth(values))))
                    # The host walk folds the same lanes in its own order, so
                    # the two agree up to the last place the width can name.
                    np.testing.assert_allclose(
                        got, float(self._host(values).trace().cpu().numpy()),
                        rtol=2e-3 if dtype is np.float16 else 1e-5,
                        atol=2e-3 if dtype is np.float16 else 1e-4,
                        err_msg="the host folded another way")

    def test_a_count_totals_into_the_width_its_answer_names(self):
        for dtype in (np.int32, np.int64):
            for index, shape in enumerate(((4, 4), (6, 2), (3, 9), (0, 5))):
                values = self._matrix(shape, 40 + index, dtype)
                answer = tensorplay.trace(self._device(values))
                with self.subTest(dtype=np.dtype(dtype).name, shape=shape):
                    self.assertEqual(answer.dtype, tensorplay.int64)
                    self.assertEqual(
                        int(answer.cpu().numpy()),
                        int(values.diagonal().astype(np.int64).sum()))

    def test_a_walk_that_is_not_a_matrix_or_a_measured_width_refuses(self):
        for shape in ((6,), (2, 3, 4), (2, 3, 4, 5)):
            with self.subTest(kind=f"rank {len(shape)}"):
                with self.assertRaisesRegex(
                        RuntimeError, "expects a matrix"):
                    tensorplay.trace(self._device(
                        np.zeros(shape, np.float32)))
        with self.subTest(kind="a payload that counts yes/no"):
            with self.assertRaisesRegex(RuntimeError, "Float32, Float16"):
                tensorplay.trace(self._device(
                    np.eye(4, dtype=bool)))

    def test_the_gradient_lands_on_the_diagonal_alone(self):
        for shape in ((4, 4), (3, 6), (6, 3)):
            for dtype in (np.float32, np.float16):
                values = self._matrix(shape, 11, dtype)
                payload = self._device(values).requires_grad_(True)
                weight = np.asarray(3.5, dtype)
                tensorplay.trace(payload).backward(self._device(weight))
                want = np.eye(shape[0], shape[1], dtype=dtype) * weight
                with self.subTest(shape=shape, dtype=np.dtype(dtype).name):
                    np.testing.assert_allclose(
                        payload.grad.cpu().numpy(), want, rtol=0, atol=0,
                        err_msg="the gradient missed the diagonal")

    def test_a_matrix_read_through_a_view_folds_its_own_diagonal(self):
        base = (np.random.default_rng(9).standard_normal((6, 5)) * 12.0).astype(
            np.float32)
        payload = self._device(base)
        for label, matrix in (
                ("a transpose", payload.transpose(0, 1)),
                ("a strided column", payload[:, 1:4]),
                ("a narrow row", payload.narrow(0, 1, 4))):
            answer = tensorplay.trace(matrix)
            want = matrix.cpu().numpy().astype(np.float64)
            with self.subTest(kind=label):
                self.assertAlmostEqual(
                    float(answer.cpu().numpy()),
                    float(want.diagonal().sum()), delta=1e-4)

    def test_the_device_answers_what_the_host_answers_over_every_width(self):
        bounds = {
            np.float32: (1e-5, 1e-4),
            np.float16: (2e-3, 2e-3),
            np.int32: (0.0, 0.0),
            np.int64: (0.0, 0.0),
        }
        for index, shape in enumerate(self.SHAPES[:6]):
            for dtype in (np.float32, np.float16, np.int32, np.int64):
                values = self._matrix(shape, 700 + index, dtype)
                rtol, atol = bounds[dtype]
                with self.subTest(dtype=np.dtype(dtype).name, shape=shape):
                    np.testing.assert_allclose(
                        tensorplay.trace(self._device(values)).cpu().numpy(
                        ).astype(np.float64),
                        tensorplay.trace(self._host(values)).cpu().numpy(
                        ).astype(np.float64),
                        rtol=rtol, atol=atol,
                        err_msg=f"{np.dtype(dtype).name} matrix {shape}")


class TestVulkanRepeatExpansion(unittest.TestCase):
    """Repeating an axis in place: the count the key-value heads reach the
    query head count by, and the half-width rotary table doubles."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _block(shape, seed):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(shape) * 4).astype(np.float32)

    def _expanded(self, data, repeats, axis, label):
        got = self._device(data).repeat_interleave(repeats, dim=axis).cpu().numpy()
        want = np.repeat(data, repeats, axis=axis)
        # Nothing is combined here, only copied, so the device answers the
        # model entry for entry rather than within a tolerance.
        np.testing.assert_array_equal(got, want, err_msg=f"expansion at {label}")
        return got

    def test_every_axis_of_a_tall_payload_expands(self):
        for shape in [(4,), (3, 5), (2, 3, 5), (2, 3, 4, 5)]:
            data = self._block(shape, 7)
            for axis in range(len(shape)):
                self._expanded(data, 3, axis, f"{shape} axis {axis}")
            self._expanded(data, 2, -1, f"{shape} negative axis")

    def test_a_grouped_query_cache_reaches_the_query_head_count(self):
        kv = self._block((1, 2, 5, 64), 8)
        queries = self._block((1, 8, 3, 64), 9)
        expanded = self._device(kv).repeat_interleave(4, dim=1)
        np.testing.assert_array_equal(
            expanded.cpu().numpy(), np.repeat(kv, 4, axis=1))
        self.assertEqual(tuple(expanded.shape), (1, 8, 5, 64))
        got = (self._device(queries) @ expanded.transpose(2, 3)).cpu().numpy()
        want = queries.astype(np.float64) @ np.repeat(
            kv, 4, axis=1).astype(np.float64).transpose(0, 1, 3, 2)
        np.testing.assert_allclose(got.astype(np.float64), want, rtol=1e-4, atol=1e-4)

    def test_a_rotary_table_doubles_across_its_last_axis(self):
        table = self._block((1, 8, 3, 32), 10)
        self._expanded(table, 2, -1, "rotary doubling")

    def test_a_view_repeats_what_it_shows(self):
        wide = self._block((2, 6, 4, 5), 11)
        self._expanded(wide[:, ::2], 2, 1, "stride two")
        permuted = np.transpose(self._block((3, 2, 5, 4), 12), (0, 2, 1, 3))
        self._expanded(permuted, 2, 2, "permuted")
        self._expanded(self._block((3, 8, 4), 13)[:, 2:6], 3, 1, "storage offset")

    def test_one_and_zero_repeats_and_a_payload_with_no_entries(self):
        data = self._block((2, 3, 4, 5), 14)
        np.testing.assert_array_equal(
            self._device(data).repeat_interleave(1, dim=2).cpu().numpy(), data)
        for axis in (0, 2, 3):
            got = self._device(data).repeat_interleave(0, dim=axis).cpu().numpy()
            np.testing.assert_array_equal(got, np.repeat(data, 0, axis=axis))
            self.assertEqual(tuple(got.shape), np.repeat(data, 0, axis=axis).shape)
        self._expanded(self._block((0, 3, 4), 15), 2, 1, "empty payload")
        self._expanded(self._block((2, 3, 0, 5), 16), 2, 2, "empty repeated axis")

    def test_the_flat_form_matches_the_axis_form(self):
        data = self._block((2, 3, 4, 5), 17)
        flat = self._device(data).repeat_interleave(3).cpu().numpy()
        np.testing.assert_array_equal(flat, np.repeat(data.reshape(-1), 3))

    def test_a_half_payload_keeps_its_element_type(self):
        half = self._block((1, 2, 8, 64), 18).astype(np.float16)
        got = self._device(half).repeat_interleave(4, dim=1)
        np.testing.assert_array_equal(got.cpu().numpy(), np.repeat(half, 4, axis=1))
        self.assertEqual(got.dtype, tensorplay.float16)

    def test_a_count_payload_keeps_its_width(self):
        rng = np.random.default_rng(19)
        wide = rng.integers(-20, 20, size=(2, 3, 4, 5)).astype(np.int64)
        got = self._device(wide).repeat_interleave(3, dim=2)
        np.testing.assert_array_equal(got.cpu().numpy(), np.repeat(wide, 3, axis=2))
        self.assertEqual(got.dtype, tensorplay.int64)
        narrow = rng.integers(-20, 20, size=(1, 2, 6, 8)).astype(np.int32)
        got = self._device(narrow).repeat_interleave(4, dim=1)
        np.testing.assert_array_equal(got.cpu().numpy(), np.repeat(narrow, 4, axis=1))

    def test_a_cache_tall_past_an_image_expands_anyway(self):
        # The folded walk names a quarter of a million entries across, which no
        # texel axis can span, so the expansion has to answer from plain memory.
        kv = self._block((1, 2, 4096, 64), 20)
        self._expanded(kv, 4, 1, "long cache")

    def test_a_count_that_does_not_fit_refuses(self):
        data = self._block((2, 3, 4), 21)
        with self.assertRaisesRegex(RuntimeError, "output_size"):
            self._device(data).repeat_interleave(3, dim=1, output_size=6)
        with self.assertRaises(ValueError):
            self._device(data).repeat_interleave(-1, dim=1)
        with self.assertRaises(IndexError):
            self._device(data).repeat_interleave(2, dim=4)


class TestVulkanRepeatIndices(unittest.TestCase):
    """The source position of every slot in an expansion whose counts are
    themselves a payload.  The length of the answer is a property of the whole
    count tensor, so it is known only once the counts have been totalled; the
    fill then writes one slot per answer position and none of them twice."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values).copy()).to("vulkan")

    @staticmethod
    def _back(payload):
        return np.asarray(payload.cpu())

    def test_the_slots_of_a_count_payload(self):
        import tensorplay.functional as F

        counts = np.array([2, 3, 1, 4], dtype=np.int32)
        got = F.repeat_interleave(self._device(counts))
        self.assertEqual(got.dtype, tensorplay.int32)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(np.arange(4, dtype=np.int32), counts))
        self.assertEqual(tuple(got.shape), (counts.sum(),))

        for host in (np.array([1], dtype=np.int32),
                    np.array([5, 1, 0, 2, 7, 1], dtype=np.int32),
                    np.array([3, 3, 3], dtype=np.int32)):
            got = F.repeat_interleave(self._device(host))
            np.testing.assert_array_equal(
                self._back(got),
                np.repeat(np.arange(host.size, dtype=np.int32), host),
                err_msg=f"counts {host.tolist()}")

    def test_a_wide_count_answers_in_the_wide_code(self):
        import tensorplay.functional as F

        wide = np.array([1, 0, 3, 2], dtype=np.int64)
        got = F.repeat_interleave(self._device(wide))
        self.assertEqual(got.dtype, tensorplay.int64)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(np.arange(4, dtype=np.int64), wide))

        # A total that no single 32-bit slot count could carry on its own still
        # answers position by position.
        long = np.full(70000, 3, dtype=np.int64)
        got = F.repeat_interleave(self._device(long))
        self.assertEqual(got.dtype, tensorplay.int64)
        self.assertEqual(tuple(got.shape), (210000,))
        answer = self._back(got)
        np.testing.assert_array_equal(answer[:3], [0, 0, 0])
        np.testing.assert_array_equal(answer[-3:], [69999, 69999, 69999])
        self.assertEqual(int(answer.min()), 0)
        self.assertEqual(int(np.diff(answer).astype(np.int64).max()), 1)

    def test_a_zero_count_holds_no_slot(self):
        import tensorplay.functional as F

        for host in (np.array([0, 2, 0, 1, 0], dtype=np.int32),
                     np.zeros(4, dtype=np.int32),
                     np.zeros(0, dtype=np.int32),
                     np.zeros(0, dtype=np.int64)):
            got = F.repeat_interleave(self._device(host))
            want = np.repeat(np.arange(host.size, dtype=host.dtype), host)
            np.testing.assert_array_equal(self._back(got), want)
            self.assertEqual(got.dtype, self._device(host).dtype)

    def test_a_long_run_of_counts(self):
        import tensorplay.functional as F

        counts = (np.arange(2000, dtype=np.int32) % 5) + 1
        got = F.repeat_interleave(self._device(counts))
        np.testing.assert_array_equal(
            self._back(got), np.repeat(np.arange(2000, dtype=np.int32), counts))

        # The fill walks segments, so a run of counts with one segment each would
        # leave most of the answer unwritten; a few wide segments must cover it.
        few = np.array([50000, 1, 40000], dtype=np.int32)
        answer = self._back(F.repeat_interleave(self._device(few)))
        np.testing.assert_array_equal(
            answer, np.repeat(np.arange(3, dtype=np.int32), few))

        # A run of more segments than the bounded grid covers has to walk the
        # rest with the whole grid as its stride, so no span may be skipped or
        # written by two segments at once.
        many = np.full(20000, 20, dtype=np.int32)
        answer = self._back(F.repeat_interleave(self._device(many)))
        self.assertEqual(tuple(answer.shape), (400000,))
        np.testing.assert_array_equal(
            answer, np.repeat(np.arange(20000, dtype=np.int32), many))

    def test_the_expansion_of_a_payload_follows_device_counts(self):
        import tensorplay.functional as F

        grid = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        for axis, counts in ((0, [2, 1]), (1, [1, 3, 2]), (2, [2, 2, 1, 1])):
            got = F.repeat_interleave(
                self._device(grid), self._device(np.array(counts, np.int32)),
                axis)
            np.testing.assert_array_equal(
                self._back(got), np.repeat(grid, counts, axis=axis),
                err_msg=f"axis {axis}")
            self.assertEqual(got.dtype, tensorplay.float32)
        got = F.repeat_interleave(
            self._device(grid), self._device(np.array([1, 2, 3, 4], np.int32)),
            -1)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(grid, [1, 2, 3, 4], axis=-1))

        half = grid.astype(np.float16)
        counts = np.array([0, 2, 1], dtype=np.int32)
        got = F.repeat_interleave(self._device(half), self._device(counts), 1)
        self.assertEqual(got.dtype, tensorplay.float16)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(half, counts, axis=1))

    def test_one_count_spread_over_a_block(self):
        import tensorplay.functional as F

        rows = np.arange(8, dtype=np.float32).reshape(4, 2)
        for count in (3, 1, 0):
            counts = np.array([count], dtype=np.int32)
            got = F.repeat_interleave(
                self._device(rows), self._device(counts), 0)
            np.testing.assert_array_equal(
                self._back(got), np.repeat(rows, count, axis=0),
                err_msg=f"count {count}")

        empty = np.zeros((0, 3), dtype=np.float32)
        got = F.repeat_interleave(
            self._device(empty), self._device(np.zeros(0, np.int32)), 0)
        self.assertEqual(tuple(got.shape), (0, 3))

    def test_a_stated_length_matched_and_refused(self):
        import tensorplay.functional as F

        counts = np.array([2, 3, 1, 4], dtype=np.int32)
        got = F.repeat_interleave(self._device(counts), output_size=10)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(np.arange(4, dtype=np.int32), counts))
        self.assertEqual(tuple(F.repeat_interleave(
            self._device(np.zeros(4, np.int32)), output_size=0).shape), (0,))
        with self.assertRaisesRegex(RuntimeError, "answer length"):
            F.repeat_interleave(self._device(counts), output_size=9)
        with self.assertRaisesRegex(RuntimeError, "answer length"):
            F.repeat_interleave(self._device(counts), output_size=-1)

    def test_a_count_payload_that_is_a_view(self):
        import tensorplay.functional as F

        source = (np.arange(36, dtype=np.int32) % 4)
        strided = source[2:20:3]
        got = F.repeat_interleave(self._device(strided))
        np.testing.assert_array_equal(
            self._back(got),
            np.repeat(np.arange(strided.size, dtype=np.int32), strided))

        wide_rows = np.array([[1, 2, 3, 4]], dtype=np.int64)
        counts = wide_rows[0, 1:4]
        got = F.repeat_interleave(self._device(counts))
        self.assertEqual(got.dtype, tensorplay.int64)
        np.testing.assert_array_equal(
            self._back(got),
            np.repeat(np.arange(counts.size, dtype=np.int64), counts))

        block = self._device(np.arange(12, dtype=np.int32).reshape(3, 4))
        sliced = block[:, 3:4]
        counts = sliced.reshape(-1)
        got = F.repeat_interleave(counts)
        np.testing.assert_array_equal(
            self._back(got),
            np.repeat(np.arange(3, dtype=np.int32), [3, 7, 11]))

    def test_the_positions_drive_a_row_gather(self):
        import tensorplay.functional as F

        rng = np.random.default_rng(23)
        table = rng.standard_normal((64, 32)).astype(np.float32)
        counts = (rng.integers(0, 4, size=64)).astype(np.int32)
        indices = F.repeat_interleave(self._device(counts))
        got = F.index_select(self._device(table), 0, indices)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(table, counts, axis=0))
        self.assertEqual(tuple(got.shape), (int(counts.sum()), 32))

    def test_counts_built_on_the_device(self):
        import tensorplay.functional as F

        counts = tensorplay.ones(6, dtype=tensorplay.int32, device="vulkan") * 2
        got = F.repeat_interleave(counts)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(np.arange(6, dtype=np.int32), 2))
        self.assertEqual(got.device.type, "vulkan")

        # A count payload may be the answer of another walk rather than a host
        # upload, so the totals read back have to come from device memory.
        wide = (tensorplay.ones(4, dtype=tensorplay.int32, device="vulkan") * 3
                ).to(tensorplay.int64)
        got = F.repeat_interleave(wide)
        self.assertEqual(got.dtype, tensorplay.int64)
        np.testing.assert_array_equal(
            self._back(got), np.repeat(np.arange(4, dtype=np.int64), 3))

    def test_a_count_vocabulary_outside_the_surface_refuses(self):
        import tensorplay.functional as F

        with self.assertRaisesRegex(RuntimeError, "Int32 or Int64"):
            F.repeat_interleave(self._device(np.ones(3, dtype=bool)))
        with self.assertRaisesRegex(RuntimeError, "Int32 or Int64"):
            F.repeat_interleave(self._device(np.ones(3, dtype=np.uint8)))
        with self.assertRaisesRegex(RuntimeError, "Int32 or Int64"):
            F.repeat_interleave(self._device(np.array([1.0, 2.0], np.float32)))
        with self.assertRaisesRegex(RuntimeError, "1d count payload"):
            F.repeat_interleave(self._device(np.ones((2, 2), dtype=np.int32)))
        with self.assertRaisesRegex(RuntimeError, "negative"):
            F.repeat_interleave(self._device(np.array([2, -1, 3], np.int32)))
        with self.assertRaisesRegex(RuntimeError, "negative"):
            F.repeat_interleave(self._device(np.array([2, 0, -5, 9], np.int64)))


class TestVulkanTake(unittest.TestCase):
    """A flat lookup: every code names one entry of the payload's row-major
    order, and the answer has the list's shape."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    def test_every_position_of_the_payload_is_reachable(self):
        rng = np.random.default_rng(31)
        payload = (rng.standard_normal((5, 8)) * 10).astype(np.float32)
        flat = payload.reshape(-1)
        codes = np.array([0, 7, 39, -1, -40], dtype=np.int64)
        got = self._device(payload).take(self._device(codes)).cpu().numpy()
        np.testing.assert_array_equal(got, flat[codes])

    def test_the_answer_keeps_the_list_shape(self):
        rng = np.random.default_rng(32)
        payload = (rng.standard_normal((4, 6)) * 10).astype(np.float32)
        codes = rng.integers(0, 24, (3, 2, 4))
        got = tensorplay.take(self._device(payload), self._device(codes))
        self.assertEqual(tuple(got.shape), (3, 2, 4))
        np.testing.assert_array_equal(got.cpu().numpy(), payload.reshape(-1)[codes])

    def test_a_narrow_list_and_a_wide_one_answer_alike(self):
        rng = np.random.default_rng(33)
        payload = (rng.standard_normal((7, 9)) * 10).astype(np.float32)
        codes = rng.integers(0, 63, 40)
        wide = self._device(codes.astype(np.int64))
        narrow = self._device(codes.astype(np.int32))
        np.testing.assert_array_equal(
            self._device(payload).take(wide).cpu().numpy(),
            payload.reshape(-1)[codes])
        np.testing.assert_array_equal(
            self._device(payload).take(narrow).cpu().numpy(),
            payload.reshape(-1)[codes])

    def test_a_half_payload_keeps_its_element_type(self):
        rng = np.random.default_rng(34)
        half = (rng.standard_normal((3, 5)) * 10).astype(np.float16)
        got = self._device(half).take(self._device([1, 4, -1]))
        np.testing.assert_array_equal(got.cpu().numpy(), half.reshape(-1)[[1, 4, -1]])
        self.assertEqual(got.dtype, tensorplay.float16)

    def test_a_list_with_no_positions_and_a_payload_with_no_entries(self):
        payload = self._device(np.zeros((2, 3), np.float32) + 1.0)
        empty = payload.take(self._device(np.zeros(0, np.int64)))
        self.assertEqual(tuple(empty.shape), (0,))
        self.assertEqual(empty.cpu().numpy().size, 0)
        with self.assertRaisesRegex(RuntimeError, "empty tensor"):
            tensorplay.take(self._device(np.zeros((0, 3), np.float32)),
                            self._device([0]))

    def test_a_long_list_over_a_wide_payload(self):
        rng = np.random.default_rng(35)
        table = np.arange(4096, dtype=np.float32)
        codes = rng.integers(0, 4096, 900)
        np.testing.assert_array_equal(
            tensorplay.take(self._device(table), self._device(codes)).cpu().numpy(),
            table[codes])
        rows = (rng.standard_normal((4, 256)) * 10).astype(np.float32)
        ids = rng.integers(0, 1024, (2, 3, 4))
        np.testing.assert_array_equal(
            tensorplay.take(self._device(rows.reshape(1024)), self._device(ids))
            .cpu().numpy(),
            rows.reshape(1024)[ids])

    def test_a_view_is_read_as_it_shows(self):
        rng = np.random.default_rng(36)
        payload = (rng.standard_normal((5, 8)) * 10).astype(np.float32)
        stepped = payload[:, ::2]
        np.testing.assert_array_equal(
            tensorplay.take(self._device(stepped), self._device([0, 3, -1]))
            .cpu().numpy(),
            stepped.reshape(-1)[[0, 3, -1]])

    def test_a_code_naming_no_entry_contributes_a_zero(self):
        # The row lookups of this backend answer a code outside the payload with
        # a zero rather than a host round trip to find it.
        payload = self._device(np.arange(6, dtype=np.float32) + 1.0)
        np.testing.assert_array_equal(
            payload.take(self._device([6, -7])).cpu().numpy(), [0.0, 0.0])

    def test_a_list_that_does_not_count_refuses(self):
        payload = self._device(np.zeros((3, 3), np.float32))
        with self.assertRaisesRegex(RuntimeError, "Int32 or Int64"):
            tensorplay.take(payload, self._device(np.array([1.0], np.float32)))


class TestVulkanArgsort(unittest.TestCase):
    """The permutation half of a selection: which row would be in which place."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    def _ordered(self, data, axis, descending=False):
        got = self._device(data).argsort(dim=axis, descending=descending).cpu().numpy()
        model = np.argsort(-data if descending else data,
                           axis=axis, kind="stable")
        np.testing.assert_array_equal(got, model, err_msg=f"argsort axis {axis}")
        return got

    def test_each_axis_of_a_row_is_ordered(self):
        rng = np.random.default_rng(37)
        data = (rng.standard_normal((5, 9)) * 10).astype(np.float32)
        for axis in (0, 1, -1):
            self._ordered(data, axis)
            self._ordered(data, axis, descending=True)

    def test_a_single_row_and_a_tall_payload(self):
        rng = np.random.default_rng(38)
        flat = (rng.standard_normal(45) * 10).astype(np.float32)
        np.testing.assert_array_equal(
            self._device(flat).argsort().cpu().numpy(),
            np.argsort(flat, kind="stable"))
        tall = (rng.standard_normal((2, 3, 4, 7)) * 10).astype(np.float32)
        got = self._device(tall).argsort(dim=2)
        self.assertEqual(tuple(got.shape), (2, 3, 4, 7))
        np.testing.assert_array_equal(got.cpu().numpy(),
                                      np.argsort(tall, axis=2, kind="stable"))

    def test_a_row_longer_than_one_sort_tile(self):
        rng = np.random.default_rng(39)
        long = (rng.standard_normal((3, 5000)) * 10).astype(np.float32)
        self._ordered(long, 1)
        self._ordered(long, 1, descending=True)

    def test_the_permutation_is_a_count_and_agrees_with_sort(self):
        rng = np.random.default_rng(40)
        data = (rng.standard_normal((6, 11)) * 10).astype(np.float32)
        order = self._device(data).argsort(dim=1, descending=True)
        self.assertEqual(order.dtype, tensorplay.int32)
        values, indices = tensorplay.sort(self._device(data), dim=1, descending=True)
        np.testing.assert_array_equal(order.cpu().numpy(), indices.cpu().numpy())
        # Reading the row back in that order gives the sorted values.
        np.testing.assert_array_equal(
            tensorplay.gather(self._device(data), 1, order).cpu().numpy(),
            values.cpu().numpy())

    def test_an_axis_beyond_the_payload_refuses(self):
        data = self._device(np.zeros((3, 4), np.float32))
        with self.assertRaises(RuntimeError):
            data.argsort(dim=4)
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            self._device(np.zeros((3, 4), np.int32)).argsort(dim=1)


class TestVulkanHalfSelection(unittest.TestCase):
    # A narrow row of logits sorts on the device too.  The reading of a lane
    # depends on the form that carries it: a texel a sampler widens on the way
    # in, or one half of a word.  Past the reading every walk orders those
    # widenings and keeps its own places, so a value narrows again only where
    # the answer is written.

    @staticmethod
    def _uploaded(array):
        # A payload that arrives from the host keeps the image form.
        return tensorplay.from_numpy(np.asarray(array, np.float16)).to("vulkan")

    @staticmethod
    def _built(array):
        # A payload the device lays out itself keeps its elements in words.
        return tensorplay.tensor(np.asarray(array, np.float16), device="vulkan")

    @staticmethod
    def _uses_dense_storage(payload):
        # Only single precision has both a texture and a linear form of lerp, so
        # a lerp that reports the missing image marks a payload that keeps its
        # elements in words.
        import tensorplay.functional as F
        try:
            F.lerp(payload, payload, 0.5)
            return False
        except RuntimeError:
            return True

    @staticmethod
    def _lanes(rows, cols):
        # Repeated lanes, so a tie has to be settled by the place it came from.
        rng = np.random.default_rng(41)
        return rng.choice(
            np.arange(-4.0, 4.5, 0.5, dtype=np.float16), (rows, cols))

    def _ordered(self, make, data, axis, descending):
        payload = make(data)
        values, indices = tensorplay.sort(payload, dim=axis, descending=descending)
        want_order = np.argsort(
            -data.astype(np.float32) if descending else data.astype(np.float32),
            axis=axis, kind="stable")
        want_values = np.take_along_axis(data, want_order, axis=axis)
        self.assertEqual(values.dtype, tensorplay.float16)
        self.assertEqual(indices.dtype, tensorplay.int32)
        np.testing.assert_array_equal(
            values.cpu().numpy(), want_values,
            err_msg=f"half values axis {axis} descending {descending}")
        np.testing.assert_array_equal(
            indices.cpu().numpy(), want_order,
            err_msg=f"half places axis {axis} descending {descending}")

    def test_a_half_row_orders_its_lanes_both_ways(self):
        data = self._lanes(5, 9)
        for make in (self._uploaded, self._built):
            for axis in (0, 1, -1):
                for descending in (False, True):
                    self._ordered(make, data, axis, descending)

    def test_a_half_permutation_agrees_with_the_order(self):
        data = self._lanes(6, 11)
        for make in (self._uploaded, self._built):
            payload = make(data)
            order = payload.argsort(dim=1, descending=True)
            _, indices = tensorplay.sort(payload, dim=1, descending=True)
            np.testing.assert_array_equal(
                order.cpu().numpy(), indices.cpu().numpy())
            np.testing.assert_array_equal(
                tensorplay.gather(payload, 1, order).cpu().numpy(),
                tensorplay.sort(payload, dim=1, descending=True)[0].cpu().numpy())

    def test_a_half_topk_answers_the_widest_lanes(self):
        data = self._lanes(4, 13)
        for make in (self._uploaded, self._built):
            for largest in (True, False):
                for k in (1, 3, 13):
                    values, indices = tensorplay.topk(
                        make(data), k, dim=-1, largest=largest)
                    order = np.argsort(
                        -data.astype(np.float32) if largest
                        else data.astype(np.float32), axis=-1, kind="stable")
                    np.testing.assert_array_equal(
                        values.cpu().numpy(),
                        np.take_along_axis(data, order, -1)[..., :k],
                        err_msg=f"half topk {k} largest {largest}")
                    np.testing.assert_array_equal(
                        indices.cpu().numpy(), order[..., :k],
                        err_msg=f"half topk places {k} largest {largest}")

    def test_a_half_row_longer_than_one_tile(self):
        rng = np.random.default_rng(42)
        long = (rng.standard_normal((3, 5000)) * 100).astype(np.float16)
        for make in (self._uploaded, self._built):
            self._ordered(make, long, 1, False)
            self._ordered(make, long, 1, True)

    def test_a_half_payload_beyond_an_image_is_ordered_in_words(self):
        # A plane the device cannot hold as an image is read and written a word
        # at a time, so both ends of the walk take their narrow variants.  The
        # texel of either width carries four lanes, so a single-precision twin
        # of the same shape answers whether this device has room for it.
        shape = (20000, 8)
        if not self._uses_dense_storage(
                tensorplay.from_numpy(np.zeros(shape, np.float32)).to("vulkan")):
            self.skipTest("this device holds a payload of that shape in an image")
        data = self._lanes(*shape)
        payload = self._uploaded(data)
        values, indices = tensorplay.sort(payload, dim=1)
        self.assertTrue(self._uses_dense_storage(values))
        order = np.argsort(data.astype(np.float32), axis=1, kind="stable")
        np.testing.assert_array_equal(
            values.cpu().numpy(), np.take_along_axis(data, order, 1))
        np.testing.assert_array_equal(indices.cpu().numpy(), order)

    def test_a_half_median_is_one_of_its_lanes(self):
        for size in (7, 8, 5000):
            data = self._lanes(1, size).reshape(size)
            for make in (self._uploaded, self._built):
                got = make(data).median().cpu().numpy()
                # The middle of an even row is the lower of its two centres.
                want = np.sort(data.astype(np.float32))[(size - 1) // 2]
                self.assertEqual(
                    float(got), float(want), msg=f"half median of {size}")

    def test_a_count_refuses_to_order(self):
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.from_numpy(np.zeros((3, 4), np.int64)).to("vulkan").sort()


class TestVulkanCountingRowLookup(unittest.TestCase):
    """A table whose rows hold counts is a table like any other: an id column,
    a list of positions, the row of codes a decode step reads out of a cached
    beam.  A row lookup never opens what the row carries, so the answer is the
    same codes in the order the list names - but the width of a count still
    decides which kernel moves it."""

    @staticmethod
    def _uploaded(values):
        # A payload that arrives from the host keeps the image form its shape
        # allows.
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _built(values):
        # A payload the device lays out itself keeps its elements in words.
        return tensorplay.tensor(np.asarray(values), device="vulkan")

    @staticmethod
    def _codes(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _rows(rows, cols):
        r = np.arange(rows, dtype=np.int64)[:, None]
        c = np.arange(cols, dtype=np.int64)[None, :]
        return (r * 977 + c * 131) % 4000000019 - 50000

    def _selected(self, make, table, axis, sel, dtype, label):
        import tensorplay.functional as F
        out = F.index_select(make(table), axis, self._codes(sel))
        self.assertEqual(out.dtype, dtype, msg=label)
        np.testing.assert_array_equal(
            out.cpu().numpy(), np.take(np.asarray(table), sel, axis=axis),
            err_msg=label)

    def test_a_column_of_counts_selects_its_rows(self):
        for dtype in (np.int32, np.int64):
            table = self._rows(60, 7).astype(dtype)
            element = (tensorplay.int32 if dtype is np.int32
                       else tensorplay.int64)
            for axis, sel in ((0, np.array([0, 59, 3, 3, 41, 17], np.int64)),
                              (1, np.array([0, 6, 2, 5, 1, 0], np.int64))):
                for make in (self._uploaded, self._built):
                    self._selected(
                        make, table, axis, sel, element,
                        f"{make.__name__} {np.dtype(dtype).name} axis {axis}")

    def test_a_counting_table_beyond_an_image_axis_is_read_in_words(self):
        # A flat payload this long has no texel route, so every code is moved
        # through linear storage a word (or a pair of words) at a time.
        for dtype in (np.int32, np.int64):
            table = self._rows(20000, 3).astype(dtype)
            rng = np.random.default_rng(51)
            sel = rng.integers(0, 20000, 777)
            for make in (self._uploaded, self._built):
                self._selected(
                    make, table, 0, sel,
                    tensorplay.int32 if dtype is np.int32 else tensorplay.int64,
                    f"{make.__name__} {np.dtype(dtype).name} long table")

    def test_a_row_width_that_is_not_a_group_of_words_straddles_them(self):
        # Five counts do not fill a group of four lanes, so the tail of one row
        # shares its words with the head of the next; a code that lands in that
        # tail has to be read out of the right lane.
        for dtype in (np.int32, np.int64):
            element = (tensorplay.int32 if dtype is np.int32
                       else tensorplay.int64)
            table = self._rows(900, 5).astype(dtype)
            self._selected(
                self._built, table, 1, np.array([0, 4, 1, 3], np.int64),
                element, f"{np.dtype(dtype).name} row tail, columns")
            self._selected(
                self._built, table, 0, np.arange(0, 900, 7, dtype=np.int64),
                element, f"{np.dtype(dtype).name} row tail, rows")

    def test_a_wide_count_keeps_every_bit_it_carries(self):
        # A code past the reach of one word lives in its high half: the answer
        # must show it, not the low word alone.
        r = np.arange(500, dtype=np.int64)[:, None]
        table = (r * 9000000000 + np.arange(4, dtype=np.int64)[None, :]).astype(
            np.int64)
        self.assertTrue(int(table.max()) > 2**32)
        payload = self._built(table)
        sel = np.array([0, 499, 250, 250, 7], dtype=np.int64)
        import tensorplay.functional as F
        got = F.index_select(payload, 0, self._codes(sel)).cpu().numpy()
        np.testing.assert_array_equal(got, table[sel], err_msg="wide rows")
        flat = self._built(table.reshape(-1))
        picked = tensorplay.take(
            flat, self._codes(np.array([0, -1, 1999, 0], np.int64)))
        want = table.reshape(-1)[[0, -1, 1999, 0]]
        np.testing.assert_array_equal(
            picked.cpu().numpy(), want, err_msg="wide flat lookup")

    def test_a_flat_lookup_of_counts_keeps_the_list_shape(self):
        table = self._rows(40, 9).astype(np.int32)
        rng = np.random.default_rng(52)
        codes = rng.integers(-360, 360, (3, 2, 5))
        for make in (self._uploaded, self._built):
            got = tensorplay.take(make(table), self._codes(codes))
            self.assertEqual(tuple(got.shape), (3, 2, 5))
            np.testing.assert_array_equal(
                got.cpu().numpy(), table.reshape(-1)[codes],
                err_msg=f"{make.__name__} counting take")

    def test_a_code_naming_no_row_contributes_a_zero(self):
        # The answer the other row lookups of this backend already give.
        import tensorplay.functional as F
        for dtype in (np.int32, np.int64):
            table = self._rows(24, 6).astype(dtype)
            sel = np.array([24, -1, 5, 1000, 2, -99], dtype=np.int64)
            want = np.zeros((len(sel), 6), table.dtype)
            want[2] = table[5]
            want[4] = table[2]
            for make in (self._uploaded, self._built):
                got = F.index_select(
                    make(table), 0, self._codes(sel)).cpu().numpy()
                np.testing.assert_array_equal(
                    got, want,
                    err_msg=f"{np.dtype(dtype).name} {make.__name__} dead code")

    def test_a_payload_narrower_than_a_word_still_refuses(self):
        payload = tensorplay.from_numpy(
            np.zeros((4, 3), np.bool_)).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "Float32, Float16, Int32 and Int64"):
            tensorplay.functional.index_select(
                payload, 0, self._codes(np.array([0, 1], np.int64)))
        with self.assertRaisesRegex(
                RuntimeError, "Float32, Float16, Int32 and Int64"):
            tensorplay.take(payload, self._codes(np.array([0], np.int64)))


class TestVulkanCountingGather(unittest.TestCase):
    """A per-position pick out of a table of counts: the code at each answer
    element names a row of the source, and the answer is a count again.  A
    gathered axis that is one of the two texel axes moves the fetch along that
    axis; an outer axis would have to carry a value between lanes of different
    texels and so runs over words, where a code wider than a word lives a pair
    of words at a time."""

    @staticmethod
    def _uploaded(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _built(values):
        return tensorplay.tensor(np.asarray(values), device="vulkan")

    @staticmethod
    def _table(rows, cols):
        r = np.arange(rows, dtype=np.int64)[:, None]
        c = np.arange(cols, dtype=np.int64)[None, :]
        return (r * 977 + c * 131) % 4000000019 - 50000

    def _gathered(self, make, table, codes, axis, dtype, label):
        out = tensorplay.gather(make(table), axis, self._uploaded(codes))
        self.assertEqual(out.dtype, dtype, msg=label)
        self.assertEqual(tuple(out.shape), tuple(codes.shape), msg=label)
        np.testing.assert_array_equal(
            out.cpu().numpy(), np.take_along_axis(table, codes, axis=axis),
            err_msg=label)

    def test_a_counting_table_is_picked_row_by_row(self):
        rng = np.random.default_rng(61)
        for dtype in (np.int32, np.int64):
            element = (tensorplay.int32 if dtype is np.int32
                       else tensorplay.int64)
            table = self._table(60, 7).astype(dtype)
            rows = rng.integers(0, 60, (4, 7)).astype(np.int64)
            cols = rng.integers(0, 7, (60, 3)).astype(np.int64)
            for make in (self._uploaded, self._built):
                self._gathered(make, table, rows, 0, element,
                               f"{make.__name__} {np.dtype(dtype).name} rows")
                self._gathered(make, table, cols, 1, element,
                               f"{make.__name__} {np.dtype(dtype).name} cols")

    def test_a_gather_of_an_outer_axis_crosses_lanes(self):
        # The batch and channel axes of a four-axis payload live in the depth
        # planes and the lanes, so their codes are moved over linear storage
        # even when the shape has an image form.
        rng = np.random.default_rng(62)
        table_shape = (5, 7, 4)
        for dtype in (np.int32, np.int64):
            element = (tensorplay.int32 if dtype is np.int32
                       else tensorplay.int64)
            table = np.arange(
                5 * 7 * 4, dtype=np.int64).reshape(table_shape).astype(dtype)
            outer = rng.integers(0, 5, (3, 7, 4)).astype(np.int64)
            middle = rng.integers(0, 7, (5, 2, 4)).astype(np.int64)
            inner = rng.integers(0, 4, (5, 7, 6)).astype(np.int64)
            for axis, codes in ((0, outer), (1, middle), (2, inner)):
                for make in (self._uploaded, self._built):
                    self._gathered(
                        make, table, codes, axis, element,
                        f"{np.dtype(dtype).name} rank three axis {axis}")

    def test_a_long_index_list_has_no_texel_route(self):
        rng = np.random.default_rng(63)
        for dtype in (np.int32, np.int64):
            element = (tensorplay.int32 if dtype is np.int32
                       else tensorplay.int64)
            table = np.arange(9000 * 3, dtype=np.int64).reshape(
                9000, 3).astype(dtype)
            codes = rng.integers(0, 3, (9000, 5)).astype(np.int64)
            for make in (self._uploaded, self._built):
                self._gathered(
                    make, table, codes, 1, element,
                    f"{np.dtype(dtype).name} long per-position pick")

    def test_a_wide_code_survives_a_gather(self):
        # A count past the reach of one word lives in its high half, and the
        # answer must show it.
        r = np.arange(500, dtype=np.int64)[:, None]
        table = (r * 9000000000 + np.arange(4, dtype=np.int64)[None, :]).astype(
            np.int64)
        self.assertTrue(int(table.max()) > 2**32)
        codes = np.repeat(
            np.array([[0], [499], [250], [7], [0]], np.int64), 4, axis=1)
        codes[3] = [250, 250, 7, 3]
        out = tensorplay.gather(self._built(table), 0, self._uploaded(codes))
        np.testing.assert_array_equal(
            out.cpu().numpy(), np.take_along_axis(table, codes, axis=0),
            err_msg="wide rows through a gather")

    def test_a_code_naming_no_row_contributes_a_zero(self):
        for dtype in (np.int32, np.int64):
            element = (tensorplay.int32 if dtype is np.int32
                       else tensorplay.int64)
            table = self._table(12, 4).astype(dtype)
            codes = np.array([[12, -1, 2, 0], [5, 99, 1, 11]], np.int64)
            out = tensorplay.gather(
                self._uploaded(table), 0, self._uploaded(codes))
            want = np.take_along_axis(
                table, np.where((codes < 0) | (codes >= 12), 0, codes), axis=0)
            want[(codes < 0) | (codes >= 12)] = 0
            self.assertEqual(out.dtype, element, msg="dead code dtype")
            np.testing.assert_array_equal(
                out.cpu().numpy(), want,
                err_msg=f"{np.dtype(dtype).name} dead code")

    def test_a_payload_narrower_than_a_word_still_refuses(self):
        payload = tensorplay.from_numpy(np.zeros((4, 3), np.bool_)).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "Float32, Float16, Int32 and Int64"):
            tensorplay.gather(
                payload, 0, tensorplay.from_numpy(
                    np.zeros((4, 3), np.int64)).to("vulkan"))


class TestVulkanCountingCat(unittest.TestCase):
    """Joining sequences of counts: a decode loop grows its token ids, its
    positions and its mask by appending to the end, so the walk that stitches
    payloads together has to carry a 64-bit element as well as a 32-bit one."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values, np.int64)).to("vulkan")

    def _joined(self, parts, dim):
        device = [self._device(part) for part in parts]
        got = tensorplay.cat(device, dim)
        self.assertEqual(got.dtype, tensorplay.int64)
        self.assertTrue(got.device.is_vulkan())
        model = np.concatenate([np.asarray(part, np.int64) for part in parts],
                               axis=dim)
        np.testing.assert_array_equal(
            np.asarray(got.cpu()), model, err_msg=f"cat along axis {dim}")
        return got

    def test_two_vectors_join_at_either_end(self):
        first = np.arange(7, dtype=np.int64) * 1000
        second = np.arange(5, dtype=np.int64) - 4
        joined = self._joined([first, second], 0)
        self.assertEqual(tuple(joined.shape), (12,))
        self._joined([second, first], 0)
        self._joined([first, second, first], 0)

    def test_a_row_of_a_table_joins_along_the_feature_axis(self):
        rng = np.random.default_rng(90)
        ids = rng.integers(0, 50257, size=(2, 6), dtype=np.int64)
        extra = rng.integers(0, 50257, size=(2, 1), dtype=np.int64)
        self.assertEqual(tuple(self._joined([ids, extra], 1).shape), (2, 7))
        self.assertEqual(tuple(self._joined([ids, ids], 0).shape), (4, 6))

    def test_every_axis_of_a_rank_three_or_four_payload(self):
        rng = np.random.default_rng(91)
        for shape in ((2, 3, 4), (2, 3, 4, 5)):
            base = rng.integers(-1000, 1000, size=shape, dtype=np.int64)
            for axis in range(len(shape)):
                grown = list(shape)
                grown[axis] += 2
                other = rng.integers(
                    -1000, 1000, size=tuple(grown), dtype=np.int64)
                self._joined([base, other], axis)

    def test_a_count_wider_than_a_word_travels_whole(self):
        # Values that set the high word of the element, positive and negative,
        # so a pass that moved only one word of a pair would be caught.
        wide = np.array(
            [1 << 40, -(1 << 40), (1 << 32) + 7, -((1 << 32) + 7),
             1 << 62, -(1 << 62), 0, 2 ** 63 - 1, -(2 ** 63)],
            dtype=np.int64)
        tail = np.array([1234567890123, -987654321098], dtype=np.int64)
        self._joined([wide, tail], 0)
        self._joined([tail, wide], 0)

    def test_a_payload_without_an_image_form(self):
        # Past the image axis limit the element count is only ever linear, so
        # the join runs entirely over words.
        first = np.arange(20000, dtype=np.int64)
        second = np.arange(30000, dtype=np.int64)
        got = self._joined([first, second], 0)
        self.assertEqual(tuple(got.shape), (50000,))

    def test_a_growing_sequence_walks_one_step_at_a_time(self):
        ids = np.array([[41, 59, 26]], dtype=np.int64)
        built = self._device(ids)
        model = ids.copy()
        for step in range(5):
            nxt = np.array([[step + 1]], dtype=np.int64)
            built = tensorplay.cat([built, self._device(nxt)], dim=1)
            model = np.concatenate([model, nxt], axis=1)
            np.testing.assert_array_equal(np.asarray(built.cpu()), model)
        self.assertEqual(tuple(built.shape), (1, 8))

    def test_the_shapes_outside_the_join_axis_stay_agreed(self):
        base = np.arange(12, dtype=np.int64).reshape((3, 4))
        odd = np.arange(8, dtype=np.int64).reshape((2, 4))
        with self.assertRaisesRegex(RuntimeError, "sizes must match"):
            tensorplay.cat([self._device(base), self._device(odd)], 1)
        # An element type other than the first one's is refused too.
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            tensorplay.cat(
                [self._device(base), tensorplay.from_numpy(
                    np.arange(12, dtype=np.float32).reshape((3, 4))
                ).to("vulkan")], 0)


class TestVulkanHalfTexelFormat(unittest.TestCase):
    """A half payload lives in texels of eight bytes, so every walk that writes
    its planes has to name that width.  These are the joins, reorders, picks and
    row reads a decode loop runs on a narrow table."""

    @staticmethod
    def _grid(shape, dtype=np.float16, modulus=13, scale=1):
        steps = int(np.prod(shape))
        values = ((np.arange(steps) % modulus) - modulus // 2) * scale
        return values.astype(dtype).reshape(shape)

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    def _same(self, got, model, label):
        """Both sides read as the bits they store, so a texel written in the
        wrong width cannot hide behind an approximate comparison."""
        model = np.asarray(model)
        self.assertEqual(got.dtype, tensorplay.from_numpy(model).dtype, label)
        self.assertEqual(tuple(got.shape), tuple(model.shape), label)
        np.testing.assert_array_equal(
            np.asarray(got.cpu()).view(np.uint16),
            model.view(np.uint16), err_msg=label)

    def test_a_narrow_join_lands_in_texels_of_its_own_width(self):
        plane = self._grid((4, 6))
        other = self._grid((2, 6), modulus=11)
        short = self._grid((4, 3), modulus=7)
        self._same(
            tensorplay.cat([self._device(plane), self._device(other)], 0),
            np.concatenate([plane, other], axis=0), "half cat of rows")
        self._same(
            tensorplay.cat([self._device(plane), self._device(short)], 1),
            np.concatenate([plane, short], axis=1), "half cat of columns")
        stack = self._grid((2, 3, 4))
        for axis in range(3):
            grown = list(stack.shape)
            grown[axis] += 1
            second = self._grid(tuple(grown), modulus=9)
            self._same(
                tensorplay.cat([self._device(stack), self._device(second)],
                               axis),
                np.concatenate([stack, second], axis=axis),
                f"half rank-3 cat along axis {axis}")

    def test_a_narrow_reorder_and_pick_read_their_own_width(self):
        model = self._grid((4, 6))
        device = self._device(model)
        self._same(tensorplay.permute(device, (1, 0)), model.T, "half permute")
        self._same(device.select(0, 1), model[1], "half pick of a row")
        self._same(device.select(1, 4), model[:, 4], "half pick of a column")
        self._same(device[1:4:2], model[1:4:2], "half strided slice")
        self._same(device[:, 0:6:2], model[:, 0:6:2], "half column slice")
        for row, want in zip(tensorplay.unbind(device, 0), model):
            self._same(row, want, "half row off the unbind walk")
        cube = self._grid((2, 3, 4))
        self._same(
            tensorplay.permute(self._device(cube), (2, 0, 1)),
            np.transpose(cube, (2, 0, 1)), "half rank-3 permute")

    def test_a_narrow_row_read_follows_the_codes(self):
        table = self._grid((9, 5))
        codes = np.array([4, 0, 8, 1], dtype=np.int64)
        device = self._device(table)
        self._same(
            tensorplay.index_select(device, 0, self._device(codes)),
            table[codes], "half row lookup")
        self._same(
            tensorplay.index_select(device, 1, self._device(codes[:2])),
            table[:, codes[:2]], "half column lookup")
        picked = np.tile((np.arange(9) % 5)[:, None], (1, 2))
        self._same(
            tensorplay.gather(device, 1, self._device(picked)),
            np.take_along_axis(table, picked, 1), "half gather")
        self._same(
            tensorplay.embedding(device, self._device(codes)),
            table[codes], "half embedding")
        replacement = self._grid((2, 5), modulus=5)
        written = table.copy()
        written[:2] = replacement
        self._same(
            tensorplay.index_copy(
                device, 0, self._device(np.array([0, 1], np.int64)),
                self._device(replacement)),
            written, "half row overwrite")

    def test_a_narrow_choice_keeps_both_planes_aligned(self):
        plane = self._grid((4, 6))
        second = self._grid((4, 6), modulus=11)
        condition = np.array([[1, 0, 1, 1, 0, 0]] * 4, np.bool_)
        self._same(
            tensorplay.where(self._device(condition), self._device(plane),
                             self._device(second)),
            np.where(condition, plane, second), "half choice")

    def test_a_narrow_payload_past_the_image_limit_is_still_picked(self):
        tall = self._grid((20000, 4))
        device = self._device(tall)
        self._same(device.select(0, 12345), tall[12345],
                   "half pick out of a long line")
        self._same(device[100:20000:500], tall[100:20000:500],
                   "half slice of a long line")

    def test_the_walks_other_widths_use_are_undisturbed(self):
        for dtype in (np.float32, np.int32):
            plane = self._grid((4, 6), dtype)
            device = self._device(plane)
            label = np.dtype(dtype).name
            np.testing.assert_array_equal(
                np.asarray(
                    tensorplay.cat([device, device], 1).cpu()),
                np.concatenate([plane, plane], axis=1), err_msg=label)
            np.testing.assert_array_equal(
                np.asarray(tensorplay.permute(device, (1, 0)).cpu()), plane.T,
                err_msg=label)
            np.testing.assert_array_equal(
                np.asarray(device.select(0, 2).cpu()), plane[2], err_msg=label)
            np.testing.assert_array_equal(
                np.asarray(device[1:4:2].cpu()), plane[1:4:2], err_msg=label)
            np.testing.assert_array_equal(
                np.asarray(device.select(0, 1).cpu()), plane[1],
                err_msg=f"{label} row off the unbind walk")
            np.testing.assert_array_equal(
                np.asarray(tensorplay.index_select(
                    device, 0, self._device(np.array([3, 0], np.int64))).cpu()),
                plane[[3, 0]], err_msg=label)
            np.testing.assert_array_equal(
                np.asarray(tensorplay.where(
                    self._device(np.zeros((4, 6), np.bool_)), device,
                    device).cpu()), plane, err_msg=label)
        # A code wider than a word has no image form, so its pick stays a
        # strided line of pairs.
        wide = (np.arange(24, dtype=np.int64) * 10 ** 12).reshape((4, 6))
        np.testing.assert_array_equal(
            np.asarray(self._device(wide).select(1, 3).cpu()), wide[:, 3])


class TestVulkanBernoulliOutOfPlace(unittest.TestCase):
    """A draw that leaves the probability payload alone: the answer is a fresh
    tensor of the same shape and element type, not a refill of the source."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _numpy(payload):
        """A payload read back as numpy, drawn or not."""
        if isinstance(payload, tensorplay.Tensor):
            return np.asarray(payload.cpu())
        return np.asarray(payload)

    def _flags(self, values):
        """A draw is only ever 0 or 1, in the source's own element type."""
        self.assertTrue(np.isin(values, (0.0, 1.0)).all(), values.tolist())
        return values

    def test_the_answer_is_a_fresh_tensor_of_the_same_kind(self):
        rng = np.random.default_rng(61)
        plane = (rng.random((4, 6)) * 0.9 + 0.05).astype(np.float32)
        source = self._device(plane)
        got = tensorplay.bernoulli(source)
        self.assertEqual(tuple(got.shape), (4, 6))
        self.assertEqual(got.dtype, tensorplay.float32)
        self.assertTrue(got.device.is_vulkan())
        self._flags(self._numpy(got))
        # The source still holds its own probabilities.
        np.testing.assert_array_equal(self._numpy(source), plane)

    def test_every_element_follows_its_own_probability(self):
        plane = np.concatenate([np.zeros(8, np.float32), np.ones(8, np.float32)])
        got = self._numpy(tensorplay.bernoulli(self._device(plane)))
        np.testing.assert_array_equal(got[:8], np.zeros(8))
        np.testing.assert_array_equal(got[8:], np.ones(8))

    def test_a_strided_plane_is_laid_out_before_it_is_read(self):
        # Columns 0, 2, 4 and 6 carry probabilities 0, 0, 1 and 1, so the draw
        # over that view must be exactly those flags.
        rows = np.tile(np.array([0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0, 0.5],
                                np.float32), (2, 1))
        view = self._device(rows)[:, 0:8:2]
        got = self._numpy(tensorplay.bernoulli(view))
        self.assertEqual(tuple(got.shape), (2, 4))
        np.testing.assert_array_equal(got, np.array([[0.0, 0.0, 1.0, 1.0]] * 2))

    def test_one_probability_governs_the_whole_payload(self):
        source = self._device(np.zeros((2, 7), np.float32))
        got = self._numpy(tensorplay.bernoulli(source, 0.5))
        self.assertEqual(got.shape, (2, 7))
        self._flags(got)
        np.testing.assert_array_equal(
            self._numpy(tensorplay.bernoulli(source, 0.0)), np.zeros((2, 7)))
        np.testing.assert_array_equal(
            self._numpy(tensorplay.bernoulli(source, 1.0)), np.ones((2, 7)))

    def test_a_share_of_one_is_reached_by_a_count_of_one(self):
        draws = np.asarray(
            tensorplay.bernoulli(self._device(np.zeros(40000, np.float32)), 0.3)
            .cpu())
        self.assertAlmostEqual(float(draws.mean()), 0.3, delta=0.01)
        plane = np.asarray(
            tensorplay.bernoulli(self._device(np.full(40000, 0.5, np.float32)))
            .cpu())
        self.assertAlmostEqual(float(plane.mean()), 0.5, delta=0.01)

    def test_a_wide_or_a_tall_payload_keeps_its_geometry(self):
        tall = tensorplay.bernoulli(
            self._device(np.full((2, 3, 4, 5), 0.5, np.float32)))
        self.assertEqual(tuple(tall.shape), (2, 3, 4, 5))
        # A count beyond the image axis limit stays a count of that length.
        long = np.asarray(
            tensorplay.bernoulli(self._device(np.zeros(5000, np.float32)), 0.25)
            .cpu())
        self.assertEqual(long.shape, (5000,))
        self.assertAlmostEqual(float(long.mean()), 0.25, delta=0.02)
        empty = tensorplay.bernoulli(self._device(np.zeros((0, 3), np.float32)))
        self.assertEqual(tuple(empty.shape), (0, 3))

    def test_a_seeded_draw_repeats_and_the_next_draw_moves(self):
        tensorplay.manual_seed(1234)
        first = self._numpy(tensorplay.bernoulli(self._device(
            np.zeros(256, np.float32)), 0.5))
        tensorplay.manual_seed(1234)
        again = self._numpy(tensorplay.bernoulli(self._device(
            np.zeros(256, np.float32)), 0.5))
        np.testing.assert_array_equal(first, again)
        moved = self._numpy(tensorplay.bernoulli(self._device(
            np.zeros(256, np.float32)), 0.5))
        self.assertFalse(np.array_equal(first, moved))

    def test_the_same_stream_answers_for_both_forms(self):
        # Both forms fill a payload the device allocated, so a seeded stream
        # maps every word to the same element either way.
        tensorplay.manual_seed(99)
        out = np.asarray(
            tensorplay.bernoulli(tensorplay.zeros(512, device="vulkan"), 0.5).cpu())
        tensorplay.manual_seed(99)
        inplace = tensorplay.zeros(512, device="vulkan")
        inplace.bernoulli_(0.5)
        np.testing.assert_array_equal(out, np.asarray(inplace.cpu()))

    def test_a_payload_on_another_device_is_served_there(self):
        host = tensorplay.tensor(np.full(8, 0.5, np.float32), device="cpu")
        got = tensorplay.bernoulli(host)
        self.assertEqual(got.device.type, "cpu")
        self.assertTrue(set(np.asarray(got.cpu()).tolist()) <= {0.0, 1.0})
        ones = tensorplay.bernoulli(host, 1.0)
        np.testing.assert_array_equal(np.asarray(ones.cpu()), np.ones(8))

    def test_outside_the_supported_surface_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "Float32"):
            tensorplay.bernoulli(self._device(np.arange(6, dtype=np.int64)))
        with self.assertRaisesRegex(RuntimeError, "Float32"):
            tensorplay.bernoulli(self._device(np.zeros(6, np.float16)))
        with self.assertRaisesRegex(RuntimeError, "1d to 4d"):
            tensorplay.bernoulli(self._device(np.array(0.5, np.float32)))
        with self.assertRaisesRegex(RuntimeError, "ndim <= 4"):
            tensorplay.bernoulli(
                self._device(np.zeros((1, 1, 1, 1, 2), np.float32)))
        for p in (-0.5, 1.5):
            with self.assertRaisesRegex(RuntimeError, r"\[0, 1\]"):
                tensorplay.bernoulli(self._device(np.zeros(4, np.float32)), p)


class TestVulkanConstantPad(unittest.TestCase):
    """A frame around a payload: negative counts crop, positive counts join a
    slab of the fill value, and the payload itself keeps its place."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values)).to("vulkan")

    @staticmethod
    def _framed(data, pad, value):
        """The same rule read off the axis it names: the list counts from the
        last axis inwards, and each entry is that end's signed count."""
        ndim = data.ndim
        pairs = len(pad) // 2
        cropped = data
        for axis in range(ndim - pairs, ndim):
            at = 2 * (ndim - axis - 1)
            left, right = pad[at], pad[at + 1]
            window = [slice(None)] * ndim
            window[axis] = slice(
                -left if left < 0 else 0,
                cropped.shape[axis] + (right if right < 0 else 0))
            cropped = cropped[tuple(window)]
        width = [(0, 0)] * ndim
        for axis in range(ndim - pairs, ndim):
            at = 2 * (ndim - axis - 1)
            width[axis] = (max(pad[at], 0), max(pad[at + 1], 0))
        return np.pad(cropped, width, mode="constant", constant_values=value)

    def _drawn(self, data, pad, value, payload=None):
        """data is what the payload shows; payload defaults to a fresh upload."""
        model = self._framed(np.asarray(data), pad, value)
        device = self._device(data) if payload is None else payload
        got = tensorplay.constant_pad_nd(device, pad, value)
        np.testing.assert_array_equal(
            np.asarray(got.cpu()), model, err_msg=f"pad {pad}")
        return got, model

    def test_a_single_axis_gains_slabs_of_the_fill(self):
        rng = np.random.default_rng(71)
        data = (rng.standard_normal(7) * 3).astype(np.float32)
        for pad in ([0, 0], [2, 0], [0, 3], [1, 2], [4, 5]):
            got, model = self._drawn(data, pad, -2.5)
            self.assertEqual(tuple(got.shape), tuple(model.shape))
            self.assertEqual(got.dtype, tensorplay.float32)

    def test_both_axes_gain_their_own_frame(self):
        rng = np.random.default_rng(72)
        data = (rng.standard_normal((3, 5)) * 3).astype(np.float32)
        got, _ = self._drawn(data, [1, 2, 3, 0], -7.5)
        self.assertEqual(tuple(got.shape), (6, 8))
        self._drawn(data, [0, 0, 1, 1], 4.0)

    def test_a_negative_count_crops_that_end(self):
        rng = np.random.default_rng(73)
        data = (rng.standard_normal((4, 6)) * 3).astype(np.float32)
        cropped, _ = self._drawn(data, [-1, -2, 0, 0], 0.0)
        self.assertEqual(tuple(cropped.shape), (4, 3))
        # A crop on one end of an axis and a slab on the other.
        for pad in ([-1, 2, 0, 0], [3, -2], [0, 0, -1, 4]):
            self._drawn(data, pad, 0.5)

    def test_the_frame_reaches_every_supported_rank(self):
        rng = np.random.default_rng(74)
        for shape in ((5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)):
            data = (rng.standard_normal(shape) * 3).astype(np.float32)
            got, model = self._drawn(data, [1, 1], 0.0)
            self.assertEqual(tuple(got.shape), tuple(model.shape))

    def test_a_count_beyond_the_texel_axis_limit_keeps_the_payload(self):
        rng = np.random.default_rng(75)
        short = (rng.standard_normal(37) * 3).astype(np.float32)
        self._drawn(short, [5, 6], 2.0)
        long = (rng.standard_normal(5000) * 3).astype(np.float32)
        got, model = self._drawn(long, [1, 2], 0.0)
        self.assertEqual(tuple(got.shape), (5003,))
        np.testing.assert_array_equal(np.asarray(got.cpu()), model)

    def test_a_view_is_framed_as_it_shows(self):
        rng = np.random.default_rng(76)
        base = self._device((rng.standard_normal((4, 8)) * 3).astype(np.float32))
        view = base[1:4, 0:8:2]
        got, model = self._drawn(
            np.asarray(view.cpu()), [0, 0, 1, 1], 0.0, payload=view)
        self.assertEqual(tuple(got.shape), (5, 4))
        np.testing.assert_array_equal(np.asarray(got.cpu()), model)

    def test_the_element_types_a_concat_walk_can_carry(self):
        cases = [
            (np.arange(6, dtype=np.float32), 9.0),
            (np.arange(6, dtype=np.float16), 9.0),
            (np.arange(6, dtype=np.int32), 9),
            (np.array([True, False, True]), False),
        ]
        for arr, value in cases:
            self._drawn(arr, [1, 2], value)
        # A 64-bit count has no image form, so its slabs join it to the payload
        # through the linear walk, which moves a whole pair of words per element.
        padded, _ = self._drawn(np.arange(6, dtype=np.int64), [1, 2], 9)
        self.assertEqual(padded.dtype, tensorplay.int64)

    def test_the_frontend_entry_agrees(self):
        import tensorplay.nn.functional as nnf
        rng = np.random.default_rng(77)
        data = (rng.standard_normal((3, 4)) * 3).astype(np.float32)
        device = self._device(data)
        np.testing.assert_array_equal(
            np.asarray(nnf.pad(device, [1, 2]).cpu()),
            self._framed(data, [1, 2], 0.0))
        np.testing.assert_array_equal(
            np.asarray(nnf.pad(device, [1, 2], value=-1.0).cpu()),
            self._framed(data, [1, 2], -1.0))
        # Two pairs reach the last two axes.
        np.testing.assert_array_equal(
            np.asarray(nnf.pad(device, [1, 2, 0, 1]).cpu()),
            self._framed(data, [1, 2, 0, 1], 0.0))

    def test_an_empty_axis_still_takes_its_frame(self):
        got, model = self._drawn(np.zeros((0, 3), np.float32), [1, 1], 0.0)
        self.assertEqual(tuple(got.shape), tuple(model.shape))
        outer, outer_model = self._drawn(np.zeros((2, 0), np.float32),
                                        [0, 0, 2, 2], 0.0)
        self.assertEqual(tuple(outer.shape), tuple(outer_model.shape))

    def test_outside_the_supported_surface_refuses(self):
        data = self._device(np.zeros(4, np.float32))
        with self.assertRaisesRegex(RuntimeError, "pairs of counts"):
            tensorplay.constant_pad_nd(data, [1, 2, 3], 0.0)
        with self.assertRaisesRegex(RuntimeError, "covers 4 ends"):
            tensorplay.constant_pad_nd(data, [1, 1, 1, 1], 0.0)
        with self.assertRaisesRegex(RuntimeError, "cannot take -9 and -2"):
            tensorplay.constant_pad_nd(data, [-9, -2], 0.0)
        with self.assertRaisesRegex(RuntimeError, "1d to 4d"):
            tensorplay.constant_pad_nd(
                self._device(np.array(1.0, np.float32)), [1, 1], 0.0)


class TestVulkanSearchsorted(unittest.TestCase):
    """The position every query takes in a sorted sequence, and the bucket it
    falls into once the sequence is read as a set of edges.  The walk halves one
    table per query, so the answer carries a position for each query and the
    table is judged as it lies: a sequence handed over unordered gets the
    position the halving stops at, which is why no order is verified here."""

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.asarray(values).copy()).to("vulkan")

    @staticmethod
    def _back(payload):
        return np.asarray(payload.cpu())

    @staticmethod
    def _model(sequence, queries, right=False):
        """One table answers a rank-1 sequence; otherwise every row of the
        leading block answers the row of queries it faces."""
        side = "right" if right else "left"
        table = np.ascontiguousarray(sequence)
        if np.ndim(queries) == 0:
            return np.array(
                np.searchsorted(table.ravel(), queries, side=side),
                dtype=np.int64)
        values = np.asarray(queries)
        if table.ndim == 1:
            return np.searchsorted(table, values, side=side)
        rows = table.reshape(-1, table.shape[-1])
        columns = values.reshape(rows.shape[0], values.shape[-1])
        answer = np.empty(columns.shape, dtype=np.int64)
        for row in range(rows.shape[0]):
            answer[row] = np.searchsorted(rows[row], columns[row], side=side)
        return answer.reshape(values.shape)

    def test_the_position_of_every_query(self):
        table = np.array(
            [-3.0, -1.0, 0.0, 0.5, 1.0, 1.0, 2.0, 7.5], dtype=np.float32)
        queries = np.array(
            [-4.0, -3.0, -1.5, 0.0, 0.75, 1.0, 2.0, 7.4, 7.5, 100.0],
            dtype=np.float32)
        got = tensorplay.searchsorted(self._device(table), self._device(queries))
        self.assertEqual(got.dtype, tensorplay.int64)
        self.assertEqual(tuple(got.shape), queries.shape)
        np.testing.assert_array_equal(self._back(got), self._model(table, queries))

    def test_either_bound_of_a_tied_table(self):
        table = np.array([1.0, 2.0, 2.0, 2.0, 5.0], dtype=np.float32)
        queries = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0], dtype=np.float32)
        left = self._back(
            tensorplay.searchsorted(self._device(table), self._device(queries)))
        right = self._back(
            tensorplay.searchsorted(
                self._device(table), self._device(queries), right=True))
        np.testing.assert_array_equal(left, self._model(table, queries))
        np.testing.assert_array_equal(
            right, self._model(table, queries, right=True))
        # Every tie shifts the bound by the run it covers, and nothing else moves.
        np.testing.assert_array_equal(right - left, [0, 1, 3, 0, 1, 0])
        for side in ("left", "right"):
            np.testing.assert_array_equal(
                self._back(
                    tensorplay.searchsorted(
                        self._device(table), self._device(queries), side=side)),
                left if side == "left" else right,
                err_msg=f"side={side}")

    def test_a_row_of_tables_answers_its_own_row(self):
        state = np.random.RandomState(0)
        tables = np.sort(state.randn(4, 9).astype(np.float32), axis=-1)
        queries = np.random.RandomState(1).randn(4, 6).astype(np.float32)
        for right in (False, True):
            got = tensorplay.searchsorted(
                self._device(tables), self._device(queries), right=right)
            self.assertEqual(tuple(got.shape), queries.shape)
            np.testing.assert_array_equal(
                self._back(got), self._model(tables, queries, right))

        deep = np.sort(
            np.random.RandomState(2).randn(2, 3, 11).astype(np.float32), axis=-1)
        deep_queries = np.random.RandomState(3).randn(2, 3, 5).astype(np.float32)
        np.testing.assert_array_equal(
            self._back(tensorplay.searchsorted(
                self._device(deep), self._device(deep_queries))),
            self._model(deep, deep_queries))

    def test_the_answer_narrows_to_a_count_when_asked(self):
        table = np.sort(np.random.RandomState(4).randn(64).astype(np.float32))
        queries = np.random.RandomState(5).randn(9).astype(np.float32)
        got = tensorplay.searchsorted(
            self._device(table), self._device(queries), out_int32=True)
        self.assertEqual(got.dtype, tensorplay.int32)
        np.testing.assert_array_equal(
            self._back(got), self._model(table, queries).astype(np.int32))
        wide = tensorplay.searchsorted(self._device(table), self._device(queries))
        self.assertEqual(wide.dtype, tensorplay.int64)
        np.testing.assert_array_equal(self._back(wide), self._back(got).astype(np.int64))

    def test_a_mask_a_byte_or_a_half_is_widened_first(self):
        cases = [
            (np.array([-2.5, 0.0, 1.5, 100.0], dtype=np.float16),
             np.array([-3.0, -2.5, 0.0, 1.6, 100.0, 101.0], dtype=np.float16)),
            (np.array([-5, -1, 0, 3, 100], dtype=np.int8),
             np.array([-6, -5, 0, 4, 100, 127], dtype=np.int8)),
            (np.array([3, 7, 8, 200], dtype=np.uint8),
             np.array([0, 3, 120, 201], dtype=np.uint8)),
            (np.array([False, False, True, True, True], dtype=bool),
             np.array([False, True], dtype=bool)),
            (np.array([-8, -3, 0, 0, 5, 127], dtype=np.int32),
             np.array([-9, -8, -4, 0, 4, 5, 128], dtype=np.int32)),
        ]
        for table, queries in cases:
            for right in (False, True):
                got = tensorplay.searchsorted(
                    self._device(table), self._device(queries), right=right)
                np.testing.assert_array_equal(
                    self._back(got), self._model(table, queries, right),
                    err_msg=f"{table.dtype} bound {'right' if right else 'left'}")

    def test_operands_of_two_widths_compare_in_the_wider(self):
        table = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 7.5], dtype=np.float32)
        counts = np.array([-4, -3, 0, 1, 8], dtype=np.int32)
        np.testing.assert_array_equal(
            self._back(tensorplay.searchsorted(self._device(table), self._device(counts))),
            self._model(table, counts))
        edges = np.array([-8, -3, 0, 0, 5, 127], dtype=np.int32)
        measured = np.array([-8.5, 0.0, 6.0], dtype=np.float32)
        np.testing.assert_array_equal(
            self._back(tensorplay.searchsorted(self._device(edges), self._device(measured))),
            self._model(edges, measured))

    def test_a_scalar_query_answers_one_position(self):
        table = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 1.0, 2.0, 7.5],
                         dtype=np.float32)
        got = tensorplay.searchsorted(self._device(table), 0.75)
        self.assertEqual(tuple(got.shape), ())
        self.assertEqual(got.dtype, tensorplay.int64)
        self.assertEqual(int(self._back(got)), 4)
        self.assertEqual(
            int(self._back(tensorplay.searchsorted(self._device(table), 1.0))), 4)
        self.assertEqual(
            int(self._back(tensorplay.searchsorted(
                self._device(table), 1.0, right=True))), 6)
        # A query taken in the width the table carries, not the one the host
        # number happens to have.
        narrow = np.array([-2.5, 0.0, 1.5, 100.0], dtype=np.float16)
        self.assertEqual(
            int(self._back(tensorplay.searchsorted(self._device(narrow), 1.6))), 3)
        counts = np.array([-5, -1, 0, 3, 100], dtype=np.int8)
        self.assertEqual(
            int(self._back(tensorplay.searchsorted(self._device(counts), 4,
                                                   out_int32=True))), 4)

    def test_a_not_a_number_walks_past_the_end(self):
        table = np.array([-3.0, -1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        queries = np.array([np.nan, -np.inf, np.inf, 0.0], dtype=np.float32)
        got = self._back(tensorplay.searchsorted(self._device(table), self._device(queries)))
        np.testing.assert_array_equal(got, [5, 0, 5, 2])

    def test_a_view_is_searched_as_it_shows(self):
        state = np.random.RandomState(6)
        block = np.sort(state.randn(3, 8).astype(np.float32), axis=-1)
        wide = np.stack([block, block + 4.0], axis=1)  # one table per row, two columns
        tables = self._device(wide)[:, 0, :]
        queries = self._device(block)[:, 1::2]
        np.testing.assert_array_equal(
            self._back(tensorplay.searchsorted(tables, queries)),
            self._model(wide[:, 0, :], np.ascontiguousarray(block[:, 1::2])))
        # A sequence whose rows are strided apart still answers its own row.
        shifted = self._device(wide)[:, 1, :]
        np.testing.assert_array_equal(
            self._back(tensorplay.searchsorted(shifted, self._device(block[:, ::3]))),
            self._model(wide[:, 1, :], np.ascontiguousarray(block[:, ::3])))

    def test_more_queries_than_the_grid_holds(self):
        table = np.sort(np.random.RandomState(7).randn(4096).astype(np.float32))
        queries = np.random.RandomState(8).randn(200_003).astype(np.float32)
        got = tensorplay.searchsorted(self._device(table), self._device(queries))
        self.assertEqual(tuple(got.shape), queries.shape)
        np.testing.assert_array_equal(
            self._back(got), self._model(table, queries).astype(got.cpu().numpy().dtype))

    def test_an_empty_side_answers_without_a_walk(self):
        table = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        empty = tensorplay.searchsorted(
            self._device(table), self._device(np.zeros((0,), np.float32)))
        self.assertEqual(tuple(empty.shape), (0,))
        self.assertEqual(empty.dtype, tensorplay.int64)

        # Every query takes the front of a table with no entries.
        for out_int32 in (False, True):
            for shape in ((5,), (2, 3)):
                queries = np.zeros(shape, np.float32)
                got = tensorplay.searchsorted(
                    self._device(np.zeros((0,), np.float32)),
                    self._device(queries),
                    out_int32=out_int32)
                self.assertEqual(
                    got.dtype, tensorplay.int32 if out_int32 else tensorplay.int64)
                np.testing.assert_array_equal(
                    self._back(got), np.zeros(shape, np.int64))
        rows = tensorplay.searchsorted(
            self._device(np.zeros((4, 0), np.float32)),
            self._device(np.zeros((4, 6), np.float32)))
        np.testing.assert_array_equal(self._back(rows), np.zeros((4, 6), np.int64))

    def test_bucketize_reads_the_table_as_edges(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([-4.0, 0.0, 0.5, 1.0, 2.5, 3.0, 100.0], dtype=np.float32)
        for right in (False, True):
            got = tensorplay.bucketize(
                self._device(values), self._device(edges), right=right)
            self.assertEqual(got.dtype, tensorplay.int64)
            np.testing.assert_array_equal(
                self._back(got), self._model(edges, values, right))
        for out_int32 in (False, True):
            got = tensorplay.bucketize(
                self._device(values), self._device(edges), out_int32=out_int32)
            self.assertEqual(
                got.dtype, tensorplay.int32 if out_int32 else tensorplay.int64)
        # A block bucketed by one edge list, and a lone number.
        block = np.random.RandomState(9).randn(3, 5).astype(np.float32)
        np.testing.assert_array_equal(
            self._back(tensorplay.bucketize(self._device(block), self._device(edges))),
            self._model(edges, block))
        self.assertEqual(int(self._back(tensorplay.bucketize(2.5, self._device(edges)))), 3)

    def test_the_method_and_the_function_answer_alike(self):
        table = np.sort(np.random.RandomState(10).randn(32).astype(np.float32))
        queries = np.random.RandomState(11).randn(7).astype(np.float32)
        device_table = self._device(table)
        device_queries = self._device(queries)
        np.testing.assert_array_equal(
            self._back(device_queries.searchsorted(device_table)),
            self._back(tensorplay.searchsorted(device_table, device_queries)))

    def test_outside_the_supported_surface_refuses(self):
        table = np.array([-3.0, -1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        device = self._device(table)
        wide = self._device(np.round(table * 1000).astype(np.int64))
        sorter = tensorplay.from_numpy(np.arange(len(table), dtype=np.int64))
        with self.assertRaisesRegex(RuntimeError, "sorter"):
            tensorplay.searchsorted(device, device, sorter=sorter)
        with self.assertRaisesRegex(TypeError, "single precision"):
            tensorplay.searchsorted(wide, wide)
        with self.assertRaisesRegex(RuntimeError, "positive"):
            tensorplay.searchsorted(
                tensorplay.from_numpy(np.array(1.0, np.float32)).to("vulkan"),
                device)
        with self.assertRaisesRegex(RuntimeError, "1 dimension"):
            tensorplay.searchsorted(self._device(table.reshape(1, 5)), device)
        with self.assertRaisesRegex(RuntimeError, "opposites"):
            tensorplay.searchsorted(device, device, right=True, side="left")
        with self.assertRaisesRegex(RuntimeError, "side can only be"):
            tensorplay.searchsorted(device, device, side="sideways")
        with self.assertRaisesRegex(RuntimeError, "1 dimension"):
            tensorplay.bucketize(device, self._device(table.reshape(1, 5)))


class TestVulkanMultinomial(unittest.TestCase):
    # A draw answers by ranking: every category gets an independent Exp(1)
    # wait, is scored by weight over that wait, and the highest scores are the
    # categories taken.

    @staticmethod
    def _device(array):
        return tensorplay.from_numpy(np.asarray(array, np.float32)).to("vulkan")

    @staticmethod
    def _back(tensor):
        return tensor.cpu().numpy()

    def test_the_unit_rate_shows_the_right_shape(self):
        tensorplay.manual_seed(7)
        waits = tensorplay.zeros((64, 256), device="vulkan")
        tensorplay.exponential_(waits, 1.0)
        drawn = self._back(waits)
        self.assertTrue(bool(np.all(drawn > 0.0)))
        self.assertTrue(bool(np.all(np.isfinite(drawn))))
        self.assertAlmostEqual(float(drawn.mean()), 1.0, delta=0.1)
        self.assertAlmostEqual(float(drawn.var()), 1.0, delta=0.3)

    def test_a_rate_stretches_the_wait(self):
        tensorplay.manual_seed(11)
        for rate, expected in ((0.5, 2.0), (4.0, 0.25)):
            waits = tensorplay.zeros((32, 32, 32), device="vulkan")
            tensorplay.exponential_(waits, rate)
            self.assertAlmostEqual(
                float(self._back(waits).mean()), expected, delta=expected / 4)

    def test_the_fill_reaches_a_long_payload(self):
        tensorplay.manual_seed(13)
        waits = tensorplay.zeros((100000,), device="vulkan")
        tensorplay.exponential_(waits, 1.0)
        drawn = self._back(waits)
        self.assertGreater(float(drawn.min()), 0.0)
        self.assertAlmostEqual(float(drawn.mean()), 1.0, delta=0.05)

    def test_nothing_to_fill_answers_without_a_draw(self):
        waits = tensorplay.zeros((0,), device="vulkan")
        tensorplay.exponential_(waits, 1.0)
        self.assertEqual(tuple(waits.shape), (0,))

    def test_a_seeded_wait_answers_twice(self):
        tensorplay.manual_seed(17)
        first = self._back(
            tensorplay.exponential_(tensorplay.zeros((16, 16), device="vulkan"), 1.0))
        tensorplay.manual_seed(17)
        second = self._back(
            tensorplay.exponential_(tensorplay.zeros((16, 16), device="vulkan"), 1.0))
        np.testing.assert_array_equal(first, second)

    def test_the_draw_takes_each_category_once(self):
        tensorplay.manual_seed(19)
        row = self._device(np.ones(8))
        drawn = self._back(tensorplay.multinomial(row, 3, replacement=False))
        self.assertEqual(len(set(drawn.tolist())), 3)
        whole = self._back(tensorplay.multinomial(row, 8, replacement=False))
        self.assertEqual(sorted(whole.tolist()), list(range(8)))

    def test_the_draw_follows_the_weights(self):
        tensorplay.manual_seed(23)
        weights = np.array([0.5, 0.25, 0.125, 0.125], np.float32)
        table = self._device(np.tile(weights, (4000, 1)))
        for replacement in (False, True):
            drawn = self._back(
                tensorplay.multinomial(table, 1, replacement=replacement))
            counted = np.bincount(drawn.ravel(), minlength=4) / drawn.shape[0]
            np.testing.assert_allclose(counted, weights, atol=0.04)

    def test_an_inclusion_rate_of_its_own(self):
        tensorplay.manual_seed(29)
        # Three of a fair row of eight are taken, so every category is drawn
        # three eighths of the time.
        table = self._device(np.ones((2000, 8)) / 8.0)
        batch = self._back(tensorplay.multinomial(table, 3, replacement=False))
        self.assertEqual(batch.shape, (2000, 3))
        for row in batch:
            self.assertEqual(len(set(row.tolist())), 3)
        counted = np.bincount(batch.ravel(), minlength=8) / batch.shape[0]
        np.testing.assert_allclose(counted, 3 / 8, atol=0.04)

    def test_a_batch_draws_its_own_row(self):
        tensorplay.manual_seed(31)
        rows = self._device([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        drawn = self._back(tensorplay.multinomial(rows, 1, replacement=False))
        self.assertEqual(drawn.tolist(), [[0], [1], [2]])

    def test_a_weightless_category_never_wins(self):
        tensorplay.manual_seed(37)
        drawn = self._back(
            tensorplay.multinomial(
                self._device([1.0, 1.0, 0.0, 0.0]), 2, replacement=False))
        self.assertEqual(set(drawn.tolist()), {0, 1})

    def test_a_long_row_is_ranked_too(self):
        tensorplay.manual_seed(41)
        vocab = tensorplay.ones((50257,), device="vulkan") / 50257.0
        drawn = self._back(tensorplay.multinomial(vocab, 5, replacement=False))
        self.assertEqual(len(set(drawn.tolist())), 5)
        self.assertGreaterEqual(int(drawn.min()), 0)
        self.assertLess(int(drawn.max()), 50257)
        replaced = self._back(tensorplay.multinomial(vocab, 5, replacement=True))
        self.assertEqual(replaced.shape, (5,))

    def test_the_answer_counts_categories_in_the_wide_code(self):
        tensorplay.manual_seed(43)
        row = self._device(np.ones(6))
        drawn = tensorplay.multinomial(row, 2, replacement=False)
        self.assertEqual(drawn.dtype, tensorplay.int64)
        self.assertEqual(tuple(drawn.shape), (2,))
        batch = tensorplay.multinomial(self._device(np.ones((4, 6))), 2)
        self.assertEqual(batch.dtype, tensorplay.int64)
        self.assertEqual(tuple(batch.shape), (4, 2))
        self.assertEqual(tuple(tensorplay.multinomial(row, 0).shape), (0,))

    def test_a_seeded_draw_answers_twice(self):
        table = self._device(np.ones((64, 8)) / 8.0)
        tensorplay.manual_seed(47)
        first = self._back(tensorplay.multinomial(table, 3, replacement=False))
        tensorplay.manual_seed(47)
        second = self._back(tensorplay.multinomial(table, 3, replacement=False))
        np.testing.assert_array_equal(first, second)
        tensorplay.manual_seed(53)
        other = self._back(tensorplay.multinomial(table, 3, replacement=False))
        self.assertFalse(bool((first == other).all()))

    def test_outside_the_supported_surface_refuses(self):
        row = self._device(np.ones(8))
        with self.assertRaisesRegex(RuntimeError, "without replacement"):
            tensorplay.multinomial(row, 9, replacement=False)
        with self.assertRaisesRegex(RuntimeError, "floating"):
            tensorplay.multinomial(
                tensorplay.from_numpy(np.arange(8, dtype=np.int32)).to("vulkan"), 1)
        with self.assertRaisesRegex(RuntimeError, "Float32"):
            tensorplay.exponential_(
                tensorplay.zeros((4, 4), device="vulkan", dtype=tensorplay.int32),
                1.0)
        with self.assertRaisesRegex(RuntimeError, "lambda > 0.0"):
            tensorplay.exponential_(tensorplay.zeros((4, 4), device="vulkan"), 0.0)


class TestVulkanLikelihoodLoss(unittest.TestCase):
    # A likelihood loss reads one class per row: the class its target names, and
    # the weight that class carries.  Nothing accumulates inside a row, so the
    # totals a reduction asks for are a sum of the row answers away, and every
    # gradient lands on the single class the row named.

    @staticmethod
    def _device(array, dtype=np.float32):
        return tensorplay.from_numpy(np.asarray(array, dtype)).to("vulkan")

    @staticmethod
    def _codes(array, dtype=np.int64):
        return tensorplay.from_numpy(np.asarray(array, dtype)).to("vulkan")

    @staticmethod
    def _back(tensor):
        return tensor.cpu().numpy()

    @staticmethod
    def _tolerance(dtype):
        return (6e-3, 6e-3) if dtype == np.float16 else (1e-4, 1e-5)

    @staticmethod
    def _rows(logits, codes, weight, ignore, dtype=np.float32):
        # The per-row answers and the weight total they add up to.  A code that
        # names no class, and the code asked to skip, carry nothing.  The plane
        # is read as the type it is held in, so a narrow table answers from the
        # values that type can keep.
        table = np.atleast_2d(
            np.asarray(logits, dtype).astype(np.float64))
        wanted = np.atleast_1d(np.asarray(codes))
        live = (wanted != ignore) & (wanted >= 0) & (wanted < table.shape[1])
        scale = np.where(live, 1.0, 0.0)
        if weight is not None:
            for i, code in enumerate(wanted):
                if live[i]:
                    scale[i] = weight[code]
        losses = np.zeros(wanted.size, np.float64)
        for i, code in enumerate(wanted):
            if live[i]:
                losses[i] = -table[i, code] * scale[i]
        return losses, scale, live

    def _checked(self, logits, codes, weight, reduction, ignore, grad,
                 dtype=np.float32):
        losses, scale, live = self._rows(logits, codes, weight, ignore, dtype)
        total = float(scale.sum())
        if reduction == "none" and np.asarray(codes).ndim > 0:
            # A batch of unreduced rows reports no normalizer.
            total = 0.0
        rows = np.atleast_2d(np.asarray(logits, dtype).astype(np.float64))
        rtol, atol = self._tolerance(dtype)
        want_loss = losses
        if reduction == "mean":
            want_loss = losses.sum() / total if total > 0 else float("nan")
        elif reduction == "sum":
            want_loss = losses.sum()
        grad_in = np.asarray(grad, dtype).astype(np.float64)
        want_grad = np.zeros_like(rows)
        for i, code in enumerate(np.atleast_1d(np.asarray(codes))):
            if not live[i]:
                continue
            carried = grad_in[i] if reduction == "none" else float(grad_in)
            if reduction == "mean" and total > 0:
                carried = carried / total
            want_grad[i, code] = -scale[i] * carried

        x = self._device(logits, dtype)
        t = self._codes(codes)
        w = None if weight is None else self._device(weight, dtype)
        out, reported = tensorplay._C.nll_loss(
            x, t, w, {"none": 0, "mean": 1, "sum": 2}[reduction], ignore)
        answered = (tensorplay.float16 if dtype == np.float16
                    else tensorplay.float32)
        self.assertEqual(out.dtype, answered)
        self.assertEqual(reported.dtype, answered)
        np.testing.assert_allclose(
            self._back(out), want_loss, rtol=rtol, atol=atol, equal_nan=True)
        self.assertAlmostEqual(
            float(self._back(reported)), total,
            delta=max(atol, abs(total) * rtol))
        back = tensorplay._C.nll_loss_backward(
            self._device(grad, dtype), x, t, w,
            {"none": 0, "mean": 1, "sum": 2}[reduction], ignore, reported)
        self.assertEqual(back.dtype, answered)
        np.testing.assert_allclose(
            self._back(back), want_grad, rtol=rtol, atol=atol)
        return out, reported, back

    def test_a_row_answers_with_the_class_it_names(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        self._checked(logits, [0, 3, 1, 4], None, "mean", -100, np.float32(1.0))
        self._checked(logits, [0, 3, 1, 4], None, "sum", -100, np.float32(1.0))

    def test_unreduced_rows_keep_their_own_loss(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        out, reported, _ = self._checked(
            logits, [0, 3, 1, 4], None, "none", -100,
            np.array([0.2, 0.4, 0.6, 0.8], np.float32))
        self.assertEqual(tuple(out.shape), (4,))
        self.assertEqual(float(self._back(reported)), 0.0)

    def test_a_skipped_row_carries_nothing(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        losses, _, _ = self._rows(logits, [0, -100, 1, -100], None, -100)
        self.assertEqual(float(losses[1]), 0.0)
        self.assertEqual(float(losses[3]), 0.0)
        self._checked(logits, [0, -100, 1, -100], None, "mean", -100, np.float32(1.0))
        # A skip may be named by any code, not just the usual one.
        self._checked(logits, [0, 3, 1, 4], None, "mean", 3, np.float32(1.0))

    def test_class_weights_scale_both_loss_and_total(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        weight = np.array([1.5, 0.5, 2.0, 0.25, 1.0], np.float32)
        for reduction in ("none", "mean", "sum"):
            grad = (np.arange(4, dtype=np.float32) + 1.0
                    if reduction == "none" else np.float32(1.0))
            self._checked(logits, [0, 3, 1, -100], weight, reduction, -100, grad)

    def test_a_code_naming_no_class_marks_nothing(self):
        # The row lookups of this backend answer a request outside their rows
        # with nothing, so no bounds check has to travel back to the host.
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        out, reported, back = self._checked(
            logits, [0, 9, -3, 4], None, "sum", -100, np.float32(1.0))
        self.assertAlmostEqual(
            float(self._back(out)),
            float(-logits[0, 0] - logits[3, 4]), places=4)
        self.assertAlmostEqual(float(self._back(reported)), 2.0, places=4)
        np.testing.assert_array_equal(self._back(back)[1], np.zeros(5))
        np.testing.assert_array_equal(self._back(back)[2], np.zeros(5))

    def test_a_wide_negative_code_is_not_the_skipped_one(self):
        # A count wider than a word is matched word against word: only the pair
        # the skipped code is stored in skips.
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        _, reported, _ = self._checked(
            logits, [0, -100 - 2 ** 32, 1, 4], None, "sum", -100, np.float32(1.0))
        self.assertAlmostEqual(float(self._back(reported)), 3.0, places=4)

    def test_a_narrow_target_is_read_as_a_count(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        x = self._device(logits)
        losses, _, _ = self._rows(logits, [0, 3, 1, 4], None, -100)
        out, _ = tensorplay._C.nll_loss(
            x, self._codes([0, 3, 1, 4], np.int32), None, 2, -100)
        self.assertAlmostEqual(
            float(self._back(out)), float(losses.sum()), places=4)

    def test_a_single_row_reports_the_weight_it_carries(self):
        logits = np.log(np.linspace(0.05, 0.95, 5))
        weight = np.array([1.5, 0.5, 2.0, 0.25, 1.0], np.float32)
        x = self._device(logits)
        t = self._codes(2)
        out, reported = tensorplay._C.nll_loss(x, t, self._device(weight), 1, -100)
        self.assertEqual(tuple(out.shape), ())
        self.assertAlmostEqual(
            float(self._back(out)), float(-logits[2] * weight[2]) / weight[2],
            places=4)
        self.assertAlmostEqual(float(self._back(reported)), float(weight[2]), places=4)
        # Unreduced, the same row keeps its loss and still reports its weight.
        out, reported = tensorplay._C.nll_loss(x, t, None, 0, -100)
        self.assertEqual(tuple(out.shape), ())
        self.assertAlmostEqual(float(self._back(out)), float(-logits[2]), places=4)
        self.assertAlmostEqual(float(self._back(reported)), 1.0, places=4)

    def test_the_averaged_gradient_needs_no_total_from_the_host(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        x = self._device(logits)
        t = self._codes([0, 3, 1, -100])
        grad = self._device(np.float32(1.0))
        with_total = tensorplay._C.nll_loss_backward(
            grad, x, t, None, 1, -100, self._device(np.array(3.0, np.float32)))
        measured = tensorplay._C.nll_loss_backward(grad, x, t, None, 1, -100, None)
        np.testing.assert_allclose(
            self._back(with_total), self._back(measured), rtol=1e-4, atol=1e-5)

    def test_nothing_to_average_answers_with_nothing(self):
        empty = np.zeros((0, 5), np.float32)
        out, reported = tensorplay._C.nll_loss(
            self._device(empty), self._codes([]), None, 1, -100)
        self.assertTrue(bool(np.isnan(float(self._back(out)))))
        self.assertEqual(float(self._back(reported)), 0.0)
        out, reported = tensorplay._C.nll_loss(
            self._device(empty), self._codes([]), None, 2, -100)
        self.assertEqual(float(self._back(out)), 0.0)
        self.assertEqual(float(self._back(reported)), 0.0)
        # Rows that all skip divide nothing by nothing too.
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        self._checked(logits, [-100] * 4, None, "mean", -100, np.float32(1.0))

    def test_a_long_walk_reaches_every_row(self):
        table = np.log(
            np.random.RandomState(5).rand(300, 517).astype(np.float32) + 0.1)
        codes = np.random.RandomState(7).randint(0, 517, size=300).astype(np.int64)
        codes[10] = -100
        codes[299] = -100
        self._checked(table, codes, None, "mean", -100, np.float32(1.7))
        wide = np.log(
            np.random.RandomState(9).rand(9, 4096).astype(np.float32) + 0.1)
        self._checked(
            wide, np.arange(9, dtype=np.int64) * 37, None, "sum", -100,
            np.float32(1.0))

    def test_the_gradient_lands_only_on_the_named_class(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        x = self._device(logits)
        x.requires_grad_(True)
        loss = tensorplay.nn.functional.nll_loss(
            x, self._codes([0, 3, 1, 4]), reduction="mean")
        loss.backward()
        grad = self._back(x.grad)
        # Four rows share one unit of gradient, and the average divides it by
        # the four weights the rows carry.
        self.assertAlmostEqual(float(grad.sum()), -1.0, places=4)
        for row, code in enumerate([0, 3, 1, 4]):
            self.assertAlmostEqual(float(grad[row, code]), -0.25, places=4)
            others = np.delete(grad[row], code)
            np.testing.assert_array_equal(others, np.zeros(4))

    def test_a_narrow_plane_answers_the_rows_it_names(self):
        # A training run keeps its likelihood plane in the narrow float, so the
        # row it reads and the gradient it hands back are both that type.
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        weight = np.array([1.5, 0.5, 2.0, 0.25, 1.0], np.float32)
        for reduction in ("none", "mean", "sum"):
            grad = (np.arange(4, dtype=np.float32) + 1.0
                    if reduction == "none" else np.float32(1.0))
            self._checked(
                logits, [0, 3, 1, 4], None, reduction, -100, grad, np.float16)
            self._checked(
                logits, [0, 3, 1, -100], weight, reduction, -100, grad,
                np.float16)
        self._checked(
            logits, [0, 9, -3, 4], None, "sum", -100, np.float32(1.0),
            np.float16)
        self._checked(
            logits, [-100] * 4, None, "mean", -100, np.float32(1.0),
            np.float16)
        self._checked(
            logits, [0, 3, 1, 4], None, "mean", 3, np.float32(1.0),
            np.float16)

    def test_a_narrow_row_and_a_narrow_target_type(self):
        logits = np.log(np.linspace(0.05, 0.95, 5))
        x = self._device(logits, np.float16)
        losses, _, _ = self._rows(logits, [2], None, -100, np.float16)
        for codes in (self._codes(2), self._codes(2, np.int32)):
            out, reported = tensorplay._C.nll_loss(x, codes, None, 2, -100)
            self.assertEqual(tuple(out.shape), ())
            self.assertEqual(out.dtype, tensorplay.float16)
            self.assertEqual(reported.dtype, tensorplay.float16)
            self.assertAlmostEqual(
                float(self._back(out)), float(losses.sum()), delta=6e-3)
        # Unreduced, the same row keeps its loss and reports its own weight.
        out, reported = tensorplay._C.nll_loss(x, self._codes(2), None, 0, -100)
        self.assertEqual(out.dtype, tensorplay.float16)
        self.assertAlmostEqual(float(self._back(out)), float(-logits[2]),
                               delta=6e-3)
        self.assertAlmostEqual(float(self._back(reported)), 1.0, delta=6e-3)

    def test_a_narrow_plane_with_no_image_form_shares_its_words(self):
        # A dense plane keeps two classes per word, so a walk over it answers a
        # pair at a time.  An odd count leaves the last word one lane short, and
        # that word may hold the one class carrying a gradient.
        for rows, classes, seed in ((8, 40000, 9), (5, 40001, 11)):
            table = np.log(
                np.random.RandomState(seed).rand(rows, classes).astype(
                    np.float32) + 0.1)
            wanted = np.arange(rows, dtype=np.int64) * (classes // rows)
            wanted[-1] = classes - 1
            for dtype in (np.float32, np.float16):
                self._checked(
                    table, wanted, None, "sum", -100, np.float32(1.0), dtype)
                self._checked(
                    table, wanted, np.full(classes, 0.5, np.float32), "sum",
                    -100, np.float32(1.0), dtype)

    def test_the_narrow_gradient_of_a_training_loss(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        x = self._device(logits, np.float16)
        x.requires_grad_(True)
        loss = tensorplay.nn.functional.nll_loss(
            x, self._codes([0, 3, 1, 4]), reduction="mean")
        self.assertEqual(loss.dtype, tensorplay.float16)
        loss.backward()
        self.assertEqual(x.grad.dtype, tensorplay.float16)
        grad = self._back(x.grad).astype(np.float64)
        np.testing.assert_allclose(grad.sum(), -1.0, rtol=6e-3, atol=6e-3)
        for row, code in enumerate([0, 3, 1, 4]):
            self.assertAlmostEqual(float(grad[row, code]), -0.25, delta=6e-3)
            np.testing.assert_array_equal(
                np.delete(grad[row], code), np.zeros(4))

    def test_outside_the_supported_surface_refuses(self):
        logits = np.log(np.linspace(0.05, 0.95, 20).reshape(4, 5))
        x = self._device(logits)
        t = self._codes([0, 3, 1, 4])
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay._C.nll_loss(
                tensorplay.zeros((4, 5), device="vulkan",
                                 dtype=tensorplay.int32),
                t, None, 1, -100)
        with self.assertRaisesRegex(RuntimeError, "Float32 or Float16 gradient"):
            tensorplay._C.nll_loss_backward(
                self._codes([1, 1, 1, 1]), x, t, None, 0, -100, None)
        with self.assertRaisesRegex(RuntimeError, "1d row"):
            tensorplay._C.nll_loss(
                self._device(np.ones((2, 3, 4), np.float32)),
                self._codes([0, 3]), None, 1, -100)
        with self.assertRaisesRegex(RuntimeError, "expected a target of"):
            tensorplay._C.nll_loss(x, self._codes([0, 3]), None, 1, -100)
        with self.assertRaisesRegex(RuntimeError, "cover every class"):
            tensorplay._C.nll_loss(
                x, t, self._device(np.ones(3, np.float32)), 1, -100)
        with self.assertRaisesRegex(RuntimeError, "invalid reduction"):
            tensorplay._C.nll_loss(x, t, None, 5, -100)
        with self.assertRaisesRegex(RuntimeError, "one value to every row"):
            tensorplay._C.nll_loss_backward(
                self._device(np.float32(1.0)), x, t, None, 0, -100, None)


class TestVulkanSliceScatter(unittest.TestCase):
    # Writing a block back into a payload is the slice read backwards: the same
    # bounds decide where the block belongs, and this is the route the gradient
    # of a slice takes.

    @staticmethod
    def _window(shape, dim, start, end, step):
        length = shape[dim]
        first = 0 if start is None else start
        last = length if end is None else end
        if first < 0:
            first += length
        if last < 0:
            last += length
        first = max(0, min(length, first))
        last = max(first, min(length, last))
        return first, last, (last - first + step - 1) // step

    def _scatter(self, self_np, src_np, dim, start, end, step,
                 dtype=tensorplay.float32):
        payload = tensorplay.tensor(self_np, dtype=dtype).to("vulkan")
        block = tensorplay.tensor(src_np, dtype=dtype).to("vulkan")
        out = tensorplay.slice_scatter(payload, block, dim, start, end, step)
        first, last, _ = self._window(self_np.shape, dim, start, end, step)
        want = self_np.copy()
        key = [slice(None)] * self_np.ndim
        key[dim] = np.arange(first, last, step)
        want[tuple(key)] = src_np
        got = out.cpu().numpy()
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(
            got, want, rtol=0,
            atol=2.0e-3 if dtype == tensorplay.float16 else 0.0)
        return got

    def test_a_window_on_either_axis(self):
        rng = np.random.default_rng(3)
        base = (rng.standard_normal((6, 3)) * 3).astype(np.float32)
        self._scatter(base, base[1:3], 0, 1, 3, 1)
        self._scatter(base, base[:, 2:5], 1, 2, 5, 1)
        four = (rng.standard_normal((2, 3, 5, 4))).astype(np.float32)
        self._scatter(four, four[:, :, 2:4], 2, 2, 4, 1)
        # A payload with no axis above its last one keeps its own row order.
        line = (rng.standard_normal((11,))).astype(np.float32)
        self._scatter(line, line[4:8], 0, 4, 8, 1)

    def test_bounds_that_lean_on_the_clamp(self):
        rng = np.random.default_rng(5)
        base = (rng.standard_normal((6, 3))).astype(np.float32)
        self._scatter(base, base[:3], 0, None, 3, 1)
        self._scatter(base, base[3:], 0, 3, None, 1)
        self._scatter(base, base[3:5], 0, -3, -1, 1)
        self._scatter(base, base[3:], 0, 3, 99, 1)
        self._scatter(base, base[:0], 0, -99, 0, 1)
        # A window that closes before it opens writes nothing at all.
        self._scatter(base, np.zeros((0, 3), np.float32), 0, 4, 2, 1)
        self._scatter(base, base, 0, 0, 6, 1)
        self._scatter(
            np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
            0, 0, 0, 1)

    def test_a_step_wider_than_one(self):
        # Every nth row the window keeps is written and the rows between them
        # keep the value they were read with.
        rng = np.random.default_rng(7)
        rows = (rng.standard_normal((9, 3))).astype(np.float32)
        self._scatter(rows, rows[1:8:2], 0, 1, 8, 2)
        cols = (rng.standard_normal((4, 9))).astype(np.float32)
        self._scatter(cols, cols[:, 0:9:3], 1, 0, 9, 3)
        line = (rng.standard_normal((10,))).astype(np.float32)
        self._scatter(line, line[2:9:3], 0, 2, 9, 3)
        four = (rng.standard_normal((2, 3, 8, 4))).astype(np.float32)
        self._scatter(four, four[:, :, 1:7:2], 2, 1, 7, 2)
        self._scatter(rows, rows[0:9:9], 0, 0, 9, 9)

    def test_the_narrow_float_and_a_source_of_another_type(self):
        rng = np.random.default_rng(11)
        base = (rng.standard_normal((6, 4))).astype(np.float32)
        self._scatter(base, base[2:4], 0, 2, 4, 1, dtype=tensorplay.float16)
        # The result carries the dtype the payload was read with, so a source
        # of another is converted before it is written.
        block = (rng.standard_normal((2, 4))).astype(np.float16)
        out = tensorplay.slice_scatter(
            tensorplay.tensor(base).to("vulkan"),
            tensorplay.tensor(block, dtype=tensorplay.float16).to("vulkan"),
            0, 1, 3, 1)
        got = out.cpu().numpy()
        self.assertEqual(got.dtype, np.float32)
        want = base.copy()
        want[1:3] = block.astype(np.float32)
        np.testing.assert_allclose(got, want, rtol=0, atol=2.0e-3)

    def test_a_payload_with_no_image_form(self):
        rng = np.random.default_rng(13)
        big = (rng.standard_normal((40, 4096, 101)) * 0.1).astype(np.float32)
        self._scatter(big, big[5:9], 0, 5, 9, 1)
        self._scatter(big, big[:, 0:40000:4], 1, 0, 40000, 4)

    def test_the_gradient_of_a_slice(self):
        # The gradient of a slice is the slice's own write: the rows the slice
        # took receive it, and the payload keeps zeros everywhere else.
        rng = np.random.default_rng(17)
        for shape, dim, start, end, step in [
                ((7, 5), 0, 2, 5, 1),
                ((5, 8), 1, 1, 6, 1),
                ((6, 4), 0, 3, None, 1),
                ((6, 4), 0, -3, -1, 1),
                ((9, 4), 0, 1, 8, 2),
                ((4, 9, 2), 1, 0, 9, 3),
                ((3, 5, 4), 1, 1, 4, 1),
                ((5, 3), 0, None, None, 1)]:
            base = (rng.standard_normal(shape)).astype(np.float32)
            first, last, kept = self._window(shape, dim, start, end, step)
            block_shape = list(shape)
            block_shape[dim] = kept
            block = (rng.standard_normal(block_shape)).astype(np.float32)
            answers = {}
            for device in ("cpu", "vulkan"):
                a = tensorplay.tensor(base, device=device).detach().requires_grad_(True)
                b = tensorplay.tensor(block, device=device).detach().requires_grad_(True)
                out = tensorplay.slice_scatter(a, b, dim, start, end, step)
                out.backward(tensorplay.ones_like(out))
                answers[device] = (
                    out.detach().cpu().numpy(),
                    a.grad.cpu().numpy(),
                    b.grad.cpu().numpy())
            for name, on_device, on_host in zip(
                    ("result", "grad_self", "grad_src"),
                    answers["cpu"], answers["vulkan"]):
                np.testing.assert_allclose(
                    on_device, on_host, rtol=1.0e-6, atol=1.0e-5,
                    err_msg=f"{shape} dim={dim} step={step} {name}")

    def test_outside_the_supported_surface_refuses(self):
        payload = tensorplay.zeros((4, 3), device="vulkan")
        block = tensorplay.zeros((2, 3), device="vulkan")
        with self.assertRaisesRegex(RuntimeError, "step must be positive"):
            tensorplay.slice_scatter(payload, block, 0, 0, 4, 0)
        with self.assertRaisesRegex(RuntimeError, "step must be positive"):
            tensorplay.slice_scatter(payload, block, 0, 0, 4, -2)
        with self.assertRaisesRegex(RuntimeError, "size equal to the slice"):
            tensorplay.slice_scatter(
                payload, tensorplay.zeros((3, 3), device="vulkan"),
                0, 0, 2, 1)
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.slice_scatter(
                tensorplay.zeros(
                    (4, 3), dtype=tensorplay.int32, device="vulkan"),
                tensorplay.zeros(
                    (2, 3), dtype=tensorplay.int32, device="vulkan"),
                0, 0, 2, 1)
        with self.assertRaisesRegex(RuntimeError, "dim out of range"):
            tensorplay.slice_scatter(payload, block, 2, 0, 2, 1)


class TestVulkanEmbeddingBackward(unittest.TestCase):
    # The gradient of a table lookup: a row answers with the sum of the
    # gradients of every lookup that named it, and a row the list never read
    # keeps a zero.

    @staticmethod
    def _named(codes, grad, rows, padding=-1, scaled=False):
        # The answer the walk owes: one row per code the table carries, a
        # padding entry left out, and a frequency-scaled row divided by how
        # many times the list named it.
        flat = np.asarray(codes).reshape(-1)
        features = tuple(np.asarray(grad).shape)[-1]
        out = np.zeros((rows, features), np.float32)
        source = np.asarray(grad).reshape(flat.size, features).astype(np.float32)
        counts = np.zeros(rows, np.int32)
        for position, code in enumerate(flat):
            if code == padding or code < 0 or code >= rows:
                continue
            out[code] += source[position]
            counts[code] += 1
        if scaled:
            out /= np.maximum(counts, 1)[:, None]
        return out

    def _against_reference(
            self, codes_np, grad_np, rows, dtype=tensorplay.float32,
            padding=-1, scaled=False, atol=0.0):
        device = tensorplay.device("vulkan")
        grad = tensorplay.tensor(grad_np, dtype=dtype).to(device)
        codes = tensorplay.tensor(codes_np).to(device)
        out = tensorplay.embedding_dense_backward(
            grad, codes, rows, padding, scaled)
        got = out.cpu().numpy()
        want = self._named(codes_np, grad_np, rows, padding, scaled)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, np.float32 if dtype == tensorplay.float32
                         else np.float16)
        np.testing.assert_allclose(
            got.astype(np.float32), want, rtol=0,
            atol=atol if atol else (2.0e-3
                                    if dtype == tensorplay.float16 else 1.0e-6))
        return got

    def test_the_rows_a_list_names_add_in_the_order_they_read(self):
        rng = np.random.default_rng(23)
        for codes in ([0, 3, 3, 10, 1, 3], [4, 4, 4, 4], list(range(12)),
                      [11, 0, 11, 5]):
            grad = (rng.standard_normal((len(codes), 6)) * 2).astype(np.float32)
            self._against_reference(
                np.array(codes, np.int64), grad, 12)

    def test_a_padding_row_neither_adds_nor_is_counted(self):
        rng = np.random.default_rng(29)
        codes = np.array([0, 2, 2, 4, 2, 7], np.int64)
        grad = (rng.standard_normal((6, 5)) * 2).astype(np.float32)
        for scaled in (False, True):
            got = self._against_reference(
                codes, grad, 9, padding=2, scaled=scaled)
            np.testing.assert_array_equal(got[2], np.zeros(5, np.float32))
        # A padding entry outside the table changes nothing about the answer.
        self._against_reference(codes, grad, 9, padding=99, scaled=True)

    def test_a_proportion_scaled_by_frequency_divides_each_row(self):
        rng = np.random.default_rng(31)
        codes = np.array([0, 0, 0, 5, 5, 7, 3], np.int64)
        grad = (rng.standard_normal((7, 8)) * 2).astype(np.float32)
        self._against_reference(codes, grad, 20, scaled=True)
        # A row the list never named divides by one, so its zeros stay there.
        got = self._against_reference(codes, grad, 20, scaled=True)
        np.testing.assert_array_equal(
            got[[1, 2, 4, 6, 8]], np.zeros((5, 8), np.float32))

    def test_a_list_of_any_rank_and_element_type(self):
        rng = np.random.default_rng(37)
        for shape in ((4, 3), (2, 3, 4)):
            codes = rng.integers(0, 20, size=shape).astype(np.int64)
            grad = (rng.standard_normal(shape + (8,)) * 2).astype(np.float32)
            self._against_reference(codes, grad, 20)
            self._against_reference(codes, grad, 20, scaled=True)
        self._against_reference(
            rng.integers(0, 20, size=(9,)).astype(np.int32),
            (rng.standard_normal((9, 8)) * 2).astype(np.float32), 20,
            scaled=True)

    def test_the_narrow_float_of_a_scaled_row(self):
        rng = np.random.default_rng(41)
        codes = np.array([1, 1, 1, 6, 6, 2], np.int64)
        grad = (rng.standard_normal((6, 8)) * 2).astype(np.float16)
        self._against_reference(
            codes, grad, 16, dtype=tensorplay.float16, scaled=True)
        self._against_reference(codes, grad, 16, dtype=tensorplay.float16)

    def test_a_code_the_table_does_not_carry_answers_no_row(self):
        # The lookup answers no row for a code outside it and the walk adds no
        # row for the same code: both ends of a lookup agree on what a code
        # with no row means.
        rng = np.random.default_rng(43)
        codes = np.array([-1, 14, 2, 0, -7, 14, 5], np.int64)
        grad = (rng.standard_normal((7, 4)) * 2).astype(np.float32)
        device = tensorplay.device("vulkan")
        got = tensorplay.embedding_dense_backward(
            tensorplay.tensor(grad).to(device),
            tensorplay.tensor(codes).to(device), 12, -1, False).cpu().numpy()
        want = np.zeros((12, 4), np.float32)
        for position, code in enumerate(codes):
            if 0 <= code < 12:
                want[code] += grad[position]
        np.testing.assert_allclose(got, want, rtol=0, atol=1.0e-6)
        np.testing.assert_array_equal(got[[1, 3, 4, 6, 7]], np.zeros((5, 4)))

    def test_a_table_with_no_image_form(self):
        rng = np.random.default_rng(47)
        codes = np.array([0, 7, 7, 19999, 1234, 7], np.int64)
        grad = (rng.standard_normal((6, 300)) * 0.2).astype(np.float32)
        self._against_reference(codes, grad, 20000, atol=1.0e-6)
        self._against_reference(codes, grad, 20000, scaled=True, atol=1.0e-6)

    def test_an_empty_list_or_an_empty_table(self):
        empty = tensorplay.zeros((0, 5), device="vulkan")
        codes = tensorplay.tensor(np.zeros((0,), np.int64)).to("vulkan")
        out = tensorplay.embedding_dense_backward(empty, codes, 7, -1, False)
        self.assertEqual(tuple(out.shape), (7, 5))
        np.testing.assert_array_equal(out.cpu().numpy(), np.zeros((7, 5)))
        # A table with no row holds nothing for a list to name.
        out = tensorplay.embedding_dense_backward(
            tensorplay.zeros((3, 4), device="vulkan"),
            tensorplay.tensor(np.array([0, 1, 2], np.int64)).to("vulkan"),
            0, -1, True)
        self.assertEqual(tuple(out.shape), (0, 4))

    def test_the_gradient_of_a_lookup_layer(self):
        import tensorplay.nn.functional as Fn
        rng = np.random.default_rng(53)
        table = (rng.standard_normal((24, 6)) * 0.5).astype(np.float32)
        codes = rng.integers(0, 24, size=(3, 5)).astype(np.int64)
        for padding, scaled in ((None, False), (3, False), (None, True),
                                (3, True)):
            answers = {}
            for device in ("cpu", "vulkan"):
                weight = tensorplay.tensor(table, device=device)\
                    .detach().requires_grad_(True)
                out = Fn.embedding(
                    tensorplay.tensor(codes, device=device), weight,
                    padding_idx=padding, scale_grad_by_freq=scaled)
                out.backward(tensorplay.ones_like(out))
                answers[device] = (
                    out.detach().cpu().numpy(), weight.grad.cpu().numpy())
            np.testing.assert_allclose(
                answers["cpu"][0], answers["vulkan"][0], rtol=0, atol=0)
            np.testing.assert_allclose(
                answers["cpu"][1], answers["vulkan"][1], rtol=1.0e-6,
                atol=1.0e-5, err_msg=f"padding={padding} scaled={scaled}")

    def test_a_narrow_float_scales_the_same_way_on_both_devices(self):
        # The frequency proportion belongs to the narrow float too: a row read
        # three times divides by three whichever device summed it.
        rng = np.random.default_rng(59)
        table = (rng.standard_normal((16, 6)) * 0.5).astype(np.float16)
        codes = np.array([1, 9, 1, 4, 1, 9, 15], np.int64)
        grad = (rng.standard_normal((7, 6)) * 2).astype(np.float16)
        answers = {}
        for device in ("cpu", "vulkan"):
            weight = tensorplay.tensor(table, device=device)\
                .detach().requires_grad_(True)
            tensorplay.nn.functional.embedding(
                tensorplay.tensor(codes, device=device), weight,
                scale_grad_by_freq=True).backward(
                    tensorplay.tensor(grad, device=device))
            answers[device] = weight.grad.cpu().numpy().astype(np.float32)
        np.testing.assert_allclose(answers["cpu"], answers["vulkan"],
                                   rtol=0, atol=2.0e-3)
        want = self._named(codes, grad, 16, scaled=True)
        np.testing.assert_allclose(answers["cpu"], want, rtol=0, atol=2.0e-3)

    def test_outside_the_supported_surface_refuses(self):
        device = tensorplay.device("vulkan")
        grad = tensorplay.zeros((4, 6), device=device)
        codes = tensorplay.tensor(np.array([0, 1, 2, 3], np.int64)).to(device)
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.embedding_dense_backward(
                tensorplay.zeros(
                    (4, 6), dtype=tensorplay.int32, device=device),
                codes, 8, -1, False)
        with self.assertRaisesRegex(RuntimeError, "Int64 or Int32"):
            tensorplay.embedding_dense_backward(
                grad, tensorplay.tensor(
                    np.array([0, 1, 2, 3], np.float32)).to(device),
                8, -1, False)
        # A gradient that does not carry one row per entry of the list.
        with self.assertRaisesRegex(RuntimeError, "one row per"):
            tensorplay.embedding_dense_backward(
                tensorplay.zeros((5, 6), device=device), codes, 8, -1, False)
        with self.assertRaisesRegex(RuntimeError, "negative count of rows"):
            tensorplay.embedding_dense_backward(grad, codes, -3, -1, False)
        with self.assertRaisesRegex(RuntimeError, "on the device"):
            tensorplay.embedding_dense_backward(
                grad, tensorplay.tensor(np.array([0, 1, 2, 3], np.int64)),
                8, -1, False)


class TestVulkanProductStorageClasses(unittest.TestCase):
    # A product meets three operands, each written by whichever route had to
    # carry it: a weight whose axes pass the device's image limit lies in words,
    # a batch the device laid out itself has no other place to be, and a small
    # addend keeps the form its own shape answers.  Kernels walk texels or words
    # according to what they were written to read, so a mismatch crosses the
    # operand the walk touches least instead of ending the product.

    @staticmethod
    def _uploaded(array, dtype=np.float32):
        # A payload that arrives from the host keeps the image form.
        return tensorplay.from_numpy(np.asarray(array, dtype)).to("vulkan")

    @staticmethod
    def _built(array, dtype=np.float32):
        # A payload the device lays out itself keeps its elements in words.
        return tensorplay.tensor(np.asarray(array, dtype), device="vulkan")

    @staticmethod
    def _uses_dense_storage(payload):
        # Only single precision has both a texture and a linear form of lerp, so
        # a lerp that reports the missing image marks a payload that keeps its
        # elements in words.
        import tensorplay.functional as F
        try:
            F.lerp(payload, payload, 0.5)
            return False
        except RuntimeError:
            return True

    @classmethod
    def _rows_without_an_image_form(cls, width):
        # The smallest row count whose plane the device cannot hold in an image,
        # which is where a product over such rows has to write words.
        rows = 4096
        while rows <= 262144:
            probe = tensorplay.from_numpy(
                np.zeros((rows, width), np.float32)).to("vulkan")
            if cls._uses_dense_storage(probe):
                return rows
            rows *= 2
        return None

    @staticmethod
    def _relerr(got, want):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        denom = max(float(np.abs(want).max()), 1.0)
        return float(np.abs(got - want).max()) / denom

    def _back(self, tensor):
        return tensor.detach().cpu().numpy().astype(np.float64)

    def test_a_product_meets_operands_of_either_storage(self):
        from tensorplay.nn.functional import linear

        rng = np.random.default_rng(3)
        M, N, K = 3, 5, 6
        host_x = rng.standard_normal((M, K)).astype(np.float32)
        host_w = rng.standard_normal((N, K)).astype(np.float32)
        host_wt = np.ascontiguousarray(host_w.T)
        for dtype, tol in ((np.float32, 1e-4), (np.float16, 6e-3)):
            x = host_x.astype(dtype)
            w = host_w.astype(dtype)
            wt = host_wt.astype(dtype)
            # The answer is walked in the wide plane from the values the narrow
            # one actually carries, so the width only costs rounding.
            want_linear = x.astype(np.float32) @ w.astype(np.float32).T
            want_mm = x.astype(np.float32) @ wt.astype(np.float32)
            for act_name, make_act in (("word", self._built),
                                       ("image", self._uploaded)):
                probe = make_act(x, np.float32)
                self.assertIs(
                    self._uses_dense_storage(probe), make_act == self._built,
                    f"a {act_name} activation was not the storage its builder "
                    "answers")
                for wt_name, make_wt in (("word", self._built),
                                         ("image", self._uploaded)):
                    activation = make_act(x, dtype)
                    answers = (
                        linear(activation, make_wt(w, dtype)),
                        tensorplay.mm(activation, make_wt(wt, dtype)))
                    for answer, want in zip(answers, (want_linear, want_mm)):
                        self.assertTrue(
                            answer.device.is_vulkan(),
                            f"a {act_name} x {wt_name} product left the device")
                        self.assertLessEqual(
                            self._relerr(self._back(answer), want), tol,
                            f"a {act_name} activation against a {wt_name} "
                            f"weight missed the product ({dtype.__name__})")

    def test_an_addend_of_either_storage_feeds_the_product(self):
        from tensorplay.nn.functional import linear

        rng = np.random.default_rng(5)
        M, N, K = 3, 5, 6
        host_x = rng.standard_normal((M, K)).astype(np.float32)
        host_w = rng.standard_normal((N, K)).astype(np.float32)
        host_b = rng.standard_normal((N,)).astype(np.float32)
        for dtype, tol in ((np.float32, 1e-4), (np.float16, 6e-3)):
            x = host_x.astype(dtype)
            w = host_w.astype(dtype)
            b = host_b.astype(dtype)
            want = (x.astype(np.float32) @ w.astype(np.float32).T
                    + b.astype(np.float32))
            for act_name, make_act in (("word", self._built),
                                       ("image", self._uploaded)):
                for wt_name, make_wt in (("word", self._built),
                                         ("image", self._uploaded)):
                    for bias_name, make_bias in (("word", self._built),
                                                 ("image", self._uploaded)):
                        answer = linear(
                            make_act(x, dtype), make_wt(w, dtype),
                            make_bias(b, dtype))
                        self.assertLessEqual(
                            self._relerr(self._back(answer), want), tol,
                            f"a {bias_name} addend against a {act_name} "
                            f"activation and a {wt_name} weight missed the "
                            f"product ({dtype.__name__})")

    def test_a_product_wide_enough_for_words_reads_an_image_addend(self):
        # Rows past the device's image axes leave a product no image to write,
        # so its result lies in words while the addend broadcast over those rows
        # still has a texture of its own.
        from tensorplay.nn.functional import linear

        rows = self._rows_without_an_image_form(8)
        if rows is None:
            self.skipTest("no row count here is beyond the image axes")
        width, features = 8, 6
        rng = np.random.default_rng(9)
        host_x = rng.standard_normal((rows, features)).astype(np.float32)
        host_w = rng.standard_normal((width, features)).astype(np.float32)
        host_b = rng.standard_normal((width,)).astype(np.float32)
        want = host_x @ host_w.T + host_b

        answer = linear(
            self._built(host_x), self._built(host_w), self._uploaded(host_b))
        self.assertTrue(
            self._uses_dense_storage(answer),
            "a product over this many rows was expected to write words")
        self.assertLessEqual(
            self._relerr(self._back(answer), want), 1e-4,
            "a word-shaped product missed the image addend")

    def test_a_model_reads_a_batch_built_on_the_device(self):
        from tensorplay.nn import Linear

        tensorplay.manual_seed(7)
        rng = np.random.default_rng(13)
        M, features, units = 4, 6, 5
        host_x = rng.standard_normal((M, features)).astype(np.float32)
        model = Linear(features, units)
        host_w = model.weight.detach().cpu().numpy().astype(np.float32)
        host_b = model.bias.detach().cpu().numpy().astype(np.float32)
        model.to("vulkan")
        self.assertFalse(
            self._uses_dense_storage(model.weight.detach()),
            "an uploaded layer was expected to keep its image")

        feed = self._built(host_x).detach().requires_grad_(True)
        answer = model(feed)
        self.assertTrue(
            answer.device.is_vulkan(),
            "a layer answer for a device-built batch left the device")
        self.assertLessEqual(
            self._relerr(self._back(answer), host_x @ host_w.T + host_b), 1e-4,
            "a device-built batch through an uploaded layer missed the walk")

        answer.sum().backward()
        self.assertIsNotNone(feed.grad, "the batch kept no gradient")
        moved = feed.grad.detach().cpu().numpy().astype(np.float32)
        self.assertTrue(
            np.isfinite(moved).all(),
            "the gradient a device-built batch was handed went non-finite")
        self.assertLessEqual(
            self._relerr(moved, np.ones((M, units), np.float32) @ host_w),
            1e-3,
            "the gradient of a device-built batch missed the layer")


class TestVulkanSquaredErrorLoss(unittest.TestCase):
    # How far an element sits from the value it is matched with is squared in
    # one walk, so an averaged answer is a reduction away and a gradient is the
    # same distance counted twice, shared by the count it was averaged over.

    @staticmethod
    def _uploaded(array, dtype=np.float32):
        # A payload that arrives from the host keeps the image form.
        return tensorplay.from_numpy(np.asarray(array, dtype)).to("vulkan")

    @staticmethod
    def _built(array, dtype=np.float32):
        # A payload the device lays out itself keeps its elements in words.
        return tensorplay.tensor(np.asarray(array, dtype), device="vulkan")

    @staticmethod
    def _uses_dense_storage(payload):
        # Only single precision has both a texture and a linear form of lerp, so
        # a lerp that reports the missing image marks a payload that keeps its
        # elements in words.  Both squared-error walks are asked for below, and
        # this is what says they were.
        import tensorplay.functional as F
        try:
            F.lerp(payload, payload, 0.5)
            return False
        except RuntimeError:
            return True

    @staticmethod
    def _half(array):
        return np.asarray(array, np.float16).astype(np.float32)

    @staticmethod
    def _back(tensor):
        return tensor.cpu().numpy().astype(np.float64)

    def test_the_squared_error_walks_both_storages(self):
        from tensorplay.nn.functional import mse_loss

        rng = np.random.default_rng(4)
        for shape in ((5, 7), (24,), (3, 5, 2)):
            values = rng.standard_normal(shape).astype(np.float32)
            compared = rng.standard_normal(shape).astype(np.float32)
            squares = (values - compared) ** 2
            for name, make in (("image", self._uploaded),
                               ("word", self._built)):
                first = make(values)
                second = make(compared)
                self.assertIs(
                    self._uses_dense_storage(first), make == self._built,
                    f"a {name} plane was not the storage its builder answers")
                for reduction, want in (
                        ("none", squares),
                        ("mean", float(squares.mean())),
                        ("sum", float(squares.sum()))):
                    got = mse_loss(first, second, reduction)
                    self.assertTrue(
                        got.device.is_vulkan(),
                        f"a {name} {reduction} loss left the device")
                    if reduction == "none":
                        self.assertEqual(
                            list(got.shape), list(shape),
                            f"a {name} unreduced loss changed the plane it walks")
                    np.testing.assert_allclose(
                        self._back(got), np.asarray(want, np.float64),
                        rtol=1e-4, atol=1e-5,
                        err_msg=f"a {name} {reduction} loss missed the squares")

    def test_the_squared_error_reads_a_narrow_plane(self):
        from tensorplay.nn.functional import mse_loss

        rng = np.random.default_rng(9)
        values = self._half(rng.standard_normal((6, 11)))
        compared = self._half(rng.standard_normal((6, 11)))
        squares = (values - compared) ** 2
        for make in (self._uploaded, self._built):
            first = make(values, np.float16)
            second = make(compared, np.float16)
            self.assertEqual(first.dtype, tensorplay.float16,
                             "a narrow plane left its own width")
            for reduction, want in (
                    ("none", squares),
                    ("mean", float(squares.mean())),
                    ("sum", float(squares.sum()))):
                got = mse_loss(first, second, reduction)
                self.assertEqual(
                    got.dtype, tensorplay.float16,
                    f"a narrow {reduction} loss answered wider than its plane")
                np.testing.assert_allclose(
                    self._back(got), np.asarray(want, np.float64),
                    rtol=6e-3, atol=6e-3,
                    err_msg=f"a narrow {reduction} loss missed the squares")

    def test_the_squared_error_gradient_counts_the_distance_twice(self):
        from tensorplay.nn.functional import mse_loss

        rng = np.random.default_rng(13)
        values = rng.standard_normal((6, 9)).astype(np.float32)
        compared = rng.standard_normal((6, 9)).astype(np.float32)
        incoming = rng.standard_normal((6, 9)).astype(np.float32)
        count = values.size
        distance = 2.0 * (values - compared)
        for make in (self._uploaded, self._built):
            for reduction, want in (
                    ("mean", distance * (1.0 / count)),
                    ("sum", distance),
                    ("none", distance * incoming)):
                plane = make(values).detach().requires_grad_(True)
                loss = mse_loss(plane, make(compared), reduction)
                if reduction == "none":
                    loss.backward(make(incoming))
                else:
                    loss.backward()
                self.assertIsNotNone(
                    plane.grad, f"a {reduction} loss handed back no gradient")
                self.assertTrue(
                    plane.grad.device.is_vulkan(),
                    f"a {reduction} gradient left the device")
                np.testing.assert_allclose(
                    self._back(plane.grad), want.astype(np.float64),
                    rtol=1e-4, atol=1e-6,
                    err_msg=f"a {reduction} gradient miscounted the distance")

    def test_a_narrower_target_is_matched_element_by_element(self):
        from tensorplay.nn.functional import mse_loss

        rng = np.random.default_rng(17)
        values = rng.standard_normal((5, 8)).astype(np.float32)
        column = rng.standard_normal((5, 1)).astype(np.float32)
        squares = (values - column) ** 2
        plane = self._uploaded(values)
        want = float(squares.mean())
        np.testing.assert_allclose(
            self._back(mse_loss(plane, self._uploaded(column), "mean")), want,
            rtol=1e-4, atol=1e-5,
            err_msg="a single-column target was not matched across the plane")
        # A target still held by the host is read on the device it compares with.
        host_target = tensorplay.from_numpy(column.copy())
        np.testing.assert_allclose(
            self._back(mse_loss(plane, host_target, "sum")),
            float(squares.sum()), rtol=1e-4, atol=1e-5,
            err_msg="a host target was not carried onto the plane")

    def test_a_model_trains_on_the_squared_error(self):
        from tensorplay.nn import Linear, MSELoss
        from tensorplay.optim import AdamW

        tensorplay.manual_seed(5)
        rng = np.random.default_rng(21)
        model = Linear(24, 8).to(tensorplay.float16).to("vulkan")
        feed = tensorplay.from_numpy(
            self._half(rng.standard_normal((32, 24)))
        ).to("vulkan").to(tensorplay.float16)
        wanted = tensorplay.from_numpy(
            self._half(rng.standard_normal((32, 8)))
        ).to("vulkan").to(tensorplay.float16)
        opt = AdamW(model.parameters(), lr=3.0e-3)
        crit = MSELoss()

        losses = []
        for _ in range(12):
            opt.zero_grad()
            loss = crit(model(feed), wanted)
            losses.append(float(loss))
            loss.backward()
            opt.step()

        self.assertTrue(all(np.isfinite(v) for v in losses),
                        f"losses left the finite range: {losses}")
        self.assertLess(losses[-1], losses[0],
                        f"loss did not fall: {losses[0]} then {losses[-1]}")
        for name, parameter in model.named_parameters():
            self.assertTrue(
                np.isfinite(parameter.detach().cpu().numpy()).all(),
                f"{name} left the finite range")


class TestVulkanNarrowTraining(unittest.TestCase):
    # A walk that answers one parameter at a time keeps the running averages of
    # a narrow parameter one plane wider, because the share a small gradient
    # adds to the second average has no name in the narrow plane.  The device
    # carries half planes only, so the narrow case is measured there.

    def test_averages_of_a_half_parameter_are_kept_wide(self):
        from tensorplay.optim import AdamW

        narrow = tensorplay.float16
        param = tensorplay.nn.Parameter(
            tensorplay.zeros((3, 5), dtype=narrow, device="vulkan"))
        param.grad = tensorplay.full((3, 5), 4.0e-3, dtype=narrow,
                                     device="vulkan")
        opt = AdamW([param], lr=3.0e-4)
        opt.step()

        state = opt.state[param]
        self.assertEqual(state["exp_avg"].dtype, tensorplay.float32,
                         "a half first average kept the narrow plane")
        self.assertEqual(state["exp_avg_sq"].dtype, tensorplay.float32,
                         "a half second average kept the narrow plane")
        self.assertEqual(param.dtype, narrow,
                         "a half parameter left its own plane")

        # The share one step of a 4e-3 gradient adds to the second average is
        # (1 - 0.999) * g * g.  A half plane cannot name it at all: the average
        # would read zero, its root zero, and the step divides by that zero.
        # Wide, the share is answered.
        step = float(param.grad.to(tensorplay.float32)[0, 0].cpu())
        share = (1.0 - 0.999) * step * step
        smallest_half = float(np.nextafter(np.float16(0.0), np.float16(1.0)))
        self.assertLess(share, smallest_half,
                        "the share this test measures is not a case")
        answers = state["exp_avg_sq"].cpu().numpy()
        np.testing.assert_allclose(
            answers, np.full((3, 5), share), rtol=1.0e-3, atol=1.0e-12,
            err_msg="the second average answered the wrong share")

        moved = param.detach().to(tensorplay.float32).cpu().numpy()
        self.assertTrue(np.isfinite(moved).all(), "a half parameter went non-finite")
        self.assertGreater(float(np.max(np.abs(moved))), 0.0,
                           "a half parameter never moved")

    def test_a_narrow_model_trains_on_the_device(self):
        from tensorplay.nn import CrossEntropyLoss, Linear
        from tensorplay.optim import AdamW

        tensorplay.manual_seed(3)
        model = Linear(24, 16).to(tensorplay.float16).to("vulkan")
        opt = AdamW(model.parameters(), lr=3.0e-3)
        crit = CrossEntropyLoss()
        codes = tensorplay.tensor(
            np.random.default_rng(7).integers(0, 16, size=(64,)).astype(
                np.int64), device="vulkan")
        host = np.random.default_rng(11).standard_normal(
            (64, 24)).astype(np.float32)
        feed = tensorplay.tensor(host, device="vulkan").to(
            tensorplay.float16).detach().requires_grad_(False)

        losses = []
        for _ in range(12):
            opt.zero_grad()
            loss = crit(model(feed), codes)
            losses.append(float(loss))
            loss.backward()
            opt.step()

        self.assertTrue(all(np.isfinite(v) for v in losses),
                        f"losses left the finite range: {losses}")
        self.assertLess(losses[-1], losses[0],
                        f"loss did not fall: first {losses[0]}, last {losses[-1]}")
        for name, p in model.named_parameters():
            plane = p.detach().to(tensorplay.float32).cpu().numpy()
            self.assertTrue(np.isfinite(plane).all(),
                            f"{name} left the finite range")

    def test_the_host_walk_keeps_the_narrow_averages(self):
        from tensorplay.optim import AdamW

        for narrow in (tensorplay.float16, tensorplay.bfloat16):
            param = tensorplay.nn.Parameter(
                tensorplay.zeros((3, 5), dtype=narrow))
            param.grad = tensorplay.full((3, 5), 4.0e-3, dtype=narrow)
            opt = AdamW([param], lr=3.0e-4)
            opt.step()
            state = opt.state[param]
            self.assertEqual(state["exp_avg"].dtype, narrow,
                             f"a host {narrow} parameter widened its averages")
            self.assertEqual(state["exp_avg_sq"].dtype, narrow,
                             f"a host {narrow} parameter widened its averages")


class TestVulkanWideArithmetic(unittest.TestCase):
    """A count wider than a word has no texel to sit in, so the arithmetic walks
    over one read and write a pair of words.  These are the sums, differences and
    products a decode loop asks of its token, position and length counters, the
    whole-number formulas that turn, compare or measure such a pair, and the
    carries and borrows that cross the seam where the two words meet."""

    MASK = (1 << 64) - 1

    @classmethod
    def _as_int64(cls, value):
        # A total past every word lands where the counts it added land, which is
        # what a pair of words holds too.
        bits = int(value) & cls.MASK
        return bits - (1 << 64) if bits >= (1 << 63) else bits

    @classmethod
    def _wrapped(cls, values):
        host = np.array(values, dtype=object)
        flat = [cls._as_int64(v) for v in host.ravel().tolist()]
        return np.array(flat, dtype=np.int64).reshape(host.shape)

    def _device(self, values, dtype=tensorplay.int64):
        return tensorplay.from_numpy(self._wrapped(values)).to("vulkan").to(dtype)

    def _checked(self, got, want, label):
        self.assertEqual(got.dtype, tensorplay.int64,
                         f"{label} changed the width of its payload")
        self.assertTrue(got.device.is_vulkan(),
                        f"{label} left the accelerator")
        np.testing.assert_array_equal(
            np.asarray(got.cpu()), self._wrapped(want), err_msg=label)

    def test_a_sum_hands_one_across_the_seam_of_its_pair(self):
        first = [1, (1 << 32) - 1, 1 << 32, 1 << 40, -(1 << 63), (1 << 63) - 1,
                 -1, 0, 1234567890123, -(1 << 32)]
        second = [1, 1, 1 << 32, 1 << 23, -1, 1, -1, 0, 987654321098, 1]
        self._checked(
            self._device(first) + self._device(second),
            [a + b for a, b in zip(first, second)],
            "sum of two counting payloads")
        # A walk that moved only the low word of a pair would keep every value
        # below 2^32 correct, so the same payload doubled at both ends is asked
        # for as well.
        doubled = self._device(first) + self._device(first)
        self._checked(doubled, [2 * a for a in first], "doubled counting payload")

    def test_a_difference_borrows_from_the_word_that_holds_the_sign(self):
        first = [(1 << 32), (1 << 40), -(1 << 63), (1 << 63) - 1, 5, 0, -1]
        second = [1, (1 << 23), 1, 1, 9, 0, (1 << 32)]
        self._checked(
            self._device(first) - self._device(second),
            [a - b for a, b in zip(first, second)],
            "difference of two counting payloads")
        self._checked(
            self._device(first) - self._device(first),
            [0] * len(first), "a count taken from itself")

    def test_a_product_keeps_the_low_bits_of_the_full_walk(self):
        first = [1 << 32, 1 << 31, (1 << 63) - 1, -(1 << 63), -1, 0, 3, 1 << 20]
        second = [1 << 32, 1 << 31, (1 << 63) - 1, -(1 << 63), -1, 7, -4,
                  1 << 43]
        self._checked(
            self._device(first) * self._device(second),
            [a * b for a, b in zip(first, second)],
            "product of two counting payloads")

    def test_a_scaled_sum_folds_the_factor_into_the_operand_slot(self):
        counts = [1, 1 << 32, (1 << 63) - 1, -(1 << 63), -7, 0]
        addends = [3, 1 << 31, 1, 5, 2, 9]
        wide = self._device(counts)
        other = self._device(addends)
        self._checked(wide + other * 3, [c + 3 * a for c, a in
                                        zip(counts, addends)],
                      "count added to a scaled count")
        self._checked(wide - other * 2, [c - 2 * a for c, a in
                                        zip(counts, addends)],
                      "count reduced by a scaled count")

    def test_a_count_meets_a_single_number_in_both_directions(self):
        counts = [1, 1 << 32, (1 << 63) - 1, -(1 << 63), -7, 0]
        wide = self._device(counts)
        self._checked(wide + 5, [c + 5 for c in counts], "count plus a number")
        self._checked(5 + wide, [5 + c for c in counts], "number plus a count")
        self._checked(wide - 9, [c - 9 for c in counts], "count less a number")
        self._checked(wide * 7, [c * 7 for c in counts], "count times a number")
        # A number with a fraction asks for a plane a count has no image form on,
        # so the payload steps up to single precision, which holds the sum.
        widened = wide + 0.5
        self.assertEqual(widened.dtype, tensorplay.float32)
        np.testing.assert_allclose(
            np.asarray(widened.cpu()),
            np.asarray(self._wrapped(counts), np.float32) + np.float32(0.5),
            rtol=1e-6, atol=1.0, err_msg="a fraction refused to widen a count")

    def test_a_count_written_in_place_keeps_its_own_width(self):
        base = [1, 1 << 32, (1 << 63) - 1, -(1 << 63), -7, 0]
        added = [2, 1 << 31, 1, 3, 5, 7]

        target = self._device(base)
        target += self._device(added)
        self._checked(target, [a + b for a, b in zip(base, added)],
                      "in-place sum of counts")

        target = self._device(base)
        target -= self._device(added)
        self._checked(target, [a - b for a, b in zip(base, added)],
                      "in-place difference of counts")

        target = self._device(base)
        target *= self._device(added)
        self._checked(target, [a * b for a, b in zip(base, added)],
                      "in-place product of counts")

        target = self._device(base)
        target += 4
        target *= 2
        target -= 8
        self._checked(target, [(a + 4) * 2 - 8 for a in base],
                      "a counting payload scaled in place")

    def test_a_narrow_count_steps_up_to_the_pair(self):
        narrow = [1, 2, -3, 5]
        wide = [1 << 40, 1, 1, (1 << 63) - 1]
        self._checked(
            self._device(narrow, tensorplay.int32) + self._device(wide),
            [a + b for a, b in zip(narrow, wide)],
            "a narrow count widened to meet its partner")
        self._checked(
            self._device(wide) * self._device(narrow, tensorplay.int32),
            [a * b for a, b in zip(wide, narrow)],
            "a product across the two widths")

    def test_a_broadcast_count_repeats_without_a_materialized_pair(self):
        rows = [1, 1 << 32, (1 << 63) - 1]
        columns = [10, 20, 30, 1 << 33]
        want = [[r + c for c in columns] for r in rows]
        self._checked(
            self._device([[r] for r in rows]) + self._device([columns]),
            want, "a counting column against a counting row")

        target = self._device([[r + j for j in range(4)] for r in rows])
        target += self._device([columns])
        self._checked(
            target, [[r + j + c for j, c in enumerate(columns)] for r in rows],
            "a counting row added in place across a broadcast")

    def test_a_rank_one_walk_over_two_counting_lines(self):
        # Both operands need their index remapped here, so every element of the
        # answer is read from a pair of words chosen by a stride.
        left = [1, 1 << 32, -3]
        right = [5, 1 << 31, -(1 << 63)]
        self._checked(
            tensorplay.outer(self._device(left), self._device(right)),
            [[a * b for b in right] for a in left],
            "a rank-one product of counting lines")

    def test_the_magnitude_turns_the_word_that_carries_the_sign(self):
        counts = [0, 1, -1, 1 << 32, -(1 << 32), (1 << 32) + 7,
                  (1 << 63) - 1, -(1 << 63), 1234567890123, -1234567890123]
        self._checked(
            self._device(counts).abs(), [abs(a) for a in counts],
            "magnitude of a counting payload")
        # Only the word above the seam tells a count from its negation, so a
        # payload whose low words are all none still has a magnitude.
        self._checked(
            self._device([1 << 32, -(1 << 32), 2 << 32, -(2 << 32)]).abs(),
            [1 << 32, 1 << 32, 2 << 32, 2 << 32],
            "magnitude of a pair whose low word is none")
        # A magnitude past every count a pair holds lands where the count it
        # turned away lands.
        self._checked(self._device([-(1 << 63)]).abs(), [-(1 << 63)],
                      "magnitude of the least count a pair holds")
        target = self._device(counts)
        target.abs_()
        self._checked(target, [abs(a) for a in counts],
                      "magnitude written back into a count")

    def test_a_negated_pair_walks_back_across_the_seam(self):
        counts = [0, 1, -1, 1 << 32, -(1 << 32), (1 << 63) - 1, -(1 << 63),
                  1234567890123]
        self._checked(self._device(counts).neg(), [-a for a in counts],
                      "negation of a counting payload")
        self._checked(-self._device(counts), [-a for a in counts],
                      "a count read through a minus sign")
        self._checked(
            self._device([1 << 32, (1 << 32) + 1, -(2 << 32)]).neg(),
            [-(1 << 32), -((1 << 32) + 1), 2 << 32],
            "negation across the seam of a pair")
        target = self._device(counts)
        target.neg_()
        self._checked(target, [-a for a in counts],
                      "negation written back into a count")

    def test_the_sign_counts_the_whole_pair(self):
        counts = [0, 1, -1, 5, -5, 1 << 32, -(1 << 32), (1 << 63) - 1,
                  -(1 << 63)]
        self._checked(
            self._device(counts).sign(),
            [(a > 0) - (a < 0) for a in counts],
            "sign of a counting payload")
        # A pair is none only when both of its words are, so a low word of none
        # under a high word that is not answers one way or the other.
        self._checked(self._device([1 << 32, -(1 << 32)]).sign(), [1, -1],
                      "sign of a pair whose low word is none")

    def test_a_squared_pair_keeps_the_low_half_of_the_walk(self):
        counts = [0, -1, 3, 12345, 1 << 32, (1 << 32) + 1, 1 << 31,
                  -(1 << 63), (1 << 63) - 1]
        self._checked(self._device(counts).square(), [a * a for a in counts],
                      "square of a counting payload")
        # A factor whose low word is none squares to nothing a pair can hold, so
        # this asks the walk for the terms that live above the seam.
        self._checked(self._device([1 << 32, 12345]).square(),
                      [0, 12345 * 12345], "square of a pair at the seam")
        target = self._device(counts)
        target.square_()
        self._checked(target, [a * a for a in counts],
                      "square written back into a count")

    def test_a_clamp_at_none_leaves_the_positive_half_alone(self):
        counts = [0, 1, -1, 5, -5, 1 << 32, -(1 << 32), (1 << 63) - 1,
                  -(1 << 63)]
        self._checked(self._device(counts).relu(), [max(a, 0) for a in counts],
                      "a counting payload clamped at none")
        target = self._device(counts)
        target.relu_()
        self._checked(target, [max(a, 0) for a in counts],
                      "clamp at none written back into a count")

    def test_the_roundings_answer_a_count_with_itself(self):
        counts = [-(1 << 63), -(1 << 32), -7, 0, 1, 1 << 32, (1 << 63) - 1]
        for name in ("floor", "ceil", "round", "trunc"):
            self._checked(
                getattr(self._device(counts), name)(), counts,
                f"{name} of a counting payload")
            target = self._device(counts)
            getattr(target, name + "_")()
            self._checked(target, counts,
                          f"{name} written back into a count")

    def test_the_greater_and_lesser_of_two_pairs_compare_the_high_word(self):
        first = [1 << 32, -1, (1 << 63) - 1, 5, 0, -(1 << 32), (1 << 32) + 7,
                 -(1 << 63)]
        second = [1, 1 << 32, -1, 5, -(1 << 63), 1, (1 << 32) + 3, 3]
        self._checked(
            tensorplay.maximum(self._device(first), self._device(second)),
            [max(a, b) for a, b in zip(first, second)],
            "greater of two counting payloads")
        self._checked(
            tensorplay.minimum(self._device(first), self._device(second)),
            [min(a, b) for a, b in zip(first, second)],
            "lesser of two counting payloads")
        # A pair with the larger low word is no greater than its partner unless
        # the word above it agrees, and a negative count sets every bit of its
        # high word, so comparing those words without a sign would turn both
        # answers over.
        self._checked(
            tensorplay.maximum(self._device([-(1 << 32), 1 << 32]),
                               self._device([1, 1])),
            [1, 1 << 32], "extremum of pairs that share no word")

    def test_a_narrow_count_meets_a_wide_one_at_an_extremum(self):
        narrow = [1, 2, -3, 5, 0, -(1 << 30)]
        wide = [1 << 40, -1, (1 << 63) - 1, -(1 << 63), 0, 1 << 30]
        self._checked(
            tensorplay.maximum(self._device(narrow, tensorplay.int32),
                               self._device(wide)),
            [max(a, b) for a, b in zip(narrow, wide)],
            "greater of a narrow count and a wide one")
        self._checked(
            tensorplay.minimum(self._device(narrow, tensorplay.int32),
                               self._device(wide)),
            [min(a, b) for a, b in zip(narrow, wide)],
            "lesser of a narrow count and a wide one")

    def test_a_broadcast_extremum_repeats_a_pair_it_never_materializes(self):
        rows = [1 << 32, -1, (1 << 63) - 1]
        columns = [10, 20, -(1 << 33), (1 << 32) + 3]
        self._checked(
            tensorplay.maximum(self._device([[r] for r in rows]),
                               self._device([columns])),
            [[max(r, c) for c in columns] for r in rows],
            "a counting column against a counting row")
        self._checked(
            tensorplay.minimum(self._device([[r] for r in rows]),
                               self._device([columns])),
            [[min(r, c) for c in columns] for r in rows],
            "least of a counting column and a counting row")

    def test_a_count_subtracted_from_a_number_walks_the_pair_backwards(self):
        counts = [0, 1, -7, 1 << 32, (1 << 63) - 1, -(1 << 63)]
        self._checked(5 - self._device(counts), [5 - a for a in counts],
                      "a count taken from a number")
        self._checked(
            (1 << 40) - self._device(counts),
            [(1 << 40) - a for a in counts],
            "a count taken from a wide number")
        self._checked(
            -(1 << 40) - self._device(counts),
            [-(1 << 40) - a for a in counts],
            "a count taken from a wide negative number")

    def test_a_magnitude_over_a_tail_reads_every_pair(self):
        rng = np.random.default_rng(77)
        for length in (1, 2, 3, 5, 17, 64, 65):
            left = rng.integers(-(2**62), 2**62, size=length)
            right = rng.integers(-(2**62), 2**62, size=length)
            self._checked(
                self._device(left).abs(), [abs(int(a)) for a in left],
                f"magnitude of {length} counts")
            self._checked(
                tensorplay.maximum(self._device(left), self._device(right)),
                [max(int(a), int(b)) for a, b in zip(left, right)],
                f"greater of {length} counts")

    def test_a_formula_with_fractions_still_refuses_a_pair(self):
        counts = [0, 1, 2, 5, 30]
        for name in ("exp", "sqrt", "sin", "tanh", "log"):
            widened = getattr(self._device(counts), name)()
            self.assertEqual(widened.dtype, tensorplay.float32, name)
            np.testing.assert_allclose(
                np.asarray(widened.cpu()),
                getattr(np, name)(np.asarray(counts, np.float32)),
                rtol=1e-5, atol=1e-6, err_msg=name)
        for name in ("exp_", "sqrt_", "sin_", "reciprocal_", "log_"):
            with self.assertRaisesRegex(
                    RuntimeError, "no kernel for a count wider than a word"):
                getattr(self._device(counts), name)()

    def test_lengths_that_fill_no_whole_number_of_slots(self):
        rng = np.random.default_rng(76)
        for length in (1, 2, 3, 5, 7, 13, 17, 64, 65):
            left = rng.integers(-(2**62), 2**62, size=length)
            right = rng.integers(-(2**62), 2**62, size=length)
            self._checked(
                self._device(left) + self._device(right),
                [int(a) + int(b) for a, b in zip(left, right)],
                f"sum of {length} counts")
            self._checked(
                self._device(left) * self._device(right),
                [int(a) * int(b) for a, b in zip(left, right)],
                f"product of {length} counts")

    def test_a_wide_payload_past_the_image_limit_still_counts(self):
        host = np.full((20001,), 1 << 31, dtype=np.int64)
        got = self._device(host) + self._device(host)
        self._checked(got, [1 << 32] * 20001, "a long counting payload doubled")

    def test_a_clip_between_two_counts_of_its_own_width(self):
        # A closed range over a pair of words is settled word against word on each
        # side: a count outside it lands on the edge, one inside is left alone.
        counts = [0, 1, -1, 5, -7, 1 << 32, -(1 << 32), (1 << 40) + 7,
                  -(1 << 63), (1 << 63) - 1]
        lo, hi = -(1 << 40), 1 << 40
        self._checked(
            self._device(counts).clamp(lo, hi),
            [min(max(a, lo), hi) for a in counts],
            "counting payload clipped to a range of its own width")
        # The extremes of the width clip nothing, which is how a caller leaves one
        # side of the range open.
        self._checked(
            self._device(counts).clamp(-(2 ** 63), 2 ** 63 - 1),
            counts,
            "clip at the ends of the width")
        import tensorplay.functional as F
        self._checked(
            F.clamp(self._device(counts), min=-5),
            [max(a, -5) for a in counts],
            "counting payload given only a floor")
        self._checked(
            F.clamp(self._device(counts), max=3),
            [min(a, 3) for a in counts],
            "counting payload given only a ceiling")

    def test_a_clip_written_back_keeps_the_pair_it_was_allocated(self):
        counts = [0, 1, -1, 5, -7, 1 << 32, -(1 << 32), (1 << 63) - 1, -(2 ** 63)]
        target = self._device(counts)
        target.clamp_(0, 1 << 32)
        self._checked(
            target, [min(max(a, 0), 1 << 32) for a in counts],
            "clip written into a counting payload")
        target = self._device(counts)
        target.clamp_min_(1 << 32)
        self._checked(
            target, [max(a, 1 << 32) for a in counts],
            "floor written into a counting payload")
        target = self._device(counts)
        target.clamp_max_(-(1 << 32))
        self._checked(
            target, [min(a, -(1 << 32)) for a in counts],
            "ceiling written into a counting payload")

    def test_a_clip_of_a_narrow_count_keeps_its_word(self):
        # The wide walk is chosen by the width of the payload, not by the shape of
        # the request: a count that fits a word answers with the kernels it did
        # before a wider one was added.
        narrow = self._device([0, 5, -7, 100, -200], tensorplay.int32)
        got = narrow.clamp(-10, 10)
        self.assertEqual(got.dtype, tensorplay.int32,
                         "clip changed the width of a narrow count")
        np.testing.assert_array_equal(
            np.asarray(got.cpu()),
            np.array([min(max(a, -10), 10) for a in [0, 5, -7, 100, -200]],
                     dtype=np.int32),
            err_msg="narrow count clipped to a range")

    def test_a_clip_over_a_grid_and_a_tail_that_fills_no_whole_word(self):
        grid = [[1 << 32, -1, 5], [-(1 << 40), 0, (1 << 63) - 1]]
        lo, hi = -(1 << 36), 1 << 36
        self._checked(
            self._device(grid).clamp(lo, hi),
            [[min(max(a, lo), hi) for a in row] for row in grid],
            "counting grid clipped to a range")
        for length in (1, 2, 3, 5, 17, 64, 65, 2001):
            rng = np.random.default_rng(length)
            host = rng.integers(-(2 ** 62), 2 ** 62, size=length)
            self._checked(
                self._device(host).clamp(-(1 << 40), 1 << 40),
                [min(max(int(a), -(1 << 40)), 1 << 40) for a in host],
                f"clip over {length} counts")

    def test_a_selection_between_two_counting_payloads(self):
        # A lane takes the whole pair of words it is told to, so the low word of
        # a held element never lands beside the high word of the rejected one.
        held = [0, 1, -1, 1 << 32, -(1 << 32), (1 << 40) + 7, (1 << 63) - 1,
                -(2 ** 63), 5, -7]
        taken = [7] * len(held)
        truth = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        mask = tensorplay.from_numpy(np.array(truth, dtype=np.bool_)).to("vulkan")
        self._checked(
            tensorplay.where(mask, self._device(held), self._device(taken)),
            [a if t else b for a, b, t in zip(held, taken, truth)],
            "selection between two counting payloads")
        self._checked(
            tensorplay.where(mask, self._device(taken), self._device(held)),
            [b if t else a for a, b, t in zip(held, taken, truth)],
            "the same selection turned around")

    def test_a_selection_that_repeats_a_plane_of_one_count(self):
        # A number folded into a plane of one is a counting payload too: the walk
        # reads its pair again for every element rather than materializing it.
        held = [0, 1, -1, 1 << 32, -(1 << 32), (1 << 40) + 7, 5, -7]
        truth = [1, 0, 1, 1, 0, 0, 1, 0]
        mask = tensorplay.from_numpy(np.array(truth, dtype=np.bool_)).to("vulkan")
        self._checked(
            tensorplay.where(mask, self._device(held), 3),
            [a if t else 3 for a, t in zip(held, truth)],
            "a count taken beside a number")
        self._checked(
            tensorplay.where(mask, -1, self._device(held)),
            [-1 if t else a for a, t in zip(held, truth)],
            "a number taken beside a count")

    def test_a_mixed_pair_selects_at_the_wider_width(self):
        # The width that holds the pair decides, so a count beside a word moves
        # the word up rather than the count down: the answer keeps both.
        narrow = [1, 2, 3, 4, 5, 6, 7, 8]
        wide = [0, 1, -1, 1 << 32, -(1 << 32), (1 << 40) + 7, 5, -7]
        truth = [1, 1, 0, 0, 1, 0, 1, 0]
        mask = tensorplay.from_numpy(np.array(truth, dtype=np.bool_)).to("vulkan")
        self._checked(
            tensorplay.where(
                mask, self._device(narrow, tensorplay.int32), self._device(wide)),
            [a if t else b for a, b, t in zip(narrow, wide, truth)],
            "a word beside a counting payload")
        got = tensorplay.where(
            mask, self._device(wide), self._device(narrow, tensorplay.int32))
        self.assertEqual(got.dtype, tensorplay.int64,
                         "a mixed pair narrowed the counting payload")

    def test_a_selection_that_repeats_a_line_of_truth(self):
        # The condition, like a branch, is addressed through its own strides: a
        # column of truth repeats across the row and a row across the column.
        grid = [[0, 1, -1], [(1 << 40) + 7, -(1 << 32), 5]]
        filler = [[-99, -99, -99], [-99, -99, -99]]
        column = [[True], [False]]
        row = [[True, False, True]]
        self._checked(
            tensorplay.where(
                tensorplay.from_numpy(np.array(column, dtype=np.bool_))
                .to("vulkan"),
                self._device(grid), self._device(filler)),
            [[a if column[r][0] else b for a, b in zip(gr, fl)]
             for r, (gr, fl) in enumerate(zip(grid, filler))],
            "a column of truth over a counting grid")
        self._checked(
            tensorplay.where(
                tensorplay.from_numpy(np.array(row, dtype=np.bool_))
                .to("vulkan"),
                self._device(grid), self._device(filler)),
            [[a if t else b for a, b, t in zip(gr, fl, row[0])]
             for gr, fl in zip(grid, filler)],
            "a row of truth over a counting grid")

    def test_a_selection_under_a_mask_of_its_own_counts(self):
        # The mask a counting payload answers for is the same pair of words read
        # against a number, so the two walks join without a host round trip.
        counts = [0, 1, -1, 1 << 32, -(1 << 32), (1 << 40) + 7, 5, -7]
        self._checked(
            tensorplay.where(
                self._device(counts) > 0, self._device([7] * len(counts)),
                self._device(counts)),
            [7 if a > 0 else a for a in counts],
            "a counting payload measured and then replaced")

    def test_a_selection_over_a_tail_of_counting_payloads(self):
        for length in (1, 2, 3, 5, 17, 65, 257, 20001):
            rng = np.random.default_rng(length)
            left = rng.integers(-(2 ** 62), 2 ** 62, size=length)
            right = rng.integers(-(2 ** 62), 2 ** 62, size=length)
            truth = rng.random(length) > 0.5
            self._checked(
                tensorplay.where(
                    tensorplay.from_numpy(truth).to("vulkan"),
                    self._device(left), self._device(right)),
                np.where(truth, left, right),
                f"selection over {length} counts")

    def test_a_number_past_a_word_folds_as_a_pair(self):
        # Two numbers with no payload beside them keep the integer plane a count
        # fits in - and step up to a pair of words when one of them does not.
        condition = tensorplay.tensor([True, False]).to("vulkan")
        in_word = tensorplay.where(condition, 7, 3)
        self.assertEqual(in_word.dtype, tensorplay.int32,
                         "a count inside a word widened its plane")
        np.testing.assert_array_equal(
            np.asarray(in_word.cpu()), np.array([7, 3], dtype=np.int32),
            err_msg="a pair inside a word")
        past_word = tensorplay.where(condition, (1 << 40) + 7, 3)
        self.assertEqual(past_word.dtype, tensorplay.int64,
                         "a count past a word kept a plane that truncates it")
        np.testing.assert_array_equal(
            np.asarray(past_word.cpu()),
            np.array([(1 << 40) + 7, 3], dtype=np.int64),
            err_msg="a pair where one count needs two words")

    def test_a_walk_with_no_pair_of_words_to_read_says_so(self):
        counts = self._device([10, 1 << 32, -7])
        other = self._device([3, 3, 3])
        # Division and a remainder have no walk over a pair of words: the refusal
        # names the width, rather than reading the element one word at a time.
        with self.assertRaisesRegex(RuntimeError, "count wider than a word"):
            counts / other
        with self.assertRaisesRegex(RuntimeError, "count wider than a word"):
            tensorplay.remainder(counts, other)
        # A fraction has no slot in a counting payload written in place.
        with self.assertRaisesRegex(RuntimeError, "fractional operand"):
            counts.add_(0.5)
        # A payload allocated as a narrow count cannot hold a wider partner.
        narrow = self._device([1, 2, 3], tensorplay.int32)
        with self.assertRaisesRegex(RuntimeError, "wider than its destination"):
            narrow += self._device([1, 2, 3])


class TestVulkanWideTotals(unittest.TestCase):
    # A payload of counts wider than a word has no texel to read and no
    # single-precision register to fold into, so its total and its product walk
    # the pairs of words it is carried in, one row per invocation.  The fold
    # starts at the identity it names, so a short tile contributes fewer terms
    # and a row folded whole still answers.  A decode loop asks for these over
    # its token, position and length counters.

    MASK = (1 << 64) - 1
    ROW_LENGTHS = (1, 2, 3, 5, 17, 65, 255, 256, 257, 258, 513, 1024, 40001)

    @classmethod
    def _as_int64(cls, value):
        # A total past every word lands where the counts it added land, which is
        # what a pair of words holds too.
        bits = int(value) & cls.MASK
        return bits - (1 << 64) if bits >= (1 << 63) else bits

    @classmethod
    def _wrapped(cls, values):
        host = np.array(values, dtype=object)
        flat = [cls._as_int64(v) for v in host.ravel().tolist()]
        return np.array(flat, dtype=np.int64).reshape(host.shape)

    def _payload(self, shape, order, reach=2 ** 62):
        rng = np.random.default_rng(order)
        return self._wrapped(
            rng.integers(-reach, reach, size=shape).tolist())

    def _device(self, held):
        return tensorplay.from_numpy(held).to("vulkan")

    def _checked(self, got, want, label):
        self.assertEqual(
            str(got.dtype), f"tensorplay.{np.dtype(want.dtype).name}",
            f"{label} answered in the wrong width")
        np.testing.assert_array_equal(
            np.asarray(got.cpu()), want, err_msg=label)

    def test_a_whole_payload_of_counts_adds_up(self):
        for order, shape in enumerate([(2, 3), (4, 5, 6), (2, 3, 4, 5), (700,)]):
            held = self._payload(shape, 71 + order)
            with np.errstate(over="ignore"):
                want = np.sum(held, dtype=np.int64)
            got = self._device(held).sum()
            self.assertTrue(got.device.is_vulkan(),
                            "a wide total left the accelerator")
            self._checked(got, np.asarray(want, dtype=np.int64),
                          f"total of a {shape} payload")

    def test_a_whole_payload_of_counts_multiplies(self):
        for order, shape in enumerate([(2, 3), (4, 5, 6), (2, 3, 4, 5)]):
            held = self._payload(shape, 83 + order, reach=1 << 33)
            with np.errstate(over="ignore"):
                want = np.prod(held, dtype=np.int64)
            self._checked(
                self._device(held).prod(),
                np.asarray(want, dtype=np.int64),
                f"product of a {shape} payload")

    def test_one_axis_folded_keeps_the_axes_it_was_not_asked_about(self):
        held = self._payload((2, 3, 4, 5), 91)
        for axis in range(4):
            for kept in (False, True):
                for name, host_fold, device_fold in (
                        ("sum", np.sum, "sum"),
                        ("product", np.prod, "prod")):
                    with np.errstate(over="ignore"):
                        want = host_fold(
                            held, axis=axis, dtype=np.int64, keepdims=kept)
                    got = getattr(self._device(held), device_fold)(
                        axis, keepdim=kept)
                    expected_shape = (
                        tuple(held.shape[:axis]) + (1,) + tuple(held.shape[axis + 1:])
                        if kept else held.shape[:axis] + held.shape[axis + 1:])
                    self.assertEqual(tuple(got.shape), expected_shape,
                                     f"{name} over axis {axis} kept shape")
                    self._checked(
                        got, np.asarray(want, dtype=np.int64),
                        f"{name} over axis {axis}, keepdim={kept}")

    def test_a_list_of_axes_folds_them_all(self):
        held = self._payload((3, 4, 5), 97)
        for axes in ((0, 2), (0, 1), (1, 2)):
            for kept in (False, True):
                with np.errstate(over="ignore"):
                    want = np.sum(
                        held, axis=axes, dtype=np.int64, keepdims=kept)
                got = self._device(held).sum(list(axes), keepdim=kept)
                self._checked(
                    got, np.asarray(want, dtype=np.int64),
                    f"total over axes {axes}, keepdim={kept}")

    def test_a_negative_axis_folds_the_same_row(self):
        held = self._payload((3, 5, 7), 101)
        for axis in (-1, -2, -3):
            with np.errstate(over="ignore"):
                want = np.sum(held, axis=axis, dtype=np.int64)
            self._checked(
                self._device(held).sum(axis),
                np.asarray(want, dtype=np.int64), f"total over axis {axis}")

    def test_a_row_folded_whole_splits_into_tiles(self):
        # A single row would hand every element to one thread, so the axis is
        # split into fixed tiles, each folded, and the tile results folded after.
        for length in self.ROW_LENGTHS:
            held = ((np.arange(length, dtype=np.int64) - length // 2)
                    * 1000000007).astype(np.int64)
            with np.errstate(over="ignore"):
                total = np.sum(held, dtype=np.int64)
                product = np.prod(
                    ((np.arange(length, dtype=np.int64) % 5) - 2).astype(
                        np.int64), dtype=np.int64)
            self._checked(
                self._device(held.reshape(1, length)).sum(1),
                np.asarray([total], dtype=np.int64),
                f"a tiled row of {length} counts")
            self._checked(
                self._device(
                    ((np.arange(length, dtype=np.int64) % 5) - 2)
                    .astype(np.int64)).prod(),
                np.asarray(product, dtype=np.int64),
                f"a tiled product of {length} counts")

    def test_a_column_folded_with_a_stride(self):
        # Consecutive elements of a row need not sit beside each other: the walk
        # strides by the size of the axes that follow the folded one.
        held = self._payload((3, 400), 103)
        for axis in (0, 1):
            with np.errstate(over="ignore"):
                want = np.sum(held, axis=axis, dtype=np.int64)
            self._checked(
                self._device(held).sum(axis),
                np.asarray(want, dtype=np.int64), f"strided total on axis {axis}")
        middle = self._payload((2, 7, 1, 5), 107)
        with np.errstate(over="ignore"):
            want = np.prod(middle, axis=1, dtype=np.int64)
        self._checked(
            self._device(middle).prod(1), np.asarray(want, dtype=np.int64),
            "a product folded across a stride of five")

    def test_an_answer_carried_in_another_width(self):
        held = np.array([[1, 2, 3], [4, 5, 7]], dtype=np.int64)
        host = self._device(held)
        for wanted, name, inexact in ((tensorplay.int32, "int32", np.int32),
                                      (tensorplay.float32, "float32", np.float32)):
            total = host.sum(dtype=wanted)
            self.assertEqual(str(total.dtype), f"tensorplay.{name}",
                             f"a total asked for as {name}")
            self.assertEqual(int(total.cpu().numpy()), int(held.sum()),
                             "a widened total moved the wrong number")
            product = host.prod(dtype=wanted)
            self.assertEqual(str(product.dtype), f"tensorplay.{name}",
                             f"a product asked for as {name}")
            self.assertEqual(int(product.cpu().numpy()), int(held.prod()),
                             "a widened product moved the wrong number")
            folded = host.sum(1, keepdim=True, dtype=wanted)
            self.assertEqual(str(folded.dtype), f"tensorplay.{name}",
                             f"an axis total asked for as {name}")
            np.testing.assert_array_equal(
                np.asarray(folded.cpu()),
                held.sum(axis=1, keepdims=True).astype(inexact),
                err_msg=f"an axis total as {name}")

    def test_a_total_that_crosses_the_seam_of_its_pair(self):
        # A walk that carried only the low word of a pair keeps every count below
        # 2^32 right, so the cases asked for are exactly the ones that carry.
        cases = (
            ([1 << 32, 1 << 32], 1 << 33),
            ([(1 << 63) - 1, 1], -(1 << 63)),
            ([-(1 << 63), -1], (1 << 63) - 1),
            ([2 ** 62, 2 ** 62], -(1 << 63)),
            ([-(1 << 62), -(1 << 62), -(1 << 62)], -(1 << 63) - (1 << 62)),
        )
        for values, want in cases:
            held = self._wrapped(values)
            got = self._device(held).sum()
            self._checked(
                got, np.array(self._as_int64(want), dtype=np.int64),
                f"a sum of {values}")

    def test_a_product_keeps_the_bits_past_both_words(self):
        cases = (
            ([-(1 << 40), -(1 << 40)], (1 << 80)),
            ([1 << 32, 1 << 32], 1 << 64),
            ([(1 << 32) - 1, (1 << 32) - 1], ((1 << 32) - 1) ** 2),
            ([0, 1 << 40, -3], 0),
            ([-7, -11, 1 << 45], 77 * (1 << 45)),
        )
        for values, want in cases:
            held = self._wrapped(values)
            got = self._device(held).prod()
            self._checked(
                got, np.array(self._as_int64(want), dtype=np.int64),
                f"a product of {values}")

    def test_a_fold_of_one_count_answers_with_the_count(self):
        held = self._payload((1, 1), 109)
        self._checked(
            self._device(held).sum(),
            np.asarray(np.sum(held, dtype=np.int64), dtype=np.int64),
            "a lone count added up")
        self._checked(
            self._device(held).prod(),
            np.asarray(np.prod(held, dtype=np.int64), dtype=np.int64),
            "a lone count multiplied")

    def test_an_empty_payload_adds_to_nothing(self):
        held = np.zeros((0,), dtype=np.int64)
        got = self._device(held).sum()
        self.assertEqual(str(got.dtype), "tensorplay.int64")
        self.assertEqual(int(got.cpu().numpy()), 0,
                         "nothing added to nothing")

    def test_a_fold_over_a_shape_the_walk_cannot_name_says_so(self):
        # A payload carried in pairs has no image form at all, yet the fold still
        # needs a plane it can index: a rank beyond four is refused by name.
        held = tensorplay.tensor(
            np.arange(2 * 3 * 4 * 5 * 6, dtype=np.int64)
            .reshape(2, 3, 4, 5, 6),
            dtype=tensorplay.int64, device="vulkan")
        self.assertEqual(tuple(held.shape), (2, 3, 4, 5, 6))
        with self.assertRaisesRegex(
                RuntimeError, "reductions support 1d to 4d tensors"):
            held.sum()
        with self.assertRaisesRegex(
                RuntimeError, "reductions support 1d to 4d tensors"):
            held.prod()


class TestVulkanRangeOfAPayload(unittest.TestCase):
    # The range a payload covers is the first thing a quantization observer asks
    # of a weight plane before it names a scale, and a calibration step that
    # cannot answer it on the device carries the whole plane back to the host.
    # Both ends answer in the width the payload is carried in, so the fold covers
    # the same element types and both storages the single extrema walk.

    SHAPES = ((2, 3), (4, 5, 6), (2, 3, 4, 5), (700,), (100003,))
    WIDTHS = (
        (tensorplay.float32, np.float32),
        (tensorplay.float16, np.float16),
        (tensorplay.int32, np.int32),
        (tensorplay.int64, np.int64),
    )

    @staticmethod
    def _plane(shape, order, np_dtype):
        rng = np.random.default_rng(order)
        if np_dtype is np.float32:
            return (rng.standard_normal(shape) * 7).astype(np.float32)
        if np_dtype is np.float16:
            return (rng.standard_normal(shape) * 4).astype(np.float16)
        if np_dtype is np.int32:
            return rng.integers(-(2 ** 30), 2 ** 30, shape).astype(np.int32)
        return rng.integers(-(2 ** 62), 2 ** 62, shape).astype(np.int64)

    def _checked(self, host, call, label):
        import tensorplay.functional as F
        want = call(tensorplay.tensor(host))
        got = call(tensorplay.tensor(host).to("vulkan"))
        for name, a, b in (("low", got[0], want[0]), ("high", got[1], want[1])):
            self.assertEqual(
                a.dtype, b.dtype, f"{label}: the {name} end changed its width")
            self.assertTrue(
                a.device.is_vulkan(), f"{label}: the {name} end left the device")
            np.testing.assert_array_equal(
                np.asarray(a.cpu()), np.asarray(b.cpu()),
                err_msg=f"{label}: the {name} end")

    def test_a_whole_payload_answers_both_ends_of_its_range(self):
        import tensorplay.functional as F
        for order, shape in enumerate(self.SHAPES):
            for w, (dtype, np_dtype) in enumerate(self.WIDTHS):
                host = self._plane(shape, 300 + order * 5 + w, np_dtype)
                self._checked(
                    host, lambda t: F.aminmax(t),
                    f"a range over a whole {str(dtype).split('.')[-1]} {shape}")
                # The method form and the function form answer the same pair.
                self._checked(
                    host, lambda t: t.aminmax(),
                    f"a range asked of the payload itself {shape}")

    def test_one_axis_answers_both_ends_of_the_range_it_covers(self):
        for shape in ((4, 5, 6), (2, 3, 4, 5)):
            host = self._plane(shape, 11, np.float32)
            for axis in range(len(shape)):
                for kept in (False, True):
                    self._checked(
                        host,
                        lambda t, axis=axis, kept=kept: t.aminmax(
                            axis, keepdim=kept),
                        f"the range of axis {axis} of {shape}, keepdim={kept}")

    def test_the_two_ends_keep_the_shape_the_fold_names(self):
        host = self._plane((3, 4, 5), 12, np.float32)
        device = tensorplay.tensor(host).to("vulkan")
        low, high = device.aminmax(1)
        self.assertEqual(tuple(low.shape), (3, 5))
        self.assertEqual(tuple(high.shape), (3, 5))
        low, high = device.aminmax(1, keepdim=True)
        self.assertEqual(tuple(low.shape), (3, 1, 5))
        self.assertEqual(tuple(high.shape), (3, 1, 5))

    def test_a_list_of_axes_answers_the_range_they_cover_together(self):
        import tensorplay.functional as F
        host = self._plane((2, 3, 4, 5), 13, np.int32)
        for axes in ([0, 3], [0, 1], [1, 2, 3]):
            for kept in (False, True):
                self._checked(
                    host,
                    lambda t, axes=axes, kept=kept: F.aminmax(
                        t, axes, keepdim=kept),
                    f"the range over axes {axes}, keepdim={kept}")

    def test_a_negative_axis_names_the_same_range(self):
        host = self._plane((3, 5, 7), 14, np.float16)
        for axis in (-1, -2, -3):
            self._checked(
                host, lambda t, axis=axis: t.aminmax(axis),
                f"the range of axis {axis}")

    def test_a_range_agrees_with_the_two_folds_asked_apart(self):
        # The pair is the two extrema of the payload, so a fold that answered a
        # different shape or a different lane would show up here.
        for order, shape in enumerate(self.SHAPES):
            for dtype, np_dtype in self.WIDTHS:
                host = self._plane(shape, 400 + order, np_dtype)
                device = tensorplay.tensor(host, dtype=dtype).to("vulkan")
                low, high = device.aminmax()
                np.testing.assert_array_equal(
                    np.asarray(low.cpu()), np.asarray(device.min().cpu()),
                    err_msg=f"a range whose low end differs from amin {shape}")
                np.testing.assert_array_equal(
                    np.asarray(high.cpu()), np.asarray(device.max().cpu()),
                    err_msg=f"a range whose high end differs from amax {shape}")

    def test_the_range_of_a_long_row_is_read_from_linear_storage(self):
        import tensorplay.functional as F
        # A row past the image axis limit lives in words, and a calibration step
        # over it must not ask the host for the payload.
        for width in (100000, 100001, 100003):
            host = self._plane((2, width), width, np.float32)
            self._checked(
                host, lambda t: t.aminmax(1), f"a range over {width} lanes")
            self._checked(
                host, lambda t: F.aminmax(t), f"a whole range over {width}")

    def test_the_range_of_a_plane_read_out_as_a_step_reaches_its_own_ends(self):
        # The point of asking for both ends in one call: an observer turns the
        # range into a step and measures every lane from the bottom of it.
        host = self._plane((3, 400), 21, np.float32)
        device = tensorplay.tensor(host).to("vulkan")
        low, high = device.aminmax()
        scaled = (device - low) / ((high - low) / 255.0)
        self.assertEqual(float(scaled.min().cpu().numpy()), 0.0,
                         "the bottom of a plane is not the step's zero")
        np.testing.assert_allclose(
            float(scaled.max().cpu().numpy()), 255.0, rtol=1e-6,
            err_msg="a range whose top lane is not 255 steps above its bottom")

    def test_a_lone_lane_answers_itself_at_both_ends(self):
        for dtype, np_dtype in self.WIDTHS:
            value = -(2 ** 62) if np_dtype is np.int64 else -7
            host = np.array([[value]], dtype=np_dtype)
            low, high = tensorplay.tensor(
                host, dtype=dtype).to("vulkan").aminmax()
            self.assertEqual(tuple(low.shape), (), "a lone lane kept a plane")
            np.testing.assert_array_equal(
                np.asarray(low.cpu()), np.asarray(high.cpu()),
                err_msg="a lone lane split into two ends")
            np.testing.assert_array_equal(
                np.asarray(low.cpu()), np.asarray(value, np_dtype),
                err_msg="a lone lane moved")

    def test_a_range_with_no_lane_or_no_number_refuses_by_name(self):
        with self.assertRaisesRegex(
                RuntimeError, "Expected reduction dim to be specified"):
            tensorplay.tensor(np.zeros((0, 4), np.float32)).to("vulkan").aminmax()
        with self.assertRaisesRegex(
                IndexError, "to have non-zero size"):
            tensorplay.tensor(np.zeros((0, 4), np.float32)).to("vulkan").aminmax(0)
        # A plane of pure truth has no fold that orders its lanes, so the range
        # names the widths it does cover rather than answering zero and one.
        with self.assertRaisesRegex(
                RuntimeError, "extrema support Float32, Float16"):
            tensorplay.tensor(np.array([True, False])).to("vulkan").aminmax()


class TestVulkanFoldOfASpanWithoutLanes(unittest.TestCase):
    # A span with no lane has no number to read, and yet the folds a decode
    # builds around it still have to answer: a total starts at zero, a product
    # starts at one, and an average divides nothing by nothing, which is not a
    # number.  A fold that names a lane -- an end of a range, the position of a
    # winner -- has no lane to name and says so with the error the shape itself
    # names, not with a number no lane was read for.  A normalization walks its
    # rows by counting them out of the length of the axis it walks, so a length
    # of zero used to stop it with no answer at all; it now releases the empty
    # payload its own shape names, on both storages.

    SPANS = (
        ((0, 3), 0),
        ((0, 3), 1),
        ((2, 0, 3), 1),
        ((4, 0), 1),
        ((0,), 0),
    )
    WIDE_SPANS = (
        ((1, 1, 30000, 0), 3),
        ((1, 1, 30000, 0), 2),
    )

    @staticmethod
    def _device(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload)).to("vulkan")

    @staticmethod
    def _host(payload):
        return tensorplay.from_numpy(np.ascontiguousarray(payload))

    def _compare(self, got, wanted, message):
        self.assertEqual(str(got.dtype), str(wanted.dtype), message)
        self.assertEqual(tuple(got.shape), tuple(wanted.shape), message)
        np.testing.assert_array_equal(
            got.cpu().numpy(), wanted.cpu().numpy(), err_msg=message)

    def test_the_starting_value_of_a_span_with_no_lane(self):
        for shape, axis in self.SPANS + self.WIDE_SPANS:
            for np_dtype in (np.float32, np.float16):
                payload = np.zeros(shape, np_dtype)
                label = f"{shape} over axis {axis}, {np_dtype.__name__}"
                for keepdim in (False, True):
                    self._compare(
                        self._device(payload).sum(axis, keepdim=keepdim),
                        self._host(payload).sum(axis, keepdim=keepdim),
                        f"a total of no lane, {label}")
                    self._compare(
                        self._device(payload).prod(axis, keepdim=keepdim),
                        self._host(payload).prod(axis, keepdim=keepdim),
                        f"a product of no lane, {label}")
                    self._compare(
                        self._device(payload).mean(axis, keepdim=keepdim),
                        self._host(payload).mean(axis, keepdim=keepdim),
                        f"an average of no lane, {label}")

    def test_a_fold_of_no_lane_answers_the_value_it_starts_from(self):
        # Read directly, with no walk to compare against: the starting values a
        # decode looks for when a window holds no token at all.
        held = self._device(np.zeros((4, 0), np.float32))
        np.testing.assert_array_equal(
            held.sum(1).cpu().numpy(), np.zeros(4, np.float32),
            err_msg="a total of no lane was not the value a sum starts at")
        np.testing.assert_array_equal(
            held.prod(1).cpu().numpy(), np.ones(4, np.float32),
            err_msg="a product of no lane was not the value a product starts at")
        for name, spread in (("variance", held.var(dim=1)),
                             ("spread", held.std(dim=1)),
                             ("average", held.mean(1))):
            np.testing.assert_array_equal(
                np.isnan(spread.cpu().numpy()), np.ones(4, np.bool_),
                err_msg=f"a {name} of no lane was a number")

    def test_a_normalization_of_no_lanes_answers_the_shape_it_walks(self):
        for shape, axis in self.SPANS + self.WIDE_SPANS:
            for np_dtype in (np.float32, np.float16):
                payload = np.zeros(shape, np_dtype)
                label = f"{shape} over axis {axis}, {np_dtype.__name__}"
                for log_mode in (False, True):
                    name = "log-softmax" if log_mode else "softmax"
                    device = self._device(payload)
                    answer = (device.log_softmax(axis) if log_mode
                              else device.softmax(axis))
                    self.assertEqual(tuple(answer.shape), shape,
                                     f"a {name} of no lanes changed its shape, {label}")
                    self.assertEqual(answer.numel(), 0,
                                     f"a {name} of no lanes was not empty, {label}")
                    plain = self._host(payload)
                    self._compare(
                        answer,
                        plain.log_softmax(axis) if log_mode else plain.softmax(axis),
                        f"a {name} of no lanes, {label}")

    def test_a_fold_that_names_a_lane_says_so_where_a_lane_is_kept(self):
        # A fold that names a lane is refused where the answer would carry one,
        # and answers an empty payload where the kept shape has no lane either.
        folds = (
            ("amax", lambda t, axis: t.amax(axis)),
            ("amin", lambda t, axis: t.amin(axis)),
            ("max", lambda t, axis: t.max(axis)[0]),
            ("min", lambda t, axis: t.min(axis)[0]),
            ("range", lambda t, axis: t.aminmax(axis)[0]),
            ("argmax", lambda t, axis: t.argmax(axis)),
            ("argmin", lambda t, axis: t.argmin(axis)),
            ("logsumexp", lambda t, axis: t.logsumexp(axis)),
        )
        for shape, axis in self.SPANS:
            payload = np.zeros(shape, np.float32)
            device = self._device(payload)
            host = self._host(payload)
            # The fold is refused only where an answer would have to name a
            # lane; where the kept shape has no lane either, no lane is named.
            names_a_lane = any(size == 0 for index, size in enumerate(shape)
                               if index != axis)
            for name, fold in folds:
                with self.subTest(fold=name, span=(shape, axis)):
                    if names_a_lane:
                        self._compare(fold(device, axis), fold(host, axis),
                                      f"a {name} that names a lane of no span")
                    else:
                        with self.assertRaisesRegex(
                                IndexError, "to have non-zero size"):
                            fold(device, axis)
                        with self.assertRaisesRegex(
                                IndexError, "to have non-zero size"):
                            fold(host, axis)

    def test_a_whole_fold_of_no_lanes_asks_for_the_axis_that_names_one(self):
        refusals = (
            ("amax", lambda t: t.amax(), RuntimeError,
             "Expected reduction dim to be specified"),
            ("amin", lambda t: t.amin(), RuntimeError,
             "Expected reduction dim to be specified"),
            ("max", lambda t: t.max(), RuntimeError,
             "Expected reduction dim to be specified"),
            ("min", lambda t: t.min(), RuntimeError,
             "Expected reduction dim to be specified"),
            ("argmax", lambda t: t.argmax(), IndexError,
             "Expected reduction dim to be specified"),
            ("argmin", lambda t: t.argmin(), IndexError,
             "Expected reduction dim to be specified"),
            ("range", lambda t: t.aminmax()[0], RuntimeError, None),
        )
        for shape, np_dtype in (((0, 3), np.float32), ((0,), np.float16)):
            payload = np.zeros(shape, np_dtype)
            device = self._device(payload)
            host = self._host(payload)
            for name, fold, error, text in refusals:
                for side, tensor in (("device", device), ("host", host)):
                    with self.subTest(fold=name, side=side,
                                      shape=shape, dtype=np_dtype.__name__):
                        if text is None:
                            with self.assertRaises(error):
                                fold(tensor)
                        else:
                            with self.assertRaisesRegex(error, text):
                                fold(tensor)

    def test_a_span_of_no_lane_still_answers_a_fold_of_a_whole_axis(self):
        # An axis that holds lanes folded against an axis that holds none: the
        # kept shape decides how many answers there are, and every one of them
        # is the value the walk would have started from.
        for shape, axis, answer_shape in (((0, 3), 0, (3,)), ((0, 3), 1, (0,)),
                                          ((2, 0, 3), 1, (2, 3)),
                                          ((4, 0), 1, (4,))):
            payload = np.zeros(shape, np.float32)
            device = self._device(payload)
            with self.subTest(span=(shape, axis)):
                self.assertEqual(tuple(device.sum(axis).shape), answer_shape)
                self.assertEqual(tuple(device.prod(axis).shape), answer_shape)
                self.assertEqual(
                    tuple(device.count_nonzero(axis).shape), answer_shape)
                self.assertEqual(tuple(device.softmax(axis).shape), shape)

    def test_a_counted_span_folds_its_starting_value_too(self):
        # A total and a product carry counts the way they carry measurements.
        # The average of a count has no counted walk yet, so it is refused by
        # name rather than answered with a number no lane was read for.
        for np_dtype in (np.int32, np.int64):
            payload = np.zeros((4, 0), np_dtype)
            device = self._device(payload)
            host = self._host(payload)
            label = f"a span of no counted lanes, {np_dtype.__name__}"
            self._compare(device.sum(1), host.sum(1), f"total {label}")
            self._compare(device.prod(1), host.prod(1), f"product {label}")
            with self.assertRaisesRegex(IndexError, "to have non-zero size"):
                device.amax(1)


class TestVulkanExtremumOfAnUnnumberedLane(unittest.TestCase):
    # A lane that is not a number is the answer a range or a greedy step has to
    # give: it names both ends of the range and the position that holds it.  A
    # fold that only compares trades it for the largest or the smallest number
    # beside it, and the plane then looks better behaved on the device than it
    # is.  The rule reaches every width that measures and both storages the
    # folds walk, including the tile split a whole long row goes through.

    WIDTHS = ((tensorplay.float32, np.float32),
              (tensorplay.float16, np.float16))
    PLANES = ((2, 3), (4, 5, 6), (2, 3, 4, 5))

    @staticmethod
    def _plane(shape, np_dtype, order):
        rng = np.random.default_rng(order)
        return (rng.standard_normal(shape) * 6).astype(np_dtype)

    @staticmethod
    def _hold(payload, flats, sign=1):
        """Put a lane that is not a number at each flat position asked for."""
        held = payload.copy()
        for flat in flats:
            np.put(held, flat % held.size, np.nan if sign > 0 else -np.nan)
        return held

    def _checked(self, host, call, label, dtype=None):
        # The host fold is the authority on which lane wins; a lane that is not
        # a number is only ever compared as "is it one", because the payload it
        # carries is not part of the answer.
        make = (lambda payload: tensorplay.tensor(payload, dtype=dtype)) if dtype \
            else (lambda payload: tensorplay.tensor(payload))
        want = call(make(host))
        got = call(make(host).to("vulkan"))
        ends = (got if isinstance(got, tuple) else (got,)), \
               (want if isinstance(want, tuple) else (want,))
        for index, (a, b) in enumerate(zip(*ends)):
            held, asked = np.asarray(a.cpu()), np.asarray(b.cpu())
            self.assertEqual(
                tuple(held.shape), tuple(asked.shape),
                f"{label}: end {index} changed its shape")
            self.assertTrue(
                a.device.is_vulkan(), f"{label}: end {index} left the device")
            if asked.dtype.kind == "f":
                self.assertEqual(
                    a.dtype, b.dtype, f"{label}: end {index} changed its width")
                np.testing.assert_array_equal(
                    np.isnan(held), np.isnan(asked),
                    err_msg=f"{label}: end {index} dropped or invented a lane "
                            "that is not a number")
                # A measured end is compared as the number it is: narrowing it
                # to a count would hide a different lane held behind the whole
                # part. An unnumbered lane already answered as one, so the
                # compare only sees the numbered lanes left.
                np.testing.assert_array_equal(
                    held[~np.isnan(held)], asked[~np.isnan(asked)],
                    err_msg=f"{label}: end {index}")
            else:
                # A position is a count the two ends carry in different widths.
                np.testing.assert_array_equal(
                    held.astype(np.int64), asked.astype(np.int64),
                    err_msg=f"{label}: end {index}")

    def test_an_unnumbered_lane_names_both_ends_of_the_range(self):
        import tensorplay.functional as F
        for dtype, np_dtype in self.WIDTHS:
            for order, shape in enumerate(self.PLANES):
                plane = self._plane(shape, np_dtype, 500 + order)
                lanes = plane.size
                for flats in ((0,), (lanes // 2,), (lanes - 1,), (1, lanes - 2)):
                    payload = self._hold(plane, flats)
                    self._checked(
                        payload, lambda t: F.aminmax(t),
                        f"the range of a {np_dtype.__name__} {shape} plane "
                        f"holding a lane that is not a number at {flats}")
                    # Both ends answer, and neither keeps a number beside it.
                    low, high = tensorplay.tensor(
                        payload, dtype=dtype).to("vulkan").aminmax()
                    for name, end in (("low", low), ("high", high)):
                        self.assertTrue(
                            np.isnan(np.asarray(end.cpu())),
                            f"{np_dtype.__name__} {shape} at {flats}: the {name} "
                            "end answered a number")

    def test_an_unnumbered_lane_names_both_ends_of_a_linear_row(self):
        import tensorplay.functional as F
        for length in (100000, 100001, 100003):
            row = self._plane((length,), np.float32, 17)
            for flat in (0, 256, length // 2, length - 257, length - 1):
                payload = self._hold(row, (flat,))
                self._checked(
                    payload, lambda t: F.aminmax(t),
                    f"the range of a {length}-lane row folded in tiles with the "
                    f"unnumbered lane at {flat}")

    def test_the_fold_names_the_lane_a_number_could_not(self):
        # An unbounded measure is the easiest thing to mistake for the answer a
        # row of unnumbered lanes gives, so both bounds sit beside the lane.
        for dtype, np_dtype in self.WIDTHS:
            payload = np.array([np.inf, np.nan, -np.inf, 1.5], dtype=np_dtype)
            low, high = tensorplay.tensor(
                payload, dtype=dtype).to("vulkan").aminmax()
            for name, end in (("low", low), ("high", high)):
                self.assertTrue(
                    np.isnan(np.asarray(end.cpu())),
                    f"a {np_dtype.__name__} range kept a bound instead of "
                    "naming the lane that is not a number")
            self._checked(payload, lambda t: t.aminmax(),
                          f"a {np_dtype.__name__} range over both bounds and an "
                          "unnumbered lane")

    def test_one_axis_names_the_unnumbered_lane_where_it_is_kept(self):
        import tensorplay.functional as F
        for dtype, np_dtype in self.WIDTHS:
            payload = self._plane((3, 4, 5), np_dtype, 91)
            payload[1, 2, :] = np.nan
            for axis in (0, 1, 2):
                for kept in (False, True):
                    self._checked(
                        payload,
                        lambda t, axis=axis, kept=kept: F.aminmax(t, [axis], kept),
                        f"the range of axis {axis} of a {np_dtype.__name__} plane "
                        "with a whole unnumbered row", dtype=dtype)
            # Only the rows that hold the lane answer in kind.
            kept = F.amax(tensorplay.tensor(payload, dtype=dtype).to("vulkan"), 2)
            answered = np.asarray(kept.cpu())
            self.assertEqual(
                list(np.isnan(answered).flat),
                [lane == 6 for lane in range(answered.size)],
                f"a {np_dtype.__name__} axis fold spread the lane across the rows "
                "that never held it")

    def test_the_position_named_for_an_unnumbered_lane_is_the_earliest(self):
        import tensorplay.functional as F
        for dtype, np_dtype in self.WIDTHS:
            plane = self._plane((2, 6), np_dtype, 33)
            for flats, first in (((3,), 3), ((1, 4), 1), ((4, 1), 1),
                                 ((5,), 5), ((0, 2), 0)):
                payload = self._hold(plane, flats)
                self._checked(payload, lambda t: F.argmax(t, 1),
                              f"an argmax over a {np_dtype.__name__} row holding "
                              f"unnumbered lanes at {flats}", dtype=dtype)
                self._checked(payload, lambda t: F.argmin(t, 1),
                              f"an argmin over a {np_dtype.__name__} row holding "
                              f"unnumbered lanes at {flats}", dtype=dtype)
                named = tensorplay.tensor(
                    payload, dtype=dtype).to("vulkan")
                self.assertEqual(
                    int(F.argmax(named, 1).cpu()[0]), first,
                    f"a {np_dtype.__name__} argmax did not name the earliest "
                    f"unnumbered lane of {flats}")
                self.assertEqual(
                    int(F.argmin(named, 1).cpu()[0]), first,
                    f"a {np_dtype.__name__} argmin did not name the earliest "
                    f"unnumbered lane of {flats}")

    def test_the_bound_and_the_position_name_the_same_lane(self):
        # The bound and the position come from two walks over the same row, so
        # reading the payload back at the position named has to give the bound
        # answered: a row cannot be unnumbered at one end and numbered at the
        # position the other end points at.
        for dtype, np_dtype in self.WIDTHS:
            payload = self._hold(self._plane((4, 5), np_dtype, 7), (2, 17))
            device = tensorplay.tensor(payload, dtype=dtype).to("vulkan")
            for name, asked in (("max", lambda t: t.max(1)),
                                ("min", lambda t: t.min(1))):
                bounds, positions = asked(device)
                held = np.asarray(bounds.cpu())
                walked = np.asarray(positions.cpu())
                for row in range(payload.shape[0]):
                    lane = payload[row, walked[row]]
                    if np.isnan(held[row]):
                        self.assertTrue(
                            np.isnan(lane),
                            f"a {np_dtype.__name__} {name}(1) answered an "
                            "unnumbered bound for a numbered position")
                    else:
                        self.assertEqual(
                            held[row], lane,
                            f"a {np_dtype.__name__} {name}(1) bound and position "
                            "named different lanes")

    def test_a_plane_of_only_unnumbered_lanes_answers_in_kind(self):
        import tensorplay.functional as F
        for dtype, np_dtype in self.WIDTHS:
            payload = np.full((2, 3), np.nan, dtype=np_dtype)
            device = tensorplay.tensor(payload, dtype=dtype).to("vulkan")
            low, high = device.aminmax()
            for end in (low, high):
                self.assertTrue(
                    bool(np.isnan(np.asarray(end.cpu())).all()),
                    f"a {np_dtype.__name__} plane of unnumbered lanes did not "
                    "answer both ends in kind")
            for name, asked in (("argmax", F.argmax), ("argmin", F.argmin)):
                self.assertEqual(
                    int(asked(device).cpu()), 0,
                    f"a {np_dtype.__name__} {name} over a plane of unnumbered "
                    "lanes named a position past the first")

    def test_a_measured_fold_still_names_the_largest_number(self):
        # The rule is about a lane that is not a number; every other answer a
        # fold gives is the number it gave before, bounds included.
        import tensorplay.functional as F
        for dtype, np_dtype in self.WIDTHS:
            for shape in self.PLANES + ((700,), (100003,)):
                payload = self._plane(shape, np_dtype, 4 + len(shape))
                for call, label in ((lambda t: F.aminmax(t), "range"),
                                    (lambda t: t.max(), "high"),
                                    (lambda t: F.argmax(t), "argmax"),
                                    (lambda t: F.argmin(t), "argmin")):
                    self._checked(payload, call,
                                  f"a clean {np_dtype.__name__} {shape} {label}")
            bounded = np.array(
                [np.inf, -np.inf, 1.5, -7.25, 0.0], dtype=np_dtype)
            for call, label in ((lambda t: t.max(), "high"),
                                (lambda t: t.min(), "low"),
                                (lambda t: t.aminmax(), "range")):
                self._checked(bounded, call,
                              f"a {np_dtype.__name__} {label} over both bounds")
            self.assertEqual(
                float(tensorplay.tensor(bounded, dtype=dtype).to("vulkan").max()),
                float("inf"), f"a {np_dtype.__name__} high lost an unbound lane")

    def test_a_counting_payload_keeps_its_own_ends(self):
        # A count has no lane that is not a number, so the rule changes nothing
        # it answers; both widths the fold walks keep their own ends.
        import tensorplay.functional as F
        rng = np.random.default_rng(5)
        for dtype, np_dtype in ((tensorplay.int32, np.int32),
                                (tensorplay.int64, np.int64)):
            payload = rng.integers(-(2 ** 30), 2 ** 30, (3, 4, 5)).astype(np_dtype)
            self._checked(payload, lambda t: F.aminmax(t),
                          f"the range of a {np_dtype.__name__} plane")
            self._checked(payload, lambda t: F.amax(t, 1),
                          f"a {np_dtype.__name__} axis high")



class TestVulkanFoldAtTheLaneAnIndexNames(unittest.TestCase):
    # A scatter that folds rather than adds hands the number of every entry of a
    # list to the element it names and grows that element the way the fold says -
    # a total, a product, an average, the largest or the smallest of them.  Several
    # entries can name one element, so the fold walks the destination a whole word
    # at a time and reads the list in the order it states: nothing is dropped, no
    # race decides the answer, and the same call answers the same number twice over.
    # Whether the element already held counts as one of the contributions is the
    # whole difference between keeping it and discarding it.

    MODES = ("sum", "prod", "mean", "amin", "amax")
    SHAPES = ((5,), (5, 4), (2, 5, 3), (2, 3, 5, 2))

    @staticmethod
    def _device(payload):
        arr = np.asarray(payload)
        if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return tensorplay.from_numpy(arr).to("vulkan")

    @staticmethod
    def _codes(values, dtype=np.int64):
        return TestVulkanFoldAtTheLaneAnIndexNames._device(
            np.asarray(values, dtype=dtype))

    @staticmethod
    def _canvas(shape, dtype=np.float32, seed=0):
        # Small whole numbers keep either element type and every intermediate
        # exact, so an entry handed to the wrong element shows up as a wrong
        # number rather than as rounding.
        rng = np.random.default_rng([len(shape), seed, *shape])
        return rng.integers(-3, 4, size=shape).astype(dtype)

    @classmethod
    def _list(cls, shape, dim, dtype=np.float32, seed=1):
        axis = dim if dim >= 0 else dim + len(shape)
        rng = np.random.default_rng([seed, *shape, axis])
        return rng.integers(0, shape[axis], size=shape).astype(np.int64)

    @classmethod
    def _want(cls, base, dim, codes, src, mode, include_self):
        axis = dim if dim >= 0 else dim + base.ndim
        rows = base.shape[axis]
        # A fold of a half payload is kept at full width and rounded on the way
        # out, which is how the accumulating scatter already treats one.
        acc = base.astype(
            np.float32 if base.dtype == np.float16 else base.dtype).copy()
        values = np.asarray(src, dtype=acc.dtype)
        held = np.zeros(acc.shape, bool)
        hits = np.zeros(acc.shape, np.int64)
        start = {"sum": 0.0, "prod": 1.0, "mean": 0.0,
                 "amin": float("inf"), "amax": float("-inf")}[mode]
        for flat in range(codes.size):
            code = int(codes.flat[flat])
            if code < 0 or code >= rows:
                continue
            coord = list(np.unravel_index(flat, codes.shape))
            slot = list(coord)
            slot[axis] = code
            slot = tuple(slot)
            if not held[slot]:
                held[slot] = True
                if not include_self:
                    acc[slot] = start
            value = values[tuple(coord)]
            current = acc[slot]
            if mode == "sum" or mode == "mean":
                acc[slot] = current + value
            elif mode == "prod":
                acc[slot] = current * value
            elif mode == "amin":
                acc[slot] = current if (np.isnan(current) or current < value) else value
            else:
                acc[slot] = current if (np.isnan(current) or current > value) else value
            hits[slot] += 1
        if mode == "mean":
            divisor = hits + (1 if include_self else 0)
            divisor[divisor == 0] = 1
            acc = np.where(held, acc / divisor, acc)
        return acc.astype(base.dtype)

    @staticmethod
    def _checked(got, want, mode, dtype, label):
        # A fold of sums, products and extrema lands on the same number either
        # way, and an average over small whole numbers lands on the nearest one
        # the width can hold - which the division the device performs may reach
        # from either side of it.
        if mode != "mean":
            np.testing.assert_array_equal(got, want, err_msg=label)
            return
        rtol = 2e-3 if dtype == np.float16 else 1e-6
        np.testing.assert_allclose(got, want, rtol=rtol, err_msg=label)

    def _folded(self, shape, dim, mode, include_self, dtype=np.float32, label=""):
        base = self._canvas(shape, dtype, seed=2)
        codes = self._list(shape, dim, dtype, seed=3)
        src = self._canvas(shape, dtype, seed=4)
        name = (f"{label} shape={shape} dim={dim} mode={mode} "
                f"keep={include_self} dtype={np.dtype(dtype).name}")
        got = self._device(base).scatter_reduce_(
            dim, self._codes(codes), self._device(src), mode,
            include_self=include_self).cpu().numpy()
        self._checked(got, self._want(base, dim, codes, src, mode, include_self),
                      mode, dtype, name)
        return name

    def test_every_fold_a_mode_states_is_answered_at_the_element_it_names(self):
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                for mode in self.MODES:
                    for include_self in (True, False):
                        self._folded(shape, dim, mode, include_self)

    def test_a_fold_of_a_half_payload_is_answered_in_the_width_it_keeps(self):
        for shape in ((5,), (5, 4), (3, 7)):
            for mode in self.MODES:
                self._folded(shape, 0, mode, True, np.float16)
                self._folded(shape, -1, mode, False, np.float16)

    def test_a_fold_at_the_host_answers_the_same_number(self):
        for shape in self.SHAPES:
            for dim in range(-len(shape), len(shape)):
                for mode in self.MODES:
                    for include_self in (True, False):
                        base = self._canvas(shape, np.float32, seed=6)
                        codes = self._list(shape, dim, np.float32, seed=7)
                        src = self._canvas(shape, np.float32, seed=8)
                        host = tensorplay.scatter_reduce(
                            tensorplay.from_numpy(base.copy()), dim,
                            tensorplay.from_numpy(codes),
                            tensorplay.from_numpy(src), mode,
                            include_self=include_self).cpu().numpy()
                        np.testing.assert_array_equal(
                            host,
                            self._want(base, dim, codes, src, mode, include_self),
                            err_msg=f"host: shape={shape} dim={dim} mode={mode}")

    def test_an_answer_twice_over_is_the_same_answer(self):
        base = self._canvas((5, 4), np.float32, seed=9)
        codes = self._list((5, 4), 0, np.float32, seed=10)
        src = self._canvas((5, 4), np.float32, seed=11)
        first = self._device(base).scatter_reduce_(
            0, self._codes(codes), self._device(src), "sum").cpu().numpy()
        second = self._device(base).scatter_reduce_(
            0, self._codes(codes), self._device(src), "sum").cpu().numpy()
        np.testing.assert_array_equal(first, second)

    def test_a_lane_the_list_never_names_keeps_what_it_held(self):
        # Only the first row is named, so the rest stands as it did on a canvas
        # nothing reached - which is the answer whether or not the fold kept it.
        base = np.arange(20, dtype=np.float32).reshape(5, 4)
        codes = np.zeros((1, 4), np.int64)
        src = np.full((1, 4), 100.0, np.float32)
        for mode in self.MODES:
            for include_self in (True, False):
                for dim in (0, 1):
                    got = self._device(base).scatter_reduce_(
                        dim, self._codes(codes), self._device(src), mode,
                        include_self=include_self).cpu().numpy()
                    self._checked(
                        got, self._want(base, dim, codes, src, mode, include_self),
                        mode, np.float32, f"mode={mode} keep={include_self} dim={dim}")
                    if dim == 0:
                        np.testing.assert_array_equal(got[1:], base[1:])

    def test_a_fold_handed_the_same_element_several_times_is_answered_in_order(self):
        # One lane named four times: the total is the sum of the four, the
        # product their product, the average their average over four (plus the
        # destination when it is kept).
        base = np.array([[3.0, 5.0]], np.float32)
        codes = np.array([[0], [0], [0], [0]], np.int64)
        src = np.array([[1.0], [2.0], [3.0], [4.0]], np.float32)
        for mode in self.MODES:
            for include_self in (True, False):
                got = self._device(base).scatter_reduce_(
                    0, self._codes(codes), self._device(src), mode,
                    include_self=include_self).cpu().numpy()
                self._checked(
                    got, self._want(base, 0, codes, src, mode, include_self),
                    mode, np.float32, f"mode={mode} keep={include_self}")
        answer = self._device(base).scatter_reduce_(
            0, self._codes(codes), self._device(src), "sum").cpu().numpy()
        self.assertEqual(answer[0][0], np.float32(13.0))
        average = self._device(base).scatter_reduce_(
            0, self._codes(codes), self._device(src), "mean",
            include_self=False).cpu().numpy()
        self.assertEqual(average[0][0], np.float32(2.5))

    def test_a_fold_that_names_a_lane_that_is_not_a_number_hands_it_over(self):
        # A lane that is not a number stays one: neither an entry beside it nor
        # the destination can trade it away, and the order the list states makes
        # no difference to which end of the fold holds it.
        base = np.array([[1.0, 9.0]], np.float32)
        codes = np.array([[0], [0]], np.int64)
        for values in ((float("nan"), 3.0), (3.0, float("nan"))):
            src = np.array([[values[0]], [values[1]]], np.float32)
            for mode in ("amin", "amax"):
                for include_self in (True, False):
                    got = self._device(base).scatter_reduce_(
                        0, self._codes(codes), self._device(src), mode,
                        include_self=include_self).cpu().numpy()
                    np.testing.assert_array_equal(
                        got,
                        self._want(base, 0, codes, src, mode, include_self),
                        err_msg=f"mode={mode} keep={include_self} values={values}")
                    self.assertTrue(np.isnan(got[0][0]))
                    self.assertEqual(got[0][1], np.float32(9.0))

    def test_a_payload_that_reaches_past_the_list_is_read_only_where_an_entry_stands(
            self):
        base = np.zeros((3, 2), np.float32)
        codes = np.array([[0], [2]], np.int64)
        src = np.array([[5.0], [7.0], [9.0], [1.0]], np.float32)
        got = self._device(base).scatter_reduce_(
            0, self._codes(codes), self._device(src), "sum").cpu().numpy()
        np.testing.assert_array_equal(got, np.array([[5.0, 0.0],
                                                     [0.0, 0.0],
                                                     [7.0, 0.0]], np.float32))

    def test_an_empty_list_answers_the_destination_as_it_stood(self):
        base = self._canvas((5, 4), np.float32, seed=12)
        codes = np.zeros((0, 4), np.int64)
        src = np.zeros((0, 4), np.float32)
        for mode in self.MODES:
            for include_self in (True, False):
                got = self._device(base).scatter_reduce_(
                    0, self._codes(codes), self._device(src), mode,
                    include_self=include_self).cpu().numpy()
                np.testing.assert_array_equal(got, base)

    def test_a_fold_answered_off_the_device_matches_the_one_off_it(self):
        # The out-of-place spelling leaves the destination alone and answers a
        # new tensor; the in-place one rewrites what it was given.
        base = self._canvas((5, 4), np.float32, seed=13)
        codes = self._list((5, 4), 0, np.float32, seed=14)
        src = self._canvas((5, 4), np.float32, seed=15)
        for mode in self.MODES:
            kept = self._device(base)
            answer = kept.scatter_reduce(
                0, self._codes(codes), self._device(src), mode).cpu().numpy()
            np.testing.assert_array_equal(kept.cpu().numpy(), base)
            in_place = self._device(base).scatter_reduce_(
                0, self._codes(codes), self._device(src), mode).cpu().numpy()
            np.testing.assert_array_equal(answer, in_place)

    def test_a_long_list_within_the_budget_is_answered_and_a_longer_one_refused(
            self):
        length = 4000000
        rng = np.random.default_rng(17)
        held = self._device(np.zeros((length,), np.float32))
        held.scatter_reduce_(
            0, self._codes(rng.integers(0, 7, size=(60000,))),
            self._device(np.ones((60000,), np.float32)), "sum")
        self.assertEqual(held.cpu().numpy().sum(), 60000.0)
        canvas = self._device(np.zeros((length,), np.float32))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            canvas.scatter_reduce_(
                0, self._codes(rng.integers(0, 7, size=(200000,))),
                self._device(np.ones((200000,), np.float32)), "sum")

    def test_operands_that_do_not_line_up_are_answered_off_it(self):
        payload = self._device(self._canvas((5, 4)))
        codes = self._codes(np.zeros((5, 4), np.int64))
        with self.assertRaisesRegex(RuntimeError, "same rank"):
            payload.scatter_reduce(
                0, codes, self._device(np.ones((5,), np.float32)), "sum")
        with self.assertRaisesRegex(RuntimeError, "reaches past the payload"):
            payload.scatter_reduce(
                0, self._codes(np.zeros((4, 4), np.int64)),
                self._device(np.zeros((3, 4), np.float32)), "sum")
        # A payload of another width would be folded in the wrong arithmetic, so
        # it is answered off it rather than widened on the way in.
        with self.assertRaisesRegex(RuntimeError, "element type the"):
            self._device(self._canvas((5, 4), np.float16)).scatter_reduce(
                0, codes, self._device(np.ones((5, 4), np.float32)), "sum")
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            payload.scatter_reduce(
                0, codes, self._device(np.ones((5, 4), np.int64)), "sum")

    def test_a_mode_not_among_the_folds_is_answered_with_the_names_it_offers(self):
        payload = self._device(self._canvas((5, 4)))
        with self.assertRaisesRegex(
                ValueError,
                "reduce argument must be one of 'sum', 'prod', 'mean', 'amin', "
                "'amax' but got: maximum"):
            payload.scatter_reduce(
                0, self._codes(np.zeros((5, 4), np.int64)),
                self._device(np.ones((5, 4), np.float32)), "maximum")

class TestVulkanInfinityAndNotANumber(unittest.TestCase):
    # The probes answer lane by lane in the payload's own width: a lane that
    # leaves the finite range is told apart from one that only changed sign,
    # and a lane that is not a number answers every ordering false.  Both
    # storages need a case: a row past the image-axis limit lives in a buffer,
    # where a half payload packs two lanes into one word.

    LENGTHS = (64, 200003)
    SPECIALS = np.array(
        [np.inf, -np.inf, np.nan, 0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0,
         2.0 ** -24, 1.0009765625, 1e-8, 7.5], dtype=np.float32)
    PROBES = {
        "isinf": np.isinf,
        "isfinite": np.isfinite,
        "isnan": np.isnan,
        # The sign probe asks whether a lane sits below zero, so a negative
        # zero reports no sign.
        "signbit": lambda values: values < 0,
        "isposinf": lambda values: values == np.inf,
        "isneginf": lambda values: values == -np.inf,
    }

    def _values(self, count, dtype, seed=21):
        rng = np.random.default_rng(seed)
        values = (rng.standard_normal(count) * 8.0).astype(dtype)
        values[:self.SPECIALS.size] = self.SPECIALS.astype(dtype)
        return values

    def _device(self, values):
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    def test_each_probe_answers_lane_by_lane_over_both_storages(self):
        for dtype in (np.float32, np.float16):
            for length in self.LENGTHS:
                values = self._values(length, dtype)
                device = self._device(values)
                for name, want in self.PROBES.items():
                    np.testing.assert_array_equal(
                        getattr(tensorplay, name)(device).cpu().numpy(),
                        want(values),
                        err_msg=f"{name} over {np.dtype(dtype).name} "
                                f"length {length}")

    def test_a_rank_three_payload_keeps_its_own_geometry(self):
        values = self._values(24, np.float16).reshape(2, 3, 4)
        device = self._device(values)
        for name, want in self.PROBES.items():
            np.testing.assert_array_equal(
                getattr(tensorplay, name)(device).cpu().numpy(), want(values),
                err_msg=name)

    def test_an_integral_or_bool_lane_never_leaves_the_number_line(self):
        cases = (
            (np.int8, np.array([0, 1, -1, -128, 127])),
            (np.uint8, np.array([0, 1, 2, 255])),
            (np.int32, np.array([0, 1, -1, -7, 1 << 30])),
            (np.int64, np.array([0, 1, -1, -(1 << 60), 1 << 40])),
            (np.bool_, np.array([True, False, True])),
        )
        for dtype, raw in cases:
            values = raw.astype(dtype)
            device = self._device(values)
            for name in ("isinf", "isnan", "isposinf", "isneginf"):
                np.testing.assert_array_equal(
                    getattr(tensorplay, name)(device).cpu().numpy(),
                    np.zeros(values.shape, dtype=bool),
                    err_msg=f"{name} over {np.dtype(dtype).name}")
            np.testing.assert_array_equal(
                tensorplay.isfinite(device).cpu().numpy(),
                np.ones(values.shape, dtype=bool),
                err_msg=f"isfinite over {np.dtype(dtype).name}")
            np.testing.assert_array_equal(
                tensorplay.signbit(device).cpu().numpy(),
                values.astype(np.int64) < 0,
                err_msg=f"signbit over {np.dtype(dtype).name}")

    def test_nan_to_num_takes_each_uncarried_value_to_its_bound(self):
        for dtype in (np.float32, np.float16):
            values = self._values(37, dtype)
            device = self._device(values)
            got = tensorplay.nan_to_num(device)
            bound = np.finfo(dtype).max
            want = values.copy()
            want[np.isnan(want)] = 0.0
            want[np.isposinf(want)] = bound
            want[np.isneginf(want)] = -bound
            np.testing.assert_array_equal(
                got.cpu().numpy(), want, err_msg=np.dtype(dtype).name)
            self.assertEqual(got.dtype, device.dtype)
            # The allocating form leaves the payload it read untouched.
            np.testing.assert_array_equal(device.cpu().numpy(), values)

    def test_nan_to_num_takes_the_replacements_it_is_given(self):
        values = np.array([np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
        got = tensorplay.nan_to_num(
            self._device(values), nan=2.5, posinf=9.0, neginf=-9.0)
        np.testing.assert_array_equal(
            got.cpu().numpy(), np.array([2.5, 9.0, -9.0, 2.0], np.float32))
        half = tensorplay.nan_to_num(
            self._device(values.astype(np.float16)),
            nan=2.5, posinf=9.0, neginf=-9.0)
        np.testing.assert_array_equal(
            half.cpu().numpy(),
            np.array([2.5, 9.0, -9.0, 2.0], np.float16))

    def test_a_replacement_that_leaves_the_range_does_not_feed_another(self):
        # A lane that is not a number takes the replacement it was given with
        # no further question, even when that replacement is itself infinite.
        values = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
        got = tensorplay.nan_to_num(
            self._device(values), nan=np.inf, posinf=5.0, neginf=-5.0)
        np.testing.assert_array_equal(
            got.cpu().numpy(), np.array([np.inf, 5.0, -5.0], np.float32))

    def test_nan_to_num_in_place_rewrites_the_same_payload(self):
        for dtype, length in ((np.float32, 64), (np.float16, 200003)):
            values = self._values(length, dtype)
            device = self._device(values)
            device.nan_to_num_(nan=1.0)
            bound = np.finfo(dtype).max
            want = values.copy()
            want[np.isnan(want)] = 1.0
            want[np.isposinf(want)] = bound
            want[np.isneginf(want)] = -bound
            np.testing.assert_array_equal(
                device.cpu().numpy(), want,
                err_msg=np.dtype(dtype).name)

    def test_an_integral_payload_keeps_its_values(self):
        for dtype in (np.int32, np.int64, np.bool_):
            values = (np.array([3, -4, 0], dtype=dtype) if dtype != np.bool_
                      else np.array([True, False, True]))
            device = self._device(values)
            got = tensorplay.nan_to_num(
                device, nan=7.0, posinf=8.0, neginf=-8.0)
            np.testing.assert_array_equal(got.cpu().numpy(), values)
            self.assertEqual(got.dtype, device.dtype)
            device.nan_to_num_()
            np.testing.assert_array_equal(device.cpu().numpy(), values)

    def test_the_device_answers_what_the_host_answers(self):
        for dtype in (np.float32, np.float16):
            values = self._values(48, dtype)
            host = tensorplay.from_numpy(values.copy())
            device = self._device(values)
            for name in self.PROBES:
                np.testing.assert_array_equal(
                    getattr(tensorplay, name)(device).cpu().numpy(),
                    getattr(tensorplay, name)(host).numpy(),
                    err_msg=f"{name} over {np.dtype(dtype).name}")
            np.testing.assert_array_equal(
                tensorplay.nan_to_num(device).cpu().numpy(),
                tensorplay.nan_to_num(host).numpy(),
                err_msg=f"nan_to_num over {np.dtype(dtype).name}")

class TestVulkanFrac(unittest.TestCase):
    # frac answers what a lane keeps once the part toward zero is taken away,
    # so the answer carries the sign of its lane and never leaves (-1, 1).  A
    # count holds no fraction of itself and is turned away rather than read
    # wide and answered with a zero.  Both storages need a case: a row past
    # the image axis limit lives in a buffer, where a half payload shares one
    # word with its neighbour and a length that is not a multiple of four
    # leaves a tail a single invocation owns.

    LENGTHS = (64, 200003)
    SPECIALS = np.array(
        [-0.0, 0.0, -3.0, 3.0, -17.0, 0.99999994, -0.99999994, 2.5, -2.5,
         65504.0, -65504.0, np.inf, -np.inf, np.nan],
        dtype=np.float32)

    def _values(self, count, dtype, seed=41):
        rng = np.random.default_rng(seed)
        values = (rng.standard_normal(count) * 12.0).astype(dtype)
        values[: self.SPECIALS.size] = self.SPECIALS.astype(dtype)
        return values

    def _device(self, values):
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _want(values):
        wide = values.astype(np.float64)
        return wide - np.trunc(wide)

    def test_a_lane_keeps_what_the_whole_toward_zero_leaves(self):
        for dtype in (np.float32, np.float16):
            for length in self.LENGTHS:
                values = self._values(length, dtype)
                payload = self._device(values)
                answer = tensorplay.frac(payload)
                self.assertEqual(answer.dtype, payload.dtype)
                self.assertEqual(tuple(answer.shape), (length,))
                np.testing.assert_allclose(
                    answer.cpu().numpy().astype(np.float64),
                    self._want(values), rtol=0, atol=0, equal_nan=True,
                    err_msg=f"{np.dtype(dtype).name} length {length}")
                # The allocating form leaves the payload it read standing.
                np.testing.assert_array_equal(payload.cpu().numpy(), values)

    def test_the_answer_carries_the_sign_of_its_lane_and_stays_inside_one(self):
        for dtype in (np.float32, np.float16):
            values = self._values(256, dtype)[self.SPECIALS.size:]
            answer = tensorplay.frac(self._device(values)).cpu().numpy()
            self.assertTrue(
                np.all(np.abs(answer) < 1.0),
                msg="a fraction left the open unit interval")
            live = answer != 0.0
            np.testing.assert_array_equal(
                np.signbit(answer[live]), np.signbit(values[live]),
                err_msg=f"{np.dtype(dtype).name} kept another sign")

    def test_a_lane_with_no_fraction_answers_zero(self):
        whole = np.array(
            [-0.0, 0.0, -1.0, 1.0, -256.0, 256.0, 65504.0, -65504.0],
            np.float32)
        answer = tensorplay.frac(self._device(whole)).cpu().numpy()
        np.testing.assert_array_equal(answer, np.zeros(whole.shape, np.float32))
        # A lane that is not a number and an endless lane both answer a
        # fraction that cannot be named.
        endless = np.array([np.inf, -np.inf, np.nan], np.float32)
        self.assertTrue(
            np.all(np.isnan(
                tensorplay.frac(self._device(endless)).cpu().numpy())))

    def test_the_walk_writes_into_the_payload_it_read(self):
        for dtype in (np.float32, np.float16):
            for length in self.LENGTHS:
                values = self._values(length, dtype, seed=length)
                payload = self._device(values)
                answer = tensorplay.frac_(payload)
                self.assertEqual(tuple(answer.shape), (length,))
                np.testing.assert_allclose(
                    payload.cpu().numpy().astype(np.float64),
                    self._want(values), rtol=0, atol=0, equal_nan=True,
                    err_msg=f"{np.dtype(dtype).name} length {length} in place")

    def test_a_count_has_no_fraction_to_take(self):
        import tensorplay.functional as F
        counting = {
            "int32": np.arange(-4, 5, dtype=np.int32),
            "int64": np.arange(-4, 5, dtype=np.int64),
            "bool": np.arange(9) % 2 == 0,
        }
        for label, values in counting.items():
            payload = self._device(values)
            with self.subTest(kind=f"frac over {label}"):
                with self.assertRaisesRegex(RuntimeError, "measured payload"):
                    F.frac(payload)
            with self.subTest(kind=f"frac_ over {label}"):
                with self.assertRaisesRegex(RuntimeError, "measured payload"):
                    F.frac_(payload)

    def test_the_gradient_of_a_fraction_is_the_gradient_it_was_given(self):
        for dtype in (np.float32, np.float16):
            values = (np.random.default_rng(9).standard_normal(200003) * 9.0
                      ).astype(dtype)
            grad = (np.random.default_rng(10).standard_normal(200003) * 3.0
                    ).astype(dtype)
            payload = self._device(values).requires_grad_(True)
            tensorplay.frac(payload).backward(self._device(grad))
            np.testing.assert_array_equal(
                payload.grad.cpu().numpy(), grad,
                err_msg=np.dtype(dtype).name)

    def test_a_view_is_answered_in_its_own_shape(self):
        base = (np.random.default_rng(13).standard_normal((4, 7, 5)) * 12.0
                ).astype(np.float32)
        payload = self._device(base)
        for label, view, want in (
                ("a transpose", payload.transpose(0, 2),
                 np.transpose(base, (2, 1, 0))),
                ("a strided slice", payload[:, 1:5, ::2], base[:, 1:5, ::2]),
                ("a narrow plane", payload.narrow(2, 1, 3), base[:, :, 1:4])):
            answer = tensorplay.frac(view)
            self.assertEqual(tuple(answer.shape), want.shape)
            np.testing.assert_allclose(
                answer.cpu().numpy(), self._want(want), rtol=0, atol=0,
                equal_nan=True, err_msg=label)

    def test_the_device_answers_what_the_host_answers(self):
        for dtype in (np.float32, np.float16):
            for length in self.LENGTHS:
                values = self._values(length, dtype, seed=length + 5)
                host = tensorplay.from_numpy(values.copy())
                np.testing.assert_allclose(
                    tensorplay.frac(self._device(values)).cpu().numpy(
                    ).astype(np.float64),
                    tensorplay.frac(host).numpy().astype(np.float64),
                    rtol=0, atol=0, equal_nan=True,
                    err_msg=f"frac over {np.dtype(dtype).name} length {length}")
                inplace_host = tensorplay.from_numpy(values.copy())
                np.testing.assert_allclose(
                    tensorplay.frac_(
                        self._device(values)).cpu().numpy().astype(np.float64),
                    tensorplay.frac_(inplace_host).numpy().astype(np.float64),
                    rtol=0, atol=0, equal_nan=True,
                    err_msg=f"frac_ over {np.dtype(dtype).name}")

    def test_a_payload_with_no_lane_answers_no_lane(self):
        for shape in ((0,), (0, 5), (2, 0, 3)):
            answer = tensorplay.frac(
                self._device(np.zeros(shape, np.float32)))
            self.assertEqual(tuple(answer.shape), shape)
            self.assertEqual(answer.numel(), 0)


class TestVulkanLogSigmoid(unittest.TestCase):
    # log_sigmoid answers min(x, 0) - log(1 + exp(-|x|)) in one pass, so a lane
    # far below zero keeps its value and a lane far above zero approaches zero
    # without the exponential overflowing.  Both storages need a case: a row
    # past the image-axis limit lives in a buffer, where a half payload packs
    # two lanes into one word.  A count is read through the float formula it
    # has no lane for, and the backward scales the gradient by sigmoid(-x).

    LENGTHS = (64, 200003)
    SPECIALS = np.array(
        [-80.0, -50.0, -20.0, -5.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 5.0, 20.0,
         50.0, 80.0, np.inf, -np.inf, np.nan], dtype=np.float32)

    def _values(self, count, dtype, seed=33):
        rng = np.random.default_rng(seed)
        values = (rng.standard_normal(count) * 9.0).astype(dtype)
        values[: self.SPECIALS.size] = self.SPECIALS.astype(dtype)
        return values

    def _device(self, values):
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _want(values):
        wide = values.astype(np.float64)
        return np.minimum(wide, 0.0) - np.log1p(np.exp(-np.abs(wide)))

    def test_the_fold_is_negative_softplus_over_both_storages(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            for length in self.LENGTHS:
                values = self._values(length, dtype)
                device = self._device(values)
                got = tensorplay.log_sigmoid(device)
                self.assertEqual(got.dtype, device.dtype)
                np.testing.assert_allclose(
                    got.cpu().numpy().astype(np.float64), self._want(values),
                    rtol=tol, atol=tol, equal_nan=True,
                    err_msg=f"{np.dtype(dtype).name} length {length}")
                # The allocating form leaves the payload it read untouched.
                np.testing.assert_array_equal(device.cpu().numpy(), values)

    def test_a_rank_three_or_four_payload_keeps_its_geometry(self):
        for shape, dtype in (((2, 3, 4), np.float16), ((2, 3, 2, 5), np.float32)):
            values = self._values(int(np.prod(shape)), dtype).reshape(shape)
            got = tensorplay.log_sigmoid(self._device(values))
            self.assertEqual(tuple(got.shape), shape)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), self._want(values),
                rtol=4e-3 if dtype == np.float16 else 2e-6,
                atol=4e-3 if dtype == np.float16 else 2e-6,
                equal_nan=True, err_msg=str(shape))

    def test_a_count_is_read_through_the_float_formula(self):
        raw = np.array([-40, -7, -1, 0, 1, 7, 40, 1 << 20, -(1 << 20)])
        for dtype in (np.int32, np.int64):
            values = raw.astype(dtype)
            device = self._device(values)
            got = tensorplay.log_sigmoid(device)
            self.assertEqual(got.dtype, tensorplay.float32)
            np.testing.assert_allclose(
                got.cpu().numpy(), self._want(values), rtol=1e-6, atol=1e-6,
                err_msg=np.dtype(dtype).name)
        # A count wide enough for linear storage takes the same road.
        long_values = (np.arange(200003, dtype=np.int32) % 41 - 20).astype(
            np.int32)
        got = tensorplay.log_sigmoid(self._device(long_values))
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64), self._want(long_values),
            rtol=1e-6, atol=1e-6)

    def test_the_gradient_scales_by_the_sigmoid_of_the_negated_input(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            for length in self.LENGTHS:
                values = self._values(length, dtype)
                grad = (np.random.default_rng(7).standard_normal(length) * 2.0
                        ).astype(dtype)
                device = tensorplay.log_sigmoid_backward(
                    self._device(grad), self._device(values))
                self.assertEqual(device.dtype, self._device(values).dtype)
                wide = values.astype(np.float64)
                want = grad.astype(np.float64) / (1.0 + np.exp(wide))
                np.testing.assert_allclose(
                    device.cpu().numpy().astype(np.float64), want,
                    rtol=tol, atol=tol, equal_nan=True,
                    err_msg=f"{np.dtype(dtype).name} length {length}")

    def test_the_device_answers_what_the_host_answers(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            values = self._values(48, dtype)
            grad = (np.random.default_rng(11).standard_normal(48) * 2.0
                    ).astype(dtype)
            host = tensorplay.from_numpy(values.copy())
            host_grad = tensorplay.from_numpy(grad.copy())
            device = host.to("vulkan")
            device_grad = host_grad.to("vulkan")
            np.testing.assert_allclose(
                tensorplay.log_sigmoid(device).cpu().numpy().astype(np.float64),
                tensorplay.log_sigmoid(host).numpy().astype(np.float64),
                rtol=tol, atol=tol, equal_nan=True,
                err_msg=f"log_sigmoid over {np.dtype(dtype).name}")
            np.testing.assert_allclose(
                tensorplay.log_sigmoid_backward(
                    device_grad, device).cpu().numpy().astype(np.float64),
                tensorplay.log_sigmoid_backward(
                    host_grad, host).numpy().astype(np.float64),
                rtol=tol, atol=tol, equal_nan=True,
                err_msg=f"log_sigmoid_backward over {np.dtype(dtype).name}")

class TestVulkanLogit(unittest.TestCase):
    # logit answers log(p / (1 - p)) in one walk, holding p inside
    # [eps, 1 - eps] when a bound is given.  A lane at either end of the unit
    # range has no finite ratio, a lane outside the range has no logarithm at
    # all without a bound, and a lane that is not a number stays one through
    # the clamp.  Both storages need a case, and a count is read through the
    # float formula it has no lane for.

    LENGTHS = (64, 200003)
    SPECIALS = np.array(
        [-np.inf, -2.0, -1.0, -1e-8, -0.0, 0.0, 1e-8, 0.01, 0.1, 0.5, 0.9,
         0.99, 1.0, 1.0 + 1e-7, 2.0, np.inf, np.nan], dtype=np.float32)

    def _values(self, count, dtype, seed=41):
        rng = np.random.default_rng(seed)
        values = rng.random(count).astype(dtype)
        values[: self.SPECIALS.size] = self.SPECIALS.astype(dtype)
        return values

    def _device(self, values):
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _want(values, eps=None):
        wide = values.astype(np.float64)
        if eps is not None:
            # The host fold keeps a lane that is not a number, so the
            # comparison-based bounds leave it alone too.
            wide = np.minimum(np.maximum(wide, eps), 1.0 - eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(wide / (1.0 - wide))

    @staticmethod
    def _grad_want(grad, values, eps=None):
        g = grad.astype(np.float64)
        s = values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = g / (s * (1.0 - s))
        # Either end carries the whole slope rather than the quotient.
        endpoint = np.where((s == 0.0) | (s == 1.0), g * np.inf, slope)
        if eps is None:
            return np.where((s < 0.0) | (s > 1.0), np.nan, endpoint)
        # The bounds are read back at the payload's own width before the
        # comparison, so a single-precision payload rounds them first.
        lo = np.float32(eps)
        hi = np.float32(np.float32(1.0) - lo)
        return np.where((s < lo) | (s > hi), 0.0, endpoint)

    def test_the_odd_ratio_is_one_walk_over_both_storages(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            for length in self.LENGTHS:
                values = self._values(length, dtype)
                device = self._device(values)
                got = tensorplay.logit(device)
                self.assertEqual(got.dtype, device.dtype)
                np.testing.assert_allclose(
                    got.cpu().numpy().astype(np.float64), self._want(values),
                    rtol=tol, atol=tol, equal_nan=True,
                    err_msg=f"{np.dtype(dtype).name} length {length}")
                # The allocating form leaves the payload it read untouched.
                np.testing.assert_array_equal(device.cpu().numpy(), values)

    def test_a_rank_three_or_four_payload_keeps_its_geometry(self):
        for shape, dtype in (((2, 3, 4), np.float16), ((2, 3, 2, 5), np.float32)):
            values = self._values(int(np.prod(shape)), dtype).reshape(shape)
            got = tensorplay.logit(self._device(values))
            self.assertEqual(tuple(got.shape), shape)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), self._want(values),
                rtol=4e-3 if dtype == np.float16 else 2e-6,
                atol=4e-3 if dtype == np.float16 else 2e-6,
                equal_nan=True, err_msg=str(shape))

    def test_a_count_is_read_through_the_float_formula(self):
        raw = np.array([-7, -1, 0, 1, 2, 1 << 20, -(1 << 20)])
        for dtype in (np.int32, np.int64):
            values = raw.astype(dtype)
            device = self._device(values)
            got = tensorplay.logit(device)
            self.assertEqual(got.dtype, tensorplay.float32)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), self._want(values),
                rtol=1e-6, atol=1e-6, equal_nan=True,
                err_msg=np.dtype(dtype).name)
        # A count wide enough for linear storage takes the same road.
        long_values = (np.arange(200003, dtype=np.int32) % 5 - 2).astype(
            np.int32)
        got = tensorplay.logit(self._device(long_values))
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64), self._want(long_values),
            rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_a_bound_holds_every_lane_inside_the_band(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            for length in self.LENGTHS:
                values = self._values(length, dtype)
                device = self._device(values)
                for eps in (0.01, 0.0):
                    got = tensorplay.logit(device, eps=eps)
                    np.testing.assert_allclose(
                        got.cpu().numpy().astype(np.float64),
                        self._want(values, eps=eps),
                        rtol=tol, atol=tol, equal_nan=True,
                        err_msg=f"eps={eps} over {np.dtype(dtype).name} "
                                f"length {length}")

    def test_the_in_place_form_rewrites_the_same_payload(self):
        for dtype, length in ((np.float32, 64), (np.float16, 200003)):
            values = self._values(length, dtype)
            device = self._device(values)
            device.logit_(eps=0.02)
            tol = 4e-3 if dtype == np.float16 else 2e-6
            np.testing.assert_allclose(
                device.cpu().numpy().astype(np.float64),
                self._want(values, eps=0.02), rtol=tol, atol=tol,
                equal_nan=True, err_msg=np.dtype(dtype).name)

    def test_the_gradient_follows_the_inverse_slope(self):
        values = np.array(
            [-1.0, -0.0, 0.0, 1e-8, 0.005, 0.5, 0.995, 1.0, 1.0 + 1e-7,
             2.0, np.nan], dtype=np.float32)
        grad = np.array(
            [2.0, 0.0, -3.0, 1.0, 0.5, -1.5, 2.5, 0.0, 1.0, -2.0, 1.0],
            dtype=np.float32)
        for eps in (None, 0.01, 0.0):
            got = tensorplay.logit_backward(
                self._device(grad), self._device(values), eps=eps)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64),
                self._grad_want(grad, values, eps),
                rtol=2e-6, atol=2e-6, equal_nan=True, err_msg=f"eps={eps}")
        # The same lanes over linear storage.
        wide_values = np.resize(values, 200003)
        wide_grad = np.resize(grad, 200003)
        for eps in (None, 0.01):
            got = tensorplay.logit_backward(
                self._device(wide_grad), self._device(wide_values), eps=eps)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64),
                self._grad_want(wide_grad, wide_values, eps),
                rtol=2e-6, atol=2e-6, equal_nan=True,
                err_msg=f"eps={eps} over linear storage")

    def test_the_device_answers_what_the_host_answers(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            values = self._values(48, dtype)
            grad = (np.random.default_rng(13).standard_normal(48) * 2.0
                    ).astype(dtype)
            host = tensorplay.from_numpy(values.copy())
            host_grad = tensorplay.from_numpy(grad.copy())
            device = host.to("vulkan")
            device_grad = host_grad.to("vulkan")
            for eps in (None, 0.01, 0.0):
                np.testing.assert_allclose(
                    tensorplay.logit(
                        device, eps=eps).cpu().numpy().astype(np.float64),
                    tensorplay.logit(
                        host, eps=eps).numpy().astype(np.float64),
                    rtol=tol, atol=tol, equal_nan=True,
                    err_msg=f"logit eps={eps} over {np.dtype(dtype).name}")
                np.testing.assert_allclose(
                    tensorplay.logit_backward(
                        device_grad, device,
                        eps=eps).cpu().numpy().astype(np.float64),
                    tensorplay.logit_backward(
                        host_grad, host, eps=eps).numpy().astype(np.float64),
                    rtol=tol, atol=tol, equal_nan=True,
                    err_msg=f"logit_backward eps={eps} over "
                            f"{np.dtype(dtype).name}")


class TestVulkanLogCumsumExp(unittest.TestCase):
    # logcumsumexp answers the logarithm of the running sum of exponentiated
    # elements along one axis, so every position depends on its whole prefix.
    # The walk leads with the larger of the pair, answers with the incoming
    # element when both are the same infinity, and lets an accumulator that is
    # not a number stay one.  Both storages and both widths are covered, as is
    # the shape-less input that has no axis to walk.

    SHAPES = (
        ((64,), 0),
        ((3, 7), 0),
        ((3, 7), -1),
        ((2, 3, 4), 1),
        ((2, 2, 3, 5), -1),
    )
    SPECIALS = np.array(
        [-np.inf, -100.0, -1.0, -0.0, 0.0, 1.0, 100.0, np.inf, np.nan],
        dtype=np.float32)

    def _values(self, shape, dtype, seed=57):
        rng = np.random.default_rng(seed)
        count = int(np.prod(shape))
        values = (rng.standard_normal(count) * 4.0).astype(dtype)
        values[: self.SPECIALS.size] = self.SPECIALS.astype(dtype)
        return values.reshape(shape)

    def _device(self, values):
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _want(values, dim):
        # Each step keeps the larger operand as the lead term and carries the
        # smaller in a correction, all in single precision, which is the width
        # the device walk accumulates in.  A half result is the rounded store
        # of that accumulator at every position.
        wide = values.astype(np.float32)
        moved = np.moveaxis(wide, dim, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        out = np.empty_like(flat)
        for row in range(flat.shape[0]):
            acc = -np.inf
            for i in range(flat.shape[1]):
                x = float(flat[row, i])
                if math.isnan(acc):
                    acc = float("nan")
                else:
                    lo = acc if acc < x else x
                    hi = acc if x < acc else x
                    if lo != hi or math.isfinite(lo):
                        acc = np.float32(math.log1p(math.exp(lo - hi)) + hi)
                    else:
                        acc = np.float32(x)
                out[row, i] = acc
        result = np.moveaxis(out.reshape(moved.shape), -1, dim)
        return result.astype(values.dtype, copy=False)

    def test_the_scan_follows_the_running_log_sum_exp(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            for shape, dim in self.SHAPES:
                values = self._values(shape, dtype)
                device = self._device(values)
                got = tensorplay.logcumsumexp(device, dim)
                self.assertEqual(got.dtype, device.dtype)
                np.testing.assert_allclose(
                    got.cpu().numpy().astype(np.float64),
                    self._want(values, dim).astype(np.float64),
                    rtol=tol, atol=tol, equal_nan=True,
                    err_msg=f"{np.dtype(dtype).name} {shape} dim {dim}")
                # The allocating form leaves the payload it read untouched.
                np.testing.assert_array_equal(device.cpu().numpy(), values)

    def test_the_special_lanes_answer_the_reference(self):
        rows = np.array([
            [-np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf],
            [-np.inf, 2.0, -np.inf],
            [np.nan, -np.inf, 0.5],
            [-np.inf, np.nan, -np.inf],
            [np.inf, -np.inf, 0.0],
        ], dtype=np.float32)
        for dtype in (np.float32, np.float16):
            values = rows.astype(dtype)
            got = tensorplay.logcumsumexp(self._device(values), 1)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64),
                self._want(values, 1).astype(np.float64),
                rtol=2e-6, atol=2e-6, equal_nan=True,
                err_msg=np.dtype(dtype).name)

    def test_linear_storage_walks_its_rows_the_same_way(self):
        # A payload past the image-axis limit lives in a buffer; the long axis
        # keeps its rows short, and the row itself stays a single walk.
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            values = self._values((50000, 4), dtype)
            got = tensorplay.logcumsumexp(self._device(values), 1)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64),
                self._want(values, 1).astype(np.float64),
                rtol=tol, atol=tol, equal_nan=True,
                err_msg=np.dtype(dtype).name)
        long_values = (
            np.random.default_rng(5).standard_normal(200003) * 3.0
        ).astype(np.float32)
        got = tensorplay.logcumsumexp(self._device(long_values), 0)
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64),
            self._want(long_values, 0).astype(np.float64),
            rtol=5e-5, atol=5e-5, equal_nan=True)

    def test_a_requested_width_sets_the_result(self):
        values = self._values((3, 7), np.float32)
        got = tensorplay.logcumsumexp(
            self._device(values), 0, dtype=tensorplay.float16)
        self.assertEqual(got.dtype, tensorplay.float16)
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64),
            self._want(values.astype(np.float16), 0).astype(np.float64),
            rtol=4e-3, atol=4e-3, equal_nan=True)
        half = self._values((3, 7), np.float16)
        got = tensorplay.logcumsumexp(
            self._device(half), -1, dtype=tensorplay.float32)
        self.assertEqual(got.dtype, tensorplay.float32)
        # The walk keeps the requested width, so the result is not rounded
        # back to the payload's own width.
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64),
            self._want(half.astype(np.float32), -1).astype(np.float64),
            rtol=2e-6, atol=2e-6, equal_nan=True)

    def test_a_scalar_has_no_axis_and_answers_with_itself(self):
        host = tensorplay.from_numpy(np.array(2.5, dtype=np.float32))
        device = host.to("vulkan")
        for dim in (0, -1):
            got = tensorplay.logcumsumexp(device, dim)
            self.assertEqual(tuple(got.shape), ())
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64), np.float64(2.5),
                rtol=2e-6, atol=2e-6)
        narrow = tensorplay.logcumsumexp(device, 0, dtype=tensorplay.float16)
        self.assertEqual(narrow.dtype, tensorplay.float16)
        self.assertAlmostEqual(float(narrow.cpu().numpy()), 2.5, places=3)

    def test_an_empty_payload_keeps_its_shape(self):
        for shape, dim in (((0, 3), 1), ((3, 0), 0), ((2, 0, 4), -2)):
            values = np.zeros(shape, dtype=np.float32)
            got = tensorplay.logcumsumexp(self._device(values), dim)
            self.assertEqual(tuple(got.shape), shape)

    def test_a_count_is_refused(self):
        values = np.arange(6, dtype=np.int32).reshape(2, 3)
        with self.assertRaises(TypeError):
            tensorplay.logcumsumexp(self._device(values), 1)

    def test_the_device_answers_what_the_host_answers(self):
        for dtype, tol in ((np.float32, 2e-6), (np.float16, 4e-3)):
            values = self._values((6, 8), dtype)
            host = tensorplay.from_numpy(values.copy())
            device = host.to("vulkan")
            np.testing.assert_allclose(
                tensorplay.logcumsumexp(
                    device, 1).cpu().numpy().astype(np.float64),
                tensorplay.logcumsumexp(
                    host, 1).numpy().astype(np.float64),
                rtol=tol, atol=tol, equal_nan=True,
                err_msg=np.dtype(dtype).name)


class TestVulkanCumExtremum(unittest.TestCase):
    # cummax and cummin answer the running extremum along one axis together
    # with the position each running value came from.  An incoming non-number
    # takes the row over and never yields, and a later element that ties the
    # running winner takes over too, so the index names the latest position
    # attaining the extremum.  The values keep the payload's width; the places
    # answer in the backend's index width, a single word.  Both storages, both
    # float widths, the counting widths and the shape-less form are covered.

    SHAPES = (
        ((64,), 0),
        ((3, 7), 0),
        ((3, 7), -1),
        ((2, 3, 4), 1),
        ((2, 2, 3, 5), -1),
    )

    def _device(self, values):
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _want(values, dim, is_max):
        floating = np.issubdtype(values.dtype, np.floating)
        moved = np.moveaxis(values, dim, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        out = np.empty_like(flat)
        places = np.empty(flat.shape, dtype=np.int32)
        for row in range(flat.shape[0]):
            best = flat[row, 0]
            best_at = 0
            out[row, 0] = best
            places[row, 0] = 0
            for i in range(1, flat.shape[1]):
                current = flat[row, i]
                current_nan = floating and bool(np.isnan(current))
                best_nan = floating and bool(np.isnan(best))
                wins = current_nan or (
                    not best_nan
                    and (current >= best if is_max else current <= best))
                if wins:
                    best = current
                    best_at = i
                out[row, i] = best
                places[row, i] = best_at
        return (
            np.moveaxis(out.reshape(moved.shape), -1, dim),
            np.moveaxis(places.reshape(moved.shape), -1, dim))

    @staticmethod
    def _check(case, values, dim, is_max):
        want_values, want_places = case._want(values, dim, is_max)
        op = tensorplay.cummax if is_max else tensorplay.cummin
        device = case._device(values)
        got = op(device, dim)
        case.assertEqual(got.values.dtype, device.dtype)
        case.assertEqual(got.indices.dtype, tensorplay.int32)
        if np.issubdtype(values.dtype, np.floating):
            np.testing.assert_allclose(
                got.values.cpu().numpy().astype(np.float64),
                want_values.astype(np.float64),
                rtol=0, atol=0, equal_nan=True,
                err_msg=f"{np.dtype(values.dtype).name} {values.shape} "
                        f"dim {dim} max {is_max}")
        else:
            np.testing.assert_array_equal(
                got.values.cpu().numpy(), want_values,
                err_msg=f"{np.dtype(values.dtype).name} {values.shape} "
                        f"dim {dim} max {is_max}")
        np.testing.assert_array_equal(
            got.indices.cpu().numpy(), want_places,
            err_msg=f"{np.dtype(values.dtype).name} {values.shape} "
                    f"dim {dim} max {is_max}")

    def _random(self, shape, dtype, seed=61):
        rng = np.random.default_rng(seed)
        count = int(np.prod(shape))
        if np.issubdtype(dtype, np.floating):
            values = (rng.standard_normal(count) * 4.0).astype(dtype)
            values[:4] = np.array(
                [-np.inf, np.nan, -0.0, np.inf], dtype=dtype)
        else:
            info = np.iinfo(dtype)
            values = rng.integers(
                info.min, info.max, count, dtype=dtype)
        return values.reshape(shape)

    def test_the_scan_follows_the_running_extremum(self):
        for dtype in (np.float32, np.float16):
            for shape, dim in self.SHAPES:
                values = self._random(shape, dtype)
                for is_max in (True, False):
                    self._check(self, values, dim, is_max)
                # The allocating forms leave the payload they read untouched.
                device = self._device(values)
                tensorplay.cummax(device, dim)
                np.testing.assert_allclose(
                    device.cpu().numpy().astype(np.float64),
                    values.astype(np.float64),
                    rtol=0, atol=0, equal_nan=True)

    def test_a_later_tie_takes_over(self):
        rows = np.array([
            [3.0, 1.0, 3.0, 3.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [-1.0, -1.0, 5.0, 5.0, -1.0],
        ], dtype=np.float32)
        got_max = tensorplay.cummax(self._device(rows), 1)
        got_min = tensorplay.cummin(self._device(rows), 1)
        np.testing.assert_array_equal(
            got_max.indices.cpu().numpy(),
            np.array([[0, 0, 2, 3, 3],
                      [0, 1, 2, 3, 4],
                      [0, 1, 2, 3, 3]], dtype=np.int32))
        np.testing.assert_array_equal(
            got_min.indices.cpu().numpy(),
            np.array([[0, 1, 1, 1, 4],
                      [0, 1, 2, 3, 4],
                      [0, 1, 1, 1, 4]], dtype=np.int32))
        self._check(self, rows, 1, True)
        self._check(self, rows, 1, False)

    def test_a_non_number_keeps_the_row(self):
        rows = np.array([
            [-np.inf, np.nan, 5.0, 0.0],
            [np.nan, np.nan, 1.0, 0.0],
            [1.0, np.nan, np.nan, 7.0],
            [np.nan, -np.inf, 0.5, -2.0],
        ], dtype=np.float32)
        for dtype in (np.float32, np.float16):
            payload = rows.astype(dtype)
            for dim in (0, 1):
                for is_max in (True, False):
                    self._check(self, payload, dim, is_max)
        got = tensorplay.cummax(self._device(rows), 1)
        self.assertTrue(np.isnan(got.values.cpu().numpy()[2, 1:]).all())
        # A non-number always takes over, so a run of them walks the index
        # forward and the ordinary value after them never wins it back.
        np.testing.assert_array_equal(
            got.indices.cpu().numpy()[2, 1:], np.array([1, 2, 2]))

    def test_a_wide_count_folds_in_its_own_type(self):
        big = 1 << 60
        base = np.array([
            [big + 5, big + 3, -(1 << 61)],
            [big + 1, -(1 << 62), big + 9],
            [big + 7, big + 4, -(1 << 63)],
        ], dtype=np.int64)
        for dim in (0, 1, -1):
            for is_max in (True, False):
                self._check(self, base, dim, is_max)
        # The same values through single precision would lose their order.
        got = tensorplay.cummax(self._device(base), 0)
        self.assertEqual(got.values.dtype, tensorplay.int64)
        np.testing.assert_array_equal(
            got.values.cpu().numpy(),
            np.array([[big + 5, big + 3, -(1 << 61)],
                      [big + 5, big + 3, big + 9],
                      [big + 7, big + 4, big + 9]], dtype=np.int64))
        np.testing.assert_array_equal(
            got.indices.cpu().numpy(),
            np.array([[0, 0, 0], [0, 0, 1], [2, 2, 1]], dtype=np.int32))

    def test_a_byte_lane_keeps_its_width(self):
        for dtype in (np.int8, np.uint8):
            values = np.array([
                [-3, 5, -128, 127, 0],
                [9, -9, 9, 9, -9],
            ], dtype=np.int64)
            if dtype is np.uint8:
                values = np.abs(values)
            values = values.astype(dtype)
            for dim in (0, 1):
                for is_max in (True, False):
                    self._check(self, values, dim, is_max)
        mask = np.array([
            [True, False, True, True],
            [False, False, True, False],
        ], dtype=bool)
        for is_max in (True, False):
            self._check(self, mask, 1, is_max)

    def test_an_axis_beyond_the_payload_refuses(self):
        values = np.zeros((3, 4), dtype=np.float32)
        for dim in (2, -3):
            with self.assertRaisesRegex(
                    RuntimeError, "dimension out of range"):
                tensorplay.cummax(self._device(values), dim)
            with self.assertRaisesRegex(
                    RuntimeError, "dimension out of range"):
                tensorplay.cummin(self._device(values), dim)

    def test_linear_storage_walks_its_rows_the_same_way(self):
        for dtype in (np.float32, np.float16):
            values = self._random((50000, 4), dtype)
            for is_max in (True, False):
                self._check(self, values, 1, is_max)
        # A long row past every image limit stays a single walk.
        long_values = (
            np.random.default_rng(7).standard_normal(200003) * 3.0
        ).astype(np.float32)
        for is_max in (True, False):
            self._check(self, long_values, 0, is_max)

    def test_a_scalar_has_no_axis_and_answers_with_itself(self):
        device = self._device(np.array(2.5, dtype=np.float32))
        for dim in (0, -1):
            for op in (tensorplay.cummax, tensorplay.cummin):
                got = op(device, dim)
                self.assertEqual(tuple(got.values.shape), ())
                self.assertEqual(tuple(got.indices.shape), ())
                self.assertAlmostEqual(float(got.values.cpu().numpy()), 2.5)
                self.assertEqual(int(got.indices.cpu().numpy()), 0)
        with self.assertRaisesRegex(RuntimeError, "dimension out of range"):
            tensorplay.cummax(device, 1)

    def test_an_empty_payload_keeps_its_shape(self):
        for shape, dim in (((0, 3), 1), ((3, 0), 0), ((2, 0, 4), -2)):
            values = np.zeros(shape, dtype=np.float32)
            for op in (tensorplay.cummax, tensorplay.cummin):
                got = op(self._device(values), dim)
                self.assertEqual(tuple(got.values.shape), shape)
                self.assertEqual(tuple(got.indices.shape), shape)

    def test_the_device_answers_what_the_host_answers(self):
        for dtype in (np.float32, np.float16, np.int32, np.int64):
            values = self._random((6, 8), dtype)
            host = tensorplay.from_numpy(values.copy())
            device = host.to("vulkan")
            for dim in (0, 1):
                for op in (tensorplay.cummax, tensorplay.cummin):
                    got = op(device, dim)
                    want = op(host, dim)
                    if np.issubdtype(dtype, np.floating):
                        np.testing.assert_allclose(
                            got.values.cpu().numpy().astype(np.float64),
                            want.values.numpy().astype(np.float64),
                            rtol=0, atol=0, equal_nan=True)
                    else:
                        np.testing.assert_array_equal(
                            got.values.cpu().numpy(),
                            want.values.numpy())
                    np.testing.assert_array_equal(
                        got.indices.cpu().numpy(),
                        want.indices.numpy().astype(np.int32))


class TestVulkanMoments(unittest.TestCase):
    # var_mean and std_mean answer the mean and the centered second moment in
    # one call.  The variance divides the squared deviations by the element
    # count less the correction, floored at zero: one element without a
    # correction has no degrees of freedom left, so its variance is NaN, while
    # a correction past the count leaves a positive deviation sum over a zero
    # divisor.  An empty reduced axis answers NaN and an empty unreduced axis
    # leaves an empty answer.  A scalar has no axis at all, so its variance is
    # NaN while its mean is the value itself.  Counts and truth values widen to
    # single precision; narrow floats narrow only their answers.  Both
    # storages answer alike.

    CASES = (
        ((3, 5), [0], False),
        ((3, 5), [-1], False),
        ((2, 3, 4), [1], True),
        ((2, 3, 4), [0, 2], False),
        ((2, 3, 4), [], False),
        ((2, 3, 4), [1, 1], False),
    )

    @staticmethod
    def _device(values, dense=False):
        if dense:
            return tensorplay.tensor(values.copy(), device="vulkan")
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(values.copy())

    def _compare(self, values, dim, keepdim, dense=False, tol=1e-5):
        got_var, got_mean = tensorplay.var_mean(
            self._device(values, dense), dim, keepdim=keepdim)
        want_var, want_mean = tensorplay.var_mean(
            self._host(values), dim, keepdim=keepdim)
        self.assertEqual(tuple(got_var.shape), tuple(want_var.shape))
        self.assertEqual(tuple(got_mean.shape), tuple(want_mean.shape))
        np.testing.assert_allclose(
            got_var.cpu().numpy().astype(np.float64),
            want_var.numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True)
        np.testing.assert_allclose(
            got_mean.cpu().numpy().astype(np.float64),
            want_mean.numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True)
        return got_var, got_mean

    def test_the_pair_follows_the_host_over_every_axis_set(self):
        rng = np.random.default_rng(29)
        for shape, dim, keepdim in self.CASES:
            for dtype, tol in ((np.float32, 1e-5), (np.float16, 2e-3)):
                values = rng.standard_normal(shape).astype(dtype)
                with self.subTest(shape=shape, dim=dim, dtype=dtype):
                    got_var, got_mean = self._compare(
                        values, dim, keepdim, tol=tol)
                    want_dtype = (tensorplay.float32
                                  if dtype == np.float32
                                  else tensorplay.float16)
                    self.assertEqual(got_var.dtype, want_dtype)
                    self.assertEqual(got_mean.dtype, want_dtype)

    def test_counts_and_truth_values_answer_in_single_precision(self):
        rng = np.random.default_rng(31)
        counts = rng.integers(-8, 8, (4, 6)).astype(np.int32)
        got_var, got_mean = self._compare(counts, [1], False)
        self.assertEqual(got_var.dtype, tensorplay.float32)
        self.assertEqual(got_mean.dtype, tensorplay.float32)
        truth = rng.integers(0, 2, (4, 6)).astype(bool)
        got_var, got_mean = self._compare(truth, [0], False)
        self.assertEqual(got_var.dtype, tensorplay.float32)
        self.assertEqual(got_mean.dtype, tensorplay.float32)

    def test_a_line_without_degrees_of_freedom_answers_nan(self):
        one = self._device(np.array([5.0], np.float32))
        var, mean = tensorplay.var_mean(one, [0])
        self.assertTrue(math.isnan(float(var.cpu())))
        self.assertEqual(float(mean.cpu()), 5.0)
        std, _ = tensorplay.std_mean(one, [0])
        self.assertTrue(math.isnan(float(std.cpu())))
        self.assertTrue(math.isnan(float(tensorplay.var(one, [0]).cpu())))
        self.assertTrue(math.isnan(float(tensorplay.std(one, [0]).cpu())))

        var, _ = tensorplay.var_mean(one, [0], False)
        self.assertEqual(float(var.cpu()), 0.0)

        # A correction past the count keeps a positive deviation sum over a
        # zero divisor, which answers infinity.
        two = self._device(np.array([5.0, 7.0], np.float32))
        var, mean = tensorplay.var_mean(two, [0], correction=3)
        self.assertTrue(math.isinf(float(var.cpu())))
        self.assertEqual(float(mean.cpu()), 6.0)

    def test_a_shape_less_payload_has_no_axis(self):
        scalar = tensorplay.tensor(3.0, device="vulkan")
        var, mean = tensorplay.var_mean(scalar, [])
        self.assertTrue(math.isnan(float(var.cpu())))
        self.assertEqual(float(mean.cpu()), 3.0)
        var_keep, mean_keep = tensorplay.var_mean(scalar, [], keepdim=True)
        self.assertEqual(tuple(var_keep.shape), ())
        self.assertEqual(tuple(mean_keep.shape), ())
        with self.assertRaises(RuntimeError):
            tensorplay.var_mean(scalar, [0])

    def test_an_empty_span_answers_nan_or_emptiness(self):
        for shape, dim, out_shape in (
                ((0, 3), [0], (3,)),
                ((2, 0), [1], (2,)),
                ((0, 3), [1], (0,)),
                ((2, 0), [], ())):
            payload = tensorplay.zeros(*shape, device="vulkan")
            with self.subTest(shape=shape, dim=dim):
                var, mean = tensorplay.var_mean(payload, dim)
                self.assertEqual(tuple(var.shape), out_shape)
                self.assertEqual(tuple(mean.shape), out_shape)
                got = var.cpu().numpy()
                if got.size:
                    self.assertTrue(np.isnan(got).all())
                    self.assertTrue(np.isnan(mean.cpu().numpy()).all())

    def test_the_word_storage_answers_the_same(self):
        rng = np.random.default_rng(37)
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 2e-3)):
            values = rng.standard_normal((5, 7, 3)).astype(dtype)
            for dim, keepdim in (([0], False), ([2], True), ([], False)):
                with self.subTest(dtype=dtype, dim=dim):
                    self._compare(values, dim, keepdim, dense=True, tol=tol)

    def test_std_mean_is_the_root_of_the_variance(self):
        rng = np.random.default_rng(41)
        values = rng.standard_normal((4, 9)).astype(np.float32)
        device = self._device(values)
        std, mean = tensorplay.std_mean(device, [1])
        var, var_mean_value = tensorplay.var_mean(device, [1])
        np.testing.assert_allclose(
            std.cpu().numpy(), np.sqrt(var.cpu().numpy()),
            rtol=1e-6, atol=1e-6, equal_nan=True)
        np.testing.assert_array_equal(
            mean.cpu().numpy(), var_mean_value.cpu().numpy())
        want_std, want_mean = tensorplay.std_mean(self._host(values), [1])
        np.testing.assert_allclose(
            std.cpu().numpy(), want_std.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            mean.cpu().numpy(), want_mean.numpy(), rtol=1e-5, atol=1e-5)


class TestVulkanChannelReduction(unittest.TestCase):
    # The channel axis is the third from the end, and four consecutive
    # channels share one texel.  A channel count that is not a multiple of
    # four leaves padding lanes in the last texel, while a count above four
    # spans several texels of the same line.  Sum and mean must ignore the
    # padding lanes, and variance must fold the whole line into one mean
    # before squaring the deviations: squaring per-texel means drops the
    # cross-lane terms.  Both storage classes answer the same values.

    SHAPES = (
        ((5, 3, 7), [0]),   # tail of three padding lanes
        ((6, 3, 7), [0]),   # tail of two padding lanes
        ((8, 3, 7), [0]),   # no padding, two texels per line
        ((2, 5, 3, 4), [1]),
    )

    @staticmethod
    def _device(values, dense=False):
        if dense:
            return tensorplay.tensor(values.copy(), device="vulkan")
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(values.copy())

    def _compare(self, name, values, dim, keepdim, dense, tol):
        device = self._device(values, dense)
        host = self._host(values)
        if name in ("sum", "mean"):
            got = getattr(device, name)(dim, keepdim=keepdim)
            want = getattr(host, name)(dim, keepdim=keepdim)
        elif name == "var":
            got = tensorplay.var(device, dim, keepdim=keepdim)
            want = tensorplay.var(host, dim, keepdim=keepdim)
        else:
            got = tensorplay.std(device, dim, keepdim=keepdim)
            want = tensorplay.std(host, dim, keepdim=keepdim)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64),
            want.numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True)
        return got

    def test_sum_and_mean_ignore_the_padding_lanes(self):
        rng = np.random.default_rng(43)
        for shape, dim in self.SHAPES:
            values = rng.standard_normal(shape).astype(np.float32)
            for keepdim in (False, True):
                for name in ("sum", "mean"):
                    with self.subTest(
                            shape=shape, dim=dim, name=name,
                            keepdim=keepdim):
                        self._compare(
                            name, values, dim, keepdim, False, 1e-5)

    def test_the_variance_folds_the_whole_channel_line(self):
        rng = np.random.default_rng(47)
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            for shape, dim in self.SHAPES:
                values = rng.standard_normal(shape).astype(dtype)
                for keepdim in (False, True):
                    for name in ("var", "std"):
                        with self.subTest(
                                shape=shape, dim=dim, name=name,
                                dtype=dtype, keepdim=keepdim):
                            self._compare(
                                name, values, dim, keepdim, False, tol)

    def test_a_channel_line_honours_the_correction(self):
        rng = np.random.default_rng(53)
        # The narrow fused walk divides on the device, so every reachable
        # correction is checked against the widened host composition.
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            values = rng.standard_normal((5, 3, 7)).astype(dtype)
            for correction in (0, 2, 4):
                with self.subTest(dtype=dtype, correction=correction):
                    got = tensorplay.var(
                        self._device(values), [0], correction=correction)
                    want = tensorplay.var(
                        self._host(values), [0], correction=correction)
                    np.testing.assert_allclose(
                        got.cpu().numpy().astype(np.float64),
                        want.numpy().astype(np.float64),
                        rtol=tol, atol=tol, equal_nan=True)

    def test_a_line_without_degrees_of_freedom_answers_nan_or_infinity(self):
        line = self._device(
            np.arange(1, 6, dtype=np.float16).reshape(5, 1, 1))
        self.assertTrue(math.isinf(
            float(tensorplay.var(line, [0], correction=5).cpu())))
        flat = self._device(
            np.full((5, 1, 1), 2.5, dtype=np.float16))
        self.assertTrue(math.isnan(
            float(tensorplay.var(flat, [0], correction=5).cpu())))

    def test_the_word_storage_agrees_on_a_channel_axis(self):
        rng = np.random.default_rng(59)
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            for shape, dim in (((5, 3, 7), [0]), ((2, 5, 3, 4), [1])):
                values = rng.standard_normal(shape).astype(dtype)
                for name in ("sum", "mean", "var", "std"):
                    with self.subTest(
                            shape=shape, dim=dim, name=name, dtype=dtype):
                        self._compare(
                            name, values, dim, False, True, tol)


class TestVulkanChannelSoftmax(unittest.TestCase):
    # Softmax over the channel axis shares one maximum and one sum across the
    # four lanes of a texel.  The channel count comes from the tensor shape,
    # and a count that is not a multiple of four leaves padding lanes in the
    # last texel; neither the maximum nor the sum may see them.  Counts on
    # either side of a texel boundary are checked over both storage classes.

    CASES = (
        ((2, 3, 7), 0),
        ((3, 3, 7), 0),
        ((4, 3, 7), 0),
        ((5, 3, 7), 0),
        ((6, 3, 7), 0),
        ((8, 3, 7), 0),
        ((2, 5, 3, 4), 1),
    )

    @staticmethod
    def _device(values, dense=False):
        if dense:
            return tensorplay.tensor(values.copy(), device="vulkan")
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    def _compare(self, name, fn, values, dim, dense, tol):
        device = self._device(values, dense)
        host = tensorplay.from_numpy(values.copy())
        np.testing.assert_allclose(
            fn(device, dim).cpu().numpy().astype(np.float64),
            fn(host, dim).numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True,
            err_msg=f"{name} over a channel axis of shape {values.shape}")

    def test_the_channel_row_normalizes_over_every_lane(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(61)
        for shape, dim in self.CASES:
            for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
                values = rng.standard_normal(shape).astype(dtype)
                for name, fn in (("softmax", F.softmax),
                                 ("log_softmax", F.log_softmax)):
                    with self.subTest(
                            shape=shape, dim=dim, name=name, dtype=dtype):
                        self._compare(name, fn, values, dim, False, tol)

    def test_the_word_storage_agrees_on_a_channel_row(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(67)
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            values = rng.standard_normal((5, 3, 7)).astype(dtype)
            for name, fn in (("softmax", F.softmax),
                             ("log_softmax", F.log_softmax)):
                with self.subTest(name=name, dtype=dtype):
                    self._compare(name, fn, values, 0, True, tol)


class TestVulkanVectorNorm(unittest.TestCase):
    # A vector norm folds the magnitudes of a span.  The order picks the fold:
    # the zero order counts the lanes that are not zero, where a lane that is
    # not a number is not zero and so counts; the unit order sums magnitudes;
    # every other finite order sums the magnitudes raised to that power before
    # the root; and the two infinite orders fold the largest and the smallest
    # magnitude, both of which carry a special lane outward.  An empty span
    # answers with the identity of its fold, zero for a sum or a largest
    # magnitude and infinity for a smallest one; a walk over named axes, and a
    # whole payload that keeps its axes, then take the root of that identity,
    # so a negative order there answers infinity, while a whole payload that
    # sheds every axis reports the identity itself.  An empty axis that
    # survives the walk leaves an empty answer.  A scalar has no axis at all,
    # so its norm is its magnitude.  A half payload widens for the walk and
    # narrows only its answer; a count has no magnitude to measure and is
    # refused.  Both storages answer alike.

    CASES = (
        ((3, 5), None, False),
        ((3, 5), [0], False),
        ((3, 5), [-1], True),
        ((2, 3, 4), [1], True),
        ((2, 3, 4), [0, 2], False),
        ((2, 3, 4), [], False),
        ((2, 3, 4, 2), [-1, 1], True),
    )
    ORDERS = (0, 1, 2, 3, -1.5, float("inf"), float("-inf"))

    @staticmethod
    def _device(values, dense=False):
        if dense:
            return tensorplay.tensor(values.copy(), device="vulkan")
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(values.copy())

    def _compare(self, values, ord, dim, keepdim, dense=False, tol=1e-5):
        got = tensorplay.linalg_vector_norm(
            self._device(values, dense), ord, dim, keepdim)
        want = tensorplay.linalg_vector_norm(
            self._host(values), ord, dim, keepdim)
        self.assertEqual(got.dtype, want.dtype)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64),
            want.numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True,
            err_msg=f"order {ord} over dims {dim} of shape {values.shape}")
        return got

    def test_the_norm_follows_the_host_over_every_order_and_axis_set(self):
        rng = np.random.default_rng(71)
        for shape, dim, keepdim in self.CASES:
            for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
                values = rng.standard_normal(shape).astype(dtype)
                for ord in self.ORDERS:
                    with self.subTest(shape=shape, dim=dim, ord=ord,
                                      dtype=dtype):
                        self._compare(values, ord, dim, keepdim, tol=tol)

    def test_both_storages_answer_the_same_norm(self):
        rng = np.random.default_rng(73)
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            values = rng.standard_normal((4, 6)).astype(dtype)
            for ord in self.ORDERS:
                for dim, keepdim in ((None, False), ([0], True), ([1], False)):
                    with self.subTest(ord=ord, dim=dim, dtype=dtype):
                        wide = self._compare(values, ord, dim, keepdim, tol=tol)
                        tight = self._compare(
                            values, ord, dim, keepdim, dense=True, tol=tol)
                        np.testing.assert_allclose(
                            tight.cpu().numpy().astype(np.float64),
                            wide.cpu().numpy().astype(np.float64),
                            rtol=tol, atol=tol, equal_nan=True)

    def test_a_wider_answer_is_offered_on_request(self):
        rng = np.random.default_rng(79)
        values = rng.standard_normal((4, 5)).astype(np.float16)
        for target in (tensorplay.float16, tensorplay.float32):
            for ord in (1, 2, 3):
                got = tensorplay.linalg_vector_norm(
                    self._device(values), ord, [1], False, dtype=target)
                want = tensorplay.linalg_vector_norm(
                    self._host(values), ord, [1], False, dtype=target)
                with self.subTest(target=target, ord=ord):
                    self.assertEqual(got.dtype, target)
                    np.testing.assert_allclose(
                        got.cpu().numpy().astype(np.float64),
                        want.numpy().astype(np.float64),
                        rtol=1e-5, atol=1e-5)
        # An answer may not be narrower than the payload it measured.
        with self.assertRaises(TypeError):
            tensorplay.linalg_vector_norm(
                self._device(values.astype(np.float32)), 2, [1], False,
                dtype=tensorplay.float16)

    def test_a_scalar_has_no_axis_to_fold(self):
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            for value in (3.0, 0.0, -2.5, float("nan"), float("inf")):
                values = np.array(value, dtype=dtype)
                for ord in self.ORDERS:
                    with self.subTest(value=value, ord=ord, dtype=dtype):
                        got = self._compare(values, ord, None, False, tol=tol)
                        self.assertEqual(tuple(got.shape), ())

    def test_an_empty_walk_answers_the_identity_of_its_fold(self):
        empties = (
            ((0,), None, False),
            ((0,), None, True),
            ((0,), [], True),
            ((0,), [0], False),
            ((0,), [0], True),
            ((0, 3), None, True),
            ((0, 3), [0], False),
            ((0, 3), [0], True),
            ((0, 3), [1], False),
            ((2, 0, 3), [1], True),
            ((2, 0, 3), [1, 2], False),
        )
        for dense in (False, True):
            for shape, dim, keepdim in empties:
                values = np.zeros(shape, dtype=np.float32)
                for ord in self.ORDERS:
                    with self.subTest(shape=shape, dim=dim, ord=ord,
                                      dense=dense):
                        got = self._compare(values, ord, dim, keepdim,
                                            dense=dense)
                        if not got.numel():
                            continue
                        # A walk over named axes, and a whole payload that
                        # keeps its axes, both take the root of the identity.
                        axis_walk = bool(dim) or keepdim
                        want = (
                            float("inf")
                            if (ord < 0 if axis_walk
                                else ord == float("-inf"))
                            else 0.0)
                        self.assertTrue(
                            np.all(got.cpu().numpy() == want))

    def test_the_special_lanes_reach_the_extrema(self):
        values = np.array([[np.nan, 1.0, -2.0],
                           [float("inf"), float("-inf"), 0.0]],
                          dtype=np.float32)
        for ord in self.ORDERS:
            for dim, keepdim in ((None, False), ([0], True), ([1], False)):
                for dense in (False, True):
                    with self.subTest(ord=ord, dim=dim, dense=dense):
                        self._compare(values, ord, dim, keepdim, dense=dense)

    def test_a_zero_order_counts_every_lane_that_is_not_zero(self):
        values = np.array([0.0, -0.0, 2.5, float("nan"), 0.0],
                          dtype=np.float32)
        got = self._compare(values, 0, None, False)
        self.assertEqual(float(got.cpu().numpy()), 2.0)
        got = self._compare(values, 0, [0], False)
        np.testing.assert_allclose(got.cpu().numpy(), np.array(2.0))

    def test_a_count_has_no_magnitude_to_measure(self):
        for values in (np.arange(6, dtype=np.int32).reshape(2, 3),
                       np.array([True, False, True])):
            for ord in (1, 2):
                with self.subTest(dtype=values.dtype, ord=ord):
                    with self.assertRaises(TypeError):
                        tensorplay.linalg_vector_norm(
                            self._device(values), ord, [0], False)

    def test_the_axis_list_must_name_distinct_axes(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        with self.assertRaises(ValueError):
            tensorplay.linalg_vector_norm(self._device(values), 2, [1, 1])
        with self.assertRaises(IndexError):
            tensorplay.linalg_vector_norm(self._device(values), 2, [3])
        with self.assertRaises(IndexError):
            tensorplay.linalg_vector_norm(self._device(values), 2, [-4])

    # The two older spellings of the same measurement: a whole payload and a
    # payload with a listed set of axes.  An absent or empty list folds every
    # axis, so the fold table an order selects is the only difference from the
    # named-order entry.

    def _norm_pair(self, values, p, dim, keepdim, dense=False):
        device = self._device(values, dense)
        host = self._host(values)
        if dim is None:
            return device.norm(p), host.norm(p)
        return (device.norm(p, dim, keepdim), host.norm(p, dim, keepdim))

    def _compare_norm(self, values, p, dim, keepdim, dense=False, tol=1e-5):
        got, want = self._norm_pair(values, p, dim, keepdim, dense)
        self.assertEqual(got.dtype, want.dtype)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        np.testing.assert_allclose(
            got.cpu().numpy().astype(np.float64),
            want.numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True,
            err_msg=(f"norm {p} over dims {dim} (keepdim {keepdim}) of shape "
                     f"{values.shape}"))
        return got

    def test_the_norm_entries_follow_the_same_fold_table(self):
        rng = np.random.default_rng(83)
        plans = (
            ((3, 5), ((None, False), ([], False), ([], True), ([0], False),
                      ([-1], True), ([0, 1], True))),
            ((2, 3, 4), ((None, True), ([], False), ([0], False),
                         ([-1], False), ([0, 2], False), ([1, 2], True))),
        )
        for shape, axes in plans:
            for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
                values = rng.standard_normal(shape).astype(dtype)
                for p in self.ORDERS:
                    for dim, keepdim in axes:
                        with self.subTest(shape=shape, dtype=dtype, p=p,
                                          dim=dim, keepdim=keepdim):
                            self._compare_norm(values, p, dim, keepdim,
                                               tol=tol)

    def test_the_norm_entries_answer_for_a_scalar(self):
        for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
            for value in (2.5, 0.0, -4.0, float("nan"), float("inf")):
                values = np.array(value, dtype=dtype)
                for p in self.ORDERS:
                    with self.subTest(value=value, p=p, dtype=dtype):
                        got = self._compare_norm(values, p, None, False,
                                                 tol=tol)
                        self.assertEqual(tuple(got.shape), ())

    def test_the_norm_entries_answer_the_empty_identity(self):
        empties = (
            ((0,), None, False),
            ((0,), None, True),
            ((0,), [], True),
            ((0,), [0], False),
            ((0, 3), [0], True),
            ((0, 3), [1], False),
            ((2, 0, 4), [], False),
            ((2, 0, 4), [1], False),
            ((2, 0, 4), [1], True),
        )
        for shape, dim, keepdim in empties:
            values = np.zeros(shape, dtype=np.float32)
            for p in self.ORDERS:
                with self.subTest(shape=shape, dim=dim, p=p, keepdim=keepdim):
                    got = self._compare_norm(values, p, dim, keepdim)
                    if not got.numel():
                        continue
                    # Only a list that names an axis walks under a root; a
                    # whole payload, and a list that names none, report the
                    # identity of the fold itself.
                    axis_walk = bool(dim)
                    want = (
                        float("inf")
                        if (p < 0 if axis_walk else p == float("-inf"))
                        else 0.0)
                    self.assertTrue(np.all(got.cpu().numpy() == want))

    def test_the_norm_entries_fold_the_special_lanes(self):
        values = np.array([[np.nan, 1.0, -2.0],
                           [float("inf"), 0.0, -float("inf")]],
                          dtype=np.float32)
        for p in self.ORDERS:
            for dim, keepdim in ((None, False), ([], True), ([0], True),
                                 ([1], False)):
                for dense in (False, True):
                    with self.subTest(p=p, dim=dim, dense=dense):
                        self._compare_norm(values, p, dim, keepdim,
                                           dense=dense)

    def test_the_norm_entries_refuse_a_count_payload(self):
        # A count has no magnitude to measure, so the fold is refused rather
        # than composed out of integer powers.
        for values in (np.arange(6, dtype=np.int32).reshape(2, 3),
                       np.arange(6, dtype=np.int64).reshape(2, 3),
                       np.array([True, False, True, False, True, False])):
            for p in (1, 2, float("inf")):
                with self.subTest(dtype=values.dtype, p=p):
                    with self.assertRaises(TypeError):
                        self._device(values).norm(p)
                    with self.assertRaises(TypeError):
                        self._device(values).norm(p, [0], False)


class TestVulkanKthValue(unittest.TestCase):
    # The k-th smallest entry of a span, counted from one, together with the
    # zero-based lane that holds it.  A tie names the earlier lane, and a lane
    # that is not a number ranks past every ordered lane, so the last rank of a
    # span that holds one answers with the earliest of them.  The answer keeps
    # the element type of the payload it came from while the lane is a wide
    # count.  A span is a listed axis of the payload, so a lone entry answers
    # itself at the first rank.

    CASES = (
        ((7,), (0, -1)),
        ((2, 5), (0, 1, -2, -1)),
        ((3, 4, 2), (0, 1, 2, -1)),
        ((2, 3, 4, 2), (3, -1)),
    )

    @staticmethod
    def _device(values, dense=False):
        if dense:
            return tensorplay.tensor(values.copy(), device="vulkan")
        return tensorplay.from_numpy(values.copy()).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(values.copy())

    def _compare(self, values, k, dim, keepdim, dense=False, tol=1e-5):
        got = self._device(values, dense).kthvalue(k, dim, keepdim)
        want = self._host(values).kthvalue(k, dim, keepdim)
        self.assertEqual(got[0].dtype, want[0].dtype)
        self.assertEqual(got[1].dtype, want[1].dtype)
        self.assertEqual(tuple(got[0].shape), tuple(want[0].shape))
        self.assertEqual(tuple(got[1].shape), tuple(want[1].shape))
        np.testing.assert_array_equal(
            got[1].cpu().numpy(), want[1].numpy(),
            err_msg=f"lane of rank {k} at dim {dim} of shape {values.shape}")
        np.testing.assert_allclose(
            got[0].cpu().numpy().astype(np.float64),
            want[0].numpy().astype(np.float64),
            rtol=tol, atol=tol, equal_nan=True,
            err_msg=f"value of rank {k} at dim {dim} of shape {values.shape}")
        return got

    def test_every_rank_of_the_span_follows_the_host(self):
        rng = np.random.default_rng(101)
        for shape, dims in self.CASES:
            for dtype, tol in ((np.float32, 1e-5), (np.float16, 3e-3)):
                values = (rng.standard_normal(shape) * 8).astype(dtype)
                for dim in dims:
                    length = shape[dim % len(shape)]
                    for k in range(1, length + 1):
                        for keepdim in (False, True):
                            with self.subTest(shape=shape, dtype=dtype, dim=dim,
                                              k=k, keepdim=keepdim):
                                self._compare(values, k, dim, keepdim, tol=tol)

    def test_both_storages_answer_the_same_selection(self):
        rng = np.random.default_rng(103)
        values = (rng.standard_normal((3, 6)) * 4).astype(np.float32)
        for dim, ranks in ((0, (1, 2, 3)), (1, (1, 4, 6)), (-1, (2, 5, 6))):
            keepdim = dim == 0
            with self.subTest(dim=dim, ranks=ranks):
                for k in ranks:
                    wide = self._compare(values, k, dim, keepdim)
                    tight = self._compare(values, k, dim, keepdim, dense=True)
                    np.testing.assert_array_equal(
                        tight[0].cpu().numpy(), wide[0].cpu().numpy())
                    np.testing.assert_array_equal(
                        tight[1].cpu().numpy(), wide[1].cpu().numpy())

    def test_a_tie_names_the_earlier_lane(self):
        values = np.array([2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        ordered = [(1, 1.0, 1), (2, 1.0, 3), (3, 2.0, 0), (4, 2.0, 2),
                   (5, 2.0, 4)]
        for k, value, lane in ordered:
            with self.subTest(k=k):
                got = self._compare(values, k, 0, False)
                self.assertEqual(float(got[0].cpu().numpy()), value)
                self.assertEqual(int(got[1].cpu().numpy()), lane)

    def test_a_lane_that_is_not_a_number_ranks_last(self):
        values = np.array([np.nan, 1.0, np.nan, -2.0], dtype=np.float32)
        # The ordered lanes come first, then the two lanes that are not a
        # number, the earlier of them at the earlier rank.
        ordered = [(1, -2.0, 3), (2, 1.0, 1), (3, float("nan"), 0),
                   (4, float("nan"), 2)]
        for k, value, lane in ordered:
            for dtype in (np.float32, np.float16):
                with self.subTest(k=k, dtype=dtype):
                    got = self._compare(values.astype(dtype), k, 0, False,
                                        tol=1e-5 if dtype is np.float32 else 3e-3)
                    self.assertEqual(int(got[1].cpu().numpy()), lane)
                    if np.isnan(value):
                        self.assertTrue(np.isnan(float(got[0].cpu().numpy())))
                    else:
                        self.assertEqual(float(got[0].cpu().numpy()), value)
        edges = np.array([float("inf"), float("-inf"), 1.0], np.float32)
        for k, lane in ((1, 1), (2, 2), (3, 0)):
            with self.subTest(k=k):
                got = self._compare(edges, k, 0, False)
                self.assertEqual(int(got[1].cpu().numpy()), lane)
                self.assertEqual(float(got[0].cpu().numpy()),
                                 [-float("inf"), 1.0, float("inf")][k - 1])

    def test_a_lone_entry_answers_itself(self):
        for value in (2.5, -1.0, 0.0, float("nan"), float("inf")):
            for dim in (0, -1):
                for keepdim in (False, True):
                    with self.subTest(value=value, dim=dim, keepdim=keepdim):
                        got = self._compare(np.array(value, np.float32), 1, dim,
                                            keepdim)
                        self.assertEqual(tuple(got[0].shape), ())
                        self.assertEqual(int(got[1].cpu().numpy()), 0)

    def test_the_selection_unlocks_a_median_over_an_axis(self):
        rng = np.random.default_rng(107)
        values = (rng.standard_normal((4, 5)) * 6).astype(np.float32)
        values[1] = np.array([3.0, 3.0, -1.0, 3.0, np.nan], np.float32)
        for dim in (0, 1, -1):
            for keepdim in (False, True):
                with self.subTest(dim=dim, keepdim=keepdim):
                    got = self._device(values).median(dim, keepdim)
                    want = self._host(values).median(dim, keepdim)
                    self.assertEqual(got[1].dtype, want[1].dtype)
                    self.assertEqual(tuple(got[0].shape), tuple(want[0].shape))
                    np.testing.assert_array_equal(got[1].cpu().numpy(),
                                                 want[1].numpy())
                    np.testing.assert_allclose(
                        got[0].cpu().numpy(), want[0].numpy(),
                        rtol=1e-5, atol=1e-5, equal_nan=True)

    def test_a_rank_or_axis_outside_the_span_refuses(self):
        payload = self._device(np.zeros((2, 3), np.float32))
        with self.assertRaisesRegex(RuntimeError, "out of range for dim"):
            payload.kthvalue(0, 1)
        with self.assertRaisesRegex(RuntimeError, "out of range for dim"):
            payload.kthvalue(4, 1)
        with self.assertRaisesRegex(RuntimeError, "out of range for dim"):
            # An axis with no lane holds no rank at all.
            self._device(np.zeros((2, 0), np.float32)).kthvalue(1, 1)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            payload.kthvalue(1, 2)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            payload.kthvalue(1, -3)
        scalar = self._device(np.array(1.0, np.float32))
        with self.assertRaisesRegex(IndexError, "for scalar kthvalue input"):
            scalar.kthvalue(1, 1)
        with self.assertRaisesRegex(RuntimeError, "out of range for dim"):
            scalar.kthvalue(2, 0)

    def test_a_count_has_no_order_to_walk(self):
        for values in (np.arange(6, dtype=np.int32).reshape(2, 3),
                       np.arange(6, dtype=np.int64).reshape(2, 3),
                       np.array([True, False, True], np.bool_)):
            with self.subTest(dtype=values.dtype):
                with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                    self._device(values).kthvalue(1, 0)


class TestVulkanGroupNorm(unittest.TestCase):
    # Group normalization folds a channel axis into a number of whole groups and
    # centers each group on its own two moments.  The walk reads the payload as
    # (N, G, (C/G)*HxW), so a group is one axis and everything a group covers is
    # one span; the affine terms then ride on (N, C, HxW), where one channel owns
    # one axis.  The moments are kept in the width the payload has, except for a
    # narrow payload, whose moments stay at single precision while its answer
    # returns to its own width.  A batch with no rows has nothing to walk and so
    # answers with empty moment tables.

    SHAPES = (
        (2, 6, 4),
        (1, 8, 3, 5),
        (3, 4, 2, 3),
        (2, 12, 7),
        (5, 6, 1),
        (1, 2, 2, 2),
        (2, 8),
    )

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _walk(values, num_groups, weight=None, bias=None, eps=1e-5):
        # The arithmetic definition, walked in double precision: the mean and the
        # biased variance of everything a group covers, then the affine terms per
        # channel.
        wide = np.asarray(values).astype(np.float64)
        rows, channels = wide.shape[0], wide.shape[1]
        spatial = int(np.prod(wide.shape[2:])) if wide.ndim > 2 else 1
        folded = wide.reshape(rows, num_groups, (channels // num_groups) * spatial)
        mean = folded.mean(axis=-1, keepdims=True)
        var = ((folded - mean) ** 2).mean(axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        out = ((folded - mean) * rstd).reshape(wide.shape)
        shape = [1, channels] + [1] * (wide.ndim - 2)
        if weight is not None:
            out = out * np.asarray(weight).astype(np.float64).reshape(shape)
        if bias is not None:
            out = out + np.asarray(bias).astype(np.float64).reshape(shape)
        return (out.astype(np.float32),
                mean.reshape(rows, num_groups).astype(np.float32),
                rstd.reshape(rows, num_groups).astype(np.float32))

    def _affine(self, channels, rng):
        weight = (rng.random(channels) + 0.5).astype(np.float32)
        bias = (rng.random(channels) - 0.5).astype(np.float32)
        return weight, bias

    def test_a_group_answers_the_span_it_covers(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(101)
        for shape in self.SHAPES:
            payload = rng.standard_normal(shape).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            for num_groups in (g for g in (1, 2, 3, 4, 6, 8)
                               if shape[1] % g == 0):
                for name, args in (
                        ("none", (None, None)),
                        ("weight", (weight, None)),
                        ("bias", (None, bias)),
                        ("both", (weight, bias))):
                    with self.subTest(shape=shape, num_groups=num_groups,
                                      affine=name):
                        got = F.group_norm(
                            self._device(payload), num_groups,
                            weight=None if args[0] is None else self._device(args[0]),
                            bias=None if args[1] is None else self._device(args[1]),
                            eps=1e-5).cpu().numpy()
                        want = self._walk(payload, num_groups, args[0], args[1])
                        np.testing.assert_allclose(
                            got, want[0], rtol=1e-4, atol=1e-4,
                            err_msg=f"group of {num_groups} over {shape}")

    def test_the_host_walk_answers_the_same(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(103)
        for shape in ((2, 6, 4), (1, 8, 3, 5), (3, 4, 2, 3)):
            payload = rng.standard_normal(shape).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            for num_groups in (1, 2):
                with self.subTest(shape=shape, num_groups=num_groups):
                    device = F.group_norm(
                        self._device(payload), num_groups,
                        weight=self._device(weight), bias=self._device(bias),
                        eps=1e-5).cpu().numpy()
                    host = F.group_norm(
                        self._host(payload), num_groups,
                        weight=self._host(weight), bias=self._host(bias),
                        eps=1e-5).numpy()
                    np.testing.assert_allclose(
                        device, host, rtol=1e-4, atol=1e-4)

    def test_a_walk_releases_the_moments_it_walked(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(107)
        for shape, num_groups in (((2, 6, 4), 3), ((1, 8, 3, 5), 4),
                                  ((4, 8, 16, 16), 2), ((2, 8), 2)):
            payload = rng.standard_normal(shape).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            spatial = int(np.prod(shape[2:])) if len(shape) > 2 else 1
            with self.subTest(shape=shape, num_groups=num_groups):
                out, mean, rstd = F.native_group_norm(
                    self._device(payload), self._device(weight),
                    self._device(bias), shape[0], shape[1], spatial,
                    num_groups, 1e-5)
                want = self._walk(payload, num_groups, weight, bias)
                self.assertEqual(tuple(mean.shape), (shape[0], num_groups))
                self.assertEqual(tuple(rstd.shape), (shape[0], num_groups))
                for got, expected, name in (
                        (out.cpu().numpy(), want[0], "answer"),
                        (mean.cpu().numpy(), want[1], "mean"),
                        (rstd.cpu().numpy(), want[2], "reciprocal spread")):
                    np.testing.assert_allclose(
                        got, expected, rtol=1e-4, atol=1e-4,
                        err_msg=f"{name} of group {num_groups}")

    def test_a_narrow_lane_keeps_its_moments_wide(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(109)
        for shape, num_groups in (((2, 6, 4), 3), ((1, 8, 3, 5), 4)):
            payload = (rng.standard_normal(shape) * 3).astype(np.float16)
            weight = (rng.random(shape[1]) + 0.5).astype(np.float16)
            bias = (rng.random(shape[1]) - 0.5).astype(np.float16)
            spatial = int(np.prod(shape[2:]))
            with self.subTest(shape=shape):
                out, mean, rstd = F.native_group_norm(
                    self._device(payload), self._device(weight),
                    self._device(bias), shape[0], shape[1], spatial,
                    num_groups, 1e-5)
                want = self._walk(payload, num_groups, weight, bias)
                self.assertEqual(out.dtype, tensorplay.float16)
                self.assertEqual(mean.dtype, tensorplay.float32)
                self.assertEqual(rstd.dtype, tensorplay.float32)
                np.testing.assert_allclose(
                    out.cpu().numpy().astype(np.float32), want[0],
                    rtol=2e-3, atol=2e-3)

    def test_a_group_far_from_zero_keeps_its_spread(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(113)
        # A spread read off a large offset is lost when the second moment is
        # folded as the difference of two large sums.  The walk here subtracts
        # the mean first and folds the deviations, so the answers spread over a
        # group keeps the ratio the definition gives -- the reciprocal spread
        # shrinks a group whose own spread is small beside eps.
        for offset, spread in ((1e3, 1.0), (1e4, 1.0), (3e4, 0.5), (1e2, 1e-3)):
            payload = (offset + spread *
                       rng.standard_normal((2, 6, 8))).astype(np.float32)
            with self.subTest(offset=offset, spread=spread):
                got = F.group_norm(self._device(payload), 3,
                                   eps=1e-5).cpu().numpy()
                want = self._walk(payload, 3)[0]
                self.assertTrue(np.isfinite(got).all(),
                                "a far offset lost the spread")
                np.testing.assert_allclose(
                    got, want, rtol=2e-2, atol=2e-2)
                np.testing.assert_allclose(
                    got.reshape(2, 3, -1).std(axis=-1),
                    want.reshape(2, 3, -1).std(axis=-1),
                    rtol=2e-2, atol=2e-2)

    def test_a_strided_payload_is_walked_in_its_own_order(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(127)
        block = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        moved = np.ascontiguousarray(block.transpose(0, 3, 1, 2))
        seen = self._device(block).permute(0, 3, 1, 2)
        for num_groups in (1, 5):
            with self.subTest(num_groups=num_groups, source="strided view"):
                got = F.group_norm(seen, num_groups).cpu().numpy()
                np.testing.assert_allclose(
                    got, self._walk(moved, num_groups)[0],
                    rtol=1e-4, atol=1e-4)

    def test_a_batch_without_rows_walks_nothing(self):
        import tensorplay.functional as F
        for dtype in (np.float32, np.float16):
            payload = np.zeros((0, 6, 4), dtype)
            with self.subTest(dtype=dtype):
                out, mean, rstd = F.native_group_norm(
                    self._device(payload), None, None, 0, 6, 4, 3, 1e-5)
                self.assertEqual(tuple(out.shape), (0, 6, 4))
                self.assertEqual(tuple(mean.shape), (0, 3))
                self.assertEqual(tuple(rstd.shape), (0, 3))
                self.assertEqual(out.dtype, tensorplay.float16
                                 if dtype is np.float16 else tensorplay.float32)
                self.assertEqual(mean.dtype, tensorplay.float32)
                self.assertEqual(rstd.dtype, tensorplay.float32)
                got = F.group_norm(self._device(payload), 3)
                self.assertEqual(tuple(got.shape), (0, 6, 4))

    def test_a_group_or_parameter_out_of_shape_refuses(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(131)
        payload = rng.standard_normal((2, 6, 4)).astype(np.float32)
        device = self._device(payload)
        with self.subTest(kind="channels not divisible"):
            with self.assertRaisesRegex(
                    RuntimeError, "channels must be divisible"):
                F.group_norm(device, 4)
        with self.subTest(kind="no groups"):
            with self.assertRaisesRegex(
                    RuntimeError, "channels must be divisible"):
                F.group_norm(device, 0)
        with self.subTest(kind="a parameter of the wrong length"):
            with self.assertRaisesRegex(
                    RuntimeError, "one lane per channel"):
                F.group_norm(device, 3, weight=self._device(
                    np.ones(5, np.float32)))
        with self.subTest(kind="a parameter of the wrong rank"):
            with self.assertRaisesRegex(
                    RuntimeError, "one lane per channel"):
                F.group_norm(device, 3, weight=self._device(
                    np.ones((2, 3), np.float32)))
        with self.subTest(kind="a spread with no lane"):
            with self.assertRaisesRegex(
                    RuntimeError, "the spatial span must be positive"):
                F.group_norm(self._device(np.zeros((2, 4, 0), np.float32)), 2)
        with self.subTest(kind="a payload with one axis"):
            with self.assertRaisesRegex(
                    RuntimeError, "supports 2d to 4d tensors"):
                F.group_norm(self._device(np.zeros(4, np.float32)), 2)

    def test_a_payload_that_is_not_a_measurement_refuses(self):
        import tensorplay.functional as F
        for values, pattern in (
                (np.zeros((2, 4, 3), np.int32), "Float32 and Float16"),
                (np.zeros((2, 4, 3), np.int64), "Float32 and Float16"),
                (np.zeros((2, 4, 3), np.float64), "Not a supported Vulkan dtype"),
                (np.zeros((1, 4, 2, 2, 2), np.float32), "ndim <= 4")):
            with self.subTest(dtype=values.dtype, shape=values.shape):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    F.group_norm(self._device(values), 2)


class TestVulkanGroupNormBackward(unittest.TestCase):
    # The gradient of a group normalization owes each lane two corrections: the
    # mean of the weighted gradient over the span the lane's group covers, and
    # the mean of that gradient times the centered and scaled activation.  The
    # parameter gradients need no moments at all -- a channel folds its own lane
    # over the spatial span and then over the rows it appears in, which is why
    # they are walked in the four-axis form (N, G, C/G, HxW): the two folds leave
    # the channels in their own order.  An answer a caller did not ask for stays
    # undefined instead of being filled with zeros.

    SHAPES = (
        (2, 6, 4),
        (1, 8, 3, 5),
        (3, 4, 2, 3),
        (2, 12, 7),
        (2, 8),
    )

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _absent(values):
        # A skipped answer arrives as a tensor with no element type.
        return (values is None
                or str(values.dtype).startswith("tensorplay.undefined"))

    @staticmethod
    def _definition(values, grad, num_groups, weight=None, eps=1e-5):
        # The arithmetic definition, walked in double precision.
        wide = np.asarray(values).astype(np.float64)
        drive = np.asarray(grad).astype(np.float64)
        rows, channels = wide.shape[0], wide.shape[1]
        spatial = int(np.prod(wide.shape[2:])) if wide.ndim > 2 else 1
        per_group = channels // num_groups
        span = per_group * spatial
        folded = wide.reshape(rows, num_groups, span)
        mean = folded.mean(axis=-1, keepdims=True)
        var = ((folded - mean) ** 2).mean(axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        hat = (folded - mean) * rstd
        gamma = (np.ones(channels, np.float64) if weight is None
                 else np.asarray(weight).astype(np.float64).reshape(channels))
        # The group fold runs channel-major: lane d*spatial + s covers channel d.
        weighted = drive.reshape(rows, num_groups, span) * np.repeat(
            gamma.reshape(num_groups, per_group), spatial, axis=-1).reshape(
                1, num_groups, span)
        dx = (rstd * (weighted
                      - weighted.mean(axis=-1, keepdims=True)
                      - hat * (weighted * hat).mean(axis=-1,
                                                    keepdims=True)))
        dx = dx.reshape(wide.shape)
        four = (rows, num_groups, per_group, spatial)
        centered = ((folded - mean).reshape(four)
                    * rstd.reshape(rows, num_groups, 1, 1))
        dweight = (drive.reshape(four) * centered).sum(axis=(0, 3)).reshape(
            channels)
        dbias = drive.reshape(four).sum(axis=(0, 3)).reshape(channels)
        return (dx.astype(np.float32), dweight.astype(np.float32),
                dbias.astype(np.float32),
                mean.reshape(rows, num_groups).astype(np.float32),
                rstd.reshape(rows, num_groups).astype(np.float32))

    def _affine(self, channels, rng):
        weight = (rng.random(channels) + 0.5).astype(np.float32)
        bias = (rng.random(channels) - 0.5).astype(np.float32)
        return weight, bias

    def test_a_gradient_answers_the_walk_of_its_span(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(151)
        for shape in self.SHAPES:
            payload = rng.standard_normal(shape).astype(np.float32)
            grad = (rng.standard_normal(shape) * 0.7).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            for num_groups in (g for g in (1, 2, 3, 4, 6, 8)
                               if shape[1] % g == 0):
                for name, args in (
                        ("none", (None, None)),
                        ("weight", (weight, None)),
                        ("bias", (None, bias)),
                        ("both", (weight, bias))):
                    with self.subTest(shape=shape, num_groups=num_groups,
                                      affine=name):
                        def walked(on_device):
                            make = (self._device if on_device
                                    else self._host)
                            outs = F.group_norm_backward(
                                make(grad), make(payload), num_groups,
                                weight=None if args[0] is None else make(args[0]),
                                bias=None if args[1] is None else make(args[1]),
                                eps=1e-5)
                            return [None if self._absent(t)
                                    else (t.cpu().numpy() if on_device
                                          else t.numpy()) for t in outs]
                        got = walked(True)
                        want = walked(False)
                        truth = self._definition(payload, grad, num_groups,
                                                 args[0])
                        for index, (label, tol) in enumerate(
                                (("dx", 1e-3), ("dweight", 1e-2),
                                 ("dbias", 1e-2))):
                            self.assertEqual(
                                (got[index] is None), (want[index] is None),
                                f"{label} answered when the host stayed away")
                            if got[index] is None:
                                continue
                            np.testing.assert_allclose(
                                got[index], want[index], rtol=tol, atol=tol,
                                err_msg=f"{label} of group {num_groups}")
                            if args[0] is not None or index == 2:
                                np.testing.assert_allclose(
                                    got[index], truth[index], rtol=2e-2,
                                    atol=2e-2, err_msg=f"{label} definition")

    def test_a_gradient_answers_the_moments_it_saved(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(163)
        for shape, num_groups in (((2, 6, 4), 3), ((1, 8, 3, 5), 4),
                                  ((3, 4, 2, 3), 2), ((2, 12, 7), 3),
                                  ((2, 8), 4)):
            payload = rng.standard_normal(shape).astype(np.float32)
            grad = rng.standard_normal(shape).astype(np.float32)
            weight = (rng.random(shape[1]) + 0.5).astype(np.float32)
            spatial = int(np.prod(shape[2:])) if len(shape) > 2 else 1
            truth = self._definition(payload, grad, num_groups, weight)
            with self.subTest(shape=shape, num_groups=num_groups):
                dx, dweight, dbias = F.native_group_norm_backward(
                    self._device(grad), self._device(payload),
                    self._device(truth[3]), self._device(truth[4]),
                    self._device(weight), shape[0], shape[1], spatial,
                    num_groups, [True, True, True])
                for got, expected, label in (
                        (dx, truth[0], "dx"),
                        (dweight, truth[1], "dweight"),
                        (dbias, truth[2], "dbias")):
                    np.testing.assert_allclose(
                        got.cpu().numpy(), expected, rtol=2e-3, atol=2e-3,
                        err_msg=f"{label} over {shape} group {num_groups}")

    def test_the_gradient_moves_the_answer_it_walked(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(167)
        shape, num_groups = (1, 4, 3), 2
        payload = rng.standard_normal(shape).astype(np.float32)
        grad = rng.standard_normal(shape).astype(np.float32)
        weight, bias = self._affine(shape[1], rng)
        dx = F.group_norm_backward(
            self._device(grad), self._device(payload), num_groups,
            weight=self._device(weight), bias=self._device(bias),
            eps=1e-5)[0].cpu().numpy()

        def walked(values):
            answer = F.group_norm(
                self._device(values), num_groups,
                weight=self._device(weight), bias=self._device(bias),
                eps=1e-5).cpu().numpy()
            return float(np.sum(grad * answer))

        step = 1e-2
        flat = payload.reshape(-1)
        for index in range(flat.size):
            up = flat.copy()
            down = flat.copy()
            up[index] += step
            down[index] -= step
            numeric = (walked(up.reshape(shape))
                       - walked(down.reshape(shape))) / (2 * step)
            np.testing.assert_allclose(
                dx.reshape(-1)[index], numeric, rtol=2e-2, atol=2e-3,
                err_msg=f"lane {index} is not the slope of the walk")

    def test_a_skipped_answer_stays_away(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(173)
        shape, num_groups = (2, 6, 4), 3
        payload = rng.standard_normal(shape).astype(np.float32)
        grad = rng.standard_normal(shape).astype(np.float32)
        weight = (rng.random(shape[1]) + 0.5).astype(np.float32)
        truth = self._definition(payload, grad, num_groups, weight)
        for mask in ([True, False, False], [False, True, False],
                     [False, False, True], [True, True, False],
                     [False, False, False]):
            with self.subTest(mask=mask):
                outs = F.native_group_norm_backward(
                    self._device(grad), self._device(payload),
                    self._device(truth[3]), self._device(truth[4]),
                    self._device(weight), 2, 6, 4, num_groups, mask)
                for index, (wanted, label) in enumerate(
                        zip(mask, ("dx", "dweight", "dbias"))):
                    if not wanted:
                        self.assertTrue(
                            self._absent(outs[index]),
                            f"{label} answered although it was skipped")
                        continue
                    np.testing.assert_allclose(
                        outs[index].cpu().numpy(), truth[index],
                        rtol=2e-3, atol=2e-3, err_msg=label)

    def test_a_narrow_lane_graduates_in_its_own_width(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(181)
        for shape, num_groups in (((2, 6, 4), 3), ((1, 8, 3, 5), 4)):
            payload = (rng.standard_normal(shape) * 2).astype(np.float16)
            grad = (rng.standard_normal(shape) * 0.5).astype(np.float16)
            weight = (rng.random(shape[1]) + 0.5).astype(np.float16)
            spatial = int(np.prod(shape[2:]))
            truth = self._definition(payload, grad, num_groups, weight)
            with self.subTest(shape=shape):
                _, mean, rstd = F.native_group_norm(
                    self._device(payload), self._device(weight), None,
                    shape[0], shape[1], spatial, num_groups, 1e-5)
                self.assertEqual(mean.dtype, tensorplay.float32)
                self.assertEqual(rstd.dtype, tensorplay.float32)
                dx, dweight, dbias = F.native_group_norm_backward(
                    self._device(grad), self._device(payload), mean, rstd,
                    self._device(weight), shape[0], shape[1], spatial,
                    num_groups, [True, True, True])
                for got, expected, label in (
                        (dx, truth[0], "dx"),
                        (dweight, truth[1], "dweight"),
                        (dbias, truth[2], "dbias")):
                    self.assertEqual(got.dtype, tensorplay.float16, label)
                    np.testing.assert_allclose(
                        got.cpu().numpy().astype(np.float32), expected,
                        rtol=4e-3, atol=4e-3,
                        err_msg=f"{label} of a narrow lane")

    def test_a_batch_without_rows_graduates_nothing(self):
        import tensorplay.functional as F
        payload = np.zeros((0, 6, 4), np.float32)
        grad = np.zeros((0, 6, 4), np.float32)
        weight = np.ones(6, np.float32)
        with self.subTest(kind="moments released for no row"):
            dx, dweight, dbias = F.native_group_norm_backward(
                self._device(grad), self._device(payload),
                self._device(np.zeros((0, 3), np.float32)),
                self._device(np.zeros((0, 3), np.float32)),
                self._device(weight), 0, 6, 4, 3, [True, True, True])
            for got, shape in ((dx, (0, 6, 4)), (dweight, (6,)), (dbias, (6,))):
                array = got.cpu().numpy()
                self.assertEqual(tuple(array.shape), shape)
                self.assertTrue(np.all(array == 0))
        with self.subTest(kind="the walk of no row"):
            outs = F.group_norm_backward(
                self._device(grad), self._device(payload), 3,
                weight=self._device(weight), bias=self._device(weight),
                eps=1e-5)
            # An activation with no row carries no gradient, but a parameter
            # still answers one lane per channel, and every lane totals nothing.
            for got, shape in zip(outs, ((0, 6, 4), (6,), (6,))):
                self.assertFalse(self._absent(got))
                array = got.cpu().numpy()
                self.assertEqual(tuple(array.shape), shape)
                self.assertTrue(np.all(array == 0), "a row-less walk paid a lane")

    def test_a_gradient_out_of_shape_refuses(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(191)
        payload = rng.standard_normal((2, 6, 4)).astype(np.float32)
        grad = rng.standard_normal((2, 6, 4)).astype(np.float32)
        moments = np.zeros((2, 3), np.float32)
        device = self._device(payload)
        grad_device = self._device(grad)
        wide_moments = self._device(np.zeros((2, 2), np.float32))
        with self.subTest(kind="a gradient of another width"):
            with self.assertRaisesRegex(
                    RuntimeError, "a gradient is read in the width"):
                F.native_group_norm_backward(
                    self._device(grad.astype(np.float16)), device,
                    self._device(moments), self._device(moments), None,
                    2, 6, 4, 3, [True, False, False])
        with self.subTest(kind="moments that miss a group"):
            with self.assertRaisesRegex(
                    RuntimeError, "must name every group"):
                F.native_group_norm_backward(
                    grad_device, device, wide_moments, wide_moments, None,
                    2, 6, 4, 3, [True, False, False])
        with self.subTest(kind="a row count that is not the payload's"):
            with self.assertRaisesRegex(
                    RuntimeError, "do not match input"):
                F.native_group_norm_backward(
                    grad_device, device, self._device(moments),
                    self._device(moments), None, 3, 6, 4, 3,
                    [True, False, False])
        with self.subTest(kind="channels not divisible"):
            with self.assertRaisesRegex(
                    RuntimeError, "channels must be divisible"):
                F.group_norm_backward(grad_device, device, 4)
        with self.subTest(kind="a payload with one axis"):
            with self.assertRaisesRegex(
                    RuntimeError, "supports 2d to 4d tensors"):
                F.group_norm_backward(self._device(np.zeros(4, np.float32)),
                                      self._device(np.zeros(4, np.float32)), 2)
        with self.subTest(kind="a payload that counts"):
            with self.assertRaisesRegex(
                    RuntimeError, "Float32 and Float16"):
                F.group_norm_backward(
                    self._device(grad.astype(np.int32)),
                    self._device(payload.astype(np.int32)), 3)


class TestVulkanInstanceNorm(unittest.TestCase):
    # One channel plane of one sample owns both of its moments: the payload folds
    # to (N, C, S) so that a plane is one axis, its mean centers it, and the mean
    # of the squared deviations is its spread -- which is why a plane that sits
    # far from zero keeps its spread.  A record kept for a later walk averages the
    # planes a channel appears in, and the spread it keeps is the unbiased one:
    # the second moment scaled by S/(S-1).  A plane of a single element therefore
    # answers as not a number rather than as zero, and a batch with no plane has
    # no average at all.  The affine terms ride per channel, a narrow payload is
    # walked at single precision and answers in its own width, and a record the
    # caller keeps on the host is written home again.

    SHAPES = (
        (2, 3, 5),
        (1, 4, 3, 5),
        (3, 6, 2, 3),
        (2, 8, 7),
        (5, 3, 4, 2),
    )

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _absent(values):
        # A skipped answer arrives as a tensor with no element type.
        return (values is None
                or str(values.dtype).startswith("tensorplay.undefined"))

    @classmethod
    def _planes(cls, values):
        # (N, C, S): a plane is one axis, everything it covers is one span.
        wide = np.asarray(values).astype(np.float64)
        spatial = int(np.prod(wide.shape[2:]))
        return wide.reshape(wide.shape[0], wide.shape[1], spatial)

    @classmethod
    def _lane(cls, param, channels):
        return np.asarray(param).astype(np.float64).reshape(1, channels, 1)

    @classmethod
    def _walk(cls, values, weight=None, bias=None, eps=1e-5):
        wide = np.asarray(values).astype(np.float64)
        planes = cls._planes(values)
        mean = planes.mean(axis=2)
        biased = ((planes - mean[..., None]) ** 2).mean(axis=2)
        unbiased = biased * planes.shape[2] / (planes.shape[2] - 1.0)
        rstd = 1.0 / np.sqrt(biased + eps)
        out = (planes - mean[..., None]) * rstd[..., None]
        if weight is not None:
            out = out * cls._lane(weight, wide.shape[1])
        if bias is not None:
            out = out + cls._lane(bias, wide.shape[1])
        return (out.reshape(wide.shape).astype(np.float32), mean, unbiased)

    @classmethod
    def _from_records(cls, values, mean, var, weight=None, bias=None, eps=1e-5):
        wide = np.asarray(values).astype(np.float64)
        planes = cls._planes(values)
        rstd = (1.0 / np.sqrt(np.asarray(var).astype(np.float64) + eps)
                )[None, :, None]
        out = (planes - np.asarray(mean).astype(np.float64)[None, :, None]) * rstd
        if weight is not None:
            out = out * cls._lane(weight, wide.shape[1])
        if bias is not None:
            out = out + cls._lane(bias, wide.shape[1])
        return out.reshape(wide.shape).astype(np.float32)

    @classmethod
    def _plane_gradient(cls, values, grad, weight=None, eps=1e-5):
        planes = cls._planes(values)
        drive = cls._planes(grad)
        mean = planes.mean(axis=2, keepdims=True)
        biased = ((planes - mean) ** 2).mean(axis=2, keepdims=True)
        rstd = 1.0 / np.sqrt(biased + eps)
        hat = (planes - mean) * rstd
        gamma = (np.ones((1, planes.shape[1], 1), np.float64) if weight is None
                 else cls._lane(weight, planes.shape[1]))
        weighted = drive * gamma
        dx = (rstd * (weighted - weighted.mean(axis=2, keepdims=True)
                     - hat * (weighted * hat).mean(axis=2, keepdims=True)))
        return (dx.reshape(np.asarray(values).shape).astype(np.float32),
                (drive * hat).sum(axis=(0, 2)).astype(np.float32),
                drive.sum(axis=(0, 2)).astype(np.float32))

    @classmethod
    def _record_gradient(cls, values, grad, mean, var, weight=None, eps=1e-5):
        planes = cls._planes(values)
        drive = cls._planes(grad)
        saved_mean = np.asarray(mean).astype(np.float64)[None, :, None]
        rstd = (1.0 / np.sqrt(np.asarray(var).astype(np.float64) + eps)
                )[None, :, None]
        hat = (planes - saved_mean) * rstd
        gamma = (np.ones((1, planes.shape[1], 1), np.float64) if weight is None
                 else cls._lane(weight, planes.shape[1]))
        return ((drive * gamma * rstd).reshape(np.asarray(values).shape)
                .astype(np.float32),
                (drive * hat).sum(axis=(0, 2)).astype(np.float32),
                drive.sum(axis=(0, 2)).astype(np.float32))

    def _affine(self, channels, rng):
        weight = (rng.random(channels) + 0.5).astype(np.float32)
        bias = (rng.random(channels) - 0.5).astype(np.float32)
        return weight, bias

    def _payload(self, shape, rng):
        return (rng.standard_normal(shape) * 0.9 + 1.3).astype(np.float32)

    def _records(self, channels, rng):
        mean = (rng.standard_normal(channels) * 0.2).astype(np.float32)
        var = (rng.random(channels) + 0.5).astype(np.float32)
        return mean, var

    def test_a_plane_answers_its_own_moments(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(201)
        for shape in self.SHAPES:
            payload = self._payload(shape, rng)
            weight, bias = self._affine(shape[1], rng)
            for name, args in (("none", (None, None)),
                               ("weight", (weight, None)),
                               ("bias", (None, bias)),
                               ("both", (weight, bias))):
                with self.subTest(shape=shape, affine=name):
                    want = self._walk(payload, *args)[0]
                    terms = [None if a is None else self._device(a) for a in args]
                    got = F.instance_norm(self._device(payload), terms[0],
                                          terms[1], None, None, True, 0.1,
                                          1e-5).cpu().numpy()
                    np.testing.assert_allclose(
                        got.astype(np.float64), want.astype(np.float64),
                        rtol=1e-4, atol=1e-4,
                        err_msg=f"plane of {shape} with affine {name}")

    def test_a_record_keeps_the_running_average_of_a_channel(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(202)
        for shape in self.SHAPES[:4]:
            payload = self._payload(shape, rng)
            weight, bias = self._affine(shape[1], rng)
            kept_mean, kept_var = self._records(shape[1], rng)
            _, mean, unbiased = self._walk(payload, weight, bias)
            for momentum in (0.05, 0.2, 1.0):
                with self.subTest(shape=shape, momentum=momentum):
                    records = [self._device(kept_mean.copy()),
                               self._device(kept_var.copy())]
                    F.instance_norm(self._device(payload), self._device(weight),
                                    self._device(bias), records[0], records[1],
                                    True, momentum, 1e-5)
                    np.testing.assert_allclose(
                        records[0].cpu().numpy().astype(np.float64),
                        (1.0 - momentum) * kept_mean.astype(np.float64)
                        + momentum * mean.mean(axis=0),
                        rtol=1e-4, atol=1e-5,
                        err_msg="running mean of a channel")
                    np.testing.assert_allclose(
                        records[1].cpu().numpy().astype(np.float64),
                        (1.0 - momentum) * kept_var.astype(np.float64)
                        + momentum * unbiased.mean(axis=0),
                        rtol=1e-4, atol=1e-5,
                        err_msg="running spread kept unbiased")

    def test_a_record_the_walk_leaves_alone_stands(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(203)
        shape = (3, 4, 5)
        payload = self._payload(shape, rng)
        kept_mean, kept_var = self._records(shape[1], rng)
        held = [self._device(kept_mean.copy()), self._device(kept_var.copy())]
        before = [t.cpu().numpy().copy() for t in held]
        F.instance_norm(self._device(payload), None, None, held[0], held[1],
                        True, 0.0, 1e-5)
        for index in (0, 1):
            self.assertTrue(
                np.array_equal(before[index], held[index].cpu().numpy()),
                msg=f"record {index} moved under no weight")

        read = [self._device(kept_mean.copy()), self._device(kept_var.copy())]
        out = F.instance_norm(self._device(payload), None, None, read[0],
                              read[1], False, 0.4, 1e-5)
        self.assertEqual(out.device.is_vulkan(), True)
        for index in (0, 1):
            self.assertTrue(
                np.array_equal(before[index], read[index].cpu().numpy()),
                msg=f"record {index} moved while it was only read")

    def test_a_saved_record_names_the_plane(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(204)
        for shape in self.SHAPES:
            payload = self._payload(shape, rng)
            weight, bias = self._affine(shape[1], rng)
            saved_mean, saved_var = self._records(shape[1], rng)
            for name, args in (("none", (None, None)),
                               ("weight", (weight, None)),
                               ("bias", (None, bias)),
                               ("both", (weight, bias))):
                with self.subTest(shape=shape, affine=name):
                    want = self._from_records(payload, saved_mean, saved_var,
                                              *args)
                    terms = [None if a is None else self._device(a) for a in args]
                    got = F.instance_norm(self._device(payload), terms[0],
                                          terms[1], self._device(saved_mean),
                                          self._device(saved_var), False, 0.1,
                                          1e-5).cpu().numpy()
                    np.testing.assert_allclose(
                        got.astype(np.float64), want.astype(np.float64),
                        rtol=1e-4, atol=1e-4,
                        err_msg=f"plane read from records, affine {name}")

    def test_a_narrow_plane_keeps_its_own_width(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(205)
        shape = (2, 5, 3, 4)
        payload = self._payload(shape, rng)
        weight, bias = self._affine(shape[1], rng)
        want, mean, unbiased = self._walk(payload, weight, bias)

        narrow = F.instance_norm(self._device(np.asarray(payload, np.float16)),
                                 self._device(np.asarray(weight, np.float16)),
                                 self._device(np.asarray(bias, np.float16)),
                                 None, None, True, 0.1, 1e-5)
        self.assertEqual(str(narrow.dtype), "tensorplay.float16")
        np.testing.assert_allclose(
            narrow.cpu().numpy().astype(np.float32).astype(np.float64),
            want.astype(np.float64), rtol=6e-3, atol=6e-3,
            err_msg="narrow plane answered narrow")

        kept_mean = self._device(np.zeros(shape[1], np.float16))
        kept_var = self._device(np.ones(shape[1], np.float16))
        F.instance_norm(self._device(np.asarray(payload, np.float16)), None,
                        None, kept_mean, kept_var, True, 0.3, 1e-5)
        self.assertEqual(str(kept_mean.dtype), "tensorplay.float16")
        np.testing.assert_allclose(
            kept_mean.cpu().numpy().astype(np.float32).astype(np.float64),
            0.3 * mean.mean(axis=0), rtol=6e-3, atol=6e-3,
            err_msg="narrow record averaged wide and kept narrow")
        np.testing.assert_allclose(
            kept_var.cpu().numpy().astype(np.float32).astype(np.float64),
            0.7 + 0.3 * unbiased.mean(axis=0), rtol=6e-3, atol=6e-3,
            err_msg="narrow record of a spread")

    def test_the_gradient_of_a_plane_walk(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(206)
        for shape in self.SHAPES[:4]:
            payload = self._payload(shape, rng)
            grad = (rng.standard_normal(shape) * 0.4).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            for name, args in (("none", (None, None)),
                               ("weight", (weight, None)),
                               ("bias", (None, bias)),
                               ("both", (weight, bias))):
                with self.subTest(shape=shape, affine=name):
                    want = self._plane_gradient(payload, grad,
                                                None if args[0] is None
                                                else args[0])
                    terms = [None if a is None else self._device(a) for a in args]
                    got = F.instance_norm_backward(self._device(grad),
                                                   self._device(payload),
                                                   terms[0], terms[1], None,
                                                   None, True, 1e-5)
                    for index in (0, 1, 2):
                        if index > 0 and args[index - 1] is None:
                            self.assertTrue(
                                self._absent(got[index]),
                                msg=f"lane {index} answered when unasked")
                            continue
                        np.testing.assert_allclose(
                            got[index].cpu().numpy().astype(np.float64),
                            want[index].astype(np.float64),
                            rtol=1e-4, atol=1e-4,
                            err_msg=f"plane gradient lane {index}, {name}")
                        self.assertFalse(self._absent(got[index]))

    def test_the_gradient_read_against_the_records(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(207)
        for shape in self.SHAPES[:4]:
            payload = self._payload(shape, rng)
            grad = (rng.standard_normal(shape) * 0.4).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            saved_mean, saved_var = self._records(shape[1], rng)
            for name, args in (("none", (None, None)),
                               ("weight", (weight, None)),
                               ("bias", (None, bias)),
                               ("both", (weight, bias))):
                with self.subTest(shape=shape, affine=name):
                    dx, dw, db = self._record_gradient(payload, grad, saved_mean,
                                                       saved_var, args[0])
                    want = (dx, dw if args[0] is not None else None,
                            db if args[1] is not None else None)
                    terms = [None if a is None else self._device(a) for a in args]
                    got = F.instance_norm_backward(
                        self._device(grad), self._device(payload), terms[0],
                        terms[1], self._device(saved_mean),
                        self._device(saved_var), False, 1e-5)
                    for index in (0, 1, 2):
                        if want[index] is None:
                            self.assertTrue(self._absent(got[index]))
                            continue
                        np.testing.assert_allclose(
                            got[index].cpu().numpy().astype(np.float64),
                            want[index].astype(np.float64),
                            rtol=1e-4, atol=1e-4,
                            err_msg=f"record gradient lane {index}, {name}")

    def test_the_gradient_reaches_every_plane_of_the_walk(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(208)
        shape = (2, 5, 3, 4)
        payload = self._payload(shape, rng)
        grad = (rng.standard_normal(shape) * 0.4).astype(np.float32)
        weight, bias = self._affine(shape[1], rng)
        want = self._plane_gradient(payload, grad, weight)

        activation = self._device(payload)
        activation.requires_grad_(True)
        scale = self._device(weight)
        scale.requires_grad_(True)
        shift = self._device(bias)
        shift.requires_grad_(True)
        answered = F.instance_norm(activation, scale, shift, None, None, True,
                                   0.1, 1e-5)
        answered.backward(self._device(grad))
        for label, got, expected in (("plane", activation.grad, want[0]),
                                     ("weight", scale.grad, want[1]),
                                     ("bias", shift.grad, want[2])):
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float64),
                expected.astype(np.float64), rtol=1e-4, atol=1e-4,
                err_msg=f"gradient carried back to the {label}")

        saved_mean, saved_var = self._records(shape[1], rng)
        record_want = self._record_gradient(payload, grad, saved_mean,
                                            saved_var, weight)
        activation = self._device(payload)
        activation.requires_grad_(True)
        scale = self._device(weight)
        scale.requires_grad_(True)
        answered = F.instance_norm(activation, scale, None,
                                   self._device(saved_mean),
                                   self._device(saved_var), False, 0.1, 1e-5)
        answered.backward(self._device(grad))
        np.testing.assert_allclose(
            activation.grad.cpu().numpy().astype(np.float64),
            record_want[0].astype(np.float64), rtol=1e-4, atol=1e-4,
            err_msg="gradient carried back over the records")
        np.testing.assert_allclose(
            scale.grad.cpu().numpy().astype(np.float64),
            record_want[1].astype(np.float64), rtol=1e-4, atol=1e-4,
            err_msg="weight carried back over the records")

    def test_a_record_the_caller_keeps_on_the_host_is_written_home(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(209)
        shape = (3, 4, 6)
        payload = self._payload(shape, rng)
        weight, bias = self._affine(shape[1], rng)
        _, mean, unbiased = self._walk(payload, weight, bias)
        kept_mean = self._host(np.zeros(shape[1], np.float32))
        kept_var = self._host(np.ones(shape[1], np.float32))
        out = F.instance_norm(self._device(payload), self._device(weight),
                              self._device(bias), kept_mean, kept_var, True,
                              0.25, 1e-5)
        self.assertTrue(out.device.is_vulkan())
        self.assertFalse(kept_mean.device.is_vulkan())
        np.testing.assert_allclose(
            kept_mean.numpy().astype(np.float64), 0.25 * mean.mean(axis=0),
            rtol=1e-4, atol=1e-5, err_msg="host record of a mean")
        np.testing.assert_allclose(
            kept_var.numpy().astype(np.float64), 0.75 + 0.25 * unbiased.mean(axis=0),
            rtol=1e-4, atol=1e-5, err_msg="host record of a spread")

    def test_the_device_walk_answers_the_host_walk(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(210)
        for shape in self.SHAPES:
            payload = self._payload(shape, rng)
            grad = (rng.standard_normal(shape) * 0.4).astype(np.float32)
            weight, bias = self._affine(shape[1], rng)
            saved_mean, saved_var = self._records(shape[1], rng)
            with self.subTest(shape=shape):
                host_records = [self._host(kept.copy())
                                for kept in (saved_mean, saved_var)]
                device_records = [self._device(kept.copy())
                                  for kept in (saved_mean, saved_var)]
                on_host = F.instance_norm(self._host(payload),
                                          self._host(weight), self._host(bias),
                                          host_records[0], host_records[1],
                                          True, 0.15, 1e-5)
                on_device = F.instance_norm(self._device(payload),
                                            self._device(weight),
                                            self._device(bias),
                                            device_records[0],
                                            device_records[1], True, 0.15, 1e-5)
                np.testing.assert_allclose(
                    on_device.cpu().numpy().astype(np.float64),
                    on_host.numpy().astype(np.float64), rtol=1e-4, atol=1e-4,
                    err_msg="plane walked on the device, not on the host")
                for index in (0, 1):
                    np.testing.assert_allclose(
                        device_records[index].cpu().numpy().astype(np.float64),
                        host_records[index].numpy().astype(np.float64),
                        rtol=1e-4, atol=1e-5,
                        err_msg=f"record {index} averaged another way")
                host_grad = F.instance_norm_backward(
                    self._host(grad), self._host(payload), self._host(weight),
                    self._host(bias), host_records[0], host_records[1], False,
                    1e-5)
                device_grad = F.instance_norm_backward(
                    self._device(grad), self._device(payload),
                    self._device(weight), self._device(bias),
                    device_records[0], device_records[1], False, 1e-5)
                for index in (0, 1, 2):
                    np.testing.assert_allclose(
                        device_grad[index].cpu().numpy().astype(np.float64),
                        host_grad[index].numpy().astype(np.float64),
                        rtol=1e-4, atol=1e-4,
                        err_msg=f"record gradient lane {index} differs")

    def test_a_walk_over_no_plane_answers_as_not_a_number(self):
        import tensorplay.functional as F
        shape = (0, 4, 6)
        kept_mean = self._device(np.zeros(4, np.float32))
        kept_var = self._device(np.ones(4, np.float32))
        out = F.instance_norm(self._device(np.zeros(shape, np.float32)), None,
                              None, kept_mean, kept_var, True, 0.1, 1e-5)
        self.assertEqual(tuple(out.shape), shape)
        self.assertEqual(out.numel(), 0)
        self.assertTrue(np.all(np.isnan(kept_mean.cpu().numpy())),
                        msg="a mean averaged over no plane")
        self.assertTrue(np.all(np.isnan(kept_var.cpu().numpy())),
                        msg="a spread averaged over no plane")

        # A plane of one element has no unbiased spread; its own center is that
        # element, so the answer sits on the affine terms alone.
        single = np.full((2, 3, 1), 5.0, np.float32)
        one_mean = self._device(np.zeros(3, np.float32))
        one_var = self._device(np.ones(3, np.float32))
        answered = F.instance_norm(self._device(single), None, None, one_mean,
                                   one_var, True, 0.5, 1e-5)
        np.testing.assert_allclose(answered.cpu().numpy(),
                                   np.zeros(single.shape, np.float32),
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(one_mean.cpu().numpy(), np.full(3, 2.5),
                                   rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.isnan(one_var.cpu().numpy())),
                        msg="a plane of one element has no spread")

        # A record only read answers the payload and is left standing.
        read_mean = self._device(np.full(4, -0.5, np.float32))
        read_var = self._device(np.full(4, 2.0, np.float32))
        empty = F.instance_norm(self._device(np.zeros(shape, np.float32)), None,
                                None, read_mean, read_var, False, 0.1, 1e-5)
        self.assertEqual(empty.numel(), 0)
        np.testing.assert_allclose(read_mean.cpu().numpy(),
                                   np.full(4, -0.5, np.float32),
                                   rtol=0, atol=0)

    def test_a_shape_or_a_record_that_does_not_line_up_refuses(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(211)
        shape = (2, 4, 3)
        payload = self._payload(shape, rng)
        grad = (rng.standard_normal(shape) * 0.4).astype(np.float32)
        records = (self._device(np.zeros(4, np.float32)),
                   self._device(np.ones(4, np.float32)))
        device = self._device(payload)
        with self.subTest(kind="a payload that counts"):
            with self.assertRaisesRegex(
                    RuntimeError, "Float32 and Float16"):
                F.instance_norm(self._device(np.zeros(shape, np.int32)), None,
                                None, None, None, True, 0.1, 1e-5)
        with self.subTest(kind="a plane with no span to fold over"):
            with self.assertRaisesRegex(
                    RuntimeError, "supports 3d to 4d tensors"):
                F.instance_norm(self._device(np.zeros((3, 4), np.float32)),
                                None, None, None, None, True, 0.1, 1e-5)
        with self.subTest(kind="no channels to name a plane"):
            with self.assertRaisesRegex(
                    RuntimeError, "at least one channel"):
                F.instance_norm(self._device(np.zeros((2, 0, 4), np.float32)),
                                None, None, None, None, True, 0.1, 1e-5)
        with self.subTest(kind="records missing where they are read"):
            with self.assertRaisesRegex(
                    RuntimeError, "must both be given"):
                F.instance_norm(device, None, None, None, None, False, 0.1, 1e-5)
        with self.subTest(kind="one record missing"):
            with self.assertRaisesRegex(
                    RuntimeError, "must both be given"):
                F.instance_norm(device, None, None, records[0], None, False,
                                0.1, 1e-5)
        with self.subTest(kind="a lane count that is not the channel count"):
            with self.assertRaisesRegex(
                    RuntimeError, "one lane per channel"):
                F.instance_norm(device, self._device(np.ones(3, np.float32)),
                                None, None, None, True, 0.1, 1e-5)
        with self.subTest(kind="an affine term that counts"):
            with self.assertRaisesRegex(
                    RuntimeError, "affine parameter is kept at"):
                F.instance_norm(device, self._device(np.ones(4, np.int32)),
                                None, None, None, True, 0.1, 1e-5)
        with self.subTest(kind="a record wider than the walk"):
            with self.assertRaisesRegex(
                    RuntimeError, "running record is kept at"):
                F.instance_norm(device, None, None,
                                self._host(np.zeros(4, np.float64)),
                                self._host(np.ones(4, np.float64)), True, 0.1,
                                1e-5)
        with self.subTest(kind="a gradient of another width"):
            with self.assertRaisesRegex(
                    RuntimeError, "read in the width"):
                F.instance_norm_backward(
                    self._device(np.asarray(grad, np.float16)), device, None,
                    None, None, None, True, 1e-5)
        with self.subTest(kind="a gradient that misses a plane"):
            with self.assertRaisesRegex(
                    RuntimeError, "name the planes"):
                F.instance_norm_backward(
                    self._device(np.zeros((2, 4, 5), np.float32)), device, None,
                    None, None, None, True, 1e-5)
        with self.subTest(kind="records missing behind a gradient"):
            with self.assertRaisesRegex(
                    RuntimeError, "must both be given"):
                F.instance_norm_backward(
                    self._device(grad), device, None, None, None, None, False,
                    1e-5)
        with self.subTest(kind="a gradient over no channel"):
            with self.assertRaisesRegex(
                    RuntimeError, "at least one channel"):
                F.instance_norm_backward(
                    self._device(np.zeros((2, 0, 4), np.float32)),
                    self._device(np.zeros((2, 0, 4), np.float32)), None, None,
                    None, None, True, 1e-5)


class TestVulkanRenorm(unittest.TestCase):
    # Every span laid out along one named axis is measured as a vector, and a span
    # longer than the bound is pulled back to it: one factor scales a whole span, so
    # the answer keeps the shape the payload was read in and the axis that was named
    # stays standing.  A length is taken in single precision even from a narrow
    # payload, because a long span can leave the width of a lane, and the factor is
    # rounded once -- when it returns to that width to multiply -- which is why a
    # span exactly at the bound keeps its lanes and a length that is not a number,
    # never being over the bound, keeps the lanes it holds.  A bound of nothing
    # collapses a measuring span, a payload with one dimension has no span to
    # measure, and counting lanes are not measured at all.

    ORDERS = (1.0, 2.0, 3.0, math.inf)
    BOUNDS = (0.0, 0.5, 1.0, 4.0, 1e9)

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _wrap(values, axis):
        rank = np.asarray(values).ndim
        return axis + rank if axis < 0 else axis

    @classmethod
    def _lengths(cls, values, axis, order):
        # The measure of every span, held along the axis it leaves standing.
        wide = np.asarray(values, dtype=np.float64)
        named = cls._wrap(wide, axis)
        span = tuple(a for a in range(wide.ndim) if a != named)
        if np.isinf(order):
            return np.max(np.abs(wide), axis=span, keepdims=True)
        return (np.abs(wide) ** order).sum(axis=span, keepdims=True) ** (1.0 / order)

    @classmethod
    def _truth(cls, values, order, axis, bound):
        wide = np.asarray(values, dtype=np.float64)
        lengths = cls._lengths(wide, axis, order)
        factors = np.where(lengths > bound, bound / (lengths + 1e-7), 1.0)
        return (wide * factors).astype(np.float32)

    def test_a_span_longer_than_the_bound_is_pulled_back_to_it(self):
        rng = np.random.default_rng(11)
        matrix = (rng.standard_normal((4, 5)) * 3.0).astype(np.float32)
        for order in self.ORDERS:
            for bound in self.BOUNDS:
                for axis in (0, 1, -1):
                    truth = self._truth(matrix, order, axis, bound)
                    device = tensorplay.renorm(
                        self._device(matrix), order, axis, bound)
                    host = tensorplay.renorm(
                        self._host(matrix), order, axis, bound)
                    with self.subTest(order=order, bound=bound, axis=axis):
                        self.assertEqual(device.dtype, tensorplay.float32)
                        self.assertEqual(tuple(device.shape), matrix.shape)
                        # The length is summed in single precision, so a long span
                        # answers a hair off the double-precision measure.
                        np.testing.assert_allclose(
                            device.detach().cpu().numpy(), truth,
                            rtol=1e-5, atol=1e-5)
                        np.testing.assert_allclose(
                            host.detach().cpu().numpy(), truth,
                            rtol=1e-5, atol=1e-5)

    def test_a_span_already_under_the_bound_keeps_every_lane(self):
        rng = np.random.default_rng(19)
        for order in self.ORDERS:
            matrix = (rng.standard_normal((3, 6)) * 2.0).astype(np.float32)
            for axis in (0, 1):
                with self.subTest(order=order, axis=axis):
                    answer = tensorplay.renorm(
                        self._device(matrix), order, axis, 1e9)
                    np.testing.assert_array_equal(
                        answer.detach().cpu().numpy(), matrix)

    def test_a_bound_of_nothing_collapses_a_measuring_span(self):
        rng = np.random.default_rng(23)
        matrix = (rng.standard_normal((5, 4)) * 7.0).astype(np.float32)
        for order in self.ORDERS:
            for axis in (0, 1):
                with self.subTest(order=order, axis=axis):
                    answer = tensorplay.renorm(
                        self._device(matrix), order, axis, 0.0)
                    lanes = answer.detach().cpu().numpy()
                    self.assertTrue(np.all(lanes == 0.0))

    def test_a_narrow_span_is_measured_wider_than_its_lane(self):
        # A span of nine thousand lanes squared leaves the width a half lane
        # holds: measured there its length would be infinity, the factor zero, and
        # every lane collapsed.  Measured wide, the span is only pulled back.
        payload = np.full((9000, 8), 60000.0, dtype=np.float16)
        truth = self._truth(payload, 2.0, 1, 1.0)
        answer = tensorplay.renorm(self._device(payload), 2.0, 1, 1.0)
        lanes = answer.detach().cpu().numpy()
        self.assertEqual(answer.dtype, tensorplay.float16)
        self.assertTrue(np.all(lanes != 0.0))
        self.assertTrue(np.all(np.isfinite(lanes)))
        np.testing.assert_allclose(
            lanes.astype(np.float64), truth.astype(np.float64),
            rtol=2e-3, atol=2e-3)

    def test_a_rank_three_walk_scales_the_axis_it_names(self):
        rng = np.random.default_rng(29)
        cube = (rng.standard_normal((2, 3, 4)) * 2.0).astype(np.float32)
        for axis in (0, 1, 2):
            for order in (1.0, 2.0, math.inf):
                with self.subTest(axis=axis, order=order):
                    truth = self._truth(cube, order, axis, 1.5)
                    answer = tensorplay.renorm(
                        self._device(cube), order, axis, 1.5)
                    np.testing.assert_allclose(
                        answer.detach().cpu().numpy(), truth,
                        rtol=1e-5, atol=1e-5)

    def test_a_rank_four_walk_scales_the_axis_it_names(self):
        rng = np.random.default_rng(31)
        stack = (rng.standard_normal((2, 2, 3, 3)) * 2.0).astype(np.float32)
        for axis in (0, 1, 2, 3):
            with self.subTest(axis=axis):
                truth = self._truth(stack, 2.0, axis, 1.0)
                answer = tensorplay.renorm(self._device(stack), 2.0, axis, 1.0)
                np.testing.assert_allclose(
                    answer.detach().cpu().numpy(), truth, rtol=1e-5, atol=1e-5)

    def test_a_span_with_no_lane_measures_as_nothing(self):
        for shape in ((0, 3), (3, 0), (1, 1)):
            payload = np.zeros(shape, dtype=np.float32)
            with self.subTest(shape=shape):
                answer = tensorplay.renorm(
                    self._device(payload), 2.0, 1, 1.0)
                self.assertEqual(tuple(answer.shape), shape)
                self.assertEqual(answer.numel(), payload.size)
                if payload.size:
                    np.testing.assert_array_equal(
                        answer.detach().cpu().numpy(), payload)

    def test_the_write_back_lands_where_the_span_was_read(self):
        rng = np.random.default_rng(37)
        matrix = (rng.standard_normal((4, 5)) * 3.0).astype(np.float32)
        truth = self._truth(matrix, 2.0, 1, 1.0)
        payload = self._device(matrix)
        answer = tensorplay.renorm_(payload, 2.0, 1, 1.0)
        self.assertIs(answer, payload)
        np.testing.assert_allclose(
            payload.detach().cpu().numpy(), truth, rtol=1e-5, atol=1e-5)
        host = self._host(matrix)
        tensorplay.renorm_(host, 2.0, 1, 1.0)
        np.testing.assert_allclose(
            host.detach().cpu().numpy(), truth, rtol=1e-5, atol=1e-5)

    def test_a_bound_carries_both_the_factor_and_the_measure_it_moves(self):
        # A pulled-back span is not rescaled by a constant held off its gradient:
        # the bound it is measured against moves with its lanes, so the gradient
        # carries the rescaled gradient less the term the moved factor takes off
        # the inner product the span holds with it.
        rng = np.random.default_rng(41)
        matrix = (rng.standard_normal((6, 4)) * 3.0).astype(np.float32)
        bound = 1.0
        # Axis one stays standing, so a span is a column: every measure is taken
        # down the rows and one factor scales a whole column.
        wide = matrix.astype(np.float64)
        length = np.sqrt((wide * wide).sum(axis=0, keepdims=True))
        ones = np.ones_like(wide)
        inner = (wide * ones).sum(axis=0, keepdims=True)
        pulled = length > bound
        factor = np.where(pulled, bound / (length + 1e-7), 1.0)
        correction = np.where(
            pulled, bound * inner / (length * (length + 1e-7) ** 2), 0.0)
        truth = factor * ones - wide * correction

        leaf = self._device(matrix).requires_grad_(True)
        answer = tensorplay.renorm(leaf, 2.0, 1, bound)
        answer.backward(self._device(np.ones_like(matrix)))
        self.assertEqual(tuple(leaf.grad.shape), matrix.shape)
        np.testing.assert_allclose(
            leaf.grad.detach().cpu().numpy(), truth, rtol=2e-3, atol=2e-4)

    def test_a_walk_that_keeps_every_lane_keeps_every_gradient(self):
        rng = np.random.default_rng(43)
        matrix = (rng.standard_normal((5, 3)) * 2.0).astype(np.float32)
        leaf = self._device(matrix).requires_grad_(True)
        answer = tensorplay.renorm(leaf, 2.0, 1, 1e9)
        answer.backward(tensorplay.ones_like(answer))
        np.testing.assert_array_equal(
            leaf.grad.detach().cpu().numpy(), np.ones_like(matrix))

    def test_a_span_that_is_not_a_number_is_not_over_the_bound(self):
        matrix = np.array(
            [[3.0, 4.0], [1.0, np.nan], [0.5, 0.5]], dtype=np.float32)
        # Axis zero stays standing, so a span is a row: the measure that is not a
        # number is not over the bound, and it does not reach into the row next
        # to it -- one is pulled back, one keeps its lanes, one stands short.
        truth = self._truth(matrix, 2.0, 0, 1.0)
        answer = tensorplay.renorm(self._device(matrix), 2.0, 0, 1.0)
        lanes = answer.detach().cpu().numpy()
        np.testing.assert_allclose(lanes[0], truth[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(lanes[1][0], 1.0)
        self.assertTrue(math.isnan(float(lanes[1][1])))
        np.testing.assert_array_equal(lanes[2], matrix[2])
        host = tensorplay.renorm(self._host(matrix), 2.0, 0, 1.0)
        np.testing.assert_array_equal(
            host.detach().cpu().numpy(), lanes)

    def test_a_walk_needs_a_measuring_payload_of_at_least_two_dimensions(self):
        flat = np.zeros(4, dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "at least two dimensions"):
            tensorplay.renorm(self._device(flat), 2.0, 0, 1.0)
        counts = np.zeros((2, 2), dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            tensorplay.renorm(self._device(counts), 2.0, 0, 1.0)
        matrix = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "positive norm order"):
            tensorplay.renorm(self._device(matrix), 0.0, 0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "positive norm order"):
            tensorplay.renorm(self._device(matrix), -2.0, 0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "non-negative length"):
            tensorplay.renorm(self._device(matrix), 2.0, 0, -1.0)
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            tensorplay.renorm(self._device(matrix), 2.0, 2, 1.0)
        # The fourth rank is the widest walk the storage carries.
        with self.assertRaisesRegex(RuntimeError, "ndim <= 4"):
            tensorplay.renorm(
                self._device(np.zeros((2, 3, 4, 5, 6), np.float32)),
                2.0, 1, 1.0)


class TestVulkanBinaryCrossEntropy(unittest.TestCase):
    # A binary likelihood reads both planes as probabilities, so every lane has to
    # sit between the bounds a logarithm is taken of.  Each logarithm is clamped at
    # -100, which is what keeps a lane exactly on a bound finite, and the gradient
    # divides by the product of the two distances to those bounds held off zero.
    # The logit form answers through log_sigmoid instead, so a saturated logit
    # stays finite without ever forming a probability.  The walk reads single
    # precision however narrow the payload is, and the answer returns to the width
    # the payload has.

    SHAPES = ((4,), (2, 5), (3, 4, 2), (2, 3, 4, 5))

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _probabilities(rng, shape):
        # A lane exactly on zero would answer the clamp, so the plane is held
        # inside the bounds.
        return (rng.random(shape) * 0.97 + 0.015).astype(np.float32)

    @staticmethod
    def _likelihood(x, t, weight=None):
        wide = np.asarray(x).astype(np.float64)
        labels = np.broadcast_to(np.asarray(t).astype(np.float64), wide.shape)
        loss = ((labels - 1.0) * np.maximum(np.log1p(-wide), -100.0)
                - labels * np.maximum(np.log(wide), -100.0))
        if weight is not None:
            loss = loss * np.broadcast_to(
                np.asarray(weight).astype(np.float64), wide.shape)
        return loss

    @staticmethod
    def _likelihood_gradient(x, t, weight=None):
        wide = np.asarray(x).astype(np.float64)
        labels = np.broadcast_to(np.asarray(t).astype(np.float64), wide.shape)
        gradient = (wide - labels) / np.maximum((1.0 - wide) * wide, 1e-12)
        if weight is not None:
            gradient = gradient * np.broadcast_to(
                np.asarray(weight).astype(np.float64), wide.shape)
        return gradient

    @staticmethod
    def _logit_likelihood(x, t, weight=None, pos_weight=None):
        wide = np.asarray(x).astype(np.float64)
        labels = np.broadcast_to(np.asarray(t).astype(np.float64), wide.shape)
        log_sigmoid = -np.logaddexp(0.0, -wide)
        if pos_weight is not None:
            log_sigmoid = log_sigmoid * ((np.broadcast_to(
                np.asarray(pos_weight).astype(np.float64), wide.shape) - 1.0)
                * labels + 1.0)
        loss = (1.0 - labels) * wide - log_sigmoid
        if weight is not None:
            loss = loss * np.broadcast_to(
                np.asarray(weight).astype(np.float64), wide.shape)
        return loss

    def test_a_probability_answers_the_loss_of_its_lane(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(211)
        for shape in self.SHAPES:
            payload = self._probabilities(rng, shape)
            labels = (rng.random(shape) > 0.5).astype(np.float32)
            weight = (rng.random(shape) + 0.5).astype(np.float32)
            for reduction in (0, 1, 2):
                for name, affine in (("none", None), ("weight", weight)):
                    with self.subTest(shape=shape, reduction=reduction,
                                      affine=name):
                        got = F.binary_cross_entropy(
                            self._device(payload), self._device(labels),
                            weight=None if affine is None
                            else self._device(affine),
                            reduction=reduction).cpu().numpy()
                        truth = self._likelihood(payload, labels, affine)
                        if reduction == 1:
                            truth = truth.mean()
                        elif reduction == 2:
                            truth = truth.sum()
                        np.testing.assert_allclose(
                            got, truth, rtol=2e-3, atol=2e-3,
                            err_msg=f"reduction {reduction} of {shape}")

    def test_the_host_likelihood_answers_the_same(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(223)
        for shape in ((4,), (2, 5), (3, 4, 2)):
            payload = self._probabilities(rng, shape)
            labels = (rng.random(shape) > 0.5).astype(np.float32)
            weight = (rng.random(shape) + 0.5).astype(np.float32)
            for reduction in (0, 1, 2):
                with self.subTest(shape=shape, reduction=reduction):
                    device = F.binary_cross_entropy(
                        self._device(payload), self._device(labels),
                        weight=self._device(weight),
                        reduction=reduction).cpu().numpy()
                    host = F.binary_cross_entropy(
                        self._host(payload), self._host(labels),
                        weight=self._host(weight),
                        reduction=reduction).numpy()
                    self.assertEqual(device.dtype, host.dtype)
                    np.testing.assert_allclose(device, host, rtol=2e-3,
                                               atol=2e-3)

    def test_a_narrower_plane_broadcasts_onto_the_walk(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(227)
        payload = self._probabilities(rng, (2, 3, 4))
        labels = (rng.random((3, 1)) > 0.5).astype(np.float32)
        weight = (rng.random((3, 1)) + 0.5).astype(np.float32)
        with self.subTest(kind="a plane narrower than the payload"):
            got = F.binary_cross_entropy(
                self._device(payload), self._device(labels),
                weight=self._device(weight)).cpu().numpy()
            np.testing.assert_allclose(
                got, self._likelihood(payload, labels, weight).mean(),
                rtol=2e-3, atol=2e-3)

    def test_a_gradient_hands_back_the_slope_of_the_loss(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(229)
        for shape in ((4,), (2, 5), (3, 4, 2)):
            payload = self._probabilities(rng, shape)
            labels = (rng.random(shape) > 0.5).astype(np.float32)
            weight = (rng.random(shape) + 0.5).astype(np.float32)
            incoming = (rng.random(shape) - 0.5).astype(np.float32)
            for reduction in (0, 1, 2):
                for name, affine in (("none", None), ("weight", weight)):
                    with self.subTest(shape=shape, reduction=reduction,
                                      affine=name):
                        got = F.binary_cross_entropy_backward(
                            self._device(incoming), self._device(payload),
                            self._device(labels),
                            weight=None if affine is None
                            else self._device(affine),
                            reduction=reduction).cpu().numpy()
                        truth = self._likelihood_gradient(
                            payload, labels, affine) * incoming.astype(np.float64)
                        if reduction == 1:
                            truth = truth / payload.size
                        np.testing.assert_allclose(
                            got, truth, rtol=2e-2, atol=2e-2,
                            err_msg=f"reduction {reduction} of {shape}")
                        host = F.binary_cross_entropy_backward(
                            self._host(incoming), self._host(payload),
                            self._host(labels),
                            weight=None if affine is None
                            else self._host(affine),
                            reduction=reduction).numpy()
                        np.testing.assert_allclose(got, host, rtol=2e-2,
                                                   atol=2e-2)

    def test_a_logit_answers_without_becoming_a_probability(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(233)
        for shape in ((5,), (2, 6), (3, 4, 2)):
            logits = (rng.standard_normal(shape) * 3).astype(np.float32)
            labels = (rng.random(shape) > 0.5).astype(np.float32)
            weight = (rng.random(shape) + 0.5).astype(np.float32)
            pos_weight = (rng.random(shape) * 4 + 0.25).astype(np.float32)
            for name, affine, positive in (
                    ("plain", None, None),
                    ("weight", weight, None),
                    ("pos_weight", None, pos_weight),
                    ("both", weight, pos_weight)):
                with self.subTest(shape=shape, affine=name):
                    got = F.binary_cross_entropy_with_logits(
                        self._device(logits), self._device(labels),
                        weight=None if affine is None else self._device(affine),
                        pos_weight=None if positive is None
                        else self._device(positive)).cpu().numpy()
                    truth = self._logit_likelihood(
                        logits, labels, affine, positive).mean()
                    np.testing.assert_allclose(got, truth, rtol=2e-3, atol=2e-3)
                    host = F.binary_cross_entropy_with_logits(
                        self._host(logits), self._host(labels),
                        weight=None if affine is None
                        else self._host(affine),
                        pos_weight=None if positive is None
                        else self._host(positive)).numpy()
                    np.testing.assert_allclose(got, host, rtol=2e-3, atol=2e-3)

    def test_a_saturated_logit_stays_finite(self):
        import tensorplay.functional as F
        logits = np.array([-60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0],
                          np.float32)
        for label in (0.0, 1.0):
            labels = np.full(logits.shape, label, np.float32)
            with self.subTest(label=label):
                got = F.binary_cross_entropy_with_logits(
                    self._device(logits), self._device(labels)).cpu().numpy()
                self.assertTrue(np.isfinite(got).all(),
                                "a saturated logit lost its headroom")
                np.testing.assert_allclose(
                    got, self._logit_likelihood(logits, labels).mean(),
                    rtol=1e-3, atol=1e-3)

    def test_a_lane_on_a_bound_answers_the_clamp(self):
        import tensorplay.functional as F
        payload = np.array([0.0, 1.0, 0.5], np.float32)
        labels = np.array([1.0, 0.0, 0.5], np.float32)
        with self.subTest(kind="the loss of a bound lane"):
            got = F.binary_cross_entropy(self._device(payload),
                                         self._device(labels),
                                         reduction=0).cpu().numpy()
            np.testing.assert_allclose(got, [100.0, 100.0, 0.6931472],
                                       rtol=1e-3, atol=1e-3)
        with self.subTest(kind="the gradient of a bound lane"):
            gradient = F.binary_cross_entropy_backward(
                self._device(np.ones(3, np.float32)), self._device(payload),
                self._device(labels), reduction=0).cpu().numpy()
            self.assertTrue(np.isfinite(gradient).all(),
                            "a bound lane divided by nothing")

    def test_a_narrow_plane_graduates_in_its_own_width(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(239)
        payload = (rng.random((2, 5)) * 0.9 + 0.05).astype(np.float16)
        labels = (rng.random((2, 5)) > 0.5).astype(np.float16)
        weight = (rng.random((2, 5)) + 0.5).astype(np.float16)
        incoming = (rng.random((2, 5)) - 0.5).astype(np.float16)
        for reduction in (0, 1, 2):
            with self.subTest(reduction=reduction, kind="the loss"):
                got = F.binary_cross_entropy(
                    self._device(payload), self._device(labels),
                    weight=self._device(weight),
                    reduction=reduction)
                host = F.binary_cross_entropy(
                    self._host(payload), self._host(labels),
                    weight=self._host(weight), reduction=reduction)
                self.assertEqual(got.dtype, host.dtype)
                np.testing.assert_allclose(
                    got.cpu().numpy().astype(np.float32),
                    host.numpy().astype(np.float32), rtol=4e-3, atol=4e-3)
        with self.subTest(kind="the gradient"):
            got = F.binary_cross_entropy_backward(
                self._device(incoming), self._device(payload),
                self._device(labels), weight=self._device(weight),
                reduction=1)
            host = F.binary_cross_entropy_backward(
                self._host(incoming), self._host(payload), self._host(labels),
                weight=self._host(weight), reduction=1)
            self.assertEqual(got.dtype, host.dtype)
            np.testing.assert_allclose(
                got.cpu().numpy().astype(np.float32),
                host.numpy().astype(np.float32), rtol=4e-3, atol=4e-3)
        with self.subTest(kind="a logit average kept wide"):
            average = F.binary_cross_entropy_with_logits(
                self._device(payload), self._device(labels))
            self.assertEqual(average.dtype, tensorplay.float32)

    def test_a_lane_outside_the_bounds_refuses(self):
        import tensorplay.functional as F
        labels = np.array([0.0, 1.0, 1.0], np.float32)
        for name, payload, pattern in (
                ("above one", np.array([0.2, 0.5, 1.5], np.float32),
                 "all elements of input"),
                ("below zero", np.array([0.2, -0.1, 0.7], np.float32),
                 "all elements of input"),
                ("not a measurement", np.array([0.2, np.nan, 0.7], np.float32),
                 "all elements of input")):
            with self.subTest(kind=name):
                with self.assertRaisesRegex(
                        RuntimeError, "should be between 0 and 1"):
                    F.binary_cross_entropy(self._device(payload),
                                           self._device(labels))
        with self.subTest(kind="a label outside the bounds"):
            with self.assertRaisesRegex(
                    RuntimeError, "all elements of target"):
                F.binary_cross_entropy(
                    self._device(np.array([0.2, 0.5, 0.7], np.float32)),
                    self._device(np.array([0.0, 2.0, 1.0], np.float32)))
        with self.subTest(kind="a count"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                F.binary_cross_entropy(
                    self._device(np.array([0, 1, 1], np.int32)),
                    self._device(np.array([0, 1, 1], np.int32)))
        with self.subTest(kind="an unknown reduction"):
            with self.assertRaisesRegex(RuntimeError, "invalid reduction mode"):
                F.binary_cross_entropy(
                    self._device(np.array([0.2], np.float32)),
                    self._device(np.array([0.0], np.float32)), reduction=5)


class TestVulkanAbsoluteDeviationLoss(unittest.TestCase):
    # The loss one element answers is how far it sits from the value it was
    # matched with, so the walk holds no bound to refuse and no denominator to
    # hold.  These tests keep the three reduction modes, the side a gradient
    # takes from a deviation, the width a narrow payload keeps, and the two
    # answers an empty plane gives.

    SHAPES = ((4,), (2, 5), (3, 4, 2), (2, 3, 4, 5))

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _plane(rng, shape):
        # A deviation is read anywhere on the line, so the plane straddles zero.
        return ((rng.random(shape) - 0.5) * 6.0).astype(np.float32)

    @staticmethod
    def _deviation(values, labels):
        wide = np.asarray(values).astype(np.float64)
        matched = np.broadcast_to(np.asarray(labels).astype(np.float64),
                                  wide.shape)
        return np.abs(wide - matched)

    @staticmethod
    def _reduced(deviation, reduction):
        if reduction == 0:
            return deviation
        return deviation.mean() if reduction == 1 else deviation.sum()

    @staticmethod
    def _deviation_gradient(values, labels, incoming, reduction):
        slope = np.sign(np.asarray(values).astype(np.float64)
                        - np.asarray(labels).astype(np.float64))
        gradient = slope * np.broadcast_to(
            np.asarray(incoming).astype(np.float64), slope.shape)
        if reduction == 1 and slope.size > 0:
            gradient = gradient / slope.size
        return gradient

    def test_a_deviation_answers_the_loss_of_its_lane(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(301)
        for shape in self.SHAPES:
            payload = self._plane(rng, shape)
            labels = self._plane(rng, shape)
            deviation = self._deviation(payload, labels)
            for reduction in (0, 1, 2):
                with self.subTest(shape=shape, reduction=reduction):
                    got = F.tp_l1_loss(
                        self._device(payload), self._device(labels),
                        reduction).cpu().numpy()
                    np.testing.assert_allclose(
                        got, self._reduced(deviation, reduction),
                        rtol=2e-3, atol=2e-3)

    def test_the_host_deviation_answers_the_same(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(307)
        payload = self._plane(rng, (2, 5))
        labels = self._plane(rng, (2, 5))
        deviation = self._deviation(payload, labels)
        for reduction in (0, 1, 2):
            with self.subTest(reduction=reduction):
                on_device = F.tp_l1_loss(
                    self._device(payload), self._device(labels),
                    reduction).cpu().numpy()
                answered = F.tp_l1_loss(
                    self._host(payload), self._host(labels),
                    reduction).cpu().numpy()
                np.testing.assert_allclose(
                    on_device, answered, rtol=2e-3, atol=2e-3)
                np.testing.assert_allclose(
                    on_device, self._reduced(deviation, reduction),
                    rtol=2e-3, atol=2e-3)

    def test_a_narrower_plane_matches_onto_the_walk(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(311)
        payload = self._plane(rng, (2, 3, 4))
        labels = self._plane(rng, (3, 1))
        with self.subTest(kind="a plane narrower than the payload"):
            got = F.tp_l1_loss(self._device(payload), self._device(labels), 1)
            np.testing.assert_allclose(
                got.cpu().numpy(), self._deviation(payload, labels).mean(),
                rtol=2e-3, atol=2e-3)
        with self.subTest(kind="a label lane that repeats"):
            got = F.tp_l1_loss(
                self._device(np.array([1.5, -2.0, 0.25], np.float32)),
                self._device(np.array(0.5, np.float32)), 2)
            np.testing.assert_allclose(
                got.cpu().numpy(), np.array([1.0, 2.5, 0.25]).sum(),
                rtol=2e-3, atol=2e-3)

    def test_a_gradient_hands_back_the_side_of_the_deviation(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(313)
        for shape in ((4,), (2, 5), (3, 4, 2)):
            payload = self._plane(rng, shape)
            labels = self._plane(rng, shape)
            incoming = (rng.random(shape) - 0.5).astype(np.float32)
            for reduction in (0, 1, 2):
                with self.subTest(shape=shape, reduction=reduction):
                    got = F.tp_l1_loss_backward(
                        self._device(incoming), self._device(payload),
                        self._device(labels), reduction).cpu().numpy()
                    np.testing.assert_allclose(
                        got,
                        self._deviation_gradient(
                            payload, labels, incoming, reduction),
                        rtol=2e-3, atol=2e-3)
        with self.subTest(kind="an incoming carried as one lane"):
            payload = self._plane(rng, (2, 5))
            labels = self._plane(rng, (2, 5))
            got = F.tp_l1_loss_backward(
                self._device(np.array(0.75, np.float32)),
                self._device(payload), self._device(labels), 1).cpu().numpy()
            np.testing.assert_allclose(
                got,
                self._deviation_gradient(payload, labels, 0.75, 1),
                rtol=2e-3, atol=2e-3)
        with self.subTest(kind="a lane that matches exactly"):
            got = F.tp_l1_loss_backward(
                self._device(np.ones(3, np.float32)),
                self._device(np.array([1.0, 2.0, 3.0], np.float32)),
                self._device(np.array([1.0, 0.0, 0.0], np.float32)),
                0).cpu().numpy()
            np.testing.assert_allclose(got, np.array([0.0, 1.0, 1.0]))

    def test_the_average_graduation_flows_back(self):
        import tensorplay.nn.functional as nnF
        rng = np.random.default_rng(317)
        payload = tensorplay.from_numpy(self._plane(rng, (2, 5))).to("vulkan")
        payload.requires_grad_(True)
        labels = tensorplay.from_numpy(self._plane(rng, (2, 5))).to("vulkan")
        average = nnF.l1_loss(payload, labels, reduction="mean")
        average.backward(tensorplay.from_numpy(np.array(1.0, np.float32)).to("vulkan"))
        flowed = payload.grad.detach().cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(
            flowed,
            np.sign(payload.detach().cpu().numpy().astype(np.float64)
                    - labels.cpu().numpy().astype(np.float64)) / 10.0,
            rtol=2e-3, atol=2e-3)

    def test_a_narrow_plane_graduates_in_its_own_width(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(331)
        payload = self._plane(rng, (3, 4)).astype(np.float16)
        labels = self._plane(rng, (3, 4)).astype(np.float16)
        with self.subTest(kind="an unreduced deviation kept narrow"):
            answered = F.tp_l1_loss(self._device(payload), self._device(labels), 0)
            self.assertEqual(answered.dtype, tensorplay.float16)
        with self.subTest(kind="an average kept narrow"):
            average = F.tp_l1_loss(self._device(payload), self._device(labels), 1)
            self.assertEqual(average.dtype, tensorplay.float16)
        with self.subTest(kind="a gradient kept narrow"):
            gradient = F.tp_l1_loss_backward(
                self._device(np.ones((3, 4), np.float16)),
                self._device(payload), self._device(labels), 1)
            self.assertEqual(gradient.dtype, tensorplay.float16)
        with self.subTest(kind="a narrow answer close to the wide one"):
            narrow = F.tp_l1_loss(
                self._device(payload), self._device(labels), 1).cpu().numpy()
            np.testing.assert_allclose(
                narrow,
                self._deviation(payload.astype(np.float32),
                               labels.astype(np.float32)).mean(),
                rtol=2e-2, atol=2e-2)

    def test_a_plane_without_lanes_answers_the_reduction(self):
        import tensorplay.functional as F
        empty = np.zeros((0, 4), np.float32)
        with self.subTest(kind="an average of nothing"):
            average = F.tp_l1_loss(self._device(empty), self._device(empty), 1)
            self.assertEqual(tuple(average.shape), ())
            self.assertTrue(np.isnan(average.cpu().numpy()))
        with self.subTest(kind="a total of nothing"):
            total = F.tp_l1_loss(self._device(empty), self._device(empty), 2)
            self.assertEqual(float(total.cpu().numpy()), 0.0)
        with self.subTest(kind="an unreduced plane of nothing"):
            answered = F.tp_l1_loss(self._device(empty), self._device(empty), 0)
            self.assertEqual(tuple(answered.shape), (0, 4))

    def test_a_lane_that_is_not_a_number_carries_through(self):
        # A difference asks nothing of the lane it reads, so a missing
        # measurement travels to the answer rather than stopping the walk.
        import tensorplay.functional as F
        payload = np.array([0.5, np.nan, -1.0], np.float32)
        labels = np.array([0.0, 1.0, 2.0], np.float32)
        with self.subTest(kind="an unreduced deviation"):
            got = F.tp_l1_loss(self._device(payload), self._device(labels), 0)
            np.testing.assert_allclose(
                got.cpu().numpy(), np.array([0.5, np.nan, 3.0]),
                rtol=2e-3, atol=2e-3, equal_nan=True)
        with self.subTest(kind="an average that carries it"):
            average = F.tp_l1_loss(self._device(payload), self._device(labels), 1)
            self.assertTrue(np.isnan(float(average.cpu().numpy())))
        with self.subTest(kind="a gradient on the missing lane"):
            # The fold a gradient is taken from answers zero for a lane that is
            # not a number, so the missing lane hands back no slope rather than
            # carrying the gap into the gradient.
            gradient = F.tp_l1_loss_backward(
                self._device(np.ones(3, np.float32)), self._device(payload),
                self._device(labels), 0).cpu().numpy()
            np.testing.assert_allclose(gradient, np.array([1.0, 0.0, -1.0]))

    def test_a_walk_out_of_kind_or_mode_refuses(self):
        import tensorplay.functional as F
        payload = self._device(np.array([0.2, 0.5], np.float32))
        labels = self._device(np.array([0.0, 1.0], np.float32))
        with self.subTest(kind="an unknown reduction"):
            with self.assertRaisesRegex(RuntimeError, "invalid reduction mode"):
                F.tp_l1_loss(payload, labels, 5)
        with self.subTest(kind="a gradient with an unknown reduction"):
            with self.assertRaisesRegex(RuntimeError, "invalid reduction mode"):
                F.tp_l1_loss_backward(
                    self._device(np.ones(2, np.float32)), payload, labels, 3)
        with self.subTest(kind="a count"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                F.tp_l1_loss(
                    self._device(np.arange(4, dtype=np.int64)),
                    self._device(np.zeros(4, dtype=np.int64)), 1)
        with self.subTest(kind="a count being graduated"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                F.tp_l1_loss_backward(
                    self._device(np.ones(4, np.float32)),
                    self._device(np.arange(4, dtype=np.int32)),
                    self._device(np.zeros(4, dtype=np.int32)), 1)


class TestVulkanSwitchedQuadraticLoss(unittest.TestCase):
    # Two losses read the same deviation twice: a magnitude inside a width
    # answers half a square, a magnitude outside it answers the deviation with
    # the area of that square taken off.  These tests keep both branches, the
    # lane that sits exactly on the width, the slope each side hands back, and
    # the width that switches the whole loss over to a plain deviation.

    SHAPES = ((4,), (2, 5), (3, 4, 2), (2, 3, 4, 5))

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _plane(rng, shape):
        # The plane straddles the width on both sides, so one walk reads the
        # quadratic branch and the line it switches to.
        return ((rng.random(shape) - 0.5) * 8.0).astype(np.float32)

    @staticmethod
    def _difference(values, labels):
        return (np.asarray(values).astype(np.float64)
                - np.asarray(labels).astype(np.float64))

    @classmethod
    def _smooth(cls, values, labels, width):
        difference = cls._difference(values, labels)
        magnitude = np.abs(difference)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside = 0.5 * difference * difference / width
        return np.where(magnitude < width, inside, magnitude - 0.5 * width)

    @classmethod
    def _smooth_slope(cls, values, labels, width):
        # Three tests on the signed deviation, read outside-in: a lane at or
        # below the negative width hands back the full slope of that side, one
        # at or above the width the full slope of its own side, and only a lane
        # strictly inside the width is scaled by it.
        difference = cls._difference(values, labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            rising = difference / width
        return np.where(difference <= -width, -1.0,
                        np.where(difference >= width, 1.0, rising))

    @classmethod
    def _huber(cls, values, labels, width):
        difference = cls._difference(values, labels)
        magnitude = np.abs(difference)
        return np.where(magnitude <= width,
                        0.5 * difference * difference,
                        width * (magnitude - 0.5 * width))

    @classmethod
    def _huber_slope(cls, values, labels, width):
        difference = cls._difference(values, labels)
        return np.where(np.abs(difference) <= width,
                        difference,
                        width * np.sign(difference))

    @staticmethod
    def _reduced(loss, reduction):
        if reduction == 0:
            return loss
        return loss.mean() if reduction == 1 else loss.sum()

    @staticmethod
    def _carried(slope, incoming, reduction):
        gradient = slope * np.broadcast_to(
            np.asarray(incoming).astype(np.float64), np.shape(slope))
        lanes = np.size(slope)
        return gradient / lanes if reduction == 1 and lanes else gradient

    def test_a_deviation_inside_the_width_answers_half_its_square(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(401)
        for shape in self.SHAPES:
            payload = self._plane(rng, shape)
            labels = self._plane(rng, shape)
            for width in (1.0, 0.4):
                loss = self._smooth(payload, labels, width)
                for reduction in (0, 1, 2):
                    with self.subTest(shape=shape, width=width,
                                      reduction=reduction):
                        got = F.smooth_l1_loss(
                            self._device(payload), self._device(labels),
                            reduction, width).cpu().numpy()
                        np.testing.assert_allclose(
                            got, self._reduced(loss, reduction),
                            rtol=2e-2, atol=2e-2)

    def test_a_lane_on_the_width_agrees_between_branches(self):
        import tensorplay.functional as F
        payload = self._device(np.array([1.0, -1.0], np.float32))
        labels = self._device(np.array([0.0, 0.0], np.float32))
        with self.subTest(kind="the smooth width"):
            np.testing.assert_allclose(
                F.smooth_l1_loss(payload, labels, 0, 1.0).cpu().numpy(),
                np.array([0.5, 0.5]))
        with self.subTest(kind="the scaled width"):
            np.testing.assert_allclose(
                F.huber_loss(payload, labels, 0, 1.0).cpu().numpy(),
                np.array([0.5, 0.5]))

    def test_a_gradient_hands_back_the_slope_its_side_of_the_width_is_on(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(409)
        for shape in ((4,), (2, 5), (3, 4, 2)):
            payload = self._plane(rng, shape)
            labels = self._plane(rng, shape)
            incoming = (rng.random(shape) - 0.5).astype(np.float32)
            for width in (1.0, 0.4):
                slope = self._smooth_slope(payload, labels, width)
                for reduction in (0, 1, 2):
                    with self.subTest(shape=shape, width=width,
                                      reduction=reduction):
                        got = F.smooth_l1_loss_backward(
                            self._device(incoming), self._device(payload),
                            self._device(labels), reduction,
                            width).cpu().numpy()
                        np.testing.assert_allclose(
                            got, self._carried(slope, incoming, reduction),
                            rtol=2e-2, atol=2e-2)

    def test_the_wider_switch_scales_its_line_instead_of_its_square(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(419)
        for shape in ((4,), (2, 5), (3, 4, 2)):
            payload = self._plane(rng, shape)
            labels = self._plane(rng, shape)
            incoming = (rng.random(shape) - 0.5).astype(np.float32)
            for width in (1.0, 0.5, 3.0):
                loss = self._huber(payload, labels, width)
                slope = self._huber_slope(payload, labels, width)
                for reduction in (0, 1, 2):
                    with self.subTest(kind="the loss", shape=shape,
                                      width=width, reduction=reduction):
                        got = F.huber_loss(
                            self._device(payload), self._device(labels),
                            reduction, width).cpu().numpy()
                        np.testing.assert_allclose(
                            got, self._reduced(loss, reduction),
                            rtol=2e-2, atol=2e-2)
                    with self.subTest(kind="the gradient", shape=shape,
                                      width=width, reduction=reduction):
                        got = F.huber_loss_backward(
                            self._device(incoming), self._device(payload),
                            self._device(labels), reduction,
                            width).cpu().numpy()
                        np.testing.assert_allclose(
                            got, self._carried(slope, incoming, reduction),
                            rtol=2e-2, atol=2e-2)

    def test_a_width_of_nothing_answers_the_deviation(self):
        # A width of nothing leaves the quadratic branch no lane to hold, so the
        # loss falls back to the plain deviation rather than a division by the
        # width, and every gradient lane is held by one of the outer tests.
        import tensorplay.functional as F
        payload = np.array([1.5, -2.0, 0.25, 3.0], np.float32)
        labels = np.array([0.5, 0.0, 0.25, 1.0], np.float32)
        deviation = np.abs(payload - labels)
        with self.subTest(kind="the loss"):
            np.testing.assert_allclose(
                F.smooth_l1_loss(self._device(payload), self._device(labels),
                                 1, 0.0).cpu().numpy(), deviation.mean(),
                rtol=2e-3, atol=2e-3)
        with self.subTest(kind="the gradient"):
            # The lane that matches exactly is at the width, so it is held by
            # the negative test and hands back the slope of that side.
            np.testing.assert_allclose(
                F.smooth_l1_loss_backward(
                    self._device(np.ones(4, np.float32)),
                    self._device(payload), self._device(labels), 0,
                    0.0).cpu().numpy(),
                self._smooth_slope(payload, labels, 0.0))

    def test_the_host_switched_loss_answers_the_same(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(421)
        payload = self._plane(rng, (2, 5))
        labels = self._plane(rng, (2, 5))
        incoming = (rng.random((2, 5)) - 0.5).astype(np.float32)
        for name, device_call, host_call in (
                ("smooth loss", F.smooth_l1_loss, F.smooth_l1_loss),
                ("huber loss", F.huber_loss, F.huber_loss),
                ("huber gradient", F.huber_loss_backward, F.huber_loss_backward)):
            for reduction in (0, 1, 2):
                with self.subTest(kind=name, reduction=reduction):
                    if "gradient" in name:
                        answered = device_call(
                            self._device(incoming), self._device(payload),
                            self._device(labels), reduction, 0.8).cpu().numpy()
                        on_host = host_call(
                            self._host(incoming), self._host(payload),
                            self._host(labels), reduction, 0.8).cpu().numpy()
                    else:
                        answered = device_call(
                            self._device(payload), self._device(labels),
                            reduction, 0.8).cpu().numpy()
                        on_host = host_call(
                            self._host(payload), self._host(labels),
                            reduction, 0.8).cpu().numpy()
                    np.testing.assert_allclose(
                        answered, on_host, rtol=2e-2, atol=2e-2)

    def test_a_narrow_plane_graduates_in_its_own_width(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(431)
        payload = self._plane(rng, (2, 5)).astype(np.float16)
        labels = self._plane(rng, (2, 5)).astype(np.float16)
        with self.subTest(kind="an unreduced smooth loss kept narrow"):
            answered = F.smooth_l1_loss(
                self._device(payload), self._device(labels), 0, 0.8)
            self.assertEqual(answered.dtype, tensorplay.float16)
        with self.subTest(kind="an averaged huber loss kept narrow"):
            average = F.huber_loss(
                self._device(payload), self._device(labels), 1, 0.8)
            self.assertEqual(average.dtype, tensorplay.float16)
        with self.subTest(kind="a huber gradient kept narrow"):
            gradient = F.huber_loss_backward(
                self._device(np.ones((2, 5), np.float16)),
                self._device(payload), self._device(labels), 1, 0.8)
            self.assertEqual(gradient.dtype, tensorplay.float16)
        with self.subTest(kind="a narrow answer close to the wide one"):
            narrow = F.huber_loss(
                self._device(payload), self._device(labels), 1,
                0.8).cpu().numpy()
            np.testing.assert_allclose(
                narrow,
                self._huber(payload.astype(np.float32),
                            labels.astype(np.float32), 0.8).mean(),
                rtol=2e-2, atol=2e-2)

    def test_a_plane_without_lanes_answers_the_reduction(self):
        import tensorplay.functional as F
        empty = np.zeros((0, 4), np.float32)
        with self.subTest(kind="a smooth average of nothing"):
            average = F.smooth_l1_loss(self._device(empty), self._device(empty),
                                       1, 1.0)
            self.assertEqual(tuple(average.shape), ())
            self.assertTrue(np.isnan(average.cpu().numpy()))
        with self.subTest(kind="a huber total of nothing"):
            total = F.huber_loss(self._device(empty), self._device(empty), 2, 1.0)
            self.assertEqual(float(total.cpu().numpy()), 0.0)

    def test_a_lane_that_is_not_a_number_carries_through(self):
        import tensorplay.functional as F
        payload = np.array([0.5, np.nan, -1.0], np.float32)
        labels = np.array([0.0, 1.0, 2.0], np.float32)
        with self.subTest(kind="a deviation inside the width"):
            got = F.smooth_l1_loss(self._device(payload), self._device(labels),
                                   0, 1.0).cpu().numpy()
            np.testing.assert_allclose(
                got, np.array([0.125, np.nan, 2.5]), equal_nan=True,
                rtol=2e-3, atol=2e-3)
        with self.subTest(kind="an average that carries it"):
            average = F.huber_loss(self._device(payload), self._device(labels),
                                   1, 1.0)
            self.assertTrue(np.isnan(float(average.cpu().numpy())))

    def test_a_width_out_of_range_or_a_count_refuses(self):
        import tensorplay.functional as F
        payload = self._device(np.array([0.2, 0.5], np.float32))
        labels = self._device(np.array([0.0, 1.0], np.float32))
        with self.subTest(kind="a negative smooth width"):
            with self.assertRaisesRegex(
                    RuntimeError, "does not support negative values for beta"):
                F.smooth_l1_loss(payload, labels, 1, -0.5)
        with self.subTest(kind="a gradient with a negative smooth width"):
            with self.assertRaisesRegex(
                    RuntimeError, "does not support negative values for beta"):
                F.smooth_l1_loss_backward(
                    self._device(np.ones(2, np.float32)), payload, labels, 1,
                    -0.5)
        for name, width in (("a zero scaled width", 0.0),
                            ("a negative scaled width", -1.0)):
            with self.subTest(kind=name):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "does not support non-positive values for delta"):
                    F.huber_loss(payload, labels, 1, width)
        with self.subTest(kind="an unknown reduction"):
            with self.assertRaisesRegex(RuntimeError, "invalid reduction mode"):
                F.smooth_l1_loss(payload, labels, 5, 1.0)
        with self.subTest(kind="a count"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                F.huber_loss(
                    self._device(np.arange(4, dtype=np.int64)),
                    self._device(np.zeros(4, dtype=np.int64)), 1, 1.0)


class TestVulkanDivergenceLoss(unittest.TestCase):
    # The divergence one element answers is the weight its match carries against
    # the log-likelihood read for it.  A lane carrying nothing answers nothing
    # for that product, and a read that reaches the far end of the scale keeps
    # the lane there: nothing times that is still that.  These tests hold both
    # readings of the match, the slope each hands back, and the corners.

    SHAPES = ((4,), (2, 5), (3, 4, 2), (2, 3, 4, 5))

    @staticmethod
    def _device(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values)).to("vulkan")

    @staticmethod
    def _host(values):
        return tensorplay.from_numpy(np.ascontiguousarray(values))

    @staticmethod
    def _read(rng, shape):
        # The walk reads log-probabilities, so every lane sits at or below zero.
        return (rng.random(shape) - 1.0).astype(np.float32)

    @staticmethod
    def _carried(rng, shape):
        weights = rng.random(shape).astype(np.float32)
        # Some lanes carry no weight at all.
        weights.ravel()[::7] = 0.0
        return weights

    @classmethod
    def _divergence(cls, read, weights, log_weights):
        x = np.asarray(read).astype(np.float64)
        t = np.asarray(weights).astype(np.float64)
        if log_weights:
            return np.exp(t) * (t - x)
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted = np.where(t == 0.0, 0.0, t * np.log(t))
        return weighted - t * x

    @classmethod
    def _slope(cls, read, weights, log_weights):
        # Only the read is being matched against, so the slope a lane hands back
        # is the weight it carries taken off its own exponent.
        t = np.asarray(weights).astype(np.float64)
        return -np.exp(t) if log_weights else -t

    @staticmethod
    def _reduced(loss, reduction):
        if reduction == 0:
            return loss
        return loss.mean() if reduction == 1 else loss.sum()

    @staticmethod
    def _carried_gradient(slope, incoming, reduction):
        gradient = slope * np.broadcast_to(
            np.asarray(incoming).astype(np.float64), np.shape(slope))
        lanes = np.size(slope)
        return gradient / lanes if reduction == 1 and lanes else gradient

    def test_a_lane_answers_its_weight_against_the_read(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(451)
        for shape in self.SHAPES:
            read = self._read(rng, shape)
            weights = self._carried(rng, shape)
            for log_weights in (False, True):
                loss = self._divergence(read, weights, log_weights)
                for reduction in (0, 1, 2):
                    with self.subTest(shape=shape, reduction=reduction,
                                      log_weights=log_weights):
                        got = F.tp_kl_div(
                            self._device(read), self._device(weights),
                            reduction, log_weights).cpu().numpy()
                        np.testing.assert_allclose(
                            got, self._reduced(loss, reduction),
                            rtol=2e-3, atol=2e-3)

    def test_a_lane_holding_nothing_answers_the_product_of_nothing(self):
        # A lane with no weight to carry answers nothing, however far the read
        # for it reaches — and a read off the end of the scale keeps it there.
        import tensorplay.functional as F
        read = np.array([0.5, np.inf, -np.inf, 2.0, -1.0, np.nan], np.float32)
        weights = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.5], np.float32)
        with self.subTest(kind="a weight of nothing on a plain read"):
            np.testing.assert_allclose(
                F.tp_kl_div(self._device(read), self._device(weights), 0,
                            False).cpu().numpy(),
                self._divergence(read, weights, False),
                rtol=2e-3, atol=2e-3, equal_nan=True)
        with self.subTest(kind="the same read taken as a log-likelihood"):
            np.testing.assert_allclose(
                F.tp_kl_div(self._device(read), self._device(weights), 0,
                            True).cpu().numpy(),
                self._divergence(read, weights, True),
                rtol=2e-3, atol=2e-3, equal_nan=True)

    def test_the_gradient_hands_back_the_weight_a_lane_carries(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(457)
        for shape in ((4,), (2, 5), (3, 4, 2)):
            read = self._read(rng, shape)
            weights = self._carried(rng, shape)
            incoming = (rng.random(shape) - 0.5).astype(np.float32)
            slope = self._slope(read, weights, False)
            logged = self._slope(read, weights, True)
            for log_weights in (False, True):
                for reduction in (0, 1, 2):
                    with self.subTest(shape=shape, reduction=reduction,
                                      log_weights=log_weights):
                        got = F.tp_kl_div_backward(
                            self._device(incoming), self._device(read),
                            self._device(weights), reduction,
                            log_weights).cpu().numpy()
                        np.testing.assert_allclose(
                            got,
                            self._carried_gradient(
                                logged if log_weights else slope,
                                incoming, reduction),
                            rtol=2e-3, atol=2e-3)

    def test_the_device_walk_answers_the_host_walk(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(461)
        read = self._read(rng, (3, 4, 2))
        weights = self._carried(rng, (3, 4, 2))
        incoming = (rng.random((3, 4, 2)) - 0.5).astype(np.float32)
        for log_weights in (False, True):
            for reduction in (0, 1, 2):
                with self.subTest(kind="the loss", reduction=reduction,
                                  log_weights=log_weights):
                    np.testing.assert_allclose(
                        F.tp_kl_div(self._device(read), self._device(weights),
                                    reduction, log_weights).cpu().numpy(),
                        F.tp_kl_div(self._host(read), self._host(weights),
                                    reduction, log_weights).cpu().numpy(),
                        rtol=2e-3, atol=2e-3)
                with self.subTest(kind="the gradient", reduction=reduction,
                                  log_weights=log_weights):
                    np.testing.assert_allclose(
                        F.tp_kl_div_backward(
                            self._device(incoming), self._device(read),
                            self._device(weights), reduction,
                            log_weights).cpu().numpy(),
                        F.tp_kl_div_backward(
                            self._host(incoming), self._host(read),
                            self._host(weights), reduction,
                            log_weights).cpu().numpy(),
                        rtol=2e-3, atol=2e-3)

    def test_a_narrow_payload_keeps_its_own_width(self):
        import tensorplay.functional as F
        rng = np.random.default_rng(467)
        read = self._read(rng, (2, 5)).astype(np.float16)
        weights = self._carried(rng, (2, 5)).astype(np.float16)
        with self.subTest(kind="an unreduced divergence kept narrow"):
            answered = F.tp_kl_div(self._device(read), self._device(weights), 0,
                                   False)
            self.assertEqual(answered.dtype, tensorplay.float16)
        with self.subTest(kind="an averaged divergence kept narrow"):
            average = F.tp_kl_div(self._device(read), self._device(weights), 1,
                                  True)
            self.assertEqual(average.dtype, tensorplay.float16)
        with self.subTest(kind="a gradient kept narrow"):
            gradient = F.tp_kl_div_backward(
                self._device(np.ones((2, 5), np.float16)), self._device(read),
                self._device(weights), 1, False)
            self.assertEqual(gradient.dtype, tensorplay.float16)
        with self.subTest(kind="a narrow answer close to the wide one"):
            narrow = F.tp_kl_div(
                self._device(read), self._device(weights), 1,
                False).cpu().numpy()
            np.testing.assert_allclose(
                narrow,
                self._reduced(
                    self._divergence(read.astype(np.float32),
                                     weights.astype(np.float32), False), 1),
                rtol=2e-2, atol=2e-2)

    def test_a_plane_without_lanes_answers_the_reduction(self):
        import tensorplay.functional as F
        empty = np.zeros((0, 4), np.float32)
        with self.subTest(kind="an average of nothing"):
            average = F.tp_kl_div(self._device(empty), self._device(empty), 1,
                                  False)
            self.assertEqual(tuple(average.shape), ())
            self.assertTrue(np.isnan(average.cpu().numpy()))
        with self.subTest(kind="a total of nothing"):
            total = F.tp_kl_div(self._device(empty), self._device(empty), 2,
                                False)
            self.assertEqual(float(total.cpu().numpy()), 0.0)
        with self.subTest(kind="an unreduced plane of nothing"):
            answered = F.tp_kl_div(self._device(empty), self._device(empty), 0,
                                   True)
            self.assertEqual(tuple(answered.shape), (0, 4))

    def test_the_frontend_carries_the_average_and_the_batch(self):
        import tensorplay.nn.functional as nnF
        rng = np.random.default_rng(471)
        shape = (3, 4, 2)
        read = self._read(rng, shape)
        weights = self._carried(rng, shape)
        for name, reduction in (("the average", "mean"),
                                ("the batch", "batchmean"),
                                ("the total", "sum")):
            with self.subTest(kind=name):
                loss = self._divergence(read, weights, False)
                want = loss.sum() if reduction == "sum" else (
                    loss.sum() / shape[0] if reduction == "batchmean"
                    else loss.mean())
                got = nnF.kl_div(self._device(read), self._device(weights),
                                 reduction=reduction)
                np.testing.assert_allclose(got.cpu().numpy(), want,
                                           rtol=2e-3, atol=2e-3)

    def test_a_walk_out_of_kind_or_mode_refuses(self):
        import tensorplay.functional as F
        read = self._device(np.array([-0.2, -0.5], np.float32))
        weights = self._device(np.array([0.4, 0.6], np.float32))
        with self.subTest(kind="an unknown reduction"):
            with self.assertRaisesRegex(RuntimeError, "invalid reduction mode"):
                F.tp_kl_div(read, weights, 5, False)
        with self.subTest(kind="a gradient with an unknown reduction"):
            with self.assertRaisesRegex(RuntimeError, "invalid reduction mode"):
                F.tp_kl_div_backward(
                    self._device(np.ones(2, np.float32)), read, weights, 3,
                    False)
        with self.subTest(kind="a count being read"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                F.tp_kl_div(
                    self._device(np.arange(4, dtype=np.int64)),
                    self._device(np.zeros(4, dtype=np.int64)), 1, False)
        with self.subTest(kind="a count carrying the weight"):
            with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
                F.tp_kl_div(
                    self._device(np.zeros(4, np.float32)),
                    self._device(np.arange(4, dtype=np.int32)), 1, True)


if __name__ == "__main__":
    unittest.main()
