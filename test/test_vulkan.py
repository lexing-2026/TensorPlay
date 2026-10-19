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
        # Two narrow types share no kernel: a conversion between them has to be
        # refused instead of reading a byte lane as something else.
        device = tensorplay.from_numpy(
            np.arange(64, dtype=np.int8)).to("vulkan")
        with self.assertRaisesRegex(
                RuntimeError, "Unsupported Vulkan dtype conversion"):
            device.to(tp.float16)


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

    def test_half_gather_rejects_integer_table(self):
        import tensorplay.functional as F
        device = tensorplay.from_numpy(
            np.arange(200003, dtype=np.int32)).to("vulkan")
        sel = tensorplay.from_numpy(np.arange(8, dtype=np.int32)).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            F.index_select(device, 0, sel)


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
        # The scan runs in the payload's own type and the result widens after
        # it, so the totals round to half on the way out even when asked for a
        # float result.
        want = np.cumsum(host.astype(np.float32), axis=0).astype(
            np.float16).astype(np.float32)
        np.testing.assert_array_equal(
            F.cumsum(device, 0, dtype=tensorplay.float32).cpu().numpy(), want,
            err_msg="half payload scanned into a float result")

    def test_cumsum_rejects_an_integer_payload(self):
        import tensorplay.functional as F
        device = tensorplay.from_numpy(
            np.arange(24, dtype=np.int32).reshape(4, 6)).to("vulkan")
        with self.assertRaisesRegex(RuntimeError, "cumsum"):
            F.cumsum(device, 0)


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
    # queue sync per call.  Reading such a payload is a transfer rather than a
    # shader, so the ops that walk a linear payload one word per element still
    # refuse it, and a range resolves to the widest code they do read.

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

    def test_wide_payloads_stop_at_the_ops_that_read_one_word_per_element(self):
        values = np.array([0, 5, 2, 7, 1], dtype=np.int64)
        payload = tensorplay.from_numpy(values.copy()).to("vulkan")
        # Slicing and reducing walk a payload one word per element: they have to
        # say so rather than read the low word of every code.
        with self.assertRaisesRegex(RuntimeError, "Vulkan slice supports"):
            tensorplay.narrow(payload, 0, 1, 3)
        with self.assertRaisesRegex(
                RuntimeError, "reductions support Float32 and Float16"):
            payload.sum()
        # A half payload packs two lanes into one word, so a wide code would
        # need three of them, and rounding a float into 64 bits needs arithmetic
        # no 32-bit shader has.
        with self.assertRaisesRegex(
                RuntimeError, "conversion from a wide payload"):
            payload.to(tensorplay.float16)
        with self.assertRaisesRegex(
                RuntimeError, "conversion over linear storage"):
            tensorplay.tensor(
                np.array([1.5, -2.25], dtype=np.float32),
                device="vulkan").to(tensorplay.int64)


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
                RuntimeError, "selects between Float32, Float16 and Int32"):
            F.where(condition, flags, flags)


if __name__ == "__main__":
    unittest.main()
