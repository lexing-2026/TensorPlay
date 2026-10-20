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

    def test_wide_payloads_stop_at_the_ops_that_fold_a_code_into_a_word(self):
        values = np.array([0, 5, 2, 7, 1], dtype=np.int64)
        payload = tensorplay.from_numpy(values.copy()).to("vulkan")
        # A total of wide lanes needs an accumulator wider than one word, and the
        # walk that adds lane by lane reads a single word per element: it has to
        # say so rather than add up the low words of every code.
        with self.assertRaisesRegex(
                RuntimeError, "reductions support Float32 and Float16"):
            payload.sum()
        # A half payload packs two lanes into one word, so a wide code would need
        # three of them and the two storages share no word boundary.
        with self.assertRaisesRegex(
                RuntimeError, "conversion from a wide payload"):
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
                RuntimeError, "Float32, Float16 and Int32"):
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
                    RuntimeError, "has no Int32 kernel"):
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

    def test_a_plane_of_wide_lanes_stays_outside_the_walk(self):
        # A lane that is already the wide word has no walk here that adds it
        # without rounding, so the plane is refused rather than answered with
        # the low half of every number.
        wide = self._device(np.arange(5, dtype=np.int64))
        with self.assertRaisesRegex(RuntimeError, "Float32 and Float16"):
            wide.sum()


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
        # Both storages settle the range in the walk: a code that names no row
        # answers zeros instead of asking the host where the rows stop.
        small = self._device(self._table((5, 4)))
        np.testing.assert_array_equal(
            self._read(small, 0, self._codes([0, 7, -2])).cpu().numpy(),
            np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]], np.float32),
            err_msg="an image payload")
        big = self._device(self._table((20000, 3)))
        np.testing.assert_array_equal(
            self._read(big, 0, self._codes([0, 20000, -1])).cpu().numpy(),
            np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]], np.float32),
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

    def test_a_shape_the_walk_would_have_to_guess_is_answered_off_it(self):
        payload = self._device(self._table((5, 4)))
        # A mask names rows by counting them, which is a compaction the walk
        # does not do; neither does a tuple of lists that has to be broadcast
        # against itself.
        with self.assertRaisesRegex(RuntimeError, "list of row codes"):
            tensorplay.index(
                payload, [self._device(np.array([True, False, True, False, False]))])
        with self.assertRaisesRegex(RuntimeError, "list of row codes"):
            tensorplay.index(payload, [self._device(np.ones(2, np.float32))])
        with self.assertRaisesRegex(RuntimeError, "several lists"):
            tensorplay.index(payload, [self._codes([0, 1]), self._codes([0, 1])])


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


if __name__ == "__main__":
    unittest.main()
