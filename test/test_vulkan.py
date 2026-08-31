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
        a = tensorplay.zeros((2, 3), device="vulkan")
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


if __name__ == "__main__":
    unittest.main()
