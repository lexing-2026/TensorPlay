"""MPS backend smoke tests: device availability, tensor factories, host/device
copy and a minimal elementwise op.

Every test is skipped when the runner has no Metal device (Linux CI, Intel
macs without a supported GPU), so the file is safe to include in the macOS
CPU wheel test matrix.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp


def mps_available():
    return tp.mps.is_available()


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSDevice(unittest.TestCase):
    def test_is_available(self):
        self.assertTrue(tp.mps.is_available())
        self.assertEqual(tp.mps.device_count(), 1)
        self.assertEqual(tp.mps.current_device(), 0)

    def test_tensor_factories(self):
        x = tp.zeros((2, 3), device="mps")
        self.assertEqual(x.device.type, "mps")
        self.assertEqual(x.dtype, tp.float32)
        self.assertTrue(tp.all(x.cpu() == 0).item())

        y = tp.ones((2, 3), device="mps")
        self.assertTrue(tp.all(y.cpu() == 1).item())

        z = tp.full((2, 3), 7, device="mps")
        self.assertTrue(tp.all(z.cpu() == 7).item())

        e = tp.empty((2, 3), device="mps")
        self.assertEqual(e.shape, (2, 3))

    def test_like_factories(self):
        x = tp.zeros((2, 3), device="mps")
        self.assertEqual(tp.zeros_like(x).device.type, "mps")
        self.assertEqual(tp.ones_like(x).device.type, "mps")
        self.assertEqual(tp.full_like(x, 3).device.type, "mps")
        self.assertTrue(tp.all(tp.full_like(x, 3).cpu() == 3).item())

    def test_move_between_devices(self):
        cpu = tp.arange(6, dtype=tp.float32).reshape(2, 3)
        mps_t = cpu.mps()
        self.assertEqual(mps_t.device.type, "mps")
        self.assertTrue(tp.all(mps_t.cpu() == cpu).item())

        mps_t2 = cpu.to("mps")
        self.assertEqual(mps_t2.device.type, "mps")
        self.assertTrue(tp.all(mps_t2.cpu() == cpu).item())

        back = mps_t.cpu()
        self.assertEqual(back.device.type, "cpu")
        self.assertTrue(tp.all(back == cpu).item())

    def test_add(self):
        a = tp.full((4, 4), 2, device="mps", dtype=tp.float32)
        b = tp.full((4, 4), 3, device="mps", dtype=tp.float32)
        c = a + b
        self.assertEqual(c.device.type, "mps")
        self.assertTrue(tp.all(c.cpu() == 5).item())

    def test_add_scalar_and_inplace(self):
        a = tp.ones(4, device="mps")
        c = a + 2
        self.assertTrue(tp.all(c.cpu() == 3).item())
        a.add_(2)
        self.assertTrue(tp.all(a.cpu() == 3).item())
        b = tp.ones(4, device="mps")
        b.add_(a)
        self.assertTrue(tp.all(b.cpu() == 4).item())

    def test_synchronize(self):
        a = tp.ones(4, 4, device="mps")
        b = a + a
        tp.mps.synchronize()
        self.assertEqual(b.shape, (4, 4))
        self.assertTrue(tp.all(b.cpu() == 2).item())

    def test_memory_accounting(self):
        tp.mps.empty_cache()
        a = tp.ones(1024, dtype=tp.float32, device="mps")
        self.assertGreaterEqual(tp.mps.memory_allocated(), 1024 * 4)
        self.assertGreaterEqual(tp.mps.memory_reserved(), 1024 * 4)
        del a
        tp.mps.empty_cache()


if __name__ == "__main__":
    unittest.main()