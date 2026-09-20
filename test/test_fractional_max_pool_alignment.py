"""

Covers the fractional pooling family added natively to close the gap against
values + flat in-plane indices, backward through autograd, direct native-op
calls, batched and unbatched inputs. Determinism comes from the
caller-provided _random_samples tensor (no internal RNG).
"""
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as torch_F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
import tensorplay.nn.functional as F

from tensorplay.testing._internal.reference import assert_reference_close, from_reference, reference_devices, to_numpy


class TestFractionalMaxPool2d(unittest.TestCase):
    def _run(self, shape, kernel_size, output_size, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        batched = input_t.dim() == 4
        B = shape[0] if batched else 1
        C = shape[1] if batched else shape[0]
        samples = torch.rand(B, C, 2)

        input_ref = input_t.clone().requires_grad_(True)
        ref, ref_idx = torch_F.fractional_max_pool2d(
            input_ref, kernel_size, output_size=output_size,
            return_indices=True, _random_samples=samples)
        g = torch.randn_like(ref)
        ref.backward(g)

        x = from_reference(input_t, dev, requires_grad=True)
        sm = from_reference(samples, dev)
        out, idx = F.fractional_max_pool2d_with_indices(
            x, kernel_size, output_size=output_size, _random_samples=sm)
        out.backward(from_reference(g, dev))

        tag = f"shape={shape} k={kernel_size} o={output_size} ({dev})"
        assert_reference_close(out, ref, msg=f"fmp2d values {tag}")
        np.testing.assert_array_equal(to_numpy(idx), to_numpy(ref_idx),
                                      err_msg=f"fmp2d indices {tag}")
        assert_reference_close(x.grad, input_ref.grad, msg=f"fmp2d grad {tag}")

    def test_configs(self):
        for dev in reference_devices():
            self._run((2, 3, 16, 16), (2, 2), (8, 8), dev, 0)
            self._run((2, 3, 15, 17), (3, 2), (5, 7), dev, 1)
            self._run((1, 4, 13, 11), (1, 1), (13, 11), dev, 2)
            self._run((3, 9, 9), (2, 3), (4, 3), dev, 3)      # unbatched
            self._run((2, 2, 12, 12), (4, 4), (3, 3), dev, 4)

    def test_output_ratio(self):
        torch.manual_seed(5)
        input_t = torch.randn(2, 2, 16, 16)
        samples = torch.rand(2, 2, 2)
        for dev in reference_devices():
            input_ref = input_t.clone().requires_grad_(True)
            ref = torch_F.fractional_max_pool2d(
                input_ref, (2, 2), output_ratio=(0.5, 0.5),
                _random_samples=samples)
            g = torch.ones_like(ref)
            ref.backward(g)
            x = from_reference(input_t, dev, requires_grad=True)
            out = F.fractional_max_pool2d(
                x, (2, 2), output_ratio=(0.5, 0.5),
                _random_samples=from_reference(samples, dev))
            out.backward(from_reference(g, dev))
            assert_reference_close(out, ref, msg=f"fmp2d ratio fwd ({dev})")
            assert_reference_close(x.grad, input_ref.grad, msg=f"fmp2d ratio bwd ({dev})")

    def test_native_op_direct(self):
        from tensorplay import _C
        torch.manual_seed(6)
        input_t = torch.randn(2, 3, 10, 10)
        samples = torch.rand(2, 3, 2)
        for dev in reference_devices():
            x = from_reference(input_t, dev)
            sm = from_reference(samples, dev)
            out, idx = _C.fractional_max_pool2d(x, [2, 2], [5, 5], sm)
            ref, ref_idx = torch_F.fractional_max_pool2d(
                input_t, (2, 2), output_size=(5, 5), return_indices=True,
                _random_samples=samples)
            assert_reference_close(out, ref, msg=f"native fmp2d fwd ({dev})")
            np.testing.assert_array_equal(to_numpy(idx), to_numpy(ref_idx),
                                          err_msg=f"native fmp2d indices ({dev})")
            g_t = torch.randn_like(ref)
            gi = _C.fractional_max_pool2d_backward(
                from_reference(g_t, dev), x, [2, 2], [5, 5], idx)
            input_ref = input_t.clone().requires_grad_(True)
            torch_F.fractional_max_pool2d(
                input_ref, (2, 2), output_size=(5, 5),
                _random_samples=samples).backward(g_t)
            assert_reference_close(gi, input_ref.grad, msg=f"native fmp2d bwd ({dev})")

    def test_module_smoke(self):
        # The module draws its own random samples, so only shape/grad flow is
        # checked here; value behavior is covered by the fixed-sample tests.
        torch.manual_seed(7)
        input_t = torch.randn(2, 2, 12, 12)
        for dev in reference_devices():
            m = tp.nn.FractionalMaxPool2d((2, 2), output_size=(6, 6))
            x = from_reference(input_t, dev, requires_grad=True)
            out = m(x)
            self.assertEqual(tuple(out.shape), (2, 2, 6, 6))
            out.backward(from_reference(torch.randn(2, 2, 6, 6), dev))
            self.assertIsNotNone(x.grad)
            self.assertEqual(tuple(x.grad.shape), tuple(input_t.shape))


class TestFractionalMaxPool3d(unittest.TestCase):
    def _run(self, shape, kernel_size, output_size, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        batched = input_t.dim() == 5
        B = shape[0] if batched else 1
        C = shape[1] if batched else shape[0]
        samples = torch.rand(B, C, 3)

        input_ref = input_t.clone().requires_grad_(True)
        ref, ref_idx = torch_F.fractional_max_pool3d(
            input_ref, kernel_size, output_size=output_size,
            return_indices=True, _random_samples=samples)
        g = torch.randn_like(ref)
        ref.backward(g)

        x = from_reference(input_t, dev, requires_grad=True)
        sm = from_reference(samples, dev)
        out, idx = F.fractional_max_pool3d_with_indices(
            x, kernel_size, output_size=output_size, _random_samples=sm)
        out.backward(from_reference(g, dev))

        tag = f"shape={shape} k={kernel_size} o={output_size} ({dev})"
        assert_reference_close(out, ref, msg=f"fmp3d values {tag}")
        np.testing.assert_array_equal(to_numpy(idx), to_numpy(ref_idx),
                                      err_msg=f"fmp3d indices {tag}")
        assert_reference_close(x.grad, input_ref.grad, msg=f"fmp3d grad {tag}")

    def test_configs(self):
        for dev in reference_devices():
            self._run((2, 2, 8, 9, 10), (2, 2, 2), (4, 4, 5), dev, 10)
            self._run((1, 3, 7, 9, 11), (2, 3, 1), (3, 3, 8), dev, 11)
            self._run((2, 6, 7, 7), (2, 2, 2), (3, 3, 3), dev, 12)   # unbatched
            self._run((1, 2, 6, 6, 6), (1, 1, 1), (5, 5, 5), dev, 13)

    def test_native_op_direct(self):
        from tensorplay import _C
        torch.manual_seed(14)
        input_t = torch.randn(2, 2, 8, 8, 8)
        samples = torch.rand(2, 2, 3)
        for dev in reference_devices():
            x = from_reference(input_t, dev)
            sm = from_reference(samples, dev)
            out, idx = _C.fractional_max_pool3d(x, [2, 2, 2], [4, 4, 4], sm)
            ref, ref_idx = torch_F.fractional_max_pool3d(
                input_t, (2, 2, 2), output_size=(4, 4, 4), return_indices=True,
                _random_samples=samples)
            assert_reference_close(out, ref, msg=f"native fmp3d fwd ({dev})")
            np.testing.assert_array_equal(to_numpy(idx), to_numpy(ref_idx),
                                          err_msg=f"native fmp3d indices ({dev})")
            g_t = torch.randn_like(ref)
            gi = _C.fractional_max_pool3d_backward(
                from_reference(g_t, dev), x, [2, 2, 2], [4, 4, 4], idx)
            input_ref = input_t.clone().requires_grad_(True)
            torch_F.fractional_max_pool3d(
                input_ref, (2, 2, 2), output_size=(4, 4, 4),
                _random_samples=samples).backward(g_t)
            assert_reference_close(gi, input_ref.grad, msg=f"native fmp3d bwd ({dev})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
