"""

  log_sigmoid, rrelu_with_noise, nll_loss2d, max_pool3d,
  max_pool2d_with_indices, max_pool3d_with_indices, adaptive_max_pool3d
(each together with its backward kernel).
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


class TestLogSigmoid(unittest.TestCase):
    def test_forward_backward(self):
        for dev in reference_devices():
            x_t = torch.randn(4, 16) * 4
            x_t.requires_grad_(True)
            ref = torch_F.logsigmoid(x_t)
            ref.sum().backward()

            x = from_reference(x_t, dev, requires_grad=True)
            out = F.logsigmoid(x)
            out.sum().backward()

            assert_reference_close(out, ref, msg=f"log_sigmoid fwd ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(x.grad, x_t.grad, msg=f"log_sigmoid bwd ({dev})", rtol=1e-4, atol=1e-5)

    def test_extreme_values(self):
        # Numerical stability: large |x| must not overflow (exp branch split).
        for dev in reference_devices():
            vals = torch.tensor([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
            ref = torch_F.logsigmoid(vals)
            x = from_reference(vals, dev)
            assert_reference_close(F.logsigmoid(x), ref, msg=f"log_sigmoid extreme ({dev})", rtol=1e-4, atol=1e-5)

    def test_native_op_direct(self):
        for dev in reference_devices():
            x_t = torch.randn(3, 7, requires_grad=True)
            ref = torch_F.logsigmoid(x_t)
            g = torch.randn_like(ref)
            ref.backward(g)

            x = from_reference(x_t, dev, requires_grad=True)
            out = tp.functional.log_sigmoid(x)
            out.backward(tp.tensor(g.numpy(), device=dev))
            assert_reference_close(out, ref, msg=f"log_sigmoid direct fwd ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(x.grad, x_t.grad, msg=f"log_sigmoid direct bwd ({dev})", rtol=1e-4, atol=1e-5)


class TestRreluWithNoise(unittest.TestCase):
    def test_eval_forward_backward(self):
        for dev in reference_devices():
            x_t = torch.randn(8, 32, requires_grad=True)
            ref = torch_F.rrelu(x_t, lower=0.125, upper=1.0 / 3, training=False)
            ref.sum().backward()

            x = from_reference(x_t, dev, requires_grad=True)
            out = F.rrelu(x, lower=0.125, upper=1.0 / 3, training=False)
            out.sum().backward()
            assert_reference_close(out, ref, msg=f"rrelu eval fwd ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(x.grad, x_t.grad, msg=f"rrelu eval bwd ({dev})", rtol=1e-4, atol=1e-5)

    def test_training_forward_backward(self):
        for dev in reference_devices():
            torch.manual_seed(0)
            x_t = torch.randn(8, 32, requires_grad=True)
            ref = torch_F.rrelu(x_t, lower=0.1, upper=0.4, training=True)
            g = torch.randn_like(ref)
            ref.backward(g)

            x = from_reference(x_t, dev, requires_grad=True)
            out = F.rrelu(x, lower=0.1, upper=0.4, training=True)
            out.backward(tp.tensor(g.numpy(), device=dev))

            # Noise is random, so check structure instead of bit equality.
            out_np, ref_np = to_numpy(out), to_numpy(ref)
            x_np = x_t.detach().numpy()
            pos = x_np > 0
            np.testing.assert_allclose(out_np[pos], x_np[pos], rtol=1e-5, atol=1e-6)
            slopes = np.where(x_np > 0, 1.0, out_np / np.where(x_np == 0, 1.0, x_np))
            neg = x_np < 0
            self.assertTrue(np.all(slopes[neg] >= 0.1 - 1e-5) and
                            np.all(slopes[neg] <= 0.4 + 1e-5),
                            f"rrelu training slopes out of range ({dev})")
            # grad: positive elements pass through, negative scale by slope in
            # [lower, upper].
            grad_np = to_numpy(x.grad)
            g_np = g.numpy()
            np.testing.assert_allclose(grad_np[pos], g_np[pos], rtol=1e-5, atol=1e-6)
            ratios = np.where(g_np == 0, 0.0, grad_np / np.where(g_np == 0, 1.0, g_np))
            self.assertTrue(np.all(ratios[neg] >= 0.1 - 1e-5) and
                            np.all(ratios[neg] <= 0.4 + 1e-5),
                            f"rrelu training grad ratios out of range ({dev})")

    def test_native_op_records_drawn_noise(self):
        for dev in reference_devices():
            torch.manual_seed(5)
            x_t = torch.randn(5, 9)
            x = from_reference(x_t, dev)
            noise = tp.full((5, 9), -7.0, device=dev)
            # Training draws slopes into noise: U(lower, upper) where x <= 0,
            # 1 elsewhere; the output is x * noise.
            out = tp.functional.rrelu_with_noise(x, noise, 0.125, 1.0 / 3, True)
            x_np = x_t.detach().numpy()
            r_np = to_numpy(noise)
            neg = x_np <= 0
            self.assertTrue(np.all(r_np[~neg] == 1.0))
            self.assertTrue(np.all((r_np[neg] >= 0.125) & (r_np[neg] <= 1.0 / 3)))
            np.testing.assert_allclose(to_numpy(out), x_np * r_np, rtol=1e-5, atol=1e-6)


class TestNllLoss2d(unittest.TestCase):
    def _run(self, reduction, weighted, ignore_index, dev):
        torch.manual_seed(3)
        N, C, H, W = 4, 5, 6, 7
        x_t = torch.randn(N, C, H, W, requires_grad=True)
        logp_t = torch_F.log_softmax(x_t, dim=1)
        tgt_t = torch.randint(0, C, (N, H, W))
        if ignore_index is not None:
            tgt_t[0, 0, 0] = ignore_index
        w_t = torch.rand(C) if weighted else None

        ref = torch_F.nll_loss(logp_t, tgt_t, weight=w_t, reduction=reduction,
                               ignore_index=ignore_index if ignore_index is not None else -100)
        g = torch.randn_like(ref)
        ref.backward(g)

        x = from_reference(x_t, dev, requires_grad=True)
        logp = F.log_softmax(x, dim=1)
        tgt = tp.tensor(tgt_t.numpy(), device=dev)
        w = tp.tensor(w_t.numpy(), device=dev) if weighted else None
        out = F.nll_loss(logp, tgt, weight=w, reduction=reduction,
                         ignore_index=ignore_index if ignore_index is not None else -100)
        out.backward(tp.tensor(g.numpy(), device=dev))

        tag = f"nll_loss2d red={reduction} weighted={weighted} ign={ignore_index} ({dev})"
        assert_reference_close(out, ref, msg=tag + " fwd", rtol=1e-4, atol=1e-5)
        assert_reference_close(x.grad, x_t.grad, rtol=2e-4, atol=1e-5, msg=tag + " bwd")

    def test_all(self):
        for dev in reference_devices():
            for reduction in ("none", "mean", "sum"):
                for weighted in (False, True):
                    for ign in (None, 2):
                        self._run(reduction, weighted, ign, dev)


class TestMaxPool3d(unittest.TestCase):
    def _run(self, kernel, stride, padding, dilation, ceil_mode, dev, unbatched=False):
        torch.manual_seed(7)
        shape = (2, 3, 9, 8, 7) if not unbatched else (3, 9, 8, 7)
        x_t = torch.randn(*shape, requires_grad=True)
        ref = torch_F.max_pool3d(x_t, kernel, stride=stride, padding=padding,
                                 dilation=dilation, ceil_mode=ceil_mode)
        g = torch.randn_like(ref)
        ref.backward(g)

        x = from_reference(x_t, dev, requires_grad=True)
        out = F.max_pool3d(x, kernel, stride=stride, padding=padding,
                           dilation=dilation, ceil_mode=ceil_mode)
        out.backward(tp.tensor(g.numpy(), device=dev))
        tag = (f"max_pool3d k={kernel} s={stride} p={padding} d={dilation} "
               f"ceil={ceil_mode} unbatched={unbatched} ({dev})")
        self.assertEqual(tuple(out.shape), tuple(ref.shape), tag + " shape")
        assert_reference_close(out, ref, msg=tag + " fwd", rtol=1e-4, atol=1e-5)
        assert_reference_close(x.grad, x_t.grad, msg=tag + " bwd", rtol=1e-4, atol=1e-5)

    def test_configs(self):
        for dev in reference_devices():
            self._run(2, None, 0, 1, False, dev)
            self._run((2, 3, 2), (1, 2, 1), 1, 1, False, dev)
            self._run(3, 2, 1, 1, True, dev)
            self._run(2, None, 0, 2, False, dev)
            self._run(3, 2, 1, 1, True, dev, unbatched=True)

    def test_with_indices(self):
        for dev in reference_devices():
            torch.manual_seed(11)
            x_t = torch.randn(2, 2, 6, 5, 4, requires_grad=True)
            ref_v, ref_i = torch_F.max_pool3d_with_indices(x_t, 2, stride=2, padding=1)
            g = torch.randn_like(ref_v)
            ref_v.backward(g)

            x = from_reference(x_t, dev, requires_grad=True)
            vals, idx = F.max_pool3d_with_indices(x, 2, stride=2, padding=1)
            vals.backward(tp.tensor(g.numpy(), device=dev))
            assert_reference_close(vals, ref_v, msg=f"max_pool3d_with_indices fwd ({dev})", rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(to_numpy(idx), ref_i.numpy(),
                                          err_msg=f"max_pool3d_with_indices indices ({dev})")
            assert_reference_close(x.grad, x_t.grad, msg=f"max_pool3d_with_indices bwd ({dev})", rtol=1e-4, atol=1e-5)

    def test_module(self):
        for dev in reference_devices():
            torch.manual_seed(13)
            x_t = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
            m_t = torch.nn.MaxPool3d(2, stride=2, return_indices=False)
            ref = m_t(x_t)
            ref.sum().backward()

            m = tp.nn.MaxPool3d(2, stride=2)
            x = from_reference(x_t, dev, requires_grad=True)
            out = m(x)
            out.sum().backward()
            assert_reference_close(out, ref, msg=f"MaxPool3d module fwd ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(x.grad, x_t.grad, msg=f"MaxPool3d module bwd ({dev})", rtol=1e-4, atol=1e-5)


class TestMaxPool2dWithIndices(unittest.TestCase):
    def test_parity(self):
        for dev in reference_devices():
            torch.manual_seed(17)
            x_t = torch.randn(2, 3, 8, 9, requires_grad=True)
            for kernel, stride, padding, dilation, ceil in (
                    (2, None, 0, 1, False),
                    ((3, 2), (2, 1), 1, 1, False),
                    (3, 2, 1, 1, True),
                    (2, 1, 0, 2, False)):
                ref_v, ref_i = torch_F.max_pool2d_with_indices(
                    x_t, kernel, stride=stride, padding=padding,
                    dilation=dilation, ceil_mode=ceil)
                x_t.grad = None
                g = torch.randn_like(ref_v)
                ref_v.backward(g)

                x = from_reference(x_t, dev, requires_grad=True)
                vals, idx = F.max_pool2d_with_indices(
                    x, kernel, stride=stride, padding=padding,
                    dilation=dilation, ceil_mode=ceil)
                vals.backward(tp.tensor(g.numpy(), device=dev))
                tag = f"max_pool2d_with_indices k={kernel} s={stride} p={padding} d={dilation} ceil={ceil} ({dev})"
                assert_reference_close(vals, ref_v, msg=tag + " fwd", rtol=1e-4, atol=1e-5)
                np.testing.assert_array_equal(to_numpy(idx), ref_i.numpy(), err_msg=tag + " indices")
                assert_reference_close(x.grad, x_t.grad, msg=tag + " bwd", rtol=1e-4, atol=1e-5)

    def test_unbatched(self):
        for dev in reference_devices():
            torch.manual_seed(19)
            x_t = torch.randn(2, 6, 6)
            ref_v, ref_i = torch_F.max_pool2d_with_indices(x_t, 2)
            x = from_reference(x_t, dev)
            vals, idx = F.max_pool2d_with_indices(x, 2)
            self.assertEqual(tuple(vals.shape), tuple(ref_v.shape))
            assert_reference_close(vals, ref_v, msg=f"max_pool2d_with_indices unbatched ({dev})", rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(to_numpy(idx), ref_i.numpy())


class TestAdaptiveMaxPool3d(unittest.TestCase):
    def test_forward_backward(self):
        for dev in reference_devices():
            torch.manual_seed(23)
            x_t = torch.randn(2, 3, 7, 6, 5, requires_grad=True)
            for out_size in ((2, 2, 2), (3, 2, 4), (1, 1, 1)):
                ref = torch_F.adaptive_max_pool3d(x_t, out_size)
                x_t.grad = None
                g = torch.randn_like(ref)
                ref.backward(g)

                x = from_reference(x_t, dev, requires_grad=True)
                out = F.adaptive_max_pool3d(x, out_size)
                out.backward(tp.tensor(g.numpy(), device=dev))
                tag = f"adaptive_max_pool3d out={out_size} ({dev})"
                assert_reference_close(out, ref, msg=tag + " fwd", rtol=1e-4, atol=1e-5)
                assert_reference_close(x.grad, x_t.grad, msg=tag + " bwd", rtol=1e-4, atol=1e-5)

    def test_return_indices(self):
        for dev in reference_devices():
            torch.manual_seed(29)
            x_t = torch.randn(1, 2, 5, 5, 5)
            ref_v, ref_i = torch_F.adaptive_max_pool3d(x_t, (2, 2, 2), return_indices=True)
            x = from_reference(x_t, dev)
            vals, idx = F.adaptive_max_pool3d(x, (2, 2, 2), return_indices=True)
            assert_reference_close(vals, ref_v, msg=f"adaptive_max_pool3d indices fwd ({dev})", rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(to_numpy(idx), ref_i.numpy(),
                                          err_msg=f"adaptive_max_pool3d indices ({dev})")

    def test_module(self):
        for dev in reference_devices():
            torch.manual_seed(31)
            x_t = torch.randn(1, 2, 6, 7, 8, requires_grad=True)
            m_t = torch.nn.AdaptiveMaxPool3d((2, 3, 4))
            ref = m_t(x_t)
            ref.sum().backward()

            m = tp.nn.AdaptiveMaxPool3d((2, 3, 4))
            x = from_reference(x_t, dev, requires_grad=True)
            out = m(x)
            out.sum().backward()
            assert_reference_close(out, ref, msg=f"AdaptiveMaxPool3d module fwd ({dev})", rtol=1e-4, atol=1e-5)
            assert_reference_close(x.grad, x_t.grad, msg=f"AdaptiveMaxPool3d module bwd ({dev})", rtol=1e-4, atol=1e-5)


class TestDoubleBackwardDtype(unittest.TestCase):
    """Float64 gradient checks for the new pointwise ops."""

    def test_log_sigmoid_f64(self):
        for dev in reference_devices():
            x_t = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
            ref = torch_F.logsigmoid(x_t)
            ref.sum().backward()
            x = from_reference(x_t, dev, requires_grad=True)
            out = F.logsigmoid(x)
            out.sum().backward()
            assert_reference_close(out, ref, rtol=1e-12, atol=1e-14, msg="log_sigmoid f64 fwd")
            assert_reference_close(x.grad, x_t.grad, rtol=1e-12, atol=1e-14, msg="log_sigmoid f64 bwd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
