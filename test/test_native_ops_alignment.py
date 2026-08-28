"""Native fwd/bwd op alignment vs torch (CPU + CUDA).

Covers the ops added natively to close the gap against ATen:
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
from tensorplay import Tensor


def _np(t):
    if isinstance(t, Tensor):
        return t.detach().cpu().numpy()
    return t.detach().numpy()


def _assert_close(test_case, tp_t, torch_t, rtol=1e-4, atol=1e-5, msg=""):
    np.testing.assert_allclose(_np(tp_t), _np(torch_t), rtol=rtol, atol=atol,
                               err_msg=msg)


def _devices():
    devs = ["cpu"]
    if tp.cuda.is_available():
        devs.append("cuda")
    return devs


def _tp_tensor(torch_t, device, requires_grad=False):
    t = tp.tensor(torch_t.detach().numpy(), device=device)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


class TestLogSigmoid(unittest.TestCase):
    def test_forward_backward(self):
        for dev in _devices():
            x_t = torch.randn(4, 16) * 4
            x_t.requires_grad_(True)
            ref = torch_F.logsigmoid(x_t)
            ref.sum().backward()

            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = F.logsigmoid(x)
            out.sum().backward()

            _assert_close(self, out, ref, msg=f"log_sigmoid fwd ({dev})")
            _assert_close(self, x.grad, x_t.grad, msg=f"log_sigmoid bwd ({dev})")

    def test_extreme_values(self):
        # Numerical stability: large |x| must not overflow (exp branch split).
        for dev in _devices():
            vals = torch.tensor([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
            ref = torch_F.logsigmoid(vals)
            x = _tp_tensor(vals, dev)
            _assert_close(self, F.logsigmoid(x), ref, msg=f"log_sigmoid extreme ({dev})")

    def test_native_op_direct(self):
        for dev in _devices():
            x_t = torch.randn(3, 7, requires_grad=True)
            ref = torch_F.logsigmoid(x_t)
            g = torch.randn_like(ref)
            ref.backward(g)

            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = tp.functional.log_sigmoid(x)
            out.backward(tp.tensor(g.numpy(), device=dev))
            _assert_close(self, out, ref, msg=f"log_sigmoid direct fwd ({dev})")
            _assert_close(self, x.grad, x_t.grad, msg=f"log_sigmoid direct bwd ({dev})")


class TestRreluWithNoise(unittest.TestCase):
    def test_eval_forward_backward(self):
        for dev in _devices():
            x_t = torch.randn(8, 32, requires_grad=True)
            ref = torch_F.rrelu(x_t, lower=0.125, upper=1.0 / 3, training=False)
            ref.sum().backward()

            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = F.rrelu(x, lower=0.125, upper=1.0 / 3, training=False)
            out.sum().backward()
            _assert_close(self, out, ref, msg=f"rrelu eval fwd ({dev})")
            _assert_close(self, x.grad, x_t.grad, msg=f"rrelu eval bwd ({dev})")

    def test_training_forward_backward(self):
        for dev in _devices():
            torch.manual_seed(0)
            x_t = torch.randn(8, 32, requires_grad=True)
            ref = torch_F.rrelu(x_t, lower=0.1, upper=0.4, training=True)
            g = torch.randn_like(ref)
            ref.backward(g)

            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = F.rrelu(x, lower=0.1, upper=0.4, training=True)
            out.backward(tp.tensor(g.numpy(), device=dev))

            # noise is random: check structural parity instead of bit equality.
            out_np, ref_np = _np(out), _np(ref)
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
            grad_np = _np(x.grad)
            g_np = g.numpy()
            np.testing.assert_allclose(grad_np[pos], g_np[pos], rtol=1e-5, atol=1e-6)
            ratios = np.where(g_np == 0, 0.0, grad_np / np.where(g_np == 0, 1.0, g_np))
            self.assertTrue(np.all(ratios[neg] >= 0.1 - 1e-5) and
                            np.all(ratios[neg] <= 0.4 + 1e-5),
                            f"rrelu training grad ratios out of range ({dev})")

    def test_native_op_with_given_noise(self):
        for dev in _devices():
            torch.manual_seed(5)
            x_t = torch.randn(5, 9)
            noise_t = torch.rand(5, 9) * 0.3 + 0.1
            # With a pre-filled noise tensor the math is x <= 0 ? x*r : x
            # (ATen's own kernel instead draws r from a generator).
            x_np = x_t.detach().numpy()
            r_np = noise_t.numpy()
            expected = np.where(x_np <= 0, x_np * r_np, x_np)

            x = _tp_tensor(x_t, dev)
            out = tp.functional.rrelu_with_noise(
                x, tp.tensor(r_np, device=dev), 0.125, 1.0 / 3, True)
            np.testing.assert_allclose(_np(out), expected, rtol=1e-5, atol=1e-6)


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

        x = _tp_tensor(x_t, dev, requires_grad=True)
        logp = F.log_softmax(x, dim=1)
        tgt = tp.tensor(tgt_t.numpy(), device=dev)
        w = tp.tensor(w_t.numpy(), device=dev) if weighted else None
        out = F.nll_loss(logp, tgt, weight=w, reduction=reduction,
                         ignore_index=ignore_index if ignore_index is not None else -100)
        out.backward(tp.tensor(g.numpy(), device=dev))

        tag = f"nll_loss2d red={reduction} weighted={weighted} ign={ignore_index} ({dev})"
        _assert_close(self, out, ref, msg=tag + " fwd")
        _assert_close(self, x.grad, x_t.grad, rtol=2e-4, atol=1e-5, msg=tag + " bwd")

    def test_all(self):
        for dev in _devices():
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

        x = _tp_tensor(x_t, dev, requires_grad=True)
        out = F.max_pool3d(x, kernel, stride=stride, padding=padding,
                           dilation=dilation, ceil_mode=ceil_mode)
        out.backward(tp.tensor(g.numpy(), device=dev))
        tag = (f"max_pool3d k={kernel} s={stride} p={padding} d={dilation} "
               f"ceil={ceil_mode} unbatched={unbatched} ({dev})")
        self.assertEqual(tuple(out.shape), tuple(ref.shape), tag + " shape")
        _assert_close(self, out, ref, msg=tag + " fwd")
        _assert_close(self, x.grad, x_t.grad, msg=tag + " bwd")

    def test_configs(self):
        for dev in _devices():
            self._run(2, None, 0, 1, False, dev)
            self._run((2, 3, 2), (1, 2, 1), 1, 1, False, dev)
            self._run(3, 2, 1, 1, True, dev)
            self._run(2, None, 0, 2, False, dev)
            self._run(3, 2, 1, 1, True, dev, unbatched=True)

    def test_with_indices(self):
        for dev in _devices():
            torch.manual_seed(11)
            x_t = torch.randn(2, 2, 6, 5, 4, requires_grad=True)
            ref_v, ref_i = torch_F.max_pool3d_with_indices(x_t, 2, stride=2, padding=1)
            g = torch.randn_like(ref_v)
            ref_v.backward(g)

            x = _tp_tensor(x_t, dev, requires_grad=True)
            vals, idx = F.max_pool3d_with_indices(x, 2, stride=2, padding=1)
            vals.backward(tp.tensor(g.numpy(), device=dev))
            _assert_close(self, vals, ref_v, msg=f"max_pool3d_with_indices fwd ({dev})")
            np.testing.assert_array_equal(_np(idx), ref_i.numpy(),
                                          err_msg=f"max_pool3d_with_indices indices ({dev})")
            _assert_close(self, x.grad, x_t.grad, msg=f"max_pool3d_with_indices bwd ({dev})")

    def test_module(self):
        for dev in _devices():
            torch.manual_seed(13)
            x_t = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
            m_t = torch.nn.MaxPool3d(2, stride=2, return_indices=False)
            ref = m_t(x_t)
            ref.sum().backward()

            m = tp.nn.MaxPool3d(2, stride=2)
            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = m(x)
            out.sum().backward()
            _assert_close(self, out, ref, msg=f"MaxPool3d module fwd ({dev})")
            _assert_close(self, x.grad, x_t.grad, msg=f"MaxPool3d module bwd ({dev})")


class TestMaxPool2dWithIndices(unittest.TestCase):
    def test_parity(self):
        for dev in _devices():
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

                x = _tp_tensor(x_t, dev, requires_grad=True)
                vals, idx = F.max_pool2d_with_indices(
                    x, kernel, stride=stride, padding=padding,
                    dilation=dilation, ceil_mode=ceil)
                vals.backward(tp.tensor(g.numpy(), device=dev))
                tag = f"max_pool2d_with_indices k={kernel} s={stride} p={padding} d={dilation} ceil={ceil} ({dev})"
                _assert_close(self, vals, ref_v, msg=tag + " fwd")
                np.testing.assert_array_equal(_np(idx), ref_i.numpy(), err_msg=tag + " indices")
                _assert_close(self, x.grad, x_t.grad, msg=tag + " bwd")

    def test_unbatched(self):
        for dev in _devices():
            torch.manual_seed(19)
            x_t = torch.randn(2, 6, 6)
            ref_v, ref_i = torch_F.max_pool2d_with_indices(x_t, 2)
            x = _tp_tensor(x_t, dev)
            vals, idx = F.max_pool2d_with_indices(x, 2)
            self.assertEqual(tuple(vals.shape), tuple(ref_v.shape))
            _assert_close(self, vals, ref_v, msg=f"max_pool2d_with_indices unbatched ({dev})")
            np.testing.assert_array_equal(_np(idx), ref_i.numpy())


class TestAdaptiveMaxPool3d(unittest.TestCase):
    def test_forward_backward(self):
        for dev in _devices():
            torch.manual_seed(23)
            x_t = torch.randn(2, 3, 7, 6, 5, requires_grad=True)
            for out_size in ((2, 2, 2), (3, 2, 4), (1, 1, 1)):
                ref = torch_F.adaptive_max_pool3d(x_t, out_size)
                x_t.grad = None
                g = torch.randn_like(ref)
                ref.backward(g)

                x = _tp_tensor(x_t, dev, requires_grad=True)
                out = F.adaptive_max_pool3d(x, out_size)
                out.backward(tp.tensor(g.numpy(), device=dev))
                tag = f"adaptive_max_pool3d out={out_size} ({dev})"
                _assert_close(self, out, ref, msg=tag + " fwd")
                _assert_close(self, x.grad, x_t.grad, msg=tag + " bwd")

    def test_return_indices(self):
        for dev in _devices():
            torch.manual_seed(29)
            x_t = torch.randn(1, 2, 5, 5, 5)
            ref_v, ref_i = torch_F.adaptive_max_pool3d(x_t, (2, 2, 2), return_indices=True)
            x = _tp_tensor(x_t, dev)
            vals, idx = F.adaptive_max_pool3d(x, (2, 2, 2), return_indices=True)
            _assert_close(self, vals, ref_v, msg=f"adaptive_max_pool3d indices fwd ({dev})")
            np.testing.assert_array_equal(_np(idx), ref_i.numpy(),
                                          err_msg=f"adaptive_max_pool3d indices ({dev})")

    def test_module(self):
        for dev in _devices():
            torch.manual_seed(31)
            x_t = torch.randn(1, 2, 6, 7, 8, requires_grad=True)
            m_t = torch.nn.AdaptiveMaxPool3d((2, 3, 4))
            ref = m_t(x_t)
            ref.sum().backward()

            m = tp.nn.AdaptiveMaxPool3d((2, 3, 4))
            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = m(x)
            out.sum().backward()
            _assert_close(self, out, ref, msg=f"AdaptiveMaxPool3d module fwd ({dev})")
            _assert_close(self, x.grad, x_t.grad, msg=f"AdaptiveMaxPool3d module bwd ({dev})")


class TestDoubleBackwardDtype(unittest.TestCase):
    """float64 parity (gradcheck-style) for the new pointwise ops."""

    def test_log_sigmoid_f64(self):
        for dev in _devices():
            x_t = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
            ref = torch_F.logsigmoid(x_t)
            ref.sum().backward()
            x = _tp_tensor(x_t, dev, requires_grad=True)
            out = F.logsigmoid(x)
            out.sum().backward()
            _assert_close(self, out, ref, rtol=1e-12, atol=1e-14, msg="log_sigmoid f64 fwd")
            _assert_close(self, x.grad, x_t.grad, rtol=1e-12, atol=1e-14, msg="log_sigmoid f64 bwd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
