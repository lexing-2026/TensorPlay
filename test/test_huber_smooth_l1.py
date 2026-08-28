"""Native smooth_l1_loss / huber_loss (+ backwards) alignment vs torch.

Closes the robust-regression loss pair against ATen (Loss.cpp +
cpu/BinaryOpsKernel.cpp smooth_l1_kernel / huber_kernel,
cpu/PointwiseOpsKernel.cpp smooth_l1_backward_cpu_kernel /
huber_backward_cpu_kernel, cuda/BinaryMiscOpsKernels.cu +
cuda/PointwiseOpsKernel.cu): forwards realigned to the ATen signature
(self, target, reduction, beta/delta), new native smooth_l1_loss_backward /
huber_loss_backward, parameter validation (beta >= 0, delta > 0), autograd
through tensorplay.nn.functional, and nn module smoke tests. Backwards use
explicit grads (no .sum().backward()) so the suite is immune to unrelated
reduction regressions.
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


def _np(t):
    return t.detach().cpu().numpy()


def _assert_close(case, tp_t, torch_t, rtol=1e-5, atol=1e-6, msg=""):
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


def _reduction_enum(reduction):
    return {"none": 0, "mean": 1, "sum": 2}[reduction]


class TestSmoothL1Forward(unittest.TestCase):
    def _run(self, shape, reduction, beta, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        target_t = torch.randn(*shape)
        # sprinkle exact-boundary diffs (|x-t| == beta) to exercise the branch
        input_t.view(-1)[::7] = target_t.view(-1)[::7] + beta
        input_t.view(-1)[::11] = target_t.view(-1)[::11] - beta
        ref = torch_F.smooth_l1_loss(input_t, target_t, reduction=reduction,
                                     beta=beta)
        got = F.smooth_l1_loss(_tp_tensor(input_t, dev),
                               _tp_tensor(target_t, dev),
                               reduction=reduction, beta=beta)
        _assert_close(self, got, ref,
                      msg=f"smooth_l1 shape={shape} red={reduction} beta={beta} ({dev})")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for beta in (0.5, 1.0, 2.0):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, beta, dev, 5)


class TestHuberForward(unittest.TestCase):
    def _run(self, shape, reduction, delta, dev, seed):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        target_t = torch.randn(*shape)
        input_t.view(-1)[::7] = target_t.view(-1)[::7] + delta
        input_t.view(-1)[::11] = target_t.view(-1)[::11] - delta
        ref = torch_F.huber_loss(input_t, target_t, reduction=reduction,
                                 delta=delta)
        got = F.huber_loss(_tp_tensor(input_t, dev), _tp_tensor(target_t, dev),
                           reduction=reduction, delta=delta)
        _assert_close(self, got, ref,
                      msg=f"huber shape={shape} red={reduction} delta={delta} ({dev})")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for delta in (0.5, 1.0, 2.0):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, delta, dev, 7)

    def test_validation(self):
        x = _tp_tensor(torch.randn(4), "cpu")
        t = _tp_tensor(torch.randn(4), "cpu")
        with self.assertRaises((ValueError, RuntimeError)):
            F.huber_loss(x, t, delta=0.0)
        with self.assertRaises((ValueError, RuntimeError)):
            F.smooth_l1_loss(x, t, beta=-1.0)


class TestBackwardNative(unittest.TestCase):
    def _run(self, shape, reduction, dev, seed, aten_fn, tp_fn_name, thresh):
        from tensorplay import _C
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        target_t = torch.randn(*shape)
        input_t.view(-1)[::7] = target_t.view(-1)[::7] + thresh
        input_t.view(-1)[::11] = target_t.view(-1)[::11] - thresh
        grad_t = torch.rand(*shape) if reduction == "none" else torch.rand(1).sum()
        ref = aten_fn(grad_t, input_t, target_t, _reduction_enum(reduction), thresh)
        got = getattr(_C, tp_fn_name)(
            _tp_tensor(grad_t, dev), _tp_tensor(input_t, dev),
            _tp_tensor(target_t, dev), _reduction_enum(reduction), thresh)
        _assert_close(self, got, ref,
                      msg=f"{tp_fn_name} shape={shape} red={reduction} thr={thresh} ({dev})")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for thresh in (0.5, 1.0, 2.0):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, dev, 13,
                                  torch.ops.aten.smooth_l1_loss_backward,
                                  "smooth_l1_loss_backward", thresh)
                        self._run(shape, reduction, dev, 14,
                                  torch.ops.aten.huber_loss_backward,
                                  "huber_loss_backward", thresh)


class TestAutograd(unittest.TestCase):
    def _run(self, shape, reduction, dev, seed, fn_t, fn_tp, thresh, kw):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        target_t = torch.randn(*shape)
        ref_in = input_t.clone().requires_grad_(True)
        ref_out = fn_t(ref_in, target_t, reduction=reduction, **{kw: thresh})
        if reduction == "none":
            g_t = torch.randn_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)
        else:
            g_t = torch.tensor(1.0)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        out = fn_tp(x, _tp_tensor(target_t, dev), reduction=reduction,
                    **{kw: thresh})
        out.backward(_tp_tensor(g_t, dev))

        tag = f"{fn_t.__name__} shape={shape} red={reduction} {kw}={thresh} ({dev})"
        _assert_close(self, out, ref_out, msg=f"fwd {tag}")
        _assert_close(self, x.grad, ref_grad, msg=f"grad {tag}")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for thresh in (0.5, 1.0, 2.0):
                    for shape in ((16,), (4, 5)):
                        self._run(shape, reduction, dev, 21,
                                  torch_F.smooth_l1_loss, F.smooth_l1_loss,
                                  thresh, "beta")
                        self._run(shape, reduction, dev, 22,
                                  torch_F.huber_loss, F.huber_loss,
                                  thresh, "delta")


class TestModules(unittest.TestCase):
    def test_modules(self):
        for dev in _devices():
            torch.manual_seed(31)
            input_t = torch.randn(4, 6)
            target_t = torch.randn(4, 6)
            for mod_t_cls, mod_tp_cls, kw, thr in (
                    (torch.nn.SmoothL1Loss, tp.nn.SmoothL1Loss, "beta", 0.7),
                    (torch.nn.HuberLoss, tp.nn.HuberLoss, "delta", 1.3)):
                for reduction in ("mean", "sum", "none"):
                    ref_in = input_t.clone().requires_grad_(True)
                    ref_mod = mod_t_cls(reduction=reduction, **{kw: thr})
                    ref_out = ref_mod(ref_in, target_t)
                    if reduction == "none":
                        g_t = torch.randn_like(ref_out)
                        (ref_grad,) = torch.autograd.grad(
                            ref_out, ref_in, grad_outputs=g_t)
                    else:
                        g_t = torch.tensor(1.0)
                        (ref_grad,) = torch.autograd.grad(ref_out, ref_in)

                    mod = mod_tp_cls(reduction=reduction, **{kw: thr})
                    x = _tp_tensor(input_t, dev, requires_grad=True)
                    out = mod(x, _tp_tensor(target_t, dev))
                    out.backward(_tp_tensor(g_t, dev))
                    name = mod_t_cls.__name__
                    tag = f"{name} red={reduction} {kw}={thr} ({dev})"
                    _assert_close(self, out, ref_out, msg=f"fwd {tag}")
                    _assert_close(self, x.grad, ref_grad, msg=f"grad {tag}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
