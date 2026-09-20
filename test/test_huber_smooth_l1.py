"""

cpu/BinaryOpsKernel.cpp smooth_l1_kernel / huber_kernel,
cpu/PointwiseOpsKernel.cpp smooth_l1_backward_cpu_kernel /
huber_backward_cpu_kernel, cuda/BinaryMiscOpsKernels.cu +
(self, target, reduction, beta/delta), new native smooth_l1_loss_backward /
huber_loss_backward, parameter validation (beta >= 0, delta > 0), autograd
through tensorplay.nn.functional, and nn module smoke tests. Backwards use
explicit grads (no .sum().backward()) so the suite is immune to unrelated
reduction regressions.
"""
import os
import sys
import unittest

import torch
import torch.nn.functional as torch_F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.testing._internal.reference import assert_reference_close, from_reference, reference_devices


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
        got = F.smooth_l1_loss(from_reference(input_t, dev),
                               from_reference(target_t, dev),
                               reduction=reduction, beta=beta)
        assert_reference_close(got, ref,
                      msg=f"smooth_l1 shape={shape} red={reduction} beta={beta} ({dev})")

    def test_configs(self):
        for dev in reference_devices():
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
        got = F.huber_loss(from_reference(input_t, dev), from_reference(target_t, dev),
                           reduction=reduction, delta=delta)
        assert_reference_close(got, ref,
                      msg=f"huber shape={shape} red={reduction} delta={delta} ({dev})")

    def test_configs(self):
        for dev in reference_devices():
            for reduction in ("none", "mean", "sum"):
                for delta in (0.5, 1.0, 2.0):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, delta, dev, 7)

    def test_validation(self):
        x = from_reference(torch.randn(4), "cpu")
        t = from_reference(torch.randn(4), "cpu")
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
            from_reference(grad_t, dev), from_reference(input_t, dev),
            from_reference(target_t, dev), _reduction_enum(reduction), thresh)
        assert_reference_close(got, ref,
                      msg=f"{tp_fn_name} shape={shape} red={reduction} thr={thresh} ({dev})")

    def test_configs(self):
        for dev in reference_devices():
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

        x = from_reference(input_t, dev, requires_grad=True)
        out = fn_tp(x, from_reference(target_t, dev), reduction=reduction,
                    **{kw: thresh})
        out.backward(from_reference(g_t, dev))

        tag = f"{fn_t.__name__} shape={shape} red={reduction} {kw}={thresh} ({dev})"
        assert_reference_close(out, ref_out, msg=f"fwd {tag}")
        assert_reference_close(x.grad, ref_grad, msg=f"grad {tag}")

    def test_configs(self):
        for dev in reference_devices():
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
        for dev in reference_devices():
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
                    x = from_reference(input_t, dev, requires_grad=True)
                    out = mod(x, from_reference(target_t, dev))
                    out.backward(from_reference(g_t, dev))
                    name = mod_t_cls.__name__
                    tag = f"{name} red={reduction} {kw}={thr} ({dev})"
                    assert_reference_close(out, ref_out, msg=f"fwd {tag}")
                    assert_reference_close(x.grad, ref_grad, msg=f"grad {tag}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
