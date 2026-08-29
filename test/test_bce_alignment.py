"""

binary_cross_entropy_backward_cpu, cuda/Loss.cu): the forward is realigned to
(input/target must lie in [0, 1], per-element logs clamped at -100 instead of
input clamping), the new native binary_cross_entropy_backward matches
grad * (x - t) / max((1 - x) * x, 1e-12) with weight multiply and
1/numel scaling for mean, autograd flows through tensorplay.nn.functional,
and nn.BCELoss smoke tests. Backwards use explicit grads (no
.sum().backward()) so the suite is immune to unrelated reduction regressions.
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


class TestBCEForward(unittest.TestCase):
    def _run(self, shape, reduction, dev, seed, with_weight):
        torch.manual_seed(seed)
        input_t = torch.rand(*shape).clamp(0.01, 0.99)
        target_t = torch.rand(*shape).round()
        weight_t = torch.rand(*shape) if with_weight else None
        ref = torch_F.binary_cross_entropy(input_t, target_t, weight_t,
                                           reduction=reduction)
        got = F.binary_cross_entropy(
            _tp_tensor(input_t, dev), _tp_tensor(target_t, dev),
            _tp_tensor(weight_t, dev) if with_weight else None,
            reduction=reduction)
        tag = f"bce shape={shape} reduction={reduction} weight={with_weight} ({dev})"
        _assert_close(self, got, ref, msg=tag)

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for with_weight in (False, True):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, dev, 5, with_weight)

    def test_boundary_zero_one(self):
        # input=0/target=1 and input=1/target=0 contribute exactly 100.
        for dev in _devices():
            input_t = torch.tensor([0.0, 1.0, 0.5, 0.25])
            target_t = torch.tensor([1.0, 0.0, 1.0, 0.0])
            for reduction in ("none", "mean", "sum"):
                ref = torch_F.binary_cross_entropy(input_t, target_t,
                                                   reduction=reduction)
                got = F.binary_cross_entropy(_tp_tensor(input_t, dev),
                                             _tp_tensor(target_t, dev),
                                             reduction=reduction)
                _assert_close(self, got, ref,
                              msg=f"bce boundary reduction={reduction} ({dev})")

    def test_input_out_of_range_raises(self):
        for bad_in, bad_tgt in ((torch.tensor([1.5]), torch.tensor([1.0])),
                                (torch.tensor([0.5]), torch.tensor([-0.1]))):
            with self.assertRaises(RuntimeError) as cm:
                F.binary_cross_entropy(_tp_tensor(bad_in, "cpu"),
                                       _tp_tensor(bad_tgt, "cpu"))
            self.assertIn("between 0 and 1", str(cm.exception))


class TestBCEBackwardNative(unittest.TestCase):
    def _run(self, shape, reduction, dev, seed, with_weight):
        from tensorplay import _C
        torch.manual_seed(seed)
        input_t = torch.rand(*shape).clamp(0.01, 0.99)
        target_t = torch.rand(*shape).round()
        weight_t = torch.rand(*shape) if with_weight else None
        grad_t = torch.rand(*shape) if reduction == "none" else torch.rand(1).sum()
        ref = torch.ops.aten.binary_cross_entropy_backward(
            grad_t, input_t, target_t, weight_t, _reduction_enum(reduction))
        got = _C.binary_cross_entropy_backward(
            _tp_tensor(grad_t, dev), _tp_tensor(input_t, dev),
            _tp_tensor(target_t, dev),
            _tp_tensor(weight_t, dev) if with_weight else None,
            _reduction_enum(reduction))
        tag = f"bce_backward shape={shape} reduction={reduction} weight={with_weight} ({dev})"
        _assert_close(self, got, ref, msg=tag)

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for with_weight in (False, True):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, dev, 13, with_weight)


class TestBCEAutograd(unittest.TestCase):
    def _run(self, shape, reduction, dev, seed, with_weight):
        torch.manual_seed(seed)
        input_t = torch.rand(*shape).clamp(0.01, 0.99)
        target_t = torch.rand(*shape).round()
        weight_t = torch.rand(*shape) if with_weight else None

        ref_in = input_t.clone().requires_grad_(True)
        ref_out = torch_F.binary_cross_entropy(ref_in, target_t, weight_t,
                                               reduction=reduction)
        if reduction == "none":
            g_t = torch.randn_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)
        else:
            g_t = torch.tensor(1.0)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        out = F.binary_cross_entropy(
            x, _tp_tensor(target_t, dev),
            _tp_tensor(weight_t, dev) if with_weight else None,
            reduction=reduction)
        out.backward(_tp_tensor(g_t, dev))

        tag = f"bce shape={shape} reduction={reduction} weight={with_weight} ({dev})"
        _assert_close(self, out, ref_out, msg=f"fwd {tag}")
        _assert_close(self, x.grad, ref_grad, msg=f"grad {tag}")

    def test_configs(self):
        for dev in _devices():
            for reduction in ("none", "mean", "sum"):
                for with_weight in (False, True):
                    for shape in ((16,), (4, 5), (2, 3, 4)):
                        self._run(shape, reduction, dev, 21, with_weight)


class TestBCELossModule(unittest.TestCase):
    def test_modules(self):
        for dev in _devices():
            torch.manual_seed(31)
            input_t = torch.rand(4, 6).clamp(0.01, 0.99)
            target_t = torch.rand(4, 6).round()
            for reduction in ("mean", "sum", "none"):
                for weight_t in (None, torch.rand(4, 6)):
                    ref_in = input_t.clone().requires_grad_(True)
                    ref_mod = torch.nn.BCELoss(weight=weight_t,
                                               reduction=reduction)
                    ref_out = ref_mod(ref_in, target_t)
                    if reduction == "none":
                        g_t = torch.randn_like(ref_out)
                        (ref_grad,) = torch.autograd.grad(
                            ref_out, ref_in, grad_outputs=g_t)
                    else:
                        g_t = torch.tensor(1.0)
                        (ref_grad,) = torch.autograd.grad(ref_out, ref_in)

                    mod = tp.nn.BCELoss(
                        weight=_tp_tensor(weight_t, dev) if weight_t is not None else None,
                        reduction=reduction)
                    x = _tp_tensor(input_t, dev, requires_grad=True)
                    out = mod(x, _tp_tensor(target_t, dev))
                    out.backward(_tp_tensor(g_t, dev))
                    tag = f"BCELoss reduction={reduction} weight={weight_t is not None} ({dev})"
                    _assert_close(self, out, ref_out, msg=f"fwd {tag}")
                    _assert_close(self, x.grad, ref_grad, msg=f"grad {tag}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
