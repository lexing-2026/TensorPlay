"""

hardshrink/softshrink + the shared shrink_backward_kernel, cuda
ActivationHardshrinkKernel.cu / ActivationSoftshrinkKernel.cu): forward
behavior checks (including the NaN pass-through and inclusive +/-lambd zero band),
the new native hardshrink_backward / softshrink_backward ops, autograd
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


class TestShrinkForward(unittest.TestCase):
    def test_configs(self):
        for dev in _devices():
            for lambd in (0.5, 0.3, 1.0):
                for shape in ((16,), (3, 5), (2, 3, 4)):
                    for fn_t, fn_tp in ((torch_F.hardshrink, F.hardshrink),
                                        (torch_F.softshrink, F.softshrink)):
                        torch.manual_seed(3)
                        input_t = torch.randn(*shape)
                        # sprinkle exact-boundary and zero values to exercise
                        # the inclusive +/-lambd zero band
                        input_t.view(-1)[::7] = lambd
                        input_t.view(-1)[::11] = -lambd
                        input_t.view(-1)[::13] = 0.0
                        ref = fn_t(input_t, lambd)
                        got = fn_tp(_tp_tensor(input_t, dev), lambd)
                        tag = f"{fn_t.__name__} shape={shape} lambd={lambd} ({dev})"
                        _assert_close(self, got, ref, msg=tag)

    def test_nan_inf_passthrough(self):
        # inf passes through as outside the band.
        for dev in _devices():
            vals = torch.tensor([float("nan"), float("inf"), float("-inf"),
                                 0.3, -0.3, 0.5, -0.5])
            for fn_t, fn_tp in ((torch_F.hardshrink, F.hardshrink),
                                (torch_F.softshrink, F.softshrink)):
                ref = fn_t(vals, 0.5)
                got = fn_tp(_tp_tensor(vals, dev), 0.5)
                tag = f"{fn_t.__name__} nan/inf ({dev})"
                np.testing.assert_allclose(
                    _np(got), _np(ref), rtol=1e-5, atol=1e-6, equal_nan=True,
                    err_msg=tag)


class TestShrinkBackwardNative(unittest.TestCase):
    def _run(self, shape, lambd, dev, seed, aten_fn, tp_fn_name):
        from tensorplay import _C
        torch.manual_seed(seed)
        grad_t = torch.randn(*shape)
        input_t = torch.randn(*shape)
        input_t.view(-1)[::7] = lambd
        input_t.view(-1)[::11] = -lambd

        ref = aten_fn(grad_t, input_t, lambd)
        tp_fn = getattr(_C, tp_fn_name)
        got = tp_fn(_tp_tensor(grad_t, dev), _tp_tensor(input_t, dev), lambd)
        tag = f"{tp_fn_name} shape={shape} lambd={lambd} ({dev})"
        _assert_close(self, got, ref, msg=tag)

    def test_configs(self):
        for dev in _devices():
            for lambd in (0.5, 0.3, 1.0):
                for shape in ((16,), (3, 5), (2, 3, 4)):
                    self._run(shape, lambd, dev, 11,
                              torch.ops.aten.hardshrink_backward,
                              "hardshrink_backward")
                    self._run(shape, lambd, dev, 12,
                              torch.ops.aten.softshrink_backward,
                              "softshrink_backward")

    def test_broadcast(self):
        from tensorplay import _C
        for dev in _devices():
            grad_t = torch.randn(4, 1)
            input_t = torch.randn(4, 5)
            for aten_fn, name in ((torch.ops.aten.hardshrink_backward,
                                   "hardshrink_backward"),
                                  (torch.ops.aten.softshrink_backward,
                                   "softshrink_backward")):
                ref = aten_fn(grad_t, input_t, 0.5)
                got = getattr(_C, name)(_tp_tensor(grad_t, dev),
                                        _tp_tensor(input_t, dev), 0.5)
                _assert_close(self, got, ref, msg=f"{name} broadcast ({dev})")


class TestShrinkAutograd(unittest.TestCase):
    def _run(self, shape, lambd, dev, seed, fn_t, fn_tp):
        torch.manual_seed(seed)
        input_t = torch.randn(*shape)
        input_t.view(-1)[::7] = lambd
        input_t.view(-1)[::11] = -lambd

        ref_in = input_t.clone().requires_grad_(True)
        ref_out = fn_t(ref_in, lambd)
        g_t = torch.randn_like(ref_out)
        (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        out = fn_tp(x, lambd)
        out.backward(_tp_tensor(g_t, dev))

        tag = f"{fn_t.__name__} shape={shape} lambd={lambd} ({dev})"
        _assert_close(self, out, ref_out, msg=f"fwd {tag}")
        _assert_close(self, x.grad, ref_grad, msg=f"grad {tag}")

    def test_configs(self):
        for dev in _devices():
            for lambd in (0.5, 0.3, 1.0):
                for shape in ((16,), (3, 5), (2, 3, 4)):
                    self._run(shape, lambd, dev, 21,
                              torch_F.hardshrink, F.hardshrink)
                    self._run(shape, lambd, dev, 22,
                              torch_F.softshrink, F.softshrink)

    def test_negative_grad_sign(self):
        # Regression: the old composite softshrink derivative multiplied by
        # sign(self), flipping the gradient for negative out-of-band inputs.
        for dev in _devices():
            vals = torch.tensor([-2.0, -0.7, -0.3, 0.0, 0.3, 0.7, 2.0])
            ref_in = vals.clone().requires_grad_(True)
            ref_out = torch_F.softshrink(ref_in, lambd=0.5)
            g_t = torch.ones_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                              grad_outputs=g_t)
            x = _tp_tensor(vals, dev, requires_grad=True)
            out = F.softshrink(x, lambd=0.5)
            out.backward(_tp_tensor(g_t, dev))
            _assert_close(self, x.grad, ref_grad,
                          msg=f"softshrink negative-side grad ({dev})")


class TestShrinkModules(unittest.TestCase):
    def test_modules(self):
        for dev in _devices():
            torch.manual_seed(31)
            input_t = torch.randn(4, 6)
            for mod_t_cls, mod_tp_cls, lambd in (
                    (torch.nn.Hardshrink, tp.nn.Hardshrink, 0.5),
                    (torch.nn.Softshrink, tp.nn.Softshrink, 0.7)):
                ref_in = input_t.clone().requires_grad_(True)
                ref_mod = mod_t_cls(lambd=lambd)
                ref_out = ref_mod(ref_in)
                g_t = torch.randn_like(ref_out)
                (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                                  grad_outputs=g_t)

                mod = mod_tp_cls(lambd=lambd)
                x = _tp_tensor(input_t, dev, requires_grad=True)
                out = mod(x)
                out.backward(_tp_tensor(g_t, dev))
                name = mod_t_cls.__name__
                _assert_close(self, out, ref_out, msg=f"{name} fwd ({dev})")
                _assert_close(self, x.grad, ref_grad,
                              msg=f"{name} grad ({dev})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
