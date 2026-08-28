"""Native sigmoid / tanh / logit backward alignment vs torch.

Covers the activation-backward family closed against ATen (cpu/Activation.cpp
sigmoid_backward_kernel / tanh_backward_kernel, cpu/LogitKernel.cpp
logit_backward_kernel, and the matching CUDA kernels): native op parity for
sigmoid_backward / tanh_backward / logit_backward, logit forward eps
semantics (eps=None -> no clamp, eps>=0 -> clamp to [eps, 1-eps]), autograd
through tensorplay's sigmoid / tanh / logit now routing to the native
backwards, and broadcast behavior. Backwards use explicit grads (no
.sum().backward()) so the suite is immune to unrelated reduction regressions.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp


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


class TestSigmoidTanhBackwardNative(unittest.TestCase):
    def _run(self, shape, dev, seed, aten_fn, tp_fn_name, out_fn):
        from tensorplay import _C
        torch.manual_seed(seed)
        grad_t = torch.randn(*shape)
        # ATen sigmoid_backward / tanh_backward take the saved forward OUTPUT.
        output_t = out_fn(torch.randn(*shape))
        ref = aten_fn(grad_t, output_t)
        got = getattr(_C, tp_fn_name)(_tp_tensor(grad_t, dev),
                                      _tp_tensor(output_t, dev))
        _assert_close(self, got, ref,
                      msg=f"{tp_fn_name} shape={shape} ({dev})")

    def test_configs(self):
        for dev in _devices():
            for shape in ((16,), (3, 5), (2, 3, 4)):
                self._run(shape, dev, 11,
                          torch.ops.aten.sigmoid_backward,
                          "sigmoid_backward", torch.sigmoid)
                self._run(shape, dev, 12,
                          torch.ops.aten.tanh_backward,
                          "tanh_backward", torch.tanh)


class TestLogitBackwardNative(unittest.TestCase):
    def _run(self, shape, dev, seed, eps):
        from tensorplay import _C
        torch.manual_seed(seed)
        grad_t = torch.randn(*shape)
        self_t = torch.rand(*shape).clamp(0.01, 0.99)
        if eps is None:
            ref = torch.ops.aten.logit_backward(grad_t, self_t)
        else:
            ref = torch.ops.aten.logit_backward(grad_t, self_t, eps)
        got = _C.logit_backward(_tp_tensor(grad_t, dev),
                                _tp_tensor(self_t, dev), eps)
        _assert_close(self, got, ref,
                      msg=f"logit_backward shape={shape} eps={eps} ({dev})")

    def test_configs(self):
        for dev in _devices():
            for eps in (None, 0.1, 0.3, 0.0):
                for shape in ((16,), (3, 5), (2, 3, 4)):
                    self._run(shape, dev, 13, eps)

    def test_eps_masking(self):
        # With eps>=0 the gradient is zero outside [eps, 1-eps] (the clamped
        # region of the forward) and grad/(x(1-x)) inside. 0.8 sits exactly
        # on the float32 1-eps boundary for eps=0.2 (1.0f - 0.2f == 0.8f),
        # exercising the scalar_t band comparison.
        from tensorplay import _C
        for dev in _devices():
            vals = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99])
            grad_t = torch.ones_like(vals)
            for eps in (0.1, 0.2):
                ref = torch.ops.aten.logit_backward(grad_t, vals, eps)
                got = _C.logit_backward(_tp_tensor(grad_t, dev),
                                        _tp_tensor(vals, dev), eps)
                _assert_close(self, got, ref,
                              msg=f"logit_backward eps masking eps={eps} ({dev})")

    def test_out_of_domain(self):
        # ATen without eps: NaN outside [0, 1], dy*inf at exact 0/1.
        from tensorplay import _C
        for dev in _devices():
            vals = torch.tensor([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
            grad_t = torch.ones_like(vals)
            ref = torch.ops.aten.logit_backward(grad_t, vals)
            got = _C.logit_backward(_tp_tensor(grad_t, dev),
                                    _tp_tensor(vals, dev), None)
            np.testing.assert_allclose(
                _np(got), _np(ref), rtol=1e-5, atol=1e-6, equal_nan=True,
                err_msg=f"logit_backward out-of-domain ({dev})")

    def test_broadcast(self):
        from tensorplay import _C
        for dev in _devices():
            grad_t = torch.randn(4, 1)
            self_t = torch.rand(4, 5).clamp(0.05, 0.95)
            ref = torch.ops.aten.logit_backward(grad_t, self_t, 0.1)
            got = _C.logit_backward(_tp_tensor(grad_t, dev),
                                    _tp_tensor(self_t, dev), 0.1)
            _assert_close(self, got, ref, msg=f"logit_backward broadcast ({dev})")


class TestLogitForwardEps(unittest.TestCase):
    def test_configs(self):
        # ATen: eps=None -> no clamp; eps>=0 -> clamp into [eps, 1-eps]
        # (including eps=0, which clamps into [0, 1]).
        for dev in _devices():
            vals = torch.tensor([-0.5, 0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, 1.5])
            for eps in (None, 0.1, 0.0):
                ref = torch.logit(vals, eps)
                got = tp.logit(_tp_tensor(vals, dev), eps)
                _assert_close(self, got, ref,
                              msg=f"logit fwd eps={eps} ({dev})")


class TestActivationAutograd(unittest.TestCase):
    def _run(self, input_t, dev, seed, fn_t, fn_tp, eps=None):
        torch.manual_seed(seed)
        g_t = torch.randn_like(input_t)
        ref_in = input_t.clone().requires_grad_(True)
        ref_out = fn_t(ref_in) if eps is None else fn_t(ref_in, eps)
        (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)

        x = _tp_tensor(input_t, dev, requires_grad=True)
        out = fn_tp(x) if eps is None else fn_tp(x, eps)
        out.backward(_tp_tensor(g_t, dev))

        name = fn_t.__name__
        tag = f"{name} shape={tuple(input_t.shape)} eps={eps} ({dev})"
        _assert_close(self, out, ref_out, msg=f"fwd {tag}")
        _assert_close(self, x.grad, ref_grad, msg=f"grad {tag}")

    def test_configs(self):
        for dev in _devices():
            for shape in ((16,), (3, 5), (2, 3, 4)):
                torch.manual_seed(7)
                input_t = torch.randn(*shape)
                self._run(input_t, dev, 21, torch.sigmoid, tp.sigmoid)
                self._run(input_t, dev, 22, torch.tanh, tp.tanh)
                # logit is only defined on (0, 1) without eps
                logit_in = torch.rand(*shape).clamp(0.01, 0.99)
                self._run(logit_in, dev, 23, torch.logit, tp.logit)

    def test_logit_eps_grad_masking(self):
        # Regression: the old inline logit derivative was grad/(x(1-x)) and
        # ignored eps, producing nonzero gradients in the clamped region.
        for dev in _devices():
            vals = torch.tensor([0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
            for eps in (0.1, 0.2):
                ref_in = vals.clone().requires_grad_(True)
                ref_out = torch.logit(ref_in, eps)
                g_t = torch.ones_like(ref_out)
                (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                                  grad_outputs=g_t)
                x = _tp_tensor(vals, dev, requires_grad=True)
                out = tp.logit(x, eps)
                out.backward(_tp_tensor(g_t, dev))
                _assert_close(self, x.grad, ref_grad,
                              msg=f"logit eps grad masking eps={eps} ({dev})")

    def test_logit_out_of_domain_grad(self):
        # ATen: without eps, inputs outside [0, 1] give NaN forward and NaN
        # gradient (logit_backward masks out-of-domain to NaN).
        for dev in _devices():
            vals = torch.tensor([-0.5, 0.25, 0.5, 0.75, 1.5])
            ref_in = vals.clone().requires_grad_(True)
            ref_out = torch.logit(ref_in)
            g_t = torch.ones_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                              grad_outputs=g_t)
            x = _tp_tensor(vals, dev, requires_grad=True)
            out = tp.logit(x)
            out.backward(_tp_tensor(g_t, dev))
            np.testing.assert_allclose(
                _np(out), _np(ref_out), rtol=1e-5, atol=1e-6, equal_nan=True,
                err_msg=f"logit fwd out-of-domain ({dev})")
            np.testing.assert_allclose(
                _np(x.grad), _np(ref_grad), rtol=1e-5, atol=1e-6,
                equal_nan=True,
                err_msg=f"logit grad out-of-domain ({dev})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
