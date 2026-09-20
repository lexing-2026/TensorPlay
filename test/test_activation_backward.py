"""

sigmoid_backward_kernel / tanh_backward_kernel, cpu/LogitKernel.cpp
logit_backward_kernel, and the matching CUDA kernels): native op behavior for
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
from tensorplay.testing._internal.reference import assert_reference_close, from_reference, reference_devices, to_numpy


class TestSigmoidTanhBackwardNative(unittest.TestCase):
    def _run(self, shape, dev, seed, aten_fn, tp_fn_name, out_fn):
        from tensorplay import _C
        torch.manual_seed(seed)
        grad_t = torch.randn(*shape)
        output_t = out_fn(torch.randn(*shape))
        ref = aten_fn(grad_t, output_t)
        got = getattr(_C, tp_fn_name)(from_reference(grad_t, dev),
                                      from_reference(output_t, dev))
        assert_reference_close(got, ref,
                      msg=f"{tp_fn_name} shape={shape} ({dev})")

    def test_configs(self):
        for dev in reference_devices():
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
        got = _C.logit_backward(from_reference(grad_t, dev),
                                from_reference(self_t, dev), eps)
        assert_reference_close(got, ref,
                      msg=f"logit_backward shape={shape} eps={eps} ({dev})")

    def test_configs(self):
        for dev in reference_devices():
            for eps in (None, 0.1, 0.3, 0.0):
                for shape in ((16,), (3, 5), (2, 3, 4)):
                    self._run(shape, dev, 13, eps)

    def test_eps_masking(self):
        # With eps>=0 the gradient is zero outside [eps, 1-eps] (the clamped
        # region of the forward) and grad/(x(1-x)) inside. 0.8 sits exactly
        # on the float32 1-eps boundary for eps=0.2 (1.0f - 0.2f == 0.8f),
        # exercising the scalar_t band comparison.
        from tensorplay import _C
        for dev in reference_devices():
            vals = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99])
            grad_t = torch.ones_like(vals)
            for eps in (0.1, 0.2):
                ref = torch.ops.aten.logit_backward(grad_t, vals, eps)
                got = _C.logit_backward(from_reference(grad_t, dev),
                                        from_reference(vals, dev), eps)
                assert_reference_close(got, ref,
                              msg=f"logit_backward eps masking eps={eps} ({dev})")

    def test_out_of_domain(self):
        from tensorplay import _C
        for dev in reference_devices():
            vals = torch.tensor([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
            grad_t = torch.ones_like(vals)
            ref = torch.ops.aten.logit_backward(grad_t, vals)
            got = _C.logit_backward(from_reference(grad_t, dev),
                                    from_reference(vals, dev), None)
            np.testing.assert_allclose(
                to_numpy(got), to_numpy(ref), rtol=1e-5, atol=1e-6, equal_nan=True,
                err_msg=f"logit_backward out-of-domain ({dev})")

    def test_broadcast(self):
        from tensorplay import _C
        for dev in reference_devices():
            grad_t = torch.randn(4, 1)
            self_t = torch.rand(4, 5).clamp(0.05, 0.95)
            ref = torch.ops.aten.logit_backward(grad_t, self_t, 0.1)
            got = _C.logit_backward(from_reference(grad_t, dev),
                                    from_reference(self_t, dev), 0.1)
            assert_reference_close(got, ref, msg=f"logit_backward broadcast ({dev})")


class TestLogitForwardEps(unittest.TestCase):
    def test_configs(self):
        # (including eps=0, which clamps into [0, 1]).
        for dev in reference_devices():
            vals = torch.tensor([-0.5, 0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, 1.5])
            for eps in (None, 0.1, 0.0):
                ref = torch.logit(vals, eps)
                got = tp.logit(from_reference(vals, dev), eps)
                assert_reference_close(got, ref,
                              msg=f"logit fwd eps={eps} ({dev})")


class TestActivationAutograd(unittest.TestCase):
    def _run(self, input_t, dev, seed, fn_t, fn_tp, eps=None):
        torch.manual_seed(seed)
        g_t = torch.randn_like(input_t)
        ref_in = input_t.clone().requires_grad_(True)
        ref_out = fn_t(ref_in) if eps is None else fn_t(ref_in, eps)
        (ref_grad,) = torch.autograd.grad(ref_out, ref_in, grad_outputs=g_t)

        x = from_reference(input_t, dev, requires_grad=True)
        out = fn_tp(x) if eps is None else fn_tp(x, eps)
        out.backward(from_reference(g_t, dev))

        name = fn_t.__name__
        tag = f"{name} shape={tuple(input_t.shape)} eps={eps} ({dev})"
        assert_reference_close(out, ref_out, msg=f"fwd {tag}")
        assert_reference_close(x.grad, ref_grad, msg=f"grad {tag}")

    def test_configs(self):
        for dev in reference_devices():
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
        for dev in reference_devices():
            vals = torch.tensor([0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
            for eps in (0.1, 0.2):
                ref_in = vals.clone().requires_grad_(True)
                ref_out = torch.logit(ref_in, eps)
                g_t = torch.ones_like(ref_out)
                (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                                  grad_outputs=g_t)
                x = from_reference(vals, dev, requires_grad=True)
                out = tp.logit(x, eps)
                out.backward(from_reference(g_t, dev))
                assert_reference_close(x.grad, ref_grad,
                              msg=f"logit eps grad masking eps={eps} ({dev})")

    def test_logit_out_of_domain_grad(self):
        # gradient (logit_backward masks out-of-domain to NaN).
        for dev in reference_devices():
            vals = torch.tensor([-0.5, 0.25, 0.5, 0.75, 1.5])
            ref_in = vals.clone().requires_grad_(True)
            ref_out = torch.logit(ref_in)
            g_t = torch.ones_like(ref_out)
            (ref_grad,) = torch.autograd.grad(ref_out, ref_in,
                                              grad_outputs=g_t)
            x = from_reference(vals, dev, requires_grad=True)
            out = tp.logit(x)
            out.backward(from_reference(g_t, dev))
            np.testing.assert_allclose(
                to_numpy(out), to_numpy(ref_out), rtol=1e-5, atol=1e-6, equal_nan=True,
                err_msg=f"logit fwd out-of-domain ({dev})")
            np.testing.assert_allclose(
                to_numpy(x.grad), to_numpy(ref_grad), rtol=1e-5, atol=1e-6,
                equal_nan=True,
                err_msg=f"logit grad out-of-domain ({dev})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
