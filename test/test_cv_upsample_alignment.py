"""
Forward value and gradient checks for the CV 2-D upsampling paths that used
to silently alias nearest (mode='nearest-exact'), or to have no kernel at all
(antialiasing on bilinear / bicubic).

The forward path is dispatched to the native CPU kernels registered under the
``_upsample_*`` ops in ``p10/src/backend/cpu/UpsampleKernels.cpp``; gradients
are attached by the autograd nodes generated from ``config/derivatives.yaml``
(no Python autograd.Function shim).
"""
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as TF

import tensorplay as tp
import tensorplay.nn.functional as F

TOL_F32 = 5e-5
TOL_F64 = 1e-12


def _make(shape, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


def _to_tp(t):
    return tp.tensor(t.detach().numpy())


def _assert_close(a_tp, a_torch, msg, tol):
    a = a_tp.detach().numpy() if hasattr(a_tp, 'numpy') else np.asarray(a_tp)
    b = a_torch.detach().numpy()
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol, err_msg=msg)


def _grads_tp(fn, *inputs, tangent):
    t_inputs = [x.clone().requires_grad_(True) for x in inputs]
    out = fn(*t_inputs)
    grad = tp.tensor(tangent.detach().numpy())
    out.backward(grad)
    return [x.grad for x in t_inputs], out.detach()


def _grads_torch(fn, *inputs):
    t_inputs = [x.clone().requires_grad_(True) for x in inputs]
    out = fn(*t_inputs)
    grad = torch.randn_like(out)
    out.backward(grad)
    return [x.grad for x in t_inputs], out.detach(), grad


# ---------------------------------------------------------------------------
# nearest-exact: forward + gradient must match reference exactly (Pillow
# convention), and the output must NOT be identical to plain nearest when
# scale != 1.
# ---------------------------------------------------------------------------

class TestNearestExact2D(unittest.TestCase):
    def test_forward_distinct_from_nearest(self):
        x = _make((1, 1, 3, 3))
        g_nn = TF.interpolate(x, size=(5, 5), mode='nearest')
        g_ne = TF.interpolate(x, size=(5, 5), mode='nearest-exact')
        self.assertFalse(torch.equal(g_nn, g_ne),
                         msg='nearest-exact must not silently alias nearest')

    def test_forward_value(self):
        for insh, outsh in [((3, 3), (5, 5)), ((8, 8), (3, 3)),
                             ((64, 64), (22, 22)), ((1, 1), (16, 16))]:
            x = _make((2, 3, *insh))
            ref = TF.interpolate(x, size=outsh, mode='nearest-exact')
            got = F.interpolate(_to_tp(x), size=list(outsh), mode='nearest-exact')
            _assert_close(got, ref, f"nearest-exact {insh}->{outsh}", TOL_F32)

    def test_forward_value_f64(self):
        x = _make((1, 2, 7, 5), dtype=torch.float64)
        ref = TF.interpolate(x, size=(13, 9), mode='nearest-exact')
        got = F.interpolate(_to_tp(x), size=[13, 9], mode='nearest-exact')
        _assert_close(got, ref, 'nearest-exact f64', TOL_F64)

    def test_grad(self):
        x = _make((1, 2, 5, 7))
        ref = lambda z: TF.interpolate(z, size=(9, 11), mode='nearest-exact')
        got = lambda z: F.interpolate(z, size=[9, 11], mode='nearest-exact')
        g_t, _, g_seed = _grads_torch(ref, x)
        g_p, _ = _grads_tp(got, _to_tp(x), tangent=g_seed)
        _assert_close(g_p[0], g_t[0], 'nearest-exact grad', TOL_F32)


class TestNearestExact3D(unittest.TestCase):
    def test_forward_value(self):
        x = _make((1, 1, 3, 4, 5))
        ref = TF.interpolate(x, size=(5, 7, 9), mode='nearest-exact')
        got = F.interpolate(_to_tp(x), size=[5, 7, 9], mode='nearest-exact')
        _assert_close(got, ref, 'nearest-exact 3d', TOL_F32)


# ---------------------------------------------------------------------------
# antialiasing on bilinear / bicubic: forward is a separable
# windowed-filter sum (not the same as plain bilinear); gradient is the
# exact adjoint scatter with the same per-axis weights.
# ---------------------------------------------------------------------------

class TestAntialiasBilinear2D(unittest.TestCase):
    def test_forward_value(self):
        for insh, outsh in [((8, 8), (5, 5)), ((64, 64), (22, 22)),
                            ((8, 8), (8, 8)), ((1, 1), (16, 16))]:
            x = _make((2, 3, *insh))
            ref = TF.interpolate(x, size=outsh, mode='bilinear',
                                 align_corners=False, antialias=True)
            got = F.interpolate(_to_tp(x), size=list(outsh), mode='bilinear',
                                align_corners=False, antialias=True)
            _assert_close(got, ref, f"aa-bilinear {insh}->{outsh}", TOL_F32)

    def test_grad(self):
        x = _make((1, 2, 32, 32))
        ref = lambda z: TF.interpolate(z, size=(12, 12), mode='bilinear',
                                       align_corners=False, antialias=True)
        got = lambda z: F.interpolate(z, size=[12, 12], mode='bilinear',
                                      align_corners=False, antialias=True)
        g_t, _, g_seed = _grads_torch(ref, x)
        g_p, _ = _grads_tp(got, _to_tp(x), tangent=g_seed)
        _assert_close(g_p[0], g_t[0], 'aa-bilinear grad', TOL_F32)


class TestAntialiasBicubic2D(unittest.TestCase):
    def test_forward_value(self):
        for insh, outsh in [((8, 8), (16, 16)), ((64, 64), (22, 22)),
                            ((8, 8), (8, 8)), ((1, 1), (16, 16))]:
            x = _make((2, 3, *insh))
            ref = TF.interpolate(x, size=outsh, mode='bicubic',
                                 align_corners=False, antialias=True)
            got = F.interpolate(_to_tp(x), size=list(outsh), mode='bicubic',
                                align_corners=False, antialias=True)
            _assert_close(got, ref, f"aa-bicubic {insh}->{outsh}", TOL_F32)

    def test_grad(self):
        x = _make((1, 2, 32, 32))
        ref = lambda z: TF.interpolate(z, size=(12, 12), mode='bicubic',
                                       align_corners=False, antialias=True)
        got = lambda z: F.interpolate(z, size=[12, 12], mode='bicubic',
                                      align_corners=False, antialias=True)
        g_t, _, g_seed = _grads_torch(ref, x)
        g_p, _ = _grads_tp(got, _to_tp(x), tangent=g_seed)
        _assert_close(g_p[0], g_t[0], 'aa-bicubic grad', TOL_F32)


if __name__ == '__main__':
    unittest.main()
