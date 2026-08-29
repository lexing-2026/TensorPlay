import unittest

import numpy as np

import tensorplay as tp

if not (hasattr(tp, "cuda") and tp.cuda.is_available()):
    raise unittest.SkipTest("CUDA not available")

import torch


def _cplx(shape, seed, dtype=np.complex64):
    rng = np.random.RandomState(seed)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    re = rng.randn(*shape) + 1
    im = rng.randn(*shape) + 1
    z = (re + 1j * im) * 0.4
    return z.astype(dtype)


def _to_tp(x):
    x = np.ascontiguousarray(x)
    t = tp.tensor(np.stack([x.real.copy(), x.imag.copy()], -1))
    out = t.view_as_complex()
    return out.cuda()


def _to_th(x):
    return torch.from_numpy(np.array(x, copy=True)).cuda()


def _close(tc, got, want, atol=2e-5, rtol=1e-4, msg=None):
    got = np.asarray(got)
    want = np.asarray(want)
    tc.assertEqual(got.shape, want.shape, msg)
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol, err_msg=msg)


class TestCudaArithmetic(unittest.TestCase):
    def setUp(self):
        self.a = _cplx((8,), 0)
        self.b = _cplx((8,), 1)

    def test_binary(self):
        ta, tb = _to_tp(self.a), _to_tp(self.b)
        at, bt = _to_th(self.a), _to_th(self.b)
        for nm, f in [("add", lambda a, b: a + b), ("sub", lambda a, b: a - b),
                      ("mul", lambda a, b: a * b), ("div", lambda a, b: a / b)]:
            _close(self, f(ta, tb).cpu().numpy(), f(at, bt).cpu().numpy())

    def test_scalar_and_python_ops(self):
        ta = _to_tp(self.a)
        at = _to_th(self.a)
        _close(self, (ta * 2j).cpu().numpy(), (at * 2j).cpu().numpy())
        _close(self, (1j + ta).cpu().numpy(), (1j + at).cpu().numpy())
        _close(self, (ta / (1 + 1j)).cpu().numpy(), (at / (1 + 1j)).cpu().numpy())
        t = ta.clone(); t.add_(tb := _to_tp(self.b))
        _close(self, t.cpu().numpy(), self.a + self.b)

    def test_broadcast(self):
        a2 = _cplx((4, 5), 2)
        col = _cplx((4, 1), 3)
        row = _cplx((1, 5), 4)
        got = (_to_tp(a2) * _to_tp(col)).cpu().numpy()
        want = (_to_th(a2) * _to_th(col)).cpu().numpy()
        _close(self, got, want)
        got = (_to_tp(a2) / _to_tp(row)).cpu().numpy()
        want = (_to_th(a2) / _to_th(row)).cpu().numpy()
        _close(self, got, want)


class TestCudaMath(unittest.TestCase):
    FUNCS = ["exp", "log", "log2", "log10", "sqrt", "rsqrt",
             "sin", "cos", "tan", "asin", "acos", "atan",
             "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
             "sigmoid", "expm1", "log1p", "reciprocal"]

    def test_matches_torch_c128(self):
        rng = np.random.RandomState(7)
        x = (rng.uniform(-0.3, 0.3, 64) + 1.05 + 0.15j +
             1j * rng.uniform(-0.3, 0.3, 64)).astype(np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        for name in self.FUNCS:
            got = getattr(tp, name)(t).cpu().numpy()
            want = getattr(torch, name)(tt).cpu().numpy()
            _close(self, got, want, atol=1e-9, rtol=1e-7, msg=name)

    def test_abs_angle_neg_square_pow(self):
        x = _cplx((10,), 5, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        self.assertEqual(tp.abs(t).dtype, tp.float64)
        _close(self, tp.abs(t).cpu().numpy(), np.abs(x))
        _close(self, tp.angle(t).cpu().numpy(), torch.angle(tt).cpu().numpy())
        _close(self, tp.neg(t).cpu().numpy(), torch.neg(tt).cpu().numpy())
        _close(self, tp.square(t).cpu().numpy(), torch.square(tt).cpu().numpy())
        _close(self, tp.pow(t, 3.5).cpu().numpy(),
               torch.pow(tt, 3.5).cpu().numpy(), atol=1e-9)

    def test_rejected_ops(self):
        t = _to_tp(_cplx(4, 6))
        for name in ("erf", "floor", "max"):
            with self.assertRaises(Exception, msg=name):
                getattr(tp, name)(t)


class TestCudaReductions(unittest.TestCase):
    def test_sum_mean_prod_norm_eq(self):
        x = _cplx((5, 6), 8, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        _close(self, tp.sum(t).cpu().numpy(), tt.sum().cpu().numpy(),
               atol=1e-9, rtol=1e-8)
        self.assertEqual(tp.sum(t).dtype, tp.complex128)
        _close(self, tp.mean(t).cpu().numpy(), tt.mean().cpu().numpy(),
               atol=1e-9, rtol=1e-8)
        _close(self, tp.sum(t, [1]).cpu().numpy(), tt.sum(dim=1).cpu().numpy(),
               atol=1e-9, rtol=1e-8)
        small = x[:3]
        _close(self, tp.prod(_to_tp(small)).cpu().numpy(),
               tt[:3].prod().cpu().numpy(), atol=1e-9, rtol=1e-8)
        b = _cplx((6,), 9)
        np.testing.assert_array_equal(
            tp.eq(_to_tp(x[0]), _to_tp(b)).cpu().numpy(),
            x[0] == b)

    def test_matmul(self):
        A, Bm = _cplx((4, 6), 10, np.complex128), _cplx((6, 3), 11, np.complex128)
        got = (_to_tp(A) @ _to_tp(Bm)).cpu().numpy()
        want = (_to_th(A) @ _to_th(Bm)).cpu().numpy()
        _close(self, got, want, atol=1e-9, rtol=1e-8)


class TestCudaFactories(unittest.TestCase):
    def test_randn_component_variance(self):
        t = tp.randn(200_000, dtype=tp.complex64, device="cuda")
        x = t.cpu().numpy()
        self.assertAlmostEqual(x.real.var(), 0.5, delta=0.03)
        self.assertAlmostEqual(x.imag.var(), 0.5, delta=0.03)

    def test_rand_range(self):
        t = tp.rand(20_000, dtype=tp.complex64, device="cuda").cpu().numpy()
        self.assertGreaterEqual(t.real.min(), 0.0)
        self.assertLessEqual(t.real.max(), 1.0)


class TestCudaAutograd(unittest.TestCase):
    def test_exp_mul_chain_matches_torch(self):
        x = _cplx((10,), 12)
        z = _to_tp(x); z.requires_grad_(True)
        zt = _to_th(x); zt.requires_grad_(True)
        (tp.exp(z) * z).sum().backward()
        (torch.exp(zt) * zt).sum().backward(
            torch.tensor(1, dtype=torch.complex64, device="cuda"))
        _close(self, z.grad.cpu().numpy(), zt.grad.cpu().numpy())

    def test_gradcheck_holomorphic(self):
        from tensorplay.autograd import gradcheck
        x = _to_tp(_cplx(6, 13, np.complex128))
        x.requires_grad_(True)
        self.assertTrue(gradcheck(lambda z: (tp.exp(z) * z).sum(), (x,)))


class TestCudaViews(unittest.TestCase):
    def test_adjoint_real_imag(self):
        x = _cplx((3, 5), 14, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        _close(self, t.adjoint().cpu().numpy(), np.conj(x.T))
        _close(self, t.real.cpu().numpy(), x.real)
        _close(self, t.imag.cpu().numpy(), x.imag)


if __name__ == "__main__":
    unittest.main()
