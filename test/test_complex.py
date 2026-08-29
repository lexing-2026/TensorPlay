"""

Covers construction, views, arithmetic, transcendental math, reductions,
comparisons and autograd over the complex dtypes, checking values against
"""
import unittest

import numpy as np
import torch

import tensorplay as tp


def _to_tp(x: np.ndarray) -> tp.Tensor:
    """Complex ndarray -> tp tensor via the interleaved view."""
    x = np.ascontiguousarray(x)
    return tp.tensor(np.stack([x.real.copy(), x.imag.copy()], -1)).view_as_complex()


def _to_th(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(x, copy=True))


def _cplx(shape, seed, dtype=np.complex64):
    rng = np.random.RandomState(seed)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    re = rng.randn(*shape).astype(np.float64 if dtype == np.complex128 else np.float32)
    im = rng.randn(*shape).astype(np.float64 if dtype == np.complex128 else np.float32)
    return ((re + 1j * im) * 0.4 + 0.3 + 0.2j).astype(dtype)


def _assert_np_close(testcase, got, want, atol=2e-5, rtol=1e-4, msg=None):
    got = np.asarray(got)
    want = np.asarray(want)
    testcase.assertEqual(got.shape, want.shape)
    if want.size == 0:
        return
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol,
                               err_msg=msg or "")


class TestConstruction(unittest.TestCase):
    def test_complex_from_parts(self):
        r, i = tp.randn(8), tp.randn(8)
        t = tp.complex(r, i)
        self.assertEqual(t.dtype, tp.complex64)
        np.testing.assert_allclose(np.asarray(t.real), r.numpy())
        np.testing.assert_allclose(np.asarray(t.imag), i.numpy())

    def test_complex_promotes_to_double(self):
        t = tp.complex(tp.randn(4, dtype=tp.float64), tp.randn(4, dtype=tp.float64))
        self.assertEqual(t.dtype, tp.complex128)

    def test_polar(self):
        rho = tp.abs(tp.randn(6)) + 0.5
        theta = tp.randn(6)
        t = tp.polar(rho, theta)
        want = torch.polar(_to_th(rho.numpy()), _to_th(theta.numpy())).numpy()
        _assert_np_close(self, t.numpy(), want)

    def test_python_list_construction(self):
        t = tp.tensor([1 + 2j, 3 - 4j])
        self.assertEqual(t.dtype, tp.complex64)
        np.testing.assert_allclose(t.numpy(), np.array([1 + 2j, 3 - 4j], np.complex64))

    def test_factories(self):
        for factory, val in ((tp.zeros, 0), (tp.ones, 1)):
            t = factory(3, dtype=tp.complex64)
            self.assertEqual(t.dtype, tp.complex64)
            np.testing.assert_allclose(t.numpy(), np.full(3, val, np.complex64))

    def test_randn_component_variance(self):
        t = tp.randn(200_000, dtype=tp.complex64)
        x = t.numpy()
        self.assertAlmostEqual(x.real.var(), 0.5, delta=0.02)
        self.assertAlmostEqual(x.imag.var(), 0.5, delta=0.02)

    def test_rand_uniform_components(self):
        t = tp.rand(50_000, dtype=tp.complex128)
        x = t.numpy()
        self.assertGreaterEqual(x.real.min(), 0.0)
        self.assertLessEqual(x.real.max(), 1.0)
        self.assertGreaterEqual(x.imag.min(), 0.0)
        self.assertLessEqual(x.imag.max(), 1.0)

    def test_scalar_wraps_python_complex(self):
        s = tp.Scalar(1 + 2j)
        self.assertTrue(s.is_complex())
        self.assertAlmostEqual(complex(s).real, 1.0)
        self.assertAlmostEqual(complex(s).imag, 2.0)

    def test_item_returns_python_complex(self):
        t = tp.tensor([1 + 2j])
        v = t.item()
        self.assertIsInstance(v, complex)
        self.assertEqual(v, 1 + 2j)


class TestViewsAndParts(unittest.TestCase):
    def setUp(self):
        self.x = _cplx((6, 4), seed=0)
        self.t = _to_tp(self.x)

    def test_real_imag(self):
        np.testing.assert_allclose(self.t.real.numpy(), self.x.real)
        np.testing.assert_allclose(self.t.imag.numpy(), self.x.imag)

    def test_conj(self):
        _assert_np_close(self, self.t.conj().numpy(), np.conj(self.x))

    def test_angle_abs_dtypes_and_values(self):
        th_t = torch.from_numpy(self.x)
        self.assertEqual(self.t.angle().dtype, tp.float32)
        _assert_np_close(self, self.t.angle().numpy(), torch.angle(th_t).numpy())
        self.assertEqual(tp.abs(self.t).dtype, tp.float32)
        _assert_np_close(self, self.t.abs().numpy(), torch.abs(th_t).numpy())

    def test_view_as_real_roundtrip(self):
        vr = self.t.view_as_real()
        self.assertEqual(vr.dtype, tp.float32)
        self.assertEqual(tuple(vr.shape), (6, 4, 2))
        back = vr.view_as_complex()
        _assert_np_close(self, back.numpy(), self.x)

    def test_adjoint(self):
        want = np.conj(np.transpose(self.x))
        _assert_np_close(self, self.t.adjoint().numpy(), want)
        vec = _to_tp(_cplx(5, 3))
        _assert_np_close(self, vec.adjoint().numpy(), np.conj(vec.numpy()))

    def test_is_complex_isreal(self):
        self.assertTrue(self.t.is_complex())
        self.assertFalse(self.t.real.is_complex() or False)  # real output is float
        real_t = tp.randn(4)
        self.assertFalse(real_t.is_complex())
        pure_real = tp.tensor([1 + 0j, 2 + 0j])
        got = tp.isreal(pure_real).numpy()
        self.assertTrue(got.all())


class TestBinaryArithmetic(unittest.TestCase):
    def setUp(self):
        self.a = _cplx((10,), 1)
        self.b = _cplx((10,), 2)
        self.ta, self.tb = _to_tp(self.a), _to_tp(self.b)

    def check(self, fn_tp, fn_th, atol=1e-5):
        got = fn_tp(self.ta, self.tb)
        want = fn_th(_to_th(self.a), _to_th(self.b)).numpy()
        _assert_np_close(self, got.numpy(), want, atol=atol)

    def test_tensor_tensor(self):
        self.check(lambda x, y: x + y, lambda x, y: x + y)
        self.check(lambda x, y: x - y, lambda x, y: x - y)
        self.check(lambda x, y: x * y, lambda x, y: x * y)
        self.check(lambda x, y: x / y, lambda x, y: x / y)

    def test_python_operators_with_complex_scalar(self):
        self.check(lambda x, _: x * 2j, lambda x, _: x * 2j)
        self.check(lambda x, _: 1j + x, lambda x, _: 1j + x)
        self.check(lambda x, _: 2j - x, lambda x, _: 2j - x)
        self.check(lambda x, _: x - 0.5, lambda x, _: x - 0.5)
        self.check(lambda x, _: 1j / (x + 2), lambda x, _: 1j / (x + 2))

    def test_weak_scalar_promotion(self):
        got = self.a.real.astype(np.float32) + 1j
        t = tp.tensor(self.a.real.astype(np.float32)) + 1j
        self.assertEqual(t.dtype, tp.complex64)
        np.testing.assert_allclose(t.numpy(), got)

        f64 = tp.tensor(self.a.real.astype(np.float64)) + 1j
        self.assertEqual(f64.dtype, tp.complex128)

    def test_inplace(self):
        t = _to_tp(self.a)
        t.add_(self.tb)
        _assert_np_close(self, t.numpy(), self.a + self.b)
        t.mul_(0.5)
        _assert_np_close(self, t.numpy(), (self.a + self.b) * 0.5)
        t.div_(1j)
        _assert_np_close(self, t.numpy(), (self.a + self.b) * 0.5 / 1j)
        t.fill_(1j)
        np.testing.assert_allclose(t.numpy(), np.full_like(self.a, 1j))

    def test_neg_reciprocal_square(self):
        self.check(lambda x, _: -x, lambda x, _: -x)
        self.check(lambda x, _: tp.reciprocal(x), lambda x, _: torch.reciprocal(x))
        self.check(lambda x, _: tp.square(x), lambda x, _: torch.square(x))


class TestTranscendental(unittest.TestCase):
    SUPPORTED = [
        "exp", "log", "log2", "log10", "sqrt", "rsqrt",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "sigmoid", "expm1", "log1p", "reciprocal",
    ]

    def _domain_safe(self, n, seed, width):
        rng = np.random.RandomState(seed)
        z = rng.uniform(-0.35, 0.35, n) + 1j * rng.uniform(-0.35, 0.35, n)
        z += 1.05 + 0.15j  # keep acosh/log branches happy
        return z.astype(width)

    def test_matches_torch_cdouble(self):
        x = self._domain_safe(128, 7, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        for name in self.SUPPORTED:
            got = getattr(tp, name)(t).numpy()
            want = getattr(torch, name)(tt).numpy()
            _assert_np_close(self, got, want, atol=1e-9, rtol=1e-7,
                             msg=name)

    def test_matches_torch_cfloat(self):
        x = self._domain_safe(128, 8, np.complex64)
        t, tt = _to_tp(x), _to_th(x)
        for name in self.SUPPORTED:
            got = getattr(tp, name)(t).numpy()
            want = getattr(torch, name)(tt).numpy()
            _assert_np_close(self, got, want, msg=name)

    def assertAlmostEqualNumpy(self, *args, **kwargs):  # noqa: N802 (helper hook)
        pass

    def test_pow(self):
        x = self._domain_safe(64, 9, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        _assert_np_close(self, tp.pow(t, 3.5).numpy(),
                         torch.pow(tt, 3.5).numpy(), atol=1e-9, rtol=1e-7)
        y = self._domain_safe(64, 10, np.complex128)
        _assert_np_close(self, tp.pow(t, _to_tp(y)).numpy(),
                         torch.pow(tt, _to_th(y)).numpy(), atol=1e-9, rtol=1e-7)

    def test_rejected_ops_match_torch(self):
        t = _to_tp(_cplx(4, 11))
        for name in ("erf", "floor", "ceil", "round", "trunc", "frac"):
            with self.assertRaises(Exception, msg=name):
                getattr(tp, name)(t)
        got = tp.sign(_to_tp(np.array([3 + 4j, 0j], np.complex64))).numpy()
        np.testing.assert_allclose(got, np.array([0.6 + 0.8j, 0j], np.complex64),
                                   atol=1e-6)
        with self.assertRaises(Exception):
            tp.max(t)
        with self.assertRaises(Exception):
            tp.sort(t)
        with self.assertRaises(Exception):
            tp.lt(t, t)

    def test_eq_ne(self):
        a = _cplx(20, 12)
        b = a.copy()
        b[3] = b[3] + 1j
        ta, tb = _to_tp(a), _to_tp(b)
        np.testing.assert_array_equal(tp.eq(ta, tb).numpy(), a == b)
        np.testing.assert_array_equal(tp.ne(ta, tb).numpy(), a != b)


class TestReductions(unittest.TestCase):
    def test_sum_mean_prod(self):
        x = _cplx((5, 6), 13, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        _assert_np_close(self, tp.sum(t).numpy(), tt.sum().numpy(),
                         atol=1e-9, rtol=1e-8)
        self.assertEqual(tp.sum(t).dtype, tp.complex128)
        _assert_np_close(self, tp.mean(t).numpy(), tt.mean().numpy(),
                         atol=1e-9, rtol=1e-8)
        small = x[:4]
        _assert_np_close(self, tp.prod(_to_tp(small)).numpy(),
                         _to_th(small).prod().numpy(), atol=1e-9, rtol=1e-8)
        # dim variants keep complex dtype
        sdim = tp.sum(t, [1]).numpy()
        _assert_np_close(self, sdim, tt.sum(dim=1).numpy(), atol=1e-9, rtol=1e-8)
        self.assertEqual(tp.sum(t, [1]).dtype, tp.complex128)
        mdim = tp.mean(t, [1], keepdim=True)
        self.assertEqual(mdim.dtype, tp.complex128)
        _assert_np_close(self, mdim.numpy(), tt.mean(dim=1, keepdim=True).numpy(),
                         atol=1e-9, rtol=1e-8)

    def test_norm(self):
        x = _cplx((30,), 14, np.complex128)
        t, tt = _to_tp(x), _to_th(x)
        self.assertAlmostEqual(float(tp.norm(t).item()),
                               float(torch.linalg.vector_norm(tt).item()),
                               places=8)
        self.assertEqual(tp.norm(t).dtype, tp.float64)


class TestLinearAlgebra(unittest.TestCase):
    def test_matmul_family(self):
        A, B = _cplx((5, 7), 15, np.complex128), _cplx((7, 3), 16, np.complex128)
        got = (_to_tp(A) @ _to_tp(B)).numpy()
        want = (_to_th(A) @ _to_th(B)).numpy()
        _assert_np_close(self, got, want, atol=1e-9, rtol=1e-8)

        Bb = _cplx((2, 4, 4), 17, np.complex128)
        Am = _cplx((2, 4, 4), 18, np.complex128)
        got = tp.bmm(_to_tp(Am), _to_tp(Bb)).numpy()
        want = torch.bmm(_to_th(Am), _to_th(Bb)).numpy()
        _assert_np_close(self, got, want, atol=1e-9, rtol=1e-8)

    def test_fft(self):
        x = _cplx(32, 19, np.complex128)
        got = tp.fft.fft(_to_tp(x)).numpy()
        want = np.fft.fft(x)
        _assert_np_close(self, got, want, atol=1e-9)


class TestAutograd(unittest.TestCase):
    def _compare_grads(self, fn_tp, fn_th, shape=(10,), seed=21, atol=1e-4):
        x = _cplx(shape, seed)
        z = _to_tp(x)
        z.requires_grad_(True)
        zt = _to_th(x)
        zt.requires_grad_(True)
        fn_tp(z).backward()
        fn_th(zt).backward(torch.tensor(1, dtype=zt.dtype))
        _assert_np_close(self, z.grad.numpy(), zt.grad.numpy(), atol=atol)

    def test_elementwise_grads_match_torch(self):
        self._compare_grads(lambda z: (tp.exp(z) * z).sum(),
                            lambda z: (torch.exp(z) * z).sum())
        self._compare_grads(lambda z: tp.sin(z).sum(),
                            lambda z: torch.sin(z).sum())
        self._compare_grads(lambda z: tp.sqrt(z * z + 0.5).sum(),
                            lambda z: torch.sqrt(z * z + 0.5).sum())
        self._compare_grads(lambda z: (z / (2 + 1j)).sum(),
                            lambda z: (z / (2 + 1j)).sum())
        self._compare_grads(lambda z: (z * (2 - 1j)).sum(),
                            lambda z: (z * (2 - 1j)).sum())
        self._compare_grads(lambda z: tp.log(z + (1 + 1j)).sum(),
                            lambda z: torch.log(z + (1 + 1j)).sum())
        self._compare_grads(lambda z: tp.pow(z, 2.5).sum(),
                            lambda z: torch.pow(z, 2.5).sum())

    def test_matmul_chain_grads_match_torch(self):
        A, B = _cplx((3, 4), 22), _cplx((4, 3), 23)
        Bd = _cplx((3, 3), 24)
        a = _to_tp(A)
        a.requires_grad_(True)
        at = _to_th(A)
        at.requires_grad_(True)
        loss_tp = tp.sum(tp.sqrt(a @ _to_tp(B)) / _to_tp(Bd))
        loss_th = torch.sum(torch.sqrt(at @ _to_th(B)) / _to_th(Bd))
        loss_tp.backward()
        loss_th.backward(torch.tensor(1, dtype=torch.complex64))
        _assert_np_close(self, a.grad.numpy(), at.grad.numpy())

    def test_gradcheck_elementwise(self):
        from tensorplay.autograd import gradcheck
        x = _to_tp(_cplx(6, 25, np.complex128))
        x.requires_grad_(True)
        self.assertTrue(gradcheck(lambda z: (tp.exp(z) * z).sum(), (x,)))
        # NOTE: holomorphic functions only -- the checker reconstructs the
        # Jacobian from a single conjugated-VJP pass, which is exact whenever
        # f depends on z alone (exp/log/pow/trig/matmul chains).  Mixed z/zbar
        # resolve_conj fast-mode machinery and are not covered yet.
        self.assertTrue(gradcheck(lambda z: (tp.sin(z) / (z * z + 1)).sum(),
                                  (x,)))

    def test_conj_real_alias_not_same_impl(self):
        # conj over real tensors must be a view (distinct impl, shared data)
        r = tp.randn(4, dtype=tp.float64)
        c = r.conj()
        self.assertFalse(c is r)
        np.testing.assert_allclose(c.numpy(), r.numpy())


class TestDtypePromotions(unittest.TestCase):
    def test_promotion_rules(self):
        c64 = tp.zeros(2, dtype=tp.complex64)
        c128 = tp.zeros(2, dtype=tp.complex128)
        f32 = tp.zeros(2, dtype=tp.float32)
        f64 = tp.zeros(2, dtype=tp.float64)
        i32 = tp.zeros(2, dtype=tp.int32)
        self.assertEqual((c64 + f32).dtype, tp.complex64)
        self.assertEqual((c64 + f64).dtype, tp.complex128)
        self.assertEqual((c64 + c128).dtype, tp.complex128)
        self.assertEqual((f64 + c64).dtype, tp.complex128)
        self.assertEqual((i32 + c64).dtype, tp.complex64)

    def test_cast_semantics(self):
        x = _cplx(4, 26)
        t = _to_tp(x)
        as_f = t.to(tp.float32)
        np.testing.assert_allclose(as_f.numpy(), x.real.astype(np.float32))
        back = as_f.to(tp.complex64)
        np.testing.assert_allclose(back.numpy(), x.real.astype(np.complex64))


if __name__ == "__main__":
    unittest.main()
