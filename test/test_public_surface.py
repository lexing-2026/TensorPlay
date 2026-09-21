"""

Surface-closure checks for the namespaces, legacy faces and Tensor dunder
protocol. Numeric checks run against the local reference runtime when it is
installed; the rest is pure-behavior and runs everywhere.
"""

import copy
import unittest

import numpy as np

import tensorplay as tp
import tensorplay.nn as nn

try:
    import torch
except ImportError:
    torch = None


def _same(a, b, tol=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.allclose(a, b, atol=tol, rtol=tol)


class TestNamespaceWiring(unittest.TestCase):
    def test_orphan_modules_reachable(self):
        for name in (
            "signal", "distributions", "masked", "package", "testing",
            "return_types", "cpu",
        ):
            self.assertTrue(hasattr(tp, name), name)
            import importlib
            mod = importlib.import_module(f"tensorplay.{name}")
            self.assertIs(getattr(tp, name), mod)

    def test_distributions_surface(self):
        d = tp.distributions
        for name in ("Normal", "Bernoulli", "Categorical", "Uniform"):
            self.assertTrue(hasattr(d, name), name)

    def test_return_types(self):
        rt = tp.return_types
        self.assertIn(tp.return_types.max, rt.all_return_types)
        self.assertIn(tp.return_types.linalg_svd, rt.all_return_types)
        r = tp.max(tp.randn(3, 3), 1)
        self.assertTrue(hasattr(r, "values"))
        self.assertTrue(hasattr(r, "indices"))
        self.assertIs(type(r), rt.max)

    def test_top_level_aliases(self):
        self.assertTrue(callable(tp.vmap))
        self.assertTrue(issubclass(tp.OutOfMemoryError, RuntimeError))
        self.assertIs(tp.memory_format, tp.MemoryFormat)
        self.assertIsInstance(tp.has_lapack, bool)
        self.assertIsInstance(tp.has_spectral, bool)
        self.assertIsInstance(tp.compiled_with_cxx11_abi, bool)

    def test_cpu_module(self):
        self.assertTrue(tp.cpu.is_available())
        self.assertTrue(tp.cpu.is_initialized())
        self.assertEqual(tp.cpu.device_count(), 1)
        self.assertEqual(tp.cpu.current_device(), 0)
        tp.cpu.synchronize()
        tp.cpu.set_device(0)
        caps = tp.cpu.get_capabilities()
        self.assertIsInstance(caps, dict)
        self.assertIn("isa", caps)

    def test_get_device_module(self):
        self.assertIs(tp.get_device_module("cpu"), tp.cpu)
        self.assertIs(tp.get_device_module(tp.device("cpu")), tp.cpu)
        with self.assertRaises(RuntimeError):
            tp.get_device_module(3.5)

    def test_thread_safe_generator_main_process(self):
        self.assertIsNone(tp.thread_safe_generator())


class TestReductionReturns(unittest.TestCase):
    def test_max_min_int_dim_dispatch(self):
        t = tp.tensor([[1.0, 5.0], [3.0, 2.0]])
        r = tp.max(t, 1)
        self.assertEqual(r.values.tolist(), [5.0, 3.0])
        self.assertEqual(r.indices.tolist(), [1, 0])
        r2 = tp.max(t, 0, False)
        self.assertEqual(r2.values.tolist(), [3.0, 5.0])
        m = tp.min(t, 1)
        self.assertEqual(m.values.tolist(), [1.0, 2.0])
        # elementwise face unchanged
        self.assertEqual(tp.max(t, tp.ones(2, 2)).tolist(), [[1, 5], [3, 2]])
        self.assertEqual(tp.max(t).item(), 5.0)

    def test_multi_output_named_tuples(self):
        t = tp.randn(4, 5)
        self.assertEqual(tp.cummax(t, 1).values.shape, [4, 5])
        self.assertEqual(tp.cummin(t, 1).indices.shape, [4, 5])
        self.assertEqual(tp.kthvalue(t, 2, dim=1).values.shape, [4])
        self.assertEqual(tp.mode(t, 1).values.shape, [4])
        self.assertEqual(tp.sort(t).values.shape, [4, 5])
        self.assertEqual(tp.sort(t).indices.shape, [4, 5])
        self.assertEqual(tp.topk(t, 2).values.shape, [4, 2])
        self.assertEqual(tp.aminmax(t).min.shape, [])
        self.assertEqual(tp.aminmax(t).max.shape, [])
        self.assertEqual(tp.median(t, 1).values.shape, [4])
        self.assertEqual(tp.median(t).shape, [])
        self.assertEqual(tp.nanmedian(t, 1).values.shape, [4])

    def test_max_vs_reference(self):
        if torch is None:
            self.skipTest("reference runtime not installed")
        data = np.random.RandomState(0).randn(4, 5).astype(np.float32)
        a = tp.max(tp.from_numpy(data), 1)
        b = torch.max(torch.from_numpy(data), 1)
        self.assertTrue(_same(a.values.numpy(), b.values.numpy()))
        self.assertEqual(a.indices.tolist(), b.indices.tolist())


class TestSparseLegacyMatmuls(unittest.TestCase):
    def _ab(self):
        data = np.random.RandomState(1).randn(3, 3).astype(np.float32)
        dense = tp.from_numpy(data)
        return dense.to_sparse(), dense

    def test_spmm(self):
        s, a = self._ab()
        b = tp.randn(3, 3)
        self.assertTrue(_same(tp.spmm(s, b).numpy(), (a @ b).numpy()))

    def test_dsmm(self):
        s, a = self._ab()
        b = tp.randn(3, 3)
        self.assertTrue(_same(tp.dsmm(b, s).numpy(), (b @ a).numpy()))

    def test_hsmm(self):
        s, a = self._ab()
        b = tp.randn(3, 3)
        self.assertTrue(_same(tp.hsmm(b, s).numpy(), (b @ a).numpy()))

    def test_saddmm(self):
        s, a = self._ab()
        b = tp.randn(3, 3)
        want = 0.5 * a + 2.0 * (a @ b)
        self.assertTrue(_same(tp.saddmm(s, s, b, beta=0.5, alpha=2.0).numpy(), want.numpy()))
        self.assertTrue(_same(tp.saddmm(s, s, b, beta=0).numpy(), (a @ b).numpy()))


class TestAutocastState(unittest.TestCase):
    def test_enabled_roundtrip(self):
        was = tp.is_autocast_cpu_enabled()
        try:
            tp.set_autocast_cpu_enabled(True)
            self.assertTrue(tp.is_autocast_cpu_enabled())
            tp.set_autocast_cpu_enabled(False)
            self.assertFalse(tp.is_autocast_cpu_enabled())
        finally:
            tp.set_autocast_cpu_enabled(was)

    def test_dtype_roundtrip_and_guard(self):
        was = tp._C.get_autocast_cpu_dtype()
        try:
            tp.set_autocast_cpu_dtype(tp.bfloat16)
            self.assertEqual(tp._C.get_autocast_cpu_dtype(), tp.bfloat16)
            tp.set_autocast_cpu_dtype(tp.float16)
            self.assertEqual(tp._C.get_autocast_cpu_dtype(), tp.float16)
            with self.assertRaises(ValueError):
                tp.set_autocast_cpu_dtype(tp.float32)
        finally:
            tp.set_autocast_cpu_dtype(was)


class TestWaitFuture(unittest.TestCase):
    def test_wait(self):
        f = tp.futures.Future()
        f.set_result(42)
        self.assertEqual(tp.wait(f), 42)


class TestSoftmax2d(unittest.TestCase):
    def test_forward(self):
        data = np.random.RandomState(2).randn(2, 3, 4, 4).astype(np.float32)
        out = nn.Softmax2d()(tp.from_numpy(data))
        sums = out.sum(dim=1)
        self.assertTrue(_same(sums.numpy(), np.ones((2, 4, 4))))
        if torch is not None:
            ref = torch.nn.Softmax2d()(torch.from_numpy(data))
            self.assertTrue(_same(out.numpy(), ref.numpy(), tol=1e-6))

    def test_rejects_non_spatial(self):
        with self.assertRaises(ValueError):
            nn.Softmax2d()(tp.randn(5))


class TestCrossMapLRN2d(unittest.TestCase):
    def _reference(self, data, size, alpha, beta, k):
        x = torch.from_numpy(data).requires_grad_(True)
        g = torch.from_numpy(
            np.random.RandomState(size).randn(*data.shape).astype(np.float32))
        out = torch.nn.CrossMapLRN2d(size, alpha=alpha, beta=beta, k=k)(x)
        out.backward(g)
        return out.detach().numpy(), x.grad.detach().numpy()

    def test_forward_backward(self):
        if torch is None:
            self.skipTest("reference runtime not installed")
        data = np.random.RandomState(3).randn(2, 5, 6, 6).astype(np.float32)
        for size, alpha, beta, k in ((3, 1e-4, 0.75, 1.0), (5, 1e-3, 0.5, 2.0)):
            x = tp.from_numpy(data).requires_grad_(True)
            out = nn.CrossMapLRN2d(size, alpha=alpha, beta=beta, k=k)(x)
            ref_out, ref_grad = self._reference(data, size, alpha, beta, k)
            self.assertTrue(_same(out.detach().numpy(), ref_out, tol=1e-5))
            g = tp.from_numpy(
                np.random.RandomState(size).randn(*data.shape).astype(np.float32))
            out.backward(g)
            self.assertTrue(_same(x.grad.detach().numpy(), ref_grad, tol=1e-5))

    def test_rejects_non_4d(self):
        with self.assertRaises(ValueError):
            nn.CrossMapLRN2d(3)(tp.randn(5, 5))

    def test_differs_from_local_response_norm(self):
        # The sliding window is intentionally not the local_response_norm
        # window; assert they disagree so nobody "simplifies" the routing.
        data = np.random.RandomState(4).randn(1, 4, 4, 4).astype(np.float32)
        x = tp.from_numpy(data)
        a = nn.CrossMapLRN2d(3)(x)
        b = nn.LocalResponseNorm(3)(x)
        self.assertFalse(np.allclose(a.numpy(), b.numpy()))


class TestNNGrad(unittest.TestCase):
    def _check(self, dim):
        rng = np.random.RandomState(dim + 1)
        shapes = {1: (1, 2, 8), 2: (1, 2, 7, 7), 3: (1, 2, 5, 5, 5)}
        k = {1: 3, 2: 3, 3: 2}
        ishape = shapes[dim]
        wshape = (2, 2) + (k[dim],) * dim
        inp = tp.from_numpy(rng.randn(*ishape).astype(np.float32)).requires_grad_(True)
        w = tp.from_numpy(rng.randn(*wshape).astype(np.float32)).requires_grad_(True)
        conv = getattr(nn.functional, f"conv{dim}d")
        out = conv(inp, w)
        go = tp.from_numpy(rng.randn(*out.shape).astype(np.float32))
        want_i, want_w = tp.autograd.grad(out, [inp, w], go)
        grad = nn.grad
        got_i = getattr(grad, f"conv{dim}d_input")(inp.shape, w, go)
        got_w = getattr(grad, f"conv{dim}d_weight")(inp, w.shape, go)
        self.assertTrue(_same(got_i.detach().numpy(), want_i.detach().numpy()))
        self.assertTrue(_same(got_w.detach().numpy(), want_w.detach().numpy()))

    def test_conv_1d_2d_3d(self):
        for dim in (1, 2, 3):
            with self.subTest(dim=dim):
                self._check(dim)

    def test_conv2d_strided(self):
        rng = np.random.RandomState(9)
        inp = tp.from_numpy(rng.randn(2, 2, 7, 7).astype(np.float32)).requires_grad_(True)
        w = tp.from_numpy(rng.randn(3, 2, 3, 3).astype(np.float32)).requires_grad_(True)
        out = nn.functional.conv2d(inp, w, stride=2, padding=1)
        go = tp.from_numpy(rng.randn(*out.shape).astype(np.float32))
        want_i, want_w = tp.autograd.grad(out, [inp, w], go)
        got_i = nn.grad.conv2d_input(inp.shape, w, go, stride=2, padding=1)
        got_w = nn.grad.conv2d_weight(inp, w.shape, go, stride=2, padding=1)
        self.assertTrue(_same(got_i.detach().numpy(), want_i.detach().numpy()))
        self.assertTrue(_same(got_w.detach().numpy(), want_w.detach().numpy()))

    def test_conv2d_dilated_grouped(self):
        rng = np.random.RandomState(11)
        inp = tp.from_numpy(rng.randn(1, 4, 6, 6).astype(np.float32)).requires_grad_(True)
        w = tp.from_numpy(rng.randn(4, 2, 3, 3).astype(np.float32)).requires_grad_(True)
        out = nn.functional.conv2d(inp, w, dilation=2, groups=2)
        go = tp.from_numpy(rng.randn(*out.shape).astype(np.float32))
        want_i, want_w = tp.autograd.grad(out, [inp, w], go)
        got_i = nn.grad.conv2d_input(inp.shape, w, go, dilation=2, groups=2)
        got_w = nn.grad.conv2d_weight(inp, w.shape, go, dilation=2, groups=2)
        self.assertTrue(_same(got_i.detach().numpy(), want_i.detach().numpy()))
        self.assertTrue(_same(got_w.detach().numpy(), want_w.detach().numpy()))


class TestTensorMethodFaces(unittest.TestCase):
    def test_dtype_shortcuts(self):
        t = tp.randn(2, 2)
        self.assertEqual(t.cfloat().dtype, tp.complex64)
        self.assertEqual(t.cdouble().dtype, tp.complex128)

    def test_nelement_equal(self):
        t = tp.randn(2, 3)
        self.assertEqual(t.nelement(), 6)
        self.assertTrue(t.equal(t.clone()))
        self.assertFalse(t.equal(t + 1))

    def test_istft_method_roundtrip(self):
        if torch is None:
            self.skipTest("reference runtime not installed")
        data = np.random.RandomState(5).randn(2, 64).astype(np.float32)
        w = tp.from_numpy(
            np.hanning(16).astype(np.float32))
        spec = tp.stft(tp.from_numpy(data), 16, window=w)
        back = spec.istft(16, window=w, length=64)
        ref_in = torch.from_numpy(data)
        ref_w = torch.from_numpy(np.hanning(16).astype(np.float32))
        ref = torch.stft(ref_in, 16, window=ref_w, return_complex=True)
        ref_back = torch.istft(ref, 16, window=ref_w)
        self.assertTrue(_same(back.numpy(), ref_back.numpy(), tol=1e-4))

    def test_to_sparse_coo(self):
        t = tp.randn(3, 3)
        self.assertTrue(t.to_sparse_coo().is_sparse)

    def test_module_load(self):
        z = tp.zeros(2, 2)
        out = z.module_load(tp.ones(2, 2))
        self.assertTrue(_same(out.numpy(), np.ones((2, 2))))
        self.assertIsNot(out, z)
        out2 = z.module_load(tp.ones(2, 2), assign=True)
        self.assertTrue(_same(out2.numpy(), np.ones((2, 2))))


class TestTensorDunders(unittest.TestCase):
    def test_div_family(self):
        t = tp.tensor([2.0, 4.0])
        self.assertEqual(t.__div__(2).tolist(), [1.0, 2.0])
        self.assertEqual(t.__rdiv__(2).tolist(), [1.0, 0.5])
        u = tp.tensor([2.0, 4.0])
        u.__idiv__(2)
        self.assertEqual(u.tolist(), [1.0, 2.0])

    def test_long_and_nonzero(self):
        self.assertEqual(tp.tensor([1, 2]).__long__().dtype, tp.int64)
        self.assertTrue(bool(tp.tensor(1.0)))
        self.assertFalse(bool(tp.tensor(0.0)))

    def test_reversed(self):
        r = reversed(tp.tensor([1, 2, 3]))
        self.assertTrue(tp.is_tensor(r))
        self.assertEqual(r.tolist(), [3, 2, 1])

    def test_contains(self):
        t = tp.tensor([1.0, 2.0])
        self.assertIn(1.0, t)
        self.assertNotIn(5.0, t)
        self.assertNotIn(float("nan"), tp.tensor([float("nan")]))
        with self.assertRaises(RuntimeError):
            [1] in t

    def test_rmatmul(self):
        m = tp.ones(2, 2)
        r = m.__rmatmul__(tp.ones(2, 2))
        self.assertEqual(r.shape, [2, 2])
        self.assertTrue(_same(r.numpy(), 2 * np.ones((2, 2))))

    def test_deepcopy(self):
        leaf = tp.randn(2, 2, requires_grad=True)
        d = copy.deepcopy(leaf)
        self.assertTrue(d.requires_grad)
        self.assertNotEqual(d.data_ptr(), leaf.data_ptr())
        self.assertTrue(_same(d.detach().numpy(), leaf.detach().numpy()))
        nonleaf = leaf * 2
        with self.assertRaises(RuntimeError):
            copy.deepcopy(nonleaf)

    def test_delitem(self):
        with self.assertRaises(TypeError):
            del tp.ones(2, 2)[0]

    def test_numpy_interop(self):
        t = tp.arange(6.0).reshape(2, 3)
        r = np.sin(t)
        self.assertTrue(tp.is_tensor(r))
        wrapped = t.__array_wrap__(np.ones((2, 3)))
        self.assertTrue(tp.is_tensor(wrapped))
        self.assertEqual(tp.Tensor.__array_priority__, 1000)

    def test_cuda_array_interface_absent_on_cpu(self):
        t = tp.randn(2, 2)
        self.assertFalse(hasattr(t, "__cuda_array_interface__"))
        with self.assertRaises(AttributeError):
            t.__cuda_array_interface__


if __name__ == "__main__":
    unittest.main()
