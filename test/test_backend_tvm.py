"""Tests for the Apache-TVM compiler backend."""

import unittest

import tensorplay as tp
from tensorplay.backends.tvm import has_tvm


def _require_tvm_or_skip():
    if not has_tvm():
        raise unittest.SkipTest("apache-tvm is not installed")


def _shape(t):
    s = t.shape
    return tuple(s() if callable(s) else s)


def _reference(x, y):
    return tp.sigmoid(tp.tanh(x + y)) * (x * y) - x


class TvmAvailabilityTest(unittest.TestCase):
    def test_missing_package_error_is_actionable(self):
        if has_tvm():
            self.skipTest("apache-tvm installed; error path unexercised")
        model = lambda a: tp.relu(a)  # noqa: E731

        with self.assertRaises(RuntimeError) as ctx:
            tp.compile(model, backend="tvm")(tp.tensor([1.0]))
        message = str(ctx.exception)
        self.assertIn("apache-tvm", message)
        self.assertIn("install", message.lower())

    def test_backend_registered(self):
        self.assertIn("tvm", tp.compiler.list_backends())


@unittest.skipUnless(has_tvm(), "apache-tvm is not installed")
class TvmNumericParityTest(unittest.TestCase):
    def test_pointwise_chain_matches_eager(self):
        def model(a, b):
            return tp.sigmoid(tp.tanh(a + b) * (a * b)) - a

        compiled = tp.compile(model, backend="tvm")
        x = tp.randn(16, 16)
        y = tp.randn(16, 16)
        got = compiled(x, y)
        want = model(x, y)
        self.assertTrue(bool(tp.allclose(got, want, atol=1e-5, rtol=1e-5)))

    def test_alpha_form_and_unary_mix(self):
        def model(a, b):
            return tp.abs((a - b) * 2.0)

        compiled = tp.compile(model, backend="tvm")
        x = tp.randn(8)
        y = tp.randn(8)
        self.assertTrue(
            bool(tp.allclose(compiled(x, y), model(x, y), atol=1e-6, rtol=1e-6))
        )

    def test_custom_op_boundary_falls_back(self):
        from tensorplay import library

        @library.custom_op("tvmns::opaque", mutates_args=())
        def opaque(x):
            return tp.add(x, 1.0)

        def model(a):
            return tp.mul(opaque(a), 2.0)

        compiled = tp.compile(model, backend="tvm")
        x = tp.tensor([1.0, 2.0])
        # The custom op is an unsupportable node: the backend must fall back
        # to the interpreter and still produce correct results.
        self.assertTrue(bool(tp.allclose(compiled(x), model(x))))

    def test_training_region_keeps_native_path(self):
        def model(a):
            return tp.exp(a)

        compiled = tp.compile(model, backend="tvm")
        x = tp.randn(4, requires_grad=True)
        out = compiled(x)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        expected = [float(v) for v in tp.exp(x).tolist()]
        self.assertEqual([float(g) for g in x.grad.tolist()], expected)

    def test_shape_change_triggers_fallback_not_corruption(self):
        def model(a):
            return tp.sqrt(a * a + 1.0)

        compiled = tp.compile(model, backend="tvm")
        small = tp.randn(4)
        large = tp.randn(32)
        self.assertEqual(_shape(compiled(small)), (4,))
        # A differently-shaped input misses the specialization guard; the
        # wrapper must recompile/fall back instead of writing wrong data.
        want = model(large)
        got = compiled(large)
        self.assertEqual(_shape(got), (32,))
        self.assertTrue(bool(tp.allclose(got, want, atol=1e-5, rtol=1e-5)))

    @unittest.skipUnless(tp.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_target_matches_eager(self):
        def model(a, b):
            return tp.tanh(a + b) / 2.0

        compiled = tp.compile(model, backend="tvm")
        x = tp.randn(64, device="cuda")
        y = tp.randn(64, device="cuda")
        got = compiled(x, y).cpu()
        want = model(x, y).cpu()
        self.assertTrue(bool(tp.allclose(got, want, atol=1e-4, rtol=1e-4)))


if __name__ == "__main__":
    unittest.main()
