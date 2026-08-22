import math
import unittest

import tensorplay as tp
from tensorplay.autograd.functional import (
    hessian,
    hvp,
    jacobian,
    jvp,
    vhp,
    vjp,
)


def exp_reducer(x):
    return x.exp().sum(dim=1)


class TestVJP(unittest.TestCase):
    def test_basic(self):
        inputs = tp.rand(4, 4, dtype=tp.float64)
        v = tp.ones(4, dtype=tp.float64)

        out, vjpval = vjp(exp_reducer, inputs, v)
        self.assertEqual(tuple(out.shape), (4,))
        expected = exp_reducer(inputs).detach()
        for i in range(4):
            self.assertAlmostEqual(out[i].item(), expected[i].item(), places=6)

        # vjpval == diag(v) * exp(x): row i of the jacobian scaled by v[i]
        x = inputs.detach().requires_grad_(True)
        outs = exp_reducer(x)
        for i in range(4):
            g = tp.autograd.grad(outs[i], x)[0] * v[i]
            for j in range(4):
                self.assertAlmostEqual(
                    vjpval[i][j].item(), g[j].item(), places=8
                )

    def test_create_graph(self):
        inputs = tp.rand(2, 2, dtype=tp.float64, requires_grad=True)
        v = tp.ones(2, dtype=tp.float64)
        out, vjpval = vjp(exp_reducer, inputs, v, create_graph=True)
        self.assertTrue(out.requires_grad)
        self.assertTrue(vjpval.requires_grad)

    def test_default_v_scalar_output(self):
        x = tp.tensor([3.0], dtype=tp.float64, requires_grad=True)

        def f(t):
            return (t * t).sum()

        out, vjpval = vjp(f, x)
        self.assertAlmostEqual(vjpval.item(), 6.0, places=10)

    def test_strict_independent_input(self):
        x = tp.tensor([1.0], dtype=tp.float64)
        y = tp.tensor([1.0], dtype=tp.float64, requires_grad=True)

        # y is passed but unused by the function -> strict mode must raise
        with self.assertRaises(RuntimeError):
            vjp(lambda a, b: b * b, (x, y), None, strict=True)


class TestJVP(unittest.TestCase):
    def test_matches_vjp_transpose(self):
        # For a linear f: jvp(f)(v) == vjp(f.T)... simplest check: linear map
        W = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64)

        def f(x):
            return tp.matmul(x, W.t())

        x = tp.tensor([1.0, 2.0], dtype=tp.float64)
        v = tp.tensor([1.0, 1.0], dtype=tp.float64)
        out, jvpval = jvp(f, x, v)
        expected = v @ W.t()
        for i in range(2):
            self.assertAlmostEqual(jvpval[i].item(), expected[i].item(), places=10)

    def test_quadratic(self):
        x = tp.tensor([1.0, 2.0], dtype=tp.float64)

        def f(t):
            return (t * t).sum()

        v = tp.tensor([1.0, 1.0], dtype=tp.float64)
        out, jvpval = jvp(f, x, v)
        # d/dt [t^2 sum] applied to v = 2 * t * v
        self.assertAlmostEqual(jvpval[0].item(), 2.0, places=10)
        self.assertAlmostEqual(jvpval[1].item(), 4.0, places=10)

    def test_create_graph(self):
        x = tp.rand(2, dtype=tp.float64, requires_grad=True)
        v = tp.ones(2, dtype=tp.float64)
        out, jvpval = jvp(exp_reducer, x, v, create_graph=True)
        self.assertTrue(out.requires_grad)
        self.assertTrue(jvpval.requires_grad)


class TestJacobian(unittest.TestCase):
    def test_single_input_output(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64)
        jac = jacobian(exp_reducer, x)
        self.assertEqual(tuple(jac.shape), (2, 2, 2))
        xe = x.detach()
        # row i of exp_reducer output depends only on row i of x
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    expected = math.exp(xe[i][k].item()) if j == i else 0.0
                    self.assertAlmostEqual(jac[i][j][k].item(), expected, places=8)

    def test_multiple_inputs(self):
        def exp_adder(x, y):
            return 2 * x.exp() + 3 * y

        x = tp.rand(2, dtype=tp.float64)
        y = tp.rand(2, dtype=tp.float64)
        jxx, jxy = jacobian(exp_adder, (x, y))
        self.assertIsInstance(jxx, tuple)
        self.assertEqual(tuple(jxx[0].shape), (2, 2))
        xd = x.detach()
        for i in range(2):
            for j in range(2):
                expected = 2 * math.exp(xd[i].item()) if i == j else 0.0
                self.assertAlmostEqual(jxx[0][i][j].item(), expected, places=8)
                self.assertAlmostEqual(jxy[0][i][j].item(), 3.0 if i == j else 0.0, places=8)

    def test_create_graph(self):
        x = tp.rand(2, dtype=tp.float64, requires_grad=True)
        jac = jacobian(exp_reducer, x, create_graph=True)
        self.assertTrue(jac.requires_grad)

    def test_vectorize_unsupported(self):
        x = tp.rand(2, dtype=tp.float64)
        with self.assertRaises(NotImplementedError):
            jacobian(exp_reducer, x, vectorize=True)

    def test_forward_mode_unsupported(self):
        x = tp.rand(2, dtype=tp.float64)
        with self.assertRaises(NotImplementedError):
            jacobian(exp_reducer, x, strategy="forward-mode")

    def test_strict_independent(self):
        def f(a, b):
            return a.sum()

        x = tp.rand(2, dtype=tp.float64)
        y = tp.rand(2, dtype=tp.float64)
        with self.assertRaises(RuntimeError):
            jacobian(f, (x, y), strict=True)


class TestHessian(unittest.TestCase):
    def test_cubic(self):
        # f(x) = sum(x^3); H = diag(6x)
        x = tp.tensor([1.5, -2.0], dtype=tp.float64)

        def pow_reducer(t):
            return t.pow(3).sum()

        h = hessian(pow_reducer, x)
        self.assertEqual(tuple(h.shape), (2, 2))
        for i in range(2):
            for j in range(2):
                expected = 6 * x.detach()[i].item() if i == j else 0.0
                self.assertAlmostEqual(h[i][j].item(), expected, places=8)

    def test_two_inputs(self):
        def pow_adder_reducer(x, y):
            return (2 * x.pow(2) + 3 * y.pow(2)).sum()

        x = tp.rand(2, dtype=tp.float64)
        y = tp.rand(2, dtype=tp.float64)
        (hxx, hxy), (hyx, hyy) = hessian(pow_adder_reducer, (x, y))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(hxx[i][j].item(), 4.0 if i == j else 0.0, places=8)
                self.assertAlmostEqual(hyy[i][j].item(), 6.0 if i == j else 0.0, places=8)
                self.assertAlmostEqual(hxy[i][j].item(), 0.0, places=8)

    def test_create_graph(self):
        x = tp.rand(2, dtype=tp.float64, requires_grad=True)
        h = hessian(lambda t: t.pow(3).sum(), x, create_graph=True)
        self.assertTrue(h.requires_grad)

    def test_multi_output_raises(self):
        x = tp.rand(2, dtype=tp.float64)
        with self.assertRaisesRegex(RuntimeError, "single Tensor"):
            hessian(lambda t: t * t, x)


class TestVHPHVP(unittest.TestCase):
    def test_vhp_quadratic(self):
        x = tp.tensor([1.0, 2.0], dtype=tp.float64)
        v = tp.tensor([1.0, 1.0], dtype=tp.float64)

        out, vhpval = vhp(lambda t: (t * t).sum(), x, v)
        # H = 2I -> Hv = 2v
        self.assertAlmostEqual(vhpval[0].item(), 2.0, places=10)
        self.assertAlmostEqual(vhpval[1].item(), 2.0, places=10)

    def test_hvp_matches_vhp(self):
        x = tp.rand(3, dtype=tp.float64)
        v = tp.rand(3, dtype=tp.float64)

        def f(t):
            return t.pow(3).sum()

        _out, hvpval = hvp(f, x, v)
        _out2, vhpval = vhp(f, x, v)
        for i in range(3):
            self.assertAlmostEqual(hvpval[i].item(), vhpval[i].item(), places=8)

    def test_default_v(self):
        x = tp.tensor([2.0], dtype=tp.float64, requires_grad=True)
        out, hv = hvp(lambda t: t.pow(3).sum(), x)
        self.assertAlmostEqual(hv.item(), 12.0, places=10)  # 6 * x * v, v=1


if __name__ == "__main__":
    unittest.main()
