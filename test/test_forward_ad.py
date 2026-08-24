import unittest

import tensorplay as tp
from tensorplay.autograd.functional import jacfwd, jvp


class TestJvpForward(unittest.TestCase):
    def test_matches_analytic(self):
        tp.manual_seed(0)
        x = tp.randn(4, 4)
        v = tp.ones_like(x)

        def f(x):
            return (x.tanh() * 2.0 + 1.0).sum(dim=1)

        out_f, jvp_f = jvp(f, (x,), (v,), mode="forward")
        self.assertTrue(tp.allclose(out_f, f(x)))
        # d/dx [2*tanh(x)+1] row-summed with seed 1 == 2*sech^2(x) row-sum.
        expect = (2.0 * (1.0 - x.tanh().pow(2))).sum(dim=1)
        self.assertTrue(tp.allclose(jvp_f, expect))

    def test_elementwise_matches_reversed_mode(self):
        # The double-backward trick agrees with native forward AD for
        # pointwise chains (reductions under create_graph are a known engine
        # gap tracked separately).
        x = tp.tensor([0.4, -1.2, 2.2])
        v = tp.ones_like(x)

        def f(x):
            return x.sigmoid() * x

        _, jvp_r = jvp(f, (x,), (v,), mode="reversed")
        _, jvp_f = jvp(f, (x,), (v,), mode="forward")
        self.assertTrue(tp.allclose(jvp_r, jvp_f))

    def test_multi_input(self):
        x = tp.tensor([1.0, 2.0])
        y = tp.tensor([3.0, 4.0])
        out, d = jvp(lambda a, b: a * b + b.exp(), (x, y),
                     (tp.ones_like(x), tp.zeros_like(y)), mode="forward")
        # Only x perturbed: tangent of x*y is y.
        self.assertTrue(tp.allclose(d, y))


class TestJacfwd(unittest.TestCase):
    def test_scalar_output_jacobian(self):
        x = tp.tensor([0.5, -0.25, 2.0])

        def f(x):
            return x.sin().sum()

        j = jacfwd(f, x)
        expect = tp.cos(x)
        # torch.func.jacfwd convention: scalar output -> shape (in,).
        self.assertEqual(tuple(j.shape), (3,))
        self.assertTrue(tp.allclose(j.reshape(expect.shape), expect))

    def test_vector_output_full_jacobian(self):
        x = tp.tensor([0.3, 1.7])

        def f(x):
            # Tuple output (free functions do not intercept DualTensor).
            return (x[0] * x[1], x[0].exp())

        j = jacfwd(f, x)
        # j[0] = d(out0)/dx = [x1, x0]; j[1] = d(out1)/dx = [exp(x0), 0]
        e0 = float(x[1])
        e1 = float(x[0])
        g0 = float(tp.exp(x[0]))
        self.assertAlmostEqual(float(j[0][0]), e0, places=5)
        self.assertAlmostEqual(float(j[0][1]), e1, places=5)
        self.assertAlmostEqual(float(j[1][0]), g0, places=5)
        self.assertAlmostEqual(float(j[1][1]), 0.0, places=5)

    def test_matrix_input_shape(self):
        x = tp.randn(2, 3)

        def f(x):
            return (x * x).sum()

        j = jacfwd(f, x)
        self.assertEqual(tuple(j.shape), (2, 3))
        self.assertTrue(tp.allclose(j, 2.0 * x))


if __name__ == "__main__":
    unittest.main()
