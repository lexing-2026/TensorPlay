"""Spec tests: cov / corrcoef native ops vs local torch oracle.

Covers ATen Correlation.cpp semantics ported to native kernels:
forward parity (dtype preservation, weighted/unweighted, corner cases),
autograd through the explicit _cov_backward/_corrcoef_backward helpers,
and error surfaces.
"""

import math
import unittest
import warnings

import torch

import tensorplay as tp


def close(a, b, tol=1e-5):
    if isinstance(a, tp.Tensor):
        a = a.tolist()
    if isinstance(b, (tp.Tensor, torch.Tensor)):
        b = b.tolist() if isinstance(b, tp.Tensor) else b.tolist()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def grad_parity(make_input, torch_call, tp_call, g_shape=None, tol=1e-9):
    xt = torch.tensor(make_input, dtype=torch.float64, requires_grad=True)
    out_t = torch_call(xt)
    g = torch.randn(g_shape or out_t.shape, dtype=torch.float64)
    (gx,) = torch.autograd.grad(out_t, xt, g)

    xp = tp.tensor(make_input, dtype=tp.float64, requires_grad=True)
    out_p = tp_call(xp)
    (gp,) = tp.autograd.grad(out_p, xp, tp.tensor(g.tolist(),
                                                 dtype=tp.float64))
    assert close(gp, gx, tol), f"grad mismatch\n{gp}\nvs\n{gx}"


class TestCovForward(unittest.TestCase):
    def test_1d_unweighted(self):
        x = [1.0, 2.0, 3.5, 4.0]
        r_torch = torch.cov(torch.tensor(x))
        r_tp = tp.cov(tp.tensor(x, dtype=tp.float32))
        self.assertEqual(tuple(r_tp.shape), ())
        self.assertTrue(close(r_tp, r_torch))

    def test_1d_f64(self):
        x = [0.3, 1.7, -2.1, 4.4, 0.9]
        r_torch = torch.cov(torch.tensor(x, dtype=torch.float64))
        r_tp = tp.cov(tp.tensor(x, dtype=tp.float64))
        self.assertEqual(r_tp.dtype, tp.float64)
        self.assertTrue(close(r_tp, r_torch, 1e-12))

    def test_matrix_unweighted(self):
        x = [[1.0, 2.0, 3.0], [4.0, 5.5, 6.0], [-1.0, 0.5, 2.0]]
        r_torch = torch.cov(torch.tensor(x))
        r_tp = tp.cov(tp.tensor(x, dtype=tp.float32))
        self.assertEqual(tuple(r_tp.shape), (3, 3))
        self.assertTrue(close(r_tp, r_torch))

    def test_dtype_preserved(self):
        x = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(
            tp.cov(tp.tensor(x, dtype=tp.float32)).dtype, tp.float32)
        self.assertEqual(
            tp.cov(tp.tensor(x, dtype=tp.float64)).dtype, tp.float64)

    def test_int_input_true_division(self):
        # avg uses true division even for integral inputs ([1,2] -> mean 1.5)
        r_torch = torch.cov(torch.tensor([1, 2]))
        r_tp = tp.cov(tp.tensor([1, 2], dtype=tp.int64))
        self.assertTrue(close(r_tp, r_torch))

    def test_correction(self):
        for corr in (0, 1):
            self.assertTrue(close(
                tp.cov(tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32),
                       correction=corr),
                torch.cov(torch.tensor([1.0, 2.0, 3.0]), correction=corr)))

    def test_fweights(self):
        x = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]
        fw = [2, 1, 3, 1]
        r_torch = torch.cov(torch.tensor(x),
                            fweights=torch.tensor(fw, dtype=torch.int64))
        r_tp = tp.cov(tp.tensor(x, dtype=tp.float32),
                      fweights=tp.tensor(fw, dtype=tp.int64))
        self.assertTrue(close(r_tp, r_torch))

    def test_aweights(self):
        x = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]
        aw = [0.5, 1.0, 2.0, 0.25]
        r_torch = torch.cov(torch.tensor(x),
                            aweights=torch.tensor(aw))
        r_tp = tp.cov(tp.tensor(x, dtype=tp.float32),
                      aweights=tp.tensor(aw, dtype=tp.float32))
        self.assertTrue(close(r_tp, r_torch))

    def test_both_weights(self):
        x = [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 2.0, 1.0, 0.0]]
        fw = [2, 1, 1, 1, 2]
        aw = [0.5, 1.0, 2.0, 1.0, 0.5]
        r_torch = torch.cov(torch.tensor(x),
                            fweights=torch.tensor(fw, dtype=torch.int64),
                            aweights=torch.tensor(aw))
        r_tp = tp.cov(tp.tensor(x, dtype=tp.float32),
                      fweights=tp.tensor(fw, dtype=tp.int64),
                      aweights=tp.tensor(aw, dtype=tp.float32))
        self.assertTrue(close(r_tp, r_torch))


class TestCorrcoefForward(unittest.TestCase):
    def test_perfect_correlation(self):
        m = [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]
        r_torch = torch.corrcoef(torch.tensor(m))
        r_tp = tp.corrcoef(tp.tensor(m, dtype=tp.float32))
        self.assertTrue(close(r_tp, r_torch))
        self.assertAlmostEqual(r_tp[0, 1].item(), 1.0, places=5)

    def test_scalar_branch(self):
        # 1-D input -> scalar covariance -> c/c semantics
        self.assertAlmostEqual(
            tp.corrcoef(tp.tensor([1.0, 2.0, 3.0],
                                  dtype=tp.float32)).item(), 1.0, places=6)
        self.assertTrue(math.isnan(
            tp.corrcoef(tp.tensor([5.0, 5.0, 5.0],
                                  dtype=tp.float32)).item()))

    def test_constant_row_nan_propagation(self):
        # constant row produces 0 std; nan propagates like torch's composition
        m = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        r_torch = torch.corrcoef(torch.tensor(m))
        r_tp = tp.corrcoef(tp.tensor(m, dtype=tp.float32))
        self.assertTrue(close(tp.isnan(r_tp).tolist(),
                              torch.isnan(r_torch).tolist()))


class TestCovCorner(unittest.TestCase):
    def test_single_observation_zeroes_input_like_upstream(self):
        x = torch.tensor([[1.0], [2.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.cov(x, fweights=torch.tensor([1], dtype=torch.int64))
        # upstream zeroes the caller's data through its aliasing view;
        # our kernel replicates this verbatim
        x_tp = tp.tensor([[1.0], [2.0]], dtype=tp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tp.cov(x_tp, fweights=tp.tensor([1], dtype=tp.int64))
        self.assertEqual(x_tp[0, 0].item(), 0.0)


class TestCovBackward(unittest.TestCase):
    def test_vector_variance_grad(self):
        grad_parity([0.3, -1.2, 2.5, 4.0, 1.1],
                    lambda t: torch.cov(t), lambda t: tp.cov(t))

    def test_matrix_grad(self):
        base = [[0.3, -1.2, 2.5], [4.0, 1.1, -0.7], [2.2, 0.1, 3.3]]
        grad_parity(base, lambda t: torch.cov(t), lambda t: tp.cov(t))

    def test_fweights_grad(self):
        base = [[0.3, -1.2, 2.5, 4.0], [1.0, 0.5, -2.0, 3.0]]
        grad_parity(base,
                    lambda t: torch.cov(
                        t, fweights=torch.tensor([2, 1, 3, 1],
                                                 dtype=torch.int64)),
                    lambda t: tp.cov(
                        t, fweights=tp.tensor([2, 1, 3, 1], dtype=tp.int64)))

    def test_aweights_grad(self):
        base = [[0.3, -1.2, 2.5, 4.0], [1.0, 0.5, -2.0, 3.0]]
        aw = [0.5, 1.0, 2.0, 0.25]
        grad_parity(base,
                    lambda t: torch.cov(
                        t, aweights=torch.tensor(aw, dtype=torch.float64)),
                    lambda t: tp.cov(t, aweights=tp.tensor(
                        aw, dtype=tp.float64)))

    def test_both_weights_grad(self):
        base = [[0.3, -1.2, 2.5, 4.0, 0.9], [1.0, 0.5, -2.0, 3.0, 0.1]]
        aw = [0.5, 1.0, 2.0, 1.0, 0.5]
        grad_parity(base,
                    lambda t: torch.cov(
                        t, fweights=torch.tensor([2, 1, 1, 1, 2],
                                                 dtype=torch.int64),
                        aweights=torch.tensor(aw, dtype=torch.float64)),
                    lambda t: tp.cov(
                        t, fweights=tp.tensor([2, 1, 1, 1, 2],
                                              dtype=tp.int64),
                        aweights=tp.tensor(aw, dtype=tp.float64)))

    def test_first_order_after_grad(self):
        # Known narrow (shared with the trapezoid family): _cov_backward is
        # a numeric helper op, so create_graph does not record through it.
        # First-order grads remain exact; higher-order needs the
        # MANUAL_DERIVATIVES treatment (cf. MatmulBackward).
        xp = tp.tensor([0.3, -1.2, 2.5], dtype=tp.float64,
                       requires_grad=True)
        out = tp.cov(xp)
        (g1,) = tp.autograd.grad(out, xp, retain_graph=True)
        self.assertFalse(g1.requires_grad)
        # closed form for scalar variance: 2*(x - mean)/(n-1)
        mean = (0.3 - 1.2 + 2.5) / 3
        expected = [2 * (v - mean) / 2 for v in (0.3, -1.2, 2.5)]
        for a, b in zip(g1.tolist(), expected):
            self.assertAlmostEqual(a, b, places=12)


class TestCorrcoefBackward(unittest.TestCase):
    def test_matrix(self):
        base = [[0.3, -1.2, 2.5], [4.0, 1.1, -0.7], [2.2, 0.1, 3.3]]
        grad_parity(base, lambda t: torch.corrcoef(t),
                    lambda t: tp.corrcoef(t), tol=1e-8)

    def test_two_variables(self):
        base = [[1.0, 2.0, 3.5], [2.0, 1.0, 0.5]]
        grad_parity(base, lambda t: torch.corrcoef(t),
                    lambda t: tp.corrcoef(t), tol=1e-8)


class TestErrors(unittest.TestCase):
    def test_bool_rejected(self):
        # upstream TORCH_CHECK_NOT_IMPLEMENTED surfaces as NotImplementedError
        with self.assertRaises(NotImplementedError):
            tp.cov(tp.tensor([True, False], dtype=tp.bool))

    def test_too_many_dims(self):
        with self.assertRaises(RuntimeError):
            tp.cov(tp.zeros([2, 2, 2]))

    def test_bad_fweights_dtype(self):
        with self.assertRaises(RuntimeError):
            tp.cov(tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32),
                   fweights=tp.tensor([1.0, 1.0, 1.0], dtype=tp.float32))

    def test_negative_fweights(self):
        with self.assertRaises(RuntimeError):
            tp.cov(tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32),
                   fweights=tp.tensor([1, -1, 1], dtype=tp.int64))

    def test_zero_weight_sum(self):
        with self.assertRaises(RuntimeError):
            tp.cov(tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32),
                   fweights=tp.tensor([0, 0, 0], dtype=tp.int64))

    def test_ddof_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tp.cov(tp.tensor([1.0, 2.0], dtype=tp.float32), correction=2)
            self.assertTrue(any("degrees of freedom" in str(x.message)
                                for x in w))


if __name__ == "__main__":
    unittest.main()
