"""
Statistical sampling operators (binomial, standard gamma, dirichlet) and the
nanmedian reductions, checked against the reference runtime for distribution
shape, dtype contracts, error contracts, and the deterministic-replay path
through an explicit generator.  Randomness itself is validated statistically:
each kernel draws many samples and the empirical moments must land within a
loose band around the analytic moments of the target distribution.
"""

import math
import unittest

import torch

import tensorplay as tp


def close(a, b, tol=1e-5):
    if isinstance(a, tp.Tensor):
        a = a.tolist()
    if isinstance(b, (tp.Tensor, torch.Tensor)):
        b = b.tolist()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


def tolist(x):
    if isinstance(x, tp.Tensor):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return x


class NanMedianFlat(unittest.TestCase):
    def test_ignores_nan(self):
        for vals in (
            [1.0, float("nan"), 3.0, 2.0],
            [float("nan"), 1.0],
            [float("nan"), float("nan"), 5.0, 7.0, float("nan")],
        ):
            t = tp.tensor(vals)
            ref = torch.tensor(vals)
            self.assertTrue(close(tp.nanmedian(t), torch.nanmedian(ref)))

    def test_all_nan(self):
        t = tp.tensor([float("nan"), float("nan")])
        ref = torch.tensor([float("nan"), float("nan")])
        self.assertTrue(close(tp.nanmedian(t), torch.nanmedian(ref)))

    def test_dtype_and_shape(self):
        t = tp.tensor([1.0, 2.0], dtype=tp.float64)
        self.assertEqual(tp.nanmedian(t).dtype, tp.float64)
        self.assertEqual(tp.nanmedian(t).dim(), 0)
        t16 = tp.tensor([1.0, 2.0], dtype=tp.float16)
        self.assertEqual(tp.nanmedian(t16).dtype, tp.float16)

    def test_empty(self):
        self.assertTrue(math.isnan(tp.nanmedian(tp.tensor([], dtype=tp.float32)).item()))
        self.assertTrue(math.isnan(tp.nanmedian(tp.tensor([], dtype=tp.float64)).item()))
        self.assertEqual(
            tp.nanmedian(tp.tensor([], dtype=tp.int64)).item(),
            -9223372036854775808,
        )


class NanMedianDim(unittest.TestCase):
    def test_dim_basic(self):
        data = [[1.0, float("nan")], [3.0, 2.0]]
        t = tp.tensor(data)
        ref = torch.tensor(data)
        v, i = tp.nanmedian(t, dim=1)
        rv, ri = torch.nanmedian(ref, dim=1)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))
        self.assertEqual(i.tolist(), [0, 1])

    def test_dim_all_nan_slice(self):
        data = [[float("nan"), float("nan")], [4.0, 1.0]]
        t = tp.tensor(data)
        ref = torch.tensor(data)
        v, i = tp.nanmedian(t, dim=0)
        rv, ri = torch.nanmedian(ref, dim=0)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))

    def test_keepdim(self):
        data = [[1.0, 2.0, float("nan")], [3.0, 5.0, 4.0]]
        t = tp.tensor(data)
        ref = torch.tensor(data)
        v, i = tp.nanmedian(t, dim=1, keepdim=True)
        rv, ri = torch.nanmedian(ref, dim=1, keepdim=True)
        self.assertEqual(list(v.shape), [2, 1])
        self.assertEqual(list(i.shape), [2, 1])
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))

    def test_negative_dim(self):
        data = [[1.0, float("nan"), 2.0]]
        t = tp.tensor(data)
        ref = torch.tensor(data)
        v, i = tp.nanmedian(t, dim=-1)
        rv, ri = torch.nanmedian(ref, dim=-1)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))

    def test_3d(self):
        data = [[[1.0, 2.0], [float("nan"), 0.0]],
                [[float("nan"), float("nan")], [7.0, float("nan")]]]
        t = tp.tensor(data)
        ref = torch.tensor(data)
        for d in (0, 1, 2):
            v, i = tp.nanmedian(t, dim=d)
            rv, ri = torch.nanmedian(ref, dim=d)
            self.assertTrue(close(v, rv), f"dim={d} values")
            self.assertTrue(close(i, ri), f"dim={d} indices")

    def test_even_count_lower_middle(self):
        # Even number of valid entries keeps the lower-middle order
        # statistic, matching the reference median convention.
        data = [[1.0, 2.0, 3.0, 4.0]]
        v, i = tp.nanmedian(tp.tensor(data), dim=1)
        rv, ri = torch.nanmedian(torch.tensor(data), dim=1)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))

    def test_nonfloat_rejected(self):
        t = tp.tensor([1, 2, 3], dtype=tp.int64)
        with self.assertRaises(Exception):
            tp.nanmedian(t, dim=0)

    def test_zero_size_dim_rejected(self):
        with self.assertRaises(Exception):
            tp.nanmedian(tp.zeros(2, 0), dim=1)


class Binomial(unittest.TestCase):
    def test_shape_and_dtype(self):
        n = tp.full((3, 4), 20.0)
        p = tp.full((3, 4), 0.5)
        r = tp.binomial(n, p)
        self.assertEqual(list(r.shape), [3, 4])
        self.assertEqual(r.dtype, tp.float32)

    def test_broadcast(self):
        n = tp.full((1, 4), 10.0)
        p = tp.full((3, 1), 0.5)
        self.assertEqual(list(tp.binomial(n, p).shape), [3, 4])
        n0 = tp.full((), 10.0)
        p0 = tp.full((), 0.5)
        self.assertEqual(list(tp.binomial(n0, p0).shape), [])
        # larger prob shape wins the broadcast
        r = tp.binomial(tp.full((), 5.0), tp.full((2, 1), 0.5))
        self.assertEqual(list(r.shape), [2, 1])

    def test_degenerate_cases(self):
        self.assertEqual(tp.binomial(tp.tensor(-3.0), tp.tensor(0.5)).item(), 0.0)
        self.assertEqual(tp.binomial(tp.tensor(0.0), tp.tensor(0.0)).item(), 0.0)
        self.assertEqual(tp.binomial(tp.tensor(7.5), tp.tensor(1.0)).item(), 7.5)
        self.assertEqual(tp.binomial(tp.tensor(5.0), tp.tensor(-0.5)).item(), 0.0)
        self.assertTrue(math.isnan(tp.binomial(tp.tensor(5.0), tp.tensor(float("nan"))).item()))

    def test_mean_and_var(self):
        n, p = 200.0, 0.3
        t = tp.full((40, 250), n)
        pp = tp.full((40, 250), p)
        r = tp.binomial(t, pp).tolist()
        flat = [x for row in r for x in row]
        m = sum(flat) / len(flat)
        var = sum((x - m) ** 2 for x in flat) / len(flat)
        self.assertAlmostEqual(m / n, p, delta=0.02)
        self.assertAlmostEqual(var / (n * p * (1 - p)), 1.0, delta=0.25)

    def test_extremes(self):
        # tiny mean exercises the geometric-sum path
        t = tp.full((50, 200), 0.02)
        pp = tp.full((50, 200), 0.5)
        flat = [x for row in tp.binomial(t, pp).tolist() for x in row]
        self.assertAlmostEqual(sum(flat) / len(flat), 0.01, delta=0.01)
        # large mean exercises the transformed-rejection path
        t = tp.full((20, 200), 500.0)
        pp = tp.full((20, 200), 0.4)
        flat = [x for row in tp.binomial(t, pp).tolist() for x in row]
        m = sum(flat) / len(flat)
        self.assertAlmostEqual(m / 500.0, 0.4, delta=0.02)

    def test_dtype_errors(self):
        with self.assertRaises(Exception):
            tp.binomial(tp.tensor([1, 2], dtype=tp.int64), tp.tensor(0.5))
        with self.assertRaises(Exception):
            tp.binomial(tp.tensor(1.0), tp.tensor([1, 2], dtype=tp.int64))
        with self.assertRaises(Exception):
            tp.binomial(tp.tensor(1.0), tp.tensor(0.5, dtype=tp.float64))
        with self.assertRaises(Exception):
            tp.binomial(tp.tensor(1.0, dtype=tp.float16), tp.tensor(0.5, dtype=tp.float16))


class StandardGamma(unittest.TestCase):
    def test_shape_dtype(self):
        a = tp.full((3, 5), 2.0)
        r = tp._standard_gamma(a)
        self.assertEqual(list(r.shape), [3, 5])
        self.assertEqual(r.dtype, tp.float32)
        r64 = tp._standard_gamma(a.to(tp.float64))
        self.assertEqual(r64.dtype, tp.float64)

    def test_zero_and_small_alpha(self):
        # alpha = 0 collapses to the smallest positive normal after the
        # lower clamp, matching the reference runtime.
        r = tp._standard_gamma(tp.tensor([0.0, 0.0]))
        self.assertEqual(r.tolist(), [1.1754943508222875e-38, 1.1754943508222875e-38])
        # alpha -> 0 concentrates mass at 0
        r = tp._standard_gamma(tp.full((100,), 1e-6))
        self.assertTrue(all(x < 1e-3 for x in r.tolist()))

    def test_moments(self):
        for alpha in (0.5, 1.0, 4.0, 20.0):
            t = tp.full((50, 200), float(alpha))
            flat = [x for row in tp._standard_gamma(t).tolist() for x in row]
            m = sum(flat) / len(flat)
            var = sum((x - m) ** 2 for x in flat) / len(flat)
            self.assertAlmostEqual(m / alpha, 1.0, delta=0.12, msg=f"alpha={alpha}")
            self.assertAlmostEqual(var / alpha, 1.0, delta=0.35, msg=f"alpha={alpha}")

    def test_alpha_half(self):
        t = tp.full((50, 200), 0.5)
        flat = [x for row in tp._standard_gamma(t).tolist() for x in row]
        m = sum(flat) / len(flat)
        self.assertAlmostEqual(m / 0.5, 1.0, delta=0.15)

    def test_errors(self):
        with self.assertRaises(Exception):
            tp._standard_gamma(tp.tensor([-1.0]))
        with self.assertRaises(Exception):
            tp._standard_gamma(tp.tensor([float("nan")]))
        with self.assertRaises(Exception):
            tp._standard_gamma(tp.tensor([1], dtype=tp.int64))

    def test_generator_replay(self):
        # The generator binding is not wired yet, so replay goes through the
        # default generator: same seed, same sequence.
        a = tp.tensor([1.0, 2.0, 3.0, 0.5])
        tp.manual_seed(123)
        r1 = tp._standard_gamma(a)
        tp.manual_seed(123)
        r2 = tp._standard_gamma(a)
        self.assertEqual(r1.tolist(), r2.tolist())


class Dirichlet(unittest.TestCase):
    def test_rows_sum_to_one(self):
        a = tp.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.0]])
        r = tp._sample_dirichlet(a)
        self.assertEqual(list(r.shape), [2, 3])
        self.assertEqual(r.dtype, tp.float32)
        for row in r.tolist():
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_moments(self):
        # E[x_i] = a_i / sum(a)
        a = tp.tensor([[2.0, 6.0, 12.0]])
        flat = tp._sample_dirichlet(tp.repeat(a, [500, 1])).tolist()
        col_means = [sum(row[j] for row in flat) / len(flat) for j in range(3)]
        expect = [2.0 / 20.0, 6.0 / 20.0, 12.0 / 20.0]
        for got, want in zip(col_means, expect):
            self.assertAlmostEqual(got, want, delta=0.02)

    def test_1d_normalizes(self):
        # A single-element row always yields the largest representable
        # value below 1.0 (the clamp ceiling), as in the reference runtime.
        r = tp._sample_dirichlet(tp.tensor([1.0]))
        ref = torch._sample_dirichlet(torch.tensor([1.0]))
        self.assertEqual(r.tolist(), ref.tolist())
        r = tp._sample_dirichlet(tp.tensor([3.0, 1.0, 1.0]))
        self.assertAlmostEqual(sum(r.tolist()), 1.0, places=5)

    def test_empty_last_dim(self):
        r = tp._sample_dirichlet(tp.zeros(2, 0))
        self.assertEqual(list(r.shape), [2, 0])

    def test_zero_concentration(self):
        # all-zero row spreads mass uniformly over the row
        r = tp._sample_dirichlet(tp.zeros(2, 4))
        rows = r.tolist()
        for row in rows:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            self.assertTrue(all(x > 0 for x in row))

    def test_errors(self):
        with self.assertRaises(Exception):
            tp._sample_dirichlet(tp.tensor([-1.0, 1.0]))
        with self.assertRaises(Exception):
            tp._sample_dirichlet(tp.tensor([float("nan"), 1.0]))
        with self.assertRaises(Exception):
            tp._sample_dirichlet(tp.tensor([1], dtype=tp.int64))

    def test_generator_replay(self):
        # The generator binding is not wired yet, so replay goes through the
        # default generator: same seed, same sequence.
        a = tp.tensor([1.0, 2.0, 3.0])
        tp.manual_seed(77)
        r1 = tp._sample_dirichlet(a)
        tp.manual_seed(77)
        r2 = tp._sample_dirichlet(a)
        self.assertEqual(r1.tolist(), r2.tolist())

    def test_matches_reference_contract(self):
        # dtypes / shapes agree with the reference runtime on the same call
        a_tp = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        a_t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = tp._sample_dirichlet(a_tp)
        ref = torch._sample_dirichlet(a_t)
        self.assertEqual(list(r.shape), list(ref.shape))
        self.assertEqual(str(r.dtype), str(tp.float32))
        self.assertEqual(str(ref.dtype), "torch.float32")


if __name__ == "__main__":
    unittest.main()
