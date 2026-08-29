"""Behavior tests for the native quantile / nanquantile / histogram batch.

The three ops are device-generic dispatcher composites in
(quantile_impl) and native/Histogram.cpp + cpu/HistogramKernel.cpp; one
registration serves CPU and CUDA.  These cases compare against the local
GPU machine (see .remote_build.md).
"""
import unittest

import numpy as np
import torch

import tensorplay as tp


def t2tp(t):
    return tp.tensor(t.detach().cpu().numpy())


class QuantileTestBase(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.x32 = torch.randn(6, generator=g)
        self.x64 = torch.randn(6, dtype=torch.float64, generator=g)
        self.x2d = torch.randn(2, 3, generator=g)
        self.x3d = torch.randn(2, 3, 4, generator=g)

    def assertParity(self, out_tp, out_torch, rtol=1e-6, atol=1e-7):
        self.assertEqual(list(out_tp.shape), list(out_torch.shape))
        self.assertEqual(
            str(out_tp.dtype),
            str(out_torch.dtype).replace("torch.", "tensorplay."),
        )
        a = out_tp.numpy()
        b = out_torch.detach().cpu().numpy()
        if a.dtype.kind == "f":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            self.assertTrue((nan_a == nan_b).all(), "NaN placement mismatch")
            a, b = a[~nan_a], b[~nan_b]
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


class TestQuantile(QuantileTestBase):
    def test_scalar_q_all_interpolations(self):
        for interp in ("linear", "lower", "higher", "midpoint", "nearest"):
            for x in (self.x32, self.x64):
                self.assertParity(
                    tp.quantile(t2tp(x), 0.3, interpolation=interp),
                    torch.quantile(x, 0.3, interpolation=interp),
                )

    def test_q_tensor_dim_keepdim_shapes(self):
        q = torch.tensor([0.1, 0.5, 0.9])
        qs = t2tp(q)
        for x in (self.x2d, self.x3d):
            for dim in (None, 0, 1, x.dim() - 1, -1):
                for keepdim in (False, True):
                    self.assertParity(
                        tp.quantile(t2tp(x), qs, dim=dim, keepdim=keepdim),
                        torch.quantile(x, q, dim=dim, keepdim=keepdim),
                    )

    def test_q_edge_values(self):
        q = torch.tensor([0.0, 1.0])
        self.assertParity(
            tp.quantile(t2tp(self.x2d), t2tp(q), dim=1),
            torch.quantile(self.x2d, q, dim=1),
        )

    def test_noncontiguous_input(self):
        x = torch.randn(4, 5, generator=torch.Generator().manual_seed(3)).t()
        q = torch.tensor([0.25, 0.75])
        self.assertParity(
            tp.quantile(t2tp(x), t2tp(q), dim=1),
            torch.quantile(x, q, dim=1),
        )
        self.assertParity(
            tp.quantile(t2tp(x), t2tp(q)),
            torch.quantile(x, q),
        )

    def test_nan_propagation(self):
        x = torch.tensor([1.0, float("nan"), 3.0, 2.0])
        self.assertParity(tp.quantile(t2tp(x), 0.5), torch.quantile(x, 0.5))
        self.assertParity(
            tp.nanquantile(t2tp(x), 0.5), torch.nanquantile(x, 0.5)
        )
        row = torch.tensor([[1.0, float("nan"), 3.0],
                            [4.0, 5.0, float("nan")]])
        q = torch.tensor([0.2, 0.7])
        self.assertParity(
            tp.nanquantile(t2tp(row), t2tp(q), dim=1),
            torch.nanquantile(row, q, dim=1),
        )

    def test_all_nan_nanquantile(self):
        x = torch.tensor([float("nan"), float("nan")])
        self.assertParity(
            tp.nanquantile(t2tp(x), 0.5), torch.nanquantile(x, 0.5)
        )

    def test_python_number_q_wraps_input_dtype(self):
        self.assertParity(
            tp.quantile(t2tp(self.x64), 0.5),
            torch.quantile(self.x64, 0.5),
        )
        with self.assertRaises(TypeError):
            tp.quantile(t2tp(self.x32), [0.1, 0.9])

    def test_q_range_and_shape_errors(self):
        x = t2tp(self.x32)
        with self.assertRaises(Exception):
            tp.quantile(x, t2tp(torch.tensor(1.5)))
        with self.assertRaises(Exception):
            tp.quantile(x, t2tp(torch.tensor([[0.1, 0.5]])))
        with self.assertRaises(Exception):
            tp.quantile(x, tp.tensor([0.5], dtype=tp.float64))
        with self.assertRaises(Exception):
            tp.quantile(x, 0.5, dim=7)
        with self.assertRaises(Exception):
            tp.quantile(x, 0.5, interpolation="quadratic")
        with self.assertRaises(Exception):
            tp.quantile(tp.tensor([]), 0.5)
        with self.assertRaises(Exception):
            tp.quantile(tp.arange(4), 0.5)


class TestHistogram(QuantileTestBase):
    def test_basic_bins(self):
        for nbins in (1, 4, 10):
            self.assertParity(
                tp.histogram(t2tp(self.x32), nbins)[0],
                torch.histogram(self.x32, nbins)[0],
            )
            self.assertParity(
                tp.histogram(t2tp(self.x32), nbins)[1],
                torch.histogram(self.x32, nbins)[1],
                rtol=1e-6, atol=1e-7,
            )

    def test_bins_tensor(self):
        edges = torch.linspace(float(self.x32.min()), float(self.x32.max()), 5)
        h_tp, e_tp = tp.histogram(t2tp(self.x32), t2tp(edges))
        h_t, e_t = torch.histogram(self.x32, edges)
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t)

    def test_range(self):
        lo, hi = float(self.x32.min()) - 1, float(self.x32.max()) + 1
        h_tp, e_tp = tp.histogram(t2tp(self.x32), 5, range=(lo, hi))
        h_t, e_t = torch.histogram(self.x32, 5, range=(lo, hi))
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t, rtol=1e-6, atol=1e-7)

    def test_weight_and_density(self):
        w = torch.rand(6, generator=torch.Generator().manual_seed(11))
        for density in (False, True):
            h_tp, _ = tp.histogram(t2tp(self.x32), 4, weight=t2tp(w),
                                   density=density)
            h_t, _ = torch.histogram(self.x32, 4, weight=w, density=density)
            self.assertParity(h_tp, h_t, rtol=1e-5, atol=1e-7)

    def test_rightmost_edge_included(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        h_tp, _ = tp.histogram(t2tp(x), 2, range=(0.0, 1.0))
        h_t, _ = torch.histogram(x, 2, range=(0.0, 1.0))
        self.assertParity(h_tp, h_t)
        self.assertEqual(h_t.tolist(), [1.0, 2.0])  # x == hi lands in last bin

    def test_out_of_range_and_nan_skipped(self):
        x = torch.tensor([1.0, 2.0, float("nan"), 3.0, 99.0, -5.0])
        h_tp, _ = tp.histogram(t2tp(x), 3, range=(1.0, 4.0))
        h_t, _ = torch.histogram(x, 3, range=(1.0, 4.0))
        self.assertParity(h_tp, h_t)
        self.assertEqual(h_t.tolist(), [1.0, 1.0, 1.0])

    def test_empty_input_with_range(self):
        x = torch.tensor([], dtype=torch.float32)
        h_tp, e_tp = tp.histogram(t2tp(x), 3, range=(0.0, 1.0))
        h_t, e_t = torch.histogram(x, 3, range=(0.0, 1.0))
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t)

    def test_empty_range_expansion(self):
        x = torch.tensor([1.0])
        h_tp, e_tp = tp.histogram(t2tp(x), 2, range=(2.0, 2.0))
        h_t, e_t = torch.histogram(x, 2, range=(2.0, 2.0))
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t)

    def test_f64_preserved(self):
        h_tp, e_tp = tp.histogram(t2tp(self.x64), 4)
        h_t, e_t = torch.histogram(self.x64, 4)
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t, rtol=1e-6, atol=1e-9)

    def test_all_nan_without_range_raises(self):
        x = torch.tensor([float("nan"), float("nan")])
        with self.assertRaises(Exception):
            tp.histogram(t2tp(x), 4)
        with self.assertRaises(Exception):
            torch.histogram(x, 4)

    def test_input_validation(self):
        x = t2tp(self.x32)
        with self.assertRaises(Exception):
            tp.histogram(x, 0)                       # bins must be > 0
        with self.assertRaises(Exception):
            tp.histogram(x, t2tp(torch.tensor([[1., 2.], [3., 4.]])))
        with self.assertRaises(Exception):
            tp.histogram(x, t2tp(torch.tensor([1., 2., 3.],
                                              dtype=torch.float64)))
        with self.assertRaises(Exception):
            tp.histogram(x, 3, weight=tp.tensor([1., 2., 3., 4., 5., 6.],
                                                dtype=tp.float64))
        with self.assertRaises(Exception):
            tp.histogram(x, 3, range=(0.0, 1.0, 2.0))
        with self.assertRaises(Exception):
            tp.histogram(x, 3, range=(float("nan"), 1.0))
        with self.assertRaises(Exception):
            tp.histogram(x, 3, range=(2.0, 1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
