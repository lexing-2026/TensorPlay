"""CUDA parity tests for the native quantile / nanquantile / histogram batch.

Same device-generic composite bodies as the CPU tests
(test_sorting_histogram.py) run on the CUDA backend via the shared
registration in MiscKernels.cpp.  Skipped when no GPU is present (local
CPU-only builds); on the GPU box, import torch before tensorplay (cudart
12.8-vs-12.4 coexistence, see .remote_build.md).
"""
import unittest

import numpy as np

import torch  # must precede tensorplay on the GPU box

import tensorplay as tp

DEVICES = ["cuda"] if torch.cuda.is_available() else []


def t2tp(t, dev):
    return tp.tensor(t.detach().cpu().numpy()).to(dev)


@unittest.skipUnless(DEVICES, "no CUDA device available")
class TestSortingHistogramCuda(unittest.TestCase):
    def setUp(self):
        self.dev = "cuda"
        g = torch.Generator().manual_seed(7)
        self.x32 = torch.randn(64, generator=g)
        self.x64 = torch.randn(64, dtype=torch.float64, generator=g)
        self.x2d = torch.randn(8, 16, generator=g)

    def assertParity(self, out_tp, out_torch, rtol=1e-6, atol=1e-7):
        self.assertEqual(list(out_tp.shape), list(out_torch.shape))
        a = out_tp.cpu().numpy()
        b = out_torch.detach().cpu().numpy()
        if a.dtype.kind == "f":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            self.assertTrue((nan_a == nan_b).all(), "NaN placement mismatch")
            a, b = a[~nan_a], b[~nan_b]
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    def test_quantile_interpolations(self):
        for interp in ("linear", "lower", "higher", "midpoint", "nearest"):
            x32, x64 = self.x32.to(self.dev), self.x64.to(self.dev)
            self.assertParity(
                tp.quantile(t2tp(x32, self.dev), 0.3, interpolation=interp),
                torch.quantile(x32, 0.3, interpolation=interp),
            )
            self.assertParity(
                tp.quantile(t2tp(x64, self.dev), 0.3, interpolation=interp),
                torch.quantile(x64, 0.3, interpolation=interp),
            )

    def test_quantile_dim_keepdim(self):
        q = torch.tensor([0.1, 0.5, 0.9]).to(self.dev)
        x = self.x2d.to(self.dev)
        for dim in (None, 0, 1, -1):
            for keepdim in (False, True):
                self.assertParity(
                    tp.quantile(t2tp(x, self.dev), t2tp(q, self.dev),
                                dim=dim, keepdim=keepdim),
                    torch.quantile(x, q, dim=dim, keepdim=keepdim),
                )

    def test_nanquantile(self):
        x = torch.tensor([1.0, float("nan"), 3.0, 2.0]).to(self.dev)
        self.assertParity(
            tp.nanquantile(t2tp(x, self.dev), 0.5),
            torch.nanquantile(x, 0.5),
        )

    def test_histogram_bins(self):
        x = self.x32.to(self.dev)
        h_tp, e_tp = tp.histogram(t2tp(x, self.dev), 7)
        # torch.histogram is CPU-only in ATen; take the reference there.
        h_t, e_t = torch.histogram(self.x32, 7)
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t)

    def test_histogram_range_weight_density(self):
        x = self.x32.to(self.dev)
        w = torch.rand(64, generator=torch.Generator().manual_seed(11))
        h_tp, e_tp = tp.histogram(t2tp(x, self.dev), 6,
                                  range=(float(self.x32.min()),
                                         float(self.x32.max())),
                                  weight=t2tp(w, self.dev), density=True)
        h_t, e_t = torch.histogram(self.x32, 6,
                                   range=(float(self.x32.min()),
                                          float(self.x32.max())),
                                   weight=w, density=True)
        self.assertParity(h_tp, h_t, rtol=1e-5, atol=1e-7)
        self.assertParity(e_tp, e_t)

    def test_histogram_bins_tensor(self):
        x = self.x32.to(self.dev)
        edges = torch.linspace(float(self.x32.min()), float(self.x32.max()),
                               9)
        h_tp, e_tp = tp.histogram(t2tp(x, self.dev),
                                  t2tp(edges, self.dev))
        h_t, e_t = torch.histogram(self.x32, edges)
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t)

    def test_histogram_empty_input_with_range(self):
        # Regression: empty input + explicit range used to segfault in the
        # histogram composite (zero-element searchsorted/index_add path).
        x = torch.empty(0)
        h_tp, e_tp = tp.histogram(t2tp(x, self.dev), 3, range=(0.0, 1.0))
        h_t, e_t = torch.histogram(x, 3, range=(0.0, 1.0))
        self.assertParity(h_tp, h_t)
        self.assertParity(e_tp, e_t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
