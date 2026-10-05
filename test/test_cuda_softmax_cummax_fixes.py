import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp


def _numpy_softmax(x, axis):
    """Float64 softmax along an axis as an independent reference."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def _numpy_log_softmax(x, axis):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _numpy_cum_extremum_indices(x, dim, is_min=False):
    """Index of the running extremum: equal values move the index to the
    latest position attaining it, a NaN wins from its first position, and
    ordinary positions after a NaN hold that first NaN's index."""
    x = np.asarray(x)
    accumulate = np.minimum.accumulate if is_min else np.maximum.accumulate
    values = accumulate(x, axis=dim)
    seen_nan = np.maximum.accumulate(np.isnan(x).astype(np.int64), axis=dim)
    first_nan = np.argmax(seen_nan == 1, axis=dim)
    pos = np.arange(x.shape[dim]).reshape(
        [-1 if ax == dim else 1 for ax in range(x.ndim)])
    attained = np.where((x == values) | np.isnan(x), pos, np.int64(-1))
    last_attained = np.maximum.accumulate(attained, axis=dim)
    nan_index = np.where(np.isnan(x), pos, first_nan)
    return np.where(seen_nan > 0, nan_index, last_attained).astype(np.int64)


def _to_numpy(t):
    """Device tensor to numpy; 64-bit outputs skip the float32 hop the
    narrow dtypes need, which would otherwise destroy their tolerance."""
    if t.dtype == tp.float64:
        return t.cpu().numpy()
    return t.cpu().float().numpy()


class TestCUDASoftmaxEntryGuards(unittest.TestCase):
    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def test_softmax_noncontiguous_forward(self):
        # A transposed view puts a stride inside the softmax dimension; the
        # row kernels must not address it with a contiguous-layout index.
        for dtype, tol in ((tp.float32, 1e-5), (tp.float64, 1e-12),
                           (tp.float16, 2e-3), (tp.bfloat16, 2e-2)):
            base = np.random.randn(6, 8, 10).astype(np.float32)
            x = tp.tensor(base, device=self.device, dtype=dtype)
            view = x.permute(2, 0, 1)  # softmax dim (1) is strided
            self.assertFalse(view.is_contiguous())
            got = tp.softmax(view, dim=1)
            ref = _numpy_softmax(base.transpose(2, 0, 1), axis=1)
            np.testing.assert_allclose(
                _to_numpy(got), ref, atol=tol, rtol=tol,
                err_msg=f"softmax dtype={dtype}")
            got_log = tp.log_softmax(view, dim=1)
            ref_log = _numpy_log_softmax(base.transpose(2, 0, 1), axis=1)
            np.testing.assert_allclose(
                _to_numpy(got_log), ref_log, atol=tol, rtol=tol,
                err_msg=f"log_softmax dtype={dtype}")

    def test_softmax_inner_dim_noncontiguous(self):
        base = np.random.randn(4, 7, 5).astype(np.float32)
        x = tp.tensor(base, device=self.device)
        view = x.permute(0, 2, 1)  # softmax dim (2) is the fastest one
        got = tp.softmax(view, dim=2)
        ref = _numpy_softmax(base.transpose(0, 2, 1), axis=2)
        np.testing.assert_allclose(got.cpu().numpy(), ref, atol=1e-5, rtol=1e-5)

    def test_softmax_empty_tensor(self):
        for shape, dim in (((0, 8), 1), ((8, 0), 0), ((2, 0, 3), 1)):
            x = tp.empty(shape, device=self.device)
            out = tp.softmax(x, dim=dim)
            self.assertEqual(tuple(out.shape), shape)
            out_log = tp.log_softmax(x, dim=dim)
            self.assertEqual(tuple(out_log.shape), shape)

    def test_softmax_dim_out_of_range(self):
        x = tp.randn(4, 4, device=self.device)
        for dim in (2, -3, 7):
            with self.assertRaises(RuntimeError):
                tp.softmax(x, dim=dim)
            with self.assertRaises(RuntimeError):
                tp.log_softmax(x, dim=dim)

    def test_softmax_zero_dim(self):
        x = tp.tensor(2.0, device=self.device)
        out = tp.softmax(x, dim=0)
        self.assertAlmostEqual(float(out), 1.0, places=6)
        out_log = tp.log_softmax(x, dim=0)
        self.assertAlmostEqual(float(out_log), 0.0, places=6)
        out_neg = tp.softmax(x, dim=-1)
        self.assertAlmostEqual(float(out_neg), 1.0, places=6)


class TestCUDASoftmaxWaveWidths(unittest.TestCase):
    """Row widths that exercise the shuffle-reduction tier (513..1024, where
    one lane holds more than sixteen elements) and the register-resident and
    block tiers above it."""

    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def _check(self, widths, dtype, tol):
        for width in widths:
            base = np.random.randn(33, width).astype(np.float32)
            x = tp.tensor(base, device=self.device, dtype=dtype)
            got = _to_numpy(tp.softmax(x, dim=1))
            got_log = _to_numpy(tp.log_softmax(x, dim=1))
            ref = _numpy_softmax(base, axis=1)
            np.testing.assert_allclose(
                got, ref, atol=tol, rtol=tol,
                err_msg=f"softmax width={width} dtype={dtype}")
            ref_log = _numpy_log_softmax(base, axis=1)
            np.testing.assert_allclose(
                got_log, ref_log, atol=tol, rtol=tol,
                err_msg=f"log_softmax width={width} dtype={dtype}")

    def test_float32_full_wave_band(self):
        self._check([512, 600, 1000, 1024, 1999, 2048, 2049, 4096],
                    tp.float32, 1e-5)

    def test_half_full_wave_band(self):
        self._check([512, 1000, 1024, 2048], tp.float16, 2e-3)

    def test_bfloat16_full_wave_band(self):
        self._check([512, 1000, 1024, 2048], tp.bfloat16, 2e-2)

    def test_float64_wave_cap(self):
        # 64-bit accumulators halve the wave row cap: 1024 stays on the
        # shuffle path, 1025 moves to the block kernels.
        self._check([512, 1000, 1024, 1025, 2048], tp.float64, 1e-12)


class TestCUDACummaxExtremum(unittest.TestCase):
    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def test_cummax_int64_large_values_outer_dim(self):
        # Values past 2**53 exceed double precision; the scan must compare
        # them in their own type.
        big = 1 << 60
        base = np.array([[big + 5, big + 3],
                         [big + 1, big + 9],
                         [big + 7, big + 4]], dtype=np.int64)
        x = tp.tensor(base, device=self.device)
        values, indices = tp.cummax(x, dim=0)
        np.testing.assert_array_equal(
            values.cpu().numpy(), np.maximum.accumulate(base, axis=0))
        np.testing.assert_array_equal(
            indices.cpu().numpy(),
            np.array([[0, 0], [0, 1], [2, 1]], dtype=np.int64))

    def test_cummin_int64_large_values_outer_dim(self):
        big = 1 << 60
        base = np.array([[big - 5, big - 3],
                         [big - 1, big - 9],
                         [big - 7, big - 4]], dtype=np.int64)
        x = tp.tensor(base, device=self.device)
        values, indices = tp.cummin(x, dim=0)
        np.testing.assert_array_equal(
            values.cpu().numpy(), np.minimum.accumulate(base, axis=0))
        np.testing.assert_array_equal(
            indices.cpu().numpy(),
            np.array([[0, 0], [0, 1], [2, 1]], dtype=np.int64))

    def test_cummax_tie_takes_last_occurrence(self):
        # Equal values move the running index to the latest position that
        # attains the extremum; both scan directions share the discipline.
        base = np.array([[1.0, 3.0], [1.0, 2.0]], dtype=np.float32)
        x = tp.tensor(base, device=self.device)
        for dim in (0, 1):
            values, indices = tp.cummax(x, dim=dim)
            np.testing.assert_array_equal(
                values.cpu().numpy(), np.maximum.accumulate(base, axis=dim))
            np.testing.assert_array_equal(
                indices.cpu().numpy(), _numpy_cum_extremum_indices(base, dim))
            vmin, imin = tp.cummin(x, dim=dim)
            np.testing.assert_array_equal(
                vmin.cpu().numpy(), np.minimum.accumulate(base, axis=dim))
            np.testing.assert_array_equal(
                imin.cpu().numpy(),
                _numpy_cum_extremum_indices(base, dim, is_min=True))

    def test_cummax_nan_priority(self):
        base = np.array([[1.0, np.nan], [5.0, 2.0]], dtype=np.float32)
        x = tp.tensor(base, device=self.device)
        values, indices = tp.cummax(x, dim=0)
        got = values.cpu().numpy()
        self.assertTrue(np.isnan(got[0, 1]))
        # A NaN beats every ordinary value and the scan never moves off it:
        # the trailing 2.0 stays NaN with the index frozen at the first NaN.
        self.assertTrue(np.isnan(got[1, 1]))
        self.assertEqual(indices.cpu().numpy()[1, 1], 0)


if __name__ == "__main__":
    unittest.main()
