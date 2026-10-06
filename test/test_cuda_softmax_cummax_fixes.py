import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp
import tensorplay.functional as F


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


class TestCUDASoftmaxWideOddRows(unittest.TestCase):
    """Rows past the register budget whose length is not a multiple of the
    vector packet width (typical vocabulary sizes) and whose storage starts
    off the 16-byte alignment: the streaming tier must handle them with a
    scalar head and tail instead of dropping to the strided fallback."""

    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def _check(self, width, dtype, tol, rows=6):
        base = np.random.randn(rows, width).astype(np.float32)
        base[:, 0] = -np.inf  # a lone -inf must contribute a zero sum
        base[rows - 1, width - 1] = -np.inf  # tail edge
        base[1, width // 2] = np.nan  # NaN poisons the whole row
        x = tp.tensor(base, device=self.device, dtype=dtype)
        got = _to_numpy(tp.softmax(x, dim=1))
        ref = _numpy_softmax(base, axis=1)
        np.testing.assert_allclose(
            got, ref, atol=tol, rtol=tol, equal_nan=True,
            err_msg=f"softmax width={width} dtype={dtype}")
        got_log = _to_numpy(tp.log_softmax(x, dim=1))
        ref_log = _numpy_log_softmax(base, axis=1)
        np.testing.assert_allclose(
            got_log, ref_log, atol=tol, rtol=tol, equal_nan=True,
            err_msg=f"log_softmax width={width} dtype={dtype}")

    def test_odd_widths_past_register_budget(self):
        # Widths land in the streaming tier with every residue mod the
        # packet width (4 for f32/f64, 8 for f16/bf16).
        self._check(16391, tp.float32, 1e-5)
        self._check(50257, tp.float32, 1e-5)
        self._check(16385, tp.float64, 1e-12)
        self._check(50001, tp.float64, 1e-12)
        self._check(16393, tp.float16, 2e-3)
        self._check(50257, tp.float16, 2e-3)
        self._check(50257, tp.bfloat16, 2e-2)

    def test_misaligned_storage_offset(self):
        # A contiguous slice view starts the first row (and every row after,
        # when the width keeps a remainder) off the packet alignment; the
        # input and output phases then disagree and stores fall back scalar.
        for width, dtype, tol in ((17000, tp.float32, 1e-5),
                                  (50257, tp.float16, 2e-3)):
            flat_np = np.random.randn(6 * width + 7).astype(np.float32)
            flat = tp.tensor(flat_np, device=self.device, dtype=dtype)
            x = flat[7:7 + 6 * width].reshape(6, width)
            base = flat_np[7:].reshape(6, width)
            self.assertTrue(x.is_contiguous())
            got = _to_numpy(tp.softmax(x, dim=1))
            ref = _numpy_softmax(base, axis=1)
            np.testing.assert_allclose(
                got, ref, atol=tol, rtol=tol,
                err_msg=f"softmax offset width={width} dtype={dtype}")


class TestCUDASoftmaxSpatialInnerDims(unittest.TestCase):
    """Slices that sit off the fast dimension: the spatial tier gives each
    inner slice its own y-thread row while an x-team reduces the softmax
    dim, so consecutive threads read consecutive addresses and the inner
    axis contributes parallelism instead of stride gaps."""

    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def _check(self, shape, dim, dtype, tol, poison=False):
        base = np.random.randn(*shape).astype(np.float32)
        if poison:
            flat = base.reshape(-1)
            flat[0] = -np.inf
            flat[flat.size // 2] = np.nan
        x = tp.tensor(base, device=self.device, dtype=dtype)
        got = _to_numpy(tp.softmax(x, dim=dim))
        ref = _numpy_softmax(base, axis=dim)
        np.testing.assert_allclose(
            got, ref, atol=tol, rtol=tol, equal_nan=poison,
            err_msg=f"softmax shape={shape} dim={dim} dtype={dtype}")
        got_log = _to_numpy(tp.log_softmax(x, dim=dim))
        ref_log = _numpy_log_softmax(base, axis=dim)
        np.testing.assert_allclose(
            got_log, ref_log, atol=tol, rtol=tol, equal_nan=poison,
            err_msg=f"log_softmax shape={shape} dim={dim} dtype={dtype}")

    def test_half_spatial_geometries(self):
        # A long dim teams x-threads on the reduction (inner small enough
        # that the block doubles x-threads in), a one-thread-per-slice face
        # whose last inner tile is only partly populated, a small odd face,
        # and a wide channel-last face.
        for shape, dim in (((2, 512, 32, 3), 1), ((2, 8, 2050, 5), 1),
                           ((3, 17, 7), 1), ((2, 16, 1024, 1024), 1)):
            self._check(shape, dim, tp.float16, 2e-3)

    def test_bfloat16_spatial_channel_last(self):
        self._check((2, 64, 256, 256), 1, tp.bfloat16, 2e-2)

    def test_poisoned_rows(self):
        self._check((4, 128, 33), 1, tp.float16, 2e-3, poison=True)


class TestCUDASoftmaxHalfToFloat(unittest.TestCase):
    """The _softmax.out out-variant with half_to_float: fp16 input, fp32
    result.  Every store lands directly in the wider dtype, so the output
    keeps the float accumulator's precision instead of rounding through
    fp16."""

    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def _check(self, shape, dim, tol):
        base = np.random.randn(*shape).astype(np.float32)
        x = tp.tensor(base, device=self.device, dtype=tp.float16)
        out = tp.empty(x.shape, dtype=tp.float32, device=self.device)
        got = F._softmax(x, dim, True, out=out)
        self.assertEqual(got.dtype, tp.float32)
        ref = _numpy_softmax(base, axis=dim)
        np.testing.assert_allclose(
            got.cpu().numpy(), ref, atol=tol, rtol=tol,
            err_msg=f"softmax half_to_float shape={shape} dim={dim}")
        out_log = tp.empty(x.shape, dtype=tp.float32, device=self.device)
        got_log = F._log_softmax(x, dim, True, out=out_log)
        self.assertEqual(got_log.dtype, tp.float32)
        ref_log = _numpy_log_softmax(base, axis=dim)
        np.testing.assert_allclose(
            got_log.cpu().numpy(), ref_log, atol=tol, rtol=tol,
            err_msg=f"log_softmax half_to_float shape={shape} dim={dim}")

    def test_all_tiers(self):
        # Wave-resident rows, register-resident rows, the streaming vocab
        # width and a spatial face (inner > 1) all accept the widened output.
        self._check((33, 512), 1, 2e-3)
        self._check((33, 2048), 1, 2e-3)
        self._check((6, 50257), 1, 2e-3)
        self._check((2, 8, 128, 5), 1, 2e-3)

    def test_requires_half_input(self):
        x = tp.randn(4, 8, device=self.device)
        out = tp.empty((4, 8), dtype=tp.float32, device=self.device)
        with self.assertRaises(RuntimeError):
            F._softmax(x, 1, True, out=out)
        with self.assertRaises(RuntimeError):
            F._log_softmax(x, 1, True, out=out)


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


class TestCUDAMaskedSoftmax(unittest.TestCase):
    """Fused masked softmax: the mask rides along with the row statistics, so
    the result must match the masked-fill rewrite for every mask shape the
    dispatcher accepts, including rows the mask drops entirely."""

    def setUp(self):
        self.device = "cuda" if tp.cuda.is_available() else "cpu"

    def _reference(self, x, mask):
        logits = np.where(mask, -np.inf, np.asarray(x, dtype=np.float64))
        if logits.ndim == 0:
            out = logits
        else:
            out = _numpy_softmax(logits, axis=-1)
        return np.where(mask, 0.0, out)

    def test_fused_matches_masked_fill(self):
        rng = np.random.default_rng(7)
        for shape, dtype, tol in (
                ((3, 16), tp.float32, 1e-6),
                ((2, 4, 8, 32), tp.float32, 1e-6),
                ((2, 4, 8, 32), tp.float16, 2e-3),
                ((2, 4, 8, 32), tp.bfloat16, 1e-2),
                ((2, 3, 7), tp.float64, 1e-12)):
            x = rng.standard_normal(shape)
            mask = rng.random(shape) < 0.3
            mask.reshape(-1)[0] = True
            tx = tp.tensor(x, device=self.device, dtype=dtype)
            tm = tp.tensor(mask, device=self.device)
            got = _to_numpy(F._masked_softmax(tx, tm, -1, 2))
            ref = self._reference(x, mask)
            np.testing.assert_allclose(
                got, ref, atol=tol, rtol=tol,
                err_msg=f"masked_softmax {shape} {dtype}")

    def test_all_dropped_rows_answer_zero(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((4, 8))
        mask = np.zeros((4, 8), dtype=bool)
        mask[1] = True          # one fully dropped row
        mask[3, 0] = True       # one partially dropped row
        tx = tp.tensor(x, device=self.device, dtype=tp.float32)
        tm = tp.tensor(mask, device=self.device)
        got = _to_numpy(F._masked_softmax(tx, tm, -1, 2))
        self.assertTrue(np.all(got[1] == 0.0))
        ref = self._reference(x, mask)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)

    def test_kept_entries_all_neg_inf_answer_nan(self):
        x = np.zeros((2, 6), dtype=np.float32)
        x[1] = -np.inf          # kept entries are all -inf: zero mass, NaN
        mask = np.zeros((2, 6), dtype=bool)
        mask[0, 0] = True
        tx = tp.tensor(x, device=self.device, dtype=tp.float32)
        tm = tp.tensor(mask, device=self.device)
        got = _to_numpy(F._masked_softmax(tx, tm, -1, 2))
        self.assertFalse(np.isnan(got[0]).any())
        self.assertTrue(np.isnan(got[1]).all())

    def test_long_rows_take_the_unfused_rewrite(self):
        rng = np.random.default_rng(13)
        shape = (2, 2048)       # past the fused tier's row-length bound
        x = rng.standard_normal(shape)
        mask = rng.random(shape) < 0.4
        tx = tp.tensor(x, device=self.device, dtype=tp.float32)
        tm = tp.tensor(mask, device=self.device)
        got = _to_numpy(F._masked_softmax(tx, tm, -1, 2))
        ref = self._reference(x, mask)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)

    def test_padding_mask_broadcasts_over_heads(self):
        rng = np.random.default_rng(17)
        b, h, l = 2, 4, 6
        x = rng.standard_normal((b, h, l, l))
        mask = rng.random((b, l)) < 0.3
        tx = tp.tensor(x, device=self.device, dtype=tp.float32)
        tm = tp.tensor(mask, device=self.device)
        got = _to_numpy(F._masked_softmax(tx, tm, -1, 1))
        ref = self._reference(x, mask.reshape(b, 1, 1, l))
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)

    def test_backward_matches_operand_masking(self):
        rng = np.random.default_rng(19)
        for dtype in (tp.float32, tp.float16):
            x = rng.standard_normal((2, 4, 16))
            mask = rng.random((2, 4, 16)) < 0.3
            mask[0, 0] = True      # one fully dropped row
            tx = tp.tensor(x, device=self.device, dtype=dtype)
            tm = tp.tensor(mask, device=self.device)
            out = F._masked_softmax(tx, tm, -1, 2)
            g = rng.standard_normal((2, 4, 16))
            tg = tp.tensor(g, device=self.device, dtype=dtype)
            got = _to_numpy(F._masked_softmax_backward(tg, out, tm, -1))
            x64 = x.astype(np.float64)
            out64 = out.cpu().float().numpy().astype(np.float64)
            g64 = g.astype(np.float64)
            g64 = np.where(mask, 0.0, g64)
            o64 = np.where(mask, 0.0, out64)
            dot = (g64 * o64).sum(axis=-1, keepdims=True)
            ref = np.where(mask, 0.0, o64 * (g64 - dot))
            tol = 1e-6 if dtype == tp.float32 else 2e-3
            np.testing.assert_allclose(
                got, ref, atol=tol, rtol=tol,
                err_msg=f"masked_softmax_backward {dtype}")

    def test_rejects_non_bool_mask(self):
        tx = tp.tensor(np.zeros((2, 4), np.float32), device=self.device)
        tm = tp.tensor(np.zeros((2, 4), np.int64), device=self.device)
        with self.assertRaises(RuntimeError):
            F._masked_softmax(tx, tm, -1, 2)


if __name__ == "__main__":
    unittest.main()
