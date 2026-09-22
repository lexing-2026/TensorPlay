"""CPU col2im / fold kernel tests.

Covers the fast paths (2x2 tiles, pointwise, non-overlapping tiles), the
general padded/strided/dilated path, batch flattening, thread counts,
non-contiguous inputs and signed-zero preservation.  Expected values are built
with an independent scatter-add reference in the tested dtype.
"""

import unittest

import numpy as np

import tensorplay as tp
import tensorplay.nn.functional as F

# (batch, channels, (h, w), kernel, stride, padding, dilation)
CASES = [
    (None, 2, (8, 10), (2, 2), (2, 2), (0, 0), (1, 1)),
    (2, 3, (128, 130), (2, 2), (2, 2), (0, 0), (1, 1)),
    (1, 4, (96, 99), (2, 3), (2, 3), (0, 0), (1, 1)),
    (2, 3, (127, 129), (1, 1), (1, 1), (0, 0), (1, 1)),
    (2, 3, (63, 65), (3, 3), (1, 1), (1, 1), (1, 1)),
    (2, 3, (65, 67), (2, 2), (2, 2), (0, 0), (1, 1)),
    (2, 3, (64, 66), (2, 2), (3, 3), (0, 0), (1, 1)),
    (2, 3, (63, 65), (2, 3), (1, 2), (2, 1), (2, 3)),
    (8, 1, (64, 66), (3, 3), (1, 1), (1, 1), (1, 1)),
    (2, 1, (128, 128), (1, 1), (1, 1), (0, 0), (1, 1)),
    (2, 1, (128, 129), (1, 1), (1, 1), (0, 0), (1, 1)),
    (0, 2, (8, 10), (2, 2), (2, 2), (0, 0), (1, 1)),
    (2, 1, (2, 3), (2, 3), (2, 3), (5, 7), (3, 2)),
    (1, 4, (4, 5), (3, 3), (1, 1), (1, 512), (1, 1)),
]


def _column_count(case):
    batch, channels, size, kernel, stride, padding, dilation = case
    n = 0 if batch is None else batch
    h, w = size
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return n, channels, (kh, kw), (oh, ow)


def _reference(case, np_dtype, seed=0):
    """Independent scatter-add reference; returns expected array + int source."""
    batch = case[0]
    n, channels, (kh, kw), (oh, ow) = _column_count(case)
    (h, w) = case[2]
    ph, pw = case[5]
    sh, sw = case[4]
    dh, dw = case[6]
    rng = np.random.default_rng(seed)
    if batch is None:
        ints = rng.integers(-2, 3, size=(channels * kh * kw, oh * ow))
        expected = np.zeros((channels, h * w), dtype=np_dtype)
        flat = ints.reshape(channels, kh * kw * oh * ow)
    else:
        ints = rng.integers(-2, 3, size=(n, channels * kh * kw, oh * ow))
        expected = np.zeros((n, channels, h * w), dtype=np_dtype)
        flat = ints.reshape(n, channels, kh * kw * oh * ow)
    rows = np.arange(kh)[:, None] * dh + np.arange(oh)[None, :] * sh - ph
    cols_ = np.arange(kw)[:, None] * dw + np.arange(ow)[None, :] * sw - pw
    rows, cols_ = rows[:, None, :, None], cols_[None, :, None, :]
    valid = ((rows >= 0) & (rows < h) & (cols_ >= 0) & (cols_ < w)).flatten()
    indices = (rows * w + cols_).flatten()[valid]
    flat = flat[..., valid]
    if batch is None:
        for c in range(channels):
            np.add.at(expected[c], indices, flat[c])
        return expected.reshape(channels, h, w), ints
    for n_i in range(n):
        for c in range(channels):
            np.add.at(expected[n_i, c], indices, flat[n_i, c])
    return expected.reshape(n, channels, h, w), ints


class TestCol2ImCpu(unittest.TestCase):
    def _run(self, case, np_dtype, threads=1, seed=0):
        expected, ints = _reference(case, np_dtype, seed)
        cols = tp.tensor(ints.astype(np_dtype))
        old = tp.get_num_threads()
        tp.set_num_threads(threads)
        try:
            got = F.fold(cols, output_size=case[2], kernel_size=case[3],
                 dilation=case[6], padding=case[5], stride=case[4])
        finally:
            tp.set_num_threads(old)
        got_np = np.array(got.numpy()).astype(np_dtype)
        maxdiff = float(np.abs(got_np - expected).max(initial=0.0))
        self.assertTrue(
            np.array_equal(got_np, expected),
            f"case={case} dtype={np_dtype} threads={threads} maxdiff={maxdiff:.3g}",
        )

    def test_fold_reference_cases(self):
        for case in CASES:
            for np_dtype in [np.float32, np.float64]:
                self._run(case, np_dtype)

    def test_fold_half_precision(self):
        for case in CASES[:6]:
            self._run(case, np.float16)

    def test_fold_bfloat16(self):
        # bf16 has no numpy counterpart; compare against the exact fp32
        # reference (small integers stay exact in bf16).
        for case in CASES[:5]:
            expected, ints = _reference(case, np.float32)
            cols = tp.tensor(ints.astype(np.float32)).to(tp.bfloat16)
            got = F.fold(cols, output_size=case[2], kernel_size=case[3],
                 dilation=case[6], padding=case[5], stride=case[4])
            got_np = np.array(got.float().numpy())
            self.assertTrue(
                np.array_equal(got_np, expected),
                f"case={case} maxdiff={float(np.abs(got_np - expected).max()):.3g}",
            )

    def test_fold_threads(self):
        for case in CASES[:4]:
            for threads in (1, 4):
                self._run(case, np.float32, threads=threads)

    def test_fold_noncontiguous_input(self):
        case = CASES[0]
        expected, ints = _reference(case, np.float32)
        cols = tp.tensor(ints.astype(np.float32))
        cols_nc = cols.transpose(-1, -2).contiguous().transpose(-1, -2)
        got = F.fold(cols_nc, output_size=case[2], kernel_size=case[3],
                 dilation=case[6], padding=case[5], stride=case[4])
        self.assertTrue(np.array_equal(np.array(got.numpy()), expected))

    def test_fold_out(self):
        case = CASES[6]
        expected, ints = _reference(case, np.float32)
        n, channels, (kh, kw), (oh, ow) = _column_count(case)
        cols = tp.tensor(ints.astype(np.float32))
        out = tp.zeros((n, channels, case[2][0], case[2][1]))
        tp.col2im(cols, case[2], case[3], case[6], case[5], case[4], out=out)
        self.assertTrue(np.array_equal(np.array(out.numpy()), expected))

    def test_fold_signed_zero(self):
        for kernel in [(1, 1), (2, 2), (2, 3)]:
            kh, kw = kernel
            ints = np.zeros((2, 4 * kh * kw, (96 // kh) * (96 // kw)),
                            dtype=np.float32)
            ints[...] = -0.0
            got = F.fold(tp.tensor(ints), (96, 96), kernel, stride=kernel)
            self.assertTrue(np.all(np.array(got.numpy()) == 0.0))
            self.assertFalse(np.signbit(np.array(got.numpy())).any())

    def test_fold_output_storage(self):
        case = CASES[1]
        expected, ints = _reference(case, np.float32)
        cols = tp.tensor(ints.astype(np.float32))
        got = F.fold(cols, output_size=case[2], kernel_size=case[3],
                 dilation=case[6], padding=case[5], stride=case[4])
        self.assertEqual(
            got.untyped_storage().nbytes(),
            got.numel() * got.element_size(),
        )


if __name__ == "__main__":
    unittest.main()
