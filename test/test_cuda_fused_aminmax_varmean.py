import unittest

import numpy as np

import tensorplay as tp
import tensorplay.functional as F


def _to_numpy(t):
    t = t.cpu()
    if t.dtype in (tp.float16, tp.bfloat16):
        t = t.float()
    return t.numpy()


class TestCUDAAminmaxFused(unittest.TestCase):
    """aminmax returns (min, max) from a single pass; the packed float form
    encodes both keys in one word, so NaN priority and ±0 folding live in the
    encoding."""

    def _check_pair(self, x, dim, keepdim, np_axis):
        lo, hi = F.aminmax(x, dim, keepdim=keepdim)
        npx = _to_numpy(x)
        np.testing.assert_allclose(
            _to_numpy(lo), npx.min(axis=np_axis, keepdims=keepdim), rtol=0, atol=0)
        np.testing.assert_allclose(
            _to_numpy(hi), npx.max(axis=np_axis, keepdims=keepdim), rtol=0, atol=0)

    def test_float32_dim_reductions(self):
        x = tp.randn((4096, 512), dtype=tp.float32, device="cuda")
        self._check_pair(x, -1, False, 1)
        self._check_pair(x, -1, True, 1)
        self._check_pair(x, 0, False, 0)

    def test_half_family_roundtrip(self):
        x = (tp.randn((256, 1024), dtype=tp.float32, device="cuda") * 4).to(tp.float16)
        self._check_pair(x, -1, False, 1)
        xb = (tp.randn((256, 1024), dtype=tp.float32, device="cuda") * 4).to(tp.bfloat16)
        self._check_pair(xb, -1, False, 1)

    def test_nan_claims_both_outputs(self):
        x = tp.tensor([[1.0, float("nan"), 3.0], [-2.0, 5.0, 0.0]],
                      dtype=tp.float32, device="cuda")
        lo, hi = F.aminmax(x, -1)
        self.assertTrue(np.isnan(_to_numpy(lo)[0]))
        self.assertTrue(np.isnan(_to_numpy(hi)[0]))
        self.assertEqual(_to_numpy(lo)[1], -2.0)
        self.assertEqual(_to_numpy(hi)[1], 5.0)

    def test_signed_zero_folds(self):
        x = tp.tensor([-0.0, 0.0, -0.0], dtype=tp.float32, device="cuda")
        lo, hi = F.aminmax(x, 0)
        self.assertEqual(_to_numpy(lo).item(), 0.0)
        self.assertEqual(_to_numpy(hi).item(), 0.0)

    def test_int64_fallback_large_values(self):
        # values beyond float32 mantissa precision exercise the non-packed path
        big = 1 << 40
        x = tp.tensor([[big + 3, big - 7], [big + 11, big + 1]],
                      dtype=tp.int64, device="cuda")
        lo, hi = F.aminmax(x, -1)
        self.assertEqual(_to_numpy(lo).tolist(), [big - 7, big + 1])
        self.assertEqual(_to_numpy(hi).tolist(), [big + 3, big + 11])

    def test_global_reduction(self):
        x = tp.randn((257, 129), dtype=tp.float32, device="cuda")
        lo, hi = F.aminmax(x)
        npx = _to_numpy(x)
        self.assertEqual(_to_numpy(lo).item(), npx.min())
        self.assertEqual(_to_numpy(hi).item(), npx.max())


class TestCUDAVarMeanCorrection(unittest.TestCase):
    """The correction spelling routes integer corrections through the fused
    Welford pass; results must match the definition (ddof = correction)."""

    def _check(self, x, dim, correction, keepdim):
        var, mean = F.var_mean(x, dim, correction=correction, keepdim=keepdim)
        npx = _to_numpy(x)
        axis = dim[0] if dim[0] >= 0 else x.dim() + dim[0]
        np.testing.assert_allclose(
            _to_numpy(var), npx.var(axis=axis, ddof=correction,
                                    keepdims=keepdim), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(
            _to_numpy(mean), npx.mean(axis=axis, keepdims=keepdim),
            rtol=1e-5, atol=1e-7)

    def test_dim_correction_default_and_zero(self):
        x = tp.randn((512, 1024), dtype=tp.float32, device="cuda")
        self._check(x, [-1], 1, False)
        self._check(x, [-1], 0, False)
        self._check(x, [-1], 1, True)
        self._check(x, [0], 1, False)

    def test_float64(self):
        x = tp.randn((128, 256), dtype=tp.float64, device="cuda")
        self._check(x, [-1], 1, False)
        self._check(x, [-1], 0, False)

    def test_global(self):
        x = tp.randn((97, 131), dtype=tp.float32, device="cuda")
        var, mean = F.var_mean(x)
        npx = _to_numpy(x)
        np.testing.assert_allclose(_to_numpy(var), npx.var(ddof=1), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(_to_numpy(mean), npx.mean(), rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
