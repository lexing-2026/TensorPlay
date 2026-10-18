"""CUDA searchsorted / bucketize against the reference runtime.

The CUDA search fills the caller's output buffer in place; the cases here pin
the direct write across sides, output widths, query shapes, the sorter
reindexing path, and output buffers with non-standard strides.
"""

import unittest

import numpy as np
import torch

import tensorplay as tp


def make_tp_tensor(torch_tensor):
    dtype = {
        torch.float16: tp.float16,
        torch.float32: tp.float32,
        torch.float64: tp.float64,
        torch.int32: tp.int32,
        torch.int64: tp.int64,
    }[torch_tensor.dtype]
    return tp.tensor(torch_tensor.cpu().tolist(), dtype=dtype).to("cuda")


class TestCUDASearchsorted(unittest.TestCase):
    def test_large_left_right_int32(self):
        g = torch.Generator(device="cuda").manual_seed(3)
        seq = torch.rand(1 << 17, generator=g, device="cuda").sort().values
        vals = torch.rand(1 << 17, generator=g, device="cuda")
        tseq, tvals = make_tp_tensor(seq), make_tp_tensor(vals)
        for kwargs in ({}, {"side": "right"}, {"out_int32": True},
                       {"right": True, "out_int32": True}):
            got = tp.searchsorted(tseq, tvals, **kwargs)
            want = torch.searchsorted(seq, vals, **kwargs)
            self.assertEqual(
                got.dtype, tp.int32 if kwargs.get("out_int32") else tp.int64)
            np.testing.assert_array_equal(got.cpu().numpy(),
                                          want.cpu().numpy())

    def test_queries_against_row_boundaries(self):
        g = torch.Generator(device="cuda").manual_seed(5)
        seq = torch.sort(torch.rand(4, 100, generator=g, device="cuda"),
                         dim=-1).values
        vals = torch.randn(4, 17, generator=g, device="cuda")
        got = tp.searchsorted(make_tp_tensor(seq), make_tp_tensor(vals))
        want = torch.searchsorted(seq, vals)
        np.testing.assert_array_equal(got.cpu().numpy(), want.cpu().numpy())

    def test_sorter_reindexing(self):
        g = torch.Generator(device="cuda").manual_seed(7)
        unsorted = torch.randn(4096, generator=g, device="cuda")
        sorter = torch.argsort(unsorted)
        vals = torch.randn(1000, generator=g, device="cuda")
        got = tp.searchsorted(make_tp_tensor(unsorted), make_tp_tensor(vals),
                              sorter=make_tp_tensor(sorter))
        want = torch.searchsorted(unsorted, vals, sorter=sorter)
        np.testing.assert_array_equal(got.cpu().numpy(), want.cpu().numpy())

    def test_out_buffers(self):
        seq = tp.tensor([1.0, 3.0, 5.0, 7.0], device="cuda")
        vals = tp.tensor([2.0, 5.0], device="cuda")
        ref_seq = torch.tensor([1.0, 3.0, 5.0, 7.0], device="cuda")
        ref_vals = torch.tensor([2.0, 5.0], device="cuda")

        out64 = tp.zeros(2, dtype=tp.int64, device="cuda")
        tp.searchsorted(seq, vals, out=out64)
        np.testing.assert_array_equal(
            out64.cpu().numpy(),
            torch.searchsorted(ref_seq, ref_vals).cpu().numpy())

        out32 = tp.zeros(2, dtype=tp.int32, device="cuda")
        tp.searchsorted(seq, vals, out_int32=True, out=out32)
        np.testing.assert_array_equal(
            out32.cpu().numpy(),
            torch.searchsorted(ref_seq, ref_vals, out_int32=True).cpu().numpy())

        # A transposed view has swapped strides; the result must still land
        # in the right cells of the base buffer.
        base = tp.zeros(2, 2, dtype=tp.int64, device="cuda")
        view = base.t()
        vals2d = tp.tensor([[2.0, 5.0], [0.0, 9.0]], device="cuda")
        tp.searchsorted(seq, vals2d, out=view)
        # view[i, j] == base[j, i]; the positions are [[1, 2], [0, 4]].
        np.testing.assert_array_equal(base.cpu().numpy(),
                                      np.array([[1, 0], [2, 4]], dtype=np.int64))

    def test_bucketize_matches_reference(self):
        g = torch.Generator(device="cuda").manual_seed(11)
        boundaries = torch.sort(torch.randn(512, generator=g, device="cuda")).values
        vals = torch.randn(1000, generator=g, device="cuda")
        got = tp.bucketize(make_tp_tensor(vals), make_tp_tensor(boundaries))
        want = torch.bucketize(vals, boundaries)
        np.testing.assert_array_equal(got.cpu().numpy(), want.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
