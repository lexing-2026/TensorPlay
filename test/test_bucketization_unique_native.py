"""

Native wiring for the searchsorted / bucketize family (all overloads: the
tensor and scalar query variants, the out variants, the side alias, the
sorter reindexing path) and the unique family (flat two- and three-output
forms, dim-wise unique with inverse/counts, and the consecutive dim form),
comparing against the reference runtime for values, dtypes, and error
contracts.
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


class SearchSorted(unittest.TestCase):
    def test_1d_lower_and_upper(self):
        seq = tp.tensor([1, 3, 5, 7, 9])
        vals = tp.tensor([[3, 6, 9], [3, 6, 9]])
        ref = torch.tensor([1, 3, 5, 7, 9])
        rv = torch.tensor([[3, 6, 9], [3, 6, 9]])
        self.assertTrue(close(tp.searchsorted(seq, vals), torch.searchsorted(ref, rv)))
        self.assertTrue(close(tp.searchsorted(seq, vals, right=True),
                              torch.searchsorted(ref, rv, right=True)))

    def test_side_alias(self):
        seq = tp.tensor([1, 3, 5, 7, 9])
        ref = torch.tensor([1, 3, 5, 7, 9])
        for side, right in (("left", False), ("right", True)):
            out = tp.searchsorted(seq, tp.tensor([0, 5, 10]), side=side)
            expect = torch.searchsorted(ref, torch.tensor([0, 5, 10]), side=side)
            self.assertTrue(close(out, expect))
            self.assertTrue(close(out, tp.searchsorted(seq, tp.tensor([0, 5, 10]), right=right)))

    def test_nd_boundaries(self):
        seq = tp.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        vals = tp.tensor([[3, 6, 9], [3, 6, 9]])
        ref = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        rv = torch.tensor([[3, 6, 9], [3, 6, 9]])
        self.assertTrue(close(tp.searchsorted(seq, vals), torch.searchsorted(ref, rv)))
        self.assertTrue(close(tp.searchsorted(seq, vals, right=True),
                              torch.searchsorted(ref, rv, right=True)))

    def test_scalar_query(self):
        seq = tp.tensor([1, 3, 5])
        ref = torch.tensor([1, 3, 5])
        for v, right in ((3, False), (3, True), (4, False), (0, True)):
            self.assertEqual(tp.searchsorted(seq, v, right=right).item(),
                             torch.searchsorted(ref, v, right=right).item())
        # A float scalar query promotes against integer boundaries.
        self.assertEqual(tp.searchsorted(seq, 4.0).item(),
                         torch.searchsorted(ref, 4.0).item())

    def test_scalar_bucketize(self):
        boundaries = tp.tensor([1, 2, 3])
        ref = torch.tensor([1, 2, 3])
        for v in (0.5, 2.5, 3.0):
            self.assertEqual(tp.bucketize(v, boundaries).item(),
                             torch.bucketize(torch.tensor(v), ref).item())
            self.assertEqual(tp.bucketize(v, boundaries, right=True).item(),
                             torch.bucketize(torch.tensor(v), ref, right=True).item())

    def test_nan_query_lands_at_end(self):
        seq = tp.tensor([1.0, 3.0, float("nan")])
        ref = torch.tensor([1.0, 3.0, float("nan")])
        nan = float("nan")
        self.assertEqual(tp.searchsorted(seq, nan).item(),
                         torch.searchsorted(ref, nan).item())
        self.assertEqual(tp.searchsorted(seq, nan, right=True).item(),
                         torch.searchsorted(ref, nan, right=True).item())

    def test_sorter(self):
        unsorted = tp.tensor([30, 10, 20])
        ref = torch.tensor([30, 10, 20])
        sorter = tp.argsort(unsorted)
        out = tp.searchsorted(unsorted, tp.tensor([15, 25, 35]), sorter=sorter)
        expect = torch.searchsorted(ref, torch.tensor([15, 25, 35]),
                                    sorter=torch.argsort(ref))
        self.assertTrue(close(out, expect))

    def test_out_variants(self):
        seq = tp.tensor([1, 3, 5])
        vals = tp.tensor([2, 4])
        out64 = tp.empty(2, dtype=tp.int64)
        tp.searchsorted(seq, vals, out=out64)
        self.assertTrue(close(out64, torch.searchsorted(
            torch.tensor([1, 3, 5]), torch.tensor([2, 4]))))
        out32 = tp.empty(2, dtype=tp.int32)
        tp.searchsorted(seq, vals, out_int32=True, out=out32)
        self.assertEqual(out32.dtype, tp.int32)
        bout = tp.empty(2, dtype=tp.int64)
        tp.bucketize(tp.tensor([2.5, 0.5]), seq, out=bout)
        self.assertTrue(close(bout, torch.bucketize(
            torch.tensor([2.5, 0.5]), torch.tensor([1, 3, 5]))))

    def test_out_int32(self):
        seq = tp.tensor([1, 3, 5])
        out = tp.searchsorted(seq, tp.tensor([2, 6]), out_int32=True)
        self.assertEqual(out.dtype, tp.int32)
        self.assertTrue(close(out, torch.searchsorted(
            torch.tensor([1, 3, 5]), torch.tensor([2, 6]), out_int32=True)))

    def test_non_contiguous_out(self):
        seq = tp.tensor([1, 3, 5, 7])
        buf = tp.zeros(2, 2, dtype=tp.int64)
        out = buf[1]
        tp.searchsorted(seq, tp.tensor([2, 5]), out=out)
        self.assertTrue(close(out, torch.searchsorted(
            torch.tensor([1, 3, 5, 7]), torch.tensor([2, 5]))))
        self.assertTrue(close(buf, [[0, 0], [1, 2]]))

    def test_error_contracts(self):
        seq = tp.tensor([1, 3, 5])
        with self.assertRaisesRegex(RuntimeError, "side"):
            tp.searchsorted(seq, tp.tensor([2]), side="middle")
        with self.assertRaisesRegex(RuntimeError, "opposites"):
            tp.searchsorted(seq, tp.tensor([2]), side="left", right=True)
        sorter_bad = tp.tensor([1, 0])
        with self.assertRaisesRegex(RuntimeError, "sorter"):
            tp.searchsorted(tp.tensor([[1, 3], [2, 4]]),
                            tp.tensor([[1, 2], [3, 4]]), sorter=sorter_bad)
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            tp.searchsorted(seq, tp.tensor([2]), sorter=tp.tensor([0, 5, 1]))
        bad_out = tp.empty(1, dtype=tp.float32)
        with self.assertRaisesRegex(RuntimeError, "dtype is wrong"):
            tp.searchsorted(seq, tp.tensor([2]), out=bad_out)
        with self.assertRaisesRegex(RuntimeError, "1 dimension"):
            tp.bucketize(tp.tensor([2]), tp.tensor([[1, 3], [2, 4]]))


class Bucketize(unittest.TestCase):
    def test_basic(self):
        vals = tp.tensor([1.0, 3.0, 5.0, 7.0])
        boundaries = tp.tensor([2.0, 4.0, 6.0])
        ref_v = torch.tensor([1.0, 3.0, 5.0, 7.0])
        ref_b = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(close(tp.bucketize(vals, boundaries),
                              torch.bucketize(ref_v, ref_b)))
        self.assertTrue(close(tp.bucketize(vals, boundaries, right=True),
                              torch.bucketize(ref_v, ref_b, right=True)))


class Unique(unittest.TestCase):
    def test_flat_values_inverse_counts(self):
        t = tp.tensor([3, 1, 2, 1])
        ref = torch.tensor([3, 1, 2, 1])
        v, i, c = tp.unique(t, return_inverse=True, return_counts=True)
        rv, ri, rc = torch.unique(ref, return_inverse=True, return_counts=True)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))
        self.assertTrue(close(c, rc))

    def test_two_output_alias(self):
        import tensorplay.functional as F
        v = F._unique(tp.tensor([3, 1, 2, 1]))
        rv = torch._unique(torch.tensor([3, 1, 2, 1]))
        self.assertTrue(close(v[0], rv[0]))
        self.assertTrue(close(v[1], rv[1]))
        v3 = F._unique2(tp.tensor([3, 1, 2, 1]), return_counts=True)
        rv3 = torch._unique2(torch.tensor([3, 1, 2, 1]), return_counts=True)
        self.assertTrue(close(v3[2], rv3[2]))

    def test_nan_never_equal(self):
        t = tp.tensor([1.0, float("nan"), float("nan"), 2.0])
        ref = torch.tensor([1.0, float("nan"), float("nan"), 2.0])
        v, c = tp.unique(t, return_counts=True)
        rv, rc = torch.unique(ref, return_counts=True)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(c, rc))

    def test_int64_precision(self):
        big = 2 ** 60
        v, c = tp.unique(tp.tensor([big, 1, big]), return_counts=True)
        self.assertTrue(close(v, torch.unique(torch.tensor([big, 1, big]))))
        self.assertTrue(close(c, [1, 2]))

    def test_bool_and_scalar_and_empty(self):
        self.assertTrue(close(tp.unique(tp.tensor([True, False, True])),
                              torch.unique(torch.tensor([True, False, True]))))
        self.assertTrue(close(tp.unique(tp.tensor(5)),
                              torch.unique(torch.tensor(5))))
        empty = tp.tensor([], dtype=tp.int64)
        v, i = tp.unique(empty, return_inverse=True)
        rv, ri = torch.unique(torch.tensor([], dtype=torch.int64),
                              return_inverse=True)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))

    def test_dim_wise(self):
        t = tp.tensor([[1, 2], [2, 3], [1, 2]])
        ref = torch.tensor([[1, 2], [2, 3], [1, 2]])
        v, i, c = tp.unique(t, dim=0, return_inverse=True, return_counts=True)
        rv, ri, rc = torch.unique(ref, dim=0, return_inverse=True,
                                  return_counts=True)
        self.assertTrue(close(v, rv))
        self.assertTrue(close(i, ri))
        self.assertTrue(close(c, rc))
        v1, i1, c1 = tp.unique(t, dim=1, return_inverse=True, return_counts=True)
        rv1, ri1, rc1 = torch.unique(ref, dim=1, return_inverse=True,
                                     return_counts=True)
        self.assertTrue(close(v1, rv1))
        self.assertTrue(close(i1, ri1))
        self.assertTrue(close(c1, rc1))

    def test_dim_negative_and_3d(self):
        t = tp.tensor([[[1, 1], [1, 1]], [[1, 1], [2, 2]], [[1, 1], [1, 1]]])
        ref = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [2, 2]],
                            [[1, 1], [1, 1]]])
        self.assertTrue(close(tp.unique(t, dim=0), torch.unique(ref, dim=0)))
        self.assertTrue(close(tp.unique(t, dim=2), torch.unique(ref, dim=2)))
        t2 = tp.tensor([[3, 1], [2, 3], [3, 1], [9, 9]])
        ref2 = torch.tensor([[3, 1], [2, 3], [3, 1], [9, 9]])
        self.assertTrue(close(tp.unique(t2, dim=-1), torch.unique(ref2, dim=-1)))

    def test_dim_consecutive(self):
        t = tp.tensor([[1, 2], [2, 3], [1, 2]])
        ref = torch.tensor([[1, 2], [2, 3], [1, 2]])
        v = tp.unique_consecutive(t, dim=0)
        rv = torch.unique_consecutive(ref, dim=0)
        self.assertTrue(close(v, rv))
        v1, i1 = tp.unique_consecutive(t, dim=0, return_inverse=True)
        rv1, ri1 = torch.unique_consecutive(ref, dim=0, return_inverse=True)
        self.assertTrue(close(v1, rv1))
        self.assertTrue(close(i1, ri1))

    def test_dim_error_contracts(self):
        # More than one zero-sized dimension rejects the op outright.
        bad = tp.zeros(2, 0, 0)
        with self.assertRaisesRegex(RuntimeError, "more than one"):
            tp.unique(bad, dim=1)
        # A zero-sized dimension that is not the selected one rejects too.
        unselected = tp.zeros(2, 0, 2)
        with self.assertRaisesRegex(RuntimeError, "aren't selected"):
            tp.unique(unselected, dim=0)
        # The single selected zero dimension is legal and passes through.
        passthrough = tp.unique(tp.zeros(2, 0, 2), dim=1)
        self.assertEqual(tuple(passthrough.shape), (2, 0, 2))


if __name__ == "__main__":
    unittest.main()
