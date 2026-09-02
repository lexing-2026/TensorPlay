import unittest

import tensorplay as tp
from tensorplay import sparse


def dense_from_coo(t):
    return t.to_dense() if hasattr(t, "to_dense") else tp.to_dense(t)


class TestSparseCOO(unittest.TestCase):
    def test_construct_and_inference(self):
        indices = tp.tensor([[0, 1, 2], [3, 1, 0]], dtype=tp.int64)
        values = tp.tensor([4.0, 5.0, 6.0])
        # size=None infers from coordinate maxima
        s = sparse.sparse_coo_tensor(indices, values)
        self.assertEqual(tuple(s.shape), (3, 4))
        self.assertFalse(s.is_coalesced())
        self.assertTrue(s.is_sparse)
        self.assertFalse(s.is_sparse_csr())
        self.assertEqual(s._nnz() if hasattr(s, "_nnz") else s.nnz, 3)

    def test_public_naming(self):
        indices = tp.tensor([[0, 1], [1, 0]], dtype=tp.int64)
        values = tp.tensor([7.0, 8.0])
        s = sparse.sparse_coo_tensor(indices, values, [2, 2])
        self.assertTrue(hasattr(s, "values"))
        self.assertTrue(tp.equal(s.values(), values))
        self.assertEqual(s.layout, tp.sparse_coo)

    def test_coalesce_folds_duplicates(self):
        indices = tp.tensor([[0, 0, 1], [0, 0, 1]], dtype=tp.int64)
        values = tp.tensor([1.0, 2.0, 3.0])
        s = sparse.sparse_coo_tensor(indices, values, [2, 2])
        c = sparse.coalesce(s)
        self.assertTrue(c.is_coalesced())
        self.assertEqual(c.values().numel(), 2)
        dense = dense_from_coo(c)
        self.assertTrue(tp.allclose(
            dense, tp.tensor([[3.0, 0.0], [0.0, 3.0]])))


class TestSparseConversions(unittest.TestCase):
    def test_to_dense(self):
        indices = tp.tensor([[0, 1], [1, 0]], dtype=tp.int64)
        s = sparse.sparse_coo_tensor(indices, tp.tensor([5.0, 6.0]), [2, 2])
        d = sparse.to_dense(s)
        self.assertTrue(tp.allclose(d, tp.tensor([[0.0, 5.0], [6.0, 0.0]])))

    def test_to_sparse_roundtrip(self):
        d = tp.tensor([[0.0, 2.0], [0.0, 0.0]])
        s = sparse.to_sparse(d)
        self.assertTrue(s.is_sparse)
        self.assertTrue(s.is_coalesced())
        self.assertTrue(tp.allclose(sparse.to_dense(s), d))

    def test_to_sparse_csr(self):
        d = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
        s = sparse.to_sparse_csr(d)
        self.assertTrue(s.is_sparse_csr())
        self.assertEqual(s.layout, tp.sparse_csr)
        crow = s.crow_indices()
        col = s.col_indices()
        self.assertEqual(crow.tolist(), [0, 2, 2, 5])
        self.assertEqual(col.tolist(), [0, 2, 0, 1, 2])
        self.assertEqual(s.values().tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])


class TestSparseMM(unittest.TestCase):
    def test_coo_mm(self):
        indices = tp.tensor([[0, 1], [0, 2]], dtype=tp.int64)
        s = sparse.sparse_coo_tensor(indices, tp.tensor([1.0, 2.0]), [2, 3])
        d = tp.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = sparse.mm(s, d)
        expect = tp.tensor([[1.0, 2.0], [10.0, 12.0]])
        self.assertTrue(tp.allclose(out, expect))

    def test_csr_mm(self):
        d0 = tp.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        s = sparse.to_sparse_csr(d0)
        d = tp.arange(6, dtype=tp.float32).reshape(3, 2)
        out = sparse.mm(s, d)
        self.assertTrue(tp.allclose(out, tp.matmul(d0, d)))


class TestSparseSum(unittest.TestCase):
    def make_input(self):
        # [[1, 0], [0, 2]] plus a duplicate at (0, 0) folded by reduction.
        indices = tp.tensor([[0, 0, 1], [0, 0, 1]], dtype=tp.int64)
        values = tp.tensor([1.0, 0.5, 2.0])
        return sparse.sparse_coo_tensor(indices, values, [2, 2])

    def test_sum_all(self):
        s = self.make_input()
        total = sparse.sum(s)
        self.assertFalse(total.is_sparse)
        self.assertAlmostEqual(float(total), 3.5, places=6)

    def test_sum_all_dims_dense(self):
        s = self.make_input()
        out = sparse.sum(s, dim=[0, 1])
        self.assertFalse(out.is_sparse)
        self.assertAlmostEqual(float(out.reshape(-1)[0]), 3.5, places=6)

    def test_sum_partial_dim_is_sparse(self):
        s = self.make_input()
        out = sparse.sum(s, dim=[1])
        self.assertTrue(out.is_sparse)
        self.assertEqual(tuple(out.shape), (2,))
        dense = dense_from_coo(out)
        self.assertTrue(tp.allclose(dense, tp.tensor([1.5, 2.0])))

    def test_sum_dtype(self):
        s = self.make_input()
        out = sparse.sum(s, dtype=tp.float64)
        self.assertEqual(out.dtype, tp.float64)
        self.assertAlmostEqual(float(out), 3.5, places=6)


class TestSparseBinaryOps(unittest.TestCase):
    def test_add_union(self):
        a = sparse.sparse_coo_tensor(
            tp.tensor([[0, 1], [0, 1]], dtype=tp.int64),
            tp.tensor([1.0, 2.0]), [2, 2])
        b = sparse.sparse_coo_tensor(
            tp.tensor([[0, 1], [0, 1]], dtype=tp.int64),
            tp.tensor([10.0, 20.0]), [2, 2])
        out = sparse.add(a, b)
        self.assertTrue(out.is_coalesced())
        self.assertTrue(tp.allclose(
            dense_from_coo(out), tp.tensor([[11.0, 0.0], [0.0, 22.0]])))

    def test_add_disjoint(self):
        a = sparse.sparse_coo_tensor(
            tp.tensor([[0], [0]], dtype=tp.int64), tp.tensor([1.0]), [2, 2])
        b = sparse.sparse_coo_tensor(
            tp.tensor([[1], [1]], dtype=tp.int64), tp.tensor([2.0]), [2, 2])
        out = sparse.add(a, b)
        self.assertTrue(tp.allclose(
            dense_from_coo(out), tp.tensor([[1.0, 0.0], [0.0, 2.0]])))

    def test_add_different_nnz(self):
        a = sparse.sparse_coo_tensor(
            tp.tensor([[0], [0]], dtype=tp.int64), tp.tensor([1.0]), [2, 2])
        b = sparse.sparse_coo_tensor(
            tp.tensor([[0, 1], [1, 0]], dtype=tp.int64),
            tp.tensor([2.0, 3.0]),
            [2, 2],
        )
        out = sparse.add(a, b)
        self.assertTrue(tp.allclose(
            dense_from_coo(out), tp.tensor([[1.0, 2.0], [3.0, 0.0]])))

    def test_mul_intersection(self):
        a = sparse.sparse_coo_tensor(
            tp.tensor([[0, 1, 1], [0, 1, 2]], dtype=tp.int64),
            tp.tensor([1.0, 2.0, 3.0]), [2, 3])
        b = sparse.sparse_coo_tensor(
            tp.tensor([[0, 1], [0, 2]], dtype=tp.int64),
            tp.tensor([10.0, 20.0]), [2, 3])
        out = sparse.mul(a, b)
        self.assertTrue(out.is_coalesced())
        dense = dense_from_coo(out)
        self.assertTrue(tp.allclose(
            dense, tp.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 60.0]])))

    def test_add_shape_mismatch_raises(self):
        a = sparse.sparse_coo_tensor(
            tp.tensor([[0], [0]], dtype=tp.int64), tp.tensor([1.0]), [2, 2])
        b = sparse.sparse_coo_tensor(
            tp.tensor([[0], [0]], dtype=tp.int64), tp.tensor([1.0]), [3, 3])
        with self.assertRaises(RuntimeError):
            sparse.add(a, b)


class TestSpdiags(unittest.TestCase):
    def test_negative_offsets(self):
        diagonals = tp.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0],
                               [6.0, 7.0, 8.0]])
        offsets = tp.tensor([0, -1, -2], dtype=tp.int64)
        s = sparse.spdiags(diagonals, offsets, (3, 3))
        self.assertEqual(s.values().tolist(), [0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
        dense = dense_from_coo(s)
        self.assertTrue(tp.allclose(
            dense, tp.tensor([[0.0, 0.0, 0.0], [3.0, 1.0, 0.0],
                              [6.0, 4.0, 2.0]])))

    def test_positive_offsets(self):
        diagonals = tp.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                               [1.0, 2.0, 3.0]])
        offsets = tp.tensor([0, 1, 2], dtype=tp.int64)
        s = sparse.spdiags(diagonals, offsets, (3, 3))
        dense = dense_from_coo(s)
        self.assertTrue(tp.allclose(
            dense, tp.tensor([[1.0, 2.0, 3.0], [0.0, 2.0, 3.0],
                              [0.0, 0.0, 3.0]])))

    def test_csr_layout(self):
        diagonals = tp.tensor([[1.0, 2.0, 3.0]])
        offsets = tp.tensor([0], dtype=tp.int64)
        s = sparse.spdiags(diagonals, offsets, (3, 3),
                           layout=tp.sparse_csr)
        self.assertTrue(s.is_sparse_csr())
        self.assertEqual(s.crow_indices().tolist(), [0, 1, 2, 3])

    def test_duplicate_offsets_raise(self):
        diagonals = tp.tensor([[1.0, 2.0], [1.0, 2.0]])
        offsets = tp.tensor([0, 0], dtype=tp.int64)
        with self.assertRaises(ValueError):
            sparse.spdiags(diagonals, offsets, (2, 2))


if __name__ == "__main__":
    unittest.main()
