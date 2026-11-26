import math
import unittest
import tensorplay as tp

class TestOps(unittest.TestCase):
    def test_transpose(self):
        t = tp.ones(
            (2, 3), 
            dtype=tp.float32, 
            device=tp.device("cpu"),
            requires_grad=False,
            )
        t2 = t.transpose(0, 1)
        self.assertEqual(t2.shape, tp.Size([3, 2]))
        
        t3 = tp.arange(0, 6).reshape((2, 3)) # [[0, 1, 2], [3, 4, 5]]
        t4 = t3.transpose(0, 1) # [[0, 3], [1, 4], [2, 5]]
        self.assertEqual(t4.shape, tp.Size([3, 2]))

    def test_t(self):
        t = tp.tensor([[1., 2.], [3., 4.]])
        self.assertTrue(tp.allclose(t.t(), tp.tensor([[1., 3.], [2., 4.]])))
        
        # Test error on 3D
        t3 = tp.ones((2, 3, 4))
        with self.assertRaises(RuntimeError):
            t3.t()

    def test_permute(self):
        t = tp.ones((2, 3, 4))
        t2 = t.permute((2, 0, 1))
        self.assertEqual(t2.shape, tp.Size([4, 2, 3]))
        
    def test_squeeze_unsqueeze(self):
        t = tp.ones((2, 1, 3))
        t2 = t.squeeze()
        self.assertEqual(t2.shape, tp.Size([2, 3]))
        
        t3 = t.squeeze(1)
        self.assertEqual(t3.shape, tp.Size([2, 3]))
        
        t4 = t2.unsqueeze(1)
        self.assertEqual(t4.shape, tp.Size([2, 1, 3]))

    def test_cat(self):
        t1 = tp.ones((2, 3))
        t2 = tp.ones((2, 3))
        t3 = tp.cat([t1, t2], dim=0)
        self.assertEqual(t3.shape, tp.Size([4, 3]))
        
        t4 = tp.cat([t1, t2], dim=1)
        self.assertEqual(t4.shape, tp.Size([2, 6]))

    def test_stack(self):
        t1 = tp.ones((2, 3))
        t2 = tp.ones((2, 3))
        t3 = tp.stack([t1, t2], dim=0)
        self.assertEqual(t3.shape, tp.Size([2, 2, 3]))
        
        t4 = tp.stack([t1, t2], dim=1)
        self.assertEqual(t4.shape, tp.Size([2, 2, 3]))

    def test_split(self):
        t = tp.arange(0, 10)
        # Split with int size
        chunks = t.split(3)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].shape, tp.Size([3]))
        self.assertEqual(chunks[3].shape, tp.Size([1]))
        
        # Split with sizes
        chunks2 = t.split([2, 3, 5])
        self.assertEqual(len(chunks2), 3)
        self.assertEqual(chunks2[0].shape, tp.Size([2]))
        self.assertEqual(chunks2[1].shape, tp.Size([3]))
        self.assertEqual(chunks2[2].shape, tp.Size([5]))
        
        # Test error
        with self.assertRaises(RuntimeError):
            t.split([2, 3]) # Sum != 10

    def test_chunk(self):
        t = tp.arange(0, 11)
        chunks = t.chunk(6)
        # 11 / 6 = 1.83 -> ceil 2 per chunk
        # [2, 2, 2, 2, 2, 1] -> 6 chunks
        self.assertEqual(len(chunks), 6)
        self.assertEqual(chunks[0].shape, tp.Size([2]))
        self.assertEqual(chunks[5].shape, tp.Size([1]))
        
        t2 = tp.arange(0, 12)
        chunks2 = t2.chunk(3)
        self.assertEqual(len(chunks2), 3)
        self.assertEqual(chunks2[0].shape, tp.Size([4]))

    def test_multinomial_replacement_uses_one_cdf_per_row(self):
        # One-hot rows must sample the same index for every draw; this pins
        # the CDF shape and the binary search, including the zero-mass row.
        probs = tp.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        samples = tp.multinomial(probs, 8, replacement=True)
        self.assertEqual(tuple(samples.shape), (4, 8))
        self.assertEqual(samples[0].tolist(), [1] * 8)
        self.assertEqual(samples[2].tolist(), [0] * 8)
        self.assertEqual(samples[3].tolist(), [0] * 8)

    def test_cdist_backward_p2_gathers_pairs_by_matmul(self):
        a = [[1.0, 2.0], [4.0, 6.0]]
        b = [[0.0, 0.0], [3.0, 4.0]]
        x1 = tp.tensor(a)
        x2 = tp.tensor(b)
        leaf = x1.clone().requires_grad_(True)
        tp.cdist(leaf, x2, 2.0).sum().backward()
        want = []
        for i in range(2):
            row = [0.0, 0.0]
            for j in range(2):
                diff = [a[i][k] - b[j][k] for k in range(2)]
                dist = math.sqrt(sum(v * v for v in diff))
                for k in range(2):
                    row[k] += diff[k] / dist
            want.append(row)
        for got_row, want_row in zip(leaf.grad.tolist(), want):
            for got, ref in zip(got_row, want_row):
                self.assertAlmostEqual(got, ref, places=5)

    def test_compute_linear_combination_cpu_kernel(self):
        # The implementation lived in FunctionOfAMatrixUtils.cpp but was not
        # in the CPU source list, so dispatch stopped at NotImplementedError.
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        coeff = tp.tensor([[2.0, -1.0]])
        got = tp.ops.tp._compute_linear_combination(x, coeff)
        self.assertEqual(got.tolist(), [[-1.0, 0.0]])

    def test_multinomial_seeded_rows_are_reproducible(self):
        # Parallel row walks must not share generator state; the same seed
        # has to answer the same draws.
        probs = tp.tensor([
            [0.2, 0.3, 0.5],
            [0.5, 0.25, 0.25],
            [0.1, 0.1, 0.8],
        ])
        tp.manual_seed(1234)
        first = tp.multinomial(probs, 16, replacement=True).tolist()
        tp.manual_seed(1234)
        second = tp.multinomial(probs, 16, replacement=True).tolist()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(0 <= v < 3 for row in first for v in row))

    def test_internal_to_sparse_csr_matches_the_public_conversion(self):
        d = tp.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
        s = tp.ops.tp._to_sparse_csr(d)
        self.assertTrue(s.is_sparse_csr())
        self.assertEqual(s.crow_indices().tolist(), [0, 2, 2, 5])
        self.assertEqual(s.col_indices().tolist(), [0, 2, 0, 1, 2])
        self.assertEqual(s.values().tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_native_batch_norm_legit_updates_running_stats(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        rm = tp.zeros((2,))
        rv = tp.ones((2,))
        out, save_mean, save_invstd = tp.ops.tp._native_batch_norm_legit(
            x, tp.ones((2,)), tp.zeros((2,)), rm, rv, True, 0.1, 1e-5)
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertAlmostEqual(rm[0].item(), 0.2, places=5)
        self.assertAlmostEqual(rm[1].item(), 0.3, places=5)

    def test_slow_conv_transpose3d_reaches_the_cpu_slot(self):
        x = tp.ones((1, 1, 2, 2, 2))
        w = tp.ones((1, 1, 2, 2, 2))
        out = tp.ops.tp.slow_conv_transpose3d(
            x, w, [2, 2, 2], None, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1])
        self.assertEqual(tuple(out.shape), (1, 1, 3, 3, 3))
        self.assertAlmostEqual(out.sum().item(), 8.0, places=5)

class TestOpPackets(unittest.TestCase):
    def test_inplace_packet_call_reaches_the_kernel(self):
        # Ops whose resolver is not exposed as a C function answer from
        # __call_py__; the call slot must not hand the body the packet twice.
        x = tp.tensor([1.0, -2.0, 4.0])
        tp.ops.tp.neg_(x)
        self.assertTrue(tp.allclose(x, tp.tensor([-1.0, 2.0, -4.0])))

        y = tp.tensor([4.0, 9.0])
        tp.ops.tp.sqrt_(y)
        self.assertTrue(tp.allclose(y, tp.tensor([2.0, 3.0])))

    def test_packet_call_leaves_the_source_untouched(self):
        z = tp.tensor([1.0, -2.0])
        out = tp.ops.tp.abs_(z)
        self.assertTrue(tp.allclose(z, tp.tensor([1.0, 2.0])))
        self.assertTrue(tp.allclose(out, tp.tensor([1.0, 2.0])))

    def test_packet_call_retries_the_other_overloads(self):
        # fill_ binds fill_.Tensor first; a Python number only binds
        # fill_.Scalar, and the packet must reach it instead of stopping at
        # the first entry's signature check.
        x = tp.ones((2, 2))
        tp.ops.tp.fill_(x, 3.0)
        self.assertTrue(tp.allclose(x, tp.full((2, 2), 3.0)))

        y = tp.ones((2, 2))
        tp.ops.tp.fill_(y, 3)
        self.assertTrue(tp.allclose(y, tp.full((2, 2), 3.0)))

    def test_packet_call_reports_every_candidate(self):
        with self.assertRaises(TypeError) as raised:
            tp.ops.tp.fill_(tp.ones((2, 2)), object())
        self.assertIn("no overload of", str(raised.exception))

    def test_nll_loss_returns_the_loss_tensor(self):
        logits = tp.tensor([[0.0, -1.0], [-1.0, 0.0]])
        target = tp.tensor([0, 1], dtype=tp.int64)
        loss = tp.ops.tp.nll_loss(logits, target)
        self.assertIsInstance(loss, tp.Tensor)
        self.assertEqual(tuple(loss.shape), ())
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertIsInstance(tp.nll_loss(logits, target), tp.Tensor)
if __name__ == '__main__':
    unittest.main()
