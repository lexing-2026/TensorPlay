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

if __name__ == '__main__':
    unittest.main()
