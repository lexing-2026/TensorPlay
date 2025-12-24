import unittest
import tensorplay as tp

class TestReduction(unittest.TestCase):
    def test_prod(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        print(f"test_prod x.dtype: {x.dtype}")
        self.assertAlmostEqual(x.prod().item(), 24.0)
        print(x)
        
        p0 = x.prod(dim=0)
        self.assertAlmostEqual(p0[0].item(), 3.0)
        self.assertAlmostEqual(p0[1].item(), 8.0)
        
        p1 = x.prod(dim=1)
        self.assertAlmostEqual(p1[0].item(), 2.0)
        self.assertAlmostEqual(p1[1].item(), 12.0)

    def test_all_any(self):
        x = tp.tensor([[1, 0], [1, 1]], dtype=tp.float32) # Non-bool
        self.assertTrue(x.any().item())
        self.assertFalse(x.all().item())
        
        y = tp.tensor([1, 1], dtype=tp.float32)
        self.assertTrue(y.all().item())
        
        z = tp.tensor([0, 0], dtype=tp.float32)
        self.assertFalse(z.any().item())
        
        # Test dim
        x_all0 = x.all(dim=0) # [T, F] -> [1, 0] && [1, 1] -> [1, 0]
        self.assertTrue(x_all0[0].item())
        self.assertFalse(x_all0[1].item())

    def test_argmax_argmin(self):
        x = tp.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]])
        
        # Flatten: max is 6.0 at index 5
        self.assertEqual(x.argmax().item(), 5)
        # Flatten: min is 1.0 at index 0
        self.assertEqual(x.argmin().item(), 0)
        
        # Dim 0: maxes are [4, 5, 6] -> indices [1, 0, 1]
        am0 = x.argmax(dim=0)
        self.assertEqual(am0[0].item(), 1)
        self.assertEqual(am0[1].item(), 0)
        self.assertEqual(am0[2].item(), 1)
        
        # Dim 1: maxes are [5, 6] -> indices [1, 2]
        am1 = x.argmax(dim=1)
        self.assertEqual(am1[0].item(), 1)
        self.assertEqual(am1[1].item(), 2)

    def test_sum_mean(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(x.sum().item(), 10.0)
        self.assertAlmostEqual(x.mean().item(), 2.5)
        
if __name__ == '__main__':
    unittest.main()
