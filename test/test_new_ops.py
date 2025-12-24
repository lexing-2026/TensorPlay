import unittest
import tensorplay as tp

class TestNewOps(unittest.TestCase):
    def test_reshape(self):
        t = tp.arange(6)
        r1 = t.reshape(2, 3)
        self.assertEqual(r1.shape, (2, 3))
        r2 = t.reshape((3, 2))
        self.assertEqual(r2.shape, (3, 2))
        r3 = t.reshape([3, 2])
        self.assertEqual(r3.shape, (3, 2))
        
    def test_unbind(self):
        t = tp.tensor([[1, 2, 3], [4, 5, 6]])
        tensors = tp.unbind(t, dim=0)
        self.assertEqual(len(tensors), 2)
        self.assertEqual(tensors[0].shape, (3,))
        # Check values
        self.assertEqual(tensors[0][0].item(), 1)
        self.assertEqual(tensors[0][1].item(), 2)
        self.assertEqual(tensors[0][2].item(), 3)
        
        tensors_dim1 = t.unbind(dim=1)
        self.assertEqual(len(tensors_dim1), 3)
        self.assertEqual(tensors_dim1[0].shape, (2,))
        self.assertEqual(tensors_dim1[0][0].item(), 1)
        self.assertEqual(tensors_dim1[0][1].item(), 4)
        
    def test_linspace(self):
        t = tp.linspace(0, 10, 5) # 0, 2.5, 5, 7.5, 10
        self.assertEqual(t.shape, (5,))
        self.assertAlmostEqual(t[0].item(), 0.0)
        self.assertAlmostEqual(t[4].item(), 10.0)
        self.assertAlmostEqual(t[2].item(), 5.0)
        
    def test_logspace(self):
        t = tp.logspace(0, 2, 3, base=10) # 10^0, 10^1, 10^2 -> 1, 10, 100
        self.assertEqual(t.shape, (3,))
        self.assertAlmostEqual(t[0].item(), 1.0)
        self.assertAlmostEqual(t[1].item(), 10.0)
        self.assertAlmostEqual(t[2].item(), 100.0)
        
    def test_random_seed(self):
        tp.manual_seed(123)
        t1 = tp.rand([10])
        
        tp.manual_seed(123)
        t2 = tp.rand([10])
        
        for i in range(10):
            self.assertEqual(t1[i].item(), t2[i].item())
            
        tp.manual_seed(456)
        t3 = tp.rand([10])
        # Very unlikely to be equal
        diff = False
        for i in range(10):
            if t1[i].item() != t3[i].item():
                diff = True
                break
        self.assertTrue(diff)

    def test_default_generator(self):
        gen = tp.default_generator()
        self.assertTrue(isinstance(gen, tp.Generator))
        current_seed = gen.seed()
        self.assertTrue(isinstance(current_seed, int))

if __name__ == '__main__':
    unittest.main()
