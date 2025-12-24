import unittest
import tensorplay as tp

class TestRandomInplace(unittest.TestCase):
    def test_bernoulli_(self):
        # bernoulli_ expects probabilities in self
        p = tp.full([100], 0.5, dtype=tp.float32)
        p.bernoulli_()
        # Check values are 0 or 1
        flat = p.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertIn(val, [0.0, 1.0])
            
    def test_cauchy_(self):
        t = tp.zeros([100], dtype=tp.float32)
        t.cauchy_(median=0.0, sigma=1.0)
        # Check non-zero
        self.assertNotEqual(t[0].item(), 0.0)
        
    def test_exponential_(self):
        t = tp.zeros([100], dtype=tp.float32)
        t.exponential_(lambd=1.0)
        # Check positive
        flat = t.view([-1])
        for i in range(flat.numel()):
             self.assertGreaterEqual(flat[i].item(), 0.0)
             
    def test_geometric_(self):
        t = tp.zeros([100], dtype=tp.float32)
        t.geometric_(p=0.5)
        # Check positive integers (>= 1)
        flat = t.view([-1])
        for i in range(flat.numel()):
             val = flat[i].item()
             self.assertGreaterEqual(val, 1.0)
             self.assertEqual(val, int(val))

    def test_log_normal_(self):
        t = tp.zeros([100], dtype=tp.float32)
        t.log_normal_(mean=0.0, std=1.0)
        # Check positive
        flat = t.view([-1])
        for i in range(flat.numel()):
             self.assertGreater(flat[i].item(), 0.0)

    def test_normal_(self):
        t = tp.zeros([100], dtype=tp.float32)
        t.normal_(mean=0.0, std=1.0)
        self.assertNotEqual(t[0].item(), 0.0)

    def test_random_(self):
        t = tp.zeros([100], dtype=tp.int64)
        t.random_(0, 10)
        flat = t.view([-1])
        for i in range(flat.numel()):
             val = flat[i].item()
             self.assertGreaterEqual(val, 0)
             self.assertLess(val, 10)
             
    def test_uniform_(self):
        t = tp.zeros([100], dtype=tp.float32)
        t.uniform_(0.0, 1.0)
        flat = t.view([-1])
        for i in range(flat.numel()):
             val = flat[i].item()
             self.assertGreaterEqual(val, 0.0)
             self.assertLess(val, 1.0)
             
    def test_inplace_return(self):
        # Verify it returns self
        t = tp.zeros([10], dtype=tp.float32)
        t2 = t.uniform_()
        self.assertEqual(id(t), id(t2))

if __name__ == '__main__':
    unittest.main()
