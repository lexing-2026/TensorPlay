import tensorplay as tp
import unittest

class TestRandom(unittest.TestCase):
    def test_bernoulli(self):
        t = tp.rand([100])
        b = tp.bernoulli(t)
        # Check values are 0 or 1
        flat = b.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertIn(val, [0.0, 1.0])

    def test_normal(self):
        mean = tp.full([100], 0.0, dtype=tp.float32)
        std = tp.full([100], 1.0, dtype=tp.float32)
        n = tp.normal(mean, std)
        self.assertEqual(n.shape, [100])

    def test_poisson(self):
        rates = tp.full([50], 5.0, dtype=tp.float32)
        p = rates.poisson()
        flat = p.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(val >= 0)
            self.assertEqual(val, int(val))

    def test_rand(self):
        r = tp.rand([50])
        flat = r.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(0.0 <= val < 1.0)

    def test_rand_like(self):
        x = tp.ones([10, 10])
        r = tp.rand_like(x)
        self.assertEqual(r.shape, x.shape)
        # Check one value
        self.assertTrue(0.0 <= r[0,0].item() < 1.0)

    def test_randint(self):
        r = tp.randint(0, 10, [50])
        flat = r.view([-1])
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(0 <= val < 10)
            self.assertIsInstance(val, int)

    def test_randint_like(self):
        x = tp.ones([5, 5], dtype=tp.int32)
        r = tp.randint_like(x, 0, 10)
        self.assertEqual(r.shape, x.shape)
        self.assertTrue(0 <= r[0,0].item() < 10)

    def test_randn(self):
        r = tp.randn([50])
        self.assertEqual(r.shape, [50])

    def test_randn_like(self):
        x = tp.ones([5, 5])
        r = tp.randn_like(x)
        self.assertEqual(r.shape, x.shape)

    def test_randperm(self):
        n = 10
        r = tp.randperm(n)
        self.assertEqual(r.numel(), n)
        # Check range
        flat = r.view([-1])
        vals = []
        for i in range(flat.numel()):
            val = flat[i].item()
            self.assertTrue(0 <= val < n)
            vals.append(val)
        # Check uniqueness
        self.assertEqual(len(set(vals)), n)

if __name__ == '__main__':
    unittest.main()
