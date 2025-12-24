import unittest
import math
import tensorplay as tp
import tensorplay.nn.init as init

class TestInit(unittest.TestCase):
    def test_constant(self):
        t = tp.empty([2, 2])
        init.constant_(t, 3.0)
        self.assertTrue((t == 3.0).all().item())

    def test_ones_zeros(self):
        t = tp.empty([2, 2])
        init.ones_(t)
        self.assertTrue((t == 1.0).all().item())
        
        init.zeros_(t)
        self.assertTrue((t == 0.0).all().item())

    def test_eye(self):
        t = tp.empty([3, 3])
        init.eye_(t)
        # Check diagonal
        for i in range(3):
            self.assertEqual(t[i, i].item(), 1.0)
        self.assertEqual(t.sum().item(), 3.0)

    def test_calculate_gain(self):
        self.assertEqual(init.calculate_gain('relu'), math.sqrt(2.0))
        self.assertEqual(init.calculate_gain('linear'), 1.0)
        with self.assertRaises(ValueError):
            init.calculate_gain('invalid_nonlinearity')

    def test_kaiming_uniform(self):
        # Statistical test is flaky, so just check it runs and changes values
        t = tp.zeros([100, 100])
        init.kaiming_uniform_(t)
        self.assertNotEqual(t.sum().item(), 0.0)
        self.assertNotEqual(t.std().item(), 0.0)

    def test_xavier_uniform(self):
        t = tp.zeros([100, 100])
        init.xavier_uniform_(t)
        self.assertNotEqual(t.sum().item(), 0.0)

    def test_dirac(self):
        t = tp.zeros([2, 2, 3, 3]) # Output, Input, H, W
        init.dirac_(t)
        # Should be 1 at center of diagonals
        self.assertEqual(t[0, 0, 1, 1].item(), 1.0)
        self.assertEqual(t[1, 1, 1, 1].item(), 1.0)
        self.assertEqual(t[0, 1, 1, 1].item(), 0.0)

if __name__ == '__main__':
    unittest.main()
