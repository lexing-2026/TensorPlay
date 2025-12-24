import unittest
import tensorplay as tp
import tensorplay.nn as nn


class TestLinearInit(unittest.TestCase):
    def test_linear_init(self):
        l = nn.Linear(10, 20)
        self.assertIsInstance(l.weight, nn.Parameter)
        self.assertTrue(l.weight.requires_grad)
        self.assertIsNotNone(l.bias)
        self.assertIsInstance(l.bias, nn.Parameter)
        
        # Check that weights are not all zeros (initialized)
        self.assertNotEqual(l.weight.sum().item(), 0.0)
        
        # Check basic stats (very rough check)
        # Kaiming uniform
        # std = gain / sqrt(fan_in) * sqrt(3)
        # gain for leaky_relu(sqrt(5)) -> sqrt(2/(1+5))? No, a=sqrt(5) is passed to kaiming_uniform
        # Wait, PyTorch Linear uses kaiming_uniform with a=sqrt(5)
        # The gain for a=sqrt(5) (negative_slope) is sqrt(2/(1+5)) = sqrt(1/3) approx 0.577
        # fan_in = 10
        # std = 0.577 / sqrt(10) ~= 0.18
        # range = [-0.18*sqrt(3), 0.18*sqrt(3)] = [-0.31, 0.31]
        
        # Just check it's within reason
        self.assertTrue(l.weight.abs().max().item() < 1.0)

if __name__ == '__main__':
    unittest.main()
