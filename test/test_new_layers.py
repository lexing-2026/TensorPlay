import unittest
import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.nn import functional as F


class TestNewLayers(unittest.TestCase):
    def test_flatten(self):
        print("\n--- Testing Flatten ---")
        # (N, C, H, W) -> (N, C*H*W)
        t = tp.ones([32, 512, 7, 7])
        flat = nn.Flatten()(t)
        self.assertEqual(flat.shape, [32, 512 * 7 * 7])
        
        # (N, C, H, W) -> (N, C, H*W) with start_dim=2
        flat2 = nn.Flatten(start_dim=2)(t)
        self.assertEqual(flat2.shape, [32, 512, 7 * 7])
        
        print("Flatten shapes correct")

    def test_sequential(self):
        print("\n--- Testing Sequential ---")
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 100),
            nn.PReLU(),
            nn.Linear(100, 10)
        )
        
        input = tp.ones([2, 512, 7, 7])
        output = model(input)
        
        self.assertEqual(output.shape, [2, 10])
        print("Sequential forward pass successful")
        
    def test_prelu(self):
        print("\n--- Testing PReLU ---")
        # Single parameter
        m = nn.PReLU(init=0.5)
        input = tp.tensor([-1.0, 1.0, -2.0, 2.0])
        output = m(input)
        
        # Expected: [-0.5, 1.0, -1.0, 2.0]
        # PReLU(x) = max(0, x) + 0.5 * min(0, x)
        # -1 -> 0 + 0.5 * -1 = -0.5
        # 1 -> 1
        
        self.assertAlmostEqual(output[0].item(), -0.5)
        self.assertAlmostEqual(output[1].item(), 1.0)
        
        # Multi parameter (channels)
        m2 = nn.PReLU(num_parameters=3)
        # Input (N, C, L)
        input2 = tp.ones([2, 3, 4])
        # Set weights manually to check broadcasting
        # Weights initialized to 0.25 by default
        
        output2 = m2(input2) # all positive, should be identity
        self.assertAlmostEqual(output2[0, 0, 0].item(), 1.0)
        
        input3 = tp.full([2, 3, 4], -1.0)
        output3 = m2(input3)
        # -1 * 0.25 = -0.25
        self.assertAlmostEqual(output3[0, 0, 0].item(), -0.25)
        
        print("PReLU logic correct")

if __name__ == '__main__':
    unittest.main()
