import unittest
import tensorplay as tp
import tensorplay.nn as nn
import numpy as np

class TestConvPadding(unittest.TestCase):
    def test_conv2d_valid(self):
        print("\nTesting Conv2d padding='valid'...")
        conv = nn.Conv2d(3, 6, 3, padding='valid')
        self.assertEqual(conv.padding, (0, 0))
        input = tp.randn(1, 3, 10, 10)
        out = conv(input)
        # Output size: (10 - 3)/1 + 1 = 8
        self.assertEqual(out.shape, (1, 6, 8, 8))
        print("Conv2d padding='valid' Passed!")

    def test_conv2d_same_odd(self):
        print("\nTesting Conv2d padding='same' (odd kernel)...")
        # k=3, dilation=1 => padding=(1,1)
        conv = nn.Conv2d(3, 6, 3, padding='same')
        self.assertEqual(conv.padding, (1, 1))
        input = tp.randn(1, 3, 10, 10)
        out = conv(input)
        # Output size: (10 + 2*1 - 3)/1 + 1 = 10
        self.assertEqual(out.shape, (1, 6, 10, 10))
        print("Conv2d padding='same' (odd kernel) Passed!")

    def test_conv2d_same_odd_dilation(self):
        print("\nTesting Conv2d padding='same' (odd kernel, dilation=2)...")
        # k=3, dilation=2. Total pad = 2*(3-1) = 4. Left=2, Right=2.
        conv = nn.Conv2d(3, 6, 3, dilation=2, padding='same')
        self.assertEqual(conv.padding, (2, 2))
        input = tp.randn(1, 3, 10, 10)
        out = conv(input)
        # Output size: (10 + 4 - (2*(3-1)+1))/1 + 1 = (14 - 5) + 1 = 10
        self.assertEqual(out.shape, (1, 6, 10, 10))
        print("Conv2d padding='same' (dilation) Passed!")

    def test_conv2d_same_even(self):
        print("\nTesting Conv2d padding='same' (even kernel)...")
        # k=2, dilation=1 => pad=1 (left=0, right=1) or (left=1, right=0)?
        # Our impl: total=1, left=0, right=1.
        conv = nn.Conv2d(3, 6, 2, padding='same')
        # padding should be 0 because we handle it via F.pad
        # Update: Implementation now optimizes 2D asymmetric padding by passing it to kernel
        # self.assertEqual(conv.padding, (0, 0))
        self.assertEqual(conv.padding, (0, 1, 0, 1))
        # self.assertIsNotNone(conv._reversed_padding_repeated_twice)
        if len(conv.kernel_size) > 2:
             self.assertIsNotNone(conv._reversed_padding_repeated_twice)
        
        input = tp.randn(1, 3, 10, 10)
        out = conv(input)
        # Output size: (10 + 1 - 2)/1 + 1 = 10
        self.assertEqual(out.shape, (1, 6, 10, 10))
        print("Conv2d padding='same' (even kernel) Passed!")

    def test_conv2d_same_stride_error(self):
        print("\nTesting Conv2d padding='same' (stride=2) Error...")
        with self.assertRaisesRegex(ValueError, "strided convolutions"):
            nn.Conv2d(3, 6, 3, stride=2, padding='same')
        print("Conv2d padding='same' (stride=2) Error Passed!")

    def test_conv1d_valid(self):
        print("\nTesting Conv1d padding='valid'...")
        conv = nn.Conv1d(3, 6, 3, padding='valid')
        self.assertEqual(conv.padding, (0,))
        input = tp.randn(1, 3, 10)
        out = conv(input)
        self.assertEqual(out.shape, (1, 6, 8))
        print("Conv1d padding='valid' Passed!")

    def test_conv3d_same(self):
        print("\nTesting Conv3d padding='same'...")
        conv = nn.Conv3d(3, 6, 3, padding='same')
        self.assertEqual(conv.padding, (1, 1, 1))
        input = tp.randn(1, 3, 5, 10, 10)
        out = conv(input)
        self.assertEqual(out.shape, (1, 6, 5, 10, 10))
        print("Conv3d padding='same' Passed!")
    
    def test_weight_init(self):
        print("\nTesting Weight Initialization...")
        conv = nn.Conv2d(3, 6, 3)
        # Kaiming Uniform should not be all zeros
        self.assertFalse(np.allclose(conv.weight.detach().numpy(), 0))
        # Bias should be uniform
        self.assertFalse(np.allclose(conv.bias.detach().numpy(), 0))
        print("Weight Initialization Passed!")

    def test_depthwise_conv2d_same(self):
        print("\nTesting DepthwiseConv2d padding='same'...")
        conv = nn.DepthwiseConv2d(3, 6, 3, padding='same')
        self.assertEqual(conv.padding, (1, 1))
        input = tp.randn(1, 3, 10, 10)
        out = conv(input)
        self.assertEqual(out.shape, (1, 6, 10, 10))
        print("DepthwiseConv2d padding='same' Passed!")

if __name__ == '__main__':
    unittest.main()
