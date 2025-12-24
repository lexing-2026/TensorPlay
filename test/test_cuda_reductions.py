
import os
import sys
import unittest
import math
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp

class TestCUDAReductions(unittest.TestCase):
    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
            

    # @unittest.skip("CUDNN reduction causing crashes in current env")
    def test_sum(self):
        # 1D
        t = tp.ones([10], device='cuda')
        print(f"test_sum t.shape: {t.shape}")
        s = t.sum()
        self.assertAlmostEqual(s.item(), 10.0)
        
        # 2D dim
        t = tp.ones([3, 4], device='cuda')
        s0 = t.sum(dim=[0]) # shape [4]
        self.assertEqual(s0.shape, [4])
        # Check values (need to move to cpu or item check)
        # Assuming we can't easily check array values without to_cpu which might not be fully impl?
        # But we can check single elements if indexing works? Or just check sum of sum.
        self.assertAlmostEqual(s0.sum().item(), 12.0)
        
        s1 = t.sum(dim=[1], keepdim=True) # shape [3, 1]
        self.assertEqual(s1.shape, [3, 1])
        self.assertAlmostEqual(s1.sum().item(), 12.0)

    # @unittest.skip("CUDNN reduction causing crashes in current env")
    def test_mean(self):
        t = tp.ones([3, 4], device='cuda')
        m = t.mean()
        self.assertAlmostEqual(m.item(), 1.0)
        
        t2 = t * 2.0
        m0 = t2.mean(dim=[0])
        # shape [4], all 2.0
        self.assertAlmostEqual(m0.sum().item(), 8.0)

    # @unittest.skip("CUDNN reduction causing crashes in current env")
    def test_max_min(self):
        # Create tensor with values
        # We don't have arange on CUDA yet? 
        # But we can copy from CPU.
        t_cpu = tp.Tensor([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
        t = t_cpu.cuda()
        
        self.assertAlmostEqual(t.max().item(), 4.0)
        self.assertAlmostEqual(t.min().item(), 1.0)
        
        m0 = t.max(dim=[0]) # [2, 2] -> max over dim 0 -> [max(1,3), max(2,4)] = [3, 4]
        self.assertEqual(m0.shape, [2])
        # Check values indirectly
        self.assertAlmostEqual(m0.sum().item(), 7.0) 

    def test_prod(self):
        t_cpu = tp.Tensor([1.0, 2.0, 3.0, 4.0])
        t = t_cpu.cuda()
        self.assertAlmostEqual(t.prod().item(), 24.0)

    def test_norm(self):
        t_cpu = tp.Tensor([3.0, 4.0])
        t = t_cpu.cuda()
        # L2 norm = 5
        self.assertAlmostEqual(t.norm().item(), 5.0) # default p=2

    # @unittest.skip("CUDNN reduction causing crashes in current env")
    def test_var_std(self):
        t_cpu = tp.Tensor([1.0, 2.0, 3.0, 4.0])
        t = t_cpu.cuda()
        # mean=2.5, var=(2.25 + 0.25 + 0.25 + 2.25)/3 = 5/3 = 1.666...
        v = t.var()
        print(f"var result: {v.item()}")
        
        # Test manual division
        s = tp.Tensor([5.0], device='cuda')
        d = s / 3.0
        print(f"5.0 / 3.0 = {d.item()}")
        
        self.assertAlmostEqual(v.item(), 5.0/3.0)
        self.assertAlmostEqual(t.std().item(), math.sqrt(5.0/3.0))

    def test_argmax_argmin(self):
        t_cpu = tp.Tensor([1.0, 5.0, 2.0, 8.0])
        t = t_cpu.cuda()
        
        self.assertEqual(t.argmax().item(), 3)
        self.assertEqual(t.argmin().item(), 0)

if __name__ == '__main__':
    unittest.main()
