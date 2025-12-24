import unittest
import sys
import os
import numpy as np

# Debug info
print(f"DEBUG: sys.path: {sys.path}")
try:
    import tensorplay
    print(f"DEBUG: tensorplay file: {tensorplay.__file__}")
except ImportError as e:
    print(f"DEBUG: Failed to import tensorplay: {e}")

import torch

class TestCUDA(unittest.TestCase):
    def test_cuda_availability(self):
        print(f"\nChecking CUDA availability...")
        available = tensorplay.cuda.is_available()
        print(f"tensorplay.cuda.is_available(): {available}")
        
        # Verify against torch if torch has cuda
        if torch.cuda.is_available():
            print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            self.assertTrue(available, "CUDA should be available if torch finds it (assuming environment matches)")
        else:
            print("torch.cuda.is_available(): False")
            # If torch doesn't find it but we built with it, it might be fine, or not.
            # But here we expect it to be True as we saw in config.
        
        if available:
            # Check device count (assuming API exists, if not we catch)
            if hasattr(tensorplay.cuda, 'device_count'):
                count = tensorplay.cuda.device_count()
                print(f"Device count: {count}")
                self.assertGreater(count, 0)
            
            if hasattr(tensorplay.cuda, 'current_device'):
                current = tensorplay.cuda.current_device()
                print(f"Current device: {current}")

    def test_tensor_movement(self):
        if not tensorplay.cuda.is_available():
            print("Skipping movement test as CUDA is not available")
            return

        print("\nTesting Tensor movement...")
        # CPU to CUDA
        x_cpu = tensorplay.ones((2, 3))
        print(f"x_cpu device: {x_cpu.device}")
        self.assertTrue(x_cpu.device.is_cpu())
        
        try:
            x_cuda = x_cpu.cuda()
            print(f"x_cuda device: {x_cuda.device}")
            self.assertTrue(x_cuda.device.is_cuda())
            
            # CUDA to CPU
            x_cpu_back = x_cuda.cpu()
            print(f"x_cpu_back device: {x_cpu_back.device}")
            self.assertTrue(x_cpu_back.device.is_cpu())
            
            # Check values
            self.assertTrue(tensorplay.all(x_cpu_back == x_cpu).item())
        except AttributeError as e:
            print(f"Caught expected error (if method missing): {e}")
            # We might need to implement .cuda() and .cpu()
            if "has no attribute 'cuda'" in str(e):
                self.fail("Tensor.cuda() method is missing!")
            elif "has no attribute 'cpu'" in str(e):
                self.fail("Tensor.cpu() method is missing!")
            else:
                raise e

    def test_tensor_creation_on_device(self):
        if not tensorplay.cuda.is_available():
            return
            
        print("\nTesting Tensor creation on device...")
        # Direct creation
        device = tensorplay._C.Device(tensorplay._C.DeviceType.CUDA, 0)
        x = tensorplay.zeros((2, 3), device=device)
        print(f"Created tensor on {x.device}")
        self.assertTrue(x.device.is_cuda())
        
        # Verify values
        x_cpu = x.cpu()
        np.testing.assert_array_equal(x_cpu.numpy(), np.zeros((2, 3)))

    def test_basic_ops(self):
        if not tensorplay.cuda.is_available():
            return

        print("\nTesting Basic Ops on CUDA...")
        x = tensorplay.ones((2, 2)).cuda()
        y = tensorplay.ones((2, 2)).cuda()
        
        z = x + y
        print(f"Result device: {z.device}")
        self.assertTrue(z.device.is_cuda())
        
        z_cpu = z.cpu()
        expected = np.ones((2, 2)) * 2
        np.testing.assert_array_equal(z_cpu.numpy(), expected)
        print("Addition result verified.")

    def test_align_with_torch(self):
        if not tensorplay.cuda.is_available():
            return

        print("\nAligning with Torch...")
        shape = (128, 128)
        
        # Matrix Multiplication
        tp_a = tensorplay.ones(shape).cuda()
        tp_b = tensorplay.ones(shape).cuda()
        tp_c = tp_a.matmul(tp_b) # Assuming matmul exists
        
        torch_a = torch.ones(shape)
        torch_b = torch.ones(shape)
        torch_c = torch.matmul(torch_a, torch_b)
        
        diff = np.abs(tp_c.cpu().numpy() - torch_c.numpy()).max()
        print(f"Matmul diff: {diff}")
        self.assertLess(diff, 1e-4)

if __name__ == '__main__':
    unittest.main()
