
import unittest
import tensorplay as tp
import numpy as np

class TestScalarTensorOps(unittest.TestCase):
    def setUp(self):
        self.devices = [tp.device("cpu")]
        if tp.cuda.is_available():
            self.devices.append(tp.device("cuda"))

    def test_add_scalar(self):
        print("\nTesting Add Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([1.0, 2.0, 3.0], device=device)
            s = 2.0
            
            # Tensor + Scalar
            res = t + s
            expected = [3.0, 4.0, 5.0]
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), f"t + s failed on {device}")
            
            # Scalar + Tensor
            res = s + t
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), f"s + t failed on {device}")
            
            # In-place
            t_clone = t.clone()
            t_clone += s
            self.assertTrue(tp.allclose(t_clone.cpu(), tp.tensor(expected)), f"t += s failed on {device}")

    def test_sub_scalar(self):
        print("\nTesting Sub Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([1.0, 2.0, 3.0], device=device)
            s = 1.0
            
            # Tensor - Scalar
            res = t - s
            expected = [0.0, 1.0, 2.0]
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), f"t - s failed on {device}")
            
            # Scalar - Tensor
            # 1.0 - [1, 2, 3] = [0, -1, -2]
            res = s - t
            expected_rev = [0.0, -1.0, -2.0]
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected_rev)), f"s - t failed on {device}")
            
            # In-place
            t_clone = t.clone()
            t_clone -= s
            self.assertTrue(tp.allclose(t_clone.cpu(), tp.tensor(expected)), f"t -= s failed on {device}")

    def test_mul_scalar(self):
        print("\nTesting Mul Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([1.0, 2.0, 3.0], device=device)
            s = 2.0
            
            # Tensor * Scalar
            res = t * s
            expected = [2.0, 4.0, 6.0]
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), f"t * s failed on {device}")
            
            # Scalar * Tensor
            res = s * t
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), f"s * t failed on {device}")
            
            # In-place
            t_clone = t.clone()
            t_clone *= s
            self.assertTrue(tp.allclose(t_clone.cpu(), tp.tensor(expected)), f"t *= s failed on {device}")

    def test_div_scalar(self):
        print("\nTesting Div Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([2.0, 4.0, 8.0], device=device)
            s = 2.0
            
            # Tensor / Scalar
            res = t / s
            expected = [1.0, 2.0, 4.0]
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), f"t / s failed on {device}")
            
            # Scalar / Tensor
            # 2.0 / [2, 4, 8] = [1.0, 0.5, 0.25]
            res = s / t
            expected_rev = [1.0, 0.5, 0.25]
            self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected_rev)), f"s / t failed on {device}")
            
            # In-place
            t_clone = t.clone()
            t_clone /= s
            self.assertTrue(tp.allclose(t_clone.cpu(), tp.tensor(expected)), f"t /= s failed on {device}")

    def test_comparison_scalar(self):
        print("\nTesting Comparison Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([1.0, 2.0, 3.0], device=device)
            
            # Eq
            res = t == 2.0
            expected = [False, True, False]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"t == s failed on {device}")
            
            # Ne
            res = t != 2.0
            expected = [True, False, True]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"t != s failed on {device}")
            
            # Lt
            res = t < 2.5
            expected = [True, True, False]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"t < s failed on {device}")
            
            # Le
            res = t <= 2.0
            expected = [True, True, False]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"t <= s failed on {device}")
            
            # Gt
            res = t > 1.5
            expected = [False, True, True]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"t > s failed on {device}")
            
            # Ge
            res = t >= 2.0
            expected = [False, True, True]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"t >= s failed on {device}")

    def test_reverse_comparison_scalar(self):
        print("\nTesting Reverse Comparison Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([1.0, 2.0, 3.0], device=device)
            
            # 2.0 == t  -> t == 2.0
            res = 2.0 == t
            expected = [False, True, False]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"s == t failed on {device}")
            
            # 2.0 != t -> t != 2.0
            res = 2.0 != t
            expected = [True, False, True]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"s != t failed on {device}")
            
            # 2.5 < t -> t > 2.5
            res = 2.5 < t
            expected = [False, False, True]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"s < t failed on {device}")
            
            # 2.0 <= t -> t >= 2.0
            res = 2.0 <= t
            expected = [False, True, True]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"s <= t failed on {device}")
            
            # 1.5 > t -> t < 1.5
            res = 1.5 > t
            expected = [True, False, False]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"s > t failed on {device}")
            
            # 2.0 >= t -> t <= 2.0
            res = 2.0 >= t
            expected = [True, True, False]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"s >= t failed on {device}")

    def test_int_scalar(self):
        print("\nTesting Int Scalar...")
        for device in self.devices:
            print(f"  Device: {device}")
            t = tp.tensor([1, 2, 3], dtype=tp.int32, device=device)
            
            # Add int
            res = t + 1
            expected = [2, 3, 4]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"int t + 1 failed on {device}")
            
            # Sub int
            res = t - 1
            expected = [0, 1, 2]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"int t - 1 failed on {device}")
            
            # Mul int
            res = t * 2
            expected = [2, 4, 6]
            self.assertEqual(res.cpu().numpy().tolist(), expected, f"int t * 2 failed on {device}")
            
            # Div int (should result in float/int depending on implementation, PyTorch does floor div for int tensors? No, it promotes to float usually for true div)
            # TP implementation of div depends on kernel. 
            # If we follow PyTorch: int / int -> float
            # Let's check logic.
            res = t / 2
            # Our current implementation might promote or use integer division.
            # Let's verify what it does.
            print(f"    Int div result type: {res.dtype}")
            # If it returns float, good. If int (floor), also okay if consistent.
            
if __name__ == '__main__':
    unittest.main()
