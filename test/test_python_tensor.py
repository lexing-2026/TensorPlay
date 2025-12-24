import unittest
import numpy as np
import tensorplay

class TestTensorCreation(unittest.TestCase):
    def test_from_list_float(self):
        print("\nTesting list (float)...")
        data = [1.0, 2.0, 3.0]
        t = tensorplay.tensor(data)
        print(f"Tensor: {t}")
        self.assertEqual(t.dtype, tensorplay.float32)
        self.assertEqual(t.shape, tensorplay.Size((3,)))
        self.assertAlmostEqual(t[0].item(), 1.0)
        self.assertAlmostEqual(t[1].item(), 2.0)
        self.assertAlmostEqual(t[2].item(), 3.0)

    def test_from_list_int(self):
        print("\nTesting list (int)...")
        data = [1, 2, 3]
        t = tensorplay.tensor(data)
        print(f"Tensor: {t}")
        self.assertEqual(t.dtype, tensorplay.int64)
        self.assertEqual(t.shape, tensorplay.Size((3,)))
        self.assertEqual(t[0].item(), 1)

    def test_from_nested_list(self):
        print("\nTesting nested list...")
        data = [[1, 2], [3, 4]]
        t = tensorplay.tensor(data)
        print(f"Tensor: {t}")
        self.assertEqual(t.shape, tensorplay.Size((2, 2)))
        self.assertEqual(t.dtype, tensorplay.int64)
        self.assertEqual(t[0][0].item(), 1)
        self.assertEqual(t[1][1].item(), 4)

    def test_from_numpy(self):
        print("\nTesting numpy...")
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = tensorplay.tensor(data)
        print(f"Tensor from numpy float32: {t}")
        self.assertEqual(t.shape, tensorplay.Size((2, 2)))
        self.assertEqual(t.dtype, tensorplay.float32)
        self.assertAlmostEqual(t[0][0].item(), 1.0)
        
        data_int = np.array([1, 2, 3], dtype=np.int64)
        t_int = tensorplay.tensor(data_int)
        print(f"Tensor from numpy int64: {t_int}")
        self.assertEqual(t_int.dtype, tensorplay.int64)
        
        data_int32 = np.array([1, 2, 3], dtype=np.int32)
        t_int32 = tensorplay.tensor(data_int32)
        print(f"Tensor from numpy int32: {t_int32}")
        self.assertEqual(t_int32.dtype, tensorplay.int32)

    def test_numpy_conversion(self):
        print("\nTesting numpy conversion...")
        # float64 (numpy default) -> float32
        data_double = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        t = tensorplay.tensor(data_double, dtype=tensorplay.float32)
        print(f"Tensor float32 from numpy float64: {t}")
        self.assertEqual(t.dtype, tensorplay.float32)
        self.assertAlmostEqual(t[0].item(), 1.1, places=4)
        
        # int64 -> float32
        data_int = np.array([1, 2, 3], dtype=np.int64)
        t2 = tensorplay.tensor(data_int, dtype=tensorplay.float32)
        print(f"Tensor float32 from numpy int64: {t2}")
        self.assertEqual(t2.dtype, tensorplay.float32)
        self.assertEqual(t2[0].item(), 1.0)

    def test_dtype_override(self):
        print("\nTesting dtype override...")
        data = [1, 2, 3]
        t = tensorplay.tensor(data, dtype=tensorplay.float32)
        print(f"Tensor with float32 override: {t}")
        self.assertEqual(t.dtype, tensorplay.float32)
        self.assertAlmostEqual(t[0].item(), 1.0)

    def test_irregular_list(self):
        print("\nTesting irregular list...")
        data = [[1, 2], [3]]
        with self.assertRaises(ValueError): # utils.h throws runtime_error for irregular list
            tensorplay.tensor(data)
        print("Caught expected error for irregular list")

    def test_scalar_creation(self):
        print("\nTesting scalar creation...")
        t = tensorplay.tensor(3.14)
        print(f"Scalar tensor: {t}")
        self.assertEqual(t.dim(), 0)
        self.assertAlmostEqual(t.item(), 3.14, places=4)
        
        t = tensorplay.tensor(42)
        print(f"Scalar int tensor: {t}")
        self.assertEqual(t.dim(), 0)
        self.assertEqual(t.item(), 42)

if __name__ == '__main__':
    unittest.main()
