import os
import sys
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorplay as tp

class TestCUDAOps(unittest.TestCase):
    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = 'cuda'

    def test_arithmetic_ops(self):
        print("\nTesting CUDA arithmetic operators...")
        device = self.device
        
        a = tp.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        b = tp.tensor([0.5, 1.0, 2.0, 4.0], device=device)
        scalar = 2.0
        
        # --- Tensor-Tensor ---
        # Add
        res = a + b
        expected = [1.5, 3.0, 5.0, 8.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Add failed")
        
        # Sub
        res = a - b
        expected = [0.5, 1.0, 1.0, 0.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Sub failed")
        
        # Mul
        res = a * b
        expected = [0.5, 2.0, 6.0, 16.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Mul failed")
        
        # Div
        res = a / b
        expected = [2.0, 2.0, 1.5, 1.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Div failed")
        
        # --- Tensor-Scalar ---
        # Add scalar
        res = a + scalar
        expected = [3.0, 4.0, 5.0, 6.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Add scalar failed")
        
        # Sub scalar
        res = a - scalar
        expected = [-1.0, 0.0, 1.0, 2.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Sub scalar failed")
        
        # Mul scalar
        res = a * scalar
        expected = [2.0, 4.0, 6.0, 8.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Mul scalar failed")
        
        # Div scalar
        res = a / scalar
        expected = [0.5, 1.0, 1.5, 2.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "Div scalar failed")

    def test_inplace_ops(self):
        print("\nTesting CUDA inplace operators...")
        device = self.device
        a = tp.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        b = tp.tensor([0.5, 1.0, 2.0, 4.0], device=device)
        scalar = 2.0

        # --- Inplace Tensor-Tensor ---
        # Add_
        t = a.clone()
        t += b
        expected = [1.5, 3.0, 5.0, 8.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Add_ failed")
        
        # Sub_
        t = a.clone()
        t -= b
        expected = [0.5, 1.0, 1.0, 0.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Sub_ failed")
        
        # Mul_
        t = a.clone()
        t *= b
        expected = [0.5, 2.0, 6.0, 16.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Mul_ failed")
        
        # Div_
        t = a.clone()
        t /= b
        expected = [2.0, 2.0, 1.5, 1.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Div_ failed")
        
        # --- Scalar Inplace ---
        # Add_ scalar
        t = a.clone()
        t += scalar
        expected = [3.0, 4.0, 5.0, 6.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Add_ scalar failed")
        
        # Sub_ scalar
        t = a.clone()
        t -= scalar
        expected = [-1.0, 0.0, 1.0, 2.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Sub_ scalar failed")

        # Mul_ scalar
        t = a.clone()
        t *= scalar
        expected = [2.0, 4.0, 6.0, 8.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Mul_ scalar failed")

        # Div_ scalar
        t = a.clone()
        t /= scalar
        expected = [0.5, 1.0, 1.5, 2.0]
        self.assertTrue(tp.allclose(t.cpu(), tp.tensor(expected)), "Div_ scalar failed")

    def test_embedding(self):
        print("\nTesting CUDA embedding...")
        device = self.device
        embedding_dim = 3
        num_embeddings = 10
        
        weight_cpu = tp.zeros([num_embeddings, embedding_dim])
        for i in range(num_embeddings):
            for j in range(embedding_dim):
                weight_cpu[i, j] = i * 1.0
                
        weight = weight_cpu.to(device)
        indices = tp.tensor([1, 3, 5], dtype=tp.int64, device=device)
        out = tp.embedding(weight, indices)
        
        expected_out = [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]
        self.assertTrue(tp.allclose(out.cpu(), tp.tensor(expected_out)), "Embedding failed")

    def test_matmul(self):
        print("\nTesting CUDA matmul...")
        device = self.device
        
        # (2, 3) x (3, 2) -> (2, 2)
        a = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        b = tp.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        
        # Expected:
        # [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [1+6+15, 2+8+18] = [22, 28]
        # [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [4+15+30, 8+20+36] = [49, 64]
        
        res = tp.mm(a, b)
        expected = [[22.0, 28.0], [49.0, 64.0]]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "MM failed")
        
        res2 = tp.matmul(a, b)
        self.assertTrue(tp.allclose(res2.cpu(), tp.tensor(expected)), "Matmul failed")

    def test_activations(self):
        print("\nTesting CUDA activations...")
        device = self.device
        a = tp.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
        
        # ReLU
        res = tp.relu(a)
        expected = [0.0, 0.0, 0.0, 1.0, 2.0]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected)), "ReLU failed")
        
        # Sigmoid
        # sigmoid(0) = 0.5
        res = tp.sigmoid(a)
        expected = [1/(1+np.exp(2)), 1/(1+np.exp(1)), 0.5, 1/(1+np.exp(-1)), 1/(1+np.exp(-2))]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected), atol=1e-5), "Sigmoid failed")
        
        # Tanh
        res = tp.tanh(a)
        expected = np.tanh([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected), atol=1e-5), "Tanh failed")
        
        # SiLU (x * sigmoid(x))
        res = tp.silu(a)
        expected = [-2.0/(1+np.exp(2)), -1.0/(1+np.exp(1)), 0.0, 1.0/(1+np.exp(-1)), 2.0/(1+np.exp(-2))]
        self.assertTrue(tp.allclose(res.cpu(), tp.tensor(expected), atol=1e-5), "SiLU failed")

if __name__ == "__main__":
    unittest.main()
