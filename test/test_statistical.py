import torch
import numpy as np
import unittest
import tensorplay as tp

class TestStatisticalOps(unittest.TestCase):
    def setUp(self):
        self.device = tp.device("cpu")

    def assertTensorClose(self, tp_tensor, torch_tensor, rtol=1e-5, atol=1e-8):
        self.assertEqual(tp_tensor.shape, list(torch_tensor.shape))
        self.assertEqual(str(tp_tensor.dtype), str(torch_tensor.dtype).replace("torch.", "tensorplay."))
        
        tp_np = tp_tensor.numpy()
        torch_np = torch_tensor.detach().cpu().numpy()
        
        np.testing.assert_allclose(tp_np, torch_np, rtol=rtol, atol=atol)

    def test_var_std(self):
        shapes = [(10,), (5, 5), (2, 3, 4)]
        for shape in shapes:
            print(f"Testing shape {shape}")
            data = torch.randn(shape, dtype=torch.float32)
            tp_tensor = tp.tensor(data.numpy())
            
            # Test var()
            print("  Testing var()")
            self.assertTensorClose(tp_tensor.var(), data.var())
            print("  Testing var(correction=0)")
            self.assertTensorClose(tp_tensor.var(correction=0), data.var(correction=0))
            
            # Test std()
            print("  Testing std()")
            self.assertTensorClose(tp_tensor.std(), data.std())
            print("  Testing std(correction=0)")
            self.assertTensorClose(tp_tensor.std(correction=0), data.std(correction=0))
            
            if len(shape) > 1:
                # Test var(dim)
                print("  Testing var(dim=0)")
                self.assertTensorClose(tp_tensor.var(dim=0), data.var(dim=0))
                print("  Testing var(dim=1, keepdim=True)")
                self.assertTensorClose(tp_tensor.var(dim=1, keepdim=True), data.var(dim=1, keepdim=True))
                
                # Test std(dim)
                print("  Testing std(dim=0)")
                self.assertTensorClose(tp_tensor.std(dim=0), data.std(dim=0))
                print("  Testing std(dim=1, keepdim=True)")
                self.assertTensorClose(tp_tensor.std(dim=1, keepdim=True), data.std(dim=1, keepdim=True))

    def test_norm(self):
        shapes = [(10,), (5, 5)]
        ps = [1.0, 2.0, float('inf'), float('-inf'), 3.0]
        
        for shape in shapes:
            data = torch.randn(shape, dtype=torch.float32)
            tp_tensor = tp.tensor(data.numpy())
            
            for p in ps:
                # Test norm(p)
                self.assertTensorClose(tp_tensor.norm(p=p), data.norm(p=p))
                
                if len(shape) > 1:
                    # Test norm(dim, p)
                    self.assertTensorClose(tp_tensor.norm(dim=[0], p=p), data.norm(dim=0, p=p))
                    self.assertTensorClose(tp_tensor.norm(dim=[1], p=p, keepdim=True), data.norm(dim=1, p=p, keepdim=True))

    def test_max_min(self):
        shapes = [(10,), (5, 5), (2, 3, 4)]
        for shape in shapes:
            data = torch.randn(shape, dtype=torch.float32)
            tp_tensor = tp.tensor(data.numpy())
            
            # Test max() / min() global
            self.assertTensorClose(tp_tensor.max(), data.max())
            self.assertTensorClose(tp_tensor.min(), data.min())
            
            if len(shape) > 1:
                # Check the schema for the max.dim return type.
                # - func: max.dim(Tensor self, int64_t[] dim, bool keepdim=false) -> Tensor
                
                # So we should compare with data.amax(dim=...) or data.max(dim=...).values
                
                self.assertTensorClose(tp_tensor.max(dim=0)[0], data.amax(dim=0))
                self.assertTensorClose(tp_tensor.min(dim=0, keepdim=True)[0], data.amin(dim=0, keepdim=True))

    def test_amax_amin_nan_propagates(self):
        data = torch.tensor([1.0, float("nan"), 3.0])
        x = tp.tensor(data.numpy())
        self.assertTensorClose(tp.amax(x), torch.amax(data))
        self.assertTensorClose(tp.amin(x), torch.amin(data))
        mn, mx = tp.aminmax(x)
        self.assertTensorClose(mn, torch.aminmax(data).min)
        self.assertTensorClose(mx, torch.aminmax(data).max)
        # NaN position must not matter (first-NaN wins under strict >)
        data2 = torch.tensor([float("nan"), 1.0, float("nan"), -2.0])
        x2 = tp.tensor(data2.numpy())
        self.assertTensorClose(tp.amax(x2), torch.amax(data2))
        self.assertTensorClose(tp.amin(x2), torch.amin(data2))
        # dim-reduction variant on a NaN-free column must stay exact
        m = torch.tensor([[1.0, float("nan")], [5.0, 2.0]])
        mm = tp.tensor(m.numpy())
        self.assertTensorClose(tp.amax(mm, dim=1), torch.amax(m, dim=1))

    def test_pow(self):
        shape = (5, 5)
        data = torch.abs(torch.randn(shape, dtype=torch.float32)) + 0.1 # Ensure positive for fractional pow
        tp_tensor = tp.tensor(data.numpy())
        
        exponents = [2.0, 0.5, 3, -1.0]
        for exp in exponents:
            self.assertTensorClose(tp_tensor.pow(exp), data.pow(exp))

    def test_sqrt_abs(self):
        shape = (5, 5)
        data = torch.randn(shape, dtype=torch.float32)
        tp_tensor = tp.tensor(data.numpy())
        
        # Test abs
        self.assertTensorClose(tp_tensor.abs(), data.abs())
        
        # Test sqrt (ensure positive)
        data_pos = data.abs() + 0.1
        tp_tensor_pos = tp.tensor(data_pos.numpy())
        self.assertTensorClose(tp_tensor_pos.sqrt(), data_pos.sqrt())

if __name__ == '__main__':
    unittest.main()
