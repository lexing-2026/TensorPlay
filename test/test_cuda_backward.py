import tensorplay as tp
import unittest
import numpy as np

class TestCUDABackward(unittest.TestCase):
    def setUp(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = tp.device("cuda:0")

    def test_add_backward(self):
        x = tp.ones([2, 2], device=self.device, requires_grad=True)
        y = tp.ones([2, 2], device=self.device, requires_grad=True)
        z = x + y
        loss = z.sum()
        loss.backward()
        
        # dloss/dz = 1
        # dz/dx = 1, dz/dy = 1
        # dloss/dx = 1, dloss/dy = 1
        
        self.assertTrue(x.grad is not None)
        self.assertTrue(y.grad is not None)
        
        # Check values on CPU
        x_grad = x.grad.cpu().numpy()
        y_grad = y.grad.cpu().numpy()
        
        np.testing.assert_allclose(x_grad, np.ones((2, 2)))
        np.testing.assert_allclose(y_grad, np.ones((2, 2)))

    def test_sub_backward(self):
        x = tp.ones([2], device=self.device, requires_grad=True)
        y = tp.ones([2], device=self.device, requires_grad=True)
        z = x - y
        loss = z.sum()
        loss.backward()
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2,)))
        np.testing.assert_allclose(y.grad.cpu().numpy(), -np.ones((2,)))

    def test_mul_backward(self):
        x = tp.full([2], 2.0, device=self.device, requires_grad=True)
        y = tp.full([2], 3.0, device=self.device, requires_grad=True)
        z = x * y
        loss = z.sum()
        loss.backward()
        
        # z = x * y
        # dz/dx = y = 3
        # dz/dy = x = 2
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.full((2,), 3.0))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.full((2,), 2.0))

    def test_div_backward(self):
        x = tp.full([2], 6.0, device=self.device, requires_grad=True)
        y = tp.full([2], 3.0, device=self.device, requires_grad=True)
        z = x / y
        loss = z.sum()
        loss.backward()
        
        # z = x / y
        # dz/dx = 1/y = 1/3
        # dz/dy = -x/y^2 = -6/9 = -2/3
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.full((2,), 1.0/3.0))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.full((2,), -2.0/3.0))

    def test_reshape_backward(self):
        x = tp.ones([2, 3], device=self.device, requires_grad=True)
        y = x.reshape([3, 2])
        loss = y.sum()
        loss.backward()
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 3)))

    def test_permute_backward(self):
        x = tp.randn([2, 3], device=self.device, requires_grad=True)
        y = x.permute([1, 0])
        z = y.permute([1, 0]) # Back to original shape
        loss = (z - x).sum()
        loss.backward()
        
        # z - x = 0 (numerically)
        # But logically, dz/dx = d(y_permuted)/dx - dx/dx
        # Let's do a simpler one: loss = y.sum()
        # y = x.T
        # loss = sum(x.T) = sum(x)
        # dloss/dx = 1
        
        x.grad = None
        y = x.permute([1, 0])
        loss = y.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 3)))

    def test_broadcast_add(self):
        x = tp.ones([2, 3], device=self.device, requires_grad=True)
        y = tp.ones([3], device=self.device, requires_grad=True)
        z = x + y # Broadcasting y to [2, 3]
        loss = z.sum()
        loss.backward()
        
        # z_ij = x_ij + y_j
        # loss = sum(x_ij + y_j)
        # dloss/dx_ij = 1
        # dloss/dy_j = sum_i(1) = 2
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 3)))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.full((3,), 2.0))

    def test_broadcast_sub(self):
        x = tp.ones([2, 3], device=self.device, requires_grad=True)
        y = tp.ones([3], device=self.device, requires_grad=True)
        z = x - y 
        loss = z.sum()
        loss.backward()
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.ones((2, 3)))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.full((3,), -2.0))

    def test_broadcast_mul(self):
        x = tp.full([2, 3], 2.0, device=self.device, requires_grad=True)
        y = tp.full([3], 3.0, device=self.device, requires_grad=True)
        z = x * y
        loss = z.sum()
        loss.backward()
        
        # z_ij = x_ij * y_j
        # dz/dx_ij = y_j = 3
        # dz/dy_j = sum_i(x_ij) = 2*2 = 4
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.full((2, 3), 3.0))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.full((3,), 4.0))

    def test_broadcast_div(self):
        x = tp.full([2, 3], 6.0, device=self.device, requires_grad=True)
        y = tp.full([3], 3.0, device=self.device, requires_grad=True)
        z = x / y
        loss = z.sum()
        loss.backward()
        
        # z_ij = x_ij / y_j
        # dz/dx_ij = 1/y_j = 1/3
        # dz/dy_j = sum_i(-x_ij/y_j^2) = 2 * (-6/9) = 2 * (-2/3) = -4/3
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.full((2, 3), 1.0/3.0))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.full((3,), -4.0/3.0))

    def test_matmul_backward(self):
        x = tp.eye(2, device=self.device, requires_grad=True)
        y = tp.full([2, 2], 2.0, device=self.device, requires_grad=True)
        z = x.matmul(y) # [[2, 2], [2, 2]]
        loss = z.sum()
        loss.backward()
        
        # z = x @ y
        # dloss/dx = dloss/dz @ y.T = 1 @ y.T
        # dloss/dy = x.T @ dloss/dz = x.T @ 1
        
        # grad_output is ones(2,2)
        # dx = ones(2,2) @ y.T = [[1,1],[1,1]] @ [[2,2],[2,2]] = [[4,4],[4,4]]
        # dy = x.T @ ones(2,2) = I @ [[1,1],[1,1]] = [[1,1],[1,1]]
        
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.full((2, 2), 4.0))
        np.testing.assert_allclose(y.grad.cpu().numpy(), np.ones((2, 2)))

if __name__ == "__main__":
    unittest.main()
