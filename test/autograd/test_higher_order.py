import unittest
import tensorplay as tp

class TestHigherOrderGrad(unittest.TestCase):
    def test_rosenbrock_optimization(self):
        # Rosenbrock function: f(x, y) = (a - x)^2 + b * (y - x^2)^2
        # Minimum is at (a, a^2)
        # Standard parameters: a = 1, b = 100
        # Min at (1, 1)
        
        a = 1.0
        b = 100.0
        
        # Initial guess
        x_val = -1.0
        y_val = 1.0
        
        x = tp.Tensor([x_val], requires_grad=True)
        y = tp.Tensor([y_val], requires_grad=True)
        
        # Optimization loop using Newton's method
        # x_{n+1} = x_n - H^{-1} * grad
        
        for i in range(10): # 10 iterations should be enough for Newton's method on quadratic-like
            # Forward pass
            term1 = (a - x) * (a - x)
            term2 = b * (y - x * x) * (y - x * x)
            loss = term1 + term2
            
            # First order gradients
            # We need create_graph=True to compute Hessian later
            grads = tp.autograd.grad([loss], [x, y], create_graph=True)
            dx = grads[0]
            dy = grads[1]
            
            # Compute Hessian
            # H = [[d2f/dx2, d2f/dxdy], [d2f/dydx, d2f/dy2]]
            
            # Row 1: grad(dx) w.r.t x, y
            hessian_row1 = tp.autograd.grad([dx], [x, y], retain_graph=True)
            d2x_dx2 = hessian_row1[0]
            d2x_dxdy = hessian_row1[1] # should be same as d2x_dydx
            
            # Row 2: grad(dy) w.r.t x, y
            hessian_row2 = tp.autograd.grad([dy], [x, y], retain_graph=False) # Last use of graph
            d2x_dydx = hessian_row2[0]
            d2x_dy2 = hessian_row2[1]
            
            # Construct Hessian matrix and gradient vector manually
            # H = [[H11, H12], [H21, H22]]
            # det(H) = H11*H22 - H12*H21
            # inv(H) = 1/det * [[H22, -H12], [-H21, H11]]
            
            H11 = d2x_dx2.item()
            H12 = d2x_dxdy.item()
            H21 = d2x_dydx.item()
            H22 = d2x_dy2.item()
            
            det = H11 * H22 - H12 * H21
            if abs(det) < 1e-6:
                print(f"Singular Hessian at iter {i}, adding regularization")
                det = 1e-6 # Simple regularization
            
            invDet = 1.0 / det
            
            invH11 = invDet * H22
            invH12 = invDet * -H12
            invH21 = invDet * -H21
            invH22 = invDet * H11
            
            # Update step: delta = -invH * grad
            g1 = dx.item()
            g2 = dy.item()
            
            delta_x = -(invH11 * g1 + invH12 * g2)
            delta_y = -(invH21 * g1 + invH22 * g2)
            
            # Update variables
            # We must update data directly or use no_grad to avoid tracking update step
            # Since we don't have with no_grad() yet (maybe?), we just create new leaf tensors
            # Actually, standard way is x.data.add_(...) or similar. 
            # TPX Tensor has item() and construction from data.
            
            new_x = x.item() + delta_x
            new_y = y.item() + delta_y
            
            x = tp.Tensor([new_x], requires_grad=True)
            y = tp.Tensor([new_y], requires_grad=True)
            
            print(f"Iter {i}: Loss={loss.item():.6f}, x={new_x:.6f}, y={new_y:.6f}")
            
            if loss.item() < 1e-6:
                break
                
        self.assertTrue(abs(x.item() - 1.0) < 1e-3)
        self.assertTrue(abs(y.item() - 1.0) < 1e-3)

if __name__ == '__main__':
    unittest.main()
