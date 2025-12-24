import unittest
import tensorplay as tp

class TestAutogradEngine(unittest.TestCase):
    def test_graph_cleanup(self):
        # Case 1: retain_graph=False (default)
        x = tp.Tensor([2.0], requires_grad=True)
        y = x * x
        
        # First backward
        y.backward()
        self.assertEqual(x.grad.item(), 4.0)
        
        # Second backward should fail or do nothing effectively because graph is cleared
        # In PyTorch this raises RuntimeError.
        # In our current implementation, clearing edges means the graph is disconnected.
        # So x.grad should NOT increase.
        
        # Reset grad to be sure
        x.grad = tp.Tensor([0.0])
        y.backward()
        
        # If graph was cleared, backward propagation stops at y.grad_fn because it has no edges.
        # So x.grad remains 0.0.
        self.assertEqual(x.grad.item(), 0.0)

    def test_retain_graph(self):
        # Case 2: retain_graph=True
        x = tp.Tensor([2.0], requires_grad=True)
        y = x * x
        
        # First backward with retain_graph
        y.backward(retain_graph=True)
        self.assertEqual(x.grad.item(), 4.0)
        
        # Second backward should work and accumulate
        y.backward() # retain_graph=False (default) implies we can consume it now
        self.assertEqual(x.grad.item(), 8.0)
        
        # Third backward should fail/do nothing
        x.grad = tp.Tensor([0.0])
        y.backward()
        self.assertEqual(x.grad.item(), 0.0)

    def test_multi_root_backward(self):
        x = tp.Tensor([2.0], requires_grad=True)
        y1 = x * x
        y2 = x * x * x
        
        # backward on both
        tp.autograd.backward([y1, y2])
        
        # grad should be dy1/dx + dy2/dx = 2x + 3x^2 = 4 + 12 = 16
        self.assertEqual(x.grad.item(), 16.0)

if __name__ == '__main__':
    unittest.main()
