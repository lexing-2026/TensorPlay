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

    def _check_reentrant_grad(self, device):
        from tensorplay.autograd import enable_grad, Function

        class ReentrantGrad(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                with enable_grad():
                    # gradient of a dependent graph computed inside backward
                    (g,) = tp.autograd.grad((x * x).sum(), x)
                return grad * g

        x = tp.rand(8, device=device, requires_grad=True)
        ReentrantGrad.apply(x).sum().backward()
        self.assertEqual(x.grad.shape, (8,))

    def test_reentrant_grad_inside_backward(self):
        # backward runs with grad mode off; the nested graph needs it on.
        # The engine must run the nested graph on a queue the current thread
        # drains itself instead of parking it on a busy device queue.
        self._check_reentrant_grad(tp.device("cpu"))

    def test_reentrant_grad_inside_backward_cuda(self):
        if not tp.cuda.is_available():
            self.skipTest("CUDA unavailable")
        # top-level backward parks on the device worker; the nested grad()
        # call runs on that worker, so a device-queue enqueue deadlocks.
        self._check_reentrant_grad(tp.device("cuda", 0))


class TestConversion(unittest.TestCase):
    def test_records_only_under_grad_mode(self):
        x = tp.randn(3, requires_grad=True)
        with tp.no_grad():
            self.assertFalse(x.double().requires_grad)
        with tp.inference_mode():
            self.assertFalse(x.double().requires_grad)
        self.assertIsNotNone(x.double().grad_fn)

    def test_unchanged_conversion_is_the_input_and_a_copy_is_recorded(self):
        x = tp.randn(3, requires_grad=True)
        self.assertIs(x.to(tp.float32), x)
        copied = x.to(tp.float32, copy=True)
        self.assertIsNotNone(copied.grad_fn)
        copied.sum().backward()
        self.assertEqual(x.grad.tolist(), [1.0, 1.0, 1.0])

    def test_integer_result_is_outside_the_graph(self):
        x = tp.randn(3, requires_grad=True)
        self.assertFalse(x.to(tp.int64).requires_grad)

    def test_gradient_of_the_gradient(self):
        x = tp.tensor([0.5, -1.0, 2.0], dtype=tp.float64, requires_grad=True)
        g, = tp.autograd.grad((x.float() ** 3).sum(), x, create_graph=True)
        self.assertEqual(g.dtype, tp.float64)
        gg, = tp.autograd.grad(g.sum(), x)
        self.assertEqual(gg.dtype, tp.float64)
        for got, want in zip(gg.tolist(), [3.0, -6.0, 12.0]):
            self.assertAlmostEqual(got, want, places=5)

    def test_real_source_takes_the_real_part_of_a_complex_gradient(self):
        x = tp.randn(3, requires_grad=True)
        (x.to(tp.complex64) * (1 + 2j)).real.sum().backward()
        self.assertEqual(x.grad.dtype, tp.float32)
        self.assertEqual(x.grad.tolist(), [1.0, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
