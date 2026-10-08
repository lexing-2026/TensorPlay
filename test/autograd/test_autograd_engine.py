import os
import subprocess
import sys
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
        
        # The first pass freed what the graph saved, so a second one fails
        # and leaves the gradient alone.
        x.grad = tp.Tensor([0.0])
        with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
            y.backward()
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
        
        # The pass that released the graph was the last one.
        x.grad = tp.Tensor([0.0])
        with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
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

    def test_grad_of_one_input_never_evaluates_another_inputs_formula(self):
        # igamma has no derivative in its order.  Asking only for the gradient
        # in x must not reach that formula although the order requires grad
        # too; a backward() that wants both still reports it.
        a = tp.tensor([2.0], dtype=tp.float64, requires_grad=True)
        x = tp.tensor([1.0], dtype=tp.float64, requires_grad=True)
        (gx,) = tp.autograd.grad(tp.igamma(a, x).sum(), x)
        # d/dx P(2, x) = x e^-x.
        self.assertAlmostEqual(gx.item(), 0.36787944117144233, places=12)
        with self.assertRaisesRegex(RuntimeError, "first argument"):
            tp.igamma(a, x).sum().backward()

    def test_grad_of_one_factor_leaves_the_other_untouched(self):
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = tp.tensor([[0.5], [0.25]], requires_grad=True)
        (ga,) = tp.autograd.grad((a @ b).sum(), a)
        self.assertEqual(ga.tolist(), [[0.5, 0.25], [0.5, 0.25]])
        self.assertIsNone(b.grad)
        (gb,) = tp.autograd.grad((a @ b).sum(), b)
        self.assertEqual(gb.tolist(), [[4.0], [6.0]])


class TestEngineTrace(unittest.TestCase):
    def test_value_trace_prints_complex_gradients(self):
        # TP_ENGINE_TRACE=3 prints gradient values, and the level is read once
        # per process, hence the child.  A complex gradient has to print rather
        # than fail the backward it describes.
        code = (
            "import tensorplay as tp\n"
            "x = tp.ones(3, requires_grad=True)\n"
            "(x.to(tp.complex64) * (1 + 2j)).real.sum().backward()\n"
            "print(x.grad.tolist())\n"
        )
        env = dict(os.environ, TP_ENGINE_TRACE="3")
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout.strip(), "[1.0, 1.0, 1.0]")
        # RealBackward hands its input on as complex(grad, 0).
        self.assertIn("1.000000+0.000000j", proc.stderr)


if __name__ == '__main__':
    unittest.main()
