import unittest
import tensorplay as tp
from tensorplay.autograd.gradcheck import (
    GradcheckError,
    gradcheck,
    gradgradcheck,
)


class TestGradcheck(unittest.TestCase):
    def test_correct_function_passes(self):
        def f(x):
            return (x * x).sum()

        x = tp.tensor([1.0, -2.0, 3.0], dtype=tp.float64, requires_grad=True)
        self.assertTrue(gradcheck(f, x))

    def test_wrong_backward_raises(self):
        class BadPow(tp.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                ctx.save_for_backward(inp)
                return inp.pow(2)

            @staticmethod
            def backward(ctx, grad_output):
                (inp,) = ctx.saved_tensors
                # Wrong: derivative of x^2 is 2x, not 3x.
                return grad_output * 3 * inp

        x = tp.tensor([1.5], dtype=tp.float64, requires_grad=True)
        with self.assertRaises(GradcheckError):
            gradcheck(lambda t: BadPow.apply(t), x)

    def test_raise_exception_false_returns_bool(self):
        class BadMul(tp.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 3  # wrong: should be 2

        x = tp.tensor([1.0], dtype=tp.float64, requires_grad=True)
        self.assertFalse(gradcheck(lambda t: BadMul.apply(t), x, raise_exception=False))
        self.assertTrue(
            gradcheck(lambda t: tp.mul(t, 2.0), x, raise_exception=False)
        )

    def test_no_differentiable_input_raises_value_error(self):
        x = tp.tensor([1.0], dtype=tp.float64)  # requires_grad=False
        with self.assertRaises(ValueError):
            gradcheck(lambda t: t * 2, x)

    def test_stride_zero_input_rejected(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64)
        expanded = x.as_strided((2, 4), (0, 1))
        expanded.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            gradcheck(lambda t: t.sum(), expanded)

    def test_undefined_grad_check(self):
        # check_undefined_grad is on by default and must pass for builtins.
        x = tp.tensor([1.0, 2.0], dtype=tp.float64, requires_grad=True)
        self.assertTrue(gradcheck(lambda t: (t * t).sum(), x, check_undefined_grad=True))

    def test_unsupported_flags_raise_not_implemented(self):
        x = tp.tensor([1.0], dtype=tp.float64, requires_grad=True)

        with self.assertRaises(NotImplementedError):
            gradcheck(lambda t: t.sum(), x, check_forward_ad=True)
        with self.assertRaises(NotImplementedError):
            gradcheck(lambda t: t.sum(), x, check_batched_grad=True)
        with self.assertRaises(NotImplementedError):
            gradcheck(lambda t: t.sum(), x, fast_mode=True)


class TestGradgradcheck(unittest.TestCase):
    def test_quadratic_passes(self):
        def f(x):
            return (x * x).sum()

        x = tp.tensor([1.0, 2.0], dtype=tp.float64, requires_grad=True)
        self.assertTrue(gradgradcheck(f, x))

    def test_custom_cubic_function_second_derivative_passes(self):
        class Cubic(tp.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                ctx.save_for_backward(inp)
                return inp.pow(3)

            @staticmethod
            def backward(ctx, grad_output):
                (inp,) = ctx.saved_tensors
                return grad_output * 3 * inp * inp

        x = tp.tensor([1.5], dtype=tp.float64, requires_grad=True)
        # The backward formula is linear in grad_output, so double backward
        # (6x here) is consistent and gradgradcheck passes.
        self.assertTrue(gradgradcheck(lambda t: Cubic.apply(t).sum(), x))

    def test_exp_passes(self):
        x = tp.rand(3, dtype=tp.float64, requires_grad=True)
        self.assertTrue(gradgradcheck(lambda t: t.exp().sum(dim=0), x))


if __name__ == "__main__":
    unittest.main()
