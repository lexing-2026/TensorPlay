"""Backward-helper operator regression: the autograd nodes for trace,
masked_select, cummax/cummin and cumprod route through dispatcher ops
(trace_backward, masked_select_backward, cummaxmin_backward,
cumprod_backward).  Those ops previously had no kernel registered under
any key, so backward raised "Kernel not found".  These checks pin the
Composite registrations behind actual gradient computations and compare
each gradient against the analytical form.
"""

import unittest

import tensorplay as tp


class BackwardHelperOps(unittest.TestCase):
    def _check(self, grad, expected, tol=1e-6):
        flat_g = grad.reshape([-1]).tolist()
        flat_e = expected.reshape([-1]).tolist()
        self.assertEqual(len(flat_g), len(flat_e))
        for a, b in zip(flat_g, flat_e):
            self.assertAlmostEqual(a, b, delta=tol * max(1.0, abs(b)))

    def test_trace_backward(self):
        # d(tr(A))/dA = I.  trace accepts only 2-D input, matching the
        # reference contract that rejects non-matrix tensors outright.
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tp.trace(a).backward()
        self._check(a.grad, tp.eye(2))

        b = tp.tensor([[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        with self.assertRaises(RuntimeError):
            tp.trace(b)

    def test_masked_select_backward(self):
        # Selected entries receive the incoming gradient in order; the rest 0.
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        a.masked_select(a > 2.0).backward(tp.tensor([10.0, 20.0]))
        self._check(a.grad, tp.tensor([[0.0, 0.0], [10.0, 20.0]]))

    def test_cummaxmin_backward(self):
        # Gradient flows to every position that produced the running extreme;
        # a later tie accumulates on top of the earlier winner.
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        g = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        a.cummax(0)[0].backward(g)
        # Every running max sits at its own row here: elementwise pass-through.
        self._check(a.grad, tp.tensor([[1.0, 2.0], [3.0, 4.0]]))

        b = tp.tensor([[4.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b.cummin(0)[0].backward(g)
        # col0: running min 4 then 3, each row its own winner -> 1 and 3.
        # col1: min stays at row 0 for both steps -> 2 + 4 there.
        self._check(b.grad, tp.tensor([[1.0, 6.0], [3.0, 0.0]]))

    def test_cumprod_backward(self):
        # y = cumprod(x, d): dx_i = (1/x_i) * sum_{j>=i} g_j * y_j.
        a = tp.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        g = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
        a.cumprod(0).backward(g)
        # y = [[1, 2], [3, 8]]: dx_00 = 1*1 + 3*3 = 10; dx_01 = 2*2 + 4*8/2...
        self._check(a.grad, tp.tensor([[10.0, 18.0], [3.0, 8.0]]))

        # A zero in the slice routes the whole trailing sum to that position.
        z = tp.tensor([[0.0, 2.0], [3.0, 4.0]], requires_grad=True)
        z.cumprod(0).backward(g)
        self._check(z.grad, tp.tensor([[10.0, 18.0], [0.0, 8.0]]))

    def test_backward_helper_composite_registration(self):
        from tensorplay import _C
        for op in ("trace_backward", "masked_select_backward",
                   "cummaxmin_backward", "cumprod_backward"):
            table = _C._dispatch_dump(op)
            self.assertIsNotNone(table, op)
            self.assertEqual(table.get("Composite"), "registered")

    def test_logsumexp_backward_puts_the_reduced_axis_back(self):
        # d logsumexp(x, dim) / dx is the softmax of x along dim, scaled by
        # the gradient arriving for that slice -- with or without keepdim.
        for keepdim in (False, True):
            for dim in (0, 1, -1):
                x = tp.tensor(
                    [[0.5, -1.0, 2.0], [1.5, 0.0, -0.5]], dtype=tp.float64,
                    requires_grad=True,
                )
                out = tp.logsumexp(x, dim, keepdim=keepdim)
                weight = tp.arange(1.0, out.numel() + 1.0, dtype=tp.float64).reshape(out.shape)
                (grad,) = tp.autograd.grad((out * weight).sum(), [x])
                upstream = weight if keepdim else weight.unsqueeze(dim)
                self._check(grad, tp.softmax(x.detach(), dim) * upstream, tol=1e-12)
        scalar = tp.tensor(1.5, dtype=tp.float64, requires_grad=True)
        out = tp.logsumexp(scalar, 0)
        self.assertEqual(out.item(), 1.5)
        (grad,) = tp.autograd.grad(out, [scalar])
        self.assertAlmostEqual(grad.item(), 1.0, delta=1e-12)

    def test_selu_and_celu_backward_follow_their_derivatives(self):
        scale, alpha = 1.0507009873554805, 1.6732632423543772
        x = tp.tensor([-2.0, -0.5, 0.5, 2.0], dtype=tp.float64, requires_grad=True)
        (grad,) = tp.autograd.grad(tp.nn.functional.selu(x).sum(), [x])
        v = x.detach()
        self._check(grad, tp.where(v > 0, tp.full_like(v, scale), scale * alpha * v.exp()), tol=1e-12)
        for a in (0.5, 2.0):
            (grad,) = tp.autograd.grad(tp.nn.functional.celu(x, a).sum(), [x])
            self._check(grad, tp.where(v > 0, tp.ones_like(v), (v / a).exp()), tol=1e-12)

    def test_fmax_fmin_backward_follow_the_operand_returned(self):
        # The returned operand takes the gradient, a NaN loses to a number,
        # and a tie splits it.
        nan = float("nan")
        a = tp.tensor([1.0, nan, 3.0, 2.0, nan], requires_grad=True)
        b = tp.tensor([2.0, 1.0, nan, 2.0, nan], requires_grad=True)
        ga, gb = tp.autograd.grad(tp.fmax(a, b).sum(), [a, b])
        self.assertEqual(ga.tolist(), [0.0, 0.0, 1.0, 0.5, 1.0])
        self.assertEqual(gb.tolist(), [1.0, 1.0, 0.0, 0.5, 0.0])
        ga, gb = tp.autograd.grad(tp.fmin(a, b).sum(), [a, b])
        self.assertEqual(ga.tolist(), [1.0, 0.0, 1.0, 0.5, 1.0])
        self.assertEqual(gb.tolist(), [0.0, 1.0, 0.0, 0.5, 0.0])

    def test_float_power_answers_in_double_and_differentiates(self):
        base = tp.tensor([2.0, 3.0], requires_grad=True)
        exponent = tp.tensor([3.0, 2.0], requires_grad=True)
        out = tp.float_power(base, exponent)
        self.assertEqual(out.dtype, tp.float64)
        self.assertEqual(tp.float_power(tp.tensor([2, 3]), 2).dtype, tp.float64)
        gb, ge = tp.autograd.grad(out.sum(), [base, exponent])
        self.assertEqual(gb.dtype, tp.float32)
        self._check(gb, tp.tensor([12.0, 6.0]))
        self._check(ge, tp.tensor([8.0 * 0.6931471805599453, 9.0 * 1.0986122886681098]), tol=1e-6)

    def test_msort_and_take_along_dim_differentiate_on_every_device(self):
        devices = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
        for device in devices:
            y = tp.tensor([[3.0, 1.0], [1.0, 4.0], [2.0, 0.0]], device=device, requires_grad=True)
            weight = tp.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
            (grad,) = tp.autograd.grad((tp.msort(y) * weight).sum(), [y])
            # Row k of the sorted column receives weight row k.
            self._check(grad.cpu(), tp.tensor([[5.0, 4.0], [1.0, 6.0], [3.0, 2.0]]))
            index = tp.tensor([[1, 1], [0, 1], [1, 0]], device=device)
            (grad,) = tp.autograd.grad(tp.take_along_dim(y, index, 1).sum(), [y])
            self._check(grad.cpu(), tp.tensor([[0.0, 2.0], [1.0, 1.0], [1.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
