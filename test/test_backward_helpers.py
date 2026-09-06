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


if __name__ == "__main__":
    unittest.main()
