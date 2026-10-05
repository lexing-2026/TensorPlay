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

    def test_views_and_selections_differentiate_on_every_device(self):
        devices = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
        for device in devices:
            x = tp.arange(12.0, device=device).reshape(3, 4).requires_grad_(True)
            (grad,) = tp.autograd.grad(tp.hsplit(x, 2)[1].sum() + 2 * tp.vsplit(x, 3)[0].sum(), [x])
            self._check(grad.cpu(), tp.tensor([[2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]))
            row = tp.tensor([[1.0, 2.0, 3.0]], device=device, requires_grad=True)
            weight = tp.arange(12.0, device=device).reshape(4, 3)
            (grad,) = tp.autograd.grad((tp.broadcast_to(row, (4, 3)) * weight).sum(), [row])
            self._check(grad.cpu(), tp.tensor([[18.0, 22.0, 26.0]]))
            vec = tp.tensor([1.0, 2.0], device=device, requires_grad=True)
            (grad,) = tp.autograd.grad((tp.diag(vec, 1) * tp.arange(9.0, device=device).reshape(3, 3)).sum(), [vec])
            self._check(grad.cpu(), tp.tensor([1.0, 5.0]))
            (grad,) = tp.autograd.grad(tp.diag(x, -1).sum(), [x])
            self._check(grad.cpu(), tp.tensor([[0.0] * 4, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            m = tp.tensor([[3.0, 1.0, 2.0], [float("nan"), 5.0, 4.0]], device=device, requires_grad=True)
            (grad,) = tp.autograd.grad(tp.median(m[:1], 1)[0].sum() + tp.nanmedian(m, 1)[0].sum(), [m])
            # Row 0's median is 2 (both calls); row 1 ignores the NaN: median of {5, 4} is 4.
            self._check(grad.cpu(), tp.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]))

    def test_distances_differentiate_for_every_norm(self):
        # d/dx1 ||x1 - x2||_p for each pair, summed over the pairs a row is in.
        devices = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
        a_rows = [[0.0, 0.0], [3.0, 1.0]]
        b_rows = [[1.0, 2.0]]
        for device in devices:
            for p, expected in ((1.0, [[-1.0, -1.0], [1.0, -1.0]]),
                                (2.0, [[-0.4472135954999579, -0.8944271909999159],
                                       [0.8944271909999159, -0.4472135954999579]]),
                                (float("inf"), [[0.0, -1.0], [1.0, 0.0]])):
                a = tp.tensor(a_rows, dtype=tp.float64, device=device, requires_grad=True)
                b = tp.tensor(b_rows, dtype=tp.float64, device=device, requires_grad=True)
                ga, gb = tp.autograd.grad(tp.cdist(a, b, p).sum(), [a, b])
                self._check(ga.cpu(), tp.tensor(expected, dtype=tp.float64), tol=1e-12)
                self._check(gb.cpu(), -tp.tensor(expected, dtype=tp.float64).sum(0, keepdim=True), tol=1e-12)
            pts = tp.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]], dtype=tp.float64, device=device, requires_grad=True)
            (grad,) = tp.autograd.grad(tp.pdist(pts).sum(), [pts])
            # Pairs (0,1): 5, (0,2): 4, (1,2): 3; each row gathers unit vectors away from the others.
            self._check(grad.cpu(), tp.tensor([[-0.6, -1.8], [1.6, 0.8], [-1.0, 1.0]], dtype=tp.float64), tol=1e-12)

    def test_results_asked_for_in_another_type_still_differentiate(self):
        # A reduction or product answering in a requested type, and a
        # similarity of operands of two types, pass their gradients back to
        # each operand in its own type.
        ops = tp.ops.tp
        x = tp.tensor([[1.0, 2.0, 2.0], [3.0, 0.0, 4.0]], requires_grad=True)
        (grad,) = tp.autograd.grad(ops.norm.ScalarOpt_dim_dtype(x, 2, [1], False, dtype=tp.float64).sum(), [x])
        self.assertEqual(grad.dtype, tp.float32)
        self._check(grad, tp.tensor([[1 / 3, 2 / 3, 2 / 3], [0.6, 0.0, 0.8]]))
        (grad,) = tp.autograd.grad(tp.prod(x[:, :2] + 1, 1, dtype=tp.float64).sum(), [x])
        self._check(grad, tp.tensor([[3.0, 2.0, 0.0], [1.0, 4.0, 0.0]]))
        y = tp.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=tp.float64, requires_grad=True)
        gx, gy = tp.autograd.grad(tp.nn.functional.cosine_similarity(x, y).sum(), [x, y])
        self.assertEqual((gx.dtype, gy.dtype), (tp.float32, tp.float64))
        # Row 1: x = (3, 0, 4), y along the third axis: d cos/dx = (y/|y| - cos x/|x|)/|x|.
        self._check(gx[1], tp.tensor([-0.096, 0.0, 0.072]), tol=1e-6)

    def test_powers_of_two_differentiate_in_value_and_exponent(self):
        ln2 = 0.6931471805599453
        f64 = tp.float64
        for device in ["cpu"] + (["cuda"] if tp.cuda.is_available() else []):
            e = tp.tensor([0.0, 1.0, -1.0], dtype=f64, device=device, requires_grad=True)
            (grad,) = tp.autograd.grad(tp.pow(2.0, e).sum(), [e])
            self._check(grad.cpu(), tp.tensor([ln2, 2 * ln2, ln2 / 2], dtype=f64), tol=1e-12)
            # A zero base gives nothing where the exponent is non-negative.
            e = tp.tensor([0.0, 2.0], dtype=f64, device=device, requires_grad=True)
            (grad,) = tp.autograd.grad(tp.pow(0.0, e).sum(), [e])
            self._check(grad.cpu(), tp.tensor([0.0, 0.0], dtype=f64))
            # ldexp(x, k) = x * 2^k.
            x = tp.tensor([3.0, -1.5], dtype=f64, device=device, requires_grad=True)
            k = tp.tensor([2.0, -1.0], dtype=f64, device=device, requires_grad=True)
            gx, gk = tp.autograd.grad(tp.ldexp(x, k).sum(), [x, k])
            self._check(gx.cpu(), tp.tensor([4.0, 0.5], dtype=f64), tol=1e-12)
            self._check(gk.cpu(), tp.tensor([12 * ln2, -0.75 * ln2], dtype=f64), tol=1e-12)
            # The power is taken in the type the pair promotes to.
            wide = tp.ldexp(tp.ones(2, device=device), tp.ones(2, dtype=f64, device=device))
            self.assertEqual(wide.dtype, f64)

    def test_incomplete_gamma_differentiates_in_its_second_argument(self):
        f64 = tp.float64
        for device in ["cpu"] + (["cuda"] if tp.cuda.is_available() else []):
            a = tp.tensor([2.0, 3.0], dtype=f64, device=device)
            x = tp.tensor([1.0, 2.0], dtype=f64, device=device, requires_grad=True)
            # d/dx P(a, x) = x^(a-1) e^-x / Gamma(a); the complement is its negative.
            density = tp.tensor([0.36787944117144233, 0.2706705664732254], dtype=f64)
            (grad,) = tp.autograd.grad(tp.igamma(a, x).sum(), [x])
            self._check(grad.cpu(), density, tol=1e-12)
            (grad,) = tp.autograd.grad(tp.igammac(a, x).sum(), [x])
            self._check(grad.cpu(), -density, tol=1e-12)
            # The first argument has no derivative: asking for it fails.
            a.requires_grad_(True)
            with self.assertRaises(NotImplementedError):
                tp.autograd.grad(tp.igamma(a, x).sum(), [a])

    def test_tensor_bounds_share_the_gradient_at_a_tie(self):
        for device in ["cpu"] + (["cuda"] if tp.cuda.is_available() else []):
            for clamp in (tp.clamp, tp.clip):
                x = tp.tensor([-2.0, 0.0, 0.5, 1.0, 3.0], device=device, requires_grad=True)
                lo = tp.tensor([-1.0, 0.0, -1.0, -1.0, -1.0], device=device, requires_grad=True)
                hi = tp.ones(5, device=device, requires_grad=True)
                gx, glo, ghi = tp.autograd.grad(clamp(x, lo, hi).sum(), [x, lo, hi])
                # Below, on the lower bound, inside, on the upper bound, above.
                self._check(gx.cpu(), tp.tensor([0.0, 0.5, 1.0, 0.5, 0.0]))
                self._check(glo.cpu(), tp.tensor([1.0, 0.5, 0.0, 0.0, 0.0]))
                self._check(ghi.cpu(), tp.tensor([0.0, 0.0, 0.0, 0.5, 1.0]))

    def test_other_spellings_differentiate_like_the_operator_they_name(self):
        f64 = tp.float64
        for device in ["cpu"] + (["cuda"] if tp.cuda.is_available() else []):
            y = tp.tensor([1.0, -2.0], dtype=f64, device=device, requires_grad=True)
            x = tp.tensor([1.0, 0.5], dtype=f64, device=device, requires_grad=True)
            gy, gx = tp.autograd.grad(tp.arctan2(y, x).sum(), [y, x])
            # d atan2(y, x) = (x dy - y dx) / (x^2 + y^2).
            self._check(gy.cpu(), tp.tensor([0.5, 0.5 / 4.25], dtype=f64), tol=1e-12)
            self._check(gx.cpu(), tp.tensor([-0.5, 2.0 / 4.25], dtype=f64), tol=1e-12)
            z = tp.tensor([-2.0, 0.5, 3.0], device=device, requires_grad=True)
            (grad,) = tp.autograd.grad(tp.clip(z, -1.0, 1.0).sum(), [z])
            self._check(grad.cpu(), tp.tensor([0.0, 1.0, 0.0]))
            (grad,) = tp.autograd.grad(tp.fix(z * 2).sum(), [z])
            self._check(grad.cpu(), tp.zeros(3))

    def test_grids_and_powers_of_a_vector_differentiate(self):
        f64 = tp.float64
        for device in ["cpu"] + (["cuda"] if tp.cuda.is_available() else []):
            a = tp.tensor([1.0, 2.0], dtype=f64, device=device, requires_grad=True)
            b = tp.tensor([3.0, 4.0, 5.0], dtype=f64, device=device, requires_grad=True)
            for indexing in ("ij", "xy"):
                ga, gb = tp.meshgrid(a, b, indexing=indexing)
                da, db = tp.autograd.grad(ga.sum() + 2 * gb.sum(), [a, b])
                # Each entry of a spans a row of three, each entry of b a column of two.
                self._check(da.cpu(), tp.tensor([3.0, 3.0], dtype=f64))
                self._check(db.cpu(), tp.tensor([4.0, 4.0, 4.0], dtype=f64))
            da, db = tp.autograd.grad(tp.cartesian_prod(a, b).sum(), [a, b])
            self._check(da.cpu(), tp.tensor([3.0, 3.0], dtype=f64))
            self._check(db.cpu(), tp.tensor([2.0, 2.0, 2.0], dtype=f64))
            # Rows (a^2, a, 1): the derivative of a row's sum is 2a + 1.
            (grad,) = tp.autograd.grad(tp.vander(a, 3).sum(), [a])
            self._check(grad.cpu(), tp.tensor([3.0, 5.0], dtype=f64))
            self.assertEqual(tp.vander(a.detach(), 3).tolist(), [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]])
            ints = tp.tensor([2, 3], device=device)
            self.assertEqual(tp.vander(ints, 2, increasing=True).dtype, tp.int64)
            self.assertEqual(tp.vander(ints, 0).shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
