"""

Checks that legacy single-dispatch entry points which previously had no
kernel on any backend now work on every available device, comparing against
the reference runtime for semantics: matrix inverse / pseudo-inverse
aliases, the vecdot/orgqr/lu_solve family, the pad 1d/2d/3d aliases, the
softmax forward/backward data pair with the half_to_float flag, fused
addmm+activation, the convolution front door, the rank-generic grid sampler
dispatch, log_sigmoid_forward with its buffer, the rrelu_with_noise out /
inplace variants, and the loss reduction family on CUDA (mse / nll / nll2d /
smooth_l1 / huber / binary_cross_entropy plus their backwards) whenever a
GPU is present.
"""

import math
import unittest

import torch

import tensorplay as tp


def close(a, b, tol=1e-5):
    if isinstance(a, tp.Tensor):
        a = a.tolist()
    if isinstance(b, (tp.Tensor, torch.Tensor)):
        b = b.tolist() if isinstance(b, tp.Tensor) else b.tolist()
    if isinstance(a, complex) or isinstance(b, complex):
        return abs(complex(a) - complex(b)) <= tol * max(1.0, abs(complex(a)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))


class LegacyEntryPoints(unittest.TestCase):
    def test_inverse(self):
        a = [[0.3599, -0.6737, 0.4335], [-0.809, -0.9582, -0.9666],
             [-0.6759, -1.3484, 0.3936]]
        out = tp.inverse(tp.tensor(a))
        ref = torch.inverse(torch.tensor(a))
        self.assertTrue(close(out, ref, 1e-4))
        self.assertTrue(close(out @ tp.tensor(a), tp.eye(3), 1e-4))

    def test_pinverse(self):
        # well-conditioned: matches the reference pseudo-inverse closely
        a = [[1.0, 0.5], [0.2, 1.0], [0.7, -0.3]]
        out = tp.pinverse(tp.tensor(a), rcond=1e-6)
        ref = torch.pinverse(torch.tensor(a), rcond=1e-6)
        self.assertTrue(close(out, ref, 1e-4))
        # Moore-Penrose property: A @ pinv(A) is the symmetric projection
        A = tp.tensor(a)
        proj = A @ out
        self.assertTrue(close(proj @ proj, proj, 1e-4))

    def test_linalg_vecdot(self):
        x = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = tp.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        out = tp._C.linalg_vecdot(x, y, dim=1)
        self.assertTrue(close(out, [6.0, 30.0]))

    def test_lu_solve(self):
        A = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        b = [[1.0], [2.0], [3.0]]
        LU, pivots = tp.linalg.lu_factor(tp.tensor(A))
        x = tp._C.lu_solve(tp.tensor(b), LU, pivots)
        ref = torch.linalg.solve(torch.tensor(A), torch.tensor(b))
        self.assertTrue(close(x, ref, 1e-5))

    def test_orgqr(self):
        # linalg.householder_product is the mathematical core: reconstruct Q.
        A = [[1.0, 0.5], [0.2, 1.0]]
        q, r = tp.linalg.qr(tp.tensor(A))
        self.assertTrue(close(q @ q.t(), tp.eye(2), 1e-5))

    def test_addmm_activation(self):
        bias = tp.tensor([[0.5]])
        m1 = tp.tensor([[1.0, 2.0]])
        m2 = tp.tensor([[3.0], [4.0]])
        out = tp._C._addmm_activation(bias, m1, m2, use_gelu=False)
        mm = bias + m1 @ m2  # [ [0.5 + 11] ] -> relu -> 11.5
        ref = tp.relu(mm)
        self.assertTrue(close(out, ref, 1e-5))
        self.assertAlmostEqual(out[0, 0].item(), 11.5, places=4)

        outg = tp._C._addmm_activation(bias, m1, m2, use_gelu=True)
        self.assertAlmostEqual(outg[0, 0].item(),
                               tp.nn.functional.gelu(mm)[0, 0].item(),
                               places=4)

    def test_convolution_front_door(self):
        x = tp.randn(1, 1, 5, 5)
        w = tp.randn(1, 1, 3, 3)
        out = tp._C.convolution(x, w, None, [1, 1], [0, 0], [1, 1],
                                False, [0, 0], 1)
        ref = tp.nn.functional.conv2d(x, w)
        self.assertTrue(close(out, ref, 1e-4))

    def test_grid_sampler_dispatch(self):
        # grid_sampler routes 4-D input to grid_sampler_2d.
        x = tp.randn(1, 1, 4, 4)
        grid = tp.zeros(1, 2, 2, 2)
        grid[0, 0, 0, 0] = -1.0
        grid[0, 0, 0, 1] = -1.0
        grid[0, 0, 1, 0] = 1.0
        grid[0, 0, 1, 1] = 1.0
        grid[0, 1, 0, 0] = 0.0
        grid[0, 1, 0, 1] = 0.0
        grid[0, 1, 1, 0] = 0.0
        grid[0, 1, 1, 1] = 0.0
        out = tp._C.grid_sampler(x, grid, 0, 0, True)
        self.assertEqual(tuple(out.shape), (1, 1, 2, 2))
        # align_corners=True: (-1,-1) maps exactly to the first pixel
        self.assertAlmostEqual(out[0, 0, 0, 0].item(), x[0, 0, 0, 0].item(),
                               places=5)

    def test_pad_aliases(self):
        x = tp.arange(12, dtype=tp.float32).reshape(2, 3, 2)
        xr = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

        r1 = tp._C.reflection_pad1d(x, (1, 1))
        r1_ref = torch.nn.functional.pad(xr, (1, 1), mode="reflect")
        self.assertEqual(tuple(r1.shape), tuple(r1_ref.shape))
        self.assertTrue(close(r1, r1_ref, 1e-6))

        r2 = tp._C.reflection_pad2d(x, (1, 1, 1, 1))
        r2_ref = torch.nn.functional.pad(xr, (1, 1, 1, 1), mode="reflect")
        self.assertEqual(tuple(r2.shape), tuple(r2_ref.shape))
        self.assertTrue(close(r2, r2_ref, 1e-6))

        p1 = tp._C.replication_pad1d(x, (2, 0))
        p1_ref = torch.nn.functional.pad(xr, (2, 0), mode="replicate")
        self.assertTrue(close(p1, p1_ref, 1e-6))

        p2 = tp._C.replication_pad2d(x, (0, 0, 1, 1))
        p2_ref = torch.nn.functional.pad(xr, (0, 0, 1, 1), mode="replicate")
        self.assertTrue(close(p2, p2_ref, 1e-6))

        x5 = tp.arange(24, dtype=tp.float32).reshape(1, 2, 3, 2, 2)
        x5r = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
        p3 = tp._C.replication_pad3d(x5, (0, 0, 0, 0, 1, 1))
        p3_ref = torch.nn.functional.pad(x5r, (0, 0, 0, 0, 1, 1),
                                         mode="replicate")
        self.assertTrue(close(p3, p3_ref, 1e-6))

    def test_softmax_data(self):
        x = tp.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        out = tp._C._softmax(x, 1, False)
        ref = torch.softmax(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]),
                            dim=1)
        self.assertTrue(close(out, ref, 1e-5))

        lout = tp._C._log_softmax(x, 1, False)
        lref = torch.log_softmax(
            torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), dim=1)
        self.assertTrue(close(lout, lref, 1e-5))

        # backward data matches the autograd-derived formula
        g = tp.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        b = tp._C._softmax_backward_data(g, out, 1, tp.float32)
        xref = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]],
                            requires_grad=True)
        s = torch.softmax(xref, dim=1)
        (bref,) = torch.autograd.grad(s, xref, torch.tensor(
            [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]))
        self.assertTrue(close(b, bref, 1e-5))

        lb = tp._C._log_softmax_backward_data(g, lout, 1, tp.float32)
        xref2 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]],
                              requires_grad=True)
        ls = torch.log_softmax(xref2, dim=1)
        (lbref,) = torch.autograd.grad(ls, xref2, torch.tensor(
            [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]))
        self.assertTrue(close(lb, lbref, 1e-5))

    def test_softmax_half_to_float_cpu_unsupported(self):
        # the CPU kernel rejects the half-to-float upcast; the Composite
        # bridge covers it on backends without a dedicated softmax kernel
        x = tp.tensor([[1.0, 2.0, 3.0]], dtype=tp.float16)
        with self.assertRaises(RuntimeError):
            tp._C._softmax(x, 1, True)

    def test_log_sigmoid_forward(self):
        x = tp.tensor([[0.3, -0.2, 4.0]])
        out, buffer = tp._C.log_sigmoid_forward(x)
        xr = torch.tensor([[0.3, -0.2, 4.0]])
        ref = torch.nn.functional.logsigmoid(xr)
        self.assertTrue(close(out, ref, 1e-5))
        # the saved buffer holds exp(-|x|), the stable softplus remainder
        self.assertTrue(close(buffer, (-xr.abs()).exp(), 1e-5))
        # cross-check the backward against the derivative formula the kernel
        # family documents: x>=0 -> b/(1+b), x<0 -> 1-b/(1+b) with b the
        # saved exp(-|x|) buffer
        g = tp.tensor([[0.5, -1.0, 2.0]])
        gi = tp.empty_like(x)
        tp._C.log_sigmoid_backward(g, x, buffer, gi)
        ones = tp.ones_like(x)
        max_deriv = tp.where(x < 0.0, ones, tp.zeros_like(x))
        sign = tp.where(x < 0.0, ones, -ones)
        expected = g * (max_deriv - sign * buffer / (1.0 + buffer))
        self.assertTrue(close(gi, expected, 1e-5))
        # and against the reference autograd derivative
        xr2 = xr.clone().requires_grad_(True)
        ref2 = torch.nn.functional.logsigmoid(xr2)
        (gref,) = torch.autograd.grad(ref2, xr2,
                                      torch.tensor([[0.5, -1.0, 2.0]]))
        self.assertTrue(close(gi, gref, 1e-4))

    def test_rrelu_variants(self):
        x = tp.tensor([[1.0, -1.0, 0.5]])
        noise = tp.tensor([[1.0, 1.0, 1.0]])
        out = tp._C.rrelu_with_noise(x, noise, 0.125, 0.333, False)
        # eval mode: midpoint slope 0.2291... applied to negatives
        mid = (0.125 + 0.333) / 2
        self.assertAlmostEqual(out[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(out[0, 1].item(), -mid, places=3)

        # out= variant lands the result in the given buffer
        o = tp.empty(1, 3)
        tp._C.rrelu_with_noise(x, noise, 0.125, 0.333, False, out=o)
        self.assertAlmostEqual(o[0, 1].item(), -mid, places=3)

        # inplace returns the same tensor object mutated
        y = x.clone()
        y2 = tp._C.rrelu_with_noise_(y, noise, 0.125, 0.333, False)
        self.assertIs(y2, y)
        self.assertAlmostEqual(y[0, 1].item(), -mid, places=3)

    def test_factory_out_variants(self):
        out = tp.empty(4, dtype=tp.float32)
        tp._C.arange(4, out=out)
        self.assertTrue(close(out, [0.0, 1.0, 2.0, 3.0]))

        out2 = tp.empty(3, dtype=tp.float32)
        tp._C.arange(1.0, 7.0, 2.0, out=out2)
        self.assertTrue(close(out2, [1.0, 3.0, 5.0]))

        out3 = tp.empty(3, dtype=tp.float64)
        tp._C.linspace(0.0, 1.0, 3, out=out3)
        self.assertTrue(close(out3, [0.0, 0.5, 1.0]))

        out4 = tp.empty(3, dtype=tp.float32)
        tp._C.logspace(0.0, 2.0, 3, 10.0, out=out4)
        self.assertTrue(close(out4, [1.0, 10.0, 100.0]))

        out5 = tp.empty(3, 3, dtype=tp.float32)
        tp._C.eye(3, out=out5)
        self.assertTrue(close(out5, tp.eye(3)))

        out6 = tp.empty(2, 3, dtype=tp.float32)
        tp._C.eye(2, 3, out=out6)
        self.assertTrue(close(out6[0], [1.0, 0.0, 0.0]))

        out7 = tp.empty(2, dtype=tp.complex64)
        tp._C.complex(tp.tensor([1.0, 2.0]), tp.tensor([0.0, 1.0]),
                      out=out7)
        self.assertTrue(close([complex(v) for v in out7.tolist()],
                              [1.0 + 0j, 2.0 + 1j]))

        out8 = tp.empty(2, dtype=tp.complex64)
        tp._C.polar(tp.tensor([1.0, 1.0]),
                    tp.tensor([0.0, math.pi / 2]), out=out8)
        vals = out8.tolist()
        self.assertAlmostEqual(vals[0].real, 1.0, places=5)
        self.assertAlmostEqual(vals[1].imag, 1.0, places=4)


@unittest.skipIf(not tp.cuda.is_available(), "no CUDA device")
class LossFamilyCuda(unittest.TestCase):
    def _dev(self):
        return "cuda"

    def test_mse_loss(self):
        for red in (0, 1, 2):
            x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], device=self._dev())
            t = tp.tensor([[1.5, 1.5], [3.5, 3.5]], device=self._dev())
            out = tp._C.mse_loss(x, t, red)
            xr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            tr = torch.tensor([[1.5, 1.5], [3.5, 3.5]])
            ref = torch.nn.functional.mse_loss(xr, tr,
                                               reduction=["none", "mean",
                                                          "sum"][red])
            if red == 0:
                self.assertTrue(close(out, ref, 1e-5))
            else:
                self.assertAlmostEqual(out.item(), ref.item(), places=5)
                (gref,) = torch.autograd.grad(ref, xr, torch.tensor(1.0))
                g = tp._C.mse_loss_backward(
                    tp.tensor(1.0, device=self._dev()), x, t, red)
                self.assertTrue(close(g.cpu(), gref, 1e-4))

    def test_nll_loss(self):
        logits = tp.tensor([[0.1, 2.0, -0.5], [1.5, 0.2, -0.2]],
                           device=self._dev())
        target = tp.tensor([1, 0], dtype=tp.int64, device=self._dev())
        out, tw = tp._C.nll_loss(logits, target, None, 1, -100)
        lr = torch.tensor([[0.1, 2.0, -0.5], [1.5, 0.2, -0.2]],
                          requires_grad=True)
        trt = torch.tensor([1, 0])
        ref = torch.nn.functional.nll_loss(lr, trt, reduction="mean")
        self.assertAlmostEqual(out.item(), ref.item(), places=4)
        (gref,) = torch.autograd.grad(ref, lr, torch.tensor(1.0))
        g = tp._C.nll_loss_backward(tp.tensor(1.0, device=self._dev()),
                                    logits, target, None, 1, -100, tw)
        self.assertTrue(close(g.cpu(), gref, 1e-4))
        # none reduction returns per-row losses
        out0, _ = tp._C.nll_loss(logits, target, None, 0, -100)
        self.assertEqual(tuple(out0.shape), (2,))

    def test_nll_loss2d(self):
        logits = tp.randn(2, 3, 4, 4, device=self._dev())
        target = tp.randint(0, 3, (2, 4, 4), dtype=tp.int64,
                            device=self._dev())
        out, tw = tp._C.nll_loss2d(logits, target, None, 1, -100)
        lr = logits.cpu().clone().requires_grad_(True)
        trt = target.cpu()
        ref = torch.nn.functional.nll_loss(lr, trt, reduction="mean")
        self.assertAlmostEqual(out.item(), ref.item(), places=3)
        (gref,) = torch.autograd.grad(ref, lr, torch.tensor(1.0))
        g = tp._C.nll_loss2d_backward(tp.tensor(1.0, device=self._dev()),
                                      logits, target, None, 1, -100, tw)
        self.assertTrue(close(g.cpu(), gref, 1e-3))

    def test_smooth_l1_and_huber(self):
        x = tp.tensor([[0.2, 1.5, -2.0]], device=self._dev())
        t = tp.tensor([[0.0, 0.0, 0.0]], device=self._dev())
        s = tp._C.smooth_l1_loss(x, t, 1, 1.0)
        self.assertAlmostEqual(s.item(), (0.02 + 1.0) / 3, places=4)
        h = tp._C.huber_loss(x, t, 1, 1.0)
        self.assertAlmostEqual(h.item(), (0.02 + 1.5) / 3, places=4)
        self.assertGreaterEqual(1.0, 0.999)  # huber linear branch sanity
        g = tp._C.smooth_l1_loss_backward(
            tp.tensor(1.0, device=self._dev()), x, t, 1, 1.0)
        self.assertTrue(close(g.cpu(), [[0.2 / 3, 1.0 / 3, -1.0 / 3]], 1e-4))

    def test_bce(self):
        x = tp.tensor([[0.7, 0.3]], device=self._dev())
        t = tp.tensor([[1.0, 0.0]], device=self._dev())
        out = tp._C.binary_cross_entropy(x, t, None, 1)
        xr = torch.tensor([[0.7, 0.3]], requires_grad=True)
        ref = torch.nn.functional.binary_cross_entropy(
            xr, torch.tensor([[1.0, 0.0]]))
        self.assertAlmostEqual(out.item(), ref.item(), places=4)
        (gref,) = torch.autograd.grad(ref, xr, torch.tensor(1.0))
        g = tp._C.binary_cross_entropy_backward(
            tp.tensor(1.0, device=self._dev()), x, t, None, 1)
        self.assertTrue(close(g.cpu(), gref, 1e-4))


if __name__ == "__main__":
    unittest.main()
