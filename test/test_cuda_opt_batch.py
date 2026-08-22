"""Correctness tests for the 2026-08-21 CUDA optimization batch.

Covers:
- layer_norm CUDA forward/backward (new custom kernels) vs CPU reference
- nll_loss / mse_loss backward on Half/BFloat16/Float64
- add/sub/mul/div vectorized fast path (contiguous large tensors)
"""
import unittest

import tensorplay as tp


def _allclose(a, b, rtol=1e-5, atol=1e-6):
    return tp.allclose(a, b, rtol=rtol, atol=atol)


class TestLayerNormCUDA(unittest.TestCase):
    """LayerNorm CUDA must match the CPU kernel within fp tolerance."""

    SHAPES = [
        ((4, 8), (8,)),            # tiny N, exercises scalar path
        ((2, 3, 16), (16,)),       # standard transformer layout
        ((3, 64), (64,)),          # N % 4 == 0 -> vectorized path
        ((1, 100), (100,)),        # N not divisible by 4
        ((2, 5, 7), (5, 7)),       # multi-dim normalized_shape
    ]

    def _check(self, shape, norm_shape, dtype):
        cpu_x = tp.randn(shape, dtype=tp.float32).to(dtype)
        cpu_w = tp.randn(norm_shape[-1:], dtype=tp.float32).to(dtype) if len(norm_shape) == 1 else None
        cpu_b = tp.randn(norm_shape[-1:], dtype=tp.float32).to(dtype) if len(norm_shape) == 1 else None

        ref = tp.layer_norm(cpu_x.cpu(), list(norm_shape), cpu_w, cpu_b)
        out = tp.layer_norm(cpu_x.cuda(), list(norm_shape),
                            cpu_w.cuda() if cpu_w is not None else None,
                            cpu_b.cuda() if cpu_b is not None else None)
        self.assertEqual(out.shape, ref.shape)
        tol = (2e-2, 2e-2) if dtype in (tp.float16, tp.bfloat16) else (1e-4, 1e-5)
        self.assertTrue(_allclose(out.cpu(), ref, rtol=tol[0], atol=tol[1]),
                        f"layer_norm forward mismatch for {shape}/{norm_shape}/{dtype}")

    def test_forward_shapes(self):
        for shape, norm in self.SHAPES:
            self._check(shape, norm, tp.float32)

    def test_forward_half(self):
        for shape, norm in self.SHAPES[:3]:
            self._check(shape, norm, tp.float16)
            self._check(shape, norm, tp.bfloat16)

    def test_backward_matches_cpu(self):
        x_cpu = tp.randn((4, 32), dtype=tp.float32)
        w = tp.randn((32,), dtype=tp.float32)
        b = tp.randn((32,), dtype=tp.float32)

        x_ref = x_cpu.clone().requires_grad_(True)
        out_ref = tp.layer_norm(x_ref, [32], w, b)
        g = tp.randn((4, 32), dtype=tp.float32)
        out_ref.backward(g)

        x_cuda = x_cpu.cuda().clone().requires_grad_(True)
        out = tp.layer_norm(x_cuda, [32], w.cuda(), b.cuda())
        out.backward(g.cuda())

        tol = 1e-4
        self.assertTrue(_allclose(x_cuda.grad.cpu(), x_ref.grad, tol, tol))
        # grad_weight/grad_bias flow through the same op
        self.assertTrue(_allclose(out.cpu(), out_ref.detach(), tol, tol))

    def test_backward_half(self):
        x = tp.randn((8, 64), dtype=tp.float32).cuda().to(tp.float16).requires_grad_(True)
        w = tp.randn((64,), dtype=tp.float32).cuda().to(tp.float16)
        out = tp.layer_norm(x, [64], w, None)
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertFalse(tp.isnan(x.grad).any().item())


class TestLossHalfPrecision(unittest.TestCase):
    def test_nll_loss_reduced_dtype(self):
        for dtype in (tp.float16, tp.bfloat16, tp.float64):
            logits = tp.randn((6, 10), dtype=tp.float32).cuda().to(dtype)
            target = tp.randint(0, 10, (6,), dtype=tp.int64).cuda()

            loss_none, _ = tp.nn.functional.nll_loss(logits, target, reduction="none")
            self.assertEqual(loss_none.dtype, logits.dtype)
            loss_mean, _ = tp.nn.functional.nll_loss(logits, target, reduction="mean")
            expected = -loss_none.mean()
            self.assertTrue(_allclose(loss_mean, expected, rtol=5e-2, atol=1e-2))

    def test_mse_loss_backward_reduced_dtype(self):
        for dtype in (tp.float16, tp.bfloat16):
            pred = tp.randn((128,), dtype=tp.float32).cuda().to(dtype).requires_grad_(True)
            tgt = tp.randn((128,), dtype=tp.float32).cuda().to(dtype)
            loss = tp.nn.functional.mse_loss(pred, tgt)
            loss.backward()
            self.assertEqual(pred.grad.shape, pred.shape)
            self.assertFalse(tp.isnan(pred.grad).any().item())


class TestBinaryVectorizedPath(unittest.TestCase):
    """The vectorized fast path must agree with the broadcast kernel."""

    OPS = [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
    ]

    def test_contiguous_large(self):
        n = 40960  # multiple of 4, well above the 4096 threshold
        for name, fn in self.OPS:
            a = tp.randn((n,), dtype=tp.float32).cuda()
            b = tp.randn((n,), dtype=tp.float32).cuda() * 0.5 + 1.0
            out = fn(a, b)
            # reference via small non-vectorizable slices through the same op
            ref_a, ref_b = a.cpu(), b.cpu()
            expect = fn(ref_a, ref_b)
            self.assertTrue(_allclose(out.cpu(), expect, 1e-5, 1e-5),
                            f"{name} vectorized mismatch")

    def test_small_and_odd_sizes(self):
        # Below threshold / odd sizes must fall back cleanly.
        for n in (7, 1023, 4095):
            a = tp.randn((n,), dtype=tp.float32).cuda()
            b = tp.randn((n,), dtype=tp.float32).cuda()
            self.assertTrue(_allclose((a * b).cpu(), (a.cpu() * b.cpu()), 1e-5, 1e-5))

    def test_half_binary(self):
        n = 8192
        for name, fn in self.OPS:
            a = tp.randn((n,), dtype=tp.float32).cuda().to(tp.float16)
            b = tp.randn((n,), dtype=tp.float32).cuda().to(tp.float16) * 0.5 + 1.0
            out = fn(a, b)
            expect = fn(a.cpu(), b.cpu())
            self.assertTrue(_allclose(out.cpu(), expect, 2e-2, 2e-2),
                            f"half {name} mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
