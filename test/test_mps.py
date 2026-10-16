"""MPS backend tests: device availability, tensor factories, host/device
copy, arithmetic, elementwise, comparison, activation, softmax and reduction
ops.

Every test is skipped when the runner has no Metal device (Linux CI, Intel
macs without a supported GPU), so the file is safe to include in the macOS
CPU wheel test matrix.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp


def mps_available():
    return tp.mps.is_available()


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSDevice(unittest.TestCase):
    def test_is_available(self):
        self.assertTrue(tp.mps.is_available())
        self.assertEqual(tp.mps.device_count(), 1)
        self.assertEqual(tp.mps.current_device(), 0)

    def test_tensor_factories(self):
        x = tp.zeros((2, 3), device="mps")
        self.assertEqual(x.device.type, "mps")
        self.assertEqual(x.dtype, tp.float32)
        self.assertTrue(tp.all(x.cpu() == 0).item())

        y = tp.ones((2, 3), device="mps")
        self.assertTrue(tp.all(y.cpu() == 1).item())

        z = tp.full((2, 3), 7, device="mps")
        self.assertTrue(tp.all(z.cpu() == 7).item())

        e = tp.empty((2, 3), device="mps")
        self.assertEqual(e.shape, (2, 3))

    def test_range_factories(self):
        a = tp.arange(6, dtype=tp.float32, device="mps")
        self.assertEqual(a.cpu().tolist(), [0, 1, 2, 3, 4, 5])

        l = tp.linspace(0, 1, 5, device="mps")
        self.assertEqual(l.cpu().tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])

        eye = tp.eye(3, device="mps")
        self.assertTrue(tp.all(eye.cpu() == tp.eye(3)).item())

    def test_like_factories(self):
        x = tp.zeros((2, 3), device="mps")
        self.assertEqual(tp.zeros_like(x).device.type, "mps")
        self.assertEqual(tp.ones_like(x).device.type, "mps")
        self.assertEqual(tp.full_like(x, 3).device.type, "mps")
        self.assertTrue(tp.all(tp.full_like(x, 3).cpu() == 3).item())

    def test_move_between_devices(self):
        cpu = tp.arange(6, dtype=tp.float32).reshape(2, 3)
        mps_t = cpu.mps()
        self.assertEqual(mps_t.device.type, "mps")
        self.assertTrue(tp.all(mps_t.cpu() == cpu).item())

        mps_t2 = cpu.to("mps")
        self.assertEqual(mps_t2.device.type, "mps")
        self.assertTrue(tp.all(mps_t2.cpu() == cpu).item())

        back = mps_t.cpu()
        self.assertEqual(back.device.type, "cpu")
        self.assertTrue(tp.all(back == cpu).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSArithmetic(unittest.TestCase):
    def setUp(self):
        self.a = tp.full((2, 3), 6, device="mps", dtype=tp.float32)
        self.b = tp.full((2, 3), 4, device="mps", dtype=tp.float32)

    def test_add(self):
        c = self.a + self.b
        self.assertEqual(c.device.type, "mps")
        self.assertTrue(tp.all(c.cpu() == 10).item())

    def test_sub_mul_div(self):
        self.assertTrue(tp.all((self.a - self.b).cpu() == 2).item())
        self.assertTrue(tp.all((self.a * self.b).cpu() == 24).item())
        self.assertTrue(tp.allclose((self.a / self.b).cpu(),
                                    tp.full((2, 3), 1.5)))

    def test_scalar_ops(self):
        self.assertTrue(tp.all((self.a + 2).cpu() == 8).item())
        self.assertTrue(tp.all((self.a - 2).cpu() == 4).item())
        self.assertTrue(tp.all((self.a * 2).cpu() == 12).item())
        self.assertTrue(tp.all((self.a / 2).cpu() == 3).item())

    def test_inplace_ops(self):
        x = self.a.clone()
        x.add_(2)
        self.assertTrue(tp.all(x.cpu() == 8).item())
        x = self.a.clone()
        x.sub_(2)
        self.assertTrue(tp.all(x.cpu() == 4).item())
        x = self.a.clone()
        x.mul_(2)
        self.assertTrue(tp.all(x.cpu() == 12).item())
        x = self.a.clone()
        x.div_(2)
        self.assertTrue(tp.all(x.cpu() == 3).item())

    def test_pow_remainder_fmod(self):
        self.assertTrue(tp.all(tp.pow(self.a, 2).cpu() == 36).item())
        self.assertTrue(tp.all((self.a ** self.b).cpu() == 6**4).item())
        self.assertTrue(tp.all((self.a % self.b).cpu() == 2).item())
        self.assertTrue(tp.all(tp.fmod(self.a, self.b).cpu() == 2).item())

    def test_max_min(self):
        self.assertTrue(tp.all(tp.maximum(self.a, self.b).cpu() == 6).item())
        self.assertTrue(tp.all(tp.minimum(self.a, self.b).cpu() == 4).item())

    def test_addcmul_addcdiv(self):
        x = tp.full((2, 3), 1, device="mps", dtype=tp.float32)
        y = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        z = tp.full((2, 3), 3, device="mps", dtype=tp.float32)
        self.assertTrue(tp.all(tp.addcmul(x, y, z).cpu() == 7).item())
        self.assertTrue(tp.allclose(tp.addcdiv(x, y, z).cpu(),
                                    tp.full((2, 3), 1 + 2 / 3.0)))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSUnary(unittest.TestCase):
    def test_unary(self):
        x = tp.full((2, 2), 4, device="mps", dtype=tp.float32)
        self.assertTrue(tp.all(tp.abs(-x).cpu() == 4).item())
        self.assertTrue(tp.all(tp.neg(x).cpu() == -4).item())
        self.assertTrue(tp.all(tp.square(x).cpu() == 16).item())
        self.assertTrue(tp.all(tp.sqrt(x).cpu() == 2).item())
        self.assertTrue(tp.all(tp.rsqrt(x).cpu() == 0.5).item())
        self.assertTrue(tp.all(tp.reciprocal(x).cpu() == 0.25).item())
        self.assertTrue(tp.all(tp.exp(tp.zeros((2, 2), device="mps")).cpu() == 1).item())
        self.assertTrue(tp.all(tp.log(tp.ones((2, 2), device="mps")).cpu() == 0).item())
        self.assertTrue(tp.allclose(tp.sin(tp.zeros((2, 2), device="mps")).cpu(),
                                    tp.zeros((2, 2))))
        self.assertTrue(tp.allclose(tp.cos(tp.zeros((2, 2), device="mps")).cpu(),
                                    tp.ones((2, 2))))
        self.assertTrue(tp.allclose(tp.tanh(tp.zeros((2, 2), device="mps")).cpu(),
                                    tp.zeros((2, 2))))
        self.assertTrue(tp.all(tp.floor(tp.full((2, 2), 3.7, device="mps")).cpu() == 3).item())
        self.assertTrue(tp.all(tp.ceil(tp.full((2, 2), 3.2, device="mps")).cpu() == 4).item())
        self.assertTrue(tp.all(tp.round(tp.full((2, 2), 3.5, device="mps")).cpu() == 4).item())
        self.assertTrue(tp.all(tp.trunc(tp.full((2, 2), -3.7, device="mps")).cpu() == -3).item())

    def test_activations(self):
        x = tp.full((2, 2), -1, device="mps", dtype=tp.float32)
        self.assertTrue(tp.all(tp.relu(x).cpu() == 0).item())
        self.assertTrue(tp.all(tp.relu(-x).cpu() == 1).item())
        self.assertTrue(tp.allclose(tp.sigmoid(tp.zeros((2, 2), device="mps")).cpu(),
                                    tp.full((2, 2), 0.5)))
        self.assertTrue(tp.all(tp.leaky_relu(x, 0.1).cpu() == -0.1).item())
        self.assertTrue(tp.all(tp.hardtanh(x, -2, 2).cpu() == -1).item())
        self.assertTrue(tp.allclose(tp.silu(tp.zeros((2, 2), device="mps")).cpu(),
                                    tp.zeros((2, 2))))
        self.assertTrue(tp.all(tp.softplus(tp.zeros((2, 2), device="mps")).cpu() > 0).item())

    def test_bitwise_not(self):
        x = tp.full((2,), 0, device="mps", dtype=tp.int32)
        self.assertTrue(tp.all(tp.bitwise_not(x).cpu() == -1).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSComparison(unittest.TestCase):
    def test_tensor_comparisons(self):
        a = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        b = tp.full((2, 3), 3, device="mps", dtype=tp.float32)
        self.assertTrue((a < b).cpu().all().item())
        self.assertTrue((a <= b).cpu().all().item())
        self.assertTrue((a == a).cpu().all().item())
        self.assertTrue((a != b).cpu().all().item())
        self.assertFalse((a > b).cpu().any().item())
        self.assertFalse((a >= b).cpu().any().item())

    def test_scalar_comparisons(self):
        a = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        self.assertTrue((a < 3).cpu().all().item())
        self.assertTrue((a == 2).cpu().all().item())
        self.assertTrue((a != 3).cpu().all().item())

    def test_where(self):
        cond = tp.full((2, 3), 1, device="mps", dtype=tp.float32) > 0
        a = tp.full((2, 3), 1, device="mps", dtype=tp.float32)
        b = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        w = tp.where(cond, a, b)
        self.assertTrue(tp.all(w.cpu() == 1).item())

    def test_clamp(self):
        x = tp.full((4,), 5, device="mps", dtype=tp.float32)
        self.assertTrue(tp.all(tp.clamp(x, 1, 3).cpu() == 3).item())
        self.assertTrue(tp.all(tp.clamp_min(x, 2).cpu() == 5).item())
        self.assertTrue(tp.all(tp.clamp_max(x, 2).cpu() == 2).item())
        y = x.clone()
        y.clamp_(1, 3)
        self.assertTrue(tp.all(y.cpu() == 3).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSSoftmaxReduction(unittest.TestCase):
    def test_softmax(self):
        x = tp.full((2, 3), 2, device="mps", dtype=tp.float32)
        s = tp.softmax(x, dim=1)
        self.assertTrue(tp.allclose(s.cpu().sum(dim=1), tp.ones((2,))))

    def test_reductions(self):
        x = tp.arange(6, dtype=tp.float32, device="mps").reshape(2, 3)
        self.assertEqual(tp.sum(x).item(), 15)
        self.assertEqual(tp.sum(x, dim=1).cpu().tolist(), [3, 12])
        self.assertEqual(tp.mean(x).item(), 2.5)
        self.assertEqual(tp.max(x).item(), 5)
        self.assertEqual(tp.min(x).item(), 0)
        self.assertEqual(tp.prod(x).item(), 0)
        self.assertEqual(tp.amax(x).item(), 5)
        self.assertEqual(tp.amin(x).item(), 0)
        self.assertEqual(tp.sum(x, dim=1, keepdim=True).cpu().tolist(), [[3], [12]])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSMemory(unittest.TestCase):
    def test_memory_accounting(self):
        tp.mps.empty_cache()
        a = tp.ones(1024, dtype=tp.float32, device="mps")
        self.assertGreaterEqual(tp.mps.memory_allocated(), 1024 * 4)
        self.assertGreaterEqual(tp.mps.memory_reserved(), 1024 * 4)
        del a
        tp.mps.empty_cache()

    def test_synchronize(self):
        a = tp.ones(4, 4, device="mps")
        b = a + a
        tp.mps.synchronize()
        self.assertEqual(b.shape, (4, 4))
        self.assertTrue(tp.all(b.cpu() == 2).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSRandom(unittest.TestCase):
    def test_rand(self):
        x = tp.rand(100, device="mps")
        self.assertEqual(x.shape, (100,))
        self.assertTrue(tp.all(x.cpu() >= 0).item())
        self.assertTrue(tp.all(x.cpu() < 1).item())

    def test_randn(self):
        # 100 samples make a mean assertion at places=1 inherently flaky
        # (standard error ~0.1); 10000 samples pin it down tightly.
        x = tp.randn(10000, device="mps")
        self.assertEqual(x.shape, (10000,))
        self.assertAlmostEqual(x.cpu().mean().item(), 0.0, places=1)

    def test_randint(self):
        x = tp.randint(0, 10, (100,), device="mps")
        self.assertTrue(tp.all(x.cpu() >= 0).item())
        self.assertTrue(tp.all(x.cpu() < 10).item())

    def test_randperm(self):
        x = tp.randperm(20, device="mps")
        self.assertEqual(sorted(x.cpu().tolist()), list(range(20)))

    def test_rand_like(self):
        a = tp.zeros((4, 5), device="mps")
        r = tp.rand_like(a)
        self.assertEqual(r.shape, (4, 5))
        self.assertEqual(r.device.type, "mps")


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSShapeOps(unittest.TestCase):
    def test_cat(self):
        a = tp.ones((2, 3), device="mps")
        b = tp.full((2, 3), 2, device="mps")
        c = tp.cat([a, b], dim=0)
        self.assertEqual(c.shape, (4, 3))
        self.assertTrue(tp.all(c.cpu()[:2] == 1).item())
        self.assertTrue(tp.all(c.cpu()[2:] == 2).item())

    def test_stack(self):
        a = tp.ones((2, 3), device="mps")
        b = tp.full((2, 3), 2, device="mps")
        c = tp.stack([a, b], dim=0)
        self.assertEqual(c.shape, (2, 2, 3))
        self.assertTrue(tp.all(c.cpu()[0] == 1).item())
        self.assertTrue(tp.all(c.cpu()[1] == 2).item())

    def test_embedding(self):
        weight = tp.arange(6, dtype=tp.float32, device="mps").reshape(3, 2)
        indices = tp.tensor([0, 2], dtype=tp.int64).mps()
        out = tp.embedding(weight, indices)
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(tp.all(out.cpu() == weight.cpu()[indices.cpu()]).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSBlas(unittest.TestCase):
    def test_mm(self):
        a = tp.arange(6, dtype=tp.float32, device="mps").reshape(2, 3)
        b = tp.arange(12, dtype=tp.float32, device="mps").reshape(3, 4)
        c = tp.mm(a, b)
        self.assertEqual(c.shape, (2, 4))
        self.assertTrue(tp.allclose(c.cpu(), a.cpu() @ b.cpu()))

    def test_matmul_2d(self):
        a = tp.ones((2, 3), device="mps")
        b = tp.ones((3, 4), device="mps")
        self.assertTrue(tp.all(tp.matmul(a, b).cpu() == 3).item())

    def test_bmm(self):
        a = tp.ones((2, 3, 4), device="mps")
        b = tp.ones((2, 4, 5), device="mps")
        c = tp.bmm(a, b)
        self.assertEqual(c.shape, (2, 3, 5))
        self.assertTrue(tp.all(c.cpu() == 4).item())

    def test_addmm(self):
        inp = tp.ones((2, 4), device="mps")
        m1 = tp.ones((2, 3), device="mps")
        m2 = tp.ones((3, 4), device="mps")
        c = tp.addmm(inp, m1, m2)
        # 1*input + 1*(m1@m2) = 1 + 3 = 4
        self.assertTrue(tp.all(c.cpu() == 4).item())

    def test_mv_dot(self):
        a = tp.ones((2, 3), device="mps")
        v = tp.ones((3,), device="mps")
        self.assertTrue(tp.all(tp.mv(a, v).cpu() == 3).item())
        x = tp.full((4,), 2, device="mps")
        y = tp.full((4,), 3, device="mps")
        self.assertEqual(tp.dot(x, y).item(), 24)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSAllAny(unittest.TestCase):
    def test_all_any(self):
        t = tp.ones((2, 3), dtype=tp.bool, device="mps")
        f = tp.zeros((2, 3), dtype=tp.bool, device="mps")
        self.assertTrue(tp.all(t).item())
        self.assertFalse(tp.all(f).item())
        self.assertFalse(tp.any(f).item())
        self.assertTrue(tp.all(t, dim=1).cpu().all().item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSLosses(unittest.TestCase):
    def test_mse_loss(self):
        x = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32, device="mps")
        y = tp.tensor([1.0, 2.0, 5.0], dtype=tp.float32, device="mps")
        self.assertAlmostEqual(tp.mse_loss(x, y).item(), 4.0 / 3.0, places=5)
        self.assertAlmostEqual(tp.mse_loss(x, y, reduction=0).cpu().tolist()[2], 4.0, places=5)
        self.assertAlmostEqual(tp.mse_loss(x, y, reduction=2).item(), 4.0, places=5)

    def test_l1_loss(self):
        x = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32, device="mps")
        y = tp.tensor([1.0, 2.0, 5.0], dtype=tp.float32, device="mps")
        self.assertAlmostEqual(tp.l1_loss(x, y).item(), 2.0 / 3.0, places=5)

    def test_smooth_l1_loss(self):
        x = tp.tensor([1.0, 2.0], dtype=tp.float32, device="mps")
        y = tp.tensor([3.0, 2.0], dtype=tp.float32, device="mps")
        # |diff| = [2, 0]; beta=1 -> [2-0.5, 0.5*0] = [1.5, 0]
        self.assertAlmostEqual(tp.smooth_l1_loss(x, y, reduction=0).cpu().tolist()[0], 1.5, places=5)
        self.assertAlmostEqual(tp.smooth_l1_loss(x, y, reduction=0).cpu().tolist()[1], 0.0, places=5)

    def test_huber_loss(self):
        x = tp.tensor([1.0, 2.0], dtype=tp.float32, device="mps")
        y = tp.tensor([5.0, 2.0], dtype=tp.float32, device="mps")
        # |diff| = [4, 0]; delta=1 -> [1*(4-0.5), 0] = [3.5, 0]
        self.assertAlmostEqual(tp.huber_loss(x, y, reduction=0).cpu().tolist()[0], 3.5, places=5)
        self.assertAlmostEqual(tp.huber_loss(x, y, reduction=0).cpu().tolist()[1], 0.0, places=5)

    def test_binary_cross_entropy(self):
        x = tp.tensor([0.5, 0.8], dtype=tp.float32, device="mps")
        y = tp.tensor([1.0, 0.0], dtype=tp.float32, device="mps")
        loss = tp.binary_cross_entropy(x, y)
        expected = -(0.5 * 0.5 * 0.0)  # mean of -[ln0.5, ln0.2]
        self.assertTrue(loss.item() > 0)
        self.assertAlmostEqual(tp.binary_cross_entropy(x, y, reduction=0).cpu().tolist()[0],
                               0.6931471805599453, places=5)

    def test_nll_loss(self):
        log_probs = tp.tensor([[-0.5, -1.0, -2.0], [-1.5, -0.2, -0.8]],
                              dtype=tp.float32, device="mps")
        target = tp.tensor([0, 1], dtype=tp.int64).mps()
        loss, total = tp.nll_loss(log_probs, target)
        self.assertAlmostEqual(loss.item(), 0.35, places=5)
        self.assertAlmostEqual(total.item(), 2.0, places=5)

        loss0 = tp.nll_loss(log_probs, target, reduction=0)
        self.assertEqual(loss0[0].cpu().tolist(), [0.5, 0.2])

        w = tp.tensor([2.0, 1.0, 1.0], dtype=tp.float32, device="mps")
        loss_w, total_w = tp.nll_loss(log_probs, target, weight=w)
        self.assertAlmostEqual(loss_w.item(), (1.0 + 0.2) / 3.0, places=5)

    def test_nll_loss_ignore_index(self):
        log_probs = tp.tensor([[-0.5, -1.0], [-1.5, -0.2]], dtype=tp.float32, device="mps")
        target = tp.tensor([0, -100], dtype=tp.int64).mps()
        loss, total = tp.nll_loss(log_probs, target, ignore_index=-100)
        self.assertAlmostEqual(loss.item(), 0.5, places=5)
        self.assertAlmostEqual(total.item(), 1.0, places=5)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSVarStd(unittest.TestCase):
    def test_var_std(self):
        x = tp.tensor([1.0, 2.0, 3.0, 4.0], dtype=tp.float32, device="mps")
        # population variance (correction=0): 1.25
        self.assertAlmostEqual(tp.var(x, correction=0).item(), 1.25, places=5)
        # sample variance (correction=1): 5/3
        self.assertAlmostEqual(tp.var(x).item(), 5.0 / 3.0, places=5)
        self.assertAlmostEqual(tp.std(x).item(), (5.0 / 3.0) ** 0.5, places=5)

    def test_var_dim(self):
        m = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32, device="mps")
        # population variance over dim 0: [1,3] -> mean 2, var 1
        v = tp.var(m, dim=[0], correction=0)
        self.assertAlmostEqual(v.cpu().tolist()[0], 1.0, places=5)
        self.assertAlmostEqual(v.cpu().tolist()[1], 1.0, places=5)
        vk = tp.var(m, dim=[0], correction=0, keepdim=True)
        self.assertEqual(vk.shape, (1, 2))
        s = tp.std(m, dim=[0], correction=0)
        self.assertAlmostEqual(s.cpu().tolist()[0], 1.0, places=5)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSMiscReductions(unittest.TestCase):
    def test_logsumexp(self):
        x = tp.tensor([[1.0, 2.0, 3.0]], dtype=tp.float32, device="mps")
        r = tp.logsumexp(x, dim=[1])
        self.assertAlmostEqual(r.item(), 3.4076059, places=5)
        # CPU reference: log(exp1+exp2+exp3)
        xc = x.cpu()
        rc = xc.exp().sum(dim=1).log()
        self.assertTrue(tp.allclose(r.cpu(), rc))

    def test_norm(self):
        x = tp.tensor([3.0, 4.0], dtype=tp.float32, device="mps")
        self.assertAlmostEqual(tp.norm(x, 2.0).item(), 5.0, places=5)
        self.assertAlmostEqual(tp.norm(x, 1.0).item(), 7.0, places=5)
        self.assertAlmostEqual(tp.norm(x, float("inf")).item(), 4.0, places=5)

    def test_norm_dim(self):
        m = tp.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=tp.float32, device="mps")
        n = tp.norm(m, p=2.0, dim=[1])
        self.assertAlmostEqual(n.cpu().tolist()[0], 5.0, places=5)
        self.assertAlmostEqual(n.cpu().tolist()[1], 10.0, places=5)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSViewOps(unittest.TestCase):
    def test_transpose_permute(self):
        x = tp.arange(6, dtype=tp.float32, device="mps").reshape(2, 3)
        self.assertTrue(tp.allclose(tp.transpose(x, 0, 1).cpu(), x.cpu().T))
        p = tp.permute(x, [1, 0])
        self.assertTrue(tp.allclose(p.cpu(), x.cpu().T))

    def test_squeeze_unsqueeze(self):
        x = tp.zeros((1, 3, 1), device="mps")
        self.assertEqual(tp.squeeze(x).shape, (3,))
        self.assertEqual(tp.unsqueeze(x, 0).shape, (1, 1, 3, 1))
        self.assertEqual(tp.squeeze(x, dim=0).shape, (3, 1))

    def test_reshape_slice(self):
        x = tp.arange(12, dtype=tp.float32, device="mps").reshape(3, 4)
        self.assertEqual(tp.reshape(x, [2, 6]).shape, (2, 6))
        self.assertTrue(tp.all(tp.reshape(x, [2, 6]).cpu() == x.cpu().reshape(2, 6)).item())
        s = x[1:3]
        self.assertEqual(s.shape, (2, 4))

    def test_split_chunk(self):
        x = tp.arange(6, dtype=tp.float32, device="mps")
        parts = tp.split(x, 2, 0)
        self.assertEqual(len(parts), 3)
        self.assertTrue(tp.allclose(parts[0].cpu(), tp.arange(2, dtype=tp.float32)))
        chunks = tp.chunk(x, 3, 0)
        self.assertEqual(len(chunks), 3)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSPooling(unittest.TestCase):
    def test_avg_pool2d(self):
        x = tp.ones((1, 1, 4, 4), dtype=tp.float32, device="mps")
        y = tp.avg_pool2d(x, kernel_size=[2, 2])
        self.assertEqual(y.shape, (1, 1, 2, 2))
        self.assertTrue(tp.all(y.cpu() == 1).item())

    def test_max_pool2d(self):
        x = tp.arange(16, dtype=tp.float32, device="mps").reshape(1, 1, 4, 4)
        y = tp.max_pool2d(x, kernel_size=[2, 2], stride=[2, 2])
        self.assertEqual(y.shape, (1, 1, 2, 2))
        self.assertEqual(y.cpu().tolist()[0][0][0][0], 5)
        self.assertEqual(y.cpu().tolist()[0][0][0][1], 7)
        self.assertEqual(y.cpu().tolist()[0][0][1][0], 13)
        self.assertEqual(y.cpu().tolist()[0][0][1][1], 15)

    def test_adaptive_avg_pool2d(self):
        x = tp.arange(16, dtype=tp.float32, device="mps").reshape(1, 1, 4, 4)
        y = tp.adaptive_avg_pool2d(x, (2, 2))
        self.assertEqual(y.shape, (1, 1, 2, 2))
        self.assertAlmostEqual(y.cpu().tolist()[0][0][0][0], 2.5, places=5)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSLayerNorm(unittest.TestCase):
    def test_layer_norm(self):
        x = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                      dtype=tp.float32, device="mps")
        y = tp.layer_norm(x, [3])
        mean = x.cpu().mean(dim=1, keepdim=True)
        var = x.cpu().var(dim=1, keepdim=True, correction=0)
        ref = (x.cpu() - mean) / (var + 1e-5).sqrt()
        self.assertTrue(tp.allclose(y.cpu(), ref, atol=1e-4))

    def test_layer_norm_affine(self):
        x = tp.tensor([[1.0, 2.0, 3.0]], dtype=tp.float32, device="mps")
        w = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32, device="mps")
        b = tp.tensor([0.5, 0.5, 0.5], dtype=tp.float32, device="mps")
        y = tp.layer_norm(x, [3], weight=w, bias=b)
        self.assertEqual(y.shape, (1, 3))
        self.assertTrue(tp.allclose(y.cpu(), ref_computed(x.cpu(), w.cpu(), b.cpu())))

    def test_native_layer_norm_stats(self):
        x = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                      dtype=tp.float32, device="mps")
        y = tp.layer_norm(x, [3])
        self.assertEqual(y.shape, (2, 3))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSIndexing(unittest.TestCase):
    def test_masked_fill(self):
        x = tp.ones((2, 3), dtype=tp.float32, device="mps")
        mask = tp.tensor([[True, False, True], [False, True, False]],
                         device="mps")
        y = tp.masked_fill(x, mask, tp.tensor(0.0, dtype=tp.float32))
        self.assertEqual(y.cpu().tolist()[0][0], 0.0)
        self.assertEqual(y.cpu().tolist()[0][1], 1.0)

    def test_index_select(self):
        x = tp.arange(6, dtype=tp.float32, device="mps").reshape(3, 2)
        idx = tp.tensor([0, 2], dtype=tp.int64).mps()
        y = tp.index_select(x, 0, idx)
        self.assertEqual(y.shape, (2, 2))
        self.assertTrue(tp.all(y.cpu() == x.cpu()[[0, 2]]).item())

    def test_gather(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32, device="mps")
        idx = tp.tensor([[0, 0], [1, 0]], dtype=tp.int64).mps()
        y = tp.gather(x, 1, idx)
        self.assertEqual(y.cpu().tolist(), [[1.0, 1.0], [4.0, 3.0]])

    def test_scatter(self):
        x = tp.zeros((2, 3), dtype=tp.float32, device="mps")
        idx = tp.tensor([[0, 1], [1, 2]], dtype=tp.int64).mps()
        src = tp.ones((2, 2), dtype=tp.float32, device="mps")
        y = tp.scatter(x, 1, idx, src)
        self.assertEqual(y.cpu().tolist(), [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    def test_scatter_add(self):
        x = tp.zeros((1, 3), dtype=tp.float32, device="mps")
        idx = tp.tensor([[0, 0, 1]], dtype=tp.int64).mps()
        src = tp.tensor([[1.0, 2.0, 3.0]], dtype=tp.float32, device="mps")
        y = tp.scatter_add(x, 1, idx, src)
        self.assertEqual(y.cpu().tolist(), [[3.0, 3.0, 0.0]])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSBatchNorm(unittest.TestCase):
    def test_batch_norm_eval(self):
        x = tp.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
                      dtype=tp.float32, device="mps")
        rm = tp.tensor([0.0, 0.0], dtype=tp.float32, device="mps")
        rv = tp.tensor([1.0, 1.0], dtype=tp.float32, device="mps")
        w = tp.tensor([1.0, 1.0], dtype=tp.float32, device="mps")
        b = tp.tensor([0.0, 0.0], dtype=tp.float32, device="mps")
        y = tp.batch_norm(x, w, b, rm, rv, training=False, momentum=0.1,
                          eps=1e-5)
        ref = x.cpu() / (1.0 + 1e-5) ** 0.5
        self.assertTrue(tp.allclose(y.cpu(), ref, atol=1e-4))

    def test_batch_norm_training_updates_stats(self):
        x = tp.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
                      dtype=tp.float32, device="mps")
        rm = tp.tensor([0.0, 0.0], dtype=tp.float32, device="mps")
        rv = tp.tensor([1.0, 1.0], dtype=tp.float32, device="mps")
        w = tp.tensor([1.0, 1.0], dtype=tp.float32, device="mps")
        b = tp.tensor([0.0, 0.0], dtype=tp.float32, device="mps")
        tp.batch_norm(x, w, b, rm, rv, training=True, momentum=0.1, eps=1e-5)
        # channel 0: mean 2.5, var 1.25, count=4, unbiased=4/3
        self.assertAlmostEqual(rm.cpu().tolist()[0], 0.25, places=5)
        self.assertAlmostEqual(rv.cpu().tolist()[0],
                               0.9 + 0.1 * 1.25 * 4.0 / 3.0, places=5)
        # channel 1: mean 6.5
        self.assertAlmostEqual(rm.cpu().tolist()[1], 0.65, places=5)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSGroupNorm(unittest.TestCase):
    def test_group_norm(self):
        x = tp.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
                      dtype=tp.float32, device="mps")
        y = tp.group_norm(x, 1, eps=1e-5)
        mean = 4.5
        var = 5.25
        ref = (x.cpu() - mean) / (var + 1e-5) ** 0.5
        self.assertTrue(tp.allclose(y.cpu(), ref, atol=1e-4))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSConv2d(unittest.TestCase):
    def test_conv2d(self):
        x = tp.ones((1, 1, 4, 4), dtype=tp.float32, device="mps")
        w = tp.ones((1, 1, 2, 2), dtype=tp.float32, device="mps")
        y = tp.conv2d(x, w, None, stride=[1, 1], padding=[0, 0],
                      dilation=[1, 1], groups=1)
        self.assertEqual(y.shape, (1, 1, 3, 3))
        self.assertTrue(tp.all(y.cpu() == 4).item())

    def test_conv2d_bias(self):
        x = tp.ones((1, 1, 4, 4), dtype=tp.float32, device="mps")
        w = tp.ones((1, 1, 2, 2), dtype=tp.float32, device="mps")
        b = tp.tensor([1.0], dtype=tp.float32, device="mps")
        y = tp.conv2d(x, w, b, stride=[1, 1], padding=[0, 0],
                      dilation=[1, 1], groups=1)
        self.assertTrue(tp.all(y.cpu() == 5).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSArgExtrema(unittest.TestCase):
    def test_argmax_argmin(self):
        x = tp.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]],
                      dtype=tp.float32, device="mps")
        self.assertEqual(tp.argmax(x).item(), 5)  # flattened index of 6.0
        self.assertEqual(tp.argmax(x, dim=1).cpu().tolist(), [1, 2])
        self.assertEqual(tp.argmin(x, dim=1).cpu().tolist(), [0, 1])
        self.assertEqual(tp.argmax(x, dim=0).cpu().tolist(), [1, 0, 1])

    def test_max_min_dim(self):
        x = tp.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]],
                      dtype=tp.float32, device="mps")
        values, indices = tp.max(x, dim=1)
        self.assertEqual(values.cpu().tolist(), [5.0, 6.0])
        self.assertEqual(indices.cpu().tolist(), [1, 2])
        vmin, imin = tp.min(x, dim=0)
        self.assertEqual(vmin.cpu().tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(imin.cpu().tolist(), [0, 1, 0])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSTriangular(unittest.TestCase):
    def test_tril_triu(self):
        x = tp.arange(9, dtype=tp.float32, device="mps").reshape(3, 3)
        self.assertTrue(tp.all(tp.tril(x).cpu() == x.cpu().tril()).item())
        self.assertTrue(tp.all(tp.triu(x).cpu() == x.cpu().triu()).item())
        self.assertTrue(tp.all(tp.tril(x, diagonal=1).cpu() == x.cpu().tril(1)).item())

    def test_tril_inplace(self):
        x = tp.arange(9, dtype=tp.float32, device="mps").reshape(3, 3)
        x.tril_()
        self.assertTrue(tp.all(x.cpu() == tp.tril(tp.arange(9, dtype=tp.float32, device="mps").reshape(3, 3)).cpu()).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSPad(unittest.TestCase):
    def test_constant_pad_nd(self):
        x = tp.ones((1, 2, 2), dtype=tp.float32, device="mps")
        y = tp.constant_pad_nd(x, [1, 1, 0, 0], 0.0)
        self.assertEqual(y.shape, (1, 2, 4))
        self.assertTrue(tp.all(y.cpu()[:, :, 0] == 0).item())
        self.assertTrue(tp.all(y.cpu()[:, :, -1] == 0).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSScan(unittest.TestCase):
    def test_cumsum_cumprod(self):
        x = tp.tensor([1.0, 2.0, 3.0, 4.0], dtype=tp.float32, device="mps")
        self.assertTrue(tp.allclose(tp.cumsum(x, 0).cpu(), x.cpu().cumsum(0)))
        self.assertTrue(tp.allclose(tp.cumprod(x, 0).cpu(), x.cpu().cumprod(0)))
        m = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32, device="mps")
        self.assertTrue(tp.allclose(tp.cumsum(m, dim=1).cpu(), m.cpu().cumsum(1)))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSSort(unittest.TestCase):
    def test_sort(self):
        x = tp.tensor([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]],
                      dtype=tp.float32, device="mps")
        values, indices = tp.sort(x, dim=1, descending=False)
        self.assertEqual(values.cpu().tolist(),
                         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(indices.cpu().tolist(), [[1, 2, 0], [2, 1, 0]])
        vd, _ = tp.sort(x, dim=1, descending=True)
        self.assertEqual(vd.cpu().tolist(),
                         [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])

    def test_argsort(self):
        x = tp.tensor([3.0, 1.0, 2.0], dtype=tp.float32, device="mps")
        self.assertEqual(tp.argsort(x, dim=0).cpu().tolist(), [1, 2, 0])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSBucketize(unittest.TestCase):
    def test_searchsorted(self):
        seq = tp.tensor([1, 3, 5, 7], dtype=tp.int64, device="mps")
        q = tp.tensor([0, 2, 5, 8], dtype=tp.int64, device="mps")
        r = tp.searchsorted(seq, q, out_int32=False, right=False, side=None,
                            sorter=None)
        self.assertEqual(r.cpu().tolist(), [0, 1, 2, 4])
        rr = tp.searchsorted(seq, q, out_int32=False, right=True, side=None,
                             sorter=None)
        self.assertEqual(rr.cpu().tolist(), [0, 1, 3, 4])

    def test_bucketize(self):
        boundaries = tp.tensor([1, 3, 5, 7], dtype=tp.int64, device="mps")
        x = tp.tensor([0, 2, 5, 8], dtype=tp.int64, device="mps")
        r = tp.bucketize(x, boundaries, out_int32=False, right=False)
        self.assertEqual(r.cpu().tolist(), [0, 1, 2, 4])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSDropout(unittest.TestCase):
    def test_dropout(self):
        x = tp.ones((100,), dtype=tp.float32, device="mps")
        y = tp.dropout(x, p=0.5, train=True)
        self.assertEqual(y.shape, (100,))
        vals = set(y.cpu().tolist())
        self.assertTrue(vals.issubset({0.0, 2.0}))
        y2 = tp.dropout(x, p=0.5, train=False)
        self.assertTrue(tp.all(y2.cpu() == 1.0).item())


def ref_computed(x_cpu, w_cpu, b_cpu):
    # Reference layer-norm output with affine for a single row.
    mean = x_cpu.mean(dim=1, keepdim=True)
    var = x_cpu.var(dim=1, keepdim=True, correction=0)
    norm = (x_cpu - mean) / (var + 1e-5).sqrt()
    return norm * w_cpu + b_cpu


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSQuantization(unittest.TestCase):
    def test_quantize_per_tensor_roundtrip(self):
        x = tp.tensor([-1.0, 0.0, 1.0, 2.0], dtype=tp.float32, device="mps")
        for dtype in (tp.quint8, tp.qint8, tp.qint32):
            q = tp.quantize_per_tensor(x, 0.1, 10, dtype)
            self.assertEqual(q.device.type, "mps")
            self.assertEqual(q.dtype, dtype)
            d = tp.dequantize(q)
            self.assertTrue(tp.allclose(d.cpu(), x.cpu(), atol=0.11))

    def test_quantize_per_tensor_int_repr(self):
        x = tp.tensor([-1.0, 0.0, 1.0, 2.0], dtype=tp.float32, device="mps")
        q = tp.quantize_per_tensor(x, 0.1, 10, tp.quint8)
        codes = tp.int_repr(q).cpu().tolist()
        # round(-1/0.1)+10 = 0, round(0)+10 = 10, round(1/0.1)+10 = 20,
        # round(2/0.1)+10 = 30.
        self.assertEqual(codes, [0, 10, 20, 30])

    def test_quantized_empty_like_and_copy(self):
        x = tp.tensor([1.0, 2.0], dtype=tp.float32, device="mps")
        q = tp.quantize_per_tensor(x, 0.5, 0, tp.qint8)
        y = tp.empty_like(q)
        self.assertEqual(y.device.type, "mps")
        self.assertEqual(y.dtype, tp.qint8)
        y.copy_(q)
        self.assertEqual(tp.int_repr(y).cpu().tolist(), tp.int_repr(q).cpu().tolist())
        self.assertTrue(tp.allclose(tp.dequantize(y).cpu(), x.cpu(), atol=0.51))

    def test_quantize_per_channel_roundtrip(self):
        x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32, device="mps")
        scales = tp.tensor([0.1, 0.2], dtype=tp.float32, device="mps")
        zps = tp.tensor([0, 0], dtype=tp.int64).mps()
        q = tp.quantize_per_channel(x, scales, zps, 0, tp.qint8)
        self.assertEqual(q.device.type, "mps")
        self.assertEqual(q.dtype, tp.qint8)
        codes = tp.int_repr(q).cpu().tolist()
        self.assertEqual(codes, [[10, 10], [30, 20]])
        d = tp.dequantize(q)
        self.assertTrue(tp.allclose(d.cpu(), x.cpu(), atol=1e-6))

    def test_weight_int8pack_mm(self):
        a = tp.tensor([[0.5, -1.0, 2.0, 0.25],
                       [1.0, 0.5, -0.5, 2.0]], dtype=tp.float32, device="mps")
        w = tp.tensor([[1, -2, 0, 1],
                       [0, 1, -1, 2],
                       [2, 0, 1, -1]], dtype=tp.int8, device="mps")
        s = tp.tensor([0.5, 1.0, 2.0], dtype=tp.float32, device="mps")
        c = tp._C._weight_int8pack_mm(a, w, s)
        ref = (a.cpu() @ w.cpu().to(tp.float32).T) * s.cpu()
        self.assertTrue(tp.allclose(c.cpu(), ref, atol=1e-5))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSSparse(unittest.TestCase):
    def test_sparse_coo_tensor_and_to_dense(self):
        indices = tp.tensor([[0, 1, 1], [2, 0, 2]], dtype=tp.int64).mps()
        values = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32, device="mps")
        s = tp.sparse_coo_tensor(indices, values, (2, 3))
        self.assertEqual(s.device.type, "mps")
        self.assertTrue(s.is_sparse)
        dense = s.to_dense()
        self.assertEqual(dense.cpu().tolist(), [[0.0, 0.0, 1.0],
                                                [2.0, 0.0, 3.0]])
        self.assertEqual(s._nnz(), 3)

    def test_to_sparse_roundtrip(self):
        x = tp.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]],
                      dtype=tp.float32, device="mps")
        s = x.to_sparse()
        self.assertTrue(s.is_sparse)
        self.assertEqual(s._nnz(), 3)
        back = s.to_dense()
        self.assertTrue(tp.allclose(back.cpu(), x.cpu()))

    def test_coalesce(self):
        indices = tp.tensor([[0, 0], [1, 1]], dtype=tp.int64).mps()
        values = tp.tensor([1.0, 2.0], dtype=tp.float32, device="mps")
        s = tp.sparse_coo_tensor(indices, values, (1, 2))
        c = s.coalesce()
        self.assertEqual(c._nnz(), 1)
        self.assertTrue(tp.allclose(c.to_dense().cpu(), [[3.0, 0.0]]))

    def test_sparse_mm(self):
        indices = tp.tensor([[0, 1], [0, 1]], dtype=tp.int64).mps()
        values = tp.tensor([2.0, 3.0], dtype=tp.float32, device="mps")
        s = tp.sparse_coo_tensor(indices, values, (2, 2))
        dense = tp.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=tp.float32,
                       device="mps")
        out = tp.sparse_mm(s, dense)
        self.assertEqual(out.cpu().tolist(), [[2.0, 0.0], [0.0, 3.0]])

    def test_sparse_add_mul(self):
        i = tp.tensor([[0], [0]], dtype=tp.int64).mps()
        a = tp.sparse_coo_tensor(i, tp.tensor([1.0], device="mps"), (1, 1))
        b = tp.sparse_coo_tensor(i, tp.tensor([2.0], device="mps"), (1, 1))
        s = tp.sparse_add(a, b)
        self.assertEqual(s.to_dense().cpu().tolist(), [[3.0]])
        m = tp.sparse_mul(a, b)
        self.assertEqual(m.to_dense().cpu().tolist(), [[2.0]])

    def test_sparse_sum(self):
        indices = tp.tensor([[0, 1, 1], [2, 0, 2]], dtype=tp.int64).mps()
        values = tp.tensor([1.0, 2.0, 3.0], dtype=tp.float32, device="mps")
        s = tp.sparse_coo_tensor(indices, values, (2, 3))
        self.assertAlmostEqual(tp.sparse_sum(s).item(), 6.0, places=5)
        row = tp.sparse_sum(s, dim=[0])
        self.assertEqual(row.to_dense().cpu().tolist(), [[2.0, 0.0, 4.0]])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSProfiler(unittest.TestCase):
    """MPS-specific profiler behavior.

    The shared profiler frontend (tensorplay.profiler) is backend-agnostic and
    is covered by test/profiler/test_profiler.py; these tests only pin the
    MPS-specific pieces: the activity flag, the automatic event-pair timing
    request, and the resolution of per-op GPU time from the Metal command
    buffer.

    The MPS profiler support is gated on the ProfilerActivity.MPS enum member
    being present, so these tests skip on builds where the MPS profiler has
    not been compiled in yet.
    """

    def _mps_profiler_available(self):
        from tensorplay import profiler as tp_prof

        return hasattr(tp_prof.ProfilerActivity, "MPS")

    def test_supported_activities_includes_mps(self):
        from tensorplay import profiler as tp_prof

        if not self._mps_profiler_available():
            self.skipTest("MPS profiler not available in this build")
        self.assertIn(tp_prof.ProfilerActivity.MPS,
                      tp_prof.supported_activities())

    def test_mps_activity_implies_gpu_timing(self):
        from tensorplay import profiler as tp_prof

        if not self._mps_profiler_available():
            self.skipTest("MPS profiler not available in this build")
        # MPS has no kernel-level trace collector; requesting the MPS
        # activity must turn on the event-pair timing path automatically.
        with tp_prof.profile(activities=[tp_prof.ProfilerActivity.MPS]) as prof:
            self.assertTrue(prof.gpu_timing)
            a = tp.ones((64, 64), dtype=tp.float32, device="mps")
            b = a + a

    def test_mps_gpu_timing_resolves(self):
        from tensorplay import profiler as tp_prof

        if not self._mps_profiler_available():
            self.skipTest("MPS profiler not available in this build")
        a = tp.ones((4096,), dtype=tp.float32, device="mps")
        with tp_prof.profile(
            activities=[tp_prof.ProfilerActivity.MPS],
            record_shapes=True,
        ) as prof:
            a.add(a)
            a.mul(a)
        tp.mps.synchronize()
        timed = [ev for ev in prof.events if ev[0].startswith(("add", "mul"))]
        self.assertTrue(timed)
        # Event field 8 is the per-op GPU time in milliseconds; MPS fills it
        # from the committed command buffer's GPUStartTime/GPUEndTime.
        self.assertTrue(all(ev[8] >= 0 for ev in timed))

    def test_cpu_only_profile_keeps_negative_gpu_time(self):
        from tensorplay import profiler as tp_prof

        # Without the MPS activity request the frontend must behave exactly
        # as on any other backend: no device timing, gpu_ms stays negative.
        with tp_prof.profile() as prof:
            tp.ones((4, 4)).add(tp.ones((4, 4)))
        self.assertTrue(all(ev[8] < 0 for ev in prof.events))


if __name__ == "__main__":
    unittest.main()
