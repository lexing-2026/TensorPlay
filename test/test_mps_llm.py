"""MPS LLM-oriented kernel tests: rms_norm, embedding backward,
expand/repeat, topk/sample, fused AdamW, SDPA and the GPU factory shaders.

Every test is skipped when the runner has no Metal device, matching the
conventions of test_mps.py.
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp


def mps_available():
    return tp.mps.is_available()


def close(a, b, tol=1e-3):
    return tp.all((a - b).abs() < tol).item()


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSRMSNorm(unittest.TestCase):
    def test_rms_norm(self):
        x = tp.randn(4, 8, device="mps")
        w = tp.randn(8, device="mps")
        y = tp.rms_norm(x, [8], w)
        y_cpu = tp.rms_norm(x.cpu(), [8], w.cpu())
        self.assertTrue(close(y.cpu(), y_cpu, 1e-3))

    def test_fused_rms_norm(self):
        x = tp.randn(2, 16, device="mps")
        w = tp.randn(16, device="mps")
        out, rstd = tp._fused_rms_norm(x, [16], w, 1e-5)
        out_cpu, rstd_cpu = tp._fused_rms_norm(x.cpu(), [16], w.cpu(), 1e-5)
        self.assertTrue(close(out.cpu(), out_cpu, 1e-3))
        self.assertEqual(rstd.shape, rstd_cpu.shape)
        self.assertTrue(close(rstd.cpu(), rstd_cpu, 1e-3))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSEmbedding(unittest.TestCase):
    def test_embedding_forward_backward(self):
        weight = tp.randn(10, 4, device="mps", requires_grad=True)
        indices = tp.tensor([1, 3, 5], device="mps")
        out = tp.embedding(weight, indices)
        self.assertEqual(list(out.shape), [3, 4])
        out.sum().backward()
        self.assertIsNotNone(weight.grad)
        # Compare against CPU backward.
        weight_cpu = tp.randn(10, 4, device="cpu", requires_grad=True)
        out_cpu = tp.embedding(weight_cpu, indices.cpu())
        out_cpu.sum().backward()
        # Shape and nonzero-row structure must match (grad is index-scattered).
        self.assertEqual(weight.grad.shape, weight_cpu.grad.shape)
        self.assertTrue(close(weight.grad.cpu(), weight_cpu.grad, 1e-3))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSExpandRepeat(unittest.TestCase):
    def test_expand(self):
        x = tp.randn(1, 3, device="mps")
        y = x.expand(4, 3)
        self.assertEqual(list(y.shape), [4, 3])
        self.assertTrue(close(y.cpu(), x.cpu().expand(4, 3), 1e-5))

    def test_repeat(self):
        x = tp.randn(2, 3, device="mps")
        y = x.repeat(2, 2)
        self.assertEqual(list(y.shape), [4, 6])
        self.assertTrue(close(y.cpu(), x.cpu().repeat(2, 2), 1e-5))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSTopKSample(unittest.TestCase):
    def test_topk(self):
        x = tp.randn(2, 8, device="mps")
        values, indices = tp.topk(x, 3, dim=-1)
        self.assertEqual(list(values.shape), [2, 3])
        self.assertEqual(list(indices.shape), [2, 3])
        values_cpu, indices_cpu = tp.topk(x.cpu(), 3, dim=-1)
        self.assertTrue(close(values.cpu(), values_cpu, 1e-3))

    def test_sample(self):
        logits = tp.randn(4, 16, device="mps")
        tokens = tp.sample(logits, temperature=1.0, top_k=0, top_p=1.0)
        self.assertEqual(list(tokens.shape), [4])
        self.assertEqual(tokens.dtype, tp.int64)
        self.assertTrue(tp.all((tokens >= 0) & (tokens < 16)).item())

        tokens_p = tp.sample(logits, temperature=0.8, top_k=5, top_p=0.9)
        self.assertEqual(list(tokens_p.shape), [4])


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSFusedOptimizer(unittest.TestCase):
    def _step(self, device, adamw):
        p = tp.tensor([0.5, -1.0, 2.0, -0.25, 3.0, 0.75], device=device)
        g = tp.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], device=device)
        ea = tp.zeros(6, device=device)
        es = tp.zeros(6, device=device)
        step = tp.tensor(1.0, dtype=tp.float32, device=device)
        fn = tp._fused_adamw_ if adamw else tp._fused_adam_
        # amsgrad is disabled, so the max-exp-avg-sqs list must be empty.
        fn([p], [g], [ea], [es], [], [step], lr=0.01, beta1=0.9,
           beta2=0.999, weight_decay=0.01, eps=1e-8, amsgrad=False,
           maximize=False)
        self.assertEqual(step.item(), 1.0)
        self.assertTrue(tp.all(tp.isfinite(p)).item())
        return p.cpu()

    def test_fused_adamw(self):
        p_mps = self._step("mps", adamw=True)
        p_cpu = self._step("cpu", adamw=True)
        self.assertTrue(close(p_mps, p_cpu, 1e-4))

    def test_fused_adam(self):
        p_mps = self._step("mps", adamw=False)
        p_cpu = self._step("cpu", adamw=False)
        self.assertTrue(close(p_mps, p_cpu, 1e-4))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSSDPA(unittest.TestCase):
    def test_sdpa_mha(self):
        q = tp.randn(2, 4, 6, 8, device="mps")
        k = tp.randn(2, 4, 10, 8, device="mps")
        v = tp.randn(2, 4, 10, 8, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v)
        out_cpu = tp.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
        self.assertEqual(list(out.shape), [2, 4, 6, 8])
        self.assertTrue(close(out.cpu(), out_cpu, 1e-2))

    def test_sdpa_causal(self):
        q = tp.randn(1, 2, 4, 8, device="mps")
        k = tp.randn(1, 2, 4, 8, device="mps")
        v = tp.randn(1, 2, 4, 8, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_cpu = tp.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(),
                                                  is_causal=True)
        self.assertTrue(close(out.cpu(), out_cpu, 1e-2))

    def test_sdpa_prefill(self):
        q = tp.randn(1, 2, 16, 64, device="mps")
        k = tp.randn(1, 2, 32, 64, device="mps")
        v = tp.randn(1, 2, 32, 64, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v)
        out_cpu = tp.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
        self.assertEqual(list(out.shape), [1, 2, 16, 64])
        self.assertTrue(close(out.cpu(), out_cpu, 1e-2))

    def test_sdpa_causal_rectangular(self):
        q = tp.randn(1, 2, 3, 8, device="mps")
        k = tp.randn(1, 2, 5, 8, device="mps")
        v = tp.randn(1, 2, 5, 8, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v, is_causal=True)
        expected = tp.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), is_causal=True)
        self.assertTrue(close(out.cpu(), expected, 1e-2))

    def test_sdpa_gqa_graph(self):
        q = tp.randn(1, 4, 4, 8, device="mps")
        k = tp.randn(1, 2, 6, 8, device="mps")
        v = tp.randn(1, 2, 6, 8, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        expected = tp.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), enable_gqa=True)
        self.assertTrue(close(out.cpu(), expected, 1e-2))

    def test_sdpa_2pass(self):
        q = tp.randn(1, 1, 4, 64, device="mps")
        k = tp.randn(1, 1, 1024, 64, device="mps")
        v = tp.randn(1, 1, 1024, 64, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v)
        out_cpu = tp.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
        self.assertTrue(close(out.cpu(), out_cpu, 1e-2))

    def test_sdpa_explicit_scale(self):
        for query_length, key_length in ((4, 64), (4, 1024), (16, 32)):
            with self.subTest(query_length=query_length, key_length=key_length):
                q = tp.randn(1, 2, query_length, 64, device="mps")
                k = tp.randn(1, 2, key_length, 64, device="mps")
                v = tp.randn(1, 2, key_length, 64, device="mps")
                out = tp.scaled_dot_product_attention(q, k, v, scale=0.25)
                expected = tp.scaled_dot_product_attention(
                    q.cpu(), k.cpu(), v.cpu(), scale=0.25)
                self.assertTrue(close(out.cpu(), expected, 1e-2))

    def test_sdpa_bool_mask(self):
        q = tp.randn(1, 2, 4, 64, device="mps")
        k = tp.randn(1, 2, 6, 64, device="mps")
        v = tp.randn(1, 2, 6, 64, device="mps")
        mask = tp.randint(0, 2, (1, 1, 4, 6), device="mps").bool()
        out = tp.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out_cpu = tp.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(),
                                                  attn_mask=mask.cpu())
        self.assertTrue(close(out.cpu(), out_cpu, 1e-2))

    def test_sdpa_head_broadcast_mask(self):
        # A mask that leaves the head axis at one still reaches the graph path,
        # where it is spread over the heads before the scores are shifted.
        q = tp.randn(2, 2, 4, 8, device="mps")
        k = tp.randn(2, 2, 6, 8, device="mps")
        v = tp.randn(2, 2, 6, 8, device="mps")
        mask = tp.randn(2, 1, 4, 6, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        expected = tp.scaled_dot_product_attention(
            q.cpu(), k.cpu(), v.cpu(), attn_mask=mask.cpu())
        self.assertTrue(close(out.cpu(), expected, 1e-2))

    def test_sdpa_gqa(self):
        q = tp.randn(1, 4, 4, 64, device="mps")
        k = tp.randn(1, 2, 6, 64, device="mps")
        v = tp.randn(1, 2, 6, 64, device="mps")
        out = tp.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        out_cpu = tp.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(),
                                                  enable_gqa=True)
        self.assertTrue(close(out.cpu(), out_cpu, 1e-2))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSEmbeddingBag(unittest.TestCase):
    def test_embedding_bag_sum(self):
        weight = tp.randn(10, 4, device="mps")
        indices = tp.tensor([0, 1, 2, 2, 5], device="mps")
        offsets = tp.tensor([0, 3], device="mps")
        out, offset2bag, bag_size, max_idx = tp.functional._embedding_bag(
            weight, indices, offsets, mode=0)
        out_cpu, _, _, _ = tp.functional._embedding_bag(
            weight.cpu(), indices.cpu(), offsets.cpu(), mode=0)
        self.assertEqual(list(out.shape), [2, 4])
        self.assertTrue(close(out.cpu(), out_cpu, 1e-3))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSFactoryShaders(unittest.TestCase):
    def test_rand_range(self):
        x = tp.rand(1000, device="mps")
        self.assertTrue(tp.all(x >= 0).item())
        self.assertTrue(tp.all(x < 1).item())

    def test_randn_shape(self):
        x = tp.randn(64, 64, device="mps")
        self.assertEqual(list(x.shape), [64, 64])
        self.assertTrue(tp.all(tp.isfinite(x)).item())

    def test_arange(self):
        x = tp.arange(10, device="mps")
        self.assertTrue(close(x.cpu(), tp.arange(10), 1e-5))

    def test_eye(self):
        x = tp.eye(4, device="mps")
        self.assertTrue(close(x.cpu(), tp.eye(4), 1e-5))

    def test_randint(self):
        x = tp.randint(0, 100, (50,), device="mps")
        self.assertTrue(tp.all(x >= 0).item())
        self.assertTrue(tp.all(x < 100).item())


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSFusedSGD(unittest.TestCase):
    def _step(self, device, momentum, nesterov=False, p=None, g=None):
        p = tp.randn(6, device=device) if p is None else p
        g = tp.randn(6, device=device) if g is None else g
        bufs = [tp.zeros(6, device=device)] if momentum != 0 else []
        tp._fused_sgd_([p], [g], bufs, weight_decay=0.01, momentum=momentum,
                       lr=0.01, dampening=0.0, nesterov=nesterov,
                       maximize=False, is_first_step=False)
        return p.cpu()

    def _mps_vs_cpu(self, momentum, nesterov=False):
        # The fused op updates p in place, so the CPU reference must start
        # from copies of the original parameters and gradients.
        p = tp.randn(6, device="mps")
        g = tp.randn(6, device="mps")
        p_ref = p.cpu().clone()
        g_ref = g.cpu().clone()
        p_mps = self._step("mps", momentum, nesterov, p=p, g=g)
        p_cpu = self._step("cpu", momentum, nesterov, p=p_ref, g=g_ref)
        return close(p_mps, p_cpu, 1e-4)

    def test_sgd_no_momentum(self):
        self.assertTrue(self._mps_vs_cpu(0.0))

    def test_sgd_momentum(self):
        self.assertTrue(self._mps_vs_cpu(0.9))

    def test_sgd_nesterov(self):
        self.assertTrue(self._mps_vs_cpu(0.9, True))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSAmp(unittest.TestCase):
    def test_non_finite_check_and_unscale(self):
        grads = [tp.tensor([1.0, 2.0, 3.0], device="mps"),
                 tp.tensor([4.0, float("inf"), 6.0], device="mps")]
        found = tp.zeros((), dtype=tp.float32, device="mps")
        inv_scale = tp.tensor(0.5, dtype=tp.float32, device="mps")
        tp._amp_foreach_non_finite_check_and_unscale_(grads, found, inv_scale)
        self.assertEqual(found.item(), 1.0)
        # Every element is scaled by inv_scale even when a non-finite value is
        # found, matching the GPU semantics.
        self.assertTrue(close(grads[0].cpu(), tp.tensor([0.5, 1.0, 1.5]), 1e-5))

    def test_non_finite_clean(self):
        grads = [tp.tensor([1.0, 2.0], device="mps")]
        found = tp.zeros((), dtype=tp.float32, device="mps")
        inv_scale = tp.tensor(0.5, dtype=tp.float32, device="mps")
        tp._amp_foreach_non_finite_check_and_unscale_(grads, found, inv_scale)
        self.assertEqual(found.item(), 0.0)
        self.assertTrue(close(grads[0].cpu(), tp.tensor([0.5, 1.0]), 1e-5))

    def test_update_scale_growth(self):
        scale = tp.tensor(1.0, dtype=tp.float32, device="mps")
        gt = tp.tensor(0, dtype=tp.int32, device="mps")
        found = tp.tensor(0.0, dtype=tp.float32, device="mps")
        tp._amp_update_scale_(scale, gt, found, 2.0, 0.5, 10)
        self.assertEqual(gt.item(), 1)
        self.assertTrue(close(scale.cpu(), tp.tensor(1.0), 1e-5))

    def test_update_scale_backoff(self):
        scale = tp.tensor(1.0, dtype=tp.float32, device="mps")
        gt = tp.tensor(5, dtype=tp.int32, device="mps")
        found = tp.tensor(1.0, dtype=tp.float32, device="mps")
        tp._amp_update_scale_(scale, gt, found, 2.0, 0.5, 10)
        self.assertEqual(gt.item(), 0)
        self.assertTrue(close(scale.cpu(), tp.tensor(0.5), 1e-5))


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSSampling(unittest.TestCase):
    def test_multinomial_replacement(self):
        probs = tp.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], device="mps")
        samples = tp.multinomial(probs, 1000, replacement=True)
        self.assertEqual(list(samples.shape), [2, 1000])
        counts = (samples == 2).sum(dim=1).float() / 1000
        self.assertTrue((counts[0] - 0.7).abs().item() < 0.1)

    def test_multinomial_no_replacement(self):
        probs = tp.tensor([0.1, 0.2, 0.7], device="mps")
        samples = tp.multinomial(probs, 2, replacement=False)
        self.assertEqual(list(samples.shape), [2])
        self.assertEqual(len(set(samples.tolist())), 2)


@unittest.skipUnless(mps_available(), "MPS not available")
class TestMPSEqual(unittest.TestCase):
    def test_equal(self):
        a = tp.randn(4, 4, device="mps")
        self.assertTrue(tp.equal(a, a.clone()))
        self.assertFalse(tp.equal(a, a + 1))
        self.assertFalse(tp.equal(a, tp.randn(3, 3, device="mps")))


if __name__ == "__main__":
    unittest.main()
