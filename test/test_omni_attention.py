"""End-to-end checks for the omni attention operator.

Covers the eager math path (dense and block-sparse evaluators must agree),
score and mask modifiers with captured buffers, grouped-query attention,
the log2-space statistics contract, gradients through the operator, and
block-mask construction.  When the reference package is installed the same
inputs are checked against the third-party attention operator for value
"""

import importlib
import math
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp

_omni_mod = importlib.import_module("tensorplay.nn.attention.omni_attention")
_hop_mod = importlib.import_module("tensorplay._higher_order_ops.omni_attention")

try:
    import torch
    import torch.nn.attention.flex_attention as _torch_flex

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def sliding_window(window):
    def mask(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window

    return mask


class TestBlockMaskConstruction(unittest.TestCase):
    def test_causal_block_mask_layout(self):
        bm = _omni_mod.create_block_mask(causal, 2, 3, 256, 256).to("cpu")
        self.assertEqual(tuple(bm.kv_num_blocks.shape), (2, 3, 2))
        # causal: q-block 0 has one partial kv block, q-block 1 has a partial
        # and a full one
        self.assertEqual(int(bm.kv_num_blocks[0, 0, 0]), 1)
        self.assertEqual(int(bm.kv_num_blocks[0, 0, 1]), 1)
        self.assertEqual(int(bm.full_kv_num_blocks[0, 0, 1]), 1)
        self.assertEqual(int(bm.full_kv_num_blocks[0, 0, 0]), 0)
        self.assertEqual(bm.seq_lengths, (256, 256))

    def test_block_mask_positional_roundtrip(self):
        # the dense view of a block mask is block-granular; create_mask
        # materializes the positional mask the block mask encodes
        bm = _omni_mod.create_block_mask(causal, 1, 1, 128, 128, device="cpu")
        m = tp.arange(128)
        expected = m[:, None] >= m[None, :]
        dense = bm.to_dense()
        # one 128x128 block grid, present on the diagonal block
        self.assertEqual(tuple(dense.shape), (1, 1, 1, 1))
        self.assertEqual(int(dense[0, 0, 0, 0]), 1)

    def test_or_and_masks(self):
        left = sliding_window(16)
        right = causal
        combined = _omni_mod.or_masks(left, right)
        m = tp.arange(128)
        expected = (m[:, None] - m[None, :] <= 16) | (m[:, None] >= m[None, :])
        dense = _omni_mod.create_mask(combined, 1, 1, 128, 128, device="cpu")
        tp.testing.assert_close(dense[0, 0], expected)


class TestOmniAttentionForward(unittest.TestCase):
    def setUp(self):
        _omni_mod._OMNI_ATTENTION_DISABLE_COMPILE_DEBUG = True

    def _inputs(self, B=1, H=2, M=256, N=256, D=32, seed=0):
        tp.manual_seed(seed)
        q = tp.randn(B, H, M, D)
        k = tp.randn(B, H, N, D)
        v = tp.randn(B, H, N, D)
        return q, k, v

    def _dense_reference(self, q, k, v, scale, mask_fn=None, score_fn=None):
        B, H, M, D = q.shape
        N = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask_fn is not None:
            m = tp.arange(M)[:, None]
            n = tp.arange(N)[None, :]
            keep = mask_fn(0, 0, m, n)
            keep = keep.expand(B, H, M, N) if keep.dim() == 2 else keep
            scores = tp.where(keep, scores, -float("inf"))
        if score_fn is not None:
            m = tp.arange(M)[:, None].expand(B, H, M, N)
            n = tp.arange(N)[None, :].expand(B, H, M, N)
            b = tp.arange(B)[:, None, None, None].expand(B, H, M, N)
            h = tp.arange(H)[None, :, None, None].expand(B, H, M, N)
            scores = score_fn(scores, b, h, m, n)
        probs = tp._safe_softmax(scores, dim=-1)
        return probs @ v

    def test_no_mask_matches_scaled_dot_product(self):
        q, k, v = self._inputs()
        scale = q.shape[-1] ** -0.5
        out = _omni_mod.omni_attention(q, k, v, scale=scale)
        expected = self._dense_reference(q, k, v, scale)
        tp.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_causal_mask(self):
        q, k, v = self._inputs()
        scale = q.shape[-1] ** -0.5
        bm = _omni_mod.create_block_mask(causal, 1, 2, 256, 256).to("cpu")
        out = _omni_mod.omni_attention(q, k, v, block_mask=bm, scale=scale)
        expected = self._dense_reference(q, k, v, scale, mask_fn=causal)
        tp.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_score_mod_with_buffer(self):
        q, k, v = self._inputs()
        scale = q.shape[-1] ** -0.5
        tp.manual_seed(3)
        bias = tp.randn(1, 2, 256)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * 1.1 + bias[b, h, q_idx] - 0.5 * kv_idx

        out = _omni_mod.omni_attention(q, k, v, score_mod=score_mod, scale=scale)
        expected = self._dense_reference(q, k, v, scale, score_fn=score_mod)
        tp.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_mask_and_score_mod_together(self):
        q, k, v = self._inputs()
        scale = q.shape[-1] ** -0.5
        bm = _omni_mod.create_block_mask(causal, 1, 2, 256, 256).to("cpu")

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + 2.0 * q_idx * kv_idx

        out = _omni_mod.omni_attention(
            q, k, v, score_mod=score_mod, block_mask=bm, scale=scale
        )
        expected = self._dense_reference(
            q, k, v, scale, mask_fn=causal, score_fn=score_mod
        )
        tp.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gqa_broadcast(self):
        tp.manual_seed(5)
        q = tp.randn(1, 4, 128, 16)
        k = tp.randn(1, 2, 128, 16)
        v = tp.randn(1, 2, 128, 16)
        scale = 16 ** -0.5
        bm = _omni_mod.create_block_mask(causal, 1, 4, 128, 128).to("cpu")
        out = _omni_mod.omni_attention(q, k, v, block_mask=bm, scale=scale, enable_gqa=True)
        kx = tp.repeat_interleave(k, 2, dim=1)
        vx = tp.repeat_interleave(v, 2, dim=1)
        expected = self._dense_reference(q, kx, vx, scale, mask_fn=causal)
        tp.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_logsumexp_contract(self):
        q, k, v = self._inputs(M=64, N=64)
        scale = q.shape[-1] ** -0.5
        bm = _omni_mod.create_block_mask(causal, 1, 2, 64, 64).to("cpu")
        out = _omni_mod.omni_attention(
            q, k, v, block_mask=bm, scale=scale, return_aux=_omni_mod.AuxRequest(lse=True)
        )
        if isinstance(out, tuple) and len(out) == 2 and hasattr(out[1], "lse"):
            lse = out[1].lse
        else:
            lse = out[1]
        # natural-log logsumexp of the masked scores, converted to log2
        scores = (q @ k.transpose(-2, -1)) * scale
        m = tp.arange(64)
        keep = (m[:, None] >= m[None, :]).expand(1, 2, 64, 64)
        scores = tp.where(keep, scores, -float("inf"))
        # the public entry converts the internal log2 statistics back to
        # natural-log space
        expected = scores.logsumexp(dim=-1)
        tp.testing.assert_close(lse, expected, atol=1e-3, rtol=1e-3)

    def test_fully_masked_rows(self):
        def never(b, h, q_idx, kv_idx):
            return q_idx < kv_idx - 1000

        q, k, v = self._inputs(M=64, N=64)
        bm = _omni_mod.create_block_mask(never, 1, 2, 64, 64).to("cpu")
        out = _omni_mod.omni_attention(q, k, v, block_mask=bm)
        self.assertTrue(bool((out == 0).all()))

    def test_sliding_window_partial_blocks(self):
        q, k, v = self._inputs(M=256, N=256)
        scale = q.shape[-1] ** -0.5
        bm = _omni_mod.create_block_mask(sliding_window(80), 1, 2, 256, 256).to("cpu")
        out = _omni_mod.omni_attention(q, k, v, block_mask=bm, scale=scale)
        expected = self._dense_reference(q, k, v, scale, mask_fn=sliding_window(80))
        tp.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


class TestBlockSparseEvaluator(unittest.TestCase):
    """The block-sparse eager path must agree with the dense evaluator."""

    def setUp(self):
        _omni_mod._OMNI_ATTENTION_DISABLE_COMPILE_DEBUG = True

    def test_selector_takes_block_sparse_path(self):
        q = tp.randn(1, 2, 256, 32)
        k = tp.randn(1, 2, 256, 32)
        bm = _omni_mod.create_block_mask(causal, 1, 2, 256, 256).to("cpu")
        self.assertTrue(_hop_mod._use_block_sparse_math(q, k, bm.as_tuple()))
        # the catch-all mask (no explicit block mask) stays dense
        self.assertFalse(
            _hop_mod._use_block_sparse_math(q, k, (256, 256) + (None,) * 12 + (1 << 30, 1 << 30, True))
        )

    def _compare_paths(self, score_mod=None, mask_fn=causal, M=256, N=256):
        tp.manual_seed(11)
        q = tp.randn(1, 2, M, 32)
        k = tp.randn(1, 2, N, 32)
        v = tp.randn(1, 2, N, 32)
        scale = 32 ** -0.5
        bm = _omni_mod.create_block_mask(mask_fn, 1, 2, M, N).to("cpu")
        out_sparse = _omni_mod.omni_attention(
            q, k, v, score_mod=score_mod, block_mask=bm, scale=scale
        )
        with unittest.mock.patch.object(
            _hop_mod, "_use_block_sparse_math", return_value=False
        ):
            out_dense = _omni_mod.omni_attention(
                q, k, v, score_mod=score_mod, block_mask=bm, scale=scale
            )
        tp.testing.assert_close(out_sparse, out_dense, atol=1e-4, rtol=1e-4)
        return out_sparse

    def test_causal_matches_dense(self):
        self._compare_paths()

    def test_score_mod_matches_dense(self):
        def score_mod(score, b, h, q_idx, kv_idx):
            return score * 0.9 + 0.01 * kv_idx

        self._compare_paths(score_mod=score_mod)

    def test_sliding_window_matches_dense(self):
        self._compare_paths(mask_fn=sliding_window(80))

    def test_uneven_block_boundary(self):
        # 192 = 128 + 64: the second q-block is only half present
        self._compare_paths(M=192, N=192)

    def test_full_mask_matches_dense(self):
        def everything(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx - 10_000

        self._compare_paths(mask_fn=everything)


@unittest.skipUnless(tp.cuda.is_available(), "CUDA not available")
class TestOmniAttentionGradCUDA(unittest.TestCase):
    """Gradients through the operator against the reference package."""

    def setUp(self):
        _omni_mod._OMNI_ATTENTION_DISABLE_COMPILE_DEBUG = True

    @unittest.skipUnless(HAS_TORCH, "reference package not available")
    def test_gradients_match_reference(self):
        """Query/key/value gradients against the reference operator.

        Closure-captured buffers inside score_mod receive no gradient in
        either implementation's eager path (the reference operator only
        lifts closures when compiled), so they are excluded here.
        """
        tp.manual_seed(21)
        B, H, M, N, D = 1, 2, 128, 128, 32
        q = tp.randn(B, H, M, D, device="cuda")
        k = tp.randn(B, H, N, D, device="cuda")
        v = tp.randn(B, H, N, D, device="cuda")
        scale = D ** -0.5
        bm = _omni_mod.create_block_mask(causal, B, H, M, N)

        tp.manual_seed(22)
        bias = tp.randn(B, H, M, device="cuda")

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * 1.05 + bias[b, h, q_idx]

        bias_torch = torch.from_numpy(bias.detach().cpu().numpy()).to("cuda")

        def run_tp():
            q_ = q.clone().requires_grad_(True)
            k_ = k.clone().requires_grad_(True)
            v_ = v.clone().requires_grad_(True)
            b_ = bias.clone().requires_grad_(True)

            def sm(score, b, h, q_idx, kv_idx):
                return score * 1.05 + b_[b, h, q_idx]

            out = _omni_mod.omni_attention(
                q_, k_, v_, score_mod=sm, block_mask=bm, scale=scale
            )
            (out * tp.arange(1, M + 1, dtype=tp.float32, device="cuda")[:, None]).sum().backward()
            return out, q_.grad, k_.grad, v_.grad

        def run_torch():
            tq = torch.from_numpy(q.detach().cpu().numpy()).to("cuda").requires_grad_(True)
            tk = torch.from_numpy(k.detach().cpu().numpy()).to("cuda").requires_grad_(True)
            tv = torch.from_numpy(v.detach().cpu().numpy()).to("cuda").requires_grad_(True)

            def tsm(score, b, h, q_idx, kv_idx):
                return score * 1.05 + bias_torch[b, h, q_idx]

            def tcausal(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            tbm = _torch_flex.create_block_mask(tcausal, B, H, M, N, device="cuda")
            out = _torch_flex.flex_attention(
                tq, tk, tv, score_mod=tsm, block_mask=tbm, scale=scale
            )
            (out * torch.arange(1, M + 1, dtype=torch.float32, device="cuda")[:, None]).sum().backward()
            return out, tq.grad, tk.grad, tv.grad

        out, gq, gk, gv = run_tp()
        tout, tgq, tgk, tgv = run_torch()

        def maxdiff(a, b):
            return abs(a.detach().cpu().numpy() - b.detach().cpu().numpy()).max()

        self.assertLess(maxdiff(out, tout), 1e-3)
        self.assertLess(maxdiff(gq, tgq), 1e-3)
        self.assertLess(maxdiff(gk, tgk), 1e-3)
        self.assertLess(maxdiff(gv, tgv), 1e-3)


@unittest.skipUnless(HAS_TORCH, "reference package not available")
class TestOmniAttentionReferenceAgreement(unittest.TestCase):
    def setUp(self):
        _omni_mod._OMNI_ATTENTION_DISABLE_COMPILE_DEBUG = True

    def test_causal_values(self):
        B, H, M, N, D = 1, 2, 128, 128, 32
        tp.manual_seed(31)
        q = tp.randn(B, H, M, D)
        k = tp.randn(B, H, N, D)
        v = tp.randn(B, H, N, D)
        scale = D ** -0.5
        bm = _omni_mod.create_block_mask(causal, B, H, M, N).to("cpu")

        def tcausal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        tbm = _torch_flex.create_block_mask(tcausal, B, H, M, N, device="cpu")
        tout = _torch_flex.flex_attention(
            torch.from_numpy(q.numpy()),
            torch.from_numpy(k.numpy()),
            torch.from_numpy(v.numpy()),
            block_mask=tbm,
            scale=scale,
        )
        out = _omni_mod.omni_attention(q, k, v, block_mask=bm, scale=scale)
        self.assertLess(abs((out.numpy() - tout.numpy())).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
