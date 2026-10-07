"""Tests for compressed sparse attention, modern blocks, and the CED scaffold."""

import unittest

import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.nn.cache import SlidingWindowCache
from tensorplay.nn.modules.compressed_attention import (
    CompressedSparseAttention,
    _LightningIndexer,
    _WindowCompressor,
)
from tensorplay.nn.modules.causal_encoder_decoder import CrossAttention, CausalEncoderDecoder
from tensorplay.nn.modules.gated_delta import GatedDeltaNet
from tensorplay.nn.modules.modern_block import SwiGLUMLP, TransformerBlock
from tensorplay.nn.modules.rotary import RotaryEmbedding, apply_rotary_emb
from tensorplay.nn.parameter import Parameter


class TestWindowCompressor(unittest.TestCase):
    def _compressor(self, m, head_dim):
        tp.manual_seed(0)
        hidden = 2 * head_dim
        comp = _WindowCompressor(hidden, head_dim, m, rms_norm_eps=1e-6)
        # Identity key projection and zero gates make the softmax uniform, so
        # each window's entry reduces to a plain mean of its vote vectors.
        comp.kv_proj.weight = Parameter(tp.eye(2 * head_dim))
        comp.gate_proj.weight = Parameter(tp.zeros(2 * head_dim, hidden))
        comp.eval()
        return comp

    def test_single_window_uniform_gates(self):
        m, hd = 2, 4
        comp = self._compressor(m, hd)
        rope = RotaryEmbedding(hd)
        x = tp.randn(1, m, 2 * hd)
        entries, state = comp(x, rope, seq_start=0, state={})
        expected = comp.kv_norm(x[:, :, hd:].mean(dim=1))  # [1, hd]
        expected, _ = apply_rotary_emb(
            expected.unsqueeze(1).unsqueeze(1), expected.unsqueeze(1).unsqueeze(1),
            *rope(x, tp.zeros(1, 1, dtype=tp.int64)),
        )
        self.assertEqual(entries.shape, tp.Size((1, 1, hd)))
        self.assertLess((entries - expected).abs().max().item(), 1e-5)
        # One entry whose last source token is the window's last position.
        self.assertEqual(state["entry_end"][0].item(), m - 1)
        # The tail holds nothing after a complete window.
        self.assertEqual(state["tail_kv"].shape[1], 0)

    def test_overlap_series_feeds_the_next_window(self):
        m, hd = 2, 4
        comp = self._compressor(m, hd)
        rope = RotaryEmbedding(hd)
        x = tp.randn(1, 2 * m, 2 * hd)
        entries, state = comp(x, rope, seq_start=0, state={})

        # Window 1 mixes window 0's Ca half with window 1's Cb half.
        votes = tp.cat([x[:, :m, :hd], x[:, m:, hd:]], dim=1)
        expected_0 = comp.kv_norm(x[:, :m, hd:].mean(dim=1))
        expected_1 = comp.kv_norm(votes.mean(dim=1))
        expected = tp.stack([expected_0, expected_1], dim=1)  # [1, 2, hd]
        expected, _ = apply_rotary_emb(
            expected.unsqueeze(1), expected.unsqueeze(1),
            *rope(x, tp.arange(2).unsqueeze(0) * m),
        )
        self.assertLess((entries - expected.squeeze(1)).abs().max().item(), 1e-5)
        # The stored overlap is the last window's Ca slice, ready for the
        # next forward call.
        self.assertTrue(tp.allclose(state["overlap_kv"][0], x[:, m:, :hd].squeeze(0)))

    def test_incomplete_window_is_tail_buffered(self):
        m, hd = 4, 4
        comp = self._compressor(m, hd)
        rope = RotaryEmbedding(hd)
        x = tp.randn(1, m + 2, 2 * hd)
        entries, state = comp(x, rope, seq_start=5, state={})
        self.assertEqual(entries.shape[1], 1)
        self.assertEqual(state["tail_kv"].shape[1], 2)
        self.assertEqual(state["tail_start"], 5 + m)
        # The entry's rope position follows seq_start, not 0.
        expected = comp.kv_norm(x[:, :m, hd:].mean(dim=1))
        expected, _ = apply_rotary_emb(
            expected.unsqueeze(1).unsqueeze(1), expected.unsqueeze(1).unsqueeze(1),
            *rope(x, tp.tensor([[5]])),
        )
        self.assertLess((entries - expected).abs().max().item(), 1e-5)


class TestCompressedSparseAttention(unittest.TestCase):
    def _module(self, seed=0, **kwargs):
        tp.manual_seed(seed)
        defaults = dict(
            hidden_size=32, num_heads=4, head_dim=16, q_lora_rank=16,
            compress_rate=2, window_size=6, rope_head_dim=8,
            index_n_heads=2, index_head_dim=8, index_topk=3,
        )
        defaults.update(kwargs)
        mod = CompressedSparseAttention(**defaults)
        mod.eval()
        return mod

    def test_output_shape(self):
        mod = self._module()
        self.assertEqual(mod(tp.randn(2, 9, 32)).shape, tp.Size((2, 9, 32)))

    def test_stepwise_cache_matches_full_prefill(self):
        """Every query's visible set (window + causally complete compressed
        entries) is identical whether tokens arrive at once or one by one."""
        mod = self._module(seed=1)
        x = tp.randn(1, 13, 32)
        with tp.no_grad():
            full = mod(x)
            cache = SlidingWindowCache(mod.sliding_window)
            steps = [
                mod(x[:, t : t + 1], past_key_values=cache)
                for t in range(x.shape[1])
            ]
            stepwise = tp.cat(steps, dim=1)
        self.assertLess((full - stepwise).abs().max().item(), 1e-4)
        self.assertEqual(cache.get_seq_length(), 13)
        self.assertEqual(cache.get_window_length(), mod.sliding_window)

    def test_indexer_never_selects_incomplete_windows(self):
        mod = self._module(seed=2)
        x = tp.randn(1, 9, 32)
        position_ids = tp.arange(9).unsqueeze(0)
        with tp.no_grad():
            q_residual = mod.q_a_norm(mod.q_a_proj(x))
            index_state, topk = mod.indexer(x, q_residual, position_ids, mod.rope, 0, {})
            entry_end = index_state["entry_end"]
            bias = mod._block_bias(topk, entry_end, position_ids, entry_end.shape[0])
        # Queries earlier than an entry's last source token see it masked.
        for t in range(9):
            visible = (bias[0, 0, t] == 0).nonzero()
            for idx in visible:
                self.assertLessEqual(entry_end[idx].item(), t)

    def test_sinks_change_scores_without_breaking_scale(self):
        mod = self._module(seed=3)
        x = tp.randn(1, 5, 32)
        with tp.no_grad():
            base = mod(x)
            mod.sinks.data.add_(2.0)
            shifted = mod(x)
        self.assertGreater((base - shifted).abs().max().item(), 1e-6)


class TestModernBlocks(unittest.TestCase):
    def test_swiglu_shape_and_backward(self):
        tp.manual_seed(0)
        mlp = SwiGLUMLP(24)
        x = tp.randn(2, 5, 24)
        out = mlp(x)
        self.assertEqual(out.shape, tp.Size((2, 5, 24)))
        out.square().mean().backward()
        self.assertIsNotNone(mlp.gate_proj.weight.grad)

    def test_block_residual_flow(self):
        tp.manual_seed(1)
        mixer = nn.GroupedQueryAttention(24, num_heads=4, num_kv_heads=2)
        block = TransformerBlock(24, mixer)
        block.eval()
        x = tp.randn(2, 6, 24)
        self.assertEqual(block(x).shape, tp.Size((2, 6, 24)))

    def test_block_accepts_linear_mixer(self):
        tp.manual_seed(2)
        mixer = GatedDeltaNet(hidden_size=24, num_v_heads=4, num_k_heads=2, head_k_dim=12)
        block = TransformerBlock(24, mixer)
        block.eval()
        out = block(tp.randn(2, 6, 24))
        self.assertEqual(out.shape, tp.Size((2, 6, 24)))

    def test_cross_attention(self):
        tp.manual_seed(3)
        cross = CrossAttention(24, num_heads=4)
        x = tp.randn(2, 5, 24)
        memory = tp.randn(2, 9, 24)
        self.assertEqual(cross(x, memory).shape, tp.Size((2, 5, 24)))


class TestCausalEncoderDecoder(unittest.TestCase):
    def test_shape_and_backward(self):
        tp.manual_seed(0)
        ced = CausalEncoderDecoder(32, num_heads=4, num_kv_heads=2)
        x = tp.randn(2, 7, 32)
        out = ced(x)
        self.assertEqual(out.shape, tp.Size((2, 7, 32)))
        out.square().mean().backward()
        grads = [p.grad for p in ced.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_custom_linear_mixers(self):
        tp.manual_seed(1)

        def mixer_factory(hidden_size, layer_idx):
            return GatedDeltaNet(
                hidden_size=hidden_size, num_v_heads=4, num_k_heads=2,
                head_k_dim=16, layer_idx=layer_idx,
            )

        ced = CausalEncoderDecoder(32, num_heads=4, mixer_factory=mixer_factory)
        ced.eval()
        out = ced(tp.randn(1, 6, 32))
        self.assertEqual(out.shape, tp.Size((1, 6, 32)))


if __name__ == "__main__":
    unittest.main()
