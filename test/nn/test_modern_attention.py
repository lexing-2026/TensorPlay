"""Tests for grouped-query attention and multi-head latent attention."""

import unittest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay.nn.cache import DynamicCache, QuantizedCache
from tensorplay.nn.modules.causal_attention import GroupedQueryAttention
from tensorplay.nn.modules.latent_attention import MultiheadLatentAttention
from tensorplay.nn.modules.rotary import apply_rotary_emb


def _causal_mask(seq_len, dtype):
    # Zero on and below the diagonal, -inf strictly above.
    return tp.triu(tp.full((seq_len, seq_len), float("-inf"), dtype=dtype), diagonal=1)


class TestGroupedQueryAttention(unittest.TestCase):
    def test_output_shape(self):
        tp.manual_seed(0)
        mod = GroupedQueryAttention(32, num_heads=8, num_kv_heads=2)
        x = tp.randn(2, 5, 32)
        self.assertEqual(mod(x).shape, tp.Size((2, 5, 32)))

    def test_invalid_head_config(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(30, num_heads=8)  # embed_dim not divisible
        with self.assertRaises(ValueError):
            GroupedQueryAttention(32, num_heads=8, num_kv_heads=3)  # non-dividing groups

    def test_matches_manual_attention(self):
        """Module output equals a hand-rolled grouped softmax attention."""
        tp.manual_seed(1)
        mod = GroupedQueryAttention(
            24, num_heads=6, num_kv_heads=2, qk_norm=True, rms_norm_eps=1e-5, bias=True
        )
        mod.eval()
        x = tp.randn(2, 7, 24)

        bsz, seq_len, _ = x.shape
        q = mod.q_proj(x).view(bsz, seq_len, mod.num_heads, mod.head_dim).transpose(1, 2)
        k = mod.k_proj(x).view(bsz, seq_len, mod.num_kv_heads, mod.head_dim).transpose(1, 2)
        v = mod.v_proj(x).view(bsz, seq_len, mod.num_kv_heads, mod.head_dim).transpose(1, 2)
        q = mod.q_norm(q)
        k = mod.k_norm(k)
        position_ids = tp.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
        cos, sin = mod.rope(x, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin, mod.rope.interleaved)
        reps = mod.num_heads // mod.num_kv_heads
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        scores = tp.matmul(q, k.transpose(-1, -2)) * mod.scaling + _causal_mask(seq_len, x.dtype)
        attn = F.softmax(scores, dim=-1)
        out = tp.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        expected = mod.o_proj(out)

        actual = mod(x)
        self.assertTrue(tp.allclose(actual, expected, atol=1e-5))

    def test_mha_degenerates_to_full_heads(self):
        """num_kv_heads == num_heads must match a plain per-head reference."""
        tp.manual_seed(2)
        mod = GroupedQueryAttention(16, num_heads=4)
        mod.eval()
        x = tp.randn(3, 4, 16)
        bsz, seq_len, _ = x.shape
        q = mod.q_proj(x).view(bsz, seq_len, 4, mod.head_dim).transpose(1, 2)
        k = mod.k_proj(x).view(bsz, seq_len, 4, mod.head_dim).transpose(1, 2)
        v = mod.v_proj(x).view(bsz, seq_len, 4, mod.head_dim).transpose(1, 2)
        cos, sin = mod.rope(x, tp.arange(seq_len).unsqueeze(0).expand(bsz, seq_len))
        q, k = apply_rotary_emb(q, k, cos, sin)
        scores = tp.matmul(q, k.transpose(-1, -2)) * mod.scaling + _causal_mask(seq_len, x.dtype)
        attn = F.softmax(scores, dim=-1)
        expected = mod.o_proj(tp.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1))
        self.assertTrue(tp.allclose(mod(x), expected, atol=1e-5))

    def test_cache_prefill_decode(self):
        """Cached decode at position T-1 reproduces the full recompute row."""
        tp.manual_seed(3)
        mod = GroupedQueryAttention(24, num_heads=4, num_kv_heads=1, qk_norm=True)
        mod.eval()
        x = tp.randn(2, 6, 24)
        full = mod(x)

        cache = DynamicCache()
        prefix = mod(x[:, :-1, :], past_key_values=cache)
        step = mod(x[:, -1:, :], past_key_values=cache)

        self.assertEqual(cache.get_seq_length(), 6)
        self.assertTrue(tp.allclose(prefix, full[:, :-1, :], atol=1e-5))
        self.assertTrue(tp.allclose(step.squeeze(1), full[:, -1, :], atol=1e-5))

    def test_cache_holds_kv_heads_only(self):
        tp.manual_seed(4)
        mod = GroupedQueryAttention(16, num_heads=4, num_kv_heads=2, head_dim=8)
        cache = DynamicCache()
        mod(tp.randn(2, 3, 16), past_key_values=cache)
        self.assertEqual(cache.key_cache[0].shape, tp.Size((2, 2, 3, 8)))
        self.assertEqual(cache.value_cache[0].shape, tp.Size((2, 2, 3, 8)))


class TestMultiheadLatentAttention(unittest.TestCase):
    def _make(self, **kwargs):
        tp.manual_seed(5)
        defaults = dict(
            embed_dim=32,
            num_heads=4,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=12,
            q_lora_rank=24,
        )
        defaults.update(kwargs)
        return MultiheadLatentAttention(**defaults)

    def _manual(self, mod, x):
        bsz, seq_len, _ = x.shape
        if mod.q_lora_rank is None:
            q = mod.q_proj(x)
        else:
            q = mod.q_b_proj(mod.q_a_layernorm(mod.q_a_proj(x)))
        q = q.view(bsz, seq_len, mod.num_heads, mod.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = tp.split(q, [mod.qk_nope_head_dim, mod.qk_rope_head_dim], dim=-1)

        compressed = mod.kv_a_proj_with_mqa(x)
        latent, k_rot = tp.split(compressed, [mod.kv_lora_rank, mod.qk_rope_head_dim], dim=-1)
        latent = mod.kv_a_layernorm(latent).view(bsz, 1, seq_len, mod.kv_lora_rank)
        k_rot = k_rot.view(bsz, 1, seq_len, mod.qk_rope_head_dim)

        cos, sin = mod.rope(x, tp.arange(seq_len).unsqueeze(0).expand(bsz, seq_len))
        q_rot, k_rot = apply_rotary_emb(q_rot, k_rot, cos, sin, mod.rope.interleaved)
        q = tp.cat([q_pass, q_rot], dim=-1)

        kv = mod.kv_b_proj(latent).view(
            bsz, seq_len, mod.num_heads, mod.qk_nope_head_dim + mod.v_head_dim
        ).transpose(1, 2)
        k_nope, v = tp.split(kv, [mod.qk_nope_head_dim, mod.v_head_dim], dim=-1)
        k = tp.cat([k_nope, k_rot.expand(-1, mod.num_heads, -1, -1)], dim=-1)

        scores = tp.matmul(q, k.transpose(-1, -2)) * mod.scaling + _causal_mask(seq_len, x.dtype)
        attn = F.softmax(scores, dim=-1)
        out = tp.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return mod.o_proj(out)

    def test_output_shape(self):
        mod = self._make()
        mod.eval()
        self.assertEqual(mod(tp.randn(2, 6, 32)).shape, tp.Size((2, 6, 32)))

    def test_matches_manual_attention(self):
        mod = self._make()
        mod.eval()
        x = tp.randn(2, 5, 32)
        self.assertTrue(tp.allclose(mod(x), self._manual(mod, x), atol=1e-5))

    def test_direct_query_projection(self):
        """q_lora_rank=None skips the query bottleneck."""
        mod = self._make(q_lora_rank=None)
        mod.eval()
        x = tp.randn(2, 4, 32)
        expected = self._manual(mod, x)
        self.assertTrue(tp.allclose(mod(x), expected, atol=1e-5))

    def test_cache_stores_compressed_latents(self):
        mod = self._make()
        cache = DynamicCache()
        mod(tp.randn(2, 6, 32), past_key_values=cache)
        # Per position the cache holds kv_lora_rank + qk_rope_head_dim numbers,
        # not the per-head expanded keys and values.
        self.assertEqual(cache.key_cache[0].shape, tp.Size((2, 1, 6, 16)))
        self.assertEqual(cache.value_cache[0].shape, tp.Size((2, 1, 6, 8)))

    def test_cache_prefill_decode(self):
        mod = self._make()
        mod.eval()
        x = tp.randn(2, 7, 32)
        full = mod(x)

        cache = DynamicCache()
        prefix = mod(x[:, :-1, :], past_key_values=cache)
        step = mod(x[:, -1:, :], past_key_values=cache)

        self.assertTrue(tp.allclose(prefix, full[:, :-1, :], atol=1e-5))
        self.assertTrue(tp.allclose(step.squeeze(1), full[:, -1, :], atol=1e-5))

    def test_quantized_cache_roundtrip(self):
        """The latent cache pair survives int8 quantization within tolerance."""
        mod = self._make()
        mod.eval()
        x = tp.randn(2, 6, 32)
        full = mod(x)

        cache = QuantizedCache()
        mod(x[:, :-1, :], past_key_values=cache)
        step = mod(x[:, -1:, :], past_key_values=cache)
        scale = full.abs().max().item()
        self.assertLess((step.squeeze(1) - full[:, -1, :]).abs().max().item(), 0.05 * scale)

    def test_odd_rope_dim_rejected(self):
        with self.assertRaises(ValueError):
            self._make(qk_rope_head_dim=7)


if __name__ == "__main__":
    unittest.main()
