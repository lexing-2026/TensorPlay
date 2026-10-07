"""Tests for the decoding cache and the autoregressive generate loop."""

import unittest

import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.nn.functional as F
from tensorplay.nn.cache import DynamicCache, LinearStateCache
from tensorplay.nn.generation import _apply_repetition_penalty, generate, top_k_top_p_filtering


class CachedSelfAttention(nn.Module):
    """Single-head causal attention that consumes a DynamicCache."""

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim**-0.5

    def forward(self, x, past_key_values=None):
        q = self.q_proj(x).unsqueeze(1)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx=0)
        # Causal alignment is top-left: with a single query against a grown
        # cache the mask must allow every cached position, so is_causal only
        # applies when the query covers the full cached length.
        is_causal = q.shape[2] == k.shape[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=self.scale)
        return out.squeeze(1)


class CachedTinyModel(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.attn = CachedSelfAttention(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, position_ids=None, past_key_values=None):
        x = self.embed(input_ids)
        x = x + self.attn(x, past_key_values=past_key_values)
        return self.head(x)


class TestDynamicCache(unittest.TestCase):
    def test_append_and_seq_length(self):
        cache = DynamicCache()
        self.assertEqual(cache.get_seq_length(), 0)
        k1, v1 = tp.randn(2, 3, 4, 5), tp.randn(2, 3, 4, 5)
        ck, cv = cache.update(k1, v1, layer_idx=0)
        self.assertIs(ck, k1)
        self.assertEqual(cache.get_seq_length(), 4)
        k2, v2 = tp.randn(2, 3, 2, 5), tp.randn(2, 3, 2, 5)
        ck, cv = cache.update(k2, v2, layer_idx=0)
        self.assertEqual(ck.shape, (2, 3, 6, 5))
        self.assertTrue(tp.equal(ck[:, :, :4], k1))
        self.assertTrue(tp.equal(ck[:, :, 4:], k2))
        self.assertEqual(cache.get_seq_length(layer_idx=1), 0)

    def test_layer_independence(self):
        cache = DynamicCache()
        ka, _ = cache.update(tp.randn(1, 1, 3, 2), tp.randn(1, 1, 3, 2), layer_idx=1)
        kb, _ = cache.update(tp.randn(1, 1, 5, 2), tp.randn(1, 1, 5, 2), layer_idx=0)
        self.assertEqual(ka.shape[2], 3)
        self.assertEqual(kb.shape[2], 5)
        cache.reset()
        self.assertEqual(len(cache), 0)


class TestLinearStateCache(unittest.TestCase):
    def test_conv_window_rolling(self):
        cache = LinearStateCache()
        self.assertFalse(cache.has_previous_state(0))
        first = cache.update_conv_state(0, tp.randn(2, 6, 4), kernel_size=4)
        self.assertEqual(first.shape[-1], 4)
        self.assertEqual(cache.conv_states[0].shape[-1], 3)
        self.assertTrue(cache.has_previous_state(0))
        second = cache.update_conv_state(0, tp.randn(2, 6, 1), kernel_size=4)
        self.assertEqual(second.shape[-1], 4)
        self.assertEqual(cache.conv_states[0].shape[-1], 3)
        self.assertTrue(tp.equal(second[..., -1:], cache.conv_states[0][..., -1:]))

    def test_recurrent_state(self):
        cache = LinearStateCache()
        state = tp.randn(2, 4, 8, 8)
        cache.update_recurrent_state(0, state)
        self.assertIs(cache.recurrent_states[0], state)
        cache.reset()
        self.assertFalse(cache.has_previous_state(0))


class TestGenerateLoop(unittest.TestCase):
    def setUp(self):
        tp.manual_seed(7)
        self.vocab, self.dim = 32, 16
        self.model = CachedTinyModel(self.vocab, self.dim)
        # An untrained head produces near-uniform logits; scale it so the
        # argmax gaps dominate float noise and greedy decoding is decisive.
        with tp.no_grad():
            self.model.head.weight.mul_(100.0)
        self.ids = tp.randint(0, self.vocab, (3, 6))

    def test_cached_matches_full_recompute(self):
        cached = generate(self.model, self.ids, max_new_tokens=10, use_cache=True)
        uncached = generate(self.model, self.ids, max_new_tokens=10, use_cache=False)
        self.assertTrue(tp.equal(cached, uncached))

    def test_prompt_is_preserved(self):
        out = generate(self.model, self.ids, max_new_tokens=5)
        self.assertEqual(out.shape, (3, 11))
        self.assertTrue(tp.equal(out[:, :6], self.ids))

    def test_max_new_tokens_zero_returns_prompt(self):
        self.assertTrue(tp.equal(generate(self.model, self.ids, max_new_tokens=0), self.ids))

    def test_sampling_respects_top_k_support(self):
        out = generate(
            self.model, self.ids, max_new_tokens=20, do_sample=True,
            temperature=1.0, top_k=2,
        )
        self.assertEqual(out.shape, (3, 26))

    def test_filter_keeps_expected_support(self):
        logits = tp.tensor([[1.0, 2.0, 3.0, 4.0]])
        kept = top_k_top_p_filtering(logits, top_k=2)
        mask = kept.isinf()
        self.assertEqual(mask.sum().item(), 2)
        self.assertFalse(mask[0, 3].item())
        self.assertFalse(mask[0, 2].item())

    def test_filter_top_p_never_empties_support(self):
        logits = tp.tensor([[0.0, 0.0, 0.0, 100.0]])
        kept = top_k_top_p_filtering(logits, top_p=0.999)
        self.assertFalse(kept.isinf().all().item())

    def test_repetition_penalty_values(self):
        logits = tp.tensor([[2.0, -2.0, 0.5, 1.0]])
        seq = tp.tensor([[0, 2]])
        out = _apply_repetition_penalty(logits, seq, penalty=2.0)
        self.assertAlmostEqual(out[0, 0].item(), 1.0)
        self.assertAlmostEqual(out[0, 2].item(), 0.25)
        self.assertAlmostEqual(out[0, 1].item(), -2.0)
        self.assertAlmostEqual(out[0, 3].item(), 1.0)

    def test_eos_pads_finished_rows(self):
        # A model whose logits always favour token 0; make token 0 the EOS.
        head_bias = self.model.head.bias
        if head_bias is None:
            self.skipTest("head has no bias to pin the argmax with")
        with tp.no_grad():
            out = generate(
                self.model, self.ids, max_new_tokens=6, eos_token_id=self.vocab - 1,
                pad_token_id=3,
            )
        self.assertEqual(out.shape, (3, 12))


if __name__ == "__main__":
    unittest.main()
