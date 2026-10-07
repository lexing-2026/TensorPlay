"""Tests for gated delta rule mixers (head-scalar and per-channel decay)."""

import unittest

import tensorplay as tp
from tensorplay.nn.cache import LinearStateCache
from tensorplay.nn.modules.gated_delta import (
    GatedDeltaNet,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from tensorplay.nn.modules.kimi_delta import (
    KimiDeltaAttention,
    chunk_kimi_delta_rule,
    recurrent_kimi_delta_rule,
)
from tensorplay.nn.modules.gated_delta import l2norm


def _manual_delta_rule(q, k, v, g, beta, l2norm_qk=False):
    """Plain per-token loop over the delta rule; q/k/v [B,T,H,D].

    ``g`` is either [B,T,H] (per head) or [B,T,H,dk] (per channel); the loop
    broadcasts it against the state automatically.
    """
    B, T, H, dk = k.shape
    dv = v.shape[-1]
    q = q.to(tp.float32)
    k = k.to(tp.float32)
    v = v.to(tp.float32)
    if l2norm_qk:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)
    q = q * dk**-0.5
    state = tp.zeros(B, H, dk, dv, dtype=tp.float32)
    outs = []
    for t in range(T):
        exp_g = g[:, t].exp()
        # [B,H] per-head decay needs two trailing axes, [B,H,dk] per-channel
        # decay only one, to broadcast against the [B,H,dk,dv] state.
        exp_g = exp_g[..., None, None] if exp_g.dim() == 2 else exp_g[..., None]
        state = state * exp_g
        kv_mem = (state * k[:, t].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, t].to(tp.float32) - kv_mem) * beta[:, t].unsqueeze(-1)
        state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q[:, t].unsqueeze(-1)).sum(dim=-2).unsqueeze(1))
    return tp.cat(outs, dim=1), state


class TestGatedDeltaRule(unittest.TestCase):
    def _inputs(self, seed=0, B=2, T=37, H=3, DK=12, DV=8):
        tp.manual_seed(seed)
        q = tp.randn(B, T, H, DK)
        k = tp.randn(B, T, H, DK)
        v = tp.randn(B, T, H, DV)
        g = -tp.rand(B, T, H).abs() * 0.3 - 0.01
        beta = tp.rand(B, T, H)
        return q, k, v, g, beta

    def test_chunk_matches_recurrent(self):
        q, k, v, g, beta = self._inputs()
        for chunk_size in (64, 16, 8):  # single chunk, padded, multi-chunk
            out_c, state_c = chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=chunk_size,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            out_r, state_r = recurrent_gated_delta_rule(
                q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            self.assertLess((out_c - out_r).abs().max().item(), 1e-5)
            self.assertLess((state_c - state_r).abs().max().item(), 1e-5)

    def test_matches_manual_loop(self):
        q, k, v, g, beta = self._inputs(seed=1, T=7)
        out_c, state_c = chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=4, output_final_state=True,
        )
        out_m, state_m = _manual_delta_rule(q, k, v, g, beta)
        self.assertLess((out_c - out_m).abs().max().item(), 1e-4)
        self.assertLess((state_c - state_m).abs().max().item(), 1e-4)

    def test_recurrent_matches_manual_with_l2norm(self):
        q, k, v, g, beta = self._inputs(seed=2, T=5)
        out_r, _ = recurrent_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        out_m, _ = _manual_delta_rule(q, k, v, g, beta, l2norm_qk=True)
        self.assertLess((out_r - out_m).abs().max().item(), 1e-4)

    def test_initial_state_continues_the_sequence(self):
        """Seeding the state with the first chunk's result reproduces a single
        pass over the concatenated sequence."""
        q, k, v, g, beta = self._inputs(seed=3, T=12)
        half = 6
        _, state = chunk_gated_delta_rule(
            q[:, :half], k[:, :half], v[:, :half], g[:, :half], beta[:, :half],
            chunk_size=4, output_final_state=True,
        )
        tail_out, tail_state = chunk_gated_delta_rule(
            q[:, half:], k[:, half:], v[:, half:], g[:, half:], beta[:, half:],
            chunk_size=4, initial_state=state, output_final_state=True,
        )
        full_out, full_state = chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=4, output_final_state=True,
        )
        self.assertLess((tail_out - full_out[:, half:]).abs().max().item(), 1e-4)
        self.assertLess((tail_state - full_state).abs().max().item(), 1e-4)


class TestKimiDeltaRule(unittest.TestCase):
    def _inputs(self, seed=0, B=2, T=37, H=3, DK=12, DV=6):
        tp.manual_seed(seed)
        q = tp.randn(B, T, H, DK)
        k = tp.randn(B, T, H, DK)
        v = tp.randn(B, T, H, DV)
        g = -tp.rand(B, T, H, DK).abs() * 0.3 - 0.01  # per channel
        beta = tp.rand(B, T, H)
        return q, k, v, g, beta

    def test_chunk_matches_recurrent(self):
        q, k, v, g, beta = self._inputs()
        for chunk_size in (64, 16, 8):
            out_c, state_c = chunk_kimi_delta_rule(
                q, k, v, g, beta, chunk_size=chunk_size,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            out_r, state_r = recurrent_kimi_delta_rule(
                q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            self.assertLess((out_c - out_r).abs().max().item(), 1e-5)
            self.assertLess((state_c - state_r).abs().max().item(), 1e-5)

    def test_matches_manual_loop(self):
        q, k, v, g, beta = self._inputs(seed=1, T=7)
        out_c, state_c = chunk_kimi_delta_rule(
            q, k, v, g, beta, chunk_size=4, output_final_state=True,
        )
        out_m, state_m = _manual_delta_rule(q, k, v, g, beta)
        self.assertLess((out_c - out_m).abs().max().item(), 1e-4)
        self.assertLess((state_c - state_m).abs().max().item(), 1e-4)


class TestGatedDeltaNet(unittest.TestCase):
    def _module(self, seed=0, **kwargs):
        tp.manual_seed(seed)
        defaults = dict(hidden_size=32, num_v_heads=4, num_k_heads=2, head_k_dim=16, head_v_dim=8)
        defaults.update(kwargs)
        mod = GatedDeltaNet(**defaults)
        mod.eval()
        return mod

    def test_output_shape_and_head_replication(self):
        mod = self._module()
        self.assertEqual(mod(tp.randn(2, 10, 32)).shape, tp.Size((2, 10, 32)))

    def test_invalid_head_config(self):
        with self.assertRaises(ValueError):
            GatedDeltaNet(hidden_size=32, num_v_heads=4, num_k_heads=3, head_k_dim=8)

    def test_cache_prefill_decode(self):
        mod = self._module(seed=1)
        x = tp.randn(2, 13, 32)
        with tp.no_grad():
            full = mod(x)
            cache = LinearStateCache()
            prefix = mod(x[:, :-1, :], past_key_values=cache)
            step = mod(x[:, -1:, :], past_key_values=cache)
        self.assertLess((prefix - full[:, :-1]).abs().max().item(), 1e-5)
        self.assertLess((step.squeeze(1) - full[:, -1]).abs().max().item(), 1e-5)
        # Recurrent matrix and the conv window are both stored.
        self.assertEqual(cache.recurrent_states[0].shape, tp.Size((2, 4, 16, 8)))
        self.assertEqual(cache.conv_states[0].shape[-1], mod.conv_kernel_size - 1)

    def test_state_stored_in_float32(self):
        mod = self._module(seed=2)
        cache = LinearStateCache()
        with tp.no_grad():
            mod(tp.randn(1, 5, 32), past_key_values=cache)
        self.assertEqual(cache.recurrent_states[0].dtype, tp.float32)


class TestKimiDeltaAttention(unittest.TestCase):
    def _module(self, seed=0, **kwargs):
        tp.manual_seed(seed)
        defaults = dict(hidden_size=36, num_heads=3, head_dim=12)
        defaults.update(kwargs)
        mod = KimiDeltaAttention(**defaults)
        mod.eval()
        return mod

    def test_output_shape(self):
        mod = self._module()
        self.assertEqual(mod(tp.randn(2, 10, 36)).shape, tp.Size((2, 10, 36)))

    def test_cache_prefill_decode(self):
        mod = self._module(seed=1)
        x = tp.randn(2, 13, 36)
        with tp.no_grad():
            full = mod(x)
            cache = LinearStateCache()
            prefix = mod(x[:, :-1, :], past_key_values=cache)
            step = mod(x[:, -1:, :], past_key_values=cache)
        self.assertLess((prefix - full[:, :-1]).abs().max().item(), 1e-5)
        self.assertLess((step.squeeze(1) - full[:, -1]).abs().max().item(), 1e-5)
        self.assertEqual(cache.recurrent_states[0].shape, tp.Size((2, 3, 12, 12)))

    def test_state_stored_in_float32(self):
        mod = self._module(seed=2)
        cache = LinearStateCache()
        with tp.no_grad():
            mod(tp.randn(1, 5, 36), past_key_values=cache)
        self.assertEqual(cache.recurrent_states[0].dtype, tp.float32)


if __name__ == "__main__":
    unittest.main()
