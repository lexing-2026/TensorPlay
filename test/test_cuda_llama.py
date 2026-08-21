"""CUDA acceptance test: a complete (small) Llama-style causal LM forward.

The model is intentionally tiny so the test runs on the development GPU, but
the forward path contains the pieces that exercise the allocator in a real
decoder block: token embedding, RoPE, RMSNorm, causal attention, and gated
SwiGLU feed-forward layers.
"""

import gc
import math

import numpy as np
import pytest

import tensorplay as tp


pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


class TinyLlamaForCausalLM:
    """A dependency-free, inference-only Llama configuration for smoke tests."""

    def __init__(self, *, vocab_size=64, hidden_size=32, num_heads=4,
                 intermediate_size=64, num_layers=2, max_seq_len=8,
                 device="cuda"):
        assert hidden_size % num_heads == 0
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device

        def weight(*shape):
            # Small initialization keeps this untrained smoke model in the
            # finite range while still exercising every CUDA allocation path.
            return tp.randn(*shape, device=device) * 0.02

        self.embed_tokens = weight(vocab_size, hidden_size)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                "q_proj": weight(hidden_size, hidden_size),
                "k_proj": weight(hidden_size, hidden_size),
                "v_proj": weight(hidden_size, hidden_size),
                "o_proj": weight(hidden_size, hidden_size),
                "gate_proj": weight(intermediate_size, hidden_size),
                "up_proj": weight(intermediate_size, hidden_size),
                "down_proj": weight(hidden_size, intermediate_size),
                "input_norm": tp.ones(hidden_size, device=device),
                "post_attn_norm": tp.ones(hidden_size, device=device),
            })
        # Llama RMSNorm scale starts at one.
        self.final_norm = tp.ones(hidden_size, device=device)
        self.lm_head = weight(vocab_size, hidden_size)

        inv_freq = [
            1.0 / (10000.0 ** (2.0 * i / self.head_dim))
            for i in range(self.head_dim // 2)
        ]
        cos = [[math.cos(pos * freq) for freq in inv_freq]
               for pos in range(max_seq_len)]
        sin = [[math.sin(pos * freq) for freq in inv_freq]
               for pos in range(max_seq_len)]
        # CPU construction is deliberate: the model buffers are then moved by
        # the same pageable-to-device copy path users hit when loading a model.
        self.rope_cos = tp.tensor(cos, dtype=tp.float32).to(device)
        self.rope_sin = tp.tensor(sin, dtype=tp.float32).to(device)

    def _project(self, x, weight):
        # TensorPlay's CUDA matmul currently accepts 2-D operands. Flatten the
        # batch/sequence dimensions exactly as a production linear wrapper
        # would, then restore the decoder shape.
        batch, seq, _ = x.shape
        return tp.matmul(x.reshape([batch * seq, x.shape[-1]]), weight.t()).reshape(
            [batch, seq, weight.shape[0]]
        )

    @staticmethod
    def _rms_norm(x, weight):
        variance = (x * x).mean(dim=[-1], keepdim=True)
        return x * tp.rsqrt(variance + 1e-5) * weight

    def _apply_rope(self, x, seq_len):
        cos = self.rope_cos[(slice(0, seq_len),)].unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[(slice(0, seq_len),)].unsqueeze(0).unsqueeze(0)
        even = x[(slice(None), slice(None), slice(None), slice(None, None, 2))]
        odd = x[(slice(None), slice(None), slice(None), slice(1, None, 2))]
        rotated = tp.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.reshape(x.shape)

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("sequence is longer than the configured RoPE table")

        x = tp.embedding(self.embed_tokens, input_ids)
        for layer in self.layers:
            residual = x
            xn = self._rms_norm(x, layer["input_norm"])
            q = self._project(xn, layer["q_proj"]).reshape(
                [batch, seq_len, self.num_heads, self.head_dim]
            ).permute([0, 2, 1, 3])
            k = self._project(xn, layer["k_proj"]).reshape(
                [batch, seq_len, self.num_heads, self.head_dim]
            ).permute([0, 2, 1, 3])
            v = self._project(xn, layer["v_proj"]).reshape(
                [batch, seq_len, self.num_heads, self.head_dim]
            ).permute([0, 2, 1, 3])
            q = self._apply_rope(q, seq_len)
            k = self._apply_rope(k, seq_len)
            attention = tp.scaled_dot_product_attention(
                q, k, v, is_causal=True, impl=1
            )
            attention = attention.permute([0, 2, 1, 3]).reshape(
                [batch, seq_len, self.hidden_size]
            )
            x = residual + self._project(attention, layer["o_proj"])

            residual = x
            xn = self._rms_norm(x, layer["post_attn_norm"])
            gate = self._project(xn, layer["gate_proj"])
            up = self._project(xn, layer["up_proj"])
            x = residual + self._project(tp.silu(gate) * up, layer["down_proj"])

        x = self._rms_norm(x, self.final_norm)
        return self._project(x, self.lm_head)


def test_tiny_llama_cuda_forward_and_pool_accounting():
    tp.cuda.synchronize()
    tp.cuda.empty_cache()
    gc.collect()
    allocated_before = tp.cuda.memory_allocated()

    model = TinyLlamaForCausalLM()
    input_ids = tp.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=tp.int64, device="cuda"
    )
    logits = model(input_ids)
    tp.cuda.synchronize()

    assert logits.device == tp.device("cuda")
    assert logits.shape == [1, 8, model.vocab_size]
    values = logits.cpu().numpy()
    assert np.isfinite(values).all()

    next_token = logits[:, -1, :].argmax(dim=-1)
    assert next_token.shape == [1]
    assert next_token.device == tp.device("cuda")
    assert 0 <= next_token.item() < model.vocab_size

    allocated = tp.cuda.memory_allocated()
    reserved = tp.cuda.memory_reserved()
    assert allocated >= allocated_before
    assert reserved >= allocated

    del next_token, logits, input_ids, model
    gc.collect()
    tp.cuda.synchronize()
    assert tp.cuda.memory_allocated() <= allocated_before
    tp.cuda.empty_cache()
    # Other CUDA tests in the same pytest process may own live allocations;
    # empty_cache() must preserve those while returning this model's free
    # blocks to the pool.  The invariant we can assert locally is that the
    # allocator remains internally consistent after the model is dropped.
    assert tp.cuda.memory_reserved() >= tp.cuda.memory_allocated()


def test_tiny_llama_cuda_backward():
    """The decoder path must carry gradients through embedding, RoPE and SDPA."""
    tp.cuda.synchronize()
    tp.cuda.empty_cache()
    model = TinyLlamaForCausalLM(
        vocab_size=32,
        hidden_size=16,
        num_heads=4,
        intermediate_size=32,
        num_layers=1,
        max_seq_len=4,
    )
    parameters = [model.embed_tokens, model.lm_head, model.final_norm]
    for layer in model.layers:
        parameters.extend(layer.values())
    # The smoke model initializes tensors without grad; promote them to leaves
    # here, just as a loader would do for trainable parameters.
    for parameter in parameters:
        parameter.requires_grad_(True)

    input_ids = tp.tensor([[1, 2, 3, 4]], dtype=tp.int64, device="cuda")
    logits = model(input_ids)
    (logits * logits).mean().backward()
    tp.cuda.synchronize()

    assert all(parameter.grad is not None for parameter in parameters)
    assert all(np.isfinite(parameter.grad.cpu().numpy()).all()
               for parameter in parameters)

    del logits, input_ids, model, parameters
    gc.collect()
    tp.cuda.synchronize()
    tp.cuda.empty_cache()
