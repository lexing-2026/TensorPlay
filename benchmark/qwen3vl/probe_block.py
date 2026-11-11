"""Attribute one block's cost to its parts by stubbing them out.

A block measured on the accelerator costs more than the products it issues.
Rather than guess which part is responsible, each part of a single block is
replaced in turn by a shape-correct stand-in that does no work, and the block
is timed again.  What the block loses is what that part cost.  The stand-ins
keep the shapes unchanged so the arithmetic around them runs as it would.

The step is timed the way the harness runs it: a preallocated cache of a fixed
extent whose write cursor is rewound before every call, and the language
tower's normalizations written out.

    python probe_block.py --model <dir> [--static-cache 64] [--repeat 50]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402
from tensorplay import nn  # noqa: E402
from tensorplay.nn.modules.rotary import apply_rotary_emb  # noqa: E402

from qwen3vl.model import (  # noqa: E402
    Qwen3VLDecoderLayer,
    Qwen3VLTextConfig,
    WeightOnlyInt4Linear,
    written_normalization,
)

HIDDEN = 4096


def bench(label, fn, repeat, reset=None):
    """Time ``fn``, optionally rewinding the cache cursor first."""
    def call():
        if reset is not None:
            reset()
        fn()

    for _ in range(3):
        call()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        call()
    tp.mlu.synchronize()
    print(f"  {label:40} {(time.time() - start) / repeat * 1e3:9.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--static-cache", type=int, default=64)
    ap.add_argument("--written-norm", action="store_true")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    from tensorplay.nn.cache import StaticCache

    config = Qwen3VLTextConfig.from_dir(args.model)
    layer = Qwen3VLDecoderLayer(config, 0, weight_only=True, device="mlu",
                                dtype=tp.float16)
    for module in layer.modules():
        if isinstance(module, WeightOnlyInt4Linear):
            module._dense_cache = tp.randn(
                module.out_features, module.in_features, device="mlu",
                dtype=tp.float16,
            )
    if args.written_norm:
        for module in layer.modules():
            if isinstance(module, nn.RMSNorm):
                module.forward = (
                    lambda x, _module=module: written_normalization(_module, x)
                )
    tp.mlu.synchronize()

    x = tp.randn(1, 1, HIDDEN, device="mlu", dtype=tp.float16)
    # The step under test sits at position 20 of a 21-deep cache.
    positions = tp.arange(20, 21, device="mlu").unsqueeze(0)
    depth = 20

    cache = StaticCache(args.static_cache)
    with tp.no_grad():
        for step in range(depth):
            layer(
                tp.randn(1, 1, HIDDEN, device="mlu", dtype=tp.float16),
                position_ids=tp.arange(step, step + 1, device="mlu").unsqueeze(0),
                past_key_values=cache,
            )
    tp.mlu.synchronize()
    start_cursor = list(cache._cursor)

    def rewind():
        cache._cursor = list(start_cursor)

    def step():
        return layer(x, position_ids=positions, past_key_values=cache)

    def restore(module):
        module.__dict__.pop("forward", None)

    def zero_attention(hidden_states, position_ids=None, past_key_values=None):
        return tp.zeros(hidden_states.shape, dtype=hidden_states.dtype,
                        device=hidden_states.device)

    def zero_mlp(hidden_states):
        return tp.zeros(hidden_states.shape, dtype=hidden_states.dtype,
                        device=hidden_states.device)

    def identity_norm(hidden_states):
        return hidden_states

    one = tp.ones((1, 1, layer.self_attn.mixer.head_dim // 2), dtype=tp.float16,
                  device="mlu")

    def flat_rope(hidden, position_ids=None, max_position=None):
        return one, one

    print("block, whole")
    bench("whole block", step, args.repeat, reset=rewind)

    print("\nblock with one part stubbed")
    layer.self_attn.forward = zero_attention
    bench("no attention", step, args.repeat, reset=rewind)
    restore(layer.self_attn)

    layer.mlp.forward = zero_mlp
    bench("no feed-forward", step, args.repeat, reset=rewind)
    restore(layer.mlp)

    layer.input_layernorm.forward = identity_norm
    layer.post_attention_layernorm.forward = identity_norm
    bench("no block normalization", step, args.repeat, reset=rewind)
    restore(layer.input_layernorm)
    restore(layer.post_attention_layernorm)

    layer.self_attn.mixer.rope.forward = flat_rope
    bench("no rotation", step, args.repeat, reset=rewind)
    restore(layer.self_attn.mixer.rope)

    q_norm_body = layer.self_attn.mixer.q_norm.forward
    layer.self_attn.mixer.q_norm.forward = identity_norm
    bench("no query normalization", step, args.repeat, reset=rewind)
    layer.self_attn.mixer.q_norm.forward = q_norm_body

    mixer = layer.self_attn.mixer
    q_proj_body = mixer.q_proj

    class Stub(tp.nn.Module):
        def forward(self, inputs):
            return tp.zeros(1, 1, mixer.num_heads * mixer.head_dim,
                            dtype=inputs.dtype, device=inputs.device)

    mixer.q_proj = Stub()
    bench("no query projection", step, args.repeat, reset=rewind)
    mixer.q_proj = q_proj_body

    def reduced_precision_norm(module, value):
        inv = ((value * value).mean(-1, keepdim=True) + module.eps).rsqrt()
        return value * inv * module.weight

    print("\none block's parts, timed where the block calls them")
    bench("block normalization, single-precision statistic",
          lambda: layer.post_attention_layernorm(layer.input_layernorm(x)), args.repeat)
    layer.input_layernorm.forward = (lambda value, _module=layer.input_layernorm:
                                     reduced_precision_norm(_module, value))
    layer.post_attention_layernorm.forward = (
        lambda value, _module=layer.post_attention_layernorm:
        reduced_precision_norm(_module, value))
    bench("block normalization, reduced statistic",
          lambda: layer.post_attention_layernorm(layer.input_layernorm(x)), args.repeat)
    q_heads = tp.randn(1, 32, 1, 128, device="mlu", dtype=tp.float16)
    k_heads = tp.randn(1, 8, 1, 128, device="mlu", dtype=tp.float16)
    half = tp.randn(1, 1, 64, device="mlu", dtype=tp.float16)
    bench("rotation for one position", lambda: mixer.rope(x, positions), args.repeat)
    bench("block normalizations",
          lambda: layer.post_attention_layernorm(layer.input_layernorm(x)),
          args.repeat)
    bench("query head normalization", lambda: mixer.q_norm(q_heads), args.repeat)
    bench("key head normalization", lambda: mixer.k_norm(k_heads), args.repeat)
    bench("rotation applied to q and k",
          lambda: apply_rotary_emb(q_heads, k_heads, half, half, False), args.repeat)
    bench("feed-forward", lambda: layer.mlp(x), args.repeat)
    wide = tp.randn(1, 1, config.intermediate_size, device="mlu", dtype=tp.float16)
    bench("seven products", lambda: (
        layer.mlp.gate_proj(x),
        layer.mlp.up_proj(x),
        layer.mlp.down_proj(wide),
        mixer.o_proj(x),
        mixer.v_proj(x),
        mixer.k_proj(x),
        mixer.q_proj(x),
    ), args.repeat)


if __name__ == "__main__":
    main()
