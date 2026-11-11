"""Time the parts of one decoder block on the accelerator.

A decode step is a stack of blocks, so a block that costs far more than the
sum of its operators says one of those operators is taking a slow path.  Each
part is timed in isolation at the shapes a decode step uses.

    python probe_layer.py --model <dir>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402
from tensorplay import nn  # noqa: E402

import qwen3vl.model as M  # noqa: E402
from qwen3vl.model import (  # noqa: E402
    Qwen3VLDecoderLayer,
    Qwen3VLTextConfig,
    WeightOnlyInt4Linear,
)


def bench(label, fn, repeat=30):
    for _ in range(3):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:40} {(time.time() - start) / repeat * 1e3:9.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    config = Qwen3VLTextConfig.from_dir(args.model)
    hidden = config.hidden_size
    layer = Qwen3VLDecoderLayer(config, 0, weight_only=True, device="mlu", dtype=tp.float16)
    for module in layer.modules():
        if isinstance(module, WeightOnlyInt4Linear):
            module._dense_cache = tp.randn(
                module.out_features, module.in_features, device="mlu", dtype=tp.float16
            )
    tp.mlu.synchronize()

    x = tp.randn(1, 1, hidden, device="mlu", dtype=tp.float16)
    positions = tp.arange(20, 21, device="mlu").unsqueeze(0)
    mixer = layer.self_attn.mixer

    from tensorplay.nn.cache import DynamicCache

    def prefilled():
        cache = DynamicCache()
        with tp.no_grad():
            mixer(
                tp.randn(1, 20, hidden, device="mlu", dtype=tp.float16),
                position_ids=tp.arange(0, 20, device="mlu").unsqueeze(0),
                past_key_values=cache,
            )
        tp.mlu.synchronize()
        return cache

    print("isolated operators")
    plain = nn.Linear(hidden, hidden, bias=False, device="mlu", dtype=tp.float16)
    bench("dense matmul x @ (4096, 4096)", lambda: plain(x), args.repeat)
    bench("dense matmul (1, 4096) @ (4096, 4096)",
          lambda: x.reshape(1, hidden) @ plain.weight.t(), args.repeat)
    bench("projection x @ cached (4096, 4096)",
          lambda: mixer.q_proj(x), args.repeat)
    bench("projection x @ cached (4096, 12288)",
          lambda: mixer.q_proj and layer.mlp.gate_proj(x), args.repeat)
    head = nn.Linear(hidden, config.vocab_size, bias=False, device="mlu", dtype=tp.float16)
    bench("output projection x @ (151936, 4096)", lambda: head(x), args.repeat)

    print("\nblock parts (attention cache warmed to 20)")
    bench("rotary table only", lambda: mixer.rope(x, positions), args.repeat)
    bench("input layernorm", lambda: layer.input_layernorm(x), args.repeat)
    bench("mlp", lambda: layer.mlp(x), args.repeat)
    bench("attention", lambda: mixer(x, position_ids=positions, past_key_values=prefilled()),
          max(1, args.repeat // 4))
    cache = prefilled()
    bench("attention (cache held, grows)", lambda: mixer(x, position_ids=positions,
                                                         past_key_values=cache), args.repeat)
    whole = prefilled()
    bench("whole block", lambda: layer(x, position_ids=positions,
                                       past_key_values=whole), args.repeat)

    print(f"\nrotary tables held: {len(M._ROTARY_TABLES)}")


if __name__ == "__main__":
    main()
