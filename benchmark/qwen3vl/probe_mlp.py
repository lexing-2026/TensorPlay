"""Time the small operators a block issues, module call against written out.

A block spends time on pieces that are not products: normalizing, the gated
activation, elementwise combining.  A module may express one of those as a
handful of calls, so the same arithmetic is timed both ways to see which the
block is paying for.

    python probe_mlp.py [--repeat 300]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402


def bench(label, fn, repeat):
    for _ in range(5):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:44} {(time.time() - start) / repeat * 1e3:9.4f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=300)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    x = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    wide = tp.randn(1, 1, 12288, device="mlu", dtype=tp.float16)
    eps = 1e-6

    print("gated activation, 12288 wide")
    silu = tp.nn.SiLU()
    bench("nn.SiLU", lambda: silu(wide), args.repeat)
    bench("write out: x * sigmoid(x)", lambda: wide * tp.sigmoid(wide), args.repeat)
    bench("sigmoid alone", lambda: tp.sigmoid(wide), args.repeat)
    bench("multiply alone", lambda: wide * wide, args.repeat)

    print("\nnormalization, 4096 wide")
    norm = tp.nn.RMSNorm(4096, eps=eps, device="mlu", dtype=tp.float16)
    bench("nn.RMSNorm", lambda: norm(x), args.repeat)
    weighted = norm.weight

    def written_out():
        mean = (x * x).mean(-1, keepdim=True)
        return weighted * (x * tp.rsqrt(mean + eps))

    bench("write out: rms then scale", written_out, args.repeat)
    print(f"  reference value {float(norm(x).reshape(-1)[0]):.6f} "
          f"vs {float(written_out().reshape(-1)[0]):.6f}")

    print("\nmodule call against calling the body directly")
    bench("nn.RMSNorm via __call__", lambda: norm(x), args.repeat)
    bench("nn.RMSNorm via forward", lambda: norm.forward(x), args.repeat)
    bench("nn.SiLU via __call__", lambda: silu(wide), args.repeat)
    bench("nn.SiLU via forward", lambda: silu.forward(wide), args.repeat)

    print("\nfeed-forward, as the block calls it")
    from qwen3vl.model import Qwen3VLMLP, Qwen3VLTextConfig

    config = Qwen3VLTextConfig.from_dir("/root/models/Qwen3-VL-8B-Instruct-AWQ-4bit")
    mlp = Qwen3VLMLP(config, weight_only=True, device="mlu", dtype=tp.float16)
    from qwen3vl.model import WeightOnlyInt4Linear

    for module in mlp.modules():
        if isinstance(module, WeightOnlyInt4Linear):
            module._dense_cache = tp.randn(
                module.out_features, module.in_features, device="mlu", dtype=tp.float16
            )
    tp.mlu.synchronize()
    bench("mlp (module call)", lambda: mlp(x), args.repeat)
    bench("mlp via forward", lambda: mlp.forward(x), args.repeat)
    bench("gate only", lambda: mlp.gate_proj(x), args.repeat)
    bench("gate, body only", lambda: x @ mlp.gate_proj._dense_cache.t(), args.repeat)
    bench("down only", lambda: mlp.down_proj(wide), args.repeat)

    def parts():
        return mlp.down_proj(silu(mlp.gate_proj(x)) * mlp.up_proj(x))

    bench("mlp written out", parts, args.repeat)


if __name__ == "__main__":
    main()
