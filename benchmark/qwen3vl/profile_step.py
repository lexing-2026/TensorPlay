"""Split a decode step into the parts that are not the weight products.

A cached step does one product per projection and, per layer, builds a rotary
table, normalizes heads, concatenates the cache and attends.  On a device a
step that should be dominated by the products can be dominated by everything
else, so each non-product part is replaced by a shape-correct stand-in that
does no work and the step is re-timed.  The difference is what that part cost.

    python profile_step.py --model <dir> --tokens 6
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

import qwen3vl.model as M  # noqa: E402
from qwen3vl.model import cache_dense_weights, load_qwen3vl  # noqa: E402

FLAGS = {"rope": True, "sdpa": True, "norm": True, "cache": True}

_ROPE = M.MultimodalRotaryEmbedding.forward
_SDPA = tp.nn.functional.scaled_dot_product_attention


def rope_step(self, x, position_ids=None, max_position=None):
    if not FLAGS["rope"]:
        shape = (x.shape[0], x.shape[1], self.head_dim // 2)
        return (
            tp.ones(shape, dtype=x.dtype, device=x.device),
            tp.zeros(shape, dtype=x.dtype, device=x.device),
        )
    return _ROPE(self, x, position_ids, max_position)


def sdpa_step(query, key, value, **kwargs):
    if not FLAGS["sdpa"]:
        return tp.zeros(query.shape, dtype=query.dtype, device=query.device)
    return _SDPA(query, key, value, **kwargs)


def install():
    M.MultimodalRotaryEmbedding.forward = rope_step
    tp.nn.functional.scaled_dot_product_attention = sdpa_step


def decode(model, ids, steps, device):
    cache = model.create_empty_cache()
    length = ids.shape[1]
    with tp.no_grad():
        logits = model(ids, position_ids=_pos(0, length, device), past_key_values=cache)
        tp.mlu.synchronize()
        times = []
        for offset in range(steps):
            token = logits[:, -1:].argmax(dim=-1)
            tp.mlu.synchronize()
            start = time.time()
            logits = model(
                token,
                position_ids=_pos(length + offset, length + offset + 1, device),
                past_key_values=cache,
            )
            tp.mlu.synchronize()
            times.append(time.time() - start)
    return times


def _pos(start, end, device):
    return tp.arange(start, end, device=device).unsqueeze(0)


def measure(model, ids, device, tokens):
    best = None
    for _ in range(2):
        times = decode(model, ids, tokens, device)
        median = stats.median(times)
        best = median if best is None else min(best, median)
    return best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    install()

    model, _ = load_qwen3vl(args.model, dtype=tp.float16, device="mlu", progress=print)
    cache_dense_weights(model, tp.float16)
    ids = tp.tensor([[int(v) for v in args.prompt_ids.split(",")]], dtype=tp.int64, device="mlu")

    scenarios = [
        ("all parts", {}),
        ("no rotary table", {"rope": False}),
        ("no attention", {"sdpa": False}),
        ("neither", {"rope": False, "sdpa": False}),
    ]
    results = {}
    for label, flags in scenarios:
        for key in FLAGS:
            FLAGS[key] = True
        FLAGS.update(flags)
        results[label] = measure(model, ids, "mlu", args.tokens)
        print(f"  {label:18} {results[label]:8.2f} ms  "
              f"{1e3 / results[label]:6.2f} token/s")

    base = results["all parts"]
    print("\nshare of the step")
    for label in ("no rotary table", "no attention", "neither"):
        saved = base - results[label]
        print(f"  {label:18} saves {saved:7.2f} ms  ({saved / base * 100:5.1f}%)")


if __name__ == "__main__":
    main()
