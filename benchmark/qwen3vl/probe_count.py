"""Count the operators one decode step issues.

Every operator a step issues costs its own dispatch, and on this device the
dispatch is a large share of the step.  Before removing work it helps to know
how much of it there is, so the tensor methods and the top-level operators a
single block reaches are counted while that block runs once at decode shapes.

    python probe_count.py --model <dir>
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import (  # noqa: E402
    Qwen3VLDecoderLayer,
    Qwen3VLTextConfig,
    WeightOnlyInt4Linear,
)

# Attribute names on the tensor class, and the top-level names, that are
# counted.  Anything absent is skipped.
TENSOR_NAMES = (
    "__add__", "__radd__", "__sub__", "__mul__", "__truediv__", "__matmul__",
    "__getitem__", "softmax", "mean", "sum", "rsqrt", "sigmoid", "pow",
    "reshape", "view", "transpose", "contiguous", "clone", "to", "float",
    "half", "copy_", "expand", "unsqueeze", "squeeze", "chunk", "flatten",
    "repeat_interleave", "index_select", "index_copy_", "index_copy",
    "index_add", "cat", "narrow", "size", "numel", "t", "transpose",
)

TOP_NAMES = (
    "empty", "empty_like", "zeros", "zeros_like", "ones", "full", "full_like",
    "cat", "stack", "index_select", "arange", "tensor", "where", "sigmoid",
    "rsqrt", "mm", "matmul", "softmax", "mean", "mul", "add", "sub", "to",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    from tensorplay.nn.cache import DynamicCache

    config = Qwen3VLTextConfig.from_dir(args.model)
    layer = Qwen3VLDecoderLayer(config, 0, weight_only=True, device="mlu",
                                dtype=tp.float16)
    for module in layer.modules():
        if isinstance(module, WeightOnlyInt4Linear):
            module._dense_cache = tp.randn(
                module.out_features, module.in_features, device="mlu",
                dtype=tp.float16,
            )
    tp.mlu.synchronize()

    counts: collections.Counter = collections.Counter()
    original = {}

    def wrap(target, name, label):
        value = getattr(target, name, None)
        if value is None or not callable(value):
            return
        original[(target, name)] = value

        def counted(*positional, **keywords):
            counts[label] += 1
            return value(*positional, **keywords)

        try:
            setattr(target, name, counted)
        except (AttributeError, TypeError):
            del original[(target, name)]

    tensor_type = type(tp.randn(1, device="cpu"))
    for name in TENSOR_NAMES:
        wrap(tensor_type, name, f"Tensor.{name}")
    for name in TOP_NAMES:
        wrap(tp, name, f"tp.{name}")
    for name in ("softmax", "silu", "embedding", "rms_norm"):
        wrap(tp.nn.functional, name, f"functional.{name}")

    x = tp.randn(1, 1, config.hidden_size, device="mlu", dtype=tp.float16)
    cache = DynamicCache()
    try:
        with tp.no_grad():
            for step in range(20):
                layer(x, position_ids=tp.arange(step, step + 1, device="mlu").unsqueeze(0),
                      past_key_values=cache)
            tp.mlu.synchronize()
            counts.clear()
            for step in range(args.steps):
                layer(x,
                      position_ids=tp.arange(20 + step, 21 + step, device="mlu").unsqueeze(0),
                      past_key_values=cache)
            tp.mlu.synchronize()
    finally:
        for (target, name), value in original.items():
            setattr(target, name, value)

    total = sum(
        count for label, count in counts.items()
        if not label.split(".")[-1].startswith("_") or label.startswith("Tensor.__")
    )
    print(f"\none decode step, {args.steps} step(s), {total} operators")
    for label, count in counts.most_common():
        if count:
            print(f"  {count:4}  {label}")


if __name__ == "__main__":
    main()
