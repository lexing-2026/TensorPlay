"""Time every layer of the real tower inside a real decode step.

A block measured on its own may not cost what it costs in the stack, because
the stack holds two hundred distinct weight buffers and walks them in order.
This wraps each layer so its own span is timed, alongside the embedding, the
final normalization and the output projection, and reports what each costs
beside the step as a whole.

    python probe_layers.py --model <dir> [--steps 4]
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import cache_dense_weights, load_qwen3vl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt-ids", default="104198,48,16948,3837,67071")
    ap.add_argument("--sync-each", action="store_true",
                    help="synchronize around every wrapped module")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, config = load_qwen3vl(
        args.model, dtype=tp.float16, device="mlu", progress=print
    )
    cache_dense_weights(model, tp.float16)
    sync()

    language = model.language_model
    records: dict[str, list[float]] = {}

    def wrap(module, name):
        body = module.forward

        def timed(*positional, **keywords):
            sync()
            start = time.time()
            out = body(*positional, **keywords)
            sync()
            records.setdefault(name, []).append(time.time() - start)
            return out

        module.forward = timed

    for index, layer in enumerate(language.layers):
        wrap(layer, f"layer {index}")
    wrap(language.embed_tokens, "embedding")
    wrap(language.norm, "final normalization")
    wrap(model.lm_head, "output projection")
    wrap(model.language_model, "blocks and final normalization")

    ids = tp.tensor([[int(v) for v in args.prompt_ids.split(",")]],
                    dtype=tp.int64, device="mlu")
    length = ids.shape[1]

    cache = model.create_empty_cache()
    with tp.no_grad():
        logits = model(ids, position_ids=tp.arange(0, length, device="mlu").unsqueeze(0),
                       past_key_values=cache)
    sync()

    records.clear()
    steps = []
    with tp.no_grad():
        for offset in range(args.steps):
            token = logits[:, -1:].argmax(dim=-1)
            position = tp.arange(length + offset, length + offset + 1,
                                 device="mlu").unsqueeze(0)
            sync()
            start = time.time()
            logits = model(token, position_ids=position, past_key_values=cache)
            sync()
            steps.append(time.time() - start)
        records_snapshot = {name: list(values) for name, values in records.items()}

    print(f"\nstep, without per-layer synchronization: "
          f"median {stats.median(steps) * 1e3:.1f} ms")

    print("\nthe same steps with every part synchronized around")
    total = 0.0
    for name, values in records_snapshot.items():
        if name in ("blocks and final normalization",):
            continue
        mean = sum(values) / len(values)
        total += mean
        print(f"  {name:32} {mean * 1e3:9.3f} ms")
    print(f"  {'sum of the parts':32} {total * 1e3:9.3f} ms")

    layers = [value for name, values in records_snapshot.items()
              if name.startswith("layer ") for value in values]
    if layers:
        print(f"\n  a layer, summed over 36:          {sum(layers) / args.steps * 1e3:9.3f} ms")
        print(f"  cheapest layer                    {min(layers) * 1e3:9.3f} ms")
        print(f"  dearest layer                     {max(layers) * 1e3:9.3f} ms")


if __name__ == "__main__":
    main()
