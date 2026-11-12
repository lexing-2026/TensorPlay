"""Time this implementation's decode the way the reference stack is timed.

One prefill, then one token per step, with a device synchronize on either
side of every interval so the prefill cost and the per-token cost stay
separate and can be read directly beside another stack's figures.

    python run_steps.py --model <dir> --inputs dump.npz --device cuda \
        --dtype bfloat16 --dense --tokens 16
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import load_qwen3vl  # noqa: E402

DTYPES = {"float16": tp.float16, "bfloat16": tp.bfloat16, "float32": tp.float32}


def pick_device(name: str) -> str:
    if name == "mlu":
        import tensorplay_mlu

        tensorplay_mlu.load()
        return "mlu"
    return name


def synchronize(device: str) -> None:
    if device == "cuda":
        tp.cuda.synchronize()
    elif device == "mlu":
        tp.mlu.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--inputs",
        default=None,
        help="npz with prompt_ids and, for an image prompt, pixel_values and image_grid_thw",
    )
    ap.add_argument("--prompt-ids", default=None, help="comma separated token ids")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=2, help="untimed passes before the timed one")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--cache-dense", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    device = pick_device(args.device)

    pixels = grid = None
    if args.inputs:
        dump = np.load(args.inputs)
        prompt_ids = np.asarray(dump["prompt_ids"], dtype=np.int64).reshape(1, -1)
        pixels = dump["pixel_values"] if "pixel_values" in dump else None
        grid = dump["image_grid_thw"] if "image_grid_thw" in dump else None
    else:
        prompt_ids = np.array([[int(x) for x in args.prompt_ids.split(",")]], dtype=np.int64)

    prompt_len = int(prompt_ids.shape[1])
    print(f"device={device} dtype={args.dtype} prompt tokens={prompt_len} image={pixels is not None}")

    t0 = time.time()
    model, config = load_qwen3vl(
        args.model, dtype=dtype, device=device, weight_only=not args.dense, progress=print
    )
    if args.cache_dense and not args.dense:
        from qwen3vl.model import cache_dense_weights

        print(f"recovered {cache_dense_weights(model, dtype)} projection weights once")
    print(f"load {time.time() - t0:.1f}s  {config}")

    ids = tp.from_numpy(prompt_ids).to(device=device)
    pixel_tensor = None
    if pixels is not None:
        pixel_tensor = tp.from_numpy(pixels).to(device=device, dtype=dtype)

    from tensorplay.nn.cache import DynamicCache

    def positions(start: int, length: int):
        return tp.arange(start, start + length, device=device).unsqueeze(0)

    def sweep(record: bool):
        """One prefill followed by ``--tokens`` single-token steps."""
        cache = DynamicCache()
        if pixel_tensor is not None:
            model.prepare_image(pixel_tensor, grid.tolist())

        with tp.no_grad():
            begin = 0.0
            if record:
                synchronize(device)
                begin = time.time()
            logits = model(ids, position_ids=positions(0, prompt_len), past_key_values=cache)
            if record:
                synchronize(device)
                prefill = time.time() - begin
            token = logits[:, -1, :].argmax(dim=-1).reshape(1, 1)

            times = []
            for index in range(args.tokens):
                if record:
                    synchronize(device)
                    begin = time.time()
                logits = model(
                    token,
                    position_ids=positions(prompt_len + index, 1),
                    past_key_values=cache,
                )
                if record:
                    synchronize(device)
                    times.append(time.time() - begin)
                token = logits[:, -1, :].argmax(dim=-1).reshape(1, 1)
        return (prefill if record else None), times

    for _ in range(args.warmup):
        sweep(False)

    prefill, times = sweep(True)
    median = stats.median(times)
    print(f"prefill {prefill:.3f}s")
    print("steps " + " ".join(f"{v * 1e3:.1f}" for v in times) + " ms")
    print(f"median {median * 1e3:.1f} ms  {1.0 / median:.2f} token/s")

    if args.json:
        import json

        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": device,
                    "dtype": args.dtype,
                    "prompt_tokens": prompt_len,
                    "image": pixels is not None,
                    "prefill_seconds": round(prefill, 4),
                    "step_seconds": [round(v, 6) for v in times],
                    "median_step_seconds": round(median, 6),
                    "tokens_per_second": round(1.0 / median, 3),
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
