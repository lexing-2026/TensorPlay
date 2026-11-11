"""Time the published reference stack decoding on this machine.

The same checkpoint, the same greedy contract, the standard implementation.
Reported per-token so the numbers can be put next to another backend's.

    python ref_speed.py --model <dir> --tokens 16
"""

from __future__ import annotations

import argparse
import statistics as stats
import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用一句话介绍你自己。")
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    processor = AutoProcessor.from_pretrained(args.model)
    content = [{"type": "text", "text": args.prompt}]
    picture = None
    if args.image:
        import sys

        sys.path.insert(0, ".")
        from ref_dump import build_image

        picture = build_image(args.image_size)
        content = [{"type": "image", "image": picture}] + content
    chat = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat], images=[picture] if picture is not None else None, return_tensors="pt"
    )
    ids = inputs["input_ids"]
    print(f"prompt tokens: {ids.shape[1]}  image: {picture is not None}")

    start = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, low_cpu_mem_usage=True
    ).eval()
    print(f"load {time.time() - start:.1f}s")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    for warmup in range(2):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=3, do_sample=False)
    torch.cuda.synchronize()

    # One token at a time so the prefill is separated from the decode.
    prefill_ids = ids.to(args.device)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        out = model(input_ids=prefill_ids, **{k: v for k, v in inputs.items() if k != "input_ids"})
        torch.cuda.synchronize()
    print(f"prefill {time.time() - start:.3f}s")

    past = out.past_key_values
    token = out.logits[:, -1:].argmax(-1)
    times = []
    cache_position = torch.tensor([ids.shape[1]], device=args.device)
    for step in range(args.tokens):
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            out = model(
                input_ids=token,
                past_key_values=past,
                cache_position=cache_position + step,
                use_cache=True,
            )
            torch.cuda.synchronize()
            times.append(time.time() - start)
        past = out.past_key_values
        token = out.logits[:, -1:].argmax(-1)

    print("steps " + " ".join(f"{v * 1e3:.1f}" for v in times) + " ms")
    print(f"median {stats.median(times) * 1e3:.1f} ms  {1 / stats.median(times):.2f} token/s")


if __name__ == "__main__":
    main()
