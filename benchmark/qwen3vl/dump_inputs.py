"""Write the exact inputs a prompt (and optionally a picture) turns into.

The chat template and the image preprocessing belong to the published stack;
this dumps their output so another implementation can be driven on byte-identical
tokens and patches.

    python dump_inputs.py --model <dir> --prompt "..." --out dump.npz
    python dump_inputs.py --model <dir> --image --out dump_img.npz
"""

from __future__ import annotations

import argparse

import numpy as np
from transformers import AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用一句话介绍你自己。")
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", default="dump.npz")
    args = ap.parse_args()

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

    payload = {"prompt_ids": inputs["input_ids"].cpu().numpy()[0]}
    if "image_grid_thw" in inputs:
        payload["pixel_values"] = inputs["pixel_values"].float().cpu().numpy()
        payload["image_grid_thw"] = inputs["image_grid_thw"].cpu().numpy()
        print("grid:", inputs["image_grid_thw"].tolist(), "patches:", tuple(payload["pixel_values"].shape))

    np.savez(args.out, **payload)
    print("prompt tokens:", tuple(inputs["input_ids"].shape), "wrote", args.out)


if __name__ == "__main__":
    main()
