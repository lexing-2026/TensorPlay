"""Dump a reference continuation for a Qwen3-VL checkpoint.

Runs the checkpoint on the stack it was published for and writes the prompt
tokens, the greedy continuation and the per-step scores to an ``.npz`` so
another implementation can be compared against exactly the same inputs.  With
``--image`` the checkpoint also sees a picture, and the dump then carries the
image patch batch, its grid and the token types the multimodal position
bookkeeping needs.

    python ref_dump.py --model <dir> --prompt "..." --tokens 16 --out ref.npz
    python ref_dump.py --model <dir> --image --out ref_img.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def build_image(size: int = 256, seed: int = 0):
    """A deterministic picture: colour fields with a bright diagonal."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    block = size // 8
    for row in range(8):
        for col in range(8):
            canvas[row * block : (row + 1) * block, col * block : (col + 1) * block] = (
                rng.integers(0, 256, size=3)
            )
    for step in range(size):
        canvas[step, step] = [255, 255, 255]
    return Image.fromarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用一句话介绍你自己。")
    ap.add_argument("--image", action="store_true", help="put a picture in the prompt")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ref.npz")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]
    processor = AutoProcessor.from_pretrained(args.model)

    content = [{"type": "text", "text": args.prompt}]
    picture = None
    if args.image:
        picture = build_image(args.image_size)
        content = [{"type": "image", "image": picture}] + content
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[picture] if picture is not None else None,
                       return_tensors="pt")
    ids = inputs["input_ids"]
    print("prompt tokens:", tuple(ids.shape), ids.tolist())

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, low_cpu_mem_usage=True
    )
    model.eval()
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    payload = {}
    if "image_grid_thw" in inputs:
        payload["pixel_values"] = inputs["pixel_values"].float().cpu().numpy()
        payload["image_grid_thw"] = inputs["image_grid_thw"].cpu().numpy()
        grid = inputs["image_grid_thw"].tolist()
        with torch.no_grad():
            position_ids = model.model.compute_3d_position_ids(
                input_ids=inputs["input_ids"],
                inputs_embeds=None,
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=mm_token_type_ids,
            )
        payload["position_ids"] = position_ids.cpu().numpy()
        payload["rope_delta"] = np.array(model.model.rope_deltas.cpu().numpy())
        print("grid:", grid, "positions:", tuple(position_ids.shape))

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seq = out.sequences[0].cpu().numpy()
    scores = np.stack([s[0].float().cpu().numpy() for s in out.scores]).astype(np.float16)
    text = processor.tokenizer.decode(seq[ids.shape[1] :], skip_special_tokens=False)
    np.savez(
        args.out,
        prompt_ids=ids.cpu().numpy()[0],
        sequence_ids=seq,
        scores=scores,
        prompt=args.prompt,
        text=text,
        **payload,
    )
    print("continuation ids:", seq[ids.shape[1] :].tolist())
    print("text:", text)
    print("wrote", args.out, "scores", scores.shape)


if __name__ == "__main__":
    main()
