"""Run a Qwen3-VL checkpoint on a chosen device and compare with a reference dump.

    python run_qwen3vl.py --model <dir> --ref ref.npz --device mlu
    python run_qwen3vl.py --model <dir> --ref ref.npz --device cpu --dtype float32

The prompt tokens come from the reference dump when one is given, so both
sides decode exactly the same sequence; the continuation, the per-step argmax
agreement and the first-step score distance are reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

from qwen3vl.model import load_qwen3vl  # noqa: E402


def pick_device(name: str) -> str:
    if name == "mlu":
        import tensorplay_mlu

        tensorplay_mlu.load()
        if tp.mlu.device_count() == 0:
            raise RuntimeError("no MLU device is visible")
        return "mlu"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--prompt-ids", default=None, help="comma separated token ids")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--dense",
        action="store_true",
        help="hold the language projections in the activation's element type "
        "instead of their four-bit groups",
    )
    ap.add_argument(
        "--cache-dense",
        action="store_true",
        help="keep the four-bit groups resident but recover their values once, "
        "so a forward is one plain product instead of one recovery per forward",
    )
    args = ap.parse_args()

    dtypes = {"float16": tp.float16, "bfloat16": tp.bfloat16, "float32": tp.float32}
    device = pick_device(args.device)

    if args.ref:
        ref = np.load(args.ref)
        prompt_ids = np.asarray(ref["prompt_ids"], dtype=np.int64).reshape(1, -1)
        ref_seq = ref["sequence_ids"]
        ref_scores = ref["scores"]
        tokens = args.tokens or int(ref_scores.shape[0])
        prompt = str(ref["prompt"])
        pixels = ref["pixel_values"] if "pixel_values" in ref else None
        grid = ref["image_grid_thw"] if "image_grid_thw" in ref else None
    else:
        prompt_ids = np.array([[int(x) for x in args.prompt_ids.split(",")]], dtype=np.int64)
        ref_seq = ref_scores = None
        pixels = grid = None
        tokens = args.tokens
        prompt = args.prompt_ids

    print(f"device={device} dtype={args.dtype} prompt tokens={len(prompt_ids[0])}")
    t0 = time.time()
    model, config = load_qwen3vl(
        args.model,
        dtype=dtypes[args.dtype],
        device=device,
        weight_only=not args.dense,
        progress=print,
    )
    if args.cache_dense and not args.dense:
        from qwen3vl.model import cache_dense_weights

        cached = cache_dense_weights(model, dtypes[args.dtype])
        print(f"recovered {cached} projection weights once")
    print(f"load {time.time() - t0:.1f}s  {config}")

    ids = tp.from_numpy(prompt_ids.astype(np.int64)).to(device=device)
    if pixels is not None:
        print(f"image: {pixels.shape}  grid {grid.tolist()}")
        model.prepare_image(
            tp.from_numpy(pixels).to(device=device, dtype=dtypes[args.dtype]),
            grid.tolist(),
        )
    from tensorplay.nn.generation import generate

    t0 = time.time()
    with tp.no_grad():
        out = generate(model, ids, max_new_tokens=tokens, do_sample=False)
        if device != "cpu":
            tp.mlu.synchronize() if device == "mlu" else tp.cuda.synchronize()
    elapsed = time.time() - t0
    seq = out.cpu().numpy()[0]
    new = seq[len(prompt_ids[0]):]
    print(f"generate {elapsed:.2f}s  {elapsed / max(tokens, 1):.3f}s/token")
    print("continuation ids:", new.tolist())

    result = {
        "device": device,
        "dtype": args.dtype,
        "prompt": prompt,
        "prompt_tokens": int(len(prompt_ids[0])),
        "continuation": new.tolist(),
        "seconds": round(elapsed, 3),
        "seconds_per_token": round(elapsed / max(tokens, 1), 4),
    }
    if ref_seq is not None:
        match = [int(a == b) for a, b in zip(new.tolist(), ref_seq[len(prompt_ids[0]):].tolist())]
        first = next((i for i, m in enumerate(match) if not m), None)
        result["token_match"] = match
        result["first_mismatch"] = first
        print("token agreement:", sum(match), "/", len(match), "first mismatch at", first)

        # Score comparison at the whole-prompt pass: prefill logits at the last
        # position should line up.
        with tp.no_grad():
            image = {}
            if pixels is not None:
                image = {
                    "pixel_values": tp.from_numpy(pixels).to(
                        device=device, dtype=dtypes[args.dtype]
                    ),
                    "image_grid_thw": grid.tolist(),
                }
            logits = model(ids, **image)[:, -1, :].float().cpu().numpy()
        ref_first = ref_scores[0].astype(np.float32)
        diff = np.abs(logits[0] - ref_first)
        result["first_step_max_abs"] = float(diff.max())
        result["first_step_argmax"] = int(logits[0].argmax())
        result["first_step_ref_argmax"] = int(ref_first.argmax())
        print(f"first-step max abs diff {diff.max():.4f}  argmax {logits[0].argmax()} "
              f"(ref {ref_first.argmax()})")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
