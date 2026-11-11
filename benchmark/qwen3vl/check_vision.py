"""Compare the framework's vision tower against a published reference tower.

The two sides live in different environments, so the script works in two
modes.  ``--dump-ref`` runs the reference tower and stores its patches and
outputs; ``--compare`` runs the framework's tower on the stored patches and
compares block, merged and deepstack outputs elementwise.  Without
``--compare`` the framework's tower is not imported at all, so the dump half
only needs the environment the checkpoint was published for.

    python check_vision.py --model <dir> --grid 1,16,16 --dump-ref ref.npz
    python check_vision.py --model <dir> --grid 1,16,16 --compare ref.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_grid(text: str):
    return [tuple(int(v) for v in part.split(",")) for part in text.split(";")]


def load_pixels(model_dir: str, grid, seed: int) -> np.ndarray:
    with open(os.path.join(model_dir, "config.json")) as handle:
        visual = json.load(handle)["vision_config"]
    patches = int(sum(t * h * w for t, h, w in grid))
    channels = int(visual["in_channels"]) * int(visual["temporal_patch_size"]) * int(visual["patch_size"]) ** 2
    rng = np.random.default_rng(seed)
    return rng.standard_normal((patches, channels)).astype(np.float32)


def dump_reference(args, grid, pixels, path):
    import torch
    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig as ReferenceConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    with open(os.path.join(args.model, "config.json")) as handle:
        reference_config = ReferenceConfig(**json.load(handle)["vision_config"])
    reference_config._attn_implementation = "eager"
    reference = Qwen3VLVisionModel(reference_config).eval()

    index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
    state = {}
    for shard in sorted(set(index.values())):
        for name, tensor in load_file(os.path.join(args.model, shard)).items():
            if name.startswith("model.visual."):
                state[name[len("model.visual.") :]] = tensor
    missing, unexpected = reference.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("MISSING:", missing[:6], "UNEXPECTED:", unexpected[:6])
        raise SystemExit("state dict does not line up with the tower")
    reference = reference.float()

    with torch.no_grad():
        out = reference(
            torch.from_numpy(pixels),
            grid_thw=torch.tensor(grid, dtype=torch.int64),
            return_dict=True,
        )
    np.savez(
        path,
        pixels=pixels,
        hidden=out.last_hidden_state.numpy(),
        merged=out.pooler_output.numpy(),
        deepstack=np.stack([t.numpy() for t in out.deepstack_features]),
    )
    print(f"reference: hidden {tuple(out.last_hidden_state.shape)} "
          f"merged {tuple(out.pooler_output.shape)} deepstack "
          f"{[tuple(t.shape) for t in out.deepstack_features]}")
    print("wrote", path)


def compare(args, path, grid):
    import tensorplay as tp

    from qwen3vl.vision import Qwen3VLVisionConfig, load_vision

    stored = np.load(path)
    pixels = stored["pixels"]
    config = Qwen3VLVisionConfig.from_dir(args.model)
    print(config)

    tower = load_vision(args.model, config, dtype=tp.float32, device=args.device)
    with tp.no_grad():
        hidden, merged, deepstack = tower(tp.from_numpy(pixels), grid)

    def to_numpy(value):
        return value.detach().cpu().numpy()

    def report(label, reference, got):
        if reference.shape != got.shape:
            print(f"{label}: SHAPE {reference.shape} vs {got.shape}")
            return
        diff = np.abs(reference - got)
        scale = np.abs(reference).max() + 1e-9
        status = "ok" if diff.max() < args.report else "MISMATCH"
        print(f"{label}: max abs {diff.max():.3e}  rel {diff.max() / scale:.3e}  {status}")

    report("block output", stored["hidden"], to_numpy(hidden))
    report("merged tokens", stored["merged"], to_numpy(merged))
    for index, layer in enumerate(deepstack):
        report(f"deepstack[{index}]", stored["deepstack"][index], to_numpy(layer))
    print("merged tokens:", tuple(merged.shape), "expected per image:",
          [t * h * w // config.spatial_merge_size ** 2 for t, h, w in grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", default="1,16,16", help="t,h,w per image; use ; between images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-ref", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", type=float, default=1e-3)
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    pixels = load_pixels(args.model, grid, args.seed)
    if args.dump_ref:
        dump_reference(args, grid, pixels, args.dump_ref)
    if args.compare:
        compare(args, args.compare, grid)


if __name__ == "__main__":
    main()
