"""Probe the accelerator's weight-only quantized product at accepted settings.

The product refuses the checkpoint's group of 32, so the settings it does take
have to be established before anything is built on top of them: which
activation widths run, whether the combination of a packed group and its
factor reproduces the values the group stands for, and what the product costs
at the shapes the language tower uses.

    python probe_w4a16.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def pack_pairs(codes: np.ndarray) -> np.ndarray:
    """Pack four-bit codes pairwise: the first code of a pair in the low half."""
    pairs = codes.reshape(codes.shape[0], -1, 2)
    packed = ((pairs[..., 1] & 0xF) << 4) | (pairs[..., 0] & 0xF)
    return packed.astype(np.uint8).view(np.int8).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mlu")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()

    import torch
    import torch_mlu_ops

    rng = np.random.default_rng(0)

    def run(a, b, scale):
        return torch_mlu_ops.weight_only_quant_matmul(
            a, b, scale, None, None, None, "none", 4, 1.0, 1.0, False
        )

    # --- does the pairing reproduce the group's values, and at which widths ---
    rows, cols = 256, 512
    groups = cols // args.group
    codes = rng.integers(-8, 8, size=(rows, cols)).astype(np.int64)
    scale = rng.uniform(0.005, 0.02, size=(rows, groups)).astype(np.float32)
    dense = (codes.reshape(rows, groups, args.group) * scale[:, :, None]).reshape(rows, cols)

    activation = rng.standard_normal((64, cols)).astype(np.float32)
    want_all = activation @ dense.T.astype(np.float32)

    b = torch.from_numpy(pack_pairs(codes)).to(args.device).contiguous()
    scales = torch.from_numpy(scale.astype(np.float16)).to(args.device).contiguous()

    print(f"group={args.group}  shape {rows}x{cols}  groups={groups}")
    print("\nvalue and accepted widths")
    accepted = []
    for width in [1, 2, 4, 8, 16, 32, 64]:
        act = torch.from_numpy(activation[:width].copy()).to(args.device).half()
        try:
            got = run(act, b, scales).float().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            print(f"  M={width:3} refused: {str(exc).splitlines()[-1][:80]}")
            continue
        error = float(np.abs(got - want_all[:width]).max())
        relative = error / (float(np.abs(want_all[:width]).max()) + 1e-9)
        verdict = "MATCH" if relative < 5e-3 else "wrong"
        print(f"  M={width:3} max abs {error:.5f}  rel {relative:.3e}  {verdict}")
        accepted.append(width)

    if not accepted:
        print("\nnothing accepted at this group size")
        return

    # --- cost at the shapes the language tower uses ---
    print("\ncost at real shapes")
    for out_dim, in_dim in [(4096, 4096), (4096, 12288), (151936, 4096)]:
        weight = torch.from_numpy(
            rng.integers(-128, 127, size=(out_dim, in_dim // 2), dtype=np.int8)
        ).to(args.device).contiguous()
        factor = torch.from_numpy(
            (rng.random((out_dim, in_dim // args.group)) * 0.02).astype(np.float16)
        ).to(args.device).contiguous()
        packed_bytes = out_dim * in_dim * 0.5
        for width in accepted:
            act = torch.from_numpy(
                rng.standard_normal((width, in_dim)).astype(np.float32)
            ).to(args.device).half()
            try:
                for _ in range(3):
                    run(act, weight, factor)
                torch.mlu.synchronize()
            except Exception:  # noqa: BLE001
                continue
            start = time.time()
            for _ in range(args.repeat):
                run(act, weight, factor)
            torch.mlu.synchronize()
            per_call = (time.time() - start) / args.repeat
            print(f"  {out_dim:6}x{in_dim:<6} M={width:3}: {per_call * 1e3:8.4f} ms  "
                  f"{packed_bytes / per_call / 1e9:7.1f} GB/s (packed only)")
            del act
        del weight, factor


if __name__ == "__main__":
    main()
