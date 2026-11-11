"""Work out how the accelerator's weight-only product reads a packed checkpoint.

The product recovers four-bit groups inside the kernel, so what matters is
only how a byte pair is interpreted and which activation widths it accepts.
A real projection is read from the checkpoint and put through each reading;
the reading that reproduces the recovered values is the convention to file the
checkpoint under, and the widths it runs at decide how a decode step can use it.

    python check_wqmm.py --model <dir> [--layer 0] [--name q_proj]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3vl.ct_weights import Checkpoint, pack_codes_uint8, unpack_int4  # noqa: E402

DEFAULT = "model.language_model.layers.{layer}.self_attn.{name}.weight"
WIDTHS = [1, 2, 4, 8, 16, 32, 64]


def readings(packed, codes, columns):
    """The byte layouts a packer can plausibly have produced."""
    # The stored words read as bytes: the first code of a pair sits in the low
    # half, and the four bits are signed.
    direct = packed.astype(np.int32).astype("<i4").view(np.int8).copy()
    # The same codes with the first of a pair in the high half.
    high_first = pack_codes_uint8((codes & 0xF).astype(np.uint8)).astype(np.int32)
    high_first = high_first.view(np.int8).copy()
    return {"stored bytes as they lie": direct, "first code in high half": high_first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--name", default="q_proj")
    ap.add_argument("--device", default="mlu")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args()

    import torch
    import torch_mlu_ops

    base = DEFAULT.format(layer=args.layer, name=args.name)
    ckpt = Checkpoint(args.model)
    packed = ckpt.raw(base + "_packed")
    scale = ckpt.float32(base + "_scale")
    shape = ckpt.raw(base + "_shape")
    rows, columns = int(shape[0]), int(shape[1])
    groups = scale.shape[1]
    print(f"{base}: out={rows} in={columns} groups={groups} "
          f"group_size={columns // groups}")

    codes = unpack_int4(packed, columns)                     # signed, -8..7
    dense = (codes.reshape(rows, groups, -1) * scale[:, :, None]).reshape(rows, columns)

    rng = np.random.default_rng(0)
    activation = rng.standard_normal((1, columns)).astype(np.float32)
    want = (activation @ dense.T.astype(np.float32))[0]

    scales = torch.from_numpy(scale.astype(np.float16)).to(args.device).contiguous()

    def call(act, payload):
        b = torch.from_numpy(payload).to(args.device).contiguous()
        return torch_mlu_ops.weight_only_quant_matmul(
            act, b, scales, None, None, None, "none", 4, 1.0, 1.0, False
        )

    print("\nvalue and width")
    matched = None
    for label, payload in readings(packed, codes, columns).items():
        a = torch.from_numpy(activation).to(args.device).half()
        for width in WIDTHS:
            wide = torch.from_numpy(
                np.repeat(activation, width, axis=0).astype(np.float32)
            ).to(args.device).half()
            try:
                got = call(wide, payload).float().cpu().numpy()[0]
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:26} M={width:3} refused: {str(exc).splitlines()[0][:70]}")
                continue
            error = float(np.abs(got - want).max())
            relative = error / (float(np.abs(want).max()) + 1e-9)
            verdict = "MATCH" if relative < 5e-3 else "wrong"
            print(f"  {label:26} M={width:3} max abs {error:.4f}  "
                  f"rel {relative:.3e}  {verdict}")
            if verdict == "MATCH" and matched is None:
                matched = label
        if matched is not None:
            break

    if args.no_speed:
        ckpt.close()
        return

    # What a single-token step would cost if the width were padded up.
    print("\ncost at a padded width")
    payload = readings(packed, codes, columns)["stored bytes as they lie"]
    b = torch.from_numpy(payload).to(args.device).contiguous()
    for width in WIDTHS[3:]:
        act = torch.from_numpy(
            np.repeat(activation, width, axis=0).astype(np.float32)
        ).to(args.device).half()
        try:
            for _ in range(3):
                torch_mlu_ops.weight_only_quant_matmul(
                    act, b, scales, None, None, None, "none", 4, 1.0, 1.0, False
                )
        except Exception:  # noqa: BLE001
            continue
        torch.mlu.synchronize()
        start = time.time()
        for _ in range(args.repeat):
            torch_mlu_ops.weight_only_quant_matmul(
                act, b, scales, None, None, None, "none", 4, 1.0, 1.0, False
            )
        torch.mlu.synchronize()
        per_call = (time.time() - start) / args.repeat
        print(f"  {rows}x{columns} at M={width:3}: {per_call * 1e3:7.3f} ms  "
              f"{rows * columns * 0.5 / per_call / 1e9:7.1f} GB/s")

    ckpt.close()


if __name__ == "__main__":
    main()
