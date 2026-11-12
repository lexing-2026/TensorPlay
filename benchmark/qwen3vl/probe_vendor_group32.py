"""Whether the library's own quantized product accepts this checkpoint's group.

The checkpoint keeps four-bit weights in groups of thirty-two, and the
resident-weight product is known to accept only groups of sixty-four and up.
The general product is the one that takes a group size from the caller, so it
is tried here at this checkpoint's group and at a width the library is known to
accept.  What comes back is reported along with the time it took and the
distance from a dense product of the same values.

    python probe_vendor_group32.py [--repeat 30]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch_mlu  # noqa: F401
import torch_mlu_ops

# (name, contracted width, output width, group size).
SHAPES = [
    ("q", 4096, 4096, 32),
    ("gate", 4096, 12288, 32),
    ("q64", 4096, 4096, 64),
    ("q128", 4096, 4096, 128),
    ("head", 4096, 151936, 32),
]

GENERAL = torch.ops.torch_mlu_ops.quant_matmul


def build(in_width, out_width, group, device="mlu"):
    """A four-bit weight in groups, the activation, and a dense reference.

    The codes are packed two to a byte with the first of each pair in the low
    field, which is what the stored form uses.
    """
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (out_width, in_width), dtype=torch.int32)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scale = torch.rand(out_width, in_width // group, dtype=torch.float32) * 0.05
    dense = ((codes - 8).float() * scale.repeat_interleave(group, dim=1)).half()

    x = torch.randn(1, in_width, dtype=torch.float16)
    amax = x.abs().max()
    a_scale = (amax / 127.0).to(torch.float16).to(device)
    a_q = torch.round(x / amax * 127.0).clamp(-128, 127).to(torch.int8).to(device)
    return (a_q, a_scale, packed.to(torch.int8).to(device), scale.to(device),
            dense.to(device), x.to(device))


def general(a_q, a_scale, w4, scale, a_layout):
    """The general product: quantized activations, group-wise four-bit weight."""
    return GENERAL(
        a_q, a_scale, None, w4, scale.half(), None, None, None, None, None,
        None, None, "half", None, "smooth_quant", a_layout,
        "quantize_group_wise", 4, "none", False, 1.0, 1.0, 1.0, False, True)


def timed(fn, repeat):
    torch.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    torch.mlu.synchronize()
    return (time.time() - start) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    print("device", torch.mlu.get_device_name(0))
    print(f"  {'shape':6} {'group':>5} {'entry':24} {'ms':>9} {'GB/s':>8}  note")
    for name, in_width, out_width, group in SHAPES:
        a_q, a_scale, w4, scale, dense, x = build(in_width, out_width, group)
        ref = (x @ dense.t()).float()
        weight_bytes = out_width * in_width * 0.5 + out_width * (in_width // group) * 2

        runs = [
            ("resident/weights", lambda: torch_mlu_ops.weight_only_quant_matmul(
                x, w4, scale.half(), quant_bit_size=4)),
            ("general/act-per-token", lambda: general(
                a_q, a_scale.reshape(1), w4, scale, "quantize_per_token")),
            ("general/act-per-tensor", lambda: general(
                a_q, a_scale.reshape(1), w4, scale, "quantize_per_tensor")),
        ]
        for entry, call in runs:
            try:
                out = call()
                torch.mlu.synchronize()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).replace("\n", " ")[:60]
                print(f"  {name:6} {group:5d} {entry:24} {'--':>9} {'--':>8}  "
                      f"{type(exc).__name__}: {msg}")
                continue
            rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            elapsed = timed(call, args.repeat)
            print(f"  {name:6} {group:5d} {entry:24} {elapsed * 1e3:9.4f} "
                  f"{weight_bytes / elapsed / 1e9:8.1f}  rel={rel:.2e}")
        del a_q, a_scale, w4, scale, dense, x, ref


if __name__ == "__main__":
    main()
