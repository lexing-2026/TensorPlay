"""Time the vendor resident-weight product at the shapes a decode step uses.

A step's projections cannot run on the device's own product: the backend
carries no product that reads weights narrower than the activations.  The
vendor library does carry one, so the question is what rate it reaches and how
many bytes it saves, since a step that reads fewer bytes per token can be
shorter than one that reads more.

The product is timed at the shapes the tower uses and at the several activation
row counts a decode step can present, for the group sizes the kernel accepts,
and the totals are projected onto a full tower and the output head.  The
weights and their group factors are the only tensors read; the activations are
a single row, so the reported rate is the rate a step would run its tower at.

    python probe_vendor.py [--repeat 30] [--rows 1,2,4,8]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch_mlu  # noqa: F401
import torch_mlu_ops

# (name, output width, contracted width).
SHAPES = [
    ("q", 4096, 4096),
    ("k", 1024, 4096),
    ("v", 1024, 4096),
    ("o", 4096, 4096),
    ("gate", 12288, 4096),
    ("up", 12288, 4096),
    ("down", 4096, 12288),
]
HEAD = ("lm_head", 151936, 4096)

# The group sizes the kernel accepts.
CONFIGS = [(4, 64), (4, 128), (8, 64), (8, 128)]


def build(out_width, in_width, rows, bits, group, device="mlu"):
    packed_width = in_width // 2 if bits == 4 else in_width
    # The narrow integer kinds have no generator on the device, so the codes
    # are drawn as bytes on the host and moved over.
    codes = torch.randint(
        0, 256, (out_width, packed_width), dtype=torch.int64
    ).to(torch.int8).to(device)
    scale = (torch.rand(out_width, in_width // group, dtype=torch.float32,
                        device=device) * 0.01 + 0.01).to(torch.float16)
    activation = torch.randn(rows, in_width, dtype=torch.float16, device=device)
    return activation, codes, scale


def moved_bytes(out_width, in_width, bits, group):
    return out_width * in_width * bits / 8 + out_width * (in_width // group) * 2


def timed(fn, repeat):
    for _ in range(3):
        fn()
    torch.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    torch.mlu.synchronize()
    return (time.time() - start) / repeat


def measure(shape, rows, bits, group, repeat):
    name, out_width, in_width = shape
    activation, codes, scale = build(out_width, in_width, rows, bits, group)

    def call():
        return torch_mlu_ops.weight_only_quant_matmul(
            activation, codes, scale, quant_bit_size=bits)

    elapsed = timed(call, repeat)
    return elapsed, moved_bytes(out_width, in_width, bits, group) / elapsed / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--rows", default="1,2,4,8")
    args = ap.parse_args()

    rows_list = [int(v) for v in args.rows.split(",")]
    print("device", torch.mlu.get_device_name(0))

    for bits, group in CONFIGS:
        print(f"\n{bits}-bit weights, group {group}")
        print(f"  {'shape':9} {'out':>7} {'in':>6} {'rows':>5} {'ms':>9} {'GB/s':>8}")
        best_rows = None
        for rows in rows_list:
            total = 0.0
            for shape in SHAPES:
                try:
                    elapsed, rate = measure(shape, rows, bits, group, args.repeat)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {shape[0]:9} {shape[1]:7} {shape[2]:6} {rows:5} "
                          f"{type(exc).__name__}: {str(exc)[:40]}")
                    continue
                total += elapsed
                print(f"  {shape[0]:9} {shape[1]:7} {shape[2]:6} {rows:5} "
                      f"{elapsed * 1e3:9.4f} {rate:8.1f}")
            print(f"  {'layer':9} {'':7} {'':6} {rows:5} {total * 1e3:9.4f} "
                  f"({total * 36 * 1e3:6.2f} ms/tower)")
            if rows == 1:
                best_rows = total

        if best_rows is None:
            continue
        out_width, in_width = HEAD[1], HEAD[2]
        try:
            elapsed, rate = measure(HEAD, 1, bits, group, args.repeat)
            tower = best_rows * 36
            print(f"  head one row: {elapsed * 1e3:7.3f} ms  {rate:7.1f} GB/s")
            print(f"  projected step (tower + head) at one row: "
                  f"{(tower + elapsed) * 1e3:6.2f} ms   (reference 25.3 ms)")
        except Exception as exc:  # noqa: BLE001
            print(f"  head one row: {type(exc).__name__}: {str(exc)[:50]}")


if __name__ == "__main__":
    main()
