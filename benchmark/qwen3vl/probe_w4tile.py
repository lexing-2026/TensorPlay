"""Whether the tile shape can rescue the four-bit extraction.

The extraction costs about four times what the words themselves cost, so the
question is whether the shape the work is cut into is what holds it back.
Every program is given a tile of output rows by a run of packed words, and the
tile is grown until the whole row fits, which makes the read of an output row
one contiguous run and leaves a single reduction per program.  The width of the
run and the number of rows are swept together with the number of workers, and
the codes are accumulated both as floats and as whole numbers, because the
accumulator's type decides how many conversions the loop pays for.

    python probe_w4tile.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl

OUT_WIDTH = 16384
IN_WIDTH = 8192
PER_WORD = 8


@triton.jit
def codes_float(w_ptr, y_ptr, kw, KW: tl.constexpr, PER_WORD: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Take every code out in turn, adding as floats into one tile per leg."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    total = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        part = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
        for p in tl.static_range(PER_WORD):
            part += ((word >> (4 * p)) & 0xF).to(tl.float32)
        total += tl.sum(part, axis=1)
    tl.store(y_ptr + rows, total)


@triton.jit
def codes_whole(w_ptr, y_ptr, kw, KW: tl.constexpr, PER_WORD: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Take every code out in turn, adding as whole numbers, one conversion."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    total = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        part = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.int32)
        for p in tl.static_range(PER_WORD):
            part += (word >> (4 * p)) & 0xF
        total += tl.sum(part.to(tl.float32), axis=1)
    tl.store(y_ptr + rows, total)


@triton.jit
def codes_pair(w_ptr, y_ptr, kw, KW: tl.constexpr, PER_WORD: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Take the codes of each word out in two masked halves rather than eight.

    Every other code of a word sits in one byte field of its own once the word
    is masked, so one mask yields the even codes and one shift and mask yields
    the odd ones; the fields are then separated by their byte position.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    total = tl.zeros((BLOCK_N,), dtype=tl.float32)
    byte = tl.arange(0, 4)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        even = word & 0x0F0F0F0F
        odd = (word >> 4) & 0x0F0F0F0F
        part = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
        for j in tl.static_range(4):
            part += (even >> (8 * j)) & 0xFF
            part += (odd >> (8 * j)) & 0xFF
        total += tl.sum(part, axis=1)
    tl.store(y_ptr + rows, total)


def timed(fn, repeat):
    torch.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    torch.mlu.synchronize()
    return (time.time() - start) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    print("device", torch.mlu.get_device_name(0))
    kw = IN_WIDTH // PER_WORD
    words = torch.randint(0, 2 ** 31 - 1, (OUT_WIDTH, kw),
                          dtype=torch.int32).to("mlu")
    y = torch.empty(OUT_WIDTH, dtype=torch.float32, device="mlu")
    weight_bytes = OUT_WIDTH * kw * 4

    configs = [(8, 512, 1), (16, 512, 1), (32, 512, 1),
               (8, 512, 4), (16, 512, 4),
               (4, 1024, 1), (8, 1024, 1), (16, 1024, 1)]
    print(f"  {'kernel':11} {'BN':>4} {'BW':>5} {'w':>2} {'ms':>9} {'GB/s':>8}")
    for bn, bw, warps in configs:
        grid = (OUT_WIDTH // bn,)
        for name, kernel in (("float", codes_float), ("whole", codes_whole),
                             ("pair", codes_pair)):
            call = lambda k=kernel, g=grid, bn=bn, bw=bw, w=warps: k[g](
                words, y, kw, KW=kw, PER_WORD=PER_WORD, BLOCK_N=bn, BLOCK_W=bw,
                num_warps=w)
            try:
                call()
                torch.mlu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:11} {bn:4d} {bw:5d} {warps:2d}  "
                      f"{type(exc).__name__}")
                continue
            elapsed = timed(call, args.repeat)
            print(f"  {name:11} {bn:4d} {bw:5d} {warps:2d} {elapsed * 1e3:9.4f} "
                  f"{weight_bytes / elapsed / 1e9:8.1f}")


if __name__ == "__main__":
    main()
