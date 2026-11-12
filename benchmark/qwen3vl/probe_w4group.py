"""Fused four-bit weight-only product over the checkpoint's own group layout.

The stored weights keep eight four-bit codes in every 32-bit word, little
field first, with one factor per group of 32 inputs and no zero point.  The
product below reads that layout as it sits: the codes are shifted out of the
word inside the reduction, so the weight is never widened, and each word is
read once per output row.

The reduction is written so that everything but the weights is read as a
contiguous run: a tile of words covers a contiguous run of the activation, and
the eight codes of a word meet the eight activations they contract through a
reshape rather than through eight strided reads.  Reads of the activation run
are what hold a weight-only product back, because they are small and repeated
while the weight is large and read once.

The result is checked against the values the codes stand for and timed at the
shapes a decode step uses.

    python probe_w4group.py [--repeat 50]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl

# (name, output width, contracted width) of one tower's projections.
SHAPES = [
    ("q", 4096, 4096),
    ("k", 1024, 4096),
    ("v", 1024, 4096),
    ("o", 4096, 4096),
    ("gate", 12288, 4096),
    ("up", 12288, 4096),
    ("down", 4096, 12288),
]
GROUP = 32
BITS = 4
# Four-bit codes in one 32-bit word, and words sharing one factor.
PER_WORD = 8
WORDS_PER_GROUP = GROUP // PER_WORD


@triton.jit
def w4a16_gemv(x_ptr, w_ptr, s_ptr, y_ptr,
               KW: tl.constexpr, G: tl.constexpr,
               PER_WORD: tl.constexpr, WORDS_PER_GROUP: tl.constexpr,
               BITS: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """One block of output rows, reduced over the packed words.

    A tile of ``BLOCK_W`` words holds ``BLOCK_W * PER_WORD`` activities, which
    is read as one contiguous run.  The codes of each word are shifted out
    against the activation run through a reshape of that run, and the half
    range every code is offset by is taken out before the factor is applied,
    so only the sums the factor multiplies are formed.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    shift = tl.arange(0, PER_WORD) * BITS
    inner = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, KW, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        codes = ((word[:, :, None] >> shift[None, None, :]) & 0xF).to(tl.float32) - 8.0
        run = tl.load(x_ptr + w0 * PER_WORD + tl.arange(0, BLOCK_W * PER_WORD))
        x3 = tl.reshape(run.to(tl.float32), (BLOCK_W, PER_WORD))
        local = tl.sum(codes * x3[None, :, :], axis=2)
        scale = tl.load(s_ptr + rows[:, None] * G + (ws // WORDS_PER_GROUP)[None, :])
        inner += tl.sum(local * scale.to(tl.float32), axis=1)
    tl.store(y_ptr + rows, inner.to(y_ptr.dtype.element_ty))


def build(out_width, in_width, device="mlu"):
    """A synthetic group-32 weight in the stored layout, and its dense form."""
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1 << BITS, size=(out_width, in_width), dtype=np.int32)
    words = np.zeros((out_width, in_width // PER_WORD), dtype=np.int32)
    for p in range(PER_WORD):
        words |= (codes[:, p::PER_WORD] & 0xF) << (4 * p)
    scale = (rng.random((out_width, in_width // GROUP), dtype=np.float32) * 0.05 + 0.01)
    dense = (codes - 8).astype(np.float32) * np.repeat(scale, GROUP, axis=1)

    x = torch.randn(1, in_width, dtype=torch.float16, device=device)
    w = torch.from_numpy(words).to(device=device)
    s = torch.from_numpy(scale).to(torch.float16).to(device=device)
    ref = torch.from_numpy(dense).to(torch.float16).to(device=device)
    return x, w, s, ref


def launch(x, w, s, out_width, in_width, block_n, block_w, warps):
    y = torch.empty(out_width, dtype=torch.float16, device=x.device)
    grid = (out_width // block_n,)
    w4a16_gemv[grid](x, w, s, y, KW=in_width // PER_WORD, G=in_width // GROUP,
                     PER_WORD=PER_WORD, WORDS_PER_GROUP=WORDS_PER_GROUP,
                     BITS=BITS,
                     BLOCK_N=block_n, BLOCK_W=block_w, num_warps=warps)
    return y


def timed(fn, repeat):
    torch.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    torch.mlu.synchronize()
    return (time.time() - start) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--block-w", type=int, default=32)
    ap.add_argument("--warps", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    print("device", torch.mlu.get_device_name(0))
    configs = [(args.block_n, args.block_w, args.warps)]
    shapes = SHAPES
    repeat = args.repeat
    if args.sweep:
        # Rank the tiles on two shapes rather than compiling every tile for
        # every shape; the ranking is what the sweep is for.
        configs = [(bn, bw, wp) for bn in (16, 32, 64)
                   for bw in (16, 32, 64) for wp in (1, 4)]
        shapes = SHAPES[:2]
        repeat = min(repeat, 30)

    for block_n, block_w, warps in configs:
        total = 0.0
        rel = 0.0
        for name, out_width, in_width in shapes:
            x, w, s, ref = build(out_width, in_width)
            y = launch(x, w, s, out_width, in_width, block_n, block_w, warps)
            torch.mlu.synchronize()
            reference = (x @ ref.t()).reshape(-1).float()
            rel = max(rel, (y.float() - reference).abs().max().item()
                      / reference.abs().max().item())
            total += timed(
                lambda: launch(x, w, s, out_width, in_width, block_n, block_w, warps),
                repeat)
        print(f"  BN={block_n:3d} BW={block_w:3d} warps={warps}  rel={rel:8.2e}  "
              f"layer {total * 1e3:8.3f} ms  x36 = {total * 36e3:7.2f} ms")


if __name__ == "__main__":
    main()
