"""Where the four-bit product spends its time.

The product streams only its weights, so it should run at the rate a plain read
of the same bytes reaches, and a read that reduces the words is that rate.  It
does not, and the loss must be one of two things: taking the codes out of the
packed words, or the reduction that turns a tile of codes into one output per
row.

Each step is timed on its own.  The words are read and reduced, then the codes
are taken out and reduced both one position at a time and all at once, then the
codes are taken out through the bytes of each word instead of its fields, then
the codes are contracted with the inputs.  Whichever step collapses points at
the cost; a step that does not collapse is not the one to spend effort on.

The problem is large enough that the device, not the launch, is what the clock
measures.

    python probe_w4extract.py [--repeat 30]
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
def words_only(w_ptr, y_ptr, kw, KW: tl.constexpr, PER_WORD: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Read the packed words and reduce them; the words are all that is read."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        acc += tl.sum(word.to(tl.float32), axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def codes_sums(w_ptr, y_ptr, kw, KW: tl.constexpr, PER_WORD: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Take each code out in turn and reduce it before the next is taken."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        for p in tl.static_range(PER_WORD):
            acc += tl.sum(((word >> (4 * p)) & 0xF).to(tl.float32), axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def codes_tile(w_ptr, y_ptr, kw, KW: tl.constexpr, PER_WORD: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Take every code out into one tile and reduce the tile once."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        part = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
        for p in tl.static_range(PER_WORD):
            part += ((word >> (4 * p)) & 0xF).to(tl.float32)
        acc += tl.sum(part, axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def codes_bytes(w8_ptr, y_ptr, kw4, KW4: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Read the words as bytes and take both of a byte's codes out at once."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for c0 in range(0, kw4, BLOCK_W):
        cs = c0 + tl.arange(0, BLOCK_W)
        byte = tl.load(w8_ptr + rows[:, None] * KW4 + cs[None, :]).to(tl.float32)
        high = tl.floor(byte * 0.0625)
        acc += tl.sum((byte - 16.0 * high) + high, axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def product_tile(w_ptr, x_ptr, y_ptr, kw, KW: tl.constexpr,
                 PER_WORD: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_W: tl.constexpr):
    """Take the codes out, contract them with the inputs, and reduce once.

    The inputs are laid out one code position per row, so the run each code
    position reads is contiguous.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        part = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
        for p in tl.static_range(PER_WORD):
            code = ((word >> (4 * p)) & 0xF).to(tl.float32) - 8.0
            xp = tl.load(x_ptr + p * kw + ws).to(tl.float32)
            part += code * xp[None, :]
        acc += tl.sum(part, axis=1)
    tl.store(y_ptr + rows, acc)


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
    kw = IN_WIDTH // PER_WORD
    words_np = np.random.default_rng(0).integers(
        0, 2 ** 31 - 1, (OUT_WIDTH, kw), dtype=np.int32)
    words = torch.from_numpy(words_np).to("mlu")
    w8 = torch.from_numpy(
        words_np.view(np.uint8).reshape(OUT_WIDTH, kw * 4)).to("mlu")
    x = torch.randn(IN_WIDTH, dtype=torch.float16, device="mlu")
    # One code position per row of the input, so a position's run is contiguous.
    xt = torch.zeros(PER_WORD, kw, dtype=torch.float16, device="mlu")
    xt.copy_(x.reshape(kw, PER_WORD).t())
    y = torch.empty(OUT_WIDTH, dtype=torch.float32, device="mlu")
    weight_bytes = OUT_WIDTH * kw * 4

    # The contraction against the values the stored codes stand for.  The codes
    # are symmetric about half range, so the half range is taken back out.
    plain = np.empty((OUT_WIDTH, IN_WIDTH), dtype=np.float32)
    for p in range(PER_WORD):
        plain[:, p::PER_WORD] = (words_np >> (4 * p)) & 0xF
    plain -= 8.0
    ref = (torch.from_numpy(plain).to("mlu").half() @ x).float()
    del plain

    product_tile[(OUT_WIDTH // 32,)](words, xt, y, kw, KW=kw,
                                     PER_WORD=PER_WORD, BLOCK_N=32, BLOCK_W=128,
                                     num_warps=1)
    torch.mlu.synchronize()
    rel = (y - ref).abs().max().item() / ref.abs().max().item()
    print(f"  contraction relative error {rel:.2e}")

    print(f"  {'kernel':11} {'BN':>4} {'BW':>4} {'ms':>9} {'GB/s':>8}")
    for bn in (32, 64):
        grid = (OUT_WIDTH // bn,)
        for bw in (128, 256):
            runs = [
                ("words", lambda: words_only[grid](
                    words, y, kw, KW=kw, PER_WORD=PER_WORD, BLOCK_N=bn,
                    BLOCK_W=bw, num_warps=1)),
                ("codes_sums", lambda: codes_sums[grid](
                    words, y, kw, KW=kw, PER_WORD=PER_WORD, BLOCK_N=bn,
                    BLOCK_W=bw, num_warps=1)),
                ("codes_tile", lambda: codes_tile[grid](
                    words, y, kw, KW=kw, PER_WORD=PER_WORD, BLOCK_N=bn,
                    BLOCK_W=bw, num_warps=1)),
                ("codes_bytes", lambda: codes_bytes[grid](
                    w8, y, kw * 4, KW4=kw * 4, BLOCK_N=bn, BLOCK_W=bw * 4,
                    num_warps=1)),
                ("product_tile", lambda: product_tile[grid](
                    words, xt, y, kw, KW=kw, PER_WORD=PER_WORD, BLOCK_N=bn,
                    BLOCK_W=bw, num_warps=1)),
            ]
            for name, call in runs:
                try:
                    call()
                    torch.mlu.synchronize()
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name:11} {bn:4d} {bw:4d}  {type(exc).__name__}")
                    continue
                elapsed = timed(call, args.repeat)
                print(f"  {name:11} {bn:4d} {bw:4d} {elapsed * 1e3:9.4f} "
                      f"{weight_bytes / elapsed / 1e9:8.1f}")


if __name__ == "__main__":
    main()
