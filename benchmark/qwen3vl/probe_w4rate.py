"""Where the four-bit product loses its rate.

The product only streams weights, so it should run at the rate a plain read of
the same bytes reaches.  It does not, and the loss can come from three places:
what a launch costs before any work happens, what reading the packed words
costs, and what taking the codes out of the words costs.  Each is measured on
its own, at the width of the widest projection, so the one that dominates can
be named.

Two shapes of program are tried.  One holds a block of output rows and reads a
stripe of every one of them; the other holds a single row and reads it whole,
so every read is a contiguous run.

    python probe_w4rate.py [--repeat 50]

    python probe_w4rate.py --sweep
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def noop_kernel(y_ptr):
    tl.store(y_ptr, 1.0)


@triton.jit
def words_stream(w_ptr, y_ptr, kw, KW: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Read the packed words and reduce them; no code is taken out."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        acc += tl.sum(word.to(tl.float32), axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def words_unpacked(x_ptr, w_ptr, y_ptr, kw, KW: tl.constexpr,
                   PER_WORD: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Read the words and take the codes out against a prepared activation.

    The activation is prepared once per step into the order the codes are
    stored in, so each of the eight fields of a word reads a contiguous run.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        partial = tl.zeros((BLOCK_N, BLOCK_W), dtype=tl.float32)
        for p in tl.static_range(PER_WORD):
            nib = ((word >> (4 * p)) & 0xF).to(tl.float32) - 8.0
            xp = tl.load(x_ptr + p * KW + ws)
            partial += nib * xp[None, :]
        acc += tl.sum(partial, axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def words_reshaped(x_ptr, w_ptr, y_ptr, kw, KW: tl.constexpr,
                   PER_WORD: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Read the words and take the codes out through a reshape of the run.

    The activation run is read once per tile, whole, and the codes meet it
    through a view of that run rather than through a read of their own.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    shift = tl.arange(0, PER_WORD) * 4
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        codes = ((word[:, :, None] >> shift[None, None, :]) & 0xF).to(tl.float32) - 8.0
        run = tl.load(x_ptr + w0 * PER_WORD + tl.arange(0, BLOCK_W * PER_WORD))
        x3 = tl.reshape(run.to(tl.float32), (BLOCK_W, PER_WORD))
        acc += tl.sum(tl.sum(codes * x3[None, :, :], axis=2), axis=1)
    tl.store(y_ptr + rows, acc)


@triton.jit
def words_nocvt(w_ptr, y_ptr, kw, KW: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """Read the packed words and reduce them while they stay whole numbers."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(w_ptr + rows[:, None] * KW + ws[None, :])
        acc += tl.sum(word, axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float32))


@triton.jit
def row_stream(w_ptr, y_ptr, kw, BLOCK_W: tl.constexpr):
    """One output row per program, read whole; the words are the only read."""
    pid = tl.program_id(0)
    base = w_ptr + pid * kw
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        acc += tl.load(base + w0 + tl.arange(0, BLOCK_W)).to(tl.float32)
    tl.store(y_ptr + pid, tl.sum(acc))


@triton.jit
def row_codes(x_ptr, w_ptr, y_ptr, kw, PER_WORD: tl.constexpr,
              BLOCK_W: tl.constexpr):
    """One output row per program, with the codes taken out of the words."""
    pid = tl.program_id(0)
    base = w_ptr + pid * kw
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for w0 in range(0, kw, BLOCK_W):
        ws = w0 + tl.arange(0, BLOCK_W)
        word = tl.load(base + ws)
        partial = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for p in tl.static_range(PER_WORD):
            nib = ((word >> (4 * p)) & 0xF).to(tl.float32) - 8.0
            partial += nib * tl.load(x_ptr + p * kw + ws)
        acc += partial
    tl.store(y_ptr + pid, tl.sum(acc))


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
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    print("device", torch.mlu.get_device_name(0))
    per_word = 8
    out_width, in_width = 4096, 4096
    kw = in_width // per_word

    # The narrow integer kinds have no generator on the device.
    words = torch.randint(0, 2 ** 31 - 1, (out_width, kw),
                          dtype=torch.int32).to("mlu")
    x = torch.randn(in_width, dtype=torch.float16, device="mlu")
    # The activation in the order the codes are stored: field p of word w.
    xt = torch.zeros(per_word, kw, dtype=torch.float16, device="mlu")
    xt.copy_(x.reshape(kw, per_word).t())
    y = torch.empty(out_width, dtype=torch.float32, device="mlu")

    weight_bytes = out_width * kw * 4
    tiny = torch.empty(1, dtype=torch.float32, device="mlu")
    for n in (1, 64, 512, 4096):
        elapsed = timed(lambda: noop_kernel[(n,)](tiny), args.repeat)
        print(f"  launch grid={n:5d}   {elapsed * 1e6:8.2f} us")

    stripes = [(32, 64, 1), (64, 64, 1), (64, 128, 1), (64, 128, 3), (128, 128, 1)]
    rowses = [(128, 1), (256, 1), (512, 1), (1024, 1)]
    if not args.sweep:
        stripes = stripes[:2]
        rowses = rowses[:2]

    def stripe_call(kernel, bn, bw, st, **extra):
        grid = (out_width // bn,)

        def call():
            kernel[grid](words, y, kw, KW=kw, BLOCK_N=bn, BLOCK_W=bw,
                         num_warps=1, num_stages=st, **extra)
        return call

    def stripe_call_act(kernel, bn, bw, st, act):
        grid = (out_width // bn,)

        def call():
            kernel[grid](act, words, y, kw, KW=kw, PER_WORD=per_word, BLOCK_N=bn,
                         BLOCK_W=bw, num_warps=1, num_stages=st)
        return call

    variants = [
        ("stream", lambda bn, bw, st: stripe_call(words_stream, bn, bw, st)),
        ("nocvt", lambda bn, bw, st: stripe_call(words_nocvt, bn, bw, st)),
        ("unpack", lambda bn, bw, st: stripe_call_act(words_unpacked, bn, bw, st, xt)),
        ("reshape", lambda bn, bw, st: stripe_call_act(words_reshaped, bn, bw, st, x)),
    ]
    print(f"  {'program':8} {'BN':>4} {'BW':>5} {'st':>3}  " + "  ".join(
        f"{n:>18}" for n, _ in variants))
    for bn, bw, st in stripes:
        cells = []
        for _, make in variants:
            call = make(bn, bw, st)
            try:
                call()
                torch.mlu.synchronize()
            except Exception as exc:  # noqa: BLE001
                cells.append(f"{type(exc).__name__:>18}")
                continue
            elapsed = timed(call, args.repeat)
            cells.append(f"{elapsed * 1e6:8.1f}us{weight_bytes / elapsed / 1e9:5.0f}G")
        print(f"  {'stripe':8} {bn:4d} {bw:5d} {st:3d}  " + "  ".join(cells))

    print(f"  {'program':8} {'BN':>4} {'BW':>5} {'st':>3}  " + "  ".join(
        f"{n:>18}" for n in ("stream", "codes")))
    for bw, st in rowses:
        grid = (out_width,)
        cells = []
        for kernel, act in ((row_stream, None), (row_codes, xt)):
            def call(kernel=kernel, act=act, bw=bw, st=st):
                if act is None:
                    kernel[grid](words, y, kw, BLOCK_W=bw, num_warps=1,
                                 num_stages=st)
                else:
                    kernel[grid](act, words, y, kw, PER_WORD=per_word, BLOCK_W=bw,
                                 num_warps=1, num_stages=st)
            try:
                call()
                torch.mlu.synchronize()
            except Exception as exc:  # noqa: BLE001
                cells.append(f"{type(exc).__name__:>18}")
                continue
            elapsed = timed(call, args.repeat)
            cells.append(f"{elapsed * 1e6:8.1f}us{weight_bytes / elapsed / 1e9:5.0f}G")
        print(f"  {'row':8} {'-':>4} {bw:5d} {st:3d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
