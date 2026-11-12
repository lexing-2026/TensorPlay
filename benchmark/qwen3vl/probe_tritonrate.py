"""What rate the accelerator's own kernel language reaches on this card.

A weight-only product only reads its weights, so the number that matters is
how fast the language reads, not how fast it copies.  Three kernels state it:
a plain copy reads and writes, a sum-reduce only reads, and a half-precision
product of the shapes a projection uses reads its weights under a real
reduction.  Each is swept over the block shape and the number of stages so the
best rate the language can reach is what the sweep reports, not the first
guess.

    python probe_tritonrate.py [--repeat 20] [--mib 256]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch_mlu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=mask), mask=mask)


@triton.jit
def sum_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """Read a slice of the buffer per program and reduce it; read traffic only."""
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i0 in range(pid * BLOCK, n, BLOCK * 8):
        acc += tl.load(x_ptr + i0 + tl.arange(0, BLOCK)).to(tl.float32)
    tl.store(y_ptr + pid, tl.sum(acc))


@triton.jit
def gemv_fp16(x_ptr, w_ptr, y_ptr, K: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        xk = tl.load(x_ptr + ks).to(tl.float32)
        w = tl.load(w_ptr + rows[:, None] * K + ks[None, :]).to(tl.float32)
        acc += tl.sum(w * xk[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16))


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
    ap.add_argument("--mib", type=int, default=256)
    args = ap.parse_args()

    print("device", torch.mlu.get_device_name(0))
    n = args.mib * 1024 * 1024 // 2
    x = torch.randn(n, dtype=torch.float16, device="mlu")
    y = torch.empty_like(x)

    # What the library's own copy reaches, as the device's side of the story.
    elapsed = timed(lambda: y.copy_(x), args.repeat)
    print(f"  library copy {args.mib} MiB: {elapsed * 1e3:8.3f} ms  "
          f"{n * 4 / elapsed / 1e9:8.1f} GB/s (read+write)")

    print(f"  {'kernel':8} {'BLOCK':>7} {'warps':>5} {'stages':>6} {'ms':>9} {'GB/s':>8}")
    small = torch.empty(65536, dtype=torch.float32, device="mlu")
    for block in (1024, 2048, 4096, 8192, 16384):
        for warps in (1, 2, 4, 8):
            grid = (triton.cdiv(n, block),)
            try:
                copy_kernel[grid](x, y, n, BLOCK=block, num_warps=warps)
                torch.mlu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print(f"  copy {block:7d} {warps:5d}    --  {type(exc).__name__}")
                continue
            elapsed = timed(lambda: copy_kernel[grid](x, y, n, BLOCK=block,
                                                      num_warps=warps), args.repeat)
            print(f"  copy {block:7d} {warps:5d}    --  {elapsed * 1e3:8.3f} "
                  f"{n * 4 / elapsed / 1e9:8.1f}")

    # Read-only: the traffic a weight-only product actually causes.
    for block in (1024, 2048, 4096, 8192):
        for warps in (2, 4, 8):
            grid = (triton.cdiv(n, block * 8),)
            try:
                read_kernel[grid](x, small, n, BLOCK=block, num_warps=warps)
                torch.mlu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print(f"  read {block:7d} {warps:5d}    --  {type(exc).__name__}")
                continue
            elapsed = timed(lambda: read_kernel[grid](x, small, n, BLOCK=block,
                                                      num_warps=warps), args.repeat)
            print(f"  read {block:7d} {warps:5d}    --  {elapsed * 1e3:8.3f} "
                  f"{n * 2 / elapsed / 1e9:8.1f}")

    for block_n in (64, 128):
        for block_k in (256, 512, 1024):
            for stages in (1, 2, 3, 4):
                k = 4096
                out = 4096
                w = torch.randn(out, k, dtype=torch.float16, device="mlu")
                xk = torch.randn(k, dtype=torch.float16, device="mlu")
                yk = torch.empty(out, dtype=torch.float16, device="mlu")
                g = (out // block_n,)
                try:
                    gemv_fp16[g](xk, w, yk, K=k, BLOCK_N=block_n, BLOCK_K=block_k,
                                 num_warps=4, num_stages=stages)
                    torch.mlu.synchronize()
                except Exception as exc:  # noqa: BLE001
                    print(f"  gemv {block_n:4d} {block_k:5d}    {stages:2d}  "
                          f"{type(exc).__name__}")
                    continue
                elapsed = timed(
                    lambda: gemv_fp16[g](xk, w, yk, K=k, BLOCK_N=block_n,
                                         BLOCK_K=block_k, num_warps=4,
                                         num_stages=stages), args.repeat)
                print(f"  gemv {block_n:4d} {block_k:5d}    {stages:2d}  "
                      f"{elapsed * 1e3:8.4f} {out * k * 2 / elapsed / 1e9:8.1f}")


if __name__ == "__main__":
    main()
