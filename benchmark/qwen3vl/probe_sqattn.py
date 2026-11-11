"""Check the accelerator's fused single-query cached attention.

A decode step attends one query against the whole cache.  The framework
composes that from a score product, a softmax and a value product; the
accelerator offers it as one call over a linear cache.  This checks the call
reproduces the composed result at the widths and head grouping a decode step
uses, and reports what it costs as the context grows.

    python probe_sqattn.py [--repeat 50]
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    import torch
    import torch_mlu_ops

    rng = np.random.default_rng(0)
    heads, kv_heads, head_size = 32, 8, 128
    scale = 1.0 / np.sqrt(head_size)

    def reference(q, k, v, length):
        group = heads // kv_heads
        out = np.zeros((1, 1, heads, head_size), dtype=np.float32)
        for h in range(heads):
            scores = (q[0, 0, h] @ k[0, h // group, :length].T) * scale
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            out[0, 0, h] = weights @ v[0, h // group, :length]
        return out

    print("correctness (non-paged)")
    for length in (1, 20, 33, 128):
        q = rng.standard_normal((1, 1, heads, head_size)).astype(np.float16)
        capacity = max(length, 512)
        k = rng.standard_normal((1, kv_heads, capacity, head_size)).astype(np.float16)
        v = rng.standard_normal((1, kv_heads, capacity, head_size)).astype(np.float16)
        want = reference(q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), length)

        got = torch_mlu_ops.single_query_cached_kv_attn(
            torch.from_numpy(q).mlu(),
            torch.from_numpy(k).mlu(),
            torch.from_numpy(v).mlu(),
            None,
            torch.zeros((1, 1), dtype=torch.int32).mlu(),
            torch.tensor([length], dtype=torch.int32).mlu(),
            None, None, None,
            capacity, -1, -1, scale,
        ).float().cpu().numpy()
        error = float(np.abs(got - want).max())
        print(f"  length={length:4}  max abs {error:.5f}  "
              f"{'MATCH' if error < 5e-2 else 'wrong'}")

    print("\ncost as the context grows (seq_q=1, 32/8 heads, 128 wide)")
    for length in (20, 128, 512, 2048, 4096):
        q = torch.randn(1, 1, heads, head_size, device="mlu", dtype=torch.float16)
        k = torch.randn(1, kv_heads, length, head_size, device="mlu", dtype=torch.float16)
        v = torch.randn(1, kv_heads, length, head_size, device="mlu", dtype=torch.float16)
        blocks = torch.zeros((1, 1), dtype=torch.int32, device="mlu")
        lens = torch.tensor([length], dtype=torch.int32, device="mlu")

        def call():
            return torch_mlu_ops.single_query_cached_kv_attn(
                q, k, v, None, blocks, lens, None, None, None,
                length, -1, -1, scale,
            )

        for _ in range(3):
            call()
        torch.mlu.synchronize()
        start = time.time()
        for _ in range(args.repeat):
            call()
        torch.mlu.synchronize()
        print(f"  length={length:5}  {(time.time() - start) / args.repeat * 1e3:8.4f} ms")
        del k, v


if __name__ == "__main__":
    main()
