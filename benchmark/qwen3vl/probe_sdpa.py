"""Compare the framework's attention call with the arithmetic written out.

A decode step's attention covers a long cache with a single query, so its cost
should track the cache width, not the number of calls.  This times the
framework's single call beside the same arithmetic expressed directly, to see
whether the call is taking a path that does more work than the arithmetic.

    python probe_sdpa.py [--repeat 200]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def bench(label, fn, repeat):
    for _ in range(5):
        fn()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn()
    tp.mlu.synchronize()
    print(f"  {label:44} {(time.time() - start) / repeat * 1e3:9.4f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    heads, kv_heads, head_dim = 32, 8, 128
    scale = head_dim**-0.5

    for length in (20, 128, 512):
        print(f"\ncache length {length}")
        q = tp.randn(1, heads, 1, head_dim, device="mlu", dtype=tp.float16)
        k = tp.randn(1, kv_heads, length, head_dim, device="mlu", dtype=tp.float16)
        v = tp.randn(1, kv_heads, length, head_dim, device="mlu", dtype=tp.float16)

        bench(
            "framework scaled_dot_product_attention",
            lambda: tp.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True
            ),
            args.repeat,
        )
        k_big = k.repeat_interleave(heads // kv_heads, dim=1)
        v_big = v.repeat_interleave(heads // kv_heads, dim=1)
        bench(
            "framework, grouping written out",
            lambda: tp.nn.functional.scaled_dot_product_attention(
                q, k_big, v_big, dropout_p=0.0, is_causal=False, scale=scale,
                enable_gqa=False
            ),
            args.repeat,
        )
        bench("score product only", lambda: q @ k_big.transpose(-1, -2), args.repeat)
        bench("softmax only", lambda: tp.nn.functional.softmax(
            q @ k_big.transpose(-1, -2), dim=-1), args.repeat)

        def written_out():
            scores = (q @ k_big.transpose(-1, -2)) * scale
            weights = tp.nn.functional.softmax(scores, dim=-1)
            return weights @ v_big

        bench("written out: product, softmax, product", written_out, args.repeat)


if __name__ == "__main__":
    main()
