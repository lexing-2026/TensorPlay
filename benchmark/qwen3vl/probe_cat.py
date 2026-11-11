"""Measure how a step can join and append pieces without a concatenation.

Joining two tensors along the last axis appears throughout a block: the cache
append, the half-swap inside the rotation, the width doubling of the rotation
tables.  If that join is the expensive part of each, the joins have to be
replaced by writes into a buffer that already has the final shape.  This
measures the join beside those writes, at the shapes a step uses, and checks
that writing into a slice lands where it should.

    python probe_cat.py [--repeat 300]
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
    print(f"  {label:46} {(time.time() - start) / repeat * 1e6:9.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=300)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    small = tp.randn(1, 1, 64, device="mlu", dtype=tp.float16)
    out128 = tp.empty(1, 1, 128, device="mlu", dtype=tp.float16)

    print("joining along the last axis")
    bench("cat two 64-wide into 128", lambda: tp.cat([small, small], dim=-1), args.repeat)
    bench("write two halves into a ready buffer",
          lambda: (out128[..., :64].copy_(small), out128[..., 64:].copy_(small)),
          args.repeat)
    bench("copy_ the first half only", lambda: out128[..., :64].copy_(small), args.repeat)
    bench("allocate a 128-wide empty",
          lambda: tp.empty(1, 1, 128, device="mlu", dtype=tp.float16), args.repeat)
    bench("allocate and fill a 128-wide zero",
          lambda: tp.zeros(1, 1, 128, device="mlu", dtype=tp.float16), args.repeat)

    print("\nappending a step to a cache")
    keys = tp.randn(1, 8, 20, 128, device="mlu", dtype=tp.float16)
    step = tp.randn(1, 8, 1, 128, device="mlu", dtype=tp.float16)
    buffer = tp.zeros(1, 8, 64, 128, device="mlu", dtype=tp.float16)
    buffer[:, :, :20].copy_(keys)
    bench("cat a cache with one more step",
          lambda: tp.cat([keys, step], dim=2), args.repeat)
    bench("write the step into its slot",
          lambda: buffer[:, :, 20:21].copy_(step), args.repeat)
    bench("index_copy_ the step into its slot",
          lambda: tp.index_copy_(buffer, 2,
                                 tp.tensor([20], dtype=tp.int64, device="mlu"), step),
          args.repeat)

    print("\nthe rotation, joined against written")
    heads, kv_heads, head_dim = 32, 8, 128
    q = tp.randn(1, heads, 1, head_dim, device="mlu", dtype=tp.float16)
    k = tp.randn(1, kv_heads, 1, head_dim, device="mlu", dtype=tp.float16)
    cos = tp.randn(1, 1, head_dim // 2, device="mlu", dtype=tp.float16)
    sin = tp.randn(1, 1, head_dim // 2, device="mlu", dtype=tp.float16)

    from tensorplay.nn.modules.rotary import apply_rotary_emb

    bench("composed helper", lambda: apply_rotary_emb(q, k, cos, sin, False), args.repeat)

    def written(x):
        """Rotate one operand by writing both halves into its own buffer."""
        first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        result = tp.empty_like(x)
        result[..., : head_dim // 2].copy_(first * cos - second * sin)
        result[..., head_dim // 2 :].copy_(second * cos + first * sin)
        return result

    bench("written out for both operands",
          lambda: (written(q), written(k)), args.repeat)

    print("\nvalue check")
    reference_q, reference_k = apply_rotary_emb(q, k, cos, sin, False)
    got_q, got_k = written(q), written(k)
    print("  largest difference q:",
          float((got_q.float() - reference_q.float()).abs().max().cpu().numpy()))
    print("  largest difference k:",
          float((got_k.float() - reference_k.float()).abs().max().cpu().numpy()))

    print("\nvalue check for the write into a slot")
    probe = tp.zeros(1, 8, 64, 128, device="mlu", dtype=tp.float16)
    probe[:, :, 20:21].copy_(step)
    print("  slot 20 holds the step:",
          float((probe[:, :, 20:21].float() - step.float()).abs().max().cpu().numpy()))
    print("  slot 19 stays zero:",
          float(probe[:, :, 19:20].float().abs().max().cpu().numpy()))


if __name__ == "__main__":
    main()
