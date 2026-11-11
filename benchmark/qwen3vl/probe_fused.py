"""Look for single-launch kernels for the block's non-product pieces.

A block spends most of its time on operators that are not products, and each
of those is a sequence of small dependent launches.  The build carries native
kernels for two of those sequences — the head/normalization reduction and the
half-split rotation — reached only through guards that name particular
devices.  This checks whether this device is served by them, whether they
reproduce the composed result, and what they cost beside the composed form.
It also checks the preallocated cache, whose write is in place instead of a
concatenation per step.

    python probe_fused.py [--repeat 200]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
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
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    eps = 1e-6

    print("native single-launch normalizations")
    for rows, width in ((1, 4096), (32, 128), (8, 128)):
        x = tp.randn(1, rows, 1, width, device="mlu", dtype=tp.float16)
        w = tp.randn(width, device="mlu", dtype=tp.float16)
        composed = tp.nn.functional.rms_norm(x, [width], w, eps)
        try:
            fused, _ = tp._C._fused_rms_norm(x, [width], w, eps)
            error = float((fused.float() - composed.float()).abs().max().cpu().numpy())
            print(f"  rows={rows:3} width={width}: fused present, "
                  f"largest difference against the composed form {error:.6f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  rows={rows:3} width={width}: fused refused "
                  f"({type(exc).__name__}: {str(exc)[:70]})")
            continue
        bench(f"  rows={rows:3} width={width}: composed", 
              lambda x=x, w=w, width=width: tp.nn.functional.rms_norm(x, [width], w, eps),
              args.repeat)
        bench(f"  rows={rows:3} width={width}: native kernel",
              lambda x=x, w=w, width=width: tp._C._fused_rms_norm(x, [width], w, eps),
              args.repeat)

    print("\nnative single-launch half-split rotation")
    q = tp.randn(1, 32, 1, 128, device="mlu", dtype=tp.float16)
    k = tp.randn(1, 8, 1, 128, device="mlu", dtype=tp.float16)
    cos = tp.randn(1, 1, 64, device="mlu", dtype=tp.float16)
    sin = tp.randn(1, 1, 64, device="mlu", dtype=tp.float16)

    from tensorplay.nn.modules.rotary import apply_rotary_emb

    composed_q, composed_k = apply_rotary_emb(q, k, cos, sin, False)
    try:
        fused_q = tp._C._fused_rope_neox(q, cos, sin)
        fused_k = tp._C._fused_rope_neox(k, cos, sin)
        eq = float((fused_q.float() - composed_q.float()).abs().max().cpu().numpy())
        ek = float((fused_k.float() - composed_k.float()).abs().max().cpu().numpy())
        print(f"  fused present; largest difference q {eq:.6f} k {ek:.6f}")
        bench("  composed rotation", lambda: apply_rotary_emb(q, k, cos, sin, False),
              args.repeat)
        bench("  native kernel, both operands",
              lambda: (tp._C._fused_rope_neox(q, cos, sin),
                       tp._C._fused_rope_neox(k, cos, sin)), args.repeat)
    except Exception as exc:  # noqa: BLE001
        print(f"  fused refused ({type(exc).__name__}: {str(exc)[:70]})")
        bench("  composed rotation", lambda: apply_rotary_emb(q, k, cos, sin, False),
              args.repeat)

    print("\ncache write")
    from tensorplay.nn.cache import DynamicCache, StaticCache

    dynamic = DynamicCache()
    static = StaticCache(256)
    with tp.no_grad():
        for step in range(20):
            dynamic.update(q, k, 0)
            static.update(q, k, 0)
    tp.mlu.synchronize()
    bench("dynamic cache, append per step",
          lambda: dynamic.update(q, k, 0), args.repeat)
    bench("preallocated cache, write in place",
          lambda: static.update(q, k, 0), args.repeat)

    print("\nwhere the preallocated cache lands")
    static2 = StaticCache(256)
    with tp.no_grad():
        for step in range(20):
            key, value = static2.update(q, k, 0)
    tp.mlu.synchronize()
    want = float(q.reshape(-1)[0])
    got = float(key[:, :, 19, :].reshape(-1)[0])
    print(f"  written value read back at its slot: {got} (expected {want})")


if __name__ == "__main__":
    main()
