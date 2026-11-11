"""Compare small operators as written calls against a compiled region.

The pieces of a step that are not products are chains of small elementwise
calls, and each call costs its own dispatch.  A compiled region lowers a chain
into fewer launches, so the same arithmetic is timed both ways at the shapes a
step uses.

    python probe_compile.py [--repeat 200]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def timed(run, repeat):
    for _ in range(5):
        run()
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        run()
    tp.mlu.synchronize()
    return (time.time() - start) / repeat * 1e6


def compare(label, eager, compiled, repeat, note=""):
    eager_us = timed(eager, repeat)
    compiled_us = timed(compiled, repeat)
    print(f"  {label:34} eager {eager_us:8.1f} us   compiled {compiled_us:8.1f} us"
          f"   {eager_us / compiled_us:5.2f}x   {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--backends", default="magicmind,stax")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    row = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    wide = tp.randn(1, 1, 12288, device="mlu", dtype=tp.float16)
    q = tp.randn(1, 1, 32, 128, device="mlu", dtype=tp.float16)
    k = tp.randn(1, 1, 8, 128, device="mlu", dtype=tp.float16)
    weight = tp.randn(4096, device="mlu", dtype=tp.float16)
    q_weight = tp.randn(128, device="mlu", dtype=tp.float16)
    k_weight = tp.randn(128, device="mlu", dtype=tp.float16)

    cos = tp.randn(1, 1, 1, 64, device="mlu", dtype=tp.float16)
    sin = tp.randn(1, 1, 1, 64, device="mlu", dtype=tp.float16)

    try:
        from tensorplay.compiler import list_backends

        print("available backends")
        for name in list_backends():
            print(f"  {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  backend listing failed: {type(exc).__name__}: {exc}")

    for backend in args.backends.split(","):
        print(f"\nbackend {backend}")
        try:
            run_backend(backend, row, wide, q, k, weight, wide, q_weight, k_weight,
                        cos, sin, args.repeat)
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {type(exc).__name__}: {str(exc)[:160]}")


def run_backend(backend, row, wide, q, k, weight, wide_weight, q_weight, k_weight,
                cos, sin, repeat):
        def eager_norm(x, w):
            compute = tp.float32 if x.dtype in (tp.float16, tp.bfloat16) else x.dtype
            inv = ((x.to(compute) * x.to(compute)).mean(-1, keepdim=True) + 1e-6).rsqrt()
            return (x * inv.to(x.dtype)) * w

        def plain_norm(x, w):
            inv = ((x * x).mean(-1, keepdim=True) + 1e-6).rsqrt()
            return x * inv * w

        def eager_silu_mul(gate, up):
            return tp.nn.functional.silu(gate) * up

        def eager_rotate(x, c, s):
            half = x.shape[-1] // 2
            first, second = x[..., :half], x[..., half:]
            out = tp.empty_like(x)
            out[..., :half].copy_(first * c - second * s)
            out[..., half:].copy_(second * c + first * s)
            return out

        compiled_norm = tp.compile(eager_norm, backend=backend)
        compiled_norm_plain = tp.compile(plain_norm, backend=backend)
        compiled_silu_mul = tp.compile(eager_silu_mul, backend=backend)
        compiled_rotate = tp.compile(eager_rotate, backend=backend)

        # Warm the guards with the shapes they will see.
        expected = eager_norm(row, weight)
        got = compiled_norm(row, weight)
        print(f"  norm largest difference against the written form "
              f"{(got - expected).abs().max().item():.3e}")

        compare("normalization, 4096 row",
                lambda: eager_norm(row, weight),
                lambda: compiled_norm(row, weight), repeat)
        compare("normalization, reduced statistic",
                lambda: plain_norm(row, weight),
                lambda: compiled_norm_plain(row, weight), repeat)
        compare("normalization, 32 heads of 128",
                lambda: eager_norm(q, q_weight),
                lambda: compiled_norm(q, q_weight), repeat)
        compare("gated activation, 12288 wide",
                lambda: eager_silu_mul(wide, wide),
                lambda: compiled_silu_mul(wide, wide), repeat)
        compare("rotation, 32 heads",
                lambda: eager_rotate(q, cos, sin),
                lambda: compiled_rotate(q, cos, sin), repeat)
        compare("rotation, 8 heads",
                lambda: eager_rotate(k, cos, sin),
                lambda: compiled_rotate(k, cos, sin), repeat)

        def both(x, c, s):
            return eager_rotate(x, c, s), eager_rotate(x, c, s)

        compiled_both = tp.compile(both, backend=backend)
        compare("rotation twice",
                lambda: both(q, cos, sin),
                lambda: compiled_both(q, cos, sin), repeat)

        # Everything a block does that is not a product, in one region.
        def block_extras(x, wide, q, k, w_in, w_post, w_q, w_k, c, s):
            h = eager_norm(x, w_in)
            qq = eager_norm(q, w_q)
            kk = eager_norm(k, w_k)
            rotated_q = eager_rotate(qq, c, s)
            rotated_k = eager_rotate(kk, c, s)
            hidden = eager_norm(x, w_post)
            gated = eager_silu_mul(hidden, hidden)
            scores = (rotated_q.reshape(1, 32, 1, 128).transpose(1, 2)
                      @ rotated_k.reshape(1, 8, 1, 128).transpose(1, 2).transpose(-1, -2))
            weights_ = tp.nn.functional.softmax(scores, dim=-1)
            attn = weights_ @ rotated_k.reshape(1, 8, 1, 128).transpose(1, 2)
            return h.sum() + gated.sum() + attn.sum() + rotated_k.sum()

        compiled_extras = tp.compile(block_extras, backend=backend)
        extras_args = (row, wide, q, k, weight, weight, q_weight, k_weight, cos, sin)
        compare("a block's non-product work",
                lambda: block_extras(*extras_args),
                lambda: compiled_extras(*extras_args), repeat)


if __name__ == "__main__":
    main()
