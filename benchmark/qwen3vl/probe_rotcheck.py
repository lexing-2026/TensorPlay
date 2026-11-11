"""Check the rotation and the normalization against their plain definitions.

Both the rotation and the written normalization are only useful if they agree
with the arithmetic they stand for.  A rotation written against the two halves
of a head leans on a slice being a view whose in-place store lands in the
buffer it came from; if the store lands somewhere else the result is silently
wrong, so the comparison is made here on the device a step actually runs on.

    python probe_rotcheck.py
"""

from __future__ import annotations

import torch  # noqa: F401
import torch_mlu  # noqa: F401

import tensorplay as tp


def rot_reference(x, cos, sin):
    """The rotation stated with a paired read, the widest way to write it.

    ``cos`` and ``sin`` arrive one half-head wide and are widened here by
    joining the head axis of the two contributions instead.
    """
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cosine, sine = cos.unsqueeze(2), sin.unsqueeze(2)
    return tp.cat([first * cosine - second * sine,
                   second * cosine + first * sine], dim=-1)


def rot_slices(x, cos, sin):
    """The rotation written against views of the halves, stored in place."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cosine, sine = cos.unsqueeze(2), sin.unsqueeze(2)
    out = tp.empty_like(x)
    out[..., :half].copy_(first * cosine - second * sine)
    out[..., half:].copy_(second * cosine + first * sine)
    return out


def rot_roll(x, cos, sin):
    """The rotation stated with a wrapped read of the whole head."""
    half = x.shape[-1] // 2
    cosine = tp.cat([cos, cos], dim=-1).unsqueeze(2)
    sine = tp.cat([-sin, sin], dim=-1).unsqueeze(2)
    return x * cosine + tp.roll(x, -half, -1) * sine


def report(label, got, want):
    difference = float((got.to(tp.float32) - want.to(tp.float32)).abs().max().item())
    print(f"  {label:28} max abs diff {difference:.6f}   {'ok' if difference < 1e-2 else 'WRONG'}")


def main():
    import tensorplay_mlu

    tensorplay_mlu.load()

    batch, length, heads, head_dim = 2, 5, 4, 8
    half = head_dim // 2
    x = tp.randn(batch, length, heads, head_dim, device="mlu", dtype=tp.float16)
    cos = tp.randn(batch, length, half, device="mlu", dtype=tp.float16)
    sin = tp.randn(batch, length, half, device="mlu", dtype=tp.float16)

    print("rotation, half-width factors")
    want = rot_reference(x, cos, sin)
    report("views stored in place", rot_slices(x, cos, sin), want)
    report("whole-head wrapped read", rot_roll(x, cos, sin), want)

    print("\nnormalization")
    weight = tp.randn(64, device="mlu", dtype=tp.float16)
    wide = tp.randn(3, 7, 64, device="mlu", dtype=tp.float16) * 4.0
    eps = 1e-6

    def written(body_x, body_w, body_eps, _shape):
        inv = ((body_x * body_x).mean(-1, keepdim=True) + body_eps).rsqrt()
        return (body_x * inv) * body_w

    want_norm = tp.nn.functional.rms_norm(wide, (64,), weight, eps)
    report("written, input's own type", written(wide, weight, eps, (64,)), want_norm)


if __name__ == "__main__":
    main()
