"""Measure whole decode steps for several ways of writing the same model.

One model load costs minutes, so every variant is timed inside a single
process: the shapes stay resident and only the body of the step changes.
Each variant reports the median step and how much of it was spent handing the
calls to the device and how much draining them.

The device has a kernel for the operators it registers; everything else is
answered on the host, and a host answer cannot proceed until the device has
drained, so a step built out of unregistered operators is bounded by the host
and not by the accelerator.  The variants below are therefore chosen to differ
in which operators they reach, not only in how many: a rotation written with a
roll and a fused multiply-add reaches two operators the device does not carry,
while the same rotation written against the two halves reaches only operators it
does.

    python probe_step.py --model <dir> --steps 12
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402

import qwen3vl.model as qmodel  # noqa: E402
from qwen3vl.model import cache_dense_weights, load_qwen3vl  # noqa: E402


def factors_full(cos, sin):
    """Widen the table so both out halves can be scaled in one elementwise pass."""
    negated = -sin
    return (
        tp.cat([cos, cos], dim=-1).unsqueeze(2),
        tp.cat([negated, sin], dim=-1).unsqueeze(2),
    )


def factors_half(cos, sin):
    """Leave the table at half width, to scale each half on its own."""
    return cos.unsqueeze(2), sin.unsqueeze(2)


def factors_none(cos, sin):
    """No table at all: the rotation is skipped, bounding what it costs."""
    return None, None


def rotate_roll_addcmul(x, cos, sin):
    half = x.shape[-1] // 2
    return tp.addcmul(x * cos, tp.roll(x, -half, -1), sin)


def rotate_roll_mul(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + tp.roll(x, -half, -1) * sin


def rotate_roll_in_place(x, cos, sin):
    """Write the rotation over the projection's own output."""
    half = x.shape[-1] // 2
    rolled = tp.roll(x, -half, -1)
    rolled.mul_(sin)
    x.mul_(cos)
    x.add_(rolled)
    return x


def rotate_index_select(x, cos, sin):
    """Reach the paired element with the device's own taking of an index."""
    global _PERMUTATION
    half = x.shape[-1] // 2
    if _PERMUTATION is None or _PERMUTATION.device != x.device:
        _PERMUTATION = tp.tensor(
            list(range(half, 2 * half)) + list(range(0, half)),
            dtype=tp.int64, device=x.device,
        )
    return x * cos + x.index_select(x.dim() - 1, _PERMUTATION) * sin


_PERMUTATION = None


def rotate_slice_copy(x, cos, sin):
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    out = tp.empty_like(x)
    out[..., :half].copy_(first * cos - second * sin)
    out[..., half:].copy_(second * cos + first * sin)
    return out


def rotate_identity(x, cos, sin):
    return x


VARIANTS = {
    "roll + addcmul": (factors_full, rotate_roll_addcmul),
    "roll + mul + add": (factors_full, rotate_roll_mul),
    "roll + in place": (factors_full, rotate_roll_in_place),
    "index_select + mul + add": (factors_full, rotate_index_select),
    "slices + copy": (factors_half, rotate_slice_copy),
    "no rotation, floor": (factors_none, rotate_identity),
}


def measure(model, ids, steps, sync, static_cache):
    from tensorplay.nn.cache import StaticCache

    length = ids.shape[1]
    cache = StaticCache(static_cache)
    with tp.no_grad():
        logits = model(
            ids,
            position_ids=tp.arange(0, length, device="mlu").unsqueeze(0),
            past_key_values=cache,
        )
    sync()

    host, device, total = [], [], []
    with tp.no_grad():
        for offset in range(steps):
            # The cache advances by itself, so each step writes at the position
            # the table describes.
            token = logits[:, -1:].argmax(dim=-1)
            position = tp.arange(length + offset, length + offset + 1,
                                 device="mlu").unsqueeze(0)
            sync()
            start = time.time()
            logits = model(token, position_ids=position, past_key_values=cache)
            queued = time.time()
            sync()
            done = time.time()
            host.append(queued - start)
            device.append(done - queued)
            total.append(done - start)

    del cache
    return (stats.median(total), stats.median(host), stats.median(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--static-cache", type=int, default=512)
    ap.add_argument("--variants", default=None,
                    help="comma separated names, default is all of them")
    ap.add_argument("--norms", default="module,written,fused")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    sync = tp.mlu.synchronize

    model, config = load_qwen3vl(args.model, dtype=tp.float16, device="mlu",
                                 progress=print)
    cache_dense_weights(model, tp.float16)
    sync()

    ids = tp.tensor([[104198, 48, 16948, 3837, 67071]], dtype=tp.int64, device="mlu")

    from tensorplay import nn as tnn

    norms = [m for m in model.language_model.modules() if isinstance(m, tnn.RMSNorm)]
    original_norms = [(m, m.forward) for m in norms]

    def set_norm(mode):
        if mode == "module":
            for module, body in original_norms:
                module.forward = body
            return
        from qwen3vl.model import bind_normalization

        for module, _ in original_norms:
            module.forward = bind_normalization(module, mode)

    variants = (args.variants.split(",") if args.variants else list(VARIANTS))
    modes = args.norms.split(",")

    original_coeffs = qmodel._rotary_coefficients
    original_rotate = qmodel._rotate_heads
    print(f"\n{'variant':34} {'step':>9} {'host':>9} {'device':>9}")
    for name in variants:
        coeffs, rotate = VARIANTS[name]
        qmodel._rotary_coefficients = coeffs
        qmodel._rotate_heads = rotate
        for mode in modes:
            set_norm(mode)
            step, host, device = measure(model, ids, args.steps, sync,
                                         args.static_cache)
            print(f"  {name + ', ' + mode:32} {step * 1e3:9.2f} {host * 1e3:9.2f} "
                  f"{device * 1e3:9.2f}")
    qmodel._rotary_coefficients = original_coeffs
    qmodel._rotate_heads = original_rotate
    set_norm("module")


if __name__ == "__main__":
    main()
