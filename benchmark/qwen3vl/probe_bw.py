"""Measure how fast this device can move values, and how it likes them served.

Everything a decode step does is bounded by moving the weights past the
arithmetic units once per token, so the useful question is what rate a pure
read reaches, and whether reading several distinct buffers in turn keeps that
rate.  The device's own report is printed beside the measurements.

    python probe_bw.py [--repeat 20]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def device_report():
    import tensorplay_mlu

    torch_mlu = getattr(tensorplay_mlu, "torch", None)
    try:
        import torch
        import torch_mlu  # noqa: F401

        torch_mlu = torch
    except Exception:  # noqa: BLE001
        pass
    if torch_mlu is None:
        print("  device report unavailable")
        return
    try:
        props = torch_mlu.mlu.get_device_properties(0)
        fields = ("name", "total_memory", "multi_processor_count", "L2_cache_size")
        for field in fields:
            print(f"  {field:22} {getattr(props, field, 'n/a')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  device report failed: {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()
    print("device")
    device_report()

    def timed(run, repeat, moved):
        for _ in range(2):
            run()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(repeat):
            run()
        tp.mlu.synchronize()
        per_call = (time.time() - start) / repeat
        return per_call, moved / per_call / 1e9

    print("\npure reads and writes")
    for size_mb in (16, 64, 256, 1024):
        count = size_mb * 1024 * 1024 // 2
        buffer = tp.randn(count, device="mlu", dtype=tp.float16)
        per_call, rate = timed(lambda: buffer.sum(), args.repeat, count * 2)
        print(f"  read {size_mb:5} MB in one pass        {per_call * 1e3:8.3f} ms  "
              f"{rate:7.1f} GB/s")
        del buffer

    print("\nseveral distinct buffers read in turn")
    chunks = [tp.randn(128 * 1024 * 1024 // 2, device="mlu", dtype=tp.float16)
              for _ in range(7)]
    per_call, rate = timed(lambda: [chunk.sum() for chunk in chunks], args.repeat,
                           sum(chunk.numel() * 2 for chunk in chunks))
    print(f"  seven 128 MB buffers                   {per_call * 1e3:8.3f} ms  "
          f"{rate:7.1f} GB/s")

    print("\none buffer read seven times")
    per_call, rate = timed(lambda: [chunks[0].sum() for _ in range(7)], args.repeat,
                           chunks[0].numel() * 2 * 7)
    print(f"  the same 128 MB buffer                 {per_call * 1e3:8.3f} ms  "
          f"{rate:7.1f} GB/s")

    print("\nweight-shaped products that together read the tower's weights")
    shapes = [(4096, 4096), (4096, 1024), (4096, 1024), (4096, 4096),
              (4096, 12288), (4096, 12288), (12288, 4096)]
    # Values kept with the output extent outermost, used through a view, as
    # the tower's weights are held.
    outer = [tp.randn(out, inner, device="mlu", dtype=tp.float16)
             for inner, out in shapes]
    # The same values with the contracted extent outermost.
    inner_first = [w.t().contiguous() for w in outer]
    activations = [tp.randn(1, inner, device="mlu", dtype=tp.float16)
                   for inner, _ in shapes]
    moved = sum(inner * out * 2 for inner, out in shapes)

    for label, weights in (("output extent outermost", outer),
                           ("contracted extent outermost", inner_first)):
        per_call, rate = timed(
            lambda w=weights: [a @ v.t() if label.startswith("output") else a @ v
                               for a, v in zip(activations, w)],
            args.repeat, moved)
        print(f"  one layer's products, {label:26} {per_call * 1e3:8.3f} ms  "
              f"{rate:7.1f} GB/s  ({per_call * 36 * 1e3:6.2f} ms/tower)")

    print("\n  one at a time, isolated")
    for (inner, out), weight, activation in zip(shapes, outer, activations):
        per_call, rate = timed(lambda a=activation, w=weight: a @ w.t(), args.repeat,
                               inner * out * 2)
        print(f"    {inner:6}x{out:<6}                       {per_call * 1e3:8.3f} ms  "
              f"{rate:7.1f} GB/s")


if __name__ == "__main__":
    main()
