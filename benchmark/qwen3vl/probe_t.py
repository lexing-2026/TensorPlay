"""Compare how a weight's layout changes the cost of contracting against it.

A projection can keep its values with the output extent outermost and be used
through a transposed view, or keep them with the contracted extent outermost
and be used directly.  The two describe the same product, so any difference in
cost comes from what the kernel makes of the layout and from who has to move
the values.

    python probe_t.py [--repeat 30]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp

# (contracted width, output width) for the projections a block issues.
SHAPES = {
    "q": (4096, 4096),
    "o": (4096, 4096),
    "k": (4096, 1024),
    "v": (4096, 1024),
    "gate": (4096, 12288),
    "up": (4096, 12288),
    "down": (12288, 4096),
    "qkv": (4096, 6144),
    "lm_head": (4096, 151936),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--shapes", default="q,k,v,gate,up,down,qkv,lm_head")
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    def timed(run, repeat):
        for _ in range(3):
            run()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(repeat):
            run()
        tp.mlu.synchronize()
        return (time.time() - start) / repeat

    print(f"{'shape':>8} {'laid out':>22} {'ms':>8} {'GB/s':>8} {'relative':>9}")
    for name in args.shapes.split(","):
        inner, out = SHAPES[name]
        moved = inner * out * 2
        x = tp.randn(1, inner, device="mlu", dtype=tp.float16)
        # Values with the output extent outermost, used through a view.
        outer = tp.randn(out, inner, device="mlu", dtype=tp.float16)
        # The same values rearranged so the contracted extent is outermost.
        direct = outer.t().contiguous()

        forms = [
            ("outermost, x @ w.t()", lambda x=x, w=outer: x @ w.t()),
            ("contracted, x @ w", lambda x=x, w=direct: x @ w),
            ("contracted, x.mm(w)", lambda x=x, w=direct: x.mm(w)),
            ("outermost, x.mm(w.t())", lambda x=x, w=outer: x.mm(w.t())),
        ]
        base = None
        for label, run in forms:
            per_call = timed(run, args.repeat)
            if base is None:
                base = per_call
            print(f"{name:>8} {label:>22} {per_call * 1e3:8.3f} "
                  f"{moved / per_call / 1e9:8.1f} {per_call / base:8.2f}x")
        del outer, direct

    print("\n  the whole sequence of a block, issued each way")
    activations = {name: tp.randn(1, SHAPES[name][0], device="mlu", dtype=tp.float16)
                   for name in ("q", "k", "v", "o", "gate", "up", "down")}
    outer = {name: tp.randn(*reversed(SHAPES[name]), device="mlu", dtype=tp.float16)
             for name in activations}
    direct = {name: outer[name].t().contiguous() for name in activations}

    def through_view():
        return [activations[name] @ outer[name].t() for name in activations]

    def through_direct():
        return [activations[name] @ direct[name] for name in activations]

    for label, run in (("x @ w.t()", through_view), ("x @ w", through_direct)):
        per_call = timed(run, args.repeat)
        moved = sum(i * o * 2 for i, o in (SHAPES[n] for n in activations))
        print(f"  {label:24} {per_call * 1e3:7.3f} ms  {moved / per_call / 1e9:6.1f} GB/s"
              f"  ({per_call * 36 * 1e3:6.2f} ms/tower)")


if __name__ == "__main__":
    main()
