"""Host-overhead benchmark for the CUDA graph replay paths.

Measures pure host-side cost per iteration (GPU work is identical across
variants; the wall time of an enqueue-only loop is host overhead):

* eager:            launch one small op per input per iteration (baseline)
* graph.bulk:       CUDAGraph.stage_and_launch(...)          (one native call)
* manager.bulk:     CudaGraphManager.replay(key, ...)        (adds key/signature check)

paths are measured side by side for reference.

Usage:
    python benchmark/bench_cudagraph_replay.py [--n 8] [--size 256] [--iters 2000]
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tensorplay as tp


def _bench(fn, iters):
    fn()
    tp.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter_ns() - start
    tp.cuda.synchronize()
    return elapsed / iters / 1e3  # us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="inputs per replay")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    device = tp.Device("cuda", 0)
    shape = (args.size, args.size)

    ws = [tp.randn(shape, device=device) for _ in range(args.n)]
    xs = [tp.randn(shape, device=device) for _ in range(args.n)]
    fresh = [[tp.randn(shape, device=device) for _ in range(args.n)]
             for _ in range(2)]

    def eager():
        for w, a in zip(ws, fresh[0]):
            a.mul_(w).relu_()

    # -- capture --------------------------------------------------------------
    g = tp.cuda.CUDAGraph()
    static_ws = [w.clone() for w in ws]
    static_xs = [x.clone() for x in xs]
    with tp.cuda.graph(g):
        outs = [static_xs[i].mul_(static_ws[i]).relu_()
                for i in range(args.n)]

    def manual():
        for dst, src in zip(static_xs, fresh[1]):
            dst.copy_(src)
        g.replay()

    from tensorplay._stax import CudaGraphManager

    mgr = CudaGraphManager(max_entries=4)

    def fn(*tensors):
        acc = tensors[0] * 1.0
        for t in tensors[1:]:
            acc = acc * t
        return acc.relu()

    entry_key = "bench"
    mgr.capture(entry_key, fn, *xs)

    def bulk_mgr():
        mgr.replay(entry_key, *fresh[1])

    rows = [
        ("eager (%d ops)" % args.n, eager),
        ("graph.manual copy_+replay", manual),
        ("graph.bulk stage_and_launch",
         lambda: g.stage_and_launch(static_xs, fresh[1])),
        ("manager.bulk", bulk_mgr),
    ]

    torch_rows = []
    try:
        import torch

        if torch.cuda.is_available():
            tws = [torch.randn(shape, device="cuda") for _ in range(args.n)]
            txs = [torch.randn(shape, device="cuda") for _ in range(args.n)]
            tfresh = torch.randn(shape, device="cuda")

            def torch_eager():
                for w in tws:
                    tfresh.mul_(w).relu_()

            tg = torch.cuda.CUDAGraph()
            tsx = [x.clone() for x in txs]
            tsw = [w.clone() for w in tws]
            with torch.cuda.graph(tg):
                for i in range(args.n):
                    tsx[i].mul_(tsw[i]).relu_()

            def torch_manual():
                for dst in tsx:
                    dst.copy_(tfresh)
                tg.replay()

            torch_rows = [
                ("ref.eager (%d ops)" % args.n, torch_eager),
                ("ref.graph copy_+replay", torch_manual),
            ]
    except ImportError:
        pass

    print(f"n={args.n} size={args.size}x{args.size} iters={args.iters}")
    print(f"{'path':<32}{'us/iter':>12}")
    print("-" * 44)

    def report(name, fn):
        us = _bench(fn, args.iters)
        print(f"{name:<32}{us:>12.2f}")
        return us

    base = {}
    for name, fn in torch_rows:
        base[name] = report(name, fn)
    for name, fn in rows:
        base[name] = report(name, fn)

    tm = base.get("ref.graph copy_+replay")
    bm = base["manager.bulk"]
    if tm:
        print(f"\nTensorPlay/ref bulk-replay ratio: {bm / tm:.2f}x "
              f"({'faster' if bm < tm else 'slower'})")


if __name__ == "__main__":
    main()
