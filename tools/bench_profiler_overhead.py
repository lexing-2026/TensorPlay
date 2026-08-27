"""Repeatable profiler-overhead benchmark: tensorplay vs torch.

Usage:
    python3 tools/bench_profiler_overhead.py [--shapes 1x1,8x16,256x512]

Prints min-of-reps per-iteration microseconds for:
  * framework baseline (profiler off)
  * profiler ON

Run on a quiet machine only -- CPU contention from parallel builds dwarfs
the effect being measured.  The pass/fail criterion for "zero-cost when
off" is base_tp ~= base_torch within noise; the criterion for "usable when
on" is on/base <= ~3x (torch.profiler itself costs 2-5x on tiny ops).
"""

import argparse
import statistics
import time


def _profile_stats(framework, prof):
    if framework == "tp":
        events = list(prof.events)
        names = {ev[0] for ev in events}
    else:
        events = prof.events()
        names = {ev.key for ev in events}
    return len(events), len(names)


def bench(framework, m, k, iters=200, reps=5, record_shapes=False):
    if framework == "tp":
        # TensorPlay links against the system CUDA runtime.  On CUDA hosts,
        # loading it before torch can make torch's packaged libc10_cuda.so
        # fail to resolve cudaGetDriverEntryPointByVersion.  Import torch
        # first for the fair side-by-side process; imports are outside timing.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        import tensorplay as tp
        from tensorplay.profiler import profile as Profile
        randn = lambda *s, rg=False: tp.randn(list(s), requires_grad=rg)
        matmul = lambda a, b: a.matmul(b)
    else:
        import torch
        from torch.profiler import profile as Profile
        randn = lambda *s, rg=False: torch.randn(list(s), requires_grad=rg)
        matmul = lambda a, b: a.matmul(b)

    x = randn(m, k, rg=True)
    w = randn(k, 64)

    def work():
        h = matmul(x, w).relu()
        return (h * h).sum()

    def timed():
        best = float("inf")
        for _ in range(reps):
            work()
            t0 = time.perf_counter()
            for _ in range(iters):
                work()
            best = min(best, (time.perf_counter() - t0) / iters * 1e6)
        return best

    # warm-up outside any session
    for _ in range(40):
        work()
    base = timed()

    with Profile(record_shapes=record_shapes) as prof:
        for _ in range(20):
            work()
        on = timed()

    event_count, op_count = _profile_stats(framework, prof)
    return base, on, event_count, op_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="256x512")
    ap.add_argument(
        "--shapes",
        help="comma-separated MxK cases; overrides --shape",
    )
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--record-shapes", action="store_true")
    args = ap.parse_args()
    shape_specs = args.shapes.split(",") if args.shapes else [args.shape]
    shapes = []
    for spec in shape_specs:
        try:
            m, k = (int(v) for v in spec.strip().lower().split("x"))
        except ValueError as exc:
            raise SystemExit(f"invalid shape {spec!r}; expected MxK") from exc
        if m <= 0 or k <= 0:
            raise SystemExit(f"invalid shape {spec!r}; dimensions must be positive")
        shapes.append((spec.strip(), m, k))

    rows = {}
    for shape_name, m, k in shapes:
        for fw in ("tp", "torch"):
            samples = []
            for _ in range(args.rounds):
                try:
                    samples.append(
                        bench(fw, m, k, args.iters, record_shapes=args.record_shapes)
                    )
                except ImportError:
                    print(f"{fw}: not installed, skipped")
                    break
            if samples:
                rows[(shape_name, fw)] = samples

    print(
        f"{'shape':<12}{'framework':<10}{'base us':>10}{'profiled us':>13}"
        f"{'overhead':>10}{'events':>10}{'unique ops':>12}"
    )
    for shape_name, _m, _k in shapes:
        for fw in ("tp", "torch"):
            samples = rows.get((shape_name, fw))
            if not samples:
                continue
            b = statistics.median(x[0] for x in samples)
            o = statistics.median(x[1] for x in samples)
            ev = int(statistics.median(x[2] for x in samples))
            ops = int(statistics.median(x[3] for x in samples))
            print(
                f"{shape_name:<12}{fw:<10}{b:>10.1f}{o:>13.1f}"
                f"{(o / b - 1) * 100:>9.1f}%{ev:>10}{ops:>12}"
            )
        tp_row = rows.get((shape_name, "tp"))
        torch_row = rows.get((shape_name, "torch"))
        if tp_row and torch_row:
            tp_on = statistics.median(x[1] for x in tp_row)
            torch_on = statistics.median(x[1] for x in torch_row)
            print(f"  {shape_name}: profiled time torch/tp={torch_on / tp_on:.3f}x")


if __name__ == "__main__":
    main()
