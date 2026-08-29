"""Repeatable profiler-overhead benchmark.

Usage:
    python3 tools/bench_profiler_overhead.py [--shapes 1x1,8x16,256x512]

Clean side-by-side (recommended on shared/CUDA hosts) -- one runtime per
process so thread-pool and BLAS initialization remain isolated:

    python3 tools/bench_profiler_overhead.py --framework tp --threads 16

Prints min-of-reps per-iteration microseconds for:
  * framework baseline (profiler off)
  * profiler ON

Run on a quiet machine only -- CPU contention from parallel builds dwarfs
the effect being measured. The zero-cost criterion compares the baseline
with profiling disabled against the enabled result; the enabled overhead
should remain within the configured noise budget.
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


def bench(framework, m, k, iters=200, reps=5, record_shapes=False,
          threads=None, isolated=False):
    if framework == "tp":
        if not isolated:
            # Load the comparison runtime first on shared CUDA hosts to avoid
            # conflicting runtime-library initialization; imports are outside
            # timing.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
        import tensorplay as tp
        from tensorplay.profiler import profile as Profile
        if threads is not None:
            tp.set_num_threads(threads)
        randn = lambda *s, rg=False: tp.randn(list(s), requires_grad=rg)
        matmul = lambda a, b: a.matmul(b)
    else:
        import torch
        from torch.profiler import profile as Profile
        if threads is not None:
            torch.set_num_threads(threads)
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
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="pin both frameworks to this intra-op thread count (shared "
             "machines: default all-core pools thrash under contention)",
    )
    ap.add_argument(
        "--framework",
        choices=("tp", "torch", "both"),
        default="both",
        help="bench a single framework in this process.  Single-framework "
             "mode skips importing the other library entirely, avoiding the "
             "cross-library BLAS and thread-pool interference. Run the script "
             "twice for a clean side-by-side comparison.",
    )
    args = ap.parse_args()
    frameworks = ("tp", "torch") if args.framework == "both" \
        else (args.framework,)
    isolated = args.framework != "both"
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
        for fw in frameworks:
            samples = []
            for _ in range(args.rounds):
                try:
                    samples.append(
                        bench(fw, m, k, args.iters,
                              record_shapes=args.record_shapes,
                              threads=args.threads, isolated=isolated)
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
        for fw in frameworks:
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
            print(f"  {shape_name}: profiled time ref/tp={torch_on / tp_on:.3f}x")


if __name__ == "__main__":
    main()
