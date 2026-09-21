"""TunableOp GEMM benchmark: tuned replay versus the in-process autotune.

For every shape, three isolated subprocess runs are compared:

* baseline:  tunable disabled; the first call of a new shape pays the
             in-process per-plan autotune (the shipped default behavior)
* tuned:     tunable enabled with tuning on; the first call runs the bounded
             search and persists the winner to a shared results file
* replay:    tunable enabled with tuning off; the first call resolves the
             winner from the file the tuned run wrote — no measurement

First-call times exclude one-time CUDA and library initialization (each
child primes the context with a tiny GEMM first), so they show the
per-shape startup cost of each mode. Steady-state times show whether the
larger search budget lands on a faster algorithm than the in-process
autotune.

Usage:
    python benchmark/bench_cuda_tunable.py [--iters 50] [--tags fp16,...] [--json]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# (m, n, k, dtype, bias, transposed, tag). Projection-style shapes store the
# weight as (n, k) and multiply by its transpose, the layout training code
# actually uses.
DEFAULT_SHAPES = [
    (4096, 4096, 4096, "float32", False, False, "fp32 square"),
    (8192, 4096, 4096, "float32", True, True, "fp32 linear"),
    (4096, 11008, 4096, "float16", True, True, "fp16 mlp up"),
    (4096, 4096, 11008, "float16", True, True, "fp16 mlp down"),
    (4096, 11008, 4096, "bfloat16", True, True, "bf16 mlp up"),
    (4096, 4096, 11008, "bfloat16", True, True, "bf16 mlp down"),
    (512, 4096, 4096, "float16", True, True, "fp16 decode batch"),
    (1, 4096, 4096, "float16", True, True, "fp16 decode single"),
    (256, 256, 256, "float64", True, False, "fp64 small"),
]

def run_child(args, env, cwd):
    out = subprocess.run([sys.executable, __file__] + args,
                         capture_output=True, text=True, env=env, cwd=cwd)
    if out.returncode != 0:
        raise RuntimeError(f"child failed ({' '.join(args)}):\n{out.stderr}")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    return json.loads(lines[-1])


def child_main(args):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import tensorplay as tp

    dt = getattr(tp, args.dtype)
    a = tp.randn(args.m, args.k, dtype=dt, device="cuda")
    if args.transposed:
        mat = tp.randn(args.n, args.k, dtype=dt, device="cuda").t()
    else:
        mat = tp.randn(args.k, args.n, dtype=dt, device="cuda")
    bias = (tp.randn(args.n, dtype=dt, device="cuda")
            if args.bias else None)

    def op():
        return tp.addmm(bias, a, mat) if args.bias else a @ mat

    # Prime CUDA context creation and library initialization so the
    # measured first call only carries the per-shape cost.
    _ = tp.randn(8, 8, device="cuda") @ tp.randn(8, 8, device="cuda")
    tp.cuda.synchronize()

    t0 = time.perf_counter_ns()
    op()
    tp.cuda.synchronize()
    first_ms = (time.perf_counter_ns() - t0) / 1e6

    for _ in range(5):
        op()
    tp.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(args.iters):
        op()
    tp.cuda.synchronize()
    steady_ms = (time.perf_counter_ns() - t0) / 1e6 / args.iters

    print(json.dumps({"mode": args.mode, "first_ms": round(first_ms, 3),
                      "steady_ms": round(steady_ms, 4)}))


def winner_from_db(db_dir, m, n, k, bias, transposed):
    sig = f"ta{'T' if transposed else 'N'}_m{m}_n{n}_k{k}_bias{int(bias)}"
    for path in glob.glob(os.path.join(db_dir, "db*.csv")):
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4 and sig in parts[1]:
                    return parts[2]
    return "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--tags", type=str, default="",
                        help="comma-separated subset of shape tags to run")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable records instead of a table")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["baseline", "tuned", "replay"])
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--transposed", action="store_true")
    args = parser.parse_args()

    if args.child:
        child_main(args)
        return

    import tensorplay as tp
    if not tp.cuda.is_available():
        sys.exit("CUDA is not available")

    shapes = DEFAULT_SHAPES
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",") if t.strip()}
        shapes = [s for s in DEFAULT_SHAPES if s[-1] in wanted]

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "db.csv")
        records = []
        for m, n, k, dtype, bias, transposed, tag in shapes:
            common = ["--child", "--m", str(m), "--n", str(n), "--k", str(k),
                      "--dtype", dtype, "--iters", str(args.iters)]
            if bias:
                common.append("--bias")
            if transposed:
                common.append("--transposed")

            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("TP_TUNABLEOP_")}

            base = run_child(common + ["--mode", "baseline"], env, tmp)

            tuned_env = dict(env, TP_TUNABLEOP_ENABLED="1",
                             TP_TUNABLEOP_FILENAME=db)
            tuned = run_child(common + ["--mode", "tuned"], tuned_env, tmp)

            replay_env = dict(env, TP_TUNABLEOP_ENABLED="1",
                              TP_TUNABLEOP_TUNING="0",
                              TP_TUNABLEOP_FILENAME=db)
            replay = run_child(common + ["--mode", "replay"], replay_env, tmp)

            records.append({
                "tag": tag, "m": m, "n": n, "k": k, "dtype": dtype,
                "bias": bias, "transposed": transposed,
                "baseline": base, "tuned": tuned, "replay": replay,
                "winner": winner_from_db(tmp, m, n, k, bias, transposed),
            })

    if args.json:
        print(json.dumps(records, indent=2))
        return

    header = (f"{'shape':<28}{'mode':<10}{'first ms':>10}{'steady ms':>12}"
              f"{'vs base':>9}")
    print(header)
    print("-" * len(header))
    for r in records:
        shape = f"{r['m']}x{r['n']}x{r['k']} {r['dtype']}"
        if r["bias"]:
            shape += "+b"
        if r["transposed"]:
            shape += " ^T"
        delta = (r["tuned"]["steady_ms"] / r["baseline"]["steady_ms"] - 1) * 100
        for mode in ("baseline", "tuned", "replay"):
            print(f"{shape:<28}{mode:<10}{r[mode]['first_ms']:>10.2f}"
                  f"{r[mode]['steady_ms']:>12.4f}"
                  + (f"{delta:>+8.1f}%" if mode == "tuned" else " " * 9))
        print(f"{'':<28}winner: {r['winner']}")


if __name__ == "__main__":
    main()
