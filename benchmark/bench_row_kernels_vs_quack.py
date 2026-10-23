"""Row kernels against quack on the same device, shapes and dtypes.

Every case is run through both libraries with inputs built once from a fixed
seed, the answers are checked against a float32 reference, and both the first
call (which pays for compilation and any tuning) and the steady-state time are
recorded.  A direction the reference library refuses is reported with the
message it raised instead of being dropped, so the table says what runs.

    python benchmark/bench_row_kernels_vs_quack.py --quack-path <dir> --json out.json

The stax side reaches the kernels the way the framework does: tensors cross over
through the DLPack bridge and the calls are the ones in
``tensorplay.compiler.backends.stax.codegen.cutedsl.kernels``.

Both libraries keep their compiled objects on disk, so a repeat run times the
loader in the first-call column.  To time the compiler instead, give each
library a fresh object directory (``QUACK_CACHE_DIR`` and ``TP_STAX_CACHE_DIR``)
or turn the stax one off with ``TP_STAX_CACHE=0``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen.cutedsl.kernels import (
    ce_bwd,
    ce_fwd,
    rmsnorm_bwd,
    rmsnorm_fwd,
    softmax_bwd,
    softmax_fwd,
)

EPS = 1e-6

RMS_CASES = [
    (1024, 4096, torch.bfloat16),
    (4096, 4096, torch.bfloat16),
    (4096, 8192, torch.float16),
    (4096, 8192, torch.float32),
    (16384, 8192, torch.bfloat16),
    (4096, 16384, torch.bfloat16),
    (4096, 16384, torch.float32),
    (4096, 32000, torch.bfloat16),
]
SOFTMAX_CASES = [
    (1024, 4096, torch.bfloat16),
    (4096, 4096, torch.float32),
    (16384, 4096, torch.bfloat16),
    (4096, 8192, torch.float16),
    (4096, 8192, torch.bfloat16),
    (4096, 32000, torch.float16),
    (4096, 32000, torch.bfloat16),
    (4096, 32000, torch.float32),
]
CE_CASES = [
    (4096, 8192, torch.float16),
    (4096, 8192, torch.bfloat16),
    (4096, 8192, torch.float32),
    (4096, 32000, torch.float16),
    (4096, 32000, torch.bfloat16),
    (16384, 32000, torch.bfloat16),
    (4096, 32000, torch.float32),
]

#: Row passes over the tensor each direction performs, as read + write.  Cross
#: entropy reads the logits once going forward and both reads and writes them
#: coming back, because the gradient is written where the logits were.
TRAFFIC = {"fwd": 2, "bwd": 3}
CE_TRAFFIC = {"fwd": 1, "bwd": 2}


def from_torch(t: torch.Tensor):
    return tp.from_dlpack(t)


def to_torch(t) -> torch.Tensor:
    return torch.from_dlpack(t)


def time_call(fn, iters: int, warmup: int, rounds: int = 3):
    """Seconds per call for back-to-back calls, and the last result.

    Calls are queued without a synchronise between them, so this is the time
    the device spends on one call rather than the time the host takes to hand
    it over.  A direction is averaged over several such rounds and the median
    round is reported.
    """

    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()
    averages = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            last = fn()
        end.record()
        torch.cuda.synchronize()
        averages.append(start.elapsed_time(end) * 1e-3 / iters)
    return statistics.median(averages), last


def timed_round(fn, iters: int) -> float:
    """Seconds per call over one round of back-to-back calls."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e-3 / iters


def first_call(fn) -> tuple[float, object]:
    """Seconds the first call takes, along with whatever it returned."""

    torch.cuda.synchronize()
    begin = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - begin, out


def copy_wall(elements: int, element_size: int) -> float:
    """Seconds a read-and-write pass over the same bytes takes as a copy."""

    src = torch.empty(elements * element_size, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    return time_call(lambda: dst.copy_(src), 30, 8)[0]


def wall_gbs(elements: int, element_size: int, copy_s: float) -> float:
    return 2 * elements * element_size / 1e9 / copy_s


def max_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return (got.float() - want.float()).abs().max().item()


def brief(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:180]


def measure_pair(case_id, records, entries, gb, gb_bwd, args):
    """Time both directions of every library on one case, round by round.

    The libraries take turns inside each round and swap places between rounds,
    so a drift in the card's clock across the run cannot favour whichever one
    was measured first.  A callable returns the tensors it produced and nothing
    else, so the timing sees the operation rather than a full-tensor comparison;
    the answers are checked once, outside the timed region.  ``backward``
    builds the backward callable; whatever setup it needs (such as a forward
    pass to record a graph) is not timed either, so the two directions stay
    comparable.  A direction the library refuses leaves the other in the table.
    """

    fields: dict = {}
    fwd_fns: dict = {}
    bwd_fns: dict = {}
    for name, fwd, check_fwd, backward, check_bwd in entries:
        try:
            first, out = first_call(fwd)
            fields[name] = dict(first_fwd=first, fwd_err=list(check_fwd(out)))
            fwd_fns[name] = fwd
        except Exception as exc:
            fields[name] = dict(fwd_error=brief(exc))
    for name, fwd, check_fwd, backward, check_bwd in entries:
        if name not in fwd_fns:
            continue
        try:
            bwd = backward()
            first, out = first_call(bwd)
            fields[name].update(first_bwd=first, bwd_err=list(check_bwd(out)))
            bwd_fns[name] = bwd
        except Exception as exc:
            fields[name]["bwd_error"] = brief(exc)

    for fn in fwd_fns.values():
        for _ in range(args.warmup):
            fn()
    for fn in bwd_fns.values():
        for _ in range(args.warmup):
            fn()
    torch.cuda.synchronize()

    fwd_samples = {name: [] for name in fwd_fns}
    bwd_samples = {name: [] for name in bwd_fns}
    for round_index in range(args.rounds):
        for fns, samples in ((fwd_fns, fwd_samples), (bwd_fns, bwd_samples)):
            names = list(fns)
            if round_index % 2:
                names.reverse()
            for name in names:
                samples[name].append(timed_round(fns[name], args.iters))

    for name, entry in fields.items():
        if name in fwd_samples:
            steady = statistics.median(fwd_samples[name])
            entry.update(fwd_s=steady, fwd_gbs=gb / steady)
        if name in bwd_samples:
            steady = statistics.median(bwd_samples[name])
            entry.update(bwd_s=steady, bwd_gbs=gb_bwd / steady)
        records.append(dict(case_id, library=name,
                            ok="fwd_s" in entry or "bwd_s" in entry, **entry))


def run_rms(case, records, args):
    m, n, dtype = case
    torch.manual_seed(0)
    xt = torch.randn(m, n, device="cuda", dtype=dtype)
    wt = torch.randn(n, device="cuda", dtype=dtype)
    dyt = torch.randn(m, n, device="cuda", dtype=dtype)
    xf, wf, dyf = xt.float(), wt.float(), dyt.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    xh = xf * rstd
    ref_out = xh * wf
    wdy = dyf * wf
    ref_dx = (wdy - xh * (xh * wdy).mean(-1, keepdim=True)) * rstd
    ref_dw = (dyf * xh).sum(0)

    case_id = dict(kernel="rmsnorm", rows=m, cols=n, dtype=str(dtype).split(".")[-1])
    wall = copy_wall(m * n, xt.element_size())
    gb = TRAFFIC["fwd"] * m * n * xt.element_size() / 1e9
    gb_bwd = TRAFFIC["bwd"] * m * n * xt.element_size() / 1e9

    entries = []
    if not args.no_quack:
        import quack

        def q_fwd():
            return quack.rmsnorm(xt, wt, eps=EPS)

        def q_check(out):
            return [max_error(out, ref_out)]

        def q_backward():
            qx = xt.detach().requires_grad_(True)
            qw = wt.detach().requires_grad_(True)
            out = quack.rmsnorm(qx, qw, eps=EPS)

            def run():
                return torch.autograd.grad(out, (qx, qw), dyt, retain_graph=True)

            return run

        def q_check_bwd(grads):
            dx, dw = grads
            return [max_error(dx, ref_dx), max_error(dw, ref_dw)]

        entries.append(("quack", q_fwd, q_check, q_backward, q_check_bwd))

    x, w, dy = from_torch(xt), from_torch(wt), from_torch(dyt)
    held: dict = {}

    def s_fwd():
        out, rstd_t = rmsnorm_fwd(x, w, eps=EPS)
        held["rstd"] = rstd_t
        return out

    def s_check(out):
        return [max_error(to_torch(out), ref_out)]

    def s_backward():
        if "rstd" not in held:
            raise RuntimeError("forward produced no row scales")
        rstd_t = held["rstd"]

        def run():
            return rmsnorm_bwd(x, w, dy, rstd_t)

        return run

    def s_check_bwd(grads):
        dx, dw = grads
        return [max_error(to_torch(dx), ref_dx), max_error(to_torch(dw), ref_dw)]

    entries.append(("stax", s_fwd, s_check, s_backward, s_check_bwd))
    measure_pair(case_id, records, entries, gb, gb_bwd, args)
    records.append(dict(case_id, library="copy", ok=True,
                        fwd_gbs=wall_gbs(m * n, xt.element_size(), wall)))
    return case_id


def run_softmax(case, records, args):
    m, n, dtype = case
    torch.manual_seed(0)
    xt = torch.randn(m, n, device="cuda", dtype=dtype)
    dyt = torch.randn(m, n, device="cuda", dtype=dtype)
    y = torch.softmax(xt.float(), -1)
    ref_dx = y * (dyt.float() - (y * dyt.float()).sum(-1, keepdim=True))
    yt = y.to(dtype)

    case_id = dict(kernel="softmax", rows=m, cols=n, dtype=str(dtype).split(".")[-1])
    wall = copy_wall(m * n, xt.element_size())
    gb = TRAFFIC["fwd"] * m * n * xt.element_size() / 1e9
    gb_bwd = TRAFFIC["bwd"] * m * n * xt.element_size() / 1e9

    entries = []
    if not args.no_quack:
        import quack

        def q_fwd():
            return quack.softmax(xt)

        def q_check(out):
            return [max_error(out, y)]

        def q_backward():
            qx = xt.detach().requires_grad_(True)
            out = quack.softmax(qx)

            def run():
                return torch.autograd.grad(out, qx, dyt, retain_graph=True)

            return run

        def q_check_bwd(grads):
            (dx,) = grads
            return [max_error(dx, ref_dx)]

        entries.append(("quack", q_fwd, q_check, q_backward, q_check_bwd))

    x, dy = from_torch(xt), from_torch(dyt)
    held: dict = {}

    def s_fwd():
        out = softmax_fwd(x)
        held["y"] = out
        return out

    def s_check(out):
        return [max_error(to_torch(out), y)]

    def s_backward():
        if "y" not in held:
            raise RuntimeError("forward produced no output")
        yt = held["y"]

        def run():
            return (softmax_bwd(yt, dy),)

        return run

    def s_check_bwd(grads):
        (dx,) = grads
        return [max_error(to_torch(dx), ref_dx)]

    entries.append(("stax", s_fwd, s_check, s_backward, s_check_bwd))
    measure_pair(case_id, records, entries, gb, gb_bwd, args)
    records.append(dict(case_id, library="copy", ok=True,
                        fwd_gbs=wall_gbs(m * n, xt.element_size(), wall)))
    return case_id


def run_ce(case, records, args):
    m, n, dtype = case
    torch.manual_seed(0)
    xt = torch.randn(m, n, device="cuda", dtype=dtype)
    tt = torch.randint(0, n, (m,), device="cuda", dtype=torch.int64)
    dl = torch.randn(m, device="cuda", dtype=torch.float32)
    xf = xt.float()
    loss_ref = torch.nn.functional.cross_entropy(xf, tt, reduction="none")
    p = torch.exp(xf - torch.logsumexp(xf, -1, keepdim=True))
    p[torch.arange(m), tt] -= 1.0
    dx_ref = p * dl[:, None]

    case_id = dict(kernel="cross_entropy", rows=m, cols=n,
                   dtype=str(dtype).split(".")[-1])
    wall = copy_wall(m * n, xt.element_size())
    gb = CE_TRAFFIC["fwd"] * m * n * xt.element_size() / 1e9
    gb_bwd = CE_TRAFFIC["bwd"] * m * n * xt.element_size() / 1e9

    entries = []
    if not args.no_quack:
        import quack

        def q_fwd():
            return quack.cross_entropy(xt, tt, reduction="none")

        def q_check(out):
            return [max_error(out, loss_ref)]

        def q_backward():
            qx = xt.detach().requires_grad_(True)
            out = quack.cross_entropy(qx, tt, reduction="none")

            def run():
                return torch.autograd.grad(out, qx, dl, retain_graph=True)

            return run

        def q_check_bwd(grads):
            (dx,) = grads
            return [max_error(dx, dx_ref)]

        entries.append(("quack", q_fwd, q_check, q_backward, q_check_bwd))

    x, target, dloss = from_torch(xt), from_torch(tt), from_torch(dl)
    held: dict = {}

    def s_fwd():
        loss_t, lse_t = ce_fwd(x, target)
        held["lse"] = lse_t
        return loss_t

    def s_check(loss_t):
        return [max_error(to_torch(loss_t), loss_ref)]

    def s_backward():
        if "lse" not in held:
            raise RuntimeError("forward produced no log-sum-exp")
        lse_t = held["lse"]

        def run():
            return (ce_bwd(x, target, lse_t, dloss),)

        return run

    def s_check_bwd(grads):
        (dx,) = grads
        return [max_error(to_torch(dx), dx_ref)]

    entries.append(("stax", s_fwd, s_check, s_backward, s_check_bwd))
    measure_pair(case_id, records, entries, gb, gb_bwd, args)
    records.append(dict(case_id, library="copy", ok=True,
                        fwd_gbs=wall_gbs(m * n, xt.element_size(), wall)))
    return case_id


def fmt(value, digits=1):
    return "-" if value is None else f"{value * 1e6:.{digits}f}"


def fmt_errors(values) -> str:
    if not values:
        return "-"
    return "/".join(f"{v:.1e}" for v in values)


def fmt_gbs(record: dict, key: str) -> str:
    value = record.get(key)
    return "-" if value is None else f"{value:.0f}"


def fmt_first(record: dict) -> str:
    def one(key):
        value = record.get(key)
        return "-" if value is None else f"{value:.2f}"

    return f"{one('first_fwd')}/{one('first_bwd')}"


def print_table(records: list[dict]):
    header = (f"{'kernel':15s} {'shape':>14s} {'dtype':>9s} {'lib':>6s} "
              f"{'fwd us':>9s} {'fwd GB/s':>9s} {'bwd us':>9s} {'bwd GB/s':>9s} "
              f"{'first f/b s':>14s} {'err f/b':>20s}")
    print(header)
    print("-" * len(header))
    for r in records:
        shape = f"{r['rows']}x{r['cols']}"
        errors = fmt_errors(r.get("fwd_err")) + " " + fmt_errors(r.get("bwd_err"))
        notes = " ".join(x for x in (r.get("fwd_error"), r.get("bwd_error")) if x)
        print(f"{r['kernel']:15s} {shape:>14s} {r['dtype']:>9s} {r['library']:>6s} "
              f"{fmt(r.get('fwd_s')):>9s} {fmt_gbs(r, 'fwd_gbs'):>9s} "
              f"{fmt(r.get('bwd_s')):>9s} {fmt_gbs(r, 'bwd_gbs'):>9s} "
              f"{fmt_first(r):>14s} {errors.strip():>20s} {notes}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("rmsnorm", "softmax", "cross_entropy", "all"),
                        default="all")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-quack", action="store_true")
    parser.add_argument("--quack-path", type=str, default=None,
                        help="directory holding the reference library, put on the path first")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    if args.quack_path:
        sys.path.insert(0, args.quack_path)
    if not args.no_quack:
        try:
            import quack  # noqa: F401
        except ImportError:
            print("reference library not importable; pass --quack-path or --no-quack",
                  file=sys.stderr)
            args.no_quack = True
    if not tp.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    print(tp.cuda.get_device_properties("cuda").name)

    records: list[dict] = []
    runs = {"rmsnorm": (RMS_CASES, run_rms),
            "softmax": (SOFTMAX_CASES, run_softmax),
            "cross_entropy": (CE_CASES, run_ce)}
    for name, (cases, run) in runs.items():
        if args.kernel not in ("all", name):
            continue
        for case in cases:
            case_id = run(case, records, args)
            print(f"{name:15s} {case_id['rows']}x{case_id['cols']} "
                  f"{case_id['dtype']:>9s} done", flush=True)

    print()
    print_table(records)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(records, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
