"""End-to-end attention-operator benchmark: this tree against a second runtime.

Every case runs the same public operator on both sides -- the plain entry
point in eager mode, and the compiler-wrapped entry point in compile mode --
so the ratio is the end-to-end cost a caller actually pays, including
whatever routing the public entry point performs on the way to a kernel.

Timing is best-of-N CUDA events after a warmup pass.  Correctness is
checked once per case against the reference before anything is timed; a
case that disagrees beyond dtype tolerance is still timed and tagged BAD so
a fast-but-wrong kernel cannot hide behind a good ratio.

Usage:
    python3 benchmark/bench_attention_vs_ref.py                        # f16, eager+compile
    python3 benchmark/bench_attention_vs_ref.py --dtype f16,bf16,f32
    python3 benchmark/bench_attention_vs_ref.py --filter decode --reps 50
    python3 benchmark/bench_attention_vs_ref.py --mode eager
    python3 benchmark/bench_attention_vs_ref.py --train               # forward+backward
    python3 benchmark/bench_attention_vs_ref.py --audit                # split routing cost
    python3 benchmark/bench_attention_vs_ref.py --csv attention.csv
"""
import argparse
import csv
import time

import torch

import tensorplay as tp
import tensorplay.nn.functional as pF

DT = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
# Fused attention accumulates the softmax in fp32 and reorders the summation,
# so a bitwise match is not on the table; these are the fp16/bf16 rounding
# floors plus a little slack for different tile orders.
TOL = {torch.float16: 3e-2, torch.bfloat16: 6e-2, torch.float32: 1e-4}


def wrap(x):
    """Wrap once, outside every timed region.

    Building the wrapper costs host time that lands between the two event
    records, so a wrapper created inside the timed callable shows up as
    measured time -- a percent or so on the fastest cases here.
    """
    return tp.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))


def unwrap(x):
    return torch.from_dlpack(tp.to_dlpack(x))


def ours_fwd(tensors, is_causal, scale, gqa):
    return pF.scaled_dot_product_attention(
        tensors[0], tensors[1], tensors[2],
        is_causal=is_causal, scale=scale, enable_gqa=gqa)


def ref_fwd(tensors, is_causal, scale, gqa):
    return torch.nn.functional.scaled_dot_product_attention(
        tensors[0], tensors[1], tensors[2],
        is_causal=is_causal, scale=scale, enable_gqa=gqa)


def fused_fwd(tensors, is_causal):
    """The kernel entry point itself, bypassing the public routing.

    Timing this alongside the public call separates the cost of the kernel
    from the cost of whatever the public entry point decides to dispatch,
    which a single public-API number cannot show.
    """
    out, _lse = tp._C._scaled_dot_product_attention_with_lse(
        tensors[0], tensors[1], tensors[2], is_causal=is_causal)
    return out


def compile_ours(fn, mode):
    return tp.compiler.compile(fn, mode=mode) if mode else tp.compiler.compile(fn)


def compile_ref(fn, mode):
    return torch.compile(fn, mode=mode) if mode else torch.compile(fn)


def reset_reference_compiler():
    """Drop the reference compiler's per-shape guards between cases.

    Every case is a fresh shape for the same code object, so the guard set
    grows once per case and eventually trips the recompile limit -- past that
    point the compiled callable silently runs eager and the compile column
    stops measuring compilation at all.
    """
    torch._dynamo.reset()
    for name in ("recompile_limit", "accumulated_recompile_limit",
                 "cache_size_limit", "accumulated_cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 256)


def make_forward(forward, is_causal, scale, gqa):
    """A three-argument callable: a compiler needs a real signature to bind
    example inputs against, so the tensor triple is passed positionally
    rather than captured out of the enclosing scope."""
    def run(q, k, v):
        return forward((q, k, v), is_causal, scale, gqa)
    return run


def make_step(forward, params, is_causal, scale, gqa):
    """Forward plus a sum-reduction backward, with grads zeroed first so
    each sample pays the same write traffic."""
    def step(q, k, v):
        for p in params:
            p.grad = None
        forward((q, k, v), is_causal, scale, gqa).sum().backward()
    return step


def timeit(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best


def warm_compiled(compile_with, fn, args):
    """Build the compiled callable and pay its first-call cost.

    Compilation is host work that would otherwise land inside the first
    timed sample and swamp a sub-millisecond operator; the reported cost
    is the one-time price, not part of the steady-state number.
    """
    t0 = time.perf_counter()
    compiled = compile_with(fn)
    compiled(*args)
    torch.cuda.synchronize()
    return compiled, time.perf_counter() - t0


# Attention traffic: two matmuls (Q@K^T then P@V), two flops per multiply-add.
# Causal self-attention only visits the lower triangle, so the pair count is
# triangular rather than the full rectangle.
def flops(B, Hq, Sq, Skv, D, is_causal):
    pairs = (Sq * (Sq + 1)) // 2 if (is_causal and Sq == Skv) else Sq * Skv
    return 4.0 * B * Hq * pairs * D


# (label, B, Hq, Hkv, Sq, Skv, D, is_causal, scale, enable_gqa)
#
# Query and key/value head counts are separate so grouped-query shapes need
# no separate case kind.  Two families dominate real loads: prefill (a full
# causal pass over the prompt) and decode (one query row against the cached
# context), plus a short-sequence group that is launch-overhead bound.
CASES = [
    # ---- prefill, causal self-attention ----
    ("prefill/causal/B1H32S512D64",   1, 32, 32,  512,  512,  64, True,  None,  False),
    ("prefill/causal/B1H32S1kD64",    1, 32, 32, 1024, 1024,  64, True,  None,  False),
    ("prefill/causal/B1H32S2kD64",    1, 32, 32, 2048, 2048,  64, True,  None,  False),
    ("prefill/causal/B4H32S1kD128",   4, 32, 32, 1024, 1024, 128, True,  None,  False),
    ("prefill/causal/B2H32S2kD128",   2, 32, 32, 2048, 2048, 128, True,  None,  False),
    ("prefill/causal/B8H16S512D128",  8, 16, 16,  512,  512, 128, True,  None,  False),
    # ---- prefill, non-causal and explicit scale ----
    ("prefill/full/B1H32S1kD64",      1, 32, 32, 1024, 1024,  64, False, None,  False),
    ("prefill/full/B2H32S1kD128",     2, 32, 32, 1024, 1024, 128, False, None,  False),
    ("prefill/scaled/B2H16S1kD64",    2, 16, 16, 1024, 1024,  64, False, 0.125, False),
    # ---- grouped-query attention ----
    ("prefill/gqa8/B2H32x4S1kD128",   2, 32,  4, 1024, 1024, 128, True,  None,  True),
    ("prefill/gqa8/B4H32x4S2kD64",    4, 32,  4, 2048, 2048,  64, True,  None,  True),
    # ---- decode: one query row against the cached context ----
    ("decode/causal/B1H32x1kD128",    1, 32, 32,    1, 1024, 128, True,  None,  False),
    ("decode/causal/B8H32x2kD128",    8, 32, 32,    1, 2048, 128, True,  None,  False),
    ("decode/causal/B32H32x4kD128",  32, 32, 32,    1, 4096, 128, True,  None,  False),
    ("decode/full/B64H32x1kD64",     64, 32, 32,    1, 1024,  64, False, None,  False),
    ("decode/gqa8/B32H32x4S4kD128",  32, 32,  4,    1, 4096, 128, False, None,  True),
    # ---- short sequences: launch-overhead dominated ----
    ("short/causal/B64H16S128D64",   64, 16, 16,  128,  128,  64, True,  None,  False),
]


def build(case, dev, dtype):
    _label, B, Hq, Hkv, Sq, Skv, D = case[:7]
    g = torch.Generator(device=dev).manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g, device=dev, dtype=dtype)

    return r(B, Hq, Sq, D), r(B, Hkv, Skv, D), r(B, Hkv, Skv, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="f16",
                    help="comma list of f16,bf16,f32")
    ap.add_argument("--mode", default="eager,compile",
                    help="comma list of eager,compile")
    ap.add_argument("--filter", default="", help="substring of the case label")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", action="store_true",
                    help="time forward+backward instead of forward only")
    ap.add_argument("--audit", action="store_true",
                    help="also time the kernel entry point directly, to split "
                         "kernel cost from public-entry routing cost")
    ap.add_argument("--compile-mode", default=None,
                    help="mode string forwarded to both compilers")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    torch.cuda.init()
    reset_reference_compiler()
    tp.compiler.reset()
    modes = [m for m in args.mode.split(",") if m]
    print(f"device={torch.cuda.get_device_name(0)}  reps={args.reps}  "
          f"pass={'fwd+bwd' if args.train else 'fwd'}  modes={','.join(modes)}")
    if args.compile_mode:
        print(f"compile-mode={args.compile_mode}")

    head = f"{'case':30s} {'dt':>4s}"
    for m in modes:
        head += f" | {'ours-' + m:>10s} {'ref-' + m:>10s} {'ratio':>6s}"
    print(head + "  chk")
    print("-" * len(head))

    rows, slow = [], []
    for dt_name in [d for d in args.dtype.split(",") if d]:
        dtype, tol = DT[dt_name], TOL[DT[dt_name]]
        for case in CASES:
            label = case[0]
            if args.filter and args.filter not in label:
                continue
            _l, B, Hq, Hkv, Sq, Skv, D, is_causal, scale, gqa = case
            flop = flops(B, Hq, Sq, Skv, D, is_causal)

            # The composed path keeps a full score matrix resident, so a case
            # that just barely fits can starve the next one.  Hand the cache
            # back before each case instead of only after it.
            torch.cuda.empty_cache()
            if "compile" in modes:
                reset_reference_compiler()
                tp.compiler.reset()

            try:
                ref_t = build(case, args.device, dtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{label:30s} {dt_name:>4s} SKIP (out of memory building inputs)")
                continue

            if args.train:
                ours_t = tuple(
                    wrap(t.detach()).requires_grad_(True) for t in ref_t)
                ref_t = tuple(t.detach().requires_grad_(True) for t in ref_t)
            else:
                ours_t = tuple(wrap(t) for t in ref_t)

            # ---- correctness, forward only, on detached inputs ----
            note = "ok"
            try:
                with torch.no_grad():
                    a = unwrap(ours_fwd(tuple(wrap(t) for t in ref_t), is_causal, scale, gqa))
                    b = ref_fwd(ref_t, is_causal, scale, gqa)
                err = (a.float() - b.float()).abs().max().item()
                if not err <= tol:
                    note = f"BAD({err:.1e})"
            except Exception as exc:
                note = f"ERR({type(exc).__name__})"

            side_fwd = {"ours": ours_fwd, "ref": ref_fwd}
            side_in = {"ours": ours_t, "ref": ref_t}
            side_compile = {"ours": compile_ours, "ref": compile_ref}
            if args.train:
                side_run = {
                    s: make_step(side_fwd[s], side_in[s], is_causal, scale, gqa)
                    for s in side_fwd
                }
            else:
                side_run = {
                    s: make_forward(side_fwd[s], is_causal, scale, gqa)
                    for s in side_fwd
                }

            line = f"{label:30s} {dt_name:>4s}"
            row = {"case": label, "dtype": dt_name, "tflop": flop / 1e12}
            for m in modes:
                cell = {}
                for side in ("ours", "ref"):
                    run, inputs = side_run[side], side_in[side]
                    try:
                        if m == "eager":
                            cell[side] = (timeit(
                                lambda f=run, a=inputs: f(*a),
                                args.reps, args.warmup), None)
                        elif not args.train:
                            compiled, c = warm_compiled(
                                lambda f, s=side: side_compile[s](
                                    f, args.compile_mode),
                                run, inputs)
                            cell[side] = (timeit(
                                lambda f=compiled, a=inputs: f(*a),
                                args.reps, args.warmup), c)
                        else:
                            # The compiled region is the forward only.  A
                            # region that contains the backward pass comes
                            # back with its gradients dropped, so capturing
                            # the backward would time a no-op; autograd runs
                            # the backward outside the region instead.
                            compiled, c = warm_compiled(
                                lambda f, s=side: side_compile[s](
                                    f, args.compile_mode),
                                make_forward(side_fwd[side], is_causal,
                                             scale, gqa), inputs)
                            run = make_step(
                                lambda t, _a, _s, _g, _f=compiled: _f(*t),
                                inputs, is_causal, scale, gqa)
                            cell[side] = (timeit(
                                lambda f=run, a=inputs: f(*a),
                                args.reps, args.warmup), c)
                    except Exception as exc:
                        cell[side] = (float("nan"), None)
                        note = f"SKIP({type(exc).__name__})"
                        torch.cuda.empty_cache()
                t_ours, t_ref = cell["ours"][0], cell["ref"][0]
                if t_ours != t_ours or t_ref != t_ref:
                    line += f" | {'SKIP':>10s} {'-':>10s} {'-':>6s}"
                    continue
                ratio = t_ours / t_ref
                line += f" | {t_ours:10.4f} {t_ref:10.4f} {ratio:6.2f}"
                if m != "eager":
                    line += f" [{cell['ours'][1]:4.1f}/{cell['ref'][1]:4.1f}s]"
                row[f"ours_{m}"] = t_ours
                row[f"ref_{m}"] = t_ref
                row[f"ratio_{m}"] = ratio
                row[f"tflops_ours_{m}"] = flop / (t_ours * 1e-3) / 1e12
                row[f"tflops_ref_{m}"] = flop / (t_ref * 1e-3) / 1e12
                if ratio > 1.0:
                    slow.append((ratio, label, dt_name, m))

            # Routing audit: the same inputs through the kernel entry point
            # directly.  Only plain cases (no explicit scale, no grouped
            # heads) have a matching kernel signature.
            if args.audit and not args.train and scale is None and not gqa:
                try:
                    pub = timeit(
                        lambda a=ours_t: ours_fwd(a, is_causal, scale, gqa),
                        args.reps, args.warmup)
                    t_fused = timeit(
                        lambda a=ours_t: fused_fwd(a, is_causal),
                        args.reps, args.warmup)
                    row["fused_direct"] = t_fused
                    row["routing_overhead"] = pub / t_fused
                    try:
                        row["ref_audit"] = timeit(
                            lambda a=ref_t: ref_fwd(a, is_causal, scale, gqa),
                            args.reps, args.warmup)
                    except Exception:
                        pass
                except Exception:
                    pass

            print(line + f"  {note}")
            rows.append(row)
            del ref_t, ours_t, side_run, side_in
            torch.cuda.empty_cache()

    print(f"\nthroughput (TFLOP/s), ours / ref")
    for m in modes:
        print(f"  {m}:")
        for row in rows:
            a = row.get(f"tflops_ours_{m}")
            b = row.get(f"tflops_ref_{m}")
            if a is None:
                continue
            print(f"    {row['case']:30s} {row['dtype']:>4s} "
                  f"{a:7.1f} / {b:7.1f}")

    if slow:
        print("\n按差距排序（ours/ref，>1 即本侧更慢）：")
        for r, label, dt_name, m in sorted(slow, reverse=True)[:30]:
            print(f"  {r:5.2f}  {dt_name:>4s}  {m:<7s} {label}")

    audited = [r for r in rows if "fused_direct" in r]
    if audited:
        print("\nrouting audit -- public entry point vs the kernel it should "
              "have reached, and the reference for scale:")
        print(f"  {'case':30s} {'dt':>4s} {'public':>9s} {'kernel':>9s} "
              f"{'ref':>9s} {'pub/ker':>8s} {'ker/ref':>8s}")
        for row in audited:
            ker = row["fused_direct"]
            pub = row["routing_overhead"] * ker
            r = row.get("ref_audit")
            rtxt = f"{r:9.4f}" if r is not None else f"{'-':>9s}"
            rt = f"{ker / r:8.2f}" if r else f"{'-':>8s}"
            print(f"  {row['case']:30s} {row['dtype']:>4s} {pub:9.4f} "
                  f"{ker:9.4f} {rtxt} {pub / ker:8.2f} {rt}")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
