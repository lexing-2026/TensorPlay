"""TensorPlay on a Cambricon MLU against torch + torch_mlu on the same card.

The TensorPlay side loads the standalone ``tensorplay-mlu`` package
(https://github.com/ohtensorplay/tensorplay-mlu); make sure it is importable
(``pip install tensorplay-mlu`` or ``PYTHONPATH`` to a checkout).

Each framework runs in its own process (one framework per process keeps
allocators, thread pools and driver state from interfering), on identical
workloads: single operators at a few sizes and dtypes, and a training step
of an MLP.  Times are medians over timed iterations after warmup, with the
device synchronized around each measurement.

    python benchmark/bench_mlu_vs_torch.py            # both, then compare
    python benchmark/bench_mlu_vs_torch.py --only tp  # one side, JSON out

The frameworks default to different float32 matmul precisions (torch_mlu
allows TF32, TensorPlay keeps full float32); BENCH_FP32_PRECISION=high (or
highest) sets the same one on both sides.  A card whose driver reports a
generic product name needs the name fixed for the other side too
(TORCH_PYTHON can point at a wrapper that does so); without it that side
times wrong results or faults, and the MLP cases check against the host.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

WARMUP = 10
ITERS = 50


class Framework:
    def __init__(self, name: str):
        self.name = name
        if name == "torch":
            import torch
            import torch_mlu  # noqa: F401

            self.m = torch
            self.sync = torch.mlu.synchronize
            self.nn = torch.nn
            self.optim = torch.optim
            self.F = torch.nn.functional
            precision = os.environ.get("BENCH_FP32_PRECISION")
            if precision:
                torch.set_float32_matmul_precision(precision)
                torch.backends.mlu.matmul.allow_tf32 = precision != "highest"
        else:
            import tensorplay as tp
            import tensorplay_mlu

            tensorplay_mlu.load()
            self.m = tp
            self.sync = tp.mlu.synchronize
            self.nn = tp.nn
            self.optim = tp.optim
            self.F = tp.nn.functional
            precision = os.environ.get("BENCH_FP32_PRECISION")
            if precision:
                tp.set_float32_matmul_precision(precision)
        self.device = "mlu"

    def dtype(self, name: str):
        return {"fp32": self.m.float32, "fp16": self.m.float16, "bf16": self.m.bfloat16}[name]

    def randn(self, *shape, dtype="fp32"):
        return self.m.randn(*shape).to(self.device).to(self.dtype(dtype))


def timeit(fx: Framework, fn, warmup=WARMUP, iters=ITERS) -> float:
    for _ in range(warmup):
        fn()
    fx.sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        fx.sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def gemm_size_ok(fx: Framework, n: int, dtype: str) -> bool:
    """Whether a size-n product gives the right answer on this card.

    A card whose driver reports a generic product name runs CNNL's
    unknown-device kernels, which return wrong data (unrealistically fast)
    for some sizes; timing them would only measure the no-op.
    """
    try:
        a = fx.m.ones(n, n, device=fx.device).to(fx.dtype(dtype))
        y = fx.m.mm(a, a)
        return abs(float(y[0, 0].cpu()) - n) < 0.5
    except Exception:
        return False


def op_cases(fx: Framework):
    m, F = fx.m, fx.F
    cases = {}
    for dt in ("fp32", "fp16"):
        for n in (1024, 4096):
            if not gemm_size_ok(fx, n, dt):
                print(f"{fx.name}: skipping the {n} products in {dt}: wrong result on this "
                      f"card", file=sys.stderr)
                continue
            a, b = fx.randn(n, n, dtype=dt), fx.randn(n, n, dtype=dt)
            cases[f"mm {n} {dt}"] = lambda a=a, b=b: m.mm(a, b)
            cases[f"mm_tn {n} {dt}"] = lambda a=a, b=b: m.mm(a.t(), b)
        x = fx.randn(4096, 4096, dtype=dt)
        y = fx.randn(4096, 4096, dtype=dt)
        bias = fx.randn(4096, dtype=dt)
        cases[f"add 4096^2 {dt}"] = lambda x=x, y=y: x + y
        cases[f"add_bcast 4096^2 {dt}"] = lambda x=x, b=bias: x + b
        cases[f"mul_scalar 4096^2 {dt}"] = lambda x=x: x * 0.5
        cases[f"add_ 4096^2 {dt}"] = lambda x=x, y=y: x.add_(y, alpha=1e-3)
        cases[f"relu 4096^2 {dt}"] = lambda x=x: F.relu(x)
        cases[f"gelu 4096^2 {dt}"] = lambda x=x: F.gelu(x)
        cases[f"exp 4096^2 {dt}"] = lambda x=x: m.exp(x)
        cases[f"softmax 4096^2 {dt}"] = lambda x=x: m.softmax(x, dim=-1)
        cases[f"sum_dim 4096^2 {dt}"] = lambda x=x: x.sum(dim=-1)
        cases[f"sum_all 4096^2 {dt}"] = lambda x=x: x.sum()
        b3a, b3b = fx.randn(32, 512, 64, dtype=dt), fx.randn(32, 64, 512, dtype=dt)
        cases[f"bmm 32x512x64x512 {dt}"] = lambda a=b3a, b=b3b: m.bmm(a, b)
    x = fx.randn(4096, 4096)
    cases["cast fp32->fp16 4096^2"] = lambda x=x: x.to(m.float16)
    host = m.randn(4096, 4096)
    cases["h2d 64MiB"] = lambda h=host: h.to(fx.device)
    cases["d2h 64MiB"] = lambda x=x: x.cpu()
    small = fx.randn(16, 16)
    cases["add 16x16 (launch overhead)"] = lambda s=small: s + s
    return cases


class WrongResult(RuntimeError):
    pass


def mlp_step_case(fx: Framework, dtype="fp32", batch=256, width=1024, hidden=4096):
    nn = fx.nn

    def build():
        return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    host_model = build()
    model = build()
    model.load_state_dict(host_model.state_dict())
    model = model.to(fx.device).to(fx.dtype(dtype))
    opt = fx.optim.SGD(model.parameters(), lr=1e-3)
    x = fx.randn(batch, width, dtype=dtype)
    target = fx.randn(batch, width, dtype=dtype)

    # A device that returns wrong data would otherwise be timed as fast.
    try:
        want = host_model(x.cpu().float())
        got = model(x).cpu().float()
        err = float((got - want).abs().max()) / max(float(want.abs().max()), 1e-6)
        problem = None if err < (1e-2 if dtype == "fp32" else 5e-2) else \
            f"forward differs from the host by {err:.3g} (relative)"
    except Exception as exc:  # a device fault while checking
        problem = f"forward failed: {type(exc).__name__}: {exc}".splitlines()[0][:120]
    if problem is not None:
        def wrong():
            raise WrongResult(problem)
        return {
            f"mlp fwd b{batch} {width}-{hidden} {dtype}": wrong,
            f"mlp train step b{batch} {width}-{hidden} {dtype}": wrong,
        }

    def step():
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()

    def forward():
        return model(x)

    return {
        f"mlp fwd b{batch} {width}-{hidden} {dtype}": forward,
        f"mlp train step b{batch} {width}-{hidden} {dtype}": step,
    }


def interpreter(name: str) -> str:
    """The Python that has this framework: each side may live in its own venv."""
    return os.environ.get("TORCH_PYTHON" if name == "torch" else "TP_PYTHON",
                          sys.executable)


def case_labels(name: str) -> list[str]:
    fx = Framework(name)
    cases = op_cases(fx)
    for dt in ("fp32", "fp16"):
        cases.update(mlp_step_case(fx, dt))
    return list(cases)


def run_one(name: str, filter_text: str | None) -> dict[str, float]:
    fx = Framework(name)
    cases = op_cases(fx)
    for dt in ("fp32", "fp16"):
        cases.update(mlp_step_case(fx, dt))
    results = {}
    for label, fn in cases.items():
        if filter_text and filter_text not in label:
            continue
        try:
            results[label] = timeit(fx, fn)
        except Exception as exc:  # report and keep going
            results[label] = f"error: {type(exc).__name__}: {exc}".splitlines()[0][:160]
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["tp", "torch"])
    parser.add_argument("--list", choices=["tp", "torch"], help="print the case labels")
    parser.add_argument("-k", dest="filter", help="run cases whose label contains this text")
    parser.add_argument("--per-case", action="store_true",
                        help="run every case in its own process; a device fault in one case "
                             "then leaves the others measurable")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(case_labels(args.list)))
        return 0
    if args.only:
        print(json.dumps(run_one(args.only, args.filter)))
        return 0

    sides = {}
    for name in ("torch", "tp"):
        if args.per_case:
            py = interpreter(name)
            listing = subprocess.run([py, os.path.abspath(__file__), "--list", name],
                                     capture_output=True, text=True)
            labels = json.loads(listing.stdout.strip().splitlines()[-1])
            results = {}
            for label in labels:
                if args.filter and args.filter not in label:
                    continue
                proc = subprocess.run(
                    [py, os.path.abspath(__file__), "--only", name, "-k", label],
                    capture_output=True, text=True)
                lines = proc.stdout.strip().splitlines()
                if proc.returncode != 0 or not lines:
                    results[label] = "error: the case process failed"
                else:
                    results.update(json.loads(lines[-1]))
            sides[name] = results
            continue
        cmd = [interpreter(name), os.path.abspath(__file__), "--only", name]
        if args.filter:
            cmd += ["-k", args.filter]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines = proc.stdout.strip().splitlines()
        if proc.returncode != 0 or not lines:
            print(f"{name} failed:\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
            return 1
        sides[name] = json.loads(lines[-1])

    labels = list(sides["torch"]) + [k for k in sides["tp"] if k not in sides["torch"]]
    print(f"{'case':44s} {'torch ms':>10s} {'tp ms':>10s} {'tp/torch':>9s}")
    for label in labels:
        a, b = sides["torch"].get(label), sides["tp"].get(label)
        if isinstance(a, float) and isinstance(b, float):
            print(f"{label:44s} {a:10.3f} {b:10.3f} {b / a:9.2f}")
        else:
            print(f"{label:44s} {a!s:>10.10} {b!s:>10.10}")
            for side, value in (("torch", a), ("tp", b)):
                if isinstance(value, str):
                    print(f"    {side}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
