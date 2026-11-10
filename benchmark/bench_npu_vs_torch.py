"""TensorPlay on an Ascend NPU against torch + torch_npu on the same card.

Each framework runs in its own process (one framework per process keeps
allocators, thread pools and driver state from interfering), on identical
workloads: single operators at a few sizes and dtypes, and a training step
of an MLP.  Times are medians over timed iterations after warmup, with the
device synchronized around each measurement.

    python benchmark/bench_npu_vs_torch.py            # both, then compare
    python benchmark/bench_npu_vs_torch.py --only tp  # one side, JSON out
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
            import torch_npu  # noqa: F401

            self.m = torch
            self.sync = torch.npu.synchronize
            self.nn = torch.nn
            self.optim = torch.optim
            self.F = torch.nn.functional
        else:
            import tensorplay as tp
            import tensorplay_npu

            tensorplay_npu.load()
            self.m = tp
            self.sync = tp.npu.synchronize
            self.nn = tp.nn
            self.optim = tp.optim
            self.F = tp.nn.functional
        self.device = "npu"

    def dtype(self, name: str):
        return {"fp32": self.m.float32, "fp16": self.m.float16, "bf16": self.m.bfloat16}[name]

    def randn(self, *shape, dtype="fp32"):
        return self.m.randn(*shape, dtype=self.dtype(dtype), device=self.device)


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


def op_cases(fx: Framework):
    m, F = fx.m, fx.F
    cases = {}
    for dt in ("fp32", "fp16"):
        for n in (1024, 4096):
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


def mlp_step_case(fx: Framework, dtype="fp32", batch=256, width=1024, hidden=4096):
    nn = fx.nn
    model = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
    model = model.to(fx.device).to(fx.dtype(dtype))
    opt = fx.optim.SGD(model.parameters(), lr=1e-3)
    x = fx.randn(batch, width, dtype=dtype)
    target = fx.randn(batch, width, dtype=dtype)

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
    parser.add_argument("-k", dest="filter", help="run cases whose label contains this text")
    args = parser.parse_args()
    if args.only:
        print(json.dumps(run_one(args.only, args.filter)))
        return 0

    sides = {}
    for name in ("torch", "tp"):
        cmd = [sys.executable, os.path.abspath(__file__), "--only", name]
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
