#!/usr/bin/env python3
"""CPU autocast benchmark.

Measures the AMP-critical paths where dispatch/cast overhead dominates:
  1. context-manager enter/exit micro (per-iteration cost)
  2. single linear under autocast (cache-hit weight-cast path)
  3. MLP inference loop under autocast with version-validated weight caching
  4. MLP train step under autocast (GradMode on, leaf caching both sides)
  5. raw fp32 -> bfloat16 cast throughput

Usage: python3 benchmark/autocast_bench.py [iters]
"""

import importlib
import sys
import time

import numpy as np


def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def make_mlp(mod, nn, dim=256, depth=8, seed=0):
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(depth):
        lin = nn.Linear(dim, dim)
        with mod.no_grad():
            lin.weight.copy_(
                mod.Tensor(rng.standard_normal((dim, dim), dtype=np.float32)))
            lin.bias.copy_(
                mod.Tensor(rng.standard_normal((dim,), dtype=np.float32) * 0.01))
        layers.append(lin)
    return nn.Sequential(*layers)


def run_framework(name, iters):
    mod = importlib.import_module(name)
    nn = importlib.import_module(f"{name}.nn")
    results = {}

    # ---- 1. enter/exit micro -------------------------------------------
    def cm():
        with mod.autocast("cpu"):
            pass
    results["ctx_enter_exit_us"] = bench(cm, iters=iters) * 1e6

    # ---- 2. single linear under autocast (cache-hit path) ---------------
    net = make_mlp(mod, nn, dim=512, depth=1, seed=1)
    x = mod.Tensor(np.random.rand(128, 512).astype(np.float32))
    with mod.autocast("cpu"):
        net(x)  # prime caches
    results["linear_cached_us"] = bench(lambda: net(x), iters=iters) * 1e6

    # ---- 3. MLP inference ------------------------------------------------
    mlp = make_mlp(mod, nn, dim=256, depth=8, seed=2)
    xi = mod.Tensor(np.random.rand(64, 256).astype(np.float32))

    def infer():
        with mod.autocast("cpu"):
            mlp(xi)
    results["mlp8_infer_ac_us"] = bench(infer, iters=iters) * 1e6
    results["mlp8_infer_fp32_us"] = bench(lambda: mlp(xi), iters=iters) * 1e6

    # ---- 4. MLP train step (manual SGD, framework-agnostic) --------------
    xt = mod.Tensor(np.random.rand(32, 256).astype(np.float32))
    yt = mod.Tensor(np.random.rand(32, 256).astype(np.float32))
    params = list(mlp.parameters())

    def train_step():
        for p in params:
            p.grad = None
        with mod.autocast("cpu"):
            loss = ((mlp(xt) - yt) ** 2).mean()
        loss.backward()
        with mod.no_grad():
            for p in params:
                if p.grad is not None:
                    p -= 0.01 * p.grad
    results["mlp8_train_us"] = bench(train_step, warmup=5,
                                     iters=max(10, iters // 2)) * 1e6

    # ---- 5. raw cast throughput -------------------------------------------
    big = mod.Tensor(np.random.rand(4096, 4096).astype(np.float32))
    t_cast = bench(lambda: big.to(mod.bfloat16), warmup=5,
                   iters=max(20, iters // 4))
    gb = 4096 * 4096 * (4 + 2) / 1e9
    results["cast_f32_bf16_GBps"] = gb / t_cast

    return results


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    # Optional second arg limits to one framework so each can run in its own
    # process (shared-process OMP pools pollute each other's timings).
    only = sys.argv[2] if len(sys.argv) > 2 else None
    names = ("tensorplay",) if only == "tensorplay" else             ("torch",) if only == "torch" else ("torch", "tensorplay")
    rows = {}
    for name in names:
        try:
            rows[name] = run_framework(name, iters)
        except Exception as e:  # noqa: BLE001
            print(f"{name} unavailable: {type(e).__name__}: {e}")

    if not rows:
        sys.exit(1)

    keys = list(next(iter(rows.values())).keys())
    hdr = f"{'metric':24s}" + "".join(f"{k:>14s}" for k in rows)
    if len(rows) == 2 and "torch" in rows and "tensorplay" in rows:
        hdr += f"{'tp/ref':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        line = f"{k:24s}"
        vals = []
        for fw in ("torch", "tensorplay"):
            v = rows.get(fw, {}).get(k)
            vals.append(v)
            line += f"{v:14.2f}" if v is not None else f"{'n/a':>14s}"
        if len(vals) == 2 and all(v is not None for v in vals) and vals[0]:
            line += f"{vals[1] / vals[0]:10.2f}x"
        print(line)


if __name__ == "__main__":
    main()
