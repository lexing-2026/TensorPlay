"""CPU fp32 train-step benchmark.

Uses layer shapes that dominate a Transformer/MLP train step.
Usage: python3 benchmark/bench_train_fp32_cpu.py [--threads N] [--reps R]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F_torch

import tensorplay as tp
from tensorplay.nn import functional as F_tp


def timeit_min(fn, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


SHAPES = [
    # (tag, batch_tokens M, in_features K, out_features N)
    ("qkv-proj",   512, 4096, 4096),
    ("mlp-up",     512, 4096, 11008),
    ("mlp-down",   512, 11008, 4096),
    ("small-net",  128, 1024, 1024),
    ("decode-1",   1,   4096, 4096),
]


def run_framework(kind, m, k, n, reps):
    if kind == "torch":
        x = torch.randn(m, k)
        w = torch.randn(n, k)
        b = torch.randn(n)

        def fwd():
            return F_torch.linear(x, w, b)

        def step():
            xr = torch.tensor(x.detach(), requires_grad=True)
            wr = torch.tensor(w.detach(), requires_grad=True)
            br = torch.tensor(b.detach(), requires_grad=True)
            y = F_torch.linear(xr, wr, br)
            torch.autograd.grad(y.pow(2).sum(), [xr, wr, br])

        return (timeit_min(fwd, reps), timeit_min(step, reps))
    else:
        x = tp.randn((m, k))
        w = tp.randn((n, k))
        b = tp.randn((n,))

        def fwd():
            return F_tp.linear(x, w, b)

        def step():
            xr = x.clone()
            wr = w.clone()
            br = b.clone()
            xr.requires_grad = True
            wr.requires_grad = True
            br.requires_grad = True
            y = F_tp.linear(xr, wr, br)
            gy = y.pow(2).sum()
            gy.backward()

        return (timeit_min(fwd, reps), timeit_min(step, reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
        if hasattr(tp, "set_num_threads"):
            tp.set_num_threads(args.threads)
    print(f"threads: ref={torch.get_num_threads()} "
          f"tp={tp.get_num_threads() if hasattr(tp, 'get_num_threads') else '?'}")
    print(f"{'layer':>10} {'M':>5} {'K':>6} {'N':>6} "
          f"{'fwd_tp_ms':>10} {'fwd_th_ms':>10} {'step_tp_ms':>11} {'step_th_ms':>11}")
    behind = []
    for tag, m, k, n in SHAPES:
        # warm both frameworks once on this shape before timing anything
        run_framework("torch", m, k, n, 2)
        run_framework("tp", m, k, n, 2)
        tp_fwd, tp_step = run_framework("tp", m, k, n, args.reps)
        th_fwd, th_step = run_framework("torch", m, k, n, args.reps)
        print(f"{tag:>10} {m:>5} {k:>6} {n:>6} "
              f"{tp_fwd*1e3:>10.3f} {th_fwd*1e3:>10.3f} "
              f"{tp_step*1e3:>11.3f} {th_step*1e3:>11.3f}")
        if tp_fwd > 1.15 * th_fwd or tp_step > 1.15 * th_step:
            behind.append(tag)
    if behind:
        print("BEHIND ref (>15%):", ", ".join(behind))


if __name__ == "__main__":
    main()
