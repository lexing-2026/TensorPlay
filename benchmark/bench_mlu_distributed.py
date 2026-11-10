"""Parallel efficiency of the CNCL backend on MLU.

Runs one data-parallel training step and measures how it scales from a
single process to one process per card.  The same *global* batch is used in
both runs: with ``N`` processes each rank takes ``global_batch / N``
samples and averages its gradients with ``all_reduce``, so the wall time of
a step should stay roughly flat while the throughput grows with ``N``.

Launch it once per rank.  A single-process run needs only ``WORLD_SIZE=1``
and ``RANK=0``; a two-process run puts rank 0 on the first host and rank 1
on the second, with ``MASTER_ADDR`` pointing at rank 0::

    RANK=0 WORLD_SIZE=1 python benchmark/bench_mlu_distributed.py --json out.json

Results are printed as one JSON object; ``--json FILE`` also writes it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

WARMUP = 5
STEPS = 20


def build_model(tp, width: int, hidden: int):
    return tp.nn.Sequential(
        tp.nn.Linear(width, hidden),
        tp.nn.GELU(),
        tp.nn.Linear(hidden, width),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    import tensorplay as tp
    import tensorplay_mlu

    tensorplay_mlu.load()
    import tensorplay.distributed as dist

    backend = os.environ.get("CNCL_BACKEND", "cncl")
    dist.init_process_group(backend, init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()

    dtypes = {"fp32": tp.float32, "fp16": tp.float16, "bf16": tp.bfloat16}
    dtype = dtypes[args.dtype]

    # Split the global batch across ranks; the workload per rank shrinks as
    # ranks are added, which is what makes the throughput scale.
    local_batch = max(1, args.global_batch // world)

    model = build_model(tp, args.width, args.hidden).to("mlu").to(dtype)
    opt = tp.optim.SGD(model.parameters(), lr=1e-3)
    x = tp.randn(local_batch, args.width, device="mlu").to(dtype)
    target = tp.randn(local_batch, args.width, device="mlu").to(dtype)

    def step():
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world
        opt.step()

    for _ in range(args.warmup):
        step()
    tp.mlu.synchronize()

    samples = []
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        tp.mlu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    step_ms = statistics.median(samples)
    throughput = args.global_batch * 1e3 / step_ms
    result = {
        "rank": rank,
        "world_size": world,
        "global_batch": args.global_batch,
        "local_batch": local_batch,
        "dtype": args.dtype,
        "width": args.width,
        "hidden": args.hidden,
        "step_ms": step_ms,
        "samples_per_s": throughput,
        "steps_samples_ms": samples,
    }
    if rank == 0:
        print(json.dumps(result), flush=True)
    if args.json and rank == 0:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
