"""Optimizer step() benchmark: TensorPlay vs PyTorch.

Compares optimizer.step() wall time for identical parameter groups across
the four execution paths torch exposes (default / foreach / single-tensor /
fused).  Run on CPU locally or on a CUDA box:

    python3 benchmark/bench_optimizer.py                 # cpu
    python3 benchmark/bench_optimizer.py --device cuda   # needs torch+cuda

The parameter mix mirrors a small transformer: many (128,128) blocks plus
a few (1024,1024) matrices (~25 MiB of fp32 state per optimizer).
"""

import argparse
import sys
import time

# Load torch first on the CUDA host: TensorPlay and the vendored torch build
# may expose different CUDA runtime versions.
import torch
import tensorplay as tp

SHAPES = [(128, 128)] * 100 + [(1024, 1024)] * 4


def make_params(lib, device):
    ps = [lib.nn.Parameter(lib.randn(list(s), device=device))
          for s in SHAPES]
    for p in ps:
        p.grad = lib.randn_like(p)
    return ps


def bench(opt, lib, device, iters=30, warmup=8):
    synchronize = None
    if device == "cuda":
        synchronize = torch.cuda.synchronize if lib is torch else tp.cuda.synchronize
    for _ in range(warmup):
        opt.step()
    if synchronize is not None:
        synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.step()
    if synchronize is not None:
        synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


CASES = [
    ("Adam", dict(lr=1e-3)),
    ("AdamW", dict(lr=1e-3, weight_decay=0.1)),
    ("SGD", dict(lr=1e-2, momentum=0.9)),
    ("Adagrad", dict(lr=5e-2)),
    ("RMSprop", dict(lr=1e-2)),
    ("Adadelta", dict(lr=1.0)),
    ("Adamax", dict(lr=1e-2)),
    ("ASGD", dict(lr=1e-1)),
    ("NAdam", dict(lr=1e-2)),
    ("RAdam", dict(lr=1e-2)),
    ("Rprop", dict(lr=1e-2)),
    ("Adafactor", dict(lr=1e-2, beta2_decay=-0.8)),
    ("Muon", dict(lr=1e-3, weight_decay=0.1)),
]

IMPLS = [({}, "default"), ({"foreach": True}, "foreach"),
         ({"foreach": False}, "single"), ({"fused": True}, "fused")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; nothing to compare against")
        return 1

    print(f"device={args.device}  params={len(SHAPES)} tensors, "
          f"{sum(a * b for a, b in SHAPES) * 4 / 2**20:.1f} MiB fp32\n")

    header = f"{'case':30s} {'tp ms':>10s} {'torch ms':>10s} {'speedup':>8s}"
    print(header)
    print("-" * len(header))

    failures = []
    for name, kw in CASES:
        for impl_kw, label in IMPLS:
            row = f"{name:10s}/{label:8s}"
            res = {}
            for tag, lib in (("tp", tp), ("torch", __import__("torch"))):
                try:
                    ps = make_params(lib, args.device)
                    opt = getattr(lib.optim, name)(ps, **kw, **impl_kw)
                    res[tag] = bench(opt, lib, args.device, iters=args.iters)
                except Exception as e:  # noqa: BLE001
                    res[tag] = None
                    msg = str(e).split("\n")[0][:60]
                    if "unexpected keyword" not in msg:
                        failures.append(f"{row} [{tag}] {type(e).__name__}: {msg}")
            tp_ms, th_ms = res["tp"], res["torch"]
            if tp_ms is not None and th_ms is not None:
                print(f"{row:30s} {tp_ms:10.2f} {th_ms:10.2f} {th_ms/tp_ms:7.2f}x")
            else:
                tps = f"{tp_ms:.2f}" if tp_ms is not None else "ERR"
                ths = f"{th_ms:.2f}" if th_ms is not None else "ERR"
                print(f"{row:30s} {tps:>10s} {ths:>10s}")

    if failures:
        print("\nerrors:")
        for f in failures:
            print(" ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
