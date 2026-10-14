"""Throughput benchmark for the fused native SNN operators.

Measures the end-to-end time (forward + backward) of:

* ``snn.lif_seq`` -- the fused multi-step operator (time loop inside one
  native CPU/CUDA kernel);
* ``snn.lif_step`` in a Python time loop -- the equivalent computation that
  launches one kernel per time step.

Launch a single configuration with::

    python benchmark/snn_benchmark.py --device cpu --T 16 --B 128 --N 512

or run the full matrix with ``--all``.  Results are printed in spikes per
second (higher is better).
"""

import argparse
import os
import sys
import time

# Allow running from a source checkout without installing the package.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import tensorplay as tp
from tensorplay import snn


def timeit(fn, warmup=5, iters=30):
    """Return the best (median-of-3) run time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def bench_lif_seq(device, T, B, N):
    x = tp.rand(T, B, N, device=device, requires_grad=True)
    v_init = tp.zeros(B, N, device=device)
    decay = tp.tensor(0.8, device=device)
    threshold = tp.tensor(1.0, device=device)
    surrogate = snn.SigmoidSurrogate(alpha=2.0)

    def fused():
        if x.grad is not None:
            x.grad.zero_()
        spikes, v_final = snn.lif_seq(
            x, v_init=v_init, decay=decay, threshold=threshold,
            v_reset=None, surrogate=surrogate,
        )
        (spikes.sum() + v_final.sum()).backward()

    def looped():
        if x.grad is not None:
            x.grad.zero_()
        v = v_init
        refs = []
        for t in range(T):
            spike, v = snn.lif_step(
                x[t], v, decay=decay, threshold=threshold,
                v_reset=None, surrogate=surrogate,
            )
            refs.append(spike)
        refs = tp.stack(refs)
        (refs.sum() + v.sum()).backward()

    t_fused = timeit(fused)
    t_loop = timeit(looped)
    n_spikes = T * B * N
    return {
        "device": device,
        "T": T,
        "B": B,
        "N": N,
        "fused_ms": t_fused * 1e3,
        "loop_ms": t_loop * 1e3,
        "speedup": t_loop / t_fused,
        "fused_Mspikes_s": n_spikes / t_fused / 1e6,
        "loop_Mspikes_s": n_spikes / t_loop / 1e6,
    }


def bench_surrogate(device, n):
    x = tp.rand(n, device=device)

    def run():
        snn.surrogate_grad(x, surrogate=1, alpha=4.0)

    t = timeit(run)
    return {"device": device, "n": n, "ms": t * 1e3,
            "G_elems_s": n / t / 1e9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if tp.cuda.is_available() else "cpu")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        configs = [
            ("cpu", 8, 32, 256),
            ("cpu", 32, 64, 512),
            ("cpu", 100, 32, 1024),
        ]
        if tp.cuda.is_available():
            configs += [
                ("cuda", 8, 32, 256),
                ("cuda", 32, 64, 512),
                ("cuda", 100, 32, 1024),
            ]
        for device, T, B, N in configs:
            r = bench_lif_seq(device, T, B, N)
            print(
                f"{r['device']:>4} T={r['T']:<4} B={r['B']:<4} N={r['N']:<5} "
                f"fused={r['fused_ms']:8.3f}ms loop={r['loop_ms']:8.3f}ms "
                f"speedup={r['speedup']:6.2f}x  "
                f"({r['fused_Mspikes_s']:7.1f} Mspikes/s fused)"
            )
        return

    device = args.device
    if device == "cuda" and not tp.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine")
    r = bench_lif_seq(device, args.T, args.B, args.N)
    print(
        f"device={r['device']} T={r['T']} B={r['B']} N={r['N']}\n"
        f"  fused: {r['fused_ms']:.3f} ms  ({r['fused_Mspikes_s']:.1f} Mspikes/s)\n"
        f"  loop : {r['loop_ms']:.3f} ms  ({r['loop_Mspikes_s']:.1f} Mspikes/s)\n"
        f"  speedup: {r['speedup']:.2f}x"
    )
    s = bench_surrogate(device, 1 << 24)
    print(
        f"surrogate_grad ({device}): {s['ms']:.3f} ms  "
        f"({s['G_elems_s']:.1f} Gelems/s)"
    )


if __name__ == "__main__":
    main()