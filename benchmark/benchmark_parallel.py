# Benchmark: P0 intra-op parallel layer (parallel_for + thread pool)
#
# Validates the P0 target:
#   small-operator 8-thread overhead <= 1.5x single thread
# and checks the thread pool actually engages on large operators.
#
# Usage:
#   python benchmark/benchmark_parallel.py
#   python benchmark/benchmark_parallel.py --iters 100000
#
# Compare against PyTorch (torch.set_num_threads + torch.add).

import argparse
import os
import sys
import time

import tensorplay as tp

_CUDA_LIBS = (
    "/home/bluemoon/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib:"
    "/home/bluemoon/miniconda3/lib/python3.13/site-packages/nvidia/cudnn/lib:"
    "/home/bluemoon/miniconda3/pkgs/cuda-cudart-13.0.96-ha533d76_1/lib:"
    "/home/bluemoon/miniconda3/pkgs/libcurand-10.4.3.29-h7a5d6f9_0/lib"
)


def bench(fn, iters):
    for _ in range(2000):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e9  # ns/call


def bench_tp(nthreads, n_elem, iters, one_dnn=True):
    tp.set_num_threads(nthreads)
    _C = sys.modules["tensorplay._C"]
    if _C.is_mkldnn_enabled() != one_dnn:
        _C.set_mkldnn_enabled(one_dnn)
    a = tp.tensor([1.0] * n_elem)
    b = tp.tensor([2.0] * n_elem)
    ns = bench(lambda: a + b, iters)
    print(f"TensorPlay n={n_elem:>9} threads={nthreads} onednn={one_dnn}: {ns:9.1f} ns/call")
    return ns


def bench_torch(nthreads, n_elem, iters):
    import torch

    torch.set_num_threads(nthreads)
    a = torch.tensor([1.0] * n_elem)
    b = torch.tensor([2.0] * n_elem)
    ns = bench(lambda: a + b, iters)
    print(f"PyTorch    n={n_elem:>9} threads={nthreads}:              {ns:9.1f} ns/call")
    return ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--large-n", type=int, default=4_000_000)
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()

    for lib in _CUDA_LIBS.split(":"):
        if lib not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = (
                lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                if os.environ.get("LD_LIBRARY_PATH")
                else lib
            )

    print(f"TensorPlay: {tp.__file__}")
    print(f"default get_num_threads: {tp.get_num_threads()}")
    print()

    # --- P0: 1-element add, 8 threads vs 1 thread ---
    tp1 = bench_tp(1, 1, args.iters, one_dnn=True)
    tp8 = bench_tp(8, 1, args.iters, one_dnn=True)
    print()
    print("=== P0 check: 8-thread / 1-thread overhead, 1-element add ===")
    print(f"TensorPlay ratio (8t/1t): {tp8 / tp1:.3f}x  (target <= 1.5x)  "
          f"{'PASS' if tp8 / tp1 <= 1.5 else 'FAIL'}")
    if not args.skip_torch:
        import torch

        tr1 = bench_torch(1, 1, args.iters)
        tr8 = bench_torch(8, 1, args.iters)
        print(f"PyTorch    ratio (8t/1t): {tr8 / tr1:.3f}x")
        print(f"TensorPlay 8t vs PyTorch 8t: {tp8 / tr8:.2f}x")
        print(f"TensorPlay 1t vs PyTorch 1t: {tp1 / tr1:.2f}x")

    # --- Large add: pool should engage and speed up ---
    N = args.large_n
    tp.set_num_threads(8)
    a = tp.tensor([1.0] * N)
    b = tp.tensor([2.0] * N)
    r = a + b
    ok = all(x == 3.0 for x in r.tolist())
    print()
    print(f"large add correctness (n={N}, 8 threads): {'OK' if ok else 'FAILED'}")
    for onednn in (True, False):
        t8 = bench_tp(8, N, 100, one_dnn=onednn)
        t1 = bench_tp(1, N, 100, one_dnn=onednn)
        print(f"large add (onednn={onednn}): 8t/1t ratio = {t8 / t1:.2f}x "
              f"({'parallel engages' if t8 / t1 < 1.0 else 'no speedup'})")
    _C = sys.modules["tensorplay._C"]
    _C.set_mkldnn_enabled(True)


if __name__ == "__main__":
    main()