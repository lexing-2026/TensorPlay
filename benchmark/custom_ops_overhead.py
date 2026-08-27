"""Custom-operator dispatch overhead: TensorPlay vs torch.

Isolates the per-call framework cost of invoking a user-defined operator
in eager mode (the ``op(x)`` hot path), on top of an identical trivial
kernel body, against the frameworks' own baselines:

- ``python_fn``   : bare Python call — the theoretical floor
- ``direct``      : calling the registered kernel callable directly
- ``custom_op``   : full registration-aware dispatch (capture guard,
                    device-key selection, autograd gate, autocast gate)

Usage::

    python3 benchmark/custom_ops_overhead.py [--iters N]
"""

import argparse
import time

import tensorplay as tp


def bench_ns_per_call(fn, iters, warmup=2_000):
    for _ in range(warmup):
        fn()
    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    return (time.perf_counter_ns() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200_000)
    args = parser.parse_args()

    results = {}

    # ---- python floor -----------------------------------------------------
    def identity(x):
        return x

    x_tp = tp.tensor([1.0])
    results["python_fn"] = bench_ns_per_call(lambda: identity(x_tp), args.iters)

    # ---- tensorplay --------------------------------------------------------
    @tp.library.custom_op("benchtp::identity", mutates_args=())
    def tp_body(x):
        return tp.mul(x, 1.0)

    results["tensorplay/custom_op"] = bench_ns_per_call(
        lambda: tp_body(x_tp), args.iters
    )
    results["tensorplay/direct_kernel"] = bench_ns_per_call(
        lambda: tp.mul(x_tp, 1.0), args.iters
    )

    # ---- torch -------------------------------------------------------------
    try:
        import torch
    except ImportError:
        print("torch not installed; skipping comparison")
        torch = None

    if torch is not None:
        x_torch = torch.tensor([1.0])

        @torch.library.custom_op("benchtorch::identity", mutates_args=())
        def torch_body(x: torch.Tensor) -> torch.Tensor:
            return x * 1.0

        @torch_body.register_fake
        def _(x):
            return torch.empty_like(x)

        results["torch/custom_op"] = bench_ns_per_call(
            lambda: torch_body(x_torch), args.iters
        )
        results["torch/direct_kernel"] = bench_ns_per_call(
            lambda: x_torch * 1.0, args.iters
        )

    width = max(len(k) for k in results)
    print(f"\n{'path':<{width}}  ns/call")
    print("-" * (width + 10))
    for name, ns in results.items():
        print(f"{name:<{width}}  {ns:8.0f}")

    if torch is not None:
        ratio = results["torch/custom_op"] / results["tensorplay/custom_op"]
        print(f"\ntorch.custom_op / tensorplay.custom_op : {ratio:.2f}x")
        # Dispatch-layer overhead = custom_op call minus its own bare-kernel
        # cost; this isolates registration machinery from native-op speed.
        overhead_tp = (
            results["tensorplay/custom_op"] - results["tensorplay/direct_kernel"]
        )
        overhead_torch = (
            results["torch/custom_op"] - results["torch/direct_kernel"]
        )
        print(f"dispatch-layer overhead  tensorplay    : {overhead_tp:8.0f} ns/call")
        print(f"dispatch-layer overhead  torch         : {overhead_torch:8.0f} ns/call")
        print(f"torch overhead / tensorplay overhead   : {overhead_torch / overhead_tp:.2f}x")


if __name__ == "__main__":
    main()
