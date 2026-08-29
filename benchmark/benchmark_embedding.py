"""Compare TensorPlay CUDA embedding with a reference runtime.

Run from the repository root:

    python benchmark/benchmark_embedding.py --quick

The timings include the operator allocation, but not input construction or
host/device copies. CUDA is synchronized around every measured region.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorplay as tp


CASES = (
    ("small-D8", 10_000, 8, 1_024),
    ("medium-D64", 50_000, 64, 4_096),
    ("wide-D768", 50_000, 768, 4_096),
    ("single-D1024", 50_000, 1_024, 1),
    ("wide-D2048", 16_384, 2_048, 8),
)


def synchronize() -> None:
    torch.cuda.synchronize()
    tp.cuda.synchronize()


def measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def to_tp(values, dtype) -> tp.Tensor:
    return tp.tensor(values, dtype=dtype).to("cuda")


def check_forward(tp_weight, tp_indices, torch_weight, torch_indices) -> None:
    tp_out = tp.embedding(tp_weight, tp_indices).to(tp.float32).cpu().numpy()
    torch_out = (
        torch.nn.functional.embedding(torch_indices, torch_weight)
        .to(torch.float32)
        .cpu()
        .numpy()
    )
    np.testing.assert_allclose(tp_out, torch_out, rtol=2e-3, atol=2e-3)


def run_case(name: str, vocab: int, dim: int, n_indices: int, iterations: int) -> None:
    indices = [i % vocab for i in range(n_indices)]
    tp_indices = to_tp(indices, tp.int64)
    torch_indices = torch.tensor(indices, dtype=torch.int64, device="cuda")

    tp_weight = tp.randn((vocab, dim), device="cuda")
    torch_weight = torch.randn((vocab, dim), device="cuda")
    # Use equal deterministic weights for the validation check without making the
    # timed path pay for a conversion.
    check_vocab = min(vocab, 16)
    check_indices = [i % check_vocab for i in range(min(n_indices, 32))]
    check_values = np.arange(check_vocab * dim, dtype=np.float32).reshape(check_vocab, dim)
    tp_check_weight = to_tp(check_values.tolist(), tp.float32)
    ref_check_weight = torch.from_numpy(check_values).to("cuda")
    check_forward(
        tp_check_weight,
        to_tp(check_indices, tp.int64),
        ref_check_weight,
        torch.tensor(check_indices, dtype=torch.int64, device="cuda"),
    )

    tp_us = measure(
        lambda: tp.embedding(tp_weight, tp_indices), warmup=10, iterations=iterations
    )
    torch_us = measure(
        lambda: torch.nn.functional.embedding(torch_indices, torch_weight),
        warmup=10,
        iterations=iterations,
    )
    print(f"{name:16s} forward {tp_us:10.2f} us  {torch_us:10.2f} us  TP/ref {tp_us / torch_us:6.3f}x")

    tp_grad = tp.randn((n_indices, dim), device="cuda")
    torch_grad = torch.randn((n_indices, dim), device="cuda")
    tp_backward = lambda: tp.embedding_dense_backward(
        tp_grad, tp_indices, vocab, -1, False
    )
    torch_backward = lambda: torch.ops.aten.embedding_dense_backward(
        torch_grad, torch_indices, vocab, -1, False
    )
    tp_bwd_us = measure(tp_backward, warmup=10, iterations=iterations)
    torch_bwd_us = measure(torch_backward, warmup=10, iterations=iterations)
    print(f"{'':16s} backward {tp_bwd_us:10.2f} us  {torch_bwd_us:10.2f} us  TP/ref {tp_bwd_us / torch_bwd_us:6.3f}x")

    del tp_weight, torch_weight, tp_indices, torch_indices, tp_grad, torch_grad
    gc.collect()
    synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use fewer timed iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not tp.cuda.is_available():
        raise SystemExit("Both reference framework and TensorPlay CUDA must be available")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("case             direction       TensorPlay         ref       ratio")
    for name, vocab, dim, n_indices in CASES:
        iterations = 20 if args.quick else 100
        if n_indices == 1:
            iterations *= 2
        run_case(name, vocab, dim, n_indices, iterations)


if __name__ == "__main__":
    main()
