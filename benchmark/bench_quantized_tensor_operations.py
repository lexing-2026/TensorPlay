"""Measure quantized tensor operations with identical layouts and thread counts."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from bench_quantized_native import measure, np, tp, torch


def run(repetitions):
    rng = np.random.default_rng(906)
    results = []

    def pair(shape):
        codes = rng.integers(0, 256, size=shape, dtype='uint8')
        return (tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes), .07, 119),
                torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .07, 119))

    def record(name, x, r, native, reference, tolerance=0):
        a, b = native(), reference()
        if tuple(a.shape) != tuple(b.shape) or a.q_scale() != b.q_scale() or a.q_zero_point() != b.q_zero_point():
            raise AssertionError(f'{name}: output shape or quantizer differs')
        error = float(np.max(np.abs(a.int_repr().numpy().astype('float64') -
                                   b.int_repr().numpy().astype('float64')), initial=0))
        if error > tolerance:
            raise AssertionError(f'{name}: code error {error} exceeds {tolerance}')
        native_us = measure(native, lambda: None, repetitions)
        reference_us = measure(reference, lambda: None, repetitions)
        results.append(dict(operator=name, device='cpu', shape=list(x.shape), strides=list(x.stride()),
                            native_us=native_us, reference_us=reference_us,
                            speedup=reference_us / native_us, max_error=error))

    for width in (256, 8193):
        x, r = pair((32, width))
        record('mean', x, r, lambda: tp.ops.tp.mean.dim(x, [-1], False),
               lambda: torch.ops.aten.mean.dim(r, [-1], False))
        record('std', x, r, lambda: tp.ops.tp.std.correction(x, [-1], correction=.5),
               lambda: torch.ops.aten.std.correction(r, [-1], correction=.5))
        record('max', x, r, lambda: tp.ops.tp.max.default(x), lambda: torch.ops.aten.max.default(r))
        record('fill', x, r, lambda: tp.ops.tp.fill_.Scalar(x, -.105),
               lambda: torch.ops.aten.fill_.Scalar(r, -.105))

    for rank in (2, 3):
        x, r = pair((2, 32) + (16,) * rank)
        w, rw = pair((1, 32) + (1,) * rank)
        for channels_last in (False, True):
            if channels_last:
                x = x.contiguous(memory_format=tp.channels_last if rank == 2 else tp.channels_last_3d)
                r = r.contiguous(memory_format=torch.channels_last if rank == 2 else torch.channels_last_3d)
            name = f'avg_pool{rank}d'
            args = ([3] * rank, [2] * rank, [1] * rank, True, False, None)
            record(name, x, r, lambda: getattr(tp.ops.tp, name).default(x, *args),
                   lambda: getattr(torch.ops.aten, name).default(r, *args), 1)
            name = f'_adaptive_avg_pool{rank}d'
            record(name, x, r, lambda: getattr(tp.ops.tp, name).default(x, [4] * rank),
                   lambda: getattr(torch.ops.aten, name).default(r, [4] * rank), 1)
            record('prelu', x, r, lambda: tp.ops.tp._prelu_kernel.default(x, w),
                   lambda: torch.ops.aten._prelu_kernel.default(r, rw))
            record('mean_spatial', x, r, lambda: tp.ops.tp.mean.dim(x, list(range(2, rank + 2)), False),
                   lambda: torch.ops.aten.mean.dim(r, list(range(2, rank + 2)), False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('benchmark/quantized_tensor_operations.json'))
    args = parser.parse_args()
    tp.set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    results = run(args.repetitions)
    data = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                threads=args.threads, repetitions=args.repetitions, reference_version=torch.__version__,
                results=results)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
