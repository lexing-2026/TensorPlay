"""Measure cached integer linear compute, packing cost and fused activation."""

import argparse
import json
import statistics
import time

import tensorplay as tp
from tensorplay.ao.nn.quantized.packed import PackedWeight


def measure(fn, device, repetitions):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(5):
        if device == 'cuda':
            tp.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            fn()
        if device == 'cuda':
            tp.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / repetitions)
    return statistics.median(samples)


def benchmark(device, shape, repetitions):
    M, N, K = shape
    x = tp._C.quantize_per_tensor(tp.randn(M, K, device=device), .03, -11, tp.qint8)
    weight = tp._C.quantize_per_tensor(tp.randn(N, K, device=device), .02, 3, tp.qint8)
    bias = tp.randn(N, device=device)
    packed = PackedWeight(weight, bias)
    direct = lambda: tp._C.quantized_linear(x, weight, .03, -11, packed.scales,
                                           packed.zero_points, bias, .2, 7)
    cached = lambda: packed.run(x, .2, 7)
    rebuilding = lambda: PackedWeight(weight, bias).run(x, .2, 7)
    fused = lambda: packed.run(x, .2, 7, relu=True)
    separate = lambda: tp._C.quantized_relu(cached())
    assert tp.equal(direct().int_repr(), cached().int_repr())
    assert tp.equal(fused().int_repr(), separate().int_repr())
    times = {name: measure(fn, device, repetitions) for name, fn in
             [('direct', direct), ('cached', cached), ('rebuilding', rebuilding),
              ('fused_relu', fused), ('separate_relu', separate)]}
    return {'device': device, 'M': M, 'N': N, 'K': K, 'microseconds': times,
            'direct_over_cached': times['direct'] / times['cached']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda', 'all'], default='all')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=20)
    args = parser.parse_args()
    tp.set_num_threads(args.threads)
    devices = ['cpu', 'cuda'] if args.device == 'all' and tp.cuda.is_available() else [args.device if args.device != 'all' else 'cpu']
    records = [benchmark(device, shape, args.repetitions) for device in devices
               for shape in [(1, 512, 1024), (32, 256, 512), (256, 512, 1024)]]
    print(json.dumps({'threads': args.threads, 'repetitions': args.repetitions,
                      'results': records}, indent=2))


if __name__ == '__main__':
    main()
