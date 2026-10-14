"""Measure quantized operator families after checking codes and affine grids."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from bench_quantized_native import code_pair, measure, np, tp, torch


def run(repetitions, validate_only=False):
    rng = np.random.default_rng(907)
    results = []

    def pair(shape, scale=.07, zero_point=119):
        return code_pair(rng.integers(0, 256, size=shape, dtype='uint8'), scale, zero_point, 'cpu')

    def record(name, shape, native, reference, tolerance=1, layout='contiguous'):
        a, b = native(), reference()
        if tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f'{name}: output shapes differ')
        if a.is_quantized() != b.is_quantized:
            raise AssertionError(f'{name}: output storage kinds differ')
        if b.is_quantized:
            if a.q_scale() != b.q_scale() or a.q_zero_point() != b.q_zero_point():
                raise AssertionError(f'{name}: output affine grids differ')
            ac, bc = a.int_repr().numpy(), b.int_repr().numpy()
        else:
            ac, bc = a.float().numpy(), b.float().numpy()
        error = float(np.max(np.abs(ac.astype('float64') - bc.astype('float64')), initial=0))
        if error > tolerance:
            raise AssertionError(f'{name}: output error {error} exceeds {tolerance}')
        entry = dict(operator=name, device='cpu', shape=list(shape), layout=layout,
                     max_error=error, tolerance=tolerance)
        if not validate_only:
            entry['native_us'] = measure(native, lambda: None, repetitions)
            entry['reference_us'] = measure(reference, lambda: None, repetitions)
            entry['speedup'] = entry['reference_us'] / entry['native_us']
        results.append(entry)

    for shape in ((32, 256), (32, 8193)):
        x, r = pair(shape, .03)
        y, s = pair(shape, .05, 113)
        for name in ('add', 'add_relu', 'mul', 'mul_relu'):
            native, reference = getattr(tp.ops.quantized, name), getattr(torch.ops.quantized, name)
            record(name, shape, lambda: native(x, y, .08, 119), lambda: reference(r, s, .08, 119))
        for name, args in (('sigmoid', (.01, 13)), ('softmax', (-1, .003, 0)),
                           ('elu', (.03, 113, 1.2, .8, 1.1)), ('celu', (.03, 113, 1.2)),
                           ('leaky_relu', (.13, False, .03, 113))):
            native, reference = getattr(tp.ops.quantized, name), getattr(torch.ops.quantized, name)
            record(name, shape, lambda: native(x, *args), lambda: reference(r, *args))
        for name in ('cat', 'cat_relu'):
            native, reference = getattr(tp.ops.quantized, name), getattr(torch.ops.quantized, name)
            record(name, shape, lambda: native([x, y], 1, .08, 119), lambda: reference([r, s], 1, .08, 119))
        weight = rng.uniform(.7, 1.3, size=shape[-1]).astype('float32')
        bias = rng.uniform(-.2, .2, size=shape[-1]).astype('float32')
        w, rw, b, rb = tp.tensor(weight), torch.tensor(weight), tp.tensor(bias), torch.tensor(bias)
        record('layer_norm', shape,
               lambda: tp.ops.quantized.layer_norm(x, [shape[-1]], w, b, 1e-5, .09, 119),
               lambda: torch.ops.quantized.layer_norm(r, [shape[-1]], rw, rb, 1e-5, .09, 119))

    shape = (2, 32, 16, 16)
    x, r = pair(shape)
    w, rw, b, rb = tp.ones(32), torch.ones(32), tp.zeros(32), torch.zeros(32)
    for channels_last in (False, True):
        if channels_last:
            x = x.contiguous(memory_format=tp.channels_last)
            r = r.contiguous(memory_format=torch.channels_last)
        layout = 'channels_last' if channels_last else 'contiguous'
        for name, args in (('group_norm', (4,)), ('instance_norm', ())):
            native, reference = getattr(tp.ops.quantized, name), getattr(torch.ops.quantized, name)
            record(name, shape, lambda: native(x, *args, w, b, 1e-5, .09, 119),
                   lambda: reference(r, *args, rw, rb, 1e-5, .09, 119), layout=layout)
        record('max_pool2d', shape,
               lambda: tp.ops.quantized.max_pool2d(x, [3, 3], [2, 2], [1, 1], [1, 1], False),
               lambda: torch.ops.quantized.max_pool2d(r, [3, 3], [2, 2], [1, 1], [1, 1], False),
               tolerance=0, layout=layout)

    for rank in (1, 2, 3):
        shape = (2, 8) + (8,) * rank
        x, r = pair(shape, .05, 113)
        bias = rng.normal(0, .1, size=8).astype('float32')
        b, rb = tp.tensor(bias), torch.tensor(bias)
        for transposed in (False, True):
            name = ('conv_transpose' if transposed else 'conv') + f'{rank}d'
            codes = rng.integers(-29, 30, size=(8, 4) + (3,) * rank, dtype='int8')
            w = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes), .02, -3)
            rw = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .02, -3)
            args = [[2] * rank, [1] * rank]
            if transposed:
                args.append([1] * rank)
            args.extend([[1] * rank, 2])
            packed = getattr(tp.ops.quantized, name + '_prepack')(w, b, *args)
            rp = getattr(torch.ops.quantized, name + '_prepack')(rw, rb, *args)
            native, reference = getattr(tp.ops.quantized, name), getattr(torch.ops.quantized, name)
            if not transposed and rank > 1:
                native, reference = native.new, reference.new
            record(name, shape, lambda: native(x, packed, .07, 119), lambda: reference(r, rp, .07, 119))

    for bits in (8, 4, 2):
        shape = (1024, 128)
        values = rng.normal(0, .2, size=shape).astype('float32')
        x, r = tp.tensor(values), torch.tensor(values)
        name = 'byte' if bits == 8 else f'{bits}bit'
        native = getattr(tp.ops.quantized, f'embedding_bag_{name}_prepack')
        reference = getattr(torch.ops.quantized, f'embedding_bag_{name}_prepack')
        record(f'embedding_bag_{name}_prepack', shape, lambda: native(x), lambda: reference(r), tolerance=0)
        packed, rp = native(x), reference(r)
        indices = rng.integers(0, shape[0], size=256, dtype='int64')
        offsets = np.arange(0, 257, 8, dtype='int64')
        ind, rind, off, roff = tp.tensor(indices), torch.tensor(indices), tp.tensor(offsets), torch.tensor(offsets)
        native = getattr(tp.ops.quantized, f'embedding_bag_{name}_rowwise_offsets')
        reference = getattr(torch.ops.quantized, f'embedding_bag_{name}_rowwise_offsets')
        record(f'embedding_bag_{name}_rowwise_offsets', (32, 8, 128),
               lambda: native(packed, ind, off, False, 0, False, None, None, True),
               lambda: reference(rp, rind, roff, False, 0, False, None, None, True), tolerance=2e-6)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('benchmark/quantized_operator_families.json'))
    args = parser.parse_args()
    tp.set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    results = run(args.repetitions, args.validate_only)
    data = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                threads=args.threads, repetitions=0 if args.validate_only else args.repetitions,
                reference_version=torch.__version__, results=results)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
