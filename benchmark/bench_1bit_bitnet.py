"""Benchmark native 1-bit affine quantization and BitNet b1.58 ternary linear.

Native side:
  - ``tp.quantize_affine(..., tp.quint1x8, 0, 1)`` / ``dequantize`` packs
    eight 1-bit codes per storage byte.
  - ``bitnet_linear_packed`` runs the BitNet b1.58 recipe through the native
    signed integer linear path.

Reference side (upstream ao):
  - ``torchao.quantization.quant_primitives.quantize_affine`` /
    ``dequantize_affine`` with ``torch.uint1`` (unpacked uint8 codes).
  - vendored ``torchao.prototype.quantized_training.bitnet``
    ``_BitNetPacked2bitLinear`` (2-bit packed weights + int8 matmul).
"""
import argparse
import datetime
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party' / 'torchao'))
import time
import numpy as np
import torch
import tensorplay as tp


def measure(fn, synchronize, repetitions):
    for _ in range(10):
        fn()
    samples = []
    for _ in range(7):
        synchronize()
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            fn()
        synchronize()
        samples.append((time.perf_counter_ns() - started) / 1000 / repetitions)
    return statistics.median(samples)


def run(device, repetitions):
    rng = np.random.default_rng(1905)
    records = []

    def synchronize():
        if device == 'cuda':
            tp.cuda.synchronize()
            torch.cuda.synchronize()

    def record(name, shape, native, reference, tolerance=1e-4):
        a = native()
        b = reference()
        synchronize()
        if tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f'{name} output shapes differ')
        an = a.float().cpu().numpy() if not hasattr(a, 'int_repr') else a.dequantize().float().cpu().numpy()
        bn = b.float().cpu().numpy() if not hasattr(b, 'int_repr') else b.dequantize().float().cpu().numpy()
        error = float(np.max(np.abs(an - bn), initial=0))
        if error > tolerance:
            raise AssertionError(f'{name} output error {error} exceeds {tolerance}')
        first = measure(native, synchronize, repetitions)
        second = measure(reference, synchronize, repetitions)
        records.append(dict(operator=name, device=device, shape=list(shape),
            native_us=first, reference_us=second, speedup=second / first, max_error=error))

    def record_decoded(name, shape, native_fn, reference_fn, native_decode, reference_decode,
                       tolerance=0):
        a = native_fn()
        b = reference_fn()
        synchronize()
        an = native_decode(a).float().cpu().numpy()
        bn = reference_decode(b).float().cpu().numpy()
        if an.shape != bn.shape:
            raise AssertionError(f'{name} decoded shapes differ')
        error = float(np.max(np.abs(an - bn), initial=0))
        if error > tolerance:
            raise AssertionError(f'{name} output error {error} exceeds {tolerance}')
        first = measure(native_fn, synchronize, repetitions)
        second = measure(reference_fn, synchronize, repetitions)
        records.append(dict(operator=name, device=device, shape=list(shape),
            native_us=first, reference_us=second, speedup=second / first, max_error=error))

    import torchao.quantization.quant_primitives as qp
    torchao_version = getattr(__import__('torchao'), '__version__', 'unknown')

    # ---- 1-bit affine quantization --------------------------------------
    for shape in ((65536,), (1048577,), (1024, 1024)):
        values = rng.uniform(-1, 4, size=shape).astype('float32')
        x = tp.tensor(values, device=device)
        r = torch.tensor(values, device=device)
        scale = .03
        def native_q():
            return tp.quantize_per_tensor(x, scale, 0, tp.quint1x8)
        def reference_q():
            return qp.quantize_affine(r, tuple(shape), torch.tensor(scale, device=device),
                                      torch.tensor(0, device=device), torch.uint1)
        def native_decode(q):
            return q.dequantize()
        def reference_decode(q):
            return qp.dequantize_affine(q, tuple(shape), torch.tensor(scale, device=device),
                                        torch.tensor(0, device=device), torch.uint1)
        record_decoded('quantize_1bit', shape, native_q, reference_q,
                       native_decode, reference_decode, tolerance=0)
        q = native_q()
        rq = reference_q()
        def native_dq():
            return q.dequantize()
        def reference_dq():
            return qp.dequantize_affine(rq, tuple(shape), torch.tensor(scale, device=device),
                                        torch.tensor(0, device=device), torch.uint1)
        record('dequantize_1bit', shape, native_dq, reference_dq, tolerance=0)

    # ---- BitNet b1.58 ternary linear --------------------------------------
    try:
        from torchao.prototype.quantized_training.bitnet import (
            _BitNetPacked2bitLinear, BitNetPacked2bitLinearWeight,
            _pack_i2_in_i8, _unpack_i2_in_i8, get_bitnet_scale, quantize_bitnet_weight,
        )
    except Exception as exc:  # pragma: no cover - depends on vendored torchao
        print(f'skipping BitNet section: {exc}', file=sys.stderr)
        return records

    from tensorplay.ao.quantization.bitnet import (
        bitnet_linear_packed, pack_ternary, quantize_bitnet_weight as tp_quantize_weight,
    )
    from tensorplay.ao.nn.quantized.packed import PackedWeight
    for m, n, k in ((2, 512, 1024), (32, 256, 512), (256, 512, 1024)):
        x_values = rng.normal(0, .2, size=(m, k)).astype('float32')
        w_values = rng.normal(0, .2, size=(n, k)).astype('float32')
        x = tp.tensor(x_values, device=device)
        r = torch.tensor(x_values, device=device)

        # Native: precompute ternary codes + packed 2-bit + scale.
        codes, scale = tp_quantize_weight(tp.tensor(w_values, device=device))
        packed = pack_ternary(codes)
        # Reference: same codes packed with the upstream packer.
        rcodes = quantize_bitnet_weight(torch.tensor(w_values, device=device),
                                        get_bitnet_scale(torch.tensor(w_values, device=device)))
        rpacked = _pack_i2_in_i8(rcodes).contiguous()
        rscale = get_bitnet_scale(torch.tensor(w_values, device=device)).to(torch.float32)

        def native_full():
            return bitnet_linear_packed(x, packed, scale, k)
        def reference_full():
            if device == 'cuda':
                weight = BitNetPacked2bitLinearWeight(rpacked, rscale)
                return _BitNetPacked2bitLinear.apply(r, weight, None)
            # torchao's no-Triton fallback: float matmul over the int8 codes.
            input_i8, row_scale = _ref_rowwise_int8(r)
            return torch.mm(input_i8.float(), _unpack_i2_in_i8(rpacked).float().T) * row_scale * rscale
        record('bitnet_linear', (m, n, k), native_full, reference_full, tolerance=1e-3)

        # Cached native inference path: unpack/prepack amortized outside loop.
        x_i8, row_scale = _rowwise_int8(x)
        x_q = tp._C._make_per_tensor_quantized_tensor(x_i8, 1.0, 0)
        w_q = tp._C._make_per_tensor_quantized_tensor(codes.contiguous(), 1.0, 0)
        cached = PackedWeight(w_q, None)
        def native_infer():
            return cached.run(x_q, 1.0, 0, output_float=True) * row_scale * scale
        def reference_full_cached():
            if device == 'cuda':
                weight = BitNetPacked2bitLinearWeight(rpacked, rscale)
                return _BitNetPacked2bitLinear.apply(r, weight, None)
            input_i8, row_scale = _ref_rowwise_int8(r)
            return torch.mm(input_i8.float(), _unpack_i2_in_i8(rpacked).float().T) * row_scale * rscale
        record('bitnet_linear_cached', (m, n, k), native_infer, reference_full_cached, tolerance=1e-3)
    return records


def _rowwise_int8(x):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    row_scale = absmax / 127.0
    codes = (x.float() / row_scale).round().clamp(-127, 127).to(tp.int8)
    return codes, row_scale


def _ref_rowwise_int8(x):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    row_scale = absmax / 127.0
    codes = (x.float() / row_scale).round().clamp(-127, 127).to(torch.int8)
    return codes, row_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda', 'all'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('benchmark/quantized_1bit_results.json'))
    args = parser.parse_args()
    tp.set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    devices = ['cpu', 'cuda'] if args.device == 'all' else [args.device]
    records = [item for device in devices for item in run(device, args.repetitions)]
    data = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        processor=platform.processor(), threads=args.threads, repetitions=args.repetitions,
        reference_version=torch.__version__, results=records)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()