"""Quantized operators and their lowering paths.

Quantization is observed at the point the compiler sees the operators: the
conversions that carry a real kernel lower through the framework fallback and
must reproduce eager, and the operators whose kernels a CPU build does not
carry refuse at capture rather than compiling to a path that could not run.
"""

import pytest
import tensorplay as tp

from tensorplay.functional import (
    _dyn_quant_matmul_4bit,
    _dyn_quant_pack_4bit_weight,
    _weight_int4pack_mm_for_cpu,
    _weight_int8pack_mm,
)


def _compile_and_call(fn, *args):
    return tp.compile(fn, strict_native=True)(*args)


def test_quantize_then_dequantize_matches_eager():
    tp.manual_seed(0)

    def fn(x, scale, zero_point):
        quantized = tp.quantize_per_tensor(x, scale, zero_point, tp.qint8)
        return tp.dequantize(quantized)

    x = tp.randn(4, 8)
    scale, zero_point = 0.05, 3
    expected = fn(x, scale, zero_point)
    result = _compile_and_call(fn, x, scale, zero_point)

    assert result.shape == expected.shape
    assert tp.allclose(result, expected, atol=1e-6)


def test_int4_weight_only_cpu_matmul_matches_eager():
    tp.manual_seed(0)
    from tensorplay.ao.quantization.int4_weight_only import _pack_int4_weight

    groupsize = 32
    # The packing requires output channels divisible by sixteen and an even
    # feature count.
    weight = tp.randn(16, 64)
    packed, scale_zero = _pack_int4_weight(weight, groupsize)
    x = tp.randn(2, 64)

    def fn(x, packed, scale_zero):
        return _weight_int4pack_mm_for_cpu(x, packed, groupsize, scale_zero)

    expected = fn(x, packed, scale_zero)
    result = _compile_and_call(fn, x, packed, scale_zero)

    assert result.shape == (2, 16)
    assert tp.allclose(result, expected, atol=1e-6)


def test_int8_weight_pack_matmul_matches_dequantized_reference():
    weight_codes = tp.tensor([[1.0, -2.0], [3.0, 4.0]])
    weight = tp.quantize_per_tensor(weight_codes, 1.0, 0, tp.qint8)
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    scales = tp.tensor([1.0, 2.0])

    expected = tp.matmul(x, tp.dequantize(weight).transpose(0, 1)) * scales.unsqueeze(1)
    result = _weight_int8pack_mm(x, weight, scales)
    assert result.shape == (2, 2)
    assert tp.allclose(result, expected)


def test_dynamic_quant_matmul_4bit_refuses_on_cpu():
    def fn(inp, packed, block_size, in_features, out_features):
        return _dyn_quant_matmul_4bit(inp, packed, block_size, in_features, out_features)

    with pytest.raises(NotImplementedError, match="Kernel not found"):
        _compile_and_call(
            fn, tp.randn(2, 8), tp.zeros((4, 4), dtype=tp.uint8), 64, 8, 4
        )


def test_dynamic_quant_pack_4bit_weight_refuses_on_cpu():
    def fn(weights, scales_zeros, bias, block_size, in_features, out_features):
        return _dyn_quant_pack_4bit_weight(
            weights, scales_zeros, bias, block_size, in_features, out_features
        )

    with pytest.raises(NotImplementedError, match="Kernel not found"):
        _compile_and_call(
            fn,
            tp.randn(4, 8),
            tp.randn(1, 4, 2),
            tp.randn(4),
            64,
            8,
            4,
        )
