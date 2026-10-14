"""INT2 / 1-bit / FP8 weight-only module tests.

Covers the new low-bit weight-only operators:

* ``int2_weight_only()``: packed 2-bit codes (four per byte) with per-group
  scales and zeros, inference and straight-through-estimator training.
* ``binary_weight_only()``: packed 1-bit sign codes (eight per byte) with
  per-row absmean scales, inference and training.
* ``float8_weight_only_inference()``: pre-quantized FP8 weight plus per-row
  scales.
* INT4 additions: ``Int4WeightOnlyQuantizedLinear`` alias and the
  ``quantize_int4_weight_only()`` convenience entry point.
"""

import pytest

import tensorplay as tp
from tensorplay import nn

from tensorplay.ao.quantization import (
    BinaryWeightOnlyQuantizedTrainingLinear,
    Float8WeightOnlyInferenceLinear,
    Int2WeightOnlyQuantizedTrainingLinear,
    Int4WeightOnlyQuantizedLinear,
    WeightOnlyBinaryLinear,
    WeightOnlyInt2Linear,
    binary_weight_only,
    float8_dynamic_activation_float8_weight_inference,
    float8_weight_only_inference,
    int2_weight_only,
    int4_weight_only,
    pack_binary,
    pack_int2,
    quantize_,
    quantize_binary_weight_only,
    quantize_int2_weight_only,
    quantize_int4_weight_only,
    unpack_binary,
    unpack_int2,
)

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def _pair(K=16, N=32, B=3):
    tp.manual_seed(0)
    x = tp.randn(B, K)
    lin = nn.Linear(K, N, bias=False)
    return x, lin


# ---------------------------------------------------------------------------
# int2 weight-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('device', DEVICES)
def test_int2_pack_unpack_roundtrip(device):
    codes = tp.randint(0, 4, (5, 10), device=device).to(tp.int8)
    packed = pack_int2(codes)
    assert packed.dtype == tp.uint8
    assert packed.shape == (5, 3)  # 10 codes -> ceil(10/4)=3 bytes
    back = unpack_int2(packed, 10)
    assert bool((back == codes).all())


@pytest.mark.parametrize('device', DEVICES)
def test_int2_from_float_and_forward(device):
    x, lin = _pair()
    x = x.to(device)
    lin = lin.to(device)
    mod = WeightOnlyInt2Linear.from_float(lin, groupsize=8)
    assert mod.weight.dtype == tp.uint8
    assert mod.weight.shape == (mod.out_features, (mod.in_features + 3) // 4)
    assert mod.scales_and_zeros.shape == (2, mod.out_features, 2)
    out = mod(x)
    assert out.shape == (x.shape[0], mod.out_features)
    # dequantized weight matches manual groupwise int2
    scale = mod.scales_and_zeros[..., 0].t().contiguous()
    zero = mod.scales_and_zeros[..., 1].t().contiguous()
    codes = unpack_int2(mod.weight, mod.in_features).to(tp.float32)
    groupsize = 8
    z = zero.repeat_interleave(groupsize, dim=-1)
    s = scale.repeat_interleave(groupsize, dim=-1)
    dq = (codes - z) * s
    ref = x @ dq.T
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_int2_quantize_model(device):
    x, _ = _pair()
    x = x.to(device)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(device)
    qmodel = quantize_(model, int2_weight_only(groupsize=8))
    assert isinstance(qmodel[0], WeightOnlyInt2Linear)
    assert isinstance(qmodel[2], WeightOnlyInt2Linear)
    out = qmodel(x)
    assert out.shape == (x.shape[0], 8)


@pytest.mark.parametrize('device', DEVICES)
def test_int2_convenience_entry(device):
    model = nn.Sequential(nn.Linear(16, 16)).to(device)
    qmodel = quantize_int2_weight_only(model, groupsize=8)
    assert isinstance(qmodel[0], WeightOnlyInt2Linear)


@pytest.mark.parametrize('device', DEVICES)
def test_int2_training_ste_gradients(device):
    x = tp.randn(3, 8, device=device, requires_grad=True)
    mod = Int2WeightOnlyQuantizedTrainingLinear(8, 5, groupsize=4).to(device)
    out = mod(x)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert mod.weight.grad is not None
    assert mod.weight.grad.shape == (5, 8)
    # grad_input flows through the full-precision weight (STE)
    assert x.grad is not None
    expected = tp.ones_like(out) @ mod.weight.detach()
    tp.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# binary (1-bit) weight-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('device', DEVICES)
def test_binary_pack_unpack_roundtrip(device):
    codes = tp.randint(0, 2, (4, 10), device=device).to(tp.uint8)
    packed = pack_binary(codes)
    assert packed.dtype == tp.uint8
    assert packed.shape == (4, 2)  # 10 codes -> ceil(10/8)=2 bytes
    back = unpack_binary(packed, 10)
    assert bool((back == codes).all())


@pytest.mark.parametrize('device', DEVICES)
def test_binary_from_float_and_forward(device):
    x, lin = _pair()
    x = x.to(device)
    lin = lin.to(device)
    mod = WeightOnlyBinaryLinear.from_float(lin)
    assert mod.weight.shape == (mod.out_features, (mod.in_features + 7) // 8)
    assert mod.scale.shape == (mod.out_features, 1)
    out = mod(x)
    assert out.shape == (x.shape[0], mod.out_features)
    # dequantized weight = sign * absmean scale
    codes = unpack_binary(mod.weight, mod.in_features).to(tp.float32) * 2.0 - 1.0
    dq = codes * mod.scale
    ref = x @ dq.T
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    # weight magnitudes all equal to the row scale
    assert bool((dq.abs() - mod.scale).abs().max() < 1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_binary_quantize_model(device):
    x, _ = _pair()
    x = x.to(device)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(device)
    qmodel = quantize_(model, binary_weight_only())
    assert isinstance(qmodel[0], WeightOnlyBinaryLinear)
    assert isinstance(qmodel[2], WeightOnlyBinaryLinear)
    out = qmodel(x)
    assert out.shape == (x.shape[0], 8)


@pytest.mark.parametrize('device', DEVICES)
def test_binary_convenience_entry(device):
    model = nn.Sequential(nn.Linear(16, 16)).to(device)
    qmodel = quantize_binary_weight_only(model)
    assert isinstance(qmodel[0], WeightOnlyBinaryLinear)


@pytest.mark.parametrize('device', DEVICES)
def test_binary_training_ste_gradients(device):
    x = tp.randn(3, 8, device=device, requires_grad=True)
    mod = BinaryWeightOnlyQuantizedTrainingLinear(8, 5).to(device)
    out = mod(x)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert mod.weight.grad is not None
    assert mod.weight.grad.shape == (5, 8)
    assert x.grad is not None
    expected = tp.ones_like(out) @ mod.weight.detach()
    tp.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# float8 weight-only inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('device', DEVICES)
def test_float8_inference_from_float_and_forward(device):
    x, lin = _pair()
    x = x.to(device)
    lin = lin.to(device)
    mod = Float8WeightOnlyInferenceLinear.from_float(lin)
    assert mod.weight.dtype == tp.float8_e4m3fn
    assert mod.scale.shape == (mod.out_features, 1)
    out = mod(x)
    assert out.shape == (x.shape[0], mod.out_features)
    # dequantized weight times scale reproduces the output
    dq = tp._C.dequantize_affine_float8(mod.weight, mod.scale, tp.float32)
    ref = x @ dq.T
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_float8_inference_quantize_model(device):
    x, _ = _pair()
    x = x.to(device)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).to(device)
    qmodel = quantize_(model, float8_weight_only_inference())
    assert isinstance(qmodel[0], Float8WeightOnlyInferenceLinear)
    out = qmodel(x)
    assert out.shape == (x.shape[0], 8)
    # quantized weight is ~half the float weight's memory (1 byte vs 4)
    assert qmodel[0].weight.element_size() == 1


# ---------------------------------------------------------------------------
# int4 additions
# ---------------------------------------------------------------------------


def test_int4_quantized_linear_alias():
    from tensorplay.ao.quantization import WeightOnlyInt4Linear

    assert Int4WeightOnlyQuantizedLinear is WeightOnlyInt4Linear


@pytest.mark.parametrize('device', DEVICES)
def test_int4_convenience_entry(device):
    model = nn.Sequential(nn.Linear(64, 16)).to(device)
    qmodel = quantize_int4_weight_only(model, groupsize=32)
    from tensorplay.ao.quantization import WeightOnlyInt4Linear

    assert isinstance(qmodel[0], WeightOnlyInt4Linear)
    x = tp.randn(2, 64, device=device)
    out = qmodel(x)
    assert out.shape == (2, 16)


@pytest.mark.parametrize('device', DEVICES)
def test_int2_rejects_bad_groupsize(device):
    lin = nn.Linear(16, 8).to(device)
    with pytest.raises(ValueError):
        WeightOnlyInt2Linear.from_float(lin, groupsize=3)

# ---------------------------------------------------------------------------
# fp8 dynamic-activation inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('device', DEVICES)
def test_float8_dynamic_inference_from_float_and_forward(device):
    x, lin = _pair()
    x = x.to(device)
    lin = lin.to(device)
    mod = Float8WeightOnlyInferenceLinear  # imported above
    from tensorplay.ao.quantization import Float8DynamicActivationInferenceLinear as FDAL

    mod = FDAL.from_float(lin)
    assert mod.weight.dtype == tp.float8_e4m3fn
    out = mod(x)
    assert out.shape == (x.shape[0], mod.out_features)
    assert bool(tp.isfinite(out).all())
    # output stays close to the float linear (fp8 has ~1e-2 relative error)
    ref = lin(x)
    scale = (out.abs().max() + 1e-8)
    assert ((out - ref).abs() / scale).max() < 0.1


@pytest.mark.parametrize('device', DEVICES)
def test_float8_dynamic_inference_quantize_model(device):
    x, _ = _pair()
    x = x.to(device)
    model = nn.Sequential(nn.Linear(16, 32, bias=False), nn.ReLU(), nn.Linear(32, 8, bias=False)).to(device)
    qmodel = quantize_(model, float8_dynamic_activation_float8_weight_inference())
    from tensorplay.ao.quantization import Float8DynamicActivationInferenceLinear

    assert isinstance(qmodel[0], Float8DynamicActivationInferenceLinear)
    out = qmodel(x)
    assert out.shape == (x.shape[0], 8)


# ---------------------------------------------------------------------------
# AWQ with int2 weight-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('device', DEVICES)
def test_awq_int2_workflow(device):
    from tensorplay.ao.quantization import awq

    model = nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU(), nn.Linear(32, 16, bias=False)).to(device)
    x = tp.randn(4, 32, device=device)
    config = awq(int2_weight_only(groupsize=8), step='prepare')
    observed = quantize_(model, config)
    observed(x)
    converted = quantize_(observed, awq(int2_weight_only(groupsize=8), step='convert'))
    assert isinstance(converted[0], WeightOnlyInt2Linear)
    out = converted(x)
    assert out.shape == (4, 16)
    assert bool(tp.isfinite(out).all())


# ---------------------------------------------------------------------------
# generic n-bit weight-only (IntxWeightOnlyConfig)
# ---------------------------------------------------------------------------
from tensorplay.ao.quantization import (
    IntxWeightOnlyConfig,
    IntxWeightOnlyQuantizedTrainingLinear,
    WeightOnlyIntxLinear,
    intx_weight_only,
    pack_intx,
    quantize_intx_weight_only,
    unpack_intx,
)


@pytest.mark.parametrize('bits', [1, 2, 3, 4, 5, 6, 7, 8])
def test_intx_pack_unpack_roundtrip(bits):
    codes = tp.randint(0, 1 << bits, (3, 10)).to(tp.int8)
    packed = pack_intx(codes, bits)
    assert packed.dtype == tp.uint8
    back = unpack_intx(packed, bits, 10)
    assert bool((back == codes).all())


@pytest.mark.parametrize('bits', [1, 2, 4])
@pytest.mark.parametrize('device', DEVICES)
def test_intx_from_float_and_forward(bits, device):
    x, lin = _pair()
    x = x.to(device)
    lin = lin.to(device)
    mod = WeightOnlyIntxLinear.from_float(lin, bits=bits, groupsize=8)
    assert mod.weight.dtype == tp.uint8
    out = mod(x)
    assert out.shape == (x.shape[0], mod.out_features)
    # manual dequantized reference
    scale = mod.scales_and_zeros[..., 0].t().contiguous()
    zero = mod.scales_and_zeros[..., 1].t().contiguous()
    codes = unpack_intx(mod.weight, bits, mod.in_features).to(tp.float32)
    groupsize = 8
    z = zero.repeat_interleave(groupsize, dim=-1)
    s = scale.repeat_interleave(groupsize, dim=-1)
    dq = (codes - z) * s
    ref = x @ dq.T
    tp.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('bits', [1, 2, 3])
def test_intx_quantize_model(bits):
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    qmodel = quantize_intx_weight_only(model, bits=bits, groupsize=8)
    assert isinstance(qmodel[0], WeightOnlyIntxLinear)
    assert qmodel[0].bits == bits
    out = qmodel(tp.randn(2, 16))
    assert out.shape == (2, 8)


def test_intx_training_ste_gradients():
    x = tp.randn(3, 8, requires_grad=True)
    mod = IntxWeightOnlyQuantizedTrainingLinear(8, 5, bits=2, groupsize=4)
    out = mod(x)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert mod.weight.grad is not None
    assert mod.weight.grad.shape == (5, 8)
    assert x.grad is not None
    expected = tp.ones_like(out) @ mod.weight.detach()
    tp.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


def test_intx_config_validation():
    with pytest.raises(ValueError):
        IntxWeightOnlyConfig(bits=0)
    with pytest.raises(ValueError):
        IntxWeightOnlyConfig(bits=9)
    with pytest.raises(ValueError):
        IntxWeightOnlyConfig(groupsize=0)
    # int2_weight_only is equivalent to IntxWeightOnlyConfig(bits=2)
    assert intx_weight_only(bits=2).bits == 2


# ---------------------------------------------------------------------------
# quantize_ top-level module regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_top_level_module(device):
    m = nn.Linear(16, 8).to(device)
    qm = quantize_(m, int2_weight_only(groupsize=8))
    assert isinstance(qm, WeightOnlyInt2Linear)
    assert qm is not m
    x = tp.randn(2, 16, device=device)
    out = qm(x)
    assert out.shape == (2, 8)


@pytest.mark.parametrize('device', DEVICES)
def test_quantize_top_level_training_module(device):
    from tensorplay.ao.quantization import int8_weight_only_quantized_training

    m = nn.Linear(16, 8).to(device)
    qm = quantize_(m, int8_weight_only_quantized_training())
    # training transform mutates in place and returns the same module
    assert qm is m
    from tensorplay.ao.quantization import Int8QuantizedTrainingLinearWeight

    assert isinstance(qm.weight, Int8QuantizedTrainingLinearWeight)
