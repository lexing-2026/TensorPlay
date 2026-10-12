import io
import copy

import numpy as np
import pytest
import tensorplay as tp
import torch
from tensorplay.ao.nn import quantized as nnq, intrinsic
from tensorplay.ao.quantization import (
    default_qconfig, default_per_channel_qat_qconfig, prepare, prepare_qat,
    convert, QuantStub, DeQuantStub,
)
from tensorplay.ao.nn.quantized.packed import PackedWeight

DEVICES = ['cpu', 'cuda'] if tp.cuda.is_available() else ['cpu']


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['qint8', 'quint8', 'qint32'])
def test_quantized_rectification_preserves_grid_and_codes(device, dtype):
    codes = {'qint8': [-128, -12, -1, 0, 32, 127],
             'quint8': [0, 12, 136, 137, 200, 255],
             'qint32': [-2147483648, -12, 0, 12344, 12345, 2147483647]}[dtype]
    zp = {'qint8': 5, 'quint8': 137, 'qint32': 12345}[dtype]
    storage = {'qint8': 'int8', 'quint8': 'uint8', 'qint32': 'int32'}[dtype]
    actual = tp._C._make_per_tensor_quantized_tensor(
        tp.tensor(codes, dtype=getattr(tp, storage), device=device).reshape(2, 3), .03, zp).T
    expected = torch._make_per_tensor_quantized_tensor(
        torch.tensor(codes, dtype=getattr(torch, storage)).reshape(2, 3), .03, zp).T
    output = nnq.ReLU()(actual)
    np.testing.assert_array_equal(output.int_repr().cpu().numpy(), torch.relu(expected).int_repr().numpy())
    np.testing.assert_array_equal(actual.int_repr().cpu().numpy(), expected.int_repr().numpy())
    assert output.dtype == actual.dtype and output.q_scale() == .03 and output.q_zero_point() == zp
    assert nnq.ReLU()(actual[:0]).shape == (0, 2)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('unsigned', [False, True])
@pytest.mark.parametrize('K', [0, 3, 31, 32, 65])
@pytest.mark.parametrize('relu', [False, True])
def test_packed_linear_integer_reference(device, unsigned, K, relu):
    rng = np.random.default_rng(23)
    xcodes = rng.integers(0 if unsigned else -128, 256 if unsigned else 128, (2, 3, K), dtype=np.int16)
    wcodes = rng.integers(-128, 128, (7, K), dtype=np.int16)
    xs, xz, ys, yz = .021, 137 if unsigned else -13, .19, 101 if unsigned else 5
    scales = np.linspace(.007, .04, 7, dtype='float32')
    zps = np.arange(-3, 4, dtype='int64')
    qx = tp._C._make_per_tensor_quantized_tensor(tp.tensor(xcodes.astype('uint8' if unsigned else 'int8'), device=device), xs, xz)
    qw = tp._C._make_per_channel_quantized_tensor(tp.tensor(wcodes.astype('int8'), device=device), tp.tensor(scales, device=device), tp.tensor(zps, device=device), 0)
    bias = tp.tensor(np.linspace(-.7, .7, 7, dtype='float32'), device=device)
    packed = PackedWeight(qw, bias)
    old = packed.packed
    actual = packed.run(qx, ys, yz, relu=relu)
    acc = (xcodes.astype('int64') - xz) @ (wcodes.astype('int64') - zps[:, None]).T
    value = acc * xs * scales + bias.cpu().numpy()
    if relu:
        value = np.maximum(value, 0.)
    expected = np.clip(np.rint(value / ys) + yz, 0 if unsigned else -128, 255 if unsigned else 127)
    np.testing.assert_array_equal(actual.int_repr().cpu().numpy(), expected)
    assert packed.packed is old
    np.testing.assert_array_equal(packed.offsets.cpu().numpy(), wcodes.astype('int64').sum(1))
    assert packed.packed.shape == (7, ((K + 31) // 32) * 32)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('relu', [False, True])
def test_packed_convolution_codes_and_serialization(device, rank, groups, relu):
    rng = np.random.default_rng(17)
    shape = (2, 4) + (5,) * rank
    weight_shape = (4, 4 // groups) + (3,) * rank
    xcodes = rng.integers(0, 256, shape, dtype='uint8')
    wcodes = rng.integers(-12, 13, weight_shape, dtype='int8')
    scales = np.array([.01, .02, .03, .04], dtype='float32')
    zps = np.array([2, -3, 0, 1], dtype='int64')
    bias = np.array([-.2, .1, .3, -.4], dtype='float32')
    qx = tp._C._make_per_tensor_quantized_tensor(tp.tensor(xcodes, device=device), .013, 129)
    qw = tp._C._make_per_channel_quantized_tensor(tp.tensor(wcodes, device=device), tp.tensor(scales, device=device), tp.tensor(zps, device=device), 0)
    cls = getattr(intrinsic.quantized if relu else nnq, f'ConvReLU{rank}d' if relu else f'Conv{rank}d')
    actual = cls(4, 4, 3, stride=2, padding=1, dilation=1, groups=groups, out_scale=.17, out_zero_point=93)
    actual.set_weight_bias(qw, tp.tensor(bias, device=device))
    dx = torch.tensor(xcodes.astype('float64') - 129) * .013
    dw = torch.tensor(wcodes.astype('float64') - zps.reshape((4,) + (1,) * (rank + 1))) * torch.tensor(scales.astype('float64')).reshape((4,) + (1,) * (rank + 1))
    value = getattr(torch.nn.functional, f'conv{rank}d')(dx, dw, torch.tensor(bias.astype('float64')), stride=2, padding=1, groups=groups).numpy()
    if relu:
        value = np.maximum(value, 0.)
    expected = np.clip(np.rint(value / .17) + 93, 0, 255)
    result = actual(qx)
    np.testing.assert_array_equal(result.int_repr().cpu().numpy(), expected)
    stream = io.BytesIO()
    tp.save(actual.state_dict(), stream)
    stream.seek(0)
    state = tp.load(stream)
    assert not any(key.endswith(('.packed', '.offsets')) for key in state)
    fresh = cls(4, 4, 3, stride=2, padding=1, groups=groups).to(device)
    fresh.load_state_dict(state)
    np.testing.assert_array_equal(fresh(qx).int_repr().cpu().numpy(), result.int_repr().cpu().numpy())
    assert fresh.out_zero_point == 93
    assert fresh.weight.qscheme() == tp.per_channel_affine


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('qat', [False, True])
def test_eager_end_to_end_frozen_grid(device, qat):
    model = tp.nn.Sequential(QuantStub(), tp.nn.Linear(5, 7), tp.nn.ReLU(), DeQuantStub()).to(device)
    model.qconfig = default_per_channel_qat_qconfig if qat else default_qconfig
    prepared = prepare_qat(model) if qat else prepare(model.eval())
    x = tp.randn(3, 5, device=device)
    prepared(x)
    if qat:
        prepared[1].weight_fake_quant.disable_observer()
        scale = prepared[1].weight_fake_quant.scale.clone()
        prepared[1].weight.data *= 4
    else:
        scale, _ = prepared[1].weight_observer.calculate_qparams()
    quantized = convert(prepared.eval(), use_precomputed_fake_quant=True)
    np.testing.assert_allclose(quantized[1].weight_scales.cpu().numpy(), np.broadcast_to(scale.cpu().numpy(), (7,)), rtol=1e-6)
    assert quantized(x).shape == (3, 7)
    state = quantized.state_dict()
    restored = convert(prepared.eval(), use_precomputed_fake_quant=True)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored(x).cpu().numpy(), quantized(x).cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
def test_dynamic_linear_global_grid(device):
    floating = tp.nn.Linear(5, 4).to(device)
    floating.qconfig = default_qconfig
    actual = nnq.dynamic.Linear.from_float(floating)
    x = tp.tensor([[.1, -1., 2., .3, -2.], [10., -12., 4., .5, -3.]], device=device)
    qx = tp._C.quantize_per_tensor_dynamic(x, tp.quint8, True)
    reference = qx.dequantize() @ actual.weight.dequantize().T + actual.bias
    np.testing.assert_allclose(actual(x).cpu().numpy(), reference.cpu().numpy(), atol=1e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_packed_full_model_save_copy_and_device(device, bias, legacy):
    qweight = tp._C.quantize_per_tensor(tp.randn(4, 5), .03, -7, tp.qint8)
    model = nnq.Linear(5, 4, qweight=qweight, bias=tp.randn(4) if bias else None,
                       out_scale=.07, out_zero_point=31).to(device)
    x = tp._C.quantize_per_tensor(tp.randn(2, 5, device=device), .02, 93, tp.quint8)
    expected = model(x).int_repr().cpu().numpy()
    stream = io.BytesIO()
    tp.save(model, stream, _use_new_zipfile_serialization=not legacy)
    stream.seek(0)
    with tp.serialization.safe_globals([nnq.Linear, PackedWeight]):
        restored = tp.load(stream)
    for result in (restored, copy.deepcopy(model), model.cpu().to(device)):
        assert result.qweight.q_zero_point() == -7
        np.testing.assert_array_equal(result(x).int_repr().cpu().numpy(), expected)


@pytest.mark.parametrize('device', DEVICES)
def test_packed_empty_noncontiguous_and_validation(device):
    weight = tp._C.quantize_per_tensor(tp.randn(4, 5, device=device), .03, 0, tp.qint8)
    packed = PackedWeight(weight)
    x = tp._C.quantize_per_tensor(tp.randn(2, 5, 3, device=device), .02, 90, tp.quint8).transpose(1, 2)
    assert not x.is_contiguous()
    np.testing.assert_array_equal(packed.run(x).int_repr().cpu().numpy(), packed.run(x.contiguous()).int_repr().cpu().numpy())
    empty = tp._C.quantize_per_tensor(tp.empty(0, 5, device=device), .02, 90, tp.quint8)
    assert packed.run(empty).shape == (0, 4)
    with pytest.raises(RuntimeError):
        packed.run(tp._C.quantize_per_tensor(tp.ones(5, device=device), .02, 90, tp.quint8))
    with pytest.raises(RuntimeError):
        packed.run(x, out_scale=0.)
    with pytest.raises(RuntimeError):
        packed.run(x, out_zero_point=256)
    with pytest.raises(RuntimeError):
        packed.run(tp.ones(2, 5, device=device))
    with pytest.raises(ValueError):
        packed.set_weight_bias(weight, tp.ones(2, device=device))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [2, 3])
def test_fused_normalization_codes(device, rank):
    rng = np.random.default_rng(71)
    shape = (2, 3) + (4,) * rank
    codes = rng.integers(0, 256, shape, dtype='uint8')
    params = [np.array(a, dtype='float32') for a in ([1.1, -.8, .4], [.2, -.3, .5], [.6, -.2, .3], [.7, 1.2, .4])]
    actual = getattr(intrinsic.quantized, f'BNReLU{rank}d')(3).to(device)
    actual.weight.data.copy_(tp.tensor(params[0], device=device))
    actual.bias.data.copy_(tp.tensor(params[1], device=device))
    actual.running_mean.copy_(tp.tensor(params[2], device=device))
    actual.running_var.copy_(tp.tensor(params[3], device=device))
    actual.scale.fill_(.07); actual.zero_point.fill_(61)
    qx = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), .03, 91)
    refx = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .03, 91)
    expected = getattr(torch.ops.quantized, f'batch_norm{rank}d_relu')(refx, *[torch.tensor(p) for p in params], actual.eps, .07, 61)
    np.testing.assert_array_equal(actual(qx).int_repr().cpu().numpy(), expected.int_repr().numpy())
    fresh = getattr(intrinsic.quantized, f'BNReLU{rank}d')(3).to(device)
    fresh.load_state_dict(actual.state_dict())
    np.testing.assert_array_equal(fresh(qx).int_repr().cpu().numpy(), actual(qx).int_repr().cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('relu', [False, True])
def test_qat_normalization_folded_conversion(device, rank, relu):
    from tensorplay.ao.quantization import fuse_modules_qat
    conv = getattr(tp.nn, f'Conv{rank}d')(2, 3, 3, padding=1)
    bn = getattr(tp.nn, f'BatchNorm{rank}d')(3)
    parts = [QuantStub(), conv, bn]
    if relu:
        parts.append(tp.nn.ReLU())
    parts.append(DeQuantStub())
    model = tp.nn.Sequential(*parts).to(device)
    model.qconfig = default_per_channel_qat_qconfig
    fused = fuse_modules_qat(model, ['1', '2', '3'] if relu else ['1', '2'])
    trained = prepare_qat(fused)
    x = tp.randn((2, 2) + (4,) * rank, device=device)
    trained(x).sum().backward()
    trained.eval()
    trained[1].weight_fake_quant.disable_observer()
    expected_weight = trained[1].to_float()
    expected_weight = getattr(expected_weight, 'op', expected_weight).weight.detach()
    result = convert(trained, use_precomputed_fake_quant=True)
    target = getattr(intrinsic.quantized if relu else nnq, f'ConvReLU{rank}d' if relu else f'Conv{rank}d')
    assert isinstance(result[1], target)
    expected_grid = trained[1].weight_fake_quant(expected_weight)
    np.testing.assert_allclose(result[1].weight.dequantize().cpu().numpy(), expected_grid.cpu().numpy(), atol=1e-6)
    assert result(x).shape == (2, 3) + (4,) * rank


@pytest.mark.parametrize('device', DEVICES)
def test_dynamic_fused_eager_conversion(device):
    from tensorplay.ao.quantization import quantize_dynamic
    model = tp.nn.Sequential(intrinsic.LinearReLU(tp.nn.Linear(5, 4))).to(device)
    result = quantize_dynamic(model)
    assert isinstance(result[0], intrinsic.quantized.dynamic.LinearReLU)
    x = tp.randn(3, 5, device=device)
    qx = tp._C.quantize_per_tensor_dynamic(x, tp.quint8, True)
    expected = (qx.dequantize() @ result[0].weight.dequantize().T + result[0].bias).relu()
    np.testing.assert_allclose(result(x).cpu().numpy(), expected.cpu().numpy(), atol=1e-6)
    assert result(tp.empty(0, 5, device=device)).shape == (0, 4)


@pytest.mark.parametrize('device', DEVICES)
def test_dynamic_root_and_configuration_override(device):
    from tensorplay.ao.quantization import quantize_dynamic, default_per_channel_qconfig
    floating = tp.nn.Linear(5, 4).to(device)
    result = quantize_dynamic(floating, {tp.nn.Linear: default_per_channel_qconfig})
    assert isinstance(result, nnq.dynamic.Linear)
    assert not result.training
    assert result.weight.qscheme() == tp.per_channel_affine
    floating.qconfig = None
    untouched = quantize_dynamic(floating)
    assert type(untouched) is tp.nn.Linear


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [1, 2])
def test_integer_reflection_padding(device, rank):
    model = tp.nn.Sequential(QuantStub(), getattr(tp.nn, f'Conv{rank}d')(2, 3, 3, padding=1, padding_mode='reflect'), DeQuantStub()).to(device).eval()
    model.qconfig = default_qconfig
    prepared = prepare(model)
    x = tp.randn((2, 2) + (5,) * rank, device=device)
    prepared(x)
    result = convert(prepared)
    assert result[1].padding_mode == 'reflect'
    qx = result[0](x)
    padded = tp.nn.functional.pad(qx.int_repr(), (1, 1) * rank, mode='reflect')
    padded = tp._C._make_per_tensor_quantized_tensor(padded, qx.q_scale(), qx.q_zero_point())
    expected = result[1]._packed_params.run(padded, result[1].out_scale, result[1].out_zero_point,
                                           (1,) * rank, (0,) * rank, (1,) * rank)
    np.testing.assert_array_equal(result[1](qx).int_repr().cpu().numpy(), expected.int_repr().cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
def test_packed_accumulation_exceeds_int32(device):
    K = 131072
    x = tp._C._make_per_tensor_quantized_tensor(tp.full((2, K), 255, dtype=tp.uint8, device=device), 1., 0)
    w = tp._C._make_per_tensor_quantized_tensor(tp.full((1, K), 127, dtype=tp.int8, device=device), 1., -128)
    packed = PackedWeight(w)
    result = packed.run(x, 1.e8, 0)
    expected = round(K * 255 * 255 / 1.e8)
    np.testing.assert_array_equal(result.int_repr().cpu().numpy(), np.full((2, 1), expected))


@pytest.mark.skipif('cuda' not in DEVICES, reason='Requires a CUDA device')
def test_packed_cuda_large_leading_dimension():
    rows = 1048592
    x = tp._C.quantize_per_tensor(tp.empty(rows, 0, device='cuda'), .1, 0, tp.quint8)
    w = tp._C.quantize_per_tensor(tp.empty(1, 0, device='cuda'), .1, 0, tp.qint8)
    packed = PackedWeight(w, tp.tensor([.3], device='cuda'))
    result = packed.run(x, .01, 129)
    assert result.shape == (rows, 1)
    codes = result.int_repr()
    assert int(codes.min()) == 159 and int(codes.max()) == 159


@pytest.mark.parametrize('device', DEVICES)
def test_checkpoint_preserves_scalar_precision_and_strictness(device):
    weight = tp._C.quantize_per_tensor(tp.zeros(1, 1, device=device), 1., 0, tp.qint8)
    module = nnq.Linear(1, 1, qweight=weight, bias=tp.tensor([.15], device=device),
                        out_scale=.3, out_zero_point=66)
    x = tp._C.quantize_per_tensor(tp.zeros(2, 1, device=device), .1, 0, tp.quint8)
    state = module.state_dict()
    fresh = nnq.Linear(1, 1).to(device)
    fresh.load_state_dict(state)
    assert fresh.out_scale == .3
    np.testing.assert_array_equal(fresh(x).int_repr().cpu().numpy(), module(x).int_repr().cpu().numpy())
    incomplete = dict(state)
    del incomplete['out_scale']
    with pytest.raises(RuntimeError):
        fresh.load_state_dict(incomplete)
