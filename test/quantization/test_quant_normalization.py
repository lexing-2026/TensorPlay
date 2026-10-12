"""Native quantized normalization contracts and eager conversion coverage."""

import numpy as np
import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
DTYPES = [(tp.qint8, -3, np.int8), (tp.quint8, 123, np.uint8),
          (tp.qint32, 7, np.int32)]


def _reference(codes, scale, zero_point, kind, weight, bias, eps, out_scale, out_zp):
    x = (codes.astype(np.float32) - zero_point) * np.float32(scale)
    if kind == "layer":
        grouped = x.reshape(x.shape[0], -1)
        affine_shape = x.shape[1:]
    else:
        groups = 2 if kind == "group" else x.shape[1]
        grouped = x.reshape(x.shape[0] * groups, -1)
        affine_shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    mean = grouped.mean(axis=1, keepdims=True)
    variance = ((grouped - mean) ** 2).mean(axis=1, keepdims=True)
    y = ((grouped - mean) / np.sqrt(variance + eps)).reshape(x.shape)
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float32).reshape(affine_shape)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32).reshape(affine_shape)
    bounds = np.iinfo(codes.dtype)
    return np.clip(np.rint(y / np.float32(out_scale)) + out_zp,
                   bounds.min, bounds.max).astype(np.int64)


def _run(kind, qx, weight, bias, out_scale=0.04, out_zp=0, eps=1e-5):
    if kind == "layer":
        return tp._C.quantized_layer_norm(qx, list(qx.shape[1:]), weight, bias,
                                          eps, out_scale, out_zp)
    if kind == "group":
        return tp._C.quantized_group_norm(qx, 2, weight, bias, eps, out_scale, out_zp)
    return tp._C.quantized_instance_norm(qx, weight, bias, eps, out_scale, out_zp)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype,zero_point,storage", DTYPES)
@pytest.mark.parametrize("kind", ["layer", "group", "instance"])
@pytest.mark.parametrize("affine", [False, True])
def test_normalization_codes(device, dtype, zero_point, storage, kind, affine):
    x = np.random.default_rng(4).normal(size=(2, 4, 3, 5)).astype(np.float32)
    qx = tp.quantize_per_tensor(tp.tensor(x).to(device), 0.13, zero_point, dtype)
    shape = (4, 3, 5) if kind == "layer" else (4,)
    weight = tp.linspace(0.25, 1.5, int(np.prod(shape))).reshape(shape).to(device) if affine else None
    bias = tp.linspace(-0.3, 0.4, int(np.prod(shape))).reshape(shape).to(device) if affine else None
    y = _run(kind, qx, weight, bias, out_zp=zero_point)
    expected = _reference(np.asarray(qx.int_repr().cpu().tolist(), dtype=storage),
                          0.13, zero_point, kind,
                          weight.cpu().tolist() if affine else None,
                          bias.cpu().tolist() if affine else None,
                          1e-5, 0.04, zero_point)
    actual = np.asarray(y.int_repr().cpu().tolist(), dtype=np.int64)
    assert np.max(np.abs(actual - expected)) <= 1
    assert y.dtype == dtype
    assert y.q_scale() == 0.04
    assert y.q_zero_point() == zero_point


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["layer", "group", "instance"])
def test_strided_input_and_affine(device, kind):
    qx = tp.quantize_per_tensor(tp.randn(2, 4, 3, 5).to(device), 0.1, 0, tp.qint8).transpose(2, 3)
    if kind == "layer":
        weight = tp.randn(4, 3, 5).to(device).transpose(1, 2)
        bias = tp.randn(4, 3, 5).to(device).transpose(1, 2)
    else:
        weight = tp.randn(2, 4).to(device).transpose(0, 1)[:, 0]
        bias = tp.randn(2, 4).to(device).transpose(0, 1)[:, 0]
    actual = _run(kind, qx, weight, bias).int_repr().cpu().tolist()
    expected = _run(kind, qx.contiguous(), weight.contiguous(), bias.contiguous()).int_repr().cpu().tolist()
    assert actual == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["group", "instance"])
@pytest.mark.parametrize("dtype,zero_point,storage", DTYPES)
def test_channels_last(device, kind, dtype, zero_point, storage):
    qx = tp.quantize_per_tensor(tp.randn(2, 4, 3, 5).to(device), 0.1, zero_point, dtype)
    qlast = qx.contiguous(memory_format=tp.channels_last)
    y = _run(kind, qlast, None, None, out_zp=zero_point)
    reference = _run(kind, qx, None, None, out_zp=zero_point)
    assert y.is_contiguous(memory_format=tp.channels_last)
    assert y.int_repr().cpu().tolist() == reference.int_repr().cpu().tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(0, 4, 3), (2, 4, 0)])
@pytest.mark.parametrize("kind", ["layer", "group", "instance"])
def test_empty_output_qparams(device, shape, kind):
    qx = tp.quantize_per_tensor(tp.empty(shape).to(device), 0.1, 0, tp.qint8)
    y = _run(kind, qx, None, None, out_scale=0.25, out_zp=-7)
    assert tuple(y.shape) == shape
    assert y.numel() == 0
    assert y.q_scale() == 0.25
    assert y.q_zero_point() == -7


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_geometry_and_parameters(device):
    qx = tp.quantize_per_tensor(tp.randn(2, 4, 3).to(device), 0.1, 0, tp.qint8)
    for shape in ([], [6], [2, 6], [1, 2, 4, 3], [-3]):
        with pytest.raises(ValueError):
            tp._C.quantized_layer_norm(qx, shape, None, None, 1e-5, 0.1, 0)
    for weight, bias in [(tp.ones(12).to(device), None), (None, tp.ones(12).to(device))]:
        with pytest.raises(ValueError):
            tp._C.quantized_layer_norm(qx, [4, 3], weight, bias, 1e-5, 0.1, 0)
    for groups in (0, -1, 3):
        with pytest.raises(ValueError):
            tp._C.quantized_group_norm(qx, groups, None, None, 1e-5, 0.1, 0)
    for kind in ("group", "instance"):
        with pytest.raises(ValueError):
            _run(kind, qx, tp.ones(3).to(device), None)
    with pytest.raises(TypeError):
        _run("layer", qx, tp.ones(4, 3, dtype=tp.float64).to(device), None)
    with pytest.raises(TypeError):
        _run("layer", tp.randn(2, 4, 3).to(device), None, None)
    with pytest.raises(ValueError):
        tp._C.quantized_group_norm(qx[:, :, 0], 2, None, None, 1e-5, 0.1, 0)
    for scale, zero_point in [(0.0, 0), (-1.0, 0), (float("inf"), 0), (float("nan"), 0), (0.1, 128)]:
        with pytest.raises(ValueError):
            _run("layer", qx, None, None, out_scale=scale, out_zp=zero_point)
    qpc = tp.quantize_per_channel(tp.randn(2, 4, 3).to(device),
                                  tp.ones(4).to(device), tp.zeros(4, dtype=tp.long).to(device),
                                  1, tp.qint8)
    with pytest.raises(RuntimeError):
        _run("layer", qpc, None, None)


@pytest.mark.parametrize("device", DEVICES)
def test_wide_storage_square_sum_and_saturation(device):
    codes = tp.tensor([[-2147483648, -1000000000, 1000000000, 2147483647]], dtype=tp.int32).to(device)
    qx = tp._C._make_per_tensor_quantized_tensor(codes, 1e-8, 0)
    y = _run("layer", qx, None, None, out_scale=0.01)
    values = np.asarray(y.int_repr().cpu().tolist())[0]
    assert np.all(np.abs(values) < 200)
    assert np.all(np.diff(values) > 0)
    for dtype, _, storage in DTYPES:
        x = tp.quantize_per_tensor(tp.tensor([[-1.0, 1.0]]).to(device), 0.1, 0, dtype)
        result = _run("layer", x, None, None, out_scale=1e-10).int_repr().cpu().tolist()[0]
        limits = np.iinfo(storage)
        assert result == [limits.min, limits.max]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_grid_stride_groups_and_parameter_device():
    qx = tp.quantize_per_tensor(tp.zeros(65537, 2, device="cuda"), 0.1, 0, tp.qint8)
    y = _run("layer", qx, None, None)
    assert y.int_repr().cpu().tolist() == [[0, 0]] * 65537
    with pytest.raises(ValueError):
        _run("layer", qx, tp.ones(2), None)
