import numpy as np
import pytest
import torch
import tensorplay as tp
from tensorplay.ao.nn.quantized import functional as F

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])
DTYPES = [('int8', -17), ('uint8', 113), ('int32', -17)]


def pair(dtype, zp, device, strided=False):
    codes = np.arange(-128 if dtype != 'uint8' else 0,
                      128 if dtype != 'uint8' else 256, dtype='int64').astype(dtype)
    codes = np.tile(codes, 33).reshape(33, 256)
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), .078, zp)
    reference = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .078, zp)
    if strided:
        x, reference = x.transpose(0, 1), reference.transpose(0, 1)
    return x, reference


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', DTYPES)
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('op', ['relu', 'relu6', 'leaky_relu', 'elu', 'hardswish',
                               'hardsigmoid', 'sigmoid', 'tanh'])
def test_activation_storage_and_values(device, dtype, zp, strided, op):
    x, r = pair(dtype, zp, device, strided)
    unsigned = dtype == 'uint8'
    ys, yz = .063, 117 if unsigned else -19
    if op == 'relu':
        y, expected = F.relu(x), torch.relu(r)
    elif op == 'relu6':
        y, expected = F.relu6(x), torch.ops.quantized.relu6(r, False)
    elif op == 'leaky_relu':
        y = F.leaky_relu(x, .21, ys, yz)
        expected = torch.ops.quantized.leaky_relu(r, .21, False, ys, yz)
    elif op == 'elu':
        y = tp._C.quantized_elu(x, ys, yz, 1.3, .7, .6)
        expected = torch.ops.quantized.elu(r, ys, yz, 1.3, .7, .6)
    elif op == 'hardswish':
        y, expected = F.hardswish(x, ys, yz), torch.ops.quantized.hardswish(r, ys, yz)
    elif op == 'sigmoid':
        y, expected = F.sigmoid(x, ys, yz), torch.ops.quantized.sigmoid(r, ys, yz)
    elif op == 'tanh':
        ys, yz = (2. / 2**32 if dtype == 'int32' else 1. / 128), 128 if unsigned else 0
        y, expected = F.tanh(x, ys, yz), torch.tanh(r)
    else:
        ys = 1. / 2**32 if dtype == 'int32' else 1. / 256
        yz = -128 if dtype == 'int8' else 0
        y, expected = F.hardsigmoid(x, ys, yz), torch.nn.functional.hardsigmoid(r)
    a, b = y.int_repr().cpu().numpy().astype('int64'), expected.int_repr().numpy().astype('int64')
    tolerance = 0 if op in ('relu', 'relu6') else (512 if dtype == 'int32' else 1)
    assert np.max(np.abs(a - b)) <= tolerance
    assert y.dtype == x.dtype
    assert y.q_scale() == pytest.approx(expected.q_scale())
    assert y.q_zero_point() == expected.q_zero_point()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', DTYPES)
def test_empty_invalid_and_saturation(device, dtype, zp):
    empty = tp._C._make_per_tensor_quantized_tensor(tp.tensor(np.zeros((2, 0, 3), dtype=dtype), device=device), .1, zp)
    for fn in (F.relu, F.relu6, lambda x: F.sigmoid(x, .01, zp)):
        assert tuple(fn(empty).shape) == (2, 0, 3)
    x, _ = pair(dtype, zp, device)
    for scale in (0., -1., float('inf'), float('nan')):
        with pytest.raises((ValueError, RuntimeError)):
            F.sigmoid(x, scale, zp)
    bounds = np.iinfo(dtype)
    for bad_zp in (int(bounds.min) - 1, int(bounds.max) + 1):
        with pytest.raises((ValueError, RuntimeError)):
            F.sigmoid(x, .1, bad_zp)
    codes = np.array([bounds.min, bounds.max], dtype=dtype)
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), 1e6, zp)
    y = F.leaky_relu(x, 1., 1e-6, 0 if dtype != 'uint8' else 128)
    np.testing.assert_array_equal(y.int_repr().cpu().numpy(), codes)


def test_lookup_cache_distinguishes_activation_parameters():
    x, r = pair('int8', -17, 'cpu')
    for slope in (.2, .7, .2, -.3):
        for scale in (.03, .3, .03):
            y = F.leaky_relu(x, slope, scale, 3)
            expected = torch.ops.quantized.leaky_relu(r, slope, False, scale, 3)
            np.testing.assert_allclose(y.int_repr().numpy(), expected.int_repr().numpy(), atol=1, rtol=0)
