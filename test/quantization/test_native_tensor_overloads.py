import numpy as np
import pytest
import tensorplay as tp
import torch

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def pair(values, device, dtype='uint8', scale=.07, zero_point=117):
    codes = np.asarray(values).astype(dtype)
    return (tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), scale, zero_point),
            torch._make_per_tensor_quantized_tensor(torch.tensor(codes), scale, zero_point))


def resolve(namespace, name):
    packet, _, overload = name.partition('.')
    function = getattr(namespace, packet)
    return getattr(function, overload or 'default')


def plain_equal(actual, expected, atol=0):
    assert tuple(actual.shape) == tuple(expected.shape)
    np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=0, atol=atol)


def quantized_equal(actual, expected):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert str(actual.dtype).split('.')[-1] == str(expected.dtype).split('.')[-1]
    plain_equal(actual.int_repr(), expected.int_repr())
    if expected.qscheme() == torch.per_tensor_affine:
        assert actual.q_scale() == expected.q_scale()
        assert actual.q_zero_point() == expected.q_zero_point()
    else:
        assert actual.q_per_channel_axis() == expected.q_per_channel_axis()
        plain_equal(actual.q_per_channel_scales(), expected.q_per_channel_scales())
        plain_equal(actual.q_per_channel_zero_points(), expected.q_per_channel_zero_points())


def channel_pair(device, shape=(1, 3, 1, 4), axis=1):
    values = np.arange(np.prod(shape), dtype='float32').reshape(shape) / 9 - .7
    scales = np.linspace(.03, .09, shape[axis], dtype='float64')
    zeros = np.arange(shape[axis], dtype='int64') - 1
    return (tp.quantize_per_channel(tp.tensor(values, device=device),
                tp.tensor(scales, device=device), tp.tensor(zeros, device=device), axis, tp.qint8),
            torch.quantize_per_channel(torch.tensor(values), torch.tensor(scales),
                torch.tensor(zeros), axis, torch.qint8))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', [('int8', -11), ('uint8', 117), ('int32', -111)])
@pytest.mark.parametrize('operation', ['eq', 'ne', 'ge', 'le', 'gt', 'lt'])
@pytest.mark.parametrize('strided', [False, True])
def test_comparison_packets_broadcast_grids_and_out(device, dtype, zp, operation, strided):
    values = np.arange(48).reshape(2, 4, 6) + zp - 24
    x, r = pair(values, device, dtype, scale=.125, zero_point=zp)
    if strided:
        x, r = x[:, :, ::2], r[:, :, ::2]
    y, s = pair(np.arange(4).reshape(1, 4, 1) + zp - 2,
                device, dtype, scale=.375, zero_point=zp)
    for overload, other, reference in [('Tensor', y, s), ('Scalar', -.375, -.375)]:
        actual = resolve(tp.ops.tp, operation + '.' + overload)(x, other)
        expected = resolve(torch.ops.aten, operation + '.' + overload)(r, reference)
        assert actual.dtype == tp.bool
        plain_equal(actual, expected)
        out = tp.empty((2, 4, x.shape[-1] * 2), device=device, dtype=tp.bool)[:, :, ::2]
        rout = torch.empty((2, 4, r.shape[-1] * 2), dtype=torch.bool)[:, :, ::2]
        address, strides = out.data_ptr(), out.stride()
        returned = resolve(tp.ops.tp, operation + '.' + overload + '_out')(x, other, out=out)
        reference_returned = resolve(torch.ops.aten, operation + '.' + overload + '_out')(
            r, reference, out=rout)
        assert returned.data_ptr() == address
        assert out.stride() == strides
        assert reference_returned.data_ptr() == rout.data_ptr()
        plain_equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('per_channel', [False, True])
def test_quantizer_query_packets_and_dequantize_lists(device, per_channel):
    x, r = channel_pair(device) if per_channel else pair(np.arange(12).reshape(1, 3, 1, 4), device)
    expected_scheme = 1 if per_channel else 0
    assert tp.ops.tp.qscheme(x) == expected_scheme
    if per_channel:
        assert tp.ops.tp.q_per_channel_axis(x) == r.q_per_channel_axis()
        plain_equal(tp.ops.tp.q_per_channel_scales(x), r.q_per_channel_scales())
        plain_equal(tp.ops.tp.q_per_channel_zero_points(x), r.q_per_channel_zero_points())
    plain_equal(tp.ops.tp.dequantize.self(x), torch.ops.aten.dequantize.self(r), 1e-6)
    actual = tp.ops.tp.dequantize.tensors([x, x.clone()])
    expected = torch.ops.aten.dequantize.tensors([r, r.clone()])
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        plain_equal(a, b, 1e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('per_channel', [False, True])
@pytest.mark.parametrize('name,args', [
    ('squeeze', ()), ('squeeze.dim', (0,)), ('squeeze.dims', ([0, 2],)),
    ('unsqueeze', (0,)), ('unsqueeze', (-1,)), ('view', ([1, 3, 4],)),
    ('_reshape_alias', ([1, 3, 4], [12, 4, 1])),
])
def test_shape_packets_preserve_codes_grid_and_alias(device, per_channel, name, args):
    x, r = channel_pair(device) if per_channel else pair(np.arange(12).reshape(1, 3, 1, 4), device)
    actual = resolve(tp.ops.tp, name)(x, *args)
    expected = resolve(torch.ops.aten, name)(r, *args)
    quantized_equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert actual.data_ptr() == x.data_ptr()
    assert expected.data_ptr() == r.data_ptr()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name,args', [
    ('as_strided', ([2, 3], [6, 2], 1)), ('unfold', (1, 2, 1)),
])
def test_strided_view_packets_preserve_offset_and_storage(device, name, args):
    x, r = pair(np.arange(12).reshape(2, 6), device)
    actual = resolve(tp.ops.tp, name)(x, *args)
    expected = resolve(torch.ops.aten, name)(r, *args)
    quantized_equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.data_ptr() - x.data_ptr() == expected.data_ptr() - r.data_ptr()


@pytest.mark.parametrize('device', DEVICES)
def test_per_channel_axis_squeeze_and_stride_rejections(device):
    x, r = channel_pair(device, (2, 1, 3), 1)
    for namespace, tensor in [(tp.ops.tp, x), (torch.ops.aten, r)]:
        with pytest.raises(RuntimeError):
            namespace.squeeze.dim(tensor, 1)
        with pytest.raises(RuntimeError):
            namespace.as_strided(tensor, [2, 3], [3, 1])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('per_channel', [False, True])
@pytest.mark.parametrize('channels_last', [False, True])
def test_clone_and_factory_packets_carry_grid_and_layout(device, per_channel, channels_last):
    x, r = channel_pair(device, (2, 3, 2, 4)) if per_channel else pair(np.arange(48).reshape(2, 3, 2, 4), device)
    amf = tp.channels_last if channels_last else tp.contiguous_format
    rmf = torch.channels_last if channels_last else torch.contiguous_format
    x, r = x.contiguous(memory_format=amf), r.contiguous(memory_format=rmf)
    for name in ['clone', 'empty_like', 'empty_quantized']:
        args = [list(x.shape), x] if name == 'empty_quantized' else [x]
        refs = [list(r.shape), r] if name == 'empty_quantized' else [r]
        options = {'memory_format': amf} if name != 'empty_like' else {}
        reference_options = {'memory_format': rmf} if name != 'empty_like' else {}
        actual = resolve(tp.ops.tp, name)(*args, **options, **({'device': device} if name == 'empty_quantized' else {}))
        expected = resolve(torch.ops.aten, name)(*refs, **reference_options)
        if name != 'clone':
            actual.copy_(x)
            expected.copy_(r)
        quantized_equal(actual, expected)
        assert actual.data_ptr() != x.data_ptr()
        assert actual.stride() == expected.stride()


@pytest.mark.parametrize('device', DEVICES)
def test_unknown_quantizer_factory_and_empty_strided_rejection(device):
    x = tp.ops.tp.empty.memory_format([2, 3], dtype=tp.qint8, device=device)
    r = torch.ops.aten.empty.memory_format([2, 3], dtype=torch.qint8)
    assert x.dtype == tp.qint8 and r.dtype == torch.qint8
    assert tuple(x.shape) == tuple(r.shape)
    for namespace, dtype, options, tensor in [(tp.ops.tp, tp.qint8, {'device': device}, x),
                                              (torch.ops.aten, torch.qint8, {}, r)]:
        with pytest.raises(RuntimeError):
            namespace.qscheme(tensor)
        with pytest.raises(RuntimeError):
            namespace.empty_strided([2, 3], [3, 1], dtype=dtype, **options)


@pytest.mark.parametrize('device', DEVICES)
def test_empty_like_channel_parameters_are_independent(device):
    x, r = channel_pair(device)
    out = tp.ops.tp.empty_like.default(x)
    rout = torch.ops.aten.empty_like.default(r)
    assert out.q_per_channel_scales().data_ptr() != x.q_per_channel_scales().data_ptr()
    assert out.q_per_channel_zero_points().data_ptr() != x.q_per_channel_zero_points().data_ptr()
    out.q_per_channel_scales().fill_(.5)
    rout.q_per_channel_scales().fill_(.5)
    out.q_per_channel_zero_points().fill_(2)
    rout.q_per_channel_zero_points().fill_(2)
    plain_equal(x.q_per_channel_scales(), r.q_per_channel_scales())
    plain_equal(x.q_per_channel_zero_points(), r.q_per_channel_zero_points())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('operation', ['eq', 'ne', 'ge', 'le', 'gt', 'lt'])
def test_comparisons_per_channel_with_nonquantized_operand(device, operation):
    x, r = channel_pair(device)
    values = np.array([-.6, -.3, 0., .3], dtype='float32').reshape(1, 1, 1, 4)
    y, s = tp.tensor(values, device=device), torch.tensor(values)
    plain_equal(resolve(tp.ops.tp, operation + '.Tensor')(x, y),
                resolve(torch.ops.aten, operation + '.Tensor')(r, s))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('channels_last', [False, True])
def test_unknown_quantized_empty_memory_format(device, channels_last):
    amf = tp.channels_last if channels_last else tp.contiguous_format
    rmf = torch.channels_last if channels_last else torch.contiguous_format
    x = tp.ops.tp.empty.memory_format([2, 3, 5, 7], dtype=tp.quint8, device=device, memory_format=amf)
    r = torch.ops.aten.empty.memory_format([2, 3, 5, 7], dtype=torch.quint8, memory_format=rmf)
    assert x.stride() == r.stride()
    with pytest.raises(RuntimeError):
        x.q_scale()
    with pytest.raises(RuntimeError):
        r.q_scale()
