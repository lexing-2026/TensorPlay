import numpy as np
import pytest
import tensorplay as tp
import torch


DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])
TYPES = [('int8', -11), ('uint8', 119), ('int32', -11)]


def pair(device, dtype='uint8', zp=119, shape=(2, 4, 5, 7), strided=False):
    values = (np.arange(np.prod(shape)).reshape(shape) % 73 - 36 + zp).astype(dtype)
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(values, device=device), .07, zp)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(values), .07, zp)
    if strided:
        x, r = x[..., ::2], r[..., ::2]
    return x, r


def resolve(namespace, name):
    packet, _, overload = name.partition('.')
    return getattr(getattr(namespace, packet), overload or 'default')


def equal(actual, expected, tolerance=0):
    assert tuple(actual.shape) == tuple(expected.shape)
    if expected.is_quantized:
        assert actual.q_scale() == expected.q_scale()
        assert actual.q_zero_point() == expected.q_zero_point()
        assert str(actual.dtype).split('.')[-1] == str(expected.dtype).split('.')[-1]
        a, b = actual.int_repr().cpu().numpy(), expected.int_repr().numpy()
    else:
        a, b = actual.cpu().numpy(), expected.numpy()
    np.testing.assert_allclose(a.astype('float64'), b.astype('float64'), rtol=0, atol=tolerance)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('tensor_value', [False, True])
def test_fill_packets_round_saturate_and_alias(device, dtype, zp, strided, tensor_value):
    for value in [-1e8, -.105, 0., .105, 1e8]:
        x, r = pair(device, dtype, zp, strided=strided)
        name = 'fill_.Tensor' if tensor_value else 'fill_.Scalar'
        v, rv = (tp.tensor(value, device=device), torch.tensor(value)) if tensor_value else (value, value)
        address, strides = x.data_ptr(), x.stride()
        returned = resolve(tp.ops.tp, name)(x, v)
        resolve(torch.ops.aten, name)(r, rv)
        assert returned.data_ptr() == address and x.stride() == strides
        equal(x, r)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('tensor_value', [False, True])
def test_masked_fill_packets_broadcast_and_strides(device, dtype, zp, tensor_value):
    x, r = pair(device, dtype, zp, strided=True)
    mask = np.array([True, False, True, False]).reshape(1, 1, 1, 4)
    m, rm = tp.tensor(mask, device=device), torch.tensor(mask)
    v, rv = (tp.tensor(-.105, device=device), torch.tensor(-.105)) if tensor_value else (-.105, -.105)
    name = 'masked_fill_.Tensor' if tensor_value else 'masked_fill_.Scalar'
    address, strides = x.data_ptr(), x.stride()
    returned = resolve(tp.ops.tp, name)(x, m, v)
    resolve(torch.ops.aten, name)(r, rm, rv)
    assert returned.data_ptr() == address and x.stride() == strides
    equal(x, r)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('operation', ['max', 'min'])
@pytest.mark.parametrize('strided', [False, True])
def test_extrema_packets_indices_ties_and_out(device, dtype, zp, operation, strided):
    x, r = pair(device, dtype, zp, strided=strided)
    equal(resolve(tp.ops.tp, operation)(x), resolve(torch.ops.aten, operation)(r))
    a = resolve(tp.ops.tp, operation + '.dim')(x, -1, True)
    b = resolve(torch.ops.aten, operation + '.dim')(r, -1, True)
    equal(a[0], b[0]); equal(a[1], b[1])
    out, rout = pair(device, dtype, zp, shape=())
    returned = resolve(tp.ops.tp, operation + '.unary_out')(x, out=out)
    resolve(torch.ops.aten, operation + '.unary_out')(r, out=rout)
    assert returned.data_ptr() == out.data_ptr()
    equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('dims', [[-1], [1, 3], []])
def test_moment_packets_dims_correction_and_out(device, dtype, zp, strided, dims):
    x, r = pair(device, dtype, zp, strided=strided)
    for operation, kwargs in [('mean', {}), ('std', {'correction': .5})]:
        name = 'mean.dim' if operation == 'mean' else 'std.correction'
        a = resolve(tp.ops.tp, name)(x, dims, keepdim=True, **kwargs)
        b = resolve(torch.ops.aten, name)(r, dims, keepdim=True, **kwargs)
        equal(a, b)
        out, rout = pair(device, dtype, zp, shape=tuple(b.shape))
        name = 'mean.out' if operation == 'mean' else 'std.correction_out'
        returned = resolve(tp.ops.tp, name)(x, dims, keepdim=True, out=out, **kwargs)
        resolve(torch.ops.aten, name)(r, dims, keepdim=True, out=rout, **kwargs)
        assert returned.data_ptr() == out.data_ptr()
        equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
def test_index_sort_flip_and_shuffle_packets(device, dtype, zp):
    x, r = pair(device, dtype, zp, strided=True)
    indices = np.array([3, 1, 1, 0], dtype='int64')
    ix, ri = tp.tensor(indices, device=device), torch.tensor(indices)
    equal(tp.ops.tp.index.Tensor(x, [None, ix]), torch.ops.aten.index.Tensor(r, [None, ri]))
    equal(tp.ops.tp.index_select.default(x, 1, ix), torch.ops.aten.index_select.default(r, 1, ri))
    out, rout = pair(device, dtype, zp, shape=tuple(r.shape))
    result = tp.ops.tp.index_select.out(x, 1, ix, out=out)
    torch.ops.aten.index_select.out(r, 1, ri, out=rout)
    assert result.data_ptr() == out.data_ptr()
    equal(out, rout)
    equal(tp.ops.tp.flip.default(x, [1, 3]), torch.ops.aten.flip.default(r, [1, 3]))
    equal(tp.ops.tp.channel_shuffle.default(x, 2), torch.ops.aten.channel_shuffle.default(r, 2))
    a = tp.ops.tp.sort.stable(x, stable=True, dim=-1, descending=True)
    b = torch.ops.aten.sort.stable(r, stable=True, dim=-1, descending=True)
    equal(a[0], b[0]); equal(a[1], b[1])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
def test_index_put_packet_float_values_negative_indices(device, dtype, zp):
    x, r = pair(device, dtype, zp, strided=True)
    indices = np.array([-1, 1], dtype='int64')
    values = np.linspace(-.7, .7, 20, dtype='float32').reshape(1, 5, 4)
    ix, ri = tp.tensor(indices, device=device), torch.tensor(indices)
    v, rv = tp.tensor(values, device=device), torch.tensor(values)
    address = x.data_ptr()
    returned = tp.ops.tp._index_put_impl_.default(x, [None, ix], v, False, False)
    torch.ops.aten._index_put_impl_.default(r, [None, ri], rv, False, False)
    assert returned.data_ptr() == address
    equal(x, r)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('name,args', [
    ('gelu', ('none',)), ('gelu', ('tanh',)), ('gelu_', ('none',)),
    ('hardsigmoid', ()), ('hardtanh', (-.6, .7)), ('hardtanh_', (-.6, .7)),
    ('leaky_relu', (.17,)), ('leaky_relu_', (.17,)), ('relu_', ()),
    ('sigmoid', ()), ('tanh', ()), ('threshold', (.2, -.4)),
])
def test_activation_packets_preserve_or_replace_grid(device, dtype, zp, name, args):
    x, r = pair(device, dtype, zp, strided=True)
    address = x.data_ptr()
    a = resolve(tp.ops.tp, name)(x, *args)
    b = resolve(torch.ops.aten, name)(r, *args)
    equal(a, b, 1 if name in ('gelu', 'gelu_', 'sigmoid', 'tanh') else 0)
    if name.endswith('_'):
        assert a.data_ptr() == address
    if name in ('hardsigmoid', 'hardtanh', 'leaky_relu'):
        out, rout = pair(device, dtype, zp, shape=tuple(b.shape))
        returned = resolve(tp.ops.tp, name + '.out')(x, *args, out=out)
        resolve(torch.ops.aten, name + '.out')(r, *args, out=rout)
        assert returned.data_ptr() == out.data_ptr()
        equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
def test_prelu_packet_quantized_channel_weights(device):
    x, r = pair(device, strided=True)
    codes = np.array([10, 20, 30, 40], dtype='uint8').reshape(1, 4, 1, 1)
    w = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), .01, 0)
    rw = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .01, 0)
    equal(tp.ops.tp._prelu_kernel.default(x, w), torch.ops.aten._prelu_kernel.default(r, rw))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('rank', [2, 3])
def test_pool_packets_strides_padding_and_out(device, dtype, zp, rank):
    shape = (2, 3) + (5,) * rank
    x, r = pair(device, dtype, zp, shape=shape, strided=True)
    args = ([3] * rank, [2] * rank, [1] * rank, True, False, None)
    name = f'avg_pool{rank}d'
    equal(resolve(tp.ops.tp, name)(x, *args), resolve(torch.ops.aten, name)(r, *args), 1)
    name = f'_adaptive_avg_pool{rank}d'
    a, b = resolve(tp.ops.tp, name)(x, [2] * rank), resolve(torch.ops.aten, name)(r, [2] * rank)
    equal(a, b, 1)
    if rank == 3:
        out, rout = pair(device, dtype, zp, shape=tuple(b.shape))
        result = tp.ops.tp.adaptive_avg_pool3d.out(x, [2] * rank, out=out)
        torch.ops.aten.adaptive_avg_pool3d.out(r, [2] * rank, out=rout)
        assert result.data_ptr() == out.data_ptr()
        equal(out, rout, 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
def test_cat_out_and_reflection_padding_packets(device, dtype, zp):
    x, r = pair(device, dtype, zp, strided=True)
    out, rout = pair(device, dtype, zp, shape=(2, 8, 5, 4))
    result = tp.ops.tp.cat.out([x, x], 1, out=out)
    torch.ops.aten.cat.out([r, r], 1, out=rout)
    assert result.data_ptr() == out.data_ptr()
    equal(out, rout)
    equal(tp.ops.tp.reflection_pad2d.default(x, [1, 2, 2, 1]),
          torch.ops.aten.reflection_pad2d.default(r, [1, 2, 2, 1]))
    for rank in [1, 2]:
        source, reference = (x.reshape([8, 5, 4]), r.reshape([8, 5, 4])) if rank == 1 else (x, r)
        padding = [1, 2] if rank == 1 else [1, 2, 2, 1]
        shape = (8, 5, 7) if rank == 1 else (2, 4, 8, 7)
        out, rout = pair(device, dtype, zp, shape=shape)
        result = resolve(tp.ops.tp, f'reflection_pad{rank}d.out')(source, padding, out=out)
        resolve(torch.ops.aten, f'reflection_pad{rank}d.out')(reference, padding, out=rout)
        assert result.data_ptr() == out.data_ptr()
        equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('padding', [0, 1, 2])
@pytest.mark.parametrize('align_corners', [False, True])
def test_grid_sampler_packet_integer_interpolation(device, padding, align_corners):
    x, r = pair(device, strided=True)
    grid = np.array([[-1.2, -1.1], [.3, .7], [1.1, 1.3], [0., 0.]], dtype='float32').reshape(1, 2, 2, 2)
    grid = np.repeat(grid, 2, axis=0)
    equal(tp.ops.tp.grid_sampler_2d.default(x, tp.tensor(grid, device=device), 0, padding, align_corners),
          torch.ops.aten.grid_sampler_2d.default(r, torch.tensor(grid), 0, padding, align_corners))


@pytest.mark.parametrize('device', DEVICES)
def test_resize_and_storage_set_packets_preserve_grid(device):
    x, r = pair(device, shape=(24,))
    result = tp.ops.tp.resize_.default(x, [2, 3, 4])
    torch.ops.aten.resize_.default(r, [2, 3, 4])
    assert result.data_ptr() == x.data_ptr()
    equal(x, r)
    source, reference = pair(device, shape=(40,))
    result = tp.ops.tp.set_.source_Storage_storage_offset(x, source.untyped_storage(), 2, [2, 3], [6, 2])
    torch.ops.aten.set_.source_Storage_storage_offset(r, reference.untyped_storage(), 2, [2, 3], [6, 2])
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert x.storage_offset() == r.storage_offset()
    equal(x, r)
