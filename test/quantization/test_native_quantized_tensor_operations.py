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
        reference = r.contiguous()
        resolve(torch.ops.aten, name)(reference, rv)
        assert returned.data_ptr() == address and x.stride() == strides
        equal(x, reference)


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
        original = out.int_repr().cpu().numpy().copy()
        address = out.data_ptr()
        returned = resolve(tp.ops.tp, name)(x, dims, keepdim=True, out=out, **kwargs)
        expected = resolve(torch.ops.aten, name)(r, dims, keepdim=True, out=rout, **kwargs)
        assert returned.data_ptr() != address and out.data_ptr() == address
        equal(returned, expected)
        np.testing.assert_array_equal(out.int_repr().cpu().numpy(), original)


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
    if dtype == 'uint8':
        equal(tp.ops.tp.channel_shuffle.default(x, 2), torch.ops.aten.channel_shuffle.default(r, 2))
    a = tp.ops.tp.sort.stable(x, stable=True, dim=-1, descending=True)
    b = torch.ops.aten.sort.stable(r, stable=True, dim=-1, descending=True)
    equal(a[0], b[0]); equal(a[1], b[1])
    for largest in (False, True):
        a = tp.ops.tp.topk.default(x, 2, -1, largest, True)
        b = torch.ops.aten.topk.default(r, 2, -1, largest, True)
        equal(a[0], b[0]); equal(a[1], b[1])
    assert tp.ops.tp.equal.default(x, x) == torch.ops.aten.equal.default(r, r)
    raw = tp.ops.tp.int_repr.default(x)
    equal(raw, r.int_repr())
    raw.fill_(0)
    equal(x, r)


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
    kwargs = {'approximate': args[0]} if name.startswith('gelu') else {}
    call_args = () if kwargs else args
    a = resolve(tp.ops.tp, name)(x, *call_args, **kwargs)
    b = resolve(torch.ops.aten, name)(r, *call_args, **kwargs)
    equal(a, b, 1 if name in ('gelu', 'gelu_', 'sigmoid', 'tanh') else 0)
    if name.endswith('_'):
        assert a.data_ptr() == address
    if name in ('hardsigmoid', 'hardtanh', 'leaky_relu'):
        out, rout = pair(device, dtype, zp, shape=tuple(b.shape))
        address = out.data_ptr()
        returned = resolve(tp.ops.tp, name + '.out')(x, *args, out=out)
        expected = resolve(torch.ops.aten, name + '.out')(r, *args, out=rout)
        if name == 'hardtanh':
            assert returned.data_ptr() != address and out.data_ptr() == address
        else:
            assert returned.data_ptr() == out.data_ptr()
        equal(returned, expected)
        equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('weight_shape', [(1,), (1, 4, 1, 1), (2, 4, 5, 4), (1, 1, 5, 1)])
def test_prelu_packet_broadcast_quantized_weights(device, dtype, zp, weight_shape):
    x, r = pair(device, dtype, zp, strided=True)
    codes = ((np.arange(np.prod(weight_shape)) * 7) % 63).astype(dtype).reshape(weight_shape)
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
        address = out.data_ptr()
        result = tp.ops.tp.adaptive_avg_pool3d.out(x, [2] * rank, out=out)
        expected = torch.ops.aten.adaptive_avg_pool3d.out(r, [2] * rank, out=rout)
        assert result.data_ptr() != address and out.data_ptr() == address
        equal(result, expected, 1)
        equal(out, rout)


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


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('operation', ['cat', 'leaky_relu', 'reflection_pad2d'])
def test_output_quantizer_selection(device, operation):
    x, r = pair(device, strided=True)
    shape = (2, 8, 5, 4) if operation == 'cat' else tuple(r.shape)
    out = tp.quantize_per_tensor(tp.zeros(shape, device=device), .19, 87, tp.quint8)
    rout = torch._empty_affine_quantized(shape, scale=.19, zero_point=87, dtype=torch.quint8)
    if operation == 'cat':
        tp.ops.tp.cat.out([x, x], 1, out=out)
        torch.ops.aten.cat.out([r, r], 1, out=rout)
    elif operation == 'leaky_relu':
        tp.ops.tp.leaky_relu.out(x, .17, out=out)
        torch.ops.aten.leaky_relu.out(r, .17, out=rout)
    else:
        tp.ops.tp.reflection_pad2d.out(x, [1, 2, 2, 1], out=out)
        torch.ops.aten.reflection_pad2d.out(r, [1, 2, 2, 1], out=rout)
    equal(out, rout)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('invalid', ['fill_rank', 'mask_dtype', 'mask_value_rank',
                                     'repeated_dims', 'accumulate', 'quantized_values',
                                     'grid_interpolation', 'prelu_float_weight'])
def test_quantized_tensor_operation_rejections(device, invalid):
    x, r = pair(device)
    args = {
        'fill_rank': ('fill_.Tensor', [tp.ones([1], device=device)], [torch.ones(1)]),
        'mask_dtype': ('masked_fill_.Scalar', [tp.ones(list(x.shape), device=device), 0.],
                       [torch.ones(r.shape), 0.]),
        'mask_value_rank': ('masked_fill_.Tensor', [tp.ones(list(x.shape), dtype=tp.bool, device=device), tp.ones([1], device=device)],
                            [torch.ones(r.shape, dtype=torch.bool), torch.ones(1)]),
        'repeated_dims': ('mean.dim', [[1, 1]], [[1, 1]]),
        'accumulate': ('_index_put_impl_', [[tp.tensor([0], device=device)], tp.tensor(0., device=device), True, False],
                       [[torch.tensor([0])], torch.tensor(0.), True, False]),
        'quantized_values': ('_index_put_impl_', [[tp.tensor([0], device=device)], x[0], False, False],
                             [[torch.tensor([0])], r[0], False, False]),
        'grid_interpolation': ('grid_sampler_2d', [tp.zeros([2, 2, 2, 2], device=device), 1, 0, False],
                              [torch.zeros(2, 2, 2, 2), 1, 0, False]),
        'prelu_float_weight': ('_prelu_kernel', [tp.ones([1], device=device)], [torch.ones(1)]),
    }
    name, actual_args, reference_args = args[invalid]
    with pytest.raises((RuntimeError, ValueError, TypeError, IndexError, NotImplementedError)):
        resolve(torch.ops.aten, name)(r, *reference_args)
    with pytest.raises((RuntimeError, ValueError, TypeError, IndexError, NotImplementedError)):
        resolve(tp.ops.tp, name)(x, *actual_args)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
def test_fill_quantized_scalar_alias(device, dtype, zp):
    x, r = pair(device, dtype, zp, strided=True)
    value, rv = x[0, 0, 0, 0], r[0, 0, 0, 0]
    address, strides = x.data_ptr(), x.stride()
    tp.ops.tp.fill_.Tensor(x, value)
    reference = r.contiguous()
    torch.ops.aten.fill_.Tensor(reference, rv)
    assert x.data_ptr() == address and x.stride() == strides
    equal(x, reference)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('correction', [0., .5, 4., 7.])
def test_std_correction_degenerate_denominators(device, dtype, zp, correction):
    x, r = pair(device, dtype, zp, shape=(2, 4))
    equal(tp.ops.tp.std.correction(x, [-1], correction=correction),
          torch.ops.aten.std.correction(r, [-1], correction=correction))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
def test_inner_moments_vector_blocks_and_tails(device, dtype, zp):
    x, r = pair(device, dtype, zp, shape=(2, 3, 8193))
    equal(tp.ops.tp.mean.dim(x, [-1], True), torch.ops.aten.mean.dim(r, [-1], True))
    equal(tp.ops.tp.std.correction(x, [-1], correction=.5, keepdim=True),
          torch.ops.aten.std.correction(r, [-1], correction=.5, keepdim=True))


@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('width', [65535, 65536, 65537, 262145])
def test_byte_standard_deviation_extreme_codes_and_accumulator_chunks(dtype, zp, width):
    limits = np.iinfo(dtype)
    codes = np.where(np.arange(2 * width) % 2, limits.min, limits.max).astype(dtype).reshape(2, width)
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes), .07, zp)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .07, zp)
    equal(tp.ops.tp.std.correction(x, [-1], correction=.5), torch.ops.aten.std.correction(r, [-1], correction=.5))


@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('channels', [7, 32, 65])
def test_volumetric_pool_large_channel_major_storage(dtype, zp, channels):
    rng = np.random.default_rng(915)
    limits = np.iinfo(dtype)
    codes = rng.integers(limits.min, limits.max + 1, size=(2, channels, 17, 15, 13), dtype=dtype)
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes), .07, zp)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .07, zp)
    actual = tp.ops.tp.avg_pool3d.default(x, [3, 3, 3], [2, 2, 2], [1, 1, 1], True, False)
    expected = torch.ops.aten.avg_pool3d.default(r, [3, 3, 3], [2, 2, 2], [1, 1, 1], True, False)
    equal(actual, expected, 1)
    assert actual.stride() == expected.stride()


def test_batch_norm_coefficients_track_parameter_mutations():
    x, r = pair('cpu')
    x, r = x.contiguous(memory_format=tp.channels_last), r.contiguous(memory_format=torch.channels_last)
    data = [np.linspace(.2, .8, 4, dtype='float32'), np.zeros(4, dtype='float32'),
            np.linspace(-.1, .1, 4, dtype='float32'), np.linspace(.5, 1., 4, dtype='float32')]
    parameters, references = [tp.tensor(v) for v in data], [torch.tensor(v) for v in data]
    for changed in [None, 0, 1, 2, 3, None]:
        if changed is not None:
            parameters[changed].add_(.03)
            references[changed].add_(.03)
        equal(tp.ops.quantized.batch_norm2d(x, *parameters, 1e-5, .07, 119),
              torch.ops.quantized.batch_norm2d(r, *references, 1e-5, .07, 119))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('rank', [2, 3])
def test_channels_last_activation_prelu_and_moments(device, dtype, zp, rank):
    shape = (2, 4) + (5,) * rank
    x, r = pair(device, dtype, zp, shape=shape)
    fmt = tp.channels_last if rank == 2 else tp.channels_last_3d
    rfmt = torch.channels_last if rank == 2 else torch.channels_last_3d
    x, r = x.contiguous(memory_format=fmt), r.contiguous(memory_format=rfmt)
    w, rw = pair(device, dtype, zp, shape=(1, 4) + (1,) * rank)
    for operation, args, rargs in [('relu', (), ()), ('sigmoid', (), ()), ('tanh', (), ()),
                                   ('hardtanh', (-.6, .7), (-.6, .7)),
                                   ('_prelu_kernel', (w,), (rw,))]:
        actual = resolve(tp.ops.tp, operation)(x, *args)
        expected = resolve(torch.ops.aten, operation)(r, *rargs)
        equal(actual, expected, 1 if operation in ('sigmoid', 'tanh') else 0)
        assert actual.stride() == expected.stride()
    equal(tp.ops.tp.mean.dim(x, [-1], True), torch.ops.aten.mean.dim(r, [-1], True))
    equal(tp.ops.tp.std.correction(x, [-1], correction=.5, keepdim=True),
          torch.ops.aten.std.correction(r, [-1], correction=.5, keepdim=True))


@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('scale', [.005, .07, .5, 2.])
@pytest.mark.parametrize('shape', [(2, 7, 5, 7), (2, 33, 16, 17), (4, 64, 16, 17),
                                  (2, 129, 17, 19), (2, 64, 4, 16, 17), (2, 7, 257, 257),
                                  (1, 33, 33, 33, 33)])
def test_channels_last_spatial_mean_cascade_and_thread_partitions(dtype, zp, scale, shape):
    rng = np.random.default_rng(914)
    limits = np.iinfo(dtype)
    codes = rng.integers(limits.min, limits.max + 1, size=shape, dtype=dtype)
    fmt = tp.channels_last if len(shape) == 4 else tp.channels_last_3d
    rfmt = torch.channels_last if len(shape) == 4 else torch.channels_last_3d
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes), scale, zp).contiguous(memory_format=fmt)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), scale, zp).contiguous(memory_format=rfmt)
    reference_old = torch.get_num_threads()
    try:
        torch.set_num_threads(tp.get_num_threads())
        dims = list(range(2, len(shape)))
        for keepdim in (False, True):
            actual = tp.ops.tp.mean.dim(x, dims, keepdim)
            expected = torch.ops.aten.mean.dim(r, dims, keepdim)
            equal(actual, expected)
            assert actual.stride() == expected.stride()
    finally:
        torch.set_num_threads(reference_old)


@pytest.mark.parametrize('channels', [7, 33, 65, 129])
@pytest.mark.parametrize('dtype,zp', TYPES[:2])
def test_channels_last_spatial_mean_half_grid_rounding(channels, dtype, zp):
    codes = np.broadcast_to(np.arange(4096) % 2 + zp - 4, (2, channels, 4096)).astype(dtype).reshape(2, channels, 16, 16, 16)
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes), .07, zp).contiguous(memory_format=tp.channels_last_3d)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .07, zp).contiguous(memory_format=torch.channels_last_3d)
    reference_old = torch.get_num_threads()
    try:
        torch.set_num_threads(tp.get_num_threads())
        equal(tp.ops.tp.mean.dim(x, [2, 3, 4], False), torch.ops.aten.mean.dim(r, [2, 3, 4], False))
    finally:
        torch.set_num_threads(reference_old)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('strided', [False, True])
def test_channels_last_pool_output_layout(device, dtype, zp, rank, strided):
    x, r = pair(device, dtype, zp, shape=(2, 4) + (5,) * rank)
    fmt = tp.channels_last if rank == 2 else tp.channels_last_3d
    rfmt = torch.channels_last if rank == 2 else torch.channels_last_3d
    x, r = x.contiguous(memory_format=fmt), r.contiguous(memory_format=rfmt)
    if strided:
        x, r = x[..., ::2], r[..., ::2]
    operations = [(f'avg_pool{rank}d', ([3] * rank, [2] * rank, [1] * rank, True, False, None)),
                  (f'_adaptive_avg_pool{rank}d', ([2] * rank,))]
    for name, args in operations:
        actual = resolve(tp.ops.tp, name)(x, *args)
        expected = resolve(torch.ops.aten, name)(r, *args)
        equal(actual, expected, 1)
        assert actual.stride() == expected.stride()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(), (2, 4, 2, 2, 2, 2), (1, 4, 1, 1, 1, 33)])
@pytest.mark.parametrize('dtype,zp', [('int8', 11), ('uint8', 119), ('int32', 11)])
def test_prelu_scalar_and_high_rank_float_weights(device, shape, dtype, zp):
    x, r = pair(device, dtype, zp, shape=shape)
    for count in ([1] if not shape else [1, 4]):
        values = np.linspace(.1, .4, count, dtype='float32')
        w, rw = tp.tensor(values, device=device), torch.tensor(values)
        equal(tp.ops.tp._prelu_kernel.default(x, w), torch.ops.aten._prelu_kernel.default(r, rw))
    w, rw = pair(device, shape=(1,))
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten._prelu_kernel.default(r, rw)
    with pytest.raises((RuntimeError, TypeError)):
        tp.ops.tp._prelu_kernel.default(x, w)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('size', [1, 7, 15, 16, 17, 31, 32, 33, 200])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('operation', ['sigmoid', 'tanh'])
def test_word_activation_vector_blocks_and_scalar_tails(device, size, strided, operation):
    x, r = pair(device, 'int32', -11, shape=(size * (2 if strided else 1),))
    if strided:
        x, r = x[::2], r[::2]
    equal(resolve(tp.ops.tp, operation)(x), resolve(torch.ops.aten, operation)(r))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('channels', [7, 8, 9, 16, 17, 33, 129])
def test_pool_channels_last_vector_blocks_and_tails(device, dtype, zp, rank, channels):
    x, r = pair(device, dtype, zp, shape=(1, channels) + (5,) * rank)
    fmt = tp.channels_last if rank == 2 else tp.channels_last_3d
    rfmt = torch.channels_last if rank == 2 else torch.channels_last_3d
    x, r = x.contiguous(memory_format=fmt), r.contiguous(memory_format=rfmt)
    for include_pad, divisor in [(False, None), (True, None), (True, 5)]:
        args = ([3] * rank, [2] * rank, [1] * rank, True, include_pad, divisor)
        equal(resolve(tp.ops.tp, f'avg_pool{rank}d')(x, *args),
              resolve(torch.ops.aten, f'avg_pool{rank}d')(r, *args), 1)
    equal(resolve(tp.ops.tp, f'_adaptive_avg_pool{rank}d')(x, [2] * rank),
          resolve(torch.ops.aten, f'_adaptive_avg_pool{rank}d')(r, [2] * rank), 1)
    if rank == 3:
        x, r = x[..., ::2], r[..., ::2]
        equal(tp.ops.tp.avg_pool3d.default(x, [3] * rank, [2] * rank, [1] * rank),
              torch.ops.aten.avg_pool3d.default(r, [3] * rank, [2] * rank, [1] * rank), 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('operation', ['sigmoid', 'tanh'])
@pytest.mark.parametrize('scale', [.005, .07, .5, 2.])
def test_word_activation_fine_output_grids(device, operation, scale):
    codes = np.arange(-256, 256, dtype='int32') - 11
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), scale, -11)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), scale, -11)
    equal(resolve(tp.ops.tp, operation)(x), resolve(torch.ops.aten, operation)(r))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('channels_last', [False, True])
@pytest.mark.parametrize('code', [400000000, 1000000000, 2147483600, -2147483600])
def test_pool_word_accumulator_width_and_conversion(device, rank, channels_last, code):
    codes = np.full((1, 4) + (4,) * rank, code, dtype='int32')
    x = tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), .07, -11)
    r = torch._make_per_tensor_quantized_tensor(torch.tensor(codes), .07, -11)
    if channels_last:
        x = x.contiguous(memory_format=tp.channels_last if rank == 2 else tp.channels_last_3d)
        r = r.contiguous(memory_format=torch.channels_last if rank == 2 else torch.channels_last_3d)
    equal(resolve(tp.ops.tp, f'avg_pool{rank}d')(x, [3] * rank, [2] * rank, [1] * rank),
          resolve(torch.ops.aten, f'avg_pool{rank}d')(r, [3] * rank, [2] * rank, [1] * rank))
    equal(resolve(tp.ops.tp, f'_adaptive_avg_pool{rank}d')(x, [2] * rank),
          resolve(torch.ops.aten, f'_adaptive_avg_pool{rank}d')(r, [2] * rank))
