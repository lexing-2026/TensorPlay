import numpy as np
import pytest
import tensorplay as tp
import torch


DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])


def pair(values, device, dtype='uint8', scale=.07, zero_point=117):
    codes = np.asarray(values).astype(dtype)
    return (tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), scale, zero_point),
            torch._make_per_tensor_quantized_tensor(torch.tensor(codes), scale, zero_point))


def equal(actual, expected, tolerance=0):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert actual.q_scale() == pytest.approx(expected.q_scale(), rel=1e-12)
    assert actual.q_zero_point() == expected.q_zero_point()
    np.testing.assert_allclose(actual.int_repr().cpu().numpy().astype('int64'),
                               expected.int_repr().numpy().astype('int64'), rtol=0, atol=tolerance)


def resolve(namespace, name):
    packet, _, overload = name.partition('.')
    function = getattr(namespace, packet)
    return getattr(function, overload) if overload else function


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('strided', [False, True])
def test_public_activation_packets_and_inplace_aliases(device, strided):
    x, r = pair(np.arange(140).reshape(2, 2, 5, 7), device)
    if strided:
        x, r = x.transpose(-1, -2), r.transpose(-1, -2)
    for name, arguments in (
        ('clamp', (-.7, 1.3)), ('threshold', (.3, -.2)),
        ('elu', (.03, 113, 1.2, .8, 1.1)), ('celu', (.03, 113, 1.2)),
        ('hardswish', (.03, 113)), ('sigmoid', (.01, 13)),
        ('softmax', (-1, .01, 13)), ('dropout', (.01, 13, .9, True)),
        ('relu6', (False,)), ('leaky_relu', (.13, False, .03, 113)),
    ):
        actual = resolve(tp.ops.quantized, name)(x, *arguments)
        expected = resolve(torch.ops.quantized, name)(r, *arguments)
        equal(actual, expected, 1)
        if name == 'dropout':
            assert actual is x and expected.data_ptr() == r.data_ptr()
    w, rw = pair([5, 7], device, scale=.03, zero_point=0)
    equal(tp.ops.quantized.prelu(x, w, .03, 113),
          torch.ops.quantized.prelu(r, rw, .03, 113), 1)
    for name, args in (('relu6', (True,)), ('leaky_relu', (.13, True, .03, 113))):
        a, b = x.clone(), r.clone()
        address = a.data_ptr()
        if name == 'leaky_relu':
            with pytest.warns(UserWarning, match='Inplace execution'):
                actual = resolve(tp.ops.quantized, name)(a, *args)
        else:
            actual = resolve(tp.ops.quantized, name)(a, *args)
        expected = resolve(torch.ops.quantized, name)(b, *args)
        assert a.data_ptr() == address
        if name == 'leaky_relu':
            assert actual is not a and actual.data_ptr() != address
            assert expected.data_ptr() != b.data_ptr()
            equal(a, b)
        else:
            assert actual is a and expected.data_ptr() == b.data_ptr()
        equal(actual, expected, 1)


@pytest.mark.parametrize('device', DEVICES)
def test_public_matmul_pooling_and_private_add(device):
    x, r = pair(np.arange(70).reshape(2, 5, 7), device)
    y, s = pair(np.arange(21).reshape(7, 3), device, scale=.13, zero_point=103)
    equal(tp.ops.quantized.matmul(x, y, .1, 113), torch.ops.quantized.matmul(r, s, .1, 113), 1)
    equal(tp.ops._quantized.add(x, x, .1, 113), torch.ops._quantized.add(r, r, .1, 113), 1)
    for rank in (1, 2):
        shape = (2, 3) + (7,) * rank
        x, r = pair(np.arange(np.prod(shape)).reshape(shape) % 255, device)
        for ceil_mode in (False, True):
            args = ([2] * rank, [2] * rank, [1] * rank, [2] * rank, ceil_mode)
            equal(resolve(tp.ops.quantized, f'max_pool{rank}d')(x, *args),
                  resolve(torch.ops.quantized, f'max_pool{rank}d')(r, *args))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind,gates', [('lstm', 4), ('gru', 3)])
def test_float_cell_parameter_factory_runs_native_sequences(device, kind, gates):
    rng = np.random.default_rng(989)
    arrays = [rng.normal(0, .15, size=shape).astype('float32')
              for shape in ((gates * 5, 7), (gates * 5, 5), (gates * 5,), (gates * 5,))]
    params = tp.ops.quantized.make_quantized_cell_params(*(tp.tensor(a, device=device) for a in arrays))
    reference_params = torch.ops.quantized.make_quantized_cell_params(*(torch.tensor(a) for a in arrays))
    data = rng.normal(0, .2, size=(4, 2, 7)).astype('float32')
    hidden = rng.normal(0, .2, size=(1, 2, 5)).astype('float32')
    cell = rng.normal(0, .2, size=(1, 2, 5)).astype('float32')
    h, rh = tp.tensor(hidden, device=device), torch.tensor(hidden)
    hx = [h, tp.tensor(cell, device=device)] if kind == 'lstm' else h
    rhx = [rh, torch.tensor(cell)] if kind == 'lstm' else rh
    actual = resolve(tp.ops.tp, 'quantized_' + kind + '.input')(
        tp.tensor(data, device=device), hx, [params], True, 1, 0., False, False, False)
    expected = resolve(torch.ops.aten, 'quantized_' + kind + '.input')(
        torch.tensor(data), rhx, [reference_params], True, 1, 0., False, False, False)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.cpu().numpy(), b.numpy(), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('channels_last', [False, True])
@pytest.mark.parametrize('affine', [False, True])
def test_public_normalization_packets_layout_and_affine(device, channels_last, affine):
    x, r = pair(np.arange(280).reshape(2, 4, 5, 7) % 255, device)
    if channels_last:
        x = x.contiguous(memory_format=tp.channels_last)
        r = r.contiguous(memory_format=torch.channels_last)
    for name in ('layer_norm', 'group_norm', 'instance_norm', 'batch_norm', 'batch_norm_relu'):
        parameters = [[5, 7]] if name == 'layer_norm' else ([2] if name == 'group_norm' else [])
        shape = (5, 7) if name == 'layer_norm' else (4,)
        weight = np.linspace(.7, 1.3, np.prod(shape)).reshape(shape).astype('float32') if affine else None
        bias = np.linspace(-.3, .2, np.prod(shape)).reshape(shape).astype('float32') if affine else None
        args = [tp.tensor(weight, device=device) if affine else None, tp.tensor(bias, device=device) if affine else None]
        refs = [torch.tensor(weight) if affine else None, torch.tensor(bias) if affine else None]
        if name.startswith('batch_norm'):
            args.extend([tp.tensor([.1, -.2, .3, -.4], device=device), tp.tensor([.7, .8, .9, 1.], device=device)])
            refs.extend([torch.tensor([.1, -.2, .3, -.4]), torch.tensor([.7, .8, .9, 1.])])
            if not affine:
                with pytest.raises(RuntimeError):
                    resolve(tp.ops.quantized, name)(x, *parameters, *args, 1e-5, .09, 117)
                with pytest.raises(RuntimeError):
                    resolve(torch.ops.quantized, name)(r, *parameters, *refs, 1e-5, .09, 117)
                continue
        equal(resolve(tp.ops.quantized, name)(x, *parameters, *args, 1e-5, .09, 117),
              resolve(torch.ops.quantized, name)(r, *parameters, *refs, 1e-5, .09, 117), 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', [('int8', -11), ('uint8', 117)])
@pytest.mark.parametrize('operation', ['add', 'add_relu', 'mul', 'mul_relu'])
@pytest.mark.parametrize('scalar', [-100., -.37, 0., .37, 100.])
@pytest.mark.parametrize('strided', [False, True])
def test_scalar_overloads_transfer_grid_and_preserve_out_storage(device, dtype, zp, operation, scalar, strided):
    codes = np.linspace(np.iinfo(dtype).min, np.iinfo(dtype).max, 102).reshape(3, 34)
    x, r = pair(codes, device, dtype, zero_point=zp)
    if strided:
        x, r = x[:, ::2], r[:, ::2]
    scalar_name = operation.split('_')[0] + '_scalar' + ('_relu' if operation.endswith('_relu') else '')
    for name, tensor_arg, reverse, output in (
        (operation + '.Scalar', False, False, False),
        (operation + '.Scalar2', False, True, False),
        (operation + '.Scalar_out', False, False, True),
        (scalar_name, False, False, False),
        (scalar_name + '.Tensor', True, False, False),
        (scalar_name + '_out', False, False, True),
        (scalar_name + '_out.Tensor', True, False, True),
    ):
        a_scalar = tp.tensor(scalar, device=device) if tensor_arg else scalar
        r_scalar = torch.tensor(scalar) if tensor_arg else scalar
        args = [a_scalar, x] if reverse else [x, a_scalar]
        ref_args = [r_scalar, r] if reverse else [r, r_scalar]
        if output:
            out, ref_out = pair(np.zeros((3, 34)), device, dtype, scale=.91, zero_point=0)
            if strided:
                out, ref_out = out[:, ::2], ref_out[:, ::2]
            address = out.data_ptr()
            args.append(out)
            ref_args.append(ref_out)
        actual = resolve(tp.ops.quantized, name)(*args)
        expected = resolve(torch.ops.quantized, name)(*ref_args)
        equal(actual, expected)
        if output:
            assert out.data_ptr() == address
            if operation == 'mul_relu' and scalar > 0:
                assert actual is not out and actual.data_ptr() != address
                assert expected.data_ptr() != ref_out.data_ptr()
            else:
                assert actual is out
                assert expected.data_ptr() == ref_out.data_ptr()
            equal(out, ref_out)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', [('int8', -11), ('uint8', 117)])
@pytest.mark.parametrize('operation', ['add', 'add_relu', 'mul', 'mul_relu'])
@pytest.mark.parametrize('strided', [False, True])
def test_binary_out_overloads_use_destination_grid(device, dtype, zp, operation, strided):
    x, r = pair(np.arange(102).reshape(3, 34), device, dtype, zero_point=zp)
    y, s = pair(np.arange(102, 204).reshape(3, 34), device, dtype, scale=.13, zero_point=zp)
    if strided:
        x, r, y, s = x[:, ::2], r[:, ::2], y[:, ::2], s[:, ::2]
    for name in (operation + '.out', operation + '_out'):
        out, ref_out = pair(np.zeros((3, 34)), device, dtype, scale=.27, zero_point=zp)
        if strided:
            out, ref_out = out[:, ::2], ref_out[:, ::2]
        address, strides = out.data_ptr(), out.stride()
        actual = resolve(tp.ops.quantized, name)(x, y, out)
        expected = resolve(torch.ops.quantized, name)(r, s, ref_out)
        assert actual is out
        assert out.data_ptr() == address and out.stride() == strides
        equal(actual, expected, 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('relu', [False, True])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_cat_public_packets_mixed_grids_empty_and_out(device, relu, strided, explicit):
    x, r = pair(np.arange(102).reshape(3, 34), device)
    y, s = pair(np.arange(102, 204).reshape(3, 34), device, scale=.13, zero_point=103)
    if strided:
        x, r, y, s = x[:, ::2], r[:, ::2], y[:, ::2], s[:, ::2]
    inputs, refs = [x, x[:0], y], [r, r[:0], s]
    name = 'cat_relu' if relu else 'cat'
    grid = (.11, 97) if explicit else (None, None)
    equal(resolve(tp.ops.quantized, name)(inputs, -2, *grid),
          resolve(torch.ops.quantized, name)(refs, -2, *grid), 1)
    out, ref_out = pair(np.zeros((6, 34)), device, scale=.19, zero_point=99)
    if strided:
        out, ref_out = out[:, ::2], ref_out[:, ::2]
    address, strides = out.data_ptr(), out.stride()
    actual = resolve(tp.ops.quantized, name + '_out')(inputs, 0, out)
    expected = resolve(torch.ops.quantized, name + '_out')(refs, 0, ref_out)
    assert actual is out
    assert out.data_ptr() == address and out.stride() == strides
    equal(actual, expected, 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bits', [2, 4, 8])
@pytest.mark.parametrize('provided_ranges', [False, True])
@pytest.mark.parametrize('optimized', [False, True])
def test_rowwise_public_packing_ranges_unpack_and_bags(device, bits, provided_ranges, optimized):
    rng = np.random.default_rng(901)
    values = rng.normal(size=(7, 20)).astype('float32')
    name = f'embedding_bag_{"byte" if bits == 8 else str(bits) + "bit"}'
    ap = [tp.tensor(values, device=device)]
    rp = [torch.tensor(values)]
    suffix = '_prepack'
    if provided_ranges:
        ranges = np.stack([values.min(1) - .1, values.max(1) + .1], axis=1)
        ap.append(tp.tensor(ranges, device=device))
        rp.append(torch.tensor(ranges))
        suffix += '_with_rowwise_min_max'
    if bits != 8:
        ap.extend([optimized, 64, .2])
        rp.extend([optimized, 64, .2])
    packed = resolve(tp.ops.quantized, name + suffix)(*ap)
    ref_packed = resolve(torch.ops.quantized, name + suffix)(*rp)
    np.testing.assert_array_equal(packed.cpu().numpy(), ref_packed.numpy())
    unpacked = resolve(tp.ops.quantized, name + '_unpack')(packed)
    ref_unpacked = resolve(torch.ops.quantized, name + '_unpack')(ref_packed)
    np.testing.assert_allclose(unpacked.cpu().numpy(), ref_unpacked.numpy(), rtol=1e-6, atol=1e-6)
    indices = np.array([0, 3, 2, 2, 6, 1], dtype='int64')
    offsets = np.array([0, 0, 3, 6], dtype='int64')
    weights = np.array([.3, -.5, 1., .7, .2, -.1], dtype='float32')
    args = [packed, tp.tensor(indices, device=device), tp.tensor(offsets, device=device),
            False, 0, False, tp.tensor(weights, device=device), None, True]
    refs = [ref_packed, torch.tensor(indices), torch.tensor(offsets),
            False, 0, False, torch.tensor(weights), None, True]
    result = resolve(tp.ops.quantized, name + '_rowwise_offsets')(*args)
    expected = resolve(torch.ops.quantized, name + '_rowwise_offsets')(*refs)
    np.testing.assert_allclose(result.cpu().numpy(), expected.numpy(), rtol=1e-6, atol=2e-6)
    if bits != 2:
        np.testing.assert_allclose(resolve(tp.ops.quantized, name)(*args).cpu().numpy(),
                                   expected.numpy(), rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bits', [4, 8])
def test_logical_embedding_public_pack_unpack_and_lookup(device, bits):
    rng = np.random.default_rng(915)
    values = rng.normal(size=(7, 20)).astype('float32')
    scales = np.linspace(.03, .2, 7, dtype='float32')
    zeros = np.linspace(1.1, 5.5, 7, dtype='float32')
    qdtype = tp.quint8 if bits == 8 else tp.quint4x2
    rdtype = torch.quint8 if bits == 8 else torch.quint4x2
    weight = tp.quantize_per_channel(tp.tensor(values, device=device), tp.tensor(scales, device=device),
                                   tp.tensor(zeros, device=device), 0, qdtype)
    rw = torch.quantize_per_channel(torch.tensor(values), torch.tensor(scales), torch.tensor(zeros), 0, rdtype)
    packed = tp.ops.quantized.embedding_bag_prepack(weight)
    rp = torch.ops.quantized.embedding_bag_prepack(rw)
    unpacked = tp.ops.quantized.embedding_bag_unpack(packed)
    expected = torch.ops.quantized.embedding_bag_unpack(rp)
    assert tuple(unpacked.shape) == tuple(expected.shape)
    np.testing.assert_array_equal(unpacked.int_repr().cpu().numpy(), expected.int_repr().numpy())
    np.testing.assert_array_equal(unpacked.q_per_channel_scales().cpu().numpy(), expected.q_per_channel_scales().numpy())
    np.testing.assert_array_equal(unpacked.q_per_channel_zero_points().cpu().numpy(), expected.q_per_channel_zero_points().numpy())
    indices = np.array([[6, 2, 2], [0, 4, 1]], dtype='int64')
    name = 'embedding_byte' if bits == 8 else 'embedding_4bit'
    actual = resolve(tp.ops.quantized, name)(packed, tp.tensor(indices, device=device))
    reference = resolve(torch.ops.quantized, name)(rp, torch.tensor(indices))
    np.testing.assert_allclose(actual.cpu().numpy(), reference.numpy(), rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bias_enabled', [False, True])
@pytest.mark.parametrize('strided', [False, True])
def test_half_linear_public_unpack_dynamic_fusion_and_private_alias(device, bias_enabled, strided):
    rng = np.random.default_rng(928)
    values = rng.normal(size=(3, 34)).astype('float32')
    weights = rng.normal(size=(9, 34)).astype('float32')
    x, rx = tp.tensor(values, device=device), torch.tensor(values)
    w, rw = tp.tensor(weights, device=device), torch.tensor(weights)
    if strided:
        values, weights = values[:, ::2], weights[:, ::2]
        x, rx, w, rw = x[:, ::2], rx[:, ::2], w[:, ::2], rw[:, ::2]
        assert not x.is_contiguous() and not w.is_contiguous()
    bias = rng.normal(size=9).astype('float32') if bias_enabled else None
    b = tp.tensor(bias, device=device) if bias_enabled else None
    rb = torch.tensor(bias) if bias_enabled else None
    for namespace in ('quantized', '_quantized'):
        packed = resolve(getattr(tp.ops, namespace), 'linear_prepack_fp16')(w, b)
        rp = resolve(getattr(torch.ops, namespace), 'linear_prepack_fp16')(rw, rb)
        aw, ab = tp.ops.quantized.linear_unpack_fp16(packed)
        ew, eb = torch.ops.quantized.linear_unpack_fp16(rp)
        np.testing.assert_array_equal(aw.cpu().numpy(), weights.astype('float16'))
        np.testing.assert_array_equal(aw.cpu().numpy(), ew.numpy())
        if bias_enabled:
            np.testing.assert_array_equal(ab.cpu().numpy(), eb.numpy())
        else:
            assert ab is None and eb is None
        for name in ('linear_dynamic_fp16', 'linear_relu_dynamic_fp16'):
            actual = resolve(tp.ops.quantized, name)(x, packed)
            expected = resolve(torch.ops.quantized, name)(rx, rp)
            np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=1e-6, atol=3e-6)
    actual = tp.ops.quantized.linear_dynamic_fp16_unpacked_weight(x, w, b)
    expected = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight(rx, rw, rb)
    np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=1e-6, atol=3e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('channel', [False, True])
@pytest.mark.parametrize('bias_enabled', [False, True])
def test_integer_linear_public_unpack_and_float_output_fusions(device, channel, bias_enabled):
    rng = np.random.default_rng(931)
    values = rng.normal(size=(3, 17)).astype('float32')
    codes = rng.integers(-35, 36, size=(9, 17), dtype='int8')
    if channel:
        scales = np.linspace(.003, .037, 9, dtype='float64')
        zeros = np.arange(-4, 5, dtype='int64')
        weight = tp._C._make_per_channel_quantized_tensor(tp.tensor(codes, device=device),
            tp.tensor(scales, device=device), tp.tensor(zeros, device=device), 0)
        rw = torch._make_per_channel_quantized_tensor(torch.tensor(codes), torch.tensor(scales), torch.tensor(zeros), 0)
    else:
        weight, rw = pair(codes, device, 'int8', .023, -3)
    bias = rng.normal(0, .1, size=9).astype('float32') if bias_enabled else None
    ab = tp.tensor(bias, device=device) if bias_enabled else None
    rb = torch.tensor(bias) if bias_enabled else None
    x, rx = tp.tensor(values, device=device), torch.tensor(values)
    for namespace in ('quantized', '_quantized'):
        packed = resolve(getattr(tp.ops, namespace), 'linear_prepack')(weight, ab)
        rp = resolve(getattr(torch.ops, namespace), 'linear_prepack')(rw, rb)
        aw, unpacked_bias = tp.ops.quantized.linear_unpack(packed)
        ew, eb = torch.ops.quantized.linear_unpack(rp)
        np.testing.assert_array_equal(aw.int_repr().cpu().numpy(), ew.int_repr().numpy())
        if channel:
            np.testing.assert_array_equal(aw.q_per_channel_scales().cpu().numpy(), ew.q_per_channel_scales().numpy())
            np.testing.assert_array_equal(aw.q_per_channel_zero_points().cpu().numpy(), ew.q_per_channel_zero_points().numpy())
        else:
            equal(aw, ew)
        if bias_enabled:
            np.testing.assert_array_equal(unpacked_bias.cpu().numpy(), eb.numpy())
        else:
            assert unpacked_bias is None and eb is None
        qx = tp.quantize_per_tensor(x, .03, 113, tp.quint8)
        qrx = torch.quantize_per_tensor(rx, .03, 113, torch.quint8)
        equal(resolve(getattr(tp.ops, namespace), 'linear')(qx, packed, .09, 117),
              resolve(getattr(torch.ops, namespace), 'linear')(qrx, rp, .09, 117), 1)
        for reduced in (False, True):
            actual = resolve(getattr(tp.ops, namespace), 'linear_dynamic')(x, packed, reduced)
            expected = resolve(getattr(torch.ops, namespace), 'linear_dynamic')(rx, rp, reduced)
            np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=1e-6, atol=2e-6)
        if namespace == 'quantized':
            equal(tp.ops.quantized.linear_relu(qx, packed, .09, 117),
                  torch.ops.quantized.linear_relu(qrx, rp, .09, 117), 1)
            actual = tp.ops.quantized.linear_relu_dynamic(x, packed, True)
            expected = torch.ops.quantized.linear_relu_dynamic(rx, rp, True)
            np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=1e-6, atol=2e-6)
        stable = resolve(getattr(tp.ops, namespace), 'linear')(qx, packed, .09, 117)
        if channel:
            aw.q_per_channel_zero_points().fill_(0)
            np.testing.assert_array_equal(weight.q_per_channel_zero_points().cpu().numpy(), zeros)
            replacement = tp._C._make_per_channel_quantized_tensor(
                tp.zeros(list(aw.shape), dtype=tp.int8, device=device),
                aw.q_per_channel_scales(), aw.q_per_channel_zero_points(), 0)
        else:
            replacement = tp._C._make_per_tensor_quantized_tensor(
                tp.zeros(list(aw.shape), dtype=tp.int8, device=device), .1, 0)
        aw.copy_(replacement)
        again = resolve(getattr(tp.ops, namespace), 'linear')(qx, packed, .09, 117)
        np.testing.assert_array_equal(again.int_repr().cpu().numpy(), stable.int_repr().cpu().numpy())
    for name in ('linear_with_input_q_dq_qweight_dq_output_fp32',
                 'linear_with_input_q_dq_qweight_dq_relu_output_fp32'):
        actual = resolve(tp.ops.quantized, name)(x, .03, 113, packed)
        expected = resolve(torch.ops.quantized, name)(rx, .03, 113, rp)
        np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('bias_enabled', [False, True])
def test_convolution_packing_execution_queries_and_aliases(device, rank, transposed, bias_enabled):
    rng = np.random.default_rng(511)
    prefix = ('conv_transpose' if transposed else 'conv') + f'{rank}d'
    shape = (4, 3) + (2,) * rank if transposed else (6, 2) + (2,) * rank
    weight, rw = pair(rng.integers(-29, 30, size=shape), device, 'int8', .02, -3)
    bias = rng.normal(0, .1, size=6).astype('float32') if bias_enabled else None
    ab = tp.tensor(bias, device=device) if bias_enabled else None
    rb = torch.tensor(bias) if bias_enabled else None
    parameters = [[2] * rank, [1] * rank]
    if transposed:
        parameters.append([1] * rank)
    parameters.extend([[2] * rank, 2])
    packed = resolve(tp.ops.quantized, prefix + '_prepack')(weight, ab, *parameters)
    rp = resolve(torch.ops.quantized, prefix + '_prepack')(rw, rb, *parameters)
    aw, unpacked_bias = resolve(tp.ops.quantized, prefix + '_unpack')(packed)
    ew, eb = resolve(torch.ops.quantized, prefix + '_unpack')(rp)
    equal(aw, ew)
    assert tuple(aw.stride()) == tuple(ew.stride())
    if bias_enabled:
        np.testing.assert_array_equal(unpacked_bias.cpu().numpy(), eb.numpy())
    else:
        assert unpacked_bias is None and eb is None
    if rank > 1:
        for field in ('stride', 'padding', 'output_padding', 'dilation', 'groups', 'transpose'):
            assert resolve(tp.ops.quantized, prefix + '_' + field)(packed) == resolve(torch.ops.quantized, prefix + '_' + field)(rp)
    values = rng.normal(size=(2, 4) + (5,) * rank).astype('float32')
    x = tp.quantize_per_tensor(tp.tensor(values, device=device), .05, 113, tp.quint8)
    r = torch.quantize_per_tensor(torch.tensor(values), .05, 113, torch.quint8)
    if not transposed and rank > 1:
        for suffix in ('', '_relu'):
            name = prefix + suffix
            equal(resolve(tp.ops.quantized, name + '.new')(x, packed, .07, 119),
                  resolve(torch.ops.quantized, name + '.new')(r, rp, .07, 119), 1)
            equal(resolve(tp.ops.quantized, name)(x, packed, *parameters, .07, 119),
                  resolve(torch.ops.quantized, name)(r, rp, *parameters, .07, 119), 1)
    else:
        equal(resolve(tp.ops.quantized, prefix)(x, packed, .07, 119),
              resolve(torch.ops.quantized, prefix)(r, rp, .07, 119), 1)
        if not transposed:
            equal(resolve(tp.ops.quantized, prefix + '_relu')(x, packed, .07, 119),
                  resolve(torch.ops.quantized, prefix + '_relu')(r, rp, .07, 119), 1)
    for reduced in (False, True):
        result = resolve(tp.ops.quantized, prefix + '_dynamic')(tp.tensor(values, device=device), packed, reduced)
        expected = resolve(torch.ops.quantized, prefix + '_dynamic')(torch.tensor(values), rp, reduced)
        grid = tp._C.quantize_per_tensor_dynamic(tp.tensor(values, device=device), tp.quint8, reduced)
        np.testing.assert_allclose(result.cpu().numpy(), expected.numpy(), rtol=0, atol=grid.q_scale() + 1e-6)
    if rank == 2 and not transposed:
        legacy = tp.ops.quantized.conv_prepack(weight, ab, *parameters)
        aw, unpacked_bias = tp.ops.quantized.conv_unpack(legacy)
        equal(aw, ew)
        sizes = tp.ops.quantized.conv2d_unpack_sizes(legacy)
        assert sizes == torch.ops.quantized.conv2d_unpack_sizes(rp)
    if transposed or rank > 1:
        alias = resolve(tp.ops._quantized, prefix + '_prepack')(weight, ab, *parameters)
        ref_alias = resolve(torch.ops._quantized, prefix + '_prepack')(rw, rb, *parameters)
        reference_operator = (torch.ops.quantized.conv_transpose3d if transposed else torch.ops.quantized.conv3d.new) if rank == 3 else resolve(torch.ops._quantized, prefix)
        equal(resolve(tp.ops._quantized, prefix)(x, alias, .07, 119),
              reference_operator(r, ref_alias, .07, 119), 1)
        if not transposed:
            reference_operator = torch.ops.quantized.conv3d_relu.new if rank == 3 else torch.ops._quantized.conv2d_relu
            equal(resolve(tp.ops._quantized, prefix + '_relu')(x, alias, .07, 119),
                  reference_operator(r, ref_alias, .07, 119), 1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('rank', [1, 2, 3])
def test_per_channel_convolution_unpack_rounding_and_independence(device, rank):
    rng = np.random.default_rng(935)
    codes = rng.integers(-37, 38, size=(6, 2) + (2,) * rank, dtype='int8')
    scales = np.linspace(.003, .037, 6, dtype='float64')
    zeros = np.arange(-3, 3, dtype='int64')
    weight = tp._C._make_per_channel_quantized_tensor(tp.tensor(codes, device=device),
        tp.tensor(scales, device=device), tp.tensor(zeros, device=device), 0)
    rw = torch._make_per_channel_quantized_tensor(torch.tensor(codes), torch.tensor(scales), torch.tensor(zeros), 0)
    prefix = f'conv{rank}d'
    args = (None, [2] * rank, [1] * rank, [2] * rank, 2)
    packed = resolve(tp.ops.quantized, prefix + '_prepack')(weight, *args)
    rp = resolve(torch.ops.quantized, prefix + '_prepack')(rw, *args)
    aw, ab = resolve(tp.ops.quantized, prefix + '_unpack')(packed)
    ew, eb = resolve(torch.ops.quantized, prefix + '_unpack')(rp)
    assert ab is None and eb is None
    assert tuple(aw.stride()) == tuple(ew.stride())
    np.testing.assert_array_equal(aw.int_repr().cpu().numpy(), ew.int_repr().numpy())
    np.testing.assert_array_equal(aw.q_per_channel_scales().cpu().numpy(), ew.q_per_channel_scales().numpy())
    np.testing.assert_array_equal(aw.q_per_channel_zero_points().cpu().numpy(), ew.q_per_channel_zero_points().numpy())
    values = rng.integers(0, 255, size=(2, 4) + (5,) * rank, dtype='uint8')
    x, r = pair(values, device)
    op = prefix if rank == 1 else prefix + '.new'
    equal(resolve(tp.ops.quantized, op)(x, packed, .09, 117),
          resolve(torch.ops.quantized, op)(r, rp, .09, 117), 1)
    aw.q_per_channel_zero_points().fill_(0)
    np.testing.assert_array_equal(weight.q_per_channel_zero_points().cpu().numpy(), zeros)
