import numpy as np
import pytest
import tensorplay as tp
import torch


def equal(actual, expected, atol=3e-6):
    assert tuple(actual.shape) == tuple(expected.shape)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=atol, rtol=3e-5)


def weight_pair(values):
    actual = tp.ops.tp.fbgemm_linear_quantize_weight.default(tp.tensor(values))
    expected = torch.ops.aten.fbgemm_linear_quantize_weight.default(torch.tensor(values))
    equal(actual[0], expected[0], 0)
    equal(actual[1], expected[1], 0)
    assert actual[2] == expected[2]
    assert actual[3] == expected[3]
    return actual, expected


@pytest.mark.parametrize('values', [
    np.zeros((3, 7), dtype='float32'), np.full((3, 7), .125, dtype='float32'),
    np.full((3, 7), -.125, dtype='float32'),
    np.linspace(-.001, .001, 21, dtype='float32').reshape(3, 7),
    np.linspace(0, 1e-6, 21, dtype='float32').reshape(3, 7),
])
def test_integer_weight_grid_constant_and_small_ranges(values):
    weight_pair(values)


@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('explicit_sizes', [False, True])
def test_integer_linear_packets_weight_grid_offsets_and_input_layout(rank, strided, explicit_sizes):
    rng = np.random.default_rng(117)
    values = rng.normal(.04, .13, (11, 37)).astype('float32')
    q, rq = weight_pair(values)
    if explicit_sizes:
        packed = tp.ops.tp.fbgemm_pack_quantized_matrix.KN(q[0], 37, 11)
        rpacked = torch.ops.aten.fbgemm_pack_quantized_matrix.KN(rq[0], 37, 11)
    else:
        packed = tp.ops.tp.fbgemm_pack_quantized_matrix.default(q[0])
        rpacked = torch.ops.aten.fbgemm_pack_quantized_matrix.default(rq[0])
    shape = (3, 37 * (2 if strided else 1)) if rank == 2 else (2, 3, 37 * (2 if strided else 1))
    data = rng.normal(0, .2, shape).astype('float32')
    x, r = tp.tensor(data), torch.tensor(data)
    if strided:
        x, r = x[..., ::2], r[..., ::2]
    bias = rng.normal(0, .1, 11).astype('float32')
    for name in ['fbgemm_linear_int8_weight_fp32_activation', 'fbgemm_linear_int8_weight']:
        actual = getattr(tp.ops.tp, name).default(x, q[0], packed, q[1], q[2], q[3], tp.tensor(bias))
        expected = getattr(torch.ops.aten, name).default(r, rq[0], rpacked, rq[1], rq[2], rq[3], torch.tensor(bias))
        equal(actual, expected)


@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('with_bias', [False, True])
def test_half_linear_packets_and_out_resize(rank, strided, with_bias):
    rng = np.random.default_rng(131)
    weight = rng.normal(0, .2, (11, 37)).astype('float32')
    packed = tp.ops.tp.fbgemm_pack_gemm_matrix_fp16.default(tp.tensor(weight))
    rpacked = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16.default(torch.tensor(weight))
    shape = (3, 37 * (2 if strided else 1)) if rank == 2 else (2, 3, 37 * (2 if strided else 1))
    values = rng.normal(0, .2, shape).astype('float32')
    x, r = tp.tensor(values), torch.tensor(values)
    if strided:
        x, r = x[..., ::2], r[..., ::2]
    bias = rng.normal(0, .1, 11).astype('float32')
    b, rb = (tp.tensor(bias), torch.tensor(bias)) if with_bias else (None, None)
    names = ['fbgemm_linear_fp16_weight_fp32_activation']
    if with_bias:
        names.append('fbgemm_linear_fp16_weight')
    for name in names:
        equal(getattr(tp.ops.tp, name).default(x, packed, b),
              getattr(torch.ops.aten, name).default(r, rpacked, rb))
        out, rout = tp.empty(0), torch.empty(0)
        returned = getattr(tp.ops.tp, name).out(x, packed, b, out)
        reference_returned = getattr(torch.ops.aten, name).out(r, rpacked, rb, rout)
        assert returned.data_ptr() == out.data_ptr()
        assert reference_returned.data_ptr() == rout.data_ptr()
        equal(out, rout)


@pytest.mark.parametrize('kind,gates', [('rnn_relu', 1), ('rnn_tanh', 1), ('lstm', 4), ('gru', 3)])
@pytest.mark.parametrize('batch', [1, 3])
def test_legacy_cell_packets_use_integer_projections(kind, gates, batch):
    rng = np.random.default_rng(177)
    hidden, features = 5, 7
    wi, rwi = weight_pair(rng.normal(0, .1, (gates * hidden, features)).astype('float32'))
    wh, rwh = weight_pair(rng.normal(0, .1, (gates * hidden, hidden)).astype('float32'))
    pi = tp.ops.tp.fbgemm_pack_quantized_matrix.default(wi[0])
    ph = tp.ops.tp.fbgemm_pack_quantized_matrix.default(wh[0])
    rpi = torch.ops.aten.fbgemm_pack_quantized_matrix.default(rwi[0])
    rph = torch.ops.aten.fbgemm_pack_quantized_matrix.default(rwh[0])
    bi = rng.normal(0, .1, gates * hidden).astype('float32')
    bh = rng.normal(0, .1, gates * hidden).astype('float32')
    data = rng.normal(0, .2, (batch, features)).astype('float32')
    state = rng.normal(0, .2, (batch, hidden)).astype('float32')
    cell = rng.normal(0, .2, (batch, hidden)).astype('float32')
    hx, rhx = tp.tensor(state), torch.tensor(state)
    if kind == 'lstm':
        hx, rhx = [hx, tp.tensor(cell)], [rhx, torch.tensor(cell)]
    actual = getattr(tp.ops.tp, 'quantized_' + kind + '_cell').default(
        tp.tensor(data), hx, wi[0], wh[0], tp.tensor(bi), tp.tensor(bh),
        pi, ph, wi[1], wh[1], wi[2], wh[2], wi[3], wh[3])
    expected = getattr(torch.ops.aten, 'quantized_' + kind + '_cell').default(
        torch.tensor(data), rhx, rwi[0], rwh[0], torch.tensor(bi), torch.tensor(bh),
        rpi, rph, rwi[1], rwh[1], rwi[2], rwh[2], rwi[3], rwh[3])
    if kind == 'lstm':
        for a, b in zip(actual, expected):
            equal(a, b)
    else:
        equal(actual, expected)


@pytest.mark.parametrize('name', ['linear_prepack_legacy', 'linear_prepack_fp16_legacy'])
def test_private_legacy_prepack_is_explicitly_rejected(name):
    for namespace, value in [(tp.ops._quantized, tp.ones(3, 7)),
                              (torch.ops._quantized, torch.ones(3, 7))]:
        with pytest.raises(RuntimeError):
            getattr(namespace, name)(value)
