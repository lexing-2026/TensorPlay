import io

import numpy as np
import pytest
import tensorplay as tp
import torch

DEVICES = ['cpu', 'cuda'] if tp.cuda.is_available() else ['cpu']


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['qint8', 'quint8', 'qint32'])
@pytest.mark.parametrize('channel', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_quantized_checkpoint_grid_codes_and_map_location(device, dtype, channel, legacy):
    x = tp.randn(3, 4, device=device)
    q = (tp._C.quantize_per_channel(x, tp.tensor([.1, .2, .3], device=device),
                                  tp.tensor([0, 1, 2], dtype=tp.int64, device=device), 0, getattr(tp, dtype))
         if channel else tp._C.quantize_per_tensor(x, .1, 2, getattr(tp, dtype)))
    if channel:
        q = tp._C._make_per_channel_quantized_tensor(q.int_repr().transpose(0, 1),
                                                    q.q_per_channel_scales(), q.q_per_channel_zero_points(), 1)
    else:
        q = q.transpose(0, 1)
    stream = io.BytesIO()
    tp.save({'weight': q, 'view': q}, stream, _use_new_zipfile_serialization=not legacy)
    stream.seek(0)
    restored = tp.load(stream, map_location='cpu')
    actual = restored['weight']
    assert actual is restored['view']
    assert actual.dtype == q.dtype and actual.qscheme() == q.qscheme()
    np.testing.assert_array_equal(actual.int_repr().numpy(), q.int_repr().cpu().numpy())
    if channel:
        assert actual.q_per_channel_axis() == 1
        np.testing.assert_array_equal(actual.q_per_channel_scales().numpy(), q.q_per_channel_scales().cpu().numpy())
        np.testing.assert_array_equal(actual.q_per_channel_zero_points().numpy(), q.q_per_channel_zero_points().cpu().numpy())
    else:
        assert actual.q_scale() == q.q_scale() and actual.q_zero_point() == q.q_zero_point()


@pytest.mark.parametrize('channel', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_external_quantized_checkpoint(channel, legacy):
    x = torch.randn(3, 4)
    q = (torch.quantize_per_channel(x, torch.tensor([.1, .2, .3]), torch.tensor([0, 1, 2]), 0, torch.qint8)
         if channel else torch.quantize_per_tensor(x, .1, 2, torch.quint8))
    stream = io.BytesIO()
    torch.save(q, stream, _use_new_zipfile_serialization=not legacy)
    stream.seek(0)
    actual = tp.load(stream)
    np.testing.assert_array_equal(actual.int_repr().numpy(), q.int_repr().numpy())
    np.testing.assert_allclose(actual.dequantize().numpy(), q.dequantize().numpy(), atol=1e-7)


@pytest.mark.parametrize('device', DEVICES)
def test_empty_quantized_checkpoint(device):
    q = tp._C.quantize_per_tensor(tp.empty(0, 3, device=device), .1, 2, tp.quint8)
    stream = io.BytesIO(); tp.save(q, stream); stream.seek(0)
    actual = tp.load(stream)
    assert actual.shape == (0, 3) and actual.q_scale() == .1


def test_invalid_channel_metadata():
    from tensorplay.serialization.quantized import rebuild_quantized_tensor
    codes = tp.ones(3, 4, dtype=tp.int8)
    for params in [('per_channel_affine', [.1], [0], 0),
                   ('per_channel_affine', [.1] * 3, [0] * 3, 4)]:
        with pytest.raises(ValueError):
            rebuild_quantized_tensor(codes, params)


@pytest.mark.parametrize('legacy', [False, True])
def test_float_channel_grid_checkpoint(legacy):
    q = torch.quantize_per_channel(torch.randn(3, 4), torch.tensor([.1, .2, .3]),
                                   torch.tensor([.25, -.5, 1.75]), 0, torch.quint8)
    stream = io.BytesIO()
    torch.save(q, stream, _use_new_zipfile_serialization=not legacy)
    stream.seek(0)
    actual = tp.load(stream)
    assert actual.qscheme() == tp.per_channel_affine_float_qparams
    np.testing.assert_array_equal(actual.int_repr().numpy(), q.int_repr().numpy())
    stream = io.BytesIO()
    tp.save(actual, stream, _use_new_zipfile_serialization=not legacy)
    stream.seek(0)
    restored = tp.load(stream)
    np.testing.assert_array_equal(restored.q_per_channel_zero_points().numpy(), q.q_per_channel_zero_points().numpy())
    np.testing.assert_array_equal(restored.dequantize().numpy(), actual.dequantize().numpy())


class _QuantizedGridState:
    restores = 0

    def __setstate__(self, state):
        type(self).restores += 1
        self.__dict__.update(state)
        self.scales = self.weight.q_per_channel_scales().tolist()


def test_legacy_object_state_observes_loaded_grid():
    obj = _QuantizedGridState()
    obj.weight = tp._C.quantize_per_channel(tp.ones(2, 3), tp.tensor([.125, .25]),
                                           tp.tensor([0, 0], dtype=tp.int64), 0, tp.qint8)
    stream = io.BytesIO()
    tp.save(obj, stream, _use_new_zipfile_serialization=False)
    stream.seek(0)
    _QuantizedGridState.restores = 0
    with tp.serialization.safe_globals([_QuantizedGridState]):
        restored = tp.load(stream)
    assert restored.scales == [.125, .25]
    assert _QuantizedGridState.restores == 1


@pytest.mark.parametrize('legacy', [False, True])
def test_quantized_tensor_dictionary_key_and_offset(legacy):
    q = tp._C.quantize_per_tensor(tp.tensor([1., 2., 3.]), .1, 0, tp.qint8)
    stream = io.BytesIO()
    tp.save({q: {'weight': q, 'view': q[1:]}}, stream, _use_new_zipfile_serialization=not legacy)
    stream.seek(0)
    restored = tp.load(stream)
    key = next(iter(restored))
    assert key is restored[key]['weight']
    view = restored[key]['view']
    assert view.storage_offset() == 1
    np.testing.assert_array_equal(view.int_repr().numpy(), q.int_repr().numpy()[1:])
