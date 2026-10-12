import io

import numpy as np
import pytest
import tensorplay as tp
import torch
from tensorplay.ao.quantization import observer as obs
from tensorplay.ao.quantization import fake_quantize as fq

DEVICES = ['cpu', 'cuda'] if tp.cuda.is_available() else ['cpu']


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', ['MinMaxObserver', 'MovingAverageMinMaxObserver',
                                  'PerChannelMinMaxObserver', 'MovingAveragePerChannelMinMaxObserver'])
@pytest.mark.parametrize('scheme', ['affine', 'symmetric', 'float_qparams'])
def test_statistics_and_qparams(device, name, scheme):
    channel = 'PerChannel' in name
    if scheme == 'float_qparams' and not channel:
        pytest.skip('Floating zero points require channel statistics')
    qscheme = ('per_channel_' if channel else 'per_tensor_') + ('affine_float_qparams' if scheme == 'float_qparams' else scheme)
    kwargs = {'quant_min': -64, 'quant_max': 63}
    actual = getattr(obs, name)(dtype=tp.qint8, qscheme=getattr(tp, qscheme), **kwargs).to(device)
    expected = getattr(torch.ao.quantization, name)(dtype=torch.qint8, qscheme=getattr(torch, qscheme), **kwargs).to(device)
    rng = np.random.default_rng(7)
    for shift in [0., 2., -3.]:
        values = rng.normal(shift, 2., (4, 5, 3)).astype('float32')
        x = tp.tensor(values, device=device)
        assert actual(x) is x
        expected(torch.tensor(values, device=device))
        for key in ['min_val', 'max_val']:
            np.testing.assert_allclose(getattr(actual, key).cpu().numpy(), getattr(expected, key).cpu().numpy(), rtol=1e-6)
        for a, b in zip(actual.calculate_qparams(), expected.calculate_qparams()):
            np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), rtol=1e-6, atol=1e-6)
    fresh = getattr(obs, name)(dtype=tp.qint8, qscheme=getattr(tp, qscheme), **kwargs).to(device)
    stream = io.BytesIO()
    tp.save(actual.state_dict(), stream)
    stream.seek(0)
    fresh.load_state_dict(tp.load(stream))
    for a, b in zip(actual.calculate_qparams(), fresh.calculate_qparams()):
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())
    fresh.reset()
    with pytest.warns(UserWarning):
        scale, zp = fresh.calculate_qparams()
    assert float(scale) == 1. and int(zp) == 0


@pytest.mark.parametrize('device', DEVICES)
def test_histogram_rebin_search_and_state(device):
    actual = obs.HistogramObserver(bins=64).to(device)
    expected = torch.ao.quantization.HistogramObserver(bins=64).to(device)
    rng = np.random.default_rng(9)
    for values in [np.ones(16, dtype='float32'), rng.normal(0, 1, 500).astype('float32'), rng.normal(1, 3, 300).astype('float32')]:
        actual(tp.tensor(values, device=device))
        expected(torch.tensor(values, device=device))
        np.testing.assert_allclose(actual.histogram.cpu().numpy(), expected.histogram.cpu().numpy(), atol=1e-4)
    for a, b in zip(actual.calculate_qparams(), expected.calculate_qparams()):
        np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), rtol=1e-6)
    fresh = obs.HistogramObserver(bins=64).to(device)
    fresh.load_state_dict(actual.state_dict())
    assert float(fresh.histogram.sum()) == pytest.approx(816., abs=1e-3)
    x = tp.tensor([-float('inf'), 0., float('inf')], device=device)
    with pytest.warns(UserWarning):
        assert fresh(x) is x


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('channel', [False, True])
def test_rounding_mask_and_flags(device, channel):
    values = np.array([[-2.51, -2.49, -.5, .5, 2.49, 2.51]], dtype='float32')
    x = tp.tensor(values, device=device, requires_grad=True)
    ref = torch.tensor(values, device=device, requires_grad=True)
    if channel:
        y = fq.fake_quantize_per_channel(x, tp.tensor([1.], device=device), tp.tensor([0], device=device), 0, -2, 2)
        z = torch.fake_quantize_per_channel_affine(ref, torch.tensor([1.], device=device), torch.tensor([0], dtype=torch.int32, device=device), 0, -2, 2)
    else:
        y = fq.fake_quantize_per_tensor(x, 1., 0, -2, 2)
        z = torch.fake_quantize_per_tensor_affine(ref, 1., 0, -2, 2)
    y.sum().backward()
    z.sum().backward()
    np.testing.assert_array_equal(y.detach().cpu().numpy(), z.detach().cpu().numpy())
    np.testing.assert_array_equal(x.grad.cpu().numpy(), ref.grad.cpu().numpy())
    module = fq.default_per_channel_weight_fake_quant().to(device)
    module(x.detach())
    module.disable_observer()
    old = module.scale.clone()
    module(x.detach() * 100)
    np.testing.assert_array_equal(old.cpu().numpy(), module.scale.cpu().numpy())
    module.disable_fake_quant()
    assert module(x) is x
    restored = fq.default_per_channel_weight_fake_quant().to(device)
    restored.load_state_dict(module.state_dict())
    assert not bool(restored.observer_enabled[0])
    assert restored.scale.shape == module.scale.shape


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('channel', [False, True])
def test_fused_moving_average(device, channel):
    kwargs = {'observer': obs.MovingAveragePerChannelMinMaxObserver, 'ch_axis': 0, 'qscheme': tp.per_channel_symmetric, 'dtype': tp.qint8, 'quant_min': -128, 'quant_max': 127} if channel else {}
    actual = fq.FusedMovingAvgObsFakeQuantize(**kwargs).to(device)
    reference_kwargs = {k: (getattr(torch.ao.quantization, v.__name__) if k == 'observer' else getattr(torch, str(v).split('.')[-1]) if k in ('dtype', 'qscheme') else v) for k, v in kwargs.items()}
    expected = torch.ao.quantization.FusedMovingAvgObsFakeQuantize(**reference_kwargs).to(device)
    for values in [np.array([[-1., .2, 2.], [-4., 1., 3.]], dtype='float32'), np.array([[-3., .1, 4.], [-6., .5, 2.]], dtype='float32')]:
        x = tp.tensor(values, device=device, requires_grad=True)
        r = torch.tensor(values, device=device, requires_grad=True)
        a, b = actual(x), expected(r)
        a.sum().backward(); b.sum().backward()
        np.testing.assert_allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy(), atol=1e-6)
        np.testing.assert_array_equal(x.grad.cpu().numpy(), r.grad.cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16', 'bfloat16'])
def test_tensor_mask_precision_and_infinities(device, dtype):
    values = np.array([-float('inf'), -2.55, -1.25, -.15, .15, 1.25, 2.55, float('inf')], dtype='float32')
    x = tp.tensor(values, device=device, dtype=getattr(tp, dtype), requires_grad=True)
    r = torch.tensor(values, device=device, dtype=getattr(torch, dtype), requires_grad=True)
    a = fq.fake_quantize_per_tensor(x, .1, 0, -25, 25)
    b = torch.fake_quantize_per_tensor_affine(r, .1, 0, -25, 25)
    a.sum().backward(); b.sum().backward()
    np.testing.assert_array_equal(a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy())
    np.testing.assert_array_equal(x.grad.float().cpu().numpy(), r.grad.float().cpu().numpy())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('channel', [False, True])
def test_learnable_grid_gradients(device, channel):
    from tensorplay.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
    from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize as Ref
    kwargs = {'quant_min': -8, 'quant_max': 7, 'scale': .5, 'zero_point': 0., 'use_grad_scaling': True}
    if channel:
        kwargs['channel_len'] = 2
    actual = _LearnableFakeQuantize(obs.PerChannelMinMaxObserver if channel else obs.MinMaxObserver, dtype=tp.qint8, **kwargs).to(device)
    expected = Ref(torch.ao.quantization.PerChannelMinMaxObserver if channel else torch.ao.quantization.MinMaxObserver, dtype=torch.qint8, **kwargs).to(device)
    actual.enable_param_learning(); expected.enable_param_learning()
    x = tp.tensor([[-6., -.7, .8], [-1.3, 2., 8.]], device=device, requires_grad=True)
    r = torch.tensor([[-6., -.7, .8], [-1.3, 2., 8.]], device=device, requires_grad=True)
    a, b = actual(x), expected(r)
    a.sum().backward(); b.sum().backward()
    for lhs, rhs in [(a.detach(), b.detach()), (x.grad, r.grad), (actual.scale.grad, expected.scale.grad), (actual.zero_point.grad, expected.zero_point.grad)]:
        np.testing.assert_allclose(lhs.cpu().numpy(), rhs.cpu().numpy(), atol=1e-6)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['int16', 'uint16', 'qint32'])
def test_wide_grid_fake_quant(device, dtype):
    actual = fq.FakeQuantize(dtype=getattr(tp, dtype)).to(device)
    expected = torch.ao.quantization.FakeQuantize(dtype=getattr(torch, dtype)).to(device)
    values = np.array([-20., -.13, 0., .17, 50.], dtype='float32')
    a = actual(tp.tensor(values, device=device))
    b = expected(torch.tensor(values, device=device))
    np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), atol=1e-6)


@pytest.mark.parametrize('backend', ['x86', 'fbgemm', 'qnnpack', 'onednn'])
@pytest.mark.parametrize('version', [0, 1])
def test_training_configuration_presets(backend, version):
    from tensorplay.ao.quantization import get_default_qat_qconfig
    actual = get_default_qat_qconfig(backend, version)
    expected = torch.ao.quantization.get_default_qat_qconfig(backend, version)
    for factory, ref_factory in [(actual.activation, expected.activation), (actual.weight, expected.weight)]:
        module, ref = factory(), ref_factory()
        assert type(module).__name__ == type(ref).__name__
        assert module.quant_min == ref.quant_min and module.quant_max == ref.quant_max
        assert str(module.qscheme).split('.')[-1] == str(ref.qscheme).split('.')[-1]
