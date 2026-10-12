import numpy as np
import pytest
import tensorplay as tp
import torch
from tensorplay.ao.nn import qat, intrinsic
from tensorplay.ao.quantization import (
    prepare_qat, default_qat_qconfig, default_per_channel_qat_qconfig,
    fuse_modules, fuse_modules_qat,
)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

DEVICES = ['cpu', 'cuda'] if tp.cuda.is_available() else ['cpu']


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', ['Linear', 'Conv1d', 'Conv2d', 'Conv3d',
                                  'LinearReLU', 'ConvReLU1d', 'ConvReLU2d', 'ConvReLU3d',
                                  'ConvBn1d', 'ConvBn2d', 'ConvBn3d',
                                  'ConvBnReLU1d', 'ConvBnReLU2d', 'ConvBnReLU3d', 'LinearBn1d'])
def test_training_outputs_gradients_and_state(device, name):
    fused = name not in ('Linear', 'Conv1d', 'Conv2d', 'Conv3d')
    linear = name.startswith('Linear')
    rank = int(name[-2]) if not linear else 0
    args = (4, 3) if linear else (2, 3, 3)
    shape = (4, 4) if linear else (2, 2) + (4,) * rank
    config = default_per_channel_qat_qconfig
    actual = getattr(intrinsic.qat if fused else qat, name)(*args, qconfig=config).to(device)
    reference_config = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.default_fake_quant,
        weight=torch.ao.quantization.default_per_channel_weight_fake_quant)
    expected = getattr(torch.ao.nn.intrinsic.qat if fused else torch.ao.nn.qat, name)(*args, qconfig=reference_config).to(device)
    actual.load_state_dict({k: tp.tensor(v.cpu().tolist(), device=device, dtype=getattr(tp, str(v.dtype).split(".")[-1])) for k, v in expected.state_dict().items()})
    values = np.random.default_rng(42).normal(size=shape).astype('float32')
    for _ in range(2):
        x = tp.tensor(values, device=device, requires_grad=True)
        r = torch.tensor(values, device=device, requires_grad=True)
        a, b = actual(x), expected(r)
        a.square().sum().backward(); b.square().sum().backward()
        np.testing.assert_allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy(), atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(x.grad.cpu().numpy(), r.grad.cpu().numpy(), atol=4e-5, rtol=2e-4)
    if hasattr(actual, 'bn'):
        actual.freeze_bn_stats()
        actual.train()
        assert not actual.bn.training
        before = actual.bn.running_mean.clone()
        actual(tp.tensor(values, device=device))
        np.testing.assert_array_equal(before.cpu().numpy(), actual.bn.running_mean.cpu().numpy())
        actual.update_bn_stats()
        assert actual.bn.training
    actual.eval()
    restored = getattr(intrinsic.qat if fused else qat, name)(*args, qconfig=config).to(device)
    restored.load_state_dict(actual.state_dict())
    assert restored.weight_fake_quant.scale.shape == actual.weight_fake_quant.scale.shape
    result = actual.to_float()
    assert not result.training
    assert next(result.parameters()).device == actual.weight.device


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bn', [False, True])
def test_prepare_fused_model(device, bn):
    layers = [tp.nn.Conv2d(2, 3, 3)]
    if bn:
        layers.append(tp.nn.BatchNorm2d(3))
    layers.append(tp.nn.ReLU())
    model = tp.nn.Sequential(*layers).to(device)
    model.qconfig = default_qat_qconfig
    fused = fuse_modules_qat(model, [str(i) for i in range(len(layers))])
    prepared = prepare_qat(fused)
    assert isinstance(prepared[0], intrinsic.qat.ConvBnReLU2d if bn else intrinsic.qat.ConvReLU2d)
    assert not hasattr(prepared[1], 'activation_post_process')
    x = tp.randn(2, 2, 4, 4, device=device, requires_grad=True)
    prepared(x).square().sum().backward()
    assert x.grad is not None and prepared[0].weight.grad is not None
    assert model[0].weight.grad is None
    with pytest.raises(ValueError):
        prepare_qat(model.eval())


@pytest.mark.parametrize('device', DEVICES)
def test_eval_fusion_named_forward_and_nonaffine(device):
    class Model(tp.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = tp.nn.Conv2d(2, 3, 3)
            self.bn = tp.nn.BatchNorm2d(3, affine=False)
            self.relu = tp.nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
    model = Model().to(device).eval()
    x = tp.randn(2, 2, 5, 5, device=device)
    expected = model(x)
    fused = fuse_modules(model, ['conv', 'bn', 'relu'])
    np.testing.assert_allclose(fused(x).detach().cpu().numpy(), expected.detach().cpu().numpy(), atol=1e-6)
    assert isinstance(fused.bn, tp.nn.Identity)
