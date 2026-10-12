"""Quantized normalization module conversion and state coverage."""

import copy

import pytest

import tensorplay as tp
from tensorplay.ao.nn import quantized as nnq
from tensorplay.ao.quantization import (
    DeQuantStub, QuantStub, convert, default_qconfig, prepare,
)

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


MODULES = [(tp.nn.LayerNorm, nnq.LayerNorm, ((4, 3),), (2, 4, 3)),
           (tp.nn.GroupNorm, nnq.GroupNorm, (2, 4), (2, 4, 3)),
           (tp.nn.InstanceNorm1d, nnq.InstanceNorm1d, (4,), (2, 4, 3)),
           (tp.nn.InstanceNorm2d, nnq.InstanceNorm2d, (4,), (2, 4, 3, 5)),
           (tp.nn.InstanceNorm3d, nnq.InstanceNorm3d, (4,), (2, 4, 2, 3, 5))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("float_cls,quant_cls,args,shape", MODULES)
@pytest.mark.parametrize("affine", [False, True])
def test_eager_conversion_and_state(device, float_cls, quant_cls, args, shape, affine):
    kwargs = {"elementwise_affine" if float_cls is tp.nn.LayerNorm else "affine": affine}
    model = float_cls(*args, **kwargs).eval().to(device)
    model.qconfig = default_qconfig
    prepared = prepare(model, inplace=False)
    prepared(tp.randn(shape).to(device))
    converted = convert(prepared, inplace=False).to(device)
    assert isinstance(converted, quant_cls)
    assert converted.scale.device == tp.device(device)
    assert "scale" in converted.state_dict()
    assert "zero_point" in converted.state_dict()
    assert not hasattr(converted, "activation_post_process")
    qx = tp.quantize_per_tensor(tp.randn(shape).to(device), 0.1, 0, tp.qint8)
    y = converted(qx)
    assert y.is_quantized()
    assert y.q_scale() == float(converted.scale)
    restored = copy.deepcopy(converted)
    restored.scale.fill_(1.0)
    restored.zero_point.fill_(0)
    restored.load_state_dict(converted.state_dict())
    assert restored(qx).int_repr().cpu().tolist() == y.int_repr().cpu().tolist()
    if affine:
        assert converted.weight.shape == prepared.weight.shape
    assert isinstance(prepared, float_cls)


def test_instance_conversion_preserves_configuration():
    model = tp.nn.InstanceNorm2d(4, eps=0.003, momentum=0.25, affine=True,
                                track_running_stats=True).eval()
    model.qconfig = default_qconfig
    prepared = prepare(model)
    prepared(tp.randn(2, 4, 3, 5))
    converted = convert(prepared)
    assert converted.affine is True
    assert converted.track_running_stats is True
    assert converted.momentum == 0.25
    assert converted.eps == 0.003
    assert converted.running_mean is not None
    assert converted.running_var is not None


def test_module_export_paths():
    from tensorplay.ao.nn.quantized import modules
    from tensorplay.ao.nn.quantized.modules import normalization
    for name in normalization.__all__:
        assert getattr(nnq, name) is getattr(modules, name)
        assert getattr(nnq, name) is getattr(normalization, name)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("float_cls,quant_cls,args,shape", MODULES)
def test_eager_boundaries(device, float_cls, quant_cls, args, shape):
    model = tp.nn.Sequential(QuantStub(), float_cls(*args), DeQuantStub()).eval().to(device)
    model.qconfig = default_qconfig
    prepared = prepare(model)
    x = tp.randn(shape).to(device)
    prepared(x)
    converted = convert(prepared).to(device)
    assert isinstance(converted[1], quant_cls)
    assert converted[0].dtype == tp.quint8
    qx = converted[0](x)
    assert qx.is_quantized()
    expected = converted[1](qx).dequantize()
    actual = converted(x)
    assert actual.dtype == tp.float32
    assert actual.cpu().tolist() == expected.cpu().tolist()
    assert default_qconfig.weight().dtype == tp.qint8
