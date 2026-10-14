"""Tests for the completed AO API surface (eager, graph, sparse, pruning, ns)."""

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.ao import quantization as q
from tensorplay.ao import nn as ao_nn
from tensorplay.ao.nn import intrinsic as nni
from tensorplay.ao.nn import quantized as nnq
from tensorplay.ao.nn.quantized import reference as nnqr


# ---------------------------------------------------------------------------
# quantization utilities / qconfig
# ---------------------------------------------------------------------------


def test_quant_type_and_utils():
    from tensorplay.ao.quantization.quant_type import QuantType

    assert QuantType.STATIC == 1
    from tensorplay.ao.quantization.utils import (
        get_quant_type,
        determine_qparams,
        get_combined_dict,
    )

    assert get_quant_type(q.default_qconfig) == QuantType.STATIC
    scale, zp = determine_qparams(
        tp.tensor([-1.0]), tp.tensor([1.0]), 0, 255, tp.quint8, tp.tensor(1e-6), False
    )
    assert scale.numel() == 1 and zp.numel() == 1
    assert get_combined_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_qconfig_equals_and_default_dict():
    from tensorplay.ao.quantization.qconfig import qconfig_equals

    assert qconfig_equals(q.default_qconfig, q.default_qconfig)
    d = q.get_default_qconfig_dict()
    assert "" in d
    d2 = q.get_default_qat_qconfig_dict()
    assert "" in d2


def test_quant_wrapper():
    from tensorplay.ao.quantization.stubs import QuantWrapper

    wrapper = QuantWrapper(nn.Linear(4, 4))
    assert [name for name, _ in wrapper.named_children()] == ["quant", "dequant", "module"]
    out = wrapper(tp.rand(2, 4))
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# fusion registry
# ---------------------------------------------------------------------------


def test_fuser_method_mappings():
    from tensorplay.ao.quantization.fuser_method_mappings import (
        fuse_conv_bn,
        fuse_linear_bn,
        get_fuser_method,
    )

    conv = nn.Conv2d(3, 4, 3)
    bn = nn.BatchNorm2d(4)
    conv.eval()
    bn.eval()
    fused = fuse_conv_bn(False, conv, bn)
    x = tp.rand(1, 3, 8, 8)
    with tp.no_grad():
        assert tp.allclose(fused(x), bn(conv(x)), atol=1e-4)
    linear = nn.Linear(4, 4)
    bn1 = nn.BatchNorm1d(4)
    linear.eval()
    bn1.eval()
    fused_linear = fuse_linear_bn(False, linear, bn1)
    assert fused_linear is not None
    method = get_fuser_method((nn.Conv2d, nn.BatchNorm2d))
    assert method is not None


# ---------------------------------------------------------------------------
# reference modules and from_reference
# ---------------------------------------------------------------------------


def test_reference_modules_and_from_reference():
    ref = nnqr.Linear(
        4, 3,
        weight_qparams={"qscheme": tp.per_tensor_affine, "dtype": tp.qint8, "scale": 0.1, "zero_point": 0},
    )
    qlinear = nnq.Linear.from_reference(ref, 0.05, 0)
    assert qlinear.qweight.dtype == tp.qint8

    refc = nnqr.Conv2d(
        2, 4, 3,
        weight_qparams={"qscheme": tp.per_channel_affine, "dtype": tp.qint8,
                        "scale": tp.ones(4) * 0.1, "zero_point": tp.zeros(4, dtype=tp.int64), "axis": 0},
    )
    qc = nnq.Conv2d.from_reference(refc, 0.05, 0)
    assert qc(tp.quantize_per_tensor(tp.rand(1, 2, 5, 5), 0.1, 0, tp.qint8)).shape == (1, 4, 3, 3)

    refe = nnqr.Embedding(
        10, 4,
        weight_qparams={"qscheme": tp.per_channel_affine_float_qparams, "dtype": tp.quint8,
                        "scale": tp.ones(10) * 0.1, "zero_point": tp.zeros(10), "axis": 0},
    )
    qe = nnq.Embedding.from_reference(refe)
    assert qe(tp.tensor([1, 2, 3])).shape == (3, 4)

    ref_lstm = nnqr.LSTM(4, 6, num_layers=1, batch_first=True)
    out, (h, c) = ref_lstm(tp.rand(2, 5, 4))
    assert out.shape == (2, 5, 6)


# ---------------------------------------------------------------------------
# sparse modules
# ---------------------------------------------------------------------------


def test_sparse_quantized_linear():
    from tensorplay.ao.nn.sparse.quantized import Linear, LinearPackedParams, dynamic

    m = Linear(4, 3, 1, 4)
    x = tp.quantize_per_tensor(tp.rand(2, 4), 0.1, 0, tp.qint8)
    assert m(x).shape == (2, 3)
    d = dynamic.Linear(4, 3, 1, 4)
    assert d(tp.rand(2, 4)).shape == (2, 3)
    assert LinearPackedParams(1, 4) is not None


# ---------------------------------------------------------------------------
# intrinsic fused modules
# ---------------------------------------------------------------------------


def test_intrinsic_fused_modules():
    f = nni.LinearLeakyReLU(nn.Linear(4, 4), nn.LeakyReLU(0.1))
    assert f(tp.rand(2, 4)).shape == (2, 4)
    f2 = nni.LinearTanh(nn.Linear(4, 4), nn.Tanh())
    assert f2(tp.rand(2, 4)).shape == (2, 4)
    c = nni.ConvAdd2d(nn.Conv2d(2, 2, 3), tp.add)
    assert c(tp.rand(1, 2, 5, 5), tp.rand(1, 2, 3, 3)).shape == (1, 2, 3, 3)

    from tensorplay.ao.nn.intrinsic import quantized as nniq

    qll = nniq.LinearLeakyReLU(4, 4, 0.1)
    assert qll._get_name() == "QuantizedLinearLeakyReLU"
    assert nniq.ConvAdd2d is not None and nniq.ConvAddReLU2d is not None


# ---------------------------------------------------------------------------
# quantized modules: dropout / softmax / prelu / rnn / mha / functional
# ---------------------------------------------------------------------------


def test_quantized_new_modules():
    x = tp.quantize_per_tensor(tp.rand(2, 4), 0.1, 0, tp.qint8)
    d = nnq.Dropout(0.5)
    assert tp.equal(d(x), x)
    s = nnq.Softmax(dim=1, scale=0.01, zero_point=0)
    assert s(x).dtype == tp.qint8
    p = nnq.PReLU(0.1, 0, num_parameters=4)
    assert p._get_name() == "QuantizedPReLU"
    lstm = nnq.LSTM(4, 6, num_layers=2, batch_first=True)
    out, (h, c) = lstm(x.unsqueeze(1).expand(2, 5, 4))
    assert out.dtype == tp.qint8

    from tensorplay.ao.nn.quantized import functional as F

    w = tp.quantize_per_channel(tp.rand(3, 4), tp.ones(3) * 0.1, tp.zeros(3), 0, tp.qint8)
    assert F.linear(x, w, None, 0.05, 0).shape == (2, 3)
    xc = tp.quantize_per_tensor(tp.rand(1, 2, 5, 5), 0.1, 0, tp.qint8)
    wc = tp.quantize_per_tensor(tp.rand(3, 2, 3, 3), 0.1, 0, tp.qint8)
    assert F.conv2d(xc, wc, None, 0.05, 0).shape == (1, 3, 3, 3)
    assert F.max_pool2d(xc, 2).shape == (1, 2, 2, 2)
    assert F.hardtanh(xc, 0.1, 0, -1, 1).shape == (1, 2, 5, 5)


def test_quantizable_modules():
    from tensorplay.ao.nn import quantizable as nnqa

    lstm = nn.LSTM(4, 6, num_layers=1, batch_first=True)
    qa = nnqa.LSTM.from_float(lstm)
    assert qa is not None
    mha = nn.MultiheadAttention(8, 2, batch_first=True)
    mha.qconfig = q.QConfig(
        tp.nn.Identity, tp.nn.Identity
    )
    qmha = nnqa.MultiheadAttention.from_float(mha)
    assert qmha is not None


# ---------------------------------------------------------------------------
# backend config
# ---------------------------------------------------------------------------


def test_backend_config():
    from tensorplay.ao.quantization.backend_config import (
        BackendConfig,
        BackendPatternConfig,
        DTypeConfig,
        ObservationType,
        get_native_backend_config,
    )

    bc = get_native_backend_config()
    assert bc.name == "native"
    assert len(bc.configs) > 0
    dc = DTypeConfig(input_dtype=tp.quint8, output_dtype=tp.quint8, weight_dtype=tp.qint8)
    cfg = BackendPatternConfig(nn.Linear).set_observation_type(
        ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    ).set_dtype_configs([dc])
    bc2 = BackendConfig("test").set_backend_pattern_config(cfg)
    assert bc2.name == "test"
    d = get_native_backend_config().to_dict()
    assert "configs" in d


# ---------------------------------------------------------------------------
# graph-mode quantization (fx)
# ---------------------------------------------------------------------------


class _TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def test_graph_mode_quantization():
    from tensorplay.ao.quantization.quantize_fx import convert_fx, prepare_fx

    model = _TwoLayer().eval()
    act_obs = q.MinMaxObserver.with_args(dtype=tp.qint8, quant_min=-128, quant_max=127)
    qconfig = q.QConfig(act_obs, q.default_per_channel_weight_observer)
    qconfig_mapping = q.QConfigMapping().set_object_type(nn.Linear, qconfig)
    gm = prepare_fx(model, qconfig_mapping, (tp.rand(2, 4),))
    assert hasattr(gm.linear, "activation_post_process")
    gm(tp.rand(2, 4))
    cgm = convert_fx(gm)
    assert isinstance(cgm.linear, nnq.QuantizedLinear)
    xq = tp.quantize_per_tensor(tp.rand(2, 4), 0.01, 0, tp.qint8)
    out = cgm(xq)
    assert out.dtype == tp.qint8


def test_graph_mode_fusion():
    from tensorplay.ao.quantization.quantize_fx import fuse_fx

    class _ConvBnRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 3)
            self.bn = nn.BatchNorm2d(2)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    model = _ConvBnRelu().eval()
    x = tp.rand(1, 2, 5, 5)
    with tp.no_grad():
        ref = model(x)
        gm = fuse_fx(model)
        got = gm(x)
    assert tp.allclose(got, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# pruning sparsifier / scheduler
# ---------------------------------------------------------------------------


def test_sparsifier_and_scheduler():
    from tensorplay.ao.pruning import (
        BaseScheduler,
        CubicSL,
        LambdaSL,
        WeightNormSparsifier,
    )

    model = nn.Sequential(nn.Linear(8, 8))
    sparsifier = WeightNormSparsifier(sparsity_level=0.5, sparse_block_shape=(1, 4))
    sparsifier.prepare(model, [{"tensor_fqn": "0.weight"}])
    sparsifier.step()
    assert (model[0].weight == 0).float().mean().item() == pytest.approx(0.5, abs=0.01)
    scheduler = LambdaSL(sparsifier, lambda epoch: 1.0)
    assert scheduler.get_last_sl() is not None
    cubic = CubicSL(sparsifier, init_sl=0.0, total_t=10)
    assert cubic is not None


# ---------------------------------------------------------------------------
# experimental quantization
# ---------------------------------------------------------------------------


def test_experimental_apot():
    from tensorplay.ao.quantization.experimental import (
        APoTObserver,
        LinearAPoT,
        dequantize_APoT,
        quantize_APoT,
    )

    obs = APoTObserver(b=8, k=2)
    x = tp.tensor([0.1, -0.2, 0.3, 0.5])
    obs(x)
    alpha, gamma, levels, indices = obs.calculate_qparams(signed=False)
    assert levels.numel() > 0
    qt = quantize_APoT(x, alpha, gamma, levels, indices)
    dq = dequantize_APoT(qt)
    assert dq.shape == x.shape
    lin = LinearAPoT(tp.rand(3, 4), 8, 2)
    assert lin(tp.rand(2, 4)).shape == (2, 3)


# ---------------------------------------------------------------------------
# ns fx
# ---------------------------------------------------------------------------


def test_numeric_suite_fx():
    from tensorplay.ao.ns._numeric_suite_fx import extract_weights

    model = _TwoLayer().eval()
    results = extract_weights("a", model, "b", model)
    assert "a" in results and "b" in results


def test_correct_bias_and_equalize():
    from tensorplay.ao.quantization._equalize import equalize

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    equalize(model, [["0", "1"]], inplace=True)
    assert model is not None