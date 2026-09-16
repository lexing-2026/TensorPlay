"""Tests for the numerical suite: float versus quantized model comparison."""

import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.ao.ns import _model_utils, _numeric_suite as ns


class _FloatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class _ShadowNet(nn.Module):
    """Same structure with the linear replaced by a quantize-dequantize pair."""

    def __init__(self, float_net):
        super().__init__()
        self.fc = _QuantizedLinearLike(float_net.fc)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class _QuantizedLinearLike(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.register_buffer("_packed_params", tp.quantize_per_channel(
            linear.weight.detach(),
            tp.full((3,), 0.01),
            tp.zeros((3,), dtype=tp.int64),
            axis=0,
            dtype=tp.qint8,
        ))
        self.bias = linear.bias.detach() if linear.bias is not None else None

    def forward(self, x):
        return tp.nn.functional.linear(
            x, tp.dequantize(self._packed_params), self.bias)


def test_compare_weights_pairs_matching_modules():
    float_net = _FloatNet().eval()
    shadow = _ShadowNet(float_net).eval()
    result = ns.compare_weights(float_net.state_dict(), shadow.state_dict())
    assert "fc._packed_params" in result
    pair = result["fc._packed_params"]
    diff = (pair["float"] - tp.dequantize(pair["quantized"])).abs().max()
    assert float(diff) < 0.02


def test_compare_model_outputs_records_stats():
    float_net = _FloatNet().eval()
    shadow = _ShadowNet(float_net).eval()
    ns.prepare_model_outputs(
        float_net, shadow, allow_list={nn.Linear, _QuantizedLinearLike})
    x = tp.rand(2, 4)
    with tp.no_grad():
        float_net(x)
        shadow(x)
    float_stats = ns.get_logger_dict(float_net)
    shadow_stats = ns.get_logger_dict(shadow)
    assert "fc.stats" in shadow_stats
    assert len(shadow_stats["fc.stats"]["tensor_val"]) >= 1
    assert "fc.stats" in float_stats


def test_compare_model_stub_shadows_linear():
    float_net = _FloatNet().eval()
    shadow = _ShadowNet(float_net).eval()
    x = tp.rand(2, 4)
    results = ns.compare_model_stub(float_net, shadow, {nn.Linear}, x)
    assert "fc.stats" in results
    entry = results["fc.stats"]
    assert len(entry["float"]) == len(entry["quantized"]) == 1


def test_model_utils_get_and_swap():
    model = _FloatNet().eval()
    fc = _model_utils.get_module(model, "fc")
    assert isinstance(fc, nn.Linear)
    assert _model_utils.parent_child_names("a.b.c") == ("a.b", "c")

    from tensorplay.ao.nn.quantized import dynamic as nnqd

    swapped = _model_utils.swap_module(fc, {nn.Linear: nnqd.Linear}, {})
    assert isinstance(swapped, nnqd.Linear)
    # an unmapped module type comes back unchanged
    assert _model_utils.swap_module(model.relu, {nn.Linear: nnqd.Linear}, {}) is model.relu
