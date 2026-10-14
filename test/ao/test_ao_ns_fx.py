"""Tests for the completed FX numeric suite (ns.fx) and its helpers."""

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.graph import Graph, GraphModule
from tensorplay.ao import quantization as q
from tensorplay.ao.ns import _numeric_suite_fx as ns_fx
from tensorplay.ao.ns.fx import graph_passes, n_shadows_utils, pattern_utils
from tensorplay.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping


def _two_layer():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return M().eval()


# ---------------------------------------------------------------------------
# QConfigMultiMapping
# ---------------------------------------------------------------------------


def test_qconfig_multi_mapping():
    qc1 = q.QConfig(q.MinMaxObserver, q.default_weight_observer)
    qc2 = q.QConfig(q.HistogramObserver, q.default_per_channel_weight_observer)
    mapping = QConfigMultiMapping() \
        .set_global([qc1, qc2]) \
        .set_object_type(nn.Linear, [qc2]) \
        .set_module_name("foo.bar", [None, qc1]) \
        .set_module_name_regex("foo.*", [qc1, qc2])
    assert len(mapping.qconfig_mappings_list) == 2
    # the second mapping must have a None for linear so the global qc2 does not leak
    assert mapping.qconfig_mappings_list[1].get_object_type_qconfig(nn.Linear) is None
    assert mapping.qconfig_mappings_list[0].get_object_type_qconfig(nn.Linear) is qc2
    # [None, qc1] collapses None during dedup then pads -> [qc1, None]
    assert mapping.qconfig_mappings_list[0].get_module_name_qconfig("foo.bar") is qc1
    assert mapping.qconfig_mappings_list[1].get_module_name_qconfig("foo.bar") is None
    # regex is applied per mapping
    assert mapping.qconfig_mappings_list[0].get_module_name_regex_qconfig("foo.bar") is qc1
    assert mapping.qconfig_mappings_list[1].get_module_name_regex_qconfig("foo.bar") is qc2

    # from_list_qconfig_mapping
    single = q.QConfigMapping().set_global(qc1).set_object_type(nn.Linear, qc2)
    rebuilt = QConfigMultiMapping.from_list_qconfig_mapping([single])
    assert len(rebuilt.qconfig_mappings_list) == 1
    assert rebuilt.qconfig_mappings_list[0].get_object_type_qconfig(nn.Linear) is qc2


# ---------------------------------------------------------------------------
# pattern_utils
# ---------------------------------------------------------------------------


def test_get_type_a_related_to_b():
    from tensorplay.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops

    related = pattern_utils.get_type_a_related_to_b(get_base_name_to_sets_of_related_ops())
    assert (nn.Linear, nn.Linear) in related
    assert (nn.Conv2d, nn.Linear) not in related


def test_get_reversed_fusions():
    fusions = pattern_utils.get_reversed_fusions()
    fusion_types = [f[0] for f in fusions]
    assert any(isinstance(f, tuple) and nn.Conv2d in f for f in fusion_types)
    # BN fusion patterns present (output-first order)
    assert any(f == (nn.BatchNorm1d, nn.Conv1d) for f in fusion_types)
    assert any(f == (nn.BatchNorm2d, nn.Conv2d) for f in fusion_types)


def test_end_node_matches_reversed_fusion():
    from tensorplay.ao.quantization.fx.tracer import QuantizationTracer

    model = _two_layer()
    gm = QuantizationTracer().trace(model)
    relu_nodes = [n for n in gm.graph.nodes if n.op == "call_module" and n.target == "relu"]
    conv_pattern = (nn.ReLU, nn.Linear)  # output-first: relu(linear(x))
    assert pattern_utils.end_node_matches_reversed_fusion(relu_nodes[0], conv_pattern, gm, set())
    # an already-seen node cannot be reused
    seen = {relu_nodes[0]}
    assert not pattern_utils.end_node_matches_reversed_fusion(
        relu_nodes[0], conv_pattern, gm, seen
    )


# ---------------------------------------------------------------------------
# graph_passes
# ---------------------------------------------------------------------------


def test_remove_quant_dequant_pairs():
    g = Graph()
    x = g.placeholder("x")
    scale = g.get_attr("scale")
    zp = g.get_attr("zp")
    quant = g.call_method("quantize_per_tensor", (x, scale, zp, "qint8"))
    dequant = g.call_method("dequantize", (quant,))
    g.output(dequant)
    root = nn.Module()
    root.scale = tp.tensor(1.0)
    root.zp = tp.tensor(0)
    gm = GraphModule(root, g)
    graph_passes.remove_quant_dequant_pairs(gm)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    assert output_node.args[0].op == "placeholder"
    quant_nodes = [n for n in gm.graph.nodes if "quantize" in str(n.target)]
    assert len(quant_nodes) == 0


def test_add_quantize_dequantize_pairs():
    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x)

    g = Graph()
    x = g.placeholder("x")
    relu = g.call_module("relu", (x,))
    g.output(relu)
    gm = GraphModule(R(), g)
    relu_node = [n for n in gm.graph.nodes if n.op == "call_module"][0]
    graph_passes.add_quantize_dequantize_pairs(gm, relu_node, scale=0.5, zero_point=0, dtype="qint8")
    targets = [n.target for n in gm.graph.nodes]
    assert "quantize_per_tensor" in targets
    assert "dequantize" in targets
    out = gm(tp.rand(2, 3))
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# n_shadows_utils / numeric suite fx shadow workflow
# ---------------------------------------------------------------------------


def test_output_prop():
    from tensorplay.ao.quantization.fx.tracer import QuantizationTracer

    gm = QuantizationTracer().trace(_two_layer())
    prop = n_shadows_utils.OutputProp(gm)
    prop.propagate(tp.rand(2, 4))
    call_nodes = [n for n in gm.graph.nodes if n.op == "call_module"]
    assert len(call_nodes) > 0
    assert all(n in prop.env for n in call_nodes)


def test_add_shadows_and_extract():
    model = _two_layer()
    results = ns_fx.add_shadows(
        "float", model, "quant", model, matched_node_name_pairs=[("linear", "linear")]
    )
    gm = results["float"]["node_output"]["model"]
    gm(tp.rand(2, 4))  # execute shadowed graph
    info = ns_fx.extract_shadow_logger_info(results["float"], results["quant"], {})
    assert len(info) == 1  # only the linear node was shadowed
    for node_name, res in info.items():
        assert len(res["comparisons"]) == 1
        assert res["comparisons"][0] is not None
    weights = ns_fx.extract_shadow_weights(results["float"], results["quant"], {})
    assert len(weights) == 1
    assert len(weights["linear"]["float"]) == 1
    assert len(weights["linear"]["quant"]) == 1


def test_shadow_output_comparison_logger():
    logger = ns_fx.ShadowOutputComparisonLogger(
        ref_node_name="linear", prev_node_name="linear", model_name="a",
        ref_name="linear", prev_node_target_type="Linear",
        ref_node_target_type="Linear", results_type="node_output",
    )
    a = tp.rand(3)
    b = a + 0.01
    logger((a, b))
    assert len(logger.stats) == 1
    assert len(logger.comparisons) == 1
    assert logger.comparisons[0] > 0


def test_create_shadow_wrapper_forward():
    from tensorplay.ao.quantization.fx.tracer import QuantizationTracer
    from tensorplay.ao.ns.fx.n_shadows_utils import create_shadow_wrapper

    gm = QuantizationTracer().trace(_two_layer())
    wrapper = create_shadow_wrapper(gm, gm, "linear", "linear")
    x = tp.rand(2, 4)
    # graph usage: (*inputs, a_out) where a_out is the reference module output
    a_out, b_out = wrapper(x, x)
    assert a_out.shape == (2, 4)
    assert b_out.shape == (2, 4)