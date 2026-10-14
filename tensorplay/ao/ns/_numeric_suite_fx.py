"""FX-based numeric suite for comparing float and quantized models."""

from collections.abc import Callable

import tensorplay as tp
from tensorplay import nn
from tensorplay.graph import GraphModule

from .fx.ns_types import NSResultsType
from .fx.utils import compute_sqnr
from .fx.weight_utils import extract_weight_from_node

__all__ = [
    "OutputLogger",
    "OutputComparisonLogger",
    "extract_weights",
    "add_loggers",
    "extract_logger_info",
    "extend_logger_results_with_comparison",
    "loggers_set_enabled",
    "loggers_set_save_activations",
]


class OutputLogger(nn.Module):
    """Captures intermediate node outputs during model execution."""

    _is_impure = True

    def __init__(self, ref_node_name="", prev_node_name="", model_name="",
                 ref_name="", prev_node_target_type="", ref_node_target_type="",
                 results_type="", index_within_arg=0, index_of_arg=0, fqn=None,
                 qconfig_str=""):
        super().__init__()
        self.stats = []
        self.stats_rnn = []
        self.ref_node_name = ref_node_name
        self.prev_node_name = prev_node_name
        self.model_name = model_name
        self.ref_name = ref_name
        self.prev_node_target_type = prev_node_target_type
        self.ref_node_target_type = ref_node_target_type
        self.results_type = results_type
        self.index_within_arg = index_within_arg
        self.index_of_arg = index_of_arg
        self.fqn = fqn
        self.enabled = True
        self.save_activations = True
        self.qconfig_str = qconfig_str

    def forward(self, x):
        if not self.enabled or not self.save_activations:
            return x
        if isinstance(x, tp.Tensor):
            self.stats.append(x.detach())
        elif isinstance(x, tuple) and len(x) == 2 and len(x[1]) == 2:
            self.stats_rnn.append((x[0].detach(), (x[1][0].detach(), x[1][1].detach())))
        return x

    def __repr__(self):
        clean_dict = {k: v for k, v in self.__dict__.items() if k != "training" and not k.startswith("_")}
        return f"OutputLogger({clean_dict})"


class OutputComparisonLogger(OutputLogger):
    """Logger that also records the SQNR against a reference activation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparison_fn = compute_sqnr
        self.comparison_fn_name = "sqnr"
        self.comparisons = []

    def forward(self, x, x_ref):
        if not self.enabled or not self.save_activations:
            return x
        if isinstance(x, tp.Tensor) and isinstance(x_ref, tp.Tensor):
            self.stats.append(x.detach())
            self.comparisons.append(self.comparison_fn(x.detach(), x_ref.detach()))
        return x


def _trace_model(model):
    from tensorplay.ao.quantization.fx.tracer import QuantizationTracer

    tracer = QuantizationTracer()
    return tracer.trace(model)


def extract_weights(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
    base_name_to_sets_of_related_ops=None,
    unmatchable_types_map=None,
    op_to_type_to_weight_extraction_fn=None,
) -> NSResultsType:
    """Extract and compare weights from two models."""
    gm_a = _trace_model(model_a)
    gm_b = _trace_model(model_b)
    results = {
        model_name_a: {"weights": {}},
        model_name_b: {"weights": {}},
    }
    modules_a = {name: mod for name, mod in gm_a.named_modules() if name}
    modules_b = {name: mod for name, mod in gm_b.named_modules() if name}
    for name, mod in modules_a.items():
        if name in modules_b and type(mod) is type(modules_b[name]):
            weight_a = extract_weight_from_node(next(
                (n for n in gm_a.graph.nodes if n.op == "call_module" and n.target == name), None
            ), gm_a) if any(n.op == "call_module" and n.target == name for n in gm_a.graph.nodes) else None
            weight_b = extract_weight_from_node(next(
                (n for n in gm_b.graph.nodes if n.op == "call_module" and n.target == name), None
            ), gm_b) if any(n.op == "call_module" and n.target == name for n in gm_b.graph.nodes) else None
            results[model_name_a]["weights"][name] = weight_a
            results[model_name_b]["weights"][name] = weight_b
    return results


def _add_loggers_one_model(gm, model_name, results_type="node_output"):
    """Attach an OutputLogger after every call_module node."""
    node_name_to_logger = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            logger = OutputLogger(
                ref_node_name=node.name,
                prev_node_name=node.name,
                model_name=model_name,
                ref_name=node.target,
                prev_node_target_type=str(node.target),
                ref_node_target_type=str(node.target),
                results_type=results_type,
            )
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_module(f"__logger_{node.name}", (node,))
            gm.add_submodule(f"__logger_{node.name}", logger)
            node_name_to_logger[node.name] = new_node
    return gm, node_name_to_logger


def add_loggers(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
    base_name_to_sets_of_related_ops=None,
    unmatchable_types_map=None,
) -> NSResultsType:
    """Add output loggers to both models and return logger node names."""
    gm_a = _trace_model(model_a)
    gm_b = _trace_model(model_b)
    gm_a, loggers_a = _add_loggers_one_model(gm_a, model_name_a)
    gm_b, loggers_b = _add_loggers_one_model(gm_b, model_name_b)
    results = {
        model_name_a: {"node_output": {"model": gm_a, "loggers": loggers_a}},
        model_name_b: {"node_output": {"model": gm_b, "loggers": loggers_b}},
    }
    return results


def _extract_logger_info_one_model(gm, loggers, results_type):
    results = {}
    for node_name, logger_node in loggers.items():
        logger_module_name = f"__logger_{node_name}"
        if hasattr(gm, logger_module_name):
            logger = gm.get_submodule(logger_module_name)
            results[node_name] = {"node_output": list(logger.stats)}
    return results


def extract_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    node_name_to_results_type,
    node_name_to_io_type=None,
    results_type="node_output",
):
    """Extract the logged activations from both models."""
    results = {}
    for model_name, model in (("model_a", model_a), ("model_b", model_b)):
        if isinstance(model, dict):
            gm = model["node_output"]["model"]
            loggers = model["node_output"]["loggers"]
            results[model_name] = _extract_logger_info_one_model(gm, loggers, results_type)
    return results


def extend_logger_results_with_comparison(
    model_name_a, model_a, model_name_b, model_b, node_name_to_results_type, results_type="node_output"
):
    """Augment logger results with SQNR comparisons."""
    results = extract_logger_info(model_a, model_b, node_name_to_results_type, results_type=results_type)
    for node_name in results.get(model_name_a, {}):
        stats_a = results[model_name_a][node_name].get(results_type, [])
        stats_b = results[model_name_b].get(node_name, {}).get(results_type, [])
        if stats_a and stats_b:
            comparisons = [
                compute_sqnr(a.detach(), b.detach()) for a, b in zip(stats_a, stats_b)
            ]
            results[model_name_a][node_name]["comparisons"] = comparisons
    return results


def loggers_set_enabled(model: nn.Module, enabled: bool) -> None:
    for name, module in model.named_modules():
        if isinstance(module, OutputLogger):
            module.enabled = enabled


def loggers_set_save_activations(model: nn.Module, save: bool) -> None:
    for name, module in model.named_modules():
        if isinstance(module, OutputLogger):
            module.save_activations = save