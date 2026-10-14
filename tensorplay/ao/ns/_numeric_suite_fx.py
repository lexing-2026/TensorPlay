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
    """Extract and compare weights from two models.

    Subgraphs are matched with :func:`get_matching_subgraph_pairs`; weights are
    extracted from the base op node of each matched pair.
    """
    from .fx.graph_matcher import get_matching_subgraph_pairs

    gm_a = _trace_model(model_a)
    gm_b = _trace_model(model_b)
    if base_name_to_sets_of_related_ops is None:
        from .fx.mappings import get_base_name_to_sets_of_related_ops as _get_sets

        base_name_to_sets_of_related_ops = _get_sets()
    matched_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )
    results = {
        model_name_a: {"weights": {}},
        model_name_b: {"weights": {}},
    }
    for ref_name, (subgraph_a, subgraph_b) in matched_pairs.items():
        node_a = subgraph_a.base_op_node
        node_b = subgraph_b.base_op_node
        weight_a = extract_weight_from_node(node_a, gm_a)
        weight_b = extract_weight_from_node(node_b, gm_b)
        if weight_a is not None:
            results[model_name_a]["weights"][ref_name] = weight_a
        if weight_b is not None:
            results[model_name_b]["weights"][ref_name] = weight_b
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

class ShadowOutputComparisonLogger(OutputLogger):
    """Logger that captures both outputs of an NNShadowWrapper.

    ``forward`` receives a 2-tuple ``(output_a, output_b)`` and records both
    activations plus an SQNR comparison.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparison_fn = compute_sqnr
        self.comparison_fn_name = "sqnr"
        self.comparisons = []

    def forward(self, x):
        if not self.enabled or not self.save_activations:
            return x
        if not (isinstance(x, tuple) and len(x) == 2):
            raise AssertionError("ShadowOutputComparisonLogger expects a 2-tuple input")
        output_a, output_b = x
        if isinstance(output_a, tp.Tensor) and isinstance(output_b, tp.Tensor):
            self.stats.append(output_a.detach())
            self.stats_rnn.append((output_a.detach(), output_b.detach()))
            self.comparisons.append(
                self.comparison_fn(output_a.detach(), output_b.detach())
            )
        return x


def _call_module_nodes(gm):
    return {n.name: n for n in gm.graph.nodes if n.op == "call_module"}


def add_shadows(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
    matched_node_name_pairs: list[tuple[str, str]] | None = None,
    base_name_to_sets_of_related_ops=None,
    unmatchable_types_map=None,
) -> NSResultsType:
    """Add shadow modules and comparison loggers to ``model_a``.

    For every matched ``(node_name_a, node_name_b)`` pair, an
    :class:`NNShadowWrapper` running both submodules is inserted after the
    node in ``model_a``, followed by a :class:`ShadowOutputComparisonLogger`.
    The returned dictionary contains the modified graph module and logger node
    names for both models.
    """
    from .fx.n_shadows_utils import NNShadowWrapper

    gm_a = _trace_model(model_a)
    gm_b = _trace_model(model_b)
    nodes_a = _call_module_nodes(gm_a)
    nodes_b = _call_module_nodes(gm_b)

    if matched_node_name_pairs is None:
        matched_node_name_pairs = [
            (name_a, name_a)
            for name_a in nodes_a
            if name_a in nodes_b and type(gm_a.get_submodule(nodes_a[name_a].target))
            is type(gm_b.get_submodule(nodes_b[name_a].target))
        ]

    loggers_a: dict[str, str] = {}
    for name_a, name_b in matched_node_name_pairs:
        if name_a not in nodes_a or name_b not in nodes_b:
            continue
        node_a = nodes_a[name_a]
        mod_a = gm_a.get_submodule(node_a.target)
        mod_b = gm_b.get_submodule(nodes_b[name_b].target)
        wrapper = NNShadowWrapper(
            mod_a, mod_b, name_a, name_b,
            model_name_a=model_name_a, model_name_b=model_name_b,
        )
        wrapper_name = f"__shadow_{name_a}"
        gm_a.add_submodule(wrapper_name, wrapper)
        # the wrapper consumes the same inputs as node_a, plus node_a's output
        with gm_a.graph.inserting_after(node_a):
            shadow_node = gm_a.graph.call_module(
                wrapper_name, (*node_a.args, node_a)
            )
        logger = ShadowOutputComparisonLogger(
            ref_node_name=name_a,
            prev_node_name=name_a,
            model_name=model_name_a,
            ref_name=name_a,
            prev_node_target_type=type(mod_a).__name__,
            ref_node_target_type=type(mod_a).__name__,
            results_type="node_output",
        )
        logger_name = f"__shadow_logger_{name_a}"
        gm_a.add_submodule(logger_name, logger)
        with gm_a.graph.inserting_after(shadow_node):
            gm_a.graph.call_module(logger_name, (shadow_node,))
        loggers_a[name_a] = logger_name
    gm_a.recompile()

    results: NSResultsType = {
        model_name_a: {"node_output": {"model": gm_a, "loggers": loggers_a}},
        model_name_b: {"node_output": {"model": gm_b, "loggers": {}}},
    }
    return results


def extract_shadow_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    node_name_to_results_type,
    results_type="node_output",
):
    """Extract logged activations and comparisons from a shadowed model."""
    results: NSResultsType = {}
    if isinstance(model_a, dict):
        gm = model_a["node_output"]["model"]
        loggers = model_a["node_output"]["loggers"]
        for node_name, logger_name in loggers.items():
            logger = gm.get_submodule(logger_name)
            results[node_name] = {
                "model_a": {"node_output": list(logger.stats)},
                "model_b": {"node_output": [s for s in logger.stats_rnn]},
                "comparisons": list(getattr(logger, "comparisons", [])),
            }
    return results


def extract_shadow_weights(
    model_a: nn.Module,
    model_b: nn.Module,
    node_name_to_results_type,
    results_type="node_output",
):
    """Extract weights from a shadowed model's wrappers."""
    from .fx.n_shadows_utils import extract_weight_comparison

    results: NSResultsType = {}
    if isinstance(model_a, dict):
        gm = model_a["node_output"]["model"]
        loggers = model_a["node_output"]["loggers"]
        for node_name, logger_name in loggers.items():
            wrapper_name = f"__shadow_{node_name}"
            if hasattr(gm, wrapper_name):
                wrapper = gm.get_submodule(wrapper_name)
                results[node_name] = extract_weight_comparison(wrapper)
    return results
