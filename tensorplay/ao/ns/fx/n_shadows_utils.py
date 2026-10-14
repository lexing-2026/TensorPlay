"""Shadow-model utilities for the FX numeric suite.

A "shadow" model runs a candidate (e.g. quantized) model in parallel with a
reference (float) model, allowing per-layer numeric comparison without
rewriting the reference graph.  This module provides the module-level shadow
wrapper and helpers to extract/group comparison results.
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp
from tensorplay import nn
from tensorplay.graph import GraphModule
from tensorplay.graph.interpreter import Interpreter
from tensorplay.graph.node import Node

from .ns_types import NSResultsType
from .utils import compute_sqnr

__all__ = [
    "OutputProp",
    "NNShadowWrapper",
    "create_shadow_wrapper",
    "extract_weight_comparison",
    "group_results_by_subgraph",
    "create_results_comparison",
    "print_n_shadows_summary",
]


class OutputProp(Interpreter):
    """Run a GraphModule on example inputs and record each node's output."""

    def __init__(self, module: GraphModule):
        super().__init__(module, garbage_collect_values=False)
        self.env: dict[Node, Any] = {}

    def run_node(self, node: Node) -> Any:
        result = super().run_node(node)
        self.env[node] = result
        return result

    def propagate(self, *args, **kwargs) -> Any:
        return self.run(*args, **kwargs)


class NNShadowWrapper(nn.Module):
    """Run a reference submodule and a shadow submodule side by side.

    The reference module's output is supplied as the last positional argument
    (it is already computed by the host graph); only the shadow module is
    invoked inside the wrapper.

    Args:
        module_a: reference module (not called, kept for weight extraction).
        module_b: shadow module (e.g. quantized).
        name_a: node/module name of ``module_a`` for result keys.
        name_b: node/module name of ``module_b`` for result keys.
    """

    def __init__(self, module_a: nn.Module, module_b: nn.Module,
                 name_a: str = "a", name_b: str = "b",
                 model_name_a: str | None = None, model_name_b: str | None = None):
        super().__init__()
        self.module_a = module_a
        self.module_b = module_b
        self.name_a = name_a
        self.name_b = name_b
        self.model_name_a = model_name_a
        self.model_name_b = model_name_b

    def forward(self, *args):
        *inputs, a_out = args
        b_out = self.module_b(*inputs)
        return a_out, b_out


def create_shadow_wrapper(
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_a: str,
    node_name_b: str,
) -> NNShadowWrapper:
    """Create a shadow wrapper from two graph modules at the given node names.

    Both ``node_name_a`` and ``node_name_b`` must be ``call_module`` node names.
    """
    node_a = next(n for n in gm_a.graph.nodes if n.name == node_name_a)
    node_b = next(n for n in gm_b.graph.nodes if n.name == node_name_b)
    if node_a.op != "call_module" or node_b.op != "call_module":
        raise ValueError("shadow nodes must be call_module nodes")
    mod_a = gm_a.get_submodule(node_a.target)
    mod_b = gm_b.get_submodule(node_b.target)
    return NNShadowWrapper(mod_a, mod_b, node_name_a, node_name_b)


def extract_weight_comparison(shadow_wrapper: NNShadowWrapper) -> dict:
    """Extract weight comparison from a shadow wrapper.

    The result is keyed by the model names when available (set by
    ``add_shadows``), falling back to the node names.  Each value is a list of
    extracted weight tensors.
    """
    weights_a = []
    weights_b = []
    for idx, mod in enumerate((shadow_wrapper.module_a, shadow_wrapper.module_b)):
        weights = []
        for name, param in mod.named_parameters():
            if "weight" in name:
                weights.append(param.detach())
        if not weights and hasattr(mod, "weight"):
            weights.append(mod.weight.detach())
        if weights:
            (weights_a if idx == 0 else weights_b).extend(weights)
    key_a = shadow_wrapper.model_name_a or shadow_wrapper.name_a
    key_b = shadow_wrapper.model_name_b or shadow_wrapper.name_b
    return {key_a: weights_a, key_b: weights_b}


def group_results_by_subgraph(results: NSResultsType) -> dict[str, dict]:
    """Group shadow results by subgraph/ref name, discarding the model key."""
    grouped: dict[str, dict] = {}
    for ref_name, res in results.items():
        if ref_name in ("model_a", "model_b"):
            continue
        grouped[ref_name] = res
    return grouped


def create_results_comparison(
    results: NSResultsType,
    comparison_fn=compute_sqnr,
) -> NSResultsType:
    """Augment shadow results with comparisons between A and B outputs."""
    for ref_name, res in results.items():
        if not isinstance(res, dict):
            continue
        if "model_a" in res and "model_b" in res:
            stats_a = res.get("model_a", {}).get("node_output", [])
            stats_b = res.get("model_b", {}).get("node_output", [])
            if stats_a and stats_b:
                res["comparisons"] = [
                    comparison_fn(a.detach(), b.detach())
                    for a, b in zip(stats_a, stats_b)
                ]
    return results


def print_n_shadows_summary(results: NSResultsType, print_fn=print) -> None:
    """Print a compact summary of shadow comparison results."""
    for ref_name, res in group_results_by_subgraph(results).items():
        print_fn(f"subgraph: {ref_name}")
        if isinstance(res, dict) and "comparisons" in res:
            for i, c in enumerate(res["comparisons"]):
                print_fn(f"  comparison[{i}]: {c}")