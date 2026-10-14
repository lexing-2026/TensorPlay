"""Model quantization readiness analysis.

This module provides a public, tracing-free analyzer that reports which
layers of a float model can be quantized, which quantized modules they map
to, which qconfig dtype they would use, and whether they are fusion
candidates.  It is a convenience layer above the quantization mappings and the
fusion registry, useful for pre-flight checks before running prepare/convert.
"""

from __future__ import annotations

from typing import Any

from tensorplay import nn

from .fuser_method_mappings import _DEFAULT_OP_LIST_TO_FUSER_METHOD
from .qconfig import QConfigMapping
from .quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_qat_module_mappings,
    get_default_static_quant_module_mappings,
    no_observer_set,
)

__all__ = [
    "analyze_model",
    "print_model_analysis",
    "model_quantization_summary",
    "quantization_ready",
]


def _is_fusion_candidate(module: nn.Module) -> bool:
    """Return whether the module type appears in any fusion pattern."""
    module_type = type(module)
    for pattern in _DEFAULT_OP_LIST_TO_FUSER_METHOD:
        if module_type in pattern:
            return True
    return False


def _resolve_qconfig(module, fqn, qconfig_mapping):
    if qconfig_mapping is None:
        return None
    if isinstance(qconfig_mapping, dict):
        return qconfig_mapping.get("")
    if isinstance(qconfig_mapping, QConfigMapping):
        return qconfig_mapping._get_qconfig_for_module(module, fqn)
    return None


def analyze_model(
    model: nn.Module,
    qconfig_mapping=None,
) -> dict[str, dict[str, Any]]:
    """Return a per-module quantization readiness report.

    Args:
        model: the float model to analyze.
        qconfig_mapping: optional :class:`QConfigMapping` (or legacy qconfig
            dict) used to resolve the qconfig/dtype per module.

    Returns:
        A dictionary keyed by fully-qualified module name.  Each entry
        contains ``module_class``, ``is_leaf``, ``static_quantizable``,
        ``dynamic_quantizable``, ``qat_quantizable``, ``quantized_module``,
        ``has_observer``, ``fusion_candidate``, ``qconfig`` and ``dtype``.
    """
    static_mappings = get_default_static_quant_module_mappings()
    dynamic_mappings = get_default_dynamic_quant_module_mappings()
    qat_mappings = get_default_qat_module_mappings()
    no_observers = no_observer_set()

    report: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        num_children = len(list(module.named_children()))
        entry: dict[str, Any] = {
            "fqn": name,
            "module_class": type(module).__name__,
            "num_children": num_children,
            "is_leaf": num_children == 0,
            "static_quantizable": type(module) in static_mappings,
            "dynamic_quantizable": type(module) in dynamic_mappings,
            "qat_quantizable": type(module) in qat_mappings,
            "quantized_module": static_mappings.get(type(module)),
            "dynamic_quantized_module": dynamic_mappings.get(type(module)),
            "has_observer": hasattr(module, "activation_post_process"),
            "fusion_candidate": _is_fusion_candidate(module),
            "no_observer": type(module) in no_observers,
            "qconfig": None,
            "dtype": None,
        }
        qconfig = _resolve_qconfig(module, name, qconfig_mapping)
        entry["qconfig"] = qconfig
        if qconfig is not None and getattr(qconfig, "activation", None) is not None:
            try:
                observer = qconfig.activation()
                entry["dtype"] = getattr(observer, "dtype", None)
            except Exception:
                entry["dtype"] = None
        report[name] = entry
    return report


def model_quantization_summary(model: nn.Module, qconfig_mapping=None) -> dict[str, int]:
    """Return aggregate counts from :func:`analyze_model`."""
    report = analyze_model(model, qconfig_mapping)
    summary: dict[str, int] = {
        "total_modules": 0,
        "leaf_modules": 0,
        "static_quantizable": 0,
        "dynamic_quantizable": 0,
        "qat_quantizable": 0,
        "with_observer": 0,
        "fusion_candidates": 0,
        "no_observer": 0,
    }
    for entry in report.values():
        summary["total_modules"] += 1
        if entry["is_leaf"]:
            summary["leaf_modules"] += 1
        if entry["static_quantizable"]:
            summary["static_quantizable"] += 1
        if entry["dynamic_quantizable"]:
            summary["dynamic_quantizable"] += 1
        if entry["qat_quantizable"]:
            summary["qat_quantizable"] += 1
        if entry["has_observer"]:
            summary["with_observer"] += 1
        if entry["fusion_candidate"]:
            summary["fusion_candidates"] += 1
        if entry.get("no_observer"):
            summary["no_observer"] = summary.get("no_observer", 0) + 1
    return summary


def quantization_ready(model: nn.Module, qconfig_mapping=None):
    """Return ``(ok, blockers)`` for a model before running prepare/convert.

    ``ok`` is True when every leaf module is either statically quantizable or a
    recognized non-quantizable op (e.g. pooling, dropout).  ``blockers`` lists
    the FQNs of leaf modules that are not quantizable and have no qconfig
    exemption.
    """
    report = analyze_model(model, qconfig_mapping)
    blockers = [
        entry["fqn"]
        for entry in report.values()
        if entry["is_leaf"]
        and not entry["static_quantizable"]
        and not entry["dynamic_quantizable"]
        and not entry["fusion_candidate"]
        and not entry["no_observer"]
        and entry["module_class"] not in (
            "Dropout", "MaxPool1d", "MaxPool2d", "MaxPool3d", "Identity",
        )
    ]
    return len(blockers) == 0, blockers


def print_model_analysis(model: nn.Module, qconfig_mapping=None, print_fn=print) -> None:
    """Print a human-readable quantization readiness report."""
    report = analyze_model(model, qconfig_mapping)
    summary = model_quantization_summary(model, qconfig_mapping)
    print_fn("=== model quantization analysis ===")
    for key, value in summary.items():
        print_fn(f"  {key}: {value}")
    print_fn("--- per-module ---")
    for entry in report.values():
        status = []
        if entry["static_quantizable"]:
            status.append("static")
        if entry["dynamic_quantizable"]:
            status.append("dynamic")
        if entry["qat_quantizable"]:
            status.append("qat")
        if not status:
            status.append("skip")
        qmod = entry.get("quantized_module")
        qmod_name = qmod.__name__ if qmod is not None else "-"
        print_fn(
            f"  {entry['fqn']:<30} {entry['module_class']:<20} "
            f"[{','.join(status):<8}] -> {qmod_name}"
            f"{' (observer)' if entry['has_observer'] else ''}"
            f"{' (fusable)' if entry['fusion_candidate'] else ''}"
            f"{' dtype=' + str(entry['dtype']) if entry['dtype'] is not None else ''}"
        )