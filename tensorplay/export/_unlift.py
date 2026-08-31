"""State lifting and input validation helpers for exported modules."""

from __future__ import annotations

import copy
import operator
from typing import Any

from ..graph._pytree import TreeSpec, tree_flatten

__all__ = [
    "_check_input_constraints_for_module",
    "_check_inputs_match",
    "_convert_guards_code_to_fn",
    "_get_codegen",
    "_insert_copy_for_mutations",
    "_register_attrs_to_new_gm",
    "_unlift",
    "_unlift_inputs_as_getattr",
    "eq_spec",
]


def eq_spec(first: TreeSpec, second: TreeSpec) -> bool:
    if first.type is not second.type or first.context != second.context:
        return False
    return len(first.children_specs) == len(second.children_specs) and all(
        eq_spec(left, right) for left, right in zip(first.children_specs, second.children_specs)
    )


def _check_inputs_match(args: tuple[Any, ...], kwargs: dict[str, Any], in_spec: TreeSpec) -> list[Any]:
    flat, spec = tree_flatten((args, kwargs))
    if not eq_spec(spec, in_spec):
        raise ValueError(f"input structure does not match: got {spec!r}, expected {in_spec!r}")
    return flat


def _force_ep_signature_match(ep_guards_code: list[str], input_paths: list[str]) -> list[str]:
    names = set(input_paths)
    return [guard for guard in ep_guards_code if any(name in guard for name in names)]


def _force_gm_signature_match(ep_guards_code: list[str], signature: Any) -> list[str]:
    parameters = getattr(signature, "parameters", {})
    return _force_ep_signature_match(ep_guards_code, list(parameters))


def _convert_guards_code_to_fn(
    guards_code: list[str], locals_dict: dict[str, Any] | None = None
) -> Any:
    expressions = [compile(code, "<export-guard>", "eval") for code in guards_code]
    namespace = dict(locals_dict or {})

    def check(*args: Any, **kwargs: Any) -> bool:
        values = dict(namespace)
        values.update(kwargs)
        values["args"] = args
        return all(bool(eval(expression, {}, values)) for expression in expressions)

    return check


def _check_input_constraints_for_module(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return module(*args, **kwargs)


def _check_input_constraints_pre_hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    _check_input_constraints_for_module(module, args, kwargs)


def _unlift_inputs_as_getattr(graph_module: Any, state_names: dict[str, str] | None = None) -> Any:
    state_names = state_names or {}
    for node in list(graph_module.graph.nodes):
        if node.op != "placeholder" or node.name not in state_names:
            continue
        replacement = graph_module.graph.get_attr(state_names[node.name])
        node.replace_all_uses_with(replacement)
        if not node.users:
            node.graph.erase_node(node)
    graph_module.recompile()
    return graph_module


def _insert_copy_for_mutations(graph: Any, mutated_nodes: set[Any] | None = None) -> int:
    mutated_nodes = mutated_nodes or set()
    inserted = 0
    for node in list(graph.nodes):
        if node.op != "call_method" or not isinstance(node.target, str) or not node.target.endswith("_"):
            continue
        if not node.args:
            continue
        source = node.args[0]
        if source not in mutated_nodes:
            continue
        with graph.inserting_before(node):
            copied = graph.call_function(copy.copy, (source,))
        node.replace_input_with(source, copied)
        inserted += 1
    return inserted


def _get_codegen(graph_module: Any) -> Any:
    return getattr(graph_module.graph, "_codegen", None)


def _register_attrs_to_new_gm(old_module: Any, new_module: Any, names: list[str]) -> None:
    for name in names:
        value = old_module
        for atom in name.split("."):
            value = getattr(value, atom)
        parent_name, _, leaf = name.rpartition(".")
        parent = new_module if not parent_name else new_module
        for atom in parent_name.split(".") if parent_name else ():
            parent = getattr(parent, atom)
        setattr(parent, leaf, value)


def _unlift(exported_program: Any, *, strict: bool = False) -> Any:
    del strict
    module = copy.deepcopy(exported_program.graph_module)
    module.recompile()
    return module
