"""Normalize effect-bearing graph values into ordinary data flow."""

from __future__ import annotations

import operator
from typing import Any

from .exported_program import ExportedProgram
from .graph_signature import InputKind, OutputKind, TokenArgument

__all__ = ["_remove_effect_tokens"]


def _get_custom_obj_for_node(node: Any, inputs_to_lifted_custom_objs: Any, constants: Any) -> Any:
    metadata = getattr(node, "meta", {}).get("val")
    if metadata is not None and hasattr(metadata, "fake_val"):
        return metadata.fake_val
    name = getattr(node, "name", None)
    if name in inputs_to_lifted_custom_objs:
        return constants[inputs_to_lifted_custom_objs[name]]
    raise KeyError(f"custom value for {name!r} was not found")


def _target_name(target: Any) -> str:
    return getattr(target, "__name__", getattr(target, "name", str(target)))


def _replace_with_effects_node(node: Any, module: Any) -> None:
    if len(node.args) < 2:
        raise ValueError("effect node requires a token and callable")
    function = node.args[1]
    arguments = tuple(node.args[2:])
    if not callable(function):
        raise TypeError("effect node callable is not executable")
    with module.graph.inserting_before(node):
        replacement = module.graph.call_function(function, arguments, node.kwargs)
    for key, value in node.meta.items():
        replacement.meta[key] = value[1] if key == "val" and isinstance(value, tuple) and len(value) > 1 else value
    for user in list(node.users):
        if user.op != "call_function" or user.target is not operator.getitem or len(user.args) < 2:
            raise ValueError("effect node users must be indexed reads")
        index = user.args[1]
        if index == 0:
            user.replace_all_uses_with(node.args[0])
        else:
            user.replace_all_uses_with(replacement if index == 1 else replacement)
        if not user.users:
            user.graph.erase_node(user)
    if not node.users:
        node.graph.erase_node(node)


def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    """Remove token placeholders and effect tuple wrappers in place."""

    module = ep.graph_module
    for node in list(ep.graph.nodes):
        if node.op == "call_function" and _target_name(node.target) in {"with_effects", "with_effect"}:
            _replace_with_effects_node(node, module)
    token_nodes = {
        spec.arg.name
        for spec in ep.graph_signature.input_specs
        if spec.kind is InputKind.TOKEN and isinstance(spec.arg, TokenArgument)
    }
    for node in list(ep.graph.nodes):
        if node.op == "placeholder" and node.name in token_nodes and not node.users:
            node.graph.erase_node(node)
    ep.graph_signature.input_specs = [
        spec for spec in ep.graph_signature.input_specs if spec.kind is not InputKind.TOKEN
    ]
    ep.graph_signature.output_specs = [
        spec for spec in ep.graph_signature.output_specs if spec.kind is not OutputKind.TOKEN
    ]
    module.recompile()
    ep.validate()
    return ep
