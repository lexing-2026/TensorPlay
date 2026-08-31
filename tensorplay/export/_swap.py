"""Helpers for replacing selected call sites in an exported graph."""

from __future__ import annotations

import operator
from typing import Any

from .exported_program import ModuleCallSignature

__all__ = ["_swap_modules"]


def _get_getitem_users(node: Any) -> set[Any]:
    return {
        user
        for user in node.users
        if user.op == "call_function" and user.target is operator.getitem
    }


def _try_remove_connecting_pytrees(curr_module_node: Any) -> None:
    for user in list(curr_module_node.users):
        if user.op == "call_function" and user.target is operator.getitem and len(user.args) == 2:
            index = user.args[1]
            if isinstance(index, int) and index < 0:
                user.args = (curr_module_node, index)


def _remove_extraneous_pytrees(graph_module: Any) -> None:
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in {tuple, list, dict}:
            continue
        if len(node.args) != 1 or not isinstance(node.args[0], (tuple, list, dict)):
            continue
        node.replace_all_uses_with(node.args[0])
        if not node.users:
            node.graph.erase_node(node)


def _construct_inputs(
    graph_module: Any,
    signature: ModuleCallSignature,
    node_name_map: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    args: list[Any] = []
    for argument in signature.inputs:
        if argument.name not in node_name_map:
            raise KeyError(f"module input {argument.name!r} is not in the graph")
        args.append(node_name_map[argument.name])
    names = signature.forward_arg_names or []
    kwargs = {name: value for name, value in zip(names, args[len(names) * -1 :] if names else [])}
    del graph_module
    return args if not kwargs else args[: len(args) - len(kwargs)], kwargs


def _insert_call_module(
    graph_module: Any,
    args_nodes: list[Any],
    kwargs_nodes: dict[str, Any],
    target: str,
) -> Any:
    if not hasattr(graph_module, "add_submodule"):
        raise TypeError("graph module cannot register a replacement submodule")
    with graph_module.graph.inserting_before(None):
        return graph_module.graph.call_module(target, tuple(args_nodes), kwargs_nodes)


def _deconstruct_outputs(
    graph_module: Any,
    signature: ModuleCallSignature,
    module_node: Any,
    node_name_map: dict[str, Any],
) -> None:
    for index, argument in enumerate(signature.outputs):
        users = [user for user in list(module_node.users) if user.target is operator.getitem and user.args[1] == index]
        if users:
            node_name_map[argument.name] = users[0]
        else:
            with graph_module.graph.inserting_after(module_node):
                node_name_map[argument.name] = graph_module.graph.call_function(
                    operator.getitem, (module_node, index)
                )


def _fix_input_output_signature(graph_module: Any, signature: ModuleCallSignature) -> None:
    del graph_module
    if signature.forward_arg_names is not None and len(signature.forward_arg_names) > len(signature.inputs):
        raise ValueError("forward argument names exceed module inputs")


def _swap_module_helper(
    graph_module: Any,
    modules_to_swap: dict[str, Any],
    module_call_graph: dict[str, ModuleCallSignature],
) -> Any:
    for target, replacement in modules_to_swap.items():
        graph_module.add_submodule(target, replacement)
    for node in graph_module.graph.nodes:
        if node.op == "call_module" and node.target in modules_to_swap:
            node.meta["swapped"] = True
    _remove_extraneous_pytrees(graph_module)
    graph_module.recompile()
    return graph_module


def _swap_modules(exported_program: Any, modules_to_swap: dict[str, Any]) -> Any:
    """Install replacement modules and update matching graph call sites."""

    if not isinstance(modules_to_swap, dict):
        raise TypeError("modules_to_swap must be a dictionary")
    graph_module = exported_program.graph_module
    result = _swap_module_helper(graph_module, modules_to_swap, {})
    exported_program.validate()
    return result
