"""Autograd helpers for stage-local backward execution."""

from typing import Any, Iterable

import tensorplay as tp

__all__ = [
    "reverse_closure",
    "construct_reverse_graph",
    "get_param_groups",
    "stage_backward_input",
    "stage_backward_weight",
    "stage_backward",
]


def _get_grad_fn_or_grad_acc(value: Any) -> Any:
    if not isinstance(value, tp.Tensor) or not value.requires_grad:
        return None
    grad_fn = getattr(value, "grad_fn", None)
    if grad_fn is not None:
        return grad_fn
    try:
        viewed = value.view_as(value)
        return getattr(viewed, "grad_fn", None)
    except Exception:
        return getattr(value, "grad_accumulator", None)


def reverse_closure(roots: Iterable[Any], target_nodes: Iterable[Any], reverse_edges_dict: dict[Any, set[Any]]) -> set[Any]:
    targets = set(target_nodes)
    stack = list(roots)
    visited: set[Any] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node not in targets:
            stack.extend(reverse_edges_dict.get(node, ()))
    return visited


def construct_reverse_graph(roots: Iterable[Any]) -> dict[Any, set[Any]]:
    edges: dict[Any, set[Any]] = {}
    pending = list(roots)
    visited: set[Any] = set()
    while pending:
        value = pending.pop()
        fn = _get_grad_fn_or_grad_acc(value)
        if fn is None or fn in visited:
            continue
        visited.add(fn)
        next_functions = getattr(fn, "next_functions", ())
        for next_fn, _ in next_functions:
            if next_fn is not None:
                edges.setdefault(fn, set()).add(next_fn)
                if next_fn not in visited:
                    pending.append(getattr(next_fn, "variable", next_fn))
    return edges


def get_param_groups(inputs: Iterable[Any], params: Iterable[Any], reverse_edges_dict: dict[Any, set[Any]]) -> list[list[Any]]:
    del inputs
    groups: list[list[Any]] = []
    owners: dict[Any, list[Any]] = {}
    for param in params:
        fn = _get_grad_fn_or_grad_acc(param)
        connected = None
        if fn is not None:
            for node, descendants in reverse_edges_dict.items():
                if fn is node or fn in descendants:
                    connected = owners.get(node)
                    if connected is None:
                        connected = []
                        owners[node] = connected
                    break
        if connected is None:
            connected = []
            groups.append(connected)
        connected.append(param)
    return groups


def _autograd_grad_for_inputs(outputs: Any, inputs: Iterable[Any], grad_outputs: Any = None, retain_graph: bool = True, allow_unused: bool = True) -> tuple[Any, ...]:
    input_values = tuple(inputs)
    valid_inputs = [
        value for value in input_values
        if isinstance(value, tp.Tensor) and value.requires_grad
    ]
    if not valid_inputs:
        return tuple(None for _ in input_values)
    gradients = tuple(
        tp.autograd.grad(
            outputs,
            tuple(valid_inputs),
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )
    )
    result: list[Any] = [None] * len(input_values)
    by_id = {id(value): index for index, value in enumerate(input_values)}
    for value, gradient in zip(valid_inputs, gradients):
        result[by_id[id(value)]] = gradient
    return tuple(result)


def stage_backward_input(stage_outputs_or_loss: Any, output_grads: Any, input_values: Iterable[Any], weights: Iterable[Any]) -> tuple[Any, ...]:
    input_values = tuple(input_values)
    weight_values = tuple(weights)
    outputs = stage_outputs_or_loss
    if isinstance(outputs, tp.Tensor):
        outputs = (outputs,)
    valid_outputs = tuple(
        output for output in outputs
        if isinstance(output, tp.Tensor) and (output.requires_grad or output.grad_fn is not None)
    )
    if not valid_outputs:
        return tuple(None for _ in input_values)
    grads = _autograd_grad_for_inputs(
        valid_outputs,
        input_values,
        output_grads,
        retain_graph=True,
        allow_unused=True,
    )
    if weight_values:
        weight_grads = _autograd_grad_for_inputs(
            valid_outputs,
            weight_values,
            output_grads,
            retain_graph=True,
            allow_unused=True,
        )
        for weight, gradient in zip(weight_values, weight_grads):
            if gradient is not None:
                weight.grad = gradient if weight.grad is None else weight.grad + gradient
    for value, gradient in zip(input_values, grads):
        if gradient is not None and isinstance(value, tp.Tensor):
            value.grad = gradient if value.grad is None else value.grad + gradient
    return grads


def stage_backward_weight(weights: Iterable[Any], param_groups: Iterable[Iterable[Any]], retain_graph: bool = True) -> tuple[Any, ...]:
    del retain_graph
    values = tuple(weights)
    gradients: list[Any] = []
    for param in values:
        gradient = getattr(param, "grad", None)
        gradients.append(gradient)
    for group in param_groups:
        if isinstance(group, dict):
            group_values = group.get("params", ())
        else:
            group_values = group
        for param in group_values:
            if param in values and getattr(param, "grad", None) is not None:
                continue
    return tuple(gradients)


def stage_backward(stage_output: Any, output_grads: Any, input_values: Iterable[Any], outputs_with_grads_idxs: Iterable[int] | None = None) -> tuple[Any, ...]:
    outputs = stage_output if isinstance(stage_output, (tuple, list)) else (stage_output,)
    grads = output_grads
    if grads is not None and not isinstance(grads, (tuple, list)):
        grads = (grads,)
    if outputs_with_grads_idxs is not None:
        allowed = set(outputs_with_grads_idxs)
        outputs = tuple(value for index, value in enumerate(outputs) if index in allowed)
        if grads is not None:
            grads = tuple(value for index, value in enumerate(grads) if index in allowed)
    return stage_backward_input(outputs, grads, input_values, ())


def _null_coalesce_accumulate(lhs: Any, rhs: Any) -> Any:
    return rhs if lhs is None else lhs + rhs
