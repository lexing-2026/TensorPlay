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
    return getattr(value, "grad_fn", None) or getattr(value, "grad_accumulator", None)


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
    for root in roots:
        fn = _get_grad_fn_or_grad_acc(root)
        if fn is not None:
            edges.setdefault(root, set()).add(fn)
    return edges


def get_param_groups(inputs: Iterable[Any], params: Iterable[Any], reverse_edges_dict: dict[Any, set[Any]]) -> list[list[Any]]:
    del inputs, reverse_edges_dict
    return [[param] for param in params]


def _autograd_grad_for_inputs(outputs: Any, inputs: Iterable[Any], grad_outputs: Any = None, retain_graph: bool = True, allow_unused: bool = True) -> tuple[Any, ...]:
    return tuple(tp.autograd.grad(outputs, tuple(inputs), grad_outputs=grad_outputs, retain_graph=retain_graph, allow_unused=allow_unused))


def stage_backward_input(stage_outputs_or_loss: Any, output_grads: Any, input_values: Iterable[Any], weights: Iterable[Any]) -> tuple[Any, ...]:
    del weights
    return _autograd_grad_for_inputs(stage_outputs_or_loss, input_values, output_grads)


def stage_backward_weight(weights: Iterable[Any], param_groups: Iterable[Iterable[Any]], retain_graph: bool = True) -> tuple[Any, ...]:
    del weights, param_groups, retain_graph
    return ()


def stage_backward(stage_output: Any, output_grads: Any, input_values: Iterable[Any], outputs_with_grads_idxs: Iterable[int] | None = None) -> tuple[Any, ...]:
    del outputs_with_grads_idxs
    return stage_backward_input(stage_output, output_grads, input_values, ())


def _null_coalesce_accumulate(lhs: Any, rhs: Any) -> Any:
    return rhs if lhs is None else lhs + rhs
