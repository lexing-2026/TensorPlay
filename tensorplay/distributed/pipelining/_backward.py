"""Autograd helpers for stage-local backward execution."""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Callable

import tensorplay as tp
from ..tensor import DTensor

from ._debug import map_debug_info

__all__ = [
    "reverse_closure",
    "construct_reverse_graph",
    "get_param_groups",
    "stage_backward_input",
    "stage_backward_weight",
    "stage_backward",
]


class _HookHandle:
    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove
        self._removed = False

    def remove(self) -> None:
        if not self._removed:
            self._removed = True
            self._remove()


def _is_tensor(value: Any) -> bool:
    return isinstance(value, (tp.Tensor, DTensor))


def _local_tensor(value: Any) -> Any:
    return value.to_local() if isinstance(value, DTensor) else value


def _get_grad_fn_or_grad_acc(value: Any) -> Any:
    if not _is_tensor(value) or not value.requires_grad:
        return None
    value = _local_tensor(value)
    grad_fn = getattr(value, "grad_fn", None)
    if grad_fn is not None:
        return grad_fn
    try:
        viewed = value.view_as(value)
    except Exception:
        return getattr(value, "grad_accumulator", None)
    viewed_grad_fn = getattr(viewed, "grad_fn", None)
    if viewed_grad_fn is not None:
        next_functions = getattr(viewed_grad_fn, "next_functions", ())
        if next_functions:
            return next_functions[0][0]
    accumulator = getattr(value, "grad_accumulator", None)
    if accumulator is not None:
        return accumulator
    raise RuntimeError("unable to locate the gradient node for a tensor")


def reverse_closure(
    roots: Iterable[Any],
    target_nodes: Iterable[Any],
    reverse_edges_dict: dict[Any, Iterable[Any]],
) -> tuple[set[Any], set[Any]]:
    closure: set[Any] = set()
    visited_target_nodes: set[Any] = set()
    targets = set(target_nodes)
    queue: collections.deque[Any] = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            queue.append(node)
    while queue:
        node = queue.popleft()
        for parent in reverse_edges_dict.get(node, ()):
            if parent in closure or parent is None:
                continue
            if parent in targets:
                visited_target_nodes.add(parent)
                continue
            closure.add(parent)
            queue.append(parent)
    return closure, visited_target_nodes


def construct_reverse_graph(roots: Iterable[Any]) -> dict[Any, list[Any]]:
    queue: collections.deque[Any] = collections.deque()
    root_seen: set[Any] = set()
    reverse_edges_dict: dict[Any, list[Any]] = collections.defaultdict(list)
    for node in roots:
        if node is not None and node not in root_seen:
            queue.append(node)
            root_seen.add(node)
    while queue:
        node = queue.popleft()
        for parent, _ in getattr(node, "next_functions", ()):
            if parent is None:
                continue
            if not reverse_edges_dict[parent]:
                queue.append(parent)
            reverse_edges_dict[parent].append(node)
    return reverse_edges_dict


def get_param_groups(
    inputs: Iterable[Any],
    params: Iterable[Any],
    reverse_edges_dict: dict[Any, Iterable[Any]],
) -> list[dict[str, Any]]:
    input_nodes = tuple(inputs)
    param_nodes = tuple(params)
    inputs_closure, _ = reverse_closure(input_nodes, set(), reverse_edges_dict)
    param_groups: dict[Any, dict[str, set[Any]]] = {}
    for param in param_nodes:
        closure, intersected = reverse_closure(
            (param,), inputs_closure, reverse_edges_dict
        )
        del closure
        param_group: dict[str, set[Any]] = {
            "params": {param},
            "intermediates": intersected,
        }
        for input_node in intersected:
            existing = param_groups.get(input_node)
            if existing is None:
                param_groups[input_node] = param_group
            else:
                existing["params"].update(param_group["params"])
                existing["intermediates"].update(param_group["intermediates"])
                param_group = existing

    unique_groups: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for param_group in param_groups.values():
        if id(param_group) not in seen_ids:
            seen_ids.add(id(param_group))
            unique_groups.append(param_group)
    return unique_groups


def _autograd_grad_for_inputs(
    outputs: Sequence[Any],
    inputs: Sequence[Any],
    grad_outputs: Sequence[Any] | None = None,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[Any, ...]:
    grad_indices: list[int] = []
    inputs_requiring_grad: list[Any] = []
    for index, value in enumerate(inputs):
        if _is_tensor(value) and value.requires_grad:
            grad_indices.append(index)
            inputs_requiring_grad.append(_local_tensor(value))
    if not inputs_requiring_grad:
        return tuple(None for _ in inputs)
    local_outputs = tuple(_local_tensor(value) for value in outputs)
    local_grad_outputs = (
        None
        if grad_outputs is None
        else tuple(
            _local_tensor(value) if value is not None else None
            for value in grad_outputs
        )
    )
    grads = tp.autograd.grad(
        outputs=local_outputs,
        inputs=tuple(inputs_requiring_grad),
        grad_outputs=local_grad_outputs,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )
    result: list[Any] = [None] * len(inputs)
    for index, gradient in zip(grad_indices, grads, strict=True):
        result[index] = gradient
    return tuple(result)


def _register_prehook(node: Any, hook: Callable[[Any], Any]) -> _HookHandle:
    active = True

    def remove() -> None:
        nonlocal active
        active = False

    register = getattr(node, "register_prehook", None)
    if callable(register):
        def callback(grad_inputs: Any) -> Any:
            if not active:
                return grad_inputs
            return hook(grad_inputs)

        register(callback)
        return _HookHandle(remove)

    register = getattr(node, "add_pre_hook", None)
    if not callable(register):
        return _HookHandle(remove)

    def callback(grad_inputs: Any) -> list[Any]:
        if not active:
            return list(grad_inputs)
        result = hook(tuple(grad_inputs))
        if result is None:
            return list(grad_inputs)
        return list(result)

    register(callback)
    return _HookHandle(remove)


def _record_intermediate_grads(param_group: dict[str, Any], index: int) -> Callable[[Any], Any]:
    def hook(grad_inputs: Any) -> Any:
        intermediates = param_group.get("intermediates", ())
        if param_group.get("grads") is None:
            param_group["grads"] = [None] * len(intermediates)
        values = grad_inputs
        if isinstance(grad_inputs, tuple) and len(grad_inputs) == 1:
            candidate = grad_inputs[0]
            if isinstance(candidate, (tuple, list)):
                values = candidate
        param_group["grads"][index] = tuple(values)
        return grad_inputs

    return hook


def _accumulate_grad(value: Any, gradient: Any) -> None:
    if isinstance(value, DTensor):
        value = value.to_local()
    if gradient is None or not isinstance(value, tp.Tensor):
        return
    current = getattr(value, "grad", None)
    value.grad = gradient if current is None else current + gradient


def stage_backward_input(
    stage_outputs_or_loss: Iterable[Any] | Any,
    output_grads: Iterable[Any] | Any | None,
    input_values: Iterable[Any],
    weights: Iterator[Any] | Iterable[Any],
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    stage_outputs = (
        (stage_outputs_or_loss,)
        if _is_tensor(stage_outputs_or_loss)
        else tuple(stage_outputs_or_loss)
    )
    input_values = tuple(input_values)
    weight_values = tuple(weights)
    if output_grads is None:
        output_grad_values: tuple[Any, ...] | None = None
    elif _is_tensor(output_grads):
        output_grad_values = (output_grads,)
    else:
        output_grad_values = tuple(output_grads)

    valid_outputs: list[Any] = []
    valid_output_grads: list[Any] = []
    for index, stage_output in enumerate(stage_outputs):
        if not _is_tensor(stage_output):
            continue
        if not stage_output.requires_grad and getattr(stage_output, "grad_fn", None) is None:
            continue
        valid_outputs.append(stage_output)
        if output_grad_values is None:
            valid_output_grads.append(tp.ones_like(stage_output))
        else:
            valid_output_grads.append(
                output_grad_values[index] if index < len(output_grad_values) else None
            )

    output_nodes = tuple(
        node for node in (_get_grad_fn_or_grad_acc(value) for value in valid_outputs)
        if node is not None
    )
    input_nodes = tuple(
        node
        for node in (_get_grad_fn_or_grad_acc(value) for value in input_values)
        if node is not None
    )
    weight_node_by_id: dict[int, Any] = {}
    weight_nodes: list[Any] = []
    for weight in weight_values:
        node = _get_grad_fn_or_grad_acc(weight)
        if node is not None:
            weight_nodes.append(node)
            weight_node_by_id[id(node)] = weight

    reverse_edges_dict = construct_reverse_graph(output_nodes)
    param_groups = get_param_groups(input_nodes, weight_nodes, reverse_edges_dict)
    deferred_weights: set[int] = set()
    handles: list[_HookHandle] = []
    try:
        for param_group in param_groups:
            intermediates = list(param_group["intermediates"])
            param_group["intermediates"] = intermediates
            group_weights = tuple(
                weight_node_by_id[id(node)]
                for node in param_group["params"]
                if id(node) in weight_node_by_id
            )
            param_group["_deferred_outputs"] = tuple(valid_outputs)
            param_group["_deferred_output_grads"] = tuple(valid_output_grads)
            param_group["_deferred_weights"] = group_weights
            deferred_weights.update(id(weight) for weight in group_weights)
            for index, intermediate in enumerate(intermediates):
                handles.append(
                    _register_prehook(
                        intermediate,
                        _record_intermediate_grads(param_group, index),
                    )
                )

        if valid_outputs:
            dinputs = _autograd_grad_for_inputs(
                valid_outputs,
                input_values,
                valid_output_grads,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            dinputs = tuple(None for _ in input_values)
        for value, gradient in zip(input_values, dinputs, strict=True):
            _accumulate_grad(value, gradient)

        if not deferred_weights:
            for value in stage_outputs:
                if isinstance(value, DTensor):
                    value = value.to_local()
                if isinstance(value, tp.Tensor):
                    value.detach_()
        return dinputs, param_groups
    except Exception as exc:
        raise RuntimeError(
            "failed to run stage backward input: "
            f"outputs={map_debug_info(stage_outputs)}, "
            f"grads={map_debug_info(output_grads)}, "
            f"inputs={map_debug_info(input_values)}"
        ) from exc
    finally:
        for handle in handles:
            handle.remove()


def stage_backward_weight(
    weights: Iterator[Any] | Iterable[Any],
    param_groups: list[dict[str, Any]],
    retain_graph: bool = False,
) -> tuple[Any, ...]:
    weight_values = tuple(weights)
    weight_grads = tuple(getattr(weight, "grad", None) for weight in weight_values)
    groups = list(param_groups)
    for group_index, param_group in enumerate(groups):
        outputs = tuple(param_group.pop("_deferred_outputs", ()))
        output_grads = tuple(param_group.pop("_deferred_output_grads", ()))
        group_weights = tuple(param_group.pop("_deferred_weights", ()))
        if outputs and group_weights:
            keep_graph = retain_graph or group_index + 1 < len(groups)
            dweights = _autograd_grad_for_inputs(
                outputs,
                group_weights,
                output_grads,
                retain_graph=keep_graph,
                allow_unused=True,
            )
            for weight, gradient in zip(group_weights, dweights, strict=True):
                _accumulate_grad(weight, gradient)
        param_group.pop("grads", None)
        param_group.pop("intermediates", None)
    return weight_grads


def stage_backward(
    stage_output: Any,
    output_grads: Any,
    input_values: Iterable[Any],
    outputs_with_grads_idxs: Iterable[int] | None = None,
) -> tuple[Any, ...]:
    if outputs_with_grads_idxs is not None:
        indices = tuple(outputs_with_grads_idxs)
        stage_output = [stage_output[index] for index in indices]
        if output_grads is not None:
            output_grads = [output_grads[index] for index in indices]

    stage_output_tensors: list[Any] = []
    output_grad_tensors: list[Any] = []

    def extract_tensors_with_grads(output_value: Any, grad_value: Any) -> None:
        if _is_tensor(output_value):
            if not output_value.requires_grad and getattr(output_value, "grad_fn", None) is None:
                return
            if grad_value is not None and not _is_tensor(grad_value):
                raise AssertionError(
                    f"expected a tensor gradient or None, got {type(grad_value)}"
                )
            stage_output_tensors.append(_local_tensor(output_value))
            output_grad_tensors.append(
                _local_tensor(grad_value) if grad_value is not None else None
            )
            return
        if isinstance(output_value, (tuple, list)):
            if grad_value is None:
                return
            if not isinstance(grad_value, (tuple, list)):
                raise AssertionError(
                    f"gradient structure does not match {type(output_value)}"
                )
            if len(output_value) != len(grad_value):
                raise AssertionError(
                    f"gradient structure lengths differ: {len(output_value)} != {len(grad_value)}"
                )
            for output_item, grad_item in zip(output_value, grad_value, strict=True):
                extract_tensors_with_grads(output_item, grad_item)
            return
        if isinstance(output_value, dict):
            if grad_value is None:
                return
            if not isinstance(grad_value, dict) or set(output_value) != set(grad_value):
                raise AssertionError("gradient mapping does not match the output mapping")
            for key in output_value:
                extract_tensors_with_grads(output_value[key], grad_value[key])

    if output_grads is None:
        if isinstance(stage_output, tp.Tensor):
            extract_tensors_with_grads(stage_output, None)
        else:
            extract_tensors_with_grads(stage_output, None)
    else:
        extract_tensors_with_grads(stage_output, output_grads)

    try:
        if stage_output_tensors:
            tp.autograd.backward(
                stage_output_tensors,
                grad_tensors=output_grad_tensors,
            )
        grad_inputs: list[Any] = []
        for value in input_values:
            if _is_tensor(value):
                local_value = _local_tensor(value)
                grad_inputs.append(getattr(local_value, "grad", None))
                local_value.grad = None
            else:
                grad_inputs.append(None)
        return tuple(grad_inputs)
    except Exception as exc:
        raise RuntimeError(
            "failed to run stage backward: "
            f"outputs={map_debug_info(stage_output)}, "
            f"grads={map_debug_info(output_grads)}, "
            f"inputs={map_debug_info(input_values)}"
        ) from exc


def _null_coalesce_accumulate(lhs: Any, rhs: Any) -> Any:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return tp.add(lhs, rhs)
