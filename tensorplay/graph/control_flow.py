"""Symbolic data-dependent loops for compiled regions.

``while_loop`` runs a condition/body pair either eagerly (plain execution)
or as one higher-order graph node during ``tensorplay.compile`` capture.
The captured form is a single ``call_function`` node whose target carries
the condition and body as graph modules, so the enclosing region keeps ONE
specialization for every trip count instead of unrolling per iteration
count and recompiling for each.

Inside capture the condition and body are traced at dispatcher level on the
carry's execute-mode samples; the loop itself is never unrolled.  Training
through a captured loop is rejected for now -- a captured backward needs a
reversed loop definition -- while the eager path composes with autograd
naturally.

Carry values follow a flat pytree of tensors (tensor, tuple, list, dict);
the body must return the same structure with the same leaf count.
"""

from __future__ import annotations

import operator
from typing import Any, Callable, Optional

import tensorplay
from ._utils import GraphCaptureError, capturing
from .graph_module import GraphModule
from .node import Node
from .proxy import Proxy

__all__ = ["while_loop"]


def _is_tensor(value: Any) -> bool:
    return isinstance(value, tensorplay.Tensor)


def _is_leaf(value: Any) -> bool:
    return _is_tensor(value) or isinstance(value, Proxy)


def _flatten(value: Any, index: int = 0) -> tuple[list[Any], Any, int]:
    """Flatten a carry into leaves plus a picklable template for rebuild.

    Returns ``(leaves, template, next_index)``; leaves are tensors when
    rebuilding a carry from real values and proxies when flattening the
    carry under an active capture.
    """

    if _is_leaf(value):
        return [value], ("leaf", index), index + 1
    if isinstance(value, tuple):
        leaves: list[Any] = []
        children = []
        for item in value:
            sub_leaves, sub_template, index = _flatten(item, index)
            leaves.extend(sub_leaves)
            children.append(sub_template)
        return leaves, ("tuple", tuple(children)), index
    if isinstance(value, list):
        leaves = []
        children = []
        for item in value:
            sub_leaves, sub_template, index = _flatten(item, index)
            leaves.extend(sub_leaves)
            children.append(sub_template)
        return leaves, ("list", children), index
    if isinstance(value, dict):
        leaves = []
        children = []
        for key in sorted(value):
            sub_leaves, sub_template, index = _flatten(value[key], index)
            leaves.extend(sub_leaves)
            children.append((key, sub_template))
        return leaves, ("dict", tuple(children)), index
    raise GraphCaptureError(
        f"while_loop carry leaves must be tensors, got {type(value)!r}"
    )


def _leaf_count(template: Any) -> int:
    kind = template[0]
    if kind == "leaf":
        return 1
    if kind == "tuple":
        return sum(_leaf_count(child) for child in template[1])
    if kind == "list":
        return sum(_leaf_count(child) for child in template[1])
    if kind == "dict":
        return sum(_leaf_count(child) for _key, child in template[1])
    raise GraphCaptureError(f"unknown carry template {kind!r}")


def _build(template: Any, leaves: list[Any]) -> Any:
    kind = template[0]
    if kind == "leaf":
        return leaves[template[1]]
    if kind == "tuple":
        return tuple(_build(child, leaves) for child in template[1])
    if kind == "list":
        return [_build(child, leaves) for child in template[1]]
    if kind == "dict":
        return {key: _build(child, leaves) for key, child in template[1]}
    raise GraphCaptureError(f"unknown carry template {kind!r}")


def _truthy(condition: Any) -> bool:
    if _is_tensor(condition):
        return bool(condition.item())
    return bool(condition)


class WhileLoopOp:
    """Higher-order while-loop target recorded by ``while_loop`` capture.

    Holds the condition and body graph modules plus the carry template.
    Calling it runs the loop structurally on the given flattened carry --
    at capture time for the shape sample, at replay time for every call,
    so one compiled region serves every trip count.
    """

    def __init__(
        self,
        cond_graph: GraphModule,
        body_graph: GraphModule,
        template: Any,
    ) -> None:
        self.cond_graph = cond_graph
        self.body_graph = body_graph
        self.template = template
        self.__name__ = "while_loop"

    def __call__(self, *carry_leaves: Any) -> Any:
        carry = _build(self.template, list(carry_leaves))
        while _truthy(self.cond_graph(carry)):
            carry = self.body_graph(carry)
        return carry


def _trace_side(
    fn: Callable[..., Any], sample_carry: Any, side: str
) -> GraphModule:
    from .experimental._dispatch_trace import dispatch_make_graph

    try:
        graph_module = dispatch_make_graph(fn)(sample_carry)
    except Exception as exc:  # noqa: BLE001 - rewrap with loop context
        raise GraphCaptureError(
            f"while_loop {side} could not be traced: {exc}"
        ) from exc
    return graph_module


def _capture_while_loop(
    cond_fn: Callable[..., Any],
    body_fn: Callable[..., Any],
    carried_inputs: Any,
) -> Any:
    flat, template, _ = _flatten(carried_inputs)
    if not flat:
        raise GraphCaptureError("while_loop carry must contain at least one tensor")
    tracers = {id(getattr(proxy, "tracer", None)) for proxy in flat}
    if len(tracers) != 1 or not all(isinstance(proxy, Proxy) for proxy in flat):
        raise GraphCaptureError(
            "while_loop carry values must come from one active capture"
        )
    tracer = flat[0].tracer

    samples: list[Any] = []
    for proxy in flat:
        sample = proxy._sample()
        if sample is None or not _is_tensor(sample):
            raise GraphCaptureError(
                "while_loop needs execute-mode tensor samples for every "
                "carry leaf; plain tracing without execution cannot "
                "capture a loop"
            )
        samples.append(sample)
    if any(sample.requires_grad for sample in samples):
        raise GraphCaptureError(
            "while_loop capture does not support training regions yet; "
            "run the loop outside tensorplay.compile for autograd"
        )
    sample_carry = _build(template, samples)

    cond_graph = _trace_side(cond_fn, sample_carry, "condition")
    body_graph = _trace_side(body_fn, sample_carry, "body")

    loop_op = WhileLoopOp(cond_graph, body_graph, template)
    args = tuple(flat)
    out_proxy = tracer.create_proxy("call_function", loop_op, args, {})
    node = out_proxy.node
    out_sample = tracer._node_samples.get(node.name)
    if out_sample is None:
        raise GraphCaptureError(
            "while_loop sample execution produced no value; the condition "
            "or body failed on the captured carry"
        )
    out_flat, _out_template, _ = _flatten(out_sample)
    if _leaf_count(template) != len(out_flat):
        raise GraphCaptureError(
            "while_loop body must return the same carry structure: "
            f"{len(out_flat)} leaves != {_leaf_count(template)}"
        )
    if len(out_flat) == 1:
        return out_proxy
    leaves = []
    for index in range(len(out_flat)):
        leaf = tracer.create_proxy(
            "call_function", operator.getitem, (out_proxy, index), {}
        )
        leaves.append(leaf)
    return _build(template, leaves)


def while_loop(
    cond_fn: Callable[..., Any],
    body_fn: Callable[..., Any],
    carried_inputs: Any,
) -> Any:
    """Run ``body_fn`` while ``cond_fn`` holds, symbolically under capture.

    Eager execution runs the Python loop directly and composes with
    autograd.  Inside ``tensorplay.compile`` capture the loop becomes one
    graph node carrying the condition and body as subgraphs: the compiled
    region stays valid for every trip count, and the loop executes through
    its subgraphs at replay time instead of being unrolled.
    """

    if not capturing():
        carry = carried_inputs
        while _truthy(cond_fn(carry)):
            carry = body_fn(carry)
        return carry
    return _capture_while_loop(cond_fn, body_fn, carried_inputs)
