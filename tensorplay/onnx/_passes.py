"""IR passes run over the built ``ModelProto``.

Cleanup applied before a model is written out: fold everything computable at
export time, drop the ``Identity`` nodes introduced while wiring names, then
delete whatever the graph outputs no longer depend on.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from onnx import GraphProto, ModelProto, helper, numpy_helper

__all__ = ["optimize", "constant_folding", "eliminate_identity", "eliminate_dead_nodes"]


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


def _attrs(node: Any) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def _fold_reshape(inputs: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    data, shape = inputs[0], np.asarray(inputs[1], dtype=np.int64)
    if not attrs.get("allowzero", 0):
        shape = np.asarray(
            [data.shape[index] if dim == 0 else dim for index, dim in enumerate(shape)],
            dtype=np.int64,
        )
    return data.reshape(tuple(int(dim) for dim in shape))


def _fold_slice(inputs: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    data, starts, ends = inputs[0], inputs[1], inputs[2]
    axes = inputs[3] if len(inputs) > 3 else np.arange(len(starts))
    steps = inputs[4] if len(inputs) > 4 else np.ones(len(starts), dtype=np.int64)
    slices = [slice(None)] * data.ndim
    for start, end, axis, step in zip(starts, ends, axes, steps):
        slices[int(axis) % data.ndim] = slice(int(start), int(end), int(step))
    return data[tuple(slices)]


def _fold_squeeze(inputs: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    axes = inputs[1] if len(inputs) > 1 else attrs.get("axes")
    if axes is None:
        return np.squeeze(inputs[0])
    return np.squeeze(inputs[0], axis=tuple(int(axis) for axis in axes))


def _fold_unsqueeze(inputs: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    axes = inputs[1] if len(inputs) > 1 else attrs.get("axes")
    result = inputs[0]
    for axis in sorted(int(item) for item in axes):
        result = np.expand_dims(result, axis)
    return result


def _fold_cast(inputs: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    from ._type_mapping import _onnx_to_np_dtype

    return inputs[0].astype(_onnx_to_np_dtype(int(attrs["to"])))


_FOLDERS: dict[str, Callable[[list[np.ndarray], dict[str, Any]], np.ndarray]] = {
    "Identity": lambda inputs, attrs: inputs[0],
    "Transpose": lambda inputs, attrs: np.transpose(inputs[0], attrs.get("perm")),
    "Reshape": _fold_reshape,
    "Flatten": lambda inputs, attrs: inputs[0].reshape(
        int(np.prod(inputs[0].shape[: attrs.get("axis", 1)] or (1,))), -1
    ),
    "Cast": _fold_cast,
    "Concat": lambda inputs, attrs: np.concatenate(inputs, axis=attrs["axis"]),
    "Squeeze": _fold_squeeze,
    "Unsqueeze": _fold_unsqueeze,
    "Slice": _fold_slice,
    "Gather": lambda inputs, attrs: np.take(
        inputs[0], inputs[1].astype(np.int64), axis=attrs.get("axis", 0)
    ),
    "Shape": lambda inputs, attrs: np.asarray(inputs[0].shape, dtype=np.int64),
    "Size": lambda inputs, attrs: np.asarray(inputs[0].size, dtype=np.int64),
    "Add": lambda inputs, attrs: inputs[0] + inputs[1],
    "Sub": lambda inputs, attrs: inputs[0] - inputs[1],
    "Mul": lambda inputs, attrs: inputs[0] * inputs[1],
    "Div": lambda inputs, attrs: inputs[0] / inputs[1],
    "Pow": lambda inputs, attrs: np.power(inputs[0], inputs[1]),
    "Neg": lambda inputs, attrs: -inputs[0],
    "Abs": lambda inputs, attrs: np.abs(inputs[0]),
    "Sqrt": lambda inputs, attrs: np.sqrt(inputs[0]),
    "Exp": lambda inputs, attrs: np.exp(inputs[0]),
    "Log": lambda inputs, attrs: np.log(inputs[0]),
    "Reciprocal": lambda inputs, attrs: np.reciprocal(inputs[0]),
    "Floor": lambda inputs, attrs: np.floor(inputs[0]),
    "Ceil": lambda inputs, attrs: np.ceil(inputs[0]),
    "Range": lambda inputs, attrs: np.arange(
        inputs[0].item(), inputs[1].item(), inputs[2].item()
    ),
}


def constant_folding(graph: GraphProto) -> int:
    """Evaluate nodes whose inputs are all constants; returns the fold count."""

    constants: dict[str, np.ndarray] = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    graph_inputs = {value.name for value in graph.input}
    graph_outputs = {value.name for value in graph.output}

    kept: list[Any] = []
    folded = 0
    for node in graph.node:
        if node.op_type == "Constant" and not node.input:
            attrs = _attrs(node)
            value = attrs.get("value")
            if value is not None:
                array = numpy_helper.to_array(value)
                constants[node.output[0]] = array
                graph.initializer.append(numpy_helper.from_array(array, node.output[0]))
                folded += 1
                continue
        foldable = (
            node.op_type in _FOLDERS
            and len(node.output) == 1
            and node.output[0] not in graph_outputs
            and all(
                name in constants and name not in graph_inputs for name in node.input
            )
            and node.input
        )
        if not foldable:
            kept.append(node)
            continue
        try:
            result = _FOLDERS[node.op_type](
                [constants[name] for name in node.input], _attrs(node)
            )
        except Exception:  # noqa: BLE001 - folding is best effort
            kept.append(node)
            continue
        array = np.asarray(result)
        constants[node.output[0]] = array
        graph.initializer.append(numpy_helper.from_array(array, node.output[0]))
        folded += 1

    del graph.node[:]
    graph.node.extend(kept)
    return folded


# ---------------------------------------------------------------------------
# Identity elimination
# ---------------------------------------------------------------------------


def eliminate_identity(graph: GraphProto) -> int:
    """Remove ``Identity`` nodes, rewiring their consumers."""

    graph_inputs = {value.name for value in graph.input}
    graph_outputs = {value.name for value in graph.output}
    initializers = {initializer.name for initializer in graph.initializer}
    removed = 0

    while True:
        producers: dict[str, Any] = {}
        use_count: dict[str, int] = {}
        for node in graph.node:
            for name in node.input:
                use_count[name] = use_count.get(name, 0) + 1
            for name in node.output:
                producers[name] = node
        for name in graph_outputs:
            use_count[name] = use_count.get(name, 0) + 1

        target = None
        for node in graph.node:
            if node.op_type != "Identity":
                continue
            source, sink = node.input[0], node.output[0]
            if sink not in graph_outputs:
                target = (node, sink, source)
                break
            producer = producers.get(source)
            if (
                producer is not None
                and source not in graph_outputs
                and source not in graph_inputs
                and source not in initializers
                and use_count.get(source, 0) == 1
            ):
                # Move the graph-output name onto the producing node instead.
                target = (node, source, sink)
                break
        if target is None:
            return removed

        node, replaced, replacement = target
        graph.node.remove(node)
        _rename_value(graph, replaced, replacement)
        removed += 1


def _rename_value(graph: GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new
    for collection in (graph.input, graph.output, graph.value_info):
        for value in collection:
            if value.name == old:
                value.name = new
    for initializer in graph.initializer:
        if initializer.name == old:
            initializer.name = new


# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------


def eliminate_dead_nodes(graph: GraphProto) -> int:
    """Drop nodes, initializers and value_info the outputs do not depend on."""

    producers: dict[str, Any] = {}
    for node in graph.node:
        for name in node.output:
            producers[name] = node

    live_nodes: set[int] = set()
    live_values: set[str] = {value.name for value in graph.output}
    pending = list(live_values)
    while pending:
        name = pending.pop()
        node = producers.get(name)
        if node is None or id(node) in live_nodes:
            continue
        live_nodes.add(id(node))
        for source in node.input:
            if source and source not in live_values:
                live_values.add(source)
                pending.append(source)

    kept = [node for node in graph.node if id(node) in live_nodes]
    removed = len(graph.node) - len(kept)
    del graph.node[:]
    graph.node.extend(kept)

    used = {name for node in graph.node for name in node.input if name}
    used |= {value.name for value in graph.output}
    initializers = [item for item in graph.initializer if item.name in used]
    removed += len(graph.initializer) - len(initializers)
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    produced = {name for node in graph.node for name in node.output}
    value_info = [item for item in graph.value_info if item.name in produced]
    del graph.value_info[:]
    graph.value_info.extend(value_info)
    return removed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def optimize(model: ModelProto, *, do_constant_folding: bool = True) -> ModelProto:
    """Run the standard cleanup pipeline over ``model`` in place."""

    graph = model.graph
    eliminate_identity(graph)
    if do_constant_folding:
        # Nodes are visited in topological order, so one pass already folds
        # chains of constants; the second identity sweep only cleans up what
        # folding exposed.
        constant_folding(graph)
        eliminate_identity(graph)
    eliminate_dead_nodes(graph)
    return model
