# mypy: allow-untyped-defs
"""Convert a traced module graph into the protobuf form used by the graph plugin."""

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef

import tensorplay
from tensorplay.graph import GraphModule, symbolic_trace
from ._proto_graph import node_proto

GETATTR_KIND = "prim::GetAttr"


class NodeBase:
    def __init__(
        self,
        debugName=None,
        inputs=None,
        scope=None,
        tensor_size=None,
        op_type="UnSpecified",
        attributes="",
    ) -> None:
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        self.scope = scope

    def __repr__(self) -> str:
        repr = []
        repr.append(str(type(self)))
        repr.extend(
            m + ": " + str(getattr(self, m)) + str(type(getattr(self, m)))
            for m in dir(self)
            if "__" not in m
        )
        return "\n".join(repr) + "\n\n"


class OpNode:
    """One internal operator: one output, N inputs, an op kind and a scope."""

    def __init__(self, debugName, inputs, scope, tensor_size, kind, attributes) -> None:
        self.debugName = debugName
        self.outputs = [debugName]
        self.outputstensor_size = [tensor_size]
        self.inputs = inputs
        self.scopeName = scope
        self.tensor_size = tensor_size
        self.kind = kind
        self.attributes = attributes


class GraphPy:
    """Bookkeeping for the two-pass module-to-GraphDef conversion.

    The first pass collects input/output nodes (in ``nodes_io``) and internal
    operator nodes (in ``nodes_op``); the second resolves every node's fully
    qualified scope name from the call-site structure.
    """

    def __init__(self) -> None:
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = "default"
        self.scope_name_appeared = []

    def append(self, x) -> None:
        self.nodes_io[x.debugName] = x

    def find_common_root(self) -> None:
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split("/")[0]

    def populate_namespace_from_OP_to_IO(self) -> None:
        for node in self.nodes_op:
            for node_output, outputSize in zip(
                node.outputs, node.outputstensor_size, strict=True
            ):
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(
                    node_output,
                    node.inputs,
                    node.scopeName,
                    outputSize,
                    op_type=node.kind,
                    attributes=node.attributes,
                )

        self.find_common_root()

        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = (
                    node.scopeName + "/" + input_node_id
                )

        for key, node in self.nodes_io.items():
            if type(node) is NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
            if hasattr(node, "input_or_output"):
                self.unique_name_to_scoped_name[key] = (
                    node.input_or_output + "/" + node.debugName
                )

            if hasattr(node, "scope") and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
                if node.scope == "" and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = (
                        self.shallowest_scope_name + "/" + node.debugName
                    )

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name.get(node_input_id, node_input_id)
                for node_input_id in node.inputs
            ]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[
                    node.debugName
                ]

    def to_proto(self):
        """Convert graph representation of the GraphPy object into the
        node list expected by the graph plugin."""
        nodes = [
            node_proto(
                v.debugName,
                input=v.inputs,
                outputsize=v.tensor_size,
                op=v.kind,
                attributes=v.attributes,
            )
            for v in self.nodes_io.values()
        ]
        return nodes


def _node_output_size(node) -> list[int] | None:
    value = node.meta.get("val")
    if value is None:
        shape = node.meta.get("tensor_shape")
        return list(shape) if shape is not None else None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(d) for d in shape]
    except TypeError:
        return None


def _flatten_args(args) -> list[Any]:
    flat: list[Any] = []
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        for arg in args:
            flat.extend(_flatten_args(arg))
    elif hasattr(args, "name") and hasattr(args, "op"):
        flat.append(args)
    return flat


def _debug_name(value) -> str:
    if hasattr(value, "name") and hasattr(value, "op"):
        return str(value.name)
    return str(value)


def _kind_for(node) -> str:
    if node.op == "call_module":
        return GETATTR_KIND
    if node.op == "get_attr":
        return GETATTR_KIND
    if node.op == "call_function":
        target = getattr(node.target, "__name__", None)
        return str(target) if target else node.op
    if node.op == "call_method":
        return str(node.target)
    return node.op


def _scope_for(node) -> str:
    if node.op in ("call_module", "get_attr"):
        target = node.target if isinstance(node.target, str) else ""
        return f"__module.{target}" if target else ""
    return ""


def _attributes_for(node) -> str:
    if node.op in ("call_module", "get_attr"):
        target = node.target if isinstance(node.target, str) else ""
        return str(target).replace("'", " ")
    if node.op == "call_function":
        return str(getattr(node.target, "__name__", node.target)).replace("'", " ")
    return ""


def parse(graph_module: GraphModule, verbose=False):
    """Parse a traced module graph into node bookkeeping for the graph plugin."""
    nodes_py = GraphPy()

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            entry = NodeBase(
                node.name,
                [],
                "",
                _node_output_size(node),
                op_type="Parameter",
            )
            entry.input_or_output = "input"
            nodes_py.append(entry)

    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        nodes_py.nodes_op.append(
            OpNode(
                node.name,
                [_debug_name(arg) for arg in _flatten_args(node.args)],
                _scope_for(node),
                _node_output_size(node),
                _kind_for(node),
                _attributes_for(node),
            )
        )

    io_counter = 0
    for node in graph_module.graph.nodes:
        if node.op == "output":
            io_counter += 1
            entry = NodeBase(
                f"output.{io_counter}",
                [_debug_name(arg) for arg in _flatten_args(node.args)],
                "",
                None,
                op_type="IO Node",
            )
            entry.input_or_output = "output"
            nodes_py.append(entry)

    # Resolve module aliases so scopes read like "<ModuleName>[<attr>]".
    def module_label(module) -> str:
        return getattr(module, "original_name", None) or type(module).__name__

    base_name = module_label(graph_module)
    alias_to_name = {}
    for name, module in graph_module.named_modules(prefix="__module"):
        if name == "__module":
            continue
        alias_to_name[name] = f"{module_label(module)}[{name.split('.')[-1]}]"

    for op_node in nodes_py.nodes_op:
        module_aliases = op_node.scopeName.split("/")
        replacements = [
            alias_to_name[alias] if alias in alias_to_name else alias.split(".")[-1]
            for alias in module_aliases
        ]
        op_node.scopeName = base_name
        if any(replacements):
            op_node.scopeName += "/" + "/".join(replacements)

    nodes_py.populate_namespace_from_OP_to_IO()
    if verbose:
        print(graph_module)
    return nodes_py.to_proto()


def graph(model, args=None, verbose=False, use_strict_trace=True):
    """Process a module into the `GraphDef`/`RunMetadata` pair for the graph plugin.

    Args:
      model: The module to trace.
      args: input tensor[s] for the model (required for tracing).
      verbose: Whether to print out verbose information while processing.
      use_strict_trace: Unused; the tracer always captures the full graph.
    """
    del use_strict_trace
    if args is None:
        raise ValueError(
            "add_graph requires example inputs ('args') to trace the model."
        )
    if isinstance(args, (list, tuple)):
        trace_args = list(args)
    else:
        trace_args = [args]

    with _set_model_to_eval(model):
        graph_module = symbolic_trace(model)
        # Execute the traced module once with real inputs so shape metadata
        # ('val') is recorded on every node, placeholders included.
        placeholders = [
            node for node in graph_module.graph.nodes if node.op == "placeholder"
        ]
        for placeholder, value in zip(placeholders, trace_args):
            placeholder.meta["val"] = value
        try:
            graph_module._interpret(*trace_args, _record_meta=True)
        except TypeError:
            graph_module(*trace_args)

    list_of_nodes = parse(graph_module, verbose)
    # The device line shown in the plugin has no bearing on actual execution.
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats


@contextmanager
def _set_model_to_eval(model):
    """Context manager to temporarily set the training mode of ``model`` to eval."""
    training = getattr(model, "training", None)
    if training is None:
        try:
            yield
        finally:
            pass
        return
    model.train(False)
    try:
        yield
    finally:
        model.train(training)
