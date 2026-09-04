"""Pipeline intermediate representation and split annotations."""

import copy
import operator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import tensorplay as tp
from tensorplay.nn.modules.container import Sequential
from tensorplay.nn.modules.module import Module
from tensorplay.graph.graph import Graph
from tensorplay.graph.graph_module import GraphModule
from tensorplay.graph._utils import get_active_tracer
from tensorplay.graph.node import Node, map_arg
from tensorplay.graph.passes.split_module import split_module

from ._backward import _null_coalesce_accumulate, stage_backward
from ._unflatten import _outline_submodules
from ._utils import PipeInfo
from .stage import build_stage

__all__ = ["Pipe", "pipe_split", "SplitPoint", "pipeline"]


def get_submod_name(stage_idx: int) -> str:
    return f"submod_pp_{stage_idx}"


def _find_loss_from_output_and_spec(output_val: Any, spec_val: Any) -> Any:
    if spec_val is False:
        return None
    if spec_val is True:
        if not hasattr(output_val, "op"):
            raise RuntimeError("loss specification must select a graph value")
        return output_val
    if isinstance(output_val, dict) and isinstance(spec_val, dict):
        if set(output_val) != set(spec_val):
            raise RuntimeError("loss specification keys do not match the output")
        for key, spec in spec_val.items():
            found = _find_loss_from_output_and_spec(output_val[key], spec)
            if found is not None:
                return found
        raise RuntimeError("loss specification did not select an output value")
    if isinstance(output_val, (tuple, list)) and isinstance(spec_val, (tuple, list)):
        if len(output_val) != len(spec_val):
            raise RuntimeError("loss specification length does not match the output")
        for value, spec in zip(output_val, spec_val):
            found = _find_loss_from_output_and_spec(value, spec)
            if found is not None:
                return found
        raise RuntimeError("loss specification did not select an output value")
    raise RuntimeError("loss specification structure does not match the output")


def _find_loss_output(mod: Any, g: Any, output_loss_value_spec: Any) -> Any:
    output_nodes = [node for node in getattr(g, "nodes", ()) if getattr(node, "op", None) == "output"]
    if len(output_nodes) != 1:
        raise RuntimeError("graph must contain exactly one output node")
    output_node = output_nodes[0]
    output_value = output_node.args[0] if getattr(output_node, "args", ()) else None
    if isinstance(mod, TrivialLossWrapper):
        if len(getattr(output_node, "args", ())) != 1:
            raise RuntimeError("graph output must contain exactly one value")
        return output_value, output_node, True
    if output_loss_value_spec is None:
        if isinstance(output_value, dict) and "loss" in output_value:
            generated = {key: key == "loss" for key in output_value}
            return output_value["loss"], output_node, generated
        return None, output_node, None
    return (
        _find_loss_from_output_and_spec(output_value, output_loss_value_spec),
        output_node,
        output_loss_value_spec,
    )


def _insert_stage_symbolic_backward(g: Any, loss_node: Any, output_node: Any) -> Any:
    if loss_node is None:
        return g
    nodes = list(getattr(g, "nodes", ()))
    if not nodes or not hasattr(g, "call_function"):
        return g

    tuple_values: dict[Any, tuple[Any, ...]] = {}
    for node in reversed(nodes):
        if getattr(node, "op", None) != "call_function" or node.target is not operator.getitem:
            continue
        if len(node.args) != 2 or not isinstance(node.args[1], int):
            continue
        source, index = node.args
        previous = list(tuple_values.get(source, ()))
        if len(previous) <= index:
            previous.extend([None] * (index + 1 - len(previous)))
        previous[index] = node
        tuple_values[source] = tuple(previous)

    live_nodes: set[Any] = {loss_node}
    value_grads: dict[Any, Any] = {loss_node: None}

    def mark(value: Any) -> None:
        if hasattr(value, "op"):
            live_nodes.add(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                mark(item)
        elif isinstance(value, dict):
            for item in value.values():
                mark(item)

    def assign(node: Any, grad: Any) -> None:
        if node in value_grads and getattr(node, "op", None) != "placeholder":
            grad = g.call_function(_null_coalesce_accumulate, (value_grads[node], grad))
        value_grads[node] = grad

    with g.inserting_before(output_node):
        for node in reversed(nodes):
            if node not in live_nodes:
                continue
            mark(getattr(node, "args", ()))
            mark(getattr(node, "kwargs", {}))
            if getattr(node, "op", None) != "call_module":
                continue
            if node in tuple_values:
                stage_output = tuple(tuple_values[node])
                output_grads = tuple(value_grads.get(item) for item in stage_output)
                output_indices = [index for index, item in enumerate(stage_output) if item in live_nodes]
            else:
                stage_output = (node,)
                output_grads = (value_grads.get(node),)
                output_indices = [0]
            grad_tuple = g.call_function(
                stage_backward,
                kwargs={
                    "stage_output": stage_output,
                    "output_grads": output_grads,
                    "input_values": tuple(getattr(node, "all_input_nodes", ())),
                    "outputs_with_grads_idxs": output_indices,
                },
            )
            for index, input_node in enumerate(getattr(node, "all_input_nodes", ())):
                grad_node = g.call_function(operator.getitem, (grad_tuple, index))
                assign(input_node, grad_node)
    return g


def _move_placeholders_to_front(graph: Any) -> Any:
    nodes = list(getattr(graph, "nodes", ()))
    if not nodes:
        return graph
    placeholders = [node for node in nodes if getattr(node, "op", None) == "placeholder"]
    if not placeholders:
        return graph
    ordered = placeholders + [node for node in nodes if node not in placeholders]
    if ordered != nodes:
        try:
            graph.nodes = ordered
        except (AttributeError, TypeError):
            nodes[:] = ordered
    return graph


class PipeSequential(Sequential):
    @staticmethod
    def from_sequential(sequential_instance: Sequential) -> "PipeSequential":
        return PipeSequential(*list(sequential_instance))

    def forward(self, input: Any) -> Any:
        value = input
        for index, module in enumerate(self):
            value = module(value)
            if index + 1 < len(self):
                pipe_split()
        return value


class LossWrapper(Module):
    def __init__(self, module: Module, loss_fn: Callable[..., Any]) -> None:
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError(
            "LossWrapper.forward must be overridden to define the loss inputs"
        )


class TrivialLossWrapper(LossWrapper):
    loss_spec = True

    def forward(self, x: Any, targets: Any) -> Any:
        return self.loss_fn(self.module(x), targets)


def _pipe_split() -> None:
    tracer = get_active_tracer()
    if tracer is not None and hasattr(tracer, "create_proxy"):
        tracer.create_proxy("call_function", _pipe_split, (), {})
    return None


def pipe_split() -> None:
    return _pipe_split()


class MultiUseParameterConfig(Enum):
    TRANSMIT = auto()
    REPLICATE = auto()


class DetachExecutor:
    def __init__(self, module: Any, garbage_collect_values: bool = True) -> None:
        self.module = module
        self.garbage_collect_values = garbage_collect_values
        self.value_remap: dict[int, Any] = {}

    @staticmethod
    def _map_values(value: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(value, tuple):
            values = [DetachExecutor._map_values(item, fn) for item in value]
            if hasattr(value, "_fields"):
                return type(value)(*values)
            return tuple(values)
        if isinstance(value, list):
            return [DetachExecutor._map_values(item, fn) for item in value]
        if isinstance(value, dict):
            return {
                DetachExecutor._map_values(key, fn): DetachExecutor._map_values(item, fn)
                for key, item in value.items()
            }
        if isinstance(value, slice):
            return slice(
                DetachExecutor._map_values(value.start, fn),
                DetachExecutor._map_values(value.stop, fn),
                DetachExecutor._map_values(value.step, fn),
            )
        return fn(value)

    def _detach_tensor(self, value: Any) -> Any:
        if not isinstance(value, tp.Tensor):
            return value
        if not getattr(value, "requires_grad", False):
            return value
        key = id(value)
        if key not in self.value_remap:
            self.value_remap[key] = value.detach().requires_grad_(True)
        return self.value_remap[key]

    def run(self, *args: Any, initial_env: Any = None, **kwargs: Any) -> Any:
        self.value_remap = {}
        graph = getattr(self.module, "graph", None)
        if graph is None:
            return self.module(*args, **kwargs)

        if kwargs or getattr(self.module, "signature", None) is not None:
            signature = getattr(self.module, "signature", None)
            if signature is not None:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                bound_arguments = dict(bound.arguments)
            else:
                bound_arguments = {}
        else:
            processed = graph.process_inputs(*args)
            if isinstance(processed, (tuple, list)):
                args = tuple(processed)
            else:
                args = (processed,)
            bound_arguments = {}

        environment: dict[Any, Any] = dict(initial_env or {})
        placeholders = list(getattr(graph, "placeholders", ()))
        for index, node in enumerate(placeholders):
            parameter_name = node.target if isinstance(node.target, str) else node.name
            if parameter_name in bound_arguments:
                value = bound_arguments[parameter_name]
            elif node.name in bound_arguments:
                value = bound_arguments[node.name]
            elif index < len(args):
                value = args[index]
            elif getattr(node, "args", ()):
                value = node.args[0]
            elif parameter_name in kwargs:
                value = kwargs[parameter_name]
            elif node.name in kwargs:
                value = kwargs[node.name]
            else:
                raise TypeError(f"missing required graph input: {parameter_name}")
            environment[node] = value

        resolve = getattr(self.module, "_resolve", None)
        if not callable(resolve):
            raise TypeError("graph module does not provide argument resolution")
        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                target = self.module._resolve_target(node.target)
                node_args = resolve(node.args, environment)
                node_kwargs = resolve(node.kwargs, environment)
                environment[node] = self.call_function(target, node_args, node_kwargs)
            elif node.op == "call_method":
                node_args = resolve(node.args, environment)
                node_kwargs = resolve(node.kwargs, environment)
                receiver, *method_args = node_args
                environment[node] = getattr(receiver, node.target)(*method_args, **node_kwargs)
            elif node.op == "call_module":
                node_args = resolve(node.args, environment)
                node_kwargs = resolve(node.kwargs, environment)
                environment[node] = self.call_module(node.target, node_args, node_kwargs)
            elif node.op == "get_attr":
                environment[node] = self.module._get_attr(str(node.target))
            elif node.op == "output":
                value = resolve(node.args[0], environment)
                return graph.process_outputs(value)
            else:
                raise RuntimeError(f"unsupported graph operation: {node.op!r}")
        raise RuntimeError("graph has no output node")

    def call_module(self, target: Any, args: Any, kwargs: Any) -> Any:
        args = self._map_values(args, self._detach_tensor)
        kwargs = self._map_values(kwargs, self._detach_tensor)
        getter = getattr(self.module, "_get_attr", None)
        module = getter(str(target)) if callable(getter) else getattr(self.module, target)
        return module(*args, **kwargs)

    def call_function(self, target: Any, args: Any, kwargs: Any) -> Any:
        if target is stage_backward:
            kwargs = dict(kwargs)
            values = kwargs.get("input_values")
            if values is not None:
                kwargs["input_values"] = [
                    self.value_remap.get(id(value), value) for value in values
                ]
        return target(*args, **kwargs)


class _NodeReference:
    def __init__(self, name: str) -> None:
        self.name = name


def _resolve_node_references(value: Any, references: dict[str, Node]) -> Any:
    if isinstance(value, _NodeReference):
        return references[value.name]
    if isinstance(value, tuple):
        values = [_resolve_node_references(item, references) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*values)
        return tuple(values)
    if isinstance(value, list):
        return [_resolve_node_references(item, references) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_node_references(item, references)
            for key, item in value.items()
        }
    if isinstance(value, slice):
        return slice(
            _resolve_node_references(value.start, references),
            _resolve_node_references(value.stop, references),
            _resolve_node_references(value.step, references),
        )
    return value


class _LinearNodeList:
    def __init__(self, node_list: list[Any]) -> None:
        self.serialize_node_list: list[Node] = []
        for node in node_list:
            node_args = map_arg(node.args, lambda value: _NodeReference(value.name))
            node_kwargs = map_arg(node.kwargs, lambda value: _NodeReference(value.name))
            serialized = Node(
                None,
                node.name,
                node.op,
                node.target,
                node_args,
                node_kwargs,
                node.type,
            )
            serialized.meta = copy.copy(node.meta)
            self.serialize_node_list.append(serialized)

    def to_graph(self) -> Any:
        graph = Graph()
        references: dict[str, Node] = {}
        for node in self.serialize_node_list:
            args = _resolve_node_references(node.args, references)
            kwargs = _resolve_node_references(node.kwargs, references)
            deserialized = graph.create_node(
                op=node.op,
                target=node.target,
                args=args,
                kwargs=kwargs,
                name=node.name,
                type_expr=node.type,
            )
            deserialized.meta = copy.copy(node.meta)
            references[node.name] = deserialized
        return graph


class DummyModule(Module):
    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__()
        self.__dict__.update(body)


def _direct_serialization_deserialize(body: dict[str, Any], nodes: _LinearNodeList) -> Any:
    return GraphModule(DummyModule(body), nodes.to_graph())


def _direct_serialization_reduce(self: Any) -> tuple[Any, tuple[Any, ...]]:
    serialization_dict = dict(self.__dict__)
    serialization_dict.pop("_graph", None)
    serialization_dict.pop("graph", None)
    serialization_dict.pop("forward", None)
    serialization_dict.pop("_compiled_forward", None)
    serialization_dict.pop("_compiled_impl", None)
    serialization_dict.pop("_python_code", None)
    return (
        _direct_serialization_deserialize,
        (serialization_dict, _LinearNodeList(self.graph.nodes)),
    )


def _modify_graph_op_device(gm: Any, new_device: Any) -> None:
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if "device" in node.kwargs and node.kwargs["device"] != new_device:
                node.update_kwarg("device", new_device)
                modified = True
        elif node.op == "call_module":
            submod = gm.get_submodule(node.target)
            if hasattr(submod, "graph"):
                _modify_graph_op_device(submod, new_device)
            elif hasattr(submod, "graph_module"):
                _modify_graph_op_device(submod.graph_module, new_device)
    if modified:
        gm.recompile()


class Pipe(Module):
    def __init__(self, split_gm: Any, num_stages: int, has_loss_and_backward: bool = False, loss_spec: Any = None) -> None:
        super().__init__()
        self.split_gm = split_gm
        self.executor = DetachExecutor(split_gm)
        self.num_stages = int(num_stages)
        self.has_loss_and_backward = bool(has_loss_and_backward)
        self.loss_spec = loss_spec
        self._stages = _extract_stages(split_gm, self.num_stages)
        if len(self._stages) != self.num_stages:
            raise RuntimeError(
                f"pipeline graph contains {len(self._stages)} stages, expected {self.num_stages}"
            )
        for node in split_gm.graph.nodes:
            if not (
                node.op in {"call_module", "placeholder", "get_attr", "output"}
                or (node.op, node.target) == ("call_function", operator.getitem)
                or (node.op, node.target) == ("call_method", "backward")
                or (node.op, node.target) == ("call_function", stage_backward)
                or (node.op, node.target)
                == ("call_function", _null_coalesce_accumulate)
            ):
                raise AssertionError(f"Unexpected node: {node}")

        def named_parameters(module: Any) -> list[tuple[str, Any]]:
            result: list[tuple[str, Any]] = []
            seen_modules: set[int] = set()
            seen_parameters: set[int] = set()

            def visit(value: Any, prefix: str = "") -> None:
                if id(value) in seen_modules:
                    return
                seen_modules.add(id(value))
                graph_root = getattr(value, "root", None)
                if graph_root is not None and hasattr(value, "graph"):
                    visit(graph_root, prefix)
                    return
                for name, parameter in getattr(value, "_parameters", {}).items():
                    if parameter is None or id(parameter) in seen_parameters:
                        continue
                    seen_parameters.add(id(parameter))
                    result.append((f"{prefix}.{name}" if prefix else name, parameter))
                for name, child in getattr(value, "_modules", {}).items():
                    if child is not None:
                        child_prefix = f"{prefix}.{name}" if prefix else name
                        visit(child, child_prefix)

            visit(module)
            return result

        params_to_users: dict[int, dict[str, str]] = {}
        for module_name, module in split_gm.named_children():
            for parameter_name, parameter in named_parameters(module):
                params_to_users.setdefault(id(parameter), {})[module_name] = parameter_name
        self.replicated_params: list[dict[str, str]] = [
            users for users in params_to_users.values() if len(users) > 1
        ]
        for parameter_mapping in self.replicated_params:
            for module_name, parameter_name in parameter_mapping.items():
                module = split_gm.get_submodule(module_name)
                parts = parameter_name.split(".")
                module_root = getattr(module, "root", module)
                for part in parts[:-1]:
                    module_root = getattr(module_root, part)
                setattr(
                    module_root,
                    parts[-1],
                    copy.deepcopy(getattr(module_root, parts[-1])),
                )

        def throw(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "invoke the pipeline object directly instead of its split graph"
            )

        split_gm.forward = throw
        index = 0
        while True:
            try:
                module = getattr(split_gm, get_submod_name(index))
            except AttributeError:
                break
            module.__class__.__reduce__ = _direct_serialization_reduce
            index += 1

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.executor.run(*args, **kwargs)

    def get_stage_module(self, stage_idx: int) -> Any:
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise ValueError("stage index is outside the pipeline")
        return self._stages[stage_idx]

    @staticmethod
    def _number_and_count_forward_stages(gm: Any) -> int:
        return len(_extract_stages(gm, 0))

    @staticmethod
    def _from_traced(mod: Any, exported_program: Any, multi_use_param_spec: Any = None, output_loss_value_spec: Any = None, split_policy: Any = None) -> "Pipe":
        del multi_use_param_spec
        graph_module = exported_program.module() if hasattr(exported_program, "module") else mod
        example_inputs = getattr(exported_program, "example_inputs", None)
        record_meta = getattr(graph_module, "_interpret", None)
        if callable(record_meta) and example_inputs:
            record_meta(**dict(example_inputs), _record_meta=True)
        if split_policy is not None:
            graph_module = split_policy(graph_module)

        if not hasattr(graph_module, "graph"):
            raise TypeError("pipeline tracing must produce a graph module")

        marker_targets = {
            _pipe_split,
            pipe_split,
        }
        marker_nodes = [
            node
            for node in graph_module.graph.nodes
            if getattr(node, "op", None) == "call_function"
            and getattr(node, "target", None) in marker_targets
        ]

        stage_id = 0

        def split_callback(node: Any) -> int:
            nonlocal stage_id
            current = stage_id
            if node in marker_nodes:
                stage_id += 1
            return current

        previous_marker = False
        for node in list(graph_module.graph.nodes):
            is_marker = (
                getattr(node, "op", None) == "call_function"
                and getattr(node, "target", None) in marker_targets
            )
            if is_marker and previous_marker:
                graph_module.graph.erase_node(node)
            previous_marker = is_marker
        graph_module.recompile()

        marker_nodes = [
            node
            for node in graph_module.graph.nodes
            if getattr(node, "op", None) == "call_function"
            and getattr(node, "target", None) in marker_targets
        ]
        stage_id = 0
        split_graph = split_module(
            graph_module,
            getattr(graph_module, "root", mod),
            split_callback,
            partition_affix="pp",
        )
        for graph in _iter_graph_modules(split_graph):
            for node in list(getattr(graph, "graph", ()).nodes):
                if (
                    getattr(node, "op", None) == "call_function"
                    and getattr(node, "target", None) in marker_targets
                ):
                    graph.graph.erase_node(node)
            graph.graph.eliminate_dead_code()
            graph.recompile()
        split_graph.graph.eliminate_dead_code()
        split_graph.recompile()
        for name, submodule in list(split_graph.named_children()):
            if isinstance(submodule, GraphModule):
                _move_placeholders_to_front(submodule.graph)
                outlined = _outline_submodules(submodule.graph)
                split_graph.root.__dict__[name] = outlined
                split_graph.__dict__.setdefault("_modules", {})[name] = outlined
        split_graph.recompile()
        graph_module = split_graph

        generated_loss_spec = output_loss_value_spec
        has_loss_and_backward = False
        if output_loss_value_spec is not None:
            loss_node, output_node, generated_loss_spec = _find_loss_output(
                mod,
                graph_module.graph,
                output_loss_value_spec,
            )
            if loss_node is None:
                raise RuntimeError(
                    f"Did not find a loss value for {output_loss_value_spec!r}"
                )
            _insert_stage_symbolic_backward(
                graph_module.graph,
                loss_node,
                output_node,
            )
            graph_module.recompile()
            has_loss_and_backward = True

        stages = _extract_stages(graph_module, 0)
        if not stages:
            raise RuntimeError("pipeline graph did not produce a stage")
        return Pipe(
            graph_module,
            len(stages),
            has_loss_and_backward,
            generated_loss_spec,
        )

    def print_readable(self, print_output: bool = True) -> str:
        value = repr(self.split_gm)
        if print_output:
            print(value)
        return value

    @staticmethod
    def _trace_with_export(mod: Any, example_args: tuple[Any, ...], example_kwargs: dict[str, Any]) -> Any:
        from tensorplay.export import export

        if not callable(mod):
            raise TypeError(f"pipeline module must be callable, got {type(mod)!r}")
        try:
            return export(mod, *tuple(example_args), **dict(example_kwargs))
        except Exception as exc:
            raise RuntimeError("unable to capture the pipeline module") from exc

    @staticmethod
    def from_tracing(mod: Any, example_args: tuple[Any, ...], example_kwargs: dict[str, Any] | None = None, split_policy: Any = None) -> "Pipe":
        exported = Pipe._trace_with_export(mod, example_args, example_kwargs or {})
        return Pipe._from_traced(
            mod,
            exported,
            output_loss_value_spec=None,
            split_policy=split_policy,
        )

    def info(self) -> PipeInfo:
        return PipeInfo(self.split_gm, self.num_stages, self.has_loss_and_backward, self.loss_spec)

    def build_stage(self, stage_index: int, device: Any = None, group: Any = None) -> Any:
        stage_module = self.get_stage_module(stage_index)
        if device is not None and isinstance(stage_module, GraphModule):
            _modify_graph_op_device(stage_module, device)
        return build_stage(stage_module, stage_index, self.info(), device, group)

    def __str__(self) -> str:
        return self.split_gm.__str__()

    def __repr__(self) -> str:
        return self.split_gm.__repr__()


class SplitPoint(Enum):
    BEGINNING = auto()
    END = auto()


class PipeSplitWrapper(Module):
    SplitPoint = SplitPoint

    def __init__(self, module: Module, split_point: SplitPoint) -> None:
        super().__init__()
        self.module = module
        self.split_point = split_point

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.split_point is SplitPoint.BEGINNING:
            _pipe_split()
            return self.module(*args, **kwargs)
        if self.split_point is SplitPoint.END:
            try:
                return self.module(*args, **kwargs)
            finally:
                _pipe_split()
        raise ValueError(f"unsupported split point: {self.split_point!r}")

    def _split_before_forward(self) -> None:
        _pipe_split()

    def _split_after_forward(self) -> None:
        _pipe_split()


def _split_before_forward(self: Any) -> None:
    del self
    _pipe_split()


def _split_after_forward(self: Any) -> None:
    del self
    _pipe_split()


def annotate_split_points(mod: Any, spec: dict[str, SplitPoint]) -> Any:
    for name, point in spec.items():
        if not isinstance(point, SplitPoint):
            raise TypeError(f"split point for {name!r} must be a SplitPoint")
        parent, _, child_name = name.rpartition(".")
        owner = mod.get_submodule(parent) if parent else mod
        if not hasattr(owner, child_name):
            raise AttributeError(f"module path {name!r} does not exist")
        child = getattr(owner, child_name)
        setattr(owner, child_name, PipeSplitWrapper(child, point))
    return mod


def pipeline(module: Any, mb_args: tuple[Any, ...], mb_kwargs: dict[str, Any] | None = None, split_spec: dict[str, SplitPoint] | None = None, split_policy: Any = None) -> Pipe:
    if split_spec is not None and split_policy is not None:
        raise ValueError("split_spec and split_policy cannot be used together")
    if split_spec:
        module = annotate_split_points(module, split_spec)
    return Pipe.from_tracing(module, mb_args, mb_kwargs or {}, split_policy=split_policy)


def _iter_graph_modules(module: Any, seen: set[int] | None = None) -> list[Any]:
    seen = set() if seen is None else seen
    if id(module) in seen:
        return []
    seen.add(id(module))
    result = [module]
    graph = getattr(module, "graph", None)
    targets = {
        str(node.target)
        for node in getattr(graph, "nodes", ())
        if getattr(node, "op", None) == "call_module"
    }
    children = {name: child for name, child in getattr(module, "named_children", lambda: ())()}
    for target in targets:
        try:
            child = getattr(module, target)
        except AttributeError:
            continue
        children.setdefault(target, child)
    for child in children.values():
        if hasattr(child, "graph"):
            result.extend(_iter_graph_modules(child, seen))
    return result


def _extract_stages(module: Any, requested: int) -> list[Any]:
    graph = getattr(module, "graph", None)
    graph_stage_nodes = [
        node
        for node in getattr(graph, "nodes", ())
        if getattr(node, "op", None) == "call_module"
        and str(getattr(node, "target", "")).startswith("submod_")
    ]
    if graph_stage_nodes:
        stages = []
        seen: set[str] = set()
        for node in graph_stage_nodes:
            target = str(node.target)
            if target in seen:
                continue
            seen.add(target)
            try:
                stages.append(getattr(module, target))
            except AttributeError as exc:
                raise RuntimeError(f"stage module {target!r} is missing") from exc
        if stages:
            return stages
    if isinstance(module, PipeSequential):
        return list(module)
    if isinstance(module, Sequential):
        children = list(module)
        if not children:
            return [module]
        boundaries = [0]
        for index, child in enumerate(children):
            if isinstance(child, PipeSplitWrapper) and child.split_point is SplitPoint.BEGINNING and index > boundaries[-1]:
                boundaries.append(index)
            if isinstance(child, PipeSplitWrapper) and child.split_point is SplitPoint.END:
                boundaries.append(index + 1)
        if len(boundaries) == 1:
            return [module]
        boundaries = sorted(set(boundaries + [len(children)]))
        return [PipeSequential(*children[start:end]) for start, end in zip(boundaries, boundaries[1:]) if start < end]
    children = list(module.named_children()) if hasattr(module, "named_children") else []
    if children and all(name.startswith("submod_") for name, _ in children):
        return [child for _, child in sorted(children)]
    if children:
        marked = [
            index
            for index, (_, child) in enumerate(children)
            if isinstance(child, PipeSplitWrapper)
        ]
        if marked:
            stages: list[Any] = []
            start = 0
            boundaries = []
            for index, (_, child) in enumerate(children):
                if isinstance(child, PipeSplitWrapper) and child.split_point is SplitPoint.BEGINNING and index > start:
                    boundaries.append(index)
                if isinstance(child, PipeSplitWrapper) and child.split_point is SplitPoint.END:
                    boundaries.append(index + 1)
            for end in sorted(set(boundaries + [len(children)])):
                if start < end:
                    stages.append(PipeSequential(*(child for _, child in children[start:end])))
                    start = end
            if stages:
                return stages
    return [module] * max(1, requested or 1)
