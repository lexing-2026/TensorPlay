"""Graph transformation for tensor-parallel exported programs."""

from __future__ import annotations

import copy
import operator
from collections.abc import Sequence
from typing import Any, cast

import tensorplay

from ....export.exported_program import ExportedProgram
from ....export.graph_signature import ExportGraphSignature
from ....graph.graph_module import GraphModule
from ....graph.node import Node
from ....graph.passes.infra.pass_base import PassBase, PassResult
from ....graph.passes.shape_prop import ShapeProp
from ....utils import _pytree as pytree
from ...device_mesh import DeviceMesh
from .._api import DTensor, distribute_tensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OpSpec, OutputSharding
from .._redistribute import redistribute_local_tensor
from ..parallel.style import ColwiseParallel, ParallelStyle
from ..placement_types import Partial, Placement, Replicate, Shard


__all__ = ["tensor_parallel_transformation"]


def tensor_parallel_transformation(
    exported_program: ExportedProgram,
    rank: int,
    world_size: int,
    device_type: str,
    parallel_strategies: dict[str, ParallelStyle],
) -> ExportedProgram:
    """Transform a single-device exported program into a local-rank program."""

    del rank
    graph_module = exported_program.graph_module
    graph_signature = copy.deepcopy(exported_program.graph_signature)
    state_dict = _copy_state_dict(exported_program)

    example_inputs = _example_inputs(exported_program)
    if any(node.meta.get("val") is None for node in graph_module.graph.nodes):
        ShapeProp(graph_module).propagate(*example_inputs)

    with graph_module._set_replace_hook(graph_signature.get_replace_hook()):
        result = _TensorParallelTransformPass(
            world_size,
            device_type,
            state_dict,
            exported_program.graph_signature,
            parallel_strategies,
        )(graph_module)
        if result is None:
            raise AssertionError
        graph_module = result.graph_module

    return _update_exported_program(
        exported_program,
        graph_module,
        graph_signature,
        state_dict,
    )


def _copy_state_dict(exported_program: ExportedProgram) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind.name not in {"PARAMETER", "BUFFER"}:
            continue
        if not isinstance(spec.target, str):
            continue
        state_dict[spec.target] = exported_program._resolve_state_value(spec.target)
    return state_dict


def _example_inputs(exported_program: ExportedProgram) -> tuple[Any, ...]:
    values: list[Any] = []
    constants = exported_program.constants
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind.name in {"PARAMETER", "BUFFER"}:
            if not isinstance(spec.target, str):
                raise ValueError("state input is missing its target")
            values.append(exported_program._resolve_state_value(spec.target))
            continue
        if spec.kind.name == "CONSTANT_TENSOR":
            if not isinstance(spec.target, str) or spec.target not in constants:
                raise ValueError("constant input is missing its value")
            values.append(constants[spec.target])
            continue
        name = getattr(spec.arg, "name", None)
        if name not in exported_program.example_inputs:
            raise ValueError(f"example input {name!r} is unavailable")
        values.append(exported_program.example_inputs[name])
    return tuple(values)


def _set_state_value(root: Any, target: str, value: Any) -> None:
    parent = root
    parts = target.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _update_exported_program(
    exported_program: ExportedProgram,
    graph_module: GraphModule,
    graph_signature: ExportGraphSignature,
    state_dict: dict[str, Any],
) -> ExportedProgram:
    result = copy.copy(exported_program)
    result.graph_module = graph_module
    result.graph_signature = graph_signature
    result._bindings = None
    result._unlifted = None
    for target, value in state_dict.items():
        _set_state_value(graph_module.root, target, value)
    return result


class _TensorParallelTransformPass(PassBase):
    """Mark, partition, and materialize a tensor-parallel graph."""

    def __init__(
        self,
        world_size: int,
        device_type: str,
        state_dict: dict[str, Any],
        graph_signature: ExportGraphSignature,
        parallel_strategies: dict[str, ParallelStyle],
    ) -> None:
        super().__init__()
        self.mesh = DeviceMesh(device_type, tensorplay.arange(world_size))
        self.state_dict = state_dict
        self.graph_signature = graph_signature
        self.parallel_strategies = parallel_strategies

    def call(self, graph_module: GraphModule) -> PassResult:
        graph_module = copy.deepcopy(graph_module)
        parameter_placements = _generate_parameter_and_buffer_placements(
            list(self.state_dict.keys()), self.parallel_strategies
        )
        placement_strategies = _mark_sharding(
            graph_module,
            self.graph_signature,
            self.mesh,
            parameter_placements,
        )
        _partitioner(graph_module)
        _shard_state_dict(
            self.state_dict,
            placement_strategies,
            self.graph_signature,
            self.mesh,
        )
        return PassResult(graph_module, True)


def _generate_parameter_and_buffer_placements(
    params_and_buffers: list[str],
    parallel_strategies: dict[str, ParallelStyle],
) -> dict[str, Placement]:
    """Build placements for parameters covered by linear layer styles."""

    parameter_placements: dict[str, Placement] = {}
    for linear_fqn, parallel_style in parallel_strategies.items():
        weight_fqn = f"{linear_fqn}.weight"
        bias_fqn = f"{linear_fqn}.bias"
        if weight_fqn not in params_and_buffers:
            raise AssertionError
        parameter_placements[weight_fqn] = (
            Shard(0) if parallel_style == ColwiseParallel else Shard(1)
        )
        if bias_fqn in params_and_buffers:
            parameter_placements[bias_fqn] = (
                Shard(0) if parallel_style == ColwiseParallel else Replicate()
            )
    return parameter_placements


def _mark_tensor_parallel_shardings(
    graph_module: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
    parameter_placements: dict[str, Placement],
) -> dict[Node, OpSpec]:
    """Mark state and user placeholders with initial placements."""

    placement_strategies: dict[Node, OpSpec] = {}
    parameter_inputs = graph_signature.inputs_to_parameters
    buffer_inputs = graph_signature.inputs_to_buffers
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        fqn = parameter_inputs.get(node.name, buffer_inputs.get(node.name))
        placement = parameter_placements.get(fqn, Replicate())
        placement_strategies[node] = _create_placement_strategy(
            node,
            mesh,
            placements=(placement,),
        )
    return placement_strategies


def _get_input_node_fqn(input_name: str, graph_signature: ExportGraphSignature) -> str:
    """Return the state target associated with an input node."""

    if input_name in graph_signature.inputs_to_parameters:
        return graph_signature.inputs_to_parameters[input_name]
    if input_name in graph_signature.inputs_to_buffers:
        return graph_signature.inputs_to_buffers[input_name]
    raise ValueError(f"{input_name} is not a state input")


def _operation_is_registered(propagator: Any, operation: Any) -> bool:
    return any(
        propagator._operation_value(table, operation) is not None
        for table in (
            propagator.op_strategy_funcs,
            propagator.op_to_rules,
            propagator.op_single_dim_strategy_funcs,
        )
    ) or propagator._global_rule(operation)[1] is not None


def _adjust_linear_partial_input(
    op_schema: OpSchema,
    output_sharding: OutputSharding,
) -> OutputSharding:
    if _operation_name(op_schema.op) != "linear":
        return output_sharding
    if len(op_schema.args_schema) < 2:
        return output_sharding
    input_spec = op_schema.args_schema[0]
    weight_spec = op_schema.args_schema[1]
    if not isinstance(input_spec, DTensorSpec) or not isinstance(weight_spec, DTensorSpec):
        return output_sharding

    from .._ops._matrix_ops import linear_single_dim_strategy

    expected_args = list(op_schema.args_schema)
    output_spec = output_sharding.output_spec
    changed = False
    if any(placement.is_partial() for placement in input_spec.placements) and any(
        isinstance(placement, Shard) and placement.dim == 0
        for placement in weight_spec.placements
    ):
        replicated_input = DTensorSpec(
            input_spec.mesh,
            tuple(Replicate() for _ in input_spec.placements),
            tensor_meta=input_spec.tensor_meta,
        )
        bias = op_schema.args_schema[2] if len(op_schema.args_schema) > 2 else None
        output_spec = linear_single_dim_strategy(
            replicated_input,
            weight_spec,
            bias if isinstance(bias, DTensorSpec) else None,
        )
        expected_args[0] = replicated_input
        changed = True
    if (
        isinstance(output_spec, DTensorSpec)
        and len(expected_args) > 2
        and isinstance(expected_args[2], DTensorSpec)
        and any(isinstance(placement, Partial) for placement in output_spec.placements)
        and all(isinstance(placement, Replicate) for placement in expected_args[2].placements)
    ):
        expected_args[2] = DTensorSpec(
            expected_args[2].mesh,
            tuple(output_spec.placements),
            tensor_meta=expected_args[2].tensor_meta,
        )
        changed = True
    if not changed:
        return output_sharding
    return OutputSharding(
        output_spec=output_spec,
        redistribute_schema=OpSchema(
            op_schema.op,
            tuple(expected_args),
            op_schema.kwargs_schema,
            schema_info=op_schema.schema_info,
        ),
        needs_redistribute=True,
    )


def _operation_name(operation: Any) -> str:
    value = getattr(operation, "__name__", getattr(operation, "name", operation))
    return str(value).rsplit(".", 1)[-1].removesuffix("_default")


def _mark_sharding(
    graph_module: GraphModule,
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
    parameter_placements: dict[str, Placement],
) -> dict[Node, OpSpec]:
    """Propagate placements through every supported graph operation."""

    placement_strategies = _mark_tensor_parallel_shardings(
        graph_module,
        graph_signature,
        mesh,
        parameter_placements,
    )
    propagator = DTensor._op_dispatcher.sharding_propagator

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            node.meta["sharding"] = placement_strategies[node]
        elif node.op == "call_function":
            if node.target is operator.getitem:
                input_nodes = node.all_input_nodes
                if len(input_nodes) != 1:
                    raise AssertionError(
                        f"getitem expects one input, found {len(input_nodes)}"
                    )
                input_strategy = placement_strategies[input_nodes[0]]
                placement_strategies[node] = _create_placement_strategy(
                    node,
                    mesh,
                    placements=input_strategy.output_spec.placements,
                    input_specs=_get_input_node_specs(node, placement_strategies),
                )
                node.meta["sharding"] = placement_strategies[node]
                continue

            op_schema = _get_op_schema(node, placement_strategies)
            if _operation_is_registered(propagator, op_schema.op):
                output_sharding = propagator.propagate_op_sharding(op_schema)
            else:
                output_sharding = _generate_default_output_sharding(
                    node,
                    mesh,
                    op_schema,
                )
            if output_sharding is None:
                raise RuntimeError(f"no placement result for {node.target!r}")
            output_sharding = _adjust_linear_partial_input(
                op_schema,
                output_sharding,
            )
            placement_strategies[node] = OpSpec(
                output_specs=_get_output_spec_from_output_sharding(output_sharding),
                input_specs=(
                    output_sharding.redistribute_schema.args_spec
                    if output_sharding.redistribute_schema is not None
                    else _get_input_node_specs(node, placement_strategies)
                ),
            )
            node.meta["sharding"] = placement_strategies[node]
        elif node.op == "output":
            node.meta["sharding"] = None
        else:
            raise RuntimeError(f"op code {node.op} not supported")
    return placement_strategies


def _get_output_spec_from_output_sharding(
    output_sharding: OutputSharding,
) -> DTensorSpec:
    """Extract the first distributed output specification."""

    if isinstance(output_sharding.output_spec, DTensorSpec):
        return output_sharding.output_spec
    if not isinstance(output_sharding.output_spec, Sequence):
        raise AssertionError
    if not output_sharding.output_spec or output_sharding.output_spec[0] is None:
        raise AssertionError
    output_sharding.output_spec[0].tensor_meta = None
    return output_sharding.output_spec[0]


def _create_placement_strategy(
    node: Node,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    input_specs: Sequence[DTensorSpec] | None = None,
) -> OpSpec:
    """Create a placement specification and attach graph tensor metadata."""

    placement = OpSpec(
        input_specs=input_specs,
        output_specs=DTensorSpec(mesh=mesh, placements=placements),
    )
    _populate_tensor_meta(node, placement.output_specs)
    return placement


def _tensor_meta(value: Any) -> TensorMeta:
    stride = value.stride() if callable(getattr(value, "stride", None)) else value.stride
    return TensorMeta(
        shape=tuple(value.shape),
        stride=tuple(stride),
        dtype=value.dtype,
    )


def _populate_tensor_meta(node: Node, output_spec: Any) -> None:
    """Attach shape, stride, and dtype metadata to an output specification."""

    value = node.meta.get("val")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not isinstance(output_spec, Sequence):
            raise AssertionError
        for spec, tensor_value in zip(output_spec, value):
            if spec is None:
                raise AssertionError
            spec.tensor_meta = _tensor_meta(tensor_value)
        return
    if not isinstance(output_spec, DTensorSpec):
        raise AssertionError
    output_spec.tensor_meta = _tensor_meta(value)


def _generate_default_output_sharding(
    node: Node,
    mesh: DeviceMesh,
    op_schema: OpSchema,
) -> OutputSharding:
    """Use replicated layouts for an operation without a registered rule."""

    def update_arg_spec(arg_spec: DTensorSpec) -> DTensorSpec:
        return DTensorSpec(
            mesh=arg_spec.mesh,
            placements=(Replicate(),),
            tensor_meta=arg_spec.tensor_meta,
        )

    new_op_schema = OpSchema(
        op=op_schema.op,
        args_schema=pytree.tree_map_only(
            DTensorSpec,
            update_arg_spec,
            op_schema.args_schema,
        ),
        kwargs_schema=op_schema.kwargs_schema,
        schema_info=op_schema.schema_info,
    )

    def create_output_spec(value: Any) -> DTensorSpec:
        return DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=_tensor_meta(value),
        )

    return OutputSharding(
        output_spec=pytree.tree_map_only(
            tensorplay.Tensor,
            create_output_spec,
            node.meta["val"],
        ),
        redistribute_schema=new_op_schema,
        needs_redistribute=True,
    )


def _partitioner(graph_module: GraphModule) -> GraphModule:
    """Convert full graph values to local values and add redistributions."""

    for node in list(graph_module.graph.nodes):
        node_sharding = node.meta["sharding"]
        if node.op == "placeholder":
            node.meta["val"] = _partition_val(
                node.meta["val"],
                node_sharding.output_spec,
            )
        elif node.op == "call_function":
            output_spec = node_sharding.output_spec
            expected_input_specs = node_sharding.input_specs
            for index, input_arg in enumerate(node.all_input_nodes):
                input_spec = input_arg.meta["sharding"].output_spec
                desired_spec = (
                    output_spec
                    if expected_input_specs is None
                    else expected_input_specs[index]
                )
                if input_spec != desired_spec:
                    _insert_reshard_gm(
                        graph_module,
                        node,
                        input_arg,
                        input_spec,
                        desired_spec,
                    )
            node.meta["val"] = _partition_val(
                node.meta["val"],
                output_spec,
            )
        elif node.op == "output":
            for input_arg in node.all_input_nodes:
                input_spec = input_arg.meta["sharding"].output_spec
                desired_spec = copy.copy(input_spec)
                desired_spec.placements = (Replicate(),)
                if input_spec != desired_spec:
                    _insert_reshard_gm(
                        graph_module,
                        node,
                        input_arg,
                        input_spec,
                        desired_spec,
                    )
        else:
            raise RuntimeError(f"op code {node.op} not supported")

    _clean_up_graph_metadata(graph_module)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _partition_val(value: Any, spec: DTensorSpec) -> Any:
    """Convert a full tensor value to the local component described by a spec."""

    if isinstance(value, tensorplay.Tensor):
        local_value = value
        if int(value.dim()) == 0:
            return local_value
        for mesh_dim, placement in enumerate(spec.placements):
            if placement.is_shard():
                coordinate = spec.mesh.get_coordinate()
                if coordinate is None:
                    raise AssertionError("current rank is not in the mesh")
                local_value = placement._select_split_tensor(
                    local_value,
                    int(spec.mesh.size(mesh_dim=mesh_dim)),
                    coordinate[mesh_dim],
                    with_padding=False,
                    contiguous=True,
                    clone=False,
                )
        return local_value
    if isinstance(value, (list, tuple)):
        return type(value)(_partition_val(item, spec) for item in value)
    raise RuntimeError(f"value type {type(value)} is not supported")


def _reshard_local_tensor(
    local_tensor: Any,
    input_spec: DTensorSpec,
    desired_spec: DTensorSpec,
) -> Any:
    return redistribute_local_tensor(local_tensor, input_spec, desired_spec)


def _insert_reshard_gm(
    graph_module: GraphModule,
    node: Node,
    input_arg: Node,
    input_arg_spec: DTensorSpec,
    desired_spec: DTensorSpec,
) -> None:
    """Insert one local redistribution operation before a graph node."""

    metadata = input_arg_spec.tensor_meta or desired_spec.tensor_meta
    if metadata is None:
        value = input_arg.meta.get("val")
        if not isinstance(value, tensorplay.Tensor):
            raise ValueError("redistribution requires tensor metadata")
        metadata = _tensor_meta(value)
    input_arg_spec.tensor_meta = metadata
    desired_spec.tensor_meta = metadata
    with graph_module.graph.inserting_before(node):
        reshard_node = graph_module.graph.call_function(
            _reshard_local_tensor,
            (input_arg, input_arg_spec, desired_spec),
        )
    node.replace_input_with(input_arg, reshard_node)


def _clean_up_graph_metadata(graph_module: GraphModule) -> None:
    """Remove temporary layout metadata and refresh local tensor metadata."""

    from ....graph.passes.shape_prop import _extract_tensor_metadata

    for node in graph_module.graph.nodes:
        node.meta.pop("sharding", None)
        value = node.meta.get("val")
        if isinstance(value, tensorplay.Tensor):
            node.meta["tensor_meta"] = _extract_tensor_metadata(value)


def _get_input_node_specs(
    node: Node,
    placement_strategies: dict[Node, OpSpec],
) -> tuple[DTensorSpec, ...]:
    """Get the output specifications of all graph inputs."""

    input_specs: list[DTensorSpec] = []
    for input_arg in node.all_input_nodes:
        if input_arg not in placement_strategies:
            raise ValueError(f"{input_arg} does not have an output specification")
        output_spec = placement_strategies[input_arg].output_specs
        if not isinstance(output_spec, DTensorSpec):
            raise AssertionError
        input_specs.append(output_spec)
    return tuple(input_specs)


def _get_op_schema(
    node: Node,
    placement_strategies: dict[Node, OpSpec],
) -> OpSchema:
    """Build an operation schema from the layouts already assigned to inputs."""

    args_schema = pytree.tree_map_only(
        Node,
        lambda value: placement_strategies[value].output_specs,
        node.args,
    )
    return OpSchema(
        op=cast(Any, node.target),
        args_schema=tuple(args_schema),
        kwargs_schema=cast(dict[str, Any], node.kwargs),
    )


def _shard_state_dict(
    state_dict: dict[str, Any],
    placement_strategies: dict[Node, OpSpec],
    graph_signature: ExportGraphSignature,
    mesh: DeviceMesh,
) -> None:
    """Partition lifted parameters and buffers using their input layouts."""

    from ....nn.parameter import Parameter

    for node, op_spec in placement_strategies.items():
        if node.op != "placeholder":
            continue
        if node.name in graph_signature.inputs_to_parameters:
            fqn = graph_signature.inputs_to_parameters[node.name]
        elif node.name in graph_signature.inputs_to_buffers:
            fqn = graph_signature.inputs_to_buffers[node.name]
        else:
            continue
        if fqn not in state_dict:
            raise AssertionError(f"{fqn} is absent from the state dictionary")
        original = state_dict[fqn]
        local = distribute_tensor(
            original,
            mesh,
            op_spec.output_spec.placements,
        ).to_local()
        state_dict[fqn] = (
            Parameter(
                local,
                requires_grad=bool(getattr(original, "requires_grad", False)),
            )
            if isinstance(original, Parameter)
            else local
        )
