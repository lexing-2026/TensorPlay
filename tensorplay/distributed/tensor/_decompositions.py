"""Runtime decomposition tracing for distributed tensor propagation."""

from __future__ import annotations

import inspect
import itertools
import operator
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

import tensorplay

from ._dtensor_spec import DTensorSpec, TensorMeta
from ._op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    RuntimeSchemaInfo,
)
from .placement_types import Placement, Replicate, Shard, _StridedShard

if TYPE_CHECKING:
    from ._sharding_prop import ShardingPropagator


__all__ = [
    "DecompShardingStrategy",
    "PlacementTrackingMode",
    "PlacementTrackingTensor",
    "decomposition_table",
    "local_map",
    "register_decomposition",
    "redistribute",
    "to_local",
]


decomposition_table: dict[Any, Callable[..., Any]] = {}


_GRAPH_OPERATOR_NAMES = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "div",
    operator.pow: "pow",
    operator.neg: "neg",
    operator.eq: "eq",
    operator.ne: "ne",
    operator.lt: "lt",
    operator.le: "le",
    operator.gt: "gt",
    operator.ge: "ge",
}


def _dtensor_type() -> type[Any]:
    from ._api import DTensor

    return DTensor


def to_local(value: Any) -> Any:
    return value.to_local() if isinstance(value, _dtensor_type()) else value


def redistribute(value: Any, device_mesh: Any, placements: Any) -> Any:
    return value.redistribute(device_mesh, placements)


def local_map(function: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(value, _dtensor_type()):
        result = function(value.to_local(), *args, **kwargs)
        return _dtensor_type()(result, value.device_mesh, value.placements, shape=value.shape)
    return function(value, *args, **kwargs)


_TENSOR_PARAMETER_NAMES = frozenset(
    {
        "a",
        "b",
        "condition",
        "grad",
        "grad_output",
        "input",
        "input1",
        "input2",
        "mat1",
        "mat2",
        "other",
        "self",
        "tensor",
        "tensor1",
        "tensor2",
        "value",
        "weight",
        "x",
        "y",
    }
)

_STATIC_PARAMETER_NAMES = frozenset(
    {
        "alpha",
        "axis",
        "axes",
        "beta",
        "correction",
        "dim",
        "dims",
        "dilation",
        "end",
        "groups",
        "keepdim",
        "k",
        "mode",
        "padding",
        "padding_idx",
        "rounding_mode",
        "scale",
        "size",
        "split_size",
        "start",
        "step",
        "stride",
        "value",
    }
)


def register_decomposition(operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a callable used to expand one operation during propagation."""

    def register(function: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(function):
            raise TypeError(f"decomposition for {operation!r} must be callable")
        decomposition_table[operation] = function
        return function

    return register


def _operation_name(operation: Any) -> str:
    if isinstance(operation, str):
        value = operation
    else:
        value = getattr(
            operation,
            "__name__",
            getattr(operation, "name", type(operation).__name__),
        )
    value = str(value).rsplit(".", 1)[-1]
    for suffix in ("_default", "_out", "_functional"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def _safe_table_get(table: Mapping[Any, Any], operation: Any) -> Any:
    try:
        value = table.get(operation)
    except (TypeError, AttributeError):
        value = None
    if value is not None:
        return value
    return table.get(_operation_name(operation))


def _tree_map(value: Any, function: Callable[[Any], Any]) -> Any:
    if isinstance(value, dict):
        return {key: _tree_map(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(item, function) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, function) for item in value]
    return function(value)


def _tree_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        result: list[Any] = []
        for item in value.values():
            result.extend(_tree_values(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_tree_values(item))
        return result
    return [value]


def _tree_any(value: Any, predicate: Callable[[Any], bool]) -> bool:
    return any(predicate(item) for item in _tree_values(value))


def _spec_leaves(value: Any) -> list[DTensorSpec]:
    return [item for item in _tree_values(value) if isinstance(item, DTensorSpec)]


def _extract_input_specs(op_schema: OpSchema) -> tuple[Any, ...]:
    return op_schema.args_schema + tuple(op_schema.kwargs_schema.values())


def _infer_schema_info_from_op(operation: Any) -> RuntimeSchemaInfo:
    try:
        parameters = tuple(inspect.signature(operation).parameters.values())
    except (TypeError, ValueError):
        return RuntimeSchemaInfo(needs_pytree=True)

    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )
    static_argnum = 100
    for index, parameter in enumerate(positional):
        name = parameter.name.lower()
        if name in _STATIC_PARAMETER_NAMES and name not in _TENSOR_PARAMETER_NAMES:
            static_argnum = index
            break

    static_kwargkey = [
        parameter.name
        for parameter in parameters
        if parameter.kind == parameter.KEYWORD_ONLY
        and parameter.name.lower() in _STATIC_PARAMETER_NAMES
        and parameter.name.lower() not in _TENSOR_PARAMETER_NAMES
    ]
    return RuntimeSchemaInfo(
        static_argnum=static_argnum,
        static_kwargkey=static_kwargkey or None,
        needs_pytree=True,
    )


def _tensor_meta(value: Any) -> TensorMeta | None:
    shape = getattr(value, "shape", None)
    stride = getattr(value, "stride", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or stride is None or dtype is None:
        return None
    try:
        stride = stride() if callable(stride) else stride
        return TensorMeta(
            tuple(int(item) for item in shape),
            tuple(int(item) for item in stride),
            dtype,
        )
    except (TypeError, ValueError):
        return None


def _is_tensor(value: Any) -> bool:
    return isinstance(value, tensorplay.Tensor)


def _normalize_call_arguments(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not args and "self" in kwargs:
        try:
            parameters = tuple(inspect.signature(function).parameters.values())
        except (TypeError, ValueError):
            parameters = ()
        if parameters and parameters[0].name != "self":
            kwargs = dict(kwargs)
            args = (kwargs.pop("self"),)
    return args, kwargs


def _call_with_plain_values(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    args, kwargs = _normalize_call_arguments(function, args, kwargs)
    return function(*args, **kwargs)


class PlacementTrackingTensor(tensorplay.Tensor):
    """TensorBase wrapper carrying one placement specification."""

    def __init__(
        self,
        value: Any = None,
        spec: DTensorSpec | None = None,
        mode: "PlacementTrackingMode | None" = None,
    ) -> None:
        if value is None:
            super().__init__()
        else:
            super().__init__(value)
        if spec is not None:
            self._spec = spec
        if mode is not None:
            self._placement_mode = mode

    @classmethod
    def from_tensor(
        cls,
        value: Any,
        spec: DTensorSpec,
        mode: "PlacementTrackingMode",
    ) -> "PlacementTrackingTensor":
        if isinstance(value, cls):
            result = value
        else:
            try:
                result = value.as_subclass(cls)
            except AttributeError:
                result = tensorplay._C._make_subclass(cls, value, False)
        result._spec = spec
        result._placement_mode = mode
        return result

    @classmethod
    def __tensorplay_dispatch__(
        cls,
        function: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        tracking = next(
            (
                value
                for value in _tree_values((args, kwargs or {}))
                if isinstance(value, cls)
            ),
            None,
        )
        if tracking is None:
            return NotImplemented
        mode = getattr(tracking, "_placement_mode", None)
        if mode is None:
            return NotImplemented
        return mode.dispatch(function, args, kwargs or {})


class PlacementTrackingMode:
    """Propagate layouts through one decomposition invocation."""

    def __init__(self, sharding_prop: "ShardingPropagator", mesh: Any) -> None:
        self.sharding_prop = sharding_prop
        self.mesh = mesh
        self._current_schema: OpSchema | None = None

    def _to_plain(self, value: Any) -> Any:
        if not isinstance(value, PlacementTrackingTensor):
            return value
        try:
            return value.as_subclass(tensorplay.Tensor)
        except AttributeError:
            return value

    def _schema_info(self, function: Any) -> RuntimeSchemaInfo | None:
        value = self.sharding_prop._operation_value(
            self.sharding_prop.op_to_schema_info, function
        )
        if value is not None:
            return value
        value = self.sharding_prop._operation_value(
            self.sharding_prop.op_to_schema_info_for_single_dim_strategy,
            function,
        )
        if value is not None:
            return value
        if DecompShardingStrategy.has_decomp(function):
            self.sharding_prop.decomp_strategy.ensure_schema_info(function)
            return self.sharding_prop._operation_value(
                self.sharding_prop.op_to_schema_info, function
            )
        return None

    def _reject_redistribution(
        self, schema: OpSchema, output_sharding: OutputSharding
    ) -> None:
        redistribute_schema = output_sharding.redistribute_schema
        if not output_sharding.needs_redistribute or redistribute_schema is None:
            return
        original = _spec_leaves((schema.args_schema, schema.kwargs_schema))
        desired = _spec_leaves(
            (redistribute_schema.args_schema, redistribute_schema.kwargs_schema)
        )
        if any(
            left.placements != right.placements
            for left, right in zip(original, desired)
        ):
            raise RuntimeError(f"decomposition requires redistribution for {schema.op}")

    def _output_spec(self, value: Any) -> Any:
        if isinstance(value, OutputSharding):
            if self._current_schema is None:
                raise RuntimeError("placement tracking has no active schema")
            self._reject_redistribution(self._current_schema, value)
            return value.output_spec
        if isinstance(value, DTensorSpec):
            return value
        if isinstance(value, OpStrategy):
            selected = self.sharding_prop._select_strategy(
                value, self._current_schema
            )
            return None if selected is None else selected.output_specs
        return getattr(value, "output_spec", value)

    def _wrap_output(self, value: Any, output_spec: Any) -> Any:
        if isinstance(value, tuple):
            if not isinstance(output_spec, (tuple, list)):
                return value
            return tuple(
                self._wrap_output(item, spec)
                for item, spec in zip(value, output_spec)
            )
        if isinstance(value, list):
            if not isinstance(output_spec, (tuple, list)):
                return value
            return [
                self._wrap_output(item, spec)
                for item, spec in zip(value, output_spec)
            ]
        if isinstance(value, dict):
            if not isinstance(output_spec, dict):
                return value
            return {
                key: self._wrap_output(item, output_spec[key])
                if key in output_spec
                else item
                for key, item in value.items()
            }
        if not isinstance(output_spec, DTensorSpec) or not _is_tensor(value):
            return value
        metadata = _tensor_meta(value)
        if metadata is not None:
            output_spec = output_spec.shallow_copy_with_tensor_meta(metadata)
        return PlacementTrackingTensor.from_tensor(value, output_spec, self)

    def dispatch(
        self,
        function: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        schema_args, schema_kwargs = _normalize_call_arguments(
            function, args, kwargs
        )
        args_schema = _tree_map(
            schema_args,
            lambda value: getattr(value, "_spec", value),
        )
        kwargs_schema = _tree_map(
            schema_kwargs,
            lambda value: getattr(value, "_spec", value),
        )
        schema = OpSchema(
            function,
            args_schema,
            kwargs_schema,
            schema_info=self._schema_info(function),
        )
        self._current_schema = schema
        output_sharding = self.sharding_prop.propagate_op_sharding_non_cached(schema)
        if output_sharding is None:
            raise NotImplementedError(
                f"no placement rule is registered for decomposition operation {function}"
            )
        output_spec = self._output_spec(output_sharding)
        with _disable_dispatch_hooks():
            result = _call_with_plain_values(
                function,
                _tree_map(args, self._to_plain),
                _tree_map(kwargs, self._to_plain),
            )
        return self._wrap_output(result, output_spec)

    def make_input(
        self,
        spec: DTensorSpec,
        placement: Placement,
    ) -> PlacementTrackingTensor:
        if spec.tensor_meta is None:
            raise NotImplementedError("decomposition tracing requires tensor metadata")
        metadata = spec.tensor_meta
        device_type = str(getattr(self.mesh, "device_type", "cpu"))
        factory = getattr(tensorplay, "empty_strided", None)
        if factory is None:
            value = tensorplay.empty(
                metadata.shape,
                dtype=metadata.dtype,
                device=device_type,
            )
        else:
            try:
                value = factory(
                    metadata.shape,
                    metadata.stride,
                    dtype=metadata.dtype,
                    device=device_type,
                )
            except (RuntimeError, TypeError, ValueError):
                value = factory(
                    metadata.shape,
                    metadata.stride,
                    dtype=metadata.dtype,
                    device="cpu",
                )
        fake_spec = DTensorSpec(self.mesh, (placement,), tensor_meta=metadata)
        return PlacementTrackingTensor.from_tensor(value, fake_spec, self)


def _disable_dispatch_hooks() -> Any:
    from tensorplay.overrides import _disable_tensorplay_function

    return _disable_tensorplay_function()


def _disable_local_tensor_mode() -> Any:
    from .._local_tensor import maybe_disable_local_tensor_mode

    return maybe_disable_local_tensor_mode()


class DecompShardingStrategy:
    """Generate operation strategies by tracing registered decompositions."""

    def __init__(self, sharding_prop: "ShardingPropagator") -> None:
        self.sharding_prop = sharding_prop
        self._fake_meshes: dict[str, Any] = {}

    @staticmethod
    def _graph_decomposition(operation: Any) -> Callable[..., Any] | None:
        try:
            from ...graph.passes.decompose import _DECOMP_METHODS
        except (ImportError, AttributeError):
            return None
        return _DECOMP_METHODS.get(_operation_name(operation))

    @classmethod
    def _lookup_decomposition(
        cls, operation: Any
    ) -> tuple[str, Callable[..., Any] | None]:
        direct = _safe_table_get(decomposition_table, operation)
        if callable(direct):
            return "runtime", direct
        method = getattr(operation, "__tensorplay_decomposition__", None)
        if callable(method):
            return "runtime", method
        graph = cls._graph_decomposition(operation)
        if callable(graph):
            return "graph", graph
        return "", None

    @staticmethod
    def has_decomp(operation: Any) -> bool:
        return DecompShardingStrategy._lookup_decomposition(operation)[1] is not None

    def ensure_schema_info(self, operation: Any) -> None:
        if self.sharding_prop._operation_value(
            self.sharding_prop.op_to_schema_info, operation
        ) is None:
            self.sharding_prop.op_to_schema_info[operation] = _infer_schema_info_from_op(
                operation
            )

    def _get_fake_mesh(self, device_type: str) -> Any:
        fake_mesh = self._fake_meshes.get(device_type)
        if fake_mesh is None:
            from ..device_mesh import DeviceMesh

            fake_mesh = DeviceMesh(device_type, [0])
            self._fake_meshes[device_type] = fake_mesh
        return fake_mesh

    def propagate_strategy(
        self,
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    ) -> OpStrategy | None:
        if not _tree_any(
            (op_schema.args_schema, op_schema.kwargs_schema),
            lambda value: isinstance(value, DTensorSpec),
        ):
            return None

        specs = _spec_leaves((op_schema.args_schema, op_schema.kwargs_schema))
        if not specs:
            return None
        mesh = specs[0].mesh
        candidate_placements = self._get_candidate_placements(op_schema)
        fake_mesh = self._get_fake_mesh(str(getattr(mesh, "device_type", "cpu")))
        single_dim_strategies: list[list[Any]] = []
        output_placements: list[Any] = []
        for input_placements in candidate_placements:
            try:
                output = self._propagate_through_decomp(
                    op_schema,
                    input_placements,
                    fake_mesh,
                )
            except NotImplementedError:
                return None
            except (RuntimeError, KeyError, IndexError, TypeError, ValueError):
                continue
            output_placements = (
                [output] if not isinstance(output, tuple) else list(output)
            )
            single_dim_strategies.append(output_placements + list(input_placements))

        if not single_dim_strategies:
            raise AssertionError(
                "decomposition propagation produced no valid placement strategy"
            )

        strategy_schema = self.sharding_prop._wrap_with_op_strategy(op_schema)
        from ._ops.utils import expand_to_full_mesh_op_strategy

        return expand_to_full_mesh_op_strategy(
            mesh,
            strategy_schema,
            single_dim_strategies,
            input_index=len(output_placements),
            output_tensor_meta=output_tensor_meta,
        )

    def _propagate_through_decomp(
        self,
        op_schema: OpSchema,
        placements: tuple[Placement | None, ...],
        mesh: Any,
    ) -> Any:
        kind, decomp_fn = self._lookup_decomposition(op_schema.op)
        if decomp_fn is None:
            raise NotImplementedError(f"no decomposition is registered for {op_schema.op}")

        mode = PlacementTrackingMode(self.sharding_prop, mesh)
        placement_iter = iter(placements)

        def to_tracking(value: Any) -> Any:
            if not isinstance(value, DTensorSpec):
                return value
            placement = next(placement_iter)
            if placement is None:
                raise ValueError("a tensor input must have a placement")
            return mode.make_input(value, placement)

        args = _tree_map(op_schema.args_schema, to_tracking)
        kwargs = _tree_map(op_schema.kwargs_schema, to_tracking)
        with _disable_local_tensor_mode():
            if kind == "runtime":
                output = decomp_fn(*args, **kwargs)
            else:
                output = self._run_graph_decomposition(
                    decomp_fn,
                    op_schema.op,
                    args,
                    kwargs,
                )

        def get_placement(value: Any) -> Any:
            if isinstance(value, PlacementTrackingTensor):
                spec = getattr(value, "_spec", None)
                if spec is not None and spec.placements:
                    return spec.placements[0]
            return None

        result = _tree_map(output, get_placement)
        flat = [value for value in _tree_values(result) if value is not None]
        if not flat:
            return None
        return flat[0] if len(flat) == 1 else tuple(flat)

    @staticmethod
    def _run_graph_decomposition(
        decomp_fn: Callable[..., Any],
        operation: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        from ...graph import Graph, GraphModule

        graph = Graph()
        input_values: list[Any] = []

        def make_node(value: Any) -> Any:
            if isinstance(value, PlacementTrackingTensor):
                index = len(input_values)
                input_values.append(value)
                return graph.placeholder(f"decomp_arg_{index}")
            if isinstance(value, tuple):
                return tuple(make_node(item) for item in value)
            if isinstance(value, list):
                return [make_node(item) for item in value]
            if isinstance(value, dict):
                return {key: make_node(item) for key, item in value.items()}
            return value

        args_nodes = tuple(make_node(value) for value in args)
        kwargs_nodes = {key: make_node(value) for key, value in kwargs.items()}
        fake_op = SimpleNamespace(
            op="call_function",
            target=operation,
            args=args_nodes,
            kwargs=kwargs_nodes,
        )
        replacement = decomp_fn(graph, fake_op)
        if replacement is None:
            raise NotImplementedError("decomposition returned no replacement")
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            name = _GRAPH_OPERATOR_NAMES.get(node.target)
            if name is None:
                continue
            function = getattr(tensorplay, name, None)
            if callable(function):
                node.target = function
        graph.output(replacement)
        return GraphModule(None, graph)(*input_values)

    @staticmethod
    def _get_candidate_placements(
        op_schema: OpSchema,
    ) -> list[tuple[Placement | None, ...]]:
        flat_specs = _tree_values(_extract_input_specs(op_schema))
        all_placements: set[Placement] = {Replicate()}
        for value in flat_specs:
            if isinstance(value, DTensorSpec):
                all_placements.update(value.placements)

        candidates: list[list[Placement | None]] = []
        for value in flat_specs:
            if not isinstance(value, DTensorSpec):
                candidates.append([None])
                continue
            options = set(all_placements)
            for placement in tuple(all_placements):
                if isinstance(placement, _StridedShard):
                    options.update(
                        _StridedShard(dim, split_factor=placement.split_factor)
                        for dim in range(value.ndim)
                    )
                elif isinstance(placement, Shard):
                    options.update(Shard(dim) for dim in range(value.ndim))
            candidates.append(list(options))
        return list(itertools.product(*candidates))
