"""Operation dispatch and metadata propagation for distributed tensors."""

from __future__ import annotations

import inspect
import contextlib
import logging
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable

from ._api import DTensor, _participates
from .placement_types import Partial, Placement, Replicate, Shard, _is_shard_like

__all__ = ["OpDispatcher", "unwrap_dtensor", "wrap_dtensor"]


_dispatch_logger = logging.getLogger(__name__)
_dispatch_set_level = _dispatch_logger.setLevel


def _setLevel_and_reinit(level: int) -> None:
    _dispatch_set_level(level)


def _ignore_fresh_unbacked_symbols_for_dtensor_tracing(
    output_spec: object,
) -> contextlib.AbstractContextManager[None]:
    return contextlib.nullcontext()


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _dtensors(value: Any) -> list[DTensor]:
    return [item for item in _walk(value) if isinstance(item, DTensor)]


def unwrap_dtensor(value: Any) -> Any:
    if isinstance(value, DTensor):
        return value.to_local()
    if isinstance(value, tuple):
        return tuple(unwrap_dtensor(item) for item in value)
    if isinstance(value, list):
        return [unwrap_dtensor(item) for item in value]
    if isinstance(value, dict):
        return {key: unwrap_dtensor(item) for key, item in value.items()}
    return value


def _shape(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _stride(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "stride", None)
    if raw is None:
        return None
    try:
        raw = raw() if callable(raw) else raw
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _is_tensor(value: Any) -> bool:
    return _shape(value) is not None and hasattr(value, "dim")


def _normalize_dim(dim: Any, ndim: int) -> int:
    value = int(dim)
    if value < 0:
        value += ndim
    if value < 0 or value >= ndim:
        raise IndexError(f"dimension {dim} is outside tensor rank {ndim}")
    return value


def _broadcast_shape(shapes: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    values = list(shapes)
    if not values:
        return ()
    rank = max(len(value) for value in values)
    result = [1] * rank
    for shape in values:
        offset = rank - len(shape)
        for index, value in enumerate(shape):
            target = offset + index
            if value not in (1, result[target]) and result[target] != 1:
                raise ValueError(f"shapes {values!r} are not broadcastable")
            result[target] = max(result[target], value)
    return tuple(result)


def _merge_placement(left: Placement, right: Placement) -> Placement:
    if left == right:
        return left
    if isinstance(left, Replicate):
        return right
    if isinstance(right, Replicate):
        return left
    if isinstance(left, Partial) and isinstance(right, Partial):
        return left if left.reduce_op == right.reduce_op else Replicate()
    return Replicate()


def _broadcast_placements(
    values: list[DTensor], output_shape: tuple[int, ...]
) -> tuple[Placement, ...]:
    template = values[0]
    result: list[Placement] = [Replicate() for _ in template.placements]
    for value in values:
        offset = len(output_shape) - value.ndim
        for mesh_dim, placement in enumerate(value.placements):
            candidate: Placement = placement
            if _is_shard_like(placement):
                input_dim = placement.dim
                output_dim = offset + input_dim
                if (
                    output_dim < 0
                    or output_dim >= len(output_shape)
                    or value.shape[input_dim] == 1
                    or value.shape[input_dim] != output_shape[output_dim]
                ):
                    candidate = Replicate()
                elif isinstance(placement, Shard):
                    candidate = Shard(output_dim)
            result[mesh_dim] = _merge_placement(result[mesh_dim], candidate)
    return tuple(result)


def _map_permutation(
    placements: tuple[Placement, ...], permutation: tuple[int, ...]
) -> tuple[Placement, ...]:
    result: list[Placement] = []
    for placement in placements:
        if _is_shard_like(placement):
            try:
                target_dim = permutation.index(placement.dim)
            except ValueError:
                result.append(Replicate())
            else:
                result.append(Shard(target_dim))
        else:
            result.append(placement)
    return tuple(result)


def _first_tensor_sequence(value: Any) -> list[DTensor]:
    if isinstance(value, (tuple, list)):
        result = [item for item in _walk(value) if isinstance(item, DTensor)]
        if result:
            return result
    return []


def _value_at(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    index: int,
    name: str,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if index < len(args) else default


def _normalize_shape_arg(value: Any, total: int | None = None) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,)
    else:
        values = tuple(int(item) for item in value)
    if total is None:
        return values
    unknown = [index for index, item in enumerate(values) if item == -1]
    if len(unknown) > 1:
        raise ValueError("only one dimension can be inferred")
    known = math.prod(item for item in values if item != -1)
    if unknown:
        if known == 0 or total % known:
            raise ValueError("inferred dimension is not integral")
        values = tuple(total // known if item == -1 else item for item in values)
    if math.prod(values) != total:
        raise ValueError("view sizes do not match the number of elements")
    return values


def _view_shape_arg(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], name: str
) -> Any:
    if "shape" in kwargs:
        return kwargs["shape"]
    if name in {"view", "reshape", "view_copy", "_unsafe_view"} and len(args) != 1:
        return args
    return args[0] if args else ()


def _size_shape_arg(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], *names: str
) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs[name]
    if len(args) == 1:
        return args[0]
    return args


def _local_view_call(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    template: DTensor,
    output_spec: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name not in {"reshape", "view", "view_copy", "_unsafe_view"}:
        return args, kwargs
    if not hasattr(output_spec, "placements"):
        return args, kwargs
    target = _view_shape_arg(args, kwargs, name)
    global_shape = _normalize_shape_arg(target, math.prod(template.shape))
    from ._utils import compute_local_shape_and_global_offset

    local_shape, _ = compute_local_shape_and_global_offset(
        global_shape,
        template.device_mesh,
        output_spec.placements,
        skip_offset=True,
    )
    if "shape" in kwargs:
        local_kwargs = dict(kwargs)
        local_kwargs.pop("shape")
        return (local_shape,), local_kwargs
    if len(args) == 1:
        return (local_shape,), kwargs
    return tuple(local_shape), kwargs


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [1] * len(shape)
    running = 1
    for index in reversed(range(len(shape))):
        result[index] = running
        running *= int(shape[index])
    return tuple(result)


def _spec_tree(value: Any) -> Any:
    if isinstance(value, DTensor):
        from ._dtensor_spec import DTensorSpec, TensorMeta

        return DTensorSpec(
            value.device_mesh,
            value.placements,
            TensorMeta(value.shape, value.stride(), value.dtype),
        )
    if _is_tensor(value):
        from ._dtensor_spec import TensorMeta

        return TensorMeta(_shape(value) or (), _stride(value) or (), value.dtype)
    if isinstance(value, tuple):
        return tuple(_spec_tree(item) for item in value)
    if isinstance(value, list):
        return [_spec_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _spec_tree(item) for key, item in value.items()}
    return value


def _output_spec(value: Any, index: int | None = None) -> Any:
    value = getattr(value, "output_spec", value)
    if index is not None and isinstance(value, (tuple, list)):
        return value[index] if index < len(value) else None
    return value


def _operation_name(operation: Any) -> str:
    name = getattr(operation, "__name__", None)
    if name:
        return str(name).removesuffix("_")
    name = getattr(operation, "__qualname__", None)
    if name:
        return str(name).rsplit(".", 1)[-1].removesuffix("_")
    return str(operation).rsplit(".", 1)[-1].removesuffix("_")


def _canonicalize_operation_arguments(
    operation: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if args:
        if (
            _operation_name(operation) == "index"
            and len(args) >= 2
            and not isinstance(args[1], (tuple, list))
        ):
            return (args[0], (args[1],), *args[2:]), dict(kwargs)
        return args, dict(kwargs)
    name = _operation_name(operation)
    values = dict(kwargs)

    def take(*names: str) -> Any:
        for key in names:
            if key in values:
                return values.pop(key)
        raise KeyError(names[0])

    layouts: dict[str, tuple[tuple[str, ...], ...]] = {
        "cat": (("tensors",), ("dim",)),
        "stack": (("tensors",), ("dim",)),
        "index_select": (("input", "self"), ("dim",), ("index",)),
        "gather": (("input", "self"), ("dim",), ("index",)),
        "index": (("input", "self"), ("indices",)),
        "multinomial": (
            ("input", "self"),
            ("num_samples",),
            ("replacement",),
            ("impl",),
        ),
    }
    fields = layouts.get(name)
    if fields is None or not any(
        any(key in values for key in group) for group in fields
    ):
        return args, values
    canonical: list[Any] = []
    for group in fields:
        try:
            canonical.append(take(*group))
        except KeyError:
            if group == ("dim",):
                canonical.append(0)
            elif group == ("replacement",):
                canonical.append(False)
            elif group == ("impl",):
                canonical.append(0)
            else:
                return args, dict(kwargs)
    if name == "index" and not isinstance(canonical[1], (tuple, list)):
        canonical[1] = (canonical[1],)
    return tuple(canonical), values


def _has_basic_index(index: Any) -> bool:
    values = index if isinstance(index, (tuple, list)) else (index,)
    return any(value is not None and not _is_tensor(value) for value in values)


def _as_strided_permutation(
    tensor: DTensor,
    size: Sequence[int],
    stride: Sequence[int],
) -> list[int] | None:
    if len(size) != tensor.ndim:
        return None
    base = list(zip(tensor.shape, tensor.stride()))
    dimensions: list[int] = []
    used = [False] * len(base)
    for target in zip(size, stride):
        candidates = [
            index
            for index, current in enumerate(base)
            if not used[index] and tuple(current) == tuple(target)
        ]
        if len(candidates) != 1:
            return None
        used[candidates[0]] = True
        dimensions.append(candidates[0])
    return dimensions


def as_strided_handler(
    operation: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if not args:
        raise TypeError("as_strided requires an input tensor")
    tensor = args[0]
    size = kwargs.get("size", args[1] if len(args) > 1 else ())
    stride = kwargs.get("stride", args[2] if len(args) > 2 else ())
    storage_offset = kwargs.get(
        "storage_offset", args[3] if len(args) > 3 else None
    )
    current_offset = getattr(tensor, "storage_offset", None)
    current_offset = current_offset() if callable(current_offset) else current_offset
    if storage_offset is None or current_offset == storage_offset:
        if tuple(tensor.shape) == tuple(size) and tuple(tensor.stride()) == tuple(stride):
            return tensor
        dimensions = _as_strided_permutation(tensor, size, stride)
        if dimensions is not None:
            return tensor.permute(tuple(dimensions))
    raise RuntimeError("as_strided is not supported for distributed tensors")


def is_same_size_handler(
    operation: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    return tuple(_shape(args[0]) or ()) == tuple(_shape(args[1]) or ())


def is_pinned_handler(
    operation: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    value = args[0].to_local() if isinstance(args[0], DTensor) else args[0]
    checker = getattr(value, "is_pinned", None)
    return bool(checker() if callable(checker) else checker)


def found_inf_reduce_handler(
    operation: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if len(args) < 2:
        raise TypeError("non-finite check requires gradients and destination")
    dispatcher = DTensor._op_dispatcher
    op_info = dispatcher.unwrap_to_op_info(operation, args, kwargs)
    local_args = op_info.local_args
    operation(*local_args, **op_info.local_kwargs)
    gradients = args[0]
    gradient = next(
        (value for value in _walk(gradients) if isinstance(value, DTensor)), None
    )
    if gradient is None:
        return None
    from ._dtensor_spec import DTensorSpec, TensorMeta

    placements = tuple(
        placement if isinstance(placement, Replicate) else Partial("max")
        for placement in gradient.placements
    )
    target = args[1]
    spec = DTensorSpec(
        gradient.device_mesh,
        placements,
        tensor_meta=TensorMeta(
            _shape(target) or (),
            _stride(target) or (),
            getattr(target, "dtype", None),
        ),
    )
    reduced = DTensor(
        target,
        gradient.device_mesh,
        spec.placements,
        shape=spec.shape,
        stride=spec.stride,
    ).full_tensor()
    copy = getattr(target, "copy_", None)
    if callable(copy):
        copy(reduced)
    return None


def _operation_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    if args and isinstance(args[0], DTensor):
        return args[1:]
    return args


def _replace_identity(value: Any, old: Any, new: Any) -> Any:
    if value is old:
        return new
    if isinstance(value, tuple):
        return tuple(_replace_identity(item, old, new) for item in value)
    if isinstance(value, list):
        return [_replace_identity(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_identity(item, old, new) for key, item in value.items()
        }
    return value


def _view_layout(
    value: DTensor, output_shape: tuple[int, ...]
) -> tuple[tuple[Placement, ...], tuple[Placement, ...]]:
    try:
        from ._ops._view_ops import propagate_shape_and_sharding, view_groups

        mesh_shape = tuple(
            int(value.device_mesh.size(index))
            for index in range(len(value.placements))
        )
        rule = view_groups(value.shape, output_shape)
        return tuple(
            tuple(item)
            for item in propagate_shape_and_sharding(
                value.placements, value.shape, rule, mesh_shape
            )
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return value.placements, tuple(
            Replicate() if _is_shard_like(item) else item
            for item in value.placements
        )


def wrap_dtensor(value: Any, template: DTensor | None) -> Any:
    if template is None:
        return value
    if _is_tensor(value):
        shape = _shape(value)
        if shape is None:
            return value
        return DTensor(
            value,
            template.device_mesh,
            template.placements,
            shape=shape,
            stride=_stride(value),
        )
    if isinstance(value, tuple):
        return tuple(wrap_dtensor(item, template) for item in value)
    if isinstance(value, list):
        return [wrap_dtensor(item, template) for item in value]
    return value


class OpDispatcher:
    """Execute an operation on local tensors and restore global metadata."""

    _POINTWISE_NAMES = frozenset(
        {
            "abs", "acos", "asin", "atan", "ceil", "clamp", "cos", "cosh",
            "elu", "erf", "exp", "expm1", "floor", "gelu", "hardtanh", "log",
            "log1p", "logical_not", "neg", "pow", "relu", "relu6", "round",
            "rsqrt", "sigmoid", "sign", "sin", "sinh", "sqrt", "square", "tan",
            "tanh", "trunc", "where", "add", "addcmul", "addcdiv", "div", "mul",
            "multiply", "sub", "subtract", "maximum", "minimum", "fmod", "remainder",
            "true_divide", "eq", "ne", "ge", "gt", "le", "lt", "logical_and",
            "logical_or", "logical_xor",
        }
    )
    _REDUCTION_NAMES = frozenset(
        {
            "sum", "mean", "prod", "amax", "amin", "max", "min", "all", "any",
            "var", "std", "norm",
        }
    )
    _PLAIN_AUXILIARY_OPS = frozenset(
        {
            "gather",
            "index",
            "index_add",
            "index_copy",
            "index_fill",
            "index_put",
            "index_reduce",
            "index_select",
            "scatter",
            "scatter_add",
            "take",
            "take_along_dim",
        }
    )

    def __init__(self) -> None:
        self._rules: dict[Any, Callable[..., Any]] = {}
        from ._nonlinear_redux import argminmax_handler, minmax_dim_handler
        from . import _random as random
        from ._tp_conv import convolution_backward_handler, convolution_handler

        self._random_module = random
        self._random_ops = frozenset(
            {
                "native_dropout",
                "normal_",
                "rand_like",
                "randn_like",
                "randint_like",
                "randint_like_low_dtype",
                "uniform_",
                "bernoulli",
                "bernoulli_",
            }
        )
        self._custom_op_handlers: dict[Any, Callable[..., Any]] = {
            "as_strided": as_strided_handler,
            "is_same_size": is_same_size_handler,
            "is_pinned": is_pinned_handler,
            "_amp_foreach_non_finite_check_and_unscale_": found_inf_reduce_handler,
            "argmax": argminmax_handler,
            "argmin": argminmax_handler,
            "max": minmax_dim_handler,
            "min": minmax_dim_handler,
            "convolution": convolution_handler,
            "convolution_backward": convolution_backward_handler,
        }
        self._implicit_replication = False
        from ._sharding_prop import ShardingPropagator

        self.sharding_propagator = ShardingPropagator()

    def _random_context(
        self, operation: Any, template: DTensor, local_tensor: Any
    ) -> contextlib.AbstractContextManager[Any]:
        if _operation_name(operation) not in self._random_ops:
            return contextlib.nullcontext()
        random = self._random_module
        if random._rng_tracker is None and random.is_rng_supported_mesh(
            template.device_mesh
        ):
            random._rng_tracker = random.OffsetBasedRNGTracker(
                template.device_mesh.device_type
            )
        if random._rng_tracker is None or bool(getattr(local_tensor, "is_meta", False)):
            return contextlib.nullcontext()
        return random._rng_tracker._distribute_region(
            self._spec_from_dtensor(template)
        )

    def register(self, operation: Any, rule: Callable[..., Any]) -> Callable[..., Any]:
        self._rules[operation] = rule
        return rule

    @property
    def _allow_implicit_replication(self) -> bool:
        return self._implicit_replication

    @_allow_implicit_replication.setter
    def _allow_implicit_replication(self, value: bool) -> None:
        self._implicit_replication = bool(value)

    def _validate(
        self,
        values: list[DTensor],
        original: Any,
        *,
        allow_non_scalar_plain: bool = False,
    ) -> None:
        if not values:
            return
        mesh = values[0].device_mesh
        for value in values[1:]:
            if value.device_mesh != mesh:
                raise ValueError("DTensor operands must use the same device mesh")
        for value in _walk(original):
            if isinstance(value, DTensor) or not _is_tensor(value):
                continue
            numel = getattr(value, "numel", lambda: 0)()
            dim = getattr(value, "dim", lambda: 0)()
            if (
                int(dim) != 0
                and int(numel) != 1
                and not allow_non_scalar_plain
            ):
                raise RuntimeError(
                    "distributed operations require plain tensor operands to be scalar"
                )

    @classmethod
    def _allows_plain_auxiliary(cls, operation: Any) -> bool:
        return _operation_name(operation) in cls._PLAIN_AUXILIARY_OPS

    @staticmethod
    def _normalize_public_kwargs(operation: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "self" not in kwargs:
            return kwargs
        try:
            parameters = inspect.signature(operation).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "input" in parameters and "self" not in parameters:
            kwargs = dict(kwargs)
            kwargs["input"] = kwargs.pop("self")
        return kwargs

    def _propagate(self, operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        rule = self._rules.get(operation)
        if rule is not None:
            return rule(*unwrap_dtensor(args), **unwrap_dtensor(kwargs))
        name = _operation_name(operation)
        if name == "index":
            index = _canonicalize_operation_arguments(operation, args, kwargs)[0]
            if index and len(index) > 1 and _has_basic_index(index[1]):
                return None
        propagator = self.sharding_propagator
        if (
            propagator._operation_value(propagator.op_to_rules, operation) is None
            and propagator._operation_value(propagator.op_strategy_funcs, operation) is None
            and propagator._operation_value(
                propagator.op_single_dim_strategy_funcs, operation
            ) is None
        ):
            _, global_rule = propagator._global_rule(operation)
            if (
                global_rule is None
                and not propagator.decomp_strategy.has_decomp(operation)
            ):
                return None
        from ._op_schema import OpSchema

        schema_args, schema_kwargs = _canonicalize_operation_arguments(
            operation, args, kwargs
        )
        schema = OpSchema(
            operation, _spec_tree(schema_args), _spec_tree(schema_kwargs)
        )
        result = self.sharding_propagator.propagate_op_sharding_non_cached(schema)
        if result is None and name != operation:
            schema = OpSchema(
                name, _spec_tree(schema_args), _spec_tree(schema_kwargs)
            )
            result = self.sharding_propagator.propagate_op_sharding_non_cached(schema)
        return result

    def _propagate_op_sharding_dispatch_slow_path(
        self,
        operation: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        op_info: Any,
        try_cache: bool,
    ) -> Any:
        if getattr(op_info, "schema", None) is None:
            raise AssertionError("op_info.schema is required for sharding propagation")
        try:
            result = self.sharding_propagator.propagate_op_sharding_non_cached(
                op_info.schema
            )
        except Exception as error:
            raise RuntimeError(
                f"{error}\n\nSharding propagation failed for {op_info.schema}"
            ) from error
        op_info.output_sharding = result
        return result

    def _dispatch_get_local_results_slow_path(
        self,
        operation: Any,
        args: tuple[Any, ...],
        op_info: Any,
    ) -> Any:
        output_sharding = getattr(op_info, "output_sharding", None)
        if output_sharding is None:
            raise AssertionError("output sharding is required")
        if output_sharding.needs_redistribute:
            if output_sharding.redistribute_schema is None:
                raise AssertionError("redistribute schema is required")
            self.redistribute_local_args(
                op_info,
                output_sharding.redistribute_schema,
                bool(output_sharding.schema_suggestions),
            )
        if not _participates(getattr(op_info, "mesh", None)):
            return self._empty_local_result(output_sharding.output_spec)
        local_args = op_info.local_args
        if op_info.args_tree_spec is not None:
            from ...utils._pytree import tree_unflatten

            local_args = tree_unflatten(local_args, op_info.args_tree_spec)
        local_args = tuple(local_args)
        local_kwargs = dict(op_info.local_kwargs)
        local_operation = operation
        if (
            output_sharding.needs_redistribute
            and output_sharding.redistribute_schema is not None
            and output_sharding.redistribute_schema.func != operation
        ):
            local_operation = output_sharding.redistribute_schema.func
        if not callable(local_operation):
            import tensorplay.functional as functional

            local_operation = getattr(functional, _operation_name(local_operation))
        return local_operation(*local_args, **local_kwargs)

    @staticmethod
    def _empty_local_result(spec: Any) -> Any:
        from ._dtensor_spec import DTensorSpec

        if isinstance(spec, DTensorSpec):
            import tensorplay

            if spec.tensor_meta is None:
                raise RuntimeError("distributed output has no tensor metadata")
            if len(spec.tensor_meta.shape) == 0:
                return tensorplay.zeros((), dtype=spec.tensor_meta.dtype)
            return tensorplay.empty((0,), dtype=spec.tensor_meta.dtype)
        if isinstance(spec, tuple):
            return tuple(OpDispatcher._empty_local_result(item) for item in spec)
        if isinstance(spec, list):
            return [OpDispatcher._empty_local_result(item) for item in spec]
        return None

    def _dispatch_fast_path_python_tail(
        self,
        operation: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: Any,
        output_sharding: Any,
        local_results: Any,
        participating: bool,
        is_inplace_op: bool,
        is_out_variant_op: bool,
    ) -> Any:
        from ._dtensor_spec import DTensorSpec

        spec = output_sharding.output_spec
        if is_inplace_op:
            if not args or not isinstance(args[0], DTensor):
                return None
            if isinstance(spec, DTensorSpec):
                args[0]._placements = spec.placements
                args[0]._shape = spec.shape
                args[0]._stride = spec.stride
                return args[0]
            return None
        if is_out_variant_op:
            output_specs = spec if isinstance(spec, (tuple, list)) else (spec,)
            result = []
            index = 0
            for value in kwargs.values():
                if isinstance(value, DTensor) and index < len(output_specs):
                    value._placements = output_specs[index].placements
                    result.append(value)
                    index += 1
            if result:
                return tuple(result) if len(result) > 1 else result[0]
        return self.wrap(local_results, spec)

    @staticmethod
    def redistribute_local_args(
        op_info: Any,
        suggested_input_schema: Any,
        use_val_from_redistribute_schema: bool,
    ) -> None:
        from ...utils._pytree import tree_flatten
        from ._redistribute import redistribute_local_tensor
        from ._utils import (
            ExplicitRedistributionContext,
            _format_implicit_redistribution_msg,
        )

        if op_info.args_tree_spec is not None:
            target_args = tree_flatten(suggested_input_schema.args_schema)[0]
        else:
            target_args = list(suggested_input_schema.args_schema)
        local_args = list(op_info.local_args)
        for index, arg_spec in enumerate(op_info.flat_args_schema):
            target = target_args[index] if index < len(target_args) else arg_spec
            if isinstance(arg_spec, DTensorSpec) and isinstance(target, DTensorSpec):
                if arg_spec != target:
                    ExplicitRedistributionContext.observe_redistribution(
                        arg_spec,
                        target,
                        _format_implicit_redistribution_msg(
                            getattr(op_info, "schema", suggested_input_schema)
                        ),
                    )
                    local_args[index] = redistribute_local_tensor(
                        local_args[index], arg_spec, target
                    )
            elif use_val_from_redistribute_schema and index < len(target_args):
                local_args[index] = target
        if use_val_from_redistribute_schema:
            local_args.extend(target_args[len(op_info.flat_args_schema) :])
        op_info.local_args = tuple(local_args)

    def unwrap_to_op_info(
        self,
        operation: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return self._unwrap_to_op_info_impl(operation, args, kwargs, True)

    def _unwrap_to_op_info_impl(
        self,
        operation: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        create_schema: bool,
    ) -> Any:
        from ._op_schema import OpInfo, OpSchema
        from ...utils._pytree import tree_flatten, tree_unflatten

        schema_info = self.sharding_propagator.op_to_schema_info.get(operation)
        if schema_info is None:
            schema_info = self.sharding_propagator.op_to_schema_info_for_single_dim_strategy.get(
                operation
            )

        def contains_dtensor(value: Any) -> bool:
            return any(isinstance(item, DTensor) for item in _walk(value))

        needs_pytree = bool(getattr(schema_info, "needs_pytree", False)) or any(
            contains_dtensor(value) for value in args
        )
        if needs_pytree:
            args_list, args_tree_spec = tree_flatten(args)
        else:
            args_list, args_tree_spec = list(args), None

        args_schema: list[Any] = []
        local_args: list[Any] = []
        compute_mesh = None
        for value in args_list:
            if isinstance(value, DTensor):
                args_schema.append(
                    self._spec_from_dtensor(value)
                )
                local_args.append(value.to_local())
                compute_mesh = compute_mesh or value.device_mesh
            elif _is_tensor(value):
                if compute_mesh is None:
                    compute_mesh = self._find_mesh_from_values(args_list)
                args_schema.append(
                    self._try_replicate_spec_for_scalar_tensor(
                        operation, value, compute_mesh
                    )
                )
                local_args.append(value)
            else:
                args_schema.append(value)
                local_args.append(value)

        kwargs_schema: dict[str, Any] = {}
        local_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, DTensor):
                kwargs_schema[key] = self._spec_from_dtensor(value)
                local_kwargs[key] = value.to_local()
                compute_mesh = compute_mesh or value.device_mesh
            elif _is_tensor(value):
                if compute_mesh is None:
                    compute_mesh = self._find_mesh_from_values(args_list)
                kwargs_schema[key] = self._try_replicate_spec_for_scalar_tensor(
                    operation, value, compute_mesh
                )
                local_kwargs[key] = value
            else:
                kwargs_schema[key] = value
                local_kwargs[key] = value

        if compute_mesh is None:
            raise AssertionError("no device mesh was found in distributed arguments")
        schema_args = (
            tree_unflatten(args_schema, args_tree_spec)
            if args_tree_spec is not None
            else tuple(args_schema)
        )
        return OpInfo(
            mesh=compute_mesh,
            schema=OpSchema(
                operation,
                schema_args,
                kwargs_schema,
                schema_info=schema_info,
            )
            if create_schema
            else None,
            flat_args_schema=args_schema,
            local_args=tuple(local_args),
            local_kwargs=local_kwargs,
            args_tree_spec=args_tree_spec,
        )

    @staticmethod
    def _spec_from_dtensor(value: DTensor) -> Any:
        from ._dtensor_spec import DTensorSpec, TensorMeta

        return DTensorSpec(
            value.device_mesh,
            value.placements,
            TensorMeta(value.shape, value.stride(), value.dtype),
        )

    @staticmethod
    def _find_mesh_from_values(values: Sequence[Any]) -> Any:
        for value in values:
            if isinstance(value, DTensor):
                return value.device_mesh
        return None

    @staticmethod
    def wrap(result: Any, spec: Any) -> Any:
        from ._dtensor_spec import DTensorSpec

        if _is_tensor(result):
            if isinstance(spec, DTensorSpec):
                return DTensor(
                    result,
                    spec.mesh,
                    spec.placements,
                    shape=spec.shape,
                    stride=spec.stride,
                )
            if spec is None:
                if len(_shape(result) or ()) != 0:
                    raise AssertionError("output tensor should be scalar")
                return result
            raise AssertionError(f"output spec does not match result: {spec!r}")
        if isinstance(result, (tuple, list)):
            if not isinstance(spec, (tuple, list)):
                raise AssertionError(f"output spec does not match result: {spec!r}")
            values = [OpDispatcher.wrap(value, child) for value, child in zip(result, spec)]
            return tuple(values) if isinstance(result, tuple) else values
        return result

    def _try_replicate_spec_for_scalar_tensor(
        self,
        operation: Any,
        tensor_arg: Any,
        compute_mesh: Any,
    ) -> Any:
        if compute_mesh is None:
            raise AssertionError("a device mesh is required for a plain tensor")
        numel = int(tensor_arg.numel())
        ndim = int(tensor_arg.dim())
        if numel == 1 and ndim == 1:
            warnings.warn(
                "a one-element non-scalar tensor is treated as replicated",
                stacklevel=2,
            )
        if numel != 1 and not self._allow_implicit_replication:
            if not self._allows_plain_auxiliary(operation):
                raise RuntimeError(
                    f"{operation}: plain tensor operands must be scalar or distributed"
                )
        from ._dtensor_spec import DTensorSpec, TensorMeta

        mesh_ndim_value = getattr(compute_mesh, "ndim")
        mesh_ndim = int(mesh_ndim_value() if callable(mesh_ndim_value) else mesh_ndim_value)
        stride = tensor_arg.stride() if callable(tensor_arg.stride) else tensor_arg.stride
        return DTensorSpec(
            compute_mesh,
            (Replicate(),) * mesh_ndim,
            tensor_meta=TensorMeta(tensor_arg.shape, stride, tensor_arg.dtype),
        )

    @staticmethod
    def _apply_redistribution(
        args: tuple[Any, ...], kwargs: dict[str, Any], propagated: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        from ._dtensor_spec import DTensorSpec, TensorMeta

        schema = getattr(propagated, "redistribute_schema", None)
        if not getattr(propagated, "needs_redistribute", False) or schema is None:
            return args, kwargs
        expected = [
            value
            for value in _walk(schema.args_schema)
            if hasattr(value, "placements") and hasattr(value, "mesh")
        ]
        expected.extend(
            value
            for value in _walk(schema.kwargs_schema)
            if hasattr(value, "placements") and hasattr(value, "mesh")
        )
        replacements = iter(expected)

        def replace(value: Any) -> Any:
            if isinstance(value, DTensor):
                target = next(replacements, None)
                if target is None or tuple(target.placements) == tuple(value.placements):
                    return value
                from ._utils import (
                    ExplicitRedistributionContext,
                    _format_implicit_redistribution_msg,
                )

                current = DTensorSpec(
                    value.device_mesh,
                    tuple(value.placements),
                    tensor_meta=TensorMeta(
                        value.shape,
                        value.stride(),
                        value.dtype,
                    ),
                )
                ExplicitRedistributionContext.observe_redistribution(
                    current,
                    target,
                    _format_implicit_redistribution_msg(schema),
                )
                return value.redistribute(placements=target.placements)
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        return tuple(replace(value) for value in args), {
            key: replace(value) for key, value in kwargs.items()
        }

    def _infer(
        self,
        result: Any,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        values: list[DTensor],
    ) -> tuple[tuple[int, ...], tuple[Placement, ...], tuple[int, ...]] | None:
        if not values or not _is_tensor(result):
            return None
        template = values[0]
        local_shape = _shape(result)
        if local_shape is None:
            return None
        shapes = [value.shape for value in values]
        placements = template.placements
        output_shape: tuple[int, ...] | None = None
        output_stride: tuple[int, ...] | None = None

        if name in self._POINTWISE_NAMES:
            output_shape = _broadcast_shape(shapes)
            placements = _broadcast_placements(values, output_shape)
        elif name == "embedding":
            if len(values) < 2 or values[0].ndim < 2:
                return None
            output_shape = values[1].shape + values[0].shape[1:]
        elif name == "embedding_dense_backward":
            if len(values) < 2 or len(args) < 2:
                return None
            num_weights = int(_value_at(args, kwargs, 1, "num_weights", 0))
            output_shape = (num_weights, values[0].shape[-1])
        elif name in {"transpose", "transpose_copy"}:
            dim0 = _value_at(args, kwargs, 0, "dim0", 0)
            dim1 = _value_at(args, kwargs, 1, "dim1", 1)
            first, second = int(dim0), int(dim1)
            first = first + template.ndim if first < 0 else first
            second = second + template.ndim if second < 0 else second
            if not (0 <= first < template.ndim and 0 <= second < template.ndim):
                raise IndexError("transpose dimension is outside tensor rank")
            permutation = list(range(template.ndim))
            permutation[first], permutation[second] = permutation[second], permutation[first]
            permutation = tuple(permutation)
            output_shape = tuple(template.shape[index] for index in permutation)
            placements = _map_permutation(template.placements, permutation)
            output_stride = tuple(template.stride(index) for index in permutation)
        elif name in {"permute", "permute_copy", "movedim"}:
            if name in {"permute", "permute_copy"}:
                dims = _value_at(args, kwargs, 0, "dims", ())
                permutation = tuple(int(item) for item in dims)
                permutation = tuple(item + template.ndim if item < 0 else item for item in permutation)
            else:
                source = _value_at(args, kwargs, 0, "source", 0)
                destination = _value_at(args, kwargs, 1, "destination", 0)
                sources = (source,) if isinstance(source, int) else tuple(source)
                destinations = (destination,) if isinstance(destination, int) else tuple(destination)
                sources = tuple(int(item) + template.ndim if int(item) < 0 else int(item) for item in sources)
                destinations = tuple(int(item) + template.ndim if int(item) < 0 else int(item) for item in destinations)
                if len(sources) != len(destinations):
                    raise ValueError("source and destination must have equal length")
                order = [-1] * template.ndim
                for old, new in zip(sources, destinations):
                    if not (0 <= old < template.ndim and 0 <= new < template.ndim) or order[new] != -1:
                        raise ValueError("invalid movedim dimensions")
                    order[new] = old
                unused = iter(index for index in range(template.ndim) if index not in sources)
                permutation = tuple(next(unused) if value == -1 else value for value in order)
            if len(permutation) != template.ndim or set(permutation) != set(range(template.ndim)):
                raise ValueError("invalid dimension permutation")
            output_shape = tuple(template.shape[index] for index in permutation)
            placements = _map_permutation(template.placements, permutation)
            output_stride = tuple(template.stride(index) for index in permutation)
        elif name == "t":
            if template.ndim > 2:
                raise ValueError("t expects a tensor with at most two dimensions")
            permutation = (1, 0) if template.ndim == 2 else tuple(range(template.ndim))
            output_shape = tuple(template.shape[index] for index in permutation)
            placements = _map_permutation(template.placements, permutation)
            output_stride = tuple(template.stride(index) for index in permutation)
        elif name in {"reshape", "view", "reshape_as", "view_as"}:
            if name.endswith("_as"):
                target = values[-1].shape
            else:
                target = _view_shape_arg(args, kwargs, name)
            output_shape = _normalize_shape_arg(target, math.prod(template.shape))
            _, placements = _view_layout(template, output_shape)
        elif name in {"view_as_complex", "view_as_complex_copy"}:
            if template.ndim == 0 or template.shape[-1] != 2:
                return None
            output_shape = template.shape[:-1]
            output_stride = tuple(int(value) // 2 for value in template.stride()[:-1])
        elif name in {"view_as_real", "view_as_real_copy"}:
            output_shape = template.shape + (2,)
            output_stride = tuple(int(value) * 2 for value in template.stride()) + (1,)
        elif name in {"atleast_1d", "atleast_2d", "atleast_3d"}:
            minimum = int(name[-2])
            if template.ndim >= minimum:
                output_shape = template.shape
            elif name == "atleast_1d":
                output_shape = (1,)
            elif name == "atleast_2d":
                output_shape = (1, 1) if template.ndim == 0 else (1,) + template.shape
            else:
                output_shape = (
                    (1, 1, 1)
                    if template.ndim == 0
                    else (1,) + template.shape + (1,)
                    if template.ndim == 1
                    else template.shape + (1,)
                )
        elif name == "ravel":
            output_shape = (math.prod(template.shape),)
        elif name == "flatten":
            start = int(_value_at(args, kwargs, 0, "start_dim", 0))
            end = int(_value_at(args, kwargs, 1, "end_dim", -1))
            start = start + template.ndim if start < 0 else start
            end = end + template.ndim if end < 0 else end
            if start < 0 or end >= template.ndim or start > end:
                raise ValueError("invalid flatten dimensions")
            output_shape = template.shape[:start] + (math.prod(template.shape[start : end + 1]),) + template.shape[end + 1 :]
            mapping = tuple(range(start)) + (start,) + tuple(range(end + 1, template.ndim))
            placements = tuple(
                Shard(mapping.index(item.dim))
                if _is_shard_like(item) and item.dim in mapping and item.dim not in range(start, end + 1)
                else Replicate() if _is_shard_like(item) else item
                for item in template.placements
            )
        elif name == "unsqueeze":
            dim = int(_value_at(args, kwargs, 0, "dim", 0))
            if dim < 0:
                dim += template.ndim + 1
            if dim < 0 or dim > template.ndim:
                raise IndexError("unsqueeze dimension is outside the output rank")
            output_shape = template.shape[:dim] + (1,) + template.shape[dim:]
            placements = tuple(
                Shard(item.dim + (item.dim >= dim)) if _is_shard_like(item) else item
                for item in template.placements
            )
        elif name in {"squeeze", "squeeze_", "squeeze_copy", "squeeze_dims", "squeeze_dims_"}:
            dim = _value_at(args, kwargs, 0, "dim", None)
            if name in {"squeeze_dims", "squeeze_dims_"}:
                dim = _value_at(args, kwargs, 0, "dims", dim)
            if dim is None:
                reduced = {
                    index for index, size in enumerate(template.shape) if size == 1
                }
            else:
                dims = (dim,) if isinstance(dim, int) else tuple(dim)
                reduced = {_normalize_dim(item, template.ndim) for item in dims}
                reduced = {
                    index for index in reduced if template.shape[index] == 1
                }
            output_shape = tuple(
                size for index, size in enumerate(template.shape) if index not in reduced
            )
            placements = tuple(
                Replicate()
                if _is_shard_like(item) and item.dim in reduced
                else Shard(item.dim - sum(value < item.dim for value in reduced))
                if _is_shard_like(item)
                else item
                for item in template.placements
            )
        elif name in {"expand", "expand_copy", "expand_as", "broadcast_to"}:
            target = (
                values[-1].shape
                if name == "expand_as"
                else _size_shape_arg(args, kwargs, "size", "shape")
            )
            target = _normalize_shape_arg(target)
            if len(target) < template.ndim:
                raise ValueError("expanded shape cannot have fewer dimensions")
            offset = len(target) - template.ndim
            output_shape = tuple(
                template.shape[index - offset]
                if value == -1 and index >= offset
                else int(value)
                for index, value in enumerate(target)
            )
            placements = _broadcast_placements([template], output_shape)
        elif name in {"repeat", "tile"}:
            repeats = _size_shape_arg(args, kwargs, "repeats", "dims", "size")
            repeats = _normalize_shape_arg(repeats)
            if len(repeats) < template.ndim:
                repeats = (1,) * (template.ndim - len(repeats)) + repeats
            padded_shape = (1,) * (len(repeats) - template.ndim) + template.shape
            output_shape = tuple(
                int(size) * int(repeat)
                for size, repeat in zip(padded_shape, repeats)
            )
            placements = _broadcast_placements([template], output_shape)
        elif name == "multinomial":
            num_samples = int(_value_at(args, kwargs, 0, "num_samples", 0))
            if template.ndim == 0 or num_samples < 0:
                return None
            output_shape = template.shape[:-1] + (num_samples,)
            placements = tuple(
                Replicate()
                if _is_shard_like(item) and item.dim == template.ndim - 1
                else item
                for item in template.placements
            )
        elif name == "select":
            dim = _normalize_dim(_value_at(args, kwargs, 0, "dim", 0), template.ndim)
            output_shape = tuple(
                size for index, size in enumerate(template.shape) if index != dim
            )
            placements = tuple(
                Replicate()
                if _is_shard_like(item) and item.dim == dim
                else Shard(item.dim - (item.dim > dim))
                if _is_shard_like(item)
                else item
                for item in template.placements
            )
        elif name == "index_select":
            dim = _normalize_dim(_value_at(args, kwargs, 0, "dim", 0), template.ndim)
            index = _value_at(args, kwargs, 1, "index")
            index_shape = _shape(index)
            if index_shape is None or len(index_shape) != 1:
                return None
            output_shape = (
                template.shape[:dim]
                + (index_shape[0],)
                + template.shape[dim + 1 :]
            )
            placements = tuple(
                Replicate()
                if _is_shard_like(item) and item.dim == dim
                else item
                for item in template.placements
            )
        elif name == "gather":
            index = _value_at(args, kwargs, 1, "index")
            output_shape = _shape(index)
            if output_shape is None:
                return None
            placements = template.placements
        elif name in {
            "scatter",
            "scatter_add",
            "index_add",
            "index_copy",
            "index_fill",
            "index_put",
            "index_reduce",
        }:
            output_shape = template.shape
        elif name == "take":
            index = _value_at(args, kwargs, 0, "index")
            index_shape = _shape(index)
            if index_shape is None:
                return None
            output_shape = index_shape
            placements = tuple(Replicate() for _ in template.placements)
        elif name == "take_along_dim":
            index = _value_at(args, kwargs, 0, "indices")
            output_shape = _shape(index)
            if output_shape is None:
                return None
        elif name == "index":
            index_values = _value_at(args, kwargs, 0, "indices")
            index_values = (
                index_values
                if isinstance(index_values, (tuple, list))
                else (index_values,)
            )
            index_values = tuple(index_values) + (slice(None),) * max(
                0, template.ndim - len(index_values)
            )
            indexed_dims: list[int] = []
            index_shapes: list[tuple[int, ...]] = []
            basic_output: list[int] = []
            removed_dims: set[int] = set()
            for dim, index_value in enumerate(index_values[: template.ndim]):
                if isinstance(index_value, int):
                    removed_dims.add(dim)
                elif isinstance(index_value, slice):
                    start, stop, step = index_value.indices(template.shape[dim])
                    basic_output.append(len(range(start, stop, step)))
                elif _is_tensor(index_value):
                    indexed_dims.append(dim)
                    index_shapes.append(_shape(index_value) or ())
                else:
                    basic_output.append(template.shape[dim])
            if index_shapes:
                advanced_shape = _broadcast_shape(index_shapes)
                contiguous = all(
                    indexed_dims[index + 1] - indexed_dims[index] == 1
                    for index in range(len(indexed_dims) - 1)
                )
                if contiguous:
                    first = indexed_dims[0]
                    output_parts: list[int] = []
                    for dim in range(template.ndim):
                        if dim == first:
                            output_parts.extend(advanced_shape)
                        if dim in indexed_dims:
                            continue
                        if dim in removed_dims:
                            continue
                        value = index_values[dim]
                        if isinstance(value, slice):
                            start, stop, step = value.indices(template.shape[dim])
                            output_parts.append(len(range(start, stop, step)))
                        else:
                            output_parts.append(template.shape[dim])
                    output_shape = tuple(output_parts)
                else:
                    output_shape = advanced_shape + tuple(
                        template.shape[dim]
                        for dim in range(template.ndim)
                        if dim not in indexed_dims and dim not in removed_dims
                    )
                placements = tuple(Replicate() for _ in template.placements)
            else:
                output_shape = tuple(basic_output)
                placements = tuple(
                    Replicate()
                    if _is_shard_like(item) and item.dim in removed_dims
                    else Shard(
                        item.dim
                        - sum(removed < item.dim for removed in removed_dims)
                    )
                    if _is_shard_like(item)
                    else item
                    for item in template.placements
                )
        elif name in {"cat", "stack"}:
            sequence = _first_tensor_sequence(args[0] if args else kwargs.get("tensors"))
            sequence = sequence or values
            dim_index = int(_value_at(args, kwargs, 1, "dim", 0))
            if name == "stack":
                dim_index = dim_index if dim_index >= 0 else dim_index + sequence[0].ndim + 1
                output_shape = sequence[0].shape[:dim_index] + (len(sequence),) + sequence[0].shape[dim_index:]
                placements = tuple(
                    Shard(item.dim + (item.dim >= dim_index)) if _is_shard_like(item) else item
                    for item in template.placements
                )
            else:
                dim_index = _normalize_dim(dim_index, sequence[0].ndim)
                shape = list(sequence[0].shape)
                shape[dim_index] = sum(item.shape[dim_index] for item in sequence)
                output_shape = tuple(shape)
                if any(_is_shard_like(item) and item.dim == dim_index for item in template.placements):
                    placements = tuple(
                        Replicate() if _is_shard_like(item) and item.dim == dim_index else item
                        for item in template.placements
                    )
        elif name in self._REDUCTION_NAMES:
            dim = _value_at(args, kwargs, 0, "dim", None)
            keepdim = bool(_value_at(args, kwargs, 1, "keepdim", False))
            if dim is None:
                reduced = set(range(template.ndim))
            else:
                dims = (dim,) if isinstance(dim, int) else tuple(dim)
                reduced = {_normalize_dim(item, template.ndim) for item in dims}
            output_shape = tuple(
                1 if index in reduced and keepdim else size
                for index, size in enumerate(template.shape)
                if keepdim or index not in reduced
            )
            reduction = (
                "avg" if name == "mean" else
                "product" if name == "prod" else
                None if name in {"all", "any", "var", "std", "norm"} else name
            )
            placements = tuple(
                Partial(reduction)
                if _is_shard_like(item) and item.dim in reduced and reduction in Partial.ALL_REDUCE_OPS
                else Replicate()
                if _is_shard_like(item) and item.dim in reduced
                else item
                for item in template.placements
            )
        elif name in {"matmul", "mm", "bmm", "addmm", "baddbmm", "linear"}:
            bias = None
            if name in {"addmm", "baddbmm"} and len(args) >= 3:
                if not isinstance(args[1], DTensor) or not isinstance(args[2], DTensor):
                    return None
                bias = args[0] if isinstance(args[0], DTensor) else None
                left, right = args[1], args[2]
            elif name in {"addmm", "baddbmm"}:
                if len(values) < 2:
                    return None
                left, right = values[-2:]
            elif name == "linear":
                if len(values) < 2:
                    return None
                left, right = values[:2]
                bias = values[2] if len(values) >= 3 else None
            elif len(values) >= 2:
                left, right = values[:2]
            else:
                return None
            from ._ops._matrix_ops import linear_single_dim_strategy, mm_single_dim_strategy

            if name == "linear":
                output_shape = left.shape[:-1] + (right.shape[0],)
            else:
                a, b = left.shape, right.shape
                if len(a) == 1 and len(b) == 1:
                    output_shape = ()
                elif len(a) == 1:
                    output_shape = _broadcast_shape([a[:-1], b[:-2]]) + (b[-1],)
                elif len(b) == 1:
                    output_shape = _broadcast_shape([a[:-2], b[:-1]]) + (a[-2],)
                else:
                    output_shape = _broadcast_shape([a[:-2], b[:-2]]) + (a[-2], b[-1])
            matrix_spec = (
                linear_single_dim_strategy(left, right, bias=bias)
                if name == "linear"
                else mm_single_dim_strategy(left, right, bias=bias)
            )
            placements = tuple(matrix_spec.placements)
        elif name in {"topk", "sort", "argsort"}:
            dim = int(_value_at(args, kwargs, 1, "dim", -1))
            dim = _normalize_dim(dim, template.ndim)
            if name == "topk":
                k = int(_value_at(args, kwargs, 0, "k", 0))
                output_shape = template.shape[:dim] + (k,) + template.shape[dim + 1 :]
                placements = tuple(
                    Replicate() if _is_shard_like(item) and item.dim == dim else item
                    for item in template.placements
                )
            else:
                output_shape = template.shape
        else:
            if local_shape == template.to_local().shape:
                output_shape = template.shape
            elif all(isinstance(item, Replicate) for item in template.placements):
                output_shape = local_shape

        if output_shape is None:
            return None
        if output_stride is None:
            output_stride = template.stride() if output_shape == template.shape else _contiguous_strides(output_shape)
        return tuple(output_shape), tuple(placements), tuple(output_stride)

    def _wrap_result(
        self,
        result: Any,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        values: list[DTensor],
        propagated: Any = None,
        index: int | None = None,
    ) -> Any:
        if isinstance(result, tuple):
            return tuple(
                self._wrap_result(item, name, args, kwargs, values, propagated, item_index)
                for item_index, item in enumerate(result)
            )
        if isinstance(result, list):
            return [
                self._wrap_result(item, name, args, kwargs, values, propagated, item_index)
                for item_index, item in enumerate(result)
            ]
        if not _is_tensor(result) or not values:
            return result
        spec = _output_spec(propagated, index)
        canonical_args, canonical_kwargs = _canonicalize_operation_arguments(
            name, args, kwargs
        )
        inferred = self._infer(
            result,
            name,
            _operation_args(canonical_args),
            canonical_kwargs,
            values,
        )
        if inferred is None:
            from ._dtensor_spec import DTensorSpec

            if isinstance(spec, DTensorSpec) and spec.tensor_meta is not None:
                return DTensor(
                    result,
                    spec.mesh,
                    spec.placements,
                    shape=spec.shape,
                    stride=spec.stride,
                )
        if inferred is None:
            return result
        shape, placements, stride = inferred
        if spec is not None and hasattr(spec, "placements"):
            placements = tuple(spec.placements)
            meta = getattr(spec, "tensor_meta", None)
            if meta is not None:
                shape = tuple(int(item) for item in meta.shape)
                stride = tuple(int(item) for item in meta.stride)
        return DTensor(
            result,
            values[0].device_mesh,
            placements,
            shape=shape,
            stride=stride,
        )

    def dispatch_function(
        self,
        operation: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        call_kwargs = self._normalize_public_kwargs(operation, dict(kwargs or {}))
        call_args = tuple(args)
        values = _dtensors((call_args, call_kwargs))
        handler = None
        if values:
            handler = self._custom_op_handlers.get(operation)
            if handler is None:
                handler = self._custom_op_handlers.get(_operation_name(operation))
            if (
                handler is not None
                and _operation_name(operation) in {"max", "min"}
            ):
                from ._nonlinear_redux import is_dim_reduction_call

                if not is_dim_reduction_call(call_args, call_kwargs):
                    handler = None
        self._validate(
            values,
            (call_args, call_kwargs),
            allow_non_scalar_plain=(
                handler is not None or self._allows_plain_auxiliary(operation)
            ),
        )
        if handler is not None:
            return handler(operation, call_args, call_kwargs)
        prepared_args = call_args
        prepared_kwargs = call_kwargs
        if values and _operation_name(operation) in {"reshape", "view"}:
            template = values[0]
            metadata_args = _operation_args(call_args)
            target = _view_shape_arg(metadata_args, call_kwargs, "view")
            output_shape = _normalize_shape_arg(target, math.prod(template.shape))
            input_placements, _ = _view_layout(template, output_shape)
            if input_placements != template.placements:
                replacement = template.redistribute(placements=input_placements)
                prepared_args = _replace_identity(call_args, template, replacement)
                prepared_kwargs = _replace_identity(call_kwargs, template, replacement)
        propagated = self._propagate(operation, call_args, call_kwargs)
        prepared_args, prepared_kwargs = self._apply_redistribution(
            prepared_args, prepared_kwargs, propagated
        )
        if values and _operation_name(operation) in {
            "reshape",
            "view",
            "view_copy",
            "_unsafe_view",
        }:
            local_view_args, local_view_kwargs = _local_view_call(
                _operation_name(operation),
                tuple(prepared_args[1:]),
                prepared_kwargs,
                values[0],
                _output_spec(propagated),
            )
            prepared_args = (prepared_args[0],) + local_view_args
            prepared_kwargs = local_view_kwargs
        local_tensor = unwrap_dtensor(prepared_args[0]) if prepared_args else None
        template = values[0] if values else None
        with (
            self._random_context(operation, template, local_tensor)
            if template is not None
            else contextlib.nullcontext()
        ):
            canonical_local_args, canonical_local_kwargs = (
                _canonicalize_operation_arguments(
                    operation, prepared_args, prepared_kwargs
                )
            )
            if _operation_name(operation) == "index" and canonical_local_args:
                local_index = unwrap_dtensor(canonical_local_args[1])
                if isinstance(local_index, list):
                    local_index = tuple(local_index)
                local_result = unwrap_dtensor(canonical_local_args[0]).__getitem__(
                    local_index
                )
            else:
                local_operation = operation
                if not callable(local_operation):
                    import tensorplay.functional as functional

                    local_operation = getattr(
                        functional, _operation_name(local_operation)
                    )
                local_result = local_operation(
                    *unwrap_dtensor(prepared_args),
                    **unwrap_dtensor(prepared_kwargs),
                )
        return self._wrap_result(
            local_result,
            _operation_name(operation),
            call_args,
            call_kwargs,
            values,
            propagated,
        )

    def dispatch_method(
        self,
        receiver: DTensor,
        name: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        call_args = tuple(args)
        call_kwargs = dict(kwargs or {})
        values = _dtensors((receiver, call_args, call_kwargs))
        handler = self._custom_op_handlers.get(name)
        if handler is not None and name in {"max", "min"}:
            from ._nonlinear_redux import is_dim_reduction_call

            if not is_dim_reduction_call((receiver,) + call_args, call_kwargs):
                handler = None
        if handler is not None:
            return handler(name, (receiver,) + call_args, call_kwargs)
        self._validate(
            values,
            (receiver, call_args, call_kwargs),
            allow_non_scalar_plain=self._allows_plain_auxiliary(
                "index" if name == "__getitem__" else name
            ),
        )
        prepared_receiver = receiver
        if name in {"reshape", "view"}:
            target = _view_shape_arg(call_args, call_kwargs, name)
            output_shape = _normalize_shape_arg(target, math.prod(receiver.shape))
            input_placements, _ = _view_layout(receiver, output_shape)
            if input_placements != receiver.placements:
                prepared_receiver = receiver.redistribute(placements=input_placements)
        dispatch_name = "index" if name == "__getitem__" else name
        method = getattr(prepared_receiver.to_local(), name)
        propagated = self._propagate(
            dispatch_name, (receiver,) + call_args, call_kwargs
        )
        redistributed_args, redistributed_kwargs = self._apply_redistribution(
            (prepared_receiver,) + call_args, call_kwargs, propagated
        )
        prepared_receiver = redistributed_args[0]
        call_args = redistributed_args[1:]
        call_kwargs = redistributed_kwargs
        local_call_args, local_call_kwargs = _local_view_call(
            name,
            call_args,
            call_kwargs,
            receiver,
            _output_spec(propagated),
        )
        method = getattr(prepared_receiver.to_local(), name)
        with self._random_context(name, receiver, prepared_receiver.to_local()):
            local_result = method(
                *unwrap_dtensor(local_call_args), **unwrap_dtensor(local_call_kwargs)
            )
        return self._wrap_result(
            local_result,
            dispatch_name,
            call_args,
            call_kwargs,
            values,
            propagated,
        )

    def dispatch(
        self,
        operation: Any,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.dispatch_function(operation, args, kwargs)

    def wrap_result(
        self,
        result: Any,
        values: tuple[Any, ...],
        name: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        templates = _dtensors(values)
        return self._wrap_result(result, name, args, dict(kwargs or {}), templates)

    def __call__(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        return self.dispatch_function(operation, args, kwargs)
