"""Operation dispatch and metadata propagation for distributed tensors."""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Mapping
from typing import Any, Callable

from ._api import DTensor
from .placement_types import Partial, Placement, Replicate, Shard, _is_shard_like

__all__ = ["OpDispatcher", "unwrap_dtensor", "wrap_dtensor"]


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
        from ._ops._view_ops import _view_groups, propagate_shape_and_sharding

        mesh_shape = tuple(
            int(value.device_mesh.size(index))
            for index in range(len(value.placements))
        )
        rule = _view_groups(value.shape, output_shape)
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

    def __init__(self) -> None:
        self._rules: dict[Any, Callable[..., Any]] = {}
        from ._sharding_prop import ShardingPropagator

        self.sharding_propagator = ShardingPropagator()

    def register(self, operation: Any, rule: Callable[..., Any]) -> Callable[..., Any]:
        self._rules[operation] = rule
        return rule

    def _validate(self, values: list[DTensor], original: Any) -> None:
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
            if int(dim) != 0 and int(numel) != 1:
                raise RuntimeError(
                    "distributed operations require plain tensor operands to be scalar"
                )

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
        if (
            operation not in self.sharding_propagator.op_to_rules
            and operation not in self.sharding_propagator.op_strategy_funcs
        ):
            _, global_rule = self.sharding_propagator._global_rule(operation)
            if global_rule is None:
                return None
        from ._op_schema import OpSchema

        schema = OpSchema(operation, _spec_tree(args), _spec_tree(kwargs))
        result = self.sharding_propagator.propagate_op_sharding_non_cached(schema)
        if result is None and name != operation:
            schema = OpSchema(name, _spec_tree(args), _spec_tree(kwargs))
            result = self.sharding_propagator.propagate_op_sharding_non_cached(schema)
        return result

    @staticmethod
    def _apply_redistribution(
        args: tuple[Any, ...], kwargs: dict[str, Any], propagated: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
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
        elif name in {"permute", "movedim"}:
            if name == "permute":
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
                target = _value_at(args, kwargs, 0, "shape", ())
            output_shape = _normalize_shape_arg(target, math.prod(template.shape))
            _, placements = _view_layout(template, output_shape)
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
        elif name == "squeeze":
            dim = _value_at(args, kwargs, 0, "dim", None)
            reduced = {_normalize_dim(dim, template.ndim)} if dim is not None else {
                index for index, size in enumerate(template.shape) if size == 1
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
        elif name in {"expand", "expand_as", "repeat"}:
            target = values[-1].shape if name == "expand_as" else _value_at(args, kwargs, 0, "size", ())
            output_shape = _normalize_shape_arg(target)
            placements = _broadcast_placements([template], output_shape)
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
        inferred = self._infer(
            result, name, _operation_args(args), kwargs, values
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
        self._validate(values, (call_args, call_kwargs))
        prepared_args = call_args
        prepared_kwargs = call_kwargs
        if values and _operation_name(operation) in {"reshape", "view"}:
            template = values[0]
            metadata_args = _operation_args(call_args)
            target = _value_at(metadata_args, call_kwargs, 0, "shape", ())
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
        local_result = operation(
            *unwrap_dtensor(prepared_args), **unwrap_dtensor(prepared_kwargs)
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
        self._validate(values, (receiver, call_args, call_kwargs))
        prepared_receiver = receiver
        if name in {"reshape", "view"}:
            target = _value_at(call_args, call_kwargs, 0, "shape", ())
            output_shape = _normalize_shape_arg(target, math.prod(receiver.shape))
            input_placements, _ = _view_layout(receiver, output_shape)
            if input_placements != receiver.placements:
                prepared_receiver = receiver.redistribute(placements=input_placements)
        method = getattr(prepared_receiver.to_local(), name)
        propagated = self._propagate(name, (receiver,) + call_args, call_kwargs)
        redistributed_args, redistributed_kwargs = self._apply_redistribution(
            (prepared_receiver,) + call_args, call_kwargs, propagated
        )
        prepared_receiver = redistributed_args[0]
        call_args = redistributed_args[1:]
        call_kwargs = redistributed_kwargs
        method = getattr(prepared_receiver.to_local(), name)
        local_result = method(
            *unwrap_dtensor(tuple(call_args)), **unwrap_dtensor(call_kwargs)
        )
        return self._wrap_result(
            local_result,
            _operation_name(name),
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
