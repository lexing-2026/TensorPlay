"""Utilities for checking placement combinations against concrete results."""

from __future__ import annotations

import copy
import fnmatch
import itertools
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import tensorplay as tp

from ..._local_tensor import LocalTensor, LocalTensorMode
from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OpSpec, OpStrategy
from ..placement_types import Partial, Placement, Replicate, Shard
from .single_dim_strategy import _ShardingPlaceholder

ComboKey = tuple[tuple[str, ...], tuple[str, ...]]
PlacementCombination = tuple[tuple[Placement, ...], tuple[Placement, ...]]
PARTIAL_REDUCE_OPS = ("sum", "avg", "min", "max")

__all__ = [
    "ComparisonStats",
    "Discrepancy",
    "PlacementCombination",
    "_OperationCapture",
    "_FalsePositiveMitigations",
    "_checkerboard_mask",
    "_compare_outputs",
    "_compare_rules",
    "_create_partial_input",
    "_extract_rules_from_op_strategy",
    "_is_tensor_output",
    "_prepare_false_positive_mitigations",
    "_run_op_on_sample",
    "_shard_tensors",
    "_to_ground_truth",
    "_tree_map",
    "create_fully_negated_sample",
    "extract_tensors_from_args",
    "extract_tensors_from_sample",
    "get_1d_input_placements_for_tensor",
    "get_1d_output_placements_for_tensor",
    "get_operation_for_sample",
    "get_registered_op_names",
    "has_any_partial",
    "has_pmin_pmax",
    "is_fully_replicated",
    "is_trivial_shard",
    "normalize_combo_key",
    "normalize_placement",
    "normalize_placement_str",
    "negate_all_tensors",
    "parse_placement",
    "query_single_dim_strategy",
    "report_registrations",
    "resolve_op_names",
    "validate_operation_combination",
    "validate_combination",
]


@dataclass
class Discrepancy:
    input_placements: tuple[str, ...]
    output_placements: tuple[str, ...]
    sample_idx: int
    input_shapes: tuple[tuple[int, ...], ...]
    discrepancy_type: str
    error_msg: str = ""
    scalar_args: tuple[Any, ...] = ()
    scalar_kwargs: dict[str, Any] = field(default_factory=dict)
    operation: Any = None
    variant: str = ""
    sample: Any = None


@dataclass
class ComparisonStats:
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: list[Discrepancy] = field(default_factory=list)
    false_negatives: list[Discrepancy] = field(default_factory=list)
    total_samples: int = 0
    total_combinations: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    no_dtensor_support: bool = False
    true_positives_by_op: dict[str, int] = field(default_factory=dict)


@dataclass
class _FalsePositiveMitigations:
    negated_sample: Any = None
    negated_tensors: list[tuple[str, Any]] | None = None
    negated_ground_truth: Any = None
    non_rounded_sample: Any = None
    non_rounded_ground_truth: Any = None
    non_rounded_negated_sample: Any = None
    non_rounded_negated_tensors: list[tuple[str, Any]] | None = None
    non_rounded_negated_ground_truth: Any = None


def parse_placement(value: str) -> Placement | None:
    value = value.strip()
    if value == "R":
        return Replicate()
    match = re.fullmatch(r"S\((\d+)\)", value)
    if match is not None:
        return Shard(int(match.group(1)))
    match = re.fullmatch(r"P\((\w+)\)", value)
    if match is not None:
        try:
            return Partial(match.group(1))
        except ValueError:
            return None
    return None


def is_fully_replicated(placements: tuple[Placement, ...]) -> bool:
    return all(isinstance(placement, Replicate) for placement in placements)


def is_trivial_shard(placement: Placement, tensor_shape: tuple[int, ...]) -> bool:
    return (
        isinstance(placement, Shard)
        and placement.dim < len(tensor_shape)
        and tensor_shape[placement.dim] == 1
    )


def normalize_placement(placement: Placement, tensor_shape: tuple[int, ...]) -> Placement:
    return Replicate() if is_trivial_shard(placement, tensor_shape) else placement


def normalize_placement_str(value: str, tensor_shape: tuple[int, ...]) -> str:
    placement = parse_placement(value)
    if placement is None:
        return value
    normalized = normalize_placement(placement, tensor_shape)
    return "R" if isinstance(normalized, Replicate) else value


def normalize_combo_key(
    combo_key: ComboKey,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> ComboKey:
    input_placements, output_placements = combo_key
    return (
        tuple(
            normalize_placement_str(value, shape)
            for value, shape in zip(input_placements, input_shapes)
        ),
        tuple(
            normalize_placement_str(value, shape)
            for value, shape in zip(output_placements, output_shapes)
        ),
    )


def get_1d_input_placements_for_tensor(
    tensor: Any, include_partial: bool = False
) -> list[Placement]:
    placements: list[Placement] = [Replicate()]
    placements.extend(Shard(dim) for dim in range(int(tensor.ndim)))
    if include_partial and tensor.dtype != tp.bool:
        placements.extend(Partial(reduce_op) for reduce_op in PARTIAL_REDUCE_OPS)
    return placements


def get_1d_output_placements_for_tensor(tensor: Any) -> list[Placement]:
    return get_1d_input_placements_for_tensor(tensor, include_partial=True)


def _tree_map(function: Callable[[Any], Any], value: Any) -> Any:
    if isinstance(value, Mapping):
        return type(value)(
            (key, _tree_map(function, child)) for key, child in value.items()
        )
    if isinstance(value, tuple):
        values = [_tree_map(function, child) for child in value]
        if hasattr(value, "_fields"):
            return type(value)(*values)
        return tuple(values)
    if isinstance(value, list):
        return [_tree_map(function, child) for child in value]
    return function(value)


def _sample_parts(sample: Any) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    return (
        getattr(sample, "input"),
        tuple(getattr(sample, "args", ())),
        dict(getattr(sample, "kwargs", {})),
    )


def _replace_sample(
    sample: Any,
    sample_input: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    try:
        return replace(sample, input=sample_input, args=args, kwargs=kwargs)
    except (TypeError, ValueError):
        result = copy.copy(sample)
        result.input = sample_input
        result.args = args
        result.kwargs = kwargs
        return result


def extract_tensors_from_sample(sample_input: Any) -> list[tuple[str, Any]]:
    tensors: list[tuple[str, Any]] = []
    index = 0

    def collect(value: Any) -> Any:
        nonlocal index
        if isinstance(value, tp.Tensor):
            tensors.append((f"tensor_{index}", value))
            index += 1
        return value

    sample, args, kwargs = _sample_parts(sample_input)
    _tree_map(collect, sample)
    _tree_map(collect, args)
    _tree_map(collect, kwargs)
    return tensors


def _checkerboard_mask(
    tensor: Any, tensor_idx: int = 0, mask_shift: int = 0
) -> Any:
    ndim = int(tensor.ndim)
    if ndim == 0:
        return tp.tensor(
            [(tensor_idx + mask_shift) % 2 == 0],
            dtype=tp.bool,
            device=tensor.device,
        )
    coordinates = tp.zeros(tensor.shape, dtype=tp.int64, device=tensor.device)
    for dim, size in enumerate(tensor.shape):
        shape = [1] * ndim
        shape[dim] = int(size)
        coordinates = coordinates + tp.arange(
            int(size), dtype=tp.int64, device=tensor.device
        ).reshape(shape)
    return ((coordinates + tensor_idx + mask_shift) % 2 == 0).flatten()


def _create_partial_input(
    tensor: Any,
    placement: Partial,
    world_size: int,
    tensor_idx: int = 0,
    mask_shift: int = 0,
) -> LocalTensor:
    reduce_op = placement.reduce_op
    local_tensors: dict[int, Any] = {}
    if reduce_op in ("sum", "avg"):
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)
        flat = tensor.flatten()
        signs = tp.ones_like(flat)
        signs[_checkerboard_mask(tensor, tensor_idx, mask_shift)] = -1.0
        offset = (flat.abs() + 1.0) * signs
        scale = world_size if reduce_op == "avg" else 1
        for rank in range(world_size):
            if rank == 0:
                local_tensors[rank] = tensor.clone() * base_ratio * scale + offset.reshape(tensor.shape)
            else:
                local_tensors[rank] = tensor.clone() * (
                    (1 - base_ratio) / (world_size - 1)
                ) * scale - offset.reshape(tensor.shape) / (world_size - 1)
    elif reduce_op in ("min", "max"):
        flat = tensor.flatten()
        value_range = (flat.max() - flat.min()).item()
        offset_value = value_range * 2 + 1
        if reduce_op == "max":
            offset_value = -offset_value
        mask = _checkerboard_mask(tensor, tensor_idx, mask_shift)
        for rank in range(world_size):
            if rank == 0:
                offset = tp.where(
                    mask,
                    tp.zeros_like(flat),
                    tp.full_like(flat, offset_value),
                )
            else:
                offset = tp.where(
                    mask,
                    tp.full_like(flat, offset_value),
                    tp.zeros_like(flat),
                )
            local_tensors[rank] = (flat + offset).reshape(tensor.shape)
    else:
        local_tensors = {rank: tensor.clone() for rank in range(world_size)}
    return LocalTensor(local_tensors)


def _shard_tensors(
    tensors: list[tuple[str, Any]],
    input_placements: tuple[Placement, ...],
    world_size: int,
    mesh: Any,
    mask_shift: int = 0,
) -> list[LocalTensor]:
    del mesh
    result: list[LocalTensor] = []
    for tensor_idx, ((_, tensor), placement) in enumerate(
        zip(tensors, input_placements)
    ):
        if isinstance(placement, Partial):
            local_tensor = _create_partial_input(
                tensor, placement, world_size, tensor_idx, mask_shift
            )
        elif isinstance(placement, Replicate):
            local_tensor = LocalTensor(
                {rank: tensor.clone() for rank in range(world_size)}
            )
        elif isinstance(placement, Shard):
            dim = placement.dim if placement.dim >= 0 else placement.dim + tensor.ndim
            chunks = tp.tensor_split(tensor, world_size, dim=dim)
            local_tensor = LocalTensor(
                {rank: chunks[rank].contiguous().clone() for rank in range(world_size)}
            )
        else:
            raise TypeError(f"unsupported placement {placement!r}")
        result.append(local_tensor)
    return result


def _compare_outputs(
    local_output: Any,
    ground_truth: Any,
    output_placements: tuple[Placement, ...],
    mesh: Any,
    world_size: int,
) -> tuple[bool, str]:
    local_outputs = (
        list(local_output)
        if isinstance(local_output, (list, tuple))
        else [local_output]
    )
    ground_truths = (
        list(ground_truth) if isinstance(ground_truth, list) else [ground_truth]
    )
    if len(local_outputs) != len(ground_truths):
        return False, f"output count mismatch: got {len(local_outputs)}, expected {len(ground_truths)}"
    if len(local_outputs) != len(output_placements):
        return False, f"output placement count mismatch: got {len(local_outputs)}, expected {len(output_placements)}"
    for index, (local_value, expected, placement) in enumerate(
        zip(local_outputs, ground_truths, output_placements)
    ):
        if not isinstance(local_value, LocalTensor):
            return False, f"output[{index}] is not a LocalTensor"
        if isinstance(placement, Replicate):
            values = [local_value._local_tensors[rank] for rank in range(world_size)]
            if not all(
                tp.allclose(values[0], value, atol=1e-5, rtol=1e-5)
                for value in values[1:]
            ):
                return False, f"replicated output[{index}] differs across ranks"
        output_dt = DTensor.from_local(
            local_value,
            mesh,
            (placement,),
            shape=expected.shape,
            stride=expected.stride(),
        )
        full_value = output_dt.redistribute(mesh, (Replicate(),)).to_local()
        if isinstance(full_value, LocalTensor):
            full_value = full_value._local_tensors[0]
        if tuple(expected.shape) != tuple(full_value.shape):
            return False, f"shape mismatch[{index}]: expected {expected.shape}, got {full_value.shape}"
        if not tp.allclose(expected, full_value, atol=1e-5, rtol=1e-5, equal_nan=True):
            difference = (expected - full_value).abs().max().item()
            return False, f"value mismatch[{index}]: max_diff={difference:.6f}"
    return True, ""


def validate_combination(
    op: Callable[..., Any],
    sample_input: Any,
    tensors: list[tuple[str, Any]],
    combination: PlacementCombination,
    ground_truth: Any,
    world_size: int = 2,
    mesh: Any = None,
    mask_shift: int = 0,
) -> tuple[bool | None, str]:
    try:
        if mesh is None:
            from ..device_mesh import init_device_mesh

            device = getattr(tensors[0][1].device, "type", "cpu") if tensors else "cpu"
            mesh = init_device_mesh(device, (world_size,))
        for (_, tensor), placement in zip(tensors, combination[0]):
            if isinstance(placement, Shard):
                dim = placement.dim if placement.dim >= 0 else placement.dim + tensor.ndim
                if tensor.shape[dim] % world_size != 0:
                    return None, "uneven shard"
        local_tensors = _shard_tensors(
            tensors, combination[0], world_size, mesh, mask_shift
        )
        local_index = 0

        def replace_tensor(value: Any) -> Any:
            nonlocal local_index
            if isinstance(value, tp.Tensor):
                local = local_tensors[local_index]
                local_index += 1
                return local
            return value

        sample, args, kwargs = _sample_parts(sample_input)
        local_input = (
            replace_tensor(sample)
            if isinstance(sample, tp.Tensor)
            else _tree_map(replace_tensor, sample)
        )
        local_args = _tree_map(replace_tensor, args)
        local_kwargs = _tree_map(replace_tensor, kwargs)
        local_output = op(local_input, *local_args, **local_kwargs)
        return _compare_outputs(
            local_output, ground_truth, combination[1], mesh, world_size
        )
    except Exception as error:
        return False, f"exception: {type(error).__name__}: {error}"


def extract_tensors_from_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[str, Any]]:
    tensors: list[tuple[str, Any]] = []
    index = 0

    def collect(value: Any) -> Any:
        nonlocal index
        if isinstance(value, tp.Tensor):
            tensors.append((f"tensor_{index}", value))
            index += 1
        return value

    _tree_map(collect, args)
    _tree_map(collect, kwargs)
    return tensors


def validate_operation_combination(
    operation: Callable[..., Any],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    ground_truth: Any,
    combination: PlacementCombination,
    world_size: int,
    mesh: Any,
    mask_shift: int = 0,
) -> tuple[bool | None, str]:
    try:
        tensors = extract_tensors_from_args(captured_args, captured_kwargs)
        if not tensors:
            return False, "no tensor arguments"
        for (_, tensor), placement in zip(tensors, combination[0]):
            if isinstance(placement, Shard):
                dim = placement.dim if placement.dim >= 0 else placement.dim + tensor.ndim
                if tensor.shape[dim] % world_size != 0:
                    return None, "uneven shard"
        local_tensors = _shard_tensors(
            tensors, combination[0], world_size, mesh, mask_shift
        )
        local_index = 0

        def replace_tensor(value: Any) -> Any:
            nonlocal local_index
            if isinstance(value, tp.Tensor):
                local = local_tensors[local_index]
                local_index += 1
                return local
            return value

        local_args = _tree_map(replace_tensor, captured_args)
        local_kwargs = _tree_map(replace_tensor, captured_kwargs)
        local_output = operation(*local_args, **local_kwargs)
        return _compare_outputs(
            local_output, ground_truth, combination[1], mesh, world_size
        )
    except Exception as error:
        return False, f"exception: {type(error).__name__}: {error}"


def has_pmin_pmax(
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
) -> bool:
    return any(
        isinstance(placement, Partial)
        and placement.reduce_op in ("min", "max")
        for placement in (*input_placements, *output_placements)
    )


def has_any_partial(
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
) -> bool:
    return any(
        isinstance(placement, Partial)
        for placement in (*input_placements, *output_placements)
    )


def negate_all_tensors(tensors: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    return [(name, -tensor) for name, tensor in tensors]


def create_fully_negated_sample(sample: Any) -> Any:
    sample_input, args, kwargs = _sample_parts(sample)
    negate = lambda value: -value if isinstance(value, tp.Tensor) else value
    return _replace_sample(
        sample,
        _tree_map(negate, sample_input),
        _tree_map(negate, args),
        _tree_map(negate, kwargs),
    )


def _run_op_on_sample(op: Callable[..., Any], sample: Any) -> Any:
    sample_input, args, kwargs = _sample_parts(sample)
    if isinstance(sample_input, tp.Tensor):
        return op(sample_input, *args, **kwargs)
    return op(*sample_input, *args, **kwargs)


def _extract_rules_from_op_strategy(
    op_strategy: Any,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> set[ComboKey]:
    rules: set[ComboKey] = set()
    if not isinstance(op_strategy, OpStrategy):
        return rules
    for strategy in op_strategy.strategies:
        if strategy.input_specs is None:
            continue
        if isinstance(strategy.output_specs, (tuple, list)):
            if any(spec is None for spec in strategy.output_specs):
                continue
            output_placements = [
                spec.placements[0] for spec in strategy.output_specs
            ]
        else:
            output_placements = [
                strategy.output_spec.placements[0]
            ] * len(output_shapes)
        input_placements = tuple(
            spec.placements[0] for spec in strategy.input_specs
        )
        key: ComboKey = (
            tuple(str(placement) for placement in input_placements),
            tuple(str(placement) for placement in output_placements),
        )
        normalized = normalize_combo_key(key, input_shapes, output_shapes)
        if not is_fully_replicated(
            tuple(parse_placement(value) or Replicate() for value in normalized[0])
        ):
            rules.add(normalized)
    return rules


class _OperationCapture:
    def __init__(self, target_op_name: str = "") -> None:
        self.target_op_name = target_op_name.lower()
        self.all_ops: list[tuple[Any, tuple[Any, ...], dict[str, Any], Any]] = []
        self.best_match: Any = None
        self.best_match_args: tuple[Any, ...] | None = None
        self.best_match_kwargs: dict[str, Any] | None = None
        self.best_match_result: Any = None

    def __enter__(self) -> "_OperationCapture":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback

    def record(
        self,
        operation: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> None:
        self.all_ops.append((operation, args, kwargs, result))
        name = str(getattr(operation, "__name__", operation)).lower()
        if self.best_match is None or not self.target_op_name or self.target_op_name in name:
            self.best_match = operation
            self.best_match_args = args
            self.best_match_kwargs = kwargs
            self.best_match_result = result


def get_operation_for_sample(
    op: Callable[..., Any], sample: Any, op_name: str = ""
) -> _OperationCapture:
    capture = _OperationCapture(op_name)
    try:
        result = _run_op_on_sample(op, sample)
        sample_input, args, kwargs = _sample_parts(sample)
        capture.record(op, (sample_input, *args), kwargs, result)
    except Exception:
        pass
    return capture


def query_single_dim_strategy(
    op_overload: Any,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> list[list[Placement]] | None:
    propagator = DTensor._op_dispatcher.sharding_propagator
    strategy_func = propagator.op_single_dim_strategy_funcs.get(op_overload)
    if strategy_func is None:
        return None
    args_meta = tuple(
        TensorMeta(tuple(value.shape), tuple(value.stride()), value.dtype)
        if isinstance(value, tp.Tensor)
        else value
        for value in captured_args
    )
    kwargs_meta = {
        key: TensorMeta(tuple(value.shape), tuple(value.stride()), value.dtype)
        if isinstance(value, tp.Tensor)
        else value
        for key, value in captured_kwargs.items()
    }
    try:
        choices = strategy_func(op_overload, args_meta, kwargs_meta)
        return [
            [
                Shard(value.dim) if isinstance(value, _ShardingPlaceholder) else value
                for value in choice
            ]
            for choice in choices
        ]
    except Exception:
        return None


def _registered_operation_names() -> set[str]:
    propagator = DTensor._op_dispatcher.sharding_propagator
    result: set[str] = set()
    for table_name in (
        "op_to_rules",
        "op_strategy_funcs",
        "op_single_dim_strategy_funcs",
    ):
        for operation in getattr(propagator, table_name, {}):
            result.add(
                str(getattr(operation, "__name__", operation)).rsplit(".", 1)[-1]
            )
    return result


def resolve_op_names(patterns: list[str]) -> list[str]:
    names = sorted(_registered_operation_names())
    resolved: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for part in (value.strip() for value in pattern.split(",")):
            if not part:
                continue
            matches = (
                fnmatch.filter(names, part)
                if "*" in part or "?" in part
                else [part]
                if part in names
                else []
            )
            if not matches:
                raise ValueError(f"no registered operation matches {part!r}")
            for match in matches:
                if match not in seen:
                    resolved.append(match)
                    seen.add(match)
    return resolved


def get_registered_op_names() -> list[str]:
    return sorted(_registered_operation_names())


def _is_tensor_output(result: Any) -> bool:
    if isinstance(result, tp.Tensor):
        return True
    if isinstance(result, (list, tuple)):
        contains_tensor = any(isinstance(value, tp.Tensor) for value in result)
        all_tensors = all(isinstance(value, tp.Tensor) for value in result)
        if contains_tensor and not all_tensors:
            raise NotImplementedError("mixed tensor and non-tensor outputs are unsupported")
        return all_tensors
    return False


def _to_ground_truth(result: Any) -> Any:
    return result if isinstance(result, tp.Tensor) else list(result)


def _prepare_false_positive_mitigations(
    op: Callable[..., Any], sample: Any, tensors: list[tuple[str, Any]]
) -> _FalsePositiveMitigations:
    mitigations = _FalsePositiveMitigations()
    try:
        mitigations.negated_sample = create_fully_negated_sample(sample)
        mitigations.negated_tensors = negate_all_tensors(tensors)
        result = _run_op_on_sample(op, mitigations.negated_sample)
        if _is_tensor_output(result):
            mitigations.negated_ground_truth = _to_ground_truth(result)
        else:
            mitigations.negated_sample = None
    except Exception:
        mitigations.negated_sample = None
        mitigations.negated_tensors = None
    sample_input, args, kwargs = _sample_parts(sample)
    if "rounding_mode" not in kwargs:
        return mitigations
    try:
        non_rounded_kwargs = {
            key: value for key, value in kwargs.items() if key != "rounding_mode"
        }
        mitigations.non_rounded_sample = _replace_sample(
            sample, sample_input, args, non_rounded_kwargs
        )
        result = _run_op_on_sample(op, mitigations.non_rounded_sample)
        if not _is_tensor_output(result):
            mitigations.non_rounded_sample = None
        else:
            mitigations.non_rounded_ground_truth = _to_ground_truth(result)
            mitigations.non_rounded_negated_sample = create_fully_negated_sample(
                mitigations.non_rounded_sample
            )
            mitigations.non_rounded_negated_tensors = negate_all_tensors(tensors)
            result = _run_op_on_sample(op, mitigations.non_rounded_negated_sample)
            if _is_tensor_output(result):
                mitigations.non_rounded_negated_ground_truth = _to_ground_truth(result)
            else:
                mitigations.non_rounded_negated_sample = None
    except Exception:
        mitigations.non_rounded_sample = None
        mitigations.non_rounded_ground_truth = None
        mitigations.non_rounded_negated_sample = None
    return mitigations


def _query_dtensor_rules(
    operation: Any,
    tensors: list[tuple[str, Any]],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    world_size: int,
    verbose: bool = False,
) -> set[ComboKey]:
    propagator = DTensor._op_dispatcher.sharding_propagator
    rules: set[ComboKey] = set()
    single_dim = propagator.op_single_dim_strategy_funcs.get(operation)
    if single_dim is not None:
        choices = query_single_dim_strategy(operation, captured_args, captured_kwargs)
        if choices:
            for choice in choices:
                if len(choice) < len(output_shapes) + len(tensors):
                    continue
                key = (
                    tuple(str(value) for value in choice[len(output_shapes):len(output_shapes) + len(tensors)]),
                    tuple(str(value) for value in choice[:len(output_shapes)]),
                )
                normalized = normalize_combo_key(key, input_shapes, output_shapes)
                if not is_fully_replicated(
                    tuple(parse_placement(value) or Replicate() for value in normalized[0])
                ):
                    rules.add(normalized)
        return rules
    strategy_func = propagator.op_strategy_funcs.get(operation)
    if strategy_func is None:
        return rules
    try:
        from ..device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world_size,))
        strategies: dict[int, OpStrategy] = {}
        for _, tensor in tensors:
            specs = []
            for placement in get_1d_input_placements_for_tensor(
                tensor, include_partial=True
            ):
                spec = DTensorSpec(
                    mesh,
                    (placement,),
                    TensorMeta(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype),
                )
                specs.append(OpSpec(spec, ()))
            strategies[id(tensor)] = OpStrategy(specs)

        def to_strategy(value: Any) -> Any:
            return strategies[id(value)] if isinstance(value, tp.Tensor) else value

        schema = OpSchema(
            operation,
            _tree_map(to_strategy, captured_args),
            _tree_map(to_strategy, captured_kwargs),
        )
        output = strategy_func(schema)
        rules.update(_extract_rules_from_op_strategy(output, input_shapes, output_shapes))
    except Exception as error:
        if verbose:
            print(f"        rule query failed: {error}")
    return rules


def _validate_with_mitigations(
    op: Callable[..., Any],
    sample: Any,
    tensors: list[tuple[str, Any]],
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
    ground_truth: Any,
    world_size: int,
    mesh: Any,
    mitigations: _FalsePositiveMitigations,
) -> bool | None:
    combination = (input_placements, output_placements)
    valid, _ = validate_combination(
        op, sample, tensors, combination, ground_truth, world_size, mesh
    )
    if valid is None:
        return None
    if valid and has_any_partial(input_placements, output_placements):
        valid, _ = validate_combination(
            op, sample, tensors, combination, ground_truth, world_size, mesh, 1
        )
    if valid and mitigations.negated_sample and has_pmin_pmax(
        input_placements, output_placements
    ):
        if mitigations.negated_tensors is None or mitigations.negated_ground_truth is None:
            raise AssertionError("negated validation data is incomplete")
        valid, _ = validate_combination(
            op,
            mitigations.negated_sample,
            mitigations.negated_tensors,
            combination,
            mitigations.negated_ground_truth,
            world_size,
            mesh,
        )
    if valid and mitigations.non_rounded_sample and has_any_partial(
        input_placements, output_placements
    ):
        if mitigations.non_rounded_ground_truth is None:
            raise AssertionError("non-rounded validation data is incomplete")
        valid, _ = validate_combination(
            op,
            mitigations.non_rounded_sample,
            tensors,
            combination,
            mitigations.non_rounded_ground_truth,
            world_size,
            mesh,
        )
    if valid and mitigations.non_rounded_negated_sample and has_pmin_pmax(
        input_placements, output_placements
    ):
        if (
            mitigations.non_rounded_negated_tensors is None
            or mitigations.non_rounded_negated_ground_truth is None
        ):
            raise AssertionError("non-rounded negated data is incomplete")
        valid, _ = validate_combination(
            op,
            mitigations.non_rounded_negated_sample,
            mitigations.non_rounded_negated_tensors,
            combination,
            mitigations.non_rounded_negated_ground_truth,
            world_size,
            mesh,
        )
    return valid


@dataclass
class _OperationFalsePositiveMitigations:
    negated_args: tuple[Any, ...] | None = None
    negated_kwargs: dict[str, Any] | None = None
    negated_ground_truth: Any = None


def _negate_tensors_in_tree(value: Any) -> Any:
    return _tree_map(
        lambda item: -item if isinstance(item, tp.Tensor) else item,
        value,
    )


def _prepare_operation_mitigations(
    operation: Callable[..., Any],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> _OperationFalsePositiveMitigations:
    mitigations = _OperationFalsePositiveMitigations()
    try:
        mitigations.negated_args = _negate_tensors_in_tree(captured_args)
        mitigations.negated_kwargs = _negate_tensors_in_tree(captured_kwargs)
        result = operation(*mitigations.negated_args, **mitigations.negated_kwargs)
        if _is_tensor_output(result):
            mitigations.negated_ground_truth = _to_ground_truth(result)
        else:
            mitigations.negated_args = None
            mitigations.negated_kwargs = None
    except Exception:
        mitigations.negated_args = None
        mitigations.negated_kwargs = None
    return mitigations


def _validate_operation_with_mitigations(
    operation: Callable[..., Any],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
    ground_truth: Any,
    world_size: int,
    mesh: Any,
    mitigations: _OperationFalsePositiveMitigations,
) -> bool | None:
    combination = (input_placements, output_placements)
    valid, _ = validate_operation_combination(
        operation,
        captured_args,
        captured_kwargs,
        ground_truth,
        combination,
        world_size,
        mesh,
    )
    if valid is None:
        return None
    if valid and has_any_partial(input_placements, output_placements):
        valid, _ = validate_operation_combination(
            operation,
            captured_args,
            captured_kwargs,
            ground_truth,
            combination,
            world_size,
            mesh,
            mask_shift=1,
        )
    if (
        valid
        and mitigations.negated_args is not None
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.negated_kwargs is None or mitigations.negated_ground_truth is None:
            raise AssertionError("negated operation data is incomplete")
        valid, _ = validate_operation_combination(
            operation,
            mitigations.negated_args,
            mitigations.negated_kwargs,
            mitigations.negated_ground_truth,
            combination,
            world_size,
            mesh,
        )
    return valid


def _assert_keys_normalized(
    keys: set[ComboKey],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> None:
    for key in keys:
        if key != normalize_combo_key(key, input_shapes, output_shapes):
            raise AssertionError(f"un-normalized placement key: {key}")


def _check_ground_truth(result: Any) -> Any | None:
    if isinstance(result, (list, tuple)):
        if not all(isinstance(value, tp.Tensor) for value in result):
            return None
        ground_truth = list(result)
    elif isinstance(result, tp.Tensor):
        ground_truth = result
    else:
        return None
    first = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
    if first.numel() == 0:
        return None
    if bool((first == 0).all().item()):
        return None
    if bool(tp.isnan(first).all().item()):
        return None
    return ground_truth


def _validate_operation_for_sample(
    operation: Callable[..., Any],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    ground_truth: Any,
    world_size: int,
    incorrect_only: bool,
    verbose: bool,
    sample_idx: int,
    variant: str,
    stats: ComparisonStats,
    sample: Any = None,
) -> tuple[int, int]:
    tensors = extract_tensors_from_args(captured_args, captured_kwargs)
    if not tensors or any(0 in tuple(tensor.shape) for _, tensor in tensors):
        return 0, 0
    input_shapes = tuple(tuple(tensor.shape) for _, tensor in tensors)
    ground_truth_values = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    output_shapes = tuple(tuple(value.shape) for value in ground_truth_values)
    scalar_args = tuple(
        value for value in captured_args if not isinstance(value, tp.Tensor)
    )
    scalar_kwargs = {
        key: value
        for key, value in captured_kwargs.items()
        if not isinstance(value, tp.Tensor)
    }
    mitigations = _prepare_operation_mitigations(
        operation, captured_args, captured_kwargs
    )
    input_options = [
        get_1d_input_placements_for_tensor(tensor, include_partial=True)
        for _, tensor in tensors
    ]
    output_options = get_1d_output_placements_for_tensor(ground_truth_values[0])
    registered_rules = _query_dtensor_rules(
        operation,
        tensors,
        captured_args,
        captured_kwargs,
        input_shapes,
        output_shapes,
        world_size,
        verbose,
    )
    ground_truth_valid: set[ComboKey] = set()
    untestable: set[ComboKey] = set()
    total_combinations = 0
    from ..device_mesh import init_device_mesh

    device = getattr(tensors[0][1].device, "type", "cpu")
    with LocalTensorMode(frozenset(range(world_size))):
        mesh = init_device_mesh(device, (world_size,))
        if incorrect_only:
            combinations = []
            for key in registered_rules:
                input_values = tuple(parse_placement(value) for value in key[0])
                output_values = tuple(parse_placement(value) for value in key[1])
                if all(value is not None for value in (*input_values, *output_values)):
                    combinations.append((input_values, output_values, key))
        else:
            combinations = []
            for input_values in itertools.product(*input_options):
                if is_fully_replicated(input_values):
                    continue
                for output_value in output_options:
                    output_values = tuple(output_value for _ in output_shapes)
                    key = (
                        tuple(str(value) for value in input_values),
                        tuple(str(value) for value in output_values),
                    )
                    combinations.append((input_values, output_values, key))
        for input_values, output_values, key in combinations:
            total_combinations += 1
            valid = _validate_operation_with_mitigations(
                operation,
                captured_args,
                captured_kwargs,
                tuple(input_values),
                tuple(output_values),
                ground_truth,
                world_size,
                mesh,
                mitigations,
            )
            normalized = normalize_combo_key(key, input_shapes, output_shapes)
            if valid is None:
                untestable.add(normalized)
            elif valid and not is_fully_replicated(
                tuple(parse_placement(value) or Replicate() for value in normalized[0])
            ):
                ground_truth_valid.add(normalized)
    _compare_rules(
        ground_truth_valid,
        registered_rules,
        input_shapes,
        output_shapes,
        sample_idx,
        scalar_args,
        scalar_kwargs,
        operation,
        variant,
        stats,
        sample,
        untestable,
    )
    if verbose:
        print(
            f"      sample {sample_idx} [{operation}]: shapes={input_shapes} "
            f"valid={len(ground_truth_valid)} registered={len(registered_rules)}"
        )
    return 1, total_combinations


def _compare_rules(
    ground_truth_valid: set[ComboKey],
    registered_rules: set[ComboKey],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    sample_idx: int,
    scalar_args: tuple[Any, ...],
    scalar_kwargs: dict[str, Any],
    operation: Any,
    variant: str,
    stats: ComparisonStats,
    sample: Any = None,
    untestable: set[ComboKey] | None = None,
) -> None:
    untestable = set() if untestable is None else untestable
    for key in (*ground_truth_valid, *registered_rules):
        if key != normalize_combo_key(key, input_shapes, output_shapes):
            raise AssertionError(f"un-normalized placement key: {key}")
    for key in ground_truth_valid:
        if key in registered_rules:
            stats.true_positives += 1
            name = str(operation)
            stats.true_positives_by_op[name] = stats.true_positives_by_op.get(name, 0) + 1
        elif key not in untestable:
            stats.false_negatives.append(
                Discrepancy(
                    key[0],
                    key[1],
                    sample_idx,
                    input_shapes,
                    "false_negative",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    operation=operation,
                    variant=variant,
                    sample=sample,
                )
            )
    for key in registered_rules:
        if key not in ground_truth_valid and key not in untestable:
            stats.false_positives.append(
                Discrepancy(
                    key[0],
                    key[1],
                    sample_idx,
                    input_shapes,
                    "false_positive",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    operation=operation,
                    variant=variant,
                    sample=sample,
                )
            )


def _format_sample_repro(sample: Any) -> str:
    sample_input, args, kwargs = _sample_parts(sample)
    values = [f"input={sample_input!r}"]
    values.extend(f"args[{index}]={value!r}" for index, value in enumerate(args))
    values.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return ", ".join(values)


def _print_discrepancy_section(
    title: str, discrepancies: list[Discrepancy], show_repro: int = 0
) -> None:
    if not discrepancies:
        return
    print(f"\n{title}")
    grouped: dict[str, dict[ComboKey, list[Discrepancy]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for discrepancy in discrepancies:
        key = (discrepancy.input_placements, discrepancy.output_placements)
        grouped[str(discrepancy.operation)][key].append(discrepancy)
    for operation, entries in sorted(grouped.items()):
        print(f"\n  [{operation}]")
        for (inputs, outputs), values in sorted(entries.items(), key=str):
            output = outputs[0] if len(outputs) == 1 else "(" + ", ".join(outputs) + ")"
            print(f"    {', '.join(inputs)} -> {output}")
            if show_repro:
                limit = len(values) if show_repro < 0 else show_repro
                for value in values[:limit]:
                    if value.sample is not None:
                        print(f"      Repro: {_format_sample_repro(value.sample)}")


def _print_comparison_summary(stats: ComparisonStats, show_repro: int = 0) -> None:
    _print_discrepancy_section("Incorrect", stats.false_positives, show_repro)
    _print_discrepancy_section("Missing", stats.false_negatives, show_repro)


def report_registrations(verbose: bool = False) -> None:
    propagator = DTensor._op_dispatcher.sharding_propagator
    categories = {
        "rule": getattr(propagator, "op_to_rules", {}),
        "strategy": getattr(propagator, "op_strategy_funcs", {}),
        "single_dim": getattr(propagator, "op_single_dim_strategy_funcs", {}),
    }
    print("=" * 70)
    print("distributed operation registration report")
    print("=" * 70)
    for label, table in categories.items():
        print(f"  {label:<10}: {len(table):>4}")
        if verbose:
            for operation in sorted(table, key=str):
                print(f"    {operation}")
