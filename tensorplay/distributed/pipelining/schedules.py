"""Microbatch pipeline schedules."""

import csv
import re
from collections import defaultdict
from enum import Enum
from typing import Any

from .microbatch import TensorChunkSpec, merge_chunks, split_args_kwargs_into_chunks, _split_tensor

__all__ = [
    "get_schedule_class",
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
    "ScheduleDualPipeV",
]


class _ComputationType(str, Enum):
    FORWARD = "F"
    BACKWARD_INPUT = "I"
    BACKWARD_WEIGHT = "W"
    UNSHARD = "UNSHARD"
    RESHARD = "RESHARD"
    SEND_F = "SEND_F"
    RECV_F = "RECV_F"
    SEND_B = "SEND_B"
    RECV_B = "RECV_B"
    FULL_BACKWARD = "B"
    OVERLAP_F_B = "OVERLAP_F_B"
    REDUCE_GRAD = "REDUCE_GRAD"

    @staticmethod
    def from_str(action: str) -> "_ComputationType":
        return _ComputationType(action)


class _Action(tuple):
    __slots__ = ()

    def __new__(cls, stage_index: int, computation_type: _ComputationType, microbatch_index: int | None = None, sub_actions: tuple["_Action", ...] | None = None):
        return tuple.__new__(cls, (stage_index, computation_type, microbatch_index, sub_actions))

    @property
    def stage_index(self) -> int:
        return self[0]

    @property
    def computation_type(self) -> _ComputationType:
        return self[1]

    @property
    def microbatch_index(self) -> int | None:
        return self[2]

    @property
    def sub_actions(self) -> tuple["_Action", ...] | None:
        return self[3]

    @property
    def is_compute_op(self) -> bool:
        return self.computation_type in {_ComputationType.FORWARD, _ComputationType.BACKWARD_INPUT, _ComputationType.BACKWARD_WEIGHT, _ComputationType.FULL_BACKWARD, _ComputationType.OVERLAP_F_B}

    def __repr__(self) -> str:
        if self.sub_actions is not None:
            return f"({';'.join(map(repr, self.sub_actions))}){self.computation_type.value}"
        return f"{self.stage_index}{self.computation_type.value}{'' if self.microbatch_index is None else self.microbatch_index}"

    __str__ = __repr__

    @staticmethod
    def from_str(action_string: str) -> "_Action | None":
        action_string = action_string.strip()
        if not action_string:
            return None
        if action_string.startswith("(") and ")" in action_string:
            end = action_string.index(")")
            sub = tuple(item for item in (_Action.from_str(part) for part in action_string[1:end].split(";")) if item is not None)
            return _Action(-1, _ComputationType.from_str(action_string[end + 1:]), None, sub)
        match = re.fullmatch(r"(\d+)(F|I|B|W|UNSHARD|RESHARD|REDUCE_GRAD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)", action_string)
        if match is None:
            raise ValueError(f"invalid pipeline action: {action_string}")
        stage, kind, microbatch = match.groups()
        return _Action(int(stage), _ComputationType(kind), int(microbatch) if microbatch else None)


def _get_profiler_function_name(action: _Action) -> str:
    return f"TP:{action}"


def _format_pipeline_order(pipeline_order: dict[int, list[_Action | None]], error_step_number: int | None = None) -> str:
    steps = max((len(actions) for actions in pipeline_order.values()), default=0)
    rows = ["step " + " ".join(f"rank {rank}" for rank in sorted(pipeline_order))]
    for index in range(steps):
        values = [str(pipeline_order.get(rank, [None] * steps)[index] or "") for rank in sorted(pipeline_order)]
        suffix = " <error>" if index == error_step_number else ""
        rows.append(f"{index}: " + " ".join(values) + suffix)
    return "\n".join(rows)


class _PipelineSchedule:
    def __init__(self, n_microbatches: int, loss_fn: Any = None, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None, output_merge_spec: Any = None, scale_grads: bool = True) -> None:
        if n_microbatches <= 0:
            raise ValueError("n_microbatches must be positive")
        self._n_microbatches = int(n_microbatches)
        self._loss_fn = loss_fn
        self._args_chunk_spec = args_chunk_spec
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        self._scale_grads = scale_grads
        self._has_backward = loss_fn is not None
        self._stages: list[Any] = []

    def _maybe_compute_loss(self, stage: Any, output: Any, target_mbs: Any, mb_index: int, loss_kwargs: dict[str, Any] | None) -> Any:
        if self._loss_fn is None or target_mbs is None:
            return None
        return self._loss_fn(output, target_mbs[mb_index], **(loss_kwargs or {}))

    def _maybe_get_loss(self, stage: Any, mb_index: int) -> Any:
        del stage
        return self._losses[mb_index] if mb_index < len(self._losses) else None

    def _update_losses(self, stages: Any, losses: list[Any] | None) -> None:
        del stages
        if losses is not None:
            losses.extend(self._losses)

    def _warmup_p2p(self, stages: Any, has_backward: bool, p2p_done: Any) -> None:
        del stages, has_backward, p2p_done

    def _initialize_pp_stages(self, stages: list[Any], args: Any, kwargs: Any, target: Any, fwd_initialized: Any, bwd_initialized: Any, loss_kwargs: Any) -> None:
        del target, fwd_initialized, bwd_initialized, loss_kwargs
        if stages and hasattr(stages[0], "_prepare_forward_infra"):
            stages[0]._prepare_forward_infra(self._n_microbatches, args, kwargs, self._has_backward)

    def _step_microbatches(self, arg_mbs: list[tuple[Any, ...]], kwarg_mbs: list[dict[str, Any]], target_mbs: list[Any] | None, losses: list[Any] | None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    def step(self, *args: Any, target: Any = None, losses: list[Any] | None = None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None, arg_mbs: Any = None, kwarg_mbs: Any = None, target_mbs: Any = None, **kwargs: Any) -> Any:
        arg_mbs, kwarg_mbs, target_mbs = self._get_microbatch_inputs(args, kwargs, target, arg_mbs, kwarg_mbs, target_mbs)
        self._losses = []
        return self._step_microbatches(arg_mbs or [], kwarg_mbs or [], target_mbs, losses, return_outputs, loss_kwargs)

    def eval(self, *args: Any, target: Any = None, losses: list[Any] | None = None, arg_mbs: Any = None, kwarg_mbs: Any = None, target_mbs: Any = None, **kwargs: Any) -> Any:
        old = self._has_backward
        self._has_backward = False
        try:
            return self.step(*args, target=target, losses=losses, arg_mbs=arg_mbs, kwarg_mbs=kwarg_mbs, target_mbs=target_mbs, **kwargs)
        finally:
            self._has_backward = old

    def _check_inputs(self, arg_mbs: Any = None, kwarg_mbs: Any = None, target_mbs: Any = None, losses: Any = None) -> tuple[list[Any], list[Any]]:
        for value, name in ((arg_mbs, "arg_mbs"), (kwarg_mbs, "kwarg_mbs"), (target_mbs, "target_mbs")):
            if value is not None and (not isinstance(value, list) or len(value) != self._n_microbatches):
                raise ValueError(f"{name} must contain {self._n_microbatches} entries")
        if losses is not None and not isinstance(losses, list):
            raise TypeError("losses must be a list")
        return arg_mbs or [()] * self._n_microbatches, kwarg_mbs or [{}] * self._n_microbatches

    def _compute_loss(self, output: Any, target: Any, loss_kwargs: dict[str, Any] | None = None) -> Any:
        return self._loss_fn(output, target, **(loss_kwargs or {}))

    def _split_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
        return split_args_kwargs_into_chunks(args, kwargs, self._n_microbatches, self._args_chunk_spec, self._kwargs_chunk_spec)

    def _get_microbatch_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any], target: Any, arg_mbs: Any, kwarg_mbs: Any, target_mbs: Any) -> tuple[list[Any], list[Any], list[Any] | None]:
        if any(value is not None for value in (arg_mbs, kwarg_mbs, target_mbs)):
            if args or kwargs or target is not None:
                raise ValueError("whole-batch and pre-split inputs cannot be mixed")
            checked_args, checked_kwargs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs)
            return checked_args, checked_kwargs, target_mbs
        args_split, kwargs_split = self._split_inputs(args, kwargs)
        target_split = list(_split_tensor(target, TensorChunkSpec(0), self._n_microbatches)) if target is not None else None
        return args_split, kwargs_split, target_split

    def _merge_outputs(self, output_chunks: list[Any]) -> Any:
        if self._output_merge_spec is None:
            output_spec = _default_merge_spec(output_chunks[0])
        else:
            output_spec = self._output_merge_spec
        return merge_chunks(output_chunks, output_spec)


class PipelineScheduleSingle(_PipelineSchedule):
    def __init__(self, stage: Any, n_microbatches: int, loss_fn: Any = None, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None, output_merge_spec: Any = None, scale_grads: bool = True) -> None:
        super().__init__(n_microbatches, loss_fn, args_chunk_spec, kwargs_chunk_spec, output_merge_spec, scale_grads)
        self._stage = stage
        self._stages = [stage]

    def _initialize_stage(self, args: Any, kwargs: Any, target: Any = None, loss_kwargs: Any = None) -> None:
        del target, loss_kwargs
        self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs, self._has_backward)

    def _step_microbatches(self, arg_mbs: list[tuple[Any, ...]], kwarg_mbs: list[dict[str, Any]], target_mbs: list[Any] | None, losses: list[Any] | None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None) -> Any:
        outputs = []
        for index, (args, kwargs) in enumerate(zip(arg_mbs, kwarg_mbs)):
            output = self._stage.forward_one_chunk(index, args, kwargs)
            outputs.append(output)
            loss = self._maybe_compute_loss(self._stage, output, target_mbs, index, loss_kwargs)
            if loss is not None:
                self._losses.append(loss)
                if self._has_backward:
                    self._stage.backward_one_chunk(index, loss=loss)
        if losses is not None:
            losses.extend(self._losses)
        if not return_outputs:
            return None
        return self._merge_outputs(outputs)

    def _get_pipeline_order(self) -> dict[int, list[_Action]]:
        actions = [_Action(self._stage.stage_index, _ComputationType.FORWARD, index) for index in range(self._n_microbatches)]
        return {self._stage.stage_index: actions}


class _ScheduleForwardOnly(PipelineScheduleSingle):
    def _step_microbatches(self, *args: Any, **kwargs: Any) -> Any:
        old = self._has_backward
        self._has_backward = False
        try:
            return super()._step_microbatches(*args, **kwargs)
        finally:
            self._has_backward = old


class ScheduleGPipe(PipelineScheduleSingle):
    pass


class Schedule1F1B(PipelineScheduleSingle):
    pass


class PipelineScheduleMulti(_PipelineSchedule):
    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None, output_merge_spec: Any = None, use_full_backward: bool = True, scale_grads: bool = True, backward_requires_autograd: bool = True) -> None:
        super().__init__(n_microbatches, loss_fn, args_chunk_spec, kwargs_chunk_spec, output_merge_spec, scale_grads)
        self._stages = list(stages)
        self.use_full_backward = use_full_backward
        self.backward_requires_autograd = backward_requires_autograd

    def _initialize_stages(self, args: Any, kwargs: Any, target: Any = None, loss_kwargs: Any = None) -> None:
        del target, loss_kwargs
        for stage in self._stages:
            stage._prepare_forward_infra(self._n_microbatches, args, kwargs, self._has_backward)

    def _step_microbatches(self, arg_mbs: list[tuple[Any, ...]], kwarg_mbs: list[dict[str, Any]], target_mbs: list[Any] | None, losses: list[Any] | None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None) -> Any:
        outputs = []
        for index, (args, kwargs) in enumerate(zip(arg_mbs, kwarg_mbs)):
            value = self._stages[0].forward_one_chunk(index, args, kwargs)
            for stage_index, stage in enumerate(self._stages[1:], 1):
                value = stage.forward_one_chunk(index, (value,), {})
            outputs.append(value)
            loss = self._maybe_compute_loss(self._stages[-1], value, target_mbs, index, loss_kwargs)
            if loss is not None:
                self._losses.append(loss)
                if self._has_backward:
                    loss.backward()
        if losses is not None:
            losses.extend(self._losses)
        return self._merge_outputs(outputs) if return_outputs else None

    def _validate_adjacent_stage_communication(self) -> None:
        if any(stage.stage_index != index for index, stage in enumerate(self._stages)):
            raise ValueError("stage indices must be contiguous")

    def _validate_and_set_stage_mapping(self, actions: Any) -> None:
        self._stage_mapping = actions

    def _dump_csv(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for rank, actions in self._get_pipeline_order().items():
                writer.writerow([rank, *map(str, actions)])

    def _load_csv(self, filename: str, format: str = "auto") -> dict[int, list[_Action | None]]:
        del format
        with open(filename, newline="", encoding="utf-8") as stream:
            return {int(row[0]): [_Action.from_str(value) for value in row[1:]] for row in csv.reader(stream)}

    def _get_pipeline_order(self) -> dict[int, list[_Action]]:
        return {stage.stage_index: [_Action(stage.stage_index, _ComputationType.FORWARD, index) for index in range(self._n_microbatches)] for stage in self._stages}


class ScheduleLoopedBFS(PipelineScheduleMulti):
    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, output_merge_spec: Any = None, scale_grads: bool = True, backward_requires_autograd: bool = True, defer_pp_recv: bool = False, max_active_stages: int | None = None) -> None:
        super().__init__(stages, n_microbatches, loss_fn, output_merge_spec=output_merge_spec, scale_grads=scale_grads, backward_requires_autograd=backward_requires_autograd)
        self.defer_pp_recv = defer_pp_recv
        self.max_active_stages = max_active_stages

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action]:
        return self._get_pipeline_order().get(rank, [])


class ScheduleInterleaved1F1B(ScheduleLoopedBFS):
    pass


class ScheduleInterleavedZeroBubble(ScheduleLoopedBFS):
    def _add_bubbles_to_actions(self, num_stages_global: int) -> None:
        del num_stages_global


class ScheduleZBVZeroBubble(ScheduleLoopedBFS):
    pass


class ScheduleDualPipeV(ScheduleLoopedBFS):
    pass


def _requires_reduce_grad(action_type: _ComputationType) -> bool:
    return action_type in {_ComputationType.BACKWARD_INPUT, _ComputationType.BACKWARD_WEIGHT, _ComputationType.FULL_BACKWARD}


def _add_reduce_grad(actions: list[_Action], n_microbatches: int) -> list[_Action]:
    return actions + [_Action(-1, _ComputationType.REDUCE_GRAD, index) for index in range(n_microbatches)]


def _add_unshard_reshard(compute_actions: list[_Action], max_active_stages: int) -> list[_Action]:
    del max_active_stages
    return compute_actions


def _merge_bw(compute_actions: list[_Action]) -> list[_Action]:
    return compute_actions


def _add_send_recv(compute_actions: list[_Action], stage_to_rank: dict[int, int], num_stages: int) -> list[_Action]:
    del stage_to_rank, num_stages
    return compute_actions


def _defer_recv_ops(actions: list[_Action], stage_to_rank: dict[int, int]) -> list[_Action]:
    del stage_to_rank
    return actions


def _validate_schedule(actions: Any, pp_group_size: int, num_stages: int, num_microbatches: int) -> None:
    del actions
    if pp_group_size <= 0 or num_stages <= 0 or num_microbatches <= 0:
        raise ValueError("pipeline dimensions must be positive")


def _get_1f1b_rank_ops(*args: Any, **kwargs: Any) -> list[_Action]:
    del args, kwargs
    return []


def _get_warmup_ops(*args: Any, **kwargs: Any) -> list[_Action]:
    del args, kwargs
    return []


def get_schedule_class(schedule_name: str) -> type[_PipelineSchedule]:
    mapping = {
        "GPipe": ScheduleGPipe,
        "1F1B": Schedule1F1B,
        "Interleaved1F1B": ScheduleInterleaved1F1B,
        "LoopedBFS": ScheduleLoopedBFS,
        "InterleavedZeroBubble": ScheduleInterleavedZeroBubble,
        "ZBVZeroBubble": ScheduleZBVZeroBubble,
        "DualPipeV": ScheduleDualPipeV,
    }
    try:
        return mapping[schedule_name]
    except KeyError as exc:
        raise ValueError(f"unknown pipeline schedule {schedule_name!r}") from exc


def _simulate_comms_compute(pipeline_order: Any, stage_to_rank: Any, num_stages: int) -> Any:
    del stage_to_rank, num_stages
    return pipeline_order


def _dump_chrometrace(schedule: Any, filename: str) -> None:
    from ._schedule_visualizer import visualize_schedule
    visualize_schedule(schedule, filename)


def _check_torch_compile_compatibility(stages: Any, schedule_name: str) -> None:
    del stages, schedule_name


def _default_merge_spec(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_default_merge_spec(item) for item in value)
    if isinstance(value, list):
        return [_default_merge_spec(item) for item in value]
    if isinstance(value, dict):
        return {key: _default_merge_spec(item) for key, item in value.items()}
    return TensorChunkSpec(0) if hasattr(value, "shape") else None
