"""Microbatch pipeline schedules."""

import csv
import re
from collections import defaultdict
from enum import Enum
from typing import Any

from .. import distributed_core as dist
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
        if not getattr(stage, "is_last", False) or self._loss_fn is None or target_mbs is None:
            return None
        loss = self._loss_fn(output, target_mbs[mb_index], **(loss_kwargs or {}))
        self._losses.append(loss)
        return loss

    def _maybe_get_loss(self, stage: Any, mb_index: int) -> Any:
        if not getattr(stage, "is_last", False):
            return None
        return self._losses[mb_index] if 0 <= mb_index < len(self._losses) else None

    def _update_losses(self, stages: Any, losses: list[Any] | None) -> None:
        stage_list = stages if isinstance(stages, (list, tuple)) else [stages]
        if losses is not None and any(getattr(stage, "is_last", False) for stage in stage_list):
            if len(self._losses) != self._n_microbatches:
                raise RuntimeError(
                    f"expected {self._n_microbatches} losses, got {len(self._losses)}"
                )
            losses.clear()
            losses.extend(self._losses)
        self._losses.clear()

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
        self._initialize_for_step(args, kwargs, arg_mbs, kwarg_mbs, target, loss_kwargs)
        self._losses = []
        return self._step_microbatches(arg_mbs or [], kwarg_mbs or [], target_mbs, losses, return_outputs, loss_kwargs)

    def _initialize_for_step(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        arg_mbs: list[Any],
        kwarg_mbs: list[Any],
        target: Any,
        loss_kwargs: Any,
    ) -> None:
        init_args = tuple(arg_mbs[0]) if arg_mbs else args
        init_kwargs = dict(kwarg_mbs[0]) if kwarg_mbs else kwargs
        if isinstance(self, PipelineScheduleSingle):
            self._initialize_stage(init_args, init_kwargs, target, loss_kwargs)
        elif self._stages:
            self._initialize_stages(init_args, init_kwargs, target, loss_kwargs)

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
        self._num_stages = int(stage.num_stages)
        self._stage.has_backward = self._has_backward

    def _initialize_stage(self, args: Any, kwargs: Any, target: Any = None, loss_kwargs: Any = None) -> None:
        del target, loss_kwargs
        self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs, self._has_backward)

    def _step_microbatches(self, arg_mbs: list[tuple[Any, ...]], kwarg_mbs: list[dict[str, Any]], target_mbs: list[Any] | None, losses: list[Any] | None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None) -> Any:
        self._stage.clear_runtime_states()
        outputs = []
        forward_sends = []
        for index, (args, kwargs) in enumerate(zip(arg_mbs, kwarg_mbs)):
            for work in _run_p2p(self._stage.get_fwd_recv_ops(index)):
                work.wait()
            output = self._stage.forward_one_chunk(index, args, kwargs)
            outputs.append(output)
            forward_sends.extend(_run_p2p(self._stage.get_fwd_send_ops(index)))
        for work in forward_sends:
            work.wait()

        if self._has_backward:
            backward_sends = []
            for index in range(len(outputs)):
                for work in _run_p2p(self._stage.get_bwd_recv_ops(index)):
                    work.wait()
                loss = self._maybe_compute_loss(self._stage, outputs[index], target_mbs, index, loss_kwargs)
                if self._stage.is_last:
                    if loss is not None:
                        self._stage.backward_one_chunk(index, loss=loss)
                else:
                    self._stage.backward_one_chunk(index)
                backward_sends.extend(_run_p2p(self._stage.get_bwd_send_ops(index)))
            for work in backward_sends:
                work.wait()
            if self._scale_grads:
                self._stage.scale_grads(self._n_microbatches)
        if losses is not None and self._stage.is_last:
            losses.extend(self._losses)
        if not return_outputs or not self._stage.is_last:
            return None
        return self._merge_outputs(outputs)

    def _get_pipeline_order(self) -> dict[int, list[_Action]]:
        actions: list[_Action | None] = [
            _Action(self._stage.stage_index, _ComputationType.FORWARD, index)
            for index in range(self._n_microbatches)
        ]
        if self._has_backward:
            actions.extend(
                _Action(self._stage.stage_index, _ComputationType.FULL_BACKWARD, index)
                for index in range(self._n_microbatches)
            )
            actions = _add_reduce_grad(actions, self._n_microbatches)
        return {int(self._stage.group_rank): actions}


def _run_p2p(operations: list[Any]) -> list[Any]:
    if not operations:
        return []
    return dist.batch_isend_irecv(operations)


def _normalize_stage_args(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


class _ScheduleForwardOnly(PipelineScheduleSingle):
    def _step_microbatches(self, *args: Any, **kwargs: Any) -> Any:
        old = self._has_backward
        self._has_backward = False
        try:
            return super()._step_microbatches(*args, **kwargs)
        finally:
            self._has_backward = old


class ScheduleGPipe(PipelineScheduleSingle):
    """Execute all forward microbatches before draining their backwards."""

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        group_size = int(self._stage.group_size)
        if group_size != self._num_stages:
            raise ValueError("GPipe requires one stage per pipeline rank")
        pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(group_size):
            actions: list[_Action | None] = [None] * rank
            actions.extend(
                _Action(rank, _ComputationType.FORWARD, microbatch)
                for microbatch in range(self._n_microbatches)
            )
            if self._has_backward:
                actions.extend(
                    [None] * (3 * (group_size - 1 - rank))
                )
                actions.extend(
                    _Action(rank, _ComputationType.FULL_BACKWARD, microbatch)
                    for microbatch in range(self._n_microbatches)
                )
                pipeline_order[rank] = _add_reduce_grad(
                    actions, self._n_microbatches
                )
            else:
                pipeline_order[rank] = actions
        return pipeline_order


class Schedule1F1B(PipelineScheduleSingle):
    """Overlap forward and backward microbatches after warmup."""

    def __init__(self, stage: Any, n_microbatches: int, loss_fn: Any = None, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None, output_merge_spec: Any = None, scale_grads: bool = True) -> None:
        super().__init__(
            stage,
            n_microbatches,
            loss_fn,
            args_chunk_spec,
            kwargs_chunk_spec,
            output_merge_spec,
            scale_grads,
        )
        if self._has_backward and n_microbatches < self._num_stages:
            raise ValueError("1F1B requires at least one microbatch per stage")

    def _step_microbatches(
        self,
        arg_mbs: list[tuple[Any, ...]],
        kwarg_mbs: list[dict[str, Any]],
        target_mbs: list[Any] | None,
        losses: list[Any] | None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self._stage.clear_runtime_states()
        warmup = min(self._n_microbatches, self._num_stages - self._stage.stage_index)
        outputs: list[Any] = []
        next_forward = 0
        next_backward = 0

        def run_forward(index: int, receives_posted: bool = False) -> None:
            if not receives_posted:
                receives = _run_p2p(self._stage.get_fwd_recv_ops(index))
                for work in receives:
                    work.wait()
            output = self._stage.forward_one_chunk(
                index, arg_mbs[index], kwarg_mbs[index]
            )
            outputs.append(output)
            for work in _run_p2p(self._stage.get_fwd_send_ops(index)):
                work.wait()
            loss = self._maybe_compute_loss(
                self._stage, output, target_mbs, index, loss_kwargs
            )

        def run_backward(index: int) -> list[Any]:
            for work in _run_p2p(self._stage.get_bwd_recv_ops(index)):
                work.wait()
            loss = self._maybe_get_loss(self._stage, index)
            if self._stage.is_last:
                if loss is not None:
                    self._stage.backward_one_chunk(index, loss=loss)
            else:
                self._stage.backward_one_chunk(index)
            return _run_p2p(self._stage.get_bwd_send_ops(index))

        while next_forward < warmup:
            run_forward(next_forward)
            next_forward += 1

        if not self._has_backward:
            while next_forward < self._n_microbatches:
                run_forward(next_forward)
                next_forward += 1
        else:
            while next_forward < self._n_microbatches:
                backward_sends = run_backward(next_backward)
                next_backward += 1
                forward_receives = _run_p2p(
                    self._stage.get_fwd_recv_ops(next_forward)
                )
                for work in backward_sends + forward_receives:
                    work.wait()
                run_forward(next_forward, receives_posted=True)
                next_forward += 1
            while next_backward < self._n_microbatches:
                backward_sends = run_backward(next_backward)
                for work in backward_sends:
                    work.wait()
                next_backward += 1

        if self._has_backward and self._scale_grads:
            self._stage.scale_grads(self._n_microbatches)
        if losses is not None and self._stage.is_last:
            losses.extend(self._losses)
        if not return_outputs or not self._stage.is_last:
            return None
        return self._merge_outputs(outputs)

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        group_size = int(self._stage.group_size)
        if group_size != self._num_stages:
            raise ValueError("1F1B requires one stage per pipeline rank")
        pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(group_size):
            actions: list[_Action | None] = [None] * rank
            warmup = min(self._n_microbatches, group_size - 1 - rank)
            actions.extend(
                _Action(rank, _ComputationType.FORWARD, microbatch)
                for microbatch in range(warmup)
            )
            actions.extend([None] * (2 * (group_size - 1 - rank)))
            next_forward = warmup
            next_backward = 0
            while next_forward < self._n_microbatches:
                actions.append(
                    _Action(rank, _ComputationType.FORWARD, next_forward)
                )
                next_forward += 1
                actions.append(
                    _Action(rank, _ComputationType.FULL_BACKWARD, next_backward)
                )
                next_backward += 1
            while next_backward < self._n_microbatches:
                if rank != group_size - 1:
                    actions.append(None)
                actions.append(
                    _Action(rank, _ComputationType.FULL_BACKWARD, next_backward)
                )
                next_backward += 1
            pipeline_order[rank] = _add_reduce_grad(
                actions, self._n_microbatches
            )
        return pipeline_order


class PipelineScheduleMulti(_PipelineSchedule):
    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None, output_merge_spec: Any = None, use_full_backward: bool = True, scale_grads: bool = True, backward_requires_autograd: bool = True) -> None:
        if not stages:
            raise ValueError("at least one pipeline stage is required")
        super().__init__(n_microbatches, loss_fn, args_chunk_spec, kwargs_chunk_spec, output_merge_spec, scale_grads)
        self._stages = list(stages)
        self.use_full_backward = use_full_backward
        self.backward_requires_autograd = backward_requires_autograd
        self._num_stages = int(stages[0].num_stages)
        self.pp_group_size = int(stages[0].group_size)
        if self._num_stages <= 0 or self.pp_group_size <= 0:
            raise ValueError("pipeline dimensions must be positive")
        if any(int(stage.num_stages) != self._num_stages for stage in stages):
            raise ValueError("all pipeline stages must use the same stage count")
        if len({int(stage.stage_index) for stage in stages}) != len(stages):
            raise ValueError("a pipeline stage cannot be listed more than once")
        self.rank = int(stages[0].group_rank)
        self.stage_index_to_group_rank = {
            index: index % self.pp_group_size for index in range(self._num_stages)
        }
        for stage in self._stages:
            stage.stage_index_to_group_rank = dict(self.stage_index_to_group_rank)

    def _initialize_stages(self, args: Any, kwargs: Any, target: Any = None, loss_kwargs: Any = None) -> None:
        del target, loss_kwargs
        for stage in self._stages:
            stage._prepare_forward_infra(self._n_microbatches, args, kwargs, self._has_backward)

    def _step_microbatches(self, arg_mbs: list[tuple[Any, ...]], kwarg_mbs: list[dict[str, Any]], target_mbs: list[Any] | None, losses: list[Any] | None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None) -> Any:
        if self._stages_are_local():
            return self._step_local_stages(
                arg_mbs,
                kwarg_mbs,
                target_mbs,
                losses,
                return_outputs,
                loss_kwargs,
            )
        return self._step_distributed_stages(
            arg_mbs,
            kwarg_mbs,
            target_mbs,
            losses,
            return_outputs,
            loss_kwargs,
        )

    def _stages_are_local(self) -> bool:
        if not self._stages:
            return True
        if not dist.is_initialized():
            return True
        if len(self._stages) == 1:
            stage = self._stages[0]
            return stage.num_stages <= 1 or stage.group_size <= 1
        first_rank = self._stages[0].group_rank
        return all(stage.group_rank == first_rank for stage in self._stages)

    def _step_local_stages(
        self,
        arg_mbs: list[tuple[Any, ...]],
        kwarg_mbs: list[dict[str, Any]],
        target_mbs: list[Any] | None,
        losses: list[Any] | None,
        return_outputs: bool,
        loss_kwargs: dict[str, Any] | None,
    ) -> Any:
        for stage in self._stages:
            stage.clear_runtime_states()
        outputs = []
        for index, (args, kwargs) in enumerate(zip(arg_mbs, kwarg_mbs)):
            value = self._stages[0].forward_one_chunk(index, args, kwargs)
            for stage in self._stages[1:]:
                stage.set_local_fwd_input(value, index)
                value = stage.forward_one_chunk(
                    index, _normalize_stage_args(value), {}
                )
            outputs.append(value)
            loss = self._maybe_compute_loss(self._stages[-1], value, target_mbs, index, loss_kwargs)
        if self._has_backward:
            for index in reversed(range(len(outputs))):
                loss = self._maybe_get_loss(self._stages[-1], index)
                next_grad = self._stages[-1].backward_one_chunk(index, loss=loss)
                for stage_index in range(len(self._stages) - 2, -1, -1):
                    stage = self._stages[stage_index]
                    stage.set_local_bwd_input(next_grad, index)
                    next_grad = stage.backward_one_chunk(index)
        if self._has_backward and self._scale_grads:
            for stage in self._stages:
                stage.scale_grads(self._n_microbatches)
        if losses is not None:
            losses.extend(self._losses)
        return self._merge_outputs(outputs) if return_outputs else None

    def _step_distributed_stages(
        self,
        arg_mbs: list[tuple[Any, ...]],
        kwarg_mbs: list[dict[str, Any]],
        target_mbs: list[Any] | None,
        losses: list[Any] | None,
        return_outputs: bool,
        loss_kwargs: dict[str, Any] | None,
    ) -> Any:
        for stage in self._stages:
            stage.clear_runtime_states()
        outputs: list[Any] = []
        send_works: list[Any] = []
        for index, (args, kwargs) in enumerate(zip(arg_mbs, kwarg_mbs)):
            for stage in self._stages:
                for work in _run_p2p(stage.get_fwd_recv_ops(index)):
                    work.wait()
                output = stage.forward_one_chunk(index, args, kwargs)
                if stage.is_last:
                    outputs.append(output)
                    loss = self._maybe_compute_loss(
                        stage, output, target_mbs, index, loss_kwargs
                    )
                send_works.extend(_run_p2p(stage.get_fwd_send_ops(index)))
        for work in send_works:
            work.wait()
        if self._has_backward:
            for index in reversed(range(self._n_microbatches)):
                for stage in reversed(self._stages):
                    for work in _run_p2p(stage.get_bwd_recv_ops(index)):
                        work.wait()
                    loss = self._maybe_get_loss(stage, index)
                    stage.backward_one_chunk(index, loss=loss)
                    for work in _run_p2p(stage.get_bwd_send_ops(index)):
                        work.wait()
            if self._scale_grads:
                for stage in self._stages:
                    stage.scale_grads(self._n_microbatches)
        if losses is not None:
            losses.extend(self._losses)
        if not return_outputs or not outputs:
            return None
        return self._merge_outputs(outputs)

    def _validate_adjacent_stage_communication(self) -> None:
        if any(stage.stage_index != index for index, stage in enumerate(self._stages)):
            raise ValueError("stage indices must be contiguous")

    def _validate_and_set_stage_mapping(self, actions: Any) -> None:
        self.stage_index_to_group_rank = _validate_schedule(
            actions,
            self.pp_group_size,
            self._num_stages,
            self._n_microbatches,
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = dict(self.stage_index_to_group_rank)

    def _dump_csv(self, filename: str) -> None:
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for rank, actions in self._get_pipeline_order().items():
                writer.writerow([rank, *map(str, actions)])

    def _load_csv(self, filename: str, format: str = "auto") -> dict[int, list[_Action | None]]:
        del format
        with open(filename, newline="", encoding="utf-8") as stream:
            actions = {
                int(row[0]): [_Action.from_str(value) for value in row[1:]]
                for row in csv.reader(stream)
            }
        self._validate_and_set_stage_mapping(actions)
        return actions

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        owned: dict[int, list[int]] = {rank: [] for rank in range(self.pp_group_size)}
        for stage_index in range(self._num_stages):
            rank = self.stage_index_to_group_rank[stage_index]
            owned.setdefault(rank, []).append(stage_index)

        pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            actions: list[_Action | None] = [None] * rank
            for stage_index in owned.get(rank, ()):
                actions.extend(
                    _Action(stage_index, _ComputationType.FORWARD, microbatch)
                    for microbatch in range(self._n_microbatches)
                )
            if self._has_backward:
                actions.extend([None] * (2 * max(0, self.pp_group_size - 1 - rank)))
                for stage_index in reversed(owned.get(rank, ())):
                    actions.extend(
                        _Action(stage_index, _ComputationType.FULL_BACKWARD, microbatch)
                        for microbatch in reversed(range(self._n_microbatches))
                    )
                actions = _add_reduce_grad(actions, self._n_microbatches)
            pipeline_order[rank] = actions
        return pipeline_order


class ScheduleLoopedBFS(PipelineScheduleMulti):
    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, output_merge_spec: Any = None, scale_grads: bool = True, backward_requires_autograd: bool = True, defer_pp_recv: bool = False, max_active_stages: int | None = None) -> None:
        super().__init__(stages, n_microbatches, loss_fn, output_merge_spec=output_merge_spec, scale_grads=scale_grads, backward_requires_autograd=backward_requires_autograd)
        self.defer_pp_recv = defer_pp_recv
        self.max_active_stages = max_active_stages

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action]:
        return self._get_pipeline_order().get(rank, [])


class ScheduleInterleaved1F1B(ScheduleLoopedBFS):
    """Run local stages in depth-first microbatch order."""

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        owned = self._owned_stages_by_rank()
        local_stage_count = max((len(value) for value in owned.values()), default=0)
        if local_stage_count == 0:
            return {rank: [] for rank in range(self.pp_group_size)}
        rounds = max(1, self._n_microbatches // self.pp_group_size)
        if self._n_microbatches % rounds:
            raise ValueError(
                "interleaved schedules require a divisible microbatch round count"
            )
        microbatches_per_round = self._n_microbatches // rounds
        pipeline_order: dict[int, list[_Action | None]] = {}
        for rank in range(self.pp_group_size):
            stage_indices = owned.get(rank, ())
            if not stage_indices:
                pipeline_order[rank] = []
                continue
            if not self._has_backward:
                pipeline_order[rank] = [
                    _Action(stage_index, _ComputationType.FORWARD, microbatch)
                    for stage_index in stage_indices
                    for microbatch in range(self._n_microbatches)
                ]
                continue
            warmup = min(
                self._n_microbatches * len(stage_indices),
                max(0, (len(stage_indices) - 1) * microbatches_per_round)
                + 2 * (self.pp_group_size - 1 - rank),
            )
            total_compute = len(stage_indices) * self._n_microbatches
            forward_ops = min(total_compute, warmup)
            steady_ops = total_compute - forward_ops
            cooldown_ops = forward_ops

            def forward_stage(step: int) -> int:
                slot = (step // microbatches_per_round) % len(stage_indices)
                return stage_indices[slot]

            def backward_stage(step: int) -> int:
                slot = (
                    len(stage_indices)
                    - 1
                    - ((step - warmup) // microbatches_per_round) % len(stage_indices)
                )
                return stage_indices[slot]

            actions = _get_1f1b_rank_ops(
                len(stage_indices),
                self.pp_group_size,
                forward_ops,
                steady_ops,
                cooldown_ops,
                rank,
                forward_stage,
                backward_stage,
            )
            pipeline_order[rank] = _add_reduce_grad(
                actions, self._n_microbatches
            ) if self._has_backward else [
                action
                for action in actions
                if action is not None
                and action.computation_type is _ComputationType.FORWARD
            ]
        return pipeline_order

    def _owned_stages_by_rank(self) -> dict[int, list[int]]:
        owned: dict[int, list[int]] = {rank: [] for rank in range(self.pp_group_size)}
        for stage_index in range(self._num_stages):
            owned[self.stage_index_to_group_rank[stage_index]].append(stage_index)
        return owned


class ScheduleInterleavedZeroBubble(ScheduleLoopedBFS):
    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        if not self._has_backward:
            return super()._get_pipeline_order()
        raw = self._get_split_stage_order()
        return self._add_bubbles_to_actions(self._num_stages, raw)

    def _get_split_stage_order(self) -> dict[int, list[_Action | None]]:
        return _split_backward_pipeline_order(
            self._num_stages,
            self.pp_group_size,
            self._n_microbatches,
            self.stage_index_to_group_rank,
        )

    def _add_bubbles_to_actions(
        self,
        num_stages_global: int,
        actions: dict[int, list[_Action | None]] | None = None,
    ) -> dict[int, list[_Action | None]]:
        source = actions if actions is not None else self._get_split_stage_order()
        result: dict[int, list[_Action | None]] = {
            rank: [] for rank in range(self.pp_group_size)
        }
        pointers = {rank: 0 for rank in range(self.pp_group_size)}
        completed: set[tuple[int, _ComputationType, int | None]] = set()

        def leaves(action: _Action) -> tuple[_Action, ...]:
            return action.sub_actions or (action,)

        def ready(action: _Action) -> bool:
            for leaf in leaves(action):
                stage = leaf.stage_index
                microbatch = leaf.microbatch_index
                kind = leaf.computation_type
                if kind is _ComputationType.FORWARD and stage > 0:
                    if (
                        stage - 1,
                        _ComputationType.FORWARD,
                        microbatch,
                    ) not in completed:
                        return False
                elif kind in {
                    _ComputationType.FULL_BACKWARD,
                    _ComputationType.BACKWARD_INPUT,
                }:
                    dependency = (
                        stage + 1,
                        kind,
                        microbatch,
                    )
                    if stage == num_stages_global - 1:
                        dependency = (stage, _ComputationType.FORWARD, microbatch)
                    elif dependency not in completed:
                        alternate = (
                            stage + 1,
                            _ComputationType.FULL_BACKWARD,
                            microbatch,
                        )
                        if alternate not in completed:
                            return False
                elif kind is _ComputationType.BACKWARD_WEIGHT:
                    if (
                        stage,
                        _ComputationType.BACKWARD_INPUT,
                        microbatch,
                    ) not in completed:
                        return False
            return True

        while any(pointers[rank] < len(source.get(rank, ())) for rank in pointers):
            progressed = False
            for rank in range(self.pp_group_size):
                index = pointers[rank]
                if index >= len(source.get(rank, ())):
                    continue
                action = source[rank][index]
                if action is None:
                    result[rank].append(None)
                    pointers[rank] += 1
                    progressed = True
                    continue
                if not ready(action):
                    result[rank].append(None)
                    continue
                result[rank].append(action)
                pointers[rank] += 1
                for leaf in leaves(action):
                    completed.add(
                        (leaf.stage_index, leaf.computation_type, leaf.microbatch_index)
                    )
                progressed = True
            if not progressed:
                raise ValueError("pipeline dependencies cannot make progress")
        for rank, rank_actions in result.items():
            result[rank] = _add_reduce_grad(rank_actions, self._n_microbatches)
        return result


class ScheduleZBVZeroBubble(ScheduleLoopedBFS):
    """Run the zero-bubble-compatible local stage schedule."""

    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, output_merge_spec: Any = None, scale_grads: bool = True, backward_requires_autograd: bool = True, defer_pp_recv: bool = False, max_active_stages: int | None = None) -> None:
        super().__init__(
            stages,
            n_microbatches,
            loss_fn,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
            max_active_stages=max_active_stages,
        )
        if len(stages) != 2:
            raise ValueError("ZBV requires exactly two local stages")

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        if not self._has_backward:
            return super()._get_pipeline_order()
        return _split_backward_pipeline_order(
            self._num_stages,
            self.pp_group_size,
            self._n_microbatches,
            self.stage_index_to_group_rank,
        )


class ScheduleDualPipeV(ScheduleLoopedBFS):
    """Run the bidirectional local stage schedule."""

    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, output_merge_spec: Any = None, scale_grads: bool = True, backward_requires_autograd: bool = True, defer_pp_recv: bool = False, max_active_stages: int | None = None) -> None:
        super().__init__(
            stages,
            n_microbatches,
            loss_fn,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
            max_active_stages=max_active_stages,
        )
        if len(stages) != 2:
            raise ValueError("DualPipeV requires exactly two local stages")
        if n_microbatches < self._num_stages:
            raise ValueError("DualPipeV requires at least one microbatch per stage")

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]]:
        if not self._has_backward:
            return super()._get_pipeline_order()
        return _split_backward_pipeline_order(
            self._num_stages,
            self.pp_group_size,
            self._n_microbatches,
            self.stage_index_to_group_rank,
        )


def _requires_reduce_grad(action_type: _ComputationType) -> bool:
    return action_type in {
        _ComputationType.BACKWARD_WEIGHT,
        _ComputationType.FULL_BACKWARD,
    }


def _add_reduce_grad(
    actions: list[_Action | None], n_microbatches: int
) -> list[_Action | None]:
    if n_microbatches <= 0:
        raise ValueError("n_microbatches must be positive")
    result: list[_Action | None] = []
    counts: dict[int, int] = defaultdict(int)
    for action in actions:
        if action is None:
            result.append(None)
            continue
        result.append(action)
        leaves = action.sub_actions or (action,)
        for leaf in leaves:
            if not _requires_reduce_grad(leaf.computation_type):
                continue
            counts[leaf.stage_index] += 1
            if counts[leaf.stage_index] == n_microbatches:
                result.append(
                    _Action(leaf.stage_index, _ComputationType.REDUCE_GRAD, None)
                )
    return result


def _split_backward_pipeline_order(
    num_stages: int,
    pp_group_size: int,
    n_microbatches: int,
    stage_to_rank: dict[int, int],
) -> dict[int, list[_Action | None]]:
    owned: dict[int, list[int]] = {rank: [] for rank in range(pp_group_size)}
    for stage_index in range(num_stages):
        owned[int(stage_to_rank[stage_index])].append(stage_index)
    result: dict[int, list[_Action | None]] = {}
    for rank in range(pp_group_size):
        actions: list[_Action | None] = [None] * rank
        for stage_index in owned[rank]:
            actions.extend(
                _Action(stage_index, _ComputationType.FORWARD, microbatch)
                for microbatch in range(n_microbatches)
            )
        for stage_index in reversed(owned[rank]):
            actions.extend(
                _Action(stage_index, _ComputationType.BACKWARD_INPUT, microbatch)
                for microbatch in reversed(range(n_microbatches))
            )
            actions.extend(
                _Action(stage_index, _ComputationType.BACKWARD_WEIGHT, microbatch)
                for microbatch in reversed(range(n_microbatches))
            )
        result[rank] = _add_reduce_grad(actions, n_microbatches)
    return result


def _add_unshard_reshard(
    compute_actions: list[_Action | None], max_active_stages: int = 3
) -> list[_Action]:
    if max_active_stages <= 0:
        raise ValueError("max_active_stages must be positive")
    active: list[int] = []
    result: list[_Action] = []
    actions = list(compute_actions)
    for index, action in enumerate(actions):
        if action is None:
            continue
        upcoming: list[int] = []
        for candidate in actions[index:]:
            if candidate is not None and candidate.stage_index not in upcoming:
                upcoming.append(candidate.stage_index)
                if len(upcoming) == max_active_stages:
                    break
        for stage_index in list(active):
            if stage_index not in upcoming:
                result.append(_Action(stage_index, _ComputationType.RESHARD))
                active.remove(stage_index)
        for stage_index in upcoming:
            if stage_index not in active:
                result.append(_Action(stage_index, _ComputationType.UNSHARD))
                active.append(stage_index)
        result.append(action)
    for stage_index in list(active):
        result.append(_Action(stage_index, _ComputationType.RESHARD))
    return result


def _merge_bw(compute_actions: list[_Action | None]) -> list[_Action]:
    pending = list(compute_actions)
    result: list[_Action] = []
    while pending:
        action = pending.pop(0)
        if action is None:
            continue
        while pending and pending[0] is None:
            pending.pop(0)
        if (
            action.computation_type is _ComputationType.BACKWARD_INPUT
            and pending
            and pending[0] is not None
            and pending[0].computation_type is _ComputationType.BACKWARD_WEIGHT
            and action.stage_index == pending[0].stage_index
            and action.microbatch_index == pending[0].microbatch_index
        ):
            result.append(
                _Action(
                    action.stage_index,
                    _ComputationType.FULL_BACKWARD,
                    action.microbatch_index,
                )
            )
            pending.pop(0)
        else:
            result.append(action)
    return result


def _add_send_recv(
    compute_actions: dict[int, list[_Action | None]],
    stage_to_rank: Any,
    num_stages: int,
) -> dict[int, list[_Action | None]]:
    if callable(stage_to_rank):
        rank_of = stage_to_rank
    else:
        rank_of = lambda stage: stage_to_rank[int(stage)]
    remaining = {
        int(rank): list(actions) for rank, actions in compute_actions.items()
    }
    result: dict[int, list[_Action | None]] = {rank: [] for rank in remaining}
    previous: dict[int, set[_Action]] = {rank: set() for rank in remaining}

    def leaves(action: _Action) -> tuple[_Action, ...]:
        return action.sub_actions or (action,)

    def communicates(action: _Action) -> bool:
        if action.computation_type is _ComputationType.FORWARD:
            return (
                action.stage_index < num_stages - 1
                and rank_of(action.stage_index) != rank_of(action.stage_index + 1)
            )
        if action.computation_type in {
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.FULL_BACKWARD,
        }:
            return (
                action.stage_index > 0
                and rank_of(action.stage_index) != rank_of(action.stage_index - 1)
            )
        return False

    def ready(action: _Action, done: set[_Action]) -> bool:
        if (
            action.computation_type is _ComputationType.FORWARD
            and action.stage_index > 0
        ):
            return (
                _Action(
                    action.stage_index,
                    _ComputationType.RECV_F,
                    action.microbatch_index,
                )
                in done
                or _Action(
                    action.stage_index - 1,
                    _ComputationType.FORWARD,
                    action.microbatch_index,
                )
                in done
            )
        if (
            action.computation_type
            in {_ComputationType.BACKWARD_INPUT, _ComputationType.FULL_BACKWARD}
            and action.stage_index < num_stages - 1
        ):
            return (
                _Action(
                    action.stage_index,
                    _ComputationType.RECV_B,
                    action.microbatch_index,
                )
                in done
                or _Action(
                    action.stage_index + 1,
                    _ComputationType.BACKWARD_INPUT,
                    action.microbatch_index,
                )
                in done
                or _Action(
                    action.stage_index + 1,
                    _ComputationType.FULL_BACKWARD,
                    action.microbatch_index,
                )
                in done
            )
        return True

    while remaining:
        progress = False
        for rank in sorted(tuple(remaining)):
            actions = remaining[rank]
            if not actions:
                del remaining[rank]
                continue
            action = actions[0]
            if action is None:
                result[rank].append(None)
                actions.pop(0)
                progress = True
                if not actions:
                    del remaining[rank]
                continue
            action_leaves = leaves(action)
            if not all(ready(leaf, previous[rank]) for leaf in action_leaves):
                continue
            result[rank].append(action)
            for leaf in action_leaves:
                previous[rank].add(leaf)
                if not communicates(leaf):
                    continue
                is_forward = leaf.computation_type is _ComputationType.FORWARD
                send_kind = (
                    _ComputationType.SEND_F
                    if is_forward
                    else _ComputationType.SEND_B
                )
                recv_kind = (
                    _ComputationType.RECV_F
                    if is_forward
                    else _ComputationType.RECV_B
                )
                peer_stage = leaf.stage_index + 1 if is_forward else leaf.stage_index - 1
                send = _Action(leaf.stage_index, send_kind, leaf.microbatch_index)
                recv = _Action(peer_stage, recv_kind, leaf.microbatch_index)
                result[rank].append(send)
                previous[rank].add(send)
                peer_rank = int(rank_of(peer_stage))
                if peer_rank not in result:
                    raise ValueError(f"stage mapping points to unknown rank {peer_rank}")
                result[peer_rank].append(recv)
                previous[peer_rank].add(recv)
            actions.pop(0)
            progress = True
            if not actions:
                del remaining[rank]
        if not progress:
            raise ValueError("malformed pipeline schedule")
    return result


def _defer_recv_ops(
    actions: list[_Action | None] | dict[int, list[_Action | None]],
    stage_to_rank: Any,
) -> list[_Action | None] | dict[int, list[_Action | None]]:
    rank_of = stage_to_rank if callable(stage_to_rank) else lambda stage: stage_to_rank[int(stage)]
    was_list = isinstance(actions, list)
    by_rank = {0: list(actions)} if was_list else {
        int(rank): list(rank_actions) for rank, rank_actions in actions.items()
    }
    result: dict[int, list[_Action | None]] = {}
    recv_types = {_ComputationType.RECV_F, _ComputationType.RECV_B}
    send_types = {_ComputationType.SEND_F, _ComputationType.SEND_B}

    def recv_peer(action: _Action) -> int:
        peer_stage = action.stage_index - 1 if action.computation_type is _ComputationType.RECV_F else action.stage_index + 1
        return int(rank_of(peer_stage))

    def send_peer(action: _Action) -> int:
        peer_stage = action.stage_index + 1 if action.computation_type is _ComputationType.SEND_F else action.stage_index - 1
        return int(rank_of(peer_stage))

    for rank, rank_actions in by_rank.items():
        deferred: dict[tuple[int, _ComputationType, int | None], _Action] = {}
        output: list[_Action | None] = []
        for action in rank_actions:
            if action is None:
                output.append(None)
                continue
            if action.computation_type in recv_types:
                key = (action.stage_index, action.computation_type, action.microbatch_index)
                deferred[key] = action
                continue
            if action.computation_type in send_types:
                peer = send_peer(action)
                if rank < peer:
                    for key in tuple(deferred):
                        if recv_peer(deferred[key]) == peer:
                            output.append(deferred.pop(key))
            leaves = action.sub_actions or (action,)
            for leaf in leaves:
                if leaf.computation_type is _ComputationType.FORWARD:
                    key = (leaf.stage_index, _ComputationType.RECV_F, leaf.microbatch_index)
                elif leaf.computation_type in {
                    _ComputationType.FULL_BACKWARD,
                    _ComputationType.BACKWARD_INPUT,
                }:
                    key = (leaf.stage_index, _ComputationType.RECV_B, leaf.microbatch_index)
                else:
                    continue
                if key in deferred:
                    output.append(deferred.pop(key))
            output.append(action)
        if deferred:
            raise ValueError("every receive action must have a consuming compute action")
        result[rank] = output
    return result[0] if was_list else result


def _validate_schedule(
    actions: Any,
    pp_group_size: int,
    num_stages: int,
    num_microbatches: int,
) -> dict[int, int]:
    if pp_group_size <= 0 or num_stages <= 0 or num_microbatches <= 0:
        raise ValueError("pipeline dimensions must be positive")
    if not isinstance(actions, dict) or len(actions) != pp_group_size:
        raise ValueError("schedule must provide one action list per rank")
    if set(actions) != set(range(pp_group_size)):
        raise ValueError("schedule ranks must be contiguous")

    stage_actions: dict[int, dict[_ComputationType, set[int]]] = {
        stage: {
            _ComputationType.FORWARD: set(),
            _ComputationType.BACKWARD_INPUT: set(),
            _ComputationType.BACKWARD_WEIGHT: set(),
            _ComputationType.FULL_BACKWARD: set(),
        }
        for stage in range(num_stages)
    }
    stage_index_to_rank: dict[int, int] = {}
    seen: set[tuple[int, _ComputationType, int | None]] = set()

    compute_types = {
        _ComputationType.FORWARD,
        _ComputationType.FULL_BACKWARD,
        _ComputationType.BACKWARD_INPUT,
        _ComputationType.BACKWARD_WEIGHT,
    }
    communication_types = {
        _ComputationType.SEND_F,
        _ComputationType.RECV_F,
        _ComputationType.SEND_B,
        _ComputationType.RECV_B,
    }

    def process_action(action: _Action, rank: int, step: int) -> None:
        if action.sub_actions is not None:
            if action.computation_type is not _ComputationType.OVERLAP_F_B:
                raise ValueError("only overlap actions may contain sub-actions")
            if not action.sub_actions:
                raise ValueError("an overlap action must contain sub-actions")
            for sub_action in action.sub_actions:
                if not isinstance(sub_action, _Action):
                    raise TypeError("sub-actions must be _Action instances")
                process_action(sub_action, rank, step)
            return

        stage = action.stage_index
        kind = action.computation_type
        microbatch = action.microbatch_index
        if not 0 <= stage < num_stages:
            raise ValueError("action stage is outside the pipeline")
        if kind not in compute_types | communication_types | {
            _ComputationType.UNSHARD,
            _ComputationType.RESHARD,
            _ComputationType.REDUCE_GRAD,
        }:
            raise ValueError(f"unsupported pipeline action {kind!r}")
        if kind in compute_types | communication_types:
            if microbatch is None or not 0 <= microbatch < num_microbatches:
                raise ValueError("action microbatch is outside the schedule")
        elif microbatch is not None:
            raise ValueError("non-compute actions cannot carry a microbatch")

        previous_rank = stage_index_to_rank.get(stage)
        if previous_rank is not None and previous_rank != rank:
            raise ValueError(
                f"stage {stage} is assigned to ranks {previous_rank} and {rank}"
            )
        stage_index_to_rank[stage] = rank
        if kind not in compute_types:
            return

        key = (stage, kind, microbatch)
        if key in seen:
            raise ValueError("a compute action occurs more than once")
        seen.add(key)
        if kind is _ComputationType.FORWARD:
            stage_actions[stage][kind].add(microbatch)
            return
        if kind is _ComputationType.FULL_BACKWARD:
            if microbatch not in stage_actions[stage][_ComputationType.FORWARD]:
                raise ValueError("backward ran before its forward")
            stage_actions[stage][kind].add(microbatch)
            return
        if kind is _ComputationType.BACKWARD_INPUT:
            if microbatch not in stage_actions[stage][_ComputationType.FORWARD]:
                raise ValueError("backward input ran before its forward")
            stage_actions[stage][kind].add(microbatch)
            return
        if microbatch not in stage_actions[stage][_ComputationType.BACKWARD_INPUT]:
            raise ValueError("backward weight ran before its input backward")
        stage_actions[stage][kind].add(microbatch)

    for rank, rank_actions in actions.items():
        if not isinstance(rank_actions, list):
            raise TypeError(f"actions for rank {rank} must be a list")
        for step, action in enumerate(rank_actions):
            if action is None:
                continue
            if not isinstance(action, _Action):
                raise TypeError("schedule entries must be _Action instances")
            process_action(action, rank, step)
    for stage in range(num_stages):
        counts = stage_actions[stage]
        if len(counts[_ComputationType.FORWARD]) != num_microbatches:
            raise ValueError("schedule is missing a forward action")
        if len(counts[_ComputationType.BACKWARD_INPUT]) != len(
            counts[_ComputationType.BACKWARD_WEIGHT]
        ):
            raise ValueError("input and weight backward counts must match")
        if len(counts[_ComputationType.FULL_BACKWARD]) + len(
            counts[_ComputationType.BACKWARD_INPUT]
        ) != num_microbatches:
            raise ValueError("schedule is missing a backward action")
    if len(stage_index_to_rank) != num_stages:
        raise ValueError("schedule does not assign every pipeline stage")
    return stage_index_to_rank


def _get_1f1b_rank_ops(
    n_local_stages: int,
    pp_group_size: int,
    warmup_ops: int,
    fwd_bwd_ops: int,
    cooldown_ops: int,
    rank: int,
    forward_stage_index: Any,
    backward_stage_index: Any,
    num_1f1b_microbatches: int = 0,
    enable_zero_bubble: bool = False,
) -> list[_Action | None]:
    if min(n_local_stages, pp_group_size) <= 0:
        raise ValueError("pipeline dimensions must be positive")
    if rank < 0 or rank >= pp_group_size:
        raise ValueError("rank is outside the pipeline group")
    if min(warmup_ops, fwd_bwd_ops, cooldown_ops) < 0:
        raise ValueError("operation counts must be non-negative")
    forward_counts: dict[int, int] = defaultdict(int)
    backward_counts: dict[int, int] = defaultdict(int)
    weight_counts: dict[int, int] = defaultdict(int)
    result: list[_Action | None] = [None] * rank
    backward_ids: list[int] = []
    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
    post_warmup = (
        n_local_stages * pp_group_size
        + 2 * (pp_group_size - 1 - rank)
        - warmup_ops
        - rank
    )
    if enable_zero_bubble:
        post_warmup = pp_group_size - rank - 1

    for operation in range(total_ops):
        if operation < warmup_ops:
            stage = int(forward_stage_index(operation))
            microbatch = forward_counts[stage]
            forward_counts[stage] += 1
            result.append(_Action(stage, _ComputationType.FORWARD, microbatch))
            if operation == warmup_ops - 1:
                result.extend([None] * max(0, post_warmup))
            continue

        if operation < warmup_ops + fwd_bwd_ops:
            stage = int(forward_stage_index(operation))
            microbatch = forward_counts[stage]
            forward_counts[stage] += 1
            result.append(_Action(stage, _ComputationType.FORWARD, microbatch))
            backward_stage = int(backward_stage_index(operation))
            microbatch = backward_counts[backward_stage]
            backward_counts[backward_stage] += 1
            backward_kind = (
                _ComputationType.BACKWARD_INPUT
                if enable_zero_bubble
                else _ComputationType.FULL_BACKWARD
            )
            result.append(
                _Action(backward_stage, backward_kind, microbatch)
            )
            backward_ids.append(operation)
            if (
                enable_zero_bubble
                and operation - warmup_ops >= num_1f1b_microbatches
            ):
                weight_index = sum(weight_counts.values())
                weight_stage = int(backward_stage_index(backward_ids[weight_index]))
                microbatch = weight_counts[weight_stage]
                weight_counts[weight_stage] += 1
                result.append(
                    _Action(
                        weight_stage,
                        _ComputationType.BACKWARD_WEIGHT,
                        microbatch,
                    )
                )
            continue

        if not enable_zero_bubble:
            result.append(None)
        backward_stage = int(backward_stage_index(operation))
        microbatch = backward_counts[backward_stage]
        backward_counts[backward_stage] += 1
        backward_kind = (
            _ComputationType.BACKWARD_INPUT
            if enable_zero_bubble
            else _ComputationType.FULL_BACKWARD
        )
        result.append(_Action(backward_stage, backward_kind, microbatch))
        backward_ids.append(operation)
        if (
            enable_zero_bubble
            and operation - warmup_ops >= num_1f1b_microbatches
        ):
            weight_index = sum(weight_counts.values())
            weight_stage = int(backward_stage_index(backward_ids[weight_index]))
            microbatch = weight_counts[weight_stage]
            weight_counts[weight_stage] += 1
            result.append(
                _Action(
                    weight_stage,
                    _ComputationType.BACKWARD_WEIGHT,
                    microbatch,
                )
            )

    while enable_zero_bubble and sum(weight_counts.values()) < len(backward_ids):
        weight_index = sum(weight_counts.values())
        weight_stage = int(backward_stage_index(backward_ids[weight_index]))
        microbatch = weight_counts[weight_stage]
        weight_counts[weight_stage] += 1
        result.append(
            _Action(weight_stage, _ComputationType.BACKWARD_WEIGHT, microbatch)
        )
    return result


def _get_warmup_ops(*args: Any, **kwargs: Any) -> list[_Action]:
    values = list(args)
    if values:
        n_microbatches = int(values[0])
        num_stages = int(
            values[1] if len(values) > 1 else kwargs.get("num_stages", 1)
        )
        stage_index = int(
            values[2] if len(values) > 2 else kwargs.get("stage_index", 0)
        )
    else:
        n_microbatches = int(
            kwargs.get("n_microbatches", kwargs.get("num_microbatches", 0))
        )
        num_stages = int(kwargs.get("num_stages", 1))
        stage_index = int(kwargs.get("stage_index", 0))
    if (
        n_microbatches <= 0
        or num_stages <= 0
        or stage_index < 0
        or stage_index >= num_stages
    ):
        raise ValueError("invalid warmup dimensions")
    count = min(n_microbatches, num_stages - stage_index)
    return [
        _Action(stage_index, _ComputationType.FORWARD, index)
        for index in range(count)
    ]


def get_schedule_class(schedule_name: str) -> type[_PipelineSchedule]:
    mapping = {
        "GPipe": ScheduleGPipe,
        "1F1B": Schedule1F1B,
        "Interleaved1F1B": ScheduleInterleaved1F1B,
        "LoopedBFS": ScheduleLoopedBFS,
        "InterleavedZeroBubble": ScheduleInterleavedZeroBubble,
        "ZBVZeroBubble": ScheduleZBVZeroBubble,
        "DualPipeV": ScheduleDualPipeV,
        "PipelineScheduleSingle": PipelineScheduleSingle,
        "PipelineScheduleMulti": PipelineScheduleMulti,
    }
    if not isinstance(schedule_name, str):
        raise TypeError("schedule name must be a string")
    normalized = {name.lower(): cls for name, cls in mapping.items()}
    try:
        return normalized[schedule_name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown pipeline schedule {schedule_name!r}") from exc


def _simulate_comms_compute(pipeline_order: Any, stage_to_rank: Any, num_stages: int) -> Any:
    if not isinstance(pipeline_order, dict):
        raise TypeError("pipeline_order must be a rank-to-actions mapping")
    if callable(stage_to_rank):
        rank_of = stage_to_rank
    else:
        rank_of = lambda stage: stage_to_rank[int(stage)]
    pending = {
        int(rank): [action for action in actions if action is not None]
        for rank, actions in pipeline_order.items()
    }
    schedule: dict[int, list[_Action | None]] = {
        rank: [] for rank in sorted(pending)
    }
    completed: dict[int, set[_Action]] = {
        rank: set() for rank in pending
    }

    def leaves(action: _Action) -> tuple[_Action, ...]:
        return action.sub_actions or (action,)

    def ready_leaf(action: _Action, owner: int) -> bool:
        if action.stage_index < 0 or action.stage_index >= num_stages:
            raise ValueError("action stage is outside the pipeline")
        owner = int(rank_of(action.stage_index)) if action.stage_index >= 0 else owner
        done = completed[owner]
        kind = action.computation_type
        stage = action.stage_index
        microbatch = action.microbatch_index
        if kind is _ComputationType.FORWARD:
            if stage == 0:
                return True
            return (
                _Action(stage, _ComputationType.RECV_F, microbatch) in done
                or _Action(stage - 1, _ComputationType.FORWARD, microbatch) in done
            )
        if kind in {
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.FULL_BACKWARD,
        }:
            if stage == num_stages - 1:
                return True
            return (
                _Action(stage, _ComputationType.RECV_B, microbatch) in done
                or _Action(stage + 1, _ComputationType.BACKWARD_INPUT, microbatch) in done
                or _Action(stage + 1, _ComputationType.FULL_BACKWARD, microbatch) in done
            )
        if kind is _ComputationType.SEND_F:
            return _Action(stage, _ComputationType.FORWARD, microbatch) in done
        if kind is _ComputationType.RECV_F:
            peer = stage - 1
            return _Action(peer, _ComputationType.SEND_F, microbatch) in completed[int(rank_of(peer))]
        if kind is _ComputationType.SEND_B:
            return (
                _Action(stage, _ComputationType.BACKWARD_INPUT, microbatch) in done
                or _Action(stage, _ComputationType.FULL_BACKWARD, microbatch) in done
            )
        if kind is _ComputationType.RECV_B:
            peer = stage + 1
            return _Action(peer, _ComputationType.SEND_B, microbatch) in completed[int(rank_of(peer))]
        if kind in {
            _ComputationType.BACKWARD_WEIGHT,
        }:
            return _Action(
                action.stage_index,
                _ComputationType.BACKWARD_INPUT,
                action.microbatch_index,
            ) in done
        if kind in {
            _ComputationType.UNSHARD,
            _ComputationType.RESHARD,
            _ComputationType.REDUCE_GRAD,
        }:
            return True
        raise ValueError(f"unsupported pipeline action {kind!r}")

    def ready(action: _Action, owner: int) -> bool:
        return all(ready_leaf(leaf, owner) for leaf in leaves(action))

    def mark_completed(action: _Action, owner: int) -> None:
        completed[owner].add(action)
        for leaf in leaves(action):
            leaf_owner = int(rank_of(leaf.stage_index))
            completed[leaf_owner].add(leaf)

    while pending:
        progress = False
        for rank in sorted(tuple(pending)):
            actions = pending[rank]
            if not actions:
                del pending[rank]
                continue
            action = actions[0]
            if not ready(action, rank):
                schedule[rank].append(None)
                continue
            schedule[rank].append(action)
            mark_completed(action, rank)
            actions.pop(0)
            progress = True
            if not actions:
                del pending[rank]

        for rank in sorted(tuple(pending)):
            if not schedule[rank] or schedule[rank][-1] is not None:
                continue
            action = pending[rank][0]
            if ready(action, rank):
                schedule[rank][-1] = action
                mark_completed(action, rank)
                pending[rank].pop(0)
                if not pending[rank]:
                    del pending[rank]
                progress = True
        if not progress:
            raise ValueError("pipeline schedule cannot make progress")
    return schedule


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
