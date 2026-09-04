"""Microbatch pipeline schedules."""

import csv
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Protocol, cast

import tensorplay as tp

from .. import distributed_core as dist
from ..fsdp._fully_shard import FSDPModule, UnshardHandle
from ._utils import InferenceMode, generate_rank_to_stage_mapping, generate_stage_to_rank_mapping
from .microbatch import TensorChunkSpec, merge_chunks, split_args_kwargs_into_chunks, _split_tensor

logger = logging.getLogger(__name__)

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

    def __str__(self) -> str:
        return self.__repr__()

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
        del p2p_done
        pipeline_stages = [
            stage for stage in stages if hasattr(stage, "_user_meta")
        ]
        if len(pipeline_stages) != len(stages):
            if not dist.is_initialized():
                return
            operations = [
                operation
                for stage in stages
                for operation in stage._get_init_p2p_neighbors_ops()
            ]
            _wait_batch_p2p(_batch_p2p(operations))
            return
        if not dist.is_initialized():
            for stage in pipeline_stages:
                stage._inference_mode = (
                    InferenceMode.DYNAMIC
                    if InferenceMode.needs_dynamic(stage._user_meta, has_backward)
                    else InferenceMode.STATIC
                )
            return
        has_cross_rank = any(
            (not stage.is_first and not stage._is_same_rank(stage.stage_index - 1))
            or (not stage.is_last and not stage._is_same_rank(stage.stage_index + 1))
            for stage in pipeline_stages
        )
        if has_cross_rank and any(
            dist.get_backend(stage.group) == "fake" for stage in pipeline_stages
        ):
            for stage in pipeline_stages:
                if InferenceMode.needs_dynamic(stage._user_meta, has_backward):
                    raise RuntimeError(
                        f"Stage {stage.stage_index} requires dynamic metadata with a fake process group"
                    )
                stage._inference_mode = InferenceMode.STATIC
            return
        accumulated = None
        for stage in pipeline_stages:
            accumulated = stage._warmup_forward_vote(
                has_backward,
                received_acc=accumulated,
            )
        result = accumulated
        for stage in reversed(pipeline_stages):
            result = stage._warmup_backward_result(received_result=result)
            stage._inference_mode = (
                InferenceMode.STATIC if int(result.item()) == 1 else InferenceMode.DYNAMIC
            )

    def _initialize_pp_stages(
        self,
        stages: list[Any],
        args: Any,
        kwargs: Any,
        target: Any,
        fwd_initialized: Any,
        bwd_initialized: Any,
        loss_kwargs: Any,
    ) -> tuple[bool, bool]:
        if fwd_initialized and self._has_backward != bwd_initialized:
            fwd_initialized = False
            bwd_initialized = False
        if not fwd_initialized:
            self._warmup_p2p(stages, self._has_backward, fwd_initialized)
            for stage in stages:
                stage.has_backward = self._has_backward
                backup = getattr(stage, "_pre_metadata_inference_backup", None)
                if callable(backup):
                    backup()
            try:
                next_stage_args = None
                for stage in stages:
                    stage_args = args if stage.is_first else next_stage_args
                    next_stage_args = stage._prepare_forward_infra(
                        self._n_microbatches,
                        stage_args,
                        kwargs,
                        self._has_backward,
                    )
                    fwd_initialized = True
                if self._has_backward and not bwd_initialized:
                    previous_grad_meta = None
                    for stage in reversed(stages):
                        previous_grad_meta = stage._prepare_backward_infra(
                            self._n_microbatches,
                            loss_fn=self._loss_fn,
                            target=target,
                            received_grad_meta=previous_grad_meta,
                            loss_kwargs=loss_kwargs,
                        )
                    bwd_initialized = True
            finally:
                for stage in stages:
                    cleanup = getattr(stage, "_post_metadata_inference_cleanup", None)
                    if callable(cleanup):
                        cleanup()
        elif self._has_backward and not bwd_initialized:
            previous_grad_meta = None
            for stage in reversed(stages):
                previous_grad_meta = stage._prepare_backward_infra(
                    self._n_microbatches,
                    loss_fn=self._loss_fn,
                    target=target,
                    received_grad_meta=previous_grad_meta,
                    loss_kwargs=loss_kwargs,
                )
            bwd_initialized = True
        return fwd_initialized, bwd_initialized

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

    def _check_inputs(
        self,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        losses: Any = None,
    ) -> tuple[list[Any], list[Any]]:
        def check_type_and_len(value: Any, name: str) -> None:
            if not isinstance(value, list):
                raise TypeError(f"{name} must be a list but got a {type(value)}")
            if len(value) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(value)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None and not isinstance(losses, list):
            raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output: Any, target: Any, loss_kwargs: dict[str, Any] | None = None) -> Any:
        return self._loss_fn(output, target, **(loss_kwargs or {}))

    def _split_inputs(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
        if args or kwargs:
            return split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
        return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _get_microbatch_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        arg_mbs: Any,
        kwarg_mbs: Any,
        target_mbs: Any,
    ) -> tuple[list[Any], list[Any], list[Any] | None]:
        pre_split = any(
            value is not None for value in (arg_mbs, kwarg_mbs, target_mbs)
        )
        if not pre_split:
            args_split, kwargs_split = self._split_inputs(args, kwargs)
            target_split = (
                list(_split_tensor(target, TensorChunkSpec(0), self._n_microbatches))
                if target is not None
                else None
            )
            return args_split, kwargs_split, target_split

        if args:
            raise ValueError(
                "When using pre-split inputs, pass pre-split positional inputs "
                "through arg_mbs=... instead of positional args."
            )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Unexpected keyword arguments with pre-split inputs: {names}. "
                "Pass pre-split keyword inputs through kwarg_mbs=."
            )
        if target is not None:
            raise ValueError(
                "When using pre-split inputs, pass pre-split targets through "
                "target_mbs=... instead of target=."
            )

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs)
        for mb_index, (arg_mb, kwarg_mb) in enumerate(
            zip(arg_mbs, kwarg_mbs, strict=True)
        ):
            if not isinstance(arg_mb, tuple):
                raise TypeError(
                    "arg_mbs must be a list of tuples, but "
                    f"arg_mbs[{mb_index}] is a {type(arg_mb)}"
                )
            if not isinstance(kwarg_mb, dict):
                raise TypeError(
                    "kwarg_mbs must be a list of dicts, but "
                    f"kwarg_mbs[{mb_index}] is a {type(kwarg_mb)}"
                )
        return arg_mbs, kwarg_mbs, target_mbs

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
        self._stage_forward_initialized = False
        self._stage_backward_initialized = False
        self.pipeline_order = self._get_pipeline_order()
        self._stage.has_backward = self._has_backward

    def _initialize_stage(self, args: Any, kwargs: Any, target: Any = None, loss_kwargs: Any = None) -> None:
        (
            self._stage_forward_initialized,
            self._stage_backward_initialized,
        ) = self._initialize_pp_stages(
            [self._stage],
            args,
            kwargs,
            target,
            self._stage_forward_initialized,
            self._stage_backward_initialized,
            loss_kwargs,
        )

    def step(
        self,
        *args: Any,
        target: Any = None,
        losses: list[Any] | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._has_backward and not tp.is_grad_enabled():
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation"
            )
        self._stage.has_backward = self._has_backward
        self._stage.clear_runtime_states()
        args_split, kwargs_split, targets_split = self._get_microbatch_inputs(
            args,
            kwargs,
            target,
            arg_mbs,
            kwarg_mbs,
            target_mbs,
        )
        self._losses = []
        self._initialize_stage(
            tuple(args_split[0]) if args_split else args,
            dict(kwargs_split[0]) if kwargs_split else kwargs,
            targets_split[0] if targets_split else None,
            loss_kwargs,
        )
        self._step_microbatches(
            args_split,
            kwargs_split,
            targets_split,
            losses,
            return_outputs,
            loss_kwargs=loss_kwargs,
        )
        if self._stage.is_last and return_outputs and self._stage.output_chunks:
            return self._merge_outputs(self._stage.output_chunks)
        return None

    def _step_microbatches(self, arg_mbs: list[tuple[Any, ...]], kwarg_mbs: list[dict[str, Any]], target_mbs: list[Any] | None, losses: list[Any] | None, return_outputs: bool = True, loss_kwargs: dict[str, Any] | None = None) -> Any:
        self._stage.clear_runtime_states()
        outputs = []
        forward_sends = []
        for index, (args, kwargs) in enumerate(zip(arg_mbs, kwarg_mbs)):
            for work in _run_p2p(self._stage.get_fwd_recv_ops(index)):
                work.wait()
            output = self._stage.forward_one_chunk(
                index,
                args,
                kwargs,
                save_forward_output=return_outputs,
            )
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
                    self._stage.backward_one_chunk(
                        index,
                        last_backward=index == len(outputs) - 1,
                    )
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


def _batch_p2p(operations: list[Any], desc: str | None = None) -> list[Any]:
    del desc
    if not operations:
        return []
    operations_by_group: dict[str, list[Any]] = defaultdict(list)
    for operation in operations:
        group = operation.group
        group_name = getattr(group, "group_name", None)
        operations_by_group[str(group_name if group_name is not None else group)].append(
            operation
        )
    if len(operations_by_group) > 1:
        works: list[Any] = []
        for _, group_operations in sorted(operations_by_group.items()):
            works.extend(_batch_p2p(group_operations))
        return works

    operation_types = {operation.op for operation in operations}
    if operation_types == {dist.isend}:
        return [
            work
            for work in (
                operation.op(
                    operation.tensor,
                    group=operation.group,
                    tag=operation.tag,
                    group_dst=operation.group_peer,
                )
                for operation in operations
            )
            if work is not None
        ]
    if operation_types == {dist.irecv}:
        return [
            work
            for work in (
                operation.op(
                    operation.tensor,
                    group=operation.group,
                    tag=operation.tag,
                    group_src=operation.group_peer,
                )
                for operation in operations
            )
            if work is not None
        ]
    return dist.batch_isend_irecv(operations)


def _sorted_batch_p2p(
    operations: list[Any], desc: str | None = None
) -> dict[int, list[Any]]:
    del desc
    operations_by_peer: dict[int, list[Any]] = defaultdict(list)
    works_by_peer: dict[int, list[Any]] = {}
    for operation in operations:
        operations_by_peer[int(operation.peer)].append(operation)
    for peer, peer_operations in sorted(operations_by_peer.items()):
        works_by_peer[peer] = _batch_p2p(peer_operations)
    return works_by_peer


def _wait_batch_p2p(works: list[Any]) -> None:
    for work in works:
        work.wait()


def _run_p2p(operations: list[Any]) -> list[Any]:
    return [
        work
        for peer_works in _sorted_batch_p2p(operations).values()
        for work in peer_works
    ]


def _normalize_stage_args(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


class _ScheduleForwardOnly(PipelineScheduleSingle):
    def _step_microbatches(self, *args: Any, **kwargs: Any) -> Any:
        arg_mbs = kwargs.pop("arg_mbs", args[0] if args else None)
        kwarg_mbs = kwargs.pop("kwarg_mbs", args[1] if len(args) > 1 else None)
        target_mbs = kwargs.pop("target_mbs", args[2] if len(args) > 2 else None)
        losses = kwargs.pop("losses", args[3] if len(args) > 3 else None)
        return_outputs = kwargs.pop(
            "return_outputs", args[4] if len(args) > 4 else True
        )
        if target_mbs is not None or losses is not None:
            raise RuntimeError("forward-only schedule does not support loss computation")
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])
        self._stage.clear_runtime_states()
        send_works: list[Any] = []
        for index in range(self._n_microbatches):
            _wait_batch_p2p(
                _batch_p2p(
                    self._stage.get_fwd_recv_ops(index), desc="fwd_recv"
                )
            )
            self._stage.forward_one_chunk(
                index,
                arg_mbs[index],
                kwarg_mbs[index],
                save_forward_output=return_outputs,
            )
            send_works.extend(
                _batch_p2p(
                    self._stage.get_fwd_send_ops(index), desc="fwd_send"
                )
            )
        _wait_batch_p2p(send_works)
        if not return_outputs or not self._stage.is_last:
            return None
        return self._merge_outputs(self._stage.output_chunks)


class ScheduleGPipe(PipelineScheduleSingle):
    """Execute all forward microbatches before draining their backwards."""

    def _step_microbatches(
        self,
        arg_mbs: list[tuple[Any, ...]],
        kwarg_mbs: list[dict[str, Any]],
        target_mbs: list[Any] | None,
        losses: list[Any] | None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )
        outputs: list[Any] = []
        forward_sends: list[Any] = []
        for index in range(self._n_microbatches):
            _wait_batch_p2p(
                [
                    work
                    for works in _sorted_batch_p2p(
                        self._stage.get_fwd_recv_ops(index), desc="fwd_recv"
                    ).values()
                    for work in works
                ]
            )
            output = self._stage.forward_one_chunk(
                index,
                arg_mbs[index],
                kwarg_mbs[index],
                save_forward_output=return_outputs,
            )
            outputs.append(output)
            forward_sends.extend(
                work
                for works in _sorted_batch_p2p(
                    self._stage.get_fwd_send_ops(index), desc="fwd_send"
                ).values()
                for work in works
            )
            self._maybe_compute_loss(
                self._stage, output, target_mbs, index, loss_kwargs
            )

        _wait_batch_p2p(forward_sends)

        backward_sends: list[Any] = []
        if self._has_backward:
            for index in range(self._n_microbatches):
                _wait_batch_p2p(
                    [
                        work
                        for works in _sorted_batch_p2p(
                            self._stage.get_bwd_recv_ops(index), desc="bwd_recv"
                        ).values()
                        for work in works
                    ]
                )
                self._stage.backward_one_chunk(
                    index,
                    loss=self._maybe_get_loss(self._stage, index),
                    last_backward=index == self._n_microbatches - 1,
                )
                backward_sends.extend(
                    work
                    for works in _sorted_batch_p2p(
                        self._stage.get_bwd_send_ops(index), desc="bwd_send"
                    ).values()
                    for work in works
                )
            _wait_batch_p2p(backward_sends)
            self._stage.perform_reduce_grad(
                self._n_microbatches if self._scale_grads else 1
            )

        self._update_losses(self._stage, losses)
        if not return_outputs or not self._stage.is_last:
            return None
        return self._merge_outputs(outputs)

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
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )
        warmup = min(self._n_microbatches, self._num_stages - self._stage.stage_index)
        outputs: list[Any] = []
        fwd_mb_index = 0
        bwd_mb_index = 0
        send_work: list[Any] = []
        fwd_sends: list[Any] = []

        for _ in range(warmup):
            _wait_batch_p2p(
                _batch_p2p(
                    self._stage.get_fwd_recv_ops(fwd_mb_index), desc="fwd_recv"
                )
            )
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )
            outputs.append(output)
            _wait_batch_p2p(send_work)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if not self._has_backward or fwd_mb_index != warmup - 1:
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            self._maybe_compute_loss(
                self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
            )
            fwd_mb_index += 1

        if not self._has_backward:
            for fwd_mb_index in range(fwd_mb_index, self._n_microbatches):
                _wait_batch_p2p(
                    _batch_p2p(
                        self._stage.get_fwd_recv_ops(fwd_mb_index),
                        desc="fwd_recv",
                    )
                )
                output = self._stage.forward_one_chunk(
                    fwd_mb_index,
                    arg_mbs[fwd_mb_index],
                    kwarg_mbs[fwd_mb_index],
                    save_forward_output=return_outputs,
                )
                outputs.append(output)
                _wait_batch_p2p(send_work)
                send_work = _batch_p2p(
                    self._stage.get_fwd_send_ops(fwd_mb_index), desc="fwd_send"
                )
                self._maybe_compute_loss(
                    self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
                )
            _wait_batch_p2p(send_work)
        else:
            while True:
                _wait_batch_p2p(
                    _batch_p2p(
                        fwd_sends + self._stage.get_bwd_recv_ops(bwd_mb_index),
                        desc="fwd_send_bwd_recv",
                    )
                )
                self._stage.backward_one_chunk(
                    bwd_mb_index,
                    loss=self._maybe_get_loss(self._stage, bwd_mb_index),
                    last_backward=bwd_mb_index == self._n_microbatches - 1,
                )
                bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
                bwd_mb_index += 1
                if fwd_mb_index == self._n_microbatches:
                    break

                _wait_batch_p2p(
                    _batch_p2p(
                        bwd_sends + self._stage.get_fwd_recv_ops(fwd_mb_index),
                        desc="bwd_send_fwd_recv",
                    )
                )
                output = self._stage.forward_one_chunk(
                    fwd_mb_index,
                    arg_mbs[fwd_mb_index],
                    kwarg_mbs[fwd_mb_index],
                    save_forward_output=return_outputs,
                )
                outputs.append(output)
                self._maybe_compute_loss(
                    self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
                )
                fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
                fwd_mb_index += 1

            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            while bwd_mb_index < self._n_microbatches:
                _wait_batch_p2p(
                    _batch_p2p(
                        self._stage.get_bwd_recv_ops(bwd_mb_index),
                        desc="bwd_recv",
                    )
                )
                self._stage.backward_one_chunk(
                    bwd_mb_index,
                    loss=self._maybe_get_loss(self._stage, bwd_mb_index),
                    last_backward=bwd_mb_index == self._n_microbatches - 1,
                )
                _wait_batch_p2p(send_work)
                send_work = _batch_p2p(
                    self._stage.get_bwd_send_ops(bwd_mb_index), desc="bwd_send"
                )
                bwd_mb_index += 1
            _wait_batch_p2p(send_work)

            self._stage.perform_reduce_grad(
                self._n_microbatches if self._scale_grads else 1
            )

        self._update_losses(self._stage, losses)
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
    def __init__(self, stages: list[Any], n_microbatches: int, loss_fn: Any = None, args_chunk_spec: Any = None, kwargs_chunk_spec: Any = None, output_merge_spec: Any = None, use_full_backward: bool | None = None, scale_grads: bool = True, backward_requires_autograd: bool = True) -> None:
        if not stages:
            raise ValueError("at least one pipeline stage is required")
        super().__init__(n_microbatches, loss_fn, args_chunk_spec, kwargs_chunk_spec, output_merge_spec, scale_grads)
        self._stages = list(stages)
        self.use_full_backward = use_full_backward
        self.backward_requires_autograd = backward_requires_autograd
        self._backward_requires_autograd = backward_requires_autograd
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
        self._stages_forward_initialized = False
        self._stages_backward_initialized = False
        self.pipeline_order: dict[int, list[_Action | None]] = {}
        if use_full_backward is not None:
            logger.warning(
                "use_full_backward is no longer supported; omit it from the schedule"
            )

    def _initialize_stages(self, args: Any, kwargs: Any, target: Any = None, loss_kwargs: Any = None) -> None:
        reinit_for_mode_switch = self._stages_forward_initialized and (
            self._has_backward != self._stages_backward_initialized
        )
        forward_initialized_before = self._stages_forward_initialized
        (
            self._stages_forward_initialized,
            self._stages_backward_initialized,
        ) = self._initialize_pp_stages(
            self._stages,
            args,
            kwargs,
            target,
            self._stages_forward_initialized,
            self._stages_backward_initialized,
            loss_kwargs,
        )
        if self._stages_forward_initialized and (
            not forward_initialized_before or reinit_for_mode_switch
        ):
            self._validate_adjacent_stage_communication()

    def step(
        self,
        *args: Any,
        target: Any = None,
        losses: list[Any] | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        **kwargs: Any,
    ) -> Any:
        if (
            self._has_backward
            and self._backward_requires_autograd
            and not tp.is_grad_enabled()
        ):
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation"
            )
        for stage in self._stages:
            stage.has_backward = self._has_backward
            stage.clear_runtime_states()
        args_split, kwargs_split, targets_split = self._get_microbatch_inputs(
            args,
            kwargs,
            target,
            arg_mbs,
            kwarg_mbs,
            target_mbs,
        )
        self._losses = []
        self._initialize_stages(
            tuple(args_split[0]) if args_split else args,
            dict(kwargs_split[0]) if kwargs_split else kwargs,
            targets_split[0] if targets_split else None,
            loss_kwargs,
        )
        self._step_microbatches(
            args_split,
            kwargs_split,
            targets_split,
            losses,
            return_outputs,
            loss_kwargs=loss_kwargs,
        )
        if return_outputs:
            for stage in self._stages:
                if stage.is_last and stage.output_chunks:
                    return self._merge_outputs(stage.output_chunks)
        return None

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
        def check_stage_indices(
            stage_index: int,
            direction: str,
            actual: set[int],
            expected: set[int],
        ) -> None:
            non_adjacent = actual - expected
            if non_adjacent:
                raise RuntimeError(
                    f"stage {stage_index} has non-adjacent {direction} stages "
                    f"{sorted(non_adjacent)}; expected only {sorted(expected)}"
                )

        for stage in self._stages:
            stage_index = stage.stage_index
            forward_sources = {
                int(getattr(info, "source"))
                for info in stage.args_recv_info.get(0, ())
                if getattr(info, "source", None) is not None
            }
            expected_sources = set() if stage.is_first else {stage_index - 1}
            check_stage_indices(
                stage_index,
                "forward receive",
                forward_sources,
                expected_sources,
            )
            forward_destinations = {
                int(destination)
                for destinations in stage.act_send_info.values()
                for destination in destinations
                if destination is not None
            }
            expected_destinations = set() if stage.is_last else {stage_index + 1}
            check_stage_indices(
                stage_index,
                "forward send",
                forward_destinations,
                expected_destinations,
            )

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
            for rank in sorted(self.pipeline_order):
                writer.writerow(self.pipeline_order[rank])

    def _load_csv(
        self,
        filename: str,
        format: Literal["compute_only", "compute_comms"] = "compute_only",
    ) -> dict[int, list[_Action | None]]:
        if format != "compute_only":
            raise AssertionError(f"format must be compute_only, got {format}")
        with open(filename, newline="", encoding="utf-8") as stream:
            actions = {
                rank: [_Action.from_str(value) for value in row]
                for rank, row in enumerate(csv.reader(stream))
            }
        self.pipeline_order = actions
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


@dataclass
class _PipelineContext:
    schedule_ref: _PipelineSchedule
    arg_mbs: list[tuple[Any, ...]] | None = None
    kwarg_mbs: list[dict[str, Any]] | None = None
    target_mbs: list[Any] | None = None
    losses: list[Any] | None = None


class _CustomFunctionProtocol(Protocol):
    def __call__(self, action: _Action, ctx: _PipelineContext) -> None: ...


class _PipelineScheduleRuntime(PipelineScheduleMulti):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._defer_pp_recv = bool(kwargs.pop("defer_pp_recv", False))
        max_active_stages = kwargs.pop("max_active_stages", 3)
        self._max_active_stages = 3 if max_active_stages is None else int(max_active_stages)
        super().__init__(*args, **kwargs)
        self._comp_type_to_function_map: dict[_ComputationType, Callable[..., Any]] = {}
        self.backward_counter: Counter[int] = Counter()
        self.bwd_recv_ops: dict[tuple[int, int], list[Any]] = {}
        self.fwd_recv_ops: dict[tuple[int, int], list[Any]] = {}
        self.unshard_ops: dict[int, list[UnshardHandle]] = defaultdict(list)
        self.unsharded_stages: set[int] = set()
        self.pipeline_order_with_comms: dict[int, list[_Action | None]] | None = None

    def register_custom_function(
        self,
        computation_type: _ComputationType,
        custom_function: _CustomFunctionProtocol,
    ) -> None:
        supported = {
            _ComputationType.FORWARD,
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.BACKWARD_WEIGHT,
            _ComputationType.OVERLAP_F_B,
            _ComputationType.UNSHARD,
            _ComputationType.RESHARD,
            _ComputationType.REDUCE_GRAD,
        }
        if computation_type not in supported:
            raise ValueError(f"invalid computation type {computation_type}")
        if computation_type in self._comp_type_to_function_map:
            logger.warning(
                "computation type %s is already registered; replacing it",
                computation_type,
            )
        self._comp_type_to_function_map[computation_type] = custom_function

    def _prepare_schedule_with_comms(
        self,
        actions: dict[int, list[_Action | None]],
        format: Literal["compute_only", "compute_comms"] = "compute_only",
    ) -> None:
        super()._validate_and_set_stage_mapping(actions)
        if format == "compute_comms":
            lowered: dict[int, list[_Action | None]] = {}
            for rank, rank_actions in actions.items():
                if any(action is None for action in rank_actions):
                    raise AssertionError("communication schedules cannot contain empty actions")
                lowered[rank] = list(rank_actions)
            self.pipeline_order_with_comms = lowered
            return
        if format != "compute_only":
            raise NotImplementedError(f"{format=} is not implemented")
        for rank, rank_actions in actions.items():
            for index, action in enumerate(rank_actions):
                if action is not None and not action.is_compute_op:
                    raise ValueError(
                        f"expected compute-only action at rank {rank}, position {index}: {action}"
                    )
        lowered = {
            rank: _add_unshard_reshard(
                rank_actions, max_active_stages=self._max_active_stages
            )
            for rank, rank_actions in actions.items()
        }
        lowered = {
            rank: _add_reduce_grad(rank_actions, self._n_microbatches)
            for rank, rank_actions in lowered.items()
        }
        lowered = _add_send_recv(
            lowered,
            stage_to_rank=lambda stage: self.stage_index_to_group_rank[stage],
            num_stages=self._num_stages,
        )
        if self._defer_pp_recv:
            lowered = _defer_recv_ops(
                lowered,
                stage_to_rank=lambda stage: self.stage_index_to_group_rank[stage],
            )
        self.pipeline_order_with_comms = lowered

    def _load_csv(
        self,
        filename: str,
        format: Literal["compute_only", "compute_comms"] = "compute_only",
    ) -> None:
        if format == "compute_only":
            actions = super()._load_csv(filename)
            self.pipeline_order = actions
            self._prepare_schedule_with_comms(actions)
            return
        if format != "compute_comms":
            raise NotImplementedError(f"{format=} is not implemented")
        with open(filename, newline="", encoding="utf-8") as stream:
            actions = {
                rank: [_Action.from_str(value) for value in row]
                for rank, row in enumerate(csv.reader(stream))
            }
        self._prepare_schedule_with_comms(actions, format=format)

    def _dump_csv(
        self,
        filename: str,
        format: Literal["compute_only", "compute_comms"] = "compute_comms",
    ) -> None:
        if format == "compute_only":
            actions = self.pipeline_order
        elif format == "compute_comms":
            actions = self.pipeline_order_with_comms
        else:
            raise NotImplementedError(f"{format=} is not implemented")
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for rank in sorted(actions):
                writer.writerow(actions[rank])

    def _simulate(self) -> Any:
        return _simulate_comms_compute(
            self.pipeline_order_with_comms,
            lambda stage: self.stage_index_to_group_rank[stage],
            self._num_stages,
        )

    def _assert_unsharded(self, stage: Any) -> None:
        if not isinstance(stage.submod, FSDPModule):
            return
        stage_index = stage.stage_index
        if stage_index in self.unshard_ops:
            for handle in self.unshard_ops[stage_index]:
                handle.wait()
            del self.unshard_ops[stage_index]
            self.unsharded_stages.add(stage_index)
        if stage_index not in self.unsharded_stages:
            raise AssertionError(f"attempted to compute on sharded stage {stage_index}")

    def _step_microbatches(
        self,
        arg_mbs: list[tuple[Any, ...]] | None = None,
        kwarg_mbs: list[dict[str, Any]] | None = None,
        target_mbs: list[Any] | None = None,
        losses: list[Any] | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> None:
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stages(arg_mbs[0], kwarg_mbs[0], first_target, loss_kwargs)
        stage_index_to_stage = {
            stage.stage_index: stage for stage in self._stages
        }
        if self.pipeline_order_with_comms is None:
            raise AssertionError(
                "must prepare a schedule with communication actions before execution"
            )
        self.fwd_recv_ops.clear()
        self.bwd_recv_ops.clear()
        self.unshard_ops.clear()
        self.unsharded_stages.clear()
        send_ops: list[list[Any]] = []

        def perform_action(action: _Action) -> None:
            computation_type = action.computation_type
            microbatch = action.microbatch_index
            if microbatch is None and computation_type not in {
                _ComputationType.UNSHARD,
                _ComputationType.RESHARD,
                _ComputationType.REDUCE_GRAD,
            }:
                raise AssertionError(f"{action=} is missing a microbatch index")
            stage = stage_index_to_stage[action.stage_index]
            stage_index = action.stage_index
            stage_uses_fsdp = isinstance(stage.submod, FSDPModule)
            next_local = stage_index + 1 in stage_index_to_stage
            previous_local = stage_index - 1 in stage_index_to_stage
            mb_index = -1 if microbatch is None else microbatch

            if computation_type is _ComputationType.SEND_F:
                send_ops.append(_batch_p2p(stage.get_fwd_send_ops(mb_index)))
            elif computation_type is _ComputationType.SEND_B:
                send_ops.append(_batch_p2p(stage.get_bwd_send_ops(mb_index)))
            elif computation_type is _ComputationType.RECV_F:
                key = (stage_index, mb_index)
                if key in self.fwd_recv_ops:
                    raise AssertionError(f"forward receive repeated for {key}")
                self.fwd_recv_ops[key] = _batch_p2p(stage.get_fwd_recv_ops(mb_index))
            elif computation_type is _ComputationType.RECV_B:
                key = (stage_index, mb_index)
                if key in self.bwd_recv_ops:
                    raise AssertionError(f"backward receive repeated for {key}")
                self.bwd_recv_ops[key] = _batch_p2p(stage.get_bwd_recv_ops(mb_index))
            elif computation_type is _ComputationType.UNSHARD:
                if stage_uses_fsdp:
                    if stage_index in self.unsharded_stages or stage_index in self.unshard_ops:
                        raise AssertionError(f"unsharding stage {stage_index} twice")
                    for submodule in stage.submod.modules():
                        if isinstance(submodule, FSDPModule):
                            handle = cast(UnshardHandle, submodule.unshard(async_op=True))
                            self.unshard_ops[stage_index].append(handle)
            elif computation_type is _ComputationType.RESHARD:
                if stage_uses_fsdp:
                    if stage_index not in self.unsharded_stages:
                        raise AssertionError(f"resharding stage {stage_index} without unsharding")
                    if stage_index in self.unshard_ops:
                        raise AssertionError(f"resharding stage {stage_index} before unshard completion")
                    for submodule in stage.submod.modules():
                        if isinstance(submodule, FSDPModule):
                            submodule.reshard()
                    self.unsharded_stages.remove(stage_index)
            elif computation_type is _ComputationType.FORWARD:
                self._assert_unsharded(stage)
                if not stage.is_first and not previous_local:
                    key = (stage_index, mb_index)
                    if key not in self.fwd_recv_ops:
                        raise AssertionError(f"forward action {action} has no receive")
                    _wait_batch_p2p(self.fwd_recv_ops.pop(key))
                output = stage.forward_one_chunk(
                    mb_index,
                    arg_mbs[mb_index],
                    kwarg_mbs[mb_index],
                    save_forward_output=return_outputs,
                )
                self._maybe_compute_loss(stage, output, target_mbs, mb_index, loss_kwargs)
                if next_local:
                    stage_index_to_stage[stage_index + 1].set_local_fwd_input(output, mb_index)
            elif computation_type is _ComputationType.FULL_BACKWARD:
                self._assert_unsharded(stage)
                if not stage.is_last and not next_local:
                    key = (stage_index, mb_index)
                    if key not in self.bwd_recv_ops:
                        raise AssertionError(f"backward action {action} has no receive")
                    _wait_batch_p2p(self.bwd_recv_ops.pop(key))
                self.backward_counter[stage_index] += 1
                last_backward = self.backward_counter[stage_index] == self._n_microbatches
                stage.backward_one_chunk(
                    mb_index,
                    loss=self._maybe_get_loss(stage, mb_index),
                    full_backward=True,
                    last_backward=last_backward,
                )
                if previous_local:
                    stage_index_to_stage[stage_index - 1].set_local_bwd_input(
                        stage.get_local_bwd_output(mb_index), mb_index
                    )
            elif computation_type is _ComputationType.BACKWARD_INPUT:
                self._assert_unsharded(stage)
                if not stage.is_last and not next_local:
                    key = (stage_index, mb_index)
                    if key not in self.bwd_recv_ops:
                        raise AssertionError(f"backward action {action} has no receive")
                    _wait_batch_p2p(self.bwd_recv_ops.pop(key))
                stage.backward_one_chunk(
                    mb_index,
                    loss=self._maybe_get_loss(stage, mb_index),
                    full_backward=False,
                    last_backward=False,
                )
                if previous_local:
                    stage_index_to_stage[stage_index - 1].set_local_bwd_input(
                        stage.get_local_bwd_output(mb_index), mb_index
                    )
            elif computation_type is _ComputationType.BACKWARD_WEIGHT:
                self._assert_unsharded(stage)
                self.backward_counter[stage_index] += 1
                last_backward = self.backward_counter[stage_index] == self._n_microbatches
                stage.backward_weight_one_chunk(mb_index, last_backward=last_backward)
            elif computation_type is _ComputationType.REDUCE_GRAD:
                scale = self._n_microbatches if self._scale_grads else 1
                stage.perform_reduce_grad(scale)
            else:
                raise ValueError(f"unknown or unsupported action {action}")

        self.backward_counter.clear()
        for time_step, action in enumerate(self.pipeline_order_with_comms[self.rank]):
            if action is None:
                continue
            try:
                custom = self._comp_type_to_function_map.get(action.computation_type)
                if custom is not None:
                    custom(
                        action,
                        _PipelineContext(self, arg_mbs, kwarg_mbs, target_mbs, losses),
                    )
                elif action.computation_type is _ComputationType.OVERLAP_F_B:
                    if action.sub_actions is None:
                        raise AssertionError("overlap action must contain sub-actions")
                    for sub_action in action.sub_actions:
                        perform_action(sub_action)
                else:
                    perform_action(action)
            except Exception:
                logger.error(
                    "pipeline runtime failed at step %d on action %s",
                    time_step,
                    action,
                )
                logger.error(
                    "%s",
                    _format_pipeline_order(
                        self.pipeline_order_with_comms,
                        error_step_number=time_step,
                    ),
                )
                raise
        while send_ops:
            _wait_batch_p2p(send_ops.pop())
        if self.unshard_ops:
            raise AssertionError("unused unshard operations")
        self._update_losses(self._stages, losses)


class ScheduleLoopedBFS(_PipelineScheduleRuntime):
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
        self.defer_pp_recv = self._defer_pp_recv
        self.max_active_stages = self._max_active_stages
        self.pipeline_order = {
            rank: self._calculate_single_rank_operations(rank)
            for rank in range(self.pp_group_size)
        }
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action | None]:
        local_stage_count = len(self._stages)
        stage_indices = range(
            rank,
            self.pp_group_size * local_stage_count,
            self.pp_group_size,
        )
        rank_actions: list[_Action | None] = [None for _ in range(rank)]
        for stage_index in stage_indices:
            rank_actions.extend(
                _Action(stage_index, _ComputationType.FORWARD, microbatch)
                for microbatch in range(self._n_microbatches)
            )
        rank_actions.extend(
            [None] * (2 * (self.pp_group_size - 1 - rank))
        )
        for stage_index in reversed(stage_indices):
            rank_actions.extend(
                _Action(
                    stage_index,
                    _ComputationType.FULL_BACKWARD,
                    microbatch,
                )
                for microbatch in reversed(range(self._n_microbatches))
            )
        return rank_actions


class ScheduleInterleaved1F1B(_PipelineScheduleRuntime):
    def __init__(
        self,
        stages: list[Any],
        n_microbatches: int,
        loss_fn: Callable[..., Any] | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: Any = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
        max_active_stages: int = 3,
    ) -> None:
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
            max_active_stages=max_active_stages,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Interleaved 1F1B requires the microbatch count to be a multiple "
                f"of the round count ({self.number_of_rounds}), got {n_microbatches}"
            )
        self.pipeline_order = {
            rank: self._calculate_single_rank_operations(rank)
            for rank in range(self.pp_group_size)
        }
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action | None]:
        warmup_ops = _get_warmup_ops(
            rank,
            self.n_local_stages,
            self.microbatches_per_round,
            self.pp_group_size,
            self._n_microbatches,
            multiply_factor=2,
        )
        microbatch_ops = self.n_local_stages * self._n_microbatches
        forward_backward_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - forward_backward_ops

        def forward_stage_index(step: int) -> int:
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return local_index * self.pp_group_size + rank

        def backward_stage_index(step: int) -> int:
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round)
                % self.n_local_stages
            )
            return local_index * self.pp_group_size + rank

        logger.debug(
            "rank %s: warmup=%s steady=%s cooldown=%s",
            rank,
            warmup_ops,
            forward_backward_ops,
            cooldown_ops,
        )
        return _get_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            forward_backward_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
        )


class ScheduleInterleavedZeroBubble(_PipelineScheduleRuntime):
    def __init__(
        self,
        stages: list[Any],
        n_microbatches: int,
        loss_fn: Callable[..., Any] | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: Any = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
        max_active_stages: int = 3,
    ) -> None:
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
            max_active_stages=max_active_stages,
        )
        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Zero bubble requires the microbatch count to be a multiple of "
                f"the round count ({self.number_of_rounds}), got {n_microbatches}"
            )
        self.pipeline_order = {
            rank: self._calculate_single_rank_operations(rank)
            for rank in range(self.pp_group_size)
        }
        self.pipeline_order = self._add_bubbles_to_actions(
            self.n_local_stages * self.pp_group_size
        )
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action | None]:
        warmup_ops = _get_warmup_ops(
            rank,
            self.n_local_stages,
            self.microbatches_per_round,
            self.pp_group_size,
            self._n_microbatches,
            multiply_factor=1,
        )
        microbatch_ops = self.n_local_stages * self._n_microbatches
        forward_backward_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - forward_backward_ops

        def forward_stage_index(step: int) -> int:
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return local_index * self.pp_group_size + rank

        def backward_stage_index(step: int) -> int:
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round)
                % self.n_local_stages
            )
            return local_index * self.pp_group_size + rank

        return _get_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            forward_backward_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
            rank,
            enable_zero_bubble=True,
        )

    def _add_bubbles_to_actions(
        self, num_stages_global: int
    ) -> dict[int, list[_Action | None]]:
        actions = self.pipeline_order

        def need_bubble(
            stage: int,
            operation: _ComputationType,
            microbatch: int | None,
            seen_ops: set[tuple[int, _ComputationType, int]],
        ) -> bool:
            if operation is _ComputationType.FORWARD:
                return stage != 0 and (
                    stage - 1,
                    operation,
                    microbatch,
                ) not in seen_ops
            if operation is _ComputationType.FULL_BACKWARD:
                if stage == num_stages_global - 1:
                    return (
                        stage,
                        _ComputationType.FORWARD,
                        microbatch,
                    ) not in seen_ops
                return (
                    stage + 1,
                    operation,
                    microbatch,
                ) not in seen_ops
            return False

        seen_ops: set[tuple[int, _ComputationType, int]] = set()
        result: dict[int, list[_Action | None]] = {
            rank: [] for rank in range(self.pp_group_size)
        }
        next_pointer = {rank: 0 for rank in range(self.pp_group_size)}
        bubbles_added = {rank: 0 for rank in range(self.pp_group_size)}
        total_bubbles_added = 0

        while True:
            should_stop = True
            temporary_seen: set[tuple[int, _ComputationType, int]] = set()
            for rank in range(self.pp_group_size):
                timestamp = next_pointer[rank]
                if timestamp >= len(actions[rank]):
                    continue
                should_stop = False
                action = actions[rank][timestamp]
                if action is None:
                    result[rank].append(None)
                    next_pointer[rank] += 1
                    continue
                stage_index, operation, microbatch, _ = action
                if not need_bubble(
                    stage_index,
                    operation,
                    microbatch,
                    seen_ops,
                ):
                    result[rank].append(action)
                    if microbatch is not None:
                        temporary_seen.add((stage_index, operation, microbatch))
                    next_pointer[rank] += 1
                else:
                    result[rank].append(None)
                    bubbles_added[rank] += 1
            seen_ops.update(temporary_seen)
            if should_stop:
                break
        if total_bubbles_added > 0:
            logger.warning(
                "non-zero bubbles added: total=%s by-rank=%s",
                total_bubbles_added,
                bubbles_added,
            )
        return result


class ScheduleZBVZeroBubble(_PipelineScheduleRuntime):
    def __init__(
        self,
        stages: list[Any],
        n_microbatches: int,
        loss_fn: Callable[..., Any] | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: Any = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
        max_active_stages: int = 3,
    ) -> None:
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
            max_active_stages=max_active_stages,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size,
            self._num_stages,
            style="v",
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank
        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly two stages per rank, "
                f"got {self.n_local_stages}"
            )
        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages
        self.pipeline_order = {
            rank: self._calculate_single_rank_operations(rank)
            for rank in range(self.pp_group_size)
        }
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action | None]:
        microbatch_count = max(2 * self.pp_group_size - 1, self._n_microbatches)
        rank_actions: list[_Action | None] = [None for _ in range(rank)]
        forward_chunk0 = 0
        forward_chunk1 = 0
        backward_chunk0 = 0
        backward_chunk1 = 0

        warmup_first = 2 * (self.pp_group_size - rank) - 1
        stage_chunk0 = rank
        stage_chunk1 = self.num_stages - 1 - rank
        for _ in range(warmup_first):
            rank_actions.append(
                _Action(
                    stage_chunk0,
                    _ComputationType.FORWARD,
                    forward_chunk0,
                )
            )
            forward_chunk0 += 1

        warmup_second = rank
        for _ in range(warmup_second):
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.FORWARD, forward_chunk1)
            )
            forward_chunk1 += 1
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.FORWARD, forward_chunk0)
            )
            forward_chunk0 += 1

        warmup_third = self.pp_group_size - rank
        for _ in range(warmup_third):
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.FORWARD, forward_chunk1)
            )
            forward_chunk1 += 1
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.BACKWARD_INPUT, backward_chunk1)
            )
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.BACKWARD_WEIGHT, backward_chunk1)
            )
            backward_chunk1 += 1

        while forward_chunk1 < forward_chunk0 or forward_chunk0 < microbatch_count:
            if forward_chunk0 < microbatch_count:
                rank_actions.append(
                    _Action(stage_chunk0, _ComputationType.FORWARD, forward_chunk0)
                )
                forward_chunk0 += 1
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.BACKWARD_INPUT, backward_chunk0)
            )
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.BACKWARD_WEIGHT, backward_chunk0)
            )
            backward_chunk0 += 1
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.FORWARD, forward_chunk1)
            )
            forward_chunk1 += 1
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.BACKWARD_INPUT, backward_chunk1)
            )
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.BACKWARD_WEIGHT, backward_chunk1)
            )
            backward_chunk1 += 1

        weight_chunk0 = backward_chunk0
        weight_chunk1 = backward_chunk1
        for _ in range(rank):
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.BACKWARD_INPUT, backward_chunk0)
            )
            backward_chunk0 += 1
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.BACKWARD_INPUT, backward_chunk1)
            )
            backward_chunk1 += 1

        for _ in range(self.pp_group_size - rank):
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.BACKWARD_INPUT, backward_chunk0)
            )
            backward_chunk0 += 1
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.BACKWARD_WEIGHT, weight_chunk0)
            )
            weight_chunk0 += 1

        while weight_chunk1 < backward_chunk1:
            rank_actions.append(
                _Action(stage_chunk1, _ComputationType.BACKWARD_WEIGHT, weight_chunk1)
            )
            weight_chunk1 += 1
        while weight_chunk0 < backward_chunk0:
            rank_actions.append(
                _Action(stage_chunk0, _ComputationType.BACKWARD_WEIGHT, weight_chunk0)
            )
            weight_chunk0 += 1

        if not (weight_chunk0 == backward_chunk0 == forward_chunk0):
            raise AssertionError(
                "stage chunk 0 action counts do not match"
            )
        if not (weight_chunk1 == backward_chunk1 == forward_chunk1):
            raise AssertionError(
                "stage chunk 1 action counts do not match"
            )
        return [
            action
            if action is not None
            and action.microbatch_index is not None
            and action.microbatch_index < self._n_microbatches
            else None
            for action in rank_actions
        ]


class ScheduleDualPipeV(_PipelineScheduleRuntime):
    """Run the bidirectional local stage schedule."""

    def __init__(
        self,
        stages: list[Any],
        n_microbatches: int,
        loss_fn: Callable[..., Any] | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: Any = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
        defer_pp_recv: bool = False,
        max_active_stages: int = 3,
    ) -> None:
        _check_torch_compile_compatibility(stages, self.__class__.__name__)
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
            backward_requires_autograd=backward_requires_autograd,
            defer_pp_recv=defer_pp_recv,
            max_active_stages=max_active_stages,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size,
            self._num_stages,
            style="v",
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank
        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )
        if n_microbatches < self._num_stages:
            raise ValueError(
                "DualPipeV requires at least as many microbatches as stages, but got "
                f"{n_microbatches} microbatches and {self._num_stages} stages."
            )
        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages
        self.pipeline_order = {
            rank: self._calculate_single_rank_operations(rank)
            for rank in range(self.pp_group_size)
        }
        self._prepare_schedule_with_comms(self.pipeline_order)

    def _calculate_single_rank_operations(self, rank: int) -> list[_Action | None]:
        actions: list[_Action | None] = []
        counters: dict[tuple[int, _ComputationType], int] = {}
        weight_queue: list[tuple[int, int]] = []

        num_ranks = self.pp_group_size
        num_chunks = self._n_microbatches
        rank_to_stages = generate_rank_to_stage_mapping(
            num_ranks, num_ranks * 2, style="v"
        )
        stage0_index, stage1_index = rank_to_stages[rank]

        def increment_backward_counts(stage_index: int) -> None:
            input_key = (stage_index, _ComputationType.BACKWARD_INPUT)
            weight_key = (stage_index, _ComputationType.BACKWARD_WEIGHT)
            counters[input_key] = counters.get(input_key, 0) + 1
            counters[weight_key] = counters.get(weight_key, 0) + 1

        def add_overlap_f_b(
            forward_stage: int,
            backward_stage: int,
        ) -> None:
            forward_key = (forward_stage, _ComputationType.FORWARD)
            backward_key = (backward_stage, _ComputationType.BACKWARD_INPUT)
            forward_mb = counters.get(forward_key, 0)
            backward_mb = counters.get(backward_key, 0)
            sub_actions = (
                _Action(forward_stage, _ComputationType.FORWARD, forward_mb),
                _Action(backward_stage, _ComputationType.FULL_BACKWARD, backward_mb),
            )
            actions.append(
                _Action(-1, _ComputationType.OVERLAP_F_B, None, sub_actions)
            )
            counters[forward_key] = forward_mb + 1
            increment_backward_counts(backward_stage)

        def add_action(
            stage_index: int,
            computation_type: _ComputationType,
        ) -> None:
            key = (
                (stage_index, computation_type)
                if computation_type != _ComputationType.FULL_BACKWARD
                else (stage_index, _ComputationType.BACKWARD_INPUT)
            )
            mb_index = counters.get(key, 0)
            actions.append(_Action(stage_index, computation_type, mb_index))
            if computation_type == _ComputationType.FULL_BACKWARD:
                increment_backward_counts(stage_index)
            else:
                if computation_type == _ComputationType.BACKWARD_INPUT:
                    weight_queue.append((stage_index, mb_index))
                counters[key] = mb_index + 1

        def add_weight_action_if_pending() -> None:
            if not weight_queue:
                return
            actual_stage_index, weight_mb_index = weight_queue.pop(0)
            actions.append(
                _Action(
                    actual_stage_index,
                    _ComputationType.BACKWARD_WEIGHT,
                    weight_mb_index,
                )
            )
            weight_key = (actual_stage_index, _ComputationType.BACKWARD_WEIGHT)
            counters[weight_key] = counters.get(weight_key, 0) + 1

        step_1 = (num_ranks - rank - 1) * 2
        for _ in range(step_1):
            add_action(stage0_index, _ComputationType.FORWARD)

        step_2 = rank + 1
        for _ in range(step_2):
            add_action(stage0_index, _ComputationType.FORWARD)
            add_action(stage1_index, _ComputationType.FORWARD)

        step_3 = num_ranks - rank - 1
        for _ in range(step_3):
            add_action(stage1_index, _ComputationType.BACKWARD_INPUT)
            add_weight_action_if_pending()
            add_action(stage1_index, _ComputationType.FORWARD)

        step_4 = num_chunks - num_ranks * 2 + rank + 1
        for index in range(step_4):
            if index == 0 and rank == num_ranks - 1:
                add_action(stage0_index, _ComputationType.FORWARD)
                add_action(stage1_index, _ComputationType.FULL_BACKWARD)
            else:
                add_overlap_f_b(stage0_index, stage1_index)
            add_overlap_f_b(stage1_index, stage0_index)

        step_5 = num_ranks - rank - 1
        for _ in range(step_5):
            add_action(stage1_index, _ComputationType.FULL_BACKWARD)
            add_overlap_f_b(stage1_index, stage0_index)

        step_6 = rank + 1
        enable_zb = False
        for index in range(step_6):
            if index == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            comp_type = (
                _ComputationType.BACKWARD_INPUT
                if enable_zb
                else _ComputationType.FULL_BACKWARD
            )
            add_action(stage1_index, comp_type)
            if index == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            comp_type = (
                _ComputationType.BACKWARD_INPUT
                if enable_zb
                else _ComputationType.FULL_BACKWARD
            )
            add_action(stage0_index, comp_type)

        step_7 = num_ranks - rank - 1
        for _ in range(step_7):
            add_weight_action_if_pending()
            comp_type = (
                _ComputationType.BACKWARD_INPUT
                if enable_zb
                else _ComputationType.FULL_BACKWARD
            )
            add_action(stage0_index, comp_type)

        step_8 = rank + 1
        for _ in range(step_8):
            add_weight_action_if_pending()
        return actions


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
    active: set[int] = set()
    result: list[_Action] = []

    def next_stage_indices(
        count: int, actions: list[_Action | None]
    ) -> list[int]:
        seen: set[int] = set()
        stages: list[int] = []
        for action in actions:
            if action is None:
                continue
            leaves = action.sub_actions or (action,)
            for leaf in leaves:
                if leaf.stage_index not in seen:
                    seen.add(leaf.stage_index)
                    stages.append(leaf.stage_index)
            if len(stages) >= count:
                break
        return stages

    actions = list(compute_actions)
    for index, action in enumerate(actions):
        if action is None:
            continue
        upcoming = next_stage_indices(max_active_stages, actions[index:])
        for stage_index in [stage for stage in active if stage not in upcoming]:
            active.remove(stage_index)
            result.append(_Action(stage_index, _ComputationType.RESHARD))
        for stage_index in upcoming:
            if stage_index not in active:
                active.add(stage_index)
                result.append(_Action(stage_index, _ComputationType.UNSHARD))
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


def _get_warmup_ops(
    rank: int,
    n_local_stages: int,
    microbatches_per_round: int,
    pp_group_size: int,
    n_microbatches: int,
    multiply_factor: int = 2,
) -> int:
    warmups_last_stage = (n_local_stages - 1) * microbatches_per_round
    warmup_ops = warmups_last_stage + multiply_factor * (pp_group_size - 1 - rank)
    return min(warmup_ops, n_microbatches * n_local_stages)


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
        if kind is _ComputationType.BACKWARD_WEIGHT:
            return True
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
    events: list[dict[str, Any]] = []
    for rank in sorted(schedule):
        for timestep, action in enumerate(schedule[rank]):
            if action is None:
                continue
            events.append(
                {
                    "name": str(action),
                    "cat": (
                        "computation"
                        if action.computation_type
                        in {
                            _ComputationType.FORWARD,
                            _ComputationType.FULL_BACKWARD,
                            _ComputationType.BACKWARD_WEIGHT,
                        }
                        else "communication"
                    ),
                    "ph": "X",
                    "pid": rank,
                    "tid": rank,
                    "ts": timestep,
                    "dur": 1,
                }
            )
    import json

    with open(filename, "w", encoding="utf-8") as stream:
        json.dump({"traceEvents": events}, stream)


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
