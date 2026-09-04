"""Schedule inspection and visualization helpers."""

from __future__ import annotations

import collections
from typing import Any, NamedTuple
from unittest import mock

from .schedules import (
    _Action,
    _ComputationType,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    get_schedule_class,
)
from .stage import PipelineStage


class OpKey(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: int


def get_schedule_ops(
    schedule: str | type[_PipelineSchedule] | _PipelineSchedule,
    pp_degree: int,
    num_microbatches: int,
    num_stages_per_rank: int | None = None,
    add_spacing: bool = False,
    with_comms: bool = False,
    defer_pp_recv: bool = False,
) -> list[list[_Action | None]]:
    if add_spacing and with_comms:
        raise ValueError("cannot add spacing and communication actions together")

    if isinstance(schedule, str):
        schedule_class = get_schedule_class(schedule)
    elif isinstance(schedule, _PipelineSchedule):
        schedule_class = type(schedule)
    elif isinstance(schedule, type) and issubclass(schedule, _PipelineSchedule):
        schedule_class = schedule
    else:
        raise ValueError(f"invalid schedule: {schedule}")

    def make_mock_stage(stage_index: int) -> Any:
        stage = mock.create_autospec(PipelineStage, instance=True)
        stage.stage_index = stage_index
        stage.group_rank = 0
        stage.group_size = pp_degree
        stage.num_stages = pp_degree
        stage.submod = None
        return stage

    if issubclass(schedule_class, PipelineScheduleSingle):
        if num_stages_per_rank is None:
            num_stages_per_rank = 1
        if num_stages_per_rank != 1:
            raise AssertionError(
                f"expected num_stages_per_rank to be 1, got {num_stages_per_rank}"
            )
        stages = make_mock_stage(0)
        stages.num_stages = num_stages_per_rank * pp_degree
    elif issubclass(schedule_class, PipelineScheduleMulti):
        if num_stages_per_rank is None:
            num_stages_per_rank = 2
        if num_stages_per_rank < 2:
            raise AssertionError(
                f"expected num_stages_per_rank >= 2, got {num_stages_per_rank}"
            )
        stages = [make_mock_stage(index) for index in range(num_stages_per_rank)]
        for stage in stages:
            stage.num_stages = num_stages_per_rank * pp_degree
    else:
        raise ValueError(f"invalid schedule: {schedule_class}")

    if isinstance(schedule, _PipelineSchedule):
        schedule_instance = schedule
    else:
        schedule_instance = schedule_class(stages, num_microbatches)
    if schedule_instance.pipeline_order is None:
        raise AssertionError("expected pipeline_order to be available")

    if with_comms:
        runtime_stages = stages if isinstance(stages, list) else [stages]
        runtime = _PipelineScheduleRuntime(
            runtime_stages,
            num_microbatches,
            defer_pp_recv=defer_pp_recv,
        )
        runtime._prepare_schedule_with_comms(schedule_instance.pipeline_order)
        all_actions = [
            list(runtime.pipeline_order_with_comms.get(rank, ()))
            for rank in range(pp_degree)
        ]
    else:
        all_actions = [
            schedule_instance.pipeline_order[rank] for rank in range(pp_degree)
        ]

    if add_spacing:
        all_actions = [
            [action for action in rank_actions if action is not None]
            for rank_actions in all_actions
        ]
        all_actions = add_schedule_op_spacing(all_actions)
    return all_actions


class _ComputationTypeVisual:
    def __init__(self, color: str, text: str = "", width: int = 1) -> None:
        self.color = color
        self.width = width
        self.text = text


action_type_to_color_mapping = {
    _ComputationType.FORWARD: _ComputationTypeVisual("blue", "Forward"),
    _ComputationType.BACKWARD_INPUT: _ComputationTypeVisual(
        "teal", "Backward Input"
    ),
    _ComputationType.BACKWARD_WEIGHT: _ComputationTypeVisual(
        "green", "Backward Weight"
    ),
    _ComputationType.FULL_BACKWARD: _ComputationTypeVisual(
        "orange", "Full Backward", 2
    ),
    _ComputationType.OVERLAP_F_B: _ComputationTypeVisual(
        "purple", "Overlap F+B", 3
    ),
    _ComputationType.REDUCE_GRAD: _ComputationTypeVisual(
        "gray", "Reduce Grad"
    ),
}


def add_schedule_op_spacing(
    schedule: list[list[_Action | None]],
) -> list[list[_Action | None]]:
    if not schedule:
        return schedule

    actions = [
        action
        for rank_actions in schedule
        for action in rank_actions
        if action is not None
    ]
    if not actions:
        return [[] for _ in schedule]
    num_stages = max(action.stage_index for action in actions) + 1
    num_ranks = len(schedule)
    spaced_schedule: list[list[_Action | None]] = [[] for _ in range(num_ranks)]
    rank_ops = [collections.deque(ops) for ops in schedule]
    scheduled_ops: dict[OpKey, int] = {}

    def is_dependency_ready(dependency_key: OpKey, timestep: int) -> bool:
        return dependency_key in scheduled_ops and timestep >= scheduled_ops[dependency_key]

    def get_dependencies(action: _Action) -> list[OpKey]:
        stage_idx = action.stage_index
        comp_type = action.computation_type
        mb_idx = action.microbatch_index
        if comp_type == _ComputationType.REDUCE_GRAD:
            return []
        if mb_idx is None:
            raise AssertionError(f"action {action} has no microbatch index")
        if stage_idx == 0 and comp_type == _ComputationType.FORWARD:
            return []
        if stage_idx == num_stages - 1 and comp_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
        ):
            return [OpKey(stage_idx - 1, _ComputationType.FORWARD, mb_idx)]
        if comp_type == _ComputationType.FORWARD:
            return [OpKey(stage_idx - 1, _ComputationType.FORWARD, mb_idx)]
        if comp_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
        ):
            return [
                OpKey(stage_idx + 1, _ComputationType.FULL_BACKWARD, mb_idx),
                OpKey(stage_idx + 1, _ComputationType.BACKWARD_INPUT, mb_idx),
            ]
        if comp_type == _ComputationType.BACKWARD_WEIGHT:
            return [OpKey(stage_idx, _ComputationType.BACKWARD_INPUT, mb_idx)]
        raise RuntimeError(f"unknown computation type: {comp_type}")

    def is_action_ready(action: _Action, timestep: int) -> bool:
        if action.computation_type == _ComputationType.REDUCE_GRAD:
            return True
        if action.computation_type == _ComputationType.OVERLAP_F_B:
            if action.sub_actions is None:
                raise AssertionError(f"overlap action {action} has no sub-actions")
            return all(
                is_action_ready(sub_action, timestep)
                for sub_action in action.sub_actions
            )
        dependencies = get_dependencies(action)
        if action.computation_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.BACKWARD_WEIGHT,
        ):
            return any(is_dependency_ready(dep, timestep) for dep in dependencies)
        if action.computation_type == _ComputationType.FORWARD:
            return all(is_dependency_ready(dep, timestep) for dep in dependencies)
        raise RuntimeError(f"unknown computation type: {action.computation_type}")

    def schedule_action(action: _Action, rank: int, timestep: int) -> int:
        spaced_schedule[rank].append(action)
        visual = action_type_to_color_mapping[action.computation_type]
        completion_time = timestep + visual.width
        if action.computation_type == _ComputationType.OVERLAP_F_B:
            if action.sub_actions is None:
                raise AssertionError(f"overlap action {action} has no sub-actions")
            cumulative_time = 0
            for sub_action in action.sub_actions:
                if sub_action.microbatch_index is None:
                    raise AssertionError(
                        f"sub-action {sub_action} has no microbatch index"
                    )
                cumulative_time += action_type_to_color_mapping[
                    sub_action.computation_type
                ].width
                scheduled_ops[
                    OpKey(
                        sub_action.stage_index,
                        sub_action.computation_type,
                        sub_action.microbatch_index,
                    )
                ] = timestep + cumulative_time
        else:
            if action.microbatch_index is None:
                if action.computation_type == _ComputationType.REDUCE_GRAD:
                    return completion_time
                raise AssertionError(f"action {action} has no microbatch index")
            scheduled_ops[
                OpKey(
                    action.stage_index,
                    action.computation_type,
                    action.microbatch_index,
                )
            ] = completion_time
        return completion_time

    current_timestep = 0
    timesteps_without_progress = 0
    rank_completion_times = dict.fromkeys(range(num_ranks), 0)
    while rank_ops:
        for rank, op_queue in enumerate(rank_ops):
            if not op_queue:
                continue
            action = op_queue[0]
            if action is None:
                spaced_schedule[rank].append(None)
                op_queue.popleft()
                timesteps_without_progress = 0
            elif (
                current_timestep >= rank_completion_times[rank]
                and is_action_ready(action, current_timestep)
            ):
                rank_completion_times[rank] = schedule_action(
                    action, rank, current_timestep
                )
                op_queue.popleft()
                timesteps_without_progress = 0

        for rank in range(num_ranks):
            if current_timestep >= rank_completion_times[rank]:
                spaced_schedule[rank].append(None)

        rank_ops = [op_queue for op_queue in rank_ops if op_queue]
        current_timestep += 1
        timesteps_without_progress += 1
        if timesteps_without_progress > max(
            visual.width for visual in action_type_to_color_mapping.values()
        ):
            raise RuntimeError("schedule made no progress")
    return spaced_schedule


def visualize_schedule(
    schedule: list[list[_Action | None]],
    filename: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.family"] = "DejaVu Sans"
    num_ranks = len(schedule)
    max_actions = max(len(rank) for rank in schedule)
    fig, ax = plt.subplots(figsize=(max_actions + 2, num_ranks + 2))
    max_draw_position = -1
    font_size = min(max_actions, num_ranks) + 4
    used_computation = set()
    for rank_idx, actions in enumerate(schedule):
        draw_position = 0
        for action in actions:
            if action is not None:
                comp_type_color = action_type_to_color_mapping.get(
                    action.computation_type, _ComputationTypeVisual("black")
                )
                used_computation.add(action.computation_type)
                if action.sub_actions is not None:
                    linewidth = 2
                    text_weight = "normal"
                else:
                    linewidth = 1
                    text_weight = "normal"
                rect = Rectangle(
                    (draw_position, num_ranks - rank_idx - 1),
                    comp_type_color.width,
                    1,
                    facecolor=comp_type_color.color,
                    edgecolor="black",
                    linewidth=linewidth,
                )
                ax.add_patch(rect)
                ax.text(
                    draw_position + comp_type_color.width / 2,
                    num_ranks - rank_idx - 1 + 0.5,
                    str(action),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="white",
                    weight=text_weight,
                )
                draw_position += comp_type_color.width
            else:
                draw_position += 1
            max_draw_position = max(max_draw_position, draw_position)
    ax.set_xlim(-0.5, max_draw_position + 1)
    ax.set_ylim(-0.5, num_ranks + 0.5)
    ax.set_yticks([num_ranks - rank_idx - 0.5 for rank_idx in range(num_ranks)])
    ax.set_yticklabels(
        [f"Rank {index}" for index in range(num_ranks)], fontsize=font_size
    )
    ax.set_xticklabels([])
    ax.grid(False)
    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=action_type_to_color_mapping[comp_type].color,
            edgecolor="black",
            label=action_type_to_color_mapping[comp_type].text,
        )
        for comp_type in used_computation
        if comp_type in action_type_to_color_mapping
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=font_size)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
