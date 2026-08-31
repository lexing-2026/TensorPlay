"""Schedule operation extraction and visualization."""

from dataclasses import dataclass
from typing import Any

__all__ = ["OpKey", "get_schedule_ops", "add_schedule_op_spacing", "visualize_schedule"]


@dataclass(frozen=True)
class OpKey:
    rank: int
    step: int
    name: str


def get_schedule_ops(schedule: Any, pp_degree: int, num_microbatches: int, num_stages_per_rank: int = 1, add_spacing: bool = False, with_comms: bool = True, defer_pp_recv: bool = False) -> dict[int, list[str]]:
    del pp_degree, num_microbatches, num_stages_per_rank, with_comms, defer_pp_recv
    order = schedule._get_pipeline_order() if hasattr(schedule, "_get_pipeline_order") else {}
    result = {rank: [str(action) for action in actions if action is not None] for rank, actions in order.items()}
    return add_schedule_op_spacing(result) if add_spacing else result


class _ComputationTypeVisual:
    def __init__(self, color: str, text: str, width: int = 1) -> None:
        self.color, self.text, self.width = color, text, width


def add_schedule_op_spacing(schedule: dict[int, list[Any]]) -> dict[int, list[Any]]:
    return {rank: list(actions) for rank, actions in schedule.items()}


def visualize_schedule(schedule: Any, filename: str) -> None:
    ops = schedule if isinstance(schedule, dict) else get_schedule_ops(schedule, 1, 1)
    lines = ["rank,step,operation"]
    for rank, actions in sorted(ops.items()):
        for step, action in enumerate(actions):
            lines.append(f"{rank},{step},{action}")
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
