from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .planner import SavePlan

__all__ = ["dedup_save_plans"]


def dedup_save_plans(
    all_plans: list[SavePlan], save_to_lowest_rank: bool = False
) -> list[SavePlan]:
    write_item_to_plan_indices: dict[object, set[int]] = defaultdict(set)
    write_item_by_index: dict[object, object] = {}
    plan_to_item_indices: list[set[object]] = [
        {item.index for item in plan.items} for plan in all_plans
    ]

    for plan_index, plan in enumerate(all_plans):
        for write_item in plan.items:
            write_item_to_plan_indices[write_item.index].add(plan_index)
            write_item_by_index[write_item.index] = write_item

    plan_to_size = [0] * len(all_plans)
    for write_item_index, plan_indices in write_item_to_plan_indices.items():
        if save_to_lowest_rank:
            selected_plan_index = min(plan_indices)
        else:
            selected_plan_index = min(
                plan_indices, key=lambda plan_index: plan_to_size[plan_index]
            )
        write_item = write_item_by_index[write_item_index]
        plan_to_size[selected_plan_index] += write_item.tensor_storage_size() or 1
        for plan_index in plan_indices - {selected_plan_index}:
            plan_to_item_indices[plan_index].discard(write_item_index)

    if len(all_plans) != len(plan_to_item_indices):
        raise AssertionError("len(all_plans) != len(plan_to_item_indices)")
    return [
        replace(
            plan,
            items=[item for item in plan.items if item.index in item_indexes],
        )
        for plan, item_indexes in zip(all_plans, plan_to_item_indices)
    ]
