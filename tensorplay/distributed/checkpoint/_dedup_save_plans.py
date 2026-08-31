from __future__ import annotations

from dataclasses import replace
from typing import Any

from .planner import SavePlan

__all__ = ["dedup_save_plans"]


def dedup_save_plans(all_plans: list[SavePlan], dedup_save_to_lowest_rank: bool = False) -> list[SavePlan]:
    seen: set[Any] = set()
    result: list[SavePlan] = []
    for plan in all_plans:
        items = []
        for item in plan.items:
            key = item.index.fqn
            if key in seen and dedup_save_to_lowest_rank:
                continue
            seen.add(key)
            items.append(item)
        result.append(replace(plan, items=items))
    return result
