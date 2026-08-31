from __future__ import annotations

from dataclasses import replace

from .planner import SavePlan

__all__ = ["dedup_tensors"]


def init_logger():
    import logging
    return logging.getLogger(__name__)


def dedup_tensors(all_plans: list[SavePlan]) -> list[SavePlan]:
    seen: set[str] = set()
    result: list[SavePlan] = []
    for plan in all_plans:
        items = [item for item in plan.items if not (item.index.fqn in seen or seen.add(item.index.fqn))]
        result.append(replace(plan, items=items))
    return result
