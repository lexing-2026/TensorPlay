from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .planner import SavePlan

__all__ = ["dedup_tensors"]


def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


logger = init_logger()


def dedup_tensors(all_plans: list[SavePlan]) -> list[SavePlan]:
    plans = list(all_plans)
    key_to_plans: dict[Any, list[int]] = {}
    for plan_index, plan in enumerate(plans):
        for item in plan.items:
            key_to_plans.setdefault(item.index, []).append(plan_index)
    remove: dict[int, set[Any]] = {}
    for key, plan_indices in key_to_plans.items():
        for plan_index in plan_indices[1:]:
            remove.setdefault(plan_index, set()).add(key)
    for plan_index, keys in remove.items():
        logger.info("duplicate checkpoint keys removed: %s", keys)
        plans[plan_index] = replace(
            plans[plan_index],
            items=[item for item in plans[plan_index].items if item.index not in keys],
        )
    return plans
