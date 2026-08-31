from __future__ import annotations

from enum import IntEnum
from typing import Any

from .ilp_utils import Graph, is_submodule
from .sac_estimator import SACStats

__all__ = ["sac_milp", "SACDecision", "get_optimal_checkpointing_policy_per_module"]


def sac_milp(
    graph: Graph,
    memory_budget: float,
    world_size: int = 1,
    ac_units: list[str] | None = None,
    fsdp_units: list[str] | None = None,
) -> tuple[dict[str, float], float, int]:
    """Select non-overlapping activation regions under a byte budget."""
    if memory_budget < 0:
        raise ValueError("memory_budget must be non-negative")
    allowed = set(ac_units) if ac_units else None
    excluded = set(fsdp_units or ())
    current = 0
    baseline_runtime = 0.0
    candidates: list[tuple[float, Any]] = []
    for node in graph.nodes:
        current += int(node.get("param_per_module", 0)) // max(1, world_size)
        baseline_runtime += float(node.get("fw_runtime_per_module", 0.0))
        name = node["fqn"]
        if node.get("is_leaf") or allowed is not None and name not in allowed or any(is_submodule(name, item) or is_submodule(item, name) for item in excluded):
            continue
        saved = max(0, int(node.get("sac_memory", node.get("act_fw_per_module", 0))))
        runtime = float(node.get("sac_runtime", node.get("fw_runtime_per_module", 0.0)))
        score = saved / runtime if runtime else float("inf")
        candidates.append((score, node))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: dict[str, float] = {}
    peak = current
    recompute = 0.0
    for _, node in candidates:
        saved = max(0, int(node.get("sac_memory", node.get("act_fw_per_module", 0))))
        if peak - saved > memory_budget:
            continue
        if any(is_submodule(node["fqn"], name) or is_submodule(name, node["fqn"]) for name in selected):
            continue
        selected[node["fqn"]] = 1.0
        peak -= saved
        recompute += float(node.get("sac_runtime", 0.0))
    return selected, recompute, peak


class SACDecision(IntEnum):
    RECOMPUTE = 0
    SAVE = 1


def get_optimal_checkpointing_policy_per_module(sac_stats: SACStats, memory_budget: float) -> list[int]:
    if not 0 <= memory_budget <= 1:
        raise ValueError(f"memory_budget must be between 0 and 1, got {memory_budget}")
    total = sum(sac_stats.memory)
    target = total * memory_budget
    policy = [int(index in sac_stats.view_like_ops or index in sac_stats.saved_autograd_ops) for index in range(len(sac_stats.func_names))]
    used = sum(memory for keep, memory in zip(policy, sac_stats.memory) if keep)
    choices = sorted(range(len(policy)), key=lambda index: (sac_stats.runtimes[index] / max(sac_stats.memory[index], 1)), reverse=True)
    for index in choices:
        if index in sac_stats.rand_ops and not sac_stats.force_store_random:
            continue
        if not policy[index] and used + sac_stats.memory[index] <= target:
            policy[index] = int(SACDecision.SAVE)
            used += sac_stats.memory[index]
    for index in sac_stats.view_like_ops:
        policy[index] = int(SACDecision.SAVE)
    return policy
