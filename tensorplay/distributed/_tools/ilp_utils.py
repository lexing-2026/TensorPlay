from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, TypedDict, cast

__all__ = [
    "ModOrder",
    "ModRuntime",
    "ModStats",
    "ModuleInfo",
    "aggregate_stats",
    "Node",
    "Graph",
    "parse_module_info",
    "is_self_or_submodule",
    "is_submodule",
    "display_bytes",
    "get_peak_memory_runtime_baseline",
]


class ModOrder(TypedDict):
    fw_pre_order: list[str]
    bw_pre_order: list[str]
    fw_post_order: list[str]
    bw_post_order: list[str]


class ModRuntime(TypedDict):
    fw: float
    bw: float


class ModStats(TypedDict, total=False):
    fqn: str
    param_per_module: int
    grad_per_module: int
    grad_total: int
    act_fw_per_module: int
    act_bw_per_module: int
    act_grad_per_module: int
    act_total: int
    input_per_module: int
    output_per_module: int
    fw_runtime_per_module: float
    bw_runtime_per_module: float
    is_leaf: bool
    sac_runtime: float
    sac_memory: int
    n_segments: int
    slopes: list[float]
    intercepts: list[float]
    breakpoints: list[float]
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    mod_order: ModOrder
    mod_stats: list[ModStats]


def aggregate_stats(model: Any, mem_tracker: Any, runtime_estimator: Any, sac_estimator: Any, dev: Any) -> ModuleInfo:
    del dev
    memory_tracking = getattr(mem_tracker, "memory_tracking", {})
    runtimes = getattr(runtime_estimator, "mod_runtimes", {})
    order: ModOrder = {
        "fw_pre_order": list(getattr(runtime_estimator, "mod_fw_pre_order", [])),
        "bw_pre_order": list(getattr(runtime_estimator, "mod_bw_pre_order", [])),
        "fw_post_order": list(getattr(runtime_estimator, "mod_fw_post_order", [])),
        "bw_post_order": list(getattr(runtime_estimator, "mod_bw_post_order", [])),
    }
    tradeoffs = getattr(sac_estimator, "sac_mod_tradeoff_stats", {})
    result: ModuleInfo = {"mod_order": order, "mod_stats": []}
    for module in model.modules():
        stat = memory_tracking.get(module)
        if stat is None:
            continue
        fqn = stat.mod_fqn
        runtime = runtimes.get(fqn, {})
        tradeoff = tradeoffs.get(fqn)
        result["mod_stats"].append(
            {
                "fqn": fqn,
                "param_per_module": int(getattr(stat, "parameter_mem", 0)),
                "grad_per_module": int(getattr(stat, "parameter_mem", 0)),
                "grad_total": int(getattr(stat, "parameter_mem", 0)),
                "act_fw_per_module": int(getattr(stat, "output_mem", 0)),
                "act_bw_per_module": int(getattr(stat, "output_mem", 0)),
                "act_grad_per_module": 0,
                "act_total": int(getattr(stat, "output_mem", 0)),
                "input_per_module": int(getattr(stat, "input_mem", 0)),
                "output_per_module": int(getattr(stat, "output_mem", 0)),
                "fw_runtime_per_module": float(runtime.get("fw", 0.0)),
                "bw_runtime_per_module": float(runtime.get("bw", 0.0)),
                "is_leaf": len(list(module.children())) == 0,
                "sac_runtime": float(getattr(tradeoff, "sac_runtime", 0.0)),
                "sac_memory": int(getattr(tradeoff, "sac_memory", 0)),
                "n_segments": int(getattr(tradeoff, "n_segments", 0)),
                "slopes": list(getattr(tradeoff, "slopes", [])),
                "intercepts": list(getattr(tradeoff, "intercepts", [])),
                "breakpoints": list(getattr(tradeoff, "fit_breaks", [])),
                "tradeoff_curve": copy.deepcopy(getattr(tradeoff, "tradeoff_curve", OrderedDict())),
            }
        )
    return result


class Node(ModStats):
    index: int
    pos_fw_post_order: int


class Graph:
    def __init__(self, n: int) -> None:
        self.nodes: list[Node] = []
        self.name2node: dict[str, Node] = {}
        self.ad_matrix = [[0 for _ in range(n)] for _ in range(n)]
        self.fw_post_order: list[str] = []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node


def parse_module_info(module_info: ModuleInfo) -> Graph:
    stats = list(module_info["mod_stats"])
    pre = module_info["mod_order"]["fw_pre_order"]
    post = module_info["mod_order"]["fw_post_order"]
    if len(stats) != len(pre):
        raise AssertionError("module statistics and order have different lengths")
    stats.sort(key=lambda item: pre.index(item["fqn"]))
    graph = Graph(len(stats))
    graph.fw_post_order = list(post)
    for index, item in enumerate(stats):
        node = cast(Node, item)
        node["index"] = index
        node["pos_fw_post_order"] = post.index(node["fqn"]) if node["fqn"] in post else index
        graph.add_node(node)
    for i, ancestor in enumerate(graph.nodes):
        for j, descendant in enumerate(graph.nodes):
            graph.ad_matrix[i][j] = int(is_self_or_submodule(descendant["fqn"], ancestor["fqn"]))
    return graph


def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
    return name_descendant == name_ancestor or name_descendant.startswith(name_ancestor + ".")


def is_submodule(name_descendant: str, name_ancestor: str) -> bool:
    return name_descendant != name_ancestor and is_self_or_submodule(name_descendant, name_ancestor)


def display_bytes(b: int, unit: str = "MiB") -> str:
    divisors = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
    if unit not in divisors:
        raise ValueError(f"unsupported memory unit {unit!r}")
    return f"{b / divisors[unit]:.2f} {unit}"


def get_peak_memory_runtime_baseline(graph: Graph) -> tuple[int, float]:
    peak = 0
    runtime = 0.0
    for node in graph.nodes:
        peak = max(peak, int(node.get("act_total", 0)) + int(node.get("param_per_module", 0)))
        runtime += float(node.get("fw_runtime_per_module", 0.0))
    return peak, runtime
