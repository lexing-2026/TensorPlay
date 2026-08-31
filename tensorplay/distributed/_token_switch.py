from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import tensorplay as tp
from tensorplay.autograd.function import Function

__all__ = ["Routing", "TokenSwitch", "TokenSwitchNCCL"]


def _find_pkg_dir(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    locations = spec.submodule_search_locations if spec is not None else None
    return next(iter(locations), None) if locations else None


def _prepare_nccl4py() -> None:
    raise ImportError("the optional token routing extension is not installed")


def _import_nccl_ep() -> Any:
    raise ImportError("the optional token routing extension is not installed")


@dataclass(frozen=True)
class Routing:
    handle: object
    topk_idx: Any


def _as_ints(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [int(item) for item in value]
    return [int(value)]


class _DispatchAutograd(Function):
    @staticmethod
    def forward(ctx: Any, switch: "TokenSwitch", routing: Routing, tokens: Any, weights: Any, max_recv_tokens: int):
        rows = int(tokens.shape[0])
        hidden = int(tokens.shape[1])
        top_k = int(weights.shape[1])
        out_tokens = tp.zeros((max_recv_tokens, hidden), dtype=tokens.dtype, device=tokens.device)
        out_weights = tp.zeros((max_recv_tokens, top_k), dtype=weights.dtype, device=weights.device)
        out_indices = tp.zeros((max_recv_tokens, top_k), dtype=routing.topk_idx.dtype, device=routing.topk_idx.device)
        switch._dispatch(routing, tokens, weights, out_tokens, out_weights, out_indices)
        ctx.switch = switch
        ctx.routing = routing
        ctx.input_shape = (rows, hidden)
        return out_tokens, out_weights, out_indices

    @staticmethod
    def backward(ctx: Any, grad_tokens: Any, grad_weights: Any, grad_indices: Any):
        del grad_weights, grad_indices
        result = grad_tokens.new_zeros(ctx.input_shape)
        ctx.switch._combine(ctx.routing, grad_tokens, result)
        return None, None, result, None, None


class _CombineAutograd(Function):
    @staticmethod
    def forward(ctx: Any, switch: "TokenSwitch", routing: Routing, expert_tokens: Any):
        shape = (int(routing.topk_idx.shape[0]), int(expert_tokens.shape[1]))
        output = expert_tokens.new_zeros(shape)
        switch._combine(routing, expert_tokens, output)
        ctx.switch = switch
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Any):
        result = grad_output.new_zeros(ctx.expert_shape)
        ctx.switch._dispatch(ctx.routing, grad_output, grad_output, result, result.new_zeros((result.shape[0], ctx.routing.topk_idx.shape[1])), ctx.routing.topk_idx.new_zeros((result.shape[0], ctx.routing.topk_idx.shape[1])))
        return None, None, result


class TokenSwitch:
    def __init__(self, process_group: Any = None, num_experts: int = 1) -> None:
        self.process_group = process_group
        self.num_experts = int(num_experts)
        self._max_recv_tokens_per_rank = 0

    def create_routing(self, topk_idx: Any, per_expert_token_counts: Any = None) -> Routing:
        del per_expert_token_counts
        if len(topk_idx.shape) != 2:
            raise ValueError("topk_idx must be a two-dimensional tensor")
        values = topk_idx.tolist() if hasattr(topk_idx, "tolist") else topk_idx
        assignments: list[tuple[int, int, int]] = []
        for token, experts in enumerate(values):
            for choice, expert in enumerate(_as_ints(experts)):
                if expert < 0 or expert >= self.num_experts:
                    raise ValueError("expert index is outside the configured range")
                assignments.append((expert, token, choice))
        return Routing(assignments, topk_idx)

    def _dispatch(self, routing: Routing, tokens: Any, topk_weights: Any, out_tokens: Any, out_topk_weights: Any, out_topk_idx: Any) -> None:
        assignments = list(routing.handle)
        if len(assignments) > int(out_tokens.shape[0]):
            raise ValueError("dispatch output is too small for the routing plan")
        for output_row, (_, token, choice) in enumerate(assignments):
            out_tokens[output_row].copy_(tokens[token])
            out_topk_weights[output_row].copy_(topk_weights[token])
            out_topk_idx[output_row].copy_(routing.topk_idx[token])
        self._max_recv_tokens_per_rank = max(self._max_recv_tokens_per_rank, len(assignments))

    def _combine(self, routing: Routing, expert_tokens: Any, out_tokens: Any) -> None:
        assignments = list(routing.handle)
        if len(assignments) > int(expert_tokens.shape[0]):
            raise ValueError("combine input is too small for the routing plan")
        for source_row, (_, token, _) in enumerate(assignments):
            out_tokens[token].add_(expert_tokens[source_row])

    def dispatch(self, routing: Routing, tokens: Any, topk_weights: Any, max_recv_tokens: int | None = None, *, out: tuple[Any, Any, Any] | None = None):
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out is not provided")
        return _DispatchAutograd.apply(self, routing, tokens, topk_weights, max_recv_tokens)

    def combine(self, routing: Routing, expert_tokens: Any, *, out: Any = None):
        if out is not None:
            self._combine(routing, expert_tokens, out)
            return out
        return _CombineAutograd.apply(self, routing, expert_tokens)


class TokenSwitchNCCL(TokenSwitch):
    def __init__(self, process_group: Any, num_experts: int, max_dispatch_tokens_per_rank: int, max_recv_tokens_per_rank: int, max_token_bytes: int) -> None:
        del max_dispatch_tokens_per_rank, max_token_bytes
        super().__init__(process_group, num_experts)
        self._max_recv_tokens_per_rank = int(max_recv_tokens_per_rank)
