from __future__ import annotations

import copy
from typing import Any

import tensorplay as tp


class StateDictStager:
    def __init__(self, pin_memory: bool = False, share_memory: bool = False) -> None:
        self.pin_memory = pin_memory
        self.share_memory = share_memory
        self._keep_alive_values: list[Any] = []

    def _offload_tensor(self, x: tp.Tensor, memo: dict[int, Any], non_blocking: bool = False) -> tp.Tensor:
        del non_blocking
        value = x.detach().clone()
        memo[id(x)] = value
        return value

    def deepcopy_with_tensor_offload(self, x: Any, memo: dict[int, Any] | None = None, _nil: list[Any] = [], non_blocking: bool = False) -> Any:
        del _nil
        memo = {} if memo is None else memo
        if isinstance(x, tp.Tensor):
            return self._offload_tensor(x, memo, non_blocking)
        if isinstance(x, dict):
            return {key: self.deepcopy_with_tensor_offload(value, memo, non_blocking=non_blocking) for key, value in x.items()}
        if isinstance(x, list):
            return [self.deepcopy_with_tensor_offload(value, memo, non_blocking=non_blocking) for value in x]
        if isinstance(x, tuple):
            return tuple(self.deepcopy_with_tensor_offload(value, memo, non_blocking=non_blocking) for value in x)
        return copy.deepcopy(x, memo)

    def stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.deepcopy_with_tensor_offload(state_dict, non_blocking=bool(kwargs.get("non_blocking", False)))

    def _stage_untyped_storage(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return None

    def _keep_alive(self, x: Any, memo: Any) -> Any:
        del memo
        self._keep_alive_values.append(x)
        return x

    def _reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        del kwargs
        return args[0] if args else None

    def close(self) -> None:
        self._keep_alive_values.clear()
