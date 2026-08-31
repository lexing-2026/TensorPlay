"""Optimizer wrapper that updates local shard tensors."""

from collections.abc import Mapping
from typing import Any

from ..sharded_tensor.api import ShardedTensor

__all__ = ["ShardedOptimizer", "named_params_with_sharded_tensor"]


def named_params_with_sharded_tensor(named_params: Mapping[str, Any]):
    for name, value in named_params.items():
        if isinstance(value, ShardedTensor):
            for shard in value.local_shards():
                yield f"{name}.local", shard.tensor
        else:
            yield name, value


class ShardedOptimizer:
    def __init__(self, named_params: Mapping[str, Any], optimizer_class: Any, *optimizer_args: Any, **optimizer_kwargs: Any) -> None:
        self.named_params = named_params
        self._optim = optimizer_class([value for _, value in named_params_with_sharded_tensor(named_params)], *optimizer_args, **optimizer_kwargs)

    @property
    def param_groups(self):
        return self._optim.param_groups

    @property
    def state(self):
        return self._optim.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optim.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:
        return self._optim.step(closure)

    def state_dict(self) -> dict[str, Any]:
        state = self._optim.state_dict()
        state["named_params"] = tuple(self.named_params.keys())
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> Any:
        return self._optim.load_state_dict(dict(state_dict))

    def add_param_group(self, param_group: Any) -> None:
        self._optim.add_param_group(param_group)
