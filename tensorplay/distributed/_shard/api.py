"""Functional parameter and module sharding APIs."""

from typing import Any, Callable

from .sharded_tensor.api import ShardedTensor

__all__ = ["_shard_tensor", "shard_parameter", "load_with_process_group", "shard_module"]


def _get_current_process_group() -> Any:
    from .. import distributed_core as dist
    return dist._get_default_group()


def _shard_tensor(tensor: Any, sharding_spec: Any, src_rank: int = 0, process_group: Any = None) -> ShardedTensor:
    return sharding_spec.shard(tensor, src_rank=src_rank, process_group=process_group)


def shard_parameter(module: Any, param_name: str, sharding_spec: Any, src_rank: int = 0, process_group: Any = None) -> ShardedTensor:
    param = module.get_parameter(param_name) if hasattr(module, "get_parameter") else getattr(module, param_name)
    sharded = _shard_tensor(param, sharding_spec, src_rank, process_group)
    module._parameters[param_name] = sharded
    return sharded


def load_with_process_group(module: Any, state_dict: dict[str, Any], process_group: Any = None) -> Any:
    del process_group
    return module.load_state_dict(state_dict)


def _reshard_output(output: Any, output_spec: Any, process_group: Any = None) -> Any:
    del process_group
    return output_spec.shard(output) if output_spec is not None and hasattr(output_spec, "shard") else output


def _collect_local_shard(output: Any) -> Any:
    return output.to_local() if isinstance(output, ShardedTensor) else output


def shard_module(module: Any, plan: Any, output_plan: Any = None, return_local_tensor: Any = None, src_rank: int = 0, process_group: Any = None) -> Any:
    for name, spec in getattr(plan, "items", lambda: plan)():
        if hasattr(spec, "shard") and "." in name:
            parent_name, param_name = name.rsplit(".", 1)
            parent = module.get_submodule(parent_name)
            shard_parameter(parent, param_name, spec, src_rank, process_group)
        elif hasattr(spec, "shard"):
            shard_parameter(module, name, spec, src_rank, process_group)
    if output_plan:
        setattr(module, "_sharding_output_plan", output_plan)
    if return_local_tensor:
        setattr(module, "_return_local_tensor", set(return_local_tensor))
    return module
