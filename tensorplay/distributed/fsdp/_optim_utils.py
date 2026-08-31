"""Optimizer-state reshaping helpers for sharded parameters."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, NamedTuple

import tensorplay as tp

__all__ = [
    "FSDPParamInfo",
    "sorted_items",
    "StateInfo",
    "_ConsolidatedOptimState",
    "_PosDimTensorInfo",
    "_OptimStateKey",
]


@dataclass
class FSDPParamInfo:
    state: Any
    handle: Any
    param_indices: dict[str, int] = field(default_factory=dict)
    param_requires_grad: list[bool] = field(default_factory=list)


def sorted_items(dictionary: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for key in sorted(dictionary):
        yield key, dictionary[key]


@dataclass
class _ConsolidatedOptimState:
    tensor_state: dict[str, Any] = field(default_factory=dict)
    zero_dim_tensor_state: dict[str, Any] = field(default_factory=dict)
    non_tensor_state: dict[str, Any] = field(default_factory=dict)


class _PosDimTensorInfo(NamedTuple):
    shape: tuple[int, ...]
    dtype: Any


class _OptimStateKey(NamedTuple):
    unflat_param_names: tuple[str, ...]
    is_fsdp_managed: bool


@dataclass
class StateInfo:
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_zero_dim_tensor(value: Any) -> bool:
    return isinstance(value, tp.Tensor) and value.dim() == 0


def _clone_state(value: Any, cpu_offload: bool = False) -> Any:
    if isinstance(value, tp.Tensor):
        value = value.detach().clone()
        return value.cpu() if cpu_offload else value
    if isinstance(value, dict):
        return {key: _clone_state(item, cpu_offload) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item, cpu_offload) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item, cpu_offload) for item in value)
    return value


def _unflatten_optim_state(fsdp_param_info: FSDPParamInfo, flat_param_state: dict[str, Any], to_save: bool, shard_state: bool, cpu_offload: bool) -> list[dict[str, Any]]:
    del shard_state
    if not to_save:
        return []
    values = []
    handle = fsdp_param_info.handle
    names = list(getattr(handle, "param_module_names", lambda: ())())
    for name in names or [str(index) for index in range(len(getattr(handle, "params", ())))]:
        values.append({key: _clone_state(value, cpu_offload) for key, value in flat_param_state.items()})
    return values


def _communicate_optim_state(fsdp_param_info: FSDPParamInfo, flat_param_state: dict[str, Any]) -> dict[str, Any]:
    del fsdp_param_info
    return _clone_state(flat_param_state)


def _unflatten_communicated_optim_state(fsdp_param_info: FSDPParamInfo, state: dict[str, Any], shard_state: bool) -> list[dict[str, Any]]:
    return _unflatten_optim_state(fsdp_param_info, state, True, shard_state, False)


def _broadcast_processed_state(fsdp_state: Any, optim_state: Any, group: Any) -> Any:
    del fsdp_state, group
    return optim_state


def _broadcast_state(fsdp_state: Any, state: Any, group: Any) -> Any:
    del fsdp_state, group
    return state


def _shard_orig_param_state(fsdp_param_info: FSDPParamInfo, fqn: str, optim_state: dict[str, Any]) -> dict[str, Any]:
    del fsdp_param_info, fqn
    return optim_state


def _flatten_optim_state_dict(optim_state_dict: dict[str, Any], model: Any, use_orig_params: bool, optim: Any, rank0_only: bool, group: Any) -> dict[str, Any]:
    del model, use_orig_params, optim, rank0_only, group
    return _clone_state(optim_state_dict)


def _flatten_optim_state(fsdp_param_info: FSDPParamInfo, unflat_osd_state: dict[str, Any], unflat_param_names: Iterable[str]) -> dict[str, Any]:
    del fsdp_param_info, unflat_param_names
    return unflat_osd_state


def _flatten_tensor_optim_state(state_name: str, pos_dim_tensors: Any, unflat_param_names: Any, unflat_param_shapes: Any, handle: Any) -> Any:
    del state_name, unflat_param_names, unflat_param_shapes, handle
    return pos_dim_tensors


def _flatten_zero_dim_tensor_optim_state(state_name: str, zero_dim_tensors: Any, unflat_param_names: Any) -> Any:
    del state_name, unflat_param_names
    return zero_dim_tensors


def _flatten_non_tensor_optim_state(state_name: str, non_tensors: Any, unflat_param_names: Any) -> Any:
    del state_name, unflat_param_names
    return non_tensors


def _rekey_sharded_optim_state_dict(sharded_osd: dict[str, Any], model: Any, optim: Any, optim_input: Any, using_optim_input: bool, is_named_optimizer: bool) -> dict[str, Any]:
    del model, optim, optim_input, using_optim_input, is_named_optimizer
    return sharded_osd


def _get_param_id_to_param_from_optim_input(model: Any, optim_input: Any) -> dict[Any, Any]:
    del model
    if optim_input is None:
        return {}
    return {index: value for index, value in enumerate(optim_input)}


def _get_flat_param_to_fqn(model: Any) -> dict[Any, str]:
    return {param: name for name, param in model.named_parameters()}


def _get_param_key_to_param(optim: Any, model: Any, is_named_optimizer: bool, param_to_fqns: Any, flat_param_to_fqn: Any) -> dict[Any, Any]:
    del model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
    return {id(param): param for group in optim.param_groups for param in group["params"]}


def _get_param_to_param_key(optim: Any, model: Any, is_named_optimizer: bool, param_to_fqns: Any, flat_param_to_fqn: Any) -> dict[Any, Any]:
    return {value: key for key, value in _get_param_key_to_param(optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn).items()}


def _get_param_to_param_id_from_optim_input(model: Any, optim_input: Any) -> dict[Any, int]:
    del model
    return {value: index for index, value in enumerate(optim_input or ())}


def _check_missing_keys_on_rank(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _map_param_key_to_optim_keys(optim_state_dict: Any, group: Any, param_key_to_param: Any, param_to_fqns: Any, fqn_to_fsdp_param_info: Any, merge_keys: bool) -> Any:
    del group, param_key_to_param, param_to_fqns, fqn_to_fsdp_param_info, merge_keys
    return optim_state_dict


def _unflatten_param_groups(state_dict: Any, param_key_to_param: Any, param_to_fqns: Any) -> Any:
    del param_key_to_param, param_to_fqns
    return state_dict


def _is_named_optimizer(optim_state_dict: dict[str, Any]) -> bool:
    return bool(optim_state_dict.get("param_groups") and isinstance(optim_state_dict["param_groups"][0].get("params", [None])[0], str))


def _allgather_state_info(fsdp_state: Any, input_states: Any) -> Any:
    del fsdp_state
    return input_states


def _convert_all_state_info(fsdp_param_info: Any, gathered_state_info: Any, input_states: Any, output_states: Any) -> Any:
    del fsdp_param_info, gathered_state_info, input_states
    return output_states


def _unflatten_orig_param_states(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return {}


def _allgather_orig_param_states(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return {}


def _gather_all_orig_param_state(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return {}


def _convert_state_with_orig_params(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return {}


def _convert_state_with_flat_params(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return {}


def _optim_state_dict(model: Any, optim: Any, optim_state_dict: Any, optim_input: Any, rank0_only: bool, shard_state: bool, group: Any, using_optim_input: bool, use_orig_params: bool, cpu_offload: bool) -> dict[str, Any]:
    del model, optim_input, rank0_only, shard_state, group, using_optim_input, use_orig_params
    source = optim_state_dict if optim_state_dict is not None else optim.state_dict()
    return _clone_state(source, cpu_offload)


def _get_fqn_to_fsdp_param_info(model: Any) -> dict[str, FSDPParamInfo]:
    result = {}
    for name, param in model.named_parameters():
        result[name] = FSDPParamInfo(None, None, {name: 0}, [getattr(param, "requires_grad", False)])
    return result


def _set_optim_use_dtensor(fsdp_state: Any, state_dict_settings: Any) -> None:
    fsdp_state._use_dtensor = bool(getattr(state_dict_settings, "state_dict_config", None) and getattr(state_dict_settings.state_dict_config, "_use_dtensor", False))
