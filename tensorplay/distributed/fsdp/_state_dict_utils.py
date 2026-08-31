"""State-dict hook helpers for sharded modules."""

from contextlib import contextmanager
from typing import Any

from .api import FullStateDictConfig, StateDictType

__all__ = ["_register_all_state_dict_hooks"]


def _should_unshard_params(fsdp_state: Any) -> bool:
    return getattr(fsdp_state, "_state_dict_type", StateDictType.FULL_STATE_DICT) != StateDictType.LOCAL_STATE_DICT


def _convert_to_wrapped_module_name(module_name: str) -> str:
    return f"_fsdp_wrapped_module.{module_name}" if module_name else "_fsdp_wrapped_module"


def _param_name_infos(module: Any, fsdp_state: Any) -> list[tuple[str, Any]]:
    del fsdp_state
    return list(module.named_parameters())


def _shared_param_name_infos(module: Any, fsdp_state: Any) -> list[tuple[str, Any]]:
    return _param_name_infos(module, fsdp_state)


@contextmanager
def _enter_unshard_params_ctx(module: Any, fsdp_state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    del module, writeback, rank0_only, offload_to_cpu, with_grads
    fsdp_state._fsdp_param_group().unshard()
    try:
        yield
    finally:
        fsdp_state._fsdp_param_group().reshard()


def _exit_unshard_params_ctx(module: Any, fsdp_state: Any) -> None:
    del module
    fsdp_state._fsdp_param_group().reshard()


def _common_pre_state_dict_hook(module: Any, fsdp_state: Any) -> None:
    del module
    fsdp_state._fsdp_param_group().unshard()


def _common_unshard_pre_state_dict_hook(module: Any, fsdp_state: Any, offload_to_cpu: bool, rank0_only: bool) -> None:
    del module, offload_to_cpu, rank0_only
    fsdp_state._fsdp_param_group().unshard()


def _common_unshard_post_state_dict_hook(module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str, param_hook: Any) -> dict[str, Any]:
    del module, fsdp_state, prefix, param_hook
    return state_dict


def _full_pre_state_dict_hook(fsdp_state: Any, module: Any) -> None:
    _common_pre_state_dict_hook(module, fsdp_state)


def _full_post_state_dict_hook(module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    del module, fsdp_state, prefix
    return state_dict


_local_pre_state_dict_hook = _full_pre_state_dict_hook
_local_post_state_dict_hook = _full_post_state_dict_hook
_sharded_pre_state_dict_hook = _full_pre_state_dict_hook
_sharded_post_state_dict_hook = _full_post_state_dict_hook


def _full_pre_load_state_dict_hook(module: Any, fsdp_state: Any, state_dict: dict[str, Any], prefix: str) -> None:
    del module, state_dict, prefix
    fsdp_state._fsdp_param_group().unshard()


_local_pre_load_state_dict_hook = _full_pre_load_state_dict_hook
_sharded_pre_load_state_dict_hook = _full_pre_load_state_dict_hook


def _full_post_load_state_dict_hook(module: Any, fsdp_state: Any) -> None:
    del module
    fsdp_state._fsdp_param_group().reshard()


_local_post_load_state_dict_hook = _full_post_load_state_dict_hook
_sharded_post_load_state_dict_hook = _full_post_load_state_dict_hook


def _replace_with_full_state_dict_type(fsdp_state: Any) -> Any:
    fsdp_state._state_dict_type = StateDictType.FULL_STATE_DICT
    fsdp_state._state_dict_config = FullStateDictConfig()
    return fsdp_state


def _post_state_dict_hook(module: Any, state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    del module, prefix
    return state_dict


def _pre_state_dict_hook(module: Any) -> None:
    del module


def _set_use_dtensor(fsdp_state: Any, use_dtensor: bool) -> None:
    fsdp_state._use_dtensor = bool(use_dtensor)


def _pre_load_state_dict_hook(module: Any, state_dict: dict[str, Any], prefix: str) -> None:
    del module, state_dict, prefix


def _post_load_state_dict_hook(module: Any, incompatible_keys: Any) -> Any:
    del module
    return incompatible_keys


def _register_state_dict_hooks_base(state: Any, hook_registration_fn_name: str, hook: Any, hook_registration_fn_kwargs: dict[str, Any] | None = None) -> Any:
    register = getattr(state.module, hook_registration_fn_name, None)
    if register is None:
        return None
    return register(hook, **(hook_registration_fn_kwargs or {}))


def _register_all_state_dict_hooks(state: Any) -> None:
    state._state_dict_hooks_registered = True
