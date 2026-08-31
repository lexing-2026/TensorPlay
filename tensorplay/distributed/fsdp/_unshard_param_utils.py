"""Explicit parameter materialization helpers."""

from contextlib import contextmanager
from typing import Any

__all__ = [
    "_writeback_to_local_shard",
    "_deregister_flat_param",
    "_register_flat_param",
    "_unflatten_as_params",
    "_validate_unshard_params_args",
    "_unshard_fsdp_state_params",
    "_unshard_params_for_summon",
    "_unshard_params",
    "_deregister_orig_params",
    "_register_orig_params",
]


def _writeback_to_local_shard(handle: Any, writeback_grad: bool = False) -> None:
    del writeback_grad
    handle.reshard()


def _deregister_flat_param(state: Any, module: Any) -> None:
    del module
    state._flat_param = None


def _register_flat_param(state: Any, module: Any) -> None:
    del module
    state._flat_param = getattr(state._param_group, "flat_param", None)


def _unflatten_as_params(state: Any, module: Any) -> None:
    del module
    for fsdp_param in state._fsdp_param_group().params:
        fsdp_param._setattr_on_modules(fsdp_param.unsharded_param())


def _validate_unshard_params_args(state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    del state, writeback, rank0_only, offload_to_cpu, with_grads


def _unshard_fsdp_state_params(module: Any, state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
    state._fsdp_param_group().unshard()
    if offload_to_cpu:
        for param in state._fsdp_param_group().params:
            param._full_tensor = param._full_tensor.cpu()


@contextmanager
def _unshard_params_for_summon(module: Any, state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    _unshard_fsdp_state_params(module, state, writeback, rank0_only, offload_to_cpu, with_grads)
    try:
        yield
    finally:
        state._fsdp_param_group().reshard()


def _unshard_params(module: Any, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    modules = module.modules() if recurse else (module,)
    for item in modules:
        state = getattr(item, "_fsdp_state", None)
        if state is not None:
            _unshard_fsdp_state_params(item, state, writeback, rank0_only, offload_to_cpu, with_grads)


def _deregister_orig_params(state: Any, module: Any) -> None:
    del state, module


def _register_orig_params(state: Any, module: Any) -> None:
    del state, module
