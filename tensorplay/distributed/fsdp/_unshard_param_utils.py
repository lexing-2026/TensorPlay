"""Explicit parameter materialization helpers."""

from contextlib import contextmanager
from typing import Any

from .. import distributed_core as dist
from tensorplay.nn.parameter import Parameter

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
    if writeback_grad:
        for param in getattr(handle, "params", ()):
            gradient = getattr(param._full_tensor, "grad", None)
            if gradient is not None:
                param._capture_full_gradient(gradient)
    handle.reshard()


def _deregister_flat_param(state: Any, module: Any) -> None:
    del module
    flat_param = getattr(state, "_flat_param", None)
    if flat_param is not None:
        state._saved_flat_param = flat_param
        state._flat_param = None


def _register_flat_param(state: Any, module: Any) -> None:
    del module
    if getattr(state, "_flat_param", None) is None:
        state._flat_param = getattr(state, "_saved_flat_param", None)


def _unflatten_as_params(state: Any, module: Any) -> None:
    del module
    for fsdp_param in state._fsdp_param_group().params:
        fsdp_param._setattr_on_modules(fsdp_param.unsharded_param())


def _validate_unshard_params_args(state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    if rank0_only and writeback:
        raise ValueError("rank0_only cannot be combined with writeback")
    if with_grads and offload_to_cpu:
        raise ValueError("with_grads cannot be combined with offload_to_cpu")
    if not isinstance(writeback, bool) or not isinstance(rank0_only, bool):
        raise TypeError("writeback and rank0_only must be booleans")
    if not hasattr(state, "_fsdp_param_group"):
        raise TypeError("state does not describe a sharded module")


def _unshard_fsdp_state_params(module: Any, state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
    if rank0_only and dist.is_initialized() and dist.get_rank() != 0:
        state._fsdp_param_group().reshard()
        return
    state._fsdp_param_group().unshard()
    if offload_to_cpu:
        for param in state._fsdp_param_group().params:
            param._full_tensor = param._full_tensor.cpu()
            param._setattr_on_modules(
                Parameter(param._full_tensor, requires_grad=param.param.requires_grad)
            )
    if with_grads:
        for param in state._fsdp_param_group().params:
            local = param._gradient_hook_param
            if getattr(local, "grad", None) is not None:
                param._full_tensor.grad = local.grad.detach().clone()


@contextmanager
def _unshard_params_for_summon(module: Any, state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    _unshard_fsdp_state_params(module, state, writeback, rank0_only, offload_to_cpu, with_grads)
    try:
        yield
    finally:
        if not (rank0_only and dist.is_initialized() and dist.get_rank() != 0):
            state._fsdp_param_group().reshard()


def _unshard_params(module: Any, recurse: bool, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    modules = module.modules() if recurse else (module,)
    seen: set[int] = set()
    for item in modules:
        state = getattr(item, "_fsdp_state", None)
        if state is not None and id(state) not in seen:
            seen.add(id(state))
            _unshard_fsdp_state_params(item, state, writeback, rank0_only, offload_to_cpu, with_grads)


def _deregister_orig_params(state: Any, module: Any) -> None:
    flat_param = getattr(state, "_flat_param", None)
    if flat_param is None:
        return
    metadata = getattr(flat_param, "_param_metadata", None)
    names = {getattr(info, "name", None) for info in getattr(metadata, "param_infos", ())}
    for name in list(getattr(module, "_parameters", {})):
        if name in names:
            module._parameters[name] = flat_param


def _register_orig_params(state: Any, module: Any) -> None:
    del module
    group = getattr(state, "_param_group", None)
    if group is not None:
        group.unshard()
