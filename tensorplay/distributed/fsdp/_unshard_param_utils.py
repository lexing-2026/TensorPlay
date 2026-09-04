"""Explicit parameter materialization helpers."""

from contextlib import ExitStack, contextmanager
from typing import Any

import tensorplay as tp

from .. import distributed_core as dist
from tensorplay.nn.parameter import Parameter
from ._common_utils import (
    HandleTrainingState,
    TrainingState,
    _get_module_fsdp_state,
    _module_handle,
)

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


def _param_groups(state: Any) -> list[Any]:
    getter = getattr(state, "_all_param_groups", None)
    if callable(getter):
        return list(getter())
    group = getattr(state, "_param_group", None)
    return [group] if group is not None else []


@tp.no_grad()
def _writeback_to_local_shard(handle: Any, writeback_grad: bool = False) -> None:
    def _get_shard(flat_param_or_grad: Any) -> Any:
        uses_sharded = getattr(handle, "uses_sharded_strategy", False)
        if callable(uses_sharded):
            uses_sharded = uses_sharded()
        if uses_sharded:
            from ._flat_param import FlatParamHandle

            shard, _ = FlatParamHandle._get_unpadded_shard(
                flat_param_or_grad,
                int(getattr(handle, "rank", 0)),
                int(getattr(handle, "world_size", 1)),
            )
            return shard
        return flat_param_or_grad

    flat_param = getattr(handle, "flat_param", None)
    if flat_param is None:
        return
    param_shard = _get_shard(flat_param)
    local_shard = getattr(handle, "_local_shard", flat_param)
    local_shard[: int(param_shard.numel())].copy_(param_shard)
    if writeback_grad:
        existing_grad = getattr(handle, "sharded_grad", None)
        if callable(existing_grad):
            existing_grad = existing_grad()
        if existing_grad is not None:
            flat_grad = getattr(flat_param, "grad", None)
            if flat_grad is None:
                raise AssertionError("expected a materialized gradient")
            grad_shard = _get_shard(flat_grad)
            existing_grad[: int(grad_shard.numel())].copy_(grad_shard)


def _deregister_flat_param(state: Any, module: Any) -> None:
    handle = _module_handle(state, module)
    flat_param = getattr(handle, "flat_param", None)
    if flat_param is None:
        flat_param = getattr(state, "_flat_param", None)
    if flat_param is None:
        return
    target = getattr(module, "module", module)
    parameters = getattr(target, "_parameters", None)
    if parameters is not None:
        state._saved_flat_param = flat_param
        parameters.pop("_flat_param", None)
        state._flat_param = None


def _register_flat_param(state: Any, module: Any) -> None:
    handle = _module_handle(state, module)
    flat_param = getattr(handle, "flat_param", None)
    if flat_param is None:
        flat_param = getattr(state, "_saved_flat_param", None)
    if flat_param is None:
        flat_param = getattr(state, "_flat_param", None)
    if flat_param is None:
        return
    target = getattr(module, "module", module)
    parameters = getattr(target, "_parameters", None)
    if parameters is not None:
        parameters["_flat_param"] = flat_param
    state._flat_param = flat_param


@contextmanager
def _unflatten_as_params(state: Any, module: Any):
    handle = getattr(state, "_handle", None)
    if handle is None:
        yield
        return
    _deregister_flat_param(state, module)
    try:
        with handle.unflatten_as_params():
            yield
    finally:
        if not bool(getattr(handle, "_use_orig_params", False)):
            _register_flat_param(state, module)


def _validate_unshard_params_args(state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool) -> None:
    if rank0_only and writeback:
        raise ValueError("rank0_only cannot be combined with writeback")
    if with_grads and offload_to_cpu:
        raise ValueError("with_grads cannot be combined with offload_to_cpu")
    if not isinstance(writeback, bool) or not isinstance(rank0_only, bool):
        raise TypeError("writeback and rank0_only must be booleans")
    groups = _param_groups(state)
    handle = getattr(state, "_handle", None) or _module_handle(state)
    if not groups and handle is None:
        raise TypeError("state does not describe a sharded module")
    if groups:
        return
    use_orig_params = bool(
        getattr(state, "_use_orig_params", getattr(state, "use_orig_params", False))
    )
    if with_grads and (offload_to_cpu or not use_orig_params):
        raise NotImplementedError(
            f"with_grads={with_grads}, use_orig_params={use_orig_params}, "
            f"offload_to_cpu={offload_to_cpu} is not supported"
        )
    uses_sharded = getattr(handle, "uses_sharded_strategy", False)
    if callable(uses_sharded):
        uses_sharded = uses_sharded()
    if offload_to_cpu and not uses_sharded:
        raise NotImplementedError("offload_to_cpu requires a sharded strategy")
    if offload_to_cpu and not rank0_only:
        import warnings

        warnings.warn(
            "offload_to_cpu=True and rank0_only=False may duplicate CPU parameter storage",
            stacklevel=2,
        )


@contextmanager
def _unshard_fsdp_state_params(
    module: Any,
    state: Any,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    groups = _param_groups(state)
    rank = int(getattr(state, "rank", 0))
    if dist.is_initialized() and not hasattr(state, "rank"):
        rank = int(dist.get_rank(getattr(state, "process_group", None)))
    nonzero_rank = rank0_only and rank != 0
    previous_summoning = bool(getattr(state, "_summoning_full_params", False))
    state._summoning_full_params = True
    try:
        if groups:
            snapshots: list[tuple[Any, Any, Any]] = []
            for group in groups:
                for param in group.params:
                    local = param._sharded_local_tensor()
                    snapshots.append(
                        (param, local.detach().clone(), getattr(local, "device", None))
                    )
                group.unshard()
            if nonzero_rank:
                for group in reversed(groups):
                    group.reshard()
                yield
                return
            if offload_to_cpu:
                for param, _, _ in snapshots:
                    full = param._full_tensor
                    if (
                        getattr(full, "device", None) is not None
                        and str(full.device) != "cpu"
                    ):
                        param._full_tensor = full.to("cpu")
                        param._unsharded_param = param._full_tensor
                        param._setattr_on_modules(
                            Parameter(
                                param._full_tensor,
                                requires_grad=param.param.requires_grad,
                            )
                        )
            if with_grads:
                for param, _, _ in snapshots:
                    local = param._gradient_hook_param
                    if getattr(local, "grad", None) is not None:
                        param._full_tensor.grad = local.grad.detach().clone()
            try:
                yield
            finally:
                if not writeback:
                    from ._fully_shard._fsdp_param import ShardedState

                    for param, local, device in snapshots:
                        sharded = param._sharded_tensor
                        if sharded is None:
                            continue
                        sharded_local = sharded.to_local()
                        if (
                            device is not None
                            and str(getattr(local, "device", "")) != str(device)
                        ):
                            local = local.to(device)
                        with tp.no_grad():
                            sharded_local.copy_(local)
                        param._state = ShardedState.SHARDED
                elif offload_to_cpu:
                    for param, _, device in snapshots:
                        full = param._full_tensor
                        if (
                            device is not None
                            and getattr(full, "device", None) is not None
                            and str(full.device) != str(device)
                        ):
                            param._full_tensor = full.to(device)
                            param._unsharded_param = param._full_tensor
                            param._setattr_on_modules(
                                Parameter(
                                    param._full_tensor,
                                    requires_grad=param.param.requires_grad,
                                )
                            )
                for group in reversed(groups):
                    group.reshard()
            return

        from ._runtime_utils import (
            _reset_flat_param_grad_info_if_needed,
            _reshard,
            _reshard_grads,
            _unshard,
            _unshard_grads,
        )

        handle = _module_handle(state, module)
        if handle is None:
            yield
            return
        if (
            getattr(handle, "_training_state", HandleTrainingState.IDLE)
            != HandleTrainingState.IDLE
        ):
            raise AssertionError("handle must be idle before full parameters are summoned")
        handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS
        _reset_flat_param_grad_info_if_needed((handle,))
        free_unsharded_flat_param = bool(handle.needs_unshard())
        device_handle = getattr(state, "_device_handle", None)
        current_stream = (
            device_handle.current_stream() if device_handle is not None else None
        )
        _unshard(state, handle, current_stream, current_stream)
        if with_grads:
            _unshard_grads(handle)
        if nonzero_rank:
            _reshard(state, handle, free_unsharded_flat_param)
            if with_grads:
                _reshard_grads(handle)
            try:
                yield
            finally:
                handle._training_state = HandleTrainingState.IDLE
            return
        with ExitStack() as stack:
            if offload_to_cpu and getattr(handle, "uses_sharded_strategy", False):
                stack.enter_context(handle.to_cpu())
            if not bool(
                getattr(
                    state,
                    "_use_orig_params",
                    getattr(state, "use_orig_params", False),
                )
            ):
                stack.enter_context(_unflatten_as_params(state, module))
            try:
                yield
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handle, with_grads)
                _reshard(state, handle, free_unsharded_flat_param)
                if with_grads:
                    _reshard_grads(handle)
                handle._training_state = HandleTrainingState.IDLE
    finally:
        state._summoning_full_params = previous_summoning


@contextmanager
def _unshard_params_for_summon(module: Any, state: Any, writeback: bool, rank0_only: bool, offload_to_cpu: bool, with_grads: bool):
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    from ._runtime_utils import _lazy_init

    _lazy_init(state, module)
    if (
        getattr(state, "_training_state", TrainingState.IDLE)
        == TrainingState.FORWARD_BACKWARD
    ):
        raise AssertionError("cannot summon full parameters during forward/backward")
    if (
        getattr(state, "_training_state", TrainingState.IDLE)
        == TrainingState.SUMMON_FULL_PARAMS
    ):
        raise AssertionError("cannot summon full parameters recursively")
    if getattr(state, "_summoning_full_params", False):
        raise AssertionError("cannot summon full parameters recursively")
    previous_training_state = getattr(state, "_training_state", TrainingState.IDLE)
    state._training_state = TrainingState.SUMMON_FULL_PARAMS
    try:
        with _unshard_fsdp_state_params(
            module,
            state,
            writeback,
            rank0_only,
            offload_to_cpu,
            with_grads,
        ):
            yield
    finally:
        state._training_state = previous_training_state


@contextmanager
def _unshard_params(
    module: Any,
    recurse: bool,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    from . import _traversal_utils

    if recurse:
        states, modules = _traversal_utils._get_fsdp_states_with_modules(module)
    else:
        state = _get_module_fsdp_state(module)
        states, modules = (([state], [module]) if state is not None else ([], []))
    with ExitStack() as stack:
        for state, state_module in zip(states, modules):
            stack.enter_context(
                _unshard_params_for_summon(
                    state_module,
                    state,
                    writeback,
                    rank0_only,
                    offload_to_cpu,
                    with_grads,
                )
            )
        yield


def _deregister_orig_params(state: Any, module: Any) -> None:
    handle = getattr(state, "_handle", None)
    if handle is None:
        return
    deregister = getattr(handle, "_deregister_orig_params", None)
    if callable(deregister):
        deregister()
    _register_flat_param(state, module)


def _register_orig_params(state: Any, module: Any) -> None:
    handle = getattr(state, "_handle", None)
    if handle is None:
        return
    _deregister_flat_param(state, module)
    flat_param = getattr(handle, "flat_param", None)
    if flat_param is not None and bool(handle.is_sharded(flat_param)):
        handle._use_sharded_views()
        handle._use_sharded_grad_views()
    else:
        handle._use_unsharded_views(as_params=True)
