"""Module wrapper for fully sharded data parallel execution."""

import contextlib
import copy
import math
import warnings
from enum import Enum, auto
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.nn.modules.module import Module
from tensorplay.nn.parameter import Parameter

from .. import distributed_core as dist
from ._common_utils import TrainingState, _FSDPDeviceHandle
from ._fully_shard import FSDPModule, fully_shard
from ._fully_shard._fsdp_api import CPUOffloadPolicy, DataParallelMeshDims, MixedPrecisionPolicy, OffloadPolicy
from ._fully_shard._fsdp_init import _init_default_mesh
from ._fully_shard._fsdp_param import ShardedState
from ._optim_utils import (
    _flatten_optim_state_dict,
    _optim_state_dict,
    _rekey_sharded_optim_state_dict,
)
from ._state_dict_utils import (
    _register_all_state_dict_hooks,
)
from ._unshard_param_utils import _unshard_params_for_summon
from ._init_utils import _sync_module_params_and_buffers
from ..device_mesh import DeviceMesh
from ..tensor import Replicate
from ._wrap_utils import _auto_wrap
from .api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)

__all__ = ["FullyShardedDataParallel", "OptimStateKeyType"]


class OptimStateKeyType(Enum):
    PARAM_NAME = auto()
    PARAM_ID = auto()


def _global_rank() -> int:
    try:
        return int(dist.get_rank()) if dist.is_initialized() else 0
    except (RuntimeError, ValueError):
        return 0


def _normalize_ignored_states(
    module: Module,
    ignored_modules: Iterable[Module] | None,
    ignored_states: Iterable[Any] | None,
) -> tuple[set[Module], set[Any]]:
    modules = set(ignored_modules or ())
    states = tuple(ignored_states or ())
    if modules and any(item not in set(module.modules()) for item in modules):
        raise ValueError("ignored module must be contained in the wrapped module")
    if states and all(isinstance(item, Module) for item in states):
        if any(item not in set(module.modules()) for item in states):
            raise ValueError("ignored module must be contained in the wrapped module")
        modules.update(states)
        states = ()
    elif states and any(isinstance(item, Module) for item in states):
        raise TypeError("ignored_states must contain only modules or parameters")
    params = set(states)
    for ignored_module in modules:
        params.update(ignored_module.parameters())
    available = set(module.parameters())
    if any(param not in available for param in params):
        raise ValueError("ignored parameter must be contained in the wrapped module")
    return modules, params


def _module_device_type(module: Module, device_id: Any) -> str:
    if device_id is not None:
        if isinstance(device_id, int):
            return "cuda"
        value = getattr(device_id, "type", None)
        if value is not None:
            return str(value)
        return str(device_id).split(":", 1)[0]
    for param in module.parameters():
        device = getattr(param, "device", None)
        kind = getattr(device, "type", None)
        if kind is None:
            kind = str(device).split(":", 1)[0]
        if kind not in {"", "meta", "None"}:
            return str(kind)
    return "cpu"


def _device_value(device_id: Any, device_type: str) -> Any:
    if device_id is None:
        return None
    if isinstance(device_id, int):
        return tp.device(device_type, device_id)
    return device_id


def _prepare_module_for_sharding(
    module: Module,
    ignored_modules: set[Module],
    ignored_params: set[Any],
    param_init_fn: Any,
    device_id: Any,
) -> None:
    if param_init_fn is not None and not callable(param_init_fn):
        raise TypeError("param_init_fn must be callable")
    device_type = _module_device_type(module, device_id)
    target_device = _device_value(device_id, device_type)
    for candidate in module.modules():
        if candidate in ignored_modules:
            continue
        direct_values = list(candidate.parameters(recurse=False)) + list(candidate.buffers(recurse=False))
        if not any(_device_kind(getattr(value, "device", None)) == "meta" for value in direct_values):
            continue
        if param_init_fn is not None:
            param_init_fn(candidate)
        if any(_device_kind(getattr(value, "device", None)) == "meta" for value in direct_values):
            device = target_device or device_type
            candidate.to_empty(device=device, recurse=False)
            reset = getattr(candidate, "reset_parameters", None)
            if callable(reset):
                reset()
    if target_device is not None:
        module.to(target_device)
    elif param_init_fn is not None:
        for param in module.parameters():
            if param in ignored_params:
                continue
            if _device_kind(param.device) == "meta":
                raise RuntimeError("parameter initialization left a meta parameter")


def _device_kind(value: Any) -> str:
    kind = getattr(value, "type", None)
    return str(kind) if kind is not None else str(value).split(":", 1)[0]


def _mesh_from_process_group(process_group: Any, device_type: str) -> DeviceMesh | None:
    if process_group is None:
        return None
    groups = list(process_group) if isinstance(process_group, tuple) else [process_group]
    if not groups:
        raise ValueError("process_group cannot be empty")
    if len(groups) == 1:
        ranks = dist.get_process_group_ranks(groups[0])
        return DeviceMesh.from_group(
            groups[0],
            device_type=device_type,
            mesh=ranks,
            mesh_dim_names=("dp",),
        )
    sizes = [len(dist.get_process_group_ranks(group)) for group in groups]
    ranks = sorted({rank for group in groups for rank in dist.get_process_group_ranks(group)})
    if math.prod(sizes) != len(ranks):
        raise ValueError("hybrid process groups must describe a rectangular mesh")

    def nest(values: list[int], shape: list[int]) -> Any:
        if len(shape) == 1:
            return values
        width = math.prod(shape[1:])
        return [nest(values[index * width:(index + 1) * width], shape[1:]) for index in range(shape[0])]

    mesh = DeviceMesh(
        device_type,
        nest(ranks, sizes),
        mesh_dim_names=tuple(f"dp{index}" for index in range(len(groups))),
    )
    mesh._dim_groups = {index: group for index, group in enumerate(groups)}
    return mesh


def _get_dp_mesh_dims(strategy: ShardingStrategy, mesh: DeviceMesh) -> Any:
    if strategy != ShardingStrategy.HYBRID_SHARD and strategy != ShardingStrategy._HYBRID_SHARD_ZERO2:
        return None
    if int(mesh.ndim) < 2:
        raise ValueError("hybrid sharding requires a two-dimensional mesh")
    names = mesh.mesh_dim_names
    if names is not None:
        return DataParallelMeshDims(shard=names[0], replicate=names[1])

    class _Dims:
        shard_names = (0,)
        replicate_names = (1,)

    return _Dims()


def _materialize_summoned_grads(snapshots: Iterable[tuple[Any, Any, Any]]) -> None:
    for param, _, _ in snapshots:
        local_param = param._gradient_hook_param
        local_grad = getattr(local_param, "grad", None)
        if local_grad is None:
            continue
        placement = param._placement
        if not hasattr(placement, "dim"):
            param._full_tensor.grad = local_grad.detach().clone()
            continue
        mesh = param.mesh_info.mesh
        mesh_dim = param.mesh_info.shard_mesh_dim
        count = int(mesh.size(mesh_dim))
        if count <= 1:
            param._full_tensor.grad = local_grad.detach().clone()
            continue
        dim = int(placement.dim)
        if dim < 0:
            dim += int(local_grad.dim())
        width = (int(param.param.shape[dim]) + count - 1) // count
        padded = local_grad.detach()
        pad = width - int(padded.shape[dim])
        if pad:
            from ..tensor._collective_utils import pad_tensor

            padded = pad_tensor(padded, dim, pad)
        outputs = [padded.new_empty(tuple(padded.shape)) for _ in range(count)]
        dist.all_gather(outputs, padded, group=mesh.get_group(mesh_dim))
        local_rank = int(mesh.get_local_rank(mesh_dim))
        outputs[local_rank] = padded
        full = tp.cat(tuple(outputs), dim=dim)
        total_padding = count * width - int(param.param.shape[dim])
        if total_padding:
            from ..tensor._collective_utils import unpad_tensor

            full = unpad_tensor(full, dim, total_padding)
        param._full_tensor.grad = full


class FullyShardedDataParallel(Module):
    """Wrap a module and manage its parameter shards around each forward."""

    def __init__(
        self,
        module: Module,
        process_group: Any = None,
        sharding_strategy: ShardingStrategy | None = None,
        cpu_offload: CPUOffload | None = None,
        auto_wrap_policy: Any = None,
        backward_prefetch: BackwardPrefetch | None = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: MixedPrecision | None = None,
        ignored_modules: Iterable[Module] | None = None,
        param_init_fn: Any = None,
        device_id: Any = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Iterable[Any] | None = None,
        device_mesh: Any = None,
    ) -> None:
        if not isinstance(module, Module):
            raise TypeError("module must be an instance of Module")
        if process_group is not None and device_mesh is not None:
            raise ValueError("process_group and device_mesh are mutually exclusive")
        if ignored_modules is not None and ignored_states is not None:
            raise ValueError("ignored_modules and ignored_states cannot both be supplied")
        super().__init__()
        self.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
        self.cpu_offload = cpu_offload or CPUOffload()
        self.mixed_precision = mixed_precision or MixedPrecision()
        self.use_orig_params = bool(use_orig_params)
        self.process_group = process_group
        self.auto_wrap_policy = auto_wrap_policy
        self.backward_prefetch = backward_prefetch
        self.forward_prefetch = bool(forward_prefetch)
        self.limit_all_gathers = bool(limit_all_gathers)
        self._param_init_fn = param_init_fn
        self._device_id = device_id
        self._sync_module_states = bool(sync_module_states)
        self._ignored_modules, self._ignored_params = _normalize_ignored_states(
            module, ignored_modules, ignored_states
        )
        _prepare_module_for_sharding(
            module,
            self._ignored_modules,
            self._ignored_params,
            param_init_fn,
            device_id,
        )
        device_type = _module_device_type(module, device_id)
        mesh = device_mesh or _mesh_from_process_group(process_group, device_type)
        if mesh is None:
            mesh = _init_default_mesh(device_type)
        self.device_mesh = mesh
        if sync_module_states:
            sync_group = process_group[0] if isinstance(process_group, tuple) else process_group
            _sync_module_params_and_buffers(
                module,
                [param for param in module.parameters() if param not in self._ignored_params],
                sync_group,
            )
        if auto_wrap_policy is not None:
            if not callable(auto_wrap_policy) and not hasattr(auto_wrap_policy, "_run_policy"):
                raise TypeError("auto_wrap_policy must be callable")
            _auto_wrap(
                module,
                auto_wrap_policy,
                self._ignored_modules,
                self._ignored_params,
                {
                    "process_group": process_group,
                    "sharding_strategy": self.sharding_strategy,
                    "cpu_offload": self.cpu_offload,
                    "backward_prefetch": backward_prefetch,
                    "mixed_precision": self.mixed_precision,
                    "param_init_fn": param_init_fn,
                    "device_id": device_id,
                    "sync_module_states": sync_module_states,
                    "forward_prefetch": forward_prefetch,
                    "limit_all_gathers": limit_all_gathers,
                    "use_orig_params": use_orig_params,
                    "ignored_states": None,
                    "device_mesh": device_mesh,
                },
                FullyShardedDataParallel,
            )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.mixed_precision.param_dtype,
            reduce_dtype=self.mixed_precision.reduce_dtype,
            output_dtype=self.mixed_precision.param_dtype,
            cast_forward_inputs=self.mixed_precision.cast_forward_inputs,
        )
        offload_policy: OffloadPolicy = (
            CPUOffloadPolicy() if self.cpu_offload.offload_params else OffloadPolicy()
        )
        dp_mesh_dims = _get_dp_mesh_dims(self.sharding_strategy, mesh)
        placement_fn = (
            lambda _param: Replicate()
            if self.sharding_strategy == ShardingStrategy.NO_SHARD
            else None
        )
        self._state_dict_type = StateDictType.FULL_STATE_DICT
        self._state_dict_config: StateDictConfig = FullStateDictConfig()
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()
        self._comm_hook = None
        self._no_sync = False
        self.module = fully_shard(
            module,
            mesh=mesh,
            reshard_after_forward=self.sharding_strategy == ShardingStrategy.FULL_SHARD,
            shard_placement_fn=placement_fn,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            ignored_params=self._ignored_params,
            dp_mesh_dims=dp_mesh_dims,
        )
        state = self.module._get_fsdp_state()
        mesh_info = getattr(state, "mesh_info", None)
        state.process_group = process_group or getattr(
            mesh_info, "shard_process_group", None
        )
        state.device_mesh = mesh
        state._device_mesh = mesh
        state.rank = int(getattr(mesh_info, "shard_mesh_rank", _global_rank()))
        state.world_size = int(getattr(mesh_info, "shard_world_size", 1))
        state.compute_device = getattr(state, "_device", None)
        state._device_handle = _FSDPDeviceHandle.from_device(state.compute_device)
        state._buffer_names = {name for name, _ in self.module.named_buffers()}
        state._buffer_name_to_orig_dtype = {
            name: getattr(buffer, "dtype", None)
            for name, buffer in self.module.named_buffers()
        }
        state._ignored_buffer_names = set()
        state.sharding_strategy = self.sharding_strategy
        state._ignored_modules = self._ignored_modules
        state._ignored_params = self._ignored_params
        state.mixed_precision = self.mixed_precision
        state.cpu_offload = self.cpu_offload
        state.backward_prefetch = backward_prefetch
        state.forward_prefetch = bool(forward_prefetch)
        state.limit_all_gathers = bool(limit_all_gathers)
        state.use_orig_params = self.use_orig_params
        state._device_id = device_id
        state._state_dict_type = self._state_dict_type
        state._state_dict_config = self._state_dict_config
        state._optim_state_dict_config = self._optim_state_dict_config
        from ._init_utils import _init_extension

        _init_extension(state, device_mesh)
        for group in state._all_param_groups():
            group._reshard_after_forward_enabled = (
                self.sharding_strategy == ShardingStrategy.FULL_SHARD
            )
            group._reshard_after_backward_enabled = True
        self._fsdp_state = state
        state._state_dict_wrapped_prefix = "module."
        _register_all_state_dict_hooks(state, module=self)

    @property
    def module(self) -> Module:
        return self._modules["module"]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self._modules.get("module")
            if module is not None:
                return getattr(module, name)
            raise

    def __getitem__(self, key: int) -> Any:
        return self.module[key]

    @property
    def _has_params(self) -> bool:
        return any(True for _ in self.module.parameters())

    @property
    def _flat_param(self) -> Any:
        state = getattr(self.module, "_fsdp_state", None)
        return getattr(state, "_flat_param", None) if state is not None else None

    def check_is_root(self) -> bool:
        state = getattr(self.module, "_fsdp_state", None)
        if state is None:
            return False
        if getattr(state, "_is_root", None) is None:
            state._lazy_init()
        return bool(state._is_root)

    @staticmethod
    def fsdp_modules(module: Module, root_only: bool = False) -> list[Any]:
        result: list[Any] = []
        state_ids: set[int] = set()
        for item in module.modules():
            if isinstance(item, FullyShardedDataParallel):
                target = item.module
            elif isinstance(item, FSDPModule):
                target = item
            else:
                continue
            state = getattr(target, "_fsdp_state", None)
            state_id = id(state) if state is not None else id(target)
            if state_id in state_ids:
                continue
            if root_only:
                if state is not None and getattr(state, "_is_root", None) is None:
                    state._lazy_init()
                if state is not None and not getattr(state, "_is_root", False):
                    continue
            state_ids.add(state_id)
            result.append(item)
        return result

    def apply(self, fn: Any) -> "FullyShardedDataParallel":
        state = self.module._get_fsdp_state()
        uninitialized = getattr(state, "_is_root", None) is None
        self._assert_state(TrainingState.IDLE)
        with _unshard_params_for_summon(
            self.module,
            state,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False,
        ):
            result = super().apply(fn)
        if uninitialized and getattr(state, "_is_root", None):
            for wrapper in self.fsdp_modules(self):
                target = wrapper.module if isinstance(wrapper, FullyShardedDataParallel) else wrapper
                target_state = getattr(target, "_fsdp_state", None)
                if target_state is not None:
                    target_state._reset_lazy_init()
        return result

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        return self.mixed_precision.buffer_dtype is not None

    def _low_precision_hook_enabled(self) -> bool:
        return self._comm_hook is not None

    def _reset_lazy_init(self) -> None:
        state = getattr(self.module, "_fsdp_state", None)
        if state is not None:
            state._reset_lazy_init()

    def _assert_state(self, state: TrainingState | list[TrainingState]) -> None:
        expected = [state] if isinstance(state, TrainingState) else list(state)
        current = getattr(self.module._get_fsdp_state(), "_training_state", None)
        if current not in expected:
            raise ValueError(
                f"expected to be in states {expected} but current state is {current}"
            )

    @staticmethod
    def _warn_optim_input(optim_input: Any, *, stacklevel: int = 1) -> None:
        if optim_input is not None:
            warnings.warn(
                "optim_input is deprecated",
                FutureWarning,
                stacklevel=stacklevel + 1,
            )

    @staticmethod
    def _is_using_optim_input(optim_input: Any, optim: Any) -> bool:
        return optim_input is not None or optim is None

    @staticmethod
    def _warn_legacy_optim_state_dict(
        current_name: str, new_name: str, *, stacklevel: int = 1
    ) -> None:
        warnings.warn(
            f"{current_name} is deprecated; use {new_name}",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )

    @staticmethod
    def _optim_state_dict_impl(
        model: Module,
        optim: Any,
        optim_state_dict: dict[str, Any] | None = None,
        optim_input: Any = None,
        rank0_only: bool = True,
        full_state_dict: bool = True,
        group: Any = None,
        cpu_offload: bool = True,
        *,
        _stacklevel: int = 1,
    ) -> dict[str, Any]:
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(
                optim_input, stacklevel=_stacklevel + 1
            )
        wrappers = FullyShardedDataParallel.fsdp_modules(model)
        use_orig_params = bool(getattr(wrappers[0], "use_orig_params", False)) if wrappers else False
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim
        )
        source = optim_state_dict
        if source is None and optim is not None:
            source = optim.state_dict()
        if source is None:
            raise ValueError("an optimizer or optimizer state is required")
        return _optim_state_dict(
            model,
            optim,
            source,
            optim_input,
            rank0_only,
            not full_state_dict,
            group,
            using_optim_input,
            use_orig_params,
            cpu_offload,
        )

    @staticmethod
    def _optim_state_dict_to_load_impl(
        optim_state_dict: dict[str, Any],
        model: Module,
        optim_input: Any = None,
        optim: Any = None,
        full_state_dict: bool = True,
        rank0_only: bool = False,
        is_named_optimizer: bool = False,
        group: Any = None,
    ) -> dict[str, Any]:
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(optim_input)
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input, optim
            )
        else:
            using_optim_input = False
            if optim_input is not None or rank0_only:
                raise AssertionError(
                    "full optimizer state loading requires rank0_only=False for a sharded input"
                )
        if rank0_only and dist.is_initialized() and _global_rank() != 0:
            source = {"state": {}}
        else:
            source = optim_state_dict
        wrappers = FullyShardedDataParallel.fsdp_modules(model)
        use_orig_params = bool(getattr(wrappers[0], "use_orig_params", False)) if wrappers else False
        flattened = _flatten_optim_state_dict(
            source,
            model=model,
            use_orig_params=use_orig_params,
            optim=optim if is_named_optimizer else None,
            rank0_only=rank0_only,
            group=group,
        )
        return _rekey_sharded_optim_state_dict(
            flattened,
            model,
            optim,
            optim_input,
            using_optim_input,
            is_named_optimizer,
        )

    @staticmethod
    def set_state_dict_type(module: Module, state_dict_type: StateDictType, state_dict_config: StateDictConfig | None = None, optim_state_dict_config: OptimStateDictConfig | None = None) -> StateDictSettings:
        targets = FullyShardedDataParallel.fsdp_modules(module)
        if not targets:
            raise ValueError("module does not contain a fully sharded wrapper")
        state_dict_config_types = {
            StateDictType.FULL_STATE_DICT: FullStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
        }
        optim_state_dict_config_types = {
            StateDictType.FULL_STATE_DICT: FullOptimStateDictConfig,
            StateDictType.LOCAL_STATE_DICT: LocalOptimStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedOptimStateDictConfig,
        }
        state_dict_config_type = state_dict_config_types[state_dict_type]
        optim_state_dict_config_type = optim_state_dict_config_types[state_dict_type]
        if state_dict_config is None:
            state_dict_config = state_dict_config_type()
        if optim_state_dict_config is None:
            optim_state_dict_config = optim_state_dict_config_type()
        if type(state_dict_config) is not state_dict_config_type:
            raise RuntimeError(
                f"Expected state_dict_config of type {state_dict_config_type} "
                f"but got {type(state_dict_config)}"
            )
        if type(optim_state_dict_config) is not optim_state_dict_config_type:
            raise RuntimeError(
                f"Expected optim_state_dict_config of type {optim_state_dict_config_type} "
                f"but got {type(optim_state_dict_config)}"
            )
        previous: StateDictSettings | None = None
        for item in targets:
            candidates = (item.module,) if isinstance(item, FullyShardedDataParallel) else ()
            for candidate in (item, *candidates):
                current = StateDictSettings(
                    candidate._state_dict_type,
                    candidate._state_dict_config,
                    candidate._optim_state_dict_config,
                )
                if previous is None:
                    previous = current
                else:
                    if previous.state_dict_type != current.state_dict_type:
                        raise AssertionError(
                            "All FSDP modules should have the same state_dict_type."
                        )
                    if not isinstance(
                        current.state_dict_config, type(previous.state_dict_config)
                    ):
                        raise AssertionError(
                            "All FSDP modules must have the same type of state_dict_config."
                        )
                    if not isinstance(
                        current.optim_state_dict_config,
                        type(previous.optim_state_dict_config),
                    ):
                        raise AssertionError(
                            "All FSDP modules must have the same type of optim_state_dict_config."
                        )
        for item in targets:
            candidates = (item.module,) if isinstance(item, FullyShardedDataParallel) else ()
            for candidate in (item, *candidates):
                candidate._state_dict_type = state_dict_type
                candidate._state_dict_config = state_dict_config
                candidate._optim_state_dict_config = optim_state_dict_config
                candidate_state = getattr(candidate, "_fsdp_state", None)
                if candidate_state is not None:
                    candidate_state._state_dict_type = state_dict_type
                    candidate_state._state_dict_config = state_dict_config
                    candidate_state._optim_state_dict_config = optim_state_dict_config
        if previous is None:
            raise ValueError("module does not contain a fully sharded wrapper")
        return previous

    @staticmethod
    def get_state_dict_type(module: Module) -> StateDictSettings:
        targets = FullyShardedDataParallel.fsdp_modules(module)
        if not targets:
            raise ValueError("module does not contain a fully sharded wrapper")
        settings: StateDictSettings | None = None
        for item in targets:
            candidates = (item.module,) if isinstance(item, FullyShardedDataParallel) else ()
            for candidate in (item, *candidates):
                current = StateDictSettings(
                    candidate._state_dict_type,
                    candidate._state_dict_config,
                    candidate._optim_state_dict_config,
                )
                if settings is None:
                    settings = current
                elif settings != current:
                    raise AssertionError(
                        "All FSDP modules must have the same state dict settings."
                        f"Got {current} and {settings}."
                    )
        if settings is None:
            raise ValueError("module does not contain a fully sharded wrapper")
        return settings

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(module: Module, state_dict_type: StateDictType, state_dict_config: StateDictConfig | None = None, optim_state_dict_config: OptimStateDictConfig | None = None):
        previous = FullyShardedDataParallel.set_state_dict_type(module, state_dict_type, state_dict_config, optim_state_dict_config)
        try:
            yield
        finally:
            FullyShardedDataParallel.set_state_dict_type(module, previous.state_dict_type, previous.state_dict_config, previous.optim_state_dict_config)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return super().load_state_dict(state_dict, *args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any):
        state = getattr(self, "_fsdp_state", None)
        clean_names = bool(getattr(state, "_summoning_full_params", False))
        for name, param in super().named_parameters(*args, **kwargs):
            if clean_names:
                name = name.replace("module.", "")
            yield name, param

    def named_buffers(self, *args: Any, **kwargs: Any):
        state = getattr(self, "_fsdp_state", None)
        clean_names = bool(getattr(state, "_summoning_full_params", False))
        for name, buffer in super().named_buffers(*args, **kwargs):
            if clean_names:
                name = name.replace("module.", "")
            yield name, buffer

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(module: Module, recurse: bool = True, writeback: bool = True, rank0_only: bool = False, offload_to_cpu: bool = False, with_grads: bool = False):
        if rank0_only and writeback:
            raise ValueError("rank0_only cannot be combined with writeback")
        if with_grads and offload_to_cpu:
            raise ValueError("with_grads cannot be combined with offload_to_cpu")
        wrappers = FullyShardedDataParallel.fsdp_modules(module) if recurse else [module]
        targets: list[FSDPModule] = []
        state_ids: set[int] = set()
        for wrapper in wrappers:
            target = wrapper.module if isinstance(wrapper, FullyShardedDataParallel) else wrapper
            if not isinstance(target, FSDPModule):
                continue
            state = target._get_fsdp_state()
            if id(state) not in state_ids:
                state_ids.add(id(state))
                targets.append(target)
        snapshots: list[tuple[Any, Any, Any]] = []
        nonzero_params: set[int] = set()
        nonzero_targets: set[int] = set()
        state_flags = [
            (
                target._get_fsdp_state(),
                bool(getattr(target._get_fsdp_state(), "_summoning_full_params", False)),
                getattr(
                    target._get_fsdp_state(),
                    "_training_state",
                    TrainingState.IDLE,
                ),
            )
            for target in targets
        ]
        for state, _, _ in state_flags:
            state._summoning_full_params = True
            state._training_state = TrainingState.SUMMON_FULL_PARAMS
        try:
            for target in targets:
                state = target._get_fsdp_state()
                target_rank = int(getattr(state, "rank", _global_rank()))
                nonzero_target = rank0_only and target_rank != 0
                for group in state._all_param_groups():
                    for param in group.params:
                        if nonzero_target:
                            nonzero_params.add(id(param))
                        local = param._sharded_local_tensor()
                        snapshots.append(
                            (param, local.detach().clone(), getattr(local, "device", None))
                        )
                target.unshard()
                if nonzero_target:
                    nonzero_targets.add(id(target))
                    target.reshard()
            for param, _, _ in snapshots:
                if not offload_to_cpu or id(param) in nonzero_params:
                    continue
                full = param._full_tensor
                if getattr(full, "device", None) is not None and str(full.device) != "cpu":
                    param._full_tensor = full.to("cpu")
                    param._unsharded_param = param._full_tensor
                    param._setattr_on_modules(
                        Parameter(
                            param._full_tensor,
                            requires_grad=param.param.requires_grad,
                        )
                    )
            if with_grads:
                _materialize_summoned_grads(
                    snapshot
                    for snapshot in snapshots
                    if id(snapshot[0]) not in nonzero_params
                )
            yield
        finally:
            try:
                for param, local, device in snapshots:
                    if id(param) in nonzero_params:
                        continue
                    if not writeback:
                        sharded = param._sharded_tensor
                        if sharded is not None:
                            sharded_local = sharded.to_local()
                            if device is not None and getattr(local, "device", None) != device:
                                local = local.to(device)
                            with tp.no_grad():
                                sharded_local.copy_(local)
                        param._state = ShardedState.SHARDED
                    elif offload_to_cpu and device is not None:
                        full = param._full_tensor
                        if getattr(full, "device", None) != device:
                            param._full_tensor = full.to(device)
                            param._unsharded_param = param._full_tensor
                            param._setattr_on_modules(
                                Parameter(
                                    param._full_tensor,
                                    requires_grad=param.param.requires_grad,
                                )
                            )
                for target in reversed(targets):
                    if id(target) not in nonzero_targets:
                        target.reshard()
            finally:
                for state, previous, previous_training_state in state_flags:
                    state._summoning_full_params = previous
                    state._training_state = previous_training_state

    def _deregister_orig_params_ctx(self):
        if not self.use_orig_params:
            return contextlib.nullcontext()
        return self.summon_full_params(self, recurse=True, writeback=True)

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self.summon_full_params(self):
            return super()._apply(fn, *args, **kwargs)

    def no_sync(self):
        @contextlib.contextmanager
        def context():
            state = self.module._get_fsdp_state()
            if getattr(state, "_is_root", None) is None:
                state._lazy_init()
            if not getattr(state, "_is_root", False):
                raise RuntimeError(
                    "no_sync must be called on the root fully sharded module"
                )
            self._assert_state(TrainingState.IDLE)
            previous = self._no_sync
            self._no_sync = True
            state_snapshots = []
            group_snapshots = []
            states_seen = set()
            groups_seen = set()
            for candidate in self.module.modules():
                state = getattr(candidate, "_fsdp_state", None)
                if state is None or id(state) in states_seen:
                    continue
                states_seen.add(id(state))
                state_snapshots.append(
                    (
                        state,
                        state._requires_gradient_sync,
                        state._requires_all_reduce,
                    )
                )
                state._requires_gradient_sync = False
                state._requires_all_reduce = False
                for group in state._all_param_groups():
                    if id(group) in groups_seen:
                        continue
                    groups_seen.add(id(group))
                    group_snapshots.append(
                        (
                            group,
                            group.reduce_grads,
                            group.all_reduce_grads,
                            group._requires_gradient_sync,
                            group._requires_all_reduce,
                        )
                    )
                    group.reduce_grads = False
                    group.all_reduce_grads = False
                    group._requires_gradient_sync = False
                    group._requires_all_reduce = False
            try:
                yield
            finally:
                self._no_sync = previous
                for (
                    state,
                    previous_sync,
                    previous_all_reduce,
                ) in state_snapshots:
                    state._requires_gradient_sync = previous_sync
                    state._requires_all_reduce = previous_all_reduce
                for (
                    group,
                    previous_reduce,
                    previous_all_reduce,
                    previous_sync,
                    previous_group_all_reduce,
                ) in group_snapshots:
                    group.reduce_grads = previous_reduce
                    group.all_reduce_grads = previous_all_reduce
                    group._requires_gradient_sync = previous_sync
                    group._requires_all_reduce = previous_group_all_reduce
        return context()

    @tp.no_grad()
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> Any:
        state = self.module._get_fsdp_state()
        if state is None:
            raise RuntimeError("clip_grad_norm_ requires a sharded module")
        if getattr(state, "_is_root", None) is None:
            state._lazy_init()
        if not getattr(state, "_is_root", False):
            raise RuntimeError(
                "clip_grad_norm_ should only be called on the root fully sharded module"
            )
        self._assert_state(TrainingState.IDLE)
        try:
            norm_type = float(norm_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("norm_type must be a positive number") from exc
        if norm_type <= 0 and norm_type != math.inf:
            raise ValueError("norm_type must be positive")
        max_norm = float(max_norm)
        if max_norm < 0:
            raise ValueError("max_norm must be non-negative")

        groups: list[Any] = []
        states_seen: set[int] = set()
        groups_seen: set[int] = set()
        for candidate in self.module.modules():
            candidate_state = getattr(candidate, "_fsdp_state", None)
            if candidate_state is None or id(candidate_state) in states_seen:
                continue
            states_seen.add(id(candidate_state))
            for group in candidate_state._all_param_groups():
                if id(group) in groups_seen:
                    continue
                groups_seen.add(id(group))
                groups.append(group)

        device = getattr(state, "compute_device", None)
        if device is None:
            device = next(
                (
                    getattr(param, "device", None)
                    for param in self.module.parameters()
                    if getattr(param, "device", None) is not None
                ),
                "cpu",
            )
        zero = tp.tensor(0.0, device=device, dtype=tp.float32)
        sharded_params: list[Any] = []
        nonsharded_params: list[Any] = []
        sharded_param_ids: set[int] = set()
        nonsharded_param_ids: set[int] = set()
        sharded_norms: list[Any] = []
        sharded_norm_groups: list[Any] = []
        grads: list[Any] = []
        for group in groups:
            reduce_group = group._reduce_scatter_process_group()
            try:
                group_world_size = (
                    dist.get_world_size(reduce_group)
                    if reduce_group is not None
                    else 1
                )
            except (RuntimeError, ValueError):
                group_world_size = 1
            target = sharded_params if group_world_size > 1 else nonsharded_params
            target_ids = (
                sharded_param_ids if group_world_size > 1 else nonsharded_param_ids
            )
            for fsdp_param in group.params:
                param = fsdp_param._sharded_local_tensor()
                if id(param) in target_ids:
                    continue
                target_ids.add(id(param))
                target.append(param)
                grad = getattr(param, "grad", None)
                if grad is not None:
                    grads.append(grad)
            if group_world_size > 1:
                sharded_norms.append(
                    _get_grad_norm(target, norm_type, zero, device)
                )
                sharded_norm_groups.append(reduce_group)

        for param in self.parameters():
            param_id = id(param)
            if param_id in sharded_param_ids or param_id in nonsharded_param_ids:
                continue
            nonsharded_param_ids.add(param_id)
            nonsharded_params.append(param)
            grad = getattr(param, "grad", None)
            if grad is not None:
                grads.append(grad)

        if norm_type == math.inf:
            total_norm = zero
            for local_norm, reduce_group in zip(
                sharded_norms, sharded_norm_groups
            ):
                if reduce_group is not None:
                    dist.all_reduce(
                        local_norm, op=dist.ReduceOp.MAX, group=reduce_group
                    )
                total_norm = tp.maximum(total_norm, local_norm)
            local_nonsharded_norm = _get_grad_norm(
                nonsharded_params, norm_type, zero, device
            )
            total_norm = tp.maximum(total_norm, local_nonsharded_norm)
        else:
            total_power = zero
            for local_norm, reduce_group in zip(
                sharded_norms, sharded_norm_groups
            ):
                local_power = local_norm ** norm_type
                if reduce_group is not None:
                    dist.all_reduce(
                        local_power, op=dist.ReduceOp.SUM, group=reduce_group
                    )
                total_power = total_power + local_power
            local_nonsharded_norm = _get_grad_norm(
                nonsharded_params, norm_type, zero, device
            )
            total_norm = (total_power + local_nonsharded_norm ** norm_type) ** (
                1.0 / norm_type
            )

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = tp.clamp(clip_coef, max=1.0)
        for grad in grads:
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
        if not grads:
            return total_norm
        total_dtype = grads[0].dtype
        for grad in grads[1:]:
            total_dtype = tp.promote_types(total_dtype, grad.dtype)
        return total_norm.to(dtype=total_dtype)

    def register_comm_hook(self, state: Any, hook: Any) -> None:
        if not self.check_is_root():
            raise AssertionError("register_comm_hook must be called on a root module")
        if not callable(hook):
            raise ValueError(f"the communication hook must be callable: {hook!r}")
        if self._comm_hook is not None:
            raise AssertionError("a communication hook is already registered")
        states: list[Any] = []
        seen: set[int] = set()
        for candidate in self.module.modules():
            fsdp_state = getattr(candidate, "_fsdp_state", None)
            if fsdp_state is None or id(fsdp_state) in seen:
                continue
            seen.add(id(fsdp_state))
            states.append(fsdp_state)
        for fsdp_state in states:
            if getattr(fsdp_state, "_comm_hook", None) is not None:
                raise AssertionError("a communication hook is already registered")
            for group in fsdp_state._all_param_groups():
                if group._is_hsdp():
                    raise AssertionError(
                        "communication hooks are not supported for hybrid sharding"
                    )
        for fsdp_state in states:
            fsdp_state._comm_hook = hook
            fsdp_state._comm_hook_state = state
            for group in fsdp_state._all_param_groups():
                group._comm_hook = hook
                group._comm_hook_state = state
        self._comm_hook = (state, hook)

    def _unshard(self, async_op: bool = False) -> Any:
        class UnshardHandle:
            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def wait(self) -> None:
                if self._handle is not None:
                    waiter = getattr(self._handle, "wait", None)
                    if callable(waiter):
                        waiter()
                    self._handle = None

        result = self.module.unshard(async_op=bool(async_op))
        if async_op:
            return UnshardHandle(result)
        if result is not None:
            UnshardHandle(result).wait()
        return None

    def _wait_unshard_streams_on_current_stream(self) -> None:
        state = self.module._get_fsdp_state()
        for group in state._all_param_groups():
            group.wait_for_unshard()

    @contextlib.contextmanager
    def _use_training_state(self, state: TrainingState, handle_training_state: Any = None):
        fsdp_state = self.module._get_fsdp_state()
        previous = fsdp_state._training_state
        fsdp_state._training_state = state
        handle = getattr(fsdp_state, "_handle", None)
        if handle is not None:
            previous_handle_state = handle._training_state
            handle._training_state = handle_training_state
        try:
            yield
        finally:
            fsdp_state._training_state = previous
            if handle is not None:
                handle._training_state = previous_handle_state

    def full_optim_state_dict(self, optim: Any, optim_input: Any = None, rank0_only: bool = True, group: Any = None) -> dict[str, Any]:
        config = self._optim_state_dict_config
        return self._optim_state_dict_impl(
            self,
            optim,
            optim.state_dict(),
            optim_input,
            rank0_only,
            True,
            group,
            bool(getattr(config, "offload_to_cpu", True)),
        )

    def sharded_optim_state_dict(self, optim: Any, group: Any = None) -> dict[str, Any]:
        config = self._optim_state_dict_config
        return self._optim_state_dict_impl(
            self,
            optim,
            optim.state_dict(),
            None,
            False,
            False,
            group,
            bool(getattr(config, "offload_to_cpu", False)),
        )

    @staticmethod
    def shard_full_optim_state_dict(full_optim_state_dict: dict[str, Any], model: Module, optim_input: Any = None, optim: Any = None) -> dict[str, Any]:
        sharded = _optim_state_dict(
            model,
            optim,
            full_optim_state_dict,
            optim_input,
            False,
            True,
            None,
            optim_input is not None,
            bool(getattr(model, "use_orig_params", False)),
            False,
        )
        return _rekey_sharded_optim_state_dict(
            sharded,
            model,
            optim,
            optim_input,
            optim_input is not None,
            False,
        )

    @staticmethod
    def flatten_sharded_optim_state_dict(sharded_optim_state_dict: dict[str, Any], model: Module, optim: Any) -> dict[str, Any]:
        return _rekey_sharded_optim_state_dict(
            sharded_optim_state_dict,
            model,
            optim,
            None,
            False,
            False,
        )

    @staticmethod
    def scatter_full_optim_state_dict(full_optim_state_dict: dict[str, Any] | None, model: Module, optim_input: Any = None, optim: Any = None, group: Any = None) -> dict[str, Any]:
        if full_optim_state_dict is None:
            return {}
        sharded = _optim_state_dict(
            model,
            optim,
            full_optim_state_dict,
            optim_input,
            False,
            True,
            group,
            optim_input is not None,
            bool(getattr(model, "use_orig_params", False)),
            False,
        )
        return _rekey_sharded_optim_state_dict(
            sharded,
            model,
            optim,
            optim_input,
            optim_input is not None,
            False,
        )

    @staticmethod
    def rekey_optim_state_dict(optim_state_dict: dict[str, Any], optim_state_key_type: OptimStateKeyType, model: Module, optim_input: Any = None, optim: Any = None) -> dict[str, Any]:
        if optim_state_key_type not in (
            OptimStateKeyType.PARAM_NAME,
            OptimStateKeyType.PARAM_ID,
        ):
            raise ValueError("optim_state_key_type must identify names or ids")
        if not isinstance(optim_state_dict, dict) or "state" not in optim_state_dict:
            raise TypeError("optim_state_dict must contain a state mapping")

        state = optim_state_dict["state"]
        key_types = {type(key) for key in state}
        if key_types and not key_types.issubset({str, int}):
            raise ValueError(f"invalid optimizer parameter keys: {tuple(state)}")
        if len(key_types) > 1:
            raise ValueError(f"invalid optimizer parameter keys: {tuple(state)}")
        source_type = next(iter(key_types), None)
        target_type = str if optim_state_key_type == OptimStateKeyType.PARAM_NAME else int
        if source_type is None or source_type is target_type:
            return optim_state_dict

        names_by_identity = {
            id(param): name for name, param in model.named_parameters()
        }
        param_groups = optim_state_dict.get("param_groups", [])
        id_to_name, name_to_id = _optimizer_parameter_maps(
            model,
            optim,
            optim_input,
            param_groups,
            names_by_identity,
        )
        result = copy.deepcopy(optim_state_dict)
        if optim_state_key_type == OptimStateKeyType.PARAM_NAME:
            result["state"] = {
                id_to_name[key]: value for key, value in state.items()
            }
            for group in result.get("param_groups", []):
                group["params"] = sorted(id_to_name[key] for key in group["params"])
        else:
            result["state"] = {
                name_to_id[key]: value for key, value in state.items()
            }
            for group in result.get("param_groups", []):
                group["params"] = sorted(name_to_id[key] for key in group["params"])
        return result

    @staticmethod
    def optim_state_dict(model: Module, optim: Any, optim_state_dict: dict[str, Any] | None = None, group: Any = None) -> dict[str, Any]:
        wrappers = FullyShardedDataParallel.fsdp_modules(model)
        state_type = wrappers[0]._state_dict_type if wrappers else StateDictType.FULL_STATE_DICT
        source = optim_state_dict if optim_state_dict is not None else optim.state_dict()
        if state_type == StateDictType.FULL_STATE_DICT:
            return FullyShardedDataParallel._optim_state_dict_impl(
                model, optim, source, None, True, True, group, True
            )
        return FullyShardedDataParallel._optim_state_dict_impl(
            model, optim, source, None, False, False, group, False
        )

    @staticmethod
    def optim_state_dict_to_load(model: Module, optim: Any, optim_state_dict: dict[str, Any], is_named_optimizer: bool = False, load_directly: bool = False, group: Any = None) -> dict[str, Any]:
        wrappers = FullyShardedDataParallel.fsdp_modules(model)
        state_type = wrappers[0]._state_dict_type if wrappers else StateDictType.FULL_STATE_DICT
        result = FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict,
            model,
            None,
            optim,
            state_type == StateDictType.FULL_STATE_DICT,
            False,
            is_named_optimizer,
            group,
        )
        if load_directly:
            optim.load_state_dict(result)
        return result


def _optimizer_parameter_maps(
    model: Module,
    optim: Any,
    optim_input: Any,
    saved_groups: Any,
    names_by_identity: dict[int, str],
) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}

    if optim is not None:
        canonical_groups = optim.state_dict().get("param_groups", [])
        for physical_group, canonical_group in zip(
            getattr(optim, "param_groups", ()), canonical_groups
        ):
            for param, param_id in zip(
                physical_group.get("params", ()), canonical_group.get("params", ())
            ):
                name = names_by_identity.get(id(param))
                if name is not None:
                    id_to_name[int(param_id)] = name
                    name_to_id[name] = int(param_id)
    else:
        values: list[Any] = []
        if optim_input is not None:
            source = list(optim_input)
            if source and isinstance(source[0], dict):
                for group in source:
                    values.extend(group.get("params", ()))
            else:
                values = source
        if not values:
            values = [param for _, param in model.named_parameters()]

        saved_ids = [
            param_id
            for group in saved_groups
            for param_id in group.get("params", ())
        ]
        if not saved_ids:
            saved_ids = list(range(len(values)))
        for param_id, value in zip(saved_ids, values):
            name = value if isinstance(value, str) else names_by_identity.get(id(value))
            if name is not None:
                id_to_name[int(param_id)] = name
                name_to_id[name] = int(param_id)

    if not id_to_name:
        names = [name for name, _ in model.named_parameters()]
        saved_ids = [
            param_id
            for group in saved_groups
            for param_id in group.get("params", ())
        ]
        for param_id, name in zip(saved_ids or range(len(names)), names):
            id_to_name[int(param_id)] = name
            name_to_id[name] = int(param_id)
    return id_to_name, name_to_id


def _default_state_dict_config(state_dict_type: StateDictType) -> StateDictConfig:
    return {
        StateDictType.FULL_STATE_DICT: FullStateDictConfig(),
        StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig(),
        StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig(),
    }[state_dict_type]


def _default_optim_state_dict_config(state_dict_type: StateDictType) -> OptimStateDictConfig:
    return {
        StateDictType.FULL_STATE_DICT: FullOptimStateDictConfig(),
        StateDictType.LOCAL_STATE_DICT: LocalOptimStateDictConfig(),
        StateDictType.SHARDED_STATE_DICT: ShardedOptimStateDictConfig(),
    }[state_dict_type]


def _rank_is_zero() -> bool:
    try:
        from .. import distributed_core as dist
        return dist.get_rank() == 0
    except Exception:
        return True


def _get_grad_norm(
    parameters: Iterable[Any],
    norm_type: float,
    zero: Any,
    device: Any,
) -> Any:
    values = [
        param.grad for param in parameters if getattr(param, "grad", None) is not None
    ]
    if not values:
        return zero
    norms = [
        tp.linalg.vector_norm(value.detach(), norm_type, dtype=tp.float32)
        for value in values
    ]
    result = tp.linalg.vector_norm(tp.stack(norms), norm_type, dtype=tp.float32)
    return result.to(device=device)


def _get_param_to_fqn(model: Module) -> dict[Any, str]:
    return {param: name for name, param in model.named_parameters()}


def _get_fqn_to_param(model: Module) -> dict[str, Any]:
    return {name: param for name, param in model.named_parameters()}
