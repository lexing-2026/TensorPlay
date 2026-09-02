"""Module wrapper for fully sharded data parallel execution."""

import contextlib
import copy
import math
from enum import Enum, auto
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.nn.modules.module import Module
from tensorplay.nn.parameter import Parameter

from .. import distributed_core as dist
from ._common_utils import TrainingState
from ._fully_shard import FSDPModule, fully_shard
from ._fully_shard._fsdp_api import CPUOffloadPolicy, DataParallelMeshDims, MixedPrecisionPolicy, OffloadPolicy
from ._fully_shard._fsdp_init import _init_default_mesh
from ._fully_shard._fsdp_param import ShardedState
from ._optim_utils import _optim_state_dict, _rekey_sharded_optim_state_dict
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
        state.process_group = process_group
        state.device_mesh = mesh
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
        state._param_group._reshard_after_forward_enabled = self.sharding_strategy == ShardingStrategy.FULL_SHARD
        state._param_group._reshard_after_backward_enabled = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("module")
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
        return bool(state is not None and getattr(state, "_is_root", True))

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
            state_ids.add(state_id)
            result.append(item)
        return result[:1] if root_only and result else result

    def apply(self, fn: Any) -> "FullyShardedDataParallel":
        with self.summon_full_params(self):
            super().apply(fn)
        return self

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        return self.mixed_precision.buffer_dtype is not None

    def _low_precision_hook_enabled(self) -> bool:
        return self._comm_hook is not None

    def _reset_lazy_init(self) -> None:
        state = getattr(self.module, "_fsdp_state", None)
        if state is not None:
            state._reset_iter_state()

    @staticmethod
    def set_state_dict_type(module: Module, state_dict_type: StateDictType, state_dict_config: StateDictConfig | None = None, optim_state_dict_config: OptimStateDictConfig | None = None) -> StateDictSettings:
        targets = FullyShardedDataParallel.fsdp_modules(module)
        if not targets:
            raise ValueError("module does not contain a fully sharded wrapper")
        target = targets[0]
        previous = StateDictSettings(target._state_dict_type, target._state_dict_config, target._optim_state_dict_config)
        state_dict_config = state_dict_config or _default_state_dict_config(state_dict_type)
        optim_state_dict_config = optim_state_dict_config or _default_optim_state_dict_config(state_dict_type)
        for item in targets:
            item._state_dict_type = state_dict_type
            item._state_dict_config = state_dict_config
            item._optim_state_dict_config = optim_state_dict_config
        return previous

    @staticmethod
    def get_state_dict_type(module: Module) -> StateDictSettings:
        targets = FullyShardedDataParallel.fsdp_modules(module)
        if not targets:
            raise ValueError("module does not contain a fully sharded wrapper")
        target = targets[0]
        return StateDictSettings(target._state_dict_type, target._state_dict_config, target._optim_state_dict_config)

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(module: Module, state_dict_type: StateDictType, state_dict_config: StateDictConfig | None = None, optim_state_dict_config: OptimStateDictConfig | None = None):
        previous = FullyShardedDataParallel.set_state_dict_type(module, state_dict_type, state_dict_config, optim_state_dict_config)
        try:
            yield previous
        finally:
            FullyShardedDataParallel.set_state_dict_type(module, previous.state_dict_type, previous.state_dict_config, previous.optim_state_dict_config)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        config = self._state_dict_config
        if self._state_dict_type == StateDictType.LOCAL_STATE_DICT:
            state = self.module.state_dict(*args, **kwargs)
        else:
            with self.summon_full_params(
                self,
                rank0_only=bool(getattr(config, "rank0_only", False)),
                offload_to_cpu=bool(getattr(config, "offload_to_cpu", False)),
            ):
                state = self.module.state_dict(*args, **kwargs)
            if bool(getattr(config, "rank0_only", False)) and _global_rank() != 0:
                return {}
        if getattr(self._state_dict_config, "offload_to_cpu", False):
            state = {key: value.cpu() if isinstance(value, tp.Tensor) else value for key, value in state.items()}
        return state

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if self._state_dict_type == StateDictType.LOCAL_STATE_DICT:
            result = self.module.load_state_dict(state_dict, *args, **kwargs)
        else:
            with self.summon_full_params(self):
                result = self.module.load_state_dict(state_dict, *args, **kwargs)
            self.module.reshard()
        return result

    def named_parameters(self, *args: Any, **kwargs: Any):
        yield from super().named_parameters(*args, **kwargs)

    def named_buffers(self, *args: Any, **kwargs: Any):
        yield from super().named_buffers(*args, **kwargs)

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
        if rank0_only and _global_rank() != 0:
            for target in targets:
                target.reshard()
            yield
            return
        snapshots: list[tuple[Any, Any, Any]] = []
        for target in targets:
            state = target._get_fsdp_state()
            for param in state._fsdp_param_group().params:
                local = param._sharded_local_tensor()
                snapshots.append((param, local.detach().clone(), getattr(local, "device", None)))
            target.unshard()
        for param, _, _ in snapshots:
            if not offload_to_cpu:
                continue
            full = param._full_tensor
            if getattr(full, "device", None) is not None and str(full.device) != "cpu":
                param._full_tensor = full.to("cpu")
                param._setattr_on_modules(
                    Parameter(param._full_tensor, requires_grad=param.param.requires_grad)
                )
        if with_grads:
            _materialize_summoned_grads(snapshots)
        try:
            yield
        finally:
            for param, local, device in snapshots:
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
                        param._setattr_on_modules(
                            Parameter(param._full_tensor, requires_grad=param.param.requires_grad)
                        )
            for target in reversed(targets):
                target.reshard()

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
            previous = self._no_sync
            self._no_sync = True
            state = getattr(self.module, "_fsdp_state", None)
            previous_sync = None
            if state is not None:
                previous_sync = state._requires_gradient_sync
                state._requires_gradient_sync = False
                group = state._fsdp_param_group()
                group._requires_gradient_sync = False
            try:
                yield
            finally:
                self._no_sync = previous
                if state is not None:
                    state._requires_gradient_sync = bool(previous_sync)
                    state._fsdp_param_group()._requires_gradient_sync = bool(previous_sync)
        return context()

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> float:
        grads = [param.grad for param in self.module.parameters() if getattr(param, "grad", None) is not None]
        if not grads:
            return 0.0
        state = self.module._get_fsdp_state()
        group = state._fsdp_param_group()._all_reduce_process_group()
        if norm_type == math.inf:
            total_tensor = tp.tensor(
                max(float(grad.abs().max().item()) for grad in grads),
                device=grads[0].device,
            )
            if group is not None and int(group.size()) > 1:
                dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX, group=group)
            total = float(total_tensor.item())
        else:
            total_tensor = tp.tensor(
                sum(float((grad.abs() ** norm_type).sum().item()) for grad in grads),
                device=grads[0].device,
            )
            if group is not None and int(group.size()) > 1:
                dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM, group=group)
            total = float(total_tensor.item()) ** (1.0 / norm_type)
        scale = min(1.0, float(max_norm) / (total + 1e-6))
        if scale < 1.0:
            for grad in grads:
                grad.mul_(scale)
        return total

    def register_comm_hook(self, state: Any, hook: Any) -> None:
        if not callable(hook):
            raise TypeError("communication hook must be callable")
        self._comm_hook = (state, hook)

    def _unshard(self, async_op: bool = False) -> Any:
        return self.module.unshard(async_op)

    def _wait_unshard_streams_on_current_stream(self) -> None:
        self.module._get_fsdp_state()._fsdp_param_group().wait_for_unshard()

    def _use_training_state(self, state: TrainingState, handle_training_state: Any = None):
        del handle_training_state
        current = self.module._get_fsdp_state()._training_state
        self.module._get_fsdp_state()._training_state = state
        return current

    def full_optim_state_dict(self, optim: Any, optim_input: Any = None, rank0_only: bool = True, group: Any = None) -> dict[str, Any]:
        config = self._optim_state_dict_config
        return _optim_state_dict(
            self,
            optim,
            optim.state_dict(),
            optim_input,
            rank0_only,
            False,
            group,
            optim_input is not None,
            self.use_orig_params,
            bool(getattr(config, "offload_to_cpu", True)),
        )

    def sharded_optim_state_dict(self, optim: Any, group: Any = None) -> dict[str, Any]:
        config = self._optim_state_dict_config
        return _optim_state_dict(
            self,
            optim,
            optim.state_dict(),
            None,
            False,
            True,
            group,
            False,
            self.use_orig_params,
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
        if state_type == StateDictType.LOCAL_STATE_DICT:
            converted = _optim_state_dict(
                model,
                optim,
                optim_state_dict,
                None,
                False,
                True,
                group,
                False,
                bool(getattr(wrappers[0], "use_orig_params", False)) if wrappers else False,
                False,
            )
            return _rekey_sharded_optim_state_dict(converted, model, optim, None, False, False)
        if state_type == StateDictType.SHARDED_STATE_DICT:
            return _optim_state_dict(
                model,
                optim,
                optim_state_dict,
                None,
                False,
                True,
                group,
                False,
                bool(getattr(wrappers[0], "use_orig_params", False)) if wrappers else False,
                False,
            )
        return _optim_state_dict(
            model,
            optim,
            optim_state_dict,
            None,
            False,
            False,
            group,
            False,
            bool(getattr(wrappers[0], "use_orig_params", False)) if wrappers else False,
            False,
        )

    @staticmethod
    def optim_state_dict_to_load(model: Module, optim: Any, optim_state_dict: dict[str, Any], is_named_optimizer: bool = False, load_directly: bool = False, group: Any = None) -> dict[str, Any]:
        if load_directly:
            return copy.deepcopy(optim_state_dict)
        if is_named_optimizer:
            return copy.deepcopy(optim_state_dict)
        return FullyShardedDataParallel.rekey_optim_state_dict(
            optim_state_dict,
            OptimStateKeyType.PARAM_ID,
            model,
            optim=optim,
        )


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


def _get_grad_norm(parameters: Iterable[Any], norm_type: float = 2.0) -> float:
    values = [param.grad for param in parameters if getattr(param, "grad", None) is not None]
    if not values:
        return 0.0
    return sum(float((value.abs() ** norm_type).sum().item()) for value in values) ** (1.0 / norm_type)


def _get_param_to_fqn(model: Module) -> dict[Any, str]:
    return {param: name for name, param in model.named_parameters()}


def _get_fqn_to_param(model: Module) -> dict[str, Any]:
    return {name: param for name, param in model.named_parameters()}
