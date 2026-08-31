"""Module wrapper for fully sharded data parallel execution."""

import contextlib
import copy
import math
from enum import Enum, auto
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.nn.modules.module import Module

from ._common_utils import TrainingState
from ._fully_shard import FSDPModule, fully_shard
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
        del process_group, auto_wrap_policy, backward_prefetch, param_init_fn
        del device_id, sync_module_states, forward_prefetch, limit_all_gathers
        del ignored_modules, ignored_states
        super().__init__()
        self.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
        self.cpu_offload = cpu_offload or CPUOffload()
        self.mixed_precision = mixed_precision or MixedPrecision()
        self.use_orig_params = bool(use_orig_params)
        self._state_dict_type = StateDictType.FULL_STATE_DICT
        self._state_dict_config: StateDictConfig = FullStateDictConfig()
        self._optim_state_dict_config: OptimStateDictConfig = FullOptimStateDictConfig()
        self._comm_hook = None
        self._no_sync = False
        self.module = fully_shard(
            module,
            mesh=device_mesh,
            reshard_after_forward=False,
        )

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
        return True

    @staticmethod
    def fsdp_modules(module: Module, root_only: bool = False) -> list[Any]:
        result = [item for item in module.modules() if isinstance(item, FullyShardedDataParallel) or isinstance(item, FSDPModule)]
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
        target._state_dict_type = state_dict_type
        target._state_dict_config = state_dict_config or _default_state_dict_config(state_dict_type)
        target._optim_state_dict_config = optim_state_dict_config or _default_optim_state_dict_config(state_dict_type)
        return previous

    @staticmethod
    def get_state_dict_type(module: Module) -> StateDictSettings:
        target = FullyShardedDataParallel.fsdp_modules(module)[0]
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
        with self.summon_full_params(self):
            state = self.module.state_dict(*args, **kwargs)
        if getattr(self._state_dict_config, "offload_to_cpu", False):
            state = {key: value.cpu() if isinstance(value, tp.Tensor) else value for key, value in state.items()}
        return state

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        with self.summon_full_params(self):
            result = self.module.load_state_dict(state_dict, *args, **kwargs)
        self.module.reshard()
        return result

    def named_parameters(self, *args: Any, **kwargs: Any):
        self.module.reshard()
        yield from super().named_parameters(*args, **kwargs)

    def named_buffers(self, *args: Any, **kwargs: Any):
        yield from super().named_buffers(*args, **kwargs)

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(module: Module, recurse: bool = True, writeback: bool = True, rank0_only: bool = False, offload_to_cpu: bool = False, with_grads: bool = False):
        del writeback, rank0_only, with_grads
        wrappers = FullyShardedDataParallel.fsdp_modules(module) if recurse else [module]
        handles = []
        for wrapper in wrappers:
            target = wrapper.module if isinstance(wrapper, FullyShardedDataParallel) else wrapper
            if isinstance(target, FSDPModule):
                handles.append(target.unshard())
        try:
            yield
        finally:
            for wrapper in wrappers:
                target = wrapper.module if isinstance(wrapper, FullyShardedDataParallel) else wrapper
                if isinstance(target, FSDPModule):
                    target.reshard()
            for handle in handles:
                handle.wait()

    def _deregister_orig_params_ctx(self):
        return contextlib.nullcontext()

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self.summon_full_params(self):
            return super()._apply(fn, *args, **kwargs)

    def no_sync(self):
        @contextlib.contextmanager
        def context():
            previous = self._no_sync
            self._no_sync = True
            state = getattr(self.module, "_fsdp_state", None)
            if state is not None:
                state._requires_gradient_sync = False
            try:
                yield
            finally:
                self._no_sync = previous
                if state is not None:
                    state._requires_gradient_sync = not previous
        return context()

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> float:
        grads = [param.grad for param in self.module.parameters() if getattr(param, "grad", None) is not None]
        if not grads:
            return 0.0
        if norm_type == math.inf:
            total = max(float(grad.abs().max().item()) for grad in grads)
        else:
            total = sum(float((grad.abs() ** norm_type).sum().item()) for grad in grads) ** (1.0 / norm_type)
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
        return None

    def _use_training_state(self, state: TrainingState, handle_training_state: Any = None):
        del handle_training_state
        current = self.module._get_fsdp_state()._training_state
        self.module._get_fsdp_state()._training_state = state
        return current

    def full_optim_state_dict(self, optim: Any, optim_input: Any = None, rank0_only: bool = True, group: Any = None) -> dict[str, Any]:
        del optim_input, group
        state = copy.deepcopy(optim.state_dict())
        return state if not rank0_only or _rank_is_zero() else {}

    def sharded_optim_state_dict(self, optim: Any, group: Any = None) -> dict[str, Any]:
        del group
        return copy.deepcopy(optim.state_dict())

    @staticmethod
    def shard_full_optim_state_dict(full_optim_state_dict: dict[str, Any], model: Module, optim_input: Any = None, optim: Any = None) -> dict[str, Any]:
        del model, optim_input, optim
        return full_optim_state_dict

    @staticmethod
    def flatten_sharded_optim_state_dict(sharded_optim_state_dict: dict[str, Any], model: Module, optim: Any) -> dict[str, Any]:
        del model, optim
        return sharded_optim_state_dict

    @staticmethod
    def scatter_full_optim_state_dict(full_optim_state_dict: dict[str, Any] | None, model: Module, optim_input: Any = None, optim: Any = None, group: Any = None) -> dict[str, Any]:
        del model, optim_input, optim, group
        return full_optim_state_dict or {}

    @staticmethod
    def rekey_optim_state_dict(optim_state_dict: dict[str, Any], optim_state_key_type: OptimStateKeyType, model: Module, optim_input: Any = None, optim: Any = None) -> dict[str, Any]:
        del optim_state_key_type, model, optim_input, optim
        return optim_state_dict

    @staticmethod
    def optim_state_dict(model: Module, optim: Any, optim_state_dict: dict[str, Any] | None = None, group: Any = None) -> dict[str, Any]:
        del group
        return copy.deepcopy(optim_state_dict if optim_state_dict is not None else optim.state_dict())

    @staticmethod
    def optim_state_dict_to_load(model: Module, optim: Any, optim_state_dict: dict[str, Any], is_named_optimizer: bool = False, load_directly: bool = False, group: Any = None) -> dict[str, Any]:
        del model, is_named_optimizer, load_directly, group
        return optim_state_dict


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
