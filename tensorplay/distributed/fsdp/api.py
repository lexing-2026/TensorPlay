"""Configuration objects for fully sharded module execution."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

__all__ = [
    "ShardingStrategy",
    "BackwardPrefetch",
    "MixedPrecision",
    "CPUOffload",
    "StateDictType",
    "StateDictConfig",
    "FullStateDictConfig",
    "LocalStateDictConfig",
    "ShardedStateDictConfig",
    "OptimStateDictConfig",
    "FullOptimStateDictConfig",
    "LocalOptimStateDictConfig",
    "ShardedOptimStateDictConfig",
    "StateDictSettings",
]


class ShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


class BackwardPrefetch(Enum):
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


@dataclass
class MixedPrecision:
    param_dtype: Any = None
    reduce_dtype: Any = None
    buffer_dtype: Any = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True
    _module_classes_to_ignore: tuple[type, ...] = ()


@dataclass
class CPUOffload:
    offload_params: bool = False


class StateDictType(Enum):
    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()


@dataclass
class StateDictConfig:
    offload_to_cpu: bool = False


@dataclass
class FullStateDictConfig(StateDictConfig):
    rank0_only: bool = False


@dataclass
class LocalStateDictConfig(StateDictConfig):
    pass


@dataclass
class ShardedStateDictConfig(StateDictConfig):
    _use_dtensor: bool = False


@dataclass
class OptimStateDictConfig:
    offload_to_cpu: bool = True


@dataclass
class FullOptimStateDictConfig(OptimStateDictConfig):
    rank0_only: bool = False


@dataclass
class LocalOptimStateDictConfig(OptimStateDictConfig):
    offload_to_cpu: bool = False


@dataclass
class ShardedOptimStateDictConfig(OptimStateDictConfig):
    _use_dtensor: bool = False


@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
