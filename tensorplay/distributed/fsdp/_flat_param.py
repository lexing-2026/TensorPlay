"""Flat parameter storage and shard metadata."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.nn.parameter import Parameter

__all__ = [
    "HandleShardingStrategy",
    "ParamInfo",
    "SharedParamInfo",
    "FlatParamShardMetadata",
    "FlatParameter",
    "FlatParamHandle",
]


class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()


@dataclass
class ParamInfo:
    name: str
    module: Any
    module_name: str = ""


@dataclass
class SharedParamInfo:
    param_name: str
    module_name: str


@dataclass
class FlatParamShardMetadata:
    start: int
    end: int
    numel: int
    padding: int = 0


@dataclass
class _ShardParamInfo:
    in_shard_start: int
    in_shard_end: int
    shard_start: int
    shard_end: int


@dataclass
class _FlatParameterMeta:
    param_infos: list[ParamInfo] = field(default_factory=list)
    numels: list[int] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    fqns: list[str] = field(default_factory=list)


class FlatParameter(Parameter):
    """A parameter containing a concatenated collection of parameters."""

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)
        self._param_metadata = _FlatParameterMeta()
        self._is_sharded = False

    def _init_metadata(
        self,
        param_infos: Iterable[ParamInfo],
        numels: Iterable[int],
        shapes: Iterable[Any],
        strides: Iterable[Any] | None = None,
        contiguities: Iterable[bool] | None = None,
        fqns: Iterable[str] | None = None,
        shared_param_infos: Iterable[SharedParamInfo] | None = None,
        param_extensions: Iterable[Any] | None = None,
        params: Iterable[Any] | None = None,
        shared_params: Iterable[Any] | None = None,
        is_padding_mask: Any = None,
    ) -> None:
        del strides, contiguities, param_extensions, params, shared_params
        del is_padding_mask
        self._param_metadata = _FlatParameterMeta(
            list(param_infos),
            list(numels),
            [tuple(shape) for shape in shapes],
            list(fqns or ()),
        )
        self._shared_param_infos = list(shared_param_infos or ())


class FlatParamHandle:
    """Manages flattening, sharding, and restoration for one module."""

    def __init__(
        self,
        params: Iterable[Any],
        fully_sharded_module: Any,
        device: Any = None,
        sharding_strategy: Any = HandleShardingStrategy.FULL_SHARD,
        offload_params: bool = False,
        mp_param_dtype: Any = None,
        mp_reduce_dtype: Any = None,
        keep_low_precision_grads: bool = False,
        process_group: Any = None,
        use_orig_params: bool = False,
    ) -> None:
        del device, offload_params, mp_param_dtype, mp_reduce_dtype
        del keep_low_precision_grads, process_group, use_orig_params
        self.params = list(params)
        self.module = fully_sharded_module
        self.sharding_strategy = sharding_strategy
        self.flat_param = self.flatten_tensors_into_flat_param(self.params)
        self._shard_metadata: FlatParamShardMetadata | None = None
        self._unsharded_flat_param = self.flat_param

    @staticmethod
    def flatten_tensors(tensors: Iterable[Any], aligned_numel: int = 1) -> Any:
        values = list(tensors)
        if not values:
            return tp.empty(0)
        flattened = [value.detach().reshape(-1) for value in values]
        result = tp.cat(flattened, dim=0)
        if aligned_numel > 1:
            padding = (-int(result.numel())) % int(aligned_numel)
            if padding:
                result = tp.cat((result, result.new_zeros(padding)), dim=0)
        return result

    def flatten_tensors_into_flat_param(
        self, tensors: Iterable[Any], aligned_numel: int = 1,
        requires_grad: bool | None = None,
    ) -> FlatParameter:
        values = list(tensors)
        data = self.flatten_tensors(values, aligned_numel)
        flat = FlatParameter(
            data,
            requires_grad=any(getattr(value, "requires_grad", False) for value in values)
            if requires_grad is None else requires_grad,
        )
        flat._init_metadata(
            [ParamInfo(str(index), self.module) for index in range(len(values))],
            [int(value.numel()) for value in values],
            [tuple(value.shape) for value in values],
            fqns=[str(index) for index in range(len(values))],
        )
        return flat

    @staticmethod
    def _get_shard(tensor: Any, rank: int, world_size: int) -> Any:
        width = (int(tensor.numel()) + world_size - 1) // world_size
        start = rank * width
        end = min(start + width, int(tensor.numel()))
        shard = tensor.reshape(-1)[start:end]
        if end - start < width:
            shard = tp.cat((shard, tensor.new_zeros(width - (end - start))), dim=0)
        return shard

    @staticmethod
    def _get_unpadded_shard(tensor: Any, rank: int, world_size: int) -> Any:
        width = (int(tensor.numel()) + world_size - 1) // world_size
        start = rank * width
        end = min(start + width, int(tensor.numel()))
        return tensor.reshape(-1)[start:end]

    @staticmethod
    def _get_sharded_size(tensor: Any, rank: int, world_size: int) -> int:
        del rank
        return (int(tensor.numel()) + world_size - 1) // world_size

    def shard(self, rank: int = 0, world_size: int = 1) -> Any:
        data = self._get_shard(self.flat_param, rank, world_size)
        self._shard_metadata = FlatParamShardMetadata(
            rank * int(data.numel()),
            min((rank + 1) * int(data.numel()), int(self.flat_param.numel())),
            int(data.numel()),
            max(0, int(data.numel()) - int(self._get_unpadded_shard(self.flat_param, rank, world_size).numel())),
        )
        self.flat_param = FlatParameter(data, self.flat_param.requires_grad)
        self.flat_param._is_sharded = True
        return self.flat_param

    def unshard(self) -> Any:
        self._unsharded_flat_param = self.flat_param
        self.flat_param._is_sharded = False
        return self.flat_param

    def reshard(self, free_unsharded_flat_param: bool = True) -> Any:
        del free_unsharded_flat_param
        self.flat_param._is_sharded = True
        return self.flat_param

    def needs_unshard(self) -> bool:
        return bool(getattr(self.flat_param, "_is_sharded", False))

    def is_sharded(self, tensor: Any | None = None) -> bool:
        return bool(getattr(tensor or self.flat_param, "_is_sharded", False))

    def sharded_grad(self) -> Any:
        return getattr(self.flat_param, "grad", None)

    def unflatten_as_params(self) -> tuple[Any, ...]:
        values = []
        offset = 0
        for numel, shape in zip(
            self.flat_param._param_metadata.numels,
            self.flat_param._param_metadata.shapes,
        ):
            values.append(self.flat_param.reshape(-1)[offset:offset + numel].reshape(shape))
            offset += numel
        return tuple(values)

    def _get_unflat_views_unaligned(self, tensor: Any) -> tuple[Any, ...]:
        values = []
        offset = 0
        for numel, shape in zip(
            self.flat_param._param_metadata.numels,
            self.flat_param._param_metadata.shapes,
        ):
            values.append(tensor.reshape(-1)[offset:offset + numel].reshape(shape))
            offset += numel
        return tuple(values)

    _get_unflat_views_aligned = _get_unflat_views_unaligned

    def param_module_names(self) -> tuple[str, ...]:
        return tuple(self.flat_param._param_metadata.fqns)

    def _get_flat_param_offsets(self) -> tuple[tuple[int, int], ...]:
        offsets = []
        start = 0
        for numel in self.flat_param._param_metadata.numels:
            offsets.append((start, start + numel))
            start += numel
        return tuple(offsets)

    def shard_metadata(self) -> FlatParamShardMetadata | None:
        return self._shard_metadata

    def to_cpu(self) -> None:
        self.flat_param = FlatParameter(self.flat_param.detach().cpu(), self.flat_param.requires_grad)

    def flat_param_to(self, *args: Any, **kwargs: Any) -> Any:
        self.flat_param = FlatParameter(self.flat_param.to(*args, **kwargs), self.flat_param.requires_grad)
        return self.flat_param

    def __repr__(self) -> str:
        return f"FlatParamHandle(num_params={len(self.params)}, sharded={self.is_sharded()})"


def _unsafe_setattr_param(module: Any, name: str, param: Any) -> None:
    module._parameters[name] = param


def _unsafe_setattr_tensor(module: Any, name: str, tensor: Any) -> None:
    module._buffers[name] = tensor


def _safe_setattr_tensor_or_param(module: Any, name: str, value: Any) -> None:
    if isinstance(value, Parameter):
        _unsafe_setattr_param(module, name, value)
    else:
        _unsafe_setattr_tensor(module, name, value)


def _convert_to_params(module: Any, names: Iterable[str]) -> None:
    for name in names:
        if name in getattr(module, "_buffers", {}):
            value = module._buffers.pop(name)
            module._parameters[name] = Parameter(value, getattr(value, "requires_grad", False))


def _is_truly_contiguous(tensor: Any) -> bool:
    return bool(getattr(tensor, "is_contiguous", lambda: False)())


def _detach_if_needed(tensor: Any) -> Any:
    return tensor.detach() if getattr(tensor, "requires_grad", False) else tensor


def _get_aligned_numel(numel: int, alignment: int) -> int:
    return ((int(numel) + alignment - 1) // alignment) * alignment


def _get_dtype_size(dtype: Any) -> int:
    return int(tp.empty((), dtype=dtype).element_size())


def _construct_padding_tensor(numel: int, dtype: Any, device: Any) -> Any:
    return tp.zeros(int(numel), dtype=dtype, device=device)


def _same_storage(left: Any, right: Any) -> bool:
    return getattr(left, "data_ptr", lambda: None)() == getattr(right, "data_ptr", lambda: None)()


def _same_storage_size(left: Any, right: Any) -> bool:
    return _same_storage(left, right) and int(left.numel()) == int(right.numel())


def _storage_size_allocated(tensor: Any) -> int:
    return int(tensor.numel()) * _get_dtype_size(tensor.dtype)
