"""Flat parameter storage and shard metadata."""

import contextlib
import functools
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable, Iterator

import tensorplay as tp
from tensorplay.nn.parameter import Parameter

from .. import distributed_core as dist
from ._common_utils import HandleTrainingState, _set_fsdp_flattened

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
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


@dataclass
class ParamInfo:
    name: str
    module: Any
    module_name: str = ""

    @property
    def param_name(self) -> str:
        return self.name


@dataclass
class SharedParamInfo:
    param_name: str
    module: Any = None
    module_name: str = ""
    prim_param_name: str = ""
    prim_module: Any = None
    prim_module_name: str = ""

    @property
    def name(self) -> str:
        return self.param_name


@dataclass
class FlatParamShardMetadata:
    param_names: tuple[str, ...] = ()
    param_shapes: tuple[tuple[int, ...], ...] = ()
    param_strides: tuple[tuple[int, ...], ...] = ()
    param_contiguities: tuple[bool, ...] = ()
    param_numels: tuple[int, ...] = ()
    param_offsets: tuple[tuple[int, int], ...] = ()

    @property
    def start(self) -> int:
        return self.param_offsets[0][0] if self.param_offsets else 0

    @property
    def end(self) -> int:
        return self.param_offsets[-1][1] if self.param_offsets else 0

    @property
    def numel(self) -> int:
        return sum(self.param_numels)

    @property
    def padding(self) -> int:
        return 0


@dataclass
class _ShardParamInfo:
    in_shard: bool
    offset_in_shard: int | None = None
    numel_in_shard: int | None = None
    intra_param_start_idx: int | None = None
    intra_param_end_idx: int | None = None

    @property
    def in_shard_start(self) -> int:
        return int(self.offset_in_shard or 0)

    @property
    def in_shard_end(self) -> int:
        return self.in_shard_start + int(self.numel_in_shard or 0)

    @property
    def shard_start(self) -> int:
        return int(self.intra_param_start_idx or 0)

    @property
    def shard_end(self) -> int:
        return int(self.intra_param_end_idx or 0) + 1


logger = logging.getLogger(__name__)


class _FlatParameterMeta(type(Parameter)):
    def __instancecheck__(cls, instance: Any) -> bool:
        del cls
        return isinstance(instance, tp.Tensor) and bool(
            getattr(instance, "_is_flat_param", False)
        )


@dataclass
class _FlatParameterMetadata:
    param_infos: list[ParamInfo] = field(default_factory=list)
    numels: list[int] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    fqns: list[str] = field(default_factory=list)
    strides: list[tuple[int, ...]] = field(default_factory=list)
    contiguities: list[bool] = field(default_factory=list)


class FlatParameter(Parameter, metaclass=_FlatParameterMeta):
    """A parameter containing a concatenated collection of parameters."""

    def __new__(cls, data: Any = None, requires_grad: bool = True) -> Any:
        if cls is not FlatParameter:
            raise AssertionError("FlatParameter does not support subclasses")
        value = super().__new__(cls)
        setattr(value, "_is_flat_param", True)
        return value

    def __init__(self, data: Any, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)
        self._param_metadata = _FlatParameterMetadata()
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
        param_infos = list(param_infos)
        numels = list(numels)
        shapes = [tuple(shape) for shape in shapes]
        strides = [tuple(stride) for stride in strides] if strides is not None else []
        contiguities = list(contiguities or ())
        fqns = list(fqns or ())
        if len(strides) == 0:
            strides = [tuple(range(len(shape) - 1, -1, -1)) for shape in shapes]
        if len(contiguities) == 0:
            contiguities = [True] * len(shapes)
        if len(fqns) == 0:
            fqns = [info.name for info in param_infos]
        padding_mask = list(is_padding_mask or (False for _ in numels))
        if len(padding_mask) != len(numels):
            raise ValueError("padding metadata must match flattened values")
        if len(param_infos) != len(shapes) or len(param_infos) != len(strides):
            raise ValueError("parameter metadata has inconsistent lengths")
        extensions = list(param_extensions or (None for _ in param_infos))
        if len(extensions) != len(param_infos):
            extensions = [None] * len(param_infos)
        actual_numels = [
            numel for numel, is_padding in zip(numels, padding_mask) if not is_padding
        ]
        self._param_metadata = _FlatParameterMetadata(
            param_infos,
            actual_numels,
            shapes,
            fqns,
            strides,
            contiguities,
        )
        self._shared_param_infos = list(shared_param_infos or ())
        self._param_infos = self._param_metadata.param_infos
        self._numels = actual_numels
        self._numels_with_padding = tuple(numels)
        self._num_params = len(param_infos)
        self._shapes = self._param_metadata.shapes
        self._strides = self._param_metadata.strides
        self._contiguities = self._param_metadata.contiguities
        self._fqns = self._param_metadata.fqns
        self._param_extensions = tuple(extensions)
        self._is_padding_mask = padding_mask
        self._modules = {
            info.module for info in param_infos if getattr(info, "module", None) is not None
        }
        if params is None:
            self._params = None
            self._shared_params = None
            self._tensors = None
            self._is_grad_none_mask = None
        else:
            self._params = list(params)
            self._shared_params = list(shared_params or ())
            self._tensors = [None] * len(param_infos)
            self._is_grad_none_mask = [False] * len(param_infos)
        unpadded_numel = int(self.numel())
        self._unpadded_unsharded_size = tuple([unpadded_numel])
        self._padded_unsharded_size = tuple([sum(numels)])
        self._sharded_size = None
        self._shard_param_infos = []
        self._shard_numel_padded = 0
        self._post_backward_called = False


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
        **kwargs: Any,
    ) -> None:
        fsdp_extension = kwargs.pop("fsdp_extension", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"unexpected FlatParamHandle argument: {unexpected}")
        self.params = list(params)
        self.module = fully_sharded_module
        self._fully_sharded_module = fully_sharded_module
        self.sharding_strategy = sharding_strategy
        self._sharding_strategy = sharding_strategy
        self._use_orig_params = bool(use_orig_params)
        self._offload_params = bool(offload_params)
        self._keep_low_precision_grads = bool(keep_low_precision_grads)
        self._training_state = HandleTrainingState.IDLE
        self._use_full_prec_in_eval = False
        self._aligned_numel = 1
        self._orig_param_dtype = getattr(self.params[0], "dtype", None) if self.params else None
        self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
        self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        self._device = device or (getattr(self.params[0], "device", None) if self.params else None)
        self.device = self._device
        if process_group is not None and isinstance(process_group, tuple):
            process_group = process_group[0]
        self.process_group = process_group
        try:
            self.rank = int(dist.get_rank(process_group)) if dist.is_initialized() else 0
            self.world_size = int(dist.get_world_size(process_group)) if dist.is_initialized() else 1
        except Exception:
            self.rank, self.world_size = 0, 1
        self._pre_unshard_event = None
        self._unshard_event = None
        self._post_reshard_event = None
        self.flat_param = self.flatten_tensors_into_flat_param(self.params)
        self._fsdp_extension = fsdp_extension
        self._init_flat_param_and_metadata(
            self.params,
            fully_sharded_module,
            aligned_numel=self._aligned_numel,
            use_orig_params=self._use_orig_params,
        )
        self._shard_metadata: FlatParamShardMetadata | None = None
        self._unsharded_flat_param = self.flat_param
        self._local_shard = self.flat_param
        self.flat_param._params = self.params if self._use_orig_params else None
        self.flat_param._is_grad_none_mask = (
            [False] * len(self.params) if self._use_orig_params else None
        )
        self.flat_param._unpadded_unsharded_size = tuple([int(self.flat_param.numel())])
        self.flat_param._sharded_size = tuple([int(self.flat_param.numel())])
        self._init_setattr_fns()
        self._init_get_unflat_views_fn(
            any(getattr(self.flat_param, "_is_padding_mask", ()))
        )
        self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
        self._unsharded_flat_param_for_skipped_views = None
        self._debug_level = None

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
        _set_fsdp_flattened(flat, True)
        flat._init_metadata(
            [ParamInfo(str(index), self.module) for index in range(len(values))],
            [int(value.numel()) for value in values],
            [tuple(value.shape) for value in values],
            fqns=[str(index) for index in range(len(values))],
            params=values,
        )
        return flat

    def _init_param_reduce_dtypes(
        self, mp_param_dtype: Any = None, mp_reduce_dtype: Any = None
    ) -> None:
        self._low_prec_param_dtype_specified = mp_param_dtype is not None
        self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
        if self._low_prec_param_dtype_specified and not self._low_prec_reduce_dtype_specified:
            self._fwd_bwd_param_dtype = mp_param_dtype
            self._reduce_dtype = mp_param_dtype
        else:
            self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
            self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        self._param_reduce_dtypes = [self._reduce_dtype] * len(self.params)

    def _init_setattr_fns(self) -> None:
        self._setattr_param = _safe_setattr_tensor_or_param
        self._setattr_tensor = _safe_setattr_tensor_or_param

    def _init_get_unflat_views_fn(self, align_addresses: bool) -> None:
        self._get_unflat_views = (
            self._get_unflat_views_aligned
            if align_addresses
            else self._get_unflat_views_unaligned
        )

    def _validate_tensors_to_flatten(self, tensors: list[Any]) -> tuple[Any, bool, Any]:
        if not tensors:
            raise AssertionError("requires a non-empty parameter list")
        dtype = None
        requires_grad: bool | None = None
        device = None
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("cannot flatten an already flattened parameter")
            is_floating_point = getattr(tensor, "is_floating_point", None)
            if callable(is_floating_point) and not is_floating_point():
                raise ValueError("cannot flatten non-floating point tensors")
            if dtype is None:
                dtype = tensor.dtype
            elif tensor.dtype != dtype:
                raise ValueError("flattened parameters must have one dtype")
            tensor_requires_grad = bool(getattr(tensor, "requires_grad", False))
            if (
                not self._use_orig_params
                and requires_grad is not None
                and tensor_requires_grad != requires_grad
            ):
                raise ValueError(
                    "flattened parameters must have uniform requires_grad values"
                )
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise ValueError("flattened parameters must use one device")
            requires_grad = bool(requires_grad) or tensor_requires_grad
        return dtype, bool(requires_grad), device

    def _init_flat_param_and_metadata(
        self,
        params: list[Any],
        module: Any,
        aligned_numel: int = 1,
        use_orig_params: bool = False,
    ) -> None:
        if not params:
            raise ValueError("requires a non-empty parameter list")
        if aligned_numel < 0:
            raise ValueError("aligned_numel must be non-negative")
        dtype, requires_grad, device = self._validate_tensors_to_flatten(params)
        params_set = {id(param) for param in params}
        values: list[Any] = []
        infos: list[ParamInfo] = []
        numels: list[int] = []
        shapes: list[Any] = []
        strides: list[tuple[int, ...]] = []
        contiguities: list[bool] = []
        fqns: list[str] = []
        padding_mask: list[bool] = []
        shared_infos: list[SharedParamInfo] = []
        shared_params: list[Any] = []
        primary_by_id: dict[int, tuple[Any, str, str]] = {}
        total = 0
        for module_name, submodule in module.named_modules(remove_duplicate=False):
            for name, param in submodule.named_parameters(
                recurse=False, remove_duplicate=False
            ):
                param_id = id(param)
                if param_id not in params_set:
                    continue
                primary = primary_by_id.get(param_id)
                if primary is not None:
                    prim_module, prim_module_name, prim_param_name = primary
                    shared_params.append(param)
                    shared_infos.append(
                        SharedParamInfo(
                            name,
                            submodule,
                            module_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                    continue
                if aligned_numel > 1:
                    padding = (-total) % int(aligned_numel)
                    if padding:
                        values.append(
                            _construct_padding_tensor(padding, dtype, False, device)
                        )
                        numels.append(padding)
                        padding_mask.append(True)
                        total += padding
                values.append(param)
                numels.append(int(param.numel()))
                padding_mask.append(False)
                infos.append(ParamInfo(name, submodule, module_name))
                shapes.append(tuple(param.shape))
                strides.append(tuple(param.stride()))
                contiguities.append(_is_truly_contiguous(param))
                fqns.append(f"{module_name}.{name}" if module_name else name)
                primary_by_id[param_id] = (submodule, module_name, name)
                total += int(param.numel())
        if aligned_numel > 0 and self.world_size > 1:
            padding = (-total) % int(self.world_size)
            if padding:
                values.append(_construct_padding_tensor(padding, dtype, False, device))
                numels.append(padding)
                padding_mask.append(True)
                total += padding
        if not values:
            raise ValueError("parameters were not found in the module")
        self.flat_param = self.flatten_tensors_into_flat_param(
            values, aligned_numel=0, requires_grad=requires_grad
        )
        self.flat_param._init_metadata(
            infos,
            numels,
            shapes,
            strides,
            contiguities,
            fqns,
            shared_param_infos=shared_infos,
            params=[
                value
                for value, is_padding in zip(values, padding_mask)
                if not is_padding
            ]
            if use_orig_params
            else None,
            shared_params=shared_params if use_orig_params else None,
            is_padding_mask=padding_mask,
        )
        _set_fsdp_flattened(self.flat_param, True)
        self.params = params
        self._unsharded_flat_param = self.flat_param

    def init_flat_param_attributes(self) -> None:
        if self.flat_param is None:
            return
        if getattr(self.flat_param, "dtype", None) != self._orig_param_dtype:
            if not getattr(self, "_low_prec_param_dtype_specified", False):
                self._fwd_bwd_param_dtype = self.flat_param.dtype
            if not getattr(self, "_low_prec_reduce_dtype_specified", False) and not getattr(
                self, "_low_prec_param_dtype_specified", False
            ):
                self._reduce_dtype = self.flat_param.dtype
            self._orig_param_dtype = self.flat_param.dtype
        if self._offload_params and str(getattr(self.flat_param.device, "type", self.flat_param.device)) != "cpu":
            raise AssertionError("parameter offload requires the flat parameter on CPU")
        if not self._offload_params:
            self._check_on_compute_device(self.flat_param)
        metadata = self.flat_param._param_metadata
        metadata_numel = sum(getattr(self.flat_param, "_numels_with_padding", metadata.numels))
        full_numel = (
            int(self.flat_param.numel()) * int(self.world_size)
            if self.uses_sharded_strategy
            else int(self.flat_param.numel())
        )
        padded_numel = max(metadata_numel, full_numel)
        self.flat_param._unpadded_unsharded_size = tuple([full_numel])
        self.flat_param._padded_unsharded_size = tuple([padded_numel])
        if getattr(self.flat_param, "_sharded_size", None) is None:
            self.flat_param._sharded_size = tuple([int(self.flat_param.numel())])
        if not hasattr(self.flat_param, "_shard_param_infos"):
            self.flat_param._shard_param_infos = []
        if not hasattr(self.flat_param, "_is_padding_mask"):
            self.flat_param._is_padding_mask = [False] * len(metadata.numels)
        self.flat_param._local_shard = self.flat_param.data
        if self._offload_params:
            if getattr(self.flat_param._local_shard, "pin_memory", None) is not None:
                try:
                    self.flat_param._local_shard = self.flat_param._local_shard.pin_memory()
                except RuntimeError:
                    pass
            self.flat_param._cpu_grad = tp.zeros_like(self.flat_param._local_shard)
        if self._uses_param_mixed_precision:
            self.flat_param._mp_shard = tp.empty(
                tuple(self.flat_param._local_shard.shape),
                dtype=self._fwd_bwd_param_dtype,
                device=self.device,
            )
            _free_storage(self.flat_param._mp_shard)
        if self.uses_sharded_strategy:
            full_dtype = (
                self._fwd_bwd_param_dtype
                if self._uses_param_mixed_precision
                else self.flat_param.dtype
            )
            full_size = int(self.flat_param.numel()) * int(self.world_size)
            self.flat_param._full_param_padded = tp.empty(
                (full_size,), dtype=full_dtype, device=self.device
            )
            _free_storage(self.flat_param._full_param_padded)
            if self._uses_param_mixed_precision:
                self.flat_param._full_prec_full_param_padded = tp.empty(
                    (full_size,), dtype=self.flat_param.dtype, device=self.device
                )
                _free_storage(self.flat_param._full_prec_full_param_padded)
        self._local_shard = self.flat_param._local_shard

    def _new_flat_parameter(self, data: Any, requires_grad: bool | None = None) -> FlatParameter:
        flat = FlatParameter(
            data,
            self.flat_param.requires_grad if requires_grad is None else requires_grad,
        )
        metadata = self.flat_param._param_metadata
        flat._init_metadata(
            metadata.param_infos,
            getattr(self.flat_param, "_numels_with_padding", metadata.numels),
            metadata.shapes,
            metadata.strides,
            metadata.contiguities,
            metadata.fqns,
            shared_param_infos=getattr(self.flat_param, "_shared_param_infos", ()),
            params=self.params if self._use_orig_params else None,
            is_padding_mask=getattr(self.flat_param, "_is_padding_mask", None),
        )
        _set_fsdp_flattened(flat, True)
        flat._is_grad_none_mask = list(
            getattr(self.flat_param, "_is_grad_none_mask", [False] * len(self.params))
        )
        flat._unpadded_unsharded_size = getattr(
            self.flat_param, "_unpadded_unsharded_size", tuple([int(data.numel())])
        )
        flat._sharded_size = tuple([int(data.numel())])
        flat._padded_unsharded_size = getattr(
            self.flat_param, "_padded_unsharded_size", flat._unpadded_unsharded_size
        )
        flat._is_sharded = getattr(self.flat_param, "_is_sharded", False)
        for name in (
            "_local_shard",
            "_full_param_padded",
            "_full_prec_full_param_padded",
            "_mp_shard",
            "_cpu_grad",
        ):
            if hasattr(self.flat_param, name):
                setattr(flat, name, getattr(self.flat_param, name))
        self._init_get_unflat_views_fn(any(getattr(flat, "_is_padding_mask", ())))
        return flat

    @staticmethod
    def _get_unpadded_shard(tensor: Any, rank: int, world_size: int) -> tuple[Any, int]:
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("rank and world_size must describe a valid shard")
        chunks = tensor.reshape(-1).chunk(world_size)
        chunk = chunks[rank] if rank < len(chunks) else tensor.new_empty(0)
        padding = (chunks[0].numel() if chunks else 0) - int(chunk.numel())
        return chunk, int(padding)

    @staticmethod
    def _get_shard(tensor: Any, rank: int, world_size: int) -> tuple[Any, int]:
        chunk, padding = FlatParamHandle._get_unpadded_shard(tensor, rank, world_size)
        shard = chunk.detach().clone()
        if padding:
            shard = tp.cat((shard, tensor.new_zeros(padding)), dim=0)
        return shard, padding

    @staticmethod
    def _get_sharded_size(tensor: Any, rank: int, world_size: int) -> int:
        chunk, padding = FlatParamHandle._get_unpadded_shard(tensor, rank, world_size)
        return int(chunk.numel()) + int(padding)

    def _init_shard_metadata(
        self, numel_padded: int, unsharded_start_idx: int, unsharded_end_idx: int
    ) -> None:
        flat = self.flat_param
        flat._sharded_size = tuple(flat.shape)
        flat._shard_numel_padded = int(numel_padded)
        flat._shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )

    def _get_shard_metadata(
        self, unsharded_start_idx: int, unsharded_end_idx: int
    ) -> tuple[_ShardParamInfo, ...]:
        offsets = self._get_flat_param_offsets()
        if len(offsets) != len(self.flat_param._is_padding_mask):
            raise AssertionError(
                "flat parameter padding metadata has an invalid length"
            )
        result: list[_ShardParamInfo] = []
        shard_numel = max(0, unsharded_end_idx - unsharded_start_idx + 1)
        for (param_start, param_end), is_padding in zip(
            offsets, self.flat_param._is_padding_mask
        ):
            if is_padding:
                continue
            if (
                unsharded_start_idx > param_end
                or unsharded_end_idx < param_start
            ):
                result.append(_ShardParamInfo(False))
                continue
            start = max(unsharded_start_idx, param_start)
            end = min(unsharded_end_idx, param_end)
            offset = start - unsharded_start_idx
            count = end - start + 1
            if offset < 0 or offset >= max(1, shard_numel):
                raise ValueError("invalid shard parameter offset")
            result.append(
                _ShardParamInfo(
                    True,
                    offset,
                    count,
                    start - param_start,
                    end - param_start,
                )
            )
        return tuple(result)

    def shard(self, rank: int = 0, world_size: int = 1) -> Any:
        flat_param = self.flat_param
        if world_size == 1 and self.world_size > 1:
            world_size, rank = self.world_size, self.rank
        source = (
            self._unsharded_flat_param
            if self._unsharded_flat_param is not None
            else flat_param
        )
        if not self.uses_sharded_strategy:
            self.flat_param = source
            self.flat_param._is_sharded = False
            self._init_shard_metadata(0, 0, int(source.numel()) - 1)
            self._shard_metadata = self.shard_metadata()
            self._local_shard = self.flat_param
            self._use_sharded_views()
            return self.flat_param
        data, padding = self._get_shard(source, rank, world_size)
        if _storage_size_allocated(flat_param):
            _free_storage(flat_param)
        flat_param.set_(data)
        self.flat_param = flat_param
        flat_param._is_sharded = True
        flat_param._sharded_size = tuple([int(data.numel())])
        flat_param._local_shard = flat_param.data
        self._init_shard_metadata(
            padding,
            int(data.numel()) * rank,
            int(data.numel()) * (rank + 1) - 1,
        )
        self._shard_metadata = self.shard_metadata()
        self._local_shard = flat_param.data
        self._unsharded_flat_param = source
        self._use_sharded_views()
        return self.flat_param

    def _build_shard_param_infos(self, rank: int, world_size: int) -> list[_ShardParamInfo]:
        shard, _ = self._get_shard(self._unsharded_flat_param, rank, world_size)
        return list(
            self._get_shard_metadata(
                int(shard.numel()) * rank,
                int(shard.numel()) * (rank + 1) - 1,
            )
        )

    def unshard(self) -> Any:
        if not self.needs_unshard():
            padded = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(padded)
            return self.flat_param
        padded = self._alloc_padded_unsharded_flat_param()
        self._all_gather_flat_param(padded)
        self._use_unsharded_flat_param(padded)
        return self.flat_param

    def reshard(self, free_unsharded_flat_param: bool = True) -> Any:
        if not self.uses_sharded_strategy:
            self.flat_param._is_sharded = False
            self._use_sharded_views()
            return self.flat_param
        self._use_sharded_flat_param()
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()
        return self.flat_param

    def needs_unshard(self) -> bool:
        if not self.uses_sharded_strategy:
            return False
        return not _storage_size_allocated(self._get_padded_unsharded_flat_param())

    def _get_padded_unsharded_flat_param(self) -> Any:
        self._check_sharded_strategy()
        if self._force_full_precision and self._uses_param_mixed_precision:
            return getattr(
                self.flat_param,
                "_full_prec_full_param_padded",
                self._unsharded_flat_param,
            )
        return getattr(
            self.flat_param, "_full_param_padded", self._unsharded_flat_param
        )

    def _alloc_padded_unsharded_flat_param(self) -> Any:
        self._check_sharded_strategy()
        padded = self._get_padded_unsharded_flat_param()
        size = int(
            getattr(
                self.flat_param,
                "_padded_unsharded_size",
                (sum(getattr(self.flat_param, "_numels_with_padding", self.flat_param._numels)),),
            )[0]
        )
        if padded is None or int(padded.numel()) != size or getattr(padded, "device", None) != self.device:
            dtype = self._orig_param_dtype or self.flat_param.dtype
            padded = tp.empty((size,), dtype=dtype, device=self.device)
            if self._force_full_precision and self._uses_param_mixed_precision:
                self.flat_param._full_prec_full_param_padded = padded
            else:
                self.flat_param._full_param_padded = padded
        elif not _storage_size_allocated(padded):
            _alloc_storage(padded, (size,))
        return padded

    def _all_gather_flat_param(self, padded_unsharded_flat_param: Any) -> Any:
        self._check_sharded_strategy()
        local = self._local_shard if self._local_shard is not None else self.flat_param
        expected = int(local.numel()) * int(self.world_size)
        if int(padded_unsharded_flat_param.numel()) != expected:
            raise ValueError("all-gather destination has an invalid size")
        if self.world_size > 1 and dist.is_initialized():
            dist.all_gather_single(
                padded_unsharded_flat_param,
                local.reshape(-1),
                group=self.process_group,
            )
        else:
            padded_unsharded_flat_param.reshape(-1).copy_(local.reshape(-1))
        return padded_unsharded_flat_param

    def _use_unsharded_flat_param(self, padded_unsharded_flat_param: Any) -> None:
        flat_param = self.flat_param
        size = int(
            getattr(
                flat_param,
                "_unpadded_unsharded_size",
                (int(padded_unsharded_flat_param.numel()),),
            )[0]
        )
        if int(padded_unsharded_flat_param.numel()) < size:
            raise ValueError("padded unsharded parameter is smaller than its metadata")
        value = padded_unsharded_flat_param.reshape(-1).narrow(0, 0, size)
        self._unsharded_flat_param = padded_unsharded_flat_param
        flat_param.data = value
        flat_param._full_param_padded = padded_unsharded_flat_param
        flat_param._is_sharded = False
        flat_param._sharded_size = tuple(self._local_shard.shape)
        flat_param._local_shard = self._local_shard
        in_forward = self._training_state == HandleTrainingState.FORWARD
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        if self._use_orig_params:
            if self._skipped_use_sharded_views and in_pre_backward:
                return
            self._use_unsharded_views(as_params=not in_forward and not in_pre_backward)
        elif in_forward:
            self._use_unsharded_views(as_params=False)

    def _use_sharded_flat_param(self) -> None:
        flat_param = self.flat_param
        local = self._local_shard
        if local is None:
            local, _ = self._get_shard(self._unsharded_flat_param, self.rank, self.world_size)
            self._local_shard = local
        in_forward = self._training_state == HandleTrainingState.FORWARD
        skip_use_sharded_views = bool(
            tp.is_grad_enabled()
            and in_forward
            and self._sharding_strategy == HandleShardingStrategy.SHARD_GRAD_OP
        )
        unsharded_flat_param = self.flat_param.data if skip_use_sharded_views else None
        flat_param.data = local
        flat_param._is_sharded = True
        flat_param._sharded_size = tuple(local.shape)
        flat_param._local_shard = local
        self._init_shard_metadata(
            int(getattr(flat_param, "_shard_numel_padded", 0)),
            int(local.numel()) * self.rank,
            int(local.numel()) * (self.rank + 1) - 1,
        )
        if self._use_orig_params:
            if skip_use_sharded_views:
                self._unsharded_flat_param_for_skipped_views = unsharded_flat_param
            else:
                self._use_sharded_views()
            if in_forward and not self._skipped_use_sharded_views:
                accumulated_grad = getattr(self.flat_param, "grad", None)
                if (
                    accumulated_grad is not None
                    and self.uses_sharded_strategy
                    and tuple(getattr(accumulated_grad, "shape", ()))
                    == tuple(getattr(self.flat_param, "_unpadded_unsharded_size", ()))
                ):
                    self._use_unsharded_grad_views()
                else:
                    self._use_sharded_grad_views()
        else:
            self._use_sharded_views()

    def is_sharded(self, tensor: Any | None = None) -> bool:
        value = self.flat_param if tensor is None else tensor
        return bool(getattr(value, "_is_sharded", False))

    @contextlib.contextmanager
    def unflatten_as_params(self):
        self._use_unsharded_views(as_params=True)
        try:
            yield
        finally:
            self._use_unsharded_views(as_params=False)

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

    def _get_unflat_views_aligned(self, tensor: Any = None) -> tuple[Any, ...]:
        if tensor is None:
            tensor = self.flat_param
        values: list[Any] = []
        actual_index = 0
        for split, is_padding in zip(
            tensor.split(tuple(self.flat_param._numels_with_padding), dim=0),
            self.flat_param._is_padding_mask,
        ):
            if is_padding:
                continue
            shape = self.flat_param._param_metadata.shapes[actual_index]
            values.append(split.reshape(shape))
            actual_index += 1
        return tuple(values)

    def _use_unsharded_views(self, as_params: bool = False) -> None:
        flat = self.flat_param
        views = self._get_unflat_views(flat)
        params = flat._params if flat._params is not None else self.params
        for index, (view, info) in enumerate(
            zip(views, flat._param_metadata.param_infos)
        ):
            param = params[index]
            if as_params:
                if self._use_orig_params:
                    param.data = view
                    self._setattr_param(info.module, info.name, param)
                else:
                    self._setattr_param(
                        info.module,
                        info.name,
                        Parameter(view, requires_grad=flat.requires_grad),
                    )
            else:
                param_var = view
                if self._use_orig_params:
                    if self._training_state == HandleTrainingState.FORWARD:
                        if flat._tensors is not None:
                            flat._tensors[index] = view
                    elif self._training_state == HandleTrainingState.BACKWARD_PRE:
                        if flat._tensors is not None and flat._tensors[index] is not None:
                            tensor = flat._tensors[index]
                            tensor.data = view
                            param_var = tensor
                self._setattr_tensor(info.module, info.name, param_var)
                if self._training_state == HandleTrainingState.FORWARD:
                    info.module._parameters.pop(info.name, None)
                if self._use_orig_params and self._training_state == HandleTrainingState.FORWARD:
                    info.module._parameters[info.name] = param_var

        for shared_index, info in enumerate(flat._shared_param_infos):
            if getattr(info, "module", None) is None or getattr(info, "prim_module", None) is None:
                continue
            primary = getattr(info.prim_module, info.prim_param_name)
            if (
                as_params
                and self._use_orig_params
                and flat._shared_params is not None
                and shared_index < len(flat._shared_params)
            ):
                shared = flat._shared_params[shared_index]
                shared.data = primary
                self._setattr_param(info.module, info.param_name, shared)
            elif as_params:
                self._setattr_param(
                    info.module,
                    info.param_name,
                    primary
                    if isinstance(primary, Parameter)
                    else Parameter(primary, requires_grad=flat.requires_grad),
                )
            else:
                self._setattr_tensor(info.module, info.param_name, primary)

    def _use_sharded_views(self) -> None:
        self._unsharded_flat_param_for_skipped_views = None
        if not self._use_orig_params:
            for info in self.flat_param._param_metadata.param_infos:
                info.module._parameters.pop(info.name, None)
                info.module._buffers.pop(info.name, None)
            for info in getattr(self.flat_param, "_shared_param_infos", ()):
                if getattr(info, "module", None) is not None:
                    info.module._parameters.pop(info.param_name, None)
                    info.module._buffers.pop(info.param_name, None)
            root = getattr(self.module, "module", self.module)
            root._parameters["_flat_param"] = self.flat_param
            return
        if not self.uses_sharded_strategy:
            self._use_unsharded_views(as_params=True)
            return
        flat = self.flat_param
        empty = flat.new_empty((0,))
        params = flat._params if flat._params is not None else self.params
        for index, info in enumerate(flat._param_metadata.param_infos):
            param = params[index]
            shard_info = flat._shard_param_infos[index] if index < len(flat._shard_param_infos) else None
            if shard_info is None or not shard_info.in_shard:
                value = empty
            else:
                value = flat.reshape(-1).narrow(
                    0,
                    int(shard_info.offset_in_shard or 0),
                    int(shard_info.numel_in_shard or 0),
                )
            param.data = value
            self._setattr_param(info.module, info.name, param)

        for info, param in zip(flat._shared_param_infos, flat._shared_params or ()):
            if getattr(info, "module", None) is None or getattr(info, "prim_module", None) is None:
                continue
            primary = getattr(info.prim_module, info.prim_param_name)
            param.data = primary
            self._setattr_param(info.module, info.param_name, param)
        if self._training_state == HandleTrainingState.BACKWARD_POST and flat._tensors is not None:
            for index in range(len(flat._tensors)):
                flat._tensors[index] = None

    def _use_unsharded_grad_views(self) -> None:
        if not self._use_orig_params:
            return
        flat = self.flat_param
        grad = getattr(flat, "grad", None)
        params = flat._params if flat._params is not None else self.params
        if grad is None:
            for param in (*params, *(flat._shared_params or ())):
                param.grad = None
            return
        for param, view in zip(params, self._get_unflat_views(grad)):
            current = getattr(param, "grad", None)
            if (
                current is None
                or tuple(getattr(current, "shape", ())) != tuple(param.shape)
                or getattr(current, "dtype", None) != getattr(view, "dtype", None)
                or getattr(current, "device", None) != getattr(view, "device", None)
            ):
                param.grad = tp.empty_like(param)
                param.grad.data = view
            else:
                param.grad = view
        for info, param in zip(flat._shared_param_infos, flat._shared_params or ()):
            primary = getattr(info.prim_module, info.prim_param_name)
            param.grad = getattr(primary, "grad", None)

    def _use_sharded_grad_views(self) -> None:
        if not self._use_orig_params:
            return
        flat = self.flat_param
        grad = self.sharded_grad
        params = flat._params if flat._params is not None else self.params
        if grad is None:
            for param in (*params, *(flat._shared_params or ())):
                param.grad = None
            return
        mask = getattr(flat, "_is_grad_none_mask", None) or ()
        for index, param in enumerate(params):
            info = flat._shard_param_infos[index] if index < len(flat._shard_param_infos) else None
            is_grad_none = bool(mask[index]) if index < len(mask) else False
            if info is None or not info.in_shard or not param.requires_grad or is_grad_none:
                param.grad = None
                continue
            view = grad.narrow(
                0,
                int(info.offset_in_shard or 0),
                int(info.numel_in_shard or 0),
            ).reshape(param.shape)
            if (
                self._keep_low_precision_grads
                or getattr(param, "dtype", None) != getattr(grad, "dtype", None)
            ):
                param.grad = tp.empty_like(param)
                param.grad.data = view
            else:
                param.grad = view
        for info, param in zip(flat._shared_param_infos, flat._shared_params or ()):
            primary = getattr(info.prim_module, info.prim_param_name)
            param.grad = getattr(primary, "grad", None) if hasattr(primary, "grad") else None

    def prepare_gradient_for_backward(self) -> Any:
        flat_param = self.flat_param
        grad = getattr(flat_param, "grad", None)
        if grad is None:
            return None
        expected = tuple(getattr(flat_param, "_unpadded_unsharded_size", grad.shape))
        local_shard = getattr(flat_param, "_local_shard", self._local_shard)
        local_size = tuple(getattr(local_shard, "shape", ()))
        if tuple(grad.shape) != expected or getattr(grad, "device", None) != getattr(flat_param, "device", None):
            if not self._offload_params and getattr(grad, "device", None) != getattr(flat_param, "device", None):
                raise AssertionError("gradient is on a different device from the parameter")
            if tuple(grad.shape) == local_size:
                if getattr(grad, "device", None) == getattr(flat_param, "device", None):
                    flat_param._saved_grad_shard = grad.data
                else:
                    if getattr(flat_param, "_cpu_grad", None) is None:
                        flat_param._cpu_grad = tp.zeros_like(grad, device="cpu")
                    flat_param._cpu_grad.copy_(grad)
            elif tuple(grad.shape) != tuple(getattr(flat_param, "_padded_unsharded_size", expected)):
                raise AssertionError("gradient has an invalid flattened shape")
            flat_param.grad = None
        return flat_param.grad

    def prepare_gradient_for_optim(self) -> Any:
        def cast_grad_to_param_dtype_if_needed(flat_param: Any) -> None:
            if not self._force_full_precision and self._keep_low_precision_grads:
                if flat_param.grad is None:
                    raise AssertionError("unexpected missing gradient")
                if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                    flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                    if self._use_orig_params:
                        self._use_sharded_grad_views()

        flat_param = self.flat_param
        if hasattr(flat_param, "_cpu_grad"):
            self._check_sharded(flat_param)
            self._check_on_cpu(flat_param)
            flat_param.grad = flat_param._cpu_grad
            cast_grad_to_param_dtype_if_needed(flat_param)
        elif hasattr(flat_param, "_saved_grad_shard"):
            self._check_sharded(flat_param)
            self._check_on_compute_device(flat_param)
            saved = flat_param._saved_grad_shard
            if saved is not None:
                self._check_on_compute_device(saved)
            if getattr(flat_param, "_post_backward_called", False):
                flat_param.grad = saved
                if flat_param.grad is not None:
                    cast_grad_to_param_dtype_if_needed(flat_param)
        elif self.uses_sharded_strategy and getattr(flat_param, "_post_backward_called", False):
            raise AssertionError("sharded parameters require a saved sharded gradient")
        if hasattr(flat_param, "_saved_grad_shard"):
            delattr(flat_param, "_saved_grad_shard")
        return flat_param.grad

    def unshard_grad(self) -> Any:
        if not self.uses_sharded_strategy:
            self._use_unsharded_grad_views()
            return self.flat_param.grad
        flat_param = self.flat_param
        grad = getattr(flat_param, "grad", None)
        if grad is None and hasattr(flat_param, "_saved_grad_shard"):
            grad = flat_param._saved_grad_shard
        if self.world_size <= 1 or not dist.is_initialized():
            if grad is not None:
                flat_param.grad = grad
            return grad
        if grad is None:
            grad = tp.zeros_like(getattr(flat_param, "_local_shard", flat_param))
        flat_param._saved_grad_shard = grad
        output_size = int(getattr(flat_param, "_padded_unsharded_size", (int(grad.numel()) * self.world_size,))[0])
        output = grad.new_empty((output_size,))
        dist.all_gather_single(output, grad.reshape(-1), group=self.process_group)
        full_size = int(getattr(flat_param, "_unpadded_unsharded_size", (output_size,))[0])
        flat_param.grad = output.narrow(0, 0, full_size)
        self._use_unsharded_grad_views()
        return flat_param.grad

    def reshard_grad(self) -> Any:
        flat_param = self.flat_param
        grad = getattr(flat_param, "grad", None)
        if self._use_orig_params:
            self._use_sharded_grad_views()
        if not self.uses_sharded_strategy:
            return grad
        saved = getattr(flat_param, "_saved_grad_shard", None)
        if saved is not None:
            flat_param.grad = saved
            delattr(flat_param, "_saved_grad_shard")
            return saved
        if grad is None or self.world_size <= 1:
            return grad
        shard, _ = self._get_shard(grad, self.rank, self.world_size)
        flat_param.grad = shard
        return shard

    def post_unshard(self) -> None:
        if self._uses_param_mixed_precision and self.uses_sharded_strategy:
            self._free_low_precision_sharded_param()
        self._check_on_compute_device(self.flat_param)

    def post_reshard(self) -> None:
        if (
            self._uses_param_mixed_precision
            and not self.uses_sharded_strategy
            and not self._force_full_precision
        ):
            self._free_low_precision_sharded_param()

    def pre_unshard(self) -> bool:
        if (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            and self._skipped_use_sharded_views
        ):
            self._use_sharded_views()
        changed = False
        if self._use_orig_params and not bool(getattr(self, "_skip_writeback_check", False)):
            changed = self._writeback_orig_params()
        if (
            self.uses_sharded_strategy
            and not self._offload_params
            and not self.needs_unshard()
        ):
            pass
        elif self._uses_param_mixed_precision and not self._force_full_precision:
            self._use_low_precision_shard()
            changed = True
        elif self._offload_params and getattr(self.flat_param, "device", None) != self.device:
            self.flat_param_to(self.device, non_blocking=True)
            changed = True
        self._check_on_compute_device(self.flat_param)
        return changed

    @contextlib.contextmanager
    def _offload_to_cpu(self):
        self._check_sharded_strategy()
        self.flat_param_to("cpu")
        self._free_unsharded_flat_param()
        try:
            yield
        finally:
            self._restore_unsharded_flat_param_from_cpu()

    @contextlib.contextmanager
    def to_cpu(self):
        self._check_sharded_strategy()
        if str(getattr(self.flat_param, "device", "cpu")).split(":", 1)[0] == "cpu":
            yield
            return
        with self._offload_to_cpu():
            yield

    def _restore_unsharded_flat_param_from_cpu(self) -> None:
        if self._device is None:
            return
        padded = self._alloc_padded_unsharded_flat_param()
        source = self.flat_param.reshape(-1)
        size = min(int(source.numel()), int(padded.numel()))
        padded.reshape(-1).narrow(0, 0, size).copy_(source.narrow(0, 0, size))
        self._use_unsharded_flat_param(padded)

    def _free_unsharded_flat_param(self) -> None:
        padded = self._get_padded_unsharded_flat_param()
        if padded is not None:
            _free_storage(padded)

    def _free_low_precision_sharded_param(self) -> None:
        flat_param = self.flat_param
        mp_shard = getattr(flat_param, "_mp_shard", None)
        if mp_shard is not None:
            _free_storage(mp_shard)
        self._mp_shard = mp_shard

    def _use_low_precision_shard(self) -> None:
        flat_param = self.flat_param
        local_shard = getattr(flat_param, "_local_shard", self._local_shard)
        mp_shard = getattr(flat_param, "_mp_shard", None)
        if (
            mp_shard is None
            or int(getattr(mp_shard, "numel", lambda: 0)()) != int(local_shard.numel())
            or getattr(mp_shard, "device", None) != self.device
            or getattr(mp_shard, "dtype", None) != self._fwd_bwd_param_dtype
        ):
            mp_shard = tp.empty(
                tuple(local_shard.shape),
                dtype=self._fwd_bwd_param_dtype,
                device=self.device,
            )
        elif not _storage_size_allocated(mp_shard):
            _alloc_storage(mp_shard, tuple(local_shard.shape))
        source = local_shard.to(self.device) if getattr(local_shard, "device", None) != self.device else local_shard
        mp_shard.copy_(source)
        flat_param._mp_shard = mp_shard
        self._mp_shard = mp_shard
        flat_param.data = mp_shard

    @property
    def _force_full_precision(self) -> bool:
        return bool(
            (self._uses_param_mixed_precision or self._uses_reduce_mixed_precision)
            and getattr(self, "_training_state", HandleTrainingState.IDLE)
            == HandleTrainingState.SUMMON_FULL_PARAMS
        )

    def _get_modules(self) -> set[Any]:
        return {
            info.module for info in self.flat_param._param_metadata.param_infos
        }.union(
            {
                info.module
                for info in getattr(self.flat_param, "_shared_param_infos", ())
                if getattr(info, "module", None) is not None
            }
        )

    def shared_param_module_names(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (getattr(info, "param_name", ""), getattr(info, "module_name", ""))
            for info in getattr(self.flat_param, "_shared_param_infos", ())
        )

    @property
    def _fqns_in_shard(self) -> list[str]:
        if not self.uses_sharded_strategy:
            return list(self.flat_param._param_metadata.fqns)
        return [
            fqn for fqn, info in zip(self.flat_param._param_metadata.fqns, self.flat_param._shard_param_infos)
            if info.in_shard_end > info.in_shard_start
        ]

    @property
    def sharded_grad(self) -> Any:
        flat_param = self.flat_param
        if hasattr(flat_param, "_cpu_grad"):
            return flat_param._cpu_grad
        if hasattr(flat_param, "_saved_grad_shard"):
            return flat_param._saved_grad_shard
        return flat_param.grad

    def _reset_flat_param_grad_info_if_needed(self) -> None:
        if not self._use_orig_params:
            return
        flat_param = self.flat_param
        params = flat_param._params if flat_param._params is not None else self.params
        all_grad_none = all(param.grad is None for param in params)
        if all_grad_none:
            flat_param.grad = None
        flat_param.requires_grad = any(param.requires_grad for param in params)

    def _reset_is_grad_none(self) -> None:
        if not self._use_orig_params or self.flat_param._is_grad_none_mask is None:
            return
        for index, param in enumerate(self.params):
            if param.requires_grad and index < len(self.flat_param._is_grad_none_mask):
                self.flat_param._is_grad_none_mask[index] = False

    def _writeback_tensor(
        self,
        src_tensor: Any,
        dst_tensor: Any,
        tensor_index: int,
        expected_shape: Any,
        offset: int,
        is_param: bool,
    ) -> None:
        expected_numel = int(expected_shape.numel()) if hasattr(expected_shape, "numel") else int(expected_shape[0])
        if src_tensor is not None and tuple(src_tensor.shape) != tuple(expected_shape):
            kind = "parameter" if is_param else "gradient"
            raise RuntimeError(
                f"cannot write back {kind} with shape {tuple(src_tensor.shape)}; expected {tuple(expected_shape)}"
            )
        target = dst_tensor.reshape(-1).narrow(0, int(offset), expected_numel)
        if src_tensor is None:
            target.zero_()
            if self.flat_param._is_grad_none_mask is not None and tensor_index < len(self.flat_param._is_grad_none_mask):
                self.flat_param._is_grad_none_mask[tensor_index] = True
        else:
            target.copy_(src_tensor.reshape(-1))

    def _writeback_orig_params(self) -> bool:
        if not self._use_orig_params or self.flat_param._params is None:
            return False
        if (
            self.uses_sharded_strategy
            and not self.is_sharded(self.flat_param)
            and not self._skipped_use_sharded_views
        ):
            return False
        target = (
            self._unsharded_flat_param_for_skipped_views
            if self._skipped_use_sharded_views
            else self.flat_param
        )
        if target is None:
            raise RuntimeError("unsharded parameter storage is unavailable")
        flat_param_grad = (
            getattr(self.flat_param, "grad", None)
            if self.uses_sharded_strategy or not self._offload_params
            else getattr(self.flat_param, "_cpu_grad", None)
        )
        wrote_back = False
        for index, (param, shard_info, info) in enumerate(
            zip(
                self.flat_param._params,
                self.flat_param._shard_param_infos,
                self.flat_param._param_infos,
            )
        ):
            if not shard_info.in_shard:
                continue
            parameters = getattr(info.module, "_parameters", {})
            if info.name not in parameters:
                continue
            original = param
            if self._skipped_use_sharded_views:
                if self.flat_param._tensors is None or self.flat_param._tensors[index] is None:
                    raise RuntimeError("saved parameter view is unavailable")
                original = self.flat_param._tensors[index]
            current = parameters[info.name]
            param_changed = current is not original
            needs_param_writeback = param_changed or not _same_storage(original, target)
            if self._skipped_use_sharded_views and needs_param_writeback:
                raise RuntimeError("parameters cannot change between forward and backward")
            if param_changed:
                param = current
                self.flat_param._params[index] = param
            count = int(shard_info.numel_in_shard or 0)
            if needs_param_writeback:
                source = param if self.uses_sharded_strategy else param.reshape(-1)
                self._writeback_tensor(
                    source,
                    target,
                    index,
                    (count,),
                    int(shard_info.offset_in_shard or 0),
                    True,
                )
                wrote_back = True
            if self._skipped_use_sharded_views:
                continue
            current_grad = getattr(param, "grad", None)
            if current_grad is None and getattr(self.flat_param, "grad", None) is not None:
                self._writeback_tensor(
                    None,
                    self.flat_param.grad,
                    index,
                    (count,),
                    int(shard_info.offset_in_shard or 0),
                    False,
                )
            elif current_grad is not None:
                if not self.uses_sharded_strategy and self._offload_params:
                    continue
                needs_grad_writeback = flat_param_grad is None or not _same_storage(
                    current_grad, flat_param_grad
                )
                if needs_grad_writeback:
                    if flat_param_grad is None:
                        flat_param_grad = tp.zeros_like(self.flat_param)
                    source = (
                        current_grad
                        if self.uses_sharded_strategy
                        else current_grad.reshape(-1)
                    )
                    self._writeback_tensor(
                        source,
                        flat_param_grad,
                        index,
                        (count,),
                        int(shard_info.offset_in_shard or 0),
                        False,
                    )
                    self.flat_param.grad = flat_param_grad
        for info in self.flat_param._shared_param_infos:
            shared = getattr(info.module, "_parameters", {}).get(info.param_name)
            primary = getattr(info.prim_module, "_parameters", {}).get(info.prim_param_name)
            if shared is not primary:
                raise NotImplementedError("changing shared parameters is not supported")
        return wrote_back

    def _deregister_orig_params(self) -> None:
        for info in self.flat_param._param_metadata.param_infos:
            if hasattr(info.module, info.name):
                info.module._parameters.pop(info.name, None)
        for info in getattr(self.flat_param, "_shared_param_infos", ()):
            if getattr(info, "module", None) is not None:
                info.module._parameters.pop(info.param_name, None)

    def _check_sharded_strategy(self) -> None:
        if not self.uses_sharded_strategy:
            raise AssertionError("sharded strategy is required")

    def _check_on_compute_device(self, tensor: Any) -> None:
        if self.device is not None and getattr(tensor, "device", None) != self.device:
            raise AssertionError(f"tensor is on {tensor.device}, expected {self.device}")

    def _check_on_cpu(self, tensor: Any) -> None:
        if str(getattr(tensor, "device", "cpu")).split(":", 1)[0] != "cpu":
            raise AssertionError(f"tensor is not on CPU: {tensor.device}")

    @staticmethod
    def _check_storage_freed(tensor: Any) -> None:
        if int(getattr(tensor, "numel", lambda: 0)()) != 0:
            raise AssertionError("tensor storage is not freed")

    @staticmethod
    def _check_storage_allocated(tensor: Any) -> None:
        if int(getattr(tensor, "numel", lambda: 0)()) == 0:
            raise AssertionError("tensor storage is not allocated")

    def _check_low_precision_shard(self) -> None:
        if getattr(self.flat_param, "_mp_shard", getattr(self, "_mp_shard", None)) is None:
            raise AssertionError("low precision shard is unavailable")

    def _check_unsharded(self, tensor: Any) -> None:
        if tuple(tensor.shape) != tuple(self.flat_param._unpadded_unsharded_size):
            raise AssertionError("tensor is not unsharded")

    def _check_sharded(self, tensor: Any) -> None:
        if tuple(tensor.shape) != tuple(self.flat_param._sharded_size):
            raise AssertionError("tensor is not sharded")

    @property
    def uses_sharded_strategy(self) -> bool:
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def _uses_param_mixed_precision(self) -> bool:
        return self._fwd_bwd_param_dtype != self._orig_param_dtype

    @property
    def _uses_reduce_mixed_precision(self) -> bool:
        return self._reduce_dtype != self._orig_param_dtype

    @property
    def _skipped_use_sharded_views(self) -> bool:
        return self._unsharded_flat_param_for_skipped_views is not None

    def param_module_names(self) -> Iterator[tuple[str, str]]:
        for info in self.flat_param._param_metadata.param_infos:
            yield info.name, info.module_name

    def _get_flat_param_offsets(self) -> tuple[tuple[int, int], ...]:
        offsets = []
        start = 0
        for numel in getattr(
            self.flat_param, "_numels_with_padding", self.flat_param._param_metadata.numels
        ):
            offsets.append((start, start + numel - 1))
            start += numel
        return tuple(offsets)

    def shard_metadata(self) -> FlatParamShardMetadata | None:
        if not getattr(self.flat_param, "_shard_param_infos", None):
            return self._shard_metadata
        names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        strides: list[tuple[int, ...]] = []
        contiguities: list[bool] = []
        numels: list[int] = []
        offsets: list[tuple[int, int]] = []
        for fqn, shape, stride, contiguous, numel, info in zip(
            self.flat_param._fqns,
            self.flat_param._shapes,
            self.flat_param._strides,
            self.flat_param._contiguities,
            self.flat_param._numels,
            self.flat_param._shard_param_infos,
        ):
            if not info.in_shard:
                continue
            names.append(fqn)
            shapes.append(tuple(shape))
            strides.append(tuple(stride))
            contiguities.append(bool(contiguous))
            numels.append(int(numel))
            offsets.append((int(info.intra_param_start_idx or 0), int(info.intra_param_end_idx or 0)))
        return FlatParamShardMetadata(
            tuple(names), tuple(shapes), tuple(strides), tuple(contiguities), tuple(numels), tuple(offsets)
        )

    def flat_param_to(self, *args: Any, **kwargs: Any) -> Any:
        self.flat_param.data = self.flat_param.to(*args, **kwargs)
        if self._use_orig_params:
            if self.is_sharded(self.flat_param):
                self._use_sharded_views()
            else:
                self._use_unsharded_views(as_params=True)
        return self.flat_param

    def __repr__(self) -> str:
        return f"FlatParamHandle(num_params={len(self.params)}, sharded={self.is_sharded()})"


def _unsafe_setattr_param(module: Any, name: str, param: Any) -> None:
    module._buffers.pop(name, None)
    module._parameters[name] = param


def _unsafe_setattr_tensor(module: Any, name: str, tensor: Any) -> None:
    module._parameters.pop(name, None)
    module._buffers[name] = tensor


def _safe_setattr_tensor_or_param(module: Any, name: str, value: Any) -> None:
    if isinstance(value, Parameter):
        _unsafe_setattr_param(module, name, value)
    else:
        _unsafe_setattr_tensor(module, name, value)


def _convert_to_params(module_or_tensors: Any, names: Iterable[str] | None = None) -> Any:
    if names is None:
        return [
            value if isinstance(value, Parameter) else Parameter(value)
            for value in module_or_tensors
        ]
    module = module_or_tensors
    for name in names:
        if name in getattr(module, "_buffers", {}):
            value = module._buffers.pop(name)
            module._parameters[name] = Parameter(
                value, getattr(value, "requires_grad", False)
            )


def _is_truly_contiguous(tensor: Any) -> bool:
    return bool(getattr(tensor, "is_contiguous", lambda: False)())


def _detach_if_needed(tensor: Any) -> Any:
    return tensor.detach() if getattr(tensor, "requires_grad", False) else tensor


def _get_aligned_numel(value: Any, alignment: int | None = None) -> int:
    if alignment is None:
        size = _get_dtype_size(value)
        return max(1, 16 // size)
    return ((int(value) + alignment - 1) // alignment) * alignment


@functools.lru_cache(8)
def _get_dtype_size(dtype: Any) -> int:
    return int(tp.empty((), dtype=dtype).element_size())


def _construct_padding_tensor(
    numel: int, dtype: Any, requires_grad: bool = False, device: Any = None
) -> Any:
    value = tp.ones(
        int(numel), dtype=dtype, device=device, requires_grad=requires_grad
    )
    return value * 42


@functools.lru_cache(1)
def _warn_skip_writeback_check(log: Any, warning: str) -> None:
    log.warning(warning)


@functools.lru_cache(1)
def _warn_use_fake_all_gather(log: Any, warning: str) -> None:
    log.warning(warning)


@functools.lru_cache(1)
def _warn_use_fake_reduce(log: Any, warning: str) -> None:
    log.warning(warning)


def _same_storage(left: Any, right: Any) -> bool:
    return getattr(left, "data_ptr", lambda: None)() == getattr(right, "data_ptr", lambda: None)()


def _same_storage_size(left: Any, right: Any) -> bool:
    if isinstance(right, int):
        return _storage_size_in_numel(left) == right
    return _same_storage(left, right) and _storage_size_in_numel(left) == _storage_size_in_numel(right)


def _storage_size_allocated(tensor: Any) -> bool:
    try:
        return int(tensor.untyped_storage().size()) > 0
    except (AttributeError, RuntimeError):
        return int(tensor.numel()) > 0


def _storage_size_in_numel(tensor: Any) -> int:
    try:
        return int(tensor.untyped_storage().size()) // int(tensor.element_size())
    except (AttributeError, RuntimeError, ZeroDivisionError):
        return int(tensor.numel())


def _alloc_storage(tensor: Any, size: Any) -> None:
    numel = int(size[0]) if isinstance(size, (tuple, list)) else int(size)
    try:
        tensor.untyped_storage().resize_(numel * int(tensor.element_size()))
    except (AttributeError, RuntimeError):
        if int(tensor.numel()) != numel:
            tensor.set_(tensor.new_empty((numel,)))


def _free_storage(tensor: Any) -> None:
    try:
        tensor.untyped_storage().resize_(0)
    except (AttributeError, RuntimeError):
        tensor.set_(tensor.new_empty((0,)))
