"""Parameter state transitions for composable sharding."""

import importlib.util
import inspect
import itertools
import math
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable

import tensorplay as tp
from tensorplay.nn.modules.module import Module
from tensorplay.nn.parameter import Parameter

from ... import distributed_core as dist
from ...device_mesh import DeviceMesh
from ...tensor import DTensor, Replicate, Shard, distribute_tensor
from ...tensor._dtensor_spec import DTensorSpec, TensorMeta
from ...tensor.placement_types import _StridedShard
from ...tensor._collective_utils import pad_tensor, unpad_tensor
from ._fsdp_api import CPUOffloadPolicy
from ._fsdp_common import (
    DataParallelMeshInfo,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    ShardPlacementResult,
    _chunk_with_empty,
    _from_local_no_grad,
    _get_dim_chunked_size,
    resolve_shard_placement,
)

__all__ = [
    "ShardedState",
    "ParamModuleInfo",
    "ExtensionsData",
    "FSDPParam",
    "copy_",
    "copy__functionalize",
    "alloc_storage",
    "free_storage",
    "unsafe_setattr_param",
    "set_requires_grad_if_needed",
]


class ShardedState(Enum):
    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()


@dataclass(init=False)
class ParamModuleInfo:
    module: Any
    fqn: str
    name: str
    shared_modules: list[Any]
    shared_param_names: list[str]

    def __init__(
        self,
        module: Any,
        fqn_or_param_name: str,
        name: str | None = None,
        shared_modules: list[Any] | None = None,
        shared_param_names: list[str] | None = None,
    ) -> None:
        self.module = module
        self.fqn = fqn_or_param_name if name is not None else fqn_or_param_name
        self.name = name or fqn_or_param_name.rsplit(".", 1)[-1]
        self.shared_modules = list(shared_modules or ())
        self.shared_param_names = list(shared_param_names or ())

    @property
    def param_name(self) -> str:
        return self.name


@dataclass
class ExtensionsData:
    value: Any = None
    metadata: Any = None
    all_gather_input_sizes: tuple[Any, ...] = ()

    @property
    def all_gather_metadata(self) -> Any:
        return self.metadata

    @all_gather_metadata.setter
    def all_gather_metadata(self, value: Any) -> None:
        self.metadata = value

    def clear(self) -> None:
        self.value = None
        self.metadata = None
        self.all_gather_input_sizes = ()


_orig_param_uid_counter = itertools.count()


def _get_orig_param_uid(param: Any) -> int:
    if not hasattr(param, "_fsdp_orig_uid"):
        param._fsdp_orig_uid = next(_orig_param_uid_counter)
    return param._fsdp_orig_uid


@lru_cache(maxsize=1)
def _get_spmd_support() -> tuple[Any, Any, Any, Any] | None:
    if importlib.util.find_spec("spmd_types") is None:
        return None
    import spmd_types as spmd
    from spmd_types._mesh_axis import flatten_axes
    from spmd_types.runtime import get_partition_spec
    from spmd_types.types import partition_spec_to_shard_types

    return spmd, flatten_axes, get_partition_spec, partition_spec_to_shard_types


def _spans_same_mesh(
    lhs_axes: Any,
    rhs_axes: Any,
    spmd: Any = None,
    flatten_axes: Any = None,
) -> bool:
    if spmd is None or flatten_axes is None:
        return tuple(lhs_axes) == tuple(rhs_axes)
    lhs_mesh = spmd.normalize_mesh(frozenset(lhs_axes))
    rhs_mesh = spmd.normalize_mesh(frozenset(rhs_axes))
    if not lhs_mesh or not rhs_mesh:
        return lhs_mesh == rhs_mesh
    return flatten_axes(tuple(lhs_mesh)) == flatten_axes(tuple(rhs_mesh))


def _mesh_placements(mesh: Any, placement: Any, mesh_dim: int | str = 0) -> tuple[Any, ...]:
    value = getattr(mesh, "ndim")
    ndim = int(value() if callable(value) else value)
    if isinstance(mesh_dim, str):
        names = getattr(mesh, "mesh_dim_names", None)
        if names is None or mesh_dim not in names:
            raise KeyError(mesh_dim)
        mesh_dim = names.index(mesh_dim)
    mesh_dim = int(mesh_dim)
    if mesh_dim < 0:
        mesh_dim += ndim
    if mesh_dim < 0 or mesh_dim >= ndim:
        raise ValueError("shard mesh dimension is outside the mesh")
    placements = [Replicate() for _ in range(ndim)]
    placements[mesh_dim] = placement
    return tuple(placements)


def copy_(tensor: Any, data: Any) -> Any:
    tensor.copy_(data)
    return tensor


def copy__functionalize(tensor: Any, data: Any) -> Any:
    return copy_(tensor, data)


class FSDPParam:
    """Owns one logical parameter and its local sharded representation."""

    def __init__(
        self,
        param: Any,
        module_info: ParamModuleInfo,
        mesh_info: DataParallelMeshInfo,
        post_forward_mesh_info: DataParallelMeshInfo | None = None,
        device: Any = None,
        shard_placement_fn: Callable[[Any], Any] | None = None,
        mp_policy: Any = None,
        offload_policy: Any = None,
    ) -> None:
        self.param = param
        self.module_info = module_info
        self._module_info = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.offload_to_cpu = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = bool(
            self.offload_to_cpu and getattr(offload_policy, "pin_memory", False)
        )
        self.grad_offload_event: Any = None
        self.param_requires_grad = bool(getattr(param, "requires_grad", False))
        self._state = ShardedState.UNSHARDED
        self._full_tensor = param
        self._sharded_tensor: DTensor | None = None
        self._sharded_post_forward_tensor: DTensor | None = None
        self._sharded_grad: Any = None
        self._extensions = ExtensionsData()
        self._gradient_sync_owner: Any = None
        self._gradient_hook_param: Any = None
        self._gradient_hook_handles: list[Any] = []
        self._gradient_hook_nodes: list[Any] = []
        self._full_gradient_hook_handle: Any = None
        self._full_gradient_hook_node: Any = None
        self._unsharded_grad: Any = None
        self._unsharded_param: Any = None
        self._unsharded_inner_tensors: list[Any] = []
        self._release_all_gather_outputs_after_post_all_gather = False
        self.all_gather_outputs: list[Any] = []
        self._all_gather_outputs_ready = False
        self.unsharded_accumulated_grad: Any = None
        self._orig_param_uid = _get_orig_param_uid(param)
        self.orig_dtype = getattr(param, "dtype", None)
        self._compute_dtype = self.orig_dtype
        self._reduce_dtype = self.orig_dtype
        self.param_dtype = None
        self.reduce_dtype = None
        self._placement, self._mesh = self._resolve_placement(
            param, mesh_info, shard_placement_fn
        )
        self.fsdp_placement = self._placement
        self._shard_mesh = self._init_shard_mesh()
        self._init_sharded_param(param, device, shard_placement_fn, mesh_info)
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )

    def _resolve_placement(
        self,
        param: Any,
        mesh_info: DataParallelMeshInfo,
        shard_placement_fn: Callable[[Any], Any] | None,
    ) -> tuple[Any, Any]:
        result = shard_placement_fn(param) if callable(shard_placement_fn) else None
        if result is None and mesh_info.shard_mesh_dim is None:
            return Replicate(), mesh_info.mesh
        if isinstance(result, ShardPlacementResult) and result.mesh_info is not None:
            self.mesh_info = result.mesh_info
        placement, mesh = resolve_shard_placement(result, self.mesh_info)
        return placement, mesh

    @staticmethod
    def _mesh_ndim(mesh: Any) -> int:
        ndim = getattr(mesh, "ndim", None)
        ndim = ndim() if callable(ndim) else ndim
        return int(ndim)

    @staticmethod
    def _contiguous_strides(shape: Any) -> tuple[int, ...]:
        values = [int(value) for value in shape]
        strides = [1] * len(values)
        running = 1
        for index in range(len(values) - 1, -1, -1):
            strides[index] = running
            running *= values[index]
        return tuple(strides)

    def _shard_count_and_rank(self, mesh_info: DataParallelMeshInfo) -> tuple[int, int]:
        mesh_dim = mesh_info.shard_mesh_dim
        if mesh_dim is None or not isinstance(self._placement, Shard):
            return 1, 0
        count = int(mesh_info.mesh.size(mesh_dim))
        rank = getattr(mesh_info, "shard_mesh_rank", None)
        if rank is None:
            rank = mesh_info.mesh.get_local_rank(mesh_dim)
        return count, int(rank)

    def _placement_list(self, mesh: Any, placement: Any, mesh_dim: Any) -> tuple[Any, ...]:
        ndim = self._mesh_ndim(mesh)
        if mesh_dim is None or isinstance(placement, Replicate):
            return tuple(Replicate() for _ in range(ndim))
        return _mesh_placements(mesh, placement, mesh_dim)

    def _make_sharded_storage(self, param_data: Any, mesh_info: DataParallelMeshInfo) -> Any:
        shard_dim = int(self._placement.dim) if isinstance(self._placement, Shard) else 0
        world_size, rank = self._shard_count_and_rank(mesh_info)
        if isinstance(self._placement, Shard) and shard_dim > 0:
            if int(param_data.shape[shard_dim]) % world_size:
                raise NotImplementedError(
                    f"uneven sharding is unsupported on dimension {shard_dim}"
                )
        chunks = _chunk_with_empty(param_data, world_size, dim=shard_dim)
        selected = chunks[rank]
        self.sharded_size = tuple(int(value) for value in selected.shape)
        self.contiguous_sharded_stride = self._contiguous_strides(self.sharded_size)
        padded_size = tuple(int(value) for value in chunks[0].shape)
        self.padded_sharded_param_size = padded_size
        padded = param_data.new_zeros(padded_size)
        if int(selected.numel()) > 0:
            length = int(selected.shape[shard_dim])
            padded.narrow(shard_dim, 0, length).copy_(selected)
        if self.offload_to_cpu and getattr(padded, "device", None) != "cpu":
            padded = padded.cpu()
        if self.pin_memory and not getattr(padded, "is_pinned", lambda: False)():
            padded = padded.pin_memory()
        self._sharded_param_data = padded.view(-1)
        length = int(selected.shape[shard_dim]) if int(selected.numel()) else 0
        local = padded.narrow(shard_dim, 0, length)
        if hasattr(local, "contiguous"):
            local = local.contiguous()
        self._sharded_tensor = self.to_sharded_dtensor(local)
        self.sharded_param = Parameter(local, requires_grad=self.param_requires_grad)
        self._sharded_post_forward_tensor = None
        self._sharded_post_forward_param_data = None
        self._sharded_post_forward_param = None
        self._setattr_on_modules(self.sharded_param)
        return local

    @tp.no_grad()
    def _init_sharded_param(
        self,
        param: Any,
        device: Any,
        shard_placement_fn: Any,
        mesh_info: DataParallelMeshInfo,
    ) -> None:
        del device, shard_placement_fn
        spmd_support = _get_spmd_support()
        self.is_spmd_types = False
        if spmd_support is not None and not isinstance(param, DTensor):
            spmd = spmd_support[0]
            get_partition_spec = spmd_support[2]
            get_local_type = getattr(spmd, "get_local_type", None)
            init_local_type = (
                get_local_type(param) if callable(get_local_type) else None
            )
            if init_local_type:
                self.is_spmd_types = True
                param = self._resolve_spmd_types_for_storage(
                    param,
                    get_partition_spec(param),
                    init_local_type,
                    mesh_info,
                )
                self.param = param
        if isinstance(param, DTensor):
            param_data = param.to_local()
            self._unsharded_dtensor_spec = DTensorSpec(
                param.device_mesh,
                param.placements,
                TensorMeta(tuple(param.shape), tuple(param.stride()), param.dtype),
            )
        else:
            param_data = param
            self._unsharded_dtensor_spec = None
        self.is_dtensor = isinstance(param, DTensor)
        if not getattr(param_data, "is_contiguous", lambda: True)():
            raise NotImplementedError(
                f"non-contiguous parameters are unsupported: shape={param_data.shape}"
            )
        if isinstance(self._placement, Shard):
            shard_dim = int(self._placement.dim)
            if shard_dim < 0:
                shard_dim += int(param_data.dim())
            if shard_dim < 0 or shard_dim >= int(param_data.dim()):
                raise ValueError(
                    f"shard dimension {self._placement.dim} is outside parameter rank"
                )
            self._placement = type(self._placement)(
                shard_dim,
                getattr(self._placement, "split_factor", 1),
            ) if isinstance(self._placement, _StridedShard) else Shard(shard_dim)
            self.fsdp_placement = self._placement
        self._orig_size = tuple(int(value) for value in param_data.shape)
        self._contiguous_orig_stride = self._contiguous_strides(self._orig_size)
        self._init_sharding_spec(param, self._placement, getattr(self._placement, "dim", 0))
        self._make_sharded_storage(param_data.detach(), self.mesh_info)
        self._init_sharded_post_forward_param_metadata(param_data)
        self._init_extensions()
        self._full_tensor = None
        self._state = ShardedState.SHARDED

    def _resolve_spmd_types_for_storage(
        self,
        param: Any,
        partition_spec: Any,
        init_local_type: Any,
        mesh_info: Any,
    ) -> Any:
        spmd_support = _get_spmd_support()
        if spmd_support is None:
            raise RuntimeError("type-check metadata support is unavailable")
        spmd, flatten_axes, _, partition_spec_to_shard_types = spmd_support
        storage_mesh = getattr(mesh_info, "spmd_mesh", None)
        storage_mesh_names = getattr(storage_mesh, "mesh_dim_names", None)
        if storage_mesh is None or storage_mesh_names is None:
            raise ValueError(
                "type-check annotated parameters require a named full storage mesh "
                "and data-parallel mesh dimensions"
            )

        storage_mesh_axes = tuple(
            (name, spmd.MeshAxis.of(storage_mesh.get_group(name)))
            for name in storage_mesh_names
        )
        storage_axes = tuple(axis for _, axis in storage_mesh_axes)
        local_type = dict(init_local_type)
        if partition_spec is not None:
            local_type.update(partition_spec_to_shard_types(partition_spec))
        annotated_axes = tuple(local_type.keys())
        if _spans_same_mesh(
            annotated_axes,
            storage_axes,
            spmd,
            flatten_axes,
        ):
            restore_mesh = annotated_axes
        else:
            current_mesh = spmd.current_mesh()
            if current_mesh is None:
                raise ValueError(
                    f"parameter '{self.module_info.param_name}' has partial type-check "
                    "metadata that cannot be restored from the storage mesh"
                )
            restore_mesh = tuple(current_mesh)

        unknown_axes = tuple(
            axis for axis in init_local_type if axis not in restore_mesh
        )
        if unknown_axes:
            raise ValueError(
                f"parameter '{self.module_info.param_name}' has metadata axes "
                f"outside the compute mesh: {unknown_axes}"
            )
        if not _spans_same_mesh(
            restore_mesh,
            storage_axes,
            spmd,
            flatten_axes,
        ):
            raise ValueError(
                f"parameter '{self.module_info.param_name}' uses a compute mesh "
                "with a different rank set from the storage mesh"
            )

        dp_mesh_dims = getattr(mesh_info, "dp_mesh_dims", None)
        if dp_mesh_dims is None:
            raise ValueError("type-check annotated parameters require DP mesh dimensions")
        dp_names = set(
            itertools.chain(
                dp_mesh_dims.shard_names,
                dp_mesh_dims.replicate_names,
            )
        )
        fsdp_axis = flatten_axes(
            tuple(axis for name, axis in storage_mesh_axes if name in dp_names)
        )
        non_fsdp_storage_mesh_axes = {
            axis for name, axis in storage_mesh_axes if name not in dp_names
        }
        storage_axis_types = {
            axis: spmd.R for name, axis in storage_mesh_axes if name in dp_names
        }
        for axis, axis_type in local_type.items():
            if axis <= fsdp_axis:
                if axis_type is not spmd.R:
                    raise ValueError(
                        f"expected replicated metadata on data-parallel axis {axis}"
                    )
            else:
                if axis not in non_fsdp_storage_mesh_axes:
                    raise ValueError(
                        f"metadata axis {axis} is not a storage mesh axis"
                    )
                storage_axis_types[axis] = axis_type
        if set(storage_axes) != set(storage_axis_types):
            raise ValueError(
                f"parameter '{self.module_info.param_name}' has incomplete "
                "type-check metadata for the storage mesh"
            )

        restore_type = {
            axis: init_local_type.get(axis, spmd.R) for axis in restore_mesh
        }
        placements = []
        grad_placements = []
        for axis_type in storage_axis_types.values():
            placements.append(spmd.spmd_type_to_dtensor_placement(axis_type))
            grad_placements.append(
                spmd.spmd_type_to_dtensor_placement(axis_type.backward_type())
            )
        dtensor_param = DTensor.from_local(
            getattr(param, "data", param),
            storage_mesh,
            placements,
            run_check=False,
        )
        self._spmd_partition_spec = partition_spec
        self._spmd_init_local_type = init_local_type
        self._spmd_restore_mesh = tuple(restore_mesh)
        self._spmd_restore_type = restore_type
        self._spmd_grad_placements = tuple(grad_placements)
        return dtensor_param

    def _restore_spmd_types(self, tensor: Any) -> Any:
        if not getattr(self, "is_spmd_types", False):
            return None
        spmd_support = _get_spmd_support()
        if spmd_support is None or not spmd_support[0].is_type_checking():
            return None
        spmd = spmd_support[0]
        spmd.assert_type(
            tensor,
            self._spmd_restore_type,
            partition_spec=self._spmd_partition_spec,
        )
        return None

    def _init_sharding_spec(self, param: Any, fsdp_placement: Any, shard_dim: int) -> Any:
        if self.is_dtensor:
            self._unsharded_dtensor_spec = DTensorSpec(
                param.device_mesh,
                param.placements,
                TensorMeta(tuple(param.shape), tuple(param.stride()), param.dtype),
            )
        else:
            self._unsharded_dtensor_spec = None
        if self.mesh_info.is_spmd_mesh and not self.is_dtensor:
            raise ValueError(
                "When dp_mesh_dims is provided, every parameter must be a distributed "
                "tensor on the full SPMD mesh. "
                f"Got plain tensor for parameter '{self.module_info.param_name}'."
            )
        if self.is_dtensor and self.mesh_info.is_spmd_mesh:
            return self._init_sharding_spec_spmd(param, fsdp_placement, shard_dim)
        if self.is_dtensor:
            return self._init_sharding_spec_tp(param, fsdp_placement, shard_dim)
        return self._init_sharding_spec_plain(param, fsdp_placement)

    def _init_sharding_spec_spmd(
        self, param: Any, fsdp_placement: Any, shard_dim: int
    ) -> Any:
        if self._unsharded_dtensor_spec is None:
            raise AssertionError("distributed parameter metadata is missing")
        spmd_mesh = self._unsharded_dtensor_spec.mesh
        dp_dim_names = self.mesh_info.dp_mesh_dims
        if dp_dim_names is None:
            raise AssertionError("data-parallel mesh dimensions are missing")
        mesh_dim_names = getattr(spmd_mesh, "mesh_dim_names", None)
        if mesh_dim_names is None:
            raise AssertionError("an SPMD parameter mesh needs dimension names")
        if (
            self.mesh_info.spmd_mesh is not None
            and spmd_mesh is not self.mesh_info.spmd_mesh
        ):
            raise ValueError(
                "the parameter distributed mesh must be the full mesh passed to fully_shard"
            )

        dp_shard_indices = [
            mesh_dim_names.index(name) for name in dp_dim_names.shard_names
        ]
        original_placements = self._unsharded_dtensor_spec.placements
        for index in dp_shard_indices:
            if not isinstance(original_placements[index], Replicate):
                raise ValueError(
                    f"data-parallel shard dimension '{mesh_dim_names[index]}' "
                    f"must be replicated, got {original_placements[index]}"
                )
        dp_replicate_indices = []
        for name in dp_dim_names.replicate_names:
            index = mesh_dim_names.index(name)
            dp_replicate_indices.append(index)
            if not isinstance(original_placements[index], Replicate):
                raise ValueError(
                    f"data-parallel replicate dimension '{mesh_dim_names[index]}' "
                    f"must be replicated, got {original_placements[index]}"
                )
        self._dp_dim_indices = frozenset(dp_shard_indices + dp_replicate_indices)

        placements = list(original_placements)
        for dp_index in dp_shard_indices:
            split_factor = 1
            for mesh_index in range(dp_index + 1, int(spmd_mesh.ndim)):
                placement = original_placements[mesh_index]
                if (
                    isinstance(placement, (Shard, _StridedShard))
                    and placement.dim == shard_dim
                ):
                    split_factor *= int(spmd_mesh.size(mesh_index))
            placements[dp_index] = (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else fsdp_placement
            )

        self._spmd_mesh = spmd_mesh
        self._spmd_placements = tuple(placements)
        self._sharding_spec = self._build_spmd_sharding_spec(
            dp_dim_names, dp_shard_indices, fsdp_placement
        )
        return param.to_local()

    def _build_spmd_sharding_spec(
        self, dp_dim_names: Any, dp_shard_indices: list[int], fsdp_placement: Any
    ) -> DTensorSpec:
        del fsdp_placement
        if self._unsharded_dtensor_spec is None:
            raise AssertionError("distributed parameter metadata is missing")
        tensor_meta = self._unsharded_dtensor_spec.tensor_meta
        if len(dp_shard_indices) <= 1:
            return DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=tensor_meta,
            )
        mesh_dim_names = getattr(self._spmd_mesh, "mesh_dim_names", None)
        if mesh_dim_names is None:
            raise AssertionError("an SPMD parameter mesh needs dimension names")
        shard_names = set(dp_dim_names.shard_names)
        replicate_names = set(dp_dim_names.replicate_names)
        submeshes = []
        spec_placements = []
        skip = 0
        for index, name in enumerate(mesh_dim_names):
            if skip > 0:
                skip -= 1
                continue
            if name in shard_names:
                submeshes.append(self.mesh_info.mesh)
                if isinstance(self.mesh_info, HSDPMeshInfo):
                    spec_placements.append(Replicate())
                spec_placements.append(self._spmd_placements[index])
                skip = len(dp_dim_names.shard_names) - 1
            elif name in replicate_names and isinstance(self.mesh_info, HSDPMeshInfo):
                continue
            else:
                submeshes.append(self._spmd_mesh[name])
                spec_placements.append(self._spmd_placements[index])
        spec_mesh = DeviceMesh._concatenate(submeshes)
        return DTensorSpec(spec_mesh, tuple(spec_placements), tensor_meta=tensor_meta)

    def _init_sharding_spec_tp(
        self, param: Any, fsdp_placement: Any, shard_dim: int
    ) -> Any:
        if self._unsharded_dtensor_spec is None:
            raise AssertionError("distributed parameter metadata is missing")
        dp_mesh = self.mesh_info.mesh
        tp_mesh = self._unsharded_dtensor_spec.mesh
        if dp_mesh is None or tp_mesh is None:
            raise AssertionError("data-parallel and model-parallel meshes are required")
        self._spmd_mesh = DeviceMesh._concatenate([dp_mesh, tp_mesh])
        if len(self._unsharded_dtensor_spec.placements) > 2:
            raise NotImplementedError(
                "only one-dimensional model parallel placement or a two-dimensional "
                f"model parallel placement is supported, got {self._unsharded_dtensor_spec.placements}"
            )
        split_factor = self._unsharded_dtensor_spec.num_shards_map[shard_dim]
        if not 2 <= int(self._spmd_mesh.ndim) <= 4:
            raise AssertionError(
                "the combined data-parallel/model-parallel mesh must have between "
                f"2 and 4 dimensions, got {self._spmd_mesh.ndim}"
            )
        if isinstance(self.mesh_info, FSDPMeshInfo):
            dp_shard_tp_placements = (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else fsdp_placement,
                *self._unsharded_dtensor_spec.placements,
            )
        else:
            dp_shard_tp_placements = (
                Replicate(),
                *self._unsharded_dtensor_spec.placements,
            )
        if isinstance(self.mesh_info, HSDPMeshInfo):
            if self.mesh_info.replicate_mesh_dim != 0:
                raise AssertionError(
                    "the HSDP replicate mesh dimension must be zero"
                )
            self._spmd_placements = (Replicate(),) + dp_shard_tp_placements
        else:
            self._spmd_placements = dp_shard_tp_placements
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=self._unsharded_dtensor_spec.tensor_meta,
        )
        return param.to_local()

    def _init_sharding_spec_plain(self, param: Any, fsdp_placement: Any) -> Any:
        self._spmd_mesh = self.mesh_info.mesh
        if isinstance(self.mesh_info, HSDPMeshInfo):
            self._spmd_placements = (Replicate(), fsdp_placement)
        elif isinstance(self.mesh_info, FSDPMeshInfo):
            self._spmd_placements = (fsdp_placement,)
        elif isinstance(self.mesh_info, DDPMeshInfo):
            self._spmd_placements = (Replicate(),)
        else:
            raise TypeError(f"unsupported data-parallel mesh info {type(self.mesh_info)!r}")
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=TensorMeta(
                tuple(param.shape), tuple(param.stride()), param.dtype
            ),
        )
        return param

    def _init_sharded_post_forward_param_metadata(self, param: Any) -> None:
        if self.post_forward_mesh_info is None:
            self.sharded_post_forward_size = self.sharded_size
            self.contiguous_sharded_post_forward_stride = self.contiguous_sharded_stride
            return
        count, rank = self._shard_count_and_rank(self.post_forward_mesh_info)
        dim = int(self._placement.dim) if isinstance(self._placement, Shard) else 0
        chunks = _chunk_with_empty(param, count, dim=dim)
        self.sharded_post_forward_size = tuple(
            int(value) for value in chunks[rank].shape
        )
        self.contiguous_sharded_post_forward_stride = self._contiguous_strides(
            self.sharded_post_forward_size
        )
        self._post_forward_shape = tuple(param.shape)

    def init_dtype_attrs(self, mp_policy: Any) -> None:
        self.orig_dtype = self.sharded_param.dtype
        param_dtype = getattr(mp_policy, "param_dtype", None)
        reduce_dtype = getattr(mp_policy, "reduce_dtype", None)
        if param_dtype == self.orig_dtype or not getattr(
            self.sharded_param, "is_floating_point", lambda: False
        )():
            param_dtype = None
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self._compute_dtype = param_dtype or self.orig_dtype
        self._reduce_dtype = reduce_dtype

    def _init_extensions(self) -> None:
        local = self._sharded_local_tensor()
        has_pre = callable(getattr(local, "fsdp_pre_all_gather", None))
        has_post = callable(getattr(local, "fsdp_post_all_gather", None))
        should_release = getattr(
            local,
            "fsdp_should_release_all_gather_outputs_after_post_all_gather",
            None,
        )
        if has_pre != has_post:
            raise AssertionError(
                "pre and post all-gather extension methods must be provided together"
            )
        if should_release is not None and not has_post:
            raise AssertionError(
                "all-gather output release requires a post all-gather extension"
            )
        self._release_all_gather_outputs_after_post_all_gather = False
        self._extensions = ExtensionsData()
        if callable(should_release):
            value = should_release()
            if not isinstance(value, bool):
                raise AssertionError("all-gather output release flag must be boolean")
            if (
                value
                and self.post_forward_mesh_info is not None
                and self.post_forward_mesh_info != self.mesh_info
            ):
                raise NotImplementedError(
                    "all-gather output release is unavailable for a different post-forward mesh"
                )
            self._release_all_gather_outputs_after_post_all_gather = value

    def init_all_gather_outputs(self, all_gather_input_numels: Any, all_gather_input_dtypes: Any, world_size: int, device: Any) -> None:
        if self.all_gather_outputs:
            return
        if len(all_gather_input_numels) != len(all_gather_input_dtypes):
            raise ValueError("all-gather sizes and dtypes must have the same length")
        self.all_gather_outputs = [
            tp.empty(int(numel) * int(world_size), device=device, dtype=dtype)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]
        self._all_gather_output = self.all_gather_outputs[0] if self.all_gather_outputs else None

    def init_unsharded_param(self) -> Any:
        self._all_gather_outputs_ready = True
        local = self._sharded_local_tensor()
        post_all_gather = getattr(local, "fsdp_post_all_gather", None)
        if callable(post_all_gather):
            all_gather_outputs = self._unflatten_all_gather_outputs()
            if all_gather_outputs is None:
                raise RuntimeError("all-gather outputs are unavailable")
            if self._unsharded_param is None:
                result = post_all_gather(
                    all_gather_outputs,
                    self._extensions.metadata,
                    self.param_dtype or self.orig_dtype,
                )
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(
                        "fsdp_post_all_gather must return a tensor and inner tensors"
                    )
                unsharded_tensor, inner_tensors = result
                self._unsharded_inner_tensors = list(inner_tensors or ())
                self._set_unsharded_tensor(unsharded_tensor)
                self._register_full_gradient_hook(self._unsharded_param)
                self._state = ShardedState.UNSHARDED
            else:
                for tensor in self._unsharded_inner_tensors:
                    alloc_storage(tensor)
                post_all_gather(
                    all_gather_outputs,
                    self._extensions.metadata,
                    self.param_dtype or self.orig_dtype,
                    out=self._unsharded_param,
                )
            self._extensions.clear()
            self._release_all_gather_outputs_if_needed()
            return self._full_tensor
        if len(self.all_gather_outputs) > 1:
            raise ValueError("default all-gather requires one output tensor")
        if self.all_gather_outputs:
            gathered = self._attach_local_gradient_to_all_gather(
                self.all_gather_outputs[0]
            )
            self._set_unsharded_tensor(gathered)
            self._register_full_gradient_hook(self._unsharded_param)
            self._state = ShardedState.UNSHARDED
            self._release_all_gather_outputs_if_needed()
        else:
            self.to_unsharded()
        return self._full_tensor

    def _release_all_gather_outputs_if_needed(self) -> None:
        if self._release_all_gather_outputs_after_post_all_gather:
            self.free_all_gather_outputs()

    def _get_unsharded_dtensor_spec(self, unsharded_param: Any) -> Any:
        if self._unsharded_dtensor_spec is not None:
            return DTensorSpec(
                self._unsharded_dtensor_spec.mesh,
                self._unsharded_dtensor_spec.placements,
                TensorMeta(
                    tuple(unsharded_param.shape),
                    tuple(unsharded_param.stride()),
                    unsharded_param.dtype,
                ),
            )
        mesh = self.mesh_info.mesh
        ndim = getattr(mesh, "ndim", 1)
        ndim = int(ndim() if callable(ndim) else ndim)
        return DTensorSpec(
            mesh,
            tuple(Replicate() for _ in range(ndim)),
            TensorMeta(
                tuple(unsharded_param.shape),
                tuple(unsharded_param.stride()),
                unsharded_param.dtype,
            ),
        )

    def _unflatten_all_gather_outputs(self) -> Any:
        if not self.all_gather_outputs:
            return None
        if self._extensions.all_gather_input_sizes:
            values = []
            for output, size in zip(
                self.all_gather_outputs, self._extensions.all_gather_input_sizes
            ):
                shape = tuple(size)
                values.append(output.view(-1, *shape[1:]))
            return tuple(values)
        return self.all_gather_outputs[0]

    def _set_unsharded_tensor(self, tensor: Any) -> Any:
        if isinstance(tensor, tuple):
            if len(tensor) != 1:
                raise ValueError("one parameter must have one gathered tensor")
            tensor = tensor[0]
        if tuple(tensor.shape) != self._orig_size:
            tensor = tp.as_strided(
                tensor,
                self._orig_size,
                self._contiguous_orig_stride,
                storage_offset=0,
            )
        unsharded_param = Parameter(tensor, requires_grad=self.param_requires_grad)
        self._unsharded_param = unsharded_param
        self._full_tensor = unsharded_param
        self._setattr_on_modules(unsharded_param)
        return unsharded_param

    def _attach_local_gradient_to_all_gather(self, gathered: Any) -> Any:
        mesh_info = self._active_mesh_info()
        mesh_dim = mesh_info.shard_mesh_dim
        if mesh_dim is None or not isinstance(self._placement, Shard):
            return gathered
        count = int(mesh_info.mesh.size(mesh_dim))
        if count <= 1:
            return gathered
        gathered = gathered.reshape(-1)
        shard_dim = int(self._placement.dim)
        width = int(gathered.numel()) // count
        padded_size = tuple(int(value) for value in self.padded_sharded_param_size)
        if math.prod(padded_size) != width:
            raise RuntimeError("all-gather slot does not match the padded shard")
        local = self._sharded_local_tensor()
        if getattr(local, "dtype", None) != getattr(gathered, "dtype", None):
            local = local.to(dtype=gathered.dtype)
        if int(local.numel()) > width:
            raise RuntimeError("local shard is larger than the all-gather slot")
        if shard_dim == 0:
            local = local.reshape(-1)
            if int(local.numel()) < width:
                local = tp.cat((local, local.new_zeros(width - int(local.numel()))), dim=0)
            pieces = [
                gathered.narrow(0, index * width, width).detach()
                for index in range(count)
            ]
            rank = int(mesh_info.mesh.get_local_rank(mesh_dim))
            pieces[rank] = local
            return tp.cat(tuple(pieces), dim=0)
        local_shape = tuple(int(value) for value in local.shape)
        if local_shape != padded_size:
            if any(
                left != right
                for index, (left, right) in enumerate(zip(local_shape, padded_size))
                if index != shard_dim
            ) or local_shape[shard_dim] > padded_size[shard_dim]:
                raise RuntimeError("local shard shape does not match the padded shard")
            padding_shape = list(local_shape)
            padding_shape[shard_dim] = padded_size[shard_dim] - local_shape[shard_dim]
            local = tp.cat((local, local.new_zeros(tuple(padding_shape))), dim=shard_dim)
        pieces = [
            gathered.narrow(0, index * width, width).detach().view(padded_size)
            for index in range(count)
        ]
        rank = int(mesh_info.mesh.get_local_rank(mesh_dim))
        pieces[rank] = local
        return tp.cat(tuple(pieces), dim=shard_dim)

    def to_sharded(self) -> None:
        if self._state == ShardedState.SHARDED:
            self._setattr_on_modules(self.sharded_param)
            return
        if self._state == ShardedState.SHARDED_POST_FORWARD:
            self._sharded_post_forward_tensor = None
            self._sharded_post_forward_param_data = None
            self._sharded_post_forward_param = None
            self._setattr_on_modules(self.sharded_param)
            self._state = ShardedState.SHARDED
            return
        full_tensor = (
            self._unsharded_param
            if self._unsharded_param is not None
            else self._full_tensor
        )
        if full_tensor is None:
            raise RuntimeError("unsharded parameter storage is unavailable")
        if isinstance(full_tensor, DTensor):
            full_tensor = full_tensor.to_local()
        self._make_sharded_storage(full_tensor.detach(), self.mesh_info)
        self._unsharded_param = None
        self._full_tensor = None
        self._sharded_post_forward_tensor = None
        self._sharded_post_forward_param_data = None
        self._sharded_post_forward_param = None
        self._setattr_on_modules(self.sharded_param)
        self._state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        if self.post_forward_mesh_info is None:
            raise RuntimeError("post-forward mesh information is not configured")
        if self.post_forward_mesh_info is self.mesh_info:
            self.to_sharded()
            self._sharded_post_forward_tensor = self._sharded_tensor
            self._sharded_post_forward_param_data = self._sharded_param_data
            self._sharded_post_forward_param = self.sharded_param
            self._state = ShardedState.SHARDED_POST_FORWARD
            return
        if self._state == ShardedState.SHARDED_POST_FORWARD:
            return
        if self._state != ShardedState.UNSHARDED:
            raise RuntimeError(
                f"cannot reshard parameter from state {self._state!r}"
            )
        full_tensor = (
            self._unsharded_param
            if self._unsharded_param is not None
            else self._full_tensor
        )
        if full_tensor is None:
            raise RuntimeError("unsharded parameter storage is unavailable")
        if isinstance(full_tensor, DTensor):
            full_tensor = full_tensor.to_local()
        if not hasattr(self._placement, "_split_tensor"):
            raise RuntimeError("post-forward resharding requires a shard placement")
        chunks, pads = self._placement._split_tensor(
            full_tensor.detach(),
            self.post_forward_mesh_info.shard_world_size,
            with_padding=True,
            contiguous=True,
        )
        rank = self.post_forward_mesh_info.shard_mesh_rank
        local = self._placement._maybe_unpad_tensor_with_sizes(
            self._placement.dim,
            chunks[rank],
            pads,
            rank,
            True,
        )
        self._sharded_post_forward_param_data = chunks[rank].view(-1)
        self._sharded_post_forward_tensor = self.to_sharded_post_forward_dtensor(local)
        self._sharded_post_forward_param = Parameter(
            local, requires_grad=self.param_requires_grad
        )
        self._unsharded_param = None
        self._full_tensor = None
        self._setattr_on_modules(self._sharded_post_forward_param)
        self._state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        if self._state == ShardedState.UNSHARDED:
            return
        local_param = self._sharded_local_tensor()
        self._gradient_hook_param = local_param
        mesh_info = self._active_mesh_info()
        gathered = self._gather_with_local_gradient(local_param, mesh_info)
        if self._all_gather_outputs_ready and self.all_gather_outputs:
            gathered_outputs = self._unflatten_all_gather_outputs()
            if gathered_outputs is not None:
                candidate = gathered_outputs
                if isinstance(candidate, tuple):
                    candidate = candidate[0]
                if int(candidate.numel()) >= math.prod(self._orig_size):
                    gathered = self._attach_local_gradient_to_all_gather(candidate)
        self._set_unsharded_tensor(gathered)
        self._register_full_gradient_hook(self._unsharded_param)
        if self._compute_dtype is not None and self._compute_dtype != getattr(
            self._unsharded_param, "dtype", None
        ):
            self._unsharded_param = self._unsharded_param.to(dtype=self._compute_dtype)
            self._full_tensor = self._unsharded_param
            self._setattr_on_modules(self._unsharded_param)
        self._release_all_gather_outputs_if_needed()
        self._sharded_post_forward_tensor = None
        self._sharded_post_forward_param = None
        self._sharded_post_forward_param_data = None
        self._state = ShardedState.UNSHARDED

    def _register_full_gradient_hook(self, tensor: Any) -> None:
        if not getattr(tensor, "requires_grad", False):
            return
        if getattr(self, "_full_gradient_hook_source", None) is tensor:
            return
        register_hook = getattr(tensor, "register_hook", None)
        if register_hook is None:
            return
        node = (
            getattr(tensor, "_accumulate_grad_node", None)
            if getattr(tensor, "is_leaf", False)
            else getattr(tensor, "grad_fn", None)
        )
        def capture(gradient: Any, source: Any = tensor) -> Any:
            return self._capture_full_gradient(gradient, source)

        self._full_gradient_hook_handle = register_hook(capture)
        self._full_gradient_hook_node = node
        self._full_gradient_hook_source = tensor

    def _capture_full_gradient(self, gradient: Any, source: Any = None) -> Any:
        self._unsharded_grad = gradient
        if source is None:
            source = self._full_tensor
        current = self.module_info.module._parameters.get(self.module_info.name)
        if current is None:
            return gradient
        if current is source and getattr(current, "is_leaf", False):
            return gradient

        if tuple(current.shape) == tuple(gradient.shape):
            local_gradient = gradient
        else:
            if not hasattr(self._placement, "dim"):
                current.grad = gradient.detach().clone()
                return gradient
            dim = int(self._placement.dim)
            if dim < 0:
                dim += int(gradient.dim())
            mesh_info = self._active_mesh_info()
            count = int(mesh_info.mesh.size(mesh_info.shard_mesh_dim))
            width = (int(self.param.shape[dim]) + count - 1) // count
            rank = int(
                mesh_info.mesh.get_local_rank(mesh_info.shard_mesh_dim)
            )
            start = min(rank * width, int(self.param.shape[dim]))
            length = int(current.shape[dim])
            slices = [slice(None)] * int(gradient.dim())
            slices[dim] = slice(start, start + length)
            local_gradient = gradient[tuple(slices)]
        current.grad = local_gradient.detach().clone()
        return gradient

    def _active_mesh_info(self) -> DataParallelMeshInfo:
        if (
            self._state == ShardedState.SHARDED_POST_FORWARD
            and self.post_forward_mesh_info is not None
        ):
            return self.post_forward_mesh_info
        return self.mesh_info

    def _gather_with_local_gradient(
        self,
        local_param: Any,
        mesh_info: DataParallelMeshInfo | None = None,
    ) -> Any:
        mesh_info = mesh_info or self._active_mesh_info()
        placement = self._placement
        if not hasattr(placement, "dim"):
            return local_param
        dim = int(placement.dim)
        if dim < 0:
            dim += int(local_param.dim())
        mesh = mesh_info.mesh
        mesh_dim = mesh_info.shard_mesh_dim
        count = int(mesh.size(mesh_dim))
        if count <= 1:
            return local_param
        group = mesh.get_group(mesh_dim)
        local_rank = int(mesh.get_local_rank(mesh_dim))
        logical_size = int(self.param.shape[dim])
        width = math.ceil(logical_size / count)
        if int(local_param.shape[dim]) > width:
            raise RuntimeError("local parameter is larger than its shard width")
        local_value = local_param * 1 if getattr(local_param, "requires_grad", False) else local_param
        padded_local = pad_tensor(local_value, dim, width - int(local_param.shape[dim]))
        if getattr(padded_local, "requires_grad", False):
            node = getattr(padded_local, "grad_fn", None)
            self._gradient_hook_handles.append(
                padded_local.register_hook(
                    lambda gradient, info=mesh_info: self._reduce_gradient(
                        gradient, info
                    )
                )
            )
            self._gradient_hook_nodes.append(node)
        outputs = [
            padded_local.detach().new_empty(tuple(padded_local.shape))
            for _ in range(count)
        ]
        dist.all_gather(outputs, padded_local.detach(), group=group)
        outputs[local_rank] = padded_local
        result = tp.cat(tuple(outputs), dim=dim)
        total_padding = count * width - logical_size
        return unpad_tensor(result, dim, total_padding)

    def _reduce_gradient(
        self,
        gradient: Any,
        mesh_info: DataParallelMeshInfo | None = None,
    ) -> Any:
        owner = self._gradient_sync_owner
        if owner is None or not getattr(owner, "_requires_gradient_sync", True):
            return gradient
        mesh_info = mesh_info or self.mesh_info
        mesh = mesh_info.mesh
        mesh_dim = mesh_info.shard_mesh_dim
        count = int(mesh.size(mesh_dim))
        if count <= 1:
            return gradient
        reduced = gradient.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=mesh.get_group(mesh_dim))
        return reduced / count

    def bind_local_param(self, param: Any, owner: Any = None) -> Any:
        if owner is not None:
            self._gradient_sync_owner = owner
        self._gradient_hook_param = param
        if self._state == ShardedState.SHARDED:
            self.sharded_param = param
        elif self._state == ShardedState.SHARDED_POST_FORWARD:
            self._sharded_post_forward_param = param
        if not getattr(param, "requires_grad", False):
            return param
        if self._gradient_sync_owner is None:
            return param
        if getattr(param, "_tp_fsdp_gradient_hook", False):
            return param

        setattr(param, "_tp_fsdp_gradient_hook", True)
        return param

    def set_gradient_sync_owner(self, owner: Any) -> None:
        self._gradient_sync_owner = owner

    @property
    def sharded_state(self) -> ShardedState:
        return self._state

    @sharded_state.setter
    def sharded_state(self, value: ShardedState) -> None:
        self._state = value

    @property
    def _param_fqn(self) -> str:
        return self.module_info.fqn

    def _setattr_on_modules(self, param: Any) -> None:
        unsafe_setattr_param(self.module_info.module, self.module_info.name, param)
        for module, name in zip(
            self.module_info.shared_modules,
            self.module_info.shared_param_names,
        ):
            unsafe_setattr_param(module, name, param)

    def to_sharded_dtensor(self, tensor: Any) -> DTensor:
        if isinstance(tensor, DTensor):
            return tensor
        if not hasattr(self, "_sharding_spec"):
            placements = self._placement_list(
                self.mesh_info.mesh,
                self._placement,
                self.mesh_info.shard_mesh_dim,
            )
            self._sharding_spec = DTensorSpec(
                self.mesh_info.mesh,
                placements,
                TensorMeta(
                    tuple(self._orig_size),
                    tuple(self._contiguous_orig_stride),
                    tensor.dtype,
                ),
            )
        return _from_local_no_grad(tensor, self._sharding_spec)

    def to_sharded_post_forward_dtensor(self, tensor: Any) -> DTensor:
        mesh_info = self.post_forward_mesh_info
        if mesh_info is None:
            raise RuntimeError("post-forward mesh information is not configured")
        mesh = mesh_info.mesh
        placements = _mesh_placements(mesh, Shard(0), mesh_info.shard_mesh_dim)
        return _from_local_no_grad(
            tensor,
            DTensorSpec(
                mesh,
                placements,
                TensorMeta(
                    tuple(self._orig_size),
                    tuple(self._contiguous_orig_stride),
                    tensor.dtype,
                ),
            ),
        )

    def to_accumulated_grad_if_needed(self) -> Any:
        unsharded = self._unsharded_param
        gradient = self._unsharded_gradient()
        if (
            self.reduce_dtype is None
            or unsharded is None
            or gradient is None
            or getattr(gradient, "dtype", None) == self.reduce_dtype
        ):
            return None
        if unsharded is not None:
            unsharded.grad = None
        self._unsharded_grad = None
        self.unsharded_accumulated_grad = gradient.to(dtype=self.reduce_dtype)
        return None

    def accumulate_unsharded_grad_if_needed(self) -> Any:
        unsharded = self._unsharded_param
        gradient = self._unsharded_gradient()
        if self.unsharded_accumulated_grad is not None and gradient is not None:
            self.unsharded_accumulated_grad = (
                self.unsharded_accumulated_grad + gradient
            )
            if unsharded is not None:
                unsharded.grad = None
            self._unsharded_grad = None
        return None

    def alloc_all_gather_outputs(self) -> None:
        for tensor in self.all_gather_outputs:
            alloc_storage(tensor)

    def free_all_gather_outputs(self) -> None:
        for tensor in self.all_gather_outputs:
            free_storage(tensor)
        self._all_gather_outputs_ready = False

    def free_unsharded_param(self) -> None:
        self.free_all_gather_outputs()
        for tensor in self._unsharded_inner_tensors:
            free_storage(tensor)
        if self._state == ShardedState.UNSHARDED:
            self._unsharded_param = None
            self._full_tensor = None

    @property
    def all_gather_inputs(self) -> list[Any]:
        self._assert_in_states(
            ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD
        )
        local = self._sharded_local_tensor()
        pre_all_gather = getattr(local, "fsdp_pre_all_gather", None)
        if self._state == ShardedState.SHARDED and callable(pre_all_gather):
            parameter_count = len(inspect.signature(pre_all_gather).parameters)
            if parameter_count not in (1, 5):
                raise AssertionError(
                    "fsdp_pre_all_gather accepts one or five arguments"
                )
            if parameter_count == 1:
                result = pre_all_gather(self.shard_mesh())
            else:
                result = pre_all_gather(
                    self.shard_mesh(),
                    self._orig_size,
                    self._contiguous_orig_stride,
                    self.module_info.module,
                    self.mp_policy,
                )
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError(
                    "fsdp_pre_all_gather must return inputs and metadata"
                )
            inputs, metadata = result
            inputs = list(inputs)
            if not inputs:
                raise ValueError("fsdp_pre_all_gather must return at least one input")
            if parameter_count == 5:
                padded = tuple(self.padded_sharded_param_size)
                local_shape = tuple(local.shape)
                if local_shape != padded and any(
                    tuple(value.shape) != padded for value in inputs
                ):
                    raise ValueError(
                        "fsdp_pre_all_gather must return padded shard-shaped inputs"
                    )
            self._extensions.metadata = metadata
            self._extensions.all_gather_input_sizes = tuple(
                tuple(value.shape) for value in inputs
            )
            return [value.reshape(-1) for value in inputs]
        if self._state == ShardedState.SHARDED_POST_FORWARD and callable(pre_all_gather):
            raise NotImplementedError(
                "all-gather extensions are unavailable after forward reshard"
            )
        if self._state == ShardedState.SHARDED_POST_FORWARD:
            value = self._sharded_post_forward_param_data
            if value is None:
                value = self._sharded_local_tensor().reshape(-1)
        else:
            value = self._sharded_param_data
        if value is None:
            raise RuntimeError("all-gather input storage is unavailable")
        if self.param_dtype is not None and value.dtype != self.param_dtype:
            value = value.to(dtype=self.param_dtype)
        if self.offload_to_cpu and getattr(value, "device", None) != self.device:
            try:
                value = value.to(self.device, non_blocking=True)
            except TypeError:
                value = value.to(self.device)
        return [value.reshape(-1)]

    def unsharded_param(self) -> Any:
        if self._state != ShardedState.UNSHARDED:
            self.to_unsharded()
        return self._full_tensor

    def unsharded_grad_data(self) -> Any:
        gradient = self._unsharded_gradient()
        if gradient is None:
            raise RuntimeError("unsharded parameter gradient is unavailable")
        return self._get_grad_inner_tensor(gradient)

    def unsharded_accumulated_grad_data(self) -> Any:
        if self.unsharded_accumulated_grad is None:
            raise RuntimeError("accumulated unsharded gradient is unavailable")
        return self._get_grad_inner_tensor(self.unsharded_accumulated_grad)

    def unsharded_zero_grad_data(self) -> Any:
        return tp.zeros_like(self.unsharded_param())

    def _get_grad_inner_tensor(self, grad: Any) -> Any:
        if getattr(self, "is_spmd_types", False):
            if self._unsharded_dtensor_spec is None:
                raise AssertionError("distributed parameter metadata is missing")
            grad = DTensor.from_local(
                grad,
                self._unsharded_dtensor_spec.mesh,
                tuple(self._spmd_grad_placements),
                run_check=False,
            )
            if not self.is_dtensor:
                raise AssertionError(
                    "type-check metadata must resolve to a distributed parameter"
                )
        if isinstance(grad, DTensor):
            if self._unsharded_dtensor_spec is not None:
                placements = self._unsharded_dtensor_spec.placements
                if self.mesh_info.is_spmd_mesh:
                    dp_indices = getattr(self, "_dp_dim_indices", frozenset())
                    target_placements = tuple(
                        grad.placements[index]
                        if index in dp_indices
                        else placements[index]
                        for index in range(len(placements))
                    )
                else:
                    target_placements = placements
                if target_placements != grad.placements:
                    grad = grad.redistribute(placements=target_placements)
            return grad.to_local()
        return grad

    def _unsharded_gradient(self) -> Any:
        if self._unsharded_grad is not None:
            return self._unsharded_grad
        return getattr(self._unsharded_param, "grad", None)

    def _sharded_local_tensor(self) -> Any:
        if self._state == ShardedState.SHARDED_POST_FORWARD:
            tensor = self._sharded_post_forward_param
        else:
            tensor = self.sharded_param if hasattr(self, "sharded_param") else None
        if tensor is None:
            self.to_sharded()
            tensor = self.sharded_param
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        if tensor is None:
            raise RuntimeError("sharded parameter storage is unavailable")
        return tensor

    def _init_shard_mesh(self) -> Any:
        mesh = self.mesh_info.mesh
        ndim = self._mesh_ndim(mesh)
        if ndim == 1:
            return mesh
        names = getattr(mesh, "mesh_dim_names", None)
        if names is None:
            raise ValueError("a multi-dimensional mesh needs dimension names")
        return mesh[names[self.mesh_info.shard_mesh_dim]]

    def shard_mesh(self) -> Any:
        return getattr(self, "_shard_mesh", self.mesh_info.mesh)

    def shard_mesh_from_root(self, root_mesh: Any) -> Any:
        del root_mesh
        return self.shard_mesh()

    def _assert_in_states(self, *states: Any) -> None:
        if self._state not in states:
            raise RuntimeError(f"parameter state {self._state!r} is not one of {states!r}")

    def reset_sharded_param(self) -> None:
        module_info = self._module_info
        new_param = module_info.module._parameters.get(module_info.param_name)
        if new_param is None:
            return
        if new_param is not self.sharded_param:
            self.sharded_param = new_param
        if self._state != ShardedState.SHARDED:
            self._full_tensor = new_param
            self._unsharded_param = new_param
            self._state = ShardedState.UNSHARDED
            return
        local = new_param.to_local() if isinstance(new_param, DTensor) else new_param
        if getattr(local, "is_meta", False):
            return
        same_local_tensor = False
        old_data = getattr(self, "_sharded_param_data", None)
        storage = getattr(old_data, "untyped_storage", None)
        new_storage = getattr(local, "untyped_storage", None)
        if callable(storage) and callable(new_storage):
            try:
                old_ptr = int(storage().data_ptr())
                new_ptr = int(new_storage().data_ptr())
                same_local_tensor = old_ptr > 0 and old_ptr == new_ptr
            except (AttributeError, RuntimeError):
                same_local_tensor = False
        padded_size = tuple(self.padded_sharded_param_size)
        shard_dim = int(self._placement.dim) if isinstance(self._placement, Shard) else 0
        length = int(local.shape[shard_dim]) if int(local.numel()) else 0
        updated_local_tensor = False
        if tuple(local.shape) != padded_size and not same_local_tensor:
            if shard_dim != 0:
                raise AssertionError(
                    f"shard dimension {shard_dim} requires even sharding: {local.shape=}"
                )
            padded = local.new_zeros(padded_size)
            if length:
                padded.narrow(shard_dim, 0, length).copy_(local)
            local = padded
            updated_local_tensor = True
        if self.pin_memory and not getattr(local, "is_pinned", lambda: False)():
            local = local.cpu().pin_memory()
            updated_local_tensor = True
        if not same_local_tensor:
            self._sharded_param_data = local.reshape(-1)
        if updated_local_tensor or self._sharded_tensor is None:
            unpadded = local.narrow(shard_dim, 0, length) if length else local.narrow(shard_dim, 0, 0)
            self._sharded_tensor = self.to_sharded_dtensor(unpadded)
        if self._state == ShardedState.SHARDED:
            self._setattr_on_modules(self.sharded_param)

    def _use_unsharded_tensor(self, tensor: Any) -> None:
        self._all_gather_outputs_ready = True
        self._set_unsharded_tensor(tensor)
        self._register_full_gradient_hook(self._unsharded_param)
        self._state = ShardedState.UNSHARDED

    def _set_sharded_grad(self, grad: Any) -> None:
        self._sharded_grad = grad
        target = self._sharded_local_tensor()
        target.grad = grad

    def __repr__(self) -> str:
        return f"FSDPParam(fqn={self.module_info.fqn!r}, state={self._state!r})"


def alloc_storage(tensor: Any) -> None:
    itemsize = getattr(tensor, "itemsize")
    itemsize = itemsize() if callable(itemsize) else itemsize
    size = int(tensor.numel()) * int(itemsize)
    storage = tensor.untyped_storage()
    if int(storage.size()) != size:
        storage.resize_(size)


def free_storage(tensor: Any) -> None:
    storage = tensor.untyped_storage()
    if int(storage.size()) != 0:
        storage.resize_(0)


def unsafe_setattr_param(module: Any, param_name: str, param: Any) -> None:
    if getattr(module.__setattr__, "__func__", None) is Module.__setattr__:
        module._buffers.pop(param_name, None)
        module._parameters[param_name] = param
    else:
        setattr(module, param_name, param)


def set_requires_grad_if_needed(src_tensor: Any, dst_tensor: Any) -> None:
    value = bool(getattr(src_tensor, "requires_grad", False))
    if bool(getattr(dst_tensor, "requires_grad", False)) != value:
        requires_grad = getattr(dst_tensor, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad(value)
        else:
            dst_tensor.requires_grad = value
