"""Parameter state transitions for composable sharding."""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import tensorplay as tp
from tensorplay.nn.parameter import Parameter

from ... import distributed_core as dist
from ...tensor import DTensor, Replicate, Shard, distribute_tensor
from ...tensor._dtensor_spec import DTensorSpec, TensorMeta
from ...tensor.placement_types import _StridedShard
from ...tensor._collective_utils import pad_tensor, unpad_tensor
from ._fsdp_api import CPUOffloadPolicy
from ._fsdp_common import (
    DataParallelMeshInfo,
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


def _get_orig_param_uid(param: Any) -> int:
    return id(param)


def _spans_same_mesh(lhs_axes: Any, rhs_axes: Any) -> bool:
    return tuple(lhs_axes) == tuple(rhs_axes)


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

    def _init_sharded_param(
        self,
        param: Any,
        device: Any,
        shard_placement_fn: Any,
        mesh_info: DataParallelMeshInfo,
    ) -> None:
        del device, shard_placement_fn
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

    def _resolve_spmd_types_for_storage(self, param: Any, partition_spec: Any, init_local_type: Any, mesh_info: Any) -> Any:
        del param, partition_spec, mesh_info
        return init_local_type

    def _restore_spmd_types(self, tensor: Any) -> Any:
        return tensor

    def _init_sharding_spec(self, param: Any, fsdp_placement: Any, shard_dim: int) -> Any:
        del shard_dim
        if isinstance(param, DTensor):
            self._unsharded_dtensor_spec = DTensorSpec(
                param.device_mesh,
                param.placements,
                TensorMeta(tuple(param.shape), tuple(param.stride()), param.dtype),
            )
            self._spmd_mesh = param.device_mesh
            original_placements = list(param.placements)
            if self.mesh_info.is_spmd_mesh and self.mesh_info.dp_mesh_dims is not None:
                names = getattr(self._spmd_mesh, "mesh_dim_names", None)
                if names is None:
                    raise ValueError("an SPMD parameter mesh needs dimension names")
                for name in self.mesh_info.dp_mesh_dims.shard_names:
                    index = names.index(name)
                    original = original_placements[index]
                    if not isinstance(original, Replicate):
                        raise ValueError(
                            f"data-parallel shard dimension {name!r} must be replicated"
                        )
                    original_placements[index] = fsdp_placement
                for name in self.mesh_info.dp_mesh_dims.replicate_names:
                    index = names.index(name)
                    if not isinstance(original_placements[index], Replicate):
                        raise ValueError(
                            f"data-parallel replicate dimension {name!r} must be replicated"
                        )
            self._spmd_placements = tuple(original_placements)
            tensor_meta = self._unsharded_dtensor_spec.tensor_meta
        else:
            self._spmd_mesh = self.mesh_info.mesh
            self._spmd_placements = self._placement_list(
                self._spmd_mesh,
                fsdp_placement,
                self.mesh_info.shard_mesh_dim,
            )
            tensor_meta = TensorMeta(
                tuple(param.shape), tuple(param.stride()), param.dtype
            )
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=tensor_meta,
        )
        return param.to_local() if isinstance(param, DTensor) else param

    _init_sharding_spec_spmd = _init_sharding_spec
    _init_sharding_spec_tp = _init_sharding_spec
    _init_sharding_spec_plain = _init_sharding_spec

    def _build_spmd_sharding_spec(self, dp_dim_names: Any, dp_shard_indices: Any, fsdp_placement: Any) -> tuple[Any, ...]:
        del dp_dim_names, dp_shard_indices
        return (fsdp_placement,)

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
        self._extensions = ExtensionsData()
        if callable(should_release):
            value = should_release()
            if not isinstance(value, bool):
                raise AssertionError("all-gather output release flag must be boolean")
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
        self._unsharded_param = tensor
        self._full_tensor = tensor
        self._setattr_on_modules(tensor)
        return tensor

    def _attach_local_gradient_to_all_gather(self, gathered: Any) -> Any:
        mesh_info = self._active_mesh_info()
        mesh_dim = mesh_info.shard_mesh_dim
        if mesh_dim is None or not isinstance(self._placement, Shard):
            return gathered
        count = int(mesh_info.mesh.size(mesh_dim))
        if count <= 1:
            return gathered
        local = self._sharded_local_tensor().reshape(-1)
        width = int(gathered.numel()) // count
        if int(local.numel()) > width:
            raise RuntimeError("local shard is larger than the all-gather slot")
        if int(local.numel()) < width:
            padding = local.new_zeros(width - int(local.numel()))
            local = tp.cat((local, padding), dim=0)
        rank = int(mesh_info.mesh.get_local_rank(mesh_dim))
        pieces = [
            gathered.narrow(0, index * width, width).detach()
            for index in range(count)
        ]
        pieces[rank] = local
        return tp.cat(tuple(pieces), dim=0)

    def to_sharded(self) -> None:
        if self._state == ShardedState.SHARDED:
            self._setattr_on_modules(self.sharded_param)
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
                if int(candidate.numel()) == math.prod(self._orig_size):
                    gathered = tp.as_strided(
                        candidate,
                        self._orig_size,
                        self._contiguous_orig_stride,
                        storage_offset=0,
                    )
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
        gradient = getattr(unsharded, "grad", None)
        if (
            self.reduce_dtype is None
            or unsharded is None
            or gradient is None
            or getattr(gradient, "dtype", None) == self.reduce_dtype
        ):
            return None
        unsharded.grad = None
        self.unsharded_accumulated_grad = gradient.to(dtype=self.reduce_dtype)
        return None

    def accumulate_unsharded_grad_if_needed(self) -> Any:
        unsharded = self._unsharded_param
        gradient = getattr(unsharded, "grad", None)
        if self.unsharded_accumulated_grad is not None and gradient is not None:
            self.unsharded_accumulated_grad = (
                self.unsharded_accumulated_grad + gradient
            )
            unsharded.grad = None
        return None

    def alloc_all_gather_outputs(self) -> None:
        for tensor in self.all_gather_outputs:
            if tensor is None:
                continue
            self._all_gather_output = tensor

    def free_all_gather_outputs(self) -> None:
        self.all_gather_outputs.clear()
        self._all_gather_output = None
        self._all_gather_outputs_ready = False

    def free_unsharded_param(self) -> None:
        if self._state == ShardedState.UNSHARDED:
            self._unsharded_param = None
            self._full_tensor = None

    @property
    def all_gather_inputs(self) -> list[Any]:
        self._assert_in_states(
            ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD
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
        return [value.reshape(-1)]

    def unsharded_param(self) -> Any:
        if self._state != ShardedState.UNSHARDED:
            self.to_unsharded()
        return self._full_tensor

    def unsharded_grad_data(self) -> Any:
        return getattr(self._full_tensor, "grad", None)

    def unsharded_accumulated_grad_data(self) -> Any:
        return self._sharded_grad

    def unsharded_zero_grad_data(self) -> Any:
        return tp.zeros_like(self.unsharded_param())

    def _get_grad_inner_tensor(self, grad: Any) -> Any:
        return grad.to_local() if isinstance(grad, DTensor) else grad

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
        current = self.module_info.module._parameters.get(self.module_info.name)
        if current is None:
            return
        if isinstance(current, DTensor):
            local = current.to_local()
        else:
            local = current
        if self._state == ShardedState.SHARDED:
            self.sharded_param = current
            self._sharded_tensor = self.to_sharded_dtensor(local)
            padded_size = getattr(self, "padded_sharded_param_size", None)
            if padded_size is not None and tuple(local.shape) != tuple(padded_size):
                shard_dim = int(self._placement.dim) if isinstance(self._placement, Shard) else 0
                padded = local.new_zeros(tuple(padded_size))
                length = int(local.shape[shard_dim]) if int(local.numel()) else 0
                if length:
                    padded.narrow(shard_dim, 0, length).copy_(local)
                self._sharded_param_data = padded.reshape(-1)
            else:
                self._sharded_param_data = local.reshape(-1)
            return
        self._full_tensor = current
        self._unsharded_param = current
        self._state = ShardedState.UNSHARDED

    def _use_unsharded_tensor(self, tensor: Any) -> None:
        self._all_gather_outputs_ready = True
        self._set_unsharded_tensor(tensor)
        self._register_full_gradient_hook(self._unsharded_param)
        self._state = ShardedState.UNSHARDED

    def _set_sharded_grad(self, grad: Any) -> None:
        self._sharded_grad = grad

    def __repr__(self) -> str:
        return f"FSDPParam(fqn={self.module_info.fqn!r}, state={self._state!r})"


def alloc_storage(tensor: Any) -> Any:
    return tensor


def free_storage(tensor: Any) -> None:
    del tensor


def unsafe_setattr_param(module: Any, param_name: str, param: Any) -> None:
    module._parameters[param_name] = param


def set_requires_grad_if_needed(src_tensor: Any, dst_tensor: Any) -> None:
    value = bool(getattr(src_tensor, "requires_grad", False))
    if bool(getattr(dst_tensor, "requires_grad", False)) != value:
        requires_grad = getattr(dst_tensor, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad(value)
        else:
            dst_tensor.requires_grad = value
