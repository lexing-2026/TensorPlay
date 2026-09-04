"""Distributed tensor values and layout conversion routines."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any

import tensorplay

from ..device_mesh import DeviceMesh, _MeshEnv
from ._collective_utils import check_tensor_meta
from ._utils import (
    assert_no_mixed_partial_types,
    compute_global_tensor_info,
    compute_local_shape_and_global_offset,
    normalize_to_torch_size,
)
from .placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "from_local",
    "ones",
    "empty",
    "full",
    "linspace",
    "logspace",
    "rand",
    "randn",
    "zeros",
]


def _mesh_ndim(mesh: DeviceMesh) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def _mesh_dim(mesh: DeviceMesh, mesh_dim: int | str) -> int:
    if isinstance(mesh_dim, str):
        names = getattr(mesh, "mesh_dim_names", None)
        if names is None:
            raise KeyError(mesh_dim)
        try:
            mesh_dim = names.index(mesh_dim)
        except ValueError as error:
            raise KeyError(mesh_dim) from error
    dim = int(mesh_dim)
    if dim < 0:
        dim += _mesh_ndim(mesh)
    if dim < 0 or dim >= _mesh_ndim(mesh):
        raise ValueError(f"mesh dimension {mesh_dim} is outside the mesh")
    return dim


def _current_mesh() -> DeviceMesh:
    stack = _MeshEnv.get().mesh_stack
    if not stack:
        raise RuntimeError("a DeviceMesh is required when no mesh was provided")
    return stack[-1]


def _normalize_mesh(device_mesh: DeviceMesh | None) -> DeviceMesh:
    return device_mesh if device_mesh is not None else _current_mesh()


def _normalize_placements(
    placements: Sequence[Placement] | None, mesh: DeviceMesh, ndim: int | None = None
) -> tuple[Placement, ...]:
    mesh_ndim = _mesh_ndim(mesh)
    result = tuple(placements) if placements is not None else tuple(
        Replicate() for _ in range(mesh_ndim)
    )
    if len(result) != mesh_ndim:
        raise ValueError(
            "placements must have the same length as device_mesh.ndim; "
            f"got {len(result)} and {mesh_ndim}"
        )
    normalized: list[Placement] = []
    for placement in result:
        if not isinstance(placement, Placement):
            raise TypeError(f"invalid placement {placement!r}")
        if _is_shard_like(placement):
            if ndim is not None and not -ndim <= placement.dim < ndim:
                raise ValueError(
                    f"shard dimension {placement.dim} is outside tensor rank {ndim}"
                )
            dim = placement.dim if placement.dim >= 0 or ndim is None else placement.dim + ndim
            if isinstance(placement, _StridedShard):
                normalized.append(_StridedShard(dim, placement.split_factor))
            else:
                normalized.append(Shard(dim))
        else:
            normalized.append(placement)
    assert_no_mixed_partial_types(normalized)
    return tuple(normalized)


def _normalize_size_args(size: Sequence[Any]) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    return normalize_to_torch_size(size)


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    running = 1
    for index in reversed(range(len(shape))):
        strides[index] = running
        running *= int(shape[index])
    return tuple(strides)


def _is_meta(value: Any) -> bool:
    return bool(getattr(value, "is_meta", False))


def _participates(mesh: DeviceMesh) -> bool:
    coordinate = getattr(mesh, "get_coordinate", None)
    return coordinate is None or coordinate() is not None


def _move_to_mesh_device(value: Any, mesh: DeviceMesh) -> Any:
    if _is_meta(value):
        return value
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if device_type is None:
        device_type = str(device).split(":", 1)[0]
    if str(device_type) != str(mesh.device_type):
        return value.to(mesh.device_type)
    return value


def _new_empty_nonparticipant(value: Any) -> Any:
    requires_grad = bool(getattr(value, "requires_grad", False))
    try:
        return value.new_empty((0,), requires_grad=requires_grad)
    except TypeError:
        result = value.new_empty((0,))
        if hasattr(result, "requires_grad_"):
            result.requires_grad_(requires_grad)
        return result


def _group(mesh: DeviceMesh, mesh_dim: int) -> Any:
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    if int(mesh.size(mesh_dim)) <= 1:
        return None
    return mesh.get_group(mesh_dim)


def _validate_src_data_rank(
    mesh: DeviceMesh, mesh_dim: int, src_data_rank: int | None
) -> None:
    if src_data_rank is None:
        return
    if type(src_data_rank) is not int or src_data_rank < 0:
        raise ValueError("src_data_rank must be a non-negative mesh-relative rank")
    if src_data_rank >= int(mesh.size(_mesh_dim(mesh, mesh_dim))):
        raise ValueError("src_data_rank is outside the mesh dimension")


def _distribute_shard(
    value: Any,
    placement: Shard,
    mesh: DeviceMesh,
    mesh_dim: int,
    src_data_rank: int | None,
) -> Any:
    if not _participates(mesh):
        return _new_empty_nonparticipant(value)
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    _validate_src_data_rank(mesh, mesh_dim, src_data_rank)
    return placement._shard_tensor(value, mesh, mesh_dim, src_data_rank)


def _replicate(
    value: Any, mesh: DeviceMesh, mesh_dim: int, src_data_rank: int | None
) -> Any:
    mesh_dim = _mesh_dim(mesh, mesh_dim)
    _validate_src_data_rank(mesh, mesh_dim, src_data_rank)
    return Replicate._make_replicate_tensor(
        value, mesh, mesh_dim, src_data_rank
    )


def _normalize_grad_placements(
    placements: Sequence[Placement], mesh: DeviceMesh, ndim: int
) -> tuple[Placement, ...]:
    normalized = _normalize_placements(placements, mesh, ndim)
    return tuple(Replicate() if isinstance(item, Partial) else item for item in normalized)


class DTensor:
    """A logical tensor represented by a local value and a mesh placement."""

    __array_priority__ = 1000
    _op_dispatcher: Any = None

    def __init__(
        self,
        local_tensor: Any,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        shape: Sequence[int] | None = None,
        stride: Sequence[int] | None = None,
        grad_placements: Sequence[Placement] | None = None,
        backward_dtype: Any = None,
    ) -> None:
        if isinstance(local_tensor, DTensor):
            raise TypeError("local_tensor must be a plain tensor")
        if shape is None:
            shape, _ = compute_global_tensor_info(
                local_tensor,
                device_mesh,
                placements,
            )
        normalized_shape = tuple(int(value) for value in shape)
        rank = len(normalized_shape) if int(local_tensor.numel()) == 0 else int(local_tensor.dim())
        self._local_tensor = local_tensor
        self._device_mesh = device_mesh
        self._placements = _normalize_placements(placements, device_mesh, rank)
        if len(normalized_shape) != rank and int(local_tensor.numel()) != 0:
            raise ValueError("global tensor rank does not match the local tensor rank")
        self._shape = normalized_shape
        try:
            local_shape, local_offset = compute_local_shape_and_global_offset(
                self._shape, self._device_mesh, self._placements
            )
            self._local_chunk = (
                tuple(int(value) for value in local_offset),
                tuple(int(value) for value in local_shape),
            )
        except (RuntimeError, ValueError):
            self._local_chunk = None
        self._stride = (
            tuple(int(value) for value in stride)
            if stride is not None
            else tuple(int(value) for value in local_tensor.stride())
        )
        if len(self._stride) != len(self._shape):
            raise ValueError("tensor stride rank does not match tensor shape rank")
        self._grad_placements = (
            _normalize_grad_placements(grad_placements, device_mesh, len(self._shape))
            if grad_placements is not None
            else None
        )
        self._backward_dtype = backward_dtype

    @classmethod
    def from_local(
        cls,
        local_tensor: Any,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        run_check: bool = False,
        shape: Sequence[int] | None = None,
        stride: Sequence[int] | None = None,
        grad_placements: Sequence[Placement] | None = None,
    ) -> "DTensor":
        if isinstance(local_tensor, DTensor):
            raise TypeError("from_local expects a plain local tensor")
        if (shape is None) != (stride is None):
            raise RuntimeError("shape and stride must be provided together")
        mesh = _normalize_mesh(device_mesh)
        local_tensor = _move_to_mesh_device(local_tensor, mesh)
        normalized = _normalize_placements(placements, mesh, int(local_tensor.dim()))
        if shape is None:
            global_shape, global_stride = compute_global_tensor_info(
                local_tensor,
                mesh,
                normalized,
            )
        else:
            global_shape = normalize_to_torch_size(shape)
            global_stride = tuple(int(value) for value in stride or ())
            if len(global_shape) != len(global_stride):
                raise ValueError("tensor shape and stride ranks must match")
        if run_check:
            check_tensor_meta(local_tensor, check_shape_stride=False)
        if not _participates(mesh):
            local_tensor = _new_empty_nonparticipant(local_tensor)
        elif run_check:
            for mesh_dim, placement in enumerate(normalized):
                if isinstance(placement, Replicate):
                    _replicate(local_tensor, mesh, mesh_dim, 0)
        return cls(
            local_tensor,
            mesh,
            normalized,
            shape=global_shape,
            stride=global_stride,
            grad_placements=grad_placements,
        )

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._device_mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        return self._placements

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def stride(self, dim: int | None = None) -> Any:
        if dim is None:
            return self._stride
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"stride dimension {dim} is outside tensor rank {self.ndim}")
        return self._stride[dim]

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def dim(self) -> int:
        return self.ndim

    @property
    def dtype(self) -> Any:
        return self._local_tensor.dtype

    @property
    def device(self) -> Any:
        return self._local_tensor.device

    def numel(self) -> int:
        return math.prod(self._shape)

    def size(self, dim: int | None = None) -> Any:
        if dim is None:
            return self._shape
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise IndexError(f"dimension {dim} is outside tensor rank {self.ndim}")
        return self._shape[dim]

    def to_local(self, *, grad_placements: Sequence[Placement] | None = None) -> Any:
        if grad_placements is not None:
            _normalize_grad_placements(grad_placements, self._device_mesh, self.ndim)
        return self._local_tensor

    def full_tensor(self, *, grad_placements: Sequence[Placement] | None = None) -> Any:
        if grad_placements is not None:
            _normalize_grad_placements(grad_placements, self._device_mesh, self.ndim)
        target = tuple(Replicate() for _ in self._placements)
        if target == self._placements:
            return self._local_tensor
        return self.redistribute(placements=target).to_local()

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
        forward_dtype: Any = None,
        backward_dtype: Any = None,
    ) -> "DTensor":
        mesh = device_mesh or self._device_mesh
        if mesh != self._device_mesh:
            raise ValueError("redistribute only supports the current device mesh")
        if placements is None:
            raise RuntimeError("placements is needed for redistribute")
        target = _normalize_placements(placements, mesh, self.ndim)
        if target == self._placements and forward_dtype is None:
            return self
        from ._redistribute import Redistribute

        input_dtype = self._local_tensor.dtype
        forward_dtype = forward_dtype or input_dtype
        return Redistribute.apply(
            self,
            mesh,
            target,
            async_op,
            {
                "op_dtype": forward_dtype,
                "out_dtype": forward_dtype,
                "backward_options": {
                    "op_dtype": backward_dtype or input_dtype,
                    "out_dtype": input_dtype,
                },
            },
        )

    def detach(self) -> "DTensor":
        return type(self)(
            self._local_tensor.detach(),
            self._device_mesh,
            self._placements,
            shape=self._shape,
            stride=self._stride,
            grad_placements=self._grad_placements,
            backward_dtype=self._backward_dtype,
        )

    def clone(self) -> "DTensor":
        return type(self)(
            self._local_tensor.clone(),
            self._device_mesh,
            self._placements,
            shape=self._shape,
            stride=self._stride,
            grad_placements=self._grad_placements,
            backward_dtype=self._backward_dtype,
        )

    def __create_write_items__(self, fqn: str, object: Any) -> list[Any]:
        self._raise_if_contains_partial_placements()
        create_items = getattr(self._local_tensor, "__create_write_items__", None)
        if callable(create_items):
            return list(create_items(fqn, object))
        from ..checkpoint.planner_helpers import _create_write_item_for_dtensor

        return [_create_write_item_for_dtensor(fqn, self)]

    def __create_chunk_list__(self) -> list[Any]:
        self._raise_if_contains_partial_placements()
        create_chunks = getattr(self._local_tensor, "__create_chunk_list__", None)
        if callable(create_chunks):
            return list(create_chunks())
        from ..checkpoint.metadata import ChunkStorageMetadata

        if self._local_chunk is None:
            from ..checkpoint.planner_helpers import _create_chunk_from_dtensor

            return [_create_chunk_from_dtensor(self)]
        offsets, sizes = self._local_chunk
        return [ChunkStorageMetadata(offsets, sizes)]

    def _raise_if_contains_partial_placements(self) -> None:
        if any(isinstance(placement, Partial) for placement in self._placements):
            raise ValueError(
                "checkpoint operations do not support partial placements"
            )

    def __get_tensor_shard__(self, index: Any) -> Any:
        get_shard = getattr(self._local_tensor, "__get_tensor_shard__", None)
        if callable(get_shard):
            return get_shard(index)
        return self.to_local()

    def __repr__(self) -> str:
        return (
            f"DTensor(local_tensor={self._local_tensor!r}, "
            f"device_mesh={self._device_mesh!r}, placements={self._placements!r})"
        )

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Any:
        local_tensor = self.__dict__.get("_local_tensor")
        if local_tensor is None:
            raise AttributeError(name)
        attribute = getattr(local_tensor, name)
        if not callable(attribute):
            return attribute

        def invoke(*args: Any, **kwargs: Any) -> Any:
            return self._op_dispatcher.dispatch_method(self, name, args, kwargs)

        return invoke

    def __getitem__(self, index: Any) -> Any:
        return self._op_dispatcher.dispatch_method(
            self, "__getitem__", (index,), {}
        )

    def _binary(
        self,
        other: Any,
        operation: Callable[[Any, Any], Any],
        operation_name: str,
    ) -> "DTensor":
        if isinstance(other, DTensor):
            if other.device_mesh != self.device_mesh or other.placements != self.placements:
                raise ValueError("DTensor operands must have identical layouts")
            other = other.to_local()
        result = operation(self.to_local(), other)
        return self._op_dispatcher.wrap_result(
            result,
            (self, other),
            operation_name,
            (self, other),
        )

    def __add__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left + right, "add")

    __radd__ = __add__

    def __sub__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left - right, "sub")

    def __rsub__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: right - left, "sub")

    def __mul__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left * right, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left / right, "div")

    def __matmul__(self, other: Any) -> "DTensor":
        return self._binary(other, lambda left, right: left @ right, "matmul")


    def __tensorplay_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        return self._op_dispatcher.dispatch(func, args, kwargs or {})

from ._dispatch import OpDispatcher

DTensor._op_dispatcher = OpDispatcher()


from_local = DTensor.from_local


def distribute_tensor(
    tensor: Any,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    *,
    src_data_rank: int | None = 0,
) -> DTensor:
    mesh = _normalize_mesh(device_mesh)
    normalized = _normalize_placements(
        placements, mesh, tensor.ndim if isinstance(tensor, DTensor) else int(tensor.dim())
    )
    if isinstance(tensor, DTensor):
        if tensor.device_mesh != mesh:
            raise ValueError("cannot distribute a DTensor to a different device mesh")
        if tensor.placements != normalized:
            raise ValueError(
                "cannot distribute a DTensor to different placements; use redistribute"
            )
        return tensor
    if hasattr(tensor, "is_leaf") and not tensor.is_leaf:
        raise RuntimeError("distribute_tensor should be used with leaf tensors")
    tensor = _move_to_mesh_device(tensor, mesh)
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    if hasattr(value, "requires_grad_"):
        value.requires_grad_(bool(getattr(tensor, "requires_grad", False)))
    if not _participates(mesh):
        value = _new_empty_nonparticipant(value)
    else:
        for mesh_dim, placement in enumerate(normalized):
            if _is_shard_like(placement):
                value = _distribute_shard(
                    value, placement, mesh, mesh_dim, src_data_rank
                )
            elif isinstance(placement, Replicate):
                value = _replicate(value, mesh, mesh_dim, src_data_rank)
            elif isinstance(placement, Partial):
                value = _replicate(value, mesh, mesh_dim, src_data_rank)
                value = placement._partition_value(value, mesh, mesh_dim)
            else:
                raise RuntimeError(f"unsupported placement {placement!r}")
    return DTensor(
        value,
        mesh,
        normalized,
        shape=tuple(int(item) for item in tensor.shape),
        stride=tuple(int(item) for item in tensor.stride()),
    )


def _callback_arity(callback: Callable[..., Any], name: str, expected: int) -> None:
    try:
        actual = len(inspect.signature(callback).parameters)
    except (TypeError, ValueError) as error:
        raise TypeError(f"cannot inspect {name}") from error
    if actual != expected:
        raise ValueError(f"{name} should take {expected} arguments, got {actual}")


def _replicate_module_state(module: Any, mesh: DeviceMesh) -> None:
    full_layout = tuple(Replicate() for _ in range(_mesh_ndim(mesh)))
    for name, parameter in list(module._parameters.items()):
        if parameter is None or isinstance(parameter, DTensor):
            continue
        data = getattr(parameter, "data", parameter)
        module._parameters[name] = distribute_tensor(
            data,
            mesh,
            full_layout,
            src_data_rank=0,
        )
    for name, buffer in list(module._buffers.items()):
        if buffer is not None and not isinstance(buffer, DTensor):
            module._buffers[name] = distribute_tensor(
                buffer,
                mesh,
                full_layout,
                src_data_rank=0,
            )


def distribute_module(
    module: Any,
    device_mesh: DeviceMesh | None = None,
    partition_fn: Callable[[str, Any, DeviceMesh], Any] | None = None,
    input_fn: Callable[[Any, Any, DeviceMesh], Any] | None = None,
    output_fn: Callable[[Any, Any, DeviceMesh], Any] | None = None,
) -> Any:
    mesh = _normalize_mesh(device_mesh)
    if getattr(module, "_distribute_module_applied", False):
        raise RuntimeError("distribute_module should only be called once per module")
    if input_fn is not None:
        _callback_arity(input_fn, "input_fn", 3)
    if output_fn is not None:
        _callback_arity(output_fn, "output_fn", 3)

    if partition_fn is None:
        for child in module.modules():
            _replicate_module_state(child, mesh)
    else:
        for name, child in module.named_modules():
            partition_fn(name, child, mesh)
            _replicate_module_state(child, mesh)

    if input_fn is not None:
        def pre_hook(current: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            result = input_fn(current, inputs, mesh)
            return inputs if result is None else result

        module.register_forward_pre_hook(pre_hook)
    if output_fn is not None:
        def post_hook(current: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            result = output_fn(current, output, mesh)
            return output if result is None else result

        module.register_forward_hook(post_hook)
    module._distribute_module_applied = True
    return module


def _validate_layout(layout: Any) -> None:
    if layout == "strided":
        return
    native_layout = getattr(tensorplay, "strided", None)
    if native_layout is not None and layout is native_layout:
        return
    raise ValueError("only the strided layout is supported")


def _scalar_value(value: Any, name: str) -> Any:
    if isinstance(value, DTensor):
        if value.ndim != 0:
            raise ValueError(f"{name} only accepts a zero-dimensional tensor")
        value = value.to_local()
    if hasattr(value, "dim"):
        if int(value.dim()) != 0:
            raise ValueError(f"{name} only accepts a zero-dimensional tensor")
        if hasattr(value, "item"):
            value = value.item()
    return value


def _factory(
    operation: Callable[..., Any],
    global_size: Sequence[int],
    *,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    dtype: Any,
    requires_grad: bool,
    kind: str,
    fill_value: Any = None,
    start: Any = None,
    end: Any = None,
    base: float = 10.0,
) -> DTensor:
    global_size = tuple(int(item) for item in global_size)
    normalized = _normalize_placements(placements, mesh, len(global_size))
    global_stride = _contiguous_strides(global_size)
    kwargs: dict[str, Any] = {
        "device": tensorplay.device(mesh.device_type),
        "requires_grad": bool(requires_grad),
    }
    if dtype is not None:
        kwargs["dtype"] = dtype

    if not _participates(mesh):
        local = tensorplay.empty((0,), device=mesh.device_type, dtype=dtype, requires_grad=requires_grad)
    else:
        local_size, offset = compute_local_shape_and_global_offset(
            global_size, mesh, normalized
        )
        if kind == "full":
            local = operation(local_size, fill_value, **kwargs)
        elif kind == "linspace":
            local_steps = int(local_size[0])
            total_steps = int(global_size[0])
            local_start = start
            local_end = end
            if local_steps > 0 and total_steps > 1:
                step = (end - start) / (total_steps - 1)
                local_start = start + int(offset[0]) * step
                local_end = local_start + step * (local_steps - 1)
            elif local_steps > 0:
                local_end = start
            local = operation(local_start, local_end, local_steps, **kwargs)
        elif kind == "logspace":
            local_steps = int(local_size[0])
            total_steps = int(global_size[0])
            local_start = start
            local_end = end
            if local_steps > 0 and total_steps > 1:
                step = (end - start) / (total_steps - 1)
                local_start = start + int(offset[0]) * step
                local_end = local_start + step * (local_steps - 1)
            elif local_steps > 0:
                local_end = start
            local = operation(local_start, local_end, local_steps, base, **kwargs)
        else:
            local = operation(local_size, **kwargs)
        for mesh_dim, placement in enumerate(normalized):
            if isinstance(placement, Partial):
                local = placement._partition_value(local, mesh, mesh_dim)
    return DTensor(
        local,
        mesh,
        normalized,
        shape=global_size,
        stride=global_stride,
    )


def ones(
    *size: Any,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    shape = _normalize_size_args(size)
    normalized = _normalize_placements(placements, mesh, len(shape))
    return _factory(
        tensorplay.ones,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="ones",
    )


def empty(
    *size: Any,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    shape = _normalize_size_args(size)
    normalized = _normalize_placements(placements, mesh, len(shape))
    return _factory(
        tensorplay.empty,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="empty",
    )


def _full_dtype(fill_value: Any, dtype: Any) -> Any:
    if dtype is not None:
        return dtype
    if isinstance(fill_value, bool):
        return tensorplay.bool
    if isinstance(fill_value, int):
        return tensorplay.int64
    if isinstance(fill_value, float):
        return tensorplay.get_default_dtype()
    if isinstance(fill_value, complex):
        return getattr(tensorplay, "complex64", tensorplay.get_default_dtype())
    return getattr(fill_value, "dtype", None)


def full(
    size: Any,
    fill_value: Any,
    *,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    shape = normalize_to_torch_size(size)
    normalized = _normalize_placements(placements, mesh, len(shape))
    fill_dtype = getattr(fill_value, "dtype", None)
    fill_value = _scalar_value(fill_value, "full")
    inferred_dtype = _full_dtype(fill_value, dtype)
    if inferred_dtype is None:
        inferred_dtype = fill_dtype
    return _factory(
        tensorplay.full,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=inferred_dtype,
        requires_grad=requires_grad,
        kind="full",
        fill_value=fill_value,
    )


def linspace(
    start: Any,
    end: Any,
    steps: int,
    *,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    if placements is not None and any(isinstance(p, _StridedShard) for p in placements):
        raise ValueError("linspace does not support strided shard placements")
    start = _scalar_value(start, "linspace")
    end = _scalar_value(end, "linspace")
    if dtype is None:
        dtype = tensorplay.get_default_dtype()
    shape = (int(steps),)
    normalized = _normalize_placements(placements, mesh, 1)
    return _factory(
        tensorplay.linspace,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="linspace",
        start=start,
        end=end,
    )


def logspace(
    start: Any,
    end: Any,
    steps: int,
    base: float = 10.0,
    *,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    if placements is not None and any(isinstance(p, _StridedShard) for p in placements):
        raise ValueError("logspace does not support strided shard placements")
    start = _scalar_value(start, "logspace")
    end = _scalar_value(end, "logspace")
    if dtype is None:
        dtype = tensorplay.get_default_dtype()
    shape = (int(steps),)
    normalized = _normalize_placements(placements, mesh, 1)
    return _factory(
        tensorplay.logspace,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="logspace",
        start=start,
        end=end,
        base=base,
    )


def rand(
    *size: Any,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    shape = _normalize_size_args(size)
    normalized = _normalize_placements(placements, mesh, len(shape))
    return _factory(
        tensorplay.rand,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="rand",
    )


def randn(
    *size: Any,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    shape = _normalize_size_args(size)
    normalized = _normalize_placements(placements, mesh, len(shape))
    return _factory(
        tensorplay.randn,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="randn",
    )


def zeros(
    *size: Any,
    dtype: Any = None,
    layout: Any = "strided",
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    _validate_layout(layout)
    mesh = _normalize_mesh(device_mesh)
    shape = _normalize_size_args(size)
    normalized = _normalize_placements(placements, mesh, len(shape))
    return _factory(
        tensorplay.zeros,
        shape,
        mesh=mesh,
        placements=normalized,
        dtype=dtype,
        requires_grad=requires_grad,
        kind="zeros",
    )
