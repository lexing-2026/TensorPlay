from __future__ import annotations

import contextlib
import copy
import functools
import operator
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any, ParamSpec, TypeVar

import tensorplay as tp

from .. import distributed_core as dist

_R = TypeVar("_R")
_P = ParamSpec("_P")

__all__ = [
    "LocalIntNode",
    "LocalTensor",
    "LocalTensorMode",
    "LocalRunnerMode",
    "local_tensor_mode",
    "enabled_local_tensor_mode",
    "get_local_tensor_mode_list",
    "maybe_run_for_local_tensor",
    "maybe_disable_local_tensor_mode",
    "rank_map",
    "tensor_map",
    "reconcile_args",
    "maybe_enable_local_tracker",
    "get_generator_seed_for_device_type",
]


def _is_in_fake_tensor_mode() -> bool:
    return False


def _reduce_multidim_lists(values: list[Any], reduce_func: Callable[[list[Any]], Any]) -> Any:
    if not values:
        raise ValueError("values cannot be empty")
    first = values[0]
    if isinstance(first, list):
        if not all(isinstance(item, list) and len(item) == len(first) for item in values):
            raise ValueError("nested list shapes differ")
        return [_reduce_multidim_lists([item[index] for item in values], reduce_func) for index in range(len(first))]
    return reduce_func(values)


def _is_inplace_op(op: Any) -> bool:
    name = getattr(op, "__name__", getattr(op, "name", ""))
    return isinstance(name, str) and name.endswith("_")


def _int_on_rank(value: int | "LocalIntNode", rank: int) -> int:
    if isinstance(value, LocalIntNode):
        return value._local_ints[rank]
    return int(value)


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(value) for value in flat_args)


def _check_for_subclass_arg(value: object) -> bool:
    return isinstance(value, tp.Tensor) and type(value) is not tp.Tensor


def _map_to_rank_local_val(value: Any, rank: int) -> Any:
    if isinstance(value, LocalTensor):
        return value._local_tensors[rank]
    if isinstance(value, LocalIntNode):
        return value._local_ints[rank]
    return value


def _collect_accelerator_rng_states() -> dict[int, Any]:
    return {}


def _set_accelerator_rng_states(rng_states: dict[int, Any]) -> None:
    del rng_states


def _get_rng_state() -> tuple[Any, dict[int, Any]]:
    return tp.get_rng_state(), {}


def _set_rng_state(cpu_state: Any, accelerator_states: dict[int, Any]) -> None:
    tp.set_rng_state(cpu_state)
    _set_accelerator_rng_states(accelerator_states)


def _combine_int_rank_results(rank_results: dict[int, int]) -> int | LocalIntNode:
    values = list(rank_results.values())
    if values and all(value == values[0] for value in values):
        return values[0]
    return LocalIntNode(rank_results)


def _combine_any_rank_results(rank_results: dict[int, Any]) -> Any:
    values = list(rank_results.values())
    if not values:
        return None
    first = values[0]
    if isinstance(first, tp.Tensor):
        return LocalTensor(rank_results)
    if isinstance(first, LocalTensor):
        return LocalTensor({rank: value._local_tensors[rank] for rank, value in rank_results.items()})
    if isinstance(first, int) and not isinstance(first, bool):
        return _combine_int_rank_results(rank_results)
    if isinstance(first, (list, tuple)):
        return type(first)(_combine_rank_results({rank: value for rank, value in rank_results.items()}, None))
    if not all(value == first for value in values):
        raise AssertionError("rank results must agree for non-tensor values")
    return first


def _combine_rank_results(rank_results: dict[int, Any], default: Any | None = None) -> Any:
    first = next(iter(rank_results.values()))
    if isinstance(first, (list, tuple)):
        length = max(len(value) for value in rank_results.values())
        result = []
        for index in range(length):
            result.append(_combine_any_rank_results({rank: value[index] if index < len(value) else default for rank, value in rank_results.items()}))
        return tuple(result) if isinstance(first, tuple) else result
    return _combine_any_rank_results(rank_results)


def _zero_sized_like(tensor: Any, dim: int) -> Any:
    shape = list(tensor.shape)
    shape[dim] = 0
    return tp.empty(shape, dtype=tensor.dtype, device=tensor.device)


def _map_structure(value: Any, rank: int) -> Any:
    if isinstance(value, LocalTensor):
        return value._local_tensors[rank]
    if isinstance(value, LocalIntNode):
        return value._local_ints[rank]
    if isinstance(value, tuple):
        return tuple(_map_structure(item, rank) for item in value)
    if isinstance(value, list):
        return [_map_structure(item, rank) for item in value]
    if isinstance(value, dict):
        return {key: _map_structure(item, rank) for key, item in value.items()}
    return value


def _combine_structure(values: dict[int, Any]) -> Any:
    first = next(iter(values.values()))
    if isinstance(first, tuple):
        return tuple(_combine_structure({rank: value[index] for rank, value in values.items()}) for index in range(len(first)))
    if isinstance(first, list):
        return [_combine_structure({rank: value[index] for rank, value in values.items()}) for index in range(len(first))]
    if isinstance(first, dict):
        keys = set(first)
        if not all(set(value) == keys for value in values.values()):
            raise ValueError("rank results have different mappings")
        return {key: _combine_structure({rank: value[key] for rank, value in values.items()}) for key in first}
    return _combine_any_rank_results(values)


def _for_each_rank_run_func(func: Callable[..., Any], ranks: frozenset[int], args: Sequence[Any], kwargs: dict[str, Any], *, alias: bool = True) -> Any:
    del alias
    results: dict[int, Any] = {}
    state = _get_rng_state()
    for rank in sorted(ranks):
        _set_rng_state(*state)
        results[rank] = func(*_map_structure(tuple(args), rank), **_map_structure(kwargs, rank))
    return _combine_structure(results)


def _get_extra_dispatch_keys(tensor: Any) -> None:
    del tensor
    return None


class LocalIntNode:
    def __new__(cls, local_ints: Mapping[int, int]):
        values = {int(rank): int(value) for rank, value in local_ints.items()}
        if len(set(values.values())) == 1:
            return int(next(iter(values.values())))
        return super().__new__(cls)

    def __init__(self, local_ints: Mapping[int, int]) -> None:
        self._local_ints = {int(rank): int(value) for rank, value in local_ints.items()}

    def maybe_as_int(self) -> None:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def clone(self) -> "LocalIntNode":
        return LocalIntNode(self._local_ints)

    def _str(self) -> str:
        return f"LocalIntNode({self._local_ints})"

    __str__ = _str
    __repr__ = _str

    def _graph_repr(self) -> str:
        return self._str()

    def is_symbolic(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False

    def _binary(self, other: Any, op: Callable[[int, int], int]) -> Any:
        return LocalIntNode({rank: op(value, _int_on_rank(other, rank)) for rank, value in self._local_ints.items()})

    def sym_max(self, other: Any) -> Any:
        return self._binary(other, max)

    def sym_min(self, other: Any) -> Any:
        return self._binary(other, min)

    def sym_sum(self, other: Sequence[Any]) -> Any:
        result: Any = 0
        for value in other:
            result = self.add(value) if result == 0 else result.add(value)
        return result

    def neg(self) -> Any:
        return LocalIntNode({rank: -value for rank, value in self._local_ints.items()})

    def add(self, other: Any) -> Any:
        return self._binary(other, operator.add)

    def sub(self, other: Any) -> Any:
        return self._binary(other, operator.sub)

    def mul(self, other: Any) -> Any:
        return self._binary(other, operator.mul)

    def floordiv(self, other: Any) -> Any:
        return self._binary(other, operator.floordiv)

    def mod(self, other: Any) -> Any:
        return self._binary(other, operator.mod)

    int_floordiv = floordiv

    def eq(self, other: Any) -> bool:
        return all(value == _int_on_rank(other, rank) for rank, value in self._local_ints.items())

    def ne(self, other: Any) -> bool:
        return not self.eq(other)

    def ge(self, other: Any) -> bool:
        return all(value >= _int_on_rank(other, rank) for rank, value in self._local_ints.items())

    def le(self, other: Any) -> bool:
        return all(value <= _int_on_rank(other, rank) for rank, value in self._local_ints.items())

    def gt(self, other: Any) -> bool:
        return all(value > _int_on_rank(other, rank) for rank, value in self._local_ints.items())

    def lt(self, other: Any) -> bool:
        return all(value < _int_on_rank(other, rank) for rank, value in self._local_ints.items())

    def wrap_int(self, number: int) -> Any:
        return number

    __neg__ = neg
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __floordiv__ = floordiv
    __mod__ = mod


class _LocalDeviceHandle:
    def __init__(self, device_handle: Any, device_type: str) -> None:
        self._device_handle = device_handle
        self._device_type = device_type

    def get_rng_state(self) -> Any:
        mode = enabled_local_tensor_mode()
        if mode is None:
            return self._device_handle.get_rng_state()
        return LocalTensor({rank: self._device_handle.get_rng_state() for rank in mode.ranks})

    def set_rng_state(self, state: Any) -> None:
        if isinstance(state, LocalTensor):
            for rank, value in state._local_tensors.items():
                del rank
                self._device_handle.set_rng_state(value)
        else:
            self._device_handle.set_rng_state(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._device_handle, name)


class _LocalOffsetBasedRNGTracker:
    def __init__(self, device_type: str = "cpu") -> None:
        self._device_type = device_type
        self.distribute_region_enabled = True

    @property
    def _device(self) -> Any:
        return tp.device(self._device_type)

    def _set_pre_op_offset(self, state: Any, spec: Any) -> None:
        del state, spec

    def _set_post_op_offset(self, state: Any, spec: Any, old_offset: Any) -> None:
        del state, spec, old_offset

    @contextlib.contextmanager
    def _distribute_region(self, spec: Any, generator: Any = None):
        del spec, generator
        yield


_LOCAL_TENSOR_ATTR_PREFIX = "_local_tensor_"


def _is_local_tensor_attr(attr: str) -> bool:
    return attr.startswith(_LOCAL_TENSOR_ATTR_PREFIX)


def _to_local_tensor_attr(rank: int) -> str:
    return f"{_LOCAL_TENSOR_ATTR_PREFIX}{rank}"


def _from_local_tensor_attr(attr: str) -> int:
    if not _is_local_tensor_attr(attr):
        raise ValueError(f"invalid local tensor attribute {attr}")
    return int(attr[len(_LOCAL_TENSOR_ATTR_PREFIX):])


def _all_elements_same(values: list[Any]) -> bool:
    return not values or all(value == values[0] for value in values[1:])


def _compute_local_tensor_meta(local_tensors: dict[int, Any]):
    if not local_tensors:
        raise ValueError("LocalTensor cannot be empty")
    first = next(iter(local_tensors.values()))
    shape = tuple(first.shape)
    strides = tuple(first.stride()) if callable(getattr(first, "stride", None)) else tuple(first.stride)
    for value in local_tensors.values():
        if value.dtype != first.dtype:
            raise ValueError("local shards must use one dtype")
        if getattr(value, "layout", None) != getattr(first, "layout", None):
            raise ValueError("local shards must use one layout")
    for dim in range(len(shape)):
        sizes = [int(value.shape[dim]) for value in local_tensors.values()]
        if not _all_elements_same(sizes):
            shape = shape[:dim] + (LocalIntNode(dict(zip(local_tensors, sizes))),) + shape[dim + 1:]
        dim_strides = [int(value.stride()[dim]) for value in local_tensors.values()]
        if not _all_elements_same(dim_strides):
            strides = strides[:dim] + (LocalIntNode(dict(zip(local_tensors, dim_strides))),) + strides[dim + 1:]
    return list(shape), list(strides), first.device, first.dtype, getattr(first, "layout", None), None


class LocalTensor:
    def __init__(self, local_tensors: dict[int, Any], requires_grad: bool = False) -> None:
        if not local_tensors:
            raise ValueError("LocalTensor cannot be empty")
        self._local_tensors = {int(rank): value for rank, value in local_tensors.items()}
        self._ranks = frozenset(self._local_tensors)
        self._size, self._stride, self._device, self._dtype, self._layout, _ = _compute_local_tensor_meta(self._local_tensors)
        self.requires_grad = bool(requires_grad)

    def __deepcopy__(self, memo: dict[Any, Any] | None = None) -> "LocalTensor":
        return LocalTensor({rank: copy.deepcopy(value, memo) for rank, value in self._local_tensors.items()}, self.requires_grad)

    @property
    def shape(self) -> tuple[Any, ...]:
        return tuple(self._size)

    @property
    def ndim(self) -> int:
        return len(self._size)

    @property
    def device(self) -> Any:
        return self._device

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def layout(self) -> Any:
        return self._layout

    def stride(self, dim: int | None = None) -> Any:
        return self._stride if dim is None else self._stride[dim]

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def numel(self) -> int:
        result = 1
        for size in self._size:
            result *= int(size) if not isinstance(size, LocalIntNode) else size._local_ints[next(iter(self._ranks))]
        return result

    def __repr__(self) -> str:
        body = ",\n".join(f"  {rank}: {value!r}" for rank, value in self._local_tensors.items())
        return f"LocalTensor(\n{body}\n)"

    def __getattr__(self, name: str) -> Any:
        if _is_local_tensor_attr(name):
            rank = _from_local_tensor_attr(name)
            if rank not in self._local_tensors:
                raise AttributeError(name)
            return self._local_tensors[rank]
        first = next(iter(self._local_tensors.values()))
        value = getattr(first, name)
        if callable(value):
            @functools.wraps(value)
            def method(*args: Any, **kwargs: Any) -> Any:
                return _for_each_rank_run_func(
                    lambda tensor, *method_args, **method_kwargs: getattr(tensor, name)(
                        *method_args, **method_kwargs
                    ),
                    self._ranks,
                    (self, *args),
                    kwargs,
                )
            return method
        return value

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        return ([_to_local_tensor_attr(rank) for rank in self._ranks], ())

    @staticmethod
    def __tensor_unflatten__(inner_tensors: dict[str, Any], flatten_spec: Any, outer_size: Any, outer_stride: Any) -> "LocalTensor":
        del flatten_spec, outer_size, outer_stride
        return LocalTensor({_from_local_tensor_attr(name): value for name, value in inner_tensors.items()})

    @classmethod
    def __torch_dispatch__(cls, func: Any, types: tuple[Any, ...], args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        del types
        mode = next((value for value in _iter_local_values((args, kwargs or {})) if isinstance(value, LocalTensor)), None)
        if mode is None:
            return NotImplemented
        return _for_each_rank_run_func(func, mode._ranks, args, kwargs or {})

    def numpy(self, force: bool = False) -> Any:
        del force
        return self.reconcile().numpy()

    def contiguous(self, memory_format: Any = None) -> "LocalTensor":
        return tensor_map(self, lambda rank, value: value.contiguous() if memory_format is None else value.contiguous(memory_format=memory_format))

    def is_contiguous(self, memory_format: Any = None) -> bool:
        return all(value.is_contiguous() if memory_format is None else value.is_contiguous(memory_format=memory_format) for value in self._local_tensors.values())

    def tolist(self) -> Any:
        equal = self._equal_local_tensors()
        if equal is None:
            raise RuntimeError("local shards do not have one common value")
        return equal.tolist()

    def reconcile(self) -> Any:
        equal = self._equal_local_tensors()
        if equal is None or not isinstance(equal, tp.Tensor):
            raise RuntimeError("local shards must be equal to reconcile")
        result = equal.clone().detach()
        result.requires_grad_(self.requires_grad)
        return result

    def _equal_local_tensors(self) -> Any:
        values = list(self._local_tensors.values())
        first = values[0]
        if all(value.equal(first) for value in values[1:]):
            return first
        if all(value.shape == first.shape for value in values[1:]):
            return first.shape
        return None

    def _sync_meta(self) -> None:
        self._size, self._stride, self._device, self._dtype, self._layout, _ = _compute_local_tensor_meta(self._local_tensors)

    def clone(self) -> "LocalTensor":
        return LocalTensor({rank: value.clone() for rank, value in self._local_tensors.items()}, self.requires_grad)

    def detach(self) -> "LocalTensor":
        return LocalTensor({rank: value.detach() for rank, value in self._local_tensors.items()}, False)

    def __getitem__(self, index: Any) -> "LocalTensor":
        return tensor_map(self, lambda rank, value: value[_map_structure(index, rank)])

    def __setitem__(self, index: Any, value: Any) -> None:
        for rank, tensor in self._local_tensors.items():
            tensor[_map_structure(index, rank)] = _map_structure(value, rank)

    def _binary_tensor(self, other: Any, operation: Callable[[Any, Any], Any]) -> "LocalTensor":
        return LocalTensor({rank: operation(value, _map_structure(other, rank)) for rank, value in self._local_tensors.items()}, self.requires_grad)

    __add__ = lambda self, other: self._binary_tensor(other, operator.add)
    __radd__ = lambda self, other: self._binary_tensor(other, lambda a, b: operator.add(b, a))
    __sub__ = lambda self, other: self._binary_tensor(other, operator.sub)
    __rsub__ = lambda self, other: self._binary_tensor(other, lambda a, b: operator.sub(b, a))
    __mul__ = lambda self, other: self._binary_tensor(other, operator.mul)
    __rmul__ = __mul__
    __truediv__ = lambda self, other: self._binary_tensor(other, operator.truediv)
    __neg__ = lambda self: LocalTensor({rank: -value for rank, value in self._local_tensors.items()}, self.requires_grad)


def _iter_local_values(value: Any):
    if isinstance(value, LocalTensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_local_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_local_values(item)


class _LocalContiguous:
    @staticmethod
    def apply(value: LocalTensor, memory_format: Any = None) -> LocalTensor:
        return value.contiguous(memory_format)


_PROCESS_LOCAL_TENSOR_MODE: list["LocalTensorMode"] = []
_THREAD_LOCAL_TENSOR_MODE = threading.local()


def get_local_tensor_mode_list() -> list["LocalTensorMode"]:
    if not hasattr(_THREAD_LOCAL_TENSOR_MODE, "value"):
        _THREAD_LOCAL_TENSOR_MODE.value = []
    return _THREAD_LOCAL_TENSOR_MODE.value


class LocalTensorMode:
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(self, ranks: int | frozenset[int] | set[int]) -> None:
        self.ranks = frozenset(range(ranks)) if isinstance(ranks, int) else frozenset(ranks)
        if not self.ranks:
            raise ValueError("at least one rank is required")
        self._disable = True
        self._per_rank_rng_states: dict[int, tuple[Any, dict[int, Any]]] = {}

    def __enter__(self) -> "LocalTensorMode":
        get_local_tensor_mode_list().append(self)
        self.enable_()
        state = _get_rng_state()
        self._per_rank_rng_states = {rank: (state[0].clone(), {}) for rank in self.ranks}
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        self.disable_()
        get_local_tensor_mode_list().pop()

    def __torch_dispatch__(self, func: Any, types: tuple[Any, ...], args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> Any:
        del types
        local = next(
            (value for value in _iter_local_values((args, kwargs or {})) if isinstance(value, LocalTensor)),
            None,
        )
        if local is None or self._disable:
            return func(*args, **(kwargs or {}))
        return _for_each_rank_run_func(func, local._ranks, args, kwargs or {})

    def disable_(self) -> None:
        self._disable = True

    def enable_(self) -> None:
        self._disable = False

    @contextlib.contextmanager
    def disable(self) -> Generator[None, None, None]:
        was_disabled = self._disable
        self.disable_()
        try:
            yield
        finally:
            self._disable = was_disabled

    def rank_map(self, cb: Callable[[int], Any]) -> LocalTensor:
        with self.disable():
            return LocalTensor({rank: cb(rank) for rank in self.ranks})

    def tensor_map(self, tensor: LocalTensor, cb: Callable[[int, Any], Any | None]) -> LocalTensor:
        with self.disable():
            values = {rank: result for rank, value in tensor._local_tensors.items() if (result := cb(rank, value)) is not None}
        return LocalTensor(values)

    def _any_local_rng_state(self) -> tuple[Any, dict[int, Any]]:
        return self._per_rank_rng_states[next(iter(self.ranks))]

    def _patch_device_mesh(self) -> None:
        return None

    def _unpatch_device_mesh(self) -> None:
        return None

    def _patch_random_functions(self) -> None:
        return None

    def _unpatch_random_functions(self) -> None:
        return None

    def _patch_dist(self) -> None:
        return None

    def _unpatch_dist(self) -> None:
        return None


class _LocalRandom:
    @staticmethod
    def torch_manual_seed(seed: int) -> Any:
        return tp.manual_seed(int(seed))

    @staticmethod
    def torch_initial_seed() -> int:
        return tp.initial_seed()


class _LocalDeviceMesh:
    @staticmethod
    def get_coordinate(mesh: Any) -> tuple[Any, ...] | None:
        mode = enabled_local_tensor_mode()
        if mode is None:
            return mesh.get_coordinate()
        return tuple(LocalIntNode({rank: mesh._coords_of(mesh._rank_map.index(rank))[dim] for rank in mode.ranks if rank in mesh._rank_map}) for dim in range(mesh.ndim()))

    @staticmethod
    def _is_current_rank_part_of_mesh(mesh: Any) -> bool:
        return _LocalDeviceMesh.get_coordinate(mesh) is not None

    @staticmethod
    def _sym_get_coordinate(mesh: Any, index: int) -> Any:
        return _LocalDeviceMesh.get_coordinate(mesh)[index]

    @staticmethod
    def get_rank(mesh: Any) -> Any:
        del mesh
        mode = enabled_local_tensor_mode()
        if mode is None:
            return dist.get_rank()
        return LocalIntNode({rank: rank for rank in mode.ranks})

    @staticmethod
    def get_local_rank(mesh: Any, mesh_dim: int | str | None = None) -> Any:
        coordinate = _LocalDeviceMesh.get_coordinate(mesh)
        if coordinate is None:
            raise RuntimeError("rank is outside the mesh")
        if mesh_dim is None:
            mesh_dim = 0
        if isinstance(mesh_dim, str):
            mesh_dim = mesh.mesh_dim_names.index(mesh_dim)
        return coordinate[mesh_dim]


class _LocalDist:
    @staticmethod
    def get_rank(group: Any = None) -> Any:
        del group
        mode = enabled_local_tensor_mode()
        return dist.get_rank() if mode is None else LocalIntNode({rank: rank for rank in mode.ranks})


def reconcile_args(args: Any, kwargs: dict[str, Any] | None = None) -> Any:
    return _reconcile_structure((args, kwargs or {}))


def _reconcile_structure(value: Any) -> Any:
    if isinstance(value, LocalTensor):
        return value.reconcile()
    if isinstance(value, tuple):
        return tuple(_reconcile_structure(item) for item in value)
    if isinstance(value, list):
        return [_reconcile_structure(item) for item in value]
    if isinstance(value, dict):
        return {key: _reconcile_structure(item) for key, item in value.items()}
    return value


def local_tensor_mode() -> LocalTensorMode | None:
    modes = get_local_tensor_mode_list()
    return modes[-1] if modes else None


def enabled_local_tensor_mode() -> LocalTensorMode | None:
    mode = local_tensor_mode()
    return mode if mode is not None and not mode._disable else None


def maybe_run_for_local_tensor(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        mode = enabled_local_tensor_mode()
        if mode is None:
            return func(*args, **kwargs)
        with mode.disable():
            return _for_each_rank_run_func(func, mode.ranks, args, kwargs)  # type: ignore[return-value]
    return wrapper


def rank_map(cb: Callable[[int], Any]) -> Any:
    mode = enabled_local_tensor_mode()
    return mode.rank_map(cb) if mode is not None else cb(dist.get_rank())


def tensor_map(tensor: Any, cb: Callable[[int, Any], Any | None]) -> Any:
    mode = enabled_local_tensor_mode()
    if mode is None:
        result = cb(dist.get_rank(), tensor)
        if result is None:
            raise ValueError("callback returned None")
        return result
    if not isinstance(tensor, LocalTensor):
        raise TypeError("tensor_map expects a LocalTensor in local mode")
    return mode.tensor_map(tensor, cb)


def maybe_disable_local_tensor_mode() -> contextlib.AbstractContextManager:
    mode = local_tensor_mode()
    return mode.disable() if mode is not None else contextlib.nullcontext()


def maybe_enable_local_tracker(device_type: str, distribute_region_enabled: bool, spec: Any, generator: Any):
    mode = enabled_local_tensor_mode()
    if mode is None:
        return None
    tracker = _LocalOffsetBasedRNGTracker(device_type)
    tracker.distribute_region_enabled = distribute_region_enabled
    return tracker._distribute_region(spec, generator)


def get_generator_seed_for_device_type(device_type: str) -> int:
    del device_type
    return tp.initial_seed()


from . import _p10d


class _ExceptionRaisingThread(threading.Thread):
    def __init__(self, group: Any, target: Callable[..., Any], name: str | None = None, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None, daemon: bool | None = None) -> None:
        del group
        super().__init__(target=target, name=name, args=args, kwargs=kwargs or {}, daemon=daemon)
        self.exception: BaseException | None = None

    def run(self) -> None:
        try:
            super().run()
        except BaseException as error:
            self.exception = error

    def join(self, timeout: float | None = None) -> None:
        super().join(timeout)
        if self.exception is not None:
            raise self.exception


class LocalRunnerMode:
    _current = threading.local()

    def __init__(self, ranks: int | frozenset[int], concurrency: int | None = None, fn: Callable[..., Any] | None = None) -> None:
        self.ranks = frozenset(range(ranks)) if isinstance(ranks, int) else frozenset(ranks)
        self.concurrency = concurrency
        self.fn = fn
        self._old = None

    def __enter__(self) -> "LocalRunnerMode":
        self._old = getattr(self._current, "value", None)
        self._current.value = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        self._current.value = self._old

    def _run(self, rank: int) -> Any:
        if self.fn is None:
            return None
        return self.fn(rank)

    def _acquire_run_lock(self) -> None:
        return None

    def _release_run_lock(self) -> None:
        return None

    def _assert_holds_run_lock(self) -> None:
        return None

    def _get_recv_object(self, src: int, dst: int) -> Any:
        del src, dst
        return None

    def _signal_send(self, src: int, dst: int, obj: Any) -> None:
        del src, dst, obj

    def _wait_recv(self, src: int, dst: int, post: bool = False) -> Any:
        del src, dst, post
        return None

    @classmethod
    def current(cls) -> "LocalRunnerMode | None":
        return getattr(cls._current, "value", None)


class _LocalPhiloxState:
    def __init__(self, state: Any) -> None:
        self._state = state
        self._seed = 0
        self._offset = 0

    @property
    def state(self) -> Any:
        return self._state

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        self._offset = int(value)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = int(value)

    def apply_to_local_tensor_mode(self, device_handle: Any) -> None:
        del device_handle
