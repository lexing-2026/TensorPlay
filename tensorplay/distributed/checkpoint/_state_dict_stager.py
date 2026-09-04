from __future__ import annotations

import copy
import types
import weakref
from copyreg import dispatch_table
from typing import Any

import tensorplay as tp


def _deepcopy_atomic(value: Any, memo: dict[int, Any]) -> Any:
    del memo
    return value


def _deepcopy_dict(value: dict[Any, Any], memo: dict[int, Any], stager: "StateDictStager" | None = None, non_blocking: bool = False) -> dict[Any, Any]:
    if stager is None:
        return copy.deepcopy(value, memo)
    result: dict[Any, Any] = {}
    memo[id(value)] = result
    for key, child in value.items():
        result[stager.deepcopy_with_tensor_offload(key, memo, non_blocking=non_blocking)] = stager.deepcopy_with_tensor_offload(child, memo, non_blocking=non_blocking)
    return result


def _deepcopy_list(value: list[Any], memo: dict[int, Any], stager: "StateDictStager" | None = None, non_blocking: bool = False) -> list[Any]:
    if stager is None:
        return copy.deepcopy(value, memo)
    result: list[Any] = []
    memo[id(value)] = result
    result.extend(stager.deepcopy_with_tensor_offload(child, memo, non_blocking=non_blocking) for child in value)
    return result


def _deepcopy_tuple(value: tuple[Any, ...], memo: dict[int, Any], stager: "StateDictStager" | None = None, non_blocking: bool = False) -> tuple[Any, ...]:
    if stager is None:
        return copy.deepcopy(value, memo)
    copied = [
        stager.deepcopy_with_tensor_offload(child, memo, non_blocking=non_blocking)
        for child in value
    ]
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    if all(original is child for original, child in zip(value, copied)):
        return value
    return tuple(copied)


def _deepcopy_method(value: types.MethodType, memo: dict[int, Any], stager: "StateDictStager" | None = None, non_blocking: bool = False) -> Any:
    if stager is None:
        return copy.deepcopy(value, memo)
    return types.MethodType(value.__func__, stager.deepcopy_with_tensor_offload(value.__self__, memo, non_blocking=non_blocking))


class StateDictStager:
    def __init__(
        self,
        pin_memory: bool = False,
        share_memory: bool = False,
        pin_memory_min_bytes: int = 5,
    ) -> None:
        cuda = getattr(tp, "cuda", None)
        cuda_available = callable(getattr(cuda, "is_available", None)) and bool(
            cuda.is_available()
        )
        self.pin_memory = bool(pin_memory) and cuda_available
        self.share_memory = bool(share_memory)
        self.pin_memory_min_bytes = int(pin_memory_min_bytes)
        self._keep_alive_values: list[Any] = []
        self._storage_cache: dict[Any, Any] = {}
        self._shared_storage_markers: dict[Any, Any] = {}
        self._cached_storage_mapping = self._storage_cache
        self._deepcopy_dispatch: dict[type[Any], Any] = {
            type(None): _deepcopy_atomic,
            int: _deepcopy_atomic,
            float: _deepcopy_atomic,
            bool: _deepcopy_atomic,
            complex: _deepcopy_atomic,
            bytes: _deepcopy_atomic,
            str: _deepcopy_atomic,
            types.CodeType: _deepcopy_atomic,
            type: _deepcopy_atomic,
            range: _deepcopy_atomic,
            types.BuiltinFunctionType: _deepcopy_atomic,
            types.FunctionType: _deepcopy_atomic,
            weakref.ref: _deepcopy_atomic,
            property: _deepcopy_atomic,
            types.MethodType: _deepcopy_method,
            dict: _deepcopy_dict,
            tuple: _deepcopy_tuple,
            list: _deepcopy_list,
        }
        self._atomic_types = {
            type(None),
            int,
            float,
            bool,
            complex,
            bytes,
            str,
            types.CodeType,
            type,
            range,
            types.BuiltinFunctionType,
            types.FunctionType,
            weakref.ref,
            property,
        }

    @staticmethod
    def _storage_view(storage: Any) -> tp.Tensor:
        view = tp.empty((0,), dtype=tp.uint8, device=getattr(storage, "device", "cpu"))
        return view.set_(storage)

    def _copy_storage(self, source: Any, destination: Any) -> None:
        self._storage_view(destination).copy_(self._storage_view(source))

    def _stage_untyped_storage(self, storage: Any, non_blocking: bool = False) -> Any:
        if not tp.is_storage(storage):
            raise TypeError("storage staging expects an untyped storage")
        nbytes = int(storage.nbytes())
        if nbytes == 0:
            staged = tp.empty((0,), dtype=tp.uint8, device="cpu")
        else:
            source = tp.empty(
                (nbytes,), dtype=tp.uint8, device=getattr(storage, "device", "cpu")
            )
            source.set_(storage)
            try:
                staged = source.to(device="cpu", non_blocking=non_blocking)
            except TypeError:
                staged = source.to(device="cpu")
        if self.share_memory and not staged.is_shared():
            staged.share_memory_()
        if self.share_memory:
            handle = getattr(storage, "_cdata", None)
            if handle is None:
                handle = (
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                    str(getattr(storage, "device", "cpu")),
                )
            self._shared_storage_markers[handle] = staged
        if self.pin_memory and nbytes >= self.pin_memory_min_bytes and not staged.is_pinned():
            staged = staged.pin_memory()
        return staged.untyped_storage()

    def _offload_tensor(
        self,
        x: tp.Tensor,
        memo: dict[int, Any],
        non_blocking: bool = False,
    ) -> tp.Tensor:
        cached = memo.get(id(x))
        if cached is not None:
            return cached
        source = x.detach()
        value: tp.Tensor
        storage_getter = getattr(source, "untyped_storage", None)
        if callable(storage_getter):
            storage = storage_getter()
            storage_handle = getattr(storage, "_cdata", None)
            if storage_handle is None:
                storage_handle = (
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                    str(getattr(storage, "device", "cpu")),
                )
            staged_storage = self._storage_cache.get(storage_handle)
            if staged_storage is None:
                staged_storage = self._stage_untyped_storage(
                    storage, non_blocking=non_blocking
                )
                self._storage_cache[storage_handle] = staged_storage
            else:
                self._copy_storage(storage, staged_storage)
            try:
                value = tp.empty((0,), dtype=source.dtype, device="cpu")
                memo[id(x)] = value
                value.set_(
                    staged_storage,
                    int(source.storage_offset()),
                    tuple(int(size) for size in source.shape),
                    tuple(int(stride) for stride in source.stride()),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                value = source.to(device="cpu").clone()
                memo[id(x)] = value
            if self.share_memory:
                marker = self._shared_storage_markers.get(storage_handle)
                shared_memory = getattr(marker, "_shared_memory", None)
                if shared_memory is not None:
                    try:
                        value._shared_memory = shared_memory
                    except AttributeError:
                        pass
                elif not value.is_shared():
                    value.share_memory_()
        else:
            try:
                value = source.to(device="cpu", non_blocking=non_blocking)
            except TypeError:
                value = source.to(device="cpu")
            if value is source:
                value = value.clone()
        if hasattr(source, "__dict__"):
            for name, attribute in source.__dict__.items():
                try:
                    setattr(
                        value,
                        name,
                        self.deepcopy_with_tensor_offload(
                            attribute, memo, non_blocking=non_blocking
                        ),
                    )
                except (AttributeError, TypeError):
                    continue
        for name in getattr(source, "__slots__", ()):
            if hasattr(source, name):
                try:
                    setattr(
                        value,
                        name,
                        self.deepcopy_with_tensor_offload(
                            getattr(source, name), memo, non_blocking=non_blocking
                        ),
                    )
                except (AttributeError, TypeError):
                    continue
        memo[id(x)] = value
        return value

    def deepcopy_with_tensor_offload(
        self,
        x: Any,
        memo: dict[int, Any] | None = None,
        _nil: list[Any] | None = None,
        non_blocking: bool = False,
    ) -> Any:
        memo = {} if memo is None else memo
        sentinel = _nil if _nil is not None else []
        cached = memo.get(id(x), sentinel)
        if cached is not sentinel:
            return cached
        if isinstance(x, tp.Tensor):
            value = self._offload_tensor(x, memo, non_blocking)
        else:
            cls = type(x)
            if cls in self._atomic_types:
                value = x
            else:
                copier = self._deepcopy_dispatch.get(cls)
                if copier is not None:
                    value = copier(x, memo, self, non_blocking)
                else:
                    deepcopy = getattr(x, "__deepcopy__", None)
                    if deepcopy is not None:
                        value = deepcopy(memo)
                    else:
                        reducer = dispatch_table.get(cls)
                        if reducer is not None:
                            reduced = reducer(x)
                        else:
                            reduce_ex = getattr(x, "__reduce_ex__", None)
                            if reduce_ex is not None:
                                reduced = reduce_ex(4)
                            else:
                                reduce_fn = getattr(x, "__reduce__", None)
                                if reduce_fn is None:
                                    raise RuntimeError(
                                        f"object of type {cls} cannot be staged"
                                    )
                                reduced = reduce_fn()
                        if isinstance(reduced, str):
                            value = x
                        else:
                            if not isinstance(reduced, tuple):
                                raise RuntimeError(
                                    f"invalid reconstruction data for {cls}"
                                )
                            if len(reduced) == 2:
                                func, args = reduced
                                value = self._reconstruct(
                                    x, memo, func, args, non_blocking=non_blocking
                                )
                            elif len(reduced) == 3:
                                func, args, state = reduced
                                value = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    non_blocking=non_blocking,
                                )
                            elif len(reduced) == 4:
                                func, args, state, listiter = reduced
                                value = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    listiter,
                                    non_blocking=non_blocking,
                                )
                            elif len(reduced) == 5:
                                func, args, state, listiter, dictiter = reduced
                                value = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    listiter,
                                    dictiter,
                                    non_blocking=non_blocking,
                                )
                            else:
                                raise RuntimeError(
                                    f"invalid reconstruction tuple length {len(reduced)}"
                                )
        if value is not x:
            memo[id(x)] = value
            self._keep_alive(x, memo)
        return value

    def stage(
        self, state_dict: dict[str, Any], non_blocking: bool = False
    ) -> dict[str, Any]:
        return self.deepcopy_with_tensor_offload(
            state_dict, non_blocking=bool(non_blocking)
        )

    def _keep_alive(self, x: Any, memo: dict[int, Any]) -> None:
        self._keep_alive_values.append(x)
        memo.setdefault(id(memo), []).append(x)

    def _reconstruct(
        self,
        x: Any,
        memo: dict[int, Any],
        func: Any,
        args: tuple[Any, ...],
        state: Any = None,
        listiter: Any = None,
        dictiter: Any = None,
        non_blocking: bool = False,
    ) -> Any:
        copied_args = tuple(
            self.deepcopy_with_tensor_offload(
                arg, memo, non_blocking=non_blocking
            )
            for arg in args
        )
        value = func(*copied_args)
        memo[id(x)] = value
        if state is not None:
            copied_state = self.deepcopy_with_tensor_offload(
                state, memo, non_blocking=non_blocking
            )
            if hasattr(value, "__setstate__"):
                value.__setstate__(copied_state)
            else:
                if isinstance(copied_state, tuple) and len(copied_state) == 2:
                    copied_state, slot_state = copied_state
                else:
                    slot_state = None
                if copied_state is not None:
                    value.__dict__.update(copied_state)
                if slot_state is not None:
                    for key, child in slot_state.items():
                        setattr(value, key, child)
        if listiter is not None:
            for child in listiter:
                value.append(
                    self.deepcopy_with_tensor_offload(
                        child, memo, non_blocking=non_blocking
                    )
                )
        if dictiter is not None:
            for key, child in dictiter:
                copied_key = self.deepcopy_with_tensor_offload(
                    key, memo, non_blocking=non_blocking
                )
                copied_child = self.deepcopy_with_tensor_offload(
                    child, memo, non_blocking=non_blocking
                )
                value[copied_key] = copied_child
        return value

    def close(self) -> None:
        self._keep_alive_values.clear()
        self._storage_cache.clear()
        self._shared_storage_markers.clear()
        self._deepcopy_dispatch.clear()
