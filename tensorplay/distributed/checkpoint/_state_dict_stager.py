from __future__ import annotations

import copy
from typing import Any

import tensorplay as tp


class StateDictStager:
    def __init__(self, pin_memory: bool = False, share_memory: bool = False) -> None:
        cuda = getattr(tp, "cuda", None)
        cuda_available = callable(getattr(cuda, "is_available", None)) and bool(
            cuda.is_available()
        )
        self.pin_memory = bool(pin_memory) and cuda_available
        self.share_memory = bool(share_memory)
        self._keep_alive_values: list[Any] = []
        self._storage_cache: dict[Any, Any] = {}

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
        if self.pin_memory and not staged.is_pinned():
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
            try:
                value = tp.empty((0,), dtype=source.dtype, device="cpu")
                value.set_(
                    staged_storage,
                    int(source.storage_offset()),
                    tuple(int(size) for size in source.shape),
                    tuple(int(stride) for stride in source.stride()),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                value = source.to(device="cpu").clone()
        else:
            try:
                value = source.to(device="cpu", non_blocking=non_blocking)
            except TypeError:
                value = source.to(device="cpu")
            if value is source:
                value = value.clone()
        memo[id(x)] = value
        return value

    def deepcopy_with_tensor_offload(
        self,
        x: Any,
        memo: dict[int, Any] | None = None,
        _nil: list[Any] | None = None,
        non_blocking: bool = False,
    ) -> Any:
        del _nil
        memo = {} if memo is None else memo
        if isinstance(x, tp.Tensor):
            return self._offload_tensor(x, memo, non_blocking)
        cached = memo.get(id(x))
        if cached is not None:
            return cached
        if isinstance(x, dict):
            result: dict[Any, Any] = {}
            memo[id(x)] = result
            for key, value in x.items():
                copied_key = self.deepcopy_with_tensor_offload(
                    key, memo, non_blocking=non_blocking
                )
                result[copied_key] = self.deepcopy_with_tensor_offload(
                    value, memo, non_blocking=non_blocking
                )
            return result
        if isinstance(x, list):
            result = []
            memo[id(x)] = result
            result.extend(
                self.deepcopy_with_tensor_offload(
                    value, memo, non_blocking=non_blocking
                )
                for value in x
            )
            return result
        if isinstance(x, tuple):
            result = tuple(
                self.deepcopy_with_tensor_offload(
                    value, memo, non_blocking=non_blocking
                )
                for value in x
            )
            memo[id(x)] = result
            return result
        return copy.deepcopy(x, memo)

    def stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self._storage_cache.clear()
        return self.deepcopy_with_tensor_offload(state_dict, non_blocking=bool(kwargs.get("non_blocking", False)))

    def _keep_alive(self, x: Any, memo: Any) -> Any:
        del memo
        self._keep_alive_values.append(x)
        return x

    def _reconstruct(self, *args: Any, **kwargs: Any) -> Any:
        del kwargs
        return args[0] if args else None

    def close(self) -> None:
        self._keep_alive_values.clear()
        self._storage_cache.clear()
