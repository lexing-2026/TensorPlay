"""Weight metadata and storage-sharing utilities for model archives."""

from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from typing import Any

__all__ = [
    "TensorProperties",
    "WeightType",
    "Weights",
    "get_complete_tensor",
    "group_weights",
]


class WeightType(IntEnum):
    """Role a packaged weight plays in the captured program."""

    PARAMETER = 0
    BUFFER = 1
    OPTIONAL_STATE = 2
    USER_INPUT = 3
    USER_OUTPUT = 4
    PARAMETER_MUTATION = 5
    BUFFER_MUTATION = 6
    USER_INPUT_MUTATION = 7
    GRADIENT_TO_PARAMETER = 8
    GRADIENT_TO_USER_INPUT = 9


def _end_ptr(value: Any) -> int | None:
    if not hasattr(value, "data_ptr") or not hasattr(value, "element_size"):
        return None
    try:
        return value.data_ptr() + value.numel() * value.element_size()
    except Exception:
        return None


class TensorProperties:
    def __init__(self, tensor: Any) -> None:
        self.is_fake = False
        self.is_contiguous = bool(getattr(tensor, "is_contiguous", lambda: False)())
        storage = getattr(tensor, "untyped_storage", lambda: None)()
        self.storage_ptr = getattr(storage, "data_ptr", lambda: None)()
        self.storage_size = getattr(storage, "nbytes", lambda: None)()
        self.start = getattr(tensor, "data_ptr", lambda: None)()
        self.end = _end_ptr(tensor)
        self.shape = tuple(getattr(tensor, "shape", ()))
        stride = getattr(tensor, "stride", None)
        self.stride = tuple(stride()) if callable(stride) else None
        self.offset = int(getattr(tensor, "storage_offset", lambda: 0)())

    def is_complete(self) -> bool:
        if not self.is_contiguous or self.storage_ptr is None or self.storage_size is None:
            return False
        return self.start == self.storage_ptr and self.end == self.storage_ptr + self.storage_size


class Weights(dict[str, tuple[Any, TensorProperties]]):
    def get_weight(self, name: str) -> tuple[Any, TensorProperties]:
        return self[name]

    def get_weight_properties(self, name: str) -> TensorProperties:
        return self[name][1]


def get_complete_tensor(group: set[tuple[str, str]], models_weights: dict[str, Weights]) -> Any:
    if not group:
        raise ValueError("weight group cannot be empty")
    for model_name, weight_name in group:
        tensor, properties = models_weights[model_name][weight_name]
        if properties.is_complete():
            return tensor
    first_model, first_name = next(iter(group))
    return models_weights[first_model][first_name][0]


def group_weights(all_weights: dict[str, Weights]) -> list[set[tuple[str, str]]]:
    groups: dict[Any, set[tuple[str, str]]] = defaultdict(set)
    for model_name, weights in all_weights.items():
        for weight_name, (tensor, properties) in weights.items():
            storage = getattr(tensor, "untyped_storage", lambda: None)()
            key = getattr(storage, "data_ptr", lambda: id(tensor))()
            groups[key].add((model_name, weight_name))
            if properties.storage_ptr is None:
                groups[(model_name, weight_name)].add((model_name, weight_name))
    return list(groups.values())
