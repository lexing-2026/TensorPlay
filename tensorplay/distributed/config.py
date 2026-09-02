"""Runtime switches used by distributed execution."""

from __future__ import annotations

import contextlib
import os
import pickle
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = [
    "compile_on_one_rank",
    "use_torchcomms",
    "pipeline_per_direction_p2p",
    "patch",
    "save_config",
    "load_config",
    "get_config_copy",
]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


compile_on_one_rank = _env_bool("TP_DISTRIBUTED_COMPILE_ON_ONE_RANK", False)
use_torchcomms = _env_bool("TP_DISTRIBUTED_USE_TORCHCOMMS", False)
pipeline_per_direction_p2p = _env_bool(
    "TP_DISTRIBUTED_PIPELINE_PER_DIRECTION_P2P", False
)

_CONFIG_NAMES = frozenset(__all__[:3])


def _validate_changes(changes: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(changes, Mapping):
        raise TypeError(f"expected a mapping, got {type(changes)!r}")
    result = dict(changes)
    unknown = [key for key in result if key not in _CONFIG_NAMES]
    if unknown:
        raise KeyError(unknown[0])
    for key, value in result.items():
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be a bool")
    return result


def get_config_copy() -> dict[str, bool]:
    return {name: bool(globals()[name]) for name in _CONFIG_NAMES}


def save_config() -> bytes:
    return pickle.dumps(get_config_copy(), protocol=2)


def load_config(value: bytes | Mapping[str, Any]) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = pickle.loads(bytes(value))
    changes = _validate_changes(value)
    for key, item in changes.items():
        globals()[key] = item


def patch(
    arg1: str | Mapping[str, Any] | None = None,
    arg2: Any = None,
    **kwargs: Any,
) -> contextlib.AbstractContextManager[Any]:
    if arg1 is None:
        if arg2 is not None:
            raise TypeError("arg2 requires a configuration name")
        changes = kwargs
    elif isinstance(arg1, str):
        if not kwargs:
            changes = {arg1: arg2}
        else:
            raise TypeError("cannot combine positional and keyword changes")
    else:
        if arg2 is not None or kwargs:
            raise TypeError("cannot combine mapping and other changes")
        changes = arg1
    checked = _validate_changes(changes)

    @contextlib.contextmanager
    def apply_changes() -> Iterator[None]:
        previous = {key: globals()[key] for key in checked}
        try:
            for key, value in checked.items():
                globals()[key] = value
            yield
        finally:
            for key, value in previous.items():
                globals()[key] = value

    return apply_changes()
