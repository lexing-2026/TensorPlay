"""Garbage-collector helpers for diagnosing captured-value retention."""

from __future__ import annotations

import gc
import types
import weakref
from typing import Any

__all__ = ["find_legit_leaks_from_referrers"]

_SKIP_TYPES = (types.FrameType, types.ModuleType)


def _is_globals_or_locals(value: Any) -> bool:
    return value is globals() or value is locals()


def _is_gm_meta_like_dict(value: dict[Any, Any], object_value: Any) -> bool:
    return value.get("val") is object_value


def _dict_is_attr_of_tracked_fake(value: dict[Any, Any]) -> bool:
    return any(
        getattr(parent, "__dict__", None) is value
        for parent in gc.get_referrers(value)
        if hasattr(parent, "__dict__")
    )


def find_legit_leaks_from_referrers(active_fakes: weakref.WeakSet[Any]) -> weakref.WeakSet[Any]:
    """Return values whose remaining referrers are graph bookkeeping objects."""

    result: weakref.WeakSet[Any] = weakref.WeakSet()
    for value in list(active_fakes):
        flagged = False
        for referrer in gc.get_referrers(value):
            if _is_globals_or_locals(referrer) or isinstance(referrer, _SKIP_TYPES):
                continue
            if isinstance(referrer, dict) and (
                _is_gm_meta_like_dict(referrer, value)
                or _dict_is_attr_of_tracked_fake(referrer)
            ):
                continue
            flagged = True
            break
        if not flagged:
            result.add(value)
    return result
