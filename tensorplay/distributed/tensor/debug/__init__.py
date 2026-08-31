"""Diagnostics for distributed tensor layouts and collectives."""

from ._comm_mode import CommDebugMode
from ._visualize_sharding import visualize_sharding

__all__ = ["CommDebugMode", "visualize_sharding"]


def _clear_sharding_prop_cache() -> None:
    from .._ops.utils import _PROPAGATION_RULES

    _PROPAGATION_RULES.clear()


def _clear_python_sharding_prop_cache() -> None:
    _clear_sharding_prop_cache()


def _get_python_sharding_prop_cache_info() -> dict[str, int]:
    from .._ops.utils import _PROPAGATION_RULES

    return {"size": len(_PROPAGATION_RULES)}


def _get_fast_path_sharding_prop_cache_stats() -> tuple[int, int]:
    return (0, 0)


def _clear_fast_path_sharding_prop_cache() -> None:
    return None


def _reinit_dispatch_logger() -> None:
    return None
