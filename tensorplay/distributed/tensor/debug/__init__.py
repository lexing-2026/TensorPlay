"""Diagnostics for distributed tensor layouts and collectives."""

from ._comm_mode import CommDebugMode
from ._visualize_sharding import visualize_sharding

__all__ = ["CommDebugMode", "visualize_sharding"]


def _clear_sharding_prop_cache() -> None:
    from .._api import DTensor

    DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()


def _clear_python_sharding_prop_cache() -> None:
    _clear_sharding_prop_cache()


def _get_python_sharding_prop_cache_info() -> dict[str, int]:
    from .._api import DTensor

    return DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()


def _get_fast_path_sharding_prop_cache_stats() -> tuple[int, int]:
    return (0, 0)


def _clear_fast_path_sharding_prop_cache() -> None:
    return None


def _reinit_dispatch_logger() -> None:
    bridge = getattr(__import__("tensorplay")._C, "_reinit_DTensor_dispatch_logger", None)
    if callable(bridge):
        bridge()


CommDebugMode.__module__ = "tensorplay.distributed.tensor.debug"
visualize_sharding.__module__ = "tensorplay.distributed.tensor.debug"
