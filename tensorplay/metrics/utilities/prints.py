"""Rank-aware logging and warning helpers.

In distributed settings only rank 0 emits logs; warnings are emitted from every
rank by default so that the user sees them regardless of which process they watch.
"""

import logging
import os
import warnings
from functools import partial, wraps
from typing import Any, Callable

_logger = logging.getLogger("tensorplay.metrics")

__all__ = ["rank_zero_debug", "rank_zero_info", "rank_zero_warn", "_future_warning"]


def rank_zero_only(fn: Callable) -> Callable:
    """Call a function only on rank 0 in distributed settings."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


rank_zero_only.rank = getattr(rank_zero_only, "rank", int(os.environ.get("LOCAL_RANK", 0)))


def _warn(*args: Any, **kwargs: Any) -> None:
    warnings.warn(*args, **kwargs)


def _info(*args: Any, **kwargs: Any) -> None:
    _logger.info(*args, **kwargs)


def _debug(*args: Any, **kwargs: Any) -> None:
    _logger.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)
_future_warning = partial(warnings.warn, category=FutureWarning)