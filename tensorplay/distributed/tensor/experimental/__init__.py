"""Experimental distributed tensor transformations."""

import contextlib
from collections.abc import Iterator

from .._api import DTensor
from ._context_parallel import context_parallel, context_parallel_unshard, set_rotate_method
from ._func_map import local_map
from ._register_sharding import register_sharding

__all__ = ["context_parallel", "context_parallel_unshard", "implicit_replication", "local_map", "register_sharding", "set_rotate_method"]


@contextlib.contextmanager
def implicit_replication() -> Iterator[None]:
    dispatcher = DTensor._op_dispatcher
    previous = dispatcher._allow_implicit_replication
    dispatcher._allow_implicit_replication = True
    try:
        yield
    finally:
        dispatcher._allow_implicit_replication = previous
