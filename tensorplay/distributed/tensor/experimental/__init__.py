"""Experimental distributed tensor transformations."""

import contextlib
from collections.abc import Iterator

from ._attention import context_parallel, context_parallel_unshard, set_rotate_method
from ._func_map import local_map
from ._register_sharding import register_sharding

__all__ = ["context_parallel", "context_parallel_unshard", "implicit_replication", "local_map", "register_sharding", "set_rotate_method"]


@contextlib.contextmanager
def implicit_replication() -> Iterator[None]:
    yield
