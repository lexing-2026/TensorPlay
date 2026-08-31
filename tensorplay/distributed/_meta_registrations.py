from __future__ import annotations

import random
from typing import Any

__all__ = ["generate_unique_id", "create_fakework", "_META_FUNCTIONS"]

_used_ids: set[int] = set()


def generate_unique_id() -> int:
    while True:
        value = random.randint(1, 1_000_000_000)
        if value not in _used_ids:
            _used_ids.add(value)
            return value


class _FakeWork:
    def __init__(self) -> None:
        self.seq_id = generate_unique_id()

    def wait(self, timeout: Any = None) -> bool:
        del timeout
        return True

    def is_completed(self) -> bool:
        return True


def create_fakework(args: Any, return_first_arg: bool = True) -> Any:
    work = _FakeWork()
    return (args[0], work) if return_first_arg else work


_META_FUNCTIONS = {
    "broadcast_": lambda *args: create_fakework(args),
    "allreduce_": lambda *args: create_fakework(args),
    "allgather_": lambda *args: create_fakework(args),
    "_allgather_base_": lambda *args: create_fakework(args),
    "reduce_scatter_": lambda *args: create_fakework(args),
    "_reduce_scatter_base_": lambda *args: create_fakework(args),
    "reduce_": lambda *args: create_fakework(args, return_first_arg=False),
    "gather_": lambda *args: create_fakework(args, return_first_arg=False),
    "scatter_": lambda *args: create_fakework(args),
    "alltoall_": lambda *args: create_fakework(args),
    "alltoall_base_": lambda *args: create_fakework(args, return_first_arg=False),
    "barrier": lambda *args: create_fakework(args, return_first_arg=False),
    "monitored_barrier_": lambda *args: None,
    "send": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_any_source_": lambda *args: create_fakework(args, return_first_arg=False),
}
