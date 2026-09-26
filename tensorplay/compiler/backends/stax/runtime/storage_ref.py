"""Naming a storage by what it is rather than by what holds it.

Two tensors are the same memory when they are windows onto one run of storage,
and a report that groups tensors by the memory they share has to say so by the
storage, not by the tensor: a view and its base are different tensors over one
storage, and a grouping keyed by tensor would count them twice.

A storage outlives the key that names it here, because the tensors that own it
are the ones being grouped.  That is what makes the key usable as a dictionary
key: it is built while those tensors are alive and dropped with the grouping.
Where a key has to outlive them, the lifetime is the caller's to arrange, and
this makes no claim about it.
"""

from __future__ import annotations

from typing import Any


def _handle(storage: Any) -> int:
    """The number that names one run of storage, and only that run.

    Read from the handle the runtime keeps rather than from the address of the
    data: the address is the same number for two storages that were freed and
    handed out again, so a key built from it would find a stale group rather
    than report a miss.
    """

    handle = getattr(storage, "_cdata", None)
    if handle is not None:
        return int(handle)
    return int(storage.data_ptr())


class StorageWeakRef:
    """A hashable name for one run of storage.

    Two of these are equal when they name the same run, which is what lets a
    dictionary of them group by shared memory.
    """

    __slots__ = ("cdata",)

    def __init__(self, storage: Any) -> None:
        self.cdata = _handle(storage)

    @classmethod
    def from_handle(cls, handle: int) -> "StorageWeakRef":
        """A key for a run already named by its handle.

        For a key that arrives from somewhere else -- a report read back, a
        measurement filed under the handle it was taken with -- where there is
        no storage to ask.
        """

        instance = cls.__new__(cls)
        instance.cdata = int(handle)
        return instance

    def __hash__(self) -> int:
        return self.cdata

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StorageWeakRef):
            return NotImplemented
        return self.cdata == other.cdata

    def __repr__(self) -> str:
        return f"StorageWeakRef(cdata={self.cdata})"


__all__ = ["StorageWeakRef"]
