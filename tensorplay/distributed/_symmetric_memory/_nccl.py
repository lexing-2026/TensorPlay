from __future__ import annotations

from typing import Any

__all__ = ["NcclCommRegistration", "register_external_nccl_comm"]

_registrations: dict[tuple[str, Any], tuple[int, Any]] = {}


class NcclCommRegistration:
    def __init__(self, group_name: str, device: Any, comm: Any | None = None) -> None:
        self._group_name = group_name
        self._device = device
        self._comm = comm
        self._active = True

    def unregister(self) -> None:
        if self._active:
            _registrations.pop((self._group_name, self._device), None)
            self._active = False
            self._comm = None

    def __enter__(self) -> "NcclCommRegistration":
        return self

    def __exit__(self, *exc: object) -> None:
        self.unregister()

    def __del__(self) -> None:
        try:
            self.unregister()
        except Exception:
            pass


def register_external_nccl_comm(group_name: str, comm_ptr: int, device: Any, comm: Any | None = None) -> NcclCommRegistration:
    if not isinstance(comm_ptr, int) or comm_ptr <= 0:
        raise ValueError("comm_ptr must be a positive opaque pointer")
    handle = NcclCommRegistration(group_name, device, comm)
    _registrations[(group_name, device)] = (comm_ptr, handle)
    return handle
