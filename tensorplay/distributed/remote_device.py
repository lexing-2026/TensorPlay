from __future__ import annotations

import tensorplay as tp

__all__ = ["_remote_device"]


class _remote_device:
    def __init__(self, remote_device: str | object):
        parse_error = (
            f"Could not parse remote_device: {remote_device!r}. "
            "Expected '<worker>/<device>', 'rank:<rank>/<device>', or '<device>'"
        )
        self._worker_name: str | None = None
        self._rank: int | None = None
        self._device: object

        if isinstance(remote_device, tp.device):
            self._device = remote_device
        elif isinstance(remote_device, str):
            fields = remote_device.split("/")
            if len(fields) == 2:
                self._worker_name, device_spec = fields
                self._device = self._parse_device(device_spec, parse_error)
            elif len(fields) == 1:
                if self._is_local_device(fields[0]):
                    self._device = self._parse_device(fields[0], parse_error)
                else:
                    self._worker_name = fields[0]
                    self._device = self._parse_device("cpu", parse_error)
            else:
                raise ValueError(parse_error)
        else:
            raise TypeError(f"Invalid type for remote_device: {type(remote_device)!r}")

        if self._worker_name is not None:
            if not self._worker_name:
                raise ValueError(parse_error)
            fields = self._worker_name.split(":")
            if len(fields) == 2 and fields[0] == "rank" and fields[1].isdigit():
                self._rank = int(fields[1])
                self._worker_name = None
            elif len(fields) != 1:
                raise ValueError(parse_error)

    @staticmethod
    def _is_local_device(value: str) -> bool:
        try:
            tp.device(value)
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_device(value: str, parse_error: str) -> object:
        try:
            return tp.device(value)
        except Exception as error:
            raise ValueError(parse_error) from error

    def worker_name(self) -> str | None:
        return self._worker_name

    def rank(self) -> int | None:
        return self._rank

    def device(self) -> object:
        return self._device

    def __repr__(self) -> str:
        if self._worker_name is not None:
            return f"{self._worker_name}/{self._device}"
        if self._rank is not None:
            return f"rank:{self._rank}/{self._device}"
        return str(self._device)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _remote_device) and (
            self._worker_name == other._worker_name
            and self._device == other._device
            and self._rank == other._rank
        )

    def __hash__(self) -> int:
        return hash((self._worker_name, self._device, self._rank))
