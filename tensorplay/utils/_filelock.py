"""Taking a lock on a file, and holding it for the length of a block.

Two compilations writing the same cache directory at the same time is the case
this exists for: each takes this lock, does its work under it, and lets it go.
The third-party lock does the locking; what is added here is that the time
spent waiting and the time spent holding are counted separately, because they
answer different questions -- one says the machine is busy, the other says the
critical section is too long.
"""

from __future__ import annotations

from types import TracebackType

from filelock import FileLock as _BaseFileLock

__all__ = ["FileLock"]


class FileLock(_BaseFileLock):
    """A file lock that counts how long it was waited for and how long it was held.

    Two counters are kept: one over acquiring the lock and one over the work
    done while holding it.  They are separate because a wait that grows and a
    critical section that grows are different problems, and one number for
    both cannot tell them apart.
    """

    def __enter__(self):
        from tensorplay._C._monitor import _WaitCounter

        self._region_counter = _WaitCounter("tensorplay.filelock.region").guard()
        with _WaitCounter("tensorplay.filelock.acquire").guard():
            result = super().__enter__()
        self._region_counter.__enter__()
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._region_counter.__exit__()
        from tensorplay._C._monitor import _WaitCounter

        with _WaitCounter("tensorplay.filelock.release").guard():
            super().__exit__(exc_type, exc_value, traceback)
        return None
