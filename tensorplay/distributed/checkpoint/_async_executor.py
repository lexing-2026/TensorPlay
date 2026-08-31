from __future__ import annotations

import abc
from concurrent.futures import Future
from typing import Any


class _AsyncCheckpointExecutor(abc.ABC):
    @abc.abstractmethod
    def execute_save(self, *args: Any, **kwargs: Any) -> Future[Any]: ...
