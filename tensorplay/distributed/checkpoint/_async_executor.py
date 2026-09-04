from __future__ import annotations

import abc
import os
from concurrent.futures import Future
from typing import Any


class _AsyncCheckpointExecutor(abc.ABC):
    @abc.abstractmethod
    def execute_save(
        self,
        staging_future_or_state_dict: Any | Future[Any],
        *,
        checkpoint_id: str | os.PathLike[str] | None = None,
        storage_writer: Any = None,
        planner: Any = None,
        process_group: Any = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future[Any]: ...
