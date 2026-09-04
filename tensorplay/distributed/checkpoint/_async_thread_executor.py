from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from ._async_executor import _AsyncCheckpointExecutor


def save_wrapper(
    staging_future_or_state_dict: Future[Any] | dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike[str] | None = None,
    storage_writer: Any = None,
    planner: Any = None,
    process_group: Any = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Any:
    from .state_dict_saver import save

    staged = (
        staging_future_or_state_dict.result()
        if isinstance(staging_future_or_state_dict, Future)
        else staging_future_or_state_dict
    )
    return save(
        staged,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="AsyncCheckpointExecutor"
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Future[Any] | dict[str, Any],
        *,
        checkpoint_id: str | os.PathLike[str] | None = None,
        storage_writer: Any = None,
        planner: Any = None,
        process_group: Any = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future[Any]:
        future = self._executor.submit(
            save_wrapper,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        future.add_done_callback(lambda _: self._executor.shutdown(wait=False))
        return future

    def close(self) -> None:
        self._executor.shutdown(wait=True)
