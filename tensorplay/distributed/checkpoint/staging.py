from __future__ import annotations

import os
import pickle
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

import tensorplay as tp

from ._state_dict_stager import StateDictStager

__all__ = ["AsyncStager", "BlockingAsyncStager", "DefaultStager", "StagingOptions"]


@runtime_checkable
class AsyncStager(Protocol):
    _synchronize_after_execute: bool = True
    @property
    def should_synchronize_after_execute(self) -> bool: ...
    def stage(
        self, state_dict: dict[str, Any], **kwargs: Any
    ) -> Future[dict[str, Any]] | dict[str, Any]: ...
    def synchronize_staging(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class StagingOptions:
    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True


class DefaultStager(AsyncStager):
    def __init__(self, config: StagingOptions = StagingOptions()) -> None:
        self._config = config
        self._stager = StateDictStager(config.use_pinned_memory, config.use_shared_memory)
        self._staging_executor = None
        self._staging_stream = None
        if config.use_async_staging:
            self._staging_executor = ThreadPoolExecutor(max_workers=1)
            if tp.cuda.is_available():
                self._staging_stream = tp.cuda.Stream()
        if config.use_non_blocking_copy and not tp.cuda.is_available():
            raise AssertionError(
                "Non-blocking copy requires an available accelerator"
            )
        self._staging_future: Future[dict[str, Any]] | None = None

    @property
    def should_synchronize_after_execute(self) -> bool:
        return bool(getattr(self, "_synchronize_after_execute", True))

    def stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any] | Future[dict[str, Any]]:
        if self._staging_executor is None:
            return self._stage(state_dict, **kwargs)
        self._staging_future = self._staging_executor.submit(
            self._stage, state_dict, **kwargs
        )
        return self._staging_future

    def synchronize_staging(self) -> None:
        if self._staging_future is not None:
            self._staging_future.result()

    def _stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self._config.use_non_blocking_copy:
            if self._staging_stream is None and self._config.use_async_staging:
                raise AssertionError(
                    "non-blocking async staging requires an accelerator stream"
                )
            context = (
                self._staging_stream
                if self._staging_stream is not None
                else nullcontext()
            )
            with context:
                state_dict = self._stager.stage(
                    state_dict,
                    non_blocking=self._config.use_non_blocking_copy,
                    **kwargs,
                )
            if self._staging_stream is not None:
                self._staging_stream.synchronize()
            else:
                tp.cuda.synchronize()
            return state_dict
        return self._stager.stage(state_dict, non_blocking=False, **kwargs)

    def close(self) -> None:
        self.synchronize_staging()
        if self._staging_executor is not None:
            self._staging_executor.shutdown(wait=True)
        self._stager.close()


class BlockingAsyncStager(DefaultStager):
    _synchronize_after_execute = False

    def __init__(
        self,
        cache_staged_state_dict: bool = False,
        type_check: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        config = kwargs.pop("config", None)
        if config is None and args and isinstance(args[0], StagingOptions):
            config = args[0]
        self.cache_staged_state_dict = bool(cache_staged_state_dict)
        self.type_check = bool(type_check)
        self.state_dict_cache: dict[str, Any] | None = None
        if config is None:
            config = StagingOptions(
                use_pinned_memory=bool(cache_staged_state_dict),
                use_shared_memory=False,
                use_async_staging=False,
                use_non_blocking_copy=False,
            )
        else:
            config.use_async_staging = False
            config.use_non_blocking_copy = False
        super().__init__(config)

    def stage(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if not self.cache_staged_state_dict:
            return self._stager.stage(state_dict, **kwargs)
        if self.state_dict_cache is None:
            self.state_dict_cache = self._stager.stage(state_dict, **kwargs)
        else:
            staged = self._stager.stage(state_dict, **kwargs)
            self.state_dict_cache.clear()
            self.state_dict_cache.update(staged)
        return self.state_dict_cache

    def synchronize_staging(self) -> None:
        return None

    def close(self) -> None:
        return None


class _ReplicationStager(AsyncStager):
    _synchronize_after_execute = False

    def __init__(
        self,
        pg: Any,
        timeout: timedelta = timedelta(minutes=30),
        device: Any = "cpu",
        storage_dir: str | None = None,
    ) -> None:
        del timeout
        self._pg = pg
        self._device = device
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="tp_replica_")
        os.makedirs(self._storage_dir, exist_ok=True)

    def stage(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        from .. import distributed_core as dist
        from ._pg_transport import PGTransport

        if not dist.is_initialized():
            return state_dict
        world_size = dist.get_world_size(self._pg)
        rank = dist.get_rank(self._pg)
        if world_size < 2 or world_size % 2:
            raise ValueError("replication staging requires an even world size")
        partner = (rank + world_size // 2) % world_size
        transport = PGTransport(self._pg, self._device)
        if rank < partner:
            transport.send_checkpoint([partner], state_dict)
            received = transport.recv_checkpoint(partner)
        else:
            received = transport.recv_checkpoint(partner)
            transport.send_checkpoint([partner], state_dict)
        self._persist_state_dict(received, rank, partner)
        return received

    def _get_persisted_path(self, current_rank: int, partner_rank: int) -> str:
        return os.path.join(
            self._storage_dir,
            f"rank_{current_rank}_replica_partner_{partner_rank}.bin",
        )

    def _persist_state_dict(
        self, state_dict: dict[str, Any], current_rank: int, partner_rank: int
    ) -> None:
        final_path = self._get_persisted_path(current_rank, partner_rank)
        temporary = f"{final_path}.tmp"
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            with open(temporary, "wb") as stream:
                pickle.dump(state_dict, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        except BaseException as error:
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except OSError:
                pass
            raise RuntimeError(
                f"failed to persist replica from rank {partner_rank} to rank {current_rank}"
            ) from error

    def synchronize_staging(self) -> None:
        return None

    def close(self) -> None:
        return None
