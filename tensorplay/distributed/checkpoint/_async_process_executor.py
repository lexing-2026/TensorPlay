from __future__ import annotations

import gc
import logging
import os
import socket
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any
from uuid import uuid4

import multiprocessing as mp

import tensorplay.distributed as dist

from ._async_executor import _AsyncCheckpointExecutor
from .metadata import Metadata
from .utils import _DistWrapper

try:
    from tensorplay.distributed.elastic.utils.distributed import get_free_port
except ImportError:
    def get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            stream.bind(("", 0))
            return int(stream.getsockname()[1])


logger = logging.getLogger(__name__)


class _CheckpointSaveProcessControlOpts(Enum):
    INIT_COMPLETE = "init_complete"
    TERMINATE = "terminate"
    START = "init_complete"
    STOP = "terminate"
    SAVE = "save"


@dataclass(frozen=True, unsafe_hash=True)
class _CheckpointRequestIdentifier:
    checkpoint_id: str | os.PathLike[str] | None
    uuid: str

    def __init__(self, checkpoint_id: str | os.PathLike[str] | None) -> None:
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "uuid", str(uuid4()))


@dataclass
class _AsyncCheckpointRequest:
    staged_state_dict: Any
    checkpoint_request_id: _CheckpointRequestIdentifier
    storage_writer: Any = None
    planner: Any = None
    no_dist: bool = False
    use_collectives: bool = True


@dataclass(init=False)
class _ProcessGroupInitInfo:
    local_rank: int
    global_rank: int
    world_size: int
    tcp_store_master_addr: str
    tcp_store_master_port: int
    use_prefix_store: bool
    disable_automatic_gc: bool
    disable_manual_gc: bool

    def __init__(self, process_group: Any = None) -> None:
        initialized = dist.is_available() and dist.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(dist.get_rank(process_group)) if initialized else 0
        self.world_size = int(dist.get_world_size(process_group)) if initialized else 1
        self.use_prefix_store = os.environ.get("DCP_USE_PREFIX_STORE", "0") == "1"
        self.disable_automatic_gc = (
            os.environ.get("DCP_DISABLE_AUTOMATIC_GC", "0") == "1"
        )
        self.disable_manual_gc = (
            os.environ.get("DCP_DISABLE_MANUAL_GC", "0") == "1"
        )

        if initialized:
            wrapper = _DistWrapper(process_group, True, 0)

            def choose_master() -> tuple[str, int]:
                if self.use_prefix_store:
                    master_addr = os.environ.get("MASTER_ADDR")
                    master_port = os.environ.get("MASTER_PORT")
                    if master_addr is None:
                        raise AssertionError(
                            "checkpoint prefix store requires MASTER_ADDR"
                        )
                    if master_port is None:
                        raise AssertionError(
                            "checkpoint prefix store requires MASTER_PORT"
                        )
                    return master_addr, int(master_port)
                return os.environ.get("MASTER_ADDR") or socket.getfqdn(), int(
                    get_free_port()
                )

            self.tcp_store_master_addr, self.tcp_store_master_port = wrapper.broadcast(
                "get_master_addr_and_port", choose_master
            )
        else:
            self.tcp_store_master_addr = os.environ.get("MASTER_ADDR") or socket.getfqdn()
            raw_port = os.environ.get("MASTER_PORT") or str(get_free_port())
            try:
                self.tcp_store_master_port = int(raw_port)
            except ValueError as error:
                raise ValueError("MASTER_PORT must be an integer") from error

        if not 0 < self.tcp_store_master_port <= 65535:
            raise ValueError("checkpoint process port must be in the valid range")


def _execute_save(
    state_dict: dict[str, Any],
    *,
    checkpoint_request_id: _CheckpointRequestIdentifier,
    storage_writer: Any = None,
    planner: Any = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Metadata:
    from .state_dict_saver import save

    metadata = save(
        state_dict,
        checkpoint_id=checkpoint_request_id.checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )
    if not isinstance(metadata, Metadata):
        raise TypeError(f"checkpoint save returned {type(metadata).__name__}")
    return metadata


class _AsyncCheckpointProcess:
    def __init__(self, pg_init_info: _ProcessGroupInitInfo) -> None:
        self._context = mp.get_context("spawn")
        self._process_pipe, child_end = self._context.Pipe()
        self._save_process = self._context.Process(
            target=self._checkpointing_subprocess,
            args=(pg_init_info, child_end),
            daemon=True,
        )
        self._save_process.start()
        child_end.close()
        response = self._wait_for_response(timeout=1800)
        if response is not _CheckpointSaveProcessControlOpts.INIT_COMPLETE:
            raise RuntimeError(f"unexpected checkpoint process response {response!r}")

    def __del__(self) -> None:
        process = getattr(self, "_save_process", None)
        if process is not None and process.is_alive():
            try:
                self._send(_CheckpointSaveProcessControlOpts.TERMINATE)
                process.join(timeout=5)
            except BaseException:
                pass
            if process.is_alive():
                process.terminate()

    def _send(self, data: Any) -> None:
        try:
            self._process_pipe.send(data)
        except (BrokenPipeError, EOFError, OSError) as error:
            raise RuntimeError("checkpoint process terminated unexpectedly") from error

    def _wait_for_response(self, timeout: float | None = None) -> Any:
        if not self._save_process.is_alive():
            self._save_process.join()
            raise RuntimeError(
                "checkpoint process is dead: "
                f"exit code {self._save_process.exitcode}"
            )
        if timeout is not None and not self._process_pipe.poll(timeout):
            raise TimeoutError("timed out waiting for checkpoint process")
        try:
            response = self._process_pipe.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as error:
            raise RuntimeError("checkpoint process terminated unexpectedly") from error
        if isinstance(response, BaseException):
            raise response
        return response

    def save(
        self,
        staged_state_dict: dict[str, Any],
        *,
        checkpoint_id: str | os.PathLike[str] | None = None,
        storage_writer: Any = None,
        planner: Any = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        request = _AsyncCheckpointRequest(
            staged_state_dict=staged_state_dict,
            checkpoint_request_id=_CheckpointRequestIdentifier(checkpoint_id),
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        self._send(request)
        result = self._wait_for_response()
        if not isinstance(result, Metadata):
            raise TypeError(f"checkpoint process returned {type(result).__name__}")
        return result

    @staticmethod
    def _execute_save(
        state_dict: dict[str, Any],
        *,
        checkpoint_request_id: _CheckpointRequestIdentifier,
        storage_writer: Any = None,
        planner: Any = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        return _execute_save(
            state_dict,
            checkpoint_request_id=checkpoint_request_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )

    @staticmethod
    def _checkpointing_subprocess(
        pg_init_info: _ProcessGroupInitInfo, parent_conn: Connection
    ) -> None:
        process_group_initialized = False
        try:
            os.environ["MASTER_ADDR"] = pg_init_info.tcp_store_master_addr
            os.environ["MASTER_PORT"] = str(pg_init_info.tcp_store_master_port)
            os.environ["LOCAL_RANK"] = str(pg_init_info.local_rank)
            os.environ["RANK"] = str(pg_init_info.global_rank)
            os.environ["WORLD_SIZE"] = str(pg_init_info.world_size)

            if pg_init_info.use_prefix_store:
                store = dist.PrefixStore(
                    "AsyncCheckpointProcess/",
                    dist.TCPStore(
                        pg_init_info.tcp_store_master_addr,
                        pg_init_info.tcp_store_master_port,
                        world_size=pg_init_info.world_size,
                        is_master=pg_init_info.global_rank == 0,
                        wait_for_workers=False,
                    ),
                )
                dist.init_process_group(
                    backend="gloo",
                    store=store,
                    world_size=pg_init_info.world_size,
                    rank=pg_init_info.global_rank,
                )
            else:
                dist.init_process_group(backend="gloo")
            process_group_initialized = dist.is_initialized()
            if process_group_initialized:
                dist.barrier()
            parent_conn.send(_CheckpointSaveProcessControlOpts.INIT_COMPLETE)
            if pg_init_info.disable_automatic_gc:
                gc.disable()
        except BaseException as error:
            try:
                parent_conn.send(error)
            except (BrokenPipeError, EOFError, OSError):
                pass
            return

        first_request = True
        try:
            while True:
                request = parent_conn.recv()
                if (
                    isinstance(request, _CheckpointSaveProcessControlOpts)
                    and request is _CheckpointSaveProcessControlOpts.TERMINATE
                ):
                    return
                if not isinstance(request, _AsyncCheckpointRequest):
                    raise TypeError("invalid checkpoint process request")
                try:
                    result = _AsyncCheckpointProcess._execute_save(
                        request.staged_state_dict,
                        checkpoint_request_id=request.checkpoint_request_id,
                        storage_writer=request.storage_writer,
                        planner=request.planner,
                        no_dist=request.no_dist,
                        use_collectives=request.use_collectives,
                    )
                    parent_conn.send(result)
                    if (
                        pg_init_info.disable_automatic_gc
                        and not pg_init_info.disable_manual_gc
                    ):
                        del request
                        gc.collect()
                        if first_request:
                            gc.freeze()
                    first_request = False
                except BaseException as error:
                    try:
                        parent_conn.send(error)
                    except (BrokenPipeError, EOFError, OSError):
                        return
        except (EOFError, BrokenPipeError, OSError):
            return
        finally:
            if process_group_initialized and dist.is_initialized():
                dist.destroy_process_group()
            try:
                parent_conn.close()
            except OSError:
                pass


_CHECKPOINT_PROCESS: _AsyncCheckpointProcess | None = None


def create_checkpoint_daemon_process(
    pg_init_info: _ProcessGroupInitInfo | None = None,
) -> _AsyncCheckpointProcess:
    global _CHECKPOINT_PROCESS
    if _CHECKPOINT_PROCESS is None:
        _CHECKPOINT_PROCESS = _AsyncCheckpointProcess(
            pg_init_info or _ProcessGroupInitInfo()
        )
    return _CHECKPOINT_PROCESS


class _ProcessBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _execute_save_impl(
        *,
        pg_init_info: _ProcessGroupInitInfo | None,
        staging_future_or_state_dict: Future[Any] | Any,
        checkpoint_id: str | os.PathLike[str] | None = None,
        storage_writer: Any = None,
        planner: Any = None,
        process_group: Any = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        del process_group
        global _CHECKPOINT_PROCESS
        if _CHECKPOINT_PROCESS is None:
            if pg_init_info is None:
                raise RuntimeError("checkpoint process initialization is missing")
            _CHECKPOINT_PROCESS = _AsyncCheckpointProcess(pg_init_info)
        staged_state_dict = (
            staging_future_or_state_dict.result()
            if isinstance(staging_future_or_state_dict, Future)
            else staging_future_or_state_dict
        )
        return _CHECKPOINT_PROCESS.save(
            staged_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Future[Any] | Any,
        *,
        checkpoint_id: str | os.PathLike[str] | None = None,
        storage_writer: Any = None,
        planner: Any = None,
        process_group: Any = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future[Any]:
        global _CHECKPOINT_PROCESS
        pg_init_info = (
            _ProcessGroupInitInfo(process_group)
            if _CHECKPOINT_PROCESS is None
            else None
        )
        future = self._executor.submit(
            self._execute_save_impl,
            pg_init_info=pg_init_info,
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
