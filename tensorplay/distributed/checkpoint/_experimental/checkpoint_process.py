from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
import multiprocessing as mp
import threading
import time
from typing import Any, Callable

from multiprocessing.connection import Connection

from .checkpoint_writer import CheckpointWriter
from .types import RankInfo, STATE_DICT


@dataclass
class CheckpointProcessConfig:
    process_start_method: str = "spawn"
    subprocess_init_timeout_secs: float = 30.0
    subprocess_shutdown_timeout_secs: float = 60.0


class RequestType(Enum):
    PING = auto()
    WRITE = auto()
    CLOSE = auto()


@dataclass
class WorkerRequest:
    request_type: RequestType
    payload: dict[str, Any]


@dataclass
class WorkerResponse:
    request_type: RequestType
    success: bool
    payload: Any = None
    error: str | None = None


def _call_writer_factory(
    factory: Callable[..., CheckpointWriter],
    rank_info: RankInfo,
    init_args: dict[str, Any],
) -> CheckpointWriter:
    arguments = dict(init_args)
    arguments.setdefault("rank_info", rank_info)
    try:
        writer = factory(**arguments)
    except TypeError as keyword_error:
        if "rank_info" not in init_args:
            try:
                writer = factory(rank_info, **init_args)
            except TypeError:
                raise keyword_error
        else:
            raise
    if not isinstance(writer, CheckpointWriter):
        if not callable(getattr(writer, "write", None)) or not callable(
            getattr(writer, "close", None)
        ):
            raise TypeError("checkpoint writer factory returned an invalid writer")
    return writer


def _subprocess_entry(
    rank_info: RankInfo,
    pipe: Connection,
    subprocess_init_fn: Callable[..., None],
    subprocess_init_args: tuple[Any, ...],
    checkpoint_writer_init_fn: Callable[..., CheckpointWriter],
    checkpoint_writer_init_args: dict[str, Any],
) -> None:
    writer: Any = None
    current_request = RequestType.PING
    try:
        subprocess_init_fn(*subprocess_init_args)
        writer = _call_writer_factory(
            checkpoint_writer_init_fn, rank_info, checkpoint_writer_init_args
        )
        while True:
            request = pipe.recv()
            if not isinstance(request, WorkerRequest):
                raise TypeError("checkpoint worker received an invalid request")
            current_request = request.request_type
            if current_request is RequestType.PING:
                pipe.send(WorkerResponse(current_request, True))
                continue
            if current_request is RequestType.WRITE:
                payload = request.payload
                result = writer.write(
                    payload["path"], payload["state_dict"], **payload.get("kwargs", {})
                )
                if isinstance(result, Future):
                    result.result()
                pipe.send(WorkerResponse(current_request, True))
                continue
            if current_request is RequestType.CLOSE:
                writer.close()
                pipe.send(WorkerResponse(current_request, True))
                return
            raise ValueError(f"unknown checkpoint worker request {current_request!r}")
    except BaseException as error:
        try:
            pipe.send(
                WorkerResponse(
                    current_request,
                    False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        try:
            pipe.close()
        except OSError:
            pass


class CheckpointProcess:
    def __init__(self, rank_info: RankInfo, config: CheckpointProcessConfig, subprocess_init_fn: Callable[..., None], subprocess_init_args: tuple[Any, ...], checkpoint_writer_init_fn: Callable[..., CheckpointWriter], checkpoint_writer_init_args: dict[str, Any]) -> None:
        if config.subprocess_init_timeout_secs <= 0:
            raise ValueError("subprocess_init_timeout_secs must be positive")
        if config.subprocess_shutdown_timeout_secs <= 0:
            raise ValueError("subprocess_shutdown_timeout_secs must be positive")
        self._rank_info = rank_info
        self._config = config
        self._subprocess_init_fn = subprocess_init_fn
        self._subprocess_init_args = tuple(subprocess_init_args)
        self._checkpoint_writer_init_fn = checkpoint_writer_init_fn
        self._checkpoint_writer_init_args = dict(checkpoint_writer_init_args)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._io_lock = threading.Lock()
        self._parent_end: Connection | None = None
        self._process: mp.Process | None = None
        self._closing = False
        self._closed = False
        self.process_creation_future = self._executor.submit(self._create_subprocess)

    def _create_subprocess(self) -> None:
        try:
            context = mp.get_context(self._config.process_start_method)
        except ValueError as error:
            raise ValueError(
                f"unknown checkpoint process start method {self._config.process_start_method!r}"
            ) from error
        parent_end, child_end = context.Pipe(duplex=True)
        process = context.Process(
            target=_subprocess_entry,
            args=(
                self._rank_info,
                child_end,
                self._subprocess_init_fn,
                self._subprocess_init_args,
                self._checkpoint_writer_init_fn,
                self._checkpoint_writer_init_args,
            ),
            daemon=True,
        )
        with self._io_lock:
            if self._closed or self._closing:
                parent_end.close()
                child_end.close()
                raise RuntimeError("checkpoint process is closing")
            self._parent_end = parent_end
            self._process = process
        process.start()
        child_end.close()
        self._send(RequestType.PING, {})
        self._recv(self._config.subprocess_init_timeout_secs)

    def _send(self, request_type: RequestType, payload: dict[str, Any]) -> None:
        with self._io_lock:
            if self._closed:
                raise RuntimeError("checkpoint process is closed")
            parent_end = self._parent_end
            if parent_end is None:
                raise RuntimeError("checkpoint process is not initialized")
            try:
                parent_end.send(WorkerRequest(request_type, payload))
            except (BrokenPipeError, EOFError, OSError) as error:
                raise RuntimeError("checkpoint worker terminated unexpectedly") from error

    def _recv(self, timeout: float | None = None) -> Any:
        parent_end = self._parent_end
        if parent_end is None:
            raise RuntimeError("checkpoint process is not initialized")
        if timeout is not None and timeout < 0:
            raise ValueError("checkpoint worker timeout must be non-negative")
        if timeout is not None and not parent_end.poll(timeout):
            raise TimeoutError("timed out waiting for checkpoint worker")
        try:
            response = parent_end.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as error:
            raise RuntimeError("checkpoint worker terminated unexpectedly") from error
        if not isinstance(response, WorkerResponse):
            raise RuntimeError("checkpoint worker returned an invalid response")
        if not response.success:
            raise RuntimeError(response.error or "checkpoint worker failed")
        return response.payload

    def write(
        self,
        path: str | STATE_DICT,
        state_dict: STATE_DICT | str | Future[STATE_DICT] | None = None,
        **kwargs: Any,
    ) -> Future[Any]:
        if isinstance(path, str):
            target_path = path
            target_state = state_dict
        else:
            target_state = path
            target_path = state_dict
        if not isinstance(target_path, str):
            raise TypeError("checkpoint path must be a string")
        if not isinstance(target_state, (dict, Future)):
            raise TypeError("checkpoint state_dict must be a dictionary or Future")
        self.process_creation_future.result()
        return self._executor.submit(
            self._write, target_path, target_state, **kwargs
        )

    def _write(
        self, path: str, state_dict: STATE_DICT | Future[STATE_DICT], **kwargs: Any
    ) -> Any:
        value = state_dict.result() if isinstance(state_dict, Future) else state_dict
        if not isinstance(value, dict):
            raise TypeError("checkpoint state_dict must resolve to a dictionary")
        self._send(
            RequestType.WRITE,
            {"path": path, "state_dict": value, "kwargs": dict(kwargs)},
        )
        return self._recv()

    def close(self) -> None:
        with self._io_lock:
            if self._closed:
                return
            self._closing = True
        init_error: BaseException | None = None
        try:
            self.process_creation_future.result()
        except BaseException as error:
            init_error = error
        self._executor.shutdown(wait=True, cancel_futures=False)
        process = self._process
        parent_end = self._parent_end
        if process is None:
            with self._io_lock:
                self._closed = True
            return
        if process.is_alive() and parent_end is not None:
            try:
                with self._io_lock:
                    parent_end.send(WorkerRequest(RequestType.CLOSE, {}))
                if parent_end.poll(self._config.subprocess_shutdown_timeout_secs):
                    self._recv()
            except (BrokenPipeError, EOFError, OSError, RuntimeError, TimeoutError):
                pass
        process.join(self._config.subprocess_shutdown_timeout_secs)
        if process.is_alive():
            process.terminate()
            process.join(self._config.subprocess_shutdown_timeout_secs)
        if process.is_alive():
            process.kill()
            process.join()
        if parent_end is not None:
            parent_end.close()
        with self._io_lock:
            self._closed = True
        if init_error is not None:
            return
