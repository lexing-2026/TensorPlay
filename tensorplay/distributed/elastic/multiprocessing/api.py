"""Worker process lifecycle management for the elastic agent.

``start_processes`` launches a homogeneous group of workers either as OS
subprocesses (command entrypoints) or through ``multiprocessing`` (function
entrypoints), and returns a :class:`PContext` that monitors, signals, and
reaps them. Std streams can be redirected into per-rank files (optionally
copied to the console) and per-rank environments carry the elastic
contract (rank ids, rendezvous endpoint, error-file path).
"""
import abc
import json
import logging
import multiprocessing as py_mp
import os
import queue as py_queue
import signal
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntFlag
from types import FrameType
from typing import Any, Union

from .errors import ProcessFailure, SignalException
from .redirects import Std, to_map
from .subprocess_handler import SubprocessHandler
from .tail_log import TailLog

logger = logging.getLogger(__name__)

__all__ = [
    "SignalException",
    "RunProcsResult",
    "start_processes",
    "PContext",
    "MultiprocessContext",
    "SubprocessContext",
    "LogsDest",
    "LogsSpecs",
    "DefaultLogsSpecs",
    "Std",
]


def _terminate_process_handler(signum: int, frame: FrameType | None) -> None:
    """Signal handler raising :class:`SignalException` so agents can unwind."""
    sigval = signal.Signals(signum)
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)


def _get_kill_signal() -> signal.Signals:
    return signal.SIGKILL


def _get_default_signal() -> signal.Signals:
    return signal.SIGTERM


def _validate_full_rank(d: dict[int, Any], nprocs: int, what: str) -> None:
    if set(d.keys()) != set(range(nprocs)):
        raise ValueError(
            f"{what} must be a full-rank map 0..{nprocs - 1}, got {sorted(d.keys())}"
        )


class LogsDest:
    """Resolved destinations for one worker's standard streams."""

    def __init__(
        self,
        local_rank: int,
        log_dir: str,
        stdout: str | None = None,
        stderr: str | None = None,
        tee_mode: Std = Std.NONE,
    ) -> None:
        self.local_rank = local_rank
        self.log_dir = log_dir
        self.stdout = stdout
        self.stderr = stderr
        self.tee_mode = tee_mode


class LogsSpecs(abc.ABC):
    """Strategy producing log destinations for a worker group."""

    @abc.abstractmethod
    def reify(
        self,
        entrypoint: str,
        args: tuple,
        envs: dict[int, dict[str, str]],
        log_dir: str | None,
        redirects: Std | dict[int, Std],
        tee: Std | dict[int, Std],
    ) -> dict[int, LogsDest]:
        ...

    @property
    @abc.abstractmethod
    def root_log_dir(self) -> str:
        ...


class DefaultLogsSpecs(LogsSpecs):
    """File-backed logs with per-rank redirection and optional tee."""

    def __init__(
        self,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
    ) -> None:
        self._log_dir = log_dir
        self._redirects = redirects
        self._tee = tee

    @property
    def root_log_dir(self) -> str:
        if self._log_dir:
            return os.path.abspath(self._log_dir)
        return os.path.join(
            tempfile.gettempdir(),
            "tp_elastic_logs",
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{os.getpid()}",
        )

    def reify(
        self,
        entrypoint: str,
        args: tuple,
        envs: dict[int, dict[str, str]],
        log_dir: str | None,
        redirects: Std | dict[int, Std],
        tee: Std | dict[int, Std],
    ) -> dict[int, LogsDest]:
        local_world_size = len(envs)
        nprocs = local_world_size
        root = log_dir or self.root_log_dir
        run_dir = os.path.join(root, "run")
        os.makedirs(run_dir, exist_ok=True)
        redirects_map = to_map(redirects or Std.NONE, nprocs)
        tee_map = to_map(tee or Std.NONE, nprocs)
        out: dict[int, LogsDest] = {}
        for local_rank in range(nprocs):
            rank_dir = os.path.join(run_dir, f"local_rank_{local_rank}")
            os.makedirs(rank_dir, exist_ok=True)
            stdout_path = os.path.join(rank_dir, "stdout.log")
            stderr_path = os.path.join(rank_dir, "stderr.log")
            mode = redirects_map[local_rank] | tee_map[local_rank]
            out[local_rank] = LogsDest(
                local_rank=local_rank,
                log_dir=rank_dir,
                stdout=stdout_path if mode & Std.OUT else None,
                stderr=stderr_path if mode & Std.ERR else None,
                tee_mode=tee_map[local_rank],
            )
        return out


@dataclass
class RunProcsResult:
    """Outcome of monitoring a worker group to completion."""

    state: str = "UNKNOWN"
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        """Whether any worker failed."""
        return bool(self.failures)


class PContext(abc.ABC):
    """Base class owning a homogeneous group of worker processes."""

    def __init__(
        self,
        name: str,
        entrypoint: Callable | str,
        args: tuple,
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs | None = None,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
    ) -> None:
        self._name = name
        self._entrypoint = entrypoint
        self._args = args
        self._envs = envs
        self._stdout_tail: TailLog | None = None
        self._logs_specs = logs_specs or DefaultLogsSpecs(log_dir=log_dir, redirects=redirects, tee=tee)
        self._redirects = redirects if redirects is not None else Std.NONE
        self._tee = tee if tee is not None else Std.NONE
        self._stdout: dict[int, str | None] = {}
        self._stderr: dict[int, str | None] = {}
        self._started = False
        try:
            signal.signal(signal.SIGTERM, _terminate_process_handler)
            signal.signal(signal.SIGINT, _terminate_process_handler)
            if sys.platform != "win32":
                signal.signal(signal.SIGHUP, _terminate_process_handler)
                signal.signal(signal.SIGQUIT, _terminate_process_handler)
        except ValueError:
            # Signal handlers can only be installed from the main thread.
            pass

    @property
    def is_started(self) -> bool:
        return self._started

    def start(self) -> None:
        """Launch all workers."""
        if self._started:
            raise RuntimeError("The process context is already started")
        logs = self._logs_specs.reify(
            self._entrypoint if isinstance(self._entrypoint, str) else getattr(self._entrypoint, "__name__", "function"),
            self._args,
            self._envs,
            None,
            self._redirects,
            self._tee,
        )
        for local_rank, dest in logs.items():
            self._stdout[local_rank] = dest.stdout
            self._stderr[local_rank] = dest.stderr
        self._start()
        self._started = True
        self._stdout_tail = self._open_tails().start()

    def _open_tails(self) -> TailLog:
        files = {
            rank: {"stdout": self._stdout.get(rank), "stderr": self._stderr.get(rank)}
            for rank in self._envs
        }
        return TailLog(
            self._envs,
            self._logs_specs.root_log_dir,
            self._tee,
            files,
        )

    @abc.abstractmethod
    def _start(self) -> None:
        ...

    @abc.abstractmethod
    def _poll(self) -> RunProcsResult | None:
        ...

    @abc.abstractmethod
    def pids(self) -> dict[int, int]:
        ...

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        ...

    def wait(self, timeout: float = -1, period: float = 1) -> RunProcsResult | None:
        """Block until completion (or ``timeout`` seconds); returns the result."""
        if timeout == -1:
            timeout = sys.maxsize
        end = time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if time.monotonic() >= end:
                return None
            time.sleep(period)

    def poll(self) -> RunProcsResult | None:
        """Return the terminal result, or None while workers are running."""
        if not self._started:
            raise RuntimeError("The process context is not started")
        return self._poll()

    def close(self, death_sig: signal.Signals | None = None, timeout: int = 30) -> None:
        """Terminate all workers with ``death_sig``, escalating to kill."""
        if not death_sig:
            death_sig = _get_default_signal()
        if self._started:
            self._close(death_sig, timeout=timeout)
            if self._stdout_tail is not None:
                self._stdout_tail.stop()


def _wrap(
    local_rank: int,
    fn: Callable,
    args: tuple,
    env: dict[str, str],
    stdout: str | None,
    stderr: str | None,
    ret_queue,
    error_file: str,
) -> None:
    """Child-process body for function entrypoints."""
    os.environ.update(env)
    from .errors import record

    if stdout:
        sys.stdout = open(stdout, "w", buffering=1)
    if stderr:
        sys.stderr = open(stderr, "w", buffering=1)

    @record
    def _run() -> Any:
        return fn(*args)

    try:
        ret = _run()
        ret_queue.put((local_rank, True, ret))
    except BaseException as exc:
        ret_queue.put((local_rank, False, repr(exc)))
        raise


class MultiprocessContext(PContext):
    """Function-entrypoint workers running as ``multiprocessing`` children.

    ``args`` must contain one argument tuple per local rank; the callables
    and arguments must be picklable by the chosen start method.
    """

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: tuple,
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs | None = None,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
        start_method: str = "spawn",
    ) -> None:
        super().__init__(name, entrypoint, args, envs, logs_specs, log_dir, redirects, tee)
        self._start_method = start_method
        self._pc: dict[int, py_mp.process.BaseProcess] = {}
        self._ret_queue = py_mp.get_context(start_method).Queue()
        self._error_files: dict[int, str] = {}

    def _start(self) -> None:
        nprocs = len(self._envs)
        mp = py_mp.get_context(self._start_method)
        for local_rank in range(nprocs):
            if not isinstance(self._args[local_rank], tuple):
                raise ValueError(
                    f"Function entrypoint requires per-rank argument tuples; "
                    f"rank {local_rank} got {type(self._args[local_rank])}"
                )
            env = dict(self._envs[local_rank])
            error_file = env.get("TORCHELASTIC_ERROR_FILE", "")
            self._error_files[local_rank] = error_file
            proc = mp.Process(
                target=_wrap,
                args=(
                    local_rank,
                    self._entrypoint,
                    self._args[local_rank],
                    env,
                    self._stdout.get(local_rank),
                    self._stderr.get(local_rank),
                    self._ret_queue,
                    error_file,
                ),
                daemon=True,
            )
            proc.start()
            self._pc[local_rank] = proc

    def _is_done(self) -> bool:
        return all(proc.exitcode is not None for proc in self._pc.values())

    def _poll(self) -> RunProcsResult | None:
        # Harvest results across polls: queue data must not be dropped when
        # some workers are still running.
        if not hasattr(self, "_harvested"):
            self._harvested = {"return_values": {}, "failures": {}}
        while True:
            try:
                local_rank, ok, value = self._ret_queue.get_nowait()
                if ok:
                    self._harvested["return_values"][local_rank] = value
                else:
                    self._harvested["failures"][local_rank] = ProcessFailure(
                        local_rank=local_rank,
                        pid=self._pc[local_rank].pid or -1,
                        exitcode=1,
                        error_file=self._error_files.get(local_rank),
                        message=value,
                    )
            except py_queue.Empty:
                break
        if not self._is_done():
            return None
        result = RunProcsResult(
            return_values=dict(self._harvested["return_values"]),
            failures=dict(self._harvested["failures"]),
        )
        for local_rank, proc in self._pc.items():
            exitcode = proc.exitcode
            if exitcode not in (0, None) and local_rank not in result.failures:
                result.failures[local_rank] = ProcessFailure(
                    local_rank=local_rank,
                    pid=proc.pid or -1,
                    exitcode=exitcode,
                    error_file=self._error_files.get(local_rank),
                )
        result.state = "FAILED" if result.failures else "SUCCEEDED"
        return result

    def pids(self) -> dict[int, int]:
        return {rank: proc.pid or -1 for rank, proc in self._pc.items()}

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not death_sig or death_sig == signal.SIGKILL:
            for proc in self._pc.values():
                if proc.exitcode is None:
                    try:
                        proc.kill()
                    except (ProcessLookupError, ValueError):
                        pass
            return
        end = time.monotonic() + timeout
        for proc in self._pc.values():
            if proc.exitcode is None:
                try:
                    proc.terminate() if death_sig == signal.SIGTERM else proc.send_signal(death_sig)
                except (ProcessLookupError, ValueError):
                    pass
        while time.monotonic() < end:
            if self._is_done():
                return
            time.sleep(0.1)
        for proc in self._pc.values():
            if proc.exitcode is None:
                try:
                    proc.kill()
                except (ProcessLookupError, ValueError):
                    pass


class SubprocessContext(PContext):
    """Command-entrypoint workers running as OS subprocesses."""

    def __init__(
        self,
        name: str,
        entrypoint: str,
        args: tuple,
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs | None = None,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
    ) -> None:
        super().__init__(name, entrypoint, args, envs, logs_specs, log_dir, redirects, tee)
        self._handlers: dict[int, SubprocessHandler] = {}

    def _start(self) -> None:
        nprocs = len(self._envs)
        for local_rank in range(nprocs):
            env = dict(self._envs[local_rank])
            handler = SubprocessHandler(
                args=(str(self._entrypoint), *map(str, self._args)),
                env=env,
                stdout=self._stdout.get(local_rank),
                stderr=self._stderr.get(local_rank),
                local_rank_id=local_rank,
            )
            self._handlers[local_rank] = handler

    def _poll(self) -> RunProcsResult | None:
        result = RunProcsResult()
        done = 0
        for local_rank, handler in self._handlers.items():
            exitcode = handler.poll()
            if exitcode is None:
                continue
            done += 1
            if exitcode != 0 and local_rank not in result.failures:
                result.failures[local_rank] = ProcessFailure(
                    local_rank=local_rank,
                    pid=handler.proc.pid,
                    exitcode=exitcode,
                    error_file=self._envs[local_rank].get("TORCHELASTIC_ERROR_FILE"),
                )
        if done < len(self._handlers):
            return None
        result.state = "FAILED" if result.failures else "SUCCEEDED"
        return result

    def pids(self) -> dict[int, int]:
        return {rank: h.proc.pid for rank, h in self._handlers.items()}

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        for handler in self._handlers.values():
            handler.close(death_sig=death_sig)


def start_processes(
    name: str,
    entrypoint: Callable | str,
    args: tuple,
    envs: dict[int, dict[str, str]],
    log_dir: str | None = None,
    start_method: str = "spawn",
    logs_specs: LogsSpecs | None = None,
    redirects: Std | dict[int, Std] = Std.NONE,
    tee: Std | dict[int, Std] = Std.NONE,
) -> PContext:
    """Launch ``len(envs)`` workers and return the managing context.

    ``entrypoint`` is either a command string (subprocess workers; ``args``
    is the shared argument list) or a picklable callable (multiprocessing
    workers; ``args`` holds one argument tuple per rank).
    """
    envs = {int(rank): dict(env) for rank, env in envs.items()}
    _validate_full_rank(envs, len(envs), "envs")
    if callable(entrypoint):
        context_cls: type[PContext] = MultiprocessContext
        if len(args) != len(envs):
            raise ValueError(
                f"Function entrypoint requires {len(envs)} argument tuples, got {len(args)}"
            )
    else:
        context_cls = SubprocessContext
    context = context_cls(
        name=name,
        entrypoint=entrypoint,
        args=tuple(args),
        envs=envs,
        logs_specs=logs_specs,
        log_dir=log_dir,
        redirects=redirects,
        tee=tee,
        **({"start_method": start_method} if context_cls is MultiprocessContext else {}),
    )
    context.start()
    return context
