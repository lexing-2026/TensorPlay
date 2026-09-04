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
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from abc import ABC
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntFlag
from types import FrameType
from typing import Any, Union

from .errors import ProcessFailure, SignalException as _SignalException
from .redirects import Std, to_map as _to_map
from .subprocess_handler import SubprocessHandler
from .subprocess_handler.handlers import get_subprocess_handler
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
    "get_std_cm",
]


class SignalException(_SignalException):
    pass


def to_map(val_or_map: Std | dict[int, Std], local_world_size: int) -> dict[int, Std]:
    return _to_map(val_or_map, local_world_size)


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
        local_rank: int | None = None,
        log_dir: str = "",
        stdout: str | None = None,
        stderr: str | None = None,
        tee_mode: Std = Std.NONE,
        *source_args,
        stdouts: dict[int, str] | None = None,
        stderrs: dict[int, str] | None = None,
        tee_stdouts: dict[int, str] | None = None,
        tee_stderrs: dict[int, str] | None = None,
        error_files: dict[int, str] | None = None,
        filtered_stdout: str = "",
        filtered_stderr: str = "",
    ) -> None:
        if isinstance(local_rank, dict):
            source_filtered_stdout = source_args[0] if source_args else ""
            source_filtered_stderr = source_args[1] if len(source_args) > 1 else ""
            stdouts = local_rank
            stderrs = log_dir if isinstance(log_dir, dict) else {}
            tee_stdouts = stdout if isinstance(stdout, dict) else {}
            tee_stderrs = stderr if isinstance(stderr, dict) else {}
            error_files = tee_mode if isinstance(tee_mode, dict) else {}
            filtered_stdout = source_filtered_stdout
            filtered_stderr = source_filtered_stderr
            local_rank, log_dir, stdout, stderr, tee_mode = None, "", None, None, Std.NONE
        self.local_rank = local_rank
        self.log_dir = log_dir
        self.stdout = stdout
        self.stderr = stderr
        self.tee_mode = tee_mode
        self.stdouts = dict(stdouts or {})
        self.stderrs = dict(stderrs or {})
        self.tee_stdouts = dict(tee_stdouts or {})
        self.tee_stderrs = dict(tee_stderrs or {})
        self.error_files = dict(error_files or {})
        self.filtered_stdout = filtered_stdout
        self.filtered_stderr = filtered_stderr
        if local_rank is not None:
            if stdout is not None:
                self.stdouts.setdefault(local_rank, stdout)
            if stderr is not None:
                self.stderrs.setdefault(local_rank, stderr)


class LogsSpecs(abc.ABC):
    """Strategy producing log destinations for a worker group."""

    def __init__(
        self,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
        local_ranks_filter: set[int] | None = None,
    ) -> None:
        self._root_log_dir = log_dir
        self._redirects = redirects
        self._tee = tee
        self._local_ranks_filter = local_ranks_filter

    def reify(
        self,
        entrypoint: str | dict[int, dict[str, str]] | None = None,
        args: tuple = (),
        envs: dict[int, dict[str, str]] | None = None,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
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
        local_ranks_filter: set[int] | None = None,
    ) -> None:
        if log_dir != os.devnull:
            if log_dir is None:
                log_dir = tempfile.mkdtemp(prefix="tp_elastic_")
            elif os.path.exists(log_dir) and not os.path.isdir(log_dir):
                raise NotADirectoryError(f"log_dir: {log_dir} is a file")
            else:
                os.makedirs(log_dir, exist_ok=True)
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        self._log_dir = log_dir
        self._redirects = redirects
        self._tee = tee
        self._root_log_dir = log_dir
        self._run_log_dir: str | None = None

    @property
    def root_log_dir(self) -> str:
        if self._log_dir == os.devnull:
            return os.devnull
        if self._log_dir:
            return os.path.abspath(self._log_dir)
        return os.path.join(
            tempfile.gettempdir(),
            "tp_elastic_logs",
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{os.getpid()}",
        )

    def reify(
        self,
        entrypoint: str | dict[int, dict[str, str]] | None = None,
        args: tuple = (),
        envs: dict[int, dict[str, str]] | None = None,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
    ) -> dict[int, LogsDest] | LogsDest:
        source_style = envs is None and isinstance(entrypoint, dict)
        if source_style:
            envs = entrypoint
            log_dir = self._root_log_dir
            redirects = self._redirects
            tee = self._tee
        envs = envs or {}
        nprocs = len(envs)
        configured_redirects = self._redirects if source_style else redirects
        configured_tee = self._tee if source_style else tee
        attempt_log_dir = ""
        if self._root_log_dir != os.devnull:
            root = log_dir or self._root_log_dir or self.root_log_dir
            run_id = (envs.get(0) or {}).get("TORCHELASTIC_RUN_ID", "run")
            restart_count = (envs.get(0) or {}).get(
                "TORCHELASTIC_RESTART_COUNT", "0"
            )
            if self._run_log_dir is None:
                self._run_log_dir = self._make_log_dir(root, run_id)
            attempt_log_dir = os.path.join(
                self._run_log_dir, f"attempt_{restart_count}"
            )
            shutil.rmtree(attempt_log_dir, ignore_errors=True)
            os.makedirs(attempt_log_dir, exist_ok=True)
        else:
            attempt_log_dir = os.devnull

        redirects_map = to_map(configured_redirects or Std.NONE, nprocs)
        tee_map = to_map(configured_tee or Std.NONE, nprocs)
        for local_rank, tee_std in tee_map.items():
            redirects_map[local_rank] |= tee_std

        stdouts = {rank: "" for rank in range(nprocs)}
        stderrs = {rank: "" for rank in range(nprocs)}
        tee_stdouts: dict[int, str] = {}
        tee_stderrs: dict[int, str] = {}
        error_files: dict[int, str] = {}
        out: dict[int, LogsDest] = {}
        for local_rank in range(nprocs):
            if attempt_log_dir == os.devnull:
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = ""
                error_files[local_rank] = os.devnull
                out[local_rank] = LogsDest(
                    local_rank=local_rank, log_dir=os.devnull, tee_mode=Std.NONE
                )
                continue
            rank_dir = os.path.join(attempt_log_dir, str(local_rank))
            os.makedirs(rank_dir, exist_ok=True)
            stdout_path = os.path.join(rank_dir, "stdout.log")
            stderr_path = os.path.join(rank_dir, "stderr.log")
            redirect_std = redirects_map[local_rank]
            stdout = stdout_path if redirect_std & Std.OUT else ""
            stderr = stderr_path if redirect_std & Std.ERR else ""
            stdouts[local_rank] = stdout
            stderrs[local_rank] = stderr
            if tee_map[local_rank] & Std.OUT:
                tee_stdouts[local_rank] = stdout
            if tee_map[local_rank] & Std.ERR:
                tee_stderrs[local_rank] = stderr
            if self._local_ranks_filter and local_rank not in self._local_ranks_filter:
                if local_rank in tee_stdouts:
                    tee_stdouts.pop(local_rank)
                if local_rank in tee_stderrs:
                    tee_stderrs.pop(local_rank)
                if not stdout:
                    stdouts[local_rank] = os.devnull
                if not stderr:
                    stderrs[local_rank] = os.devnull
            error_file = os.path.join(rank_dir, "error.json")
            error_files[local_rank] = error_file
            envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file
            out[local_rank] = LogsDest(
                local_rank=local_rank,
                log_dir=rank_dir,
                stdout=stdout or None,
                stderr=stderr or None,
                tee_mode=tee_map[local_rank],
            )
        if not source_style:
            return out
        root = attempt_log_dir
        return LogsDest(
            stdouts=stdouts,
            stderrs=stderrs,
            tee_stdouts=tee_stdouts,
            tee_stderrs=tee_stderrs,
            error_files=error_files,
            filtered_stdout=os.path.join(root, "filtered_stdout.log"),
            filtered_stderr=os.path.join(root, "filtered_stderr.log"),
        )

    def _make_log_dir(self, log_dir: str | None, rdzv_run_id: str) -> str:
        base = log_dir or tempfile.mkdtemp(prefix="tp_elastic_")
        os.makedirs(base, exist_ok=True)
        return tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base)

    def __repr__(self) -> str:
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, "
            f"redirects={self._redirects}, tee={self._tee}, "
            f"local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DefaultLogsSpecs) and (
            self._root_log_dir,
            self._redirects,
            self._tee,
            self._local_ranks_filter,
        ) == (
            other._root_log_dir,
            other._redirects,
            other._tee,
            other._local_ranks_filter,
        )


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
        log_line_prefixes: dict[int, str] | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ) -> None:
        self._name = name
        self._entrypoint = entrypoint
        self._args = args
        self._envs = envs
        self._stdout_tail: TailLog | None = None
        self._logs_specs = logs_specs or DefaultLogsSpecs(log_dir=log_dir, redirects=redirects, tee=tee)
        self._redirects = (
            getattr(logs_specs, "_redirects", redirects)
            if logs_specs is not None
            else redirects
        )
        self._tee = (
            getattr(logs_specs, "_tee", tee) if logs_specs is not None else tee
        )
        self._log_line_prefixes = log_line_prefixes
        self._duplicate_stdout_filters = duplicate_stdout_filters
        self._duplicate_stderr_filters = duplicate_stderr_filters
        self._stdout: dict[int, str | None] = {}
        self._stderr: dict[int, str | None] = {}
        self._tee_stdout: dict[int, str] = {}
        self._tee_stderr: dict[int, str] = {}
        self.filtered_stdout = None
        self.filtered_stderr = None
        self._filtered_stdout_path = ""
        self._filtered_stderr_path = ""
        self._tail_logs: list[TailLog] = []
        self._error_files: dict[int, str] = {
            rank: env.get("TORCHELASTIC_ERROR_FILE", "")
            for rank, env in envs.items()
        }
        self.name = name
        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.nprocs = len(envs)
        self.stdouts = self._stdout
        self.stderrs = self._stderr
        self.error_files = self._error_files
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
        try:
            logs = self._logs_specs.reify(self._envs)
        except TypeError:
            logs = self._logs_specs.reify(
                self._entrypoint
                if isinstance(self._entrypoint, str)
                else getattr(self._entrypoint, "__name__", "function"),
                self._args,
                self._envs,
                None,
                self._redirects,
                self._tee,
            )
        if isinstance(logs, LogsDest):
            self._stdout = dict(logs.stdouts)
            self._stderr = dict(logs.stderrs)
            self._tee_stdout = dict(logs.tee_stdouts)
            self._tee_stderr = dict(logs.tee_stderrs)
            self._error_files = dict(logs.error_files)
            self._filtered_stdout_path = logs.filtered_stdout
            self._filtered_stderr_path = logs.filtered_stderr
            for local_rank, error_file in self._error_files.items():
                self._envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file
        else:
            for local_rank, dest in logs.items():
                self._stdout[local_rank] = dest.stdout
                self._stderr[local_rank] = dest.stderr
                if dest.tee_mode & Std.OUT and dest.stdout:
                    self._tee_stdout[local_rank] = dest.stdout
                if dest.tee_mode & Std.ERR and dest.stderr:
                    self._tee_stderr[local_rank] = dest.stderr
        self.stdouts = self._stdout
        self.stderrs = self._stderr
        self.error_files = self._error_files
        self._start()
        self._started = True
        self._stdout_tail = self._open_tails()
        for tail_log in self._tail_logs:
            tail_log.start()

    def _open_tails(self) -> TailLog:
        self._tail_logs = [
            TailLog(
                self._name,
                self._tee_stdout,
                sys.stdout,
                self._log_line_prefixes,
            ),
            TailLog(
                self._name,
                self._tee_stderr,
                sys.stderr,
                self._log_line_prefixes,
            ),
        ]
        if self._duplicate_stdout_filters:
            path = self._filtered_stdout_path or os.path.join(
                self._logs_specs.root_log_dir, "filtered_stdout.log"
            )
            self.filtered_stdout = open(path, "w", buffering=1, errors="replace")
            self._tail_logs.append(
                TailLog(
                    self._name,
                    self._tee_stdout,
                    self.filtered_stdout,
                    self._log_line_prefixes,
                    log_line_filter=lambda line: any(
                        needle in line for needle in self._duplicate_stdout_filters
                    ),
                )
            )
        if self._duplicate_stderr_filters:
            path = self._filtered_stderr_path or os.path.join(
                self._logs_specs.root_log_dir, "filtered_stderr.log"
            )
            self.filtered_stderr = open(path, "w", buffering=1, errors="replace")
            self._tail_logs.append(
                TailLog(
                    self._name,
                    self._tee_stderr,
                    self.filtered_stderr,
                    self._log_line_prefixes,
                    log_line_filter=lambda line: any(
                        needle in line for needle in self._duplicate_stderr_filters
                    ),
                )
            )
        return self._tail_logs[0]

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
            for tail_log in self._tail_logs:
                tail_log.stop()
            if self.filtered_stdout is not None:
                self.filtered_stdout.close()
            if self.filtered_stderr is not None:
                self.filtered_stderr.close()


def get_std_cm(std_rd: str, redirect_fn):
    if sys.platform in {"win32", "darwin"} or not std_rd:
        return nullcontext()
    return redirect_fn(std_rd)


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
        log_line_prefixes: dict[int, str] | None = None,
        numa_options: Any = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ) -> None:
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_dir,
            redirects,
            tee,
            log_line_prefixes,
            duplicate_stdout_filters,
            duplicate_stderr_filters,
        )
        self._start_method = start_method
        self._numa_options = numa_options
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
        failed_ranks = {
            local_rank: proc
            for local_rank, proc in self._pc.items()
            if proc.exitcode not in (None, 0)
        }
        if failed_ranks:
            failures = dict(self._harvested["failures"])
            for local_rank, proc in failed_ranks.items():
                failures.setdefault(
                    local_rank,
                    ProcessFailure(
                        local_rank=local_rank,
                        pid=proc.pid or -1,
                        exitcode=proc.exitcode or 1,
                        error_file=self._error_files.get(local_rank),
                    ),
                )
            self.close()
            return RunProcsResult(
                state="FAILED",
                failures=failures,
                stdouts=dict(self._stdout),
                stderrs=dict(self._stderr),
            )
        if not self._is_done():
            return None
        expected = len(self._pc)
        deadline = time.monotonic() + 1.0
        while len(self._harvested["return_values"]) < expected:
            try:
                local_rank, ok, value = self._ret_queue.get(timeout=0.02)
            except (py_queue.Empty, EOFError, OSError):
                if time.monotonic() >= deadline:
                    break
                continue
            if ok:
                self._harvested["return_values"][local_rank] = value
            else:
                self._harvested["failures"][local_rank] = ProcessFailure(
                    local_rank=local_rank,
                    pid=self._pc[local_rank].pid or -1,
                    exitcode=1,
                    error_file=self._error_files.get(local_rank),
                )
        result = RunProcsResult(
            return_values=dict(self._harvested["return_values"]),
            failures=dict(self._harvested["failures"]),
            stdouts=dict(self._stdout),
            stderrs=dict(self._stderr),
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
        log_line_prefixes: dict[int, str] | None = None,
        numa_options: Any = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ) -> None:
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_dir,
            redirects,
            tee,
            log_line_prefixes,
            duplicate_stdout_filters,
            duplicate_stderr_filters,
        )
        self._handlers: dict[int, SubprocessHandler] = {}
        self.subprocess_handlers = self._handlers
        self._running_local_ranks = set(range(len(envs)))
        self._failures: dict[int, ProcessFailure] = {}
        self._numa_options = numa_options

    def _start(self) -> None:
        nprocs = len(self._envs)
        if self._handlers:
            raise ValueError("The subprocess handlers are already initialized")
        for local_rank in range(nprocs):
            args = (
                self._args[local_rank]
                if isinstance(self._args, dict)
                else self._args
            )
            self._handlers[local_rank] = get_subprocess_handler(
                entrypoint=str(self._entrypoint),
                args=args,
                env=dict(self._envs[local_rank]),
                stdout=self._stdout.get(local_rank),
                stderr=self._stderr.get(local_rank),
                local_rank_id=local_rank,
                numa_options=self._numa_options,
            )

    def _poll(self) -> RunProcsResult | None:
        done_local_ranks: set[int] = set()
        self._capture_process_failures(done_local_ranks)
        self._running_local_ranks.difference_update(done_local_ranks)
        if self._running_local_ranks and not self._failures:
            return None
        self.close()
        self._capture_process_failures(done_local_ranks)
        result = RunProcsResult(
            failures=dict(self._failures),
            stdouts=dict(self._stdout),
            stderrs=dict(self._stderr),
        )
        if not result.failures:
            result.return_values = {rank: None for rank in self._envs}
        result.state = "FAILED" if result.failures else "SUCCEEDED"
        return result

    def _capture_process_failures(self, done_local_ranks: set[int]) -> None:
        for local_rank in self._running_local_ranks:
            handler = self._handlers[local_rank]
            exitcode = handler.poll()
            if exitcode is None:
                continue
            done_local_ranks.add(local_rank)
            if exitcode != 0:
                self._failures[local_rank] = ProcessFailure(
                    local_rank=local_rank,
                    pid=handler.proc.pid,
                    exitcode=exitcode,
                    error_file=self._envs[local_rank].get("TORCHELASTIC_ERROR_FILE"),
                )

    def pids(self) -> dict[int, int]:
        return {rank: h.proc.pid for rank, h in self._handlers.items()}

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        for handler in self._handlers.values():
            handler.close(death_sig=death_sig, timeout=timeout)


def start_processes(
    name: str,
    entrypoint: Callable | str,
    args: tuple | dict[int, tuple],
    envs: dict[int, dict[str, str]],
    log_dir: str | None = None,
    start_method: str = "spawn",
    logs_specs: LogsSpecs | None = None,
    redirects: Std | dict[int, Std] = Std.NONE,
    tee: Std | dict[int, Std] = Std.NONE,
    log_line_prefixes: dict[int, str] | None = None,
    numa_options: Any = None,
    duplicate_stdout_filters: list[str] | None = None,
    duplicate_stderr_filters: list[str] | None = None,
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
        if isinstance(args, dict):
            args_by_rank = {int(rank): tuple(values) for rank, values in args.items()}
        else:
            args_by_rank = {rank: tuple(args) for rank in envs}
        _validate_full_rank(args_by_rank, len(envs), "args")
        if len(args_by_rank) != len(envs):
            raise ValueError(
                f"Function entrypoint requires {len(envs)} argument tuples, got {len(args_by_rank)}"
            )
    else:
        context_cls = SubprocessContext
        if isinstance(args, dict):
            args_by_rank = {int(rank): tuple(values) for rank, values in args.items()}
        else:
            args_by_rank = {rank: tuple(args) for rank in envs}
        _validate_full_rank(args_by_rank, len(envs), "args")
    context = context_cls(
        name=name,
        entrypoint=entrypoint,
        args=args_by_rank,
        envs=envs,
        logs_specs=logs_specs,
        log_dir=log_dir,
        redirects=redirects,
        tee=tee,
        log_line_prefixes=log_line_prefixes,
        numa_options=numa_options,
        duplicate_stdout_filters=duplicate_stdout_filters,
        duplicate_stderr_filters=duplicate_stderr_filters,
        **({"start_method": start_method} if context_cls is MultiprocessContext else {}),
    )
    context.start()
    return context
