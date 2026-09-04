"""Console mirroring of worker log files.

When a worker's stdout/stderr is redirected to files, ``tee`` additionally
streams the file contents back to the agent console so operator-facing
output is preserved without interleaving hazards at the worker level.
"""
import logging
import os
import threading
import time
from collections.abc import Callable
from threading import Event

from .redirects import Std, to_map

logger = logging.getLogger(__name__)


def tail_logfile(
    file: str | None = None,
    dst=None,
    done=None,
    poll_interval: float = 0.05,
    *source_args,
    header: str = "",
    finished: Event | None = None,
    interval_sec: float | None = None,
    log_line_filter: Callable[[str], bool] | None = None,
) -> None:
    source_mode = finished is not None or hasattr(poll_interval, "is_set")
    if hasattr(poll_interval, "is_set"):
        header, file, dst, finished = file or "", dst, done, poll_interval
        interval_sec = source_args[0] if source_args else 0.1
        log_line_filter = source_args[1] if len(source_args) > 1 else None
    if source_mode:
        if file is None:
            return
        is_done = finished.is_set
        wait_interval = interval_sec if interval_sec is not None else 0.1
    else:
        if file is None:
            return
        is_done = done
        wait_interval = poll_interval
    while not os.path.isfile(file):
        if is_done():
            return
        time.sleep(wait_interval)
    with open(file, errors="replace") as stream:
        while True:
            line = stream.readline()
            if line:
                if source_mode:
                    if log_line_filter is None or log_line_filter(line):
                        dst.write(f"{header}{line}")
                        dst.flush()
                else:
                    dst(line.rstrip("\n"))
                continue
            if is_done():
                break
            time.sleep(wait_interval)


class TailLog:
    """Fan out per-rank log files to console streams for a running context."""

    def __init__(self, *args, **kwargs) -> None:
        self._stopped = False
        self._threads: list[threading.Thread] = []
        self._source_mode = bool(args and isinstance(args[0], str))
        if self._source_mode:
            self._name = args[0]
            self._log_files = args[1]
            self._dst = args[2]
            self._log_line_prefixes = (
                args[3] if len(args) > 3 else kwargs.get("log_line_prefixes")
            ) or {}
            self._interval_sec = (
                args[4] if len(args) > 4 else kwargs.get("interval_sec", 0.1)
            )
            self._log_line_filter = kwargs.get("log_line_filter")
            return
        self._contexts = args[0]
        self._log_dir = args[1]
        self._redirects = args[2]
        self._files = args[3]
        self._log_line_prefixes = (args[4] if len(args) > 4 else None) or {}

    def start(self) -> "TailLog":
        if self._source_mode:
            if not self._dst:
                return self
            for local_rank, file in self._log_files.items():
                header = f"[{self._name}{local_rank}]:"
                if local_rank in self._log_line_prefixes:
                    header = self._log_line_prefixes[local_rank]
                finished = Event()
                thread = threading.Thread(
                    target=tail_logfile,
                    kwargs={
                        "header": header,
                        "file": file,
                        "dst": self._dst,
                        "finished": finished,
                        "interval_sec": self._interval_sec,
                        "log_line_filter": self._log_line_filter,
                    },
                    daemon=True,
                    name=f"tp_elastic_tail_{self._name}{local_rank}",
                )
                self._threads.append(thread)
                thread._tp_finished = finished
                thread.start()
            return self
        """Start one reader thread per (rank, stream) with redirection."""
        redirects_map = to_map(self._redirects, len(self._contexts))
        for local_rank, streams in self._files.items():
            mode = redirects_map[local_rank]
            for stream_name in ("stdout", "stderr"):
                flag = Std.OUT if stream_name == "stdout" else Std.ERR
                path = streams.get(stream_name)
                if not path or not (mode & flag):
                    continue
                if local_rank in self._log_line_prefixes:
                    prefix = self._log_line_prefixes[local_rank]
                    if stream_name == "stdout":
                        dst = lambda line, prefix=prefix: print(f"{prefix}{line}")
                    else:
                        dst = lambda line, prefix=prefix: logger.warning(
                            "%s%s", prefix, line
                        )
                else:
                    dst = print if stream_name == "stdout" else (
                        lambda line: logger.warning(line)
                    )
                thread = threading.Thread(
                    target=tail_logfile,
                    args=(path, dst, lambda: self._stopped),
                    daemon=True,
                    name=f"tp_elastic_tail_{local_rank}_{stream_name}",
                )
                self._threads.append(thread)
                thread.start()
        return self

    def stop(self) -> None:
        """Stop all reader threads."""
        self._stopped = True
        for thread in self._threads:
            finished = getattr(thread, "_tp_finished", None)
            if finished is not None:
                finished.set()
        for thread in self._threads:
            thread.join(timeout=5)

    def stopped(self) -> bool:
        return self._stopped
