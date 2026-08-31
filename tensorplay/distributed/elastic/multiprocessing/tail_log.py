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

from .redirects import Std, to_map

logger = logging.getLogger(__name__)


def tail_logfile(
    file: str | None,
    dst: Callable[[str], None],
    done: Callable[[], bool],
    poll_interval: float = 0.05,
) -> None:
    """Stream appended lines of ``file`` into ``dst`` until ``done`` and EOF."""
    if file is None:
        return
    while not os.path.isfile(file):
        if done():
            return
        time.sleep(poll_interval)
    with open(file, errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                dst(line.rstrip("\n"))
                continue
            if done():
                break
            time.sleep(poll_interval)


class TailLog:
    """Fan out per-rank log files to console streams for a running context."""

    def __init__(
        self,
        contexts: dict[int, "SubprocessHandler | object"],
        log_dir: str,
        redirects: Std | dict[int, Std],
        files: dict[int, dict[str, str | None]],
    ) -> None:
        self._contexts = contexts
        self._log_dir = log_dir
        self._redirects = redirects
        self._files = files
        self._stdout_tail: threading.Thread | None = None
        self._stderr_tail: threading.Thread | None = None
        self._stopped = False
        self._threads: list[threading.Thread] = []

    def start(self) -> "TailLog":
        """Start one reader thread per (rank, stream) with redirection."""
        redirects_map = to_map(self._redirects, len(self._contexts))
        for local_rank, streams in self._files.items():
            mode = redirects_map[local_rank]
            for stream_name in ("stdout", "stderr"):
                flag = Std.OUT if stream_name == "stdout" else Std.ERR
                path = streams.get(stream_name)
                if not path or not (mode & flag):
                    continue
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

    def stopped(self) -> bool:
        return self._stopped
