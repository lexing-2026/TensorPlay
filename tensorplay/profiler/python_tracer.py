"""Optional Python stack sampling for profiler traces."""

from __future__ import annotations

import os
import sys
import threading
import time


def sample_interval_ms():
    """Read the sampling period from the process environment."""
    raw = os.environ.get("TP_PROFILER_SAMPLE_MS", "")
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        value = 0.0
    return value if value > 0 else 5.0


class PySampler:
    """Collect frame chains from all Python threads on a background thread."""

    def __init__(self, should_run):
        self.should_run = should_run
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _tick(self):
        frames_by_tid = sys._current_frames()
        timestamp_ns = time.perf_counter_ns()
        for os_tid, frame in frames_by_tid.items():
            chain = []
            depth = 0
            while frame is not None and depth < 64:
                code = frame.f_code
                chain.append(f"{code.co_filename}:{frame.f_lineno} ({code.co_name})")
                frame = frame.f_back
                depth += 1
            if chain:
                self.samples.append((timestamp_ns, os_tid, chain))

    def _run(self):
        interval = sample_interval_ms() / 1e3
        while not self._stop.wait(interval):
            try:
                if self.should_run():
                    self._tick()
            except Exception:
                continue

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="tp-profiler-sampler",
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


__all__ = ["PySampler", "sample_interval_ms"]
