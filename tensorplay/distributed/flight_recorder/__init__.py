"""Collective flight recording.

The recorder accumulates timestamped collective events per rank and dumps
them for offline analysis with :mod:`tensorplay.distributed.flight_recorder.fr_trace`.
Recording is wired through the process-group hooks this build exposes;
collectives recorded here carry ``id`` (monotonic sequence per rank),
``ts`` (timestamp), ``collective`` (op name), ``state``, and optional
``stack``/``pg`` fields.
"""
import json
import os
import time
from collections.abc import Callable

from . import fr_trace

__all__ = ["FlightRecorder", "start_flight_recorder", "dump_flight_recorder"]

_DUMP_ENV = "TP_FLIGHT_RECORDER_DUMP"


class _Record:
    __slots__ = ("id", "ts", "collective", "state", "pg", "stack")

    def __init__(self, id: int, collective: str, state: str, pg: str = "", stack: str = "") -> None:
        self.id = id
        self.ts = time.time()
        self.collective = collective
        self.state = state
        self.pg = pg
        self.stack = stack


class FlightRecorder:
    """Per-process recorder of collective lifecycle events."""

    def __init__(self, capture_stack: bool = False) -> None:
        self._records: list[_Record] = []
        self._seq = 0
        self._capture_stack = capture_stack
        self._lock = __import__("threading").Lock()

    def record(self, collective: str, state: str, pg: str = "") -> None:
        """Append one event (``state`` is ``started`` or ``finished``)."""
        with self._lock:
            if state == "started":
                self._seq += 1
                seq = self._seq
            else:
                seq = self._seq
            stack = ""
            if self._capture_stack:
                import traceback

                stack = "".join(traceback.format_stack(limit=8))
            self._records.append(_Record(seq, collective, state, pg, stack))

    def dump(self) -> list[dict]:
        """Return all records as JSON-serializable dicts."""
        with self._lock:
            return [
                {
                    "id": r.id,
                    "ts": r.ts,
                    "collective": r.collective,
                    "state": r.state,
                    "pg": r.pg,
                    "stack": r.stack,
                }
                for r in self._records
            ]


_recorder: FlightRecorder | None = None


def start_flight_recorder(capture_stack: bool = False) -> FlightRecorder:
    """Enable recording for this process (idempotent)."""
    global _recorder
    if _recorder is None:
        _recorder = FlightRecorder(capture_stack=capture_stack)
    return _recorder


def _get_recorder() -> FlightRecorder | None:
    return _recorder


def dump_flight_recorder(path: str | None = None) -> str | None:
    """Write this rank's records to ``path`` (or the configured dump path)."""
    if _recorder is None:
        return None
    path = path or os.environ.get(_DUMP_ENV)
    if not path:
        return None
    rank = 0
    try:
        from tensorplay.distributed import get_rank

        rank = get_rank()
    except Exception:
        pass
    target = path.format(rank=rank)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w") as f:
        for record in _recorder.dump():
            f.write(json.dumps(record) + "\n")
    return target
