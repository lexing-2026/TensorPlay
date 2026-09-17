"""Monitoring primitives for lightweight production instrumentation.

Events flow to registered handlers; :class:`Stat` exports fixed-window
summaries (mean, count, extrema, sum) as events, and ``_WaitCounter`` times
scoped regions.  Handlers should avoid blocking work: they run inline on the
logging thread.
"""

from tensorplay._C._monitor import (
    Aggregation as Aggregation,
    COUNT as COUNT,
    Event as Event,
    MAX as MAX,
    MEAN as MEAN,
    MIN as MIN,
    Stat as Stat,
    StatResult as StatResult,
    SUM as SUM,
    VALUE as VALUE,
    log_event as log_event,
    register_event_handler as register_event_handler,
    unregister_event_handler as unregister_event_handler,
)

__all__ = [
    "STAT_EVENT",
    "Aggregation",
    "COUNT",
    "Event",
    "MAX",
    "MEAN",
    "MIN",
    "SUM",
    "VALUE",
    "Stat",
    "StatResult",
    "log_event",
    "register_event_handler",
    "unregister_event_handler",
]

STAT_EVENT = "tensorplay.monitor.Stat"


class TensorboardEventHandler:
    """Forwards known events to a TensorBoard writer as scalars.

    The handler accepts any writer exposing ``add_scalar(tag, value,
    walltime=...)``; only :data:`STAT_EVENT` events are forwarded, one
    scalar per data key.
    """

    def __init__(self, writer):
        self._writer = writer

    def __call__(self, event):
        if event.name == STAT_EVENT:
            for k, v in event.data.items():
                self._writer.add_scalar(k, v, walltime=event.timestamp)
