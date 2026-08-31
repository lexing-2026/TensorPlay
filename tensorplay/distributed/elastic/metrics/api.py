"""Metric emission for the elastic control loop.

Metrics are named values tagged with a group; handlers decide where they
land. ``null`` (default) discards, ``console`` logs, and out-of-tree
packages may register their own handler via :func:`configure`.
"""
import abc
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


__all__ = [
    "MetricsConfig",
    "MetricHandler",
    "ConsoleMetricHandler",
    "NullMetricHandler",
    "MetricStream",
    "configure",
    "getStream",
    "prof",
    "put_metric",
    "initialize_metrics",
]


@dataclass
class MetricsConfig:
    """Metric plumbing configuration; maps groups to handler names."""

    default_handler: str = "console"
    cfg: dict[str, str] = field(default_factory=dict)


class MetricHandler(abc.ABC):
    """Sink for ``(metric_name, value, interval_ms)`` triples."""

    @abc.abstractmethod
    def emit(self, metric_name: str, metric_value: float, interval_ms: float = 1000) -> None:
        ...


class ConsoleMetricHandler(MetricHandler):
    """Write metrics to the ``tp_elastic_metrics`` logger."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("tp_elastic_metrics")

    def emit(self, metric_name: str, metric_value: float, interval_ms: float = 1000) -> None:
        self.logger.info(
            '{"metric_name": "%s", "value": %s, "interval_ms": %s}',
            metric_name,
            metric_value,
            interval_ms,
        )


class NullMetricHandler(MetricHandler):
    """Discard metrics."""

    def emit(self, metric_name: str, metric_value: float, interval_ms: float = 1000) -> None:
        return


class MetricStream:
    """Group-scoped metric emitter with simple in-memory aggregation."""

    def __init__(self, name: str, handler: MetricHandler) -> None:
        self.name = name
        self.handler = handler
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._window_start = time.time()

    def add_value(self, metric_name: str, value: float, interval_ms: float = 1000) -> None:
        self._sums[metric_name] += value
        self._counts[metric_name] += 1
        self.handler.emit(f"{self.name}.{metric_name}", value, interval_ms)

    def flush(self) -> dict[str, float]:
        """Return per-metric means since the last flush and reset the window."""
        out: dict[str, float] = {}
        for metric, total in list(self._sums.items()):
            count = max(1, self._counts[metric])
            out[metric] = total / count
        self._sums.clear()
        self._counts.clear()
        self._window_start = time.time()
        return out


_handler_by_group: dict[str, MetricHandler] = defaultdict(NullMetricHandler)
_default_handler: MetricHandler = NullMetricHandler()


def configure(handler: MetricHandler, group: str | None = None) -> None:
    """Route ``group`` (or all groups) to ``handler``."""
    global _default_handler
    if group is None:
        _default_handler = handler
    else:
        _handler_by_group[group] = handler


def initialize_metrics(cfg: MetricsConfig | None = None) -> None:
    """Install handlers named in ``cfg``; ``console`` and ``null`` are built in."""
    global _default_handler
    cfg = cfg or MetricsConfig()
    handlers: dict[str, MetricHandler] = {
        "console": ConsoleMetricHandler(),
        "null": NullMetricHandler(),
    }
    name = cfg.default_handler
    _default_handler = handlers.get(name, NullMetricHandler())
    if name not in handlers:
        warnings.warn(f"Unknown metric handler '{name}'; metrics disabled", stacklevel=2)
    for group, group_name in cfg.cfg.items():
        _handler_by_group[group] = handlers.get(group_name, NullMetricHandler())


def _get_stream(group: str) -> MetricStream:
    handler = _handler_by_group.get(group, _default_handler)
    return MetricStream(group, handler)


# Alias mirroring the historical camelCase accessor.
getStream = _get_stream


def _get_metric_name(fn) -> str:
    return f"tp_elastic.{fn.__module__}.{fn.__qualname__}.duration_ms"


def put_metric(
    name: str,
    value: float,
    interval_ms: float = 1000,
    group: str = "tp_elastic",
) -> None:
    """Emit one metric value on ``group``."""
    _get_stream(group).add_value(name, value, interval_ms)


def prof(
    fn=None,
    group: str = "tp_elastic",
    interval_ms: float = 1000,
):
    """Decorator measuring wall time of ``fn`` and emitting it as a metric.

    Works both bare (``@prof``) and parameterized (``@prof(group=...)``).
    """
    import functools

    def _decorate(func):
        metric_name = _get_metric_name(func)

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                put_metric(metric_name, (time.time() - start) * 1000, interval_ms, group)

        return _wrapper

    if fn is None:
        return _decorate
    return _decorate(fn)
