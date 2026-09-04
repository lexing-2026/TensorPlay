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
from collections import namedtuple
from dataclasses import dataclass, field
from functools import wraps
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
    "profile",
    "publish_metric",
    "get_elapsed_time_ms",
    "MetricData",
]

MetricData = namedtuple("MetricData", ["timestamp", "group_name", "name", "value"])


@dataclass(init=False)
class MetricsConfig:
    """Metric plumbing configuration; maps groups to handler names."""

    default_handler: str = "console"
    cfg: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        params: dict[str, str] | None = None,
        default_handler: str = "console",
        cfg: dict[str, str] | None = None,
    ) -> None:
        self.default_handler = default_handler
        self.cfg = dict(cfg or params or {})
        self.params = dict(params or cfg or {})
        if params and "default_handler" in params:
            self.default_handler = str(params["default_handler"])


class MetricHandler(abc.ABC):
    """Sink for ``(metric_name, value, interval_ms)`` triples."""

    @abc.abstractmethod
    def emit(self, metric_data: MetricData) -> None:
        ...


class ConsoleMetricHandler(MetricHandler):
    """Write metrics to the ``tp_elastic_metrics`` logger."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("tp_elastic_metrics")

    def emit(self, metric_data: MetricData) -> None:
        self.logger.info(
            '{"metric_name": "%s", "value": %s, "interval_ms": %s}',
            metric_data.name,
            metric_data.value,
            1000,
        )


class NullMetricHandler(MetricHandler):
    """Discard metrics."""

    def emit(self, metric_data: MetricData) -> None:
        return


class MetricStream:
    """Group-scoped metric emitter with simple in-memory aggregation."""

    def __init__(self, name: str, handler: MetricHandler) -> None:
        self.name = name
        self.group_name = name
        self.handler = handler
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._window_start = time.time()

    def add_value(self, metric_name: str, value: float, interval_ms: float = 1000) -> None:
        self._sums[metric_name] += value
        self._counts[metric_name] += 1
        data = MetricData(time.time(), self.name, metric_name, value)
        try:
            self.handler.emit(data)
        except TypeError as first_error:
            try:
                self.handler.emit(metric_name, value, interval_ms)
            except TypeError:
                raise first_error

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
        if group == "default_handler":
            continue
        _handler_by_group[group] = handlers.get(group_name, NullMetricHandler())


def _get_stream(group: str) -> MetricStream:
    handler = _handler_by_group.get(group, _default_handler)
    return MetricStream(group, handler)


def getStream(group: str) -> MetricStream:
    return _get_stream(group)


def _get_metric_name(fn) -> str:
    qualname = fn.__qualname__
    if "." in qualname:
        return qualname
    module = fn.__module__
    return f"{module.rsplit('.', 1)[-1]}.{qualname}" if module else qualname


def put_metric(
    name: str,
    value: float,
    interval_ms: float | str = 1000,
    group: str = "torchelastic",
) -> None:
    """Emit one metric value on ``group``."""
    if isinstance(interval_ms, str):
        group = interval_ms
        interval_ms = 1000
    _get_stream(group).add_value(name, value, interval_ms)


def prof(
    fn=None,
    group: str = "tp_elastic",
    interval_ms: float = 1000,
):
    """Decorator measuring wall time of ``fn`` and emitting it as a metric.

    Works both bare (``@prof``) and parameterized (``@prof(group=...)``).
    """
    def _decorate(func):
        metric_name = _get_metric_name(func)

        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                put_metric(f"{metric_name}.success", 1, group)
                return result
            except Exception:
                put_metric(f"{metric_name}.failure", 1, group)
                raise
            finally:
                put_metric(
                    f"{metric_name}.duration.ms",
                    get_elapsed_time_ms(start),
                    group,
                )

        return _wrapper

    if fn is None:
        return _decorate
    return _decorate(fn)


def profile(group=None):
    """Return the compatibility profiling decorator for a metric group."""
    def _decorate(func):
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                publish_metric(group, f"{func.__name__}.success", 1)
                return result
            except Exception:
                publish_metric(group, f"{func.__name__}.failure", 1)
                raise
            finally:
                publish_metric(
                    group, f"{func.__name__}.duration.ms", get_elapsed_time_ms(start)
                )

        return _wrapper

    return _decorate


def publish_metric(metric_group: str | None, metric_name: str, metric_value: float) -> None:
    _get_stream(metric_group or "torchelastic").add_value(metric_name, metric_value)


def get_elapsed_time_ms(start_time_in_seconds: float) -> int:
    return int((time.time() - start_time_in_seconds) * 1000)
