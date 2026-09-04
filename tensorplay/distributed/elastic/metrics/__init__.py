"""Elastic metric recording entry points."""
from .api import (
    ConsoleMetricHandler,
    MetricHandler,
    MetricStream,
    MetricData,
    MetricsConfig,
    NullMetricHandler,
    configure,
    getStream,
    initialize_metrics as _initialize_metrics,
    prof,
    profile,
    publish_metric,
    get_elapsed_time_ms,
    put_metric,
)

__all__ = [
    "MetricsConfig",
    "MetricHandler",
    "ConsoleMetricHandler",
    "NullMetricHandler",
    "MetricStream",
    "MetricData",
    "configure",
    "getStream",
    "prof",
    "profile",
    "publish_metric",
    "get_elapsed_time_ms",
    "put_metric",
    "initialize_metrics",
]


def initialize_metrics(cfg=None):
    return _initialize_metrics(cfg)
