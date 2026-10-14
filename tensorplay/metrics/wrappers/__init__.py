"""Wrapper metrics for modifying the behaviour of base metrics."""

from tensorplay.metrics.wrappers.bootstrapping import BootStrapper
from tensorplay.metrics.wrappers.classwise import ClasswiseWrapper
from tensorplay.metrics.wrappers.minmax import MinMaxMetric
from tensorplay.metrics.wrappers.multioutput import MultioutputWrapper
from tensorplay.metrics.wrappers.multitask import MultitaskWrapper
from tensorplay.metrics.wrappers.running import Running, RunningMean, RunningSum
from tensorplay.metrics.wrappers.tracker import MetricTracker

__all__ = [
    "BootStrapper",
    "ClasswiseWrapper",
    "MetricTracker",
    "MinMaxMetric",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "Running",
    "RunningMean",
    "RunningSum",
]
