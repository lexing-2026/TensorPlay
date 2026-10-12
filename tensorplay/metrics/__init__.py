"""Metrics API: evaluation metrics with an accumulating state interface.

The package exposes the familiar evaluation-metric surface used by deep
learning frameworks for the metrics implemented so far. The core
:class:`~tensorplay.metrics.Metric` base class handles state accumulation,
device placement and distributed synchronization; ``update``/``compute`` are
implemented by each metric. ``forward`` both updates the global state and
returns the batch-local value.

Covered domains in this snapshot:

- aggregation (``SumMetric``, ``MeanMetric``, ``MaxMetric``, ``MinMetric``,
  ``CatMetric``)
- classification (accuracy, precision, recall, F-beta/F-1, statistics,
  confusion matrix) as both class-based and functional APIs
- regression (MSE, MAE, R2, Pearson/Spearman correlation, explained variance,
  symmetric MAPE) as both class-based and functional APIs
"""

from tensorplay.metrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric
from tensorplay.metrics.collections import MetricCollection
from tensorplay.metrics.metric import CompositionalMetric, Metric
from tensorplay.metrics import classification
from tensorplay.metrics import functional
from tensorplay.metrics import regression

__all__ = [
    "CatMetric",
    "CompositionalMetric",
    "MaxMetric",
    "MeanMetric",
    "Metric",
    "MetricCollection",
    "MinMetric",
    "SumMetric",
    "classification",
    "functional",
    "regression",
]