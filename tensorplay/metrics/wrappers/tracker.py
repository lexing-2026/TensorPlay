"""Metric tracker wrapper."""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Optional, Union, cast

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import ModuleList

from tensorplay.metrics.collections import MetricCollection
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
from tensorplay.metrics.utilities.prints import rank_zero_warn


class MetricTracker(ModuleList):
    """A wrapper class that can help keeping track of a metric or metric collection over time.

    The wrapper implements the standard ``.update()``, ``.compute()``, ``.reset()`` methods that just
    call the corresponding method of the currently tracked metric. However, the following additional methods are
    provided:

    - ``MetricTracker.n_steps``: number of metrics being tracked
    - ``MetricTracker.increment()``: initialize a new metric for being tracked
    - ``MetricTracker.compute_all()``: get the metric value for all steps
    - ``MetricTracker.best_metric()``: returns the best value

    Out of the box, this wrapper class fully supports that the base metric being tracked is a single ``Metric``, a
    ``MetricCollection`` or another wrapper around a metric. However, multiple layers of nesting, such as using a
    ``Metric`` inside a wrapper inside a ``MetricCollection`` is not fully supported, especially the ``.best_metric``
    method that cannot auto compute the best metric and index for such nested structures.

    Args:
        metric: instance of a ``Metric`` or ``MetricCollection`` to keep track of at each timestep
        maximize: either single bool or list of bool indicating if higher metric values are
            better (``True``) or lower is better (``False``)

    """

    maximize: Union[bool, list[bool]]
    _base_metric: Union[Metric, MetricCollection]

    def __init__(self, metric: Union[Metric, MetricCollection], maximize: Union[bool, list[bool], None] = None) -> None:
        super().__init__()
        if not isinstance(metric, (Metric, MetricCollection)):
            raise TypeError(
                f"Metric arg need to be an instance of a `Metric` or `MetricCollection` but got {metric}"
            )
        self._base_metric = metric

        if maximize is None:
            if isinstance(metric, Metric):
                if getattr(metric, "higher_is_better", None) is None:
                    raise AttributeError(
                        f"The metric '{metric.__class__.__name__}' does not have a 'higher_is_better' attribute."
                        " Please provide the `maximize` argument explicitly."
                    )
                self.maximize = metric.higher_is_better  # type: ignore[assignment]
            elif isinstance(metric, MetricCollection):
                self.maximize = []
                for name, m in metric.items():
                    if getattr(m, "higher_is_better", None) is None:
                        raise AttributeError(
                            f"The metric '{name}' in the MetricCollection does not have a 'higher_is_better' attribute."
                            " Please provide the `maximize` argument explicitly."
                        )
                    # Per-class wrappers (e.g. a classwise wrapper around a metric)
                    # expand into one entry per class; regular metrics contribute
                    # a single entry.
                    num_classes = getattr(m, "num_classes", None)
                    if isinstance(m, Metric) and isinstance(num_classes, int) and getattr(m, "_per_class", False):
                        m_higher_is_better = [m.higher_is_better for _ in range(int(num_classes))]
                    else:
                        m_higher_is_better = [m.higher_is_better]
                    self.maximize.extend(m_higher_is_better)  # type: ignore[arg-type]
        else:
            if not isinstance(maximize, (bool, list)):
                raise ValueError("Argument `maximize` should either be a single bool or list of bool")
            if isinstance(maximize, list) and not all(isinstance(m, bool) for m in maximize):
                raise ValueError("Argument `maximize` is list but not type of bool.")
            if isinstance(maximize, list) and isinstance(metric, MetricCollection) and len(maximize) != len(metric):
                raise ValueError("The len of argument `maximize` should match the length of the metric collection")
            if isinstance(metric, Metric) and not isinstance(maximize, bool):
                raise ValueError("Argument `maximize` should be a single bool when `metric` is a single Metric")
            self.maximize = maximize

        self._increment_called = False

    @property
    def n_steps(self) -> int:
        """Return the number of times the tracker has been incremented."""
        return len(self) - 1  # subtract the base metric

    def increment(self) -> None:
        """Create a new instance of the input metric that will be updated next."""
        self._increment_called = True
        self.append(deepcopy(self._base_metric))

    def forward(self, *args: Any, **kwargs: Any) -> None:
        """Call forward of the current metric being tracked."""
        self._check_for_increment("forward")
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}.")
        return self[-1](*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the current metric being tracked."""
        self._check_for_increment("update")
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}.")
        self[-1].update(*args, **kwargs)

    def compute(self) -> Any:
        """Call compute of the current metric being tracked."""
        self._check_for_increment("compute")
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}.")
        return self[-1].compute()

    def compute_all(self) -> Any:
        """Compute the metric value for all tracked metrics.

        Return:
            By default will try stacking the results from all increments into a single tensor if the tracked base
            object is a single metric. If a metric collection is provided a dict of stacked tensors will be returned.
            If the stacking process fails a list of the computed results will be returned.

        Raises:
            ValueError: if ``self.increment`` has not been called before this method is called

        """
        self._check_for_increment("compute_all")
        # The i!=0 accounts for the self._base_metric should be ignored
        res: list[Any] = []
        for i, metric in enumerate(self):
            if i == 0:
                continue
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(f"Expected the item to be a Metric or MetricCollection, but got {type(metric)}.")
            res.append(metric.compute())

        try:
            if isinstance(res[0], dict):
                keys = res[0].keys()
                return {k: tp.stack([cast(Tensor, r[k]) for r in res], dim=0) for k in keys}

            if isinstance(res[0], list):
                return tp.stack([tp.stack(cast(list[Tensor], r), dim=0) for r in res], dim=0)

            return tp.stack(cast(list[Tensor], res), dim=0)

        except TypeError:  # fallback solution to just return as it is if we cannot successfully stack
            return res
        return res

    def reset(self) -> None:
        """Reset the current metric being tracked."""
        if not isinstance(self[-1], (Metric, MetricCollection)):
            raise TypeError(f"Expected the last item to be a Metric or MetricCollection, but got {type(self[-1])}.")
        self[-1].reset()

    def reset_all(self) -> None:
        """Reset all metrics being tracked."""
        for metric in self:
            if not isinstance(metric, (Metric, MetricCollection)):
                raise TypeError(f"Expected all metrics to be Metric or MetricCollection, but got {type(metric)}.")
            metric.reset()

    def best_metric(
        self, return_step: bool = False
    ) -> Union[
        None,
        float,
        Tensor,
        tuple[Union[int, float, Tensor], Union[int, float, Tensor]],
        tuple[None, None],
        dict[str, Union[float, None]],
        tuple[dict[str, Union[float, None]], dict[str, Union[int, None]]],
    ]:
        """Return the highest metric out of all tracked.

        Args:
            return_step: if ``True`` will also return the step with the highest metric value

        Returns:
            Either a single value or a tuple, depends on the value of ``return_step`` and the object being tracked:

            - If a single metric is being tracked and ``return_step=False`` then a single tensor will be returned
            - If a single metric is being tracked and ``return_step=True`` then a 2-element tuple will be returned,
              where the first value is the optimal value and second value is the corresponding optimal step
            - If a metric collection is being tracked and ``return_step=False`` then a single dict will be returned,
              where keys correspond to the different values of the collection and the values are the optimal metric
              value
            - If a metric collection is being tracked and ``return_step=True`` then a 2-element tuple will be
              returned where each is a dict, with keys corresponding to the different values of the collection and
              the values of the first dict being the optimal values and the values of the second dict being the
              optimal step

            In addition the value in all cases may be ``None`` if the underlying metric does not have a proper defined
            way of being optimal or in the case where a nested structure of metrics are being tracked.

        """
        res = self.compute_all()
        if isinstance(res, list):
            rank_zero_warn(
                "Encountered nested structure. You are probably using a metric collection inside a metric collection,"
                " or a metric wrapper inside a metric collection, which is not supported by `.best_metric()` method."
                " Returning `None` instead."
            )
            if return_step:
                return None, None
            return None

        if isinstance(self._base_metric, Metric):
            fn = tp.max if self.maximize else tp.min
            try:
                value, idx = fn(res, 0)
                if return_step:
                    return value.item(), idx.item()
                return value.item()
            except (ValueError, RuntimeError) as error:
                rank_zero_warn(
                    f"Encountered the following error when trying to get the best metric: {error}"
                    "this is probably due to the 'best' not being defined for this metric."
                    "Returning `None` instead.",
                    UserWarning,
                )
                if return_step:
                    return None, None
                return None

        else:  # this is a metric collection
            maximize = self.maximize if isinstance(self.maximize, list) else len(res) * [self.maximize]
            value, idx = {}, {}  # type: ignore[assignment]
            for i, (k, v) in enumerate(res.items()):
                try:
                    fn = tp.max if maximize[i] else tp.min
                    out = fn(v, 0)
                    value[k], idx[k] = out[0].item(), out[1].item()
                except (ValueError, RuntimeError) as error:
                    rank_zero_warn(
                        f"Encountered the following error when trying to get the best metric for metric {k}:"
                        f"{error} this is probably due to the 'best' not being defined for this metric."
                        "Returning `None` instead.",
                        UserWarning,
                    )
                    value[k], idx[k] = None, None  # type: ignore[assignment]

            if return_step:
                return value, idx
            return value

    def _check_for_increment(self, method: str) -> None:
        """Check that a metric that can be updated/used for computations has been initialized."""
        if not self._increment_called:
            raise ValueError(f"`{method}` cannot be called before `.increment()` has been called.")

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or ``metric.compute`` or a list of these
                results. If no value is provided, will automatically call ``metric.compute`` and plot that result.
            ax: a matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        """
        val = val if val is not None else self.compute_all()
        fig, ax = plot_single_or_multi_val(
            val,
            ax=ax,
            name=self.__class__.__name__,
        )
        return fig, ax