from collections.abc import Sequence
from typing import Optional, Union

from tensorplay import Tensor

from tensorplay.metrics.functional.retrieval.r_precision import retrieval_r_precision
from tensorplay.metrics.retrieval.base import RetrievalMetric
from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["RetrievalRPrecision.plot"]


class RetrievalRPrecision(RetrievalMetric):
    """Compute `IR R-Precision`_.

    Works with binary target data. Accepts float predictions from a model output.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~tensorplay.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~tensorplay.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~tensorplay.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``rp`` (:class:`~tensorplay.Tensor`): A single-value tensor with the r-precision of the predictions ``preds``
      w.r.t. the labels ``target``.

    All ``indexes``, ``preds`` and ``target`` must have the same dimension and will be flatten at the beginning,
    so that for example, a tensor of shape ``(N, M)`` is treated as ``(N * M, )``. Predictions will be first grouped by
    ``indexes`` and then will be computed as the mean of the metric over each query.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index: Ignore predictions where the target is equal to this number.
        aggregation:
            Specify how to aggregate over indexes. Can either a custom callable function that takes in a single tensor
            and returns a scalar value or one of the following strings:

            - ``'mean'``: average value is returned
            - ``'median'``: median value is returned
            - ``'max'``: max value is returned
            - ``'min'``: min value is returned

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.

    Example:
        >>> from tensorplay import tensor
        >>> from tensorplay.metrics.retrieval import RetrievalRPrecision
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> p2 = RetrievalRPrecision()
        >>> p2(preds, target, indexes=indexes)
        tensor(0.7500)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_r_precision(preds, target)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> import torch
            >>> from tensorplay.metrics.retrieval import RetrievalRPrecision
            >>> # Example plotting a single value
            >>> metric = RetrievalRPrecision()
            >>> metric.update(tensorplay.rand(10,), tensorplay.randint(2, (10,)), indexes=tensorplay.randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> import torch
            >>> from tensorplay.metrics.retrieval import RetrievalRPrecision
            >>> # Example plotting multiple values
            >>> metric = RetrievalRPrecision()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(tensorplay.rand(10,), tensorplay.randint(2, (10,)), indexes=tensorplay.randint(2,(10,))))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)