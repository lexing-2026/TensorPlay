"""Multioutput wrapper metric."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Optional, Union, cast

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.nn import ModuleList

from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import apply_to_collection
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.wrappers.abstract import WrapperMetric


def _get_nan_indices(*tensors: Tensor) -> Tensor:
    """Get indices of rows along dim 0 which have NaN values."""
    if len(tensors) == 0:
        raise ValueError("Must pass at least one tensor as argument")
    sentinel = tensors[0]
    nan_idxs = tp.zeros(len(sentinel), dtype=tp.bool, device=sentinel.device)
    for tensor in tensors:
        permuted_tensor = tensor.flatten(start_dim=1)
        nan_idxs |= tp.any(tp.isnan(permuted_tensor), dim=1)
    return nan_idxs


class MultioutputWrapper(WrapperMetric):
    """Wrap a base metric to enable it to support multiple outputs.

    This wrapper tracks one underlying metric per output dimension. It does not support any
    aggregation across outputs: with ``num_outputs=2``, ``.compute()`` returns a tensor of shape
    ``(2, ...)`` where ``...`` represents the dimensions the metric returns when not wrapped.

    Args:
        base_metric: metric being wrapped
        num_outputs: expected dimensionality of the output dimension
        output_dim: dimension on which the output is expected
        remove_nans: whether to remove the intersection of rows containing NaNs from the values
            passed through to each underlying metric
        squeeze_outputs: if ``True``, will squeeze the 1-item dimensions left after ``index_select``
            is applied

    """

    is_differentiable: bool = False

    def __init__(
        self,
        base_metric: Metric,
        num_outputs: int,
        output_dim: int = -1,
        remove_nans: bool = True,
        squeeze_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.metrics = ModuleList([deepcopy(base_metric) for _ in range(num_outputs)])
        self.output_dim = output_dim
        self.remove_nans = remove_nans
        self.squeeze_outputs = squeeze_outputs

    def _get_args_kwargs_by_output(self, *args: Tensor, **kwargs: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Get args and kwargs reshaped to be output-specific and (maybe) having NaNs stripped out."""
        args_kwargs_by_output = []
        for i in range(len(self.metrics)):
            selected_args = apply_to_collection(
                args, Tensor, tp.index_select, dim=self.output_dim, index=tp.tensor(i, device=self.device)
            )
            selected_kwargs = apply_to_collection(
                kwargs, Tensor, tp.index_select, dim=self.output_dim, index=tp.tensor(i, device=self.device)
            )
            if self.remove_nans:
                args_kwargs = selected_args + tuple(selected_kwargs.values())
                nan_idxs = _get_nan_indices(*args_kwargs)
                selected_args = [arg[~nan_idxs] for arg in selected_args]
                selected_kwargs = {k: v[~nan_idxs] for k, v in selected_kwargs.items()}
            if self.squeeze_outputs:
                selected_args = [arg.squeeze(self.output_dim) for arg in selected_args]
                selected_kwargs = {k: v.squeeze(self.output_dim) for k, v in selected_kwargs.items()}
            args_kwargs_by_output.append((selected_args, selected_kwargs))
        return args_kwargs_by_output

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update each underlying metric with the corresponding output."""
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        for metric, (selected_args, selected_kwargs) in zip(self.metrics, reshaped_args_kwargs):
            cast(Metric, metric).update(*selected_args, **cast(Mapping, selected_kwargs))

    def compute(self) -> Tensor:
        """Compute metrics."""
        return tp.stack([cast(Metric, m).compute() for m in self.metrics], 0)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call underlying forward methods and aggregate the results if they're non-null.

        We override this method to ensure that state variables get copied over on the underlying metrics.

        """
        reshaped_args_kwargs = self._get_args_kwargs_by_output(*args, **kwargs)
        results = [
            metric(*selected_args, **cast(Mapping, selected_kwargs))
            for metric, (selected_args, selected_kwargs) in zip(self.metrics, reshaped_args_kwargs)
        ]
        if results[0] is None:
            return None
        return tp.stack(results, 0)

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.metrics:
            cast(Metric, metric).reset()
        super().reset()

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
        return self._plot(val, ax)