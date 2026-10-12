"""Plotting helpers for metrics.

Plotting is optional: it only works when ``matplotlib`` is installed. All plot
APIs raise :class:`ModuleNotFoundError` otherwise.
"""

from typing import Any, Optional, Sequence, Union

from tensorplay import Tensor

from tensorplay.metrics.utilities.imports import _MATPLOTLIB_AVAILABLE

_AX_TYPE = Any
_PLOT_OUT_TYPE = Any

if _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt

    def plot_single_or_multi_val(
        val: Union[Tensor, Sequence[Tensor], dict[str, Tensor], Sequence[dict[str, Tensor]]],
        ax: Optional[Any] = None,
        higher_is_better: Optional[bool] = True,
        name: str = "Metric",
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        legend_name: Optional[str] = None,
    ) -> tuple[Any, Any]:
        """Plot a single value, a list of values or a dict of values as a bar chart."""
        import numpy as np

        def _as_array(t):
            return np.asarray(t.cpu().detach()) if isinstance(t, Tensor) else t

        if isinstance(val, dict):
            keys = list(val.keys())
            values = [_as_array(v) for v in val.values()]
        elif isinstance(val, Sequence) and all(isinstance(v, dict) for v in val):
            keys = list(val[0].keys())
            values = [[_as_array(v[k]) for v in val] for k in keys]
        elif isinstance(val, Sequence):
            keys = [str(i) for i in range(len(val))]
            values = [_as_array(v) for v in val]
        else:
            keys = [name]
            values = [_as_array(val)]

        if ax is None:
            _, ax = plt.subplots()

        if isinstance(values[0], (list, np.ndarray)) and all(np.isscalar(v) for v in values[0]):
            # multiple values per key: plot as grouped bars
            width = 0.8 / len(values)
            x = np.arange(len(values[0]))
            for i, (k, v) in enumerate(zip(keys, values)):
                ax.bar(x + i * width, v, width, label=k)
            ax.set_xticks(x)
            if legend_name:
                ax.legend(title=legend_name)
            else:
                ax.legend()
        else:
            ax.bar(keys, values)
            if legend_name:
                ax.set_xlabel(legend_name)
        ax.set_title(name)
        if lower_bound is not None:
            ax.set_ylim(bottom=lower_bound)
        if upper_bound is not None:
            ax.set_ylim(top=upper_bound)
        return ax.figure, ax

else:  # pragma: no cover
    def plot_single_or_multi_val(*_: Any, **__: Any) -> Any:
        raise ModuleNotFoundError(
            "Plotting requires `matplotlib` to be installed. Install it with `pip install matplotlib`."
        )


__all__ = ["_AX_TYPE", "_PLOT_OUT_TYPE", "plot_single_or_multi_val"]