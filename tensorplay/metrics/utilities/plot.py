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

    def _error_on_missing_matplotlib() -> None:
        return None

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
            if isinstance(t, Tensor):
                arr = np.asarray(t.cpu().detach())
            else:
                arr = t
            return arr.item() if isinstance(arr, np.ndarray) and arr.ndim == 0 else arr

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

    def plot_curve(
        curve: Union[tuple[Tensor, Tensor, Tensor], tuple[list[Tensor], list[Tensor], list[Tensor]]],
        score: Optional[Tensor] = None,
        ax: Optional[Any] = None,
        label_names: Optional[tuple[str, str]] = None,
        legend_name: Optional[str] = None,
        name: Optional[str] = None,
        labels: Optional[list[Union[int, str]]] = None,
    ) -> tuple[Any, Any]:
        """Plot a curve object.

        Args:
            curve: a tuple of ``(x, y, t)`` where ``x`` and ``y`` are the coordinates of the curve and ``t``
                are the thresholds used to compute the curve
            score: optional area under the curve added as label to the plot
            ax: Axis from a figure
            label_names: tuple containing the names of the x and y axis
            legend_name: name of the curve to be used in the legend
            name: custom name to describe the metric
            labels: optional labels for the different curves that will be added to the plot

        Returns:
            A tuple consisting of the figure and respective ax objects (or array of ax objects) of the
            generated figure

        Raises:
            ModuleNotFoundError: if ``matplotlib`` is not installed
            ValueError: if ``curve`` does not have 3 elements, being in the wrong format

        """
        if len(curve) < 2:
            raise ValueError(f"Expected 2 or 3 elements in curve but got {len(curve)}")
        x, y = curve[:2]

        fig, ax = plt.subplots() if ax is None else (None, ax)

        if isinstance(x, Tensor) and isinstance(y, Tensor) and x.ndim == 1 and y.ndim == 1:
            label = f"AUC={score.item():0.3f}" if score is not None else None
            ax.plot(x.detach().cpu(), y.detach().cpu(), linestyle="-", linewidth=2, label=label)
            if label is not None:
                ax.legend()
        elif (isinstance(x, list) and isinstance(y, list)) or (
            isinstance(x, Tensor) and isinstance(y, Tensor) and x.ndim == 2 and y.ndim == 2
        ):
            n_classes = len(x)
            if labels is not None and len(labels) != n_classes:
                raise ValueError(
                    "Expected number of elements in arg `labels` to match number of curves but "
                    f"got {len(labels)} and {n_classes}"
                )

            for i, (x_, y_) in enumerate(zip(x, y)):
                label = f"{legend_name}_{i}" if legend_name is not None else str(i) if labels is None else str(labels[i])
                label += f" AUC={score[i].item():0.3f}" if score is not None else ""
                ax.plot(x_.detach().cpu(), y_.detach().cpu(), linestyle="-", linewidth=2, label=label)
                ax.legend()
        else:
            raise ValueError(
                f"Unknown format for argument `x` and `y`. Expected either list or tensors but got {type(x)} and {type(y)}."
            )
        if label_names is not None:
            ax.set_xlabel(label_names[0])
            ax.set_ylabel(label_names[1])
        ax.grid(True)
        ax.set_title(name)

        return fig, ax

else:  # pragma: no cover
    def _error_on_missing_matplotlib() -> None:
        raise ModuleNotFoundError(
            "Plotting requires `matplotlib` to be installed. Install it with `pip install matplotlib`."
        )

    def plot_single_or_multi_val(*_: Any, **__: Any) -> Any:
        raise ModuleNotFoundError(
            "Plotting requires `matplotlib` to be installed. Install it with `pip install matplotlib`."
        )

    def plot_curve(*_: Any, **__: Any) -> Any:
        raise ModuleNotFoundError(
            "Plotting requires `matplotlib` to be installed. Install it with `pip install matplotlib`."
        )


__all__ = ["_AX_TYPE", "_PLOT_OUT_TYPE", "plot_curve", "plot_single_or_multi_val"]