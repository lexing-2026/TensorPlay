"""Learned Perceptual Image Patch Similarity (LPIPS) metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.image.lpips import _lpips_compute, _lpips_update, _NoTrainLpips
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from typing import Literal

__all__ = ["LearnedPerceptualImagePatchSimilarity"]


class LearnedPerceptualImagePatchSimilarity(Metric):
    """Calculate the Learned Perceptual Image Patch Similarity (LPIPS).

    LPIPS computes the similarity between the activations of two image patches
    under a pretrained backbone network. A low LPIPS score means that the image
    patches are perceptually similar.

    Both inputs are expected to have shape ``(N, 3, H, W)``. The minimum size of
    ``H, W`` depends on the chosen backbone.

    Args:
        net_type: backbone network type, one of ``'alex'``, ``'vgg'`` or
            ``'squeeze'``.
        reduction: how to reduce over the batch dimension, one of ``'sum'``,
            ``'mean'``, ``'none'`` or ``None``.
        normalize: if ``False`` inputs are expected in ``[-1, 1]``; if ``True``
            inputs are expected in ``[0, 1]``.
        kwargs: additional keyword arguments for :class:`Metric`.

    Raises:
        ValueError: if ``net_type`` is not one of ``'vgg'``, ``'alex'`` or
            ``'squeeze'``, if ``reduction`` is not one of ``'mean'``, ``'sum'``,
            ``'none'`` or ``None``, or if ``normalize`` is not a bool.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    all_scores: list
    feature_network: str = "net"

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Optional[Literal["sum", "mean", "none"]] = "mean",
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}.")
        self.net = _NoTrainLpips(net=net_type)

        valid_reduction = ("mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` should be an bool but got {normalize}")
        self.normalize = normalize

        self.add_state("all_scores", default=[], dist_reduce_fx=None)

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update internal states with the LPIPS score.

        Args:
            img1: first set of images.
            img2: second set of images.
        """
        loss = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        self.all_scores.append(loss)

    def compute(self) -> Tensor:
        """Compute the final perceptual similarity metric."""
        scores = dim_zero_cat(self.all_scores)
        return _lpips_compute(scores, reduction=self.reduction)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: either a single result from calling ``metric.forward`` or
                ``metric.compute`` or a list of these results. If no value is
                provided, ``metric.compute`` is called and that result plotted.
            ax: a matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object
        """
        return self._plot(val, ax)