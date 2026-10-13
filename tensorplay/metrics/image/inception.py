"""Inception Score (IS) metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor, nn

from tensorplay.metrics.image._inception import NoTrainInceptionV3
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.utilities.prints import rank_zero_warn

__all__ = ["InceptionScore"]


class InceptionScore(Metric):
    """Calculate the Inception Score (IS).

    IS assesses how realistic generated images are by measuring the KL
    divergence between the conditional label distribution :math:`p(y|x)` and
    the marginal distribution :math:`p(y)` obtained from Inception v3 features:

    .. math::
        IS = \exp(\mathbb{E}_x KL(p(y | x ) || p(y)))

    The score is computed on random splits of the images so that both a mean and
    a standard deviation are returned.

    Inputs are mini-batches of 3-channel RGB images. When ``normalize`` is
    ``True`` images are expected to be float tensors in ``[0, 1]``; otherwise
    they are expected to be uint8 tensors in ``[0, 255]``.

    Args:
        feature: a string or integer indicating the Inception v3 feature layer to
            use (one of ``'logits_unbiased'``, 64, 192, 768, 2048) or an
            ``nn.Module`` to use as a custom feature extractor returning an
            ``(N, d)`` matrix.
        splits: number of splits used when computing the score.
        kwargs: additional keyword arguments for :class:`Metric`.

    Raises:
        ValueError: if ``feature`` is an int not in ``(64, 192, 768, 2048)`` or a
            str not in ``('logits_unbiased', 64, 192, 768, 2048)``.
        TypeError: if ``feature`` is neither a str/int nor an ``nn.Module``.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    features: list
    inception: nn.Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[str, int, nn.Module] = "logits_unbiased",
        splits: int = 10,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `InceptionScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )
            self.inception = NoTrainInceptionV3([str(feature)])
        elif isinstance(feature, nn.Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        if not (isinstance(splits, int) and splits > 0):
            raise ValueError("Argument `splits` expected to be an integer larger than 0")
        self.splits = splits
        self.add_state("features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor) -> None:
        """Update the state with extracted features.

        Args:
            imgs: input images to evaluate.
        """
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.features.append(features)

    def compute(self) -> tuple[Tensor, Tensor]:
        """Compute the Inception Score.

        Returns:
            (mean, standard deviation) of the score over splits.
        """
        features = dim_zero_cat(self.features)
        # random permute the features
        idx = tp.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = tp.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()

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
        val = val or self.compute()[0]  # by default we select the mean to plot
        return self._plot(val, ax)