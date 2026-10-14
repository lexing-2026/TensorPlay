"""Kernel Inception Distance (KID) metric."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import tensorplay as tp
from tensorplay import Tensor, nn

from tensorplay.metrics.image._inception import NoTrainInceptionV3
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from tensorplay.metrics.utilities.prints import rank_zero_warn

__all__ = ["KernelInceptionDistance"]


def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Compute the maximum mean discrepancy for a polynomial kernel."""
    m = k_xx.shape[0]

    diag_x = tp.diag(k_xx)
    diag_y = tp.diag(k_yy)

    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m**2)
    return value


def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    """Polynomial kernel between feature matrices."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree


def poly_mmd(
    f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
) -> Tensor:
    """Polynomial maximum mean discrepancy between two feature sets."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)


class KernelInceptionDistance(Metric):
    """Calculate the Kernel Inception Distance (KID).

    KID compares real and generated images via the maximum mean discrepancy
    (MMD) of Inception v3 features with a polynomial kernel:

    .. math::
        KID = MMD(f_{real}, f_{fake})^2

    The MMD is evaluated over a number of random subsets so that both a mean and
    a standard deviation over subsets are returned.

    Inputs are mini-batches of 3-channel RGB images. When ``normalize`` is
    ``True`` images are expected to be float tensors in ``[0, 1]``; otherwise
    they are expected to be uint8 tensors in ``[0, 255]``.

    Args:
        feature: a string or integer indicating the Inception v3 feature layer to
            use (one of ``'logits_unbiased'``, 64, 192, 768, 2048) or an
            ``nn.Module`` to use as a custom feature extractor returning an
            ``(N, d)`` matrix.
        subsets: number of random subsets to evaluate.
        subset_size: number of randomly picked samples in each subset.
        degree: degree of the polynomial kernel.
        gamma: scale of the polynomial kernel; if ``None`` set to the inverse of
            the feature size.
        coef: bias term in the polynomial kernel.
        reset_real_features: whether to reset the real distribution features when
            ``reset`` is called.
        kwargs: additional keyword arguments for :class:`Metric`.

    Raises:
        ValueError: if ``feature`` is an int not in ``(64, 192, 768, 2048)`` or a
            str not in ``('logits_unbiased', 64, 192, 768, 2048)``, or if any of
            the numeric/boolean arguments have invalid types or values.
        TypeError: if ``feature`` is neither a str/int nor an ``nn.Module``.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    real_features: List[Tensor]
    fake_features: List[Tensor]
    inception: nn.Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[str, int, nn.Module] = 2048,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef: float = 1.0,
        reset_real_features: bool = True,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `KernelInceptionDistance` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        self.used_custom_model = False

        if isinstance(feature, (str, int)):
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )
            self.inception = NoTrainInceptionV3([str(feature)])
        elif isinstance(feature, nn.Module):
            self.inception = feature
            self.used_custom_model = True
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not (isinstance(subsets, int) and subsets > 0):
            raise ValueError("Argument `subsets` expected to be integer larger than 0")
        self.subsets = subsets

        if not (isinstance(subset_size, int) and subset_size > 0):
            raise ValueError("Argument `subset_size` expected to be integer larger than 0")
        self.subset_size = subset_size

        if not (isinstance(degree, int) and degree > 0):
            raise ValueError("Argument `degree` expected to be integer larger than 0")
        self.degree = degree

        if gamma is not None and not (isinstance(gamma, float) and gamma > 0):
            raise ValueError("Argument `gamma` expected to be `None` or float larger than 0")
        self.gamma = gamma

        if not (isinstance(coef, float) and coef > 0):
            raise ValueError("Argument `coef` expected to be float larger than 0")
        self.coef = coef

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            imgs: input images to evaluate.
            real: whether the given images belong to the real distribution.
        """
        imgs = (imgs * 255).byte() if self.normalize and (not self.used_custom_model) else imgs
        features = self.inception(imgs)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> tuple[Tensor, Tensor]:
        """Compute the KID score over random subsets.

        Returns:
            (mean, standard deviation) of the MMD over subsets.
        """
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        n_samples_real = real_features.shape[0]
        if n_samples_real < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")
        n_samples_fake = fake_features.shape[0]
        if n_samples_fake < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")

        kid_scores_ = []
        for _ in range(self.subsets):
            perm = tp.randperm(n_samples_real)
            f_real = real_features[perm[: self.subset_size]]
            perm = tp.randperm(n_samples_fake)
            f_fake = fake_features[perm[: self.subset_size]]

            o = poly_mmd(f_real, f_fake, self.degree, self.gamma, self.coef)
            kid_scores_.append(o)
        kid_scores = tp.stack(kid_scores_)
        return kid_scores.mean(), kid_scores.std(correction=0)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()

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