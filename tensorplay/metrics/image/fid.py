"""Frechet Inception Distance (FID) metric."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorplay as tp
from tensorplay import Tensor, nn

from tensorplay.metrics.image._inception import NoTrainInceptionV3
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

__all__ = ["FrechetInceptionDistance"]


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute the Frechet distance between two multivariate Gaussians.

    The Frechet Inception Distance between two multivariate Gaussians
    :math:`\mathcal{N}(\mu_1, \Sigma_1)` and :math:`\mathcal{N}(\mu_2, \Sigma_2)`
    is :math:`d^2 = \|\mu_1 - \mu_2\|^2 + \mathrm{Tr}(\Sigma_1 + \Sigma_2 -
    2\sqrt{\Sigma_1 \Sigma_2})`.

    Args:
        mu1: mean of activations of the first distribution.
        sigma1: covariance matrix of the first distribution.
        mu2: mean of activations of the second distribution.
        sigma2: covariance matrix of the second distribution.

    Returns:
        scalar value of the Frechet distance.
    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = tp.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
    return a + b - 2 * c


class FrechetInceptionDistance(Metric):
    """Calculate the Frechet inception distance (FID).

    FID assesses the quality of generated images by comparing the statistics of
    Inception v3 features computed on real and generated image sets:

    .. math::
        FID = \|\mu - \mu_w\|^2 + \mathrm{Tr}(\Sigma + \Sigma_w -
        2(\Sigma \Sigma_w)^{\frac{1}{2}})

    Inputs are mini-batches of 3-channel RGB images. When ``normalize`` is
    ``True`` images are expected to be float tensors in ``[0, 1]``; otherwise
    they are expected to be uint8 tensors in ``[0, 255]``. All images are
    resized to 299x299 internally.

    Args:
        feature: an integer indicating the Inception v3 feature layer to use
            (one of 64, 192, 768, 2048) or an ``nn.Module`` to use as a custom
            feature extractor returning an ``(N, d)`` matrix.
        reset_real_features: whether to reset the real distribution features when
            ``reset`` is called. Set to ``False`` when the real dataset does not
            change across resets to cache its features.
        normalize: whether input images are in ``[0, 1]`` (``True``) or
            ``[0, 255]`` (``False``).
        input_img_size: input image size for a custom feature extractor.
        feature_extractor_weights_path: optional path to a local Inception v3
            state dict, passed to the feature extractor.
        kwargs: additional keyword arguments for :class:`Metric`.

    Raises:
        ValueError: if ``feature`` is an integer not in (64, 192, 768, 2048),
            or ``normalize``/``reset_real_features`` are not booleans.
        TypeError: if ``feature`` is neither an int nor an ``nn.Module``.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    inception: nn.Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[int, nn.Module] = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        input_img_size: tuple[int, int, int] = (3, 299, 299),
        feature_extractor_weights_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize
        self.used_custom_model = False

        if isinstance(feature, int):
            num_features = feature
            valid_int_input = (64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )
            self.inception = NoTrainInceptionV3([str(feature)], weights_path=feature_extractor_weights_path)
        elif isinstance(feature, nn.Module):
            self.inception = feature
            self.used_custom_model = True
            if hasattr(self.inception, "num_features"):
                num_features = int(self.inception.num_features)
            else:
                dummy = tp.rand(1, *input_img_size, dtype=tp.float32) if self.normalize else tp.randint(0, 255, (1, *input_img_size), dtype=tp.uint8)
                num_features = self.inception(dummy).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        self.add_state("real_features_sum", tp.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", tp.zeros((num_features, num_features)).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", tp.tensor(0, dtype=tp.int64), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", tp.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", tp.zeros((num_features, num_features)).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", tp.tensor(0, dtype=tp.int64), dist_reduce_fx="sum")

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            imgs: input images to evaluate.
            real: whether the given images belong to the real distribution.
        """
        imgs = (imgs * 255).byte() if self.normalize and (not self.used_custom_model) else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Compute the FID score from the accumulated features."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distribution to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = self.real_features_sum.detach().clone()
            real_features_cov_sum = self.real_features_cov_sum.detach().clone()
            real_features_num_samples = self.real_features_num_samples.detach().clone()
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
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
        return self._plot(val, ax)