"""Structural similarity index measure metrics."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import tensorplay as tp
from tensorplay import Tensor

from tensorplay.metrics.functional.image.ssim import _multiscale_ssim_update, _ssim_check_inputs, _ssim_update
from tensorplay.metrics.metric import Metric
from tensorplay.metrics.utilities.data import dim_zero_cat
from tensorplay.metrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class StructuralSimilarityIndexMeasure(Metric):
    """Compute the structural similarity index measure (SSIM).

    The metric measures the similarity between two images, combining luminance, contrast and structure
    terms over local windows.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predictions from model
    - ``target`` (float tensor): ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``ssim`` (float tensor): if ``reduction!='none'`` returns a float scalar tensor with average SSIM value
      over sample else returns tensor of shape ``(N,)`` with SSIM values per sample

    Args:
        gaussian_kernel: if ``True`` (default), a gaussian kernel is used, if ``False`` a uniform kernel is used
        sigma: standard deviation of the gaussian kernel, anisotropic kernels are possible. Ignored if a uniform
            kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible. Ignored if a Gaussian kernel
            is used
        reduction: a method to reduce metric score over individual batch scores:

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: the range of the data. If ``None``, it is determined from the data (max - min). If a tuple is
            provided then the range is calculated as the difference and input is clamped between the values.
        k1: parameter of SSIM
        k2: parameter of SSIM
        return_full_image: if ``True``, the full ``ssim`` image is returned as a second argument. Mutually exclusive
            with ``return_contrast_sensitivity``
        return_contrast_sensitivity: if ``True``, the contrast term is returned as a second argument. The luminance
            term can be obtained with ``luminance=ssim/contrast``. Mutually exclusive with ``return_full_image``

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: str = "elementwise_mean",
        data_range: Optional[Union[float, tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=tp.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx=None)
        self.add_state("total", default=tp.tensor(0.0), dist_reduce_fx="sum")
        if return_contrast_sensitivity or return_full_image:
            self.add_state("image_return", default=[], dist_reduce_fx="cat")
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity_pack = _ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2,
            self.return_full_image,
            self.return_contrast_sensitivity,
        )
        if isinstance(similarity_pack, tuple):
            similarity, image = similarity_pack
        else:
            similarity = similarity_pack
        if self.return_contrast_sensitivity or self.return_full_image:
            if not isinstance(self.image_return, list):
                raise TypeError("Expected `self.image_return` to be a list when returning images.")
            self.image_return.append(image)
        if self.reduction in ("elementwise_mean", "sum"):
            if not isinstance(self.similarity, Tensor):
                raise TypeError("Expected `self.similarity` to be a Tensor for reductions.")
            self.similarity += similarity.sum()
            if not isinstance(self.total, Tensor):
                raise TypeError("Expected `self.total` to be a Tensor.")
            self.total += preds.shape[0]
        else:
            if not isinstance(self.similarity, list):
                raise TypeError("Expected `self.similarity` to be a list when reduction='none'.")
            self.similarity.append(similarity)

    def compute(self) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Compute SSIM over state."""
        if self.reduction == "elementwise_mean":
            if isinstance(self.similarity, Tensor) and isinstance(self.total, Tensor):
                similarity = self.similarity / self.total
            else:
                raise TypeError(
                    "Expected `self.similarity`and `self.total` to be of type Tensor for elementwise_mean reduction."
                )
        elif self.reduction == "sum":
            if not isinstance(self.similarity, Tensor):
                raise TypeError("Expected `self.similarity` to be a Tensor for sum reduction.")
            similarity = self.similarity
        else:
            if isinstance(self.similarity, list):
                similarity = dim_zero_cat(self.similarity)  # Concatenate list of Tensors
            else:
                raise TypeError("Expected `self.similarity` to be a list for reduction='none'.")
        if self.return_contrast_sensitivity or self.return_full_image:
            if isinstance(self.image_return, list):
                image_return = dim_zero_cat(self.image_return)  # Concatenate list of Tensors
            else:
                raise TypeError("Expected `self.image_return` to be a list when returning images.")
            return similarity, image_return
        return similarity

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


class MultiScaleStructuralSimilarityIndexMeasure(Metric):
    """Compute the multi-scale structural similarity index measure (MS-SSIM).

    This metric is a generalization of the structural similarity index measure by incorporating image
    details at different resolution scales.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (float tensor): predictions from model
    - ``target`` (float tensor): ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``msssim`` (float tensor): if ``reduction!='none'`` returns a float scalar tensor with average MS-SSIM value
      over sample else returns tensor of shape ``(N,)`` with MS-SSIM values per sample

    Args:
        gaussian_kernel: if ``True`` (default), a gaussian kernel is used, if ``False`` a uniform kernel is used
        kernel_size: size of the gaussian kernel
        sigma: standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels:

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: the range of the data. If ``None``, it is determined from the data (max - min). If a tuple is
            provided then the range is calculated as the difference and input is clamped between the values.
        k1: parameter of structural similarity index measure
        k2: parameter of structural similarity index measure
        betas: exponent parameters for individual similarities and contrast sensitivities returned by different
            image resolutions
        normalize: when the metric is used as a training loss it is desirable to normalize the values to improve
            training stability

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        gaussian_kernel: bool = True,
        kernel_size: Union[int, Sequence[int]] = 11,
        sigma: Union[float, Sequence[float]] = 1.5,
        reduction: str = "elementwise_mean",
        data_range: Optional[Union[float, tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        betas: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        normalize: Optional[str] = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        valid_reduction = ("elementwise_mean", "sum", "none", None)
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        if reduction in ("elementwise_mean", "sum"):
            self.add_state("similarity", default=tp.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("similarity", default=[], dist_reduce_fx=None)
        self.add_state("total", default=tp.tensor(0.0), dist_reduce_fx="sum")
        if not isinstance(kernel_size, (Sequence, int)):
            raise ValueError(
                f"Argument `kernel_size` expected to be a sequence or an int, or a single int. Got {kernel_size}"
            )
        if isinstance(kernel_size, Sequence) and (
            len(kernel_size) not in (2, 3) or not all(isinstance(ks, int) for ks in kernel_size)
        ):
            raise ValueError(
                "Argument `kernel_size` expected to be a sequence of size 2 or 3 where each element is an int, "
                f"or a single int. Got {kernel_size}"
            )
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        if not isinstance(betas, tuple):
            raise ValueError("Argument `betas` is expected to be of a type tuple.")
        if isinstance(betas, tuple) and not all(isinstance(beta, float) for beta in betas):
            raise ValueError("Argument `betas` is expected to be a tuple of floats.")
        self.betas = betas
        if normalize and normalize not in ("relu", "simple"):
            raise ValueError("Argument `normalize` to be expected either `None` or one of 'relu' or 'simple'")
        self.normalize = normalize

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity = _multiscale_ssim_update(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.data_range,
            self.k1,
            self.k2,
            self.betas,
            self.normalize,
        )
        if self.reduction in ("none", None):
            if not isinstance(self.similarity, list):
                raise TypeError("Expected `self.similarity` to be a list for reduction='none'.")
            self.similarity.append(similarity)
        else:
            if not isinstance(self.similarity, Tensor):
                raise TypeError("Expected `self.similarity` to be a Tensor for elementwise_mean or sum reduction.")
            self.similarity += similarity.sum()
        if not isinstance(self.total, Tensor):
            raise TypeError("Expected `self.total` to be a Tensor.")
        self.total += tp.tensor(preds.shape[0], dtype=self.total.dtype, device=self.total.device)

    def compute(self) -> Tensor:
        """Compute MS-SSIM over state."""
        if self.reduction in ("none", None):
            if isinstance(self.similarity, list):
                return dim_zero_cat(self.similarity)
            raise TypeError("Expected `self.similarity` to be a list for reduction='none'.")
        if self.reduction == "sum":
            if isinstance(self.similarity, Tensor):
                return self.similarity
            raise TypeError("Expected `self.similarity` to be a Tensor for sum reduction.")
        if isinstance(self.similarity, Tensor) and isinstance(self.total, Tensor):
            return self.similarity / self.total
        raise TypeError("Expected `self.similarity` and `self.total` to be Tensors for elementwise_mean reduction.")

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