import tensorplay
from tensorplay import Tensor

from tensorplay.metrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from tensorplay.metrics.functional.image.utils import _uniform_filter


def _rase_update(
    preds: Tensor, target: Tensor, window_size: int, rmse_map: Tensor, target_sum: Tensor, total_images: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate the sum of RMSE map values for the batch of examples and update intermediate states.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for RMSE calculation
        rmse_map: Sum of RMSE map values over all examples
        target_sum: target...
        total_images: Total number of images

    Return:
        Intermediate state of RMSE map
        Updated total number of already processed images

    """
    _, rmse_map, total_images = _rmse_sw_update(
        preds, target, window_size, rmse_val_sum=None, rmse_map=rmse_map, total_images=total_images
    )
    target_sum += tensorplay.sum(_uniform_filter(target, window_size) / (window_size**2), dim=0)
    return rmse_map, target_sum, total_images


def _rase_compute(rmse_map: Tensor, target_sum: Tensor, total_images: Tensor, window_size: int) -> Tensor:
    """Compute RASE.

    Args:
        rmse_map: Sum of RMSE map values over all examples
        target_sum: target...
        total_images: Total number of images.
        window_size: Sliding window used for rmse calculation

    Return:
        Relative Average Spectral Error (RASE)

    """
    _, rmse_map = _rmse_sw_compute(rmse_val_sum=None, rmse_map=rmse_map, total_images=total_images)
    target_mean = target_sum / total_images
    target_mean = target_mean.mean(0)  # mean over image channels
    rase_map = 100 / target_mean * tensorplay.sqrt(tensorplay.mean(rmse_map**2, 0))
    crop_slide = round(window_size / 2)

    return tensorplay.mean(rase_map[crop_slide:-crop_slide, crop_slide:-crop_slide])


def relative_average_spectral_error(preds: Tensor, target: Tensor, window_size: int = 8) -> Tensor:
    """Compute Relative Average Spectral Error (RASE) (RelativeAverageSpectralError_).

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation

    Return:
        Relative Average Spectral Error (RASE)

    Example:
        >>> from tensorplay import rand
        >>> from tensorplay.metrics.functional.image import relative_average_spectral_error
        >>> preds = rand(4, 3, 16, 16)
        >>> target = rand(4, 3, 16, 16)
        >>> relative_average_spectral_error(preds, target)
        tensor(5326.40...)

    Raises:
        ValueError: If ``window_size`` is not a positive integer.

    """
    if not isinstance(window_size, int) or (isinstance(window_size, int) and window_size < 1):
        raise ValueError("Argument `window_size` is expected to be a positive integer.")

    img_shape = target.shape[1:]  # [num_channels, width, height]
    rmse_map = tensorplay.zeros(img_shape, dtype=target.dtype, device=target.device)
    target_sum = tensorplay.zeros(img_shape, dtype=target.dtype, device=target.device)
    total_images = tensorplay.tensor(0.0, device=target.device)

    rmse_map, target_sum, total_images = _rase_update(preds, target, window_size, rmse_map, target_sum, total_images)
    return _rase_compute(rmse_map, target_sum, total_images, window_size)