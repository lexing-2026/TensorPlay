"""Functional interface for the total variation metric."""

from typing import Optional, Union

from tensorplay import Tensor


def _total_variation_update(img: Tensor) -> tuple[Tensor, int]:
    """Compute total variation statistics on the current batch."""
    if img.ndim != 4:
        raise RuntimeError(f"Expected input `img` to be an 4D tensor, but got {img.shape}")
    diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]

    res1 = diff1.abs().sum([1, 2, 3])
    res2 = diff2.abs().sum([1, 2, 3])
    score = res1 + res2
    return score, img.shape[0]


def _total_variation_compute(
    score: Tensor, num_elements: Union[int, Tensor], reduction: Optional[str]
) -> Tensor:
    """Compute the final total variation score."""
    if reduction == "mean":
        return score.sum() / num_elements
    if reduction == "sum":
        return score.sum()
    if reduction is None or reduction == "none":
        return score
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")


def total_variation(img: Tensor, reduction: Optional[str] = "sum") -> Tensor:
    """Compute the total variation loss.

    The total variation measures the sum of absolute differences between neighboring pixel values,
    providing a smoothness prior for images.

    Args:
        img: a tensor of shape ``(N, C, H, W)`` consisting of images
        reduction: a method to reduce metric score over samples:

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample

    Returns:
        A loss scalar value containing the total variation

    Raises:
        ValueError: if ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``
        RuntimeError: if ``img`` is not 4D tensor

    """
    score, num_elements = _total_variation_update(img)
    return _total_variation_compute(score, num_elements, reduction)