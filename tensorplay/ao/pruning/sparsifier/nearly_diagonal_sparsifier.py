"""Nearly-diagonal band sparsifier."""

import tensorplay as tp

from .base_sparsifier import BaseSparsifier

__all__ = ["NearlyDiagonalSparsifier"]


class NearlyDiagonalSparsifier(BaseSparsifier):
    """Keep only the bands closest to the main diagonal of a weight matrix.

    ``nearliness`` is the number of non-zero diagonal lines around the main
    diagonal and must be odd.
    """

    def __init__(self, nearliness: int = 1):
        defaults = {"nearliness": nearliness}
        super().__init__(defaults=defaults)

    def update_mask(self, module, tensor_name, nearliness, **kwargs):
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        mask.data = tp.zeros_like(mask)
        if nearliness <= 0:
            return

        tensor = getattr(module, tensor_name)
        height, width = tensor.shape

        if nearliness % 2 == 0:
            raise ValueError("nearliness can only be an odd number")
        dist_to_diagonal = nearliness // 2
        if dist_to_diagonal >= min(height, width):
            raise ValueError(
                "nearliness cannot be larger than the dimensions of tensor."
            )

        for row in range(height):
            low = max(0, row - dist_to_diagonal)
            high = min(width, row + dist_to_diagonal + 1)
            mask[row, low:high].fill_(1)