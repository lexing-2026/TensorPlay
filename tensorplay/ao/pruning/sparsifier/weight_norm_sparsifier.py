"""Weight-norm based block sparsifier."""

import operator
from collections.abc import Callable
from functools import reduce

import tensorplay as tp
from tensorplay.nn import functional as F

from .base_sparsifier import BaseSparsifier

__all__ = ["WeightNormSparsifier"]


def _flat_idx_to_2d(idx, shape):
    rows = idx // shape[1]
    cols = idx % shape[1]
    return rows, cols


class WeightNormSparsifier(BaseSparsifier):
    """Zero out sparse blocks with the lowest norm.

    Controlled by ``sparsity_level`` (fraction of blocks removed),
    ``sparse_block_shape`` (rows, cols of a block) and ``zeros_per_block``
    (target zeros inside each surviving block).
    """

    def __init__(
        self,
        sparsity_level: float = 0.5,
        sparse_block_shape: tuple[int, int] = (1, 4),
        zeros_per_block: int | None = None,
        norm: Callable | int | None = None,
    ):
        if zeros_per_block is None:
            zeros_per_block = reduce(operator.mul, sparse_block_shape)
        defaults = {
            "sparsity_level": sparsity_level,
            "sparse_block_shape": sparse_block_shape,
            "zeros_per_block": zeros_per_block,
        }
        if norm is None:
            norm = 2
        if callable(norm):
            self.norm_fn = norm
        elif norm == 1:
            self.norm_fn = lambda T: T.abs()
        elif norm == 2:
            self.norm_fn = lambda T: T * T
        else:
            raise NotImplementedError(f"L-{norm} is not yet implemented.")
        super().__init__(defaults=defaults)

    def _scatter_fold_block_mask(
        self, output_shape, dim, indices, block_shape, mask=None, input_shape=None, device=None
    ):
        if mask is None:
            if input_shape is None:
                raise AssertionError("input_shape must be provided when mask is None")
            mask = tp.ones(input_shape, device=device)
        mask.scatter_(dim=dim, index=indices, value=0)
        mask.data = F.fold(
            mask, output_size=output_shape, kernel_size=block_shape, stride=block_shape
        )
        return mask

    def _make_tensor_mask(self, data, input_shape, sparsity_level, sparse_block_shape, mask=None):
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w

        if mask is None:
            mask = tp.ones(h + dh, w + dw, device=data.device)

        if sparsity_level >= 1.0:
            mask.data = tp.zeros_like(mask)
            return mask
        elif sparsity_level <= 0.0:
            mask.data = tp.ones_like(mask)
            return mask

        values_per_block = reduce(operator.mul, sparse_block_shape)
        if values_per_block > 1:
            data = F.avg_pool2d(
                data[None, None, :],
                kernel_size=sparse_block_shape,
                stride=sparse_block_shape,
                ceil_mode=True,
            )
        data = data.flatten()
        num_blocks = len(data)

        data = data.repeat(1, values_per_block, 1)

        threshold_idx = round(sparsity_level * num_blocks)
        threshold_idx = max(0, min(num_blocks - 1, threshold_idx))
        _, sorted_idx = tp.topk(data, k=threshold_idx, dim=2, largest=False)

        mask_reshape = mask.reshape(data.shape)
        self._scatter_fold_block_mask(
            dim=2,
            output_shape=(h + dh, w + dw),
            indices=sorted_idx,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )
        mask.data = mask_reshape.squeeze().reshape(mask.shape)[:h, :w].contiguous()
        return mask

    def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w
        values_per_block = reduce(operator.mul, sparse_block_shape)

        if mask is None:
            mask = tp.ones((h + dh, w + dw), device=data.device)

        if values_per_block == zeros_per_block:
            mask.data = tp.zeros_like(mask)
            return mask

        padded_data = tp.ones(h + dh, w + dw, dtype=data.dtype, device=data.device)
        padded_data.fill_(tp.nan)
        padded_data[:h, :w] = data
        unfolded_data = F.unfold(
            padded_data[None, None, :],
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
        )

        mask_reshape = mask.reshape(unfolded_data.shape)
        _, sorted_idx = tp.topk(
            unfolded_data, k=zeros_per_block, dim=1, largest=False
        )

        self._scatter_fold_block_mask(
            dim=1,
            indices=sorted_idx,
            output_shape=padded_data.shape,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )

        mask.data = mask_reshape.squeeze().reshape(mask.shape).contiguous()
        return mask

    def update_mask(self, module, tensor_name, sparsity_level, sparse_block_shape,
                    zeros_per_block, **kwargs):
        values_per_block = reduce(operator.mul, sparse_block_shape)
        if zeros_per_block > values_per_block:
            raise ValueError(
                "Number of zeros per block cannot be more than the total number of elements in that block."
            )
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        mask = getattr(module.parametrizations, tensor_name)[0].mask
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = tp.ones_like(mask)
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = tp.zeros_like(mask)
        else:
            ww = self.norm_fn(getattr(module, tensor_name))
            tensor_mask = self._make_tensor_mask(
                data=ww,
                input_shape=ww.shape,
                sparsity_level=sparsity_level,
                sparse_block_shape=sparse_block_shape,
            )
            if values_per_block != zeros_per_block:
                block_mask = self._make_block_mask(
                    data=ww,
                    sparse_block_shape=sparse_block_shape,
                    zeros_per_block=zeros_per_block,
                )
                tensor_mask = tp.logical_or(tensor_mask, block_mask)
            mask.data = tensor_mask