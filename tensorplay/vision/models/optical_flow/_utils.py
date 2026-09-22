"""Helper ops for optical-flow models: absolute-coordinate sampling, coordinate grids, flow upsampling."""
from typing import Optional

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import Tensor


def grid_sample(img: Tensor, absolute_grid: Tensor, mode: str = "bilinear", align_corners: Optional[bool] = None) -> Tensor:
    """Sample ``img`` at absolute pixel coordinates instead of normalized ones.

    ``absolute_grid`` has shape (..., 2); the last axis holds (x, y) pixel
    coordinates. They are normalized internally against the spatial size of
    ``img`` before the underlying sampling op runs.
    """
    h, w = img.shape[-2:]

    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    # A single-row image needs no y normalization and keeps the op reusable elsewhere.
    if h > 1:
        ygrid = 2 * ygrid / (h - 1) - 1
    normalized_grid = tensorplay.cat([xgrid, ygrid], dim=-1)

    return F.grid_sample(img, normalized_grid, mode=mode, align_corners=align_corners)


def make_coords_grid(batch_size: int, h: int, w: int, device: str = "cpu") -> Tensor:
    """Build a (batch_size, 2, h, w) grid holding (x, y) pixel coordinates per position."""
    mesh = tensorplay.meshgrid([tensorplay.arange(h, device=device), tensorplay.arange(w, device=device)], indexing="ij")
    coords = tensorplay.stack(list(mesh)[::-1], dim=0).float()
    return tensorplay.unsqueeze(coords, 0).repeat(batch_size, 1, 1, 1)


def upsample_flow(flow: Tensor, up_mask: Optional[Tensor] = None, factor: int = 8) -> Tensor:
    """Upsample a flow field by ``factor`` (default 8).

    Without ``up_mask`` this is a plain bilinear interpolation scaled by
    ``factor``. With ``up_mask`` (shape (B, 9 * factor * factor, h, w)) the
    output is a convex combination of the 9 neighbouring flow values at each
    target cell, per the RAFT paper page 8 and appendix B.
    """
    batch_size, num_channels, h, w = flow.shape
    new_h, new_w = h * factor, w * factor

    if up_mask is None:
        return factor * F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=True)

    up_mask = up_mask.view(batch_size, 1, 9, factor, factor, h, w)
    up_mask = tensorplay.softmax(up_mask, dim=2)  # "convex" == weights sum to 1

    upsampled_flow = F.unfold(factor * flow, kernel_size=3, padding=1).view(
        batch_size, num_channels, 9, 1, 1, h, w
    )
    upsampled_flow = tensorplay.sum(up_mask * upsampled_flow, dim=2)

    return upsampled_flow.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, num_channels, new_h, new_w)
