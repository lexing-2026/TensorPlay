"""Multi-scale ROI align (feature-pyramid pooling)."""
import math
import warnings
from typing import List, Sequence, Tuple, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once
from .roi_align import roi_align


def _infer_scale(feature_size: Sequence[int], original_size: Sequence[int]) -> float:
    """Spatial scale of a pyramid level, inferred as a power of two."""
    ratio = float(feature_size[0]) / original_size[0]
    scale = 2 ** round(math.log2(ratio))
    if abs(ratio - scale) > 1e-3:
        warnings.warn(
            f"Building MultiScaleRoIAlign with a scale of {scale:.4f}; the "
            f"inferred scale {ratio:.4f} is not a power of two"
        )
    return float(scale)


class LevelMapper:
    """Assign each box to a pyramid level by its area.

    ``lvl = floor(lvl0 + log2(sqrt(area) / canonical_scale) + eps)``, clamped
    to ``[k_min, k_max]``.
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor], image_shapes: List[Tuple[int, int]]) -> Tensor:
        target_lvls = []
        for boxlist in boxlists:
            sqrt_area = tensorplay.sqrt((boxlist[:, 3] - boxlist[:, 1]) * (boxlist[:, 2] - boxlist[:, 0]))
            lvls = tensorplay.floor(self.lvl0 + tensorplay.log2(sqrt_area / self.s0 + self.eps))
            lvls = tensorplay.clamp(lvls, min=self.k_min, max=self.k_max)
            target_lvls.append(lvls.to(tensorplay.int64))
        return tensorplay.cat(target_lvls, dim=0)


class MultiScaleRoIAlign(tensorplay.nn.Module):
    """Pool region proposals from several pyramid levels with ROI align.

    Args:
        featmap_names: names of the pyramid levels to use, e.g.
            ``["0", "1", "2", "3"]``
        output_size: pooled spatial size (int or pair)
        sampling_ratio: samples per bin in each pooled output; ``<= 0`` means
            adaptive density
        canonical_scale: reference box side for level assignment (default 224)
        canonical_level: level at which a box of ``canonical_scale`` maps
            exactly (default 4)
    """

    __annotations__ = {"featmap_names": List[str]}

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int, int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def convert_to_roi_format(self, boxes: List[Tensor], dtype, device) -> Tensor:
        concat_boxes = tensorplay.cat(boxes, dim=0)
        ids = tensorplay.cat(
            [
                tensorplay.full((b.shape[0], 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        return tensorplay.cat([ids, concat_boxes], dim=1)

    def forward(
        self,
        x: dict,
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        _log_api_usage_once(self)
        dtypes = {t.dtype for t in x.values()}
        if len(dtypes) != 1:
            raise TypeError("All features must have the same dtype")
        devices = {t.device for t in x.values()}
        if len(devices) != 1:
            raise TypeError("All features must be on the same device")
        for name in self.featmap_names:
            if name not in x:
                raise ValueError(f"Feature map {name} not found in x")

        num_levels = len(self.featmap_names)
        num_boxes = sum(b.shape[0] for b in boxes)
        dtype = next(iter(dtypes))
        device = next(iter(devices))
        channels = x[self.featmap_names[0]].shape[1]
        res = tensorplay.zeros(
            (num_boxes, channels, self.output_size[0], self.output_size[1]),
            dtype=dtype,
            device=device,
        )
        if num_boxes == 0:
            return res

        mapper = LevelMapper(
            k_min=0,
            k_max=num_levels - 1,
            canonical_scale=self.canonical_scale,
            canonical_level=self.canonical_level,
        )
        lvls = mapper(boxes, image_shapes)
        rois = self.convert_to_roi_format(boxes, dtype, device)

        for lvl, feat_name in enumerate(self.featmap_names):
            scale = _infer_scale(x[feat_name].shape[-2:], image_shapes[0])
            assigned = tensorplay.nonzero(lvls == lvl).reshape(-1)
            if assigned.numel() == 0:
                continue
            rois_lvl = rois[assigned]
            pooled = roi_align(
                x[feat_name], rois_lvl, self.output_size, scale, self.sampling_ratio, aligned=False
            )
            res[assigned] = pooled
        return res

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(featmap_names={self.featmap_names}"
        s += f", output_size={self.output_size}"
        s += f", sampling_ratio={self.sampling_ratio}"
        if self.canonical_scale != 224:
            s += f", canonical_scale={self.canonical_scale}"
        if self.canonical_level != 4:
            s += f", canonical_level={self.canonical_level}"
        return s + ")"
