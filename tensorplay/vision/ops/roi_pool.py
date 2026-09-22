"""ROI max pooling over quantized ROI bins."""
from typing import List, Tuple, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once, _make_ntuple
from .roi_align import check_roi_boxes_shape, convert_boxes_to_roi_format


def roi_pool(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: Union[int, List[int], Tuple[int, int]],
    spatial_scale: float = 1.0,
) -> Tensor:
    """Max-pool each ROI into ``output_size`` bins; ROI coordinates are scaled
    and rounded to integers before the bin geometry is computed."""
    _log_api_usage_once(roi_pool)
    check_roi_boxes_shape(boxes)
    rois = boxes if isinstance(boxes, Tensor) else convert_boxes_to_roi_format(boxes)
    output_size = _make_ntuple(output_size, 2)
    if rois.dtype != input.dtype:
        rois = rois.to(dtype=input.dtype)
    return tensorplay.functional.roi_pool(input, rois, spatial_scale, output_size[0], output_size[1])


class RoIPool(tensorplay.nn.Module):
    """Module wrapper around :func:`roi_pool`."""

    def __init__(
        self,
        output_size: Union[int, List[int], Tuple[int, int]],
        spatial_scale: float,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Union[Tensor, List[Tensor]]) -> Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_size={self.output_size}"
        tmpstr += f", spatial_scale={self.spatial_scale}"
        tmpstr += ")"
        return tmpstr
