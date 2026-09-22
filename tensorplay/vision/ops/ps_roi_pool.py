"""Position-sensitive ROI average pooling (R-FCN style)."""
from typing import List, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once, _make_ntuple
from .roi_align import check_roi_boxes_shape, convert_boxes_to_roi_format


def ps_roi_pool(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: int,
    spatial_scale: float = 1.0,
) -> Tensor:
    """Position-sensitive ROI pooling: average pooling over quantized ROI bins
    where output bin ``(ph, pw)`` of output channel ``c`` reads input channel
    ``(c * output_size + ph) * output_size + pw``."""
    _log_api_usage_once(ps_roi_pool)
    check_roi_boxes_shape(boxes)
    rois = boxes if isinstance(boxes, Tensor) else convert_boxes_to_roi_format(boxes)
    output_size = _make_ntuple(output_size, 2)
    if rois.dtype != input.dtype:
        rois = rois.to(dtype=input.dtype)
    if input.shape[1] % (output_size[0] * output_size[1]) != 0:
        raise ValueError(
            "input channels must be divisible by output_size * output_size"
        )
    return tensorplay.functional.ps_roi_pool(
        input, rois, spatial_scale, output_size[0], output_size[1]
    )


class PSRoIPool(tensorplay.nn.Module):
    """Module wrapper around :func:`ps_roi_pool`."""

    def __init__(self, output_size: int, spatial_scale: float):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return ps_roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_size={self.output_size}"
        tmpstr += f", spatial_scale={self.spatial_scale}"
        tmpstr += ")"
        return tmpstr
