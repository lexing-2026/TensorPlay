"""Position-sensitive ROI align (R-FCN style) with fixed half-pixel alignment."""
from typing import List, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once, _make_ntuple
from .roi_align import check_roi_boxes_shape, convert_boxes_to_roi_format


def ps_roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
) -> Tensor:
    """Position-sensitive ROI align: the input channel axis must be laid out as
    ``(out_channels, output_size, output_size)`` groups; each output bin
    averages bilinear samples of exactly one input channel. ROI coordinates
    always use the half-pixel offset."""
    _log_api_usage_once(ps_roi_align)
    check_roi_boxes_shape(boxes)
    rois = boxes if isinstance(boxes, Tensor) else convert_boxes_to_roi_format(boxes)
    output_size = _make_ntuple(output_size, 2)
    if rois.dtype != input.dtype:
        rois = rois.to(dtype=input.dtype)
    if input.shape[1] % (output_size[0] * output_size[1]) != 0:
        raise ValueError(
            "input channels must be divisible by output_size * output_size"
        )
    return tensorplay.functional.ps_roi_align(
        input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio
    )


class PSRoIAlign(tensorplay.nn.Module):
    """Module wrapper around :func:`ps_roi_align`."""

    def __init__(self, output_size: int, spatial_scale: float, sampling_ratio: int):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return ps_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_size={self.output_size}"
        tmpstr += f", spatial_scale={self.spatial_scale}"
        tmpstr += f", sampling_ratio={self.sampling_ratio}"
        tmpstr += ")"
        return tmpstr
