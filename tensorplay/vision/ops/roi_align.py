"""ROI align: bilinearly sampled, grid-averaged pooling over floating ROIs."""
from typing import List, Tuple, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once, _make_ntuple


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    """Pack per-image ``[L, 4]`` boxes into a single ``[K, 5]`` ROI tensor
    whose first column carries the image (batch) index."""
    concat_boxes = tensorplay.cat(boxes, dim=0)
    batch_indices = []
    for i, b in enumerate(boxes):
        batch_indices.append(
            tensorplay.full((b.shape[0], 1), i, dtype=concat_boxes.dtype, device=concat_boxes.device)
        )
    rois = tensorplay.cat(batch_indices, dim=0)
    return tensorplay.cat([rois, concat_boxes], dim=1)


def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]) -> None:
    if isinstance(boxes, (list, tuple)):
        for b in boxes:
            if b.dim() != 2 or b.shape[-1] != 4:
                raise ValueError("The shape of the box in the box list must be (K, 4)!")
    else:
        if boxes.dim() != 2 or boxes.shape[-1] != 5:
            raise ValueError("roi boxes must be of shape (K, 5)!")


def roi_align(
    input: Tensor,
    boxes: Union[Tensor, List[Tensor]],
    output_size: Union[int, List[int], Tuple[int, int]],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    """Pool each ROI into ``output_size`` by averaging ``sampling_ratio`` x
    ``sampling_ratio`` bilinear samples per bin (adaptive density when
    ``sampling_ratio <= 0``). With ``aligned=True`` ROI coordinates are shifted
    by half a pixel so integer coordinates land on pixel centers."""
    _log_api_usage_once(roi_align)
    check_roi_boxes_shape(boxes)
    rois = boxes if isinstance(boxes, Tensor) else convert_boxes_to_roi_format(boxes)
    output_size = _make_ntuple(output_size, 2)
    if rois.dtype != input.dtype:
        rois = rois.to(dtype=input.dtype)
    return tensorplay.functional.roi_align(
        input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned
    )


class RoIAlign(tensorplay.nn.Module):
    """Module wrapper around :func:`roi_align`."""

    def __init__(
        self,
        output_size: Union[int, List[int], Tuple[int, int]],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Union[Tensor, List[Tensor]]) -> Tensor:
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_size={self.output_size}"
        tmpstr += f", spatial_scale={self.spatial_scale}"
        tmpstr += f", sampling_ratio={self.sampling_ratio}"
        tmpstr += f", aligned={self.aligned}"
        tmpstr += ")"
        return tmpstr
