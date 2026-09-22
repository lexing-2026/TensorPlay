"""Box utilities: overlap metrics, NMS, format conversion and box losses.

All axis-aligned boxes use the ``(x1, y1, x2, y2)`` corner convention unless a
format argument says otherwise. Degenerate boxes (empty intersection) produce
zero intersection area; IoU denominators are left unguarded exactly like the
reference formulas so degenerate inputs surface as inf/nan rather than being
silently reweighted.
"""
import math
from typing import List, Tuple, Union

import tensorplay
from tensorplay import Tensor

from ..utils import _log_api_usage_once


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Greedily suppress overlapping detections.

    Boxes are visited in descending score order (ties keep input order); a box
    is dropped when its IoU with any previously kept box is strictly greater
    than ``iou_threshold``. Returns int64 indices into ``boxes`` in
    descending-score order.
    """
    _log_api_usage_once(nms)
    return tensorplay.functional.nms(boxes, scores, iou_threshold)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """NMS performed independently per class id ``idxs``.

    Class ids are folded into the box coordinates as a per-class offset of
    ``max_coordinate + 1``; translated boxes of different classes can never
    overlap, so a single global suppression pass is equivalent to per-class
    NMS while keeping global score ordering.
    """
    _log_api_usage_once(batched_nms)
    if boxes.numel() == 0:
        return tensorplay.empty((0,), dtype=tensorplay.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """Indices of boxes whose width and height are both >= ``min_size``."""
    _log_api_usage_once(remove_small_boxes)
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return tensorplay.nonzero(keep).reshape(-1)


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """Clip box coordinates into the image: x to [0, width], y to [0, height]."""
    _log_api_usage_once(clip_boxes_to_image)
    height, width = size
    boxes_x = boxes[..., 0::2].clamp(0, width)
    boxes_y = boxes[..., 1::2].clamp(0, height)
    clipped = tensorplay.stack((boxes_x, boxes_y), dim=-1)
    return clipped.reshape(boxes.shape)


def box_area(boxes: Tensor, fmt: str = "xyxy") -> Tensor:
    """Area of each box under ``fmt`` in {"xyxy", "xywh", "cxcywh"}."""
    _log_api_usage_once(box_area)
    boxes = _upcast(boxes)
    if fmt == "xyxy":
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    if fmt in ("xywh", "cxcywh"):
        return boxes[..., 2] * boxes[..., 3]
    raise ValueError(f"Unsupported box format {fmt}.")


def box_iou(boxes1: Tensor, boxes2: Tensor, fmt: str = "xyxy") -> Tensor:
    """Pairwise IoU matrix of shape ``[..., N, M]``.

    Only the axis-aligned formats "xyxy", "xywh" and "cxcywh" are supported;
    rotated formats need polygon-clipping intersection area.
    """
    _log_api_usage_once(box_iou)
    if fmt not in ("xyxy", "xywh", "cxcywh"):
        raise ValueError(f"Unsupported box format {fmt} for box_iou.")
    boxes1 = _upcast(_convert_to_xyxy(boxes1, fmt))
    boxes2 = _upcast(_convert_to_xyxy(boxes2, fmt))
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    lt = tensorplay.maximum(boxes1[..., None, :2], boxes2[..., :2])
    rb = tensorplay.minimum(boxes1[..., None, 2:], boxes2[..., 2:])
    wh = _upcast(rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2 - inter
    return inter / union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """IoU penalized by the empty fraction of the enclosing box."""
    _log_api_usage_once(generalized_box_iou)
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union

    lti = tensorplay.minimum(boxes1[..., None, :2], boxes2[..., :2])
    rbi = tensorplay.maximum(boxes1[..., None, 2:], boxes2[..., 2:])
    whi = _upcast(rbi - lti).clamp(min=0)
    areai = whi[..., 0] * whi[..., 1]
    return iou - (areai - union) / areai


def distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """IoU penalized by the normalized squared distance between box centers."""
    _log_api_usage_once(distance_box_iou)
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    iou = box_iou(boxes1, boxes2)
    lti = tensorplay.minimum(boxes1[..., None, :2], boxes2[..., :2])
    rbi = tensorplay.maximum(boxes1[..., None, 2:], boxes2[..., 2:])
    whi = _upcast(rbi - lti).clamp(min=0)
    diagonal_distance_squared = whi[..., 0] ** 2 + whi[..., 1] ** 2 + eps
    # centers of the two boxes
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
    centers_distance_squared = (cx1[..., None] - cx2) ** 2 + (cy1[..., None] - cy2) ** 2
    return iou - (centers_distance_squared / diagonal_distance_squared)


def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Distance IoU plus an aspect-ratio consistency penalty."""
    _log_api_usage_once(complete_box_iou)
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[..., 2] - boxes1[..., 0]
    h_pred = boxes1[..., 3] - boxes1[..., 1]
    w_gt = boxes2[..., 2] - boxes2[..., 0]
    h_gt = boxes2[..., 3] - boxes2[..., 1]
    v = (4 / (math.pi**2)) * tensorplay.pow(
        tensorplay.atan(w_pred[..., None] / h_pred[..., None]) - tensorplay.atan(w_gt / h_gt), 2
    )
    with tensorplay.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v


def generalized_box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, reduction: str = "none", eps: float = 1e-7
) -> Tensor:
    """``1 - generalized_box_iou`` with an eps-guarded enclosing-box term."""
    _log_api_usage_once(generalized_box_iou_loss)
    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)
    int_x1 = tensorplay.maximum(boxes1[:, 0], boxes2[:, 0])
    int_y1 = tensorplay.maximum(boxes1[:, 1], boxes2[:, 1])
    int_x2 = tensorplay.minimum(boxes1[:, 2], boxes2[:, 2])
    int_y2 = tensorplay.minimum(boxes1[:, 3], boxes2[:, 3])
    intsctk = (int_x2 - int_x1).clamp(min=0) * (int_y2 - int_y1).clamp(min=0)
    unionk = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) + (
        boxes2[:, 2] - boxes2[:, 0]
    ) * (boxes2[:, 3] - boxes2[:, 1]) - intsctk
    area_c = (
        tensorplay.clamp(tensorplay.maximum(boxes1[:, 2], boxes2[:, 2]) - tensorplay.minimum(boxes1[:, 0], boxes2[:, 0]), min=0)
        * tensorplay.clamp(tensorplay.maximum(boxes1[:, 3], boxes2[:, 3]) - tensorplay.minimum(boxes1[:, 1], boxes2[:, 1]), min=0)
    )
    giou = 1 - (intsctk / unionk - (area_c - unionk) / (area_c + eps))
    return _reduce(giou, reduction)


def distance_box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, reduction: str = "none", eps: float = 1e-7
) -> Tensor:
    """``1 - distance_box_iou`` on paired boxes of shape ``[N, 4]``."""
    _log_api_usage_once(distance_box_iou_loss)
    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)
    diou = distance_box_iou(boxes1, boxes2, eps)[..., 0]
    return _reduce(1 - diou, reduction)


def complete_box_iou_loss(
    boxes1: Tensor, boxes2: Tensor, reduction: str = "none", eps: float = 1e-7
) -> Tensor:
    """``1 - complete_box_iou`` on paired boxes of shape ``[N, 4]``."""
    _log_api_usage_once(complete_box_iou_loss)
    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)
    ciou = complete_box_iou(boxes1, boxes2, eps)[..., 0]
    return _reduce(1 - ciou, reduction)


def box_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """Convert boxes between "xyxy", "xywh" and "cxcywh" formats."""
    _log_api_usage_once(box_convert)
    allowed = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed or out_fmt not in allowed:
        raise ValueError(f"Unsupported conversion from {in_fmt} to {out_fmt}.")
    if in_fmt == out_fmt:
        return boxes.clone()
    xyxy = _convert_to_xyxy(boxes, in_fmt)
    if out_fmt == "xyxy":
        return xyxy
    if out_fmt == "xywh":
        return tensorplay.stack(
            (xyxy[..., 0], xyxy[..., 1], xyxy[..., 2] - xyxy[..., 0], xyxy[..., 3] - xyxy[..., 1]),
            dim=-1,
        )
    # cxcywh
    return tensorplay.stack(
        ((xyxy[..., 0] + xyxy[..., 2]) / 2, (xyxy[..., 1] + xyxy[..., 3]) / 2,
         xyxy[..., 2] - xyxy[..., 0], xyxy[..., 3] - xyxy[..., 1]),
        dim=-1,
    )


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Bounding boxes (x1, y1, x2, y2) of each boolean/int mask in ``[N, H, W]``."""
    _log_api_usage_once(masks_to_boxes)
    if masks.numel() == 0:
        return tensorplay.zeros((0, 4), dtype=tensorplay.float32, device=masks.device)
    n, height, width = masks.shape
    m = masks > 0
    x = tensorplay.arange(width, device=masks.device)[None, None, :]
    y = tensorplay.arange(height, device=masks.device)[None, :, None]
    pos_inf = tensorplay.full((1,), float("inf"), device=masks.device, dtype=masks.dtype)
    neg_inf = tensorplay.full((1,), float("-inf"), device=masks.device, dtype=masks.dtype)
    xm = tensorplay.where(m, x, pos_inf)
    ym = tensorplay.where(m, y, pos_inf)
    xm_max = tensorplay.where(m, x, neg_inf)
    ym_max = tensorplay.where(m, y, neg_inf)
    left = xm.flatten(1).min(dim=1).values
    top = ym.flatten(1).min(dim=1).values
    right = xm_max.flatten(1).max(dim=1).values
    bottom = ym_max.flatten(1).max(dim=1).values
    boxes = tensorplay.stack([left, top, right, bottom], dim=1).to(tensorplay.float32)
    empty = m.flatten(1).sum(dim=1) == 0
    return tensorplay.where(empty[:, None], tensorplay.zeros_like(boxes), boxes)


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")


def _upcast(t: Tensor) -> Tensor:
    """Protect half precision inputs from overflow in area arithmetic."""
    if t.dtype in (tensorplay.float16, tensorplay.bfloat16):
        return t.to(tensorplay.float32)
    return t


def _upcast_non_float(t: Tensor) -> Tensor:
    if t.dtype not in (tensorplay.float32, tensorplay.float64):
        return t.to(tensorplay.float32)
    if t.dtype == tensorplay.float16:
        return t.to(tensorplay.float32)
    return t


def _convert_to_xyxy(boxes: Tensor, fmt: str) -> Tensor:
    if fmt == "xyxy":
        return boxes
    if fmt == "xywh":
        return tensorplay.stack(
            (boxes[..., 0], boxes[..., 1], boxes[..., 0] + boxes[..., 2], boxes[..., 1] + boxes[..., 3]),
            dim=-1,
        )
    if fmt == "cxcywh":
        return tensorplay.stack(
            (boxes[..., 0] - boxes[..., 2] / 2, boxes[..., 1] - boxes[..., 3] / 2,
             boxes[..., 0] + boxes[..., 2] / 2, boxes[..., 1] + boxes[..., 3] / 2),
            dim=-1,
        )
    raise ValueError(f"Unsupported box format {fmt}.")


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    lt = tensorplay.maximum(boxes1[..., None, :2], boxes2[..., :2])
    rb = tensorplay.minimum(boxes1[..., None, 2:], boxes2[..., 2:])
    wh = _upcast(rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2 - inter
    return inter, union


def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tuple[Tensor, Tensor]:
    iou = box_iou(boxes1, boxes2)
    lti = tensorplay.minimum(boxes1[..., None, :2], boxes2[..., :2])
    rbi = tensorplay.maximum(boxes1[..., None, 2:], boxes2[..., 2:])
    whi = _upcast(rbi - lti).clamp(min=0)
    diagonal_distance_squared = whi[..., 0] ** 2 + whi[..., 1] ** 2 + eps
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
    centers_distance_squared = (cx1[..., None] - cx2) ** 2 + (cy1[..., None] - cy2) ** 2
    diou = iou - (centers_distance_squared / diagonal_distance_squared)
    return diou, iou


__all__ = [
    "nms",
    "batched_nms",
    "remove_small_boxes",
    "clip_boxes_to_image",
    "box_area",
    "box_convert",
    "box_iou",
    "generalized_box_iou",
    "distance_box_iou",
    "complete_box_iou",
    "generalized_box_iou_loss",
    "distance_box_iou_loss",
    "complete_box_iou_loss",
    "masks_to_boxes",
]