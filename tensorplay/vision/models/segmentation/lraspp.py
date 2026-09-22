from collections import OrderedDict
from functools import partial
from typing import Any, Optional

import tensorplay.nn as nn
from tensorplay import Tensor
from tensorplay.nn import Module
from tensorplay.nn import functional as F

from ...transforms._presets import SemanticSegmentation
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import MobileNet_V3_Large_Weights, MobileNetV3, mobilenet_v3_large


__all__ = ["LRASPP", "LRASPP_MobileNet_V3_Large_Weights", "lraspp_mobilenet_v3_large"]


class LRASPP(Module):
    """Lite R-ASPP: a single global-pooling branch combined with a low-level
    feature map for compact semantic segmentation.

    Args:
        backbone (Module): maps the input image to feature maps, returning a
            dict with the keys ``"high"`` (high level) and ``"low"`` (low level).
        low_channels (int): number of channels of the low level features.
        high_channels (int): number of channels of the high level features.
        num_classes (int): number of output classes (including background).
        inter_channels (int): number of channels for intermediate computations.
    """

    def __init__(
        self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)
        result = OrderedDict()
        result["out"] = out
        return result


class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_classifier(low) + self.high_classifier(x)


def _lraspp_mobilenetv3(backbone: MobileNetV3, num_classes: int) -> LRASPP:
    backbone = backbone.features
    # Stage boundaries are the strided blocks; the first and last blocks are
    # always included as C0 and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # C2, output_stride 8
    high_pos = stage_indices[-1]  # C5, output_stride 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})
    return LRASPP(backbone, low_channels, high_channels, num_classes)


class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            "num_params": 3221538,
            "categories": _VOC_CATEGORIES,
            "min_size": (1, 1),
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 57.9,
                    "pixel_acc": 91.2,
                }
            },
            "_ops": 2.086,
            "_file_size": 12.49,
            "_docs": """
                These weights were trained on a subset of COCO, using only the 20 categories that are present in the
                Pascal VOC dataset.
            """,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


@register_model()
@handle_legacy_interface(
    weights=("pretrained", LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def lraspp_mobilenet_v3_large(
    *,
    weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> LRASPP:
    """Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    Args:
        weights (:class:`~tensorplay.vision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~tensorplay.vision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        weights_backbone (:class:`~tensorplay.vision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``tensorplay.vision.models.segmentation.lraspp.LRASPP``
            base class.
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")
    weights = LRASPP_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone, num_classes)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model
