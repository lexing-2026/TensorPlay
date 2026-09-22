from collections import OrderedDict
from typing import Optional

from tensorplay import Tensor
from tensorplay.nn import Module, functional as F

from ...utils import _log_api_usage_once


class _SimpleSegmentationModel(Module):
    """Backbone plus dense classifier, resizing predictions to the input size."""

    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: Module, classifier: Module, aux_classifier: Optional[Module] = None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result
