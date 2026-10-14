"""Inception v3 feature extractor used by generative model metrics.

The extractor wraps the vision Inception v3 network and returns intermediate
features at configurable layers, mirroring the layer set commonly used for
generative quality metrics: 64, 192, 768, 2048, logits_unbiased and logits.
Input images are expected to be uint8 tensors in ``[0, 255]``; they are
resized to 299x299 and normalized to ``[-1, 1]`` before being fed to the
network. The module is always in evaluation mode.
"""

from typing import List, Optional, Tuple, Union

import tensorplay as tp
from tensorplay import Tensor, nn

from tensorplay.vision.models import Inception_V3_Weights, inception_v3

__all__ = ["NoTrainInceptionV3"]


class NoTrainInceptionV3(nn.Module):
    """Inception v3 feature extractor that never leaves evaluation mode.

    Args:
        features_list: list of feature layer names to return. Supported values
            are ``"64"``, ``"192"``, ``"768"``, ``"2048"``,
            ``"logits_unbiased"`` and ``"logits"``.
        weights_path: optional path to a local state dict for the Inception v3
            network. If ``None`` the standard ImageNet pretrained weights are
            loaded from the vision model cache.

    Raises:
        ValueError: if a requested feature layer is not supported.
    """

    INPUT_IMAGE_SIZE = 299

    _SUPPORTED_FEATURES = ("64", "192", "768", "2048", "logits_unbiased", "logits")

    def __init__(self, features_list: Union[str, List[str]], weights_path: Optional[str] = None) -> None:
        super().__init__()
        if isinstance(features_list, str):
            features_list = [features_list]
        for name in features_list:
            if name not in self._SUPPORTED_FEATURES:
                raise ValueError(
                    f"Unknown feature layer '{name}'. Supported layers are {self._SUPPORTED_FEATURES}."
                )
        self.features_list = list(features_list)
        self._dtype = tp.float32

        if weights_path is not None:
            self.model = inception_v3(weights=None)
            state = tp.load(weights_path)
            self.model.load_state_dict(state)
        else:
            self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.eval()

    def train(self, mode: bool = True) -> "NoTrainInceptionV3":
        """Force the network to always be in evaluation mode."""
        return super().train(False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Extract features from ``x``.

        Args:
            x: tensor of images with dtype uint8 and values in ``[0, 255]``.

        Returns:
            a single tensor when one feature layer is requested, otherwise a
            tuple of tensors in the order of ``features_list``.
        """
        x = x.to(self._dtype) if x.dtype != self._dtype else x
        if x.shape[-2] != self.INPUT_IMAGE_SIZE or x.shape[-1] != self.INPUT_IMAGE_SIZE:
            x = tp.nn.functional.interpolate(
                x,
                size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        x = (x - 128) / 128

        features = {}
        remaining = self.features_list.copy()

        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)

        if "64" in remaining:
            features["64"] = tp.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
            remaining.remove("64")
            if not remaining:
                return self._output(features)

        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)

        if "192" in remaining:
            features["192"] = tp.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
            remaining.remove("192")
            if not remaining:
                return self._output(features)

        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)

        if "768" in remaining:
            features["768"] = tp.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
            remaining.remove("768")
            if not remaining:
                return self._output(features)

        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = self.model.dropout(x)
        x = tp.flatten(x, 1)

        if "2048" in remaining:
            features["2048"] = x
            remaining.remove("2048")
            if not remaining:
                return self._output(features)

        if "logits_unbiased" in remaining:
            features["logits_unbiased"] = x.mm(self.model.fc.weight.T)
            remaining.remove("logits_unbiased")
            if not remaining:
                return self._output(features)

        features["logits"] = self.model.fc(x)
        return self._output(features)

    def _output(self, features: dict) -> Union[Tensor, Tuple[Tensor, ...]]:
        values = tuple(features[name] for name in self.features_list)
        return values[0] if len(values) == 1 else values