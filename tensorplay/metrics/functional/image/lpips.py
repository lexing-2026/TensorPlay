"""Learned Perceptual Image Patch Similarity (LPIPS) functional."""

import os
from typing import List, NamedTuple, Optional, Union

import tensorplay as tp
from tensorplay import Tensor, nn
from tensorplay.vision import models as _vision_models

from typing import Literal

__all__ = ["learned_perceptual_image_patch_similarity"]

_weight_map = {
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    "alexnet": "AlexNet_Weights",
    "vgg16": "VGG16_Weights",
}


def _get_tv_model_features(net: str, pretrained: bool = False) -> nn.Sequential:
    """Return the convolutional feature backbone of a vision classification model."""
    if pretrained:
        model_weights = getattr(_vision_models, _weight_map[net])
        model = getattr(_vision_models, net)(weights=model_weights.DEFAULT)
    else:
        model = getattr(_vision_models, net)(weights=None)
    return model.features


class SqueezeNet(nn.Module):
    """SqueezeNet backbone sliced into feature stages."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        pretrained_features = _get_tv_model_features("squeezenet1_1", pretrained)

        self.N_slices = 7
        slices = []
        feature_ranges = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 11), range(11, 12), range(12, 13)]
        for feature_range in feature_ranges:
            seq = nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            slices.append(seq)

        self.slices = nn.ModuleList(slices)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""

        class _SqueezeOutput(NamedTuple):
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor
            relu6: Tensor
            relu7: Tensor

        relus = []
        for slice_ in self.slices:
            x = slice_(x)
            relus.append(x)
        return _SqueezeOutput(*relus)


class Alexnet(nn.Module):
    """AlexNet backbone sliced into feature stages."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        alexnet_pretrained_features = _get_tv_model_features("alexnet", pretrained)

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        class _AlexnetOutputs(NamedTuple):
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor

        return _AlexnetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class Vgg16(nn.Module):
    """VGG16 backbone sliced into feature stages."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        vgg_pretrained_features = _get_tv_model_features("vgg16", pretrained)

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        class _VGGOutputs(NamedTuple):
            relu1_2: Tensor
            relu2_2: Tensor
            relu3_3: Tensor
            relu4_3: Tensor
            relu5_3: Tensor

        return _VGGOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _spatial_average(in_tens: Tensor, keep_dim: bool = True) -> Tensor:
    """Spatially average over the height and width of a tensor."""
    return in_tens.mean([2, 3], keepdim=keep_dim)


def _upsample(in_tens: Tensor, out_hw: tuple[int, ...] = (64, 64)) -> Tensor:
    """Upsample input with bilinear interpolation."""
    return nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)(in_tens)


def _normalize_tensor(in_feat: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize the feature channels."""
    norm_factor = tp.sqrt(eps + tp.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / norm_factor


def _resize_tensor(x: Tensor, size: int = 64) -> Tensor:
    """Resize an image tensor to a square of the given side length."""
    if x.shape[-1] > size and x.shape[-2] > size:
        return tp.nn.functional.interpolate(x, (size, size), mode="area")
    return tp.nn.functional.interpolate(x, (size, size), mode="bilinear", align_corners=False)


class ScalingLayer(nn.Module):
    """Input scaling layer matching the normalization of the pretrained backbone."""

    shift: Tensor
    scale: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", tp.tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", tp.tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp: Tensor) -> Tensor:
        """Scale input."""
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single 1x1 convolution used to pool the perceptual distances."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()

        layers = [nn.Dropout()] if use_dropout else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Process input."""
        return self.model(x)


class _LPIPS(nn.Module):
    """Perceptual similarity module comparing activations of a vision backbone."""

    def __init__(
        self,
        pretrained: bool = True,
        net: Literal["alex", "vgg", "squeeze"] = "alex",
        spatial: bool = False,
        pnet_rand: bool = False,
        pnet_tune: bool = False,
        use_dropout: bool = True,
        model_path: Optional[str] = None,
        eval_mode: bool = True,
        resize: Optional[int] = None,
    ) -> None:
        """Initialize the perceptual loss module.

        Args:
            pretrained: whether the linear pooling layers use pretrained weights.
            net: backbone network to use, one of ``'alex'``, ``'vgg'`` or ``'squeeze'``.
            spatial: whether to keep the spatial dimensions of the distances.
            pnet_rand: whether the backbone uses random (instead of pretrained) weights.
            pnet_tune: whether backpropagation is enabled for the backbone.
            use_dropout: whether dropout layers are added.
            model_path: path to load the pretrained linear weights from.
            eval_mode: whether the network starts in evaluation mode.
            resize: optional size to resize inputs to.
        """
        super().__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.resize = resize
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = Alexnet  # type: ignore[assignment]
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = SqueezeNet  # type: ignore[assignment]
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)  # type: ignore[assignment]

        if pretrained:
            if model_path is None:
                model_path = os.path.join(os.path.dirname(__file__), "lpips_models", f"{net}.pth")

            self.load_state_dict(tp.load(model_path, map_location="cpu"), strict=False)

        if eval_mode:
            self.eval()

        if not self.pnet_tune:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self, in0: Tensor, in1: Tensor, retperlayer: bool = False, normalize: bool = False
    ) -> Union[Tensor, tuple[Tensor, List[Tensor]]]:
        """Compute the perceptual distance between two image sets."""
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # normalize input
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)

        # resize input if needed
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)
            in1_input = _resize_tensor(in1_input, size=self.resize)

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = _normalize_tensor(outs0[kk]), _normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = []
        for kk in range(self.L):
            if self.spatial:
                res.append(_upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:])))
            else:
                res.append(_spatial_average(self.lins[kk](diffs[kk]), keep_dim=True))

        val: Tensor = sum(res)  # type: ignore[assignment]
        if retperlayer:
            return (val, res)
        return val


class _NoTrainLpips(_LPIPS):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainLpips":
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor, normalize: bool) -> bool:
    """Check that input is a valid image tensor."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check  # type: ignore[return-value]


def _lpips_update(img1: Tensor, img2: Tensor, net: nn.Module, normalize: bool) -> Tensor:
    """Update LPIPS with a pair of images."""
    if not (_valid_img(img1, normalize) and _valid_img(img2, normalize)):
        raise ValueError(
            "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
            f" Got input with shape {img1.shape} and {img2.shape} and values in range"
            f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
            f" expected to be in the {[0, 1] if normalize else [-1, 1]} range."
        )
    return net(img1, img2, normalize=normalize).squeeze()


def _lpips_compute(scores: Tensor, reduction: Optional[Literal["sum", "mean", "none"]] = "mean") -> Tensor:
    """Reduce LPIPS scores over the batch."""
    if reduction == "mean":
        return scores.mean()
    if reduction == "sum":
        return scores.sum()
    if reduction == "none" or reduction is None:
        return scores
    raise ValueError(f"Invalid reduction type: {reduction}")


def learned_perceptual_image_patch_similarity(
    img1: Tensor,
    img2: Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
    reduction: Optional[Literal["sum", "mean", "none"]] = "mean",
    normalize: bool = False,
) -> Tensor:
    """Calculate the Learned Perceptual Image Patch Similarity (LPIPS).

    LPIPS computes the similarity between the activations of two image patches
    under a pretrained backbone network. Low values indicate perceptual
    similarity.

    Both inputs are expected to have shape ``(N, 3, H, W)``. The minimum size of
    ``H, W`` depends on the chosen backbone.

    Args:
        img1: first set of images.
        img2: second set of images.
        net_type: backbone network type, one of ``'alex'``, ``'vgg'`` or
            ``'squeeze'``.
        reduction: how to reduce over the batch dimension, one of ``'sum'``,
            ``'mean'``, ``'none'`` or ``None``.
        normalize: if ``False`` inputs are expected in ``[-1, 1]``; if ``True``
            inputs are expected in ``[0, 1]``.

    Returns:
        scalar value with the mean LPIPS score over the batch.
    """
    net = _NoTrainLpips(net=net_type).to(device=img1.device, dtype=img1.dtype)
    loss = _lpips_update(img1, img2, net, normalize)
    return _lpips_compute(loss, reduction)