"""Tests for AIGC / generative-model metrics (FID, KID, Inception Score)."""

import pytest

import tensorplay as tp
from tensorplay import nn
from tensorplay.metrics import (
    FrechetInceptionDistance,
    InceptionScore,
    KernelInceptionDistance,
)
from tensorplay.metrics.image._inception import NoTrainInceptionV3


def _imgs_lo() -> tp.Tensor:
    return tp.randint(0, 100, (6, 3, 299, 299), dtype=tp.uint8)


def _imgs_hi() -> tp.Tensor:
    return tp.randint(150, 255, (6, 3, 299, 299), dtype=tp.uint8)


def test_inception_extractor_feature_shapes():
    inc = NoTrainInceptionV3(["64", "192", "768", "2048", "logits_unbiased", "logits"])
    out = inc(_imgs_lo())
    assert len(out) == 6
    assert out[0].shape == (6, 64)
    assert out[1].shape == (6, 192)
    assert out[2].shape == (6, 768)
    assert out[3].shape == (6, 2048)
    assert out[4].shape == (6, 1000)
    assert out[5].shape == (6, 1000)


def test_inception_extractor_invalid_feature():
    with pytest.raises(ValueError, match="Unknown feature layer"):
        NoTrainInceptionV3("bad_layer")


def test_fid_basic():
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(_imgs_lo(), real=True)
    fid.update(_imgs_hi(), real=False)
    value = fid.compute().item()
    assert value > 0
    assert isinstance(value, float)


def test_fid_requires_two_samples():
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(_imgs_lo()[:1], real=True)
    fid.update(_imgs_hi()[:1], real=False)
    with pytest.raises(RuntimeError, match="More than one sample"):
        fid.compute()


def test_fid_invalid_feature():
    with pytest.raises(ValueError, match="Integer input to argument `feature`"):
        FrechetInceptionDistance(feature=123)


def test_fid_invalid_normalize():
    with pytest.raises(ValueError, match="normalize"):
        FrechetInceptionDistance(normalize="yes")  # type: ignore[arg-type]


def test_fid_reset_real_features():
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False)
    fid.update(_imgs_lo(), real=True)
    fid.update(_imgs_hi(), real=False)
    _ = fid.compute()
    fid.reset()
    # real features should be preserved after reset
    assert fid.real_features_num_samples.item() == 6


def test_fid_normalize_true():
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    x1 = tp.rand(4, 3, 299, 299, dtype=tp.float32)
    x2 = tp.rand(4, 3, 299, 299, dtype=tp.float32) + 0.5
    fid.update(x1, real=True)
    fid.update(x2, real=False)
    assert fid.compute().item() > 0


def test_fid_custom_feature_extractor():
    class _Feat(nn.Module):
        num_features = 16

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 16)

        def forward(self, x: tp.Tensor) -> tp.Tensor:
            return self.proj(x.mean(dim=(2, 3)))

    fid = FrechetInceptionDistance(feature=_Feat(), normalize=True)
    x1 = tp.rand(4, 3, 32, 32)
    x2 = tp.rand(4, 3, 32, 32)
    fid.update(x1, real=True)
    fid.update(x2, real=False)
    assert fid.compute().item() > -1e-4


def test_fid_low_feature_layers():
    for layer in (64, 192, 768):
        fid = FrechetInceptionDistance(feature=layer)
        fid.update(_imgs_lo(), real=True)
        fid.update(_imgs_hi(), real=False)
        assert fid.compute().item() > 0


def test_kid_basic():
    kid = KernelInceptionDistance(subsets=5, subset_size=4)
    kid.update(_imgs_lo(), real=True)
    kid.update(_imgs_hi(), real=False)
    mean, std = kid.compute()
    assert mean.item() > 0
    assert std.item() >= 0


def test_kid_invalid_feature():
    with pytest.raises(ValueError, match="Integer input to argument `feature`"):
        KernelInceptionDistance(feature="bad")


def test_kid_invalid_args():
    with pytest.raises(ValueError, match="subsets"):
        KernelInceptionDistance(subsets=0)
    with pytest.raises(ValueError, match="subset_size"):
        KernelInceptionDistance(subset_size=-1)
    with pytest.raises(ValueError, match="degree"):
        KernelInceptionDistance(degree=0)
    with pytest.raises(ValueError, match="gamma"):
        KernelInceptionDistance(gamma=0.0)
    with pytest.raises(ValueError, match="coef"):
        KernelInceptionDistance(coef=-1.0)


def test_kid_subset_size_too_large():
    kid = KernelInceptionDistance(subsets=2, subset_size=100)
    kid.update(_imgs_lo(), real=True)
    kid.update(_imgs_hi(), real=False)
    with pytest.raises(ValueError, match="subset_size"):
        kid.compute()


def test_kid_poly_functions():
    from tensorplay.metrics.image.kid import poly_kernel, poly_mmd

    f1 = tp.rand(4, 8)
    f2 = tp.rand(4, 8)
    k = poly_kernel(f1, f2, degree=3, gamma=0.1, coef=1.0)
    assert k.shape == (4, 4)
    mmd = poly_mmd(f1, f2, degree=3, gamma=0.1, coef=1.0)
    assert mmd.ndim == 0


def test_inception_score_basic():
    is_ = InceptionScore(splits=3)
    is_.update(_imgs_lo())
    mean, std = is_.compute()
    assert mean.item() >= 1.0
    assert std.item() >= 0


def test_inception_score_invalid_feature():
    with pytest.raises(ValueError, match="Integer input to argument `feature`"):
        InceptionScore(feature="bad")


def test_inception_score_normalize():
    is_ = InceptionScore(splits=2, normalize=True)
    is_.update(tp.rand(6, 3, 299, 299, dtype=tp.float32))
    mean, std = is_.compute()
    assert mean.item() >= 1.0
    assert std.item() >= 0

def _img_pair(n: int = 2, size: int = 64) -> tuple:
    img1 = tp.rand(n, 3, size, size) * 2 - 1
    img2 = tp.rand(n, 3, size, size) * 2 - 1
    return img1, img2


def test_lpips_functional_squeeze():
    from tensorplay.metrics.functional.image import learned_perceptual_image_patch_similarity

    img1, img2 = _img_pair()
    value = learned_perceptual_image_patch_similarity(img1, img2, net_type="squeeze")
    assert value.item() >= 0
    assert value.item() < 1.0


def test_lpips_class_squeeze():
    from tensorplay.metrics import LearnedPerceptualImagePatchSimilarity

    img1, img2 = _img_pair()
    metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
    assert metric(img1, img2).item() >= 0
    metric.reset()
    assert len(metric.all_scores) == 0


def test_lpips_normalize():
    from tensorplay.metrics import LearnedPerceptualImagePatchSimilarity

    img1 = tp.rand(2, 3, 64, 64)
    img2 = tp.rand(2, 3, 64, 64)
    metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    assert metric(img1, img2).item() >= 0


def test_lpips_reduction():
    from tensorplay.metrics import LearnedPerceptualImagePatchSimilarity

    img1, img2 = _img_pair()
    metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", reduction="none")
    out = metric(img1, img2)
    assert out.shape == (2,)


def test_lpips_invalid_net_type():
    from tensorplay.metrics import LearnedPerceptualImagePatchSimilarity

    with pytest.raises(ValueError, match="net_type"):
        LearnedPerceptualImagePatchSimilarity(net_type="resnet")


def test_lpips_invalid_reduction():
    from tensorplay.metrics import LearnedPerceptualImagePatchSimilarity

    with pytest.raises(ValueError, match="reduction"):
        LearnedPerceptualImagePatchSimilarity(reduction="median")


def test_lpips_invalid_normalize():
    from tensorplay.metrics import LearnedPerceptualImagePatchSimilarity

    with pytest.raises(ValueError, match="normalize"):
        LearnedPerceptualImagePatchSimilarity(normalize="yes")  # type: ignore[arg-type]


def test_lpips_invalid_input():
    from tensorplay.metrics.functional.image import learned_perceptual_image_patch_similarity

    img1 = tp.rand(2, 3, 64, 64) - 2.0  # min < -1 violates the [-1, 1] contract
    img2 = tp.rand(2, 3, 64, 64)
    with pytest.raises(ValueError, match="Expected both input"):
        learned_perceptual_image_patch_similarity(img1, img2, net_type="squeeze")


def test_lpips_functional_alex():
    from tensorplay.metrics.functional.image import learned_perceptual_image_patch_similarity

    img1, img2 = _img_pair()
    value = learned_perceptual_image_patch_similarity(img1, img2, net_type="alex")
    assert value.item() >= 0
    assert value.item() < 1.0


def test_lpips_functional_vgg():
    from tensorplay.metrics.functional.image import learned_perceptual_image_patch_similarity

    img1, img2 = _img_pair()
    value = learned_perceptual_image_patch_similarity(img1, img2, net_type="vgg")
    assert value.item() >= 0
    assert value.item() < 1.0
