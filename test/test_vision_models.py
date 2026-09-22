"""Forward smoke tests for the vision model builders (CPU, random weights)."""
import collections

import pytest

import tensorplay
from tensorplay import vision


def _seg_builder(name):
    return getattr(vision.models.segmentation, name)


def _video_builder(name):
    return getattr(vision.models.video, name)


@pytest.mark.parametrize(
    "name",
    ["fcn_resnet50", "fcn_resnet101", "deeplabv3_resnet50", "deeplabv3_resnet101", "deeplabv3_mobilenet_v3_large"],
)
def test_segmentation_models(name):
    model = _seg_builder(name)().eval()
    out = model(tensorplay.randn(1, 3, 128, 128))
    assert isinstance(out, collections.OrderedDict)
    assert tuple(out["out"].shape) == (1, 21, 128, 128)


def test_lraspp_mobilenet_v3_large():
    model = _seg_builder("lraspp_mobilenet_v3_large")().eval()
    out = model(tensorplay.randn(1, 3, 128, 128))
    assert tuple(out["out"].shape) == (1, 21, 128, 128)


@pytest.mark.parametrize("name", ["r3d_18", "mc3_18", "r2plus1d_18"])
def test_video_resnet_models(name):
    model = _video_builder(name)().eval()
    out = model(tensorplay.randn(1, 3, 4, 64, 64))
    assert tuple(out.shape) == (1, 400)


def test_s3d():
    model = _video_builder("s3d")().eval()
    out = model(tensorplay.randn(1, 3, 14, 224, 224))
    assert tuple(out.shape) == (1, 400)


@pytest.mark.parametrize("name", ["mvit_v1_b", "mvit_v2_s"])
def test_mvit_models(name):
    # Fixed-size positional encoding: the builder's spatial/temporal sizes must
    # match the forwarded clip.
    model = _video_builder(name)(spatial_size=(56, 56), temporal_size=4)().eval()
    out = model(tensorplay.randn(1, 3, 4, 56, 56))
    assert tuple(out.shape) == (1, 400)


@pytest.mark.parametrize("name", ["swin3d_t", "swin3d_s", "swin3d_b"])
def test_swin3d_models(name):
    model = _video_builder(name)().eval()
    out = model(tensorplay.randn(1, 3, 4, 96, 96))
    assert tuple(out.shape) == (1, 400)


@pytest.mark.parametrize("name", ["raft_large", "raft_small"])
def test_raft_models(name):
    model = getattr(vision.models, name)().eval()
    img = tensorplay.randn(1, 3, 256, 256)
    with tensorplay.no_grad():
        flows = model(img, img, num_flow_updates=1)
    assert len(flows) == 1
    assert tuple(flows[0].shape) == (1, 2, 256, 256)


def test_mobilenet_v3_large_dilated():
    model = vision.models.mobilenet_v3_large(dilated=True)
    out = model(tensorplay.randn(1, 3, 64, 64))
    assert tuple(out.shape) == (1, 1000)


@pytest.mark.parametrize(
    "name",
    [
        "fcn_resnet50",
        "lraspp_mobilenet_v3_large",
        "r3d_18",
        "s3d",
        "mvit_v1_b",
        "swin3d_t",
        "raft_large",
        "raft_small",
    ],
)
def test_task_builders_exposed_at_top_level(name):
    assert callable(getattr(vision.models, name))
