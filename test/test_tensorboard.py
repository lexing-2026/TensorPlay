"""Tests for summary logging backed by the tensorboard event-file stack."""

import importlib
import io
import sys

import numpy as np
import pytest

import tensorplay as tp
from tensorplay.utils.tensorboard import FileWriter, SummaryWriter

PILImage = pytest.importorskip("PIL.Image")

from tensorboard.backend.event_processing import event_accumulator as _ea
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


def _canonical_read(run_dir: str) -> _ea.EventAccumulator:
    """Read a run back through TensorBoard's own accumulator."""
    acc = _ea.EventAccumulator(
        run_dir,
        size_guidance={
            _ea.SCALARS: 0,
            _ea.TENSORS: 0,
            _ea.IMAGES: 0,
            _ea.HISTOGRAMS: 0,
            _ea.AUDIO: 0,
            _ea.GRAPH: 0,
        },
    )
    acc.Reload()
    return acc


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def test_add_scalar_old_style_default(tmp_path):
    run = str(tmp_path / "job")
    with SummaryWriter(run, max_queue=1) as writer:
        writer.add_scalar("train/loss", 1.0, 0)
        writer.add_scalar("train/loss", 0.5, 1)
        writer.add_scalar("val/acc", 0.9, 7)

    acc = _canonical_read(run)
    assert "train/loss" in acc.Tags()["scalars"]
    points = acc.Scalars("train/loss")
    assert [p.value for p in points] == pytest.approx([1.0, 0.5])
    assert [p.step for p in points] == [0, 1]
    assert acc.Scalars("val/acc")[0].value == pytest.approx(0.9)
    # old style carries no plugin metadata: it stays out of the tensor stream
    assert "train/loss" not in acc.Tags()["tensors"]


def test_add_scalar_new_style_tensor_summary(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_scalar("x", 2.5, 3, new_style=True)
    writer.add_scalar("xd", 2.5, 3, new_style=True, double_precision=True)
    writer.close()
    acc = _canonical_read(run)
    assert "x" in acc.Tags()["tensors"]
    assert acc.SummaryMetadata("x").plugin_data.plugin_name == "scalars"
    assert tensor_util.make_ndarray(acc.Tensors("x")[0].tensor_proto).item() == pytest.approx(2.5)
    assert acc.Tensors("x")[0].step == 3
    assert acc.Tensors("xd")[0].tensor_proto.dtype == 2  # DT_DOUBLE


def test_add_scalar_accepts_tensorplay_tensor(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_scalar("x", tp.Tensor([2.5]), 3)
    writer.close()
    acc = _canonical_read(run)
    assert acc.Scalars("x")[0].value == pytest.approx(2.5)


def test_add_scalar_rejects_non_scalar(tmp_path):
    writer = SummaryWriter(str(tmp_path / "job"), max_queue=1)
    with pytest.raises(AssertionError):
        writer.add_scalar("x", np.array([1.0, 2.0]), 0)
    writer.close()


def test_add_scalars_namespaced_writers(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_scalars("grp", {"a": 1.0, "b": 2.0}, 5)
    writer.close()
    sub_a = tmp_path / "job" / "grp_a"
    sub_b = tmp_path / "job" / "grp_b"
    assert sub_a.is_dir() and sub_b.is_dir()
    acc = _canonical_read(str(sub_a))
    assert acc.Scalars("grp")[0].value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------


def test_add_tensor_roundtrip(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_tensor("tpx", tp.tensor([1.0, 2.0, 3.0]), 1)
    writer.add_tensor("tpi", tp.tensor([[1, 2], [3, 4]]), 2)
    writer.add_tensor("tpc", tp.tensor([1 + 2j, 3 + 4j]), 3)
    writer.add_tensor("tph", tp.tensor([1.0, 2.0]).to(tp.float16), 4)
    writer.close()
    acc = _canonical_read(run)
    assert acc.SummaryMetadata("tpx").plugin_data.plugin_name == "tensor"
    assert tensor_util.make_ndarray(acc.Tensors("tpx")[0].tensor_proto).tolist() == [1.0, 2.0, 3.0]
    assert tensor_util.make_ndarray(acc.Tensors("tpi")[0].tensor_proto).tolist() == [[1, 2], [3, 4]]
    assert tensor_util.make_ndarray(acc.Tensors("tpc")[0].tensor_proto).tolist() == [1 + 2j, 3 + 4j]
    assert tensor_util.make_ndarray(acc.Tensors("tph")[0].tensor_proto).tolist() == [1.0, 2.0]


def test_add_tensor_rejects_oversized(tmp_path):
    writer = SummaryWriter(str(tmp_path / "job"), max_queue=1)
    # 1<<28 float64 = 2 GiB, at the protobuf hard limit
    big = tp.from_numpy(np.zeros((1 << 28), dtype=np.float64))
    with pytest.raises(ValueError):
        writer.add_tensor("big", big, 0)
    writer.close()


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


def test_add_histogram_legacy_roundtrip(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    data = np.random.RandomState(1).normal(size=1000)
    writer.add_histogram("dist/w", data, 1, bins=10)
    writer.add_histogram("dist/const", np.full(7, 3.14), 2, bins=5)
    writer.close()
    acc = _canonical_read(run)
    assert "dist/w" in acc.Tags()["histograms"]
    histo = acc.Histograms("dist/w")[0].histogram_value
    assert histo.num == 1000
    assert sum(histo.bucket) == pytest.approx(1000)
    const = acc.Histograms("dist/const")[0].histogram_value
    assert const.num == 7
    assert const.sum == pytest.approx(3.14 * 7)


def test_add_histogram_raw(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    values = np.arange(20.0)
    counts, limits = np.histogram(values, bins=5)
    writer.add_histogram_raw(
        tag="raw/h",
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=values.dot(values),
        bucket_limits=limits[1:].tolist(),
        bucket_counts=counts.tolist(),
        global_step=0,
    )
    writer.close()
    acc = _canonical_read(run)
    histo = acc.Histograms("raw/h")[0].histogram_value
    assert histo.num == 20
    assert sum(histo.bucket) == pytest.approx(20.0)


def test_add_histogram_raw_length_mismatch(tmp_path):
    writer = SummaryWriter(str(tmp_path / "job"), max_queue=1)
    with pytest.raises(ValueError):
        writer.add_histogram_raw("raw/bad", 0, 1, 1, 0, 0, [1, 2], [1])
    writer.close()


# ---------------------------------------------------------------------------
# Images / figures
# ---------------------------------------------------------------------------


def test_add_image_scaling(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_image("im/u8", np.full((3, 4, 4), 200, np.uint8), 0)
    writer.add_image("im/f01", np.full((3, 4, 4), 0.5, np.float32), 1)
    writer.add_image("im/hwc", np.full((4, 4, 3), 100, np.uint8), 2, dataformats="HWC")
    writer.add_image("im/hw", np.full((4, 4), 10, np.uint8), 3, dataformats="HW")
    writer.close()
    acc = _canonical_read(run)
    records = {r.step: r for r in acc.Images("im/u8") + acc.Images("im/f01")
               + acc.Images("im/hwc") + acc.Images("im/hw")}

    def decoded(step):
        rec = records[step]
        return np.asarray(PILImage.open(io.BytesIO(rec.encoded_image_string)))

    assert records[0].width == 4 and records[0].height == 4
    assert decoded(0).max() == 200
    assert decoded(1).max() == 127  # 0.5 * 255
    assert decoded(2).max() == 100
    # HW grayscale is broadcast to 3 channels
    assert decoded(3).shape[2] == 3 and decoded(3).min() == 10


def test_add_images_batch(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    batch = np.zeros((3, 3, 8, 8), np.uint8)
    batch[1, 0, 0, 0] = 255
    writer.add_images("batch/im", batch, 0)  # default NCHW
    writer.close()
    acc = _canonical_read(run)
    rec = acc.Images("batch/im")[0]
    arr = np.asarray(PILImage.open(io.BytesIO(rec.encoded_image_string)))
    assert (rec.height, rec.width) == (8, 24)  # 3 tiles in one row
    assert arr.max() == 255


def test_add_image_with_boxes(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    img = np.full((3, 12, 12), 200, np.uint8)
    boxes = np.array([[2.0, 2.0, 10.0, 10.0]])
    writer.add_image_with_boxes("box/im", img, boxes, 0, labels=["cat"])
    writer.close()
    acc = _canonical_read(run)
    rec = acc.Images("box/im")[0]
    arr = np.asarray(PILImage.open(io.BytesIO(rec.encoded_image_string)))
    # the box outline is drawn in red on the flat 200-gray image
    assert arr[:, :, 0].max() > 200 or arr[:, :, 1].min() < 200


def test_add_figure_renders_png(tmp_path):
    plt = pytest.importorskip("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_figure("fig/plot", fig, 0)
    writer.close()
    acc = _canonical_read(run)
    rec = acc.Images("fig/plot")[0]
    assert PILImage.open(io.BytesIO(rec.encoded_image_string)).format == "PNG"


# ---------------------------------------------------------------------------
# Audio / video / text
# ---------------------------------------------------------------------------


def test_add_audio(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_audio("a/x", np.sin(np.linspace(0, 100, 200)).astype(np.float32), 1, sample_rate=8000)
    writer.close()
    acc = _canonical_read(run)
    rec = acc.Audio("a/x")[0]
    assert rec.sample_rate == 8000
    assert rec.encoded_audio_string[:4] == b"RIFF"


def test_add_text_summary_tag(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_text("log/notes", "**markdown** supported", 4)
    writer.close()
    acc = _canonical_read(run)
    assert acc.SummaryMetadata("log/notes/text_summary").plugin_data.plugin_name == "text"
    payload = tensor_util.make_ndarray(acc.Tensors("log/notes/text_summary")[0].tensor_proto)
    assert payload.tolist()[0].decode() == "**markdown** supported"


def test_add_video_without_moviepy(tmp_path, capsys):
    pytest.importorskip  # moviepy is optional; absence exercises the warning path
    try:
        import moviepy  # noqa: F401
        pytest.skip("moviepy installed")
    except ImportError:
        pass
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_video("v/x", np.zeros((2, 3, 3, 8, 8), np.uint8), 1)
    writer.close()
    assert "add_video needs package moviepy" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# PR curves / custom scalars / hparams / embedding
# ---------------------------------------------------------------------------


def test_add_pr_curve(tmp_path):
    rng = np.random.RandomState(0)
    labels = rng.randint(2, size=100)
    predictions = rng.rand(100)
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_pr_curve("pr/x", labels, predictions, 1, num_thresholds=11)
    writer.close()
    acc = _canonical_read(run)
    assert acc.SummaryMetadata("pr/x").plugin_data.plugin_name == "pr_curves"
    data = tensor_util.make_ndarray(acc.Tensors("pr/x")[0].tensor_proto)
    assert data.shape == (6, 11)


def test_add_custom_scalars_layout(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_custom_scalars_multilinechart(["s/a", "s/b"])
    writer.close()
    acc = _canonical_read(run)
    assert "custom_scalars__config__" in acc.Tags()["tensors"]


def test_add_hparams_subrun(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_hparams({"lr": 0.1, "name": "exp"}, {"hparam/acc": 0.9}, run_name="hp1")
    writer.close()
    # The loader's compat layer reconstructs the plugin tensors from metadata;
    # decode the plugin payloads the same way the hparams plugin does.
    events = []
    for path in sorted((tmp_path / "job" / "hp1").glob("events.out.tfevents.*")):
        events.extend(EventFileLoader(str(path)).Load())
    from tensorboard.plugins.hparams import plugin_data_pb2 as _hpd

    experiment = None
    start = None
    end = None
    for event in events:
        if not event.HasField("summary"):
            continue
        for value in event.summary.value:
            if not value.HasField("tensor"):
                continue
            payload = _hpd.HParamsPluginData()
            payload.ParseFromString(value.metadata.plugin_data.content)
            if value.tag == "_hparams_/experiment":
                experiment = payload.experiment
            elif value.tag == "_hparams_/session_start_info":
                start = payload.session_start_info
            elif value.tag == "_hparams_/session_end_info":
                end = payload.session_end_info
    assert start is not None and end is not None and experiment is not None
    assert start.hparams["lr"].number_value == pytest.approx(0.1)
    assert start.hparams["name"].string_value == "exp"
    assert end.status == 1  # STATUS_SUCCESS
    assert [h.name for h in experiment.hparam_infos] == ["lr", "name"]
    assert experiment.metric_infos[0].name.tag == "hparam/acc"

    acc = _canonical_read(str(tmp_path / "job" / "hp1"))
    assert acc.Scalars("hparam/acc")[0].value == pytest.approx(0.9)


def test_add_hparams_type_error(tmp_path):
    writer = SummaryWriter(str(tmp_path / "job"), max_queue=1)
    with pytest.raises(TypeError):
        writer.add_hparams(["not", "a", "dict"], {"m": 1.0})
    writer.close()


def test_add_embedding_writes_projector_files(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    mat = np.random.RandomState(0).randn(4, 3).astype(np.float32)
    writer.add_embedding(mat, metadata=["a", "b", "c", "d"], global_step=1)
    writer.close()
    subdir = tmp_path / "job" / "00001" / "default"
    assert (subdir / "tensors.tsv").is_file()
    assert (subdir / "metadata.tsv").is_file()
    assert (tmp_path / "job" / "projector_config.pbtxt").is_file()
    rows = (subdir / "tensors.tsv").read_text().strip().splitlines()
    assert len(rows) == 4 and len(rows[0].split("\t")) == 3


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def test_add_graph_traces_module(tmp_path):
    from tensorplay import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=1)
    writer.add_graph(Net(), tp.rand(2, 4))
    writer.close()
    acc = _canonical_read(run)
    graph_def = acc.Graph()
    node_ops = [n.op for n in graph_def.node]
    assert len(node_ops) > 3
    assert "IO Node" in node_ops and "Parameter" in node_ops


def test_add_graph_requires_inputs(tmp_path):
    from tensorplay import nn

    class Net(nn.Module):
        def forward(self, x):
            return x

    writer = SummaryWriter(str(tmp_path / "job"), max_queue=1)
    with pytest.raises(ValueError):
        writer.add_graph(Net())
    writer.close()


def test_file_writer_events_and_summary(tmp_path):
    from tensorboard.compat.proto import event_pb2
    from tensorboard.compat.proto.summary_pb2 import Summary

    fw = FileWriter(str(tmp_path / "job"))
    fw.add_event(event_pb2.Event(step=1, file_version="brain.Event:2"))
    fw.add_summary(Summary(value=[Summary.Value(tag="t", simple_value=1.5)]), 3)
    fw.flush()
    fw.close()

    def all_events():
        events = []
        for path in sorted((tmp_path / "job").glob("events.out.tfevents.*")):
            events.extend(EventFileLoader(str(path)).Load())
        return events

    events = all_events()
    assert any(e.file_version == "brain.Event:2" for e in events)
    assert any(e.step == 3 for e in events)
    # reopen starts a second event file
    fw.reopen()
    fw.add_event(event_pb2.Event(step=2, file_version="brain.Event:2"))
    fw.flush()
    fw.close()
    event_files = list((tmp_path / "job").glob("events.out.tfevents.*"))
    assert len(event_files) >= 2
    assert any(e.step == 2 for e in all_events())


# ---------------------------------------------------------------------------
# Lifecycle / defaults / dependency handling
# ---------------------------------------------------------------------------


def test_purge_step_writes_session_start(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, purge_step=5, max_queue=1)
    writer.add_scalar("x", 1.0, 6)
    writer.close()
    events = []
    for path in sorted((tmp_path / "job").glob("events.out.tfevents.*")):
        events.extend(EventFileLoader(str(path)).Load())
    assert any(
        e.file_version == "brain.Event:2" and e.step == 5 for e in events
    )
    assert any(
        e.session_log.status == e.session_log.SessionStatus.START for e in events
    )


def test_default_log_dir_created_with_comment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writer = SummaryWriter(comment="demo")
    try:
        writer.add_scalar("m", 1.0, 0)
        writer.flush()
    finally:
        writer.close()
    runs_dir = tmp_path / "runs"
    assert runs_dir.is_dir()
    (subdir,) = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert subdir.name.endswith("demo")
    assert list(subdir.glob("events.out.tfevents.*"))


def test_get_logdir(tmp_path):
    writer = SummaryWriter(str(tmp_path / "explicit"))
    assert writer.get_logdir() == str(tmp_path / "explicit")
    fw = writer._get_file_writer()
    assert fw.get_logdir().startswith(str(tmp_path / "explicit"))
    writer.close()


def test_flush_and_recreate_after_close(tmp_path):
    run = str(tmp_path / "job")
    writer = SummaryWriter(run, max_queue=100, flush_secs=3600)
    writer.add_scalar("a", 1.0, 0)
    writer.flush()
    acc = _canonical_read(run)
    assert len(acc.Scalars("a")) == 1
    writer.close()
    writer.add_scalar("a", 2.0, 1)  # recreates the file writer
    writer.flush()
    writer.close()
    acc = _canonical_read(run)
    assert [p.value for p in acc.Scalars("a")] == pytest.approx([1.0, 2.0])


def test_missing_tensorboard_package_raises(monkeypatch):
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "tensorboard" or name.startswith("tensorplay.utils.tensorboard")
    }
    for name in saved:
        sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "tensorboard", None)
    try:
        with pytest.raises(ImportError):
            importlib.import_module("tensorplay.utils.tensorboard")
    finally:
        for name in list(sys.modules):
            if name == "tensorboard" or name.startswith("tensorplay.utils.tensorboard"):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
    # the module imports fine again once the dependency is available
    importlib.import_module("tensorplay.utils.tensorboard")


def test_old_tensorboard_version_rejected(monkeypatch):
    import tensorboard

    saved_version = tensorboard.__version__
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("tensorplay.utils.tensorboard")
    }
    for name in saved:
        sys.modules.pop(name, None)
    monkeypatch.setattr(tensorboard, "__version__", "1.0.0")
    try:
        with pytest.raises(ImportError, match="1.15"):
            importlib.import_module("tensorplay.utils.tensorboard")
    finally:
        monkeypatch.setattr(tensorboard, "__version__", saved_version)
        for name in list(sys.modules):
            if name.startswith("tensorplay.utils.tensorboard"):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_utils_lazy_exports():
    import tensorplay.utils as U
    from tensorplay.utils import tensorboard as tb_mod

    assert U.tensorboard is tb_mod
    assert tb_mod.SummaryWriter.__name__ == "SummaryWriter"
    assert not hasattr(U, "board")  # no bundled dashboard; use stock tooling
