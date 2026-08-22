"""Interoperability: torch .pt archives (zip/stream/tar) and safetensors."""

import io
import pickle
import struct
import sys
import tarfile
from pathlib import Path

import pytest

import tensorplay as tp

from tensorplay.serialization import inspect_checkpoint
from tensorplay import _serialization_torch as st


TORCH_MAGIC_NUMBER = 0x1950A86A20F9469CFC6C


def _raw_bytes(tensor) -> bytes:
    return st._tensor_bytes(tensor)


# ---------------------------------------------------------------------------
# torch zip format (PyTorch >= 1.6)
# ---------------------------------------------------------------------------


def test_torch_zip_round_trip_nested(tmp_path: Path):
    obj = {
        "layer": {"w": tp.arange(6, dtype=tp.float32).reshape((2, 3)), "b": None},
        "meta": [1, 2.5, "x", True],
        "tied_src": tp.full((4,), 2.0, dtype=tp.bfloat16),
    }
    obj["tied_dst"] = obj["tied_src"]

    filename = tmp_path / "model.pt"
    tp.save(obj, filename)
    loaded = tp.load(filename)

    assert isinstance(loaded, dict)
    assert loaded["meta"] == [1, 2.5, "x", True]
    assert loaded["layer"]["b"] is None
    assert list(loaded["layer"]["w"].shape) == [2, 3]
    assert loaded["layer"]["w"].dtype == tp.float32
    # bf16 must round-trip bit-exactly through the raw payload.
    assert _raw_bytes(loaded["tied_src"]) == _raw_bytes(obj["tied_src"])
    # Tied weights stay tied through save/load.
    assert int(loaded["tied_dst"].data_ptr()) == int(loaded["tied_src"].data_ptr())


def test_torch_zip_view_materialized_with_values(tmp_path: Path):
    base = tp.arange(10, dtype=tp.int64)
    view = base[2:]
    filename = tmp_path / "views.pt"
    tp.save({"v": view}, filename, )
    loaded = tp.load(filename)
    assert tp.allclose(loaded["v"].to(tp.int64), view)


def test_torch_zip_buffer_round_trip():
    value = {"w": tp.arange(3, dtype=tp.float64)}
    buffer = io.BytesIO()
    tp.save(value, buffer)

    buffer.seek(0)
    kind = st._sniff_format(buffer.read(4))
    buffer.seek(0)
    assert kind == "torch_zip"
    loaded = tp.load(buffer)
    assert tp.allclose(loaded["w"], value["w"])


def test_describe_torch_zip(tmp_path: Path):
    filename = tmp_path / "d.pt"
    tp.save({"a": tp.zeros((2,), dtype=tp.float32),
             "b": tp.zeros((3,), dtype=tp.int64)}, filename)
    info = inspect_checkpoint(filename)
    assert info["format"] == "torch_zip"
    assert set(info["storages"]) >= {"0", "1"}
    dtypes = {entry["dtype"] for entry in info["storages"].values()}
    assert dtypes == {"float32", "int64"}


# ---------------------------------------------------------------------------
# weights-only safety
# ---------------------------------------------------------------------------


def test_weights_only_rejects_arbitrary_globals(tmp_path: Path):
    class Evil:
        def __reduce__(self):
            import os

            return os.system, ("echo pwned",)

    filename = tmp_path / "evil.pt"
    tp.save({"e": Evil()}, filename)  # our writer pickles arbitrary objects

    with pytest.raises(pickle.UnpicklingError, match="allowlisted"):
        tp.load(filename)


def test_probe_does_not_execute_pickle_globals(tmp_path: Path):
    payload = pickle.dumps(__import__("os").system, protocol=2)
    filename = tmp_path / "global-first.bin"
    filename.write_bytes(payload)
    from tensorplay.serialization import _probe_torch_stream

    assert _probe_torch_stream(str(filename)) is False


# ---------------------------------------------------------------------------
# legacy magic-number stream format
# ---------------------------------------------------------------------------


class _StorageRef:
    def __init__(self, key: str, tensor):
        self.key = key
        self.tensor = tensor


def _storage_class_for(tensor):
    dtype_name = st._dtype_name_of(tensor)
    storage_name = st._STORAGE_NAMES_BY_DTYPE[dtype_name]
    return type(storage_name, (), {
        "__module__": "torch", "__qualname__": storage_name, "__name__": storage_name,
    })


def _build_magic_number_stream(tensors: dict) -> bytes:
    """Synthesize a torch <1.6 stream checkpoint for ``{name: tensor}``."""

    buf = io.BytesIO()
    pickle.dump(TORCH_MAGIC_NUMBER, buf, protocol=2)
    pickle.dump(1001, buf, protocol=2)
    pickle.dump({"protocol_version": 1001, "little_endian": True,
                 "type_sizes": {"short": 2, "int": 4, "long": 8}},
                buf, protocol=2)

    key_of = {id(tensor): str(index) for index, tensor in enumerate(tensors.values())}

    pickler = pickle.Pickler(buf, protocol=2)

    def persistent_id(obj):
        if isinstance(obj, _StorageRef):
            return ("storage", _storage_class_for(obj.tensor), obj.key, "cpu",
                    int(obj.tensor.numel()))
        return None

    def reduce_tensor(tensor):
        shape = [int(dim) for dim in tensor.shape]
        ref = _StorageRef(key_of[id(tensor)], tensor)
        return (st._rebuild_tensor_v2,
                (ref, 0, shape, st._contiguous_stride(shape), False, {}))

    dispatch = dict(pickle.Pickler.dispatch_table)
    dispatch[type(tp.Tensor)] = reduce_tensor
    parameter_cls = getattr(tp.nn, "Parameter", None)
    if parameter_cls is not None and parameter_cls is not tp.Tensor:
        dispatch[parameter_cls] = reduce_tensor
    pickler.dispatch_table = dispatch
    pickler.persistent_id = persistent_id
    pickler.dump(dict(tensors))
    pickle.dump([key_of[id(tensor)] for tensor in tensors.values()], buf, protocol=2)

    for tensor in tensors.values():
        buf.write(struct.pack("<q", int(tensor.numel())))
        buf.write(_raw_bytes(tensor))
    return buf.getvalue()


@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_magic_number_stream_round_trip(tmp_path: Path, dtype):
    value = {"w": tp.arange(5, dtype=dtype)}
    filename = tmp_path / "legacy.pth"
    filename.write_bytes(_build_magic_number_stream(value))

    loaded = tp.load(filename)
    assert loaded["w"].dtype == dtype
    assert tp.allclose(loaded["w"], value["w"])


def test_inspect_reports_torch_stream(tmp_path: Path):
    filename = tmp_path / "legacy.pth"
    filename.write_bytes(_build_magic_number_stream({"w": tp.ones((2,), dtype=tp.float32)}))
    info = inspect_checkpoint(filename)
    assert info["format"] == "torch_stream"


# ---------------------------------------------------------------------------
# ancient tar format
# ---------------------------------------------------------------------------


def test_ancient_tar_round_trip(tmp_path: Path):
    tensors = {"w": tp.arange(4, dtype=tp.float32)}
    specs = [("w", "0", tensors["w"])]

    storages_buf = io.BytesIO()
    pickle.dump(len(specs), storages_buf, protocol=2)
    for _name, key, tensor in specs:
        dtype_name = st._dtype_name_of(tensor)
        pickle.dump((key, "cpu",
                     st._StorageType(st._STORAGE_NAMES_BY_DTYPE[dtype_name])),
                    storages_buf, protocol=2)
        storages_buf.write(struct.pack("<q", int(tensor.numel())))
        storages_buf.write(_raw_bytes(tensor))

    tensors_buf = io.BytesIO()
    pickle.dump(len(specs), tensors_buf, protocol=2)
    for name, key, tensor in specs:
        size = [int(dim) for dim in tensor.shape]
        stride = st._contiguous_stride(size)
        pickle.dump((name, key, "torch.Tensor"), tensors_buf, protocol=2)
        tensors_buf.write(struct.pack("<i", len(size)))
        tensors_buf.write(struct.pack("<i", 0))
        for dim in size:
            tensors_buf.write(struct.pack("<q", dim))
        for step in stride:
            tensors_buf.write(struct.pack("<q", step))
        tensors_buf.write(struct.pack("<q", 0))

    main_buf = io.BytesIO()
    main_pickler = pickle.Pickler(main_buf, protocol=2)

    class _TarRef:
        def __init__(self, name: str):
            self.name = name

    def persistent_id(obj):
        if isinstance(obj, _TarRef):
            return obj.name  # reference into the prebuilt tensors table
        return None

    main_pickler.persistent_id = persistent_id
    main_pickler.dump({"w": _TarRef("w")})

    filename = tmp_path / "ancient.tar"
    with tarfile.open(filename, mode="w:", format=tarfile.PAX_FORMAT) as archive:
        for member_name, data in (("storages", storages_buf.getvalue()),
                                  ("tensors", tensors_buf.getvalue()),
                                  ("pickle", main_buf.getvalue())):
            info = tarfile.TarInfo(name=member_name)
            info.size = len(data)
            with io.BytesIO(data) as payload:
                archive.addfile(info, payload)

    loaded = tp.load(filename)
    assert tp.allclose(loaded["w"], tensors["w"])


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------


def test_safetensors_round_trip(tmp_path: Path):
    value = {
        "f32": tp.arange(4, dtype=tp.float32),
        "bf16": tp.tensor([[1.0, -2.0]], dtype=tp.bfloat16),
        "i64": tp.tensor([-5, 0, 5], dtype=tp.int64),
    }
    filename = tmp_path / "model.safetensors"
    tp.save(value, filename, metadata={"source": "test"})

    loaded = tp.load(filename)
    assert tp.allclose(loaded["f32"], value["f32"])
    assert _raw_bytes(loaded["bf16"]) == _raw_bytes(value["bf16"])
    assert tp.allclose(loaded["i64"], value["i64"])

    info = inspect_checkpoint(filename)
    assert info["format"] == "safetensors"
    assert info["metadata"] == {"source": "test"}
    assert info["tensors"]["bf16"]["dtype"] == "BF16"


def test_safetensors_header_alignment(tmp_path: Path):
    filename = tmp_path / "aligned.safetensors"
    tp.save({"w": tp.ones((2,), dtype=tp.float32)}, filename)
    (header_length,) = struct.unpack("<Q", filename.read_bytes()[:8])
    raw = filename.read_bytes()[8:8 + header_length]
    assert header_length % 8 == 0
    json.loads(raw.rstrip(b" "))


def test_safetensors_requires_flat_mapping(tmp_path: Path):
    filename = tmp_path / "nested.safetensors"
    with pytest.raises(TypeError, match="flat"):
        tp.save({"a": {"b": tp.ones((1,), dtype=tp.float32)}}, filename)


def test_safetensors_mmap(tmp_path: Path):
    value = {"w": tp.arange(64, dtype=tp.float32), "v": tp.ones((8,), dtype=tp.bfloat16)}
    filename = tmp_path / "mmap.safetensors"
    tp.save(value, filename)

    with open(filename, "rb") as handle:
        loaded = tp.load(handle, mmap=True)
    assert tp.allclose(loaded["w"], value["w"])
    assert _raw_bytes(loaded["v"]) == _raw_bytes(value["v"])


def test_safetensors_corrupt_offsets_rejected(tmp_path: Path):
    filename = tmp_path / "corrupt.safetensors"
    tp.save({"w": tp.ones((2,), dtype=tp.float32)}, filename)
    raw = bytearray(filename.read_bytes())
    # Point data_offsets past the end of the file.
    idx = raw.find(b"data_offsets")
    raw[idx + 13] = 0x7F
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        tp.load(bad)


# ---------------------------------------------------------------------------
# cross-validation against real torch when available
# ---------------------------------------------------------------------------


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_real_torch_loads_our_pt(tmp_path: Path):
    import torch

    value = {"w": tp.arange(6, dtype=tp.float32).reshape((2, 3)),
             "tied_a": tp.ones((3,), dtype=tp.bfloat16)}
    value["tied_b"] = value["tied_a"]
    filename = tmp_path / "ours.pt"
    tp.save(value, filename)

    reference = torch.load(str(filename), map_location="cpu", weights_only=True)
    expected = torch.arange(6, dtype=torch.float32).reshape((2, 3))
    assert torch.equal(reference["w"], expected)
    assert reference["w"].dtype == torch.float32
    assert reference["tied_a"].data_ptr() == reference["tied_b"].data_ptr()


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_we_load_real_torch_save(tmp_path: Path):
    import torch

    state_dict = {
        "encoder.weight": torch.randn((4, 4)),
        "encoder.bias": torch.randn((4,)),
        "header": {"version": 1},
    }
    filename = tmp_path / "torch.pt"
    torch.save(state_dict, str(filename))

    loaded = tp.load(filename, mmap=False)
    assert isinstance(loaded, dict)
    assert loaded["header"] == {"version": 1}
    assert list(loaded["encoder.weight"].shape) == [4, 4]
    assert loaded["encoder.weight"].dtype == tp.float32
