"""TensorPlay MEGA serialization: containers, dedup, checksums, mmap, inspect."""

import io
import json
import zlib
from pathlib import Path

import pytest

import tensorplay as tp

from tensorplay.serialization import inspect_checkpoint, parse_mega_header


@pytest.fixture(autouse=True)
def _require_runtime():
    pytest.importorskip("megatensors")


def test_rejects_unknown_extension(tmp_path: Path):
    value = tp.tensor([1.0, 2.0], dtype=tp.float32)
    for suffix in (".tpm", ".onnx", ".gguf", ".bin"):
        filename = tmp_path / f"weights{suffix}"
        with pytest.raises(ValueError, match="Supported"):
            tp.save({"weight": value}, filename)
        (tmp_path / f"weights{suffix}").write_bytes(b"garbage!!!")
        with pytest.raises(ValueError):
            tp.load(filename)


def test_mega_round_trip_preserves_tensor_containers(tmp_path: Path):
    value = tp.tensor([1.0, 2.0], dtype=tp.bfloat16)

    for name, original, expected_type in (
        ("dict", {"weight": value}, dict),
        ("tuple", (value,), tuple),
        ("list", [value], list),
    ):
        filename = tmp_path / f"{name}.mega"
        tp.save(original, filename)
        loaded = tp.load(filename)
        assert isinstance(loaded, expected_type)
        loaded_value = loaded["weight"] if isinstance(loaded, dict) else loaded[0]
        assert loaded_value.dtype == value.dtype
        assert tp.allclose(loaded_value, value)


def test_mega_round_trip_single_tensor(tmp_path: Path):
    value = tp.arange(6, dtype=tp.float32).reshape((2, 3))
    filename = tmp_path / "single.mega"
    tp.save(value, filename)
    loaded = tp.load(filename)
    assert isinstance(loaded, tp.Tensor)
    assert list(loaded.shape) == [2, 3]
    assert tp.allclose(loaded, value)


def test_mega_nested_tree_with_primitives(tmp_path: Path):
    obj = {
        "layer": {"weight": tp.ones((2, 2), dtype=tp.float32), "bias": None,
                  "scale": 0.5, "name": "block0"},
        "pairs": (tp.zeros((3,), dtype=tp.int64), 7),
        "flags": [True, "keep", 1.25],
    }
    filename = tmp_path / "tree.mega"
    tp.save(obj, filename)
    loaded = tp.load(filename)

    assert loaded["layer"]["scale"] == 0.5
    assert loaded["layer"]["name"] == "block0"
    assert loaded["layer"]["bias"] is None
    assert isinstance(loaded["pairs"], tuple)
    assert loaded["pairs"][1] == 7
    assert tp.allclose(loaded["layer"]["weight"], tp.ones((2, 2)))
    assert loaded["flags"] == [True, "keep", 1.25]


def test_mega_unsupported_leaf_raises(tmp_path: Path):
    filename = tmp_path / "bad.mega"
    with pytest.raises(TypeError, match="unsupported leaf"):
        tp.save({"obj": object()}, filename)


def test_mega_shared_storage_dedup_and_aliasing(tmp_path: Path):
    base = tp.arange(8, dtype=tp.float32)
    tied = {"a": base, "b": base}
    filename = tmp_path / "tied.mega"
    tp.save(tied, filename)

    header = parse_mega_header(str(filename))
    offsets = {info["payload_offset"] for info in header["tensors"].values()}
    assert len(offsets) == 1, "shared storage must occupy one payload region"

    info = inspect_checkpoint(filename)
    total_nbytes = sum(entry["nbytes"] for entry in info["tensors"].values())
    assert info["file_size"] - info["header_length"] < total_nbytes * 2 - base.numel() * 4 + 1

    loaded = tp.load(filename)
    assert int(loaded["a"].data_ptr()) == int(loaded["b"].data_ptr())
    assert tp.allclose(loaded["b"], base)


def test_mega_checksum_corruption_detected(tmp_path: Path):
    value = {"w": tp.full((16,), 3.14, dtype=tp.float32)}
    filename = tmp_path / "crc.mega"
    tp.save(value, filename, checksum="crc32")

    header = parse_mega_header(str(filename))
    info = header["tensors"]["w"]
    payload_start = header["header_length"] + info["payload_offset"]
    with open(filename, "r+b") as handle:
        handle.seek(payload_start + 8)
        original = handle.read(1)
        handle.seek(payload_start + 8)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(Exception, match="checksum"):
        tp.load(filename)


def test_mega_sha256_and_none_modes(tmp_path: Path):
    value = {"w": tp.arange(4, dtype=tp.float32)}
    for mode in ("sha256", "none"):
        filename = tmp_path / f"{mode}.mega"
        tp.save(value, filename, checksum=mode)
        header = parse_mega_header(str(filename))
        expected_type = {"sha256": 2, "none": 0}[mode]
        assert all(info["checksum_type"] == expected_type
                   for info in header["tensors"].values())
        loaded = tp.load(filename)
        assert tp.allclose(loaded["w"], value["w"])


def test_inspect_checkpoint_reports_layout(tmp_path: Path):
    meta = {"author": "tp", "step": 7}
    filename = tmp_path / "meta.mega"
    tp.save({"w": tp.zeros((2, 3), dtype=tp.float32)}, filename, metadata=meta,
            checksum="crc32")

    info = inspect_checkpoint(filename, verify_checksums=True)
    assert info["format"] == "mega"
    assert info["metadata"]["general.architecture"] == "tensorplay"
    assert info["metadata"]["author"] == "tp"
    assert info["metadata"]["step"] == 7
    assert info["tensors"]["w"]["shape"] == [2, 3]
    assert info["tensors"]["w"]["dtype"] == "F32"
    assert info["tensors"]["w"]["checksum"] == "crc32"
    assert info["checksums_verified"] is True


def test_mega_byteorder_recorded(tmp_path: Path):
    import sys

    filename = tmp_path / "bo.mega"
    tp.save({"w": tp.ones((1,), dtype=tp.float32)}, filename)
    header = parse_mega_header(str(filename))
    assert header["metadata"]["tensorplay.byteorder"] == sys.byteorder


def test_mega_mmap_load(tmp_path: Path):
    value = {"w": tp.arange(100, dtype=tp.float32)}
    filename = tmp_path / "mmap.mega"
    tp.save(value, filename)

    loaded = tp.load(filename, mmap=True)
    assert tp.allclose(loaded["w"], value["w"])
    # Views remain valid after the loader call returns.
    assert float(loaded["w"].sum()) == sum(range(100))


def test_mega_float8_round_trip(tmp_path: Path):
    names = ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
    value = {
        name: tp.arange(16, dtype=tp.uint8).reshape((4, 4)).view(getattr(tp, name))
        for name in names
    }
    filename = tmp_path / "fp8.mega"
    tp.save(value, filename)

    loaded = tp.load(filename)
    for name in names:
        assert loaded[name].dtype == getattr(tp, name)
        assert (
            loaded[name].view(tp.uint8).numpy().tobytes()
            == value[name].view(tp.uint8).numpy().tobytes()
        )

    mapped = tp.load(filename, mmap=True)
    for name in names:
        assert mapped[name].dtype == getattr(tp, name)
        assert (
            mapped[name].view(tp.uint8).numpy().tobytes()
            == value[name].view(tp.uint8).numpy().tobytes()
        )


def test_mega_load_from_buffer(tmp_path: Path):
    value = {"w": tp.arange(5, dtype=tp.float32)}
    filename = tmp_path / "buf.mega"
    tp.save(value, filename)
    buffer = io.BytesIO(filename.read_bytes())

    loaded = tp.load(buffer)
    assert tp.allclose(loaded["w"], value["w"])


def test_map_location_variants(tmp_path: Path):
    value = {"w": tp.ones((2,), dtype=tp.float32)}

    filename = tmp_path / "loc.mega"
    tp.save(value, filename)

    by_string = tp.load(filename, map_location="cpu")
    assert tp.allclose(by_string["w"], value["w"])

    by_device = tp.load(filename, map_location=tp.Device(tp.DeviceType.CPU))
    assert tp.allclose(by_device["w"], value["w"])

    by_mapping = tp.load(filename, map_location={"cpu": "cpu"})
    assert tp.allclose(by_mapping["w"], value["w"])

    def choose(storage_stub, location):
        return "cpu"

    by_callable = tp.load(filename, map_location=choose)
    assert tp.allclose(by_callable["w"], value["w"])

    received = {}

    def into_tensor(storage_stub, location):
        received["location"] = location
        return tp.empty((2,), dtype=tp.float32)

    by_tensor = tp.load(filename, map_location=into_tensor)
    assert received["location"] in {"cpu", "cuda"}
    assert tp.allclose(by_tensor["w"], value["w"])


def _compressed_mega_artifact(tmp_path: Path, values: dict, *, chunk_crc: bool = True) -> Path:
    import numpy as np

    megacpp = pytest.importorskip("megatensors").cpp
    src = tmp_path / "compressed-source.bin"
    payload = tmp_path / "compressed-payload.tmp"
    target = tmp_path / "compressed.mega"
    records, offset = [], 0
    with open(src, "wb") as handle:
        for name, array in values.items():
            raw = array.tobytes()
            handle.write(raw)
            records.append({
                "name": name,
                "shape": list(array.shape),
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "payload_offset": 0,
                "logical_nbytes": len(raw),
                "stored_nbytes": len(raw),
                "src_filename": str(src),
                "src_offset": offset,
            })
            offset += len(raw)
    records = list(megacpp.encode_file_tensors(str(payload), records, 1, 3, 1.0))
    if chunk_crc:
        import zlib

        blob = payload.read_bytes()
        for record in records:
            chunks = record.get("chunks", [])
            entries = chunks if chunks else [record]
            for entry in entries:
                begin = int(entry["payload_offset"])
                size = int(entry["stored_size"]) if chunks else int(entry["stored_nbytes"])
                if not size:
                    continue
                entry["checksum_type"] = 1
                entry["checksum"] = (
                    zlib.crc32(blob[begin:begin + size]).to_bytes(4, "little") + b"\x00" * 28
                )
    megacpp.write_file(
        str(target),
        records,
        {"general.alignment": 4096, "mega.tensor_info.format": "self_describing"},
        4096,
    )
    return target


def test_mega_compressed_chunked_load(tmp_path: Path):
    import numpy as np

    length = 3 * 1024 * 1024
    values = {
        "big": np.sin(np.arange(length, dtype=np.float64) / 1000).astype(np.float32),
        "small": np.linspace(-1.0, 1.0, 4096, dtype=np.float32),
    }
    filename = _compressed_mega_artifact(tmp_path, values)

    header = parse_mega_header(str(filename))
    assert header["tensors"]["big"]["storage_format"] == "chunked_raw_dense"
    assert len(header["tensors"]["big"]["chunks"]) >= 3
    assert header["tensors"]["small"]["compression_codec"] == 1

    for mmap in (False, True):
        loaded = tp.load(filename, map_location="cpu", mmap=mmap)
        for name, expected in values.items():
            assert np.array_equal(loaded[name].numpy(), expected), (name, mmap)


def test_mega_compressed_chunk_checksum_corruption_detected(tmp_path: Path):
    import numpy as np

    filename = _compressed_mega_artifact(
        tmp_path, {"big": np.sin(np.arange(3 * 1024 * 1024) / 1000).astype(np.float32)}
    )
    header = parse_mega_header(str(filename))
    chunk = header["tensors"]["big"]["chunks"][0]
    offset = header["header_length"] + int(chunk["payload_offset"]) + 3
    data = bytearray(filename.read_bytes())
    data[offset] ^= 0xFF
    filename.write_bytes(data)

    with pytest.raises(ValueError, match="checksum"):
        tp.load(filename, map_location="cpu")

def test_mega_crc32_fallback_writer_stores_little_endian(tmp_path: Path):
    import numpy as np

    values = np.arange(1024, dtype=np.float32).reshape((32, 32))
    filename = tmp_path / "dlpack.mega"
    tp.save({"w": tp.from_dlpack(values)}, filename, checksum="crc32")

    info = parse_mega_header(str(filename))["tensors"]["w"]
    expected = zlib.crc32(values.tobytes()) & 0xFFFFFFFF
    assert bytes(info["checksum"])[:4] == expected.to_bytes(4, "little")

    loaded = tp.load(filename, map_location="cpu")
    assert np.array_equal(loaded["w"].numpy(), values)
