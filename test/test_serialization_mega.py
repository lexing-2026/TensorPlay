from pathlib import Path

import pytest

import tensorplay as tp


def test_save_and_load_accept_only_mega(tmp_path: Path):
    value = tp.tensor([1.0, 2.0], dtype=tp.float32)
    for suffix in (".tpm", ".onnx", ".gguf", ".safetensors", ".pt", ".pth", ".bin"):
        filename = tmp_path / f"weights{suffix}"
        with pytest.raises(ValueError, match="MEGA"):
            tp.save({"weight": value}, filename)
        with pytest.raises(ValueError, match="MEGA"):
            tp.load(filename)


def test_mega_round_trip_preserves_tensor_containers(tmp_path: Path):
    pytest.importorskip("megatensors")
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
