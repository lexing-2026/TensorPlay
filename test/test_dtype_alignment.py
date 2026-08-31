import numpy as np

import tensorplay as tp


REAL_DTYPES = (
    tp.bool,
    tp.uint8,
    tp.int8,
    tp.int16,
    tp.uint16,
    tp.int32,
    tp.uint32,
    tp.int64,
    tp.uint64,
    tp.float16,
    tp.bfloat16,
    tp.float32,
    tp.float64,
)

COMPLEX_DTYPES = (tp.complex32, tp.complex64, tp.complex128, tp.bcomplex32)


def test_dtype_surface_matches_torch_naming_and_properties():
    names = (
        "bool",
        "uint8",
        "int8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex32",
        "complex64",
        "complex128",
    )
    for name in names:
        dtype = getattr(tp, name)
        assert str(dtype) == f"tensorplay.{name}"
        assert getattr(tp.DType, name) == dtype
        assert dtype.itemsize > 0

    assert tp.float16 is tp.half
    assert tp.float32 is tp.float
    assert tp.float64 is tp.double
    assert tp.int16 is tp.short
    assert tp.int32 is tp.int
    assert tp.int64 is tp.long
    assert tp.complex32 is tp.chalf
    assert tp.complex64 is tp.cfloat
    assert tp.complex128 is tp.cdouble

    assert not tp.uint32.is_signed
    assert not tp.uint64.is_signed
    assert not tp.float32.is_complex
    assert tp.float32.is_floating_point
    assert not tp.complex64.is_floating_point
    assert tp.complex64.is_complex
    assert tp.complex64.is_signed


def test_real_dtype_roundtrip_and_uint64_range():
    for dtype in REAL_DTYPES:
        values = [True, False] if dtype is tp.bool else [1, 2]
        tensor = tp.tensor(values, dtype=dtype)
        assert tensor.dtype == dtype
        assert len(tensor.tolist()) == 2
        assert tensor.itemsize() == dtype.itemsize
        assert tensor[0].item() == values[0]

    large = 2**63 + 17
    tensor = tp.tensor([large], dtype=tp.uint64)
    assert tensor.item() == large
    assert tensor.tolist() == [large]
    assert tensor.numpy().dtype == np.dtype("uint64")


def test_complex_dtype_roundtrip_numpy_and_python_scalar_inference():
    inferred = tp.tensor([1 + 2j, 3 - 4j])
    assert inferred.dtype == tp.complex64
    assert inferred.tolist() == [1 + 2j, 3 - 4j]

    expected_numpy_dtype = {
        tp.complex32: np.dtype("complex64"),
        tp.complex64: np.dtype("complex64"),
        tp.complex128: np.dtype("complex128"),
        tp.bcomplex32: np.dtype("complex64"),
    }
    for dtype in COMPLEX_DTYPES:
        tensor = tp.tensor([1 + 2j, 3 - 4j], dtype=dtype)
        assert tensor.dtype == dtype
        assert tensor.is_complex()
        assert tensor.tolist()[0] == complex(1, 2)
        assert tensor.numpy().dtype == expected_numpy_dtype[dtype]

    for numpy_dtype, dtype in ((np.complex64, tp.complex64), (np.complex128, tp.complex128)):
        source = np.array([1 + 2j, 3 - 4j], dtype=numpy_dtype)
        tensor = tp.tensor(source)
        assert tensor.dtype == dtype
        np.testing.assert_array_equal(tensor.numpy(), source)


def test_dtype_cast_keeps_complex_real_component_and_numpy_uints():
    complex_tensor = tp.tensor([1 + 2j, 3 + 4j], dtype=tp.complex64)
    real_tensor = complex_tensor.to(tp.float32)
    assert real_tensor.dtype == tp.float32
    assert real_tensor.tolist() == [1.0, 3.0]

    for numpy_dtype, dtype in (
        (np.uint16, tp.uint16),
        (np.uint32, tp.uint32),
        (np.uint64, tp.uint64),
    ):
        source = np.array([1, 2], dtype=numpy_dtype)
        tensor = tp.tensor(source)
        assert tensor.dtype == dtype
        assert tensor.numpy().dtype == np.dtype(numpy_dtype)


def test_dlpack_codes_match_torch_for_supported_real_and_complex_types():
    torch = __import__("pytest").importorskip("torch")
    cases = (
        (tp.float32, torch.float32, [1, 2]),
        (tp.float64, torch.float64, [1, 2]),
        (tp.complex64, torch.complex64, [1 + 2j, 3 + 4j]),
        (tp.complex128, torch.complex128, [1 + 2j, 3 + 4j]),
    )
    for dtype, torch_dtype, values in cases:
        tensor = tp.tensor(values, dtype=dtype)
        assert torch.from_dlpack(tensor).dtype == torch_dtype
        torch_tensor = torch.tensor(values, dtype=torch_dtype)
        assert tp.from_dlpack(torch_tensor).dtype == dtype


def test_from_dlpack_rejects_a_non_capsule_instead_of_crashing():
    """An object with a permissive ``__getattr__`` answers every attribute
    probe, so ``__dlpack__`` must be validated before it is dereferenced."""

    import pytest

    class AnswersEverything:
        def __getattr__(self, name):
            return lambda *args, **kwargs: 42

    with pytest.raises(TypeError, match="dltensor"):
        tp.from_dlpack(AnswersEverything())
    with pytest.raises(TypeError, match="dltensor"):
        tp.tensor(AnswersEverything())


def test_from_dlpack_reports_a_consumed_capsule():
    import pytest

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    capsule = array.__dlpack__()
    assert np.allclose(tp.from_dlpack(capsule).numpy(), array)
    with pytest.raises(ValueError, match="already been consumed"):
        tp.from_dlpack(capsule)
