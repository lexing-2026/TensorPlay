"""Every element type prints its values, not a placeholder."""

import math

import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def suffix(device):
    return "" if device == "cpu" else ", device='cuda:0'"


@pytest.mark.parametrize("device", DEVICES)
def test_complex_values_print_as_real_sign_imaginary_j(device):
    z = tp.tensor([1.5 - 0.25j, -2 + 0j, 0.5j], dtype=tp.complex128, device=device)
    assert repr(z) == f"tensor([1.5-0.25j, -2.+0.j, 0.+0.5j], dtype=ComplexDouble{suffix(device)})"
    w = tp.tensor([[1 + 1j], [-1 - 1j]], dtype=tp.complex64, device=device)
    assert repr(w) == f"tensor([[1.+1.j],\n        [-1.-1.j]], dtype=ComplexFloat{suffix(device)})"


def test_a_complex_scalar_and_its_non_finite_parts_print():
    assert repr(tp.tensor(3 - 4j, dtype=tp.complex128)) == "tensor(3.-4.j, dtype=ComplexDouble)"
    z = tp.tensor([complex(math.nan, math.inf)], dtype=tp.complex128)
    assert repr(z) == "tensor([nan+infj], dtype=ComplexDouble)"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype, name",
    [(tp.float16, "Float16"), (tp.bfloat16, "BFloat16")],
)
def test_half_precision_prints_its_values(device, dtype, name):
    x = tp.tensor([1.5, -2.0, 0.25], dtype=dtype, device=device)
    assert repr(x) == f"tensor([1.5, -2., 0.25], dtype={name}{suffix(device)})"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype, name",
    [(tp.int8, "Int8"), (tp.int16, "Int16"), (tp.uint8, "UInt8"),
     (tp.uint16, "UInt16"), (tp.uint32, "UInt32"), (tp.uint64, "UInt64")],
)
def test_small_and_unsigned_integers_print_their_values(device, dtype, name):
    x = tp.tensor([0, 7, 100], dtype=dtype, device=device)
    assert repr(x) == f"tensor([0, 7, 100], dtype={name}{suffix(device)})"


def test_fp8_values_print_like_the_other_floats():
    x = tp.ones(2, dtype=tp.float8_e4m3fn)
    assert repr(x) == "tensor([1., 1.], dtype=Float8_e4m3fn)"


@pytest.mark.parametrize(
    "dtype",
    [tp.float8_e4m3fn, tp.float8_e5m2, tp.float8_e4m3fnuz, tp.float8_e5m2fnuz],
)
def test_fp8_tensors_are_filled_on_the_cpu(dtype):
    assert tp.full((3,), 2.0, dtype=dtype).float().tolist() == [2.0, 2.0, 2.0]
    assert tp.zeros(2, dtype=dtype).float().tolist() == [0.0, 0.0]
    strided = tp.zeros(4, dtype=dtype)[::2]
    strided.fill_(1.0)
    assert strided.float().tolist() == [1.0, 1.0]


def test_a_summarized_complex_tensor_keeps_its_edges():
    z = tp.arange(2000, dtype=tp.float64).to(tp.complex128) * 1j
    text = repr(z)
    assert text.startswith("tensor([0.+0.j, 0.+1.j, 0.+2.j, ...")
    assert text.endswith("0.+1999.j], dtype=ComplexDouble)")
