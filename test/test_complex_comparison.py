"""Complex values compare for equality on every backend and refuse to be ordered."""

import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def values(device, dtype):
    return tp.tensor([1 + 0j, 0j, 2 - 1j, -1 + 3j], dtype=dtype, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [tp.complex64, tp.complex128])
def test_a_complex_tensor_equals_a_real_scalar_through_its_real_part(device, dtype):
    z = values(device, dtype)
    assert z.eq(0).tolist() == [False, True, False, False]
    assert z.ne(1).tolist() == [False, True, True, True]
    assert (z == 1.0).tolist() == [True, False, False, False]
    assert z.eq(2 - 1j).tolist() == [False, False, True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_a_complex_tensor_equals_a_real_tensor(device):
    z = values(device, tp.complex128)
    r = tp.tensor([1.0, 0.0, 2.0, -1.0], dtype=tp.float64, device=device)
    assert z.eq(r).tolist() == [True, True, False, False]
    assert z.ne(r).tolist() == [False, False, True, True]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["lt", "le", "gt", "ge"])
def test_complex_values_have_no_order(device, op):
    z = values(device, tp.complex64)
    with pytest.raises(NotImplementedError, match="ComplexFloat"):
        getattr(z, op)(z)
    with pytest.raises(NotImplementedError, match="ComplexFloat"):
        getattr(z, op)(0)
