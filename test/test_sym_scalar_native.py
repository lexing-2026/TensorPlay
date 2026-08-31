import pytest

import tensorplay as tp


def test_symbolic_integer_native_operations():
    value = tp.SymInt.symbolic("n")
    assert value.is_symbolic()
    assert not value.has_hint()
    with pytest.raises(RuntimeError):
        value.guard_int("test", 1)

    expression = value + tp.SymInt(2)
    assert expression.is_symbolic()
    assert "n" in str(expression)
    assert (tp.SymInt(3) + tp.SymInt(4)).expect_int() == 7
    assert value.sym_eq(value).guard_bool("test", 2)


def test_symbolic_boolean_native_operations():
    value = tp.SymBool.symbolic("flag")
    assert value.is_symbolic()
    assert (tp.SymBool(True) & tp.SymBool(True)).expect_bool()
    assert not tp.SymBool(False).sym_or(tp.SymBool(False)).expect_bool()
    assert tp.SymBool(True).sym_not().expect_bool() is False
    with pytest.raises(RuntimeError):
        value.guard_bool("test", 3)


def test_symbolic_float_native_operations():
    value = tp.SymFloat.symbolic("scale")
    assert value.is_symbolic()
    expression = value * tp.SymFloat(2.0)
    assert expression.is_symbolic()
    assert (tp.SymFloat(1.5) + tp.SymFloat(2.5)).expect_float() == 4.0


def test_symbolic_metadata_returns_are_native_types():
    value = tp.empty((2, 3))
    assert isinstance(tp.sym_size(value, 0), tp.SymInt)
    assert tp.sym_size(value, 0).expect_int() == 2
    assert isinstance(tp.sym_numel(value), tp.SymInt)
    assert tp.sym_numel(value).expect_int() == 6
    assert isinstance(tp.sym_stride(value, 1), tp.SymInt)
    assert tp.sym_stride(value, 1).expect_int() == 1
    assert isinstance(tp.sym_storage_offset(value), tp.SymInt)
    assert tp.sym_storage_offset(value).expect_int() == 0
    assert isinstance(tp.sym_is_contiguous(value), tp.SymBool)
    assert tp.sym_is_contiguous(value).expect_bool()
