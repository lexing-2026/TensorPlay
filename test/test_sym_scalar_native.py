import builtins
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


def test_symbolic_scalar_native_operator_surface():
    integer = tp.SymInt.symbolic("n")
    floating = tp.SymFloat.symbolic("scale")
    boolean = tp.SymBool.symbolic("flag")

    assert isinstance(integer / 2, tp.SymFloat)
    assert isinstance(integer // 2, tp.SymInt)
    assert isinstance(integer & 3, tp.SymInt)
    assert isinstance(integer | 3, tp.SymInt)
    assert isinstance(integer ^ 3, tp.SymInt)
    assert isinstance(integer << 2, tp.SymInt)
    assert isinstance(integer >> 2, tp.SymInt)
    assert isinstance(integer**2, tp.SymInt)
    assert isinstance(round(integer), tp.SymInt)
    assert isinstance(round(floating), tp.SymInt)
    assert isinstance(round(floating, 2), tp.SymFloat)
    assert isinstance(tp.sym_sqrt(integer), tp.SymFloat)
    assert isinstance(tp.sym_log2(floating), tp.SymFloat)
    assert isinstance(boolean ^ True, tp.SymBool)
    assert isinstance(boolean + 2, tp.SymInt)
    assert isinstance(boolean + tp.SymBool.symbolic("other"), tp.SymInt)
    assert ("flag" in str(boolean + 2))
    assert isinstance(tp.sym_ite(boolean, integer, tp.SymInt(2)), tp.SymInt)
    assert isinstance(tp.sym_sum(integer, 2), tp.SymInt)
    assert isinstance(tp.sym_sum([integer, 2]), tp.SymInt)

    assert type(tp.sym_float(2)) is builtins.float
    assert type(tp.sym_int(2.75)) is builtins.int
    assert type(tp.sym_not(True)) is builtins.bool
    assert type(tp.sym_min(1, 2)) is builtins.int
    assert type(tp.sym_max(1, 2.0)) is builtins.float
    assert tp.sym_ite(True, 1, 2) == 1
    assert tp.sym_ite(True, "yes", "no") == "yes"
