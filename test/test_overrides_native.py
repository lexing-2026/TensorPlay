import pytest

import tensorplay as tp
import tensorplay.overrides as overrides


_NATIVE_API = (
    "_get_tensor_function_state",
    "_set_tensor_function_state",
    "_exchange_tensor_function_skip_next",
    "_exchange_tensor_subclass_skip_next",
    "_push_tensor_function_mode",
    "_pop_tensor_function_mode",
    "_get_tensor_function_mode",
    "_len_tensor_function_mode",
    "_get_tensor_dispatch_layer",
)


def test_native_override_surface_is_bound():
    missing = [name for name in _NATIVE_API if not callable(getattr(tp._C, name, None))]
    assert not missing, f"missing native override bindings: {missing}"


def test_function_hook_precedes_native_argument_conversion():
    calls = []

    class FunctionLike:
        @classmethod
        def __tensorplay_function__(cls, func, types, args, kwargs):
            calls.append((func, types, args, kwargs))
            return "function"

    result = tp._C.add(tp.tensor([1.0]), FunctionLike())
    assert result == "function"
    assert calls and calls[0][1] == (FunctionLike,)
    assert calls[0][3] == {}


def test_mode_stack_and_exception_restore():
    events = []

    class Mode(overrides.TensorPlayFunctionMode):
        def __tensorplay_function__(self, func, types, args, kwargs=None):
            events.append((func, types, args, kwargs))
            return "mode"

    with Mode():
        assert tp._C.add(tp.tensor([1.0]), 2.0) == "mode"
        assert tp._C._len_tensor_function_mode() == 1
    assert tp._C._len_tensor_function_mode() == 0
    assert events and events[0][1] == ()

    class FailingMode(overrides.TensorPlayFunctionMode):
        def __tensorplay_function__(self, func, types, args, kwargs=None):
            raise ValueError("hook failure")

    with pytest.raises(ValueError, match="hook failure"):
        with FailingMode():
            tp._C.add(tp.tensor([1.0]), 2.0)
    assert tp._C._len_tensor_function_mode() == 0


def test_subclass_precedence_and_redispatch():
    calls = []

    class Base(tp.Tensor):
        @classmethod
        def __tensorplay_dispatch__(cls, func, types, args, kwargs):
            calls.append(("base", func, types))
            return "base"

    class Derived(Base):
        @classmethod
        def __tensorplay_dispatch__(cls, func, types, args, kwargs):
            calls.append(("derived", func, types))
            return NotImplemented

    base = Base(tp.tensor([1.0]))
    derived = Derived(tp.tensor([2.0]))
    assert tp._C.add(derived, base) == "base"
    assert [entry[0] for entry in calls] == ["derived", "base"]
    assert calls[0][2] == (Derived, Base)

    class Redispatch(tp.Tensor):
        @classmethod
        def __tensorplay_dispatch__(cls, func, types, args, kwargs):
            return overrides.redispatch_function(func, types, args, kwargs)

    result = tp._C.add(Redispatch(tp.tensor([1.0])), 2.0)
    assert type(result) is tp.Tensor


def test_disable_and_restore_native_state():
    class Subclass(tp.Tensor):
        @classmethod
        def __tensorplay_dispatch__(cls, func, types, args, kwargs):
            return "blocked"

    value = Subclass(tp.tensor([1.0]))
    with overrides._disable_tensorplay_function():
        assert type(tp._C.add(value, 2.0)) is tp.Tensor
        assert tp._C._get_tensor_function_state() == 2
    assert tp._C._get_tensor_function_state() == 0
