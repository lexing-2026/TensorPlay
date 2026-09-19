import pickle
import threading

import pytest

import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE, OpOverload, OpOverloadPacket
from tensorplay.utils._dispatch import (
    TensorPlayDispatchMode,
    _disable_current_modes,
    _get_current_dispatch_mode,
    is_in_tensorplay_dispatch_mode,
)

ops = getattr(tp.ops, NATIVE_NAMESPACE)


class RecordingMode(TensorPlayDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.threads = set()

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append(func)
        self.threads.add(threading.get_ident())
        return func(*args, **(kwargs or {}))

    def names(self):
        return [f.__name__ for f in self.calls]


def test_mode_receives_operator_overloads():
    x = tp.tensor([0.0, 1.0])
    with RecordingMode() as mode:
        y = x.sin()
    assert mode.calls, "the mode saw no operator"
    assert mode.calls[0] is ops.sin.default
    assert isinstance(mode.calls[0], OpOverload)
    assert (y - tp.tensor([0.0, 1.0]).sin()).abs().max().item() == 0.0


def test_python_key_follows_the_mode_stack():
    assert not tp._C._python_dispatch_key_included()
    with RecordingMode():
        assert tp._C._python_dispatch_key_included()
        assert is_in_tensorplay_dispatch_mode()
    assert not tp._C._python_dispatch_key_included()
    assert not is_in_tensorplay_dispatch_mode()


def test_backward_operators_reach_the_mode():
    x = tp.tensor([1.0, 2.0], requires_grad=True)
    with RecordingMode() as mode:
        (x * x).sum().backward()
    forward_and_backward = mode.names()
    assert "mul.Tensor" in forward_and_backward
    # The multiplication backward emits further multiplications/expands.
    assert forward_and_backward.count("mul.Tensor") >= 2
    assert x.grad.tolist() == [2.0, 4.0]


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA runtime is not available")
def test_backward_on_device_worker_threads_reaches_the_mode():
    x = tp.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with RecordingMode() as mode:
        (x.sin() * 3).sum().backward()
    assert "cos.default" in mode.names()
    assert (x.grad.cpu() - x.detach().cpu().cos() * 3).abs().max().item() < 1e-6


def test_handler_reentry_records_autograd_once():
    x = tp.tensor([1.0, 2.0], requires_grad=True)
    with RecordingMode():
        y = x.exp()
    assert y.grad_fn is not None
    y.sum().backward()
    assert (x.grad - x.detach().exp()).abs().max().item() < 1e-6


def test_nested_modes_run_innermost_first():
    order = []

    class Tag(TensorPlayDispatchMode):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
            order.append((self.tag, func.__name__))
            return func(*args, **(kwargs or {}))

    x = tp.tensor([1.0])
    with Tag("outer"), Tag("inner"):
        x.neg()
    assert order == [("inner", "neg.default"), ("outer", "neg.default")]


def test_mode_can_replace_results():
    class Zeros(TensorPlayDispatchMode):
        def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            return tp.zeros_like(out) if func is ops.sin.default else out

    with Zeros():
        y = tp.tensor([1.0, 2.0]).sin()
    assert y.tolist() == [0.0, 0.0]


def test_handler_exceptions_keep_their_type():
    class Boom(Exception):
        pass

    class Raising(TensorPlayDispatchMode):
        def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
            raise Boom("from the handler")

    with pytest.raises(Boom, match="from the handler"):
        with Raising():
            tp.tensor([1.0]).cos()
    assert _get_current_dispatch_mode() is None


def test_wrong_result_type_is_reported():
    class Wrong(TensorPlayDispatchMode):
        def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
            return "not a tensor"

    with pytest.raises(TypeError):
        with Wrong():
            tp.tensor([1.0]).cos()


def test_inplace_operator_returns_its_destination():
    x = tp.tensor([1.0, 2.0])
    with RecordingMode() as mode:
        y = x.add_(1)
    assert y is x
    assert x.tolist() == [2.0, 3.0]
    assert "add_.Tensor" in mode.names() or "add_.Scalar" in mode.names()


def test_view_operators_are_dispatched():
    x = tp.arange(12.0).reshape(3, 4)
    with RecordingMode() as mode:
        x.view(4, 3)
        x.select(0, 1)
        x[:, 1:3]
        x.t()
    names = mode.names()
    assert "view.default" in names
    assert "select.int" in names
    assert "slice.Tensor" in names


def test_disable_current_modes_restores_the_stack():
    with RecordingMode() as mode:
        with _disable_current_modes():
            tp.tensor([1.0]).sin()
        assert _get_current_dispatch_mode() is mode
    assert not mode.calls


def test_every_schema_operator_has_a_python_kernel():
    missing = [
        name
        for name, _schema, _args, _npos, _tags in tp._C._python_dispatch_entries()
        if not tp._C._dispatch_has_kernel_for_dispatch_key(name, "Python")
    ]
    assert missing == []


def test_overload_objects_expose_schemas():
    add = ops.add
    assert isinstance(add, OpOverloadPacket)
    assert "Tensor" in add.overloads() and "Scalar" in add.overloads()
    overload = add.Tensor
    schema = overload._schema
    assert schema.name == "add" and schema.overload_name == "Tensor"
    assert [a.name for a in schema.arguments] == ["self", "other", "alpha"]
    assert schema.arguments[2].kwarg_only
    assert overload.name() == f"{NATIVE_NAMESPACE}::add.Tensor"
    assert ops.add_.Tensor._schema.is_mutable
    assert ops.transpose.int._schema._is_view_op()
    x = tp.tensor([1.0, 2.0])
    assert overload(x, x, alpha=2).tolist() == [3.0, 6.0]
    assert add(x, 1).tolist() == [2.0, 3.0]
    assert pickle.loads(pickle.dumps(overload)) is overload
    assert ops.sin.default is ops.sin.default


def test_every_kernel_matches_its_operator_abi():
    # Each registered kernel must have exactly the signature every caller of
    # its operator casts the slot to; anything else is a mismatched call.
    mismatches = tp._C._dispatch_abi_mismatches()
    assert mismatches == [], "\n".join(
        f"{op} [{key}]: registered {got}, expected {want}"
        for op, key, got, want in mismatches[:50]
    )


def test_mask_indexing_is_recorded_not_baked():
    from tensorplay.graph.experimental._dispatch_trace import dispatch_make_graph

    def fn(a):
        return a[a > 0]

    graph = dispatch_make_graph(fn)(tp.tensor([1.0, -2.0, 3.0, -4.0]))
    targets = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
    assert any("index.Tensor" in t for t in targets)
    other = tp.tensor([-1.0, 2.0, -3.0, 4.0])
    assert graph(other).tolist() == [2.0, 4.0]


def test_mask_assignment_is_recorded():
    from tensorplay.graph.experimental._dispatch_trace import dispatch_make_graph

    def fn(a):
        b = a.clone()
        b[b > 0] = 0.0
        return b

    graph = dispatch_make_graph(fn)(tp.tensor([1.0, -2.0, 3.0]))
    assert graph(tp.tensor([-1.0, 2.0, -3.0])).tolist() == [-1.0, 0.0, -3.0]
