"""A shared copy answers only the node it was made for.

Values that are not windows onto named storage are shared by node identity.
An identity is only unique while its node is alive, so the cache keeps the
node and checks it on a hit: an entry reached through a recycled identity
must not hand one value's copy to another value.
"""
import types

from tensorplay.compiler.backends.stax import ir
from tensorplay.compiler.backends.stax.virtualized import V


class _Node:
    """A stand-in for an unrealized value: sized, with no named storage."""

    def __init__(self, size):
        self._size = size

    def get_size(self):
        return self._size

    def get_dtype(self):
        return "float32"


def test_layout_copy_is_shared_per_node(monkeypatch):
    made = []

    def fake_copy(x):
        made.append(x)
        return object()

    monkeypatch.setattr(ir.ExternKernel, "copy_input", staticmethod(fake_copy))
    with V.set_graph_handler(types.SimpleNamespace()):
        a = _Node([4, 8])
        first = ir.ExternKernel._shared_layout_copy(a, layout_key=("realize_input",))
        again = ir.ExternKernel._shared_layout_copy(a, layout_key=("realize_input",))
        assert first is again
        assert made == [a]
        other = ir.ExternKernel._shared_layout_copy(a, layout_key=("stride1",))
        assert other is not first


def test_layout_copy_rejects_a_recycled_identity(monkeypatch):
    monkeypatch.setattr(
        ir.ExternKernel, "copy_input", staticmethod(lambda x: ("copy-of", x)))
    with V.set_graph_handler(types.SimpleNamespace()) as _:
        a = _Node([4, 8])
        b = _Node([4, 8])
        key_layout = ("realize_input",)
        copy_a = ir.ExternKernel._shared_layout_copy(a, layout_key=key_layout)
        cache = V.graph._shared_layout_copy_cache
        # What a recycled identity looks like: b is asked for under the key
        # that a's entry sits at.
        cache[(id(b), (4, 8), key_layout)] = cache[(id(a), (4, 8), key_layout)]
        copy_b = ir.ExternKernel._shared_layout_copy(b, layout_key=key_layout)
        assert copy_a == ("copy-of", a)
        assert copy_b == ("copy-of", b)
