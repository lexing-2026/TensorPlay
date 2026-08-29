"""Forward-mode automatic differentiation support.

The per-level stack lets ``make_dual`` pair a primal with its tangent at the
current level, and arithmetic on duals propagates tangents analytically
(Jacobian-vector products computed inline -- no graph, no backward pass).

kernel. Here the propagation rules are implemented explicitly per supported
op (the seed set below covers linear/algebraic composition); unsupported
operations raise instead of silently dropping tangents.
"""

from __future__ import annotations

import threading

import tensorplay as tp

__all__ = [
    "enter_dual_level",
    "exit_dual_level",
    "current_dual_level",
    "make_dual",
    "unpack_dual",
    "is_dual_tensor",
    "DualTensor",
]

_levels = threading.local()


def _depth() -> int:
    return getattr(_levels, "d", 0)


def current_dual_level() -> int:
    """Returns the current forward-AD nesting level (-1 = none active,
"""
    return _depth() - 1


def enter_dual_level() -> int:
    """Enters a new forward gradient level and returns its index
"""
    lvl = _depth()
    _levels.d = lvl + 1
    return lvl


def exit_dual_level(level: int | None = None):
    """Exits the given (default: most recent) forward gradient level.

    Any levels nested inside ``level`` are exited as well, matching
    """
    d = _depth()
    if d == 0:
        raise RuntimeError(
            "Trying to exit a forward AD level but no level is active")
    target = d - 1 if level is None else int(level)
    if not 0 <= target < d:
        raise RuntimeError(
            f"Trying to exit a forward AD level that was not entered: {level}")
    _levels.d = target


class DualTensor:
    """(primal, tangent) pair flowing through forward-mode evaluation."""

    __slots__ = ("primal", "tangent", "level")

    def __init__(self, primal, tangent, level):
        if isinstance(tangent, (int, float)):
            # Scalar constants carry a numeric zero tangent.
            tangent = float(tangent)
        elif not isinstance(tangent, tp.Tensor):
            raise TypeError("tangent must be a tensorplay.Tensor")
        self.primal = primal
        self.tangent = tangent
        self.level = level

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise TypeError(
            f"forward_ad: operation '{name}' is not in the native seed set "
            "and would silently drop tangents")

    # -- arithmetic (seed Jacobian-vector products) --------------------
    def _co(self, other):
        if isinstance(other, DualTensor):
            return other
        return DualTensor(other, tp.zeros_like(other)
                          if isinstance(other, tp.Tensor) else 0.0,
                          self.level)

    @staticmethod
    def _unwrap(v):
        return v

    def __add__(self, other):
        o = self._co(other)
        t = self.tangent + o.tangent if isinstance(o.tangent, tp.Tensor) \
            else self.tangent
        return DualTensor(self.primal + o.primal, t, self.level)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._co(other)
        t = self.tangent - o.tangent if isinstance(o.tangent, tp.Tensor) \
            else self.tangent
        return DualTensor(self.primal - o.primal, t, self.level)

    def __rsub__(self, other):
        o = self._co(other)
        return DualTensor(o.primal - self.primal,
                          (o.tangent - self.tangent)
                          if isinstance(o.tangent, tp.Tensor) else self.tangent * -1,
                          self.level)

    def __mul__(self, other):
        o = self._co(other)
        if isinstance(o.tangent, tp.Tensor):
            t = o.tangent * self.primal + self.tangent * o.primal
        else:
            t = self.tangent * o.primal
        return DualTensor(self.primal * o.primal, t, self.level)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._co(other)
        op2 = o.primal * o.primal
        num_t = (self.tangent * o.primal - self.primal * o.tangent) \
            if isinstance(o.tangent, tp.Tensor) else self.tangent * o.primal
        return DualTensor(self.primal / o.primal, num_t / op2, self.level)

    def __neg__(self):
        return DualTensor(-self.primal, -self.tangent, self.level)

    def __pow__(self, e):
        # d(x**e) = e * x**(e-1) * dx  (constant exponent)
        base = self.primal ** (e - 1)
        return DualTensor(self.primal ** e, e * base * self.tangent,
                          self.level)

    # -- transcendental seeds ------------------------------------------
    def exp(self):
        v = self.primal.exp()
        return DualTensor(v, v * self.tangent, self.level)

    def log(self):
        return DualTensor(self.primal.log(), self.tangent / self.primal,
                          self.level)

    def sin(self):
        return DualTensor(self.primal.sin(), self.primal.cos() * self.tangent,
                          self.level)

    def cos(self):
        return DualTensor(self.primal.cos(),
                          -self.primal.sin() * self.tangent, self.level)

    def sum(self, *a, **k):
        return DualTensor(self.primal.sum(*a, **k),
                          self.tangent.sum(*a, **k), self.level)

    def reshape(self, *shape):
        return DualTensor(self.primal.reshape(*shape),
                          self.tangent.reshape(*shape), self.level)

    def view(self, *shape):
        return self.reshape(*shape)

    @property
    def shape(self):
        return self.primal.shape

    def item(self):
        raise TypeError(
            "unpack_dual() first: item() would silently drop the tangent")

    def __repr__(self):
        return (f"DualTensor(level={self.level}, "
                f"primal={self.primal!r}, tangent={self.tangent!r})")


def is_dual_tensor(obj) -> bool:
    """True when ``obj`` carries a forward-mode tangent."""
    return isinstance(obj, DualTensor)


def make_dual(primal, tangent, *, level: int | None = None):
    """Pairs ``primal`` with ``tangent`` at ``level`` (default: current).

    """
    lvl = current_dual_level() if level is None else level
    if lvl < 0:
        raise RuntimeError(
            "make_dual requires an active forward AD level "
            "(call enter_dual_level() first)")
    if not isinstance(primal, tp.Tensor) or not isinstance(tangent, tp.Tensor):
        raise TypeError("make_dual: primal and tangent must be tensors")
    if tangent.dtype != primal.dtype:
        tangent = tp.to(tangent, primal.dtype) if hasattr(tp, "to") \
            else tangent.to(primal.dtype)
    return DualTensor(primal, tangent, lvl)


def unpack_dual(dual):
    """Returns ``(tangent, primal)``; ``tangent`` is None for plain tensors."""
    if isinstance(dual, DualTensor):
        return dual.tangent, dual.primal
    if isinstance(dual, tp.Tensor):
        from .graph import _hook_stack  # noqa: F401  (import symmetry check)
        return None, dual
    raise TypeError("unpack_dual: expected a tensor or DualTensor")
