"""Forward-mode automatic differentiation support.

Forward gradients (tangents) live on the tensors themselves, keyed by an
active forward-AD *level*.  ``enter_dual_level`` hands out the level's
integer handle; ``make_dual`` pairs a primal with its tangent inside that
level; arithmetic on dual tensors then propagates tangents through each
operation's forward-derivative formula (Jacobian-vector products computed
inline -- no backward graph); ``unpack_dual`` reads the primal and tangent
back out.  Exiting the level erases every tangent registered with it.

Nesting is not supported: a single forward-AD level is active at a time.
Higher-order forward gradients can be composed by running :func:`jvp`
inside another :func:`jvp`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import tensorplay as tp
from .grad_mode import _DecoratorContextManager

__all__ = [
    "UnpackedDualTensor",
    "current_dual_level",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
    "dual_level",
    "is_dual_tensor",
]

_C = tp._C

# Python-side mirror of the active level so make_dual / unpack_dual can use
# it as their default; the C++ registry stays the source of truth.
_current_level = -1


def current_dual_level() -> int:
    """Returns the current forward-AD level (-1 when none is active)."""
    return _current_level


def enter_dual_level() -> int:
    """Enters a new forward grad level and returns its index.

    This level can be used to make and unpack dual Tensors to compute
    forward gradients.  Levels cannot nest: entering while another level is
    active raises, so all forward AD computation happens inside one level.
    """
    global _current_level
    new_level = _C._enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError(
            "Entering a new forward AD level but the current level "
            "is not valid. Make sure you did not modify it directly."
        )
    _current_level = new_level
    return new_level


def exit_dual_level(*, level: int | None = None) -> None:
    """Exits a forward grad level.

    This deletes all the tangents associated with this level.  Only exiting
    the most recently entered level is allowed.
    """
    global _current_level
    if _current_level < 0:
        raise RuntimeError(
            "Trying to exit a forward AD level but no level is active")
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError(
            "Trying to exit a forward AD level that was not the last one "
            "that was created. This is not supported."
        )
    _C._exit_dual_level(level)
    _current_level = level - 1


def make_dual(tensor: tp.Tensor, tangent: tp.Tensor, *,
              level: int | None = None) -> tp.Tensor:
    """Associates a tensor value with its tangent to create a "dual tensor".

    The result is a new tensor aliased to ``tensor`` with ``tangent``
    attached as its forward gradient.  The tangent can be recovered with
    :func:`unpack_dual`.

    Given a function ``f`` whose jacobian is ``J``, this computes the
    Jacobian-vector product (``jvp``) between ``J`` and a given vector ``v``::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, v)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)
    """
    from ..functional import _make_dual as _op_make_dual

    if level is None:
        level = _current_level

    if level < 0:
        raise RuntimeError(
            "Trying to create a dual Tensor for forward AD but no level "
            "exists, make sure to enter_dual_level() first."
        )
    return _op_make_dual(tensor, tangent, level)


class UnpackedDualTensor(NamedTuple):
    """Namedtuple with the primal and tangent parts of a dual tensor."""

    primal: tp.Tensor
    tangent: tp.Tensor | None


def unpack_dual(tensor: tp.Tensor, *,
                level: int | None = None) -> UnpackedDualTensor:
    """Unpacks a dual tensor into its primal value and forward gradient.

    Returns ``(primal, tangent)`` where ``primal`` is a view of ``tensor``'s
    primal and ``tangent`` is ``tensor``'s tangent (``None`` when ``tensor``
    carries no tangent at ``level``).
    """
    from ..functional import _unpack_dual as _op_unpack_dual

    if level is None:
        level = _current_level

    if level < 0:
        return UnpackedDualTensor(tensor, None)

    primal, tangent = _op_unpack_dual(tensor, level)
    # An absent tangent comes back as an undefined tensor; normalize it to
    # None so callers can branch on the python value.
    if tangent is not None and not tangent.defined():
        tangent = None
    return UnpackedDualTensor(primal, tangent)


def is_dual_tensor(tensor: Any) -> bool:
    """True when ``tensor`` carries a forward-mode tangent."""
    if not isinstance(tensor, tp.Tensor):
        raise TypeError(
            f"is_dual_tensor: expected a tensorplay.Tensor, got {type(tensor)}")
    if _current_level < 0:
        return False
    return _C._fw_grad(tensor, _current_level).defined()


class dual_level(_DecoratorContextManager):
    """Context manager for forward AD.

    All forward AD computation must occur within a ``dual_level`` context,
    which enters the level on entry and exits it (erasing its tangents) on
    exit.  Nested ``dual_level`` contexts are not supported; to compute
    higher-order forward gradients, use :func:`jvp`.
    """

    def __enter__(self) -> int:
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()


# Private helper to enable or disable tangent reads; transformations that
# run a function under forward AD enable it only for the duration of the
# traced call so unrelated code never observes tangents.
class _set_fwd_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = _C._get_fwd_grad_enabled()
        _C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _C._set_fwd_grad_enabled(self.prev)
