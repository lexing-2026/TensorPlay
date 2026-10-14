"""Gradients that do not go through the circuit.

A gate's parameters are differentiable, so a circuit can be trained the usual
way.  Two situations call for something else.  One is a value that is not built
out of differentiable primitives at all -- an energy estimated from a finite
number of measurements has no gradient to take.  The other is a circuit deep
enough that backpropagating through every rotation is more expensive than
evaluating the whole circuit twice.

Both are answered by the shift rule: for a value that depends on one angle
through ``exp(-i theta G)``, its derivative is a difference of the value at two
displaced angles, and it is exact.  The cost is two evaluations of the whole
value per parameter, against one backpropagation through the whole circuit --
which is why it is the rule of choice for a deep circuit measured in shots, and
why it survives a sampler in the middle.

The displacement and the weight follow from the generator's own eigenvalues, so
the general form is available as arguments rather than assumed.  A value whose
frequencies are ``omega`` satisfies
``f'(theta) = omega / (2 sin(omega * shift)) * [f(theta + shift) - f(theta - shift)]``
for any displacement that is not a zero of that sine -- which rules out the
half-period and every multiple of it.  Every angle gate in this module is a
rotation about half a Pauli word, whose eigenvalues are ``+/- 1/2``, so its
frequency is one: the default displacement of ``pi / 2`` and weight of one half
are the rule for all of them.
"""
import math
from typing import Callable, Optional

import tensorplay as tp

__all__ = [
    "param_shift",
]


def param_shift(
    value_fn: Callable[[tp.Tensor], tp.Tensor],
    params: tp.Tensor,
    *,
    shift: Optional[float] = None,
    weight: Optional[float] = None,
    axis: int = -1,
) -> tp.Tensor:
    """The gradient of a value built from one angle per parameter, by shifting.

    Exact for a value that depends on each parameter through
    ``exp(-i theta G)``, and independent of how the value itself was computed:
    measured in shots, run through a sampler, or written out by hand.

    Args:
        value_fn: takes a tensor shaped like ``params`` and returns the value,
            one entry per batch entry.
        params: the angles; the parameter axis is ``axis``, the last one by
            default, so a ``[batch, angles]`` tensor is read as one angle per
            column.
        shift: how far to displace the angle; ``pi / 2`` by default, which is
            right for a rotation about half a Pauli word.  A generator whose
            eigenvalues differ by ``omega`` wants ``pi / (2 * omega)``, and any
            displacement will do that is not a multiple of ``pi / omega``.
        weight: what to multiply the difference by; one half by default, which
            pairs with the default displacement.  For another frequency it is
            ``omega / (2 sin(omega * shift))``, the difference carrying a factor
            of that sine.
        axis: which axis of ``params`` holds the parameters.

    Returns:
        The gradient, shaped like ``params``.

    Raises:
        ValueError: if the parameter axis is not there, or the displacement is
            zero -- a difference taken at the same angle twice is not a rule.

    Example:
        >>> angles = tp.tensor([0.2, -0.4])
        >>> energy = lambda a: tq.energy(build(a), h2)
        >>> tq.param_shift(energy, angles)
    """
    if params.dim() == 0:
        raise ValueError("an angle is a tensor with at least one axis")
    if not -params.dim() <= axis < params.dim():
        raise ValueError(
            f"axis {axis} is out of range for a tensor of {params.dim()} axes"
        )
    shift = math.pi / 2.0 if shift is None else float(shift)
    weight = 0.5 if weight is None else float(weight)
    if shift == 0.0:
        raise ValueError("a displacement of zero measures no difference")
    if weight == 0.0:
        raise ValueError("a weight of zero discards the whole difference")

    positions = tp.arange(params.shape[axis], device=params.device)
    rows: list = []
    for index in range(params.shape[axis]):
        one = (positions == index).to(dtype=params.dtype) * shift
        moved = params + one
        rows.append(weight * (value_fn(moved) - value_fn(params - one)))
    gradient = tp.stack(rows, dim=axis)
    if gradient.shape != params.shape:
        raise ValueError(
            f"the value came back shaped {tuple(rows[0].shape)}, which with a "
            f"parameter axis of {axis} would put the gradient at "
            f"{tuple(gradient.shape)} rather than {tuple(params.shape)}; a value "
            "is expected with the parameter axis removed"
        )
    return gradient
