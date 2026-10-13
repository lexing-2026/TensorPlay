"""Native fast paths for the state simulation primitives.

Two things in the module have a compiled counterpart: applying a local unitary
to the targeted wires, and turning amplitudes into measurement probabilities.
Both read the flat layout ``[batch, 2**n]``, where qubit ``w`` owns bit
``n-1-w`` of the amplitude index -- the same indexing a folded view of a
``[batch, 2, 2, ...]`` state produces, so a gate becomes a single pass over the
amplitudes with no permuted intermediate copy.

A gate's gradient is carried by the decomposition instead (permute, batched
matrix multiply, permute back), which composes from existing primitives and
stays differentiable in its parameters.  These paths therefore only take over
when no gradient is being recorded; otherwise the decomposition runs.
"""
from typing import Optional, Sequence

import tensorplay as tp

from tensorplay import _C

__all__ = [
    "native_apply_gate",
    "native_state_probs",
    "records_grad",
]

# Resolved once: the extension is a build product, so a tree built before these
# ops existed simply runs the decomposition instead of failing at import.
_APPLY_GATE = getattr(_C, "_quantum_apply_gate", None)
_STATE_PROBS = getattr(_C, "_quantum_state_probs", None)


def records_grad(*tensors: Optional[tp.Tensor]) -> bool:
    """Whether a graph is being recorded for any of these tensors."""
    if not tp.is_grad_enabled():
        return False
    return any(
        isinstance(item, tp.Tensor) and item.requires_grad for item in tensors
    )


def native_apply_gate(
    state: tp.Tensor, mat: tp.Tensor, wires: Sequence[int]
) -> Optional[tp.Tensor]:
    """Applies a local unitary to a flat state, or returns ``None``.

    Args:
        state: amplitudes shaped ``[batch, 2**n]``.
        mat: the unitary, ``[2**k, 2**k]`` or ``[batch, 2**k, 2**k]``.
        wires: the target qubits, ascending.

    Returns:
        The new amplitudes in the same layout, or ``None`` when the compiled
        path is unavailable or a gradient is being recorded.
    """
    if _APPLY_GATE is None or records_grad(state, mat):
        return None
    flat = state.reshape(state.shape[0], -1).contiguous()
    return _APPLY_GATE(flat, mat.contiguous(), [int(wire) for wire in wires])


def native_state_probs(state: tp.Tensor) -> Optional[tp.Tensor]:
    """Squares the amplitudes of a flat state, or returns ``None``."""
    if _STATE_PROBS is None or records_grad(state):
        return None
    flat = state.reshape(state.shape[0], -1).contiguous()
    return _STATE_PROBS(flat)