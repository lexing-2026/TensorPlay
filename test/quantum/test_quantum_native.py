"""The compiled state primitives: agreement with the decomposition, contracts.

The gate kernel is the fast path outside a recorded graph; the decomposition
carries gradients.  These check the two produce the same amplitudes, and pin
the argument rules both paths rely on.
"""
import math

import pytest

import tensorplay as tp
from tensorplay import quantum as tq
from tensorplay import _C
from tensorplay.quantum import _native

pytestmark = pytest.mark.skipif(
    not hasattr(_C, "_quantum_apply_gate"), reason="built before the state primitives"
)


def _flat(state):
    return state.reshape(state.shape[0], -1)


def _by_decomposition(state, mat, wires):
    """The gate as the recorded graph computes it, whatever the fast path does."""
    tracked = state.clone().requires_grad_(True)
    return tq.apply_gate(tracked, mat, wires).detach()


def _pairs():
    """A spread of widths, parameter values, and register sizes.

    Each entry names the case, so a failure points at one shape of gate.
    """
    for n_qubits in (1, 2, 3, 5):
        for wire in range(n_qubits):
            yield (
                f"one wire {wire} of {n_qubits}",
                tq.plus_state(n_qubits),
                tq.rot_y_matrix(0.4),
                [wire],
            )
    yield "far wires", tq.plus_state(3), tq.controlled_matrix(tq.rot_x_matrix(-1.1)), [0, 2]
    yield "controlled rotation", tq.plus_state(3), tq.controlled_matrix(tq.rot_z_matrix(0.9)), [1, 2]
    yield "adjacent wires", tq.plus_state(3), tq.swap_matrix(), [0, 1]
    yield "three wires", tq.zeros_state(3), tq.mcx_matrix(2), [0, 1, 2]
    yield "four wires", tq.zeros_state(4), tq.mcx_matrix(3), [0, 1, 2, 3]
    yield "basis state", tq.basis_state(5, 3), tq.swap_matrix(), [0, 2]
    yield "single wire on a basis state", tq.basis_state(5, 3), tq.h_matrix(), [2]


@pytest.mark.parametrize("name,state,mat,wires", list(_pairs()), ids=[case[0] for case in _pairs()])
def test_fast_path_matches_the_decomposition(name, state, mat, wires):
    tp.testing.assert_close(
        tq.apply_gate(state, mat, wires),
        _by_decomposition(state, mat, wires),
        atol=1e-6,
        rtol=1e-6,
    )


def test_fast_path_matches_the_decomposition_in_double_precision():
    state = tq.plus_state(3, dtype=tp.complex128)
    mat = tq.rot_y_matrix(0.4, c_dtype=tp.complex128)
    tp.testing.assert_close(
        tq.apply_gate(state, mat, [1]),
        _by_decomposition(state, mat, [1]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_fast_path_carries_a_per_batch_matrix():
    theta = tp.tensor([0.0, math.pi, 2.0 * math.pi])
    states = tq.zeros_state(1, batch=3)
    out = _flat(tq.apply_gate(states, tq.rot_y_matrix(theta), [0]))
    tp.testing.assert_close(out[0], tp.tensor([1.0, 0.0], dtype=tp.complex64), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[1], tp.tensor([0.0, 1.0], dtype=tp.complex64), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[2], tp.tensor([-1.0, 0.0], dtype=tp.complex64), atol=1e-6, rtol=0)


def test_fast_path_keeps_the_state_shape():
    state = tq.plus_state(4)
    out = tq.apply_gate(state, tq.h_matrix(), [3])
    assert out.shape == state.shape


def test_fast_path_takes_the_wider_dtype():
    state = tq.plus_state(2, dtype=tp.complex64)
    out = tq.apply_gate(state, tq.h_matrix(c_dtype=tp.complex128), [0])
    assert out.dtype == tp.complex128
    tp.testing.assert_close(
        out.reshape(1, -1).to(tp.complex64),
        _flat(tq.apply_gate(state, tq.h_matrix(), [0])),
        atol=1e-6,
        rtol=1e-6,
    )


def test_state_probs_squares_the_amplitudes():
    state = tq.apply_h(tq.zeros_state(1), 0)
    tp.testing.assert_close(tq.probs(state), tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6)


def test_state_probs_of_a_bell_pair():
    bell = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    tp.testing.assert_close(tq.probs(bell), tp.tensor([[0.5, 0.0, 0.0, 0.5]]),
                            atol=1e-6, rtol=1e-6)


def test_state_probs_in_double_precision():
    state = tq.apply_h(tq.zeros_state(1, dtype=tp.complex128), 0)
    out = tq.probs(state)
    assert out.dtype == tp.float64
    tp.testing.assert_close(out, tp.tensor([[0.5, 0.5]], dtype=tp.float64), atol=1e-12, rtol=0)


def test_state_probs_over_a_batch():
    states = tq.plus_state(3, batch=4)
    out = tq.probs(states)
    assert out.shape == (4, 8)
    tp.testing.assert_close(out, tp.full((4, 8), 0.125), atol=1e-6, rtol=1e-6)


def test_gate_rejects_a_matrix_of_the_wrong_width():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, tp.eye(4, dtype=tp.complex64), [0])


def test_gate_rejects_a_repeated_wire():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, tp.eye(4, dtype=tp.complex64), [0, 0])


def test_gate_rejects_a_wire_outside_the_register():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(IndexError):
        _C._quantum_apply_gate(flat, tq.i_matrix(), [2])


def test_gate_rejects_a_state_length_that_is_not_a_whole_register():
    flat = tp.zeros((1, 3), dtype=tp.complex64)
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, tq.i_matrix(), [0])


def test_gate_rejects_a_real_state():
    flat = tp.ones((1, 2), dtype=tp.float32)
    with pytest.raises(NotImplementedError):
        _C._quantum_apply_gate(flat, tq.i_matrix(), [0])


def test_state_probs_rejects_a_real_state():
    with pytest.raises(NotImplementedError):
        _C._quantum_state_probs(tp.ones((1, 2), dtype=tp.float32))


def test_decomposed_path_is_taken_when_a_gradient_is_recorded():
    state = tq.plus_state(2)
    tracked = state.clone().requires_grad_(True)
    tq.apply_gate(tracked, tq.h_matrix(), [0]).real.sum().backward()
    assert tracked.grad is not None
    assert not bool(tp.allclose(tracked.grad, tp.zeros_like(tracked)))


def test_no_grad_takes_the_fast_path():
    state = tq.plus_state(2)
    with tp.no_grad():
        out = tq.apply_gate(state, tq.h_matrix(), [0])
    assert not out.requires_grad
    tp.testing.assert_close(
        out, _by_decomposition(state, tq.h_matrix(), [0]), atol=1e-6, rtol=1e-6
    )


def test_native_helpers_report_absence_rather_than_failing():
    saved = _native._APPLY_GATE
    try:
        _native._APPLY_GATE = None
        assert tq.apply_gate(tq.zeros_state(1), tq.h_matrix(), [0]).shape == (1, 2)
    finally:
        _native._APPLY_GATE = saved


def test_gate_rejects_more_wires_than_either_backend_allows():
    # Eight wires is the widest gate either backend tabulates.
    state = tq.zeros_state(9)
    mat = tp.eye(1 << 9, dtype=tp.complex64)
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(
            tq.flatten(state).contiguous(), mat, list(range(9))
        )


def test_gate_spans_every_width_it_allows():
    # Every control is zero on |0...0>, so the widest gate is inert there --
    # what it checks is that the width is handled at all.
    state = tq.zeros_state(8)
    mat = tq.mcx_matrix(7, c_dtype=tp.complex64)
    out = tq.apply_gate(state, mat, list(range(8)))
    tp.testing.assert_close(_flat(out), _flat(state))
    # With the controls set, the target flips.
    excited = tq.basis_state(254, 8)  # every control set, target clear
    flipped = tq.apply_gate(excited, mat, list(range(8)))
    tp.testing.assert_close(_flat(flipped), _flat(tq.basis_state(255, 8)))