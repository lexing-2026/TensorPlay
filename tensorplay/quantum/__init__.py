"""Quantum state simulation on top of the tensor primitives.

The module simulates circuits on a classical device: a state tensor carries one
amplitude per basis entry, gates are dense unitaries applied to the wires they
act upon, and measurement turns the amplitudes into a distribution that can be
evaluated exactly or sampled.

Layouts
    A state is ``[batch, 2, 2, ...]`` -- one dimension of size 2 per qubit after
    the batch, qubit 0 leading, so qubit 0 is the most significant bit of the
    amplitude index.  Density matrices use ``[batch, 2**n, 2**n]``, with the
    same bit order on the row and the column.  Complex dtypes are
    :data:`tensorplay.complex64` and :data:`tensorplay.complex128`.

Building blocks
    :func:`apply_gate` is the primitive every gate goes through; the named
    gates (``apply_x``, ``apply_cnot``, ...) and :class:`Circuit` are layers on
    top of it, and all of them stay differentiable in their parameters.

    :func:`probs` and :func:`expectation` read a state without disturbing it.
    :func:`measure` samples the distribution where the state already lives.

    Channels (:func:`apply_channel` and the named wrappers) act on density
    matrices; :class:`Hamiltonian` adds energy evaluation and time evolution,
    exact per term and by Trotter product.

Example:
    >>> from tensorplay import quantum as tq
    >>> bell = tq.Circuit(2)
    >>> bell.h(0)
    >>> bell.cx(0, 1)
    >>> state = bell()
    >>> tq.probs(state)
    tensor([[0.5000, 0.0000, 0.0000, 0.5000]])
    >>> tq.expectation(state, "ZZ")
    tensor([1.])
"""
from ._circuit import (
    GATE_TABLE,
    Circuit,
    efficient_su2,
    entanglement_pairs,
    expand_to_register,
    nlocal_circuit,
    qft_circuit,
    real_amplitudes,
)
from ._gates import (
    apply_ccx,
    apply_controlled,
    apply_cnot,
    apply_cp,
    apply_crx,
    apply_cry,
    apply_crz,
    apply_cswap,
    apply_cx,
    apply_cy,
    apply_cz,
    apply_gate,
    apply_global_phase,
    apply_h,
    apply_i,
    apply_iswap,
    apply_mcx,
    apply_multirz,
    apply_p,
    apply_pauli_rotation,
    apply_rccx,
    apply_rot,
    apply_rx,
    apply_ry,
    apply_rz,
    apply_s,
    apply_sdg,
    apply_swap,
    apply_sx,
    apply_t,
    apply_tdg,
    apply_toffoli,
    apply_u2,
    apply_u3,
    apply_x,
    apply_y,
    apply_z,
    controlled_matrix,
    dagger,
    globalphase_matrix,
    h_matrix,
    i_matrix,
    iswap_matrix,
    mcx_matrix,
    multirz_eigvals,
    multirz_matrix,
    p_matrix,
    pauli_matrix,
    pauli_rotation_matrix,
    phase_matrix,
    qft_matrix,
    rccx_matrix,
    rot_matrix,
    rot_x_matrix,
    rot_y_matrix,
    rot_z_matrix,
    s_matrix,
    sdg_matrix,
    swap_matrix,
    sx_matrix,
    t_matrix,
    tdg_matrix,
    u2_matrix,
    u3_matrix,
    x_matrix,
    y_matrix,
    z_matrix,
)
from ._hamiltonian import (
    Hamiltonian,
    energy,
    matrix_exp_evolve,
    pauli_string_matrix,
    trotter_evolve,
)
from ._measurement import (
    apply_measurement_basis,
    apply_pauli_word,
    commuting_groups,
    counts_from_indices,
    density_probs,
    expectation,
    expectation_from_counts,
    joint_expectation,
    marginal_probs,
    measure,
    probs,
    sample_indices,
)
from ._noise import (
    amplitude_damping_kraus,
    apply_amplitude_damping,
    apply_bit_flip,
    apply_channel,
    apply_depolarizing,
    apply_phase_flip,
    bit_flip_kraus,
    depolarizing_kraus,
    phase_flip_kraus,
)
from ._state import (
    basis_state,
    density_to_state,
    fidelity,
    flatten,
    inner,
    norm,
    normalize,
    partial_trace,
    plus_state,
    product_state,
    state_from_bases,
    to_density,
    unflatten,
    zeros_state,
)

__all__ = [
    # circuit
    "Circuit",
    "GATE_TABLE",
    "expand_to_register",
    "efficient_su2",
    "entanglement_pairs",
    "nlocal_circuit",
    "qft_circuit",
    "real_amplitudes",
    # state
    "basis_state",
    "density_to_state",
    "fidelity",
    "flatten",
    "inner",
    "norm",
    "normalize",
    "partial_trace",
    "plus_state",
    "product_state",
    "state_from_bases",
    "to_density",
    "unflatten",
    "zeros_state",
    # gates: primitive
    "apply_gate",
    "controlled_matrix",
    "dagger",
    # gates: matrices
    "globalphase_matrix",
    "h_matrix",
    "i_matrix",
    "iswap_matrix",
    "mcx_matrix",
    "multirz_eigvals",
    "multirz_matrix",
    "p_matrix",
    "pauli_matrix",
    "pauli_rotation_matrix",
    "phase_matrix",
    "qft_matrix",
    "rccx_matrix",
    "rot_matrix",
    "rot_x_matrix",
    "rot_y_matrix",
    "rot_z_matrix",
    "s_matrix",
    "sdg_matrix",
    "swap_matrix",
    "sx_matrix",
    "t_matrix",
    "tdg_matrix",
    "u2_matrix",
    "u3_matrix",
    "x_matrix",
    "y_matrix",
    "z_matrix",
    # gates: named applications
    "apply_ccx",
    "apply_controlled",
    "apply_cnot",
    "apply_cp",
    "apply_crx",
    "apply_cry",
    "apply_crz",
    "apply_cswap",
    "apply_cx",
    "apply_cy",
    "apply_cz",
    "apply_global_phase",
    "apply_h",
    "apply_i",
    "apply_iswap",
    "apply_mcx",
    "apply_multirz",
    "apply_p",
    "apply_pauli_rotation",
    "apply_rccx",
    "apply_rot",
    "apply_rx",
    "apply_ry",
    "apply_rz",
    "apply_s",
    "apply_sdg",
    "apply_swap",
    "apply_sx",
    "apply_t",
    "apply_tdg",
    "apply_toffoli",
    "apply_u2",
    "apply_u3",
    "apply_x",
    "apply_y",
    "apply_z",
    # measurement
    "apply_measurement_basis",
    "apply_pauli_word",
    "commuting_groups",
    "counts_from_indices",
    "density_probs",
    "expectation",
    "expectation_from_counts",
    "joint_expectation",
    "marginal_probs",
    "measure",
    "probs",
    "sample_indices",
    # noise
    "amplitude_damping_kraus",
    "apply_amplitude_damping",
    "apply_bit_flip",
    "apply_channel",
    "apply_depolarizing",
    "apply_phase_flip",
    "bit_flip_kraus",
    "depolarizing_kraus",
    "phase_flip_kraus",
    # hamiltonian
    "Hamiltonian",
    "energy",
    "matrix_exp_evolve",
    "pauli_string_matrix",
    "trotter_evolve",
]
