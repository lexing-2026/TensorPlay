"""The quantum module on the accelerator.

Each check builds the same construction on the host and on the device and
compares the amplitudes: two kernels reading the same layout have to agree, and
the host side is the reference the CPU tests already pin down.
"""

import math

import pytest

import tensorplay as tp
from tensorplay import _C
from tensorplay import quantum as tq


pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA not available")


def _flat(state):
    return state.reshape(state.shape[0], -1)


def _on_device(tensor):
    return tensor.to(tp.device("cpu"))


# Each entry builds a state from scratch on the device it is handed, so the
# device kernel does the work rather than receiving a host result.
CASES = [
    ("hadamard", lambda d: tq.apply_h(tq.zeros_state(1, device=d), 0)),
    ("bell", lambda d: tq.apply_cnot(tq.apply_h(tq.zeros_state(2, device=d), 0), 0, 1)),
    ("rotation", lambda d: tq.apply_gate(tq.plus_state(3, device=d), tq.rot_y_matrix(0.4), [1])),
    ("wide_rotation", lambda d: tq.apply_gate(
        tq.plus_state(5, device=d), tq.controlled_matrix(tq.rot_z_matrix(-0.9)), [0, 4]
    )),
    ("swap", lambda d: tq.apply_swap(tq.basis_state(9, 4, device=d), 1, 3)),
    ("toffoli", lambda d: tq.apply_toffoli(tq.basis_state(6, 3, device=d), 0, 1, 2)),
    ("mcx", lambda d: tq.apply_gate(tq.zeros_state(4, device=d), tq.mcx_matrix(3), [0, 1, 2, 3])),
    ("qft", lambda d: tq.qft_circuit(3)()),
    ("mixed", lambda d: tq.apply_cnot(tq.apply_h(tq.apply_ry(tq.zeros_state(3, device=d), 2, 0.35), 0), 0, 1)),
]


@pytest.mark.parametrize("name,build", CASES, ids=[case[0] for case in CASES])
def test_device_matches_host(name, build):
    tp.testing.assert_close(
        _on_device(_flat(build("cuda"))), _flat(build("cpu")), atol=1e-5, rtol=1e-5
    )


def test_probs_match_the_host():
    state = tq.apply_h(tq.zeros_state(1), 0)
    tp.testing.assert_close(
        tq.probs(state.to(tp.device("cuda"))).to(tp.device("cpu")),
        tq.probs(state),
        atol=1e-6,
        rtol=1e-6,
    )


def test_probs_in_double_precision_match_the_host():
    state = tq.apply_h(tq.zeros_state(1, dtype=tp.complex128), 0)
    on_device = tq.probs(state.to(tp.device("cuda")))
    assert on_device.dtype == tp.float64
    tp.testing.assert_close(on_device.to(tp.device("cpu")), tq.probs(state), atol=1e-12, rtol=0)


def test_batched_gate_matches_the_host():
    theta = tp.tensor([0.0, math.pi, 1.0])
    out = _flat(
        tq.apply_gate(tq.zeros_state(1, batch=3, device=tp.device("cuda")),
                      tq.rot_y_matrix(theta), [0])
    )
    tp.testing.assert_close(
        _on_device(out),
        _flat(tq.apply_gate(tq.zeros_state(1, batch=3), tq.rot_y_matrix(theta), [0])),
        atol=1e-6,
        rtol=1e-6,
    )


def test_widest_gate_matches_the_host():
    state = tq.zeros_state(8)
    mat = tq.mcx_matrix(7, c_dtype=tp.complex64)
    out = _flat(tq.apply_gate(state.to(tp.device("cuda")), mat, list(range(8))))
    tp.testing.assert_close(
        _on_device(out), _flat(tq.apply_gate(state, mat, list(range(8))))
    )


def test_expectation_matches_the_host():
    state = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    tp.testing.assert_close(
        tq.expectation(state.to(tp.device("cuda")), "XX").to(tp.device("cpu")),
        tq.expectation(state, "XX"),
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_pauli_expectation_with_y_factors_matches_the_host():
    # The overlap's imaginary half is what a Y factor lands in, so a word with
    # one in it is the case that tells a right answer from a conjugated one.
    state = tq.apply_s(tq.apply_h(tq.zeros_state(1), 0), 0)
    tp.testing.assert_close(
        _on_device(tq.expectation(state.to(tp.device("cuda")), "Y")),
        tq.expectation(state, "Y"),
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_pauli_expectation_of_a_batch_matches_the_host():
    states = tq.plus_state(3, batch=4)
    for wire in range(3):
        states = tq.apply_s(states, wire)
    for word in ("XXX", "ZZZ", "YZI", "IXY", "III"):
        tp.testing.assert_close(
            _on_device(tq.expectation(states.to(tp.device("cuda")), word)),
            tq.expectation(states, word),
            atol=1e-5,
            rtol=1e-5,
        )


def test_sampling_stays_on_the_device():
    # |+0> puts half its shots on |00> and half on |10>.
    state = tq.apply_h(tq.zeros_state(2), 0)
    counts = tq.measure(state.to(tp.device("cuda")), shots=4096)
    assert counts.device.type == "cuda"
    counts = counts.to(tp.device("cpu"))
    tp.testing.assert_close(counts.sum(), tp.tensor(4096.0))
    tp.testing.assert_close(counts[0, 0], tp.tensor(2048.0), atol=200.0, rtol=0)
    tp.testing.assert_close(counts[0, 2], tp.tensor(2048.0), atol=200.0, rtol=0)


def test_readout_error_matches_the_host():
    state = tq.basis_state(2, 3)
    matrix = tq.readout_error_matrix(0.1, 0.2)
    tp.testing.assert_close(
        _on_device(
            tq.apply_readout_error(
                tq.probs(state.to(tp.device("cuda"))),
                matrix.to(tp.device("cuda")),
            )
        ),
        tq.apply_readout_error(tq.probs(state), matrix),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sampled_readout_error_matches_the_exact_map():
    # Every wire is turned over with chance p, so a prepared |00> stays put with
    # probability (1 - p) squared.
    state = tq.zeros_state(2)
    matrix = tq.readout_error_matrix(0.15)
    device = tp.device("cuda")
    counts = tq.measure(state.to(device), 20000, readout=matrix.to(device))
    assert counts.device.type == "cuda"
    exact = tq.apply_readout_error(tq.probs(state), matrix)
    counts = counts.to(tp.device("cpu"))
    tp.testing.assert_close(counts.sum(), tp.tensor(20000.0), atol=1e-4, rtol=1e-6)
    tp.testing.assert_close(counts / counts.sum(), exact, atol=0.02, rtol=0)


def test_the_marginal_keeps_its_column_order_on_the_device():
    state = tq.apply_h(tq.zeros_state(3), 0)
    on_device = tq.marginal_probs(state.to(tp.device("cuda")), [2, 0, 1])
    tp.testing.assert_close(
        _on_device(on_device),
        tq.marginal_probs(state, [2, 0, 1]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_marginal_covers_a_wide_register_on_the_device():
    state = tq.basis_state(1, 10)
    counts = tq.measure(state.to(tp.device("cuda")), 512)
    assert counts.shape == (1, 1024)
    counts = counts.to(tp.device("cpu"))
    tp.testing.assert_close(counts[0, 1], tp.tensor(512.0), atol=1e-4, rtol=1e-6)


def test_the_expectation_gradient_matches_the_host_on_the_device():
    # The gradient is worked out on the device and comes back from there, so the
    # node has to travel as far as the forward does.
    real = [0.3, 0.5, -0.2, 0.7]
    imag = [-0.2, 0.7, 0.4, -0.6]

    def grads(device):
        real_leaf = tp.tensor(real, device=device, requires_grad=True)
        imag_leaf = tp.tensor(imag, device=device, requires_grad=True)
        state = tq.unflatten(tp.complex(real_leaf, imag_leaf).reshape(1, -1), 2)
        tq.expectation(state, "XY").sum().backward()
        return _on_device(real_leaf.grad), _on_device(imag_leaf.grad)

    on_device = grads(tp.device("cuda"))
    on_host = grads(tp.device("cpu"))
    for got, want in zip(on_device, on_host):
        tp.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_the_marginal_gradient_matches_the_host_on_the_device():
    angles = [
        tp.tensor(0.37 + 0.2 * wire, requires_grad=True) for wire in range(3)
    ]

    def grads(device):
        state = tq.zeros_state(3, device=device)
        for wire in range(3):
            state = tq.apply_ry(state, wire, angles[wire].to(device))
        columns = tq.marginal_probs(state, [2, 0])
        (columns * tp.tensor([0.3, 0.7, 1.1, 1.9], device=device)).sum().backward()
        return [_on_device(angle.grad) for angle in angles]

    for got, want in zip(grads(tp.device("cuda")), grads(tp.device("cpu"))):
        tp.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_channel_matches_the_host():
    density = tq.to_density(tq.apply_h(tq.zeros_state(2), 0))
    tp.testing.assert_close(
        _on_device(tq.apply_depolarizing(density.to(tp.device("cuda")), 0.3, 1)),
        tq.apply_depolarizing(density, 0.3, 1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_trotter_matches_the_host():
    state = tq.plus_state(3)
    hamiltonian = tq.Hamiltonian([(0.7, "XIZ"), (-0.4, "ZZX"), (0.2, "III")])
    tp.testing.assert_close(
        _on_device(
            _flat(tq.trotter_evolve(state.to(tp.device("cuda")), hamiltonian, 0.5, order=4))
        ),
        _flat(tq.trotter_evolve(state, hamiltonian, 0.5, order=4)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gradient_matches_the_host():
    # One rotation angle per batch entry, so the batched matrix is exercised.
    hamiltonian = tq.Hamiltonian([(1.0, "ZI"), (-0.5, "XX")])
    gradients = {}
    for device in ("cpu", "cuda"):
        theta = tp.tensor([0.3, -0.2], requires_grad=True)
        state = tq.zeros_state(2, batch=2, device=tp.device(device))
        tq.energy(
            tq.apply_ry(state, 0, theta.to(tp.device(device))), hamiltonian
        ).sum().backward()
        gradients[device] = theta.grad.to(tp.device("cpu")).clone()
    tp.testing.assert_close(gradients["cuda"], gradients["cpu"], atol=1e-6, rtol=1e-5)


def test_gate_rejects_a_repeated_wire_on_device():
    flat = tq.flatten(tq.zeros_state(2, device=tp.device("cuda"))).contiguous()
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(
            flat, tp.eye(4, dtype=tp.complex64, device=tp.device("cuda")), [0, 0]
        )


def test_gate_rejects_more_wires_than_either_backend_allows():
    flat = tq.flatten(tq.zeros_state(9, device=tp.device("cuda"))).contiguous()
    mat = tp.eye(1 << 8, dtype=tp.complex64, device=tp.device("cuda"))
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, mat, list(range(9)))

def test_the_master_equation_matches_the_host_on_the_device():
    # Every pass of the generator is a gate application on the device, so the
    # whole evolution runs there; the answer has to be the host's answer.
    hamiltonian = tq.Hamiltonian([(0.7, "XII"), (-0.4, "IZZ"), (0.3, "IIY")])
    damping = tq.amplitude_damping_kraus(0.2, c_dtype=tp.complex64)[0]
    generator = tq.Lindbladian(hamiltonian, [("XII", 0.5), ([2], damping)], c_dtype=tp.complex64)

    def evolve(device):
        state = tq.plus_state(3, device=device)
        rho = tq.to_density(state)
        return _on_device(generator.evolve(rho, 1.0, n_step=100))

    tp.testing.assert_close(
        evolve(tp.device("cuda")), evolve(tp.device("cpu")), atol=1e-5, rtol=1e-5
    )


def test_the_state_path_matches_the_host_on_the_device():
    hamiltonian = tq.Hamiltonian([(0.6, "XIZ"), (0.35, "IZI")])
    state = tq.apply_h(tq.zeros_state(3, device=tp.device("cuda")), 0)
    on_device = tq.sesolve(state, hamiltonian, 1.0, order=2, n_step=8)
    on_host = tq.sesolve(
        tq.apply_h(tq.zeros_state(3, device=tp.device("cpu")), 0),
        hamiltonian,
        1.0,
        order=2,
        n_step=8,
    )
    tp.testing.assert_close(
        _on_device(on_device), _on_device(on_host), atol=1e-5, rtol=1e-5
    )
