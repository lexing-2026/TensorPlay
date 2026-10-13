# Quantum simulation

`tensorplay.quantum` simulates quantum circuits on the tensor hardware. A
state is a tensor of amplitudes, gates are matrices applied to the qubits they
act on, and measurement turns amplitudes into a distribution you can evaluate
exactly or sample.

```python
import tensorplay as tp
from tensorplay import quantum as tq

bell = tq.Circuit(2)
bell.h(0)
bell.cx(0, 1)

state = bell()
tq.probs(state)             # [[0.5, 0., 0., 0.5]] — |00> and |11>
tq.expectation(state, "ZZ") # [1.] — the pair is perfectly correlated
tq.measure(state, shots=1000)
```

## Layout

A state is shaped `[batch, 2, 2, ...]` — one dimension of size 2 per qubit,
after the batch. **Qubit 0 leads**, so qubit 0 is the most significant bit of the
amplitude index: applying X to wire 0 of `|00>` gives `|10>`, which is index 2.

| Layout | Shape | Meaning |
| --- | --- | --- |
| State | `[batch, 2, 2, ...]` | one amplitude per qubit dimension |
| Flat state | `[batch, 2**n]` | the same amplitudes, qubit dimensions folded |
| Density matrix | `[batch, 2**n, 2**n]` | row index the ket, column index the bra |

Complex dtypes are `tensorplay.complex64` and `tensorplay.complex128`; a state
keeps whichever one it was created with. A batch dimension of 1 is always
present, so a single state is a valid batch of one.

Qubit 0 is the most significant bit throughout — in `marginal_probs`, in
`measure` with a `wires` list, and in the argument index of a Pauli word.

## States

```python
tq.zeros_state(3)            # |000>
tq.plus_state(2)             # (|00> + |01> + |10> + |11>) / 2
tq.basis_state(5, 3)         # |101> on 3 qubits
tq.state_from_bases([0, 3], 2)  # (|00> + |11>) / sqrt(2)
tq.product_state([a, b])     # the tensor product of per-qubit vectors
```

`normalize` rescales to unit norm, `norm` reports it, and `inner` is the
conjugate overlap between two states. `to_density` and `density_to_state` cross
between the ket and density views, and `partial_trace` drops qubits from a
density matrix.

## Gates

`apply_gate` is the primitive: a matrix applied to a list of wires. Everything
else — the named gates, controlled gates, circuits — is a layer on top of it.

```python
state = tq.zeros_state(2)
state = tq.apply_h(state, 0)          # Hadamard on wire 0
state = tq.apply_cnot(state, 0, 1)    # control wire 0, target wire 1
state = tq.apply_gate(state, tq.rot_y_matrix(0.3), [1])
```

Parameter-free gates: `apply_x`, `apply_y`, `apply_z`, `apply_h`, `apply_s`,
`apply_sx`, `apply_t`. Parameterized gates take the parameter as the last
argument: `apply_rx`, `apply_ry`, `apply_rz`, `apply_p`, `apply_u3`. Two-qubit
gates: `apply_cx`/`apply_cnot`, `apply_cz`, `apply_swap`, `apply_iswap`. Wider
gates: `apply_toffoli` (three wires), `apply_cswap` (control plus two targets),
and `apply_mcx`, which takes a control list and flips the remaining wire.

The matching matrix builders are available on their own — `h_matrix`,
`rot_x_matrix`, `swap_matrix`, `mcx_matrix` and the rest — so a gate can be
built, inspected, or composed without applying it. `controlled_matrix` wraps any
single-qubit unitary as a control, `pauli_matrix` builds a Pauli word, and
`dagger` is the conjugate transpose.

### Batched parameters

A gate matrix may carry one entry per batch item, which gives every state in the
batch its own parameter:

```python
theta = tp.tensor([0.0, 3.14159])          # per batch item
states = tq.zeros_state(1, batch=2)
tq.apply_ry(states, 0, theta)               # first qubit stays 0, second flips
```

## Circuits

A `Circuit` is an ordered gate list over a fixed register. It holds no state:
calling it starts from `|0...0>`, and `apply` threads a state you provide
through the same gates.

```python
qc = tq.Circuit(3)
qc.h(0)
qc.rx(1, 0.4)
qc.cx(0, 1)
qc.barrier()             # a marker for transpilation; changes nothing
qc.cx(1, 2)

state = qc()             # from |000>
state = qc(some_state)   # from a state you choose
state = qc(batch=64)     # 64 copies, all following the same gates
```

`add` takes any gate name as a string, so gates without a method are still
reachable; `gates()` returns the recorded list, and `len(qc)` counts it. The
methods cover the named gates — `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`,
`sx`, `rx`, `ry`, `rz`, `p`, `u3`, `cx`, `cz`, `cp`, `crx`, `cry`, `crz`,
`swap`, `iswap`, `cswap`, `ccx` and `mcx`.

`unitary()` multiplies the gates into the dense register-wide matrix, and
`inverse()` returns the circuit that undoes this one — reversed order, with each
gate replaced by its adjoint. `qft_circuit(n)` builds the Fourier transform as a
gate sequence rather than a dense matrix.

```python
qc.unitary().shape       # (8, 8) on 3 qubits
qc.inverse()(qc())       # back to |000>
```

## Measurement

`probs` is the squared amplitude of every basis entry — the exact measurement
distribution.

```python
tq.probs(state)                        # [batch, 2**n]
tq.marginal_probs(state, [0, 2])       # the listed qubits, in the order given
tq.expectation(state, "XIZ")           # <psi|P|psi>, real, per batch item
```

`expectation` applies the Pauli word one factor at a time, so its cost grows
with the state rather than with its square. `apply_pauli_word` gives `P|psi>`
itself.

`measure` samples the distribution, where the state already lives:

```python
counts = tq.measure(state, shots=4096)          # [batch, 2**n]
counts = tq.measure(state, shots=1024, wires=[0])  # [batch, 2]
```

Sampling walks the cumulative distribution with one uniform per shot, so a
sampled batch never leaves the device. `sample_indices` returns the raw draws
and `counts_from_indices` tallies them if you want the two steps separately.

## Noise

Channels act on density matrices, given as Kraus operators. Each one is a set
of local matrices `K_r`, and applying it means `rho -> sum_r K_r rho K_r^dagger`.

```python
rho = tq.to_density(state)
rho = tq.apply_bit_flip(rho, 0.01, wire=0)
rho = tq.apply_phase_flip(rho, 0.02, wire=1)
rho = tq.apply_amplitude_damping(rho, 0.05, wire=2)
rho = tq.apply_depolarizing(rho, 0.01, wire=0)
```

`bit_flip_kraus`, `phase_flip_kraus`, `amplitude_damping_kraus` and
`depolarizing_kraus` return the operator sets, and `apply_channel` takes any set
you build yourself, on any list of wires. A single-qubit set acts on one wire;
a set that covers several wires is the register-wide product of the individual
operators:

```python
kraus = tq.bit_flip_kraus(0.1)
rho = tq.apply_channel(rho, kraus, [0])

both = tp.stack([tp.kron(operator, tp.eye(2, dtype=operator.dtype)) for operator in kraus])
rho = tq.apply_channel(rho, both, [0, 1])
```

Because every Kraus operator is local, the cost stays linear in the register
size: no channel superoperator is ever built.

## Hamiltonians and time evolution

A `Hamiltonian` is a weighted sum of Pauli words.

```python
h2 = tq.Hamiltonian([
    (-1.0523732, "II"),
    (0.3979374, "IZ"),
    (-0.3979374, "ZI"),
    (-0.1809312, "XX"),
])
tq.energy(state, h2)     # <psi|H|psi>, real, per batch item
h2.matrix()              # the dense matrix, on the wires the words span
```

Every Pauli word squares to the identity, so `exp(-i * t * c * P)` is a
rotation about that word, `cos(t * c) I - i sin(t * c) P`, with a closed form.
`pauli_rotation_matrix` builds it, and `trotter_evolve` applies one rotation per
term in sequence:

```python
evolved = tq.trotter_evolve(state, h2, time=1.5, order=4)
```

The split is exact for commuting terms; otherwise the error shrinks as the
order grows, at a proportional cost. For small registers, `matrix_exp_evolve`
gives the exact propagator from the dense matrix instead.

## Gradients

Every gate is differentiable in its parameters, so a circuit is differentiable
in whatever tensors you hand it — `tq.energy` included. Registering the
parameters as `nn.Parameter` is all it takes to optimize a circuit: a parameter
handed to a gate becomes one of the circuit's own, named after the gate it
belongs to.

```python
import tensorplay.nn as nn

h2 = tq.Hamiltonian([(-1.0523732, "II"), (0.3979374, "IZ"),
                     (-0.3979374, "ZI"), (-0.1809312, "XX")])

qc = tq.Circuit(2)
theta0 = nn.Parameter(tp.tensor(0.2))
theta1 = nn.Parameter(tp.tensor(-0.4))
qc.ry(0, theta0)
qc.ry(1, theta1)
qc.cx(0, 1)

optimizer = tp.optim.SGD(qc.parameters(), lr=0.05)
loss = tq.energy(qc(), h2).sum()
loss.backward()
optimizer.step()
```

Inside a recorded graph a gate is evaluated as a permutation, a batched matrix
multiply, and the inverse permutation — all differentiable primitives, with the
gradient carried through them. Outside one, the compiled kernel applies the gate
in a single pass over the amplitudes.

## Reference

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.quantum.Circuit
    tensorplay.quantum.GATE_TABLE
    tensorplay.quantum.qft_circuit
    tensorplay.quantum.apply_gate
    tensorplay.quantum.controlled_matrix
    tensorplay.quantum.dagger
    tensorplay.quantum.expand_to_register
    tensorplay.quantum.zeros_state
    tensorplay.quantum.plus_state
    tensorplay.quantum.basis_state
    tensorplay.quantum.state_from_bases
    tensorplay.quantum.product_state
    tensorplay.quantum.flatten
    tensorplay.quantum.unflatten
    tensorplay.quantum.norm
    tensorplay.quantum.normalize
    tensorplay.quantum.inner
    tensorplay.quantum.fidelity
    tensorplay.quantum.to_density
    tensorplay.quantum.density_to_state
    tensorplay.quantum.partial_trace
    tensorplay.quantum.probs
    tensorplay.quantum.density_probs
    tensorplay.quantum.marginal_probs
    tensorplay.quantum.expectation
    tensorplay.quantum.apply_pauli_word
    tensorplay.quantum.sample_indices
    tensorplay.quantum.counts_from_indices
    tensorplay.quantum.measure
    tensorplay.quantum.apply_channel
    tensorplay.quantum.apply_bit_flip
    tensorplay.quantum.apply_phase_flip
    tensorplay.quantum.apply_amplitude_damping
    tensorplay.quantum.apply_depolarizing
    tensorplay.quantum.bit_flip_kraus
    tensorplay.quantum.phase_flip_kraus
    tensorplay.quantum.amplitude_damping_kraus
    tensorplay.quantum.depolarizing_kraus
    tensorplay.quantum.Hamiltonian
    tensorplay.quantum.pauli_string_matrix
    tensorplay.quantum.energy
    tensorplay.quantum.trotter_evolve
    tensorplay.quantum.matrix_exp_evolve
```

## Where to go next

- [Complex numbers](complex_numbers.md) — the dtype behind every state.
- [Autograd](autograd.md) — how gradients reach a circuit's parameters.