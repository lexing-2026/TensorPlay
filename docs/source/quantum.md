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

Two-qubit and wider gates read their arguments in the order given: the first
wire named is the first index bit of the gate matrix, which is what makes the
control the control. `apply_cnot(state, 1, 0)` drives wire 0 from wire 1, and
`apply_mcx(state, [0, 2, 1])` uses wires 0 and 2 as controls and flips wire 1.

## States

```python
tq.zeros_state(3)            # |000>
tq.plus_state(2)             # (|00> + |01> + |10> + |11>) / 2
tq.basis_state(5, 3)         # |101> on 3 qubits
tq.state_from_bases([0, 3], 2)  # (|00> + |11>) / sqrt(2)
tq.product_state([a, b])     # the tensor product of per-qubit vectors
tq.random_state(3)           # uniform on the sphere of the register
```

`normalize` rescales to unit norm, `norm` reports it, and `inner` is the
conjugate overlap between two states. `to_density` and `density_to_state` cross
between the ket and density views, and `partial_trace` drops qubits from a
density matrix.

`global_phase` answers the question that comes up whenever two operators are
computed two different ways: do they differ by a phase and nothing else? It
reads that one factor off the whole overlap, so no single entry decides it, and
raises if the two are not the same operator up to a phase.

```python
phase = tq.global_phase(fused_gate, dense_product)   # 1.0 if they are the same
```

## Gates

`apply_gate` is the primitive: a matrix applied to a list of wires. Everything
else — the named gates, controlled gates, circuits — is a layer on top of it.

```python
state = tq.zeros_state(2)
state = tq.apply_h(state, 0)          # Hadamard on wire 0
state = tq.apply_cnot(state, 0, 1)    # control wire 0, target wire 1
state = tq.apply_gate(state, tq.rot_y_matrix(0.3), [1])
```

Parameter-free gates: `apply_x`, `apply_y`, `apply_z`, `apply_i`, `apply_h`,
`apply_s`, `apply_sdg`, `apply_sx`, `apply_t`, `apply_tdg`. Parameterized gates
take the parameter as the last argument: `apply_rx`, `apply_ry`, `apply_rz`,
`apply_p`, `apply_u2`, `apply_u3`, `apply_rot` (any axis), `apply_multirz` (one
angle on a list of wires), `apply_global_phase` (scales the whole state).
Two-qubit gates: `apply_cx`/`apply_cnot`, `apply_cy`, `apply_cz`, `apply_swap`,
`apply_iswap`, `apply_dcx` (the same controlled X with the target written
first), `apply_ecr` (echoed cross resonance), the controlled square roots
`apply_cs`, `apply_csdg` and `apply_csx`, and the controlled parameter gates
`apply_cp`, `apply_crx`, `apply_cry`, `apply_crz`. Wider gates:
`apply_toffoli`/`apply_ccx` (three wires), `apply_ccz` (two controls and a phase),
`apply_rccx` (the relative-phase Toffoli), `apply_rc3x` (its four-wire version),
`apply_cswap` (control plus two targets), and `apply_mcx`, which takes a control
list and flips the remaining wire.

The chemistry pair gates mix two wires while leaving the register's excitation
count alone, which is what a particle-number-preserving ansatz needs:
`apply_xx_plus_yy`, `apply_xx_minus_yy` and `apply_single_excitation`, each
taking the pair of wires and an angle.

`random_unitary` draws a gate uniformly from all the unitaries on a register,
which is what a test or a baseline average wants rather than a matrix of random
entries: the distribution is uniform over the operators themselves, phase and
all.

Two-body rotations take a pair of wires and an angle the same way:
`apply_rxx`, `apply_ryy`, `apply_rzz` and `apply_rzx`. Each is
`exp(-i * angle * P / 2)` for a two-wire Pauli product `P` — `ZZ` is diagonal, so
it only adds a phase, which is the gate an Ising Hamiltonian is written with. The
first three treat the pair alike; `apply_rzx` puts `Z` on the wire named first.

```python
state = tq.apply_rzz(state, [0, 1], 0.4)   # a phase on each basis state
state = tq.apply_rxx(state, [0, 1], 0.4)   # mixes |01> and |10>
```

The matching matrix builders are available on their own — `h_matrix`,
`rot_x_matrix`, `swap_matrix`, `mcx_matrix` and the rest — so a gate can be
built, inspected, or composed without applying it. `controlled_matrix` wraps any
single-qubit unitary as a control, `pauli_matrix` builds a Pauli word, and
`dagger` is the conjugate transpose.

### Control readings

A control decides whether the gate fires by what it reads, and all ones is only
the default. `control_values` says what each control has to hold, in wire order:

```python
tq.apply_mcx(tq.zeros_state(3), [0, 1, 2])                      # fires on |11.>
tq.apply_mcx(tq.zeros_state(3), [0, 1, 2], control_values=[0, 0])  # fires when both are 0
```

`apply_controlled(state, mat, wires, control_values=...)` is the same idea for
an arbitrary operator, with the controls leading the wire list.

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

### Ansatz templates

Layer templates lay out a trainable circuit instead of hand by hand.
`nlocal_circuit` is the general form: `reps` blocks of "rotate every wire, then
entangle", a closing rotation layer, and one trainable angle per rotation gate.

```python
tq.real_amplitudes(3, reps=2)          # RY rotations and CNOT pairs
tq.efficient_su2(3, reps=2)            # RY and RZ rotations, CNOT pairs
tq.nlocal_circuit(3, ("ry", "p"), reps=1, entanglement="circular")
```

The entanglement pattern is one of `"linear"`, `"reverse_linear"`,
`"circular"` or `"full"`; `entanglement_pairs(n, pattern)` returns the pairs
directly. Angles arrive as module parameters in `(-pi, pi)`, so the template
can go straight to an optimizer:

```python
qc = tq.real_amplitudes(3, reps=2)
loss = tq.expectation(qc.state(), "ZI")
loss.backward()
len(list(qc.parameters()))            # 9: three wires, two blocks, one closing layer
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

### Reading values back out of shots

A Pauli word that touches an `X` or a `Y` wire cannot be read off a plain
histogram until the wires are turned: `apply_measurement_basis` puts the state
into the frame where measuring in Z reports each factor's eigenvalue. The
histogram then averages those signs, which is `expectation_from_counts`.

```python
readable = tq.apply_measurement_basis(state, "XZ")
counts = tq.measure(readable, shots=4096, wires=[0, 2])
tq.expectation_from_counts(counts, "XZ", wires=[0, 2])   # [batch]
```

Words that agree on every wire where both of them act can share one such
rotation: each is a sub-word of the group's product, so a single histogram
reads all of them. `commuting_groups` finds the groups, `joint_expectation`
runs them and returns one estimate per word — fewer settings means fewer shots
for the same noise.

```python
tq.commuting_groups(["XZ", "XI", "ZZ"])   # [['XZ', 'XI'], ['ZZ']]
tq.joint_expectation(state, ["XZ", "XI", "ZZ"], shots=4096)   # [batch, 3]
```

### A readout that lies

A device reports a bit that was prepared wrong now and then, and the two
directions usually differ. A readout matrix records that: row `p` is what gets
reported for a prepared `p`, so it is a distribution over the reported bit.

```python
noisy = tq.readout_error_matrix(0.02)               # 2% in both directions
worse = tq.readout_error_matrix(0.02, 0.05)         # 2% of 1s, 5% of 0s
```

`measure` takes one and decides every shot's reading on its own, which is how a
real readout behaves:

```python
counts = tq.measure(state, shots=4096, readout=noisy)
```

Give one matrix per wire as `[n, 2, 2]` when the wires were calibrated apart.
`apply_readout_error` applies the same map to a histogram or an exact
distribution, so an offline histogram can be corrected — or biased, which is
what it is usually wanted for.

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

### A system that does not close

A channel applies all at once. Noise that acts over time is the generator
`d rho / dt = -i [H, rho] + sum_r (L_r rho L_r^dagger - 1/2 {L_r^dagger L_r, rho})`,
and `Lindbladian` is that generator, held as an object and applied as often as an
integration needs.

```python
generator = tq.Lindbladian(h2, jump_ops=["XI", ("IZ", 0.05)])
final = generator.evolve(rho, time=2.0)
```

The generator itself is never built. Written out densely it is `2**n` times the
size of `rho` squared -- on ten qubits, a matrix of a trillion entries -- so
each term is applied on its own and the integral is formed by composition alone.

A jump operator is a Pauli word, a `(wires, matrix)` pair, or a full-register
matrix, and a bare Pauli word means a rate of one. A rate can be written next to
it, which is the short way to say how fast the error happens:

```python
tq.Lindbladian(h2, [("XI", 0.02), ("ZI", 0.01)])        # Pauli errors at rates
tq.Lindbladian(h2, [([0], tq.amplitude_damping_kraus(0.05)[0]), n_qubits=4])
```

A word of identities and `Z`s is a sign pattern rather than a general matrix, so
its whole dissipator is one factor per entry of `rho` and costs a single pass.
A channel's Kraus operators can be handed in directly as a stack; they generate
the channel once the piece of the identity they leave unused is added to them,
which `_read_jumps` does without being asked.

Two routes integrate the generator, and both keep the trace:

```python
generator.evolve(rho, 2.0, method="rk4")     # fixed fourth-order steps
generator.evolve(rho, 2.0, method="taylor")  # a truncated series of the span
```

`"rk4"` is the one that can follow a Hamiltonian that moves with time, given as a
callable of `t`, and the one that reports intermediate times. `"taylor"` sums a
series of the whole span, which costs several times less per unit of time
covered and lands closer to the exact answer — the default wherever it applies,
short spans and long ones alike.

No span is out of reach for the series. Its terms climb to about `exp(span)`
while the sum they add to stays the size of the state, so one sum can only carry
so far before it cancels itself away in the rounding; past that point the span
is covered by several sums of a size one can carry, in a row. Advancing by `a`
then by `b` is advancing by `a + b`, each half applied to the matrix the one
before left, and the propagator is still never built — a propagator can only be
squared as an operator, and this generator is never built to square.

What an experiment wants is the observables along the way:

```python
states, values = generator.trajectory(
    rho, 2.0, n_sample=101, observables=[h2, "IZ"]
)
```

`values` is `[time, observable, batch]`, real. Without a sample count,
`mesolve` answers with the final matrix on its own; `sesolve` is the closed-system
counterpart, which advances a state rather than its square and so costs `2**n`
per step instead of `4**n`.

```python
tq.sesolve(state, h2, time=2.0, order=2, n_sample=51, observables=[h2])
tq.mesolve(rho, h2, None, time=2.0, n_sample=51, observables=[h2])   # same answer
```

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

The split is exact for commuting terms. Otherwise the order is what decides how
carefully the non-commuting pieces are dropped, and each step is `order` times
the work: a first-order product has an error that falls with the step, the
palindromic second-order one with the square of it, and the fourth-order one --
that scheme composed three times with uneven weights -- with the fourth power.
For small registers, `matrix_exp_evolve` gives the exact propagator from the
dense matrix instead, and `sesolve` picks between the routes.

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

The measurement side stays compiled either way. `expectation`, `probs` and
`marginal_probs` read the amplitudes in one pass and are linear in `|psi|²`, so
inside a graph they hand the backward the arithmetic they would have done
anyway: the gradient of a Pauli word is that word applied to the state, and the
gradient of a distribution is the state times the weights the forward already
computed. A trained circuit therefore pays for one pass forward and one pass per
non-identity factor backward, rather than rebuilding a rotated copy of the state
and threading a graph through it.

### Gradients without a graph

Two situations call for something else. One is a value that is not built out of
differentiable primitives at all: an energy estimated from a finite number of
measurements has nothing to backpropagate through. The other is a circuit deep
enough that backpropagating through every rotation costs more than evaluating
the whole thing twice.

`sample_energy` answers the first — the Hamiltonian measured with as few
measurement settings as its terms allow, optionally through a readout matrix so
the estimate carries the error a device would add:

```python
value = tq.sample_energy(state, h2, shots=4000)
```

`param_shift` answers the second, and answers the first too. For a value that
depends on one angle through `exp(-i θ G)`, the derivative is a difference of
the value at two displaced angles, and it is exact — independent of how deep the
circuit is and of what sits in the middle of it, a sampler included:

```python
def energy(angles):
    return tq.energy(tq.apply_cnot(
        tq.apply_ry(tq.apply_ry(tq.zeros_state(2, batch=angles.shape[0]),
                                0, angles[:, 0]),
                    1, angles[:, 1]), 0, 1), h2)

angles = tp.tensor([[0.2, -0.4]])
gradients = tq.param_shift(energy, angles)          # no graph anywhere
sampled = lambda a: tq.sample_energy(build(a), h2, 2000)
noisy = tq.param_shift(sampled, angles)             # a value with no graph either
```

Every angle gate here is a rotation about half a Pauli word, whose eigenvalues
are `±1/2`, so the default displacement of `π/2` and weight of one half are the
rule for all of them. A generator whose eigenvalues differ by `ω` instead wants
`π / (2 ω)` and a weight of `ω / (2 sin(ω shift))`; both are arguments, so the
rule follows the gate rather than assuming it.

The rule reaches an angle that enters through one rotation and stops there. A
controlled rotation is not one exponential — where the control is unset its
generator reads zero — so the value carries half the frequency as well as the
whole one, and no single displacement reaches both. Those angles want the
differentiated gradient, which covers them.

### Precision

A register's own dtype is what a whole circuit runs at: a double-precision state
stays double through every gate, and a single-precision one stays single even
when a matrix arrives wider. The amplitude a gate produces is only as precise as
the register holding it, so the place to decide is where the state is built. An
angle written as a plain number is read in double precision on the way to the
matrix, which costs nothing and keeps a double-precision register from being fed
a single-precision matrix.

```python
wide = tq.zeros_state(2, dtype=tp.complex128)   # this is what decides
tq.apply_ry(wide, 0, 0.3)                        # double-precision amplitudes
tq.apply_ry(tq.zeros_state(2), 0, 0.3)          # single, whatever the angle says
```


## Reference

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.quantum.Circuit
    tensorplay.quantum.GATE_TABLE
    tensorplay.quantum.qft_circuit
    tensorplay.quantum.real_amplitudes
    tensorplay.quantum.efficient_su2
    tensorplay.quantum.nlocal_circuit
    tensorplay.quantum.entanglement_pairs
    tensorplay.quantum.apply_gate
    tensorplay.quantum.controlled_matrix
    tensorplay.quantum.apply_controlled
    tensorplay.quantum.rot_matrix
    tensorplay.quantum.multirz_matrix
    tensorplay.quantum.multirz_eigvals
    tensorplay.quantum.rxx_matrix
    tensorplay.quantum.ryy_matrix
    tensorplay.quantum.rzz_matrix
    tensorplay.quantum.rzx_matrix
    tensorplay.quantum.globalphase_matrix
    tensorplay.quantum.rccx_matrix
    tensorplay.quantum.dcx_matrix
    tensorplay.quantum.ecr_matrix
    tensorplay.quantum.rc3x_matrix
    tensorplay.quantum.xx_plus_yy_matrix
    tensorplay.quantum.xx_minus_yy_matrix
    tensorplay.quantum.single_excitation_matrix
    tensorplay.quantum.dagger
    tensorplay.quantum.expand_to_register
    tensorplay.quantum.zeros_state
    tensorplay.quantum.plus_state
    tensorplay.quantum.basis_state
    tensorplay.quantum.global_phase
    tensorplay.quantum.Lindbladian
    tensorplay.quantum.lindbladian
    tensorplay.quantum.mesolve
    tensorplay.quantum.sesolve
    tensorplay.quantum.random_state
    tensorplay.quantum.random_unitary
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
    tensorplay.quantum.expectation_from_density
    tensorplay.quantum.apply_pauli_word
    tensorplay.quantum.apply_measurement_basis
    tensorplay.quantum.expectation_from_counts
    tensorplay.quantum.commuting_groups
    tensorplay.quantum.joint_expectation
    tensorplay.quantum.sample_indices
    tensorplay.quantum.counts_from_indices
    tensorplay.quantum.measure
    tensorplay.quantum.readout_error_matrix
    tensorplay.quantum.apply_readout_error
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
    tensorplay.quantum.param_shift
    tensorplay.quantum.sample_energy
```

## Where to go next

- [Complex numbers](complex_numbers.md) — the dtype behind every state.
- [Autograd](autograd.md) — how gradients reach a circuit's parameters.