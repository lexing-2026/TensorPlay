"""Circuit composition: an ordered list of gates over a fixed qubit register.

A circuit holds no state of its own.  Applying it threads a state tensor
through the gates in order and returns a new tensor, so the same circuit can
be applied to many states, to a batch of them, or reused inside a larger
autograd graph.  Registering the circuit as a module is what makes its
parameters visible to an optimizer.
"""
import math
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import tensorplay as tp

from . import _gates as _g
from ._helpers import DEFAULT_C_DTYPE, ordered_wires
from ._state import zeros_state

__all__ = [
    "Circuit",
    "GATE_TABLE",
    "efficient_su2",
    "entanglement_pairs",
    "expand_to_register",
    "nlocal_circuit",
    "qft_circuit",
    "real_amplitudes",
]


def expand_to_register(mat: tp.Tensor, wires: Sequence[int], n_qubits: int) -> tp.Tensor:
    """Embed a gate matrix into a register-wide unitary on ``n_qubits`` qubits.

    The gate is lifted by Kronecker product onto the leading qubits and then
    permuted onto ``wires``, so the result acts on the whole register.  The
    matrix spells its index bits in the order the wires were named, so those
    bits are first permuted onto ascending order to match where the lifted
    gate lands.
    """
    target = list(wires)
    if len(target) > n_qubits:
        raise ValueError(f"a {len(target)}-wire gate does not fit in {n_qubits} qubit(s)")
    side = mat.shape[-1]
    if side != 1 << len(target):
        raise ValueError(
            f"a {side}x{side} matrix does not act on {len(target)} qubit(s)"
        )

    order = [target.index(wire) for wire in sorted(target)]
    if any(position != source for position, source in enumerate(order)):
        leading = mat.dim() - 2
        k = len(target)
        axes = list(range(leading))
        axes += [leading + order[bit] for bit in range(k)]
        axes += [leading + k + order[bit] for bit in range(k)]
        shape = mat.shape
        mat = mat.reshape(*([1] * leading), *([2] * (2 * k))).permute(axes).reshape(shape)
    target = sorted(target)

    if len(target) == n_qubits:
        return mat

    identity = tp.eye(1 << (n_qubits - len(target)), dtype=mat.dtype, device=mat.device)
    wide = tp.kron(mat, identity)
    free = [axis for axis in range(n_qubits) if axis not in target]
    # The product lays the gate's own axes down first, in the order they were
    # named, then the untouched wires; where[wire] is where wire `wire` sits.
    where = [0] * n_qubits
    for position, wire in enumerate(target):
        where[wire] = position
    for position, wire in enumerate(free):
        where[wire] = len(target) + position
    order = where + [n_qubits + axis for axis in where]
    return wide.reshape((2,) * (2 * n_qubits)).permute(order).reshape((1 << n_qubits,) * 2)


class _Gate:
    """One entry of the gate table: how many wires and parameters it takes."""

    __slots__ = ("name", "n_wires", "n_params", "apply", "matrix")

    def __init__(self, name, n_wires, n_params, apply, matrix):
        self.name = name
        self.n_wires = n_wires
        self.n_params = n_params
        self.apply = apply
        self.matrix = matrix


def _on_wire(builder: Callable) -> Callable:
    def apply(state, wires, *params):
        return _g.apply_gate(state, builder(*params).to(device=state.device), wires)

    return apply


def _build_gate_table() -> dict:
    table = {
        "i": _Gate("i", 1, 0, _on_wire(lambda: _g.i_matrix()), _g.i_matrix),
        "x": _Gate("x", 1, 0, _on_wire(lambda: _g.x_matrix()), _g.x_matrix),
        "y": _Gate("y", 1, 0, _on_wire(lambda: _g.y_matrix()), _g.y_matrix),
        "z": _Gate("z", 1, 0, _on_wire(lambda: _g.z_matrix()), _g.z_matrix),
        "h": _Gate("h", 1, 0, _on_wire(lambda: _g.h_matrix()), _g.h_matrix),
        "sx": _Gate("sx", 1, 0, _on_wire(lambda: _g.sx_matrix()), _g.sx_matrix),
        "s": _Gate("s", 1, 0, _on_wire(lambda: _g.s_matrix()), _g.s_matrix),
        "sdg": _Gate("sdg", 1, 0, _on_wire(lambda: _g.sdg_matrix()), _g.sdg_matrix),
        "t": _Gate("t", 1, 0, _on_wire(lambda: _g.t_matrix()), _g.t_matrix),
        "tdg": _Gate("tdg", 1, 0, _on_wire(lambda: _g.tdg_matrix()), _g.tdg_matrix),
        "rx": _Gate("rx", 1, 1, _on_wire(_g.rot_x_matrix), _g.rot_x_matrix),
        "ry": _Gate("ry", 1, 1, _on_wire(_g.rot_y_matrix), _g.rot_y_matrix),
        "rz": _Gate("rz", 1, 1, _on_wire(_g.rot_z_matrix), _g.rot_z_matrix),
        "p": _Gate("p", 1, 1, _on_wire(_g.phase_matrix), _g.phase_matrix),
        "u3": _Gate("u3", 1, 3, _on_wire(_g.u3_matrix), _g.u3_matrix),
    }

    def _controlled(unitary, n_controls=1):
        return lambda *args, **kwargs: _g.controlled_matrix(unitary(*args, **kwargs), n_controls)

    table["cx"] = _Gate(
        "cx", 2, 0, lambda s, w, *p: _g.apply_cnot(s, *w), _controlled(_g.x_matrix)
    )
    table["cy"] = _Gate(
        "cy", 2, 0, lambda s, w, *p: _g.apply_gate(
            s, _g.controlled_matrix(_g.y_matrix(c_dtype=s.dtype)), w), _controlled(_g.y_matrix)
    )
    table["cz"] = _Gate(
        "cz", 2, 0, lambda s, w, *p: _g.apply_cz(s, *w), _controlled(_g.z_matrix)
    )
    table["swap"] = _Gate(
        "swap", 2, 0, lambda s, w, *p: _g.apply_swap(s, *w), _g.swap_matrix
    )
    table["iswap"] = _Gate(
        "iswap", 2, 0, lambda s, w, *p: _g.apply_iswap(s, *w), _g.iswap_matrix
    )
    table["cswap"] = _Gate(
        "cswap", 3, 0, lambda s, w, *p: _g.apply_cswap(s, *w), _controlled(_g.swap_matrix)
    )
    table["cp"] = _Gate(
        "cp", 2, 1,
        lambda s, w, *p: _g.apply_gate(
            s, _g.controlled_matrix(_g.phase_matrix(*p).to(device=s.device)).to(s.dtype), w),
        lambda *p: _controlled(_g.phase_matrix)(*p),
    )
    table["crx"] = _Gate(
        "crx", 2, 1,
        lambda s, w, *p: _g.apply_gate(
            s, _g.controlled_matrix(_g.rot_x_matrix(*p).to(device=s.device)).to(s.dtype), w),
        lambda *p: _controlled(_g.rot_x_matrix)(*p),
    )
    table["cry"] = _Gate(
        "cry", 2, 1,
        lambda s, w, *p: _g.apply_gate(
            s, _g.controlled_matrix(_g.rot_y_matrix(*p).to(device=s.device)).to(s.dtype), w),
        lambda *p: _controlled(_g.rot_y_matrix)(*p),
    )
    table["crz"] = _Gate(
        "crz", 2, 1,
        lambda s, w, *p: _g.apply_gate(
            s, _g.controlled_matrix(_g.rot_z_matrix(*p).to(device=s.device)).to(s.dtype), w),
        lambda *p: _controlled(_g.rot_z_matrix)(*p),
    )
    table["ccx"] = _Gate(
        "ccx", 3, 0, lambda s, w, *p: _g.apply_toffoli(s, *w),
        lambda: _g.mcx_matrix(2),
    )
    table["mcx"] = _Gate(
        "mcx", None, 0, lambda s, w, *p: _g.apply_mcx(s, w),
        lambda n: _g.mcx_matrix(n - 1),
    )
    return table


GATE_TABLE = _build_gate_table()

_INVERSES = {
    "s": "sdg",
    "sdg": "s",
    "t": "tdg",
    "tdg": "t",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "sx": "sx",
    "i": "i",
    "swap": "swap",
    "iswap": "iswap",
}


class Circuit(tp.nn.Module):
    """An ordered gate list over a register of ``n_qubits`` qubits.

    Args:
        n_qubits: register width.
        batch: batch size of the states :meth:`forward` starts from.
        device: where the untouched start state is created.
        dtype: complex dtype of the start state.

    Example:
        >>> bell = Circuit(2)
        >>> bell.h(0)
        >>> bell.cx(0, 1)
        >>> state = bell()
    """

    def __init__(self, n_qubits: int, *, batch: int = 1, device=None,
                 dtype=DEFAULT_C_DTYPE, name: Optional[str] = None):
        super().__init__()
        if n_qubits < 1:
            raise ValueError(f"a register holds at least one qubit, got {n_qubits}")
        self.n_qubits = n_qubits
        self.batch = batch
        self.device = device
        self.dtype = dtype
        if name is not None:
            self.name = name
        self._entries: List[Tuple[str, Tuple[int, ...], tuple]] = []
        # Gate parameters handed over as tensors are attached to the module, so
        # `parameters()` finds them and an optimizer can move them.
        self._param_names: dict = {}

    # -- construction -------------------------------------------------------

    def add(self, gate: str, wires, *params) -> "Circuit":
        """Append a gate; returns ``self`` so calls can be chained.

        Args:
            gate: a name from :data:`GATE_TABLE`.
            wires: target qubits, ascending.
            params: gate parameters -- floats, or tensors when the gate is
                parameterized and the parameters should carry gradients.
        """
        key = gate.lower()
        if key not in GATE_TABLE:
            raise KeyError(
                f"unknown gate {gate!r}; available: {sorted(GATE_TABLE)}"
            )
        entry = GATE_TABLE[key]
        wires_tuple = tuple(ordered_wires(wires, self.n_qubits))
        if entry.n_wires is not None and len(wires_tuple) != entry.n_wires:
            raise ValueError(
                f"gate {key!r} acts on {entry.n_wires} qubit(s), got {len(wires_tuple)}"
            )
        if len(params) != entry.n_params:
            raise ValueError(
                f"gate {key!r} takes {entry.n_params} parameter(s), got {len(params)}"
            )
        self._entries.append((key, wires_tuple, tuple(params)))
        for index, param in enumerate(params):
            self._attach_parameter(param, key, wires_tuple, index)
        return self

    def _attach_parameter(self, param, gate: str, wires, index: int) -> None:
        """Register a gate parameter with the module, once per tensor.

        The entry keeps the object it was handed, so a step that writes the
        parameter in place is seen by the gate that owns it.  The name says
        which gate the parameter belongs to, which is what shows up in a
        ``state_dict``.
        """
        if not isinstance(param, tp.Tensor):
            return
        key = id(param)
        if key in self._param_names:
            return
        base = "gate_{}_{}_{}".format(gate, "_".join(str(wire) for wire in wires), index)
        name, attempt = base, 1
        while hasattr(self, name):
            name = f"{base}_{attempt}"
            attempt += 1
        self._param_names[key] = name
        setattr(self, name, param)

    def h(self, wire):
        """Append a Hadamard gate."""
        return self.add("h", wire)

    def x(self, wire):
        """Append an X gate."""
        return self.add("x", wire)

    def y(self, wire):
        """Append a Y gate."""
        return self.add("y", wire)

    def z(self, wire):
        """Append a Z gate."""
        return self.add("z", wire)

    def rx(self, wire, theta):
        """Append an X-axis rotation."""
        return self.add("rx", wire, theta)

    def ry(self, wire, theta):
        """Append a Y-axis rotation."""
        return self.add("ry", wire, theta)

    def rz(self, wire, theta):
        """Append a Z-axis rotation."""
        return self.add("rz", wire, theta)

    def p(self, wire, theta):
        """Append a phase shift."""
        return self.add("p", wire, theta)

    def u3(self, wire, theta, phi, lam):
        """Append a general single-qubit gate."""
        return self.add("u3", wire, theta, phi, lam)

    def s(self, wire):
        """Append a quarter turn around Z."""
        return self.add("s", wire)

    def sdg(self, wire):
        """Append the inverse quarter turn around Z."""
        return self.add("sdg", wire)

    def t(self, wire):
        """Append an eighth turn around Z."""
        return self.add("t", wire)

    def tdg(self, wire):
        """Append the inverse eighth turn around Z."""
        return self.add("tdg", wire)

    def sx(self, wire):
        """Append a square root of X."""
        return self.add("sx", wire)

    def cx(self, control, target):
        """Append a controlled X (CNOT)."""
        return self.add("cx", (control, target))

    def cz(self, control, target):
        """Append a controlled Z."""
        return self.add("cz", (control, target))

    def cp(self, control, target, theta):
        """Append a controlled phase shift."""
        return self.add("cp", (control, target), theta)

    def crx(self, control, target, theta):
        """Append a controlled X-axis rotation."""
        return self.add("crx", (control, target), theta)

    def cry(self, control, target, theta):
        """Append a controlled Y-axis rotation."""
        return self.add("cry", (control, target), theta)

    def crz(self, control, target, theta):
        """Append a controlled Z-axis rotation."""
        return self.add("crz", (control, target), theta)

    def swap(self, wire_a, wire_b):
        """Append a swap."""
        return self.add("swap", (wire_a, wire_b))

    def iswap(self, wire_a, wire_b):
        """Append an iswap."""
        return self.add("iswap", (wire_a, wire_b))

    def cswap(self, control, wire_a, wire_b):
        """Append a swap controlled by a third wire."""
        return self.add("cswap", (control, wire_a, wire_b))

    def ccx(self, control_a, control_b, target):
        """Append a Toffoli gate."""
        return self.add("ccx", (control_a, control_b, target))

    def mcx(self, *wires):
        """Append a multi-controlled X; the last wire is the target."""
        return self.add("mcx", wires)

    def barrier(self):
        """A no-op marker: keeps gate boundaries visible when transpiling."""
        self._entries.append(("barrier", (), ()))
        return self

    # -- inspection ---------------------------------------------------------

    def gates(self) -> List[Tuple[str, Tuple[int, ...], tuple]]:
        """The gate list, in order."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Tuple[str, Tuple[int, ...], tuple]]:
        return iter(self._entries)

    def __repr__(self) -> str:
        body = ", ".join(
            name if not wires else f"{name}{list(wires)}" for name, wires, _ in self._entries
        )
        return f"Circuit(n_qubits={self.n_qubits}, gates=[{body}])"

    # -- execution ----------------------------------------------------------

    def apply(self, state: tp.Tensor) -> tp.Tensor:
        """Thread ``state`` through the gate list and return the result."""
        for name, wires, params in self._entries:
            if name == "barrier":
                continue
            gate = GATE_TABLE[name]
            state = gate.apply(state, wires, *params)
        return state

    def forward(self, state: Optional[tp.Tensor] = None, batch: Optional[int] = None) -> tp.Tensor:
        """Run the circuit, starting from ``|0...0>`` when no state is given."""
        if state is None:
            size = self.batch if batch is None else batch
            state = zeros_state(self.n_qubits, size, device=self.device, dtype=self.dtype)
        return self.apply(state)

    def state(self, batch: Optional[int] = None) -> tp.Tensor:
        """The state this circuit prepares from the all-zero state."""
        return self.forward(None, batch)

    def unitary(self) -> tp.Tensor:
        """The dense unitary this circuit applies to a register.

        Built by multiplying the register-wide gate matrices, so the register
        has to stay small enough for a dense ``2**n`` by ``2**n`` result.
        """
        dim = 1 << self.n_qubits
        total = tp.eye(dim, dtype=self.dtype, device=self.device)
        for name, wires, params in self._entries:
            if name == "barrier":
                continue
            gate = GATE_TABLE[name]
            local = gate.matrix(*(params or (len(wires),))) if gate.n_wires is None else gate.matrix(*params)
            # Gates act on the state from the left, so the product grows on the
            # left too and the first gate ends up rightmost.
            total = expand_to_register(local.to(total.dtype), wires, self.n_qubits) @ total
        return total

    def inverse(self) -> "Circuit":
        """The circuit that undoes this one: reversed order, adjoint gates."""
        out = Circuit(self.n_qubits, batch=self.batch, device=self.device, dtype=self.dtype)
        for name, wires, params in reversed(self._entries):
            if name == "barrier":
                continue
            flipped = _INVERSES.get(name, name)
            if flipped in ("rx", "ry", "rz", "p", "cp", "crx", "cry", "crz"):
                params = tuple(-tp.as_tensor(p) for p in params)
            elif flipped == "u3":
                theta, phi, lam = (tp.as_tensor(p) for p in params)
                params = (-theta, -lam, -phi)
            out.add(flipped, wires, *params)
        return out

    def expectation(self, word: str) -> tp.Tensor:
        """``<psi|P|psi>`` for the Pauli word, on the state this circuit prepares."""
        from ._measurement import expectation

        return expectation(self.state(), word)


def qft_circuit(n_qubits: int, inverse: bool = False) -> Circuit:
    """The quantum Fourier transform as a gate sequence.

    Each qubit takes a Hadamard plus one controlled phase from every qubit above
    it, which is the ``O(n**2)`` construction rather than the dense matrix; the
    closing swaps undo the bit reversal the rotations leave behind.
    """
    circuit = Circuit(n_qubits)
    for wire in range(n_qubits):
        circuit.h(wire)
        for partner in range(wire + 1, n_qubits):
            circuit.add("cp", (wire, partner), 2.0 * math.pi / (1 << (partner - wire + 1)))
    for wire in range(n_qubits // 2):
        circuit.add("swap", (wire, n_qubits - 1 - wire))
    return circuit.inverse() if inverse else circuit


_ENTANGLEMENT_PATTERNS = ("linear", "reverse_linear", "circular", "full")


def entanglement_pairs(n_qubits: int, pattern: str = "reverse_linear") -> List[Tuple[int, int]]:
    """The ordered ``(control, target)`` pairs of an entanglement pattern.

    Args:
        n_qubits: register width.
        pattern: ``"linear"`` walks up the register, ``"reverse_linear"`` walks
            down, ``"circular"`` adds the closing pair, and ``"full"`` takes
            every unordered pair.

    Returns:
        One pair per two-qubit gate the pattern applies.
    """
    if pattern not in _ENTANGLEMENT_PATTERNS:
        raise ValueError(
            f"unknown entanglement pattern {pattern!r}; "
            f"available: {list(_ENTANGLEMENT_PATTERNS)}"
        )
    if pattern == "linear":
        return [(wire, wire + 1) for wire in range(n_qubits - 1)]
    if pattern == "reverse_linear":
        return [(wire + 1, wire) for wire in range(n_qubits - 1)]
    if pattern == "circular":
        pairs = [(wire, wire + 1) for wire in range(n_qubits - 1)]
        # The closing pair is only a second direction through an existing edge,
        # so a single-wire register has nothing to close.
        return pairs + ([(n_qubits - 1, 0)] if n_qubits > 1 else [])
    return [(low, high) for low in range(n_qubits) for high in range(low + 1, n_qubits)]


def _angle(device) -> tp.Tensor:
    """One trainable rotation angle, drawn from the half-open range (-pi, pi)."""
    return tp.nn.Parameter(tp.uniform_(tp.empty(1, device=device), -math.pi, math.pi))


def nlocal_circuit(
    n_qubits: int,
    rotations: Sequence[str],
    reps: int = 3,
    entanglement: str = "reverse_linear",
    skip_final_rotation_layer: bool = False,
    *,
    device=None,
) -> Circuit:
    """Repeats a rotation layer and an entanglement layer over a register.

    The layout is the hardware-efficient one: ``reps`` blocks of "rotate every
    wire, then entangle", followed by one more rotation layer unless it is
    skipped.  Each rotation gate gets its own trainable angle, so the result is
    ready to hand to an optimizer.

    Args:
        n_qubits: register width.
        rotations: the rotation gates of one layer, in the order they run.
        reps: how many rotation/entanglement blocks to repeat; at least one.
        entanglement: the pair pattern from :func:`entanglement_pairs`.
        skip_final_rotation_layer: leave off the closing rotation layer.
        device: where the angles are created.

    Returns:
        The assembled circuit, whose ``parameters()`` are the rotation angles.
    """
    circuit = Circuit(n_qubits, device=device)
    if reps < 1:
        raise ValueError(f"an ansatz repeats at least once, got {reps}")
    pairs = entanglement_pairs(n_qubits, entanglement)

    def rotate() -> None:
        for gate in rotations:
            for wire in range(n_qubits):
                circuit.add(gate, wire, _angle(device))

    for _ in range(reps):
        rotate()
        for control, target in pairs:
            circuit.cx(control, target)
    if not skip_final_rotation_layer:
        rotate()
    return circuit


def real_amplitudes(
    n_qubits: int,
    reps: int = 3,
    entanglement: str = "reverse_linear",
    skip_final_rotation_layer: bool = False,
    *,
    device=None,
) -> Circuit:
    """Y rotations interleaved with CNOT pairs, the variational starting point.

    The state it prepares stays real for real angles, which is why the template
    uses ``RY`` alone: half the parameters of a general rotation layer would be
    dead weight.

    Args:
        n_qubits: register width.
        reps: how many rotation/entanglement blocks to repeat.
        entanglement: the pair pattern from :func:`entanglement_pairs`.
        skip_final_rotation_layer: leave off the closing rotation layer.
        device: where the angles are created.

    Returns:
        The assembled circuit, whose ``parameters()`` are the rotation angles.
    """
    return nlocal_circuit(
        n_qubits,
        ("ry",),
        reps,
        entanglement,
        skip_final_rotation_layer,
        device=device,
    )


def efficient_su2(
    n_qubits: int,
    reps: int = 3,
    entanglement: str = "reverse_linear",
    skip_final_rotation_layer: bool = False,
    *,
    device=None,
) -> Circuit:
    """Y and Z rotations interleaved with CNOT pairs.

    A layer here is every ``RY`` followed by every ``RZ``, so the two rotation
    types stay in blocks rather than alternating wire by wire.  That gives a
    state complex enough to reach the ground states the real-only ansatz cannot
    see, at twice the parameters per layer.

    Args:
        n_qubits: register width.
        reps: how many rotation/entanglement blocks to repeat.
        entanglement: the pair pattern from :func:`entanglement_pairs`.
        skip_final_rotation_layer: leave off the closing rotation layer.
        device: where the angles are created.

    Returns:
        The assembled circuit, whose ``parameters()`` are the rotation angles.
    """
    return nlocal_circuit(
        n_qubits,
        ("ry", "rz"),
        reps,
        entanglement,
        skip_final_rotation_layer,
        device=device,
    )
