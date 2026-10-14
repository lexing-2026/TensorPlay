"""Open-system evolution: the master equation and its integrators.

An open system is described by a generator rather than a unitary::

    d rho / dt = -i [H, rho] + sum_r ( L_r rho L_r^dagger - 1/2 {L_r^dagger L_r, rho} )

The dense form of that generator is ``2**n`` times the size of ``rho`` squared,
which is out of reach a few wires past the first: on ten qubits it is a matrix
of a trillion entries.  So the generator here is never built.  It is applied term
by term, and the integral is formed by composition alone -- a matrix-free
exponential, which is what keeps open systems affordable at research register
sizes.

Two routes to the same end are offered.  ``"rk4"`` takes small steps of a
fourth-order scheme, follows a generator that moves with time, and is the one to
trust by default.  ``"taylor"`` composes a truncated series with repeated
squaring and needs no step size at all, taking the same handful of generator
applications however long the evolution runs -- the right choice for a stiff
system, where a fixed-step scheme would have to be sized by the fastest decay
rather than by the accuracy wanted.

Everything a term needs is worked out once, when the generator is built: the
matrix of each Pauli word, the product ``L^dagger L`` of each jump operator, and
a single scaling pattern that stands in for every dissipator that happens to be
diagonal.  An application then moves ``rho`` once per surviving term and reads
nothing else.
"""
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import tensorplay as tp

from ._circuit import expand_to_register
from ._gates import apply_gate, dagger, pauli_matrix
from ._hamiltonian import Hamiltonian
from ._helpers import DEFAULT_C_DTYPE, check_pauli_string, pauli_wires
from ._state import flatten, to_density

__all__ = [
    "Lindbladian",
    "dissipator",
    "lindbladian",
    "mesolve",
    "sesolve",
]

# How small the truncated series is asked to leave the result, relative to the
# largest entry it starts from.  Both integrators are far inside the precision
# of the arithmetic they run in, so this is a floor on the model rather than a
# target the output has to meet.
_SERIES_TOLERANCE = 1e-10

# The largest angle any one term turns through in a step of the fourth-order
# route.  That scheme is stable well past this, so the value is set by accuracy
# rather than by stability, and the error of a run falls as its fourth power.
_TARGET_ANGLE = 0.1

# A dense generator applied to a whole register is one matrix multiply per pass;
# past this width the term-by-term route is the only one that stays affordable,
# so a dense Hamiltonian is refused rather than quietly attempted.
_MAX_DENSE_WIRES = 8


def _qubit_count(dim: int, what: str = "matrix side") -> int:
    """The number of qubits a side of length ``dim`` describes."""
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"{dim} does not describe a whole number of qubits ({what})")
    return dim.bit_length() - 1


def _as_density(matrix: tp.Tensor, what: str = "density matrix") -> tp.Tensor:
    """Check that a tensor is a density matrix and report its width."""
    if matrix.dim() != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"a {what} is [batch, 2**n, 2**n], got {tuple(matrix.shape)}"
        )
    _qubit_count(matrix.shape[-1], f"{what} side")
    return matrix


def _pauli_signs(word: str, n_qubits: int, *, device, dtype) -> tp.Tensor:
    """The diagonal of a word of identities and ``Z``s, as a register vector.

    A word with an ``X`` or a ``Y`` in it has no diagonal to speak of and never
    reaches this; the ones that do are pure sign patterns, and that is what makes
    their dissipator a scaling rather than a matrix.

    The vector is laid out along the amplitude index, so axis ``w`` of the
    register -- which is bit ``n - 1 - w`` of that index -- is where wire
    ``w``'s sign belongs.
    """
    signs = tp.ones([2] * n_qubits, device=device, dtype=dtype)
    pair = tp.tensor([1.0, -1.0], device=device, dtype=dtype)
    for wire, letter in enumerate(word):
        if letter != "Z":
            continue
        shape = [1] * n_qubits
        shape[wire] = 2
        signs = signs * pair.reshape(shape)
    return signs.reshape(-1)


def _word_is_diagonal(word: str) -> bool:
    """Whether a Pauli word is a sign pattern rather than a general matrix."""
    return all(letter in "IZ" for letter in word)


def _sample_marks(steps: int, wanted: int) -> set:
    """The step indices at which a run reports.

    The reporting times are evenly spaced across the run, so the steps that land
    on them are worked out once as integers -- the same set on every machine,
    which matters because a floating-point comparison here would put a sample on
    one side of a step on one machine and not the next.
    """
    if wanted <= 1:
        return {steps}
    return {int(round(index * steps / (wanted - 1))) for index in range(wanted)}


def _register_matrix(word: str, n_qubits: int, device) -> tp.Tensor:
    """A Pauli word as a dense matrix over the whole register."""
    wires = pauli_wires(word)
    if not wires:
        return tp.eye(1 << n_qubits, device=device, dtype=DEFAULT_C_DTYPE)
    return expand_to_register(pauli_matrix(word, device=device), wires, n_qubits)


def _observable_reader(observable, device):
    """A reader of one expectation value off a density matrix.

    The words are turned into dense matrices on the first read rather than up
    front, so they come out in the precision of the matrix they are read from
    instead of the default one.

    Args:
        observable: a :class:`Hamiltonian` or a Pauli word.
        device: where the dense form of the words is put.

    Returns:
        A function from a density matrix to a real vector over the batch.
    """
    if isinstance(observable, Hamiltonian):
        terms = [(coeff, word) for coeff, word in observable.terms]
        n_qubits = observable.n_qubits
    elif isinstance(observable, str):
        word = check_pauli_string(observable)
        terms = [(1.0, word)]
        n_qubits = len(word)
    else:
        raise ValueError(
            "an observable is a Hamiltonian or a Pauli word, got "
            f"{type(observable).__name__}"
        )
    built: Optional[list] = None

    def read(density: tp.Tensor) -> tp.Tensor:
        nonlocal built
        if built is None:
            built = [
                (coeff, _register_matrix(word, n_qubits, device, density.dtype))
                for coeff, word in terms
            ]
        total = None
        for coeff, matrix in built:
            piece = coeff * (density * matrix).sum(dim=(-1, -2))
            total = piece if total is None else total + piece
        return total.real

    return read


def _register_matrix(
    word: str, n_qubits: int, device, dtype=DEFAULT_C_DTYPE
) -> tp.Tensor:
    """A Pauli word as a dense matrix over the whole register."""
    wires = pauli_wires(word)
    if not wires:
        return tp.eye(1 << n_qubits, device=device, dtype=dtype)
    local = pauli_matrix(word, device=device, c_dtype=dtype)
    return expand_to_register(local, wires, n_qubits)


def _checked_wires(wires, matrix: tp.Tensor) -> List[int]:
    """Normalize the wires a local jump operator acts on."""
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"a jump operator is square, got {tuple(matrix.shape)}")
    n_wires = _qubit_count(matrix.shape[0], "jump operator side")
    listed = [int(wire) for wire in wires]
    if len(listed) != n_wires or sorted(set(listed)) != listed:
        raise ValueError(
            f"a jump operator on {n_wires} wire(s) needs {n_wires} distinct wires "
            f"in ascending order, got {listed}"
        )
    return listed


def _read_terms(hamiltonian, device, c_dtype):
    """Split a Hamiltonian into Pauli terms or a whole-register matrix.

    A word of identities only is left out.  It commutes with everything, so it
    has no say in the evolution of a density matrix -- it is a global phase on
    the state, and no observable on this side can see one.  ``None`` means no
    Hamiltonian at all, which is a pure dissipation.

    Returns:
        ``(terms, dense, n_qubits)`` where ``terms`` are
        ``(coefficient, matrix, wires)`` triples and ``dense`` is a
        full-register matrix or ``None``.
    """
    if hamiltonian is None:
        return [], None, None
    if isinstance(hamiltonian, Hamiltonian):
        terms: List[Tuple[float, tp.Tensor, List[int]]] = []
        for coeff, word in hamiltonian.terms:
            wires = pauli_wires(word)
            if not wires:
                continue
            matrix = pauli_matrix(word, device=device, c_dtype=c_dtype)
            terms.append((coeff, matrix, wires))
        return terms, None, hamiltonian.n_qubits
    if not isinstance(hamiltonian, tp.Tensor) or hamiltonian.dim() != 2:
        raise ValueError(
            "a Hamiltonian is a Hamiltonian, a square matrix, or a callable of "
            f"time, got {type(hamiltonian).__name__}"
        )
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(
            f"a Hamiltonian is square, got {tuple(hamiltonian.shape)}"
        )
    n_qubits = _qubit_count(hamiltonian.shape[0], "Hamiltonian side")
    if n_qubits > _MAX_DENSE_WIRES:
        raise ValueError(
            f"a dense Hamiltonian may span at most {_MAX_DENSE_WIRES} wires, got "
            f"{n_qubits}; pass a Hamiltonian so its terms can be applied one at a time"
        )
    dense = hamiltonian.to(dtype=c_dtype, device=device)
    return [], dense, n_qubits


def _read_jumps(jump_ops, n_qubits: int, device, c_dtype):
    """Put every jump term into one of the three forms the action knows.

    A jump operator carries its own rate in its size -- ``sqrt(gamma) P`` is the
    operator of a Pauli error happening at rate ``gamma`` -- so a bare Pauli word
    means unit rate.  A rate may also be given alongside the operator, as
    ``(word, rate)`` or ``(wires, matrix, rate)``, which is the shorter way to say
    the same thing.

    Returns:
        ``(scaling, words, operators, n_scaling)``.  ``scaling`` is the single
        factor that stands in for every diagonal dissipator, ``words`` are the
        Pauli words that are not diagonal as ``(rate, matrix, wires)``, and
        ``operators`` are the rest as ``(rate, operator, gram, wires)`` with
        ``gram = L^dagger L`` already formed.  ``n_scaling`` counts the operators
        merged into ``scaling``, which is what tells the generator's rate how much
        decay it carries.
    """
    scaling = None
    n_scaling = 0
    words: List[Tuple[float, tp.Tensor, List[int]]] = []
    operators: List[Tuple[float, tp.Tensor, tp.Tensor, List[int]]] = []
    for entry in jump_ops:
        rate = 1.0
        if isinstance(entry, str):
            word, matrix_wires = check_pauli_string(entry), None
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) in (2, 3)
            and isinstance(entry[0], str)
        ):
            word, matrix_wires = check_pauli_string(entry[0]), None
            rate = _checked_rate(entry[1], entry[0])
        else:
            word = None
            if isinstance(entry, (tuple, list)) and len(entry) in (2, 3):
                matrix_wires, matrix = entry[0], entry[1]
                if len(entry) == 3:
                    rate = _checked_rate(entry[2], "jump operator")
            else:
                matrix_wires, matrix = None, entry
            if not isinstance(matrix, tp.Tensor):
                raise ValueError(
                    "a jump operator is a Pauli word, a (wires, matrix) pair, or a "
                    f"square matrix, got {type(entry).__name__}"
                )
            if matrix.dim() == 3:
                # A stack of jump operators, read one at a time.
                side = _qubit_count(matrix.shape[1], "jump operator side")
                wires = _checked_wires(
                    matrix_wires
                    if matrix_wires is not None
                    else range(side),
                    matrix[0],
                )
                stack = matrix.to(dtype=c_dtype, device=device)
                for operator, wires in _as_lindblad_jumps(stack, wires):
                    gram = dagger(operator) @ operator
                    operators.append((rate, operator, gram, wires))
                continue
            wires = _checked_wires(
                matrix_wires
                if matrix_wires is not None
                else range(_qubit_count(matrix.shape[0], "jump operator side")),
                matrix,
            )
            # L^dagger L does not depend on the density matrix, so it is formed
            # once here rather than on every application.
            operator = matrix.to(dtype=c_dtype, device=device)
            operators.append((rate, operator, dagger(operator) @ operator, wires))
            continue
        if word is None:
            continue
        if len(word) > n_qubits:
            raise ValueError(
                f"jump operator {word!r} spans {len(word)} qubit(s), the generator "
                f"is {n_qubits} wide"
            )
        if _word_is_diagonal(word):
            factor = _diagonal_scaling(
                word,
                n_qubits,
                device,
                dtype=tp.float64 if c_dtype == tp.complex128 else tp.float32,
            )
            factor = factor * rate if rate != 1.0 else factor
            scaling = factor if scaling is None else scaling + factor
            n_scaling += 1
        else:
            wires = pauli_wires(word)
            words.append(
                (rate, pauli_matrix(word, device=device, c_dtype=c_dtype), wires)
            )
    return scaling, words, operators, n_scaling


def _as_lindblad_jumps(stack: tp.Tensor, wires: Sequence[int]):
    """A channel's operators, read as the jumps of a generator.

    A generator is a sum of dissipators ``D[J] = J rho J^dag - {J^dag J, rho}/2``,
    one per jump, and that is how a stack is read here: each operator is one
    jump at its own rate.  The single-operator case is the usual way to name a
    generator -- ``{sqrt(gamma) * |0><1|}`` is exactly an excitation decaying at
    rate ``gamma``, which is the master equation's own form of the statement.

    Nothing is re-weighted and nothing is added, and the reason is worth
    writing down.  A channel is the exponential of a generator, not a generator:
    summing dissipators over a channel's operators gives the channel applied to
    the state *minus the state*, once per operator, so that sum is not the
    exponential of anything.  A channel can be handed in where a step of a
    circuit applies it to a density matrix -- there it belongs, and it is exact
    -- but a generator has to be written as jumps.
    """
    count = stack.shape[0]
    if count == 0:
        raise ValueError("a jump stack needs at least one operator")
    return [(stack[index], wires) for index in range(count)]


def _checked_rate(value, name) -> float:
    """Read a jump rate, which is a number and cannot be negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"a rate for {name!r} is a number, got {type(value).__name__}"
        )
    if value < 0.0:
        raise ValueError(f"a rate cannot be negative, got {value}")
    return float(value)


def _series_span_limit(dtype) -> float:
    """The largest span one sum of a truncated series can carry.

    The terms of the series climb to about ``exp(span)`` while the sum they add
    up to stays of the order of the state, so a span far past the point where
    the arithmetic can tell the terms apart gives an answer made entirely of
    rounding.  Six tenths of the range the precision covers leaves a few digits
    of headroom for that cancellation.

    It is a limit on one *sum*, not on the evolution: a longer span is covered
    by several sums of this size in a row (see :meth:`Lindbladian._compose`),
    which is exact to the precision of each and never builds the propagator.
    """
    real = tp.float64 if dtype == tp.complex128 else tp.float32
    return 0.6 * math.log10(1.0 / float(tp.finfo(real).eps))


def _series_degree(span: float, tolerance: float = _SERIES_TOLERANCE) -> int:
    """How many terms of its own series a generator of size ``span`` needs.

    The series of ``exp`` truncated after ``p`` terms leaves the rest of the
    argument's series behind, bounded by ``a**(p+1) / (p+1)!`` once the argument
    is under ``p + 2``.  The smallest ``p`` whose bound falls under the tolerance
    is found by walking upwards, which costs nothing next to an application.
    """
    if span < 1.0:
        # A span this small is inside the first two terms of its own series; the
        # factorial argument below wants more room than that.
        return 8
    order = int(span)
    target = math.log10(tolerance)
    while True:
        log_span = (order + 1) * math.log10(span)
        log_factorial = math.lgamma(order + 2) / math.log(10.0)
        log_tail = math.log10(max(1.0 - span / (order + 2), 1e-12))
        if log_span - log_factorial - log_tail <= target:
            return order
        order += 1


def _diagonal_scaling(word: str, n_qubits: int, device, dtype) -> tp.Tensor:
    """The factor that stands in for one diagonal jump's whole dissipator.

    For ``L = diag(l)`` the dissipator of ``rho`` is
    ``(l_a conj(l_b) - 1/2 (|l_a|^2 + |l_b|^2)) rho_ab``, which depends on
    nothing but the two indices -- a scaling, with no matrix in it.  For a word
    of identities and ``Z``s the entries are signs, so the squared magnitudes
    are one and the factor is ``l_a l_b - 1``.
    """
    side = 1 << n_qubits
    signs = _pauli_signs(word, n_qubits, device=device, dtype=dtype)
    return signs.reshape([side, 1]) * signs.reshape([1, side]) - 1.0


def _read_width(hamiltonian, jump_ops, given: Optional[int]) -> Optional[int]:
    """The register width, from whatever names one.

    A Hamiltonian spans its own words and a dense matrix spans its own side; a
    word of dissipative operators spans at least as far as the longest of them.
    What is left over -- a time-dependent Hamiltonian that has not been read yet,
    or nothing but local matrices -- has to be told, or waited for.
    """
    widths = []
    if isinstance(hamiltonian, Hamiltonian):
        widths.append(hamiltonian.n_qubits)
    elif isinstance(hamiltonian, tp.Tensor) and hamiltonian.dim() == 2:
        if hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError(f"a Hamiltonian is square, got {tuple(hamiltonian.shape)}")
        widths.append(_qubit_count(hamiltonian.shape[0], "Hamiltonian side"))
    elif hamiltonian is not None and not callable(hamiltonian):
        raise ValueError(
            "a Hamiltonian is a Hamiltonian, a square matrix, None, or a callable "
            f"of time, got {type(hamiltonian).__name__}"
        )
    for entry in jump_ops:
        if isinstance(entry, str):
            widths.append(len(check_pauli_string(entry)))
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) in (2, 3)
            and isinstance(entry[0], str)
        ):
            widths.append(len(check_pauli_string(entry[0])))
    if given is not None:
        widest = max(widths) if widths else 0
        if widest > given:
            raise ValueError(
                f"the terms span {widest} qubit(s), the register is declared "
                f"as {given}"
            )
        return given
    if not widths:
        return None
    # Terms of different lengths are ordinary -- a damping written as "Z" says
    # nothing about the qubits after it, and is read on the first -- so the
    # register is as wide as the widest of them and no wider.
    return max(widths)


class Lindbladian:
    """The generator of a Lindblad master equation, as it acts on matrices.

    The object holds the terms and applies them, so it is built once and can be
    applied as often as an integration needs -- which is where the savings are,
    since every invariant is worked out here rather than on each application.

    Args:
        hamiltonian: the Hamiltonian, as a :class:`Hamiltonian`, as a dense
            ``[2**n, 2**n]`` matrix, as ``None`` for pure dissipation, or as a
            callable of time returning either.  A callable makes the generator
            time-dependent, which the fourth-order route can follow and the
            series route cannot.
        jump_ops: the dissipative terms.  Each is a Pauli word, a local
            ``[2**k, 2**k]`` matrix paired with the wires it acts on, or a
            full-register matrix.  A word of identities and ``Z``s -- the shape a
            phase error takes -- is recognized as the scaling it is and costs one
            pass rather than four.
        n_qubits: the register width, needed only when nothing else names it:
            a time-dependent Hamiltonian with no word to read yet, or nothing but
            local matrices.
        device: where the derived pieces go.
        c_dtype: the complex type of the derived pieces.

    Example:
        >>> generator = Lindbladian(Hamiltonian([(1.0, "ZI")]), ["XI"])
        >>> tuple(generator.apply(to_density(zeros_state(2))).shape)
        (1, 4, 4)
        >>> tuple(Lindbladian(None, ["IZ"]).apply(to_density(zeros_state(2))).shape)
        (1, 4, 4)
    """

    def __init__(
        self,
        hamiltonian: Union[Hamiltonian, tp.Tensor, Callable[[float], Hamiltonian], None],
        jump_ops: Optional[Sequence] = None,
        *,
        n_qubits: Optional[int] = None,
        device=None,
        c_dtype=DEFAULT_C_DTYPE,
    ):
        self.hamiltonian = hamiltonian
        self.device = device
        self.c_dtype = c_dtype
        self._time_dependent = callable(hamiltonian)
        self._entries = list(jump_ops or ())
        # A second copy of the terms in double precision, for the summation that
        # needs it; built on first use and kept, since a run is usually repeated.
        self._wider = None

        self.n_qubits = _read_width(hamiltonian, self._entries, n_qubits)
        if self.n_qubits is None and not self._time_dependent and self._entries:
            raise ValueError(
                "nothing here says how wide the register is: every jump operator "
                "names wires rather than a register, so pass n_qubits"
            )
        if self.n_qubits is not None and not self._time_dependent:
            self._read()
        else:
            # A time-dependent Hamiltonian is only named when the time is known,
            # so its terms are read per evaluation and the width is taken from
            # the first one seen -- or from what the caller declared.
            self._terms, self._dense = [], None
            self._scaling, self._words, self._operators, self._n_scaling = None, [], [], 0
            self._prepared_for = None
            self._rate = None

    def _read(self) -> None:
        """Work out every term once, at the width the register turned out to be."""
        self._terms, self._dense, width = _read_terms(
            self.hamiltonian, self.device, self.c_dtype
        )
        if self.n_qubits is not None and width is not None and width != self.n_qubits:
            raise ValueError(
                f"the Hamiltonian is {width} qubit(s) wide, the register is "
                f"{self.n_qubits}"
            )
        self.n_qubits = width if self.n_qubits is None else self.n_qubits
        self._scaling, self._words, self._operators, self._n_scaling = _read_jumps(
            self._entries, self.n_qubits, self.device, self.c_dtype
        )
        self._rate = self._measure_rate()
        self._prepared_for = None

    # -- the action ----------------------------------------------------------

    def apply(self, density: tp.Tensor, time: Optional[float] = None) -> tp.Tensor:
        """``d rho/dt`` for one density matrix, or for a batch of them.

        Args:
            density: shaped ``[batch, 2**n, 2**n]``.
            time: the instant to evaluate a time-dependent Hamiltonian at;
                ignored otherwise.

        Returns:
            The generator's action, in the same layout.
        """
        _as_density(density)
        batch, dim, _ = density.shape
        if self._time_dependent:
            self._prepare(float(time))
        if self.n_qubits != _qubit_count(dim):
            raise ValueError(
                f"the generator is {self.n_qubits} qubit(s) wide, the density "
                f"matrix describes {_qubit_count(dim)}"
            )
        n_qubits = self.n_qubits

        working = density if tp.is_complex(density) else density.to(dtype=self.c_dtype)
        # The ket axes and the bra axes lie side by side, so a matrix is pushed
        # through either of them by naming the axes it acts on.
        flat = working.reshape((batch, *([2] * (2 * n_qubits))))
        bra = tuple(wire + n_qubits for wire in range(n_qubits))

        out = tp.zeros_like(flat)
        for coeff, matrix, wires in self._terms:
            # P rho and rho P, each read off the matrix unchanged: on the bra
            # side the operator enters transposed, which for a Hermitian one is
            # its conjugate.  The two products cannot be chained -- a matrix
            # pushed through the bra side of the first would land between the
            # two operators instead of beside them.
            bra_wires = tuple(wire + n_qubits for wire in wires)
            left = apply_gate(flat, matrix, wires)
            right = apply_gate(flat, matrix.conj(), bra_wires)
            out = out + (-1j * coeff) * (left - right)
        if self._dense is not None:
            left = apply_gate(flat, self._dense, bra)
            right = apply_gate(flat, self._dense.conj(), bra)
            out = out + (-1j) * (left - right)

        # Every dissipator that is a sign pattern is one scaling, however many of
        # them there are.  The factor is indexed by (ket index, bra index) and the
        # matrix is laid out as one axis per wire per side, so it is read in that
        # shape rather than the square one.
        if self._scaling is not None:
            out = out + flat * self._scaling.reshape([2] * (2 * n_qubits))
        for rate, matrix, wires in self._words:
            turned = apply_gate(
                apply_gate(flat, matrix, wires),
                matrix.conj(),
                tuple(wire + n_qubits for wire in wires),
            )
            # P^dagger P is the identity for a Pauli word, so the anticommutator
            # term is the whole of rho.
            out = out + rate * (turned - flat)
        for rate, operator, gram, wires in self._operators:
            bra_wires = tuple(wire + n_qubits for wire in wires)
            turned = apply_gate(
                apply_gate(flat, operator, wires), operator.conj(), bra_wires
            )
            out = out + rate * (
                turned
                - 0.5 * (
                    apply_gate(flat, gram, wires) + apply_gate(flat, gram, bra_wires)
                )
            )
        return out.reshape((batch, dim, dim))

    def _prepare(self, time: float) -> None:
        """Read the terms of a time-dependent Hamiltonian at one instant."""
        if self._prepared_for == time:
            return
        hamiltonian = self.hamiltonian(time)
        terms, dense, n_qubits = _read_terms(hamiltonian, self.device, self.c_dtype)
        self._terms, self._dense = terms, dense
        if n_qubits is not None and n_qubits != self.n_qubits:
            if self.n_qubits is not None:
                raise ValueError(
                    f"the generator changed width at t={time}: {self.n_qubits} "
                    f"qubit(s) became {n_qubits}"
                )
            self.n_qubits = n_qubits
            (
                self._scaling,
                self._words,
                self._operators,
                self._n_scaling,
            ) = _read_jumps(self._entries, n_qubits, self.device, self.c_dtype)
        self._rate = self._measure_rate()
        self._prepared_for = time

    def _measure_rate(self) -> float:
        """How fast the generator can move a state, per unit of time.

        One unit is one radian of rotation or one unit of decay.  A step count
        and a series' span are chosen from it, so a loose estimate costs a little
        accuracy and nothing else -- but an estimate that is too *small* is not
        merely loose: a series told its span is half what it is cancels away the
        precision it needs.  The commutator reads a term from both sides of the
        matrix, so a Hamiltonian turns twice as fast as its largest coefficient
        says, and a dissipator likewise.
        """
        rate = 2.0 * sum(abs(coeff) for coeff, _, _ in self._terms)
        if self._dense is not None:
            rate += 2.0 * float(self._dense.abs().max())
        # Each dissipator moves a matrix by at most its rate doubled, whether it
        # is written as a matrix or as the scaling it reduces to.
        rate += 2.0 * (self._n_scaling + sum(spec[0] for spec in self._words))
        for spec in self._operators:
            rate += 2.0 * spec[0] * float(spec[1].abs().max())
        return rate

    @property
    def rate(self) -> float:
        """The generator's characteristic rate, in radians or units per time.

        A time-dependent generator has no rate until its terms have been read at
        some instant, which is what asking for one does -- at the start of an
        evolution, that is the same instant every step would begin from.
        """
        if self._rate is None:
            self._prepare(0.0)
        return self._rate or 0.0

    # -- integration ---------------------------------------------------------

    def n_step_for(self, time: float, *, target: float = _TARGET_ANGLE) -> int:
        """A step count that keeps every term turning through ``target``.

        Args:
            time: the span to cover.
            target: the largest angle one term should turn through in a step.

        Returns:
            The number of fourth-order steps, at least one.
        """
        return max(1, int(math.ceil(abs(float(time)) * max(self.rate, 1e-12) / target)))

    def evolve(
        self,
        density: tp.Tensor,
        time: float,
        *,
        method: str = "auto",
        n_step: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> tp.Tensor:
        """Advance a density matrix by ``time`` under this generator.

        Args:
            density: shaped ``[batch, 2**n, 2**n]``.
            time: how far to advance; negative runs backwards.
            method: ``"auto"`` to take the series where it fits and fixed
                fourth-order steps where it does not, ``"rk4"`` for the steps
                alone, or ``"taylor"`` for the series alone -- which needs a
                generator that does not move with time.
            n_step: how many fourth-order steps to take; chosen from the rate
                when omitted.
            dt: the step to take, as an alternative to ``n_step``.

        Returns:
            The evolved density matrix.
        """
        steps, method = self._plan(time, method, n_step, dt)
        return self._run(density, time, method, steps, set())[-1]

    def trajectory(
        self,
        density: tp.Tensor,
        time: float,
        *,
        n_sample: int = 0,
        observables: Sequence = (),
        method: str = "auto",
        n_step: Optional[int] = None,
        dt: Optional[float] = None,
    ):
        """Evolve and report the density matrices along the way.

        This is the shape an experiment wants: the observables read off at a set
        of times, not one matrix at the end of it.

        Args:
            density: shaped ``[batch, 2**n, 2**n]``.
            time: the span to cover.
            n_sample: how many evenly spaced times to report, the first at zero
                and the last at ``time``.  Zero reports only the end point.
            observables: what to read off each reported time, as
                :class:`Hamiltonian` instances or Pauli words.
            method: ``"auto"``, ``"rk4"`` or ``"taylor"``; see :meth:`evolve`.
            n_step: fourth-order steps, or ``None`` to choose from the rate.
            dt: the step to take, as an alternative to ``n_step``.

        Returns:
            ``(states, values)``, where ``states`` is
            ``[n_sample, batch, 2**n, 2**n]`` and ``values`` is
            ``[n_sample, observables, batch]`` and real, or ``None`` when no
            observable was asked for.  With ``n_sample`` at zero only the final
            matrix is reported, so ``states`` has one entry.
        """
        _as_density(density)
        if n_sample < 0:
            raise ValueError(
                f"a trajectory reports at least zero times, got {n_sample}"
            )
        wanted = max(n_sample, 1)
        steps, method = self._plan(
            time, method, n_step, dt, report_interior=n_sample > 1
        )
        marks = _sample_marks(steps, wanted) - {steps}
        collected = self._run(density, time, method, steps, marks)
        states = collected if n_sample else collected[-1:]
        readers = [_observable_reader(item, density.device) for item in observables]
        values = None
        if readers:
            values = tp.stack(
                [
                    tp.stack([reader(matrix) for matrix in states], dim=0)
                    for reader in readers
                ],
                dim=1,
            )
        return tp.stack(states), values

    def _plan(
        self,
        time: float,
        method: str,
        n_step: Optional[int],
        dt: Optional[float],
        report_interior: bool = False,
    ):
        """Settle on the method and the step count for one run.

        ``"auto"`` takes the series wherever it can be composed and the
        step-by-step route where it cannot: on a generator that does not move
        with time the series costs several times less per unit of time covered
        *and* comes out closer to the exact answer, so reaching for the stepping
        first would be paying seven times as much for a worse result.

        A run that has to report times in between takes the steps as well.  The
        series is one composition, so it has no interior to report from -- the
        only points it can answer for are the ends.
        """
        if method == "auto":
            method = (
                "rk4" if self._time_dependent or report_interior else "taylor"
            )
        if method not in ("rk4", "taylor"):
            raise ValueError(
                "an integration method is 'auto', 'rk4' or 'taylor', got "
                f"{method!r}"
            )
        if method == "taylor" and self._time_dependent:
            raise ValueError(
                "a time-dependent generator cannot be composed into a series; "
                "use method='rk4'"
            )
        if dt is not None and n_step is not None:
            raise ValueError("give either dt or n_step, not both")
        if method == "taylor":
            # A series has no steps to choose: one composition is the whole run,
            # and the samples are read off it afterwards.
            return 1, method
        if dt is not None:
            return max(1, int(math.ceil(abs(float(time)) / abs(float(dt))))), method
        return (n_step if n_step is not None else self.n_step_for(time)), method

    def _run(
        self,
        density: tp.Tensor,
        time: float,
        method: str,
        steps: int,
        marks: set,
    ) -> List[tp.Tensor]:
        """Integrate over ``steps`` of the chosen method, reporting at ``marks``.

        The states at the marked steps are returned in order, followed by the
        final one whether or not it was marked.
        """
        collected: List[tp.Tensor] = []
        current = density
        if 0 in marks:
            collected.append(density)
        if method == "taylor":
            return collected + [self._compose(current, float(time))]
        step = float(time) / steps
        for index in range(steps):
            current = self._rk4_step(current, step, float(time) * index / steps)
            if (index + 1) in marks:
                collected.append(current)
        return collected + [current]

    def _rk4_step(self, density: tp.Tensor, step: float, time: float) -> tp.Tensor:
        """One fourth-order step of the master equation."""
        half = 0.5 * step
        first = self.apply(density, time)
        second = self.apply(density + half * first, time + half)
        third = self.apply(density + half * second, time + half)
        fourth = self.apply(density + step * third, time + step)
        return density + (step / 6.0) * (first + 2.0 * second + 2.0 * third + fourth)

    def _compose(self, density: tp.Tensor, time: float) -> tp.Tensor:
        """``exp(time * G) rho`` for a generator that does not move with time.

        A truncated series of the whole span, with the number of terms read off
        the span itself: the series of an argument ``a`` is good to ``a**p / p!``
        after ``p`` terms, so the count grows with the time covered and with
        nothing else.

        There is no squaring step here, and there cannot be one.  A propagator
        can only be squared as an operator, and this generator is never built --
        it exists only as something that acts on a matrix, which squares to
        nothing.  The cost per unit of span is therefore several times better
        than a fourth-order step of the same accuracy, and it grows the same way.

        A span past what one sum can carry is covered by several sums of a size
        one can carry, in a row: advancing by ``a`` then by ``b`` is advancing by
        ``a + b``, and each half is applied to the matrix the other half left.
        So there is no span this route cannot take, and it is still the cheap one
        -- what a longer span buys is more applications of the generator, at the
        same cost per application as a short span's.

        The sum is done in double precision whatever the generator was built in.
        The terms of a series climb to about ``exp(span)`` while the answer they
        add up to stays the size of the state, so the summation cancels; in
        single precision that cancellation runs out a few times out and the sum
        becomes a row of rounding errors.  The result is handed back in the
        precision it came in.

        Args:
            density: shaped ``[batch, 2**n, 2**n]``.
            time: how far to advance.

        Returns:
            The evolved density matrix.
        """
        wider = self._in_wider_precision()
        dtype = density.dtype
        limit = _series_span_limit(tp.complex128)
        # Each piece is a span the sum can carry, so no single sum ever cancels
        # away the precision it needs; the pieces are exact in composition.
        pieces = max(1, int(math.ceil(abs(float(time)) * max(self.rate, 1e-12) / limit)))
        piece = float(time) / pieces
        current = density.to(dtype=wider.c_dtype)
        for _ in range(pieces):
            current = self._sum(current, piece)
        return current.to(dtype=dtype)

    def _sum(self, density: tp.Tensor, time: float) -> tp.Tensor:
        """One composition of a span small enough to sum in a single series."""
        span = abs(float(time)) * max(self.rate, 1e-12)
        wider = self._in_wider_precision()
        result = density
        term = density
        for order in range(1, _series_degree(span) + 1):
            term = wider.apply(term) * (time / order)
            result = result + term
        return result

    def _in_wider_precision(self) -> "Lindbladian":
        """The same generator, in the widest precision it can be summed in."""
        if self.c_dtype == tp.complex128:
            return self
        if self._wider is None:
            self._wider = Lindbladian(
                self.hamiltonian,
                self._entries,
                device=self.device,
                c_dtype=tp.complex128,
            )
        return self._wider

    def __repr__(self) -> str:
        if self._time_dependent:
            body = "<time-dependent>"
        elif self._terms:
            body = f"Hamiltonian(n_qubits={self.n_qubits}, terms={len(self._terms)})"
        elif self._dense is not None:
            body = f"matrix {tuple(self._dense.shape)}"
        else:
            body = "zero"
        n_jumps = len(self._words) + len(self._operators) + int(self._scaling is not None)
        return f"Lindbladian({body}, jumps={n_jumps})"


def lindbladian(
    hamiltonian,
    jump_ops=None,
    *,
    n_qubits: Optional[int] = None,
    device=None,
    c_dtype=DEFAULT_C_DTYPE,
) -> Lindbladian:
    """The Lindblad generator of a master equation.

    A constructor for :class:`Lindbladian`, for a call that wants the generator
    as a value rather than as a place to hold one.

    Args:
        hamiltonian: a :class:`Hamiltonian`, a dense matrix, ``None``, or a
            callable of time returning either.
        jump_ops: the dissipative terms, as :class:`Lindbladian` takes them.
        n_qubits: the register width, when nothing else names it.
        device: where the derived pieces go.
        c_dtype: the complex type of the derived pieces.

    Returns:
        The generator.
    """
    return Lindbladian(
        hamiltonian, jump_ops, n_qubits=n_qubits, device=device, c_dtype=c_dtype
    )


def dissipator(
    jump_ops,
    density: tp.Tensor,
    *,
    device=None,
    c_dtype=DEFAULT_C_DTYPE,
) -> tp.Tensor:
    """The Lindblad dissipator of ``jump_ops``, applied to ``density``.

    This is the ``sum_j (L_j rho L_j^dagger - 1/2 {L_j^dagger L_j, rho})`` half
    of a master equation, with the Hamiltonian's commutator left off.  Reading
    it on its own is how a steady state is checked, how fast a channel decays
    is measured, or how a custom generator is put together.

    Jump operators are read the way :class:`Lindbladian` reads them: a Pauli
    word carries unit rate, ``sqrt(gamma) P`` carries ``gamma``, and a matrix
    names its wires or spans the whole register.

    Args:
        jump_ops: the dissipative terms, as :class:`Lindbladian` takes them.
        density: shaped ``[batch, 2**n, 2**n]``.
        device: where to build the operators.
        c_dtype: the complex type of the operators.

    Returns:
        The dissipator's action, in the same layout as ``density``.
    """
    density = _as_density(density)
    batch, dim, _ = density.shape
    n_qubits = _qubit_count(dim)
    if tp.is_complex(density):
        c_dtype = density.dtype
    device = density.device if device is None else device

    if isinstance(jump_ops, (str, tp.Tensor)):
        entries = [jump_ops]
    elif isinstance(jump_ops, tuple) and len(jump_ops) in (2, 3):
        entries = [jump_ops]
    else:
        entries = list(jump_ops)
    scaling, words, operators, _ = _read_jumps(entries, n_qubits, device, c_dtype)

    flat = density.to(dtype=c_dtype).reshape((batch, *([2] * (2 * n_qubits))))
    out = tp.zeros_like(flat)
    if scaling is not None:
        out = out + flat * scaling.reshape([2] * (2 * n_qubits))
    for rate, matrix, wires in words:
        turned = apply_gate(
            apply_gate(flat, matrix, wires),
            matrix.conj(),
            tuple(wire + n_qubits for wire in wires),
        )
        # L^dagger L is the identity for a Pauli word, so the anticommutator
        # term is the whole of rho.
        out = out + rate * (turned - flat)
    for rate, operator, gram, wires in operators:
        bra_wires = tuple(wire + n_qubits for wire in wires)
        turned = apply_gate(
            apply_gate(flat, operator, wires), operator.conj(), bra_wires
        )
        out = out + rate * (
            turned
            - 0.5 * (
                apply_gate(flat, gram, wires) + apply_gate(flat, gram, bra_wires)
            )
        )
    return out.reshape((batch, dim, dim))


def mesolve(
    density: tp.Tensor,
    hamiltonian,
    jump_ops=None,
    time: float = 0.0,
    *,
    n_sample: int = 0,
    observables: Sequence = (),
    method: str = "auto",
    n_step: Optional[int] = None,
    dt: Optional[float] = None,
    n_qubits: Optional[int] = None,
):
    """Solve the master equation over ``time``.

    Args:
        density: the initial density matrix, shaped ``[batch, 2**n, 2**n]``.
        hamiltonian: a :class:`Hamiltonian`, a dense matrix, ``None``, or a
            callable of time returning either.
        jump_ops: the dissipative terms, as :class:`Lindbladian` takes them.
        time: how long to run for.
        n_sample: how many times to report, evenly spaced and including both
            ends; zero reports only the final matrix.
        observables: what to read off each reported time, as
            :class:`Hamiltonian` instances or Pauli words.
        method: ``"auto"``, ``"rk4"`` or ``"taylor"``; see
            :meth:`Lindbladian.evolve`.
        n_step: fourth-order steps, or ``None`` to choose from the rate.
        dt: the step to take, as an alternative to ``n_step``.
        n_qubits: the register width, when nothing else names it.

    Returns:
        The final density matrix when ``n_sample`` is zero, otherwise the pair
        ``(states, values)`` :meth:`Lindbladian.trajectory` documents.
    """
    generator = Lindbladian(hamiltonian, jump_ops, n_qubits=n_qubits)
    if n_sample:
        return generator.trajectory(
            density,
            time,
            n_sample=n_sample,
            observables=observables,
            method=method,
            n_step=n_step,
            dt=dt,
        )
    return generator.evolve(density, time, method=method, n_step=n_step, dt=dt)


def sesolve(
    state: tp.Tensor,
    hamiltonian,
    time: float = 0.0,
    *,
    n_sample: int = 0,
    observables: Sequence = (),
    method: str = "trotter",
    order: int = 1,
    n_step: Optional[int] = None,
):
    """Evolve a closed system, where the generator is the Hamiltonian alone.

    A pure state is advanced as a state, which is ``2**n`` times less work than
    advancing its density matrix -- the difference between an answer that costs
    ``2**n`` per step and one that costs ``4**n``.

    Only a state vector comes in, so what goes in is what comes out.  A density
    matrix, mixed or not, is a different object and takes :func:`mesolve` with no
    jump operators, which advances it the same way this would.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        hamiltonian: a :class:`Hamiltonian` or a dense matrix.
        time: how long to run for.
        n_sample: how many times to report; zero reports only the last.
        observables: what to read off each reported time, as
            :class:`Hamiltonian` instances or Pauli words.  A state is read
            through its density matrix, which costs one square of it per sample.
        method: ``"trotter"`` for a Trotter product of exact per-term rotations,
            which needs no step size, or ``"matrix_exp"`` for the dense
            exponential, which is exact and confined to small registers.
        order: how many times the Trotter product splits.
        n_step: the number of steps, or ``None`` for one per term.

    Returns:
        The final state, or the pair ``(states, values)`` when ``n_sample`` is
        positive, with the states in the layout that came in.

    Raises:
        ValueError: if the input is a density matrix rather than a state, or if
            the method is not one of the two.
    """
    if state.dim() < 2 or any(dim != 2 for dim in state.shape[1:]):
        raise ValueError(
            f"sesolve evolves a state, shaped [batch, 2, 2, ...], got "
            f"{tuple(state.shape)}; a density matrix is [batch, 2**n, 2**n] and "
            "belongs in mesolve, which advances it the same way"
        )
    if method not in ("trotter", "matrix_exp"):
        raise ValueError(f"a method is 'trotter' or 'matrix_exp', got {method!r}")
    from ._hamiltonian import matrix_exp_evolve, trotter_evolve

    if isinstance(hamiltonian, Hamiltonian):
        if method == "matrix_exp":
            def advance(current, step):
                return matrix_exp_evolve(current, hamiltonian, step)
        else:
            def advance(current, step):
                return trotter_evolve(current, hamiltonian, step, order)
        steps = n_step if n_step is not None else max(1, len(hamiltonian))
    else:
        # A dense generator has no terms to walk, so it is propagated by its own
        # exponential.  Splitting the time leaves that exponential exact, so the
        # split costs nothing in accuracy.
        steps = max(n_step or 1, 1)
        propagator = tp.matrix_exp(-1j * (float(time) / steps) * hamiltonian)

        def advance(current, step):
            return (propagator @ flatten(current).transpose(-1, -2)).transpose(
                -1, -2
            ).reshape(current.shape)

    wanted = max(n_sample, 1)
    marks = _sample_marks(steps, wanted)
    collected = []
    current = state
    if n_sample > 0:
        collected.append(current)
    step = float(time) / steps
    for index in range(steps):
        current = advance(current, step)
        if n_sample > 0 and (index + 1) in marks:
            collected.append(current)
    if not n_sample:
        return current
    states = tp.stack(collected)
    readers = [_observable_reader(item, state.device) for item in observables]
    values = None
    if readers:
        values = tp.stack(
            [
                tp.stack([reader(to_density(one)) for one in states], dim=0)
                for reader in readers
            ],
            dim=1,
        )
    return states, values
