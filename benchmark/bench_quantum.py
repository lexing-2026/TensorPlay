"""Measure the state-preparation kernels against the steps they replace.

Every check here is a comparison rather than an absolute: the compiled path
against the same quantity assembled from the smaller pieces the module already
exposes.  A Pauli word read out of a wide register can be computed by rotating
the state and taking an inner product, and a gate can be applied by building the
whole register's unitary and multiplying -- both of which are correct and both
of which cost the square of the register.
"""

import argparse
import json
import statistics
import time

import tensorplay as tp
from tensorplay import quantum as tq


def measure(fn, device, repetitions):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(5):
        if device == 'cuda':
            tp.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            fn()
        if device == 'cuda':
            tp.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / repetitions)
    return statistics.median(samples)


def _rotated_inner_product(state, word):
    """<psi|P|psi> by rotating the state and overlapping it with itself."""
    rotated = tq.apply_pauli_word(state, word)
    overlap = rotated.conj() * state
    return overlap.sum(dim=tuple(range(1, state.dim()))).real


def _permute_and_fold(state, wires):
    """The marginal by moving the listed qubits to the front and summing."""
    n_qubits = state.dim() - 1
    kept = list(wires)
    dropped = [w for w in range(n_qubits) if w not in kept]
    axes = (0, *(w + 1 for w in kept), *(w + 1 for w in dropped))
    # The square is taken before the move, so only the weights are rearranged.
    folded = tq.unflatten(tq.probs(state), n_qubits).permute(axes)
    if dropped:
        folded = folded.sum(dim=tuple(range(1 + len(kept), 1 + n_qubits)))
    return folded.reshape(state.shape[0], 1 << len(kept))


def _decomposed_control(state, matrix, wires):
    """A controlled rotation without a two-wire gate: rotate, copy, unrotate."""
    control, target = wires
    angle = 0.4
    out = tq.apply_ry(state, target, angle)
    out = tq.apply_cnot(out, control, target)
    return tq.apply_ry(out, target, -angle)


def _dense_gate(state, matrix, wires):
    """A gate applied as one dense unitary over the whole register.

    Only available while the register fits the gate width the compiled path
    accepts, which is the whole point: past that the dense alternative stops
    being expressible rather than merely slow.
    """
    n_qubits = state.dim() - 1
    full = tq.expand_to_register(matrix, wires, n_qubits)
    dim = 1 << n_qubits
    rows = tq.flatten(state).reshape(state.shape[0], dim)
    return tq.unflatten(rows @ full, n_qubits)


def benchmark(device, n_qubits, repetitions):
    state = tq.plus_state(n_qubits, device=device)
    word = 'I' * (n_qubits - 1) + 'Z'
    all_wires = list(range(n_qubits))
    matrix = tq.controlled_matrix(tq.rot_y_matrix(0.4))

    cases = {
        'expectation': (
            lambda: tq.expectation(state, word),
            lambda: _rotated_inner_product(state, word),
        ),
        'expectation_half_register': (
            lambda: tq.expectation(state, 'Z' * (n_qubits // 2) + 'I' * (n_qubits - n_qubits // 2)),
            lambda: _rotated_inner_product(
                state, 'Z' * (n_qubits // 2) + 'I' * (n_qubits - n_qubits // 2)
            ),
        ),
        'marginal_probs': (
            lambda: tq.marginal_probs(state, all_wires),
            lambda: _permute_and_fold(state, all_wires),
        ),
        'marginal_probs_few_wires': (
            lambda: tq.marginal_probs(state, [0, 1]),
            lambda: _permute_and_fold(state, [0, 1]),
        ),
        'controlled_gate': (
            lambda: tq.apply_gate(state, matrix, [0, n_qubits - 1]),
            lambda: _decomposed_control(state, matrix, [0, n_qubits - 1]),
        ),
    }
    if n_qubits <= 8:
        cases['controlled_gate_dense'] = (
            lambda: tq.apply_gate(state, matrix, [0, n_qubits - 1]),
            lambda: _dense_gate(state, matrix, [0, n_qubits - 1]),
        )

    times = {}
    for name, (fast, slow) in cases.items():
        difference = abs(complex(tp.reshape(fast(), [-1])[0]) - complex(tp.reshape(slow(), [-1])[0]))
        if difference > 1e-4:
            raise AssertionError(f'{name} on {device}: the two paths differ by {difference}')
        fused = measure(fast, device, repetitions)
        baseline = measure(slow, device, repetitions)
        times[name] = {
            'fused_microseconds': fused,
            'baseline_microseconds': baseline,
            'speedup': baseline / fused,
        }
    return {'device': device, 'n_qubits': n_qubits, 'timings': times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda', 'all'], default='all')
    parser.add_argument('--qubits', type=int, nargs='+', default=[12, 16, 20])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    tp.set_num_threads(args.threads)
    if args.device == 'all':
        devices = ['cpu', 'cuda'] if tp.cuda.is_available() else ['cpu']
    else:
        devices = [args.device]
    records = [benchmark(device, n_qubits, args.repetitions)
               for device in devices for n_qubits in args.qubits]
    print(json.dumps({'threads': args.threads, 'repetitions': args.repetitions,
                      'results': records}, indent=2))


if __name__ == '__main__':
    main()