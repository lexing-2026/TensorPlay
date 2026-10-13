// State simulation primitives: local unitary application and amplitude
// probabilities.
//
// Both work on the flat computational-basis layout [batch, 2**n], where qubit
// w owns bit n-1-w of the amplitude index -- qubit 0 is the most significant
// bit.  The frontend carries states as [batch, 2, 2, ...] and folds the qubit
// dimensions into this one before the call, so a gate costs a single pass over
// the amplitudes with no intermediate copy of the state.
//
// A gate on k wires reads, for each output amplitude, the 2**k input amplitudes
// that differ from it only inside the targeted bit positions.  Those source
// indices are rebuilt from the wire shifts per term, which keeps the kernel
// free of any precomputed table while a wire is at most a shift and a test.

#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

constexpr int64_t kQuantumGrain = 4096;
// Past this width a gate is better expressed as a sequence of narrower ones;
// the limit is shared with the accelerator path so one gate behaves the same
// on both.
constexpr size_t kQuantumMaxWires = 8;

struct WireSet {
    // Shift of each targeted wire inside an amplitude index, plus the bit
    // pattern of those wires alone.
    std::vector<int64_t> shifts;
    uint64_t mask = 0;
    int64_t side = 0;
    int64_t dim = 0;
};

int64_t checked_log2(int64_t value, const char* what) {
    if (value <= 0 || (value & (value - 1)) != 0) {
        TP_THROW(ValueError, std::string(what) + " must be a power of two, got " +
                 std::to_string(value));
    }
    int64_t bits = 0;
    while ((int64_t(1) << bits) < value) {
        ++bits;
    }
    return bits;
}

DType quantum_result_dtype(DType state_dtype, DType mat_dtype) {
    if (state_dtype == DType::ComplexDouble || mat_dtype == DType::ComplexDouble) {
        return DType::ComplexDouble;
    }
    return DType::ComplexFloat;
}

void check_complex(DType dtype, const char* what) {
    if (dtype != DType::ComplexFloat && dtype != DType::ComplexDouble) {
        TP_THROW(NotImplementedError, std::string(what) +
                 " acts on complex tensors, got ", toString(dtype));
    }
}

// Resolves the targeted wires into bit shifts.  A repeated wire would alias
// two bits onto each other and make the gate act on fewer qubits than it
// claims, so it is rejected here.
WireSet resolve_wires(int64_t n_qubits, std::vector<int64_t> wires) {
    if (wires.size() > kQuantumMaxWires) {
        TP_THROW(ValueError, "a gate may span at most " +
                 std::to_string(kQuantumMaxWires) + " wires, got " +
                 std::to_string(wires.size()));
    }
    std::sort(wires.begin(), wires.end());
    WireSet set;
    for (size_t i = 0; i < wires.size(); ++i) {
        if (wires[i] < 0 || wires[i] >= n_qubits) {
            TP_THROW(IndexError, "wire " + std::to_string(wires[i]) +
                     " is outside a register of " + std::to_string(n_qubits) +
                     " qubit(s)");
        }
        if (i > 0 && wires[i] == wires[i - 1]) {
            TP_THROW(ValueError, "a gate cannot name wire " +
                     std::to_string(wires[i]) + " twice");
        }
        const int64_t shift = n_qubits - 1 - wires[i];
        set.shifts.push_back(shift);
        set.mask |= uint64_t(1) << shift;
    }
    set.side = int64_t(1) << wires.size();
    set.dim = int64_t(1) << n_qubits;
    return set;
}

void check_gate_arguments(const Tensor& state, const Tensor& mat,
                          const std::vector<int64_t>& wires, int64_t* n_qubits) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_apply_gate expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    check_complex(state.dtype(), "_quantum_apply_gate");
    check_complex(mat.dtype(), "_quantum_apply_gate");
    const int64_t bits = checked_log2(state.size(1), "_quantum_apply_gate state length");
    const int64_t side = int64_t(1) << wires.size();
    if (mat.dim() != 2 && mat.dim() != 3) {
        TP_THROW(ValueError, "_quantum_apply_gate expects a 2-D or batched 3-D gate "
                 "matrix, got " + std::to_string(mat.dim()) + " dimension(s)");
    }
    if (mat.size(-1) != side || mat.size(-2) != side) {
        TP_THROW(ValueError, "_quantum_apply_gate expects a " + std::to_string(side) +
                 "x" + std::to_string(side) + " gate matrix for " +
                 std::to_string(wires.size()) + " wire(s)");
    }
    if (mat.dim() == 3 && mat.size(0) != state.size(0)) {
        TP_THROW(ValueError, "_quantum_apply_gate was given " + std::to_string(mat.size(0)) +
                 " gate matrices for a batch of " + std::to_string(state.size(0)));
    }
    *n_qubits = bits;
}

template <typename scalar_t>
void apply_gate_kernel(const scalar_t* state, const scalar_t* mat, scalar_t* out,
                       const WireSet& wires, int64_t batch, int64_t mat_stride) {
    const int64_t dim = wires.dim;
    const int64_t side = wires.side;
    const size_t n_shifts = wires.shifts.size();

    parallel::parallel_for(0, batch * dim, kQuantumGrain, [&](int64_t first, int64_t last) {
        for (int64_t flat = first; flat < last; ++flat) {
            const int64_t b = flat / dim;
            const int64_t index = flat - b * dim;

            // Row of the gate matrix this output amplitude belongs to: the
            // targeted bits of the index, read in ascending wire order.
            int64_t row = 0;
            for (size_t j = 0; j < n_shifts; ++j) {
                row = (row << 1) | ((index >> wires.shifts[j]) & 1);
            }
            const uint64_t rest = static_cast<uint64_t>(index) & ~wires.mask;
            const scalar_t* gate_row = mat + b * mat_stride + row * side;

            scalar_t accumulator(0.0, 0.0);
            for (int64_t column = 0; column < side; ++column) {
                // Ascending wires read as one gate-matrix row, most
                // significant bit first.
                uint64_t bits = 0;
                for (size_t j = 0; j < n_shifts; ++j) {
                    if ((column >> (n_shifts - 1 - j)) & 1) {
                        bits |= uint64_t(1) << wires.shifts[j];
                    }
                }
                const int64_t source =
                    static_cast<int64_t>(rest | bits);
                accumulator += gate_row[column] * state[b * dim + source];
            }
            out[flat] = accumulator;
        }
    });
}

template <typename scalar_t, typename real_t>
void state_probs_kernel(const scalar_t* state, real_t* out, int64_t n) {
    parallel::parallel_for(0, n, kQuantumGrain, [&](int64_t first, int64_t last) {
        for (int64_t i = first; i < last; ++i) {
            const scalar_t value = state[i];
            out[i] = value.real() * value.real() + value.imag() * value.imag();
        }
    });
}

}  // namespace

// Applies a local unitary to the qubits named by wires.  The state is flat
// [batch, 2**n] and qubit w owns bit n-1-w, so one output amplitude gathers
// exactly the 2**k inputs that differ only in the targeted bits.
Tensor quantum_apply_gate_cpu(const Tensor& state, const Tensor& mat,
                              const std::vector<int64_t>& wires) {
    int64_t n_qubits = 0;
    check_gate_arguments(state, mat, wires, &n_qubits);
    if (state.numel() == 0) {
        return state.clone();
    }
    const WireSet resolved = resolve_wires(n_qubits, wires);
    const int64_t batch = state.size(0);

    const DType dtype = quantum_result_dtype(state.dtype(), mat.dtype());
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(state.shape()), dtype,
                               state.device());
    // One matrix per batch entry when the gate carries a per-batch parameter,
    // otherwise the same matrix for all of them.
    const int64_t mat_stride = mat.dim() == 2 ? 0 : resolved.side * resolved.side;
    if (dtype == DType::ComplexDouble) {
        apply_gate_kernel<complex<double>>(state.data_ptr<complex<double>>(),
                                           mat.data_ptr<complex<double>>(),
                                           out.data_ptr<complex<double>>(), resolved,
                                           batch, mat_stride);
    } else {
        apply_gate_kernel<complex<float>>(state.data_ptr<complex<float>>(),
                                          mat.data_ptr<complex<float>>(),
                                          out.data_ptr<complex<float>>(), resolved,
                                          batch, mat_stride);
    }
    return out;
}

// |amplitude|**2 per basis entry, in the matching real dtype.  Fusing the
// square into one pass keeps the measurement distribution from materializing
// the intermediate real and imaginary halves.
Tensor quantum_state_probs_cpu(const Tensor& state) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_state_probs expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    check_complex(state.dtype(), "_quantum_state_probs");
    checked_log2(state.size(1), "_quantum_state_probs state length");

    const DType dtype = state.dtype() == DType::ComplexDouble ? DType::Float64
                                                              : DType::Float32;
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(state.shape()), dtype,
                               state.device());
    const int64_t n = state.numel();
    if (n == 0) {
        return out;
    }
    if (state.dtype() == DType::ComplexDouble) {
        state_probs_kernel<complex<double>, double>(state.data_ptr<complex<double>>(),
                                                    out.data_ptr<double>(), n);
    } else {
        state_probs_kernel<complex<float>, float>(state.data_ptr<complex<float>>(),
                                                   out.data_ptr<float>(), n);
    }
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantumKernels) {
    m.impl("_quantum_apply_gate", quantum_apply_gate_cpu);
    m.impl("_quantum_state_probs", quantum_state_probs_cpu);
}

}  // namespace cpu
}  // namespace tensorplay