// State simulation primitives on the accelerator: local unitary application
// and amplitude probabilities.
//
// The flat layout is [batch, 2**n] with qubit w on bit n-1-w.  One thread owns
// one output amplitude and gathers the 2**k inputs that differ only inside the
// targeted bits, so a gate is a single pass with no intermediate permuted copy
// of the state.  Neighbouring amplitudes gather overlapping inputs, so the
// reads stay in cache even when the targeted wires are far apart.

#include "Tensor.h"
#include "Complex.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Exception.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kQuantumBlock = 256;
// Past this width a gate is better expressed as a sequence of narrower ones;
// the limit is shared with the host path so one gate behaves the same on both.
constexpr int kQuantumMaxWires = 8;

// Passed to the kernel by value: the wire shifts are the same for every
// thread, and the struct is a few dozen bytes.
struct WireSet {
    int shifts[kQuantumMaxWires];
    int n_shifts;
    int side;
    int dim;
    unsigned long long mask;
};

int64_t quantum_log2(int64_t value, const char* what) {
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

void quantum_check_complex(DType dtype, const char* what) {
    if (dtype != DType::ComplexFloat && dtype != DType::ComplexDouble) {
        TP_THROW(NotImplementedError, std::string(what) +
                 " acts on complex tensors, got ", toString(dtype));
    }
}

DType quantum_result_dtype(DType state_dtype, DType mat_dtype) {
    if (state_dtype == DType::ComplexDouble || mat_dtype == DType::ComplexDouble) {
        return DType::ComplexDouble;
    }
    return DType::ComplexFloat;
}

// Resolves the wires into bit shifts.  A repeated wire would alias two bits
// and make the gate act on fewer qubits than it claims, so it is rejected.
WireSet resolve_wires(int64_t n_qubits, std::vector<int64_t> wires) {
    if (wires.size() > static_cast<size_t>(kQuantumMaxWires)) {
        TP_THROW(ValueError, "a gate may span at most " +
                 std::to_string(kQuantumMaxWires) + " wires, got " +
                 std::to_string(wires.size()));
    }
    std::sort(wires.begin(), wires.end());
    WireSet set;
    set.n_shifts = static_cast<int>(wires.size());
    set.mask = 0;
    for (int i = 0; i < set.n_shifts; ++i) {
        if (wires[i] < 0 || wires[i] >= n_qubits) {
            TP_THROW(IndexError, "wire " + std::to_string(wires[i]) +
                     " is outside a register of " + std::to_string(n_qubits) +
                     " qubit(s)");
        }
        if (i > 0 && wires[i] == wires[i - 1]) {
            TP_THROW(ValueError, "a gate cannot name wire " +
                     std::to_string(wires[i]) + " twice");
        }
        set.shifts[i] = static_cast<int>(n_qubits - 1 - wires[i]);
        set.mask |= 1ull << set.shifts[i];
    }
    set.side = 1 << set.n_shifts;
    set.dim = 1 << n_qubits;
    return set;
}

template <typename scalar_t, typename index_t>
__global__ void quantum_apply_gate_kernel(index_t count, WireSet wires,
                                          index_t mat_stride,
                                          const scalar_t* __restrict__ state,
                                          const scalar_t* __restrict__ mat,
                                          scalar_t* __restrict__ out) {
    const int dim = wires.dim;
    const int side = wires.side;
    const int n_shifts = wires.n_shifts;
    const index_t stride = static_cast<index_t>(blockDim.x) * gridDim.x;
    for (index_t flat = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < count; flat += stride) {
        const index_t b = flat / dim;
        const int index = static_cast<int>(flat - b * dim);

        // Row of the gate matrix this amplitude belongs to: the targeted bits
        // of the index, read in ascending wire order.
        int row = 0;
        for (int j = 0; j < n_shifts; ++j) {
            row = (row << 1) |
                  ((static_cast<unsigned long long>(index) >> wires.shifts[j]) & 1ull);
        }
        const unsigned long long rest =
            static_cast<unsigned long long>(index) & ~wires.mask;
        const scalar_t* gate_row = mat + b * mat_stride + static_cast<index_t>(row) * side;

        scalar_t accumulator(0.0, 0.0);
        for (int column = 0; column < side; ++column) {
            // Ascending wires read as one gate-matrix row, most significant
            // bit first.
            unsigned long long bits = 0;
            for (int j = 0; j < n_shifts; ++j) {
                if ((column >> (n_shifts - 1 - j)) & 1) {
                    bits |= 1ull << wires.shifts[j];
                }
            }
            const int source = static_cast<int>(rest | bits);
            accumulator += gate_row[column] * state[b * dim + source];
        }
        out[flat] = accumulator;
    }
}

template <typename scalar_t, typename real_t, typename index_t>
__global__ void quantum_state_probs_kernel(index_t count, const scalar_t* __restrict__ state,
                                           real_t* __restrict__ out) {
    const index_t stride = static_cast<index_t>(blockDim.x) * gridDim.x;
    for (index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const scalar_t value = state[i];
        out[i] = value.real() * value.real() + value.imag() * value.imag();
    }
}

void quantum_check_gate_arguments(const Tensor& state, const Tensor& mat,
                                  const std::vector<int64_t>& wires, int64_t* n_qubits) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_apply_gate expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    quantum_check_complex(state.dtype(), "_quantum_apply_gate");
    quantum_check_complex(mat.dtype(), "_quantum_apply_gate");
    const int64_t bits = quantum_log2(state.size(1), "_quantum_apply_gate state length");
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

}  // namespace

// Applies a local unitary to the qubits named by wires.  The state is flat
// [batch, 2**n] and qubit w owns bit n-1-w, so one output amplitude gathers
// exactly the 2**k inputs that differ only in the targeted bits.
Tensor quantum_apply_gate_cuda(const Tensor& state, const Tensor& mat,
                               const std::vector<int64_t>& wires) {
    int64_t n_qubits = 0;
    quantum_check_gate_arguments(state, mat, wires, &n_qubits);
    if (state.numel() == 0) {
        return state.clone();
    }
    const WireSet resolved = resolve_wires(n_qubits, wires);
    const DType dtype = quantum_result_dtype(state.dtype(), mat.dtype());
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(state.shape()), dtype,
                               state.device());

    const auto stream = getCurrentCUDAStream().stream();
    const int64_t count = state.numel();
    // One matrix per batch entry when the gate carries a per-batch parameter,
    // otherwise the same matrix for all of them.
    const int64_t mat_stride = mat.dim() == 2 ? 0 : resolved.side * resolved.side;
    const unsigned blocks =
        static_cast<unsigned>((count + kQuantumBlock - 1) / kQuantumBlock);
    const dim3 block(kQuantumBlock);

#define TP_QUANTUM_GATE(ctype_, itype_)                                           \
    {                                                                             \
        const ctype_* source = state.data_ptr<ctype_>();                          \
        const ctype_* gate = mat.data_ptr<ctype_>();                              \
        ctype_* dest = out.data_ptr<ctype_>();                                    \
        quantum_apply_gate_kernel<ctype_, itype_><<<blocks, block, 0, stream>>>(   \
            static_cast<itype_>(count), resolved,                                 \
            static_cast<itype_>(mat_stride), source, gate, dest);                \
    }

    if (dtype == DType::ComplexDouble) {
        if (count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            TP_QUANTUM_GATE(complex<double>, int32_t)
        } else {
            TP_QUANTUM_GATE(complex<double>, int64_t)
        }
    } else {
        if (count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            TP_QUANTUM_GATE(complex<float>, int32_t)
        } else {
            TP_QUANTUM_GATE(complex<float>, int64_t)
        }
    }
#undef TP_QUANTUM_GATE
    checkCuda(cudaGetLastError(), "_quantum_apply_gate launch");

    return out;
}

// |amplitude|**2 per basis entry, in the matching real dtype.  Fusing the
// square into one pass keeps the measurement distribution from materializing
// the real and imaginary halves separately.
Tensor quantum_state_probs_cuda(const Tensor& state) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_state_probs expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    quantum_check_complex(state.dtype(), "_quantum_state_probs");
    quantum_log2(state.size(1), "_quantum_state_probs state length");

    const DType dtype = state.dtype() == DType::ComplexDouble ? DType::Float64
                                                              : DType::Float32;
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(state.shape()), dtype,
                               state.device());
    const int64_t count = state.numel();
    if (count == 0) {
        return out;
    }

    const auto stream = getCurrentCUDAStream().stream();
    const unsigned blocks =
        static_cast<unsigned>((count + kQuantumBlock - 1) / kQuantumBlock);
    const dim3 block(kQuantumBlock);

#define TP_QUANTUM_PROBS(ctype_, rtype_)                                          \
    {                                                                             \
        const ctype_* source = state.data_ptr<ctype_>();                          \
        rtype_* dest = out.data_ptr<rtype_>();                                    \
        if (count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) { \
            quantum_state_probs_kernel<ctype_, rtype_, int32_t>                   \
                <<<blocks, block, 0, stream>>>(static_cast<int32_t>(count), source, dest); \
        } else {                                                                  \
            quantum_state_probs_kernel<ctype_, rtype_, int64_t>                   \
                <<<blocks, block, 0, stream>>>(static_cast<int64_t>(count), source, dest); \
        }                                                                         \
    }

    if (dtype == DType::Float64) {
        TP_QUANTUM_PROBS(complex<double>, double)
    } else {
        TP_QUANTUM_PROBS(complex<float>, float)
    }
#undef TP_QUANTUM_PROBS
    checkCuda(cudaGetLastError(), "_quantum_state_probs launch");

    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantumKernels) {
    m.impl("_quantum_apply_gate", quantum_apply_gate_cuda);
    m.impl("_quantum_state_probs", quantum_state_probs_cuda);
}

}  // namespace cuda
}  // namespace tensorplay