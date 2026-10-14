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
// How much of a column's walk one block covers, and how many pieces a walk is
// cut into at most.
constexpr int64_t kQuantumWalkChunk = 1024;
constexpr int64_t kQuantumMaxChunks = 256;
// Past this width a gate is better expressed as a sequence of narrower ones;
// the limit is shared with the host path so one gate behaves the same on both.
constexpr int kQuantumMaxWires = 8;

// A histogram needs no matrix, so it is only bounded by the register -- and the
// mask that picks its wires out of an amplitude index is 64 bits wide, which is
// where that ceiling comes from.  The shifts array is sized for that case
// rather than for the gate case: it is the larger of the two that has to fit.
constexpr int kQuantumMaxHistogramWires = 64;

// Passed to the kernel by value: the wire shifts are the same for every
// thread, and the struct is a few hundred bytes.
struct WireSet {
    int shifts[kQuantumMaxHistogramWires];
    int n_shifts;
    int side;
    int dim;
    int identity;
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
// and make the operation act on fewer qubits than it claims, so it is rejected.
//
// A gate reads its matrix bits against ascending wires, which is what the sort
// below arranges.  A histogram has no matrix: its columns are the listed wires
// in the order they were listed, so that caller keeps the order it was given and
// asks for no sort.
WireSet resolve_wires(int64_t n_qubits, std::vector<int64_t> wires, bool ascending = true) {
    // A gate's matrix is 2**k by 2**k, so its wire count is capped: past that the
    // matrix alone outweighs the state it would be applied to.  A histogram has
    // no matrix, and its only ceiling is the register itself.
    const size_t limit = ascending
                             ? static_cast<size_t>(kQuantumMaxWires)
                             : static_cast<size_t>(
                                   std::min<int64_t>(n_qubits, kQuantumMaxHistogramWires));
    if (wires.size() > limit) {
        TP_THROW(ValueError,
                 std::string(ascending ? "a gate may span at most "
                                       : "a histogram may cover at most ") +
                 std::to_string(limit) + " wire(s), got " +
                 std::to_string(wires.size()));
    }
    if (ascending) {
        std::sort(wires.begin(), wires.end());
    }
    WireSet set;
    set.n_shifts = static_cast<int>(wires.size());
    set.mask = 0;
    for (int i = 0; i < set.n_shifts; ++i) {
        if (wires[i] < 0 || wires[i] >= n_qubits) {
            TP_THROW(IndexError, "wire " + std::to_string(wires[i]) +
                     " is outside a register of " + std::to_string(n_qubits) +
                     " qubit(s)");
        }
        for (int j = 0; j < i; ++j) {
            if (wires[j] == wires[i]) {
                TP_THROW(ValueError, "wire " + std::to_string(wires[i]) +
                         " is named twice");
            }
        }
        set.shifts[i] = static_cast<int>(n_qubits - 1 - wires[i]);
        set.mask |= 1ull << set.shifts[i];
    }
    set.side = 1 << set.n_shifts;
    set.dim = 1 << n_qubits;
    // Listing every wire in order makes the column index the amplitude index:
    // the bit scatter the caller normally pays for collapses to a copy.
    set.identity = wires.size() == static_cast<size_t>(n_qubits) ? 1 : 0;
    for (int j = 0; set.identity && j < set.n_shifts; ++j) {
        set.identity = set.shifts[j] == static_cast<int>(n_qubits) - 1 - j ? 1 : 0;
    }
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

// A Pauli operator sends the amplitude at index x to a single other index: the X
// and Y factors flip their bit, the Z factors leave it and contribute their
// sign, and each Y factor contributes a quarter turn of its own.  Every row of
// the matrix holds exactly one entry, so the whole overlap collapses to one
// pass of two reads per amplitude -- no gathered sum and no rotated state
// written out in between.
struct PauliWord {
    unsigned long long flip_mask;
    unsigned long long sign_mask;
    int n_y;
};

// One block owns one state and reduces it in shared memory; the block's partial
// is written out so the final sum has a fixed order and the same state always
// gives the same number.
template <typename scalar_t, typename real_t, typename index_t>
__global__ void quantum_pauli_partial_kernel(int dim, PauliWord word,
                                            const scalar_t* __restrict__ state,
                                            real_t* __restrict__ partial) {
    __shared__ real_t scratch[kQuantumBlock];
    const scalar_t* row = state + static_cast<size_t>(blockIdx.y) * dim;
    real_t acc_re = 0;
    real_t acc_im = 0;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < dim;
         x += blockDim.x * gridDim.x) {
        const scalar_t amplitude = row[x];
        const scalar_t turned =
            row[static_cast<int>(static_cast<unsigned long long>(x) ^ word.flip_mask)];
        // conj(amplitude) * turned, before the sign the Z and Y factors add.
        // The real part of that product is symmetric in the two, the imaginary
        // part is not: it reads the amplitude's imaginary half against the
        // turned one, or the overlap comes out conjugated.
        const real_t re = amplitude.real() * turned.real() +
                          amplitude.imag() * turned.imag();
        const real_t im = amplitude.real() * turned.imag() -
                          amplitude.imag() * turned.real();
        const int ones = __popcll(static_cast<unsigned long long>(x) & word.sign_mask);
        if (ones & 1) {
            acc_re -= re;
            acc_im -= im;
        } else {
            acc_re += re;
            acc_im += im;
        }
    }
    scratch[threadIdx.x] = acc_re;
    __shared__ real_t scratch_im[kQuantumBlock];
    scratch_im[threadIdx.x] = acc_im;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            scratch_im[threadIdx.x] += scratch_im[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // One quarter turn per Y factor; the overlap of a Hermitian operator is
        // real, so only the matching part is kept.
        const int turns = word.n_y & 3;
        const real_t re_part[4] = {real_t(1), real_t(0), real_t(-1), real_t(0)};
        const real_t im_part[4] = {real_t(0), real_t(-1), real_t(0), real_t(1)};
        partial[static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x] =
            scratch[0] * re_part[turns] - scratch_im[0] * im_part[turns];
    }
}

// Sums the per-chunk partials of each entry in a fixed order, so two runs on
// the same input agree bit for bit.
template <typename real_t, typename index_t>
__global__ void quantum_chunk_reduce_kernel(index_t count, index_t n_chunks,
                                            const real_t* __restrict__ partial,
                                            real_t* __restrict__ out) {
    for (index_t b = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         b < count; b += static_cast<index_t>(blockDim.x) * gridDim.x) {
        real_t total = 0;
        for (index_t c = 0; c < n_chunks; ++c) {
            total += partial[static_cast<size_t>(b) * n_chunks + c];
        }
        out[b] = total;
    }
}

// Every amplitude is read once and its weight lands in the column its targeted
// bits name.  The order the wires were listed in is the column order, first
// wire most significant.
template <typename scalar_t, typename real_t, typename index_t>
__global__ void quantum_marginal_probs_kernel(index_t total, WireSet wires, int n_bits,
                                              int chunk, int n_chunks,
                                              const scalar_t* __restrict__ state,
                                              real_t* __restrict__ out) {
    // The wires that were not listed are free; their shifts are listed once per
    // block instead of being rediscovered on every step of the walk.
    __shared__ int free_shifts[64];
    if (threadIdx.x == 0) {
        int found = 0;
        for (int s = 0; s < n_bits && found < 64; ++s) {
            if (((wires.mask >> s) & 1ull) == 0) {
                free_shifts[found++] = s;
            }
        }
    }
    __syncthreads();
    const int n_free = n_bits - wires.n_shifts;

    const int64_t side = wires.side;
    const size_t n_shifts = static_cast<size_t>(wires.n_shifts);
    for (index_t flat = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total; flat += static_cast<index_t>(blockDim.x) * gridDim.x) {
        // One block per (state, column, chunk): a column whose walk covers the
        // whole register would otherwise keep the device idle, and a handful of
        // columns is exactly what a two-wire marginal asks for.
        // The piece count is a power of two, so the piece and the column are a
        // shift and a mask apart.
        const int chunk_shift = __ffs(n_chunks) - 1;
        const index_t c = flat & static_cast<index_t>(n_chunks - 1);
        const index_t column_flat = flat >> chunk_shift;
        const index_t b = column_flat / side;
        const index_t column = column_flat - b * side;
        // The column's bits settle the listed wires; the rest start clear.
        unsigned long long base = 0;
        if (wires.identity) {
            base = static_cast<unsigned long long>(column);
        } else {
            for (size_t j = 0; j < n_shifts; ++j) {
                if ((column >> (n_shifts - 1 - j)) & 1) {
                    base |= 1ull << wires.shifts[j];
                }
            }
        }
        const scalar_t* row = state + static_cast<size_t>(b) * wires.dim;
        real_t total_weight = 0;
        // Free wires are flipped one at a time as the counter walks up, so each
        // step of the walk costs a single exclusive or.
        unsigned long long bits = 0;
        const int64_t combos = int64_t(1) << n_free;
        const int64_t begin = static_cast<int64_t>(c) * chunk;
        const int64_t end = min(begin + chunk, combos);
        for (int64_t combo = begin; combo < end; ++combo) {
            // A chunk that does not start at zero rebuilds the bits it was
            // handed before its first flip.
            if (combo != 0) {
                int which = 0;
                unsigned long long probe = static_cast<unsigned long long>(combo);
                while ((probe & 1) == 0) {
                    probe >>= 1;
                    ++which;
                }
                bits ^= 1ull << free_shifts[which];
            }
            const scalar_t value = row[static_cast<int>(base | bits)];
            total_weight += value.real() * value.real() + value.imag() * value.imag();
        }
        out[flat] = total_weight;
    }
}

PauliWord resolve_pauli_word(const Tensor& word, int64_t n_qubits) {
    if (word.dim() != 1 || word.size(0) != n_qubits) {
        TP_THROW(ValueError, "_quantum_pauli_expectation expects a word of " +
                 std::to_string(n_qubits) + " code(s), got " + std::to_string(word.size(0)) +
                 " in " + std::to_string(word.dim()) + " dimension(s)");
    }
    const bool narrow = word.dtype() == DType::Int32;
    if (!narrow && word.dtype() != DType::Int64) {
        TP_THROW(NotImplementedError, std::string(
                     "_quantum_pauli_expectation reads its word as int32 or int64, got ") +
                 toString(word.dtype()));
    }
    // The two bit masks are decided here, on the host, so the codes have to be
    // readable here: a word that arrived on the device comes across first.
    const Tensor codes = word.device().is_cpu()
                             ? word
                             : word.to(Device(DeviceType::CPU)).contiguous();
    const int32_t* narrow_codes = narrow ? codes.data_ptr<int32_t>() : nullptr;
    const int64_t* wide_codes = narrow ? nullptr : codes.data_ptr<int64_t>();
    PauliWord resolved;
    resolved.flip_mask = 0;
    resolved.sign_mask = 0;
    resolved.n_y = 0;
    for (int64_t w = 0; w < n_qubits; ++w) {
        const int64_t code = narrow ? narrow_codes[w] : wide_codes[w];
        if (code < 0 || code > 3) {
            TP_THROW(ValueError, "a Pauli code is 0 for I, 1 for X, 2 for Y, 3 for Z; got " +
                     std::to_string(code));
        }
        const unsigned long long bit = 1ull << (n_qubits - 1 - w);
        if (code == 1 || code == 2) {
            resolved.flip_mask |= bit;
        }
        if (code == 2) {
            resolved.sign_mask |= bit;
            ++resolved.n_y;
        } else if (code == 3) {
            resolved.sign_mask |= bit;
        }
    }
    return resolved;
}

DType quantum_real_dtype(DType state_dtype) {
    return state_dtype == DType::ComplexDouble ? DType::Float64 : DType::Float32;
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

Tensor quantum_marginal_probs_cuda(const Tensor& state, const std::vector<int64_t>& wires) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_marginal_probs expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    quantum_check_complex(state.dtype(), "_quantum_marginal_probs");
    const int64_t n_qubits = quantum_log2(state.size(1),
                                          "_quantum_marginal_probs state length");
    // The columns follow the order the wires were listed in, so the sort a gate
    // needs is not applied here.
    const WireSet resolved = resolve_wires(n_qubits, wires, /*ascending=*/false);

    const int64_t batch = state.size(0);
    const int64_t total = batch * resolved.side;
    const DType dtype = quantum_real_dtype(state.dtype());
    Tensor out = Tensor::empty({batch, resolved.side}, dtype, state.device());
    if (state.numel() == 0 || total == 0) {
        return out;
    }

    // A column whose walk covers the whole register would leave the rest of the
    // device idle, so the walk is cut into chunks that a second pass sums.  The
    // sum runs in a fixed order, so two runs on the same state agree bit for bit.
    const int64_t n_free = n_qubits - static_cast<int64_t>(wires.size());
    const int64_t combos = int64_t(1) << n_free;
    // A power-of-two piece count keeps the work item's piece and column a shift
    // and a mask apart; the last piece may be short, which the walk bounds off.
    int64_t n_chunks = 1;
    while (n_chunks < kQuantumMaxChunks && n_chunks * kQuantumWalkChunk < combos) {
        n_chunks <<= 1;
    }
    const int64_t chunk = (combos + n_chunks - 1) / n_chunks;
    // One piece covers the whole walk, so its partial is the answer: writing it
    // in place saves both the copy and the pass that sums the pieces together.
    Tensor partial = n_chunks == 1 ? out : Tensor::empty({total * n_chunks}, dtype,
                                                          state.device());

    const auto stream = getCurrentCUDAStream().stream();
    const unsigned blocks = static_cast<unsigned>(
        std::min<int64_t>((total * n_chunks + kQuantumBlock - 1) / kQuantumBlock,
                          65535));
    const dim3 block(kQuantumBlock);

#define TP_QUANTUM_MARGINAL(ctype_, rtype_, itype_)                                   \
    {                                                                                 \
        const ctype_* source = state.data_ptr<ctype_>();                               \
        rtype_* scratch = partial.data_ptr<rtype_>();                                  \
        rtype_* dest = out.data_ptr<rtype_>();                                         \
        quantum_marginal_probs_kernel<ctype_, rtype_, itype_>                          \
            <<<blocks, block, 0, stream>>>(static_cast<itype_>(total * n_chunks),      \
                                           resolved, static_cast<int>(n_qubits),      \
                                           static_cast<int>(chunk),                    \
                                           static_cast<int>(n_chunks), source,        \
                                           scratch);                                  \
        if (n_chunks > 1) {                                                            \
            quantum_chunk_reduce_kernel<rtype_, itype_>                                \
                <<<blocks, block, 0, stream>>>(static_cast<itype_>(total),             \
                                               static_cast<itype_>(n_chunks), scratch, \
                                               dest);                                  \
        }                                                                             \
    }

    if (state.dtype() == DType::ComplexDouble) {
        if (total <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            TP_QUANTUM_MARGINAL(complex<double>, double, int32_t)
        } else {
            TP_QUANTUM_MARGINAL(complex<double>, double, int64_t)
        }
    } else {
        if (total <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            TP_QUANTUM_MARGINAL(complex<float>, float, int32_t)
        } else {
            TP_QUANTUM_MARGINAL(complex<float>, float, int64_t)
        }
    }
#undef TP_QUANTUM_MARGINAL
    checkCuda(cudaGetLastError(), "_quantum_marginal_probs launch");
    return out;
}

// <psi|P|psi> for a Pauli word given as one code per qubit: 0 for I, 1 for X,
// 2 for Y, 3 for Z.  One pass over the amplitudes with nothing materialized in
// between, which is what makes a sum over many terms cheap.
Tensor quantum_pauli_expectation_cuda(const Tensor& state, const Tensor& word) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_pauli_expectation expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    quantum_check_complex(state.dtype(), "_quantum_pauli_expectation");
    const int64_t n_qubits = quantum_log2(state.size(1),
                                          "_quantum_pauli_expectation state length");
    const PauliWord resolved = resolve_pauli_word(word, n_qubits);

    const int64_t batch = state.size(0);
    const int64_t dim = state.size(1);
    const DType dtype = quantum_real_dtype(state.dtype());
    Tensor out = Tensor::empty({batch}, dtype, state.device());
    if (state.numel() == 0 || batch == 0) {
        return out;
    }

    // Enough blocks to fill the device but few enough that the final reduction
    // stays short; each one writes a single partial per state.
    const int64_t per_block = kQuantumBlock;
    int64_t n_chunks = (dim + per_block - 1) / per_block;
    const int64_t max_chunks = 512;
    if (n_chunks > max_chunks) {
        n_chunks = max_chunks;
    }
    if (n_chunks < 1) {
        n_chunks = 1;
    }
    Tensor partial = Tensor::empty({batch * n_chunks}, dtype, state.device());

    const auto stream = getCurrentCUDAStream().stream();
    const dim3 partial_grid(static_cast<unsigned>(n_chunks),
                            static_cast<unsigned>(batch));
    const unsigned reduce_blocks =
        static_cast<unsigned>((batch + kQuantumBlock - 1) / kQuantumBlock);
    const dim3 reduce_block(kQuantumBlock);

#define TP_QUANTUM_PAULI(ctype_, rtype_)                                                 \
    {                                                                                    \
        const ctype_* source = state.data_ptr<ctype_>();                                  \
        rtype_* scratch = partial.data_ptr<rtype_>();                                     \
        rtype_* dest = out.data_ptr<rtype_>();                                            \
        quantum_pauli_partial_kernel<ctype_, rtype_, int32_t>                             \
            <<<partial_grid, kQuantumBlock, 0, stream>>>(static_cast<int>(dim),           \
                                                         resolved, source, scratch);    \
        quantum_chunk_reduce_kernel<rtype_, int32_t>                                      \
            <<<reduce_blocks, reduce_block, 0, stream>>>(                                 \
                static_cast<int32_t>(batch), static_cast<int32_t>(n_chunks), scratch,     \
                dest);                                                                   \
    }

    if (state.dtype() == DType::ComplexDouble) {
        TP_QUANTUM_PAULI(complex<double>, double)
    } else {
        TP_QUANTUM_PAULI(complex<float>, float)
    }
#undef TP_QUANTUM_PAULI
    checkCuda(cudaGetLastError(), "_quantum_pauli_expectation launch");
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantumKernels) {
    m.impl("_quantum_apply_gate", quantum_apply_gate_cuda);
    m.impl("_quantum_state_probs", quantum_state_probs_cuda);
    m.impl("_quantum_marginal_probs", quantum_marginal_probs_cuda);
    m.impl("_quantum_pauli_expectation", quantum_pauli_expectation_cuda);
}

}  // namespace cuda
}  // namespace tensorplay