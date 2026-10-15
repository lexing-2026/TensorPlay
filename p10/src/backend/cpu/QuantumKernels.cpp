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

#include <bit>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

constexpr int64_t kQuantumGrain = 4096;
// How much of a column's walk one work item covers, and how many pieces a walk
// is cut into at most.
constexpr int64_t kQuantumWalkChunk = 1024;
constexpr int64_t kQuantumMaxChunks = 256;
// Past this width a gate is better expressed as a sequence of narrower ones;
// the limit is shared with the accelerator path so one gate behaves the same
// on both.
constexpr size_t kQuantumMaxWires = 8;
// The mask that picks a histogram's wires out of an amplitude index is 64 bits
// wide, so that is as many wires as one can name.
constexpr int64_t kQuantumMaxHistogramWires = 64;

struct WireSet {
    // Shift of each targeted wire inside an amplitude index, plus the bit
    // pattern of those wires alone.
    std::vector<int64_t> shifts;
    uint64_t mask = 0;
    int64_t side = 0;
    int64_t dim = 0;
    // Whether the listed wires are the whole register in order, which turns the
    // column-to-index scatter into a copy.
    bool identity = false;
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
// two bits onto each other and make the operation act on fewer qubits than it
// claims, so it is rejected here.
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
                             ? kQuantumMaxWires
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
    for (size_t i = 0; i < wires.size(); ++i) {
        if (wires[i] < 0 || wires[i] >= n_qubits) {
            TP_THROW(IndexError, "wire " + std::to_string(wires[i]) +
                     " is outside a register of " + std::to_string(n_qubits) +
                     " qubit(s)");
        }
        for (size_t j = 0; j < i; ++j) {
            if (wires[j] == wires[i]) {
                TP_THROW(ValueError, "wire " + std::to_string(wires[i]) +
                         " is named twice");
            }
        }
        const int64_t shift = n_qubits - 1 - wires[i];
        set.shifts.push_back(shift);
        set.mask |= uint64_t(1) << shift;
    }
    set.side = int64_t(1) << wires.size();
    set.dim = int64_t(1) << n_qubits;
    // Listing every wire in order makes the column index the amplitude index:
    // the bit scatter the caller normally pays for collapses to a copy.
    set.identity = wires.size() == static_cast<size_t>(n_qubits);
    for (size_t j = 0; set.identity && j < wires.size(); ++j) {
        set.identity = set.shifts[j] == static_cast<int64_t>(n_qubits) - 1 - static_cast<int64_t>(j);
    }
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

// A Pauli operator sends the amplitude at index x to a single other index: the
// X and Y factors flip their bit, the Z factors leave it and contribute their
// sign, and the sign of the Y factors contributes a quarter turn each.  Every
// row of the matrix holds exactly one entry, so the whole overlap collapses to
// one pass of two reads per amplitude -- no gathered sum, no materialized
// rotated state.
struct PauliWord {
    uint64_t flip_mask = 0;  // wires whose factor flips the amplitude index
    uint64_t sign_mask = 0;  // wires whose factor contributes a sign
    int64_t n_y = 0;
};

template <typename scalar_t>
void apply_gate_kernel(const scalar_t* state, const scalar_t* mat, scalar_t* out,
                       const WireSet& wires, int64_t batch, int64_t mat_stride) {
    const int64_t dim = wires.dim;
    const int64_t side = wires.side;
    const size_t n_shifts = wires.shifts.size();

    // A register length is a power of two, so the batch entry and the amplitude
    // index are a shift and a mask apart -- a division by a value only known at
    // run time costs more than the gather that follows it.  The shift table is
    // held by a plain pointer for the same reason: reaching it through the set
    // on every amplitude costs more than the gather itself.
    const int64_t dim_bits = checked_log2(dim, "state");
    const int64_t index_mask = dim - 1;
    const int64_t* shifts = wires.shifts.data();

    parallel::parallel_for(0, batch * dim, kQuantumGrain, [&](int64_t first, int64_t last) {
        for (int64_t flat = first; flat < last; ++flat) {
            const int64_t b = flat >> dim_bits;
            const int64_t index = flat & index_mask;

            // Row of the gate matrix this output amplitude belongs to: the
            // targeted bits of the index, read in ascending wire order.
            int64_t row = 0;
            for (size_t j = 0; j < n_shifts; ++j) {
                row = (row << 1) | ((index >> shifts[j]) & 1);
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

// Marginal distribution over a chosen list of wires: every amplitude is read
// once, its weight landing in the column its targeted bits name.  The order the
// wires were listed in is the column order, first wire most significant.
template <typename scalar_t, typename real_t>
void marginal_probs_kernel(const scalar_t* state, real_t* partial, int64_t batch,
                           int64_t chunk, int64_t chunk_shift, int64_t n_chunks,
                           const WireSet& wires) {
    const int64_t dim = wires.dim;
    const int64_t side = wires.side;
    // The wires that were not listed are free; their shifts drive the walk over
    // the remaining basis entries.
    std::vector<int64_t> free_shifts;
    const int64_t n_bits = checked_log2(dim, "state");
    for (int64_t shift = 0; shift < n_bits; ++shift) {
        if (((wires.mask >> shift) & 1) == 0) {
            free_shifts.push_back(shift);
        }
    }
    const size_t n_free = free_shifts.size();
    const int64_t combos = int64_t(1) << n_free;
    // A column count is a power of two, so the batch entry and the column are a
    // shift and a mask apart.  Dividing by a value only known at run time costs
    // more than the walk that follows it.
    const int64_t batch_shift = static_cast<int64_t>(wires.shifts.size());
    const int64_t column_mask = side - 1;
    // Held by a plain pointer: reaching the table through the set on every
    // column costs more than the walk that follows it.
    const int64_t* shifts = wires.shifts.data();
    const size_t n_shifts = wires.shifts.size();
    const bool identity = wires.identity;

    // One work item per (state, column, chunk): a column whose walk is long
    // would otherwise own a whole core while the rest of the register sits
    // idle, and a handful of columns is exactly what a two-wire marginal asks
    // for.
    parallel::parallel_for(0, batch * side * n_chunks, kQuantumGrain,
                           [&](int64_t first, int64_t last) {
        for (int64_t flat = first; flat < last; ++flat) {
            // The piece count is a power of two, so the piece and the column
            // are a shift and a mask apart.
            const int64_t c = flat & (n_chunks - 1);
            const int64_t column_flat = flat >> chunk_shift;
            const int64_t b = column_flat >> batch_shift;
            const int64_t column = column_flat & column_mask;
            // The column's bits settle the listed wires; the rest start clear.
            uint64_t base = 0;
            if (identity) {
                base = static_cast<uint64_t>(column);
            } else {
                for (size_t j = 0; j < n_shifts; ++j) {
                    if ((column >> (n_shifts - 1 - j)) & 1) {
                        base |= uint64_t(1) << shifts[j];
                    }
                }
            }
            const scalar_t* row = state + b * dim;
            real_t total = 0;
            uint64_t bits = 0;
            const int64_t begin = c * chunk;
            const int64_t end = std::min(begin + chunk, combos);
            for (int64_t combo = begin; combo < end; ++combo) {
                // Walking the free wires one flip at a time turns the inner
                // loop into a single exclusive or per step.  A chunk that does
                // not start at zero rebuilds the bits it was handed first.
                if (combo != 0) {
                    int64_t which = 0;
                    uint64_t probe = uint64_t(combo);
                    while ((probe & 1) == 0) {
                        probe >>= 1;
                        ++which;
                    }
                    bits ^= uint64_t(1) << free_shifts[which];
                }
                const scalar_t value = row[static_cast<int64_t>(base | bits)];
                total += value.real() * value.real() + value.imag() * value.imag();
            }
            partial[flat] = total;
        }
    });
}

// Sums the overlap of each state with its Pauli-rotated self.  The per-chunk
// partial sums keep the reduction order fixed, so the result does not depend on
// how the work was split.
template <typename scalar_t, typename real_t>
void pauli_expectation_kernel(const scalar_t* state, real_t* partial, int64_t batch,
                              int64_t dim, int64_t chunk, int64_t n_chunks,
                              const PauliWord& word) {
    // (-i) per Y factor, the one part of the sign that does not depend on the
    // amplitude index.
    const int quarter = static_cast<int>(word.n_y % 4);
    const real_t turns[4] = {real_t(1), real_t(0), real_t(-1), real_t(0)};
    const real_t quarter_imag[4] = {real_t(0), real_t(-1), real_t(0), real_t(1)};
    const real_t fixed_re = turns[quarter];
    const real_t fixed_im = quarter_imag[quarter];

    parallel::parallel_for(0, batch * n_chunks, kQuantumGrain,
                           [&](int64_t first, int64_t last) {
        for (int64_t flat = first; flat < last; ++flat) {
            const int64_t b = flat / n_chunks;
            const int64_t c = flat - b * n_chunks;
            const int64_t begin = c * chunk;
            const int64_t end = std::min(begin + chunk, dim);
            const scalar_t* row = state + b * dim;
            real_t acc_re = 0;
            real_t acc_im = 0;
            for (int64_t x = begin; x < end; ++x) {
                const scalar_t amplitude = row[x];
                const scalar_t turned = row[static_cast<int64_t>(static_cast<uint64_t>(x) ^
                                                                 word.flip_mask)];
                // conj(amplitude) * turned, before the sign.  The real part of
                // that product is symmetric in the two, but the imaginary part
                // is not: it has to read the amplitude's imaginary half against
                // the turned one, or the overlap comes out conjugated and a
                // Y factor turns the answer upside down.
                const real_t re = amplitude.real() * turned.real() +
                                  amplitude.imag() * turned.imag();
                const real_t im = amplitude.real() * turned.imag() -
                                  amplitude.imag() * turned.real();
                // Z and Y factors each contribute a sign where their bit reads 1.
                const int64_t ones = std::popcount(
                    static_cast<uint64_t>(x) & word.sign_mask);
                if (ones & 1) {
                    acc_re -= re;
                    acc_im -= im;
                } else {
                    acc_re += re;
                    acc_im += im;
                }
            }
            // The overlap of a Hermitian operator is real; the imaginary part
            // left is rounding noise.
            partial[flat] = acc_re * fixed_re - acc_im * fixed_im;
        }
    });
}

PauliWord resolve_pauli_word(const Tensor& word, int64_t n_qubits) {
    if (word.dim() != 1 || word.size(0) != n_qubits) {
        TP_THROW(ValueError, "_quantum_pauli_expectation expects a word of " +
                 std::to_string(n_qubits) + " code(s), got shape " +
                 std::to_string(word.size(0)) + " in " + std::to_string(word.dim()) +
                 " dimension(s)");
    }
    const bool narrow = word.dtype() == DType::Int32;
    if (!narrow && word.dtype() != DType::Int64) {
        TP_THROW(NotImplementedError, std::string(
                     "_quantum_pauli_expectation reads its word as int32 or int64, got ") +
                 toString(word.dtype()));
    }
    const int32_t* narrow_codes = narrow ? word.data_ptr<int32_t>() : nullptr;
    const int64_t* wide_codes = narrow ? nullptr : word.data_ptr<int64_t>();
    PauliWord resolved;
    for (int64_t w = 0; w < n_qubits; ++w) {
        const int64_t code = narrow ? narrow_codes[w] : wide_codes[w];
        if (code < 0 || code > 3) {
            TP_THROW(ValueError, "a Pauli code is 0 for I, 1 for X, 2 for Y, 3 for Z; got " +
                     std::to_string(code));
        }
        const uint64_t bit = uint64_t(1) << (n_qubits - 1 - w);
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

Tensor quantum_marginal_probs_cpu(const Tensor& state, const std::vector<int64_t>& wires) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_marginal_probs expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    check_complex(state.dtype(), "_quantum_marginal_probs");
    const int64_t n_qubits = checked_log2(state.size(1), "_quantum_marginal_probs state length");
    // The columns follow the order the wires were listed in, so the sort a gate
    // needs is not applied here.
    const WireSet resolved = resolve_wires(n_qubits, wires, /*ascending=*/false);

    const int64_t batch = state.size(0);
    const DType dtype = quantum_real_dtype(state.dtype());
    Tensor out = Tensor::empty({batch, resolved.side}, dtype, state.device());
    if (state.numel() == 0) {
        return out;
    }

    // A column whose walk covers the whole register would leave every other core
    // idle, so the walk is cut into chunks that are summed afterwards.  The sum
    // runs in a fixed order, so two runs on the same state agree bit for bit.
    const int64_t n_free = n_qubits - static_cast<int64_t>(wires.size());
    const int64_t combos = int64_t(1) << n_free;
    // A power-of-two piece count keeps the work item's piece and column a shift
    // and a mask apart; the last piece may be short, which the walk bounds off.
    int64_t n_chunks = 1;
    while (n_chunks < kQuantumMaxChunks && n_chunks * kQuantumWalkChunk < combos) {
        n_chunks <<= 1;
    }
    const int64_t chunk_shift = checked_log2(n_chunks, "piece count");
    const int64_t chunk = (combos + n_chunks - 1) / n_chunks;
    // One piece covers the whole walk, so its partial is the answer: writing it
    // in place saves both the copy and the pass that sums the pieces together.
    Tensor partial = n_chunks == 1
                         ? out
                         : Tensor::empty({batch * resolved.side * n_chunks}, dtype,
                                         state.device());

    if (state.dtype() == DType::ComplexDouble) {
        marginal_probs_kernel<complex<double>, double>(
            state.data_ptr<complex<double>>(), partial.data_ptr<double>(), batch, chunk,
            chunk_shift, n_chunks, resolved);
    } else {
        marginal_probs_kernel<complex<float>, float>(
            state.data_ptr<complex<float>>(), partial.data_ptr<float>(), batch, chunk,
            chunk_shift, n_chunks, resolved);
    }

    if (n_chunks == 1) {
        return out;
    }
    const double* wide = dtype == DType::Float64 ? partial.data_ptr<double>() : nullptr;
    const float* narrow = dtype == DType::Float32 ? partial.data_ptr<float>() : nullptr;
    for (int64_t entry = 0; entry < batch * resolved.side; ++entry) {
        double total = 0.0;
        for (int64_t c = 0; c < n_chunks; ++c) {
            total += wide ? wide[entry * n_chunks + c] : narrow[entry * n_chunks + c];
        }
        if (wide) {
            out.data_ptr<double>()[entry] = total;
        } else {
            out.data_ptr<float>()[entry] = static_cast<float>(total);
        }
    }
    return out;
}

// <psi|P|psi> for a Pauli word given as one code per qubit: 0 for I, 1 for X,
// 2 for Y, 3 for Z.  One pass over the amplitudes with nothing materialized in
// between, which is what makes a sum over many terms cheap.
Tensor quantum_pauli_expectation_cpu(const Tensor& state, const Tensor& word) {
    if (state.dim() != 2) {
        TP_THROW(ValueError, "_quantum_pauli_expectation expects a state shaped "
                 "[batch, 2**n], got " + std::to_string(state.dim()) + " dimension(s)");
    }
    check_complex(state.dtype(), "_quantum_pauli_expectation");
    const int64_t n_qubits = checked_log2(state.size(1),
                                          "_quantum_pauli_expectation state length");
    const PauliWord resolved = resolve_pauli_word(word, n_qubits);

    const int64_t batch = state.size(0);
    const int64_t dim = state.size(1);
    const DType dtype = quantum_real_dtype(state.dtype());
    Tensor out = Tensor::empty({batch}, dtype, state.device());
    if (state.numel() == 0) {
        return out;
    }

    const int64_t n_chunks = std::max<int64_t>(1, (dim + kQuantumGrain - 1) / kQuantumGrain);
    const int64_t chunk = (dim + n_chunks - 1) / n_chunks;
    Tensor partial = Tensor::empty({batch * n_chunks}, dtype, state.device());

    if (state.dtype() == DType::ComplexDouble) {
        pauli_expectation_kernel<complex<double>, double>(
            state.data_ptr<complex<double>>(), partial.data_ptr<double>(), batch, dim,
            chunk, n_chunks, resolved);
    } else {
        pauli_expectation_kernel<complex<float>, float>(
            state.data_ptr<complex<float>>(), partial.data_ptr<float>(), batch, dim,
            chunk, n_chunks, resolved);
    }

    // The partials are summed in a fixed order so two runs on the same state
    // agree bit for bit.
    const double* wide = dtype == DType::Float64 ? partial.data_ptr<double>() : nullptr;
    const float* narrow = dtype == DType::Float32 ? partial.data_ptr<float>() : nullptr;
    for (int64_t b = 0; b < batch; ++b) {
        double total = 0.0;
        for (int64_t c = 0; c < n_chunks; ++c) {
            total += wide ? wide[b * n_chunks + c] : narrow[b * n_chunks + c];
        }
        if (wide) {
            out.data_ptr<double>()[b] = total;
        } else {
            out.data_ptr<float>()[b] = static_cast<float>(total);
        }
    }
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantumKernels) {
    m.impl("_quantum_apply_gate", quantum_apply_gate_cpu);
    m.impl("_quantum_state_probs", quantum_state_probs_cpu);
    m.impl("_quantum_marginal_probs", quantum_marginal_probs_cpu);
    m.impl("_quantum_pauli_expectation", quantum_pauli_expectation_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
