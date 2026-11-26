#include "Tensor.h"
#include "Dispatcher.h"
#include "ErrorReporting.h"
#include "Exception.h"
#include "Utils.h"
#include "Parallel.h"
#include "Macros.h"
#include "ReductionKernels.h"
#include "TensorIterator.h"
#include "Complex.h"
#include "cpu/CascadeSum.h"
#include "cpu/Reduce.h"
#include "cpu/vec/vec.h"
#include "cpu/vec/vec_reduce.h"
#include "cpu/VecReducedMath.h"
#include "cpu/VecComplex.h"
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tensorplay {
namespace cpu {
namespace {
using namespace vec;
// parallel_for / GRAIN_SIZE moved under tensorplay::parallel.
using namespace tensorplay::parallel;

template <typename T>
struct Accumulator {
    static void add(T& acc, T val) { acc += val; }
    static void mul(T& acc, T val) { acc *= val; }
};

template <>
struct Accumulator<bool> {
    static void add(bool& acc, bool val) { acc = acc || val; }
    static void mul(bool& acc, bool val) { acc = acc && val; }
};

template <typename T>
struct AccumulateType { using type = T; };

template <> struct AccumulateType<float> { using type = double; };
template <> struct AccumulateType<int32_t> { using type = int64_t; };
template <> struct AccumulateType<int16_t> { using type = int64_t; };
template <> struct AccumulateType<int8_t> { using type = int64_t; };
template <> struct AccumulateType<uint8_t> { using type = int64_t; };
template <> struct AccumulateType<bool> { using type = int64_t; };

// Helper to convert any type to Scalar safely
template <typename T>
Scalar to_scalar(T val) {
    if constexpr (std::is_integral_v<T>) {
        return Scalar(static_cast<int64_t>(val));
    } else if constexpr (is_complex_type_v<T>) {
        using vt = typename is_complex_type<T>::value_type;
        if constexpr (std::is_same_v<vt, float> || std::is_same_v<vt, double>) {
            return Scalar(val);
        } else {
            // Scalar storage only holds cfloat/cdouble; widen the reduced
            // complexes through complex64.
            return Scalar(complex<float>(
                static_cast<float>(val.real()), static_cast<float>(val.imag())));
        }
    } else {
        return Scalar(val);
    }
}

// Helper to compute output shape for reduction
std::vector<int64_t> compute_reduction_shape(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<bool> is_reduced(shape.size(), false);

    for (int64_t d : dims) {
        int64_t dim = d;
        if (shape.empty()) {
            // A 0-d tensor accepts -1 and 0 as its single (virtual) dimension.
            if (dim != 0 && dim != -1) {
                TP_THROW(IndexError, format_dim_range(shape.size(), d));
            }
            is_reduced.push_back(true);
            continue;
        }
        if (dim < 0) dim += shape.size();
        if (dim < 0 || dim >= (int64_t)shape.size()) {
             TP_THROW(IndexError, format_dim_range(shape.size(), d));
        }
        is_reduced[dim] = true;
    }

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < is_reduced.size(); ++i) {
        if (is_reduced[i]) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape.empty() ? 0 : shape[i]);
        }
    }
    return out_shape;
}

// True when the named dimensions are exactly every axis of an ndim tensor, so
// the reduction spans the whole payload.  The axes are tracked in a bitmask,
// which also rejects a repeated axis.
inline bool names_every_axis(const std::vector<int64_t>& dims, int64_t ndim) {
    if (ndim <= 0 || ndim > 63 ||
        static_cast<int64_t>(dims.size()) != ndim) {
        return false;
    }
    uint64_t seen = 0;
    for (int64_t d : dims) {
        const int64_t rank = d < 0 ? d + ndim : d;
        if (rank < 0 || rank >= ndim) return false;
        const uint64_t bit = uint64_t(1) << rank;
        if ((seen & bit) != 0) return false;
        seen |= bit;
    }
    return seen == ((uint64_t(1) << ndim) - 1);
}

// ndim, inserting size-1 dims with stride 0 at the reduced positions so the
// iterator can identify the reduced dims from the output's strides.
Tensor review_reduce_result(const Tensor& result, int64_t ndim, const std::vector<bool>& mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(result.shape());
  std::vector<int64_t> stride = static_cast<std::vector<int64_t>>(result.strides());
  for (int64_t dim = 0; dim < ndim; ++dim) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  // The generated as_strided takes a mutable view of the source tensor; the
  // view it returns never aliases the caller's operand, so a shallow copy of
  // the reference satisfies the signature without touching the underlying
  // storage.
  Tensor as_strided_src = result;
  return Tensor::as_strided(as_strided_src, shape, stride, std::nullopt);
}

// ops-based accumulator for the complex dtypes (no Vectorized<complex>
// operator+ with per-thread partials combined by binary_kernel_reduce.
template <typename scalar_t>
struct CxSumOps {
    scalar_t reduce(scalar_t acc, scalar_t data, int64_t) const { return acc + data; }
    scalar_t combine(scalar_t a, scalar_t b) const { return a + b; }
    scalar_t project(scalar_t a) const { return a; }
    scalar_t translate_idx(scalar_t acc, int64_t) const { return acc; }
};

template <typename scalar_t>
struct NormAccumulatorType {
    using value_t = typename scalar_value_type<scalar_t>::type;
    using type = std::conditional_t<
        std::is_same_v<value_t, Half> || std::is_same_v<value_t, BFloat16>,
        float, value_t>;
};

template <typename scalar_t, typename acc_t>
inline acc_t norm_abs_value(scalar_t value) {
    if constexpr (is_complex_type_v<scalar_t>) {
        return static_cast<acc_t>(std::hypot(
            static_cast<acc_t>(value.real()), static_cast<acc_t>(value.imag())));
    } else {
        const acc_t converted = static_cast<acc_t>(value);
        return converted < acc_t(0) ? -converted : converted;
    }
}

template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct NormZeroOps {
    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        return acc + (data == scalar_t(0) ? acc_t(0) : acc_t(1));
    }
    acc_t combine(acc_t a, acc_t b) const { return a + b; }
    out_t project(acc_t value) const { return static_cast<out_t>(value); }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct NormOneOps {
    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        return acc + norm_abs_value<scalar_t, acc_t>(data);
    }
    acc_t combine(acc_t a, acc_t b) const { return a + b; }
    out_t project(acc_t value) const { return static_cast<out_t>(value); }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct NormTwoOps {
    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        const acc_t value = norm_abs_value<scalar_t, acc_t>(data);
        return acc + value * value;
    }
    acc_t combine(acc_t a, acc_t b) const { return a + b; }
    out_t project(acc_t value) const {
        return static_cast<out_t>(std::sqrt(value));
    }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct AbsMinOps {
    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        return combine(acc, norm_abs_value<scalar_t, acc_t>(data));
    }
    acc_t combine(acc_t a, acc_t b) const {
        if (std::isnan(a)) return a;
        if (std::isnan(b)) return b;
        return a < b ? a : b;
    }
    out_t project(acc_t value) const { return static_cast<out_t>(value); }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct AbsMaxOps {
    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        return combine(acc, norm_abs_value<scalar_t, acc_t>(data));
    }
    acc_t combine(acc_t a, acc_t b) const {
        if (std::isnan(a)) return a;
        if (std::isnan(b)) return b;
        return a > b ? a : b;
    }
    out_t project(acc_t value) const { return static_cast<out_t>(value); }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct NormOps {
    acc_t norm;

    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        return acc + static_cast<acc_t>(std::pow(
            norm_abs_value<scalar_t, acc_t>(data), norm));
    }
    acc_t combine(acc_t a, acc_t b) const { return a + b; }
    out_t project(acc_t value) const {
        return static_cast<out_t>(std::pow(value, acc_t(1) / norm));
    }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

// L2 norm fast path. The previous TensorPlay implementation composed
// native path reduces squares directly; this is particularly important for
// Muon's bfloat16 normalization step.

// --- tier-compiled sum of squares over a contiguous stream ------------------
namespace {

// The reduced widths are squared and summed in single precision: the square
// of an element of either width does not fit the layout it came in.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma,f16c"), noinline))
float sumsq_f16_chunk(const uint16_t* x, int64_t b, int64_t e) {
    // Four accumulators stay in flight: one chain would wait on the latency
    // of the fold, not on the stream it reads.
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int64_t i = b;
    for (; i + 32 <= e; i += 32) {
        const __m256 v0 = tensorplay::cpu::vecreduced::widen8(x + i);
        const __m256 v1 = tensorplay::cpu::vecreduced::widen8(x + i + 8);
        const __m256 v2 = tensorplay::cpu::vecreduced::widen8(x + i + 16);
        const __m256 v3 = tensorplay::cpu::vecreduced::widen8(x + i + 24);
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
        a2 = _mm256_fmadd_ps(v2, v2, a2);
        a3 = _mm256_fmadd_ps(v3, v3, a3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    for (; i + 8 <= e; i += 8) {
        const __m256 v = tensorplay::cpu::vecreduced::widen8(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float s = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
              ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; i < e; ++i) {
        const float v = tensorplay::cpu::vecreduced::widen_one(x[i]);
        s += v * v;
    }
    return s;
}

__attribute__((target("avx2,fma"), noinline))
float sumsq_bf16_chunk(const uint16_t* x, int64_t b, int64_t e) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int64_t i = b;
    for (; i + 32 <= e; i += 32) {
        const __m256 v0 = tensorplay::cpu::vecreduced::widen_bf16_8(x + i);
        const __m256 v1 = tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 8);
        const __m256 v2 = tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 16);
        const __m256 v3 = tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 24);
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
        a2 = _mm256_fmadd_ps(v2, v2, a2);
        a3 = _mm256_fmadd_ps(v3, v3, a3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    for (; i + 8 <= e; i += 8) {
        const __m256 v = tensorplay::cpu::vecreduced::widen_bf16_8(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float s = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
              ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; i < e; ++i) {
        const float v = tensorplay::cpu::vecreduced::widen_bf16_one(x[i]);
        s += v * v;
    }
    return s;
}

// The magnitudes of a contiguous reduced-width buffer summed in single
// precision, four chains of lanes in flight.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
float sum_abs_reduced_chunk(const uint16_t* x, int64_t b, int64_t e) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int64_t i = b;
    for (; i + 32 <= e; i += 32) {
        const __m256 v0 = BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i)
            : tensorplay::cpu::vecreduced::widen8(x + i);
        const __m256 v1 = BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 8)
            : tensorplay::cpu::vecreduced::widen8(x + i + 8);
        const __m256 v2 = BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 16)
            : tensorplay::cpu::vecreduced::widen8(x + i + 16);
        const __m256 v3 = BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 24)
            : tensorplay::cpu::vecreduced::widen8(x + i + 24);
        a0 = _mm256_add_ps(a0, _mm256_andnot_ps(sign, v0));
        a1 = _mm256_add_ps(a1, _mm256_andnot_ps(sign, v1));
        a2 = _mm256_add_ps(a2, _mm256_andnot_ps(sign, v2));
        a3 = _mm256_add_ps(a3, _mm256_andnot_ps(sign, v3));
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    for (; i + 8 <= e; i += 8) {
        const __m256 v = BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i)
            : tensorplay::cpu::vecreduced::widen8(x + i);
        acc = _mm256_add_ps(acc, _mm256_andnot_ps(sign, v));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float s = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
              ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; i < e; ++i) {
        const float v = BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_one(x[i])
            : tensorplay::cpu::vecreduced::widen_one(x[i]);
        s += std::abs(v);
    }
    return s;
}

// The squares of a contiguous reduced-width buffer, one slot per worker.
inline bool normsq_reduced(const void* xv, int64_t n, DType dt, int64_t nslots,
                           double* out) {
    constexpr int64_t kGrain = 32768;
    const bool half_width = dt == DType::Float16;
    if (half_width) {
        if (!tensorplay::cpu::vecreduced::f16c_available()) return false;
    } else if (dt != DType::BFloat16) {
        return false;
    }
    const uint16_t* x = static_cast<const uint16_t*>(xv);
    std::vector<float> part(static_cast<size_t>(nslots), 0.f);
    tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
        float& slot = part[static_cast<size_t>(tensorplay::parallel::get_thread_num())];
        slot += half_width ? sumsq_f16_chunk(x, b, e)
                           : sumsq_bf16_chunk(x, b, e);
    });
    float s = 0.f;
    for (int64_t k = 0; k < nslots; ++k) s += part[static_cast<size_t>(k)];
    *out = static_cast<double>(s);
    return true;
}
#else
inline bool normsq_reduced(const void*, int64_t, DType, int64_t, double*) {
    return false;
}
#endif

// Streams the raw sum of squares over a contiguous buffer; returns false when
// the caller should fall back to the iterator path (short inputs or a dtype
// without a streaming path).  The caller applies the square root in the
// element dtype.
static bool try_normsq_real(const void* xv, int64_t n, DType dt, double* out) {
    if (n < 4096) return false;
    constexpr int64_t kGrain = 32768;
    // One slot per worker, indexed by the worker id: a chunk boundary is the
    // team's even split, not a multiple of the grain, so the chunk start is
    // not a slot number.
    const int64_t nslots = std::max<int64_t>(get_num_threads(), 1);
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        std::vector<float> part(nslots, 0.f);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[get_thread_num()] += vec::sum_of_squares(x, b, e);
        });
        float s = 0.f;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = static_cast<double>(s);
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        std::vector<double> part(nslots, 0.0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[get_thread_num()] += vec::sum_of_squares(x, b, e);
        });
        double s = 0.0;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = s;
        return true;
    }
    return normsq_reduced(xv, n, dt, nslots, out);
}

// --- 256-bit packed reduction walks (x86, per-function target) --------------
#if defined(TP_VECREDUCED_X86)

// The plain sum of a reduced-width stream, in single precision like the
// squares above: one row of twenty-four lanes folded through four chains.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
float sum_reduced_chunk(const uint16_t* x, int64_t b, int64_t e) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int64_t i = b;
    for (; i + 32 <= e; i += 32) {
        a0 = _mm256_add_ps(a0, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i)
            : tensorplay::cpu::vecreduced::widen8(x + i));
        a1 = _mm256_add_ps(a1, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 8)
            : tensorplay::cpu::vecreduced::widen8(x + i + 8));
        a2 = _mm256_add_ps(a2, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 16)
            : tensorplay::cpu::vecreduced::widen8(x + i + 16));
        a3 = _mm256_add_ps(a3, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 24)
            : tensorplay::cpu::vecreduced::widen8(x + i + 24));
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    for (; i + 8 <= e; i += 8) {
        acc = _mm256_add_ps(acc, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i)
            : tensorplay::cpu::vecreduced::widen8(x + i));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float s = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
              ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; i < e; ++i) {
        s += BFloat ? tensorplay::cpu::vecreduced::widen_bf16_one(x[i])
                    : tensorplay::cpu::vecreduced::widen_one(x[i]);
    }
    return s;
}

inline bool reduced_sum_takes(DType dt) {
    if (dt == DType::Float16) {
        return tensorplay::cpu::vecreduced::f16c_available();
    }
    return dt == DType::BFloat16;
}

// How a thin row of packed values folds down to its one answer.
enum class ThinRowFold { Sum, AbsSum, Product, TwoNorm };

// A last axis narrower than one step: a whole block of rows is widened into a
// float scratch at once, so the crossings are a streaming walk rather than a
// call per row, and the rows are folded in single precision there.
template <bool BFloat, ThinRowFold Fold>
__attribute__((target("avx2,fma,f16c"), noinline))
void fold_thin_rows(const uint16_t* in, int64_t rows, int64_t run,
                    uint16_t* out) {
    static thread_local std::vector<float> scratch;
    const int64_t block_rows =
        std::max<int64_t>(1, 2048 / std::max<int64_t>(run, 1));
    const int64_t capacity = std::min(rows, block_rows) * run;
    if (static_cast<int64_t>(scratch.size()) < capacity) {
        scratch.resize(static_cast<size_t>(capacity));
    }
    float* w = scratch.data();
    for (int64_t r0 = 0; r0 < rows; r0 += block_rows) {
        const int64_t nr = std::min(block_rows, rows - r0);
        const uint16_t* src = in + r0 * run;
        const int64_t n = nr * run;
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(w + i, BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(src + i)
                : tensorplay::cpu::vecreduced::widen8(src + i));
        }
        for (; i < n; ++i) {
            w[i] = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_one(src[i])
                          : tensorplay::cpu::vecreduced::widen_one(src[i]);
        }
        for (int64_t r = 0; r < nr; ++r) {
            const float* rp = w + r * run;
            float value = Fold == ThinRowFold::Product ? 1.f : 0.f;
            for (int64_t k = 0; k < run; ++k) {
                value = Fold == ThinRowFold::Product
                    ? value * rp[k]
                    : Fold == ThinRowFold::TwoNorm
                        ? value + rp[k] * rp[k]
                        : Fold == ThinRowFold::AbsSum ? value + std::abs(rp[k])
                                                      : value + rp[k];
            }
            const float answer =
                Fold == ThinRowFold::TwoNorm ? std::sqrt(value) : value;
            out[r0 + r] = BFloat
                ? tensorplay::cpu::vecreduced::narrow_bf16_one(answer)
                : tensorplay::cpu::vecreduced::narrow_one_hw(answer);
        }
    }
}

// A running product that has decayed into the range a float no longer holds
// with full precision is multiplied far more slowly than a normal one, and a
// column of small factors spends most of its walk there.  Each column of a band
// is therefore scaled up by a power of two as soon as it falls below 2^-64 --
// its value is unchanged, since the factor is exact -- and scaled back once
// when its totals leave; a column that has grown past 2^60 is scaled back at
// once so that the scaling can never overflow.  Columns are scaled one at a
// time, so a band whose columns decay at different rates keeps every one of
// them in the range where a multiply answers at full speed.
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
void keep_totals_in_range(__m256& a, __m256i& shiftv) {
    // A column is scaled by its own magnitude: a negative column of ordinary
    // size is not small, and a column that is already zero stays there.
    const __m256 magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.f), a);
    const __m256 small = _mm256_and_ps(
        _mm256_cmp_ps(magnitude, _mm256_set1_ps(0x1p-64f), _CMP_LT_OQ),
        _mm256_cmp_ps(magnitude, _mm256_setzero_ps(), _CMP_GT_OQ));
    // The ordinary walk holds a total that is neither small nor carried, and
    // it answers that case without the compare that asks whether a carried
    // total has grown past the range it would be brought back from.
    const bool is_small = _mm256_movemask_ps(small) != 0;
    const bool is_live = _mm256_testz_si256(shiftv, shiftv) == 0;
    if (!is_small && !is_live) return;
    const __m256i sixty_four = _mm256_set1_epi32(64);
    if (is_small) {
        a = _mm256_blendv_ps(a, _mm256_mul_ps(a, _mm256_set1_ps(0x1p64f)),
                             small);
        shiftv = _mm256_add_epi32(
            shiftv, _mm256_and_si256(_mm256_castps_si256(small), sixty_four));
    }
    const __m256i grown = _mm256_and_si256(
        _mm256_castps_si256(
            _mm256_cmp_ps(magnitude, _mm256_set1_ps(0x1p60f), _CMP_GT_OQ)),
        _mm256_cmpgt_epi32(shiftv, _mm256_setzero_si256()));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(grown)) != 0) {
        a = _mm256_blendv_ps(a, _mm256_mul_ps(a, _mm256_set1_ps(0x1p-64f)),
                             _mm256_castsi256_ps(grown));
        shiftv = _mm256_sub_epi32(shiftv, _mm256_and_si256(grown, sixty_four));
    }
}

// A packed row of a reduced width multiplied through single precision, four
// chains of lanes in flight.  A long row of small factors walks a chain down
// into the range a float no longer holds with full precision, where a multiply
// costs far more than a normal one and most of the row is spent; each chain is
// therefore held in range by an exact power of two, and the scaling its lanes
// carry is applied to the row's answer once, as one factor, so no intermediate
// is ever written below the range.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
float product_reduced_chunk(const uint16_t* x, int64_t b, int64_t e) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 a0 = one, a1 = one, a2 = one, a3 = one;
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = s0, s2 = s0, s3 = s0;
    int64_t i = b;
    // A check reads the magnitude of every lane, so it is made once over four
    // factors: a lane that has decayed this far stays inside the range a float
    // carries for another four, and a chain of eight columns is what pays for
    // the check otherwise.
    for (; i + 128 <= e; i += 128) {
        for (int64_t k = 0; k < 4; ++k) {
            const uint16_t* p = x + i + 32 * k;
            a0 = _mm256_mul_ps(a0, BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(p)
                : tensorplay::cpu::vecreduced::widen8(p));
            a1 = _mm256_mul_ps(a1, BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(p + 8)
                : tensorplay::cpu::vecreduced::widen8(p + 8));
            a2 = _mm256_mul_ps(a2, BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(p + 16)
                : tensorplay::cpu::vecreduced::widen8(p + 16));
            a3 = _mm256_mul_ps(a3, BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(p + 24)
                : tensorplay::cpu::vecreduced::widen8(p + 24));
        }
        keep_totals_in_range(a0, s0);
        keep_totals_in_range(a1, s1);
        keep_totals_in_range(a2, s2);
        keep_totals_in_range(a3, s3);
    }
    for (; i + 32 <= e; i += 32) {
        a0 = _mm256_mul_ps(a0, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i)
            : tensorplay::cpu::vecreduced::widen8(x + i));
        a1 = _mm256_mul_ps(a1, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 8)
            : tensorplay::cpu::vecreduced::widen8(x + i + 8));
        a2 = _mm256_mul_ps(a2, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 16)
            : tensorplay::cpu::vecreduced::widen8(x + i + 16));
        a3 = _mm256_mul_ps(a3, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i + 24)
            : tensorplay::cpu::vecreduced::widen8(x + i + 24));
        keep_totals_in_range(a0, s0);
        keep_totals_in_range(a1, s1);
        keep_totals_in_range(a2, s2);
        keep_totals_in_range(a3, s3);
    }
    __m256 acc = _mm256_mul_ps(_mm256_mul_ps(a0, a1), _mm256_mul_ps(a2, a3));
    __m256i sa = _mm256_setzero_si256();
    for (; i + 8 <= e; i += 8) {
        acc = _mm256_mul_ps(acc, BFloat
            ? tensorplay::cpu::vecreduced::widen_bf16_8(x + i)
            : tensorplay::cpu::vecreduced::widen8(x + i));
        keep_totals_in_range(acc, sa);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float p = ((lanes[0] * lanes[1]) * (lanes[2] * lanes[3])) *
              ((lanes[4] * lanes[5]) * (lanes[6] * lanes[7]));
    for (; i < e; ++i) {
        p *= BFloat ? tensorplay::cpu::vecreduced::widen_bf16_one(x[i])
                    : tensorplay::cpu::vecreduced::widen_one(x[i]);
    }
    // Every lane of the answer above carries its own power of two; the row's
    // answer takes them back as a single factor, which is exact while the
    // answer stays in range and rounds once when it does not.
    const __m256i total = _mm256_add_epi32(
        _mm256_add_epi32(s0, s1), _mm256_add_epi32(_mm256_add_epi32(s2, s3), sa));
    __m128i held = _mm_add_epi32(_mm256_castsi256_si128(total),
                                 _mm256_extracti128_si256(total, 1));
    held = _mm_add_epi32(held, _mm_shuffle_epi32(held, 0x4E));
    held = _mm_add_epi32(held, _mm_shuffle_epi32(held, 0xB1));
    const int scaling = _mm_cvtsi128_si32(held);
    if (scaling != 0 && std::isfinite(p)) {
        p = static_cast<float>(std::ldexp(static_cast<double>(p), -scaling));
    }
    return p;
}

// A packed row of a reduced width summed through a range of columns, folded
// into the column totals a caller keeps.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), always_inline))
inline __m256 widen_row(const uint16_t* p) {
    if constexpr (BFloat) return tensorplay::cpu::vecreduced::widen_bf16_8(p);
    return tensorplay::cpu::vecreduced::widen8(p);
}

template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
void accumulate_rows_reduced(const uint16_t* x, int64_t begin, int64_t end,
                             int64_t cols, float* acc) {
    // Four rows land in one column total before that total is read back, so
    // the column buffer is touched a quarter as often.
    int64_t r = begin;
    const int64_t r4 = begin + (end - begin) / 4 * 4;
    for (; r < r4; r += 4) {
        const uint16_t* r0 = x + r * cols;
        const uint16_t* r1 = r0 + cols;
        const uint16_t* r2 = r1 + cols;
        const uint16_t* r3 = r2 + cols;
        int64_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256 sum = _mm256_add_ps(
                _mm256_add_ps(widen_row<BFloat>(r0 + c), widen_row<BFloat>(r1 + c)),
                _mm256_add_ps(widen_row<BFloat>(r2 + c), widen_row<BFloat>(r3 + c)));
            _mm256_storeu_ps(acc + c, _mm256_add_ps(_mm256_loadu_ps(acc + c), sum));
        }
        // A row that ends mid-band crosses its last few columns here.  The
        // crossing is the instruction the rest of the walk uses, so it is
        // written out at the width the walk holds rather than reached through
        // a call that cannot be inlined into the loop.
        if constexpr (BFloat) {
            for (; c < cols; ++c) {
                acc[c] += ((tensorplay::cpu::vecreduced::widen_bf16_one(r0[c]) +
                            tensorplay::cpu::vecreduced::widen_bf16_one(r1[c])) +
                           (tensorplay::cpu::vecreduced::widen_bf16_one(r2[c]) +
                            tensorplay::cpu::vecreduced::widen_bf16_one(r3[c])));
            }
        } else {
            for (; c < cols; ++c) {
                acc[c] += ((_cvtsh_ss(r0[c]) + _cvtsh_ss(r1[c])) +
                           (_cvtsh_ss(r2[c]) + _cvtsh_ss(r3[c])));
            }
        }
    }
    for (; r < end; ++r) {
        const uint16_t* rp = x + r * cols;
        int64_t c = 0;
        for (; c + 8 <= cols; c += 8) {
            _mm256_storeu_ps(acc + c,
                             _mm256_add_ps(_mm256_loadu_ps(acc + c),
                                           widen_row<BFloat>(rp + c)));
        }
        for (; c < cols; ++c) {
            acc[c] += BFloat ? tensorplay::cpu::vecreduced::widen_bf16_one(rp[c])
                             : _cvtsh_ss(rp[c]);
        }
    }
}

// The column totals of every worker, merged in the order the workers were
// numbered and crossed to the layout of the answer a vector at a time.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
void store_column_totals(const float* const* partials, int64_t nthreads,
                         int64_t cols, uint16_t* out) {
    int64_t c = 0;
    for (; c + 8 <= cols; c += 8) {
        __m256 s = _mm256_setzero_ps();
        for (int64_t t = 0; t < nthreads; ++t) {
            s = _mm256_add_ps(s, _mm256_loadu_ps(partials[t] + c));
        }
        if constexpr (BFloat) {
            tensorplay::cpu::vecreduced::store_bf16_8(out + c, s);
        } else {
            tensorplay::cpu::vecreduced::lay_f16_8<false>(out + c, s);
        }
    }
    for (; c < cols; ++c) {
        float s = 0.f;
        for (int64_t t = 0; t < nthreads; ++t) s += partials[t][c];
        if constexpr (BFloat) {
            out[c] = tensorplay::cpu::vecreduced::narrow_bf16_one(s);
        } else {
            out[c] = tensorplay::cpu::vecreduced::narrow_one_hw(s);
        }
    }
}

// The column totals of a product narrowed a band at a time: the totals of every
// worker multiply a lane at a time and the packed answer is formed once per
// band rather than once per column.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
void store_column_products(const float* const* partials, int64_t nthreads,
                           int64_t cols, uint16_t* out) {
    int64_t c = 0;
    for (; c + 8 <= cols; c += 8) {
        __m256 p = _mm256_loadu_ps(partials[0] + c);
        for (int64_t t = 1; t < nthreads; ++t) {
            p = _mm256_mul_ps(p, _mm256_loadu_ps(partials[t] + c));
        }
        if constexpr (BFloat) {
            tensorplay::cpu::vecreduced::store_bf16_8(out + c, p);
        } else {
            tensorplay::cpu::vecreduced::lay_f16_8<false>(out + c, p);
        }
    }
    for (; c < cols; ++c) {
        float p = 1.f;
        for (int64_t t = 0; t < nthreads; ++t) {
            p *= partials[t][c];
        }
        if constexpr (BFloat) {
            out[c] = tensorplay::cpu::vecreduced::narrow_bf16_one(p);
        } else {
            out[c] = tensorplay::cpu::vecreduced::narrow_one_hw(p);
        }
    }
}

// The last axis of a packed tensor, multiplied a row at a time, with the
// thin case widened whole.
inline std::optional<Tensor> try_product_lastdim_reduced(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    if (!input.is_contiguous() || !reduced_sum_takes(input.dtype())) {
        return std::nullopt;
    }
    const int64_t d_size = input.size(input.dim() - 1);
    if (d_size < 1 || input.numel() == 0) return std::nullopt;
    const int64_t rows = input.numel() / d_size;
    const bool half_width = input.dtype() == DType::Float16;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    uint16_t* outp = half_width
        ? reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::BFloat16>());
    const int64_t row_grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (d_size < 8) {
        parallel_for(0, rows, row_grain, [&](int64_t b, int64_t e) {
            if (half_width) {
                fold_thin_rows<false, ThinRowFold::Product>(in + b * d_size, e - b, d_size,
                                            outp + b);
            } else {
                fold_thin_rows<true, ThinRowFold::Product>(in + b * d_size, e - b, d_size,
                                           outp + b);
            }
        });
        return out;
    }
    parallel_for(0, rows, row_grain, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const uint16_t* row = in + r * d_size;
            if (half_width) {
                outp[r] = tensorplay::cpu::vecreduced::narrow_one_hw(
                    product_reduced_chunk<false>(row, 0, d_size));
            } else {
                outp[r] = tensorplay::cpu::vecreduced::narrow_bf16_one(
                    product_reduced_chunk<true>(row, 0, d_size));
            }
        }
    });
    return out;
}

// One step of eight packed values widened into single precision.  It is a free
// function so that a caller keeping its own target attribute can reach it.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
__m256 widen_row8(const uint16_t* p) {
    using tensorplay::cpu::vecreduced::widen8;
    using tensorplay::cpu::vecreduced::widen_bf16_8;
    return BFloat ? widen_bf16_8(p) : widen8(p);
}

// The product of a small group of rows, one band of eight columns at a time,
// folded as a balanced tree.  A total that runs through a long column of
// factors passes through the range a float cannot hold without losing
// precision, and every multiply inside that range costs far more than a normal
// one; folding a group first keeps the group's own products in the normal range
// so the total crosses the range a group at a time.
template <bool BFloat, int64_t Group>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
__m256 fold_rows_hw(const uint16_t* p, int64_t cols) {
    if constexpr (Group == 4) {
        return _mm256_mul_ps(
            _mm256_mul_ps(widen_row8<BFloat>(p), widen_row8<BFloat>(p + cols)),
            _mm256_mul_ps(widen_row8<BFloat>(p + 2 * cols),
                          widen_row8<BFloat>(p + 3 * cols)));
    } else {
        return _mm256_mul_ps(fold_rows_hw<BFloat, Group / 2>(p, cols),
                             fold_rows_hw<BFloat, Group / 2>(
                                 p + (Group / 2) * cols, cols));
    }
}

// The scaling a band carries applied to its totals, one power of two at a time:
// each step leaves a value no smaller than the answer that follows, so an
// intermediate cannot flush to zero while the answer would not.
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
void rescale_totals(__m256& a, __m256i shiftv) {
    __m256i live = _mm256_cmpgt_epi32(shiftv, _mm256_setzero_si256());
    while (_mm256_movemask_ps(_mm256_castsi256_ps(live)) != 0) {
        a = _mm256_blendv_ps(a, _mm256_mul_ps(a, _mm256_set1_ps(0x1p-64f)),
                             _mm256_castsi256_ps(live));
        shiftv = _mm256_sub_epi32(
            shiftv, _mm256_and_si256(live, _mm256_set1_epi32(64)));
        live = _mm256_cmpgt_epi32(shiftv, _mm256_setzero_si256());
    }
}

// The same scaling applied in one step: a power of two as far below the range
// of a float as the walk wants is formed exactly in double precision and the
// totals cross back in a single multiply, so a total that has been carried a
// long way does not spend a dependent multiply per step it was scaled by.
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
void rescale_totals_wide(__m256& a, __m256i shiftv) {
    if (_mm256_testz_si256(shiftv, shiftv) != 0) return;
    shiftv = _mm256_min_epi32(shiftv, _mm256_set1_epi32(1022));
    const __m256i bias = _mm256_set1_epi64x(1023);
    const __m256i e_lo = _mm256_slli_epi64(
        _mm256_sub_epi64(bias,
                         _mm256_cvtepi32_epi64(_mm256_castsi256_si128(shiftv))),
        52);
    const __m256i e_hi = _mm256_slli_epi64(
        _mm256_sub_epi64(
            bias, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(shiftv, 1))),
        52);
    const __m256d lo = _mm256_mul_pd(
        _mm256_cvtps_pd(_mm256_castps256_ps128(a)), _mm256_castsi256_pd(e_lo));
    const __m256d hi = _mm256_mul_pd(
        _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)), _mm256_castsi256_pd(e_hi));
    a = _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo));
}

// The column totals carry a scaling of their own: a total that has decayed
// below the range a float holds with full precision is kept by an exact power
// of two, and the power its columns carry is remembered alongside them, so a
// walk over many blocks of rows never has to cross back into that range and
// return, which would cost a dependent multiply on a value a float carries
// without full precision at the head of every block.  The totals are taken back
// once, after the last block.
__attribute__((target("avx2,fma,f16c"), noinline))
void unravel_column_totals(float* acc, const int32_t* shifts, int64_t cols) {
    int64_t c = 0;
    for (; c + 8 <= cols; c += 8) {
        __m256 a = _mm256_loadu_ps(acc + c);
        const __m256i s = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(shifts + c));
        rescale_totals_wide(a, s);
        _mm256_storeu_ps(acc + c, a);
    }
    for (; c < cols; ++c) {
        acc[c] = static_cast<float>(std::ldexp(
            static_cast<double>(acc[c]),
            -std::min<int64_t>(shifts[c], 1022)));
    }
}

// The leading axis of a packed tensor, multiplied a whole row into per-worker
// column totals.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
void multiply_rows_reduced(const uint16_t* x, int64_t begin, int64_t end,
                           int64_t cols, float* acc, int32_t* shifts) {
    // A block of rows folds into the column totals while four bands of eight
    // columns stay in registers, so the totals are read and written once per
    // block rather than once per row.  Sixteen rows at a time are multiplied
    // among themselves first and that product then folded into the total: over
    // a long column of factors the running total must pass through the range
    // a float carries without full precision, and each multiply made there
    // costs far more than a normal one.  A tree leaves the group's own
    // products in the normal range, so the total crosses that range a group of
    // rows at a time, while a wider group would no longer pay for itself.
    constexpr int64_t kFold = 16;
    const int64_t rows = end - begin;
    int64_t block = std::max<int64_t>(
        kFold, 24576 / std::max<int64_t>(cols, 1));
    // One block cannot decay far enough for a scaling to be worth carrying
    // past it, so its totals are taken back where the block ends and the
    // scaling columns are never touched.
    const bool carried = rows > block;
    block = std::min<int64_t>(block, rows);
    for (int64_t r0 = begin; r0 < end; r0 += block) {
        const int64_t r1 = std::min(end, r0 + block);
        int64_t c = 0;
        for (; c + 32 <= cols; c += 32) {
            __m256 a[4];
            __m256i shift[4];
            for (int64_t k = 0; k < 4; ++k) {
                a[k] = _mm256_loadu_ps(acc + c + 8 * k);
                shift[k] = carried
                    ? _mm256_loadu_si256(
                          reinterpret_cast<const __m256i*>(shifts + c + 8 * k))
                    : _mm256_setzero_si256();
            }
            int64_t r = r0;
            for (; r + kFold <= r1; r += kFold) {
                for (int64_t k = 0; k < 4; ++k) {
                    const uint16_t* rp = x + r * cols + c + 8 * k;
                    a[k] = _mm256_mul_ps(a[k],
                                         fold_rows_hw<BFloat, kFold>(rp, cols));
                    keep_totals_in_range(a[k], shift[k]);
                }
            }
            for (; r + 4 <= r1; r += 4) {
                for (int64_t k = 0; k < 4; ++k) {
                    const uint16_t* rp = x + r * cols + c + 8 * k;
                    a[k] = _mm256_mul_ps(a[k],
                                         fold_rows_hw<BFloat, 4>(rp, cols));
                    keep_totals_in_range(a[k], shift[k]);
                }
            }
            for (; r < r1; ++r) {
                const uint16_t* rp = x + r * cols + c;
                for (int64_t k = 0; k < 4; ++k) {
                    a[k] = _mm256_mul_ps(a[k], widen_row8<BFloat>(rp + 8 * k));
                    keep_totals_in_range(a[k], shift[k]);
                }
            }
            for (int64_t k = 0; k < 4; ++k) {
                if (!carried) rescale_totals_wide(a[k], shift[k]);
                _mm256_storeu_ps(acc + c + 8 * k, a[k]);
                if (carried) {
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(shifts + c + 8 * k),
                        shift[k]);
                }
            }
        }
        for (; c + 8 <= cols; c += 8) {
            __m256 a = _mm256_loadu_ps(acc + c);
            __m256i shift = carried
                ? _mm256_loadu_si256(
                      reinterpret_cast<const __m256i*>(shifts + c))
                : _mm256_setzero_si256();
            int64_t r = r0;
            for (; r + kFold <= r1; r += kFold) {
                a = _mm256_mul_ps(
                    a, fold_rows_hw<BFloat, kFold>(x + r * cols + c, cols));
                keep_totals_in_range(a, shift);
            }
            for (; r + 4 <= r1; r += 4) {
                a = _mm256_mul_ps(
                    a, fold_rows_hw<BFloat, 4>(x + r * cols + c, cols));
                keep_totals_in_range(a, shift);
            }
            for (; r < r1; ++r) {
                a = _mm256_mul_ps(a, widen_row8<BFloat>(x + r * cols + c));
                keep_totals_in_range(a, shift);
            }
            if (!carried) rescale_totals_wide(a, shift);
            _mm256_storeu_ps(acc + c, a);
            if (carried) {
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(shifts + c), shift);
            }
        }
        if (c < cols) {
            // The last columns of a row do not fill a band.  A band read from
            // there reaches into the next row, and those lanes are multiplied
            // as well, but a product is formed lane by lane, so only the
            // columns the row really has are written back.  The tensor's own
            // last row has no row after it and reads through a copy.
            const int64_t keep = cols - c;
            alignas(32) float held[8] = {1.f, 1.f, 1.f, 1.f,
                                         1.f, 1.f, 1.f, 1.f};
            alignas(32) int32_t held_shift[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            std::memcpy(held, acc + c,
                        static_cast<size_t>(keep) * sizeof(float));
            __m256 a = _mm256_load_ps(held);
            __m256i shift = _mm256_setzero_si256();
            if (carried) {
                std::memcpy(held_shift, shifts + c,
                            static_cast<size_t>(keep) * sizeof(int32_t));
                shift = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(held_shift));
            }
            int64_t r = r0;
            for (; r + kFold <= r1; r += kFold) {
                a = _mm256_mul_ps(
                    a, fold_rows_hw<BFloat, kFold>(x + r * cols + c, cols));
                keep_totals_in_range(a, shift);
            }
            for (; r < r1; ++r) {
                const uint16_t* rp = x + r * cols + c;
                __m256 b;
                if (r == rows - 1) {
                    alignas(16) uint16_t copy[8];
                    std::memcpy(copy, rp,
                                static_cast<size_t>(keep) * sizeof(uint16_t));
                    for (int64_t k = keep; k < 8; ++k) {
                        copy[k] = BFloat ? 0x3F80u : 0x3C00u;
                    }
                    b = widen_row8<BFloat>(copy);
                } else {
                    b = widen_row8<BFloat>(rp);
                }
                a = _mm256_mul_ps(a, b);
                keep_totals_in_range(a, shift);
            }
            if (!carried) rescale_totals_wide(a, shift);
            _mm256_store_ps(held, a);
            _mm256_store_si256(reinterpret_cast<__m256i*>(held_shift), shift);
            std::memcpy(acc + c, held,
                        static_cast<size_t>(keep) * sizeof(float));
            if (carried) {
                std::memcpy(shifts + c, held_shift,
                            static_cast<size_t>(keep) * sizeof(int32_t));
            }
        }
    }
}

inline std::optional<Tensor> try_product_leading_reduced(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    if (!input.is_contiguous() || input.dim() < 1 ||
        !reduced_sum_takes(input.dtype())) {
        return std::nullopt;
    }
    const int64_t rows = input.size(0);
    const int64_t cols = rows > 0 ? input.numel() / rows : 0;
    if (rows == 0 || cols < 1) return std::nullopt;
    const bool half_width = input.dtype() == DType::Float16;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    const int64_t nthreads = std::max<int64_t>(1, get_num_threads());
    std::vector<std::vector<float>> partials(
        static_cast<size_t>(nthreads),
        std::vector<float>(static_cast<size_t>(cols), 1.f));
    const int64_t rows_per_block =
        std::max<int64_t>(16, 24576 / std::max<int64_t>(cols, 1));
    const bool carried = rows > rows_per_block;
    std::vector<std::vector<int32_t>> carried_shifts(
        static_cast<size_t>(carried ? nthreads : 0),
        std::vector<int32_t>(static_cast<size_t>(cols), 0));
    const int64_t row_grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(cols, 1));
    parallel_for(0, rows, row_grain, [&](int64_t b, int64_t e) {
        const size_t worker = static_cast<size_t>(get_thread_num());
        float* acc = partials[worker].data();
        if (!carried) {
            if (half_width) {
                multiply_rows_reduced<false>(in, b, e, cols, acc, nullptr);
            } else {
                multiply_rows_reduced<true>(in, b, e, cols, acc, nullptr);
            }
            return;
        }
        int32_t* shifts = carried_shifts[worker].data();
        if (half_width) {
            multiply_rows_reduced<false>(in, b, e, cols, acc, shifts);
        } else {
            multiply_rows_reduced<true>(in, b, e, cols, acc, shifts);
        }
        unravel_column_totals(acc, shifts, cols);
    });
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    uint16_t* outp = half_width
        ? reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::BFloat16>());
    std::vector<const float*> columns(static_cast<size_t>(nthreads), nullptr);
    for (int64_t t = 0; t < nthreads; ++t) {
        columns[static_cast<size_t>(t)] = partials[static_cast<size_t>(t)].data();
    }
#if defined(__x86_64__)
    if (tensorplay::cpu::vecreduced::f16c_available()) {
        if (half_width) {
            store_column_products<false>(columns.data(), nthreads, cols, outp);
        } else {
            store_column_products<true>(columns.data(), nthreads, cols, outp);
        }
        return out;
    }
#endif
    for (int64_t c = 0; c < cols; ++c) {
        float p = 1.f;
        for (int64_t t = 0; t < nthreads; ++t) {
            p *= columns[static_cast<size_t>(t)][static_cast<size_t>(c)];
        }
        if (half_width) {
            outp[c] = tensorplay::cpu::vecreduced::narrow_one_hw(p);
        } else {
            outp[c] = tensorplay::cpu::vecreduced::narrow_bf16_one(p);
        }
    }
    return out;
}

#endif  // TP_VECREDUCED_X86

// A slice of columns multiplied row after row in the slice's own width; four
// rows are folded among themselves first, which is what keeps the chain of
// multiplies on one column from being the loop's critical path.
template <typename T>
inline void multiply_rows_real(const T* TP_RESTRICT x, int64_t rows,
                               int64_t cols, T* TP_RESTRICT acc) {
    int64_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const T* TP_RESTRICT r0 = x + r * cols;
        const T* TP_RESTRICT r1 = r0 + cols;
        const T* TP_RESTRICT r2 = r1 + cols;
        const T* TP_RESTRICT r3 = r2 + cols;
        for (int64_t j = 0; j < cols; ++j) {
            acc[j] *= ((r0[j] * r1[j]) * (r2[j] * r3[j]));
        }
    }
    for (; r < rows; ++r) {
        const T* TP_RESTRICT row = x + r * cols;
        for (int64_t j = 0; j < cols; ++j) acc[j] *= row[j];
    }
}

// A reduced axis under an outer block: every block multiplies its own rows into
// its own column totals, so the blocks are what the team splits, every element
// is read exactly once, and no block waits on another's totals.
template <typename T>
inline void product_blocks_real(const Tensor& input, Tensor& out, int64_t outer,
                                int64_t red, int64_t cols) {
    const T* in = input.data_ptr<T>();
    T* op = out.data_ptr<T>();
    const int64_t grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(red * cols, 1));
    parallel_for(0, outer, grain, [&](int64_t b, int64_t e) {
        std::vector<T> acc(static_cast<size_t>(cols));
        for (int64_t o = b; o < e; ++o) {
            const T* TP_RESTRICT block = in + o * red * cols;
            T* TP_RESTRICT dst = op + o * cols;
            if (red == 1) {
                std::memcpy(dst, block, static_cast<size_t>(cols) * sizeof(T));
                continue;
            }
            std::fill(acc.begin(), acc.end(), T(1));
            multiply_rows_real<T>(block, red, cols, acc.data());
            std::memcpy(dst, acc.data(),
                        static_cast<size_t>(cols) * sizeof(T));
        }
    });
}

#if defined(TP_VECREDUCED_X86)
// The scaling a half-precision band carries applied to its totals, with the
// columns whose answer the width cannot tell from zero taken out first: a
// scaling step on such a column would produce a value below the smallest a
// float holds, and that step costs far more than the ones above it, while the
// answer is a zero either way.  The bound is the largest magnitude that rounds
// to zero in the packed width.
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
void rescale_totals_half(__m256& a, __m256i& shiftv) {
    const __m256i wide =
        _mm256_cmpgt_epi32(shiftv, _mm256_set1_epi32(128));
    const __m256i bound = _mm256_slli_epi32(
        _mm256_add_epi32(shiftv, _mm256_set1_epi32(102)), 23);
    const __m256 magnitude =
        _mm256_andnot_ps(_mm256_set1_ps(-0.f), a);
    __m256 vanish = _mm256_cmp_ps(magnitude, _mm256_castsi256_ps(bound),
                                  _CMP_LE_OQ);
    vanish = _mm256_or_ps(vanish, _mm256_castsi256_ps(wide));
    a = _mm256_andnot_ps(vanish, a);
    rescale_totals(a, shiftv);
}

// One group of eight-column bands of a block, multiplied out in registers and
// answered once.  Four bands run side by side so that the multiplies of one
// band are not what the loop waits on.
template <bool BFloat, int64_t NB>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
void product_band_group(const uint16_t* first, int64_t red, int64_t cols,
                        uint16_t* dst) {
    using tensorplay::cpu::vecreduced::lay_f16_8;
    using tensorplay::cpu::vecreduced::store_bf16_8;
    __m256 acc[NB];
    __m256i shift[NB];
    if (red >= 16) {
        // A long column of factors decays below the range a float carries
        // without full precision, and every multiply made there costs far more
        // than a normal one.  Sixteen rows are multiplied among themselves
        // first, which leaves a group's own product in the normal range, and
        // the running total is held there by an exact power of two.
        for (int64_t k = 0; k < NB; ++k) {
            acc[k] = fold_rows_hw<BFloat, 16>(first + 8 * k, cols);
            shift[k] = _mm256_setzero_si256();
        }
        int64_t r = 16;
        for (; r + 16 <= red; r += 16) {
            for (int64_t k = 0; k < NB; ++k) {
                acc[k] = _mm256_mul_ps(
                    acc[k], fold_rows_hw<BFloat, 16>(first + 8 * k + r * cols,
                                                     cols));
                keep_totals_in_range(acc[k], shift[k]);
            }
        }
        for (; r < red; ++r) {
            for (int64_t k = 0; k < NB; ++k) {
                acc[k] = _mm256_mul_ps(
                    acc[k],
                    widen_row8<BFloat>(first + 8 * k + r * cols));
                keep_totals_in_range(acc[k], shift[k]);
            }
        }
        for (int64_t k = 0; k < NB; ++k) {
            if constexpr (BFloat) {
                rescale_totals(acc[k], shift[k]);
            } else {
                rescale_totals_half(acc[k], shift[k]);
            }
        }
    } else {
        // A short run cannot decay far enough for the range to matter, so the
        // rows are multiplied in order with nothing else on the path.
        for (int64_t k = 0; k < NB; ++k) {
            acc[k] = widen_row8<BFloat>(first + 8 * k);
        }
        for (int64_t r = 1; r < red; ++r) {
            for (int64_t k = 0; k < NB; ++k) {
                acc[k] = _mm256_mul_ps(
                    acc[k],
                    widen_row8<BFloat>(first + 8 * k + r * cols));
            }
        }
    }
    for (int64_t k = 0; k < NB; ++k) {
        if constexpr (BFloat) {
            store_bf16_8(dst + 8 * k, acc[k]);
        } else {
            lay_f16_8<false>(dst + 8 * k, acc[k]);
        }
    }
}

// A run of rows too short to pay for a widened total walked in memory: each
// band of eight columns keeps its product in a register over the whole run and
// answers once, so the rows are read exactly once, no identity is laid down
// first, and nothing but the answer passes through memory.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
void product_short_packed_hw(const uint16_t* in, uint16_t* out, int64_t outer,
                             int64_t red, int64_t inner) {
    for (int64_t o = 0; o < outer; ++o) {
        const uint16_t* base = in + o * red * inner;
        uint16_t* dst = out + o * inner;
        int64_t b = 0;
        for (; b + 32 <= inner; b += 32) {
            product_band_group<BFloat, 4>(base + b, red, inner, dst + b);
        }
        for (; b + 8 <= inner; b += 8) {
            product_band_group<BFloat, 1>(base + b, red, inner, dst + b);
        }
        // The columns past the last whole band are answered one at a time; a
        // slice that is a whole number of bands, which is the common case,
        // never enters this walk.
        for (int64_t c = b; c < inner; ++c) {
            float value = BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_one(base[c])
                : tensorplay::cpu::vecreduced::widen_one(base[c]);
            for (int64_t r = 1; r < red; ++r) {
                value *= BFloat
                    ? tensorplay::cpu::vecreduced::widen_bf16_one(
                          base[r * inner + c])
                    : tensorplay::cpu::vecreduced::widen_one(
                          base[r * inner + c]);
            }
            dst[c] = BFloat
                ? tensorplay::cpu::vecreduced::narrow_bf16_one(value)
                : tensorplay::cpu::vecreduced::narrow_one_hw(value);
        }
    }
}

// The same over every block of the outer axis, split among the workers the way
// the block walk above splits.
template <bool BFloat>
inline void product_short_rows_packed(const Tensor& input, Tensor& out,
                                      int64_t outer, int64_t red,
                                      int64_t cols) {
    using Packed = std::conditional_t<BFloat, tensorplay::BFloat16,
                                       tensorplay::Half>;
    const uint16_t* in =
        reinterpret_cast<const uint16_t*>(input.data_ptr<Packed>());
    uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr<Packed>());
    const int64_t grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(red * cols, 1));
    parallel_for(0, outer, grain, [&](int64_t b, int64_t e) {
        product_short_packed_hw<BFloat>(in + b * red * cols, op + b * cols,
                                        e - b, red, cols);
    });
}

// The last axis of a packed tensor, one row of lanes at a time.
inline std::optional<Tensor> try_sum_lastdim_reduced(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    if (!input.is_contiguous() || !reduced_sum_takes(input.dtype())) {
        return std::nullopt;
    }
    const int64_t d_size = input.size(input.dim() - 1);
    if (d_size < 1 || input.numel() == 0) return std::nullopt;
    const int64_t rows = input.numel() / d_size;
    const bool half_width = input.dtype() == DType::Float16;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    uint16_t* outp = half_width
        ? reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::BFloat16>());
    const int64_t row_grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (d_size < 8) {
        parallel_for(0, rows, row_grain, [&](int64_t b, int64_t e) {
            if (half_width) {
                fold_thin_rows<false, ThinRowFold::Sum>(in + b * d_size, e - b, d_size,
                                             outp + b);
            } else {
                fold_thin_rows<true, ThinRowFold::Sum>(in + b * d_size, e - b, d_size,
                                            outp + b);
            }
        });
        return out;
    }
    parallel_for(0, rows, row_grain, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const uint16_t* row = in + r * d_size;
            if (half_width) {
                outp[r] = tensorplay::cpu::vecreduced::narrow_one_hw(
                    sum_reduced_chunk<false>(row, 0, d_size));
            } else {
                outp[r] = tensorplay::cpu::vecreduced::narrow_bf16_one(
                    sum_reduced_chunk<true>(row, 0, d_size));
            }
        }
    });
    return out;
}

// The leading axis of a packed tensor: whole rows land in per-worker column
// totals, which are then merged and narrowed once.
inline std::optional<Tensor> try_sum_leading_reduced(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    if (!input.is_contiguous() || input.dim() < 1 ||
        !reduced_sum_takes(input.dtype())) {
        return std::nullopt;
    }
    const int64_t rows = input.size(0);
    const int64_t cols = rows > 0 ? input.numel() / rows : 0;
    // A narrow column count leaves too little work between the column totals
    // being read back and written again; the four-rows-in-one-register walk
    // keeps the totals in flight instead and answers that case faster.  A
    // column count that is a whole number of bands keeps the earlier gate, so
    // a slice wide enough to amortise the read-back is unaffected.
    if (rows == 0) return std::nullopt;
    if (cols < 128 && cols % 8 == 0) return std::nullopt;
    if (cols < 8) return std::nullopt;
    const bool half_width = input.dtype() == DType::Float16;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    const int64_t nthreads = std::max<int64_t>(1, get_num_threads());
    std::vector<std::vector<float>> partials(
        static_cast<size_t>(nthreads), std::vector<float>(static_cast<size_t>(cols), 0.f));
    // A task takes at least four rows, so the four-row fold below engages
    // even when a row already fills a task's worth of work.
    const int64_t row_grain = std::max<int64_t>(
        4, GRAIN_SIZE / std::max<int64_t>(cols, 1));
    parallel_for(0, rows, row_grain, [&](int64_t b, int64_t e) {
        float* acc = partials[static_cast<size_t>(get_thread_num())].data();
        if (half_width) {
            accumulate_rows_reduced<false>(in, b, e, cols, acc);
        } else {
            accumulate_rows_reduced<true>(in, b, e, cols, acc);
        }
    });
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    uint16_t* outp = half_width
        ? reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::BFloat16>());
    std::vector<const float*> columns(static_cast<size_t>(nthreads), nullptr);
    for (int64_t t = 0; t < nthreads; ++t) {
        columns[static_cast<size_t>(t)] = partials[static_cast<size_t>(t)].data();
    }
    if (half_width) {
        store_column_totals<false>(columns.data(), nthreads, cols, outp);
    } else {
        store_column_totals<true>(columns.data(), nthreads, cols, outp);
    }
    return out;
}

#endif  // TP_VECREDUCED_X86

}  // namespace

template <typename scalar_t, typename acc_t, typename out_t>
void run_norm_reduce(TensorIterator& iter, double p) {
    if (p == 0.0) {
        binary_kernel_reduce(iter, NormZeroOps<scalar_t, acc_t, out_t>{}, acc_t(0));
    } else if (p == 1.0) {
        binary_kernel_reduce(iter, NormOneOps<scalar_t, acc_t, out_t>{}, acc_t(0));
    } else if (p == 2.0) {
        binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t>{}, acc_t(0));
    } else if (p == std::numeric_limits<double>::infinity()) {
        binary_kernel_reduce(iter, AbsMaxOps<scalar_t, acc_t, out_t>{}, acc_t(0));
    } else if (p == -std::numeric_limits<double>::infinity()) {
        binary_kernel_reduce(
            iter, AbsMinOps<scalar_t, acc_t, out_t>{},
            std::numeric_limits<acc_t>::infinity());
    } else {
        binary_kernel_reduce(
            iter, NormOps<scalar_t, acc_t, out_t>{static_cast<acc_t>(p)}, acc_t(0));
    }
}

template <typename scalar_t>
Tensor norm_full_typed(const Tensor& self, double p);

template <typename scalar_t>
Tensor norm_dim_typed(const Tensor& self, const std::vector<int64_t>& dims,
                     double p, bool keepdim);

Tensor norm_kernel_impl(const Tensor& self, double p) {
    const DType out_dtype = toRealValueType(self.dtype());
    if (self.numel() == 0) {
        Tensor out = Tensor::zeros({}, out_dtype, self.device());
        if (p == -std::numeric_limits<double>::infinity()) {
            out.fill_(Scalar(std::numeric_limits<double>::infinity()));
        }
        return out;
    }

    if (p == 2.0 && self.is_contiguous()) {
        double r = 0.0;
        if (try_normsq_real(self.data_ptr(), self.numel(), self.dtype(), &r)) {
            Tensor out = Tensor::empty({}, out_dtype, self.device());
            if (self.dtype() == DType::Float32 ||
                isReducedFloatingType(self.dtype())) {
                out.fill_(Scalar(std::sqrt(static_cast<float>(r))));
            } else {
                out.fill_(Scalar(std::sqrt(r)));
            }
            return out;
        }
    }

    #define TP_NORM_FULL_CASE(ctype, dtype_name) \
        case DType::dtype_name: \
            return norm_full_typed<ctype>(self, p)
    switch (self.dtype()) {
        TP_NORM_FULL_CASE(float, Float32);
        TP_NORM_FULL_CASE(double, Float64);
        TP_NORM_FULL_CASE(Half, Float16);
        TP_NORM_FULL_CASE(BFloat16, BFloat16);
        TP_NORM_FULL_CASE(complex<Half>, ComplexHalf);
        TP_NORM_FULL_CASE(complex<float>, ComplexFloat);
        TP_NORM_FULL_CASE(complex<double>, ComplexDouble);
        TP_NORM_FULL_CASE(complex<BFloat16>, BComplex32);
        default:
            TP_THROW(NotImplementedError, "norm: unsupported dtype on CPU");
    }
    #undef TP_NORM_FULL_CASE
}

template <typename scalar_t>
Tensor norm_full_typed(const Tensor& self, double p) {
    using acc_t = typename NormAccumulatorType<scalar_t>::type;
    using out_t = typename scalar_value_type<scalar_t>::type;
    Tensor out = Tensor::zeros({}, toRealValueType(self.dtype()), self.device());
    TensorIterator iter = TensorIterator::reduce_op(out, self);
    run_norm_reduce<scalar_t, acc_t, out_t>(iter, p);
    return out;
}

template <typename scalar_t>
Tensor norm_dim_typed(const Tensor& self, const std::vector<int64_t>& dims,
                     double p, bool keepdim) {
    using acc_t = typename NormAccumulatorType<scalar_t>::type;
    using out_t = typename scalar_value_type<scalar_t>::type;
    const std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::zeros(out_shape, toRealValueType(self.dtype()), self.device());

    std::vector<bool> mask(self.dim(), false);
    for (int64_t d : dims) {
        if (d < 0) d += self.dim();
        TP_CHECK(d >= 0 && d < self.dim(), "norm: dimension out of range");
        TP_CHECK(!mask[static_cast<size_t>(d)], "norm: duplicate dimension");
        mask[static_cast<size_t>(d)] = true;
    }
    Tensor viewed = review_reduce_result(out, self.dim(), mask, keepdim);
    TensorIterator iter = TensorIterator::reduce_op(viewed, self);
    run_norm_reduce<scalar_t, acc_t, out_t>(iter, p);
    return out;
}

// What a column total folds from its elements: a two-norm folds their squares
// and takes one square root of the total, a one-norm folds their magnitudes,
// and a plain sum folds the element itself.
enum class TotalMode { Sum, Abs, Square };

template <typename T, TotalMode Mode>
inline Vectorized<T> fold_total_step(const Vectorized<T>& q) {
    if constexpr (Mode == TotalMode::Square) {
        return q * q;
    } else if constexpr (Mode == TotalMode::Abs) {
        return q.abs();
    } else {
        return q;
    }
}

template <typename T, TotalMode Mode>
inline T fold_total_one(T v) {
    if constexpr (Mode == TotalMode::Square) {
        return v * v;
    } else if constexpr (Mode == TotalMode::Abs) {
        return std::abs(v);
    } else {
        return v;
    }
}

// One reduced axis of a real tensor, walked as (outer, reduce, inner): every
// element of a row folds into a per-column total over the inner direction,
// which is the contiguous one, and the totals are read back once.  A two-norm
// takes one square root of its totals.  Four rows land in a total before it is
// read back, so the walk is limited by the input stream rather than by the
// total's own traffic.
template <typename T, TotalMode Mode>
inline void norm_totals_real_range(const T* in, T* out, int64_t red,
                                   int64_t inner, int64_t block, int64_t nblk,
                                   int64_t begin, int64_t end) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = Vec::size();
    // A task's totals are its own: the buffer is claimed here so every worker
    // thread holds the one it will write, not the one the caller allocated.
    static thread_local std::vector<T> totals;
    if (static_cast<int64_t>(totals.size()) < block) {
        totals.resize(static_cast<size_t>(block));
    }
    {
        T* acc = totals.data();
        for (int64_t task = begin; task < end; ++task) {
            const int64_t o = task / nblk;
            const int64_t c0 = (task % nblk) * block;
            const int64_t n = std::min(block, inner - c0);
            const T* base = in + (o * red) * inner + c0;
            T* dst = out + o * inner + c0;
            std::fill(acc, acc + n, T(0));
            const int64_t vec = n & ~(W - 1);
            int64_t r = 0;
            for (; r + 4 <= red; r += 4) {
                const T* p0 = base + (r + 0) * inner;
                const T* p1 = base + (r + 1) * inner;
                const T* p2 = base + (r + 2) * inner;
                const T* p3 = base + (r + 3) * inner;
                int64_t j = 0;
                for (; j < vec; j += W) {
                    const Vec q0 = fold_total_step<T, Mode>(Vec::loadu(p0 + j));
                    const Vec q1 = fold_total_step<T, Mode>(Vec::loadu(p1 + j));
                    const Vec q2 = fold_total_step<T, Mode>(Vec::loadu(p2 + j));
                    const Vec q3 = fold_total_step<T, Mode>(Vec::loadu(p3 + j));
                    (Vec::loadu(acc + j) + ((q0 + q1) + (q2 + q3)))
                        .store(acc + j);
                }
                for (; j < n; ++j) {
                    acc[j] += ((fold_total_one<T, Mode>(p0[j]) +
                                fold_total_one<T, Mode>(p1[j])) +
                               (fold_total_one<T, Mode>(p2[j]) +
                                fold_total_one<T, Mode>(p3[j])));
                }
            }
            for (; r < red; ++r) {
                const T* p = base + r * inner;
                int64_t j = 0;
                for (; j < vec; j += W) {
                    const Vec q = fold_total_step<T, Mode>(Vec::loadu(p + j));
                    (Vec::loadu(acc + j) + q).store(acc + j);
                }
                for (; j < n; ++j) acc[j] += fold_total_one<T, Mode>(p[j]);
            }
            for (int64_t j = 0; j < n; ++j) {
                dst[j] = Mode == TotalMode::Square
                    ? static_cast<T>(std::sqrt(acc[j]))
                    : static_cast<T>(acc[j]);
            }
        }
    }
}

template <typename T, TotalMode Mode>
inline void norm_totals_real_impl(const T* in, T* out, int64_t outer,
                                  int64_t red, int64_t inner) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const int64_t block = std::max<int64_t>(
        4 * W, GRAIN_SIZE / std::max<int64_t>(red, 1));
    const int64_t nblk = (inner + block - 1) / block;
    parallel_for(0, outer * nblk, 1, [&](int64_t begin, int64_t end) {
        norm_totals_real_range<T, Mode>(in, out, red, inner, block, nblk,
                                        begin, end);
    });
}

#if defined(__x86_64__)
// One reduced axis of a packed tensor: the block of rows is widened a step at
// a time and every element folds into single-precision per-column totals,
// which then narrow once into the width the answer is carried in.  A two-norm
// folds the square of each element and takes one square root of the totals; a
// sum folds the element itself.  The walk is a free function so that its
// target carries into the work a parallel task runs.
template <TotalMode Mode>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
__m256 fold_total_step8(const __m256& q) {
    if constexpr (Mode == TotalMode::Square) {
        return _mm256_mul_ps(q, q);
    } else if constexpr (Mode == TotalMode::Abs) {
        return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), q);
    } else {
        return q;
    }
}

template <bool BFloat, TotalMode Mode, bool Root>
__attribute__((target("avx2,fma,f16c"), noinline))
void column_totals_reduced_range(const uint16_t* in, uint16_t* out,
                                 int64_t outer, int64_t red, int64_t inner,
                                 int64_t block, int64_t nblk, int64_t begin,
                                 int64_t end) {
    using tensorplay::cpu::vecreduced::widen8;
    using tensorplay::cpu::vecreduced::widen_bf16_8;
    constexpr int64_t W = 8;
    static thread_local std::vector<float> totals;
    if (static_cast<int64_t>(totals.size()) < block) {
        totals.resize(static_cast<size_t>(block));
    }
    float* acc = totals.data();
    for (int64_t task = begin; task < end; ++task) {
        const int64_t o = task / nblk;
        const int64_t c0 = (task % nblk) * block;
        const int64_t n = std::min(block, inner - c0);
        const uint16_t* base = in + (o * red) * inner + c0;
        uint16_t* dst = out + o * inner + c0;
        std::fill(acc, acc + n, 0.f);
        const int64_t vec = n & ~(W - 1);
        if (n < W) {
            // A run narrower than one step is widened from where it lies: the
            // lanes past the run belong to the next row and are dropped, and
            // only the tensor's own last row needs a padded copy to read.  Four
            // rows' totals stay in flight at once, so no row waits for the one
            // before it to finish its add.
            const bool last_plane = o + 1 == outer;
            // The tensor's own last row is the one read past; every other row
            // is followed by data this walk already owns.
            const int64_t bulk = last_plane ? red - 1 : red;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int64_t rr = 0;
            for (; rr + 4 <= bulk; rr += 4) {
                const __m256 q0 = widen_row8<BFloat>(base + (rr + 0) * inner);
                const __m256 q1 = widen_row8<BFloat>(base + (rr + 1) * inner);
                const __m256 q2 = widen_row8<BFloat>(base + (rr + 2) * inner);
                const __m256 q3 = widen_row8<BFloat>(base + (rr + 3) * inner);
                a0 = _mm256_add_ps(a0, fold_total_step8<Mode>(q0));
                a1 = _mm256_add_ps(a1, fold_total_step8<Mode>(q1));
                a2 = _mm256_add_ps(a2, fold_total_step8<Mode>(q2));
                a3 = _mm256_add_ps(a3, fold_total_step8<Mode>(q3));
            }
            for (; rr < bulk; ++rr) {
                const __m256 q = widen_row8<BFloat>(base + rr * inner);
                a0 = _mm256_add_ps(a0, fold_total_step8<Mode>(q));
            }
            if (last_plane) {
                alignas(16) uint16_t pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                const uint16_t* p = base + (red - 1) * inner;
                for (int64_t k = 0; k < n; ++k) pad[k] = p[k];
                const __m256 q = widen_row8<BFloat>(pad);
                a0 = _mm256_add_ps(a0, fold_total_step8<Mode>(q));
            }
            _mm256_storeu_ps(acc, _mm256_add_ps(_mm256_add_ps(a0, a1),
                                                _mm256_add_ps(a2, a3)));
            for (int64_t j = 0; j < n; ++j) {
                const float v = Root ? std::sqrt(acc[j]) : acc[j];
                dst[j] = BFloat
                    ? tensorplay::cpu::vecreduced::narrow_bf16_one(v)
                    : tensorplay::cpu::vecreduced::narrow_one_hw(v);
            }
            continue;
        }
        // The columns past the last whole step are widened too: the lanes past
        // the run belong to the next row, and only the tensor's own last row
        // needs a padded copy, so the totals take one extra step per row rather
        // than a converted element at a time.
        const bool last_plane = o + 1 == outer;
        __m256 tail = _mm256_setzero_ps();
        int64_t r = 0;
        for (; r + 4 <= red; r += 4) {
            const uint16_t* p0 = base + (r + 0) * inner;
            const uint16_t* p1 = base + (r + 1) * inner;
            const uint16_t* p2 = base + (r + 2) * inner;
            const uint16_t* p3 = base + (r + 3) * inner;
            int64_t j = 0;
            for (; j < vec; j += W) {
                const __m256 q0 =
                    BFloat ? widen_bf16_8(p0 + j) : widen8(p0 + j);
                const __m256 q1 =
                    BFloat ? widen_bf16_8(p1 + j) : widen8(p1 + j);
                const __m256 q2 =
                    BFloat ? widen_bf16_8(p2 + j) : widen8(p2 + j);
                const __m256 q3 =
                    BFloat ? widen_bf16_8(p3 + j) : widen8(p3 + j);
                _mm256_storeu_ps(
                    acc + j,
                    _mm256_add_ps(
                        _mm256_loadu_ps(acc + j),
                        _mm256_add_ps(
                            _mm256_add_ps(fold_total_step8<Mode>(q0),
                                          fold_total_step8<Mode>(q1)),
                            _mm256_add_ps(fold_total_step8<Mode>(q2),
                                          fold_total_step8<Mode>(q3)))));
            }
            if (vec < n) {
                const __m256 q0 = widen_row8<BFloat>(p0 + vec);
                const __m256 q1 = widen_row8<BFloat>(p1 + vec);
                const __m256 q2 = widen_row8<BFloat>(p2 + vec);
                __m256 q3;
                if (last_plane && r + 3 == red - 1) {
                    alignas(16) uint16_t pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int64_t k = vec; k < n; ++k) pad[k - vec] = p3[k];
                    q3 = widen_row8<BFloat>(pad);
                } else {
                    q3 = widen_row8<BFloat>(p3 + vec);
                }
                tail = _mm256_add_ps(
                    tail,
                    _mm256_add_ps(
                        _mm256_add_ps(fold_total_step8<Mode>(q0),
                                      fold_total_step8<Mode>(q1)),
                        _mm256_add_ps(fold_total_step8<Mode>(q2),
                                      fold_total_step8<Mode>(q3))));
            }
        }
        for (; r < red; ++r) {
            const uint16_t* p = base + r * inner;
            int64_t j = 0;
            for (; j < vec; j += W) {
                const __m256 q =
                    BFloat ? widen_bf16_8(p + j) : widen8(p + j);
                _mm256_storeu_ps(
                    acc + j,
                    _mm256_add_ps(_mm256_loadu_ps(acc + j),
                                  fold_total_step8<Mode>(q)));
            }
            if (vec < n) {
                __m256 q;
                if (last_plane && r == red - 1) {
                    alignas(16) uint16_t pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int64_t k = vec; k < n; ++k) pad[k - vec] = p[k];
                    q = widen_row8<BFloat>(pad);
                } else {
                    q = widen_row8<BFloat>(p + vec);
                }
                tail = _mm256_add_ps(tail, fold_total_step8<Mode>(q));
            }
        }
        if (vec < n) {
            alignas(32) float lanes[W];
            _mm256_storeu_ps(lanes, tail);
            for (int64_t j = vec; j < n; ++j) acc[j] = lanes[j - vec];
        }
        for (int64_t j = 0; j < n; ++j) {
            const float v = Root ? std::sqrt(acc[j]) : acc[j];
            dst[j] = BFloat
                ? tensorplay::cpu::vecreduced::narrow_bf16_one(v)
                : tensorplay::cpu::vecreduced::narrow_one_hw(v);
        }
    }
}

template <bool BFloat, TotalMode Mode, bool Root>
inline void column_totals_reduced_impl(const uint16_t* in, uint16_t* out,
                                       int64_t outer, int64_t red,
                                       int64_t inner) {
    const int64_t block = std::max<int64_t>(
        4 * 8, GRAIN_SIZE / std::max<int64_t>(red, 1));
    const int64_t nblk = (inner + block - 1) / block;
    tensorplay::parallel::parallel_for(
        0, outer * nblk, 1, [&](int64_t begin, int64_t end) {
            column_totals_reduced_range<BFloat, Mode, Root>(
                in, out, outer, red, inner, block, nblk, begin, end);
        });
}

// A single reduced axis of a packed tensor that the leading and trailing
// walks declined: rows widen into per-column totals over the contiguous
// direction, so the walk never steps across the reduced axis.
inline std::optional<Tensor> try_sum_dim_reduced_half(
        const Tensor& input, const std::vector<int64_t>& out_shape,
        int64_t dim) {
    if (!input.is_contiguous() || !reduced_sum_takes(input.dtype())) {
        return std::nullopt;
    }
    const int64_t ndim = input.dim();
    if (ndim < 2 || dim < 0 || dim >= ndim - 1) return std::nullopt;
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= input.size(i);
    const int64_t red = input.size(dim);
    if (outer <= 0 || red <= 0) return std::nullopt;
    const int64_t inner = input.numel() / (outer * red);
    if (inner <= 0) return std::nullopt;
    // The leading axis of a whole number of wide steps already streams through
    // the per-worker totals; a ragged or short run is what this walk answers.
    if (dim == 0 && inner >= 32 && inner % 32 == 0) return std::nullopt;
    if (!tensorplay::cpu::vecreduced::f16c_available() &&
        input.dtype() == DType::Float16) {
        return std::nullopt;
    }
    const bool half_width = input.dtype() == DType::Float16;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    uint16_t* outp = half_width
        ? reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(out.data_ptr<tensorplay::BFloat16>());
    if (half_width) {
        column_totals_reduced_impl<false, TotalMode::Sum, false>(in, outp, outer, red,
                                                        inner);
    } else {
        column_totals_reduced_impl<true, TotalMode::Sum, false>(in, outp, outer, red,
                                                       inner);
    }
    return out;
}
#endif

// A single reduced axis of a real tensor at the two-norm answers the same
// value the per-element fold does, but walks the contiguous direction of a
// row instead of stepping across the reduced axis.
inline std::optional<Tensor> try_norm_totals_dim(
        const Tensor& self, int64_t dim, bool keepdim,
        const std::vector<int64_t>& dims, double p) {
    const DType dt = self.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16) {
        return std::nullopt;
    }
    if (!self.is_contiguous() || self.numel() == 0 || self.dim() == 0) {
        return std::nullopt;
    }
    const int64_t nd = self.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    if (d < 0 || d >= nd) return std::nullopt;
    const bool reduced_fast =
        dt == DType::BFloat16 ||
        (dt == DType::Float16 &&
         tensorplay::cpu::vecreduced::f16c_available());
    if (!reduced_fast && (dt == DType::Float16 || dt == DType::BFloat16)) {
        return std::nullopt;
    }
    // A packed run over the last axis has one column per row: the widened step
    // would read a whole lane-vector for every element it keeps, and the row
    // walk answers that shape from four rows of its own.
    if ((dt == DType::Float16 || dt == DType::BFloat16) && d == nd - 1) {
        return std::nullopt;
    }
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < d; ++i) outer *= self.size(i);
    for (int64_t i = d + 1; i < nd; ++i) inner *= self.size(i);
    const int64_t red = self.size(d);
    if (red == 0 || inner == 0 || outer == 0) return std::nullopt;
    Tensor out = Tensor::empty(compute_reduction_shape(self, dims, keepdim),
                               dt, self.device());
    const bool squared = p == 2.0;
    if (dt == DType::Float32) {
        if (squared) {
            norm_totals_real_impl<float, TotalMode::Square>(
                self.data_ptr<float>(), out.data_ptr<float>(), outer, red,
                inner);
        } else {
            norm_totals_real_impl<float, TotalMode::Abs>(
                self.data_ptr<float>(), out.data_ptr<float>(), outer, red,
                inner);
        }
        return out;
    }
    if (dt == DType::Float64) {
        if (squared) {
            norm_totals_real_impl<double, TotalMode::Square>(
                self.data_ptr<double>(), out.data_ptr<double>(), outer, red,
                inner);
        } else {
            norm_totals_real_impl<double, TotalMode::Abs>(
                self.data_ptr<double>(), out.data_ptr<double>(), outer, red,
                inner);
        }
        return out;
    }
#if defined(__x86_64__)
    if (dt == DType::Float16) {
        const uint16_t* in = reinterpret_cast<const uint16_t*>(
            self.data_ptr<tensorplay::Half>());
        uint16_t* op = reinterpret_cast<uint16_t*>(
            out.data_ptr<tensorplay::Half>());
        if (squared) {
            column_totals_reduced_impl<false, TotalMode::Square, true>(
                in, op, outer, red, inner);
        } else {
            column_totals_reduced_impl<false, TotalMode::Abs, false>(
                in, op, outer, red, inner);
        }
        return out;
    }
    const uint16_t* in = reinterpret_cast<const uint16_t*>(
        self.data_ptr<tensorplay::BFloat16>());
    uint16_t* op = reinterpret_cast<uint16_t*>(
        out.data_ptr<tensorplay::BFloat16>());
    if (squared) {
        column_totals_reduced_impl<true, TotalMode::Square, true>(
            in, op, outer, red, inner);
    } else {
        column_totals_reduced_impl<true, TotalMode::Abs, false>(
            in, op, outer, red, inner);
    }
    return out;
#else
    return std::nullopt;
#endif
}

Tensor norm_dim_kernel_impl(const Tensor& self,
                            const std::vector<int64_t>& dims,
                            double p, bool keepdim) {
    if (dims.empty()) return norm_kernel_impl(self, p);

    // Reducing every dimension of a packed tensor is one pass over the whole
    // buffer, which the full-reduction form already answers as a single
    // stream; walking it one reduced dimension at a time reaches a fraction
    // of that rate.
    if (self.numel() > 0 && self.is_contiguous() && self.dim() > 1 &&
        static_cast<int64_t>(dims.size()) == self.dim()) {
        bool covers_all = true;
        std::vector<bool> seen(static_cast<size_t>(self.dim()), false);
        for (int64_t d : dims) {
            const int64_t v = d < 0 ? d + self.dim() : d;
            if (v < 0 || v >= self.dim() || seen[static_cast<size_t>(v)]) {
                covers_all = false;
                break;
            }
            seen[static_cast<size_t>(v)] = true;
        }
        if (covers_all) {
            Tensor result = norm_kernel_impl(self, p);
            if (keepdim) {
                result = result.reshape(
                    std::vector<int64_t>(static_cast<size_t>(self.dim()), 1));
            }
            return result;
        }
    }

    // An L2 norm that folds a trailing block of a contiguous real tensor is
    // one independent stream per row: each row walks the wide sum-of-squares
    // pass and takes a single square root, while a pass per reduced dimension
    // reaches only a fraction of that rate.
    const int64_t nd = self.dim();
    const bool reduced_fast =
        self.dtype() == DType::BFloat16 ||
        (self.dtype() == DType::Float16 &&
         tensorplay::cpu::vecreduced::f16c_available());
    if ((p == 1.0 || p == 2.0) && self.is_contiguous() && self.numel() > 0 &&
        nd >= 1 && !dims.empty() &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64 ||
         reduced_fast)) {
        int64_t first = nd;
        bool trailing = true;
        for (int64_t d : dims) {
            const int64_t v = d < 0 ? d + nd : d;
            if (v < 0 || v >= nd) {
                trailing = false;
                break;
            }
            first = std::min(first, v);
        }
        if (trailing) {
            // The reduced axes must be exactly the trailing block, each one
            // named once; a gap would fold more elements than were asked for.
            if (static_cast<int64_t>(dims.size()) != nd - first) {
                trailing = false;
            } else {
                std::vector<bool> seen(static_cast<size_t>(nd), false);
                for (int64_t d : dims) {
                    const int64_t v = d < 0 ? d + nd : d;
                    if (v < first || seen[static_cast<size_t>(v)]) {
                        trailing = false;
                        break;
                    }
                    seen[static_cast<size_t>(v)] = true;
                }
            }
        }
        if (trailing) {
            int64_t row = 1;
            for (int64_t d = first; d < nd; ++d) row *= self.size(d);
            const int64_t outer = row > 0 ? self.numel() / row : 0;
            const int64_t row_grain =
                std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(row, 1));
            Tensor out = Tensor::empty(compute_reduction_shape(self, dims, keepdim),
                                       self.dtype(), self.device());
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            // A row narrower than a step is widened as a block of rows and fold
            // there, which turns one call per row into one streaming walk.
            if (self.dtype() == DType::Float16) {
                const uint16_t* in = reinterpret_cast<const uint16_t*>(
                    self.data_ptr<tensorplay::Half>());
                uint16_t* op = reinterpret_cast<uint16_t*>(
                    out.data_ptr<tensorplay::Half>());
                if (row < 16) {
                    parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                        if (p == 2.0) {
                            fold_thin_rows<false, ThinRowFold::TwoNorm>(
                                in + b * row, e - b, row, op + b);
                        } else {
                            fold_thin_rows<false, ThinRowFold::AbsSum>(
                                in + b * row, e - b, row, op + b);
                        }
                    });
                    return out;
                }
                parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                    for (int64_t r = b; r < e; ++r) {
                        const float value = p == 2.0
                            ? std::sqrt(sumsq_f16_chunk(in + r * row, 0, row))
                            : sum_abs_reduced_chunk<false>(in + r * row, 0, row);
                        op[r] = tensorplay::cpu::vecreduced::narrow_one_hw(value);
                    }
                });
                return out;
            }
            if (self.dtype() == DType::BFloat16) {
                const uint16_t* in = reinterpret_cast<const uint16_t*>(
                    self.data_ptr<tensorplay::BFloat16>());
                uint16_t* op = reinterpret_cast<uint16_t*>(
                    out.data_ptr<tensorplay::BFloat16>());
                if (row < 16) {
                    parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                        if (p == 2.0) {
                            fold_thin_rows<true, ThinRowFold::TwoNorm>(
                                in + b * row, e - b, row, op + b);
                        } else {
                            fold_thin_rows<true, ThinRowFold::AbsSum>(
                                in + b * row, e - b, row, op + b);
                        }
                    });
                    return out;
                }
                parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                    for (int64_t r = b; r < e; ++r) {
                        const float value = p == 2.0
                            ? std::sqrt(sumsq_bf16_chunk(in + r * row, 0, row))
                            : sum_abs_reduced_chunk<true>(in + r * row, 0, row);
                        op[r] = tensorplay::cpu::vecreduced::narrow_bf16_one(value);
                    }
                });
                return out;
            }
#endif
            if (self.dtype() == DType::Float32) {
                const float* in = self.data_ptr<float>();
                float* op = out.data_ptr<float>();
                parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                    for (int64_t r = b; r < e; ++r) {
                        op[r] = p == 2.0
                            ? std::sqrt(vec::sum_of_squares(in + r * row, 0, row))
                            : vec::sum_of_abs(in + r * row, 0, row);
                    }
                });
            } else {
                const double* in = self.data_ptr<double>();
                double* op = out.data_ptr<double>();
                parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                    for (int64_t r = b; r < e; ++r) {
                        op[r] = p == 2.0
                            ? std::sqrt(vec::sum_of_squares(in + r * row, 0, row))
                            : vec::sum_of_abs(in + r * row, 0, row);
                    }
                });
            }
            return out;
        }
    }

    // One named axis that is not the trailing block walks its totals over the
    // contiguous direction of a row instead of stepping across the reduced
    // one, which is the order the per-element fold pays for.
    if ((p == 1.0 || p == 2.0) && dims.size() == 1) {
        if (std::optional<Tensor> totals =
                try_norm_totals_dim(self, dims[0], keepdim, dims, p)) {
            return std::move(*totals);
        }
    }

    #define TP_NORM_DIM_CASE(ctype, dtype_name) \
        case DType::dtype_name: \
            return norm_dim_typed<ctype>(self, dims, p, keepdim)
    switch (self.dtype()) {
        TP_NORM_DIM_CASE(float, Float32);
        TP_NORM_DIM_CASE(double, Float64);
        TP_NORM_DIM_CASE(Half, Float16);
        TP_NORM_DIM_CASE(BFloat16, BFloat16);
        TP_NORM_DIM_CASE(complex<Half>, ComplexHalf);
        TP_NORM_DIM_CASE(complex<float>, ComplexFloat);
        TP_NORM_DIM_CASE(complex<double>, ComplexDouble);
        TP_NORM_DIM_CASE(complex<BFloat16>, BComplex32);
        default:
            TP_THROW(NotImplementedError, "norm: unsupported dtype on CPU");
    }
    #undef TP_NORM_DIM_CASE
}

// --- AVX-512 runtime-dispatched full-reduction sum (real dtypes) -----------
#if defined(__x86_64__)
namespace {

inline bool reduce_avx512_available() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

inline bool byte_reduce_avx512_available() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512bw") != 0;
    return ok;
}

__attribute__((target("avx512f")))
float sum_f32_chunk_avx512(const float* x, int64_t b, int64_t e) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int64_t i = b;
    for (; i + 64 <= e; i += 64) {
        a0 = _mm512_add_ps(a0, _mm512_loadu_ps(x + i));
        a1 = _mm512_add_ps(a1, _mm512_loadu_ps(x + i + 16));
        a2 = _mm512_add_ps(a2, _mm512_loadu_ps(x + i + 32));
        a3 = _mm512_add_ps(a3, _mm512_loadu_ps(x + i + 48));
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    for (; i + 16 <= e; i += 16)
        acc = _mm512_add_ps(acc, _mm512_loadu_ps(x + i));
    alignas(64) float buf[16];
    _mm512_storeu_ps(buf, acc);
    float s = ((buf[0] + buf[1]) + (buf[2] + buf[3])) +
              ((buf[4] + buf[5]) + (buf[6] + buf[7])) +
              ((buf[8] + buf[9]) + (buf[10] + buf[11])) +
              ((buf[12] + buf[13]) + (buf[14] + buf[15]));
    for (; i < e; ++i) s += x[i];
    return s;
}

__attribute__((target("avx512f")))
double sum_f64_chunk_avx512(const double* x, int64_t b, int64_t e) {
    __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
    int64_t i = b;
    for (; i + 16 <= e; i += 16) {
        a0 = _mm512_add_pd(a0, _mm512_loadu_pd(x + i));
        a1 = _mm512_add_pd(a1, _mm512_loadu_pd(x + i + 8));
    }
    __m512d acc = _mm512_add_pd(a0, a1);
    alignas(64) double buf[8];
    _mm512_storeu_pd(buf, acc);
    double s = (buf[0] + buf[1]) + (buf[2] + buf[3]) +
               (buf[4] + buf[5]) + (buf[6] + buf[7]);
    for (; i < e; ++i) s += x[i];
    return s;
}

__attribute__((target("avx512f")))
float product_f32_chunk_avx512(const float* x, int64_t b, int64_t e) {
    __m512 a0 = _mm512_set1_ps(1.0f), a1 = a0;
    __m512 a2 = a0, a3 = a0;
    int64_t i = b;
    for (; i + 64 <= e; i += 64) {
        a0 = _mm512_mul_ps(a0, _mm512_loadu_ps(x + i));
        a1 = _mm512_mul_ps(a1, _mm512_loadu_ps(x + i + 16));
        a2 = _mm512_mul_ps(a2, _mm512_loadu_ps(x + i + 32));
        a3 = _mm512_mul_ps(a3, _mm512_loadu_ps(x + i + 48));
    }
    __m512 acc = _mm512_mul_ps(_mm512_mul_ps(a0, a1),
                               _mm512_mul_ps(a2, a3));
    for (; i + 16 <= e; i += 16)
        acc = _mm512_mul_ps(acc, _mm512_loadu_ps(x + i));
    alignas(64) float lanes[16];
    _mm512_storeu_ps(lanes, acc);
    float product = lanes[0];
    for (int j = 1; j < 16; ++j) product *= lanes[j];
    for (; i < e; ++i) product *= x[i];
    return product;
}

__attribute__((target("avx512f")))
double product_f64_chunk_avx512(const double* x, int64_t b, int64_t e) {
    __m512d a0 = _mm512_set1_pd(1.0), a1 = a0;
    int64_t i = b;
    for (; i + 16 <= e; i += 16) {
        a0 = _mm512_mul_pd(a0, _mm512_loadu_pd(x + i));
        a1 = _mm512_mul_pd(a1, _mm512_loadu_pd(x + i + 8));
    }
    __m512d acc = _mm512_mul_pd(a0, a1);
    for (; i + 8 <= e; i += 8)
        acc = _mm512_mul_pd(acc, _mm512_loadu_pd(x + i));
    alignas(64) double lanes[8];
    _mm512_storeu_pd(lanes, acc);
    double product = lanes[0];
    for (int j = 1; j < 8; ++j) product *= lanes[j];
    for (; i < e; ++i) product *= x[i];
    return product;
}

__attribute__((target("avx2,fma")
))
float product_f32_chunk_avx2(const float* x, int64_t b, int64_t e) {
    __m256 a0 = _mm256_set1_ps(1.0f), a1 = a0;
    int64_t i = b;
    for (; i + 16 <= e; i += 16) {
        a0 = _mm256_mul_ps(a0, _mm256_loadu_ps(x + i));
        a1 = _mm256_mul_ps(a1, _mm256_loadu_ps(x + i + 8));
    }
    __m256 acc = _mm256_mul_ps(a0, a1);
    alignas(32) float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    float product = lanes[0];
    for (int j = 1; j < 8; ++j) product *= lanes[j];
    for (; i < e; ++i) product *= x[i];
    return product;
}

__attribute__((target("avx2,fma"
)))
double product_f64_chunk_avx2(const double* x, int64_t b, int64_t e) {
    __m256d a0 = _mm256_set1_pd(1.0), a1 = a0;
    int64_t i = b;
    for (; i + 8 <= e; i += 8) {
        a0 = _mm256_mul_pd(a0, _mm256_loadu_pd(x + i));
        a1 = _mm256_mul_pd(a1, _mm256_loadu_pd(x + i + 4));
    }
    __m256d acc = _mm256_mul_pd(a0, a1);
    alignas(32) double lanes[4];
    _mm256_storeu_pd(lanes, acc);
    double product = lanes[0];
    for (int j = 1; j < 4; ++j) product *= lanes[j];
    for (; i < e; ++i) product *= x[i];
    return product;
}

inline bool reduce_avx2_available() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
}

// The AVX2 tier of the product reduction: same shape as the AVX512 one,
// for hosts whose lanes stop at 256 bits.  Small sizes answer serially —
// the partials plumbing costs more than the walk itself.
static bool try_product_real_avx2(const void* xv, int64_t n, DType dt,
                                  double* out) {
    if (!reduce_avx2_available() || n < 512) return false;
    constexpr int64_t kGrain = 32768;
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        float product;
        if (n < kGrain) {
            product = product_f32_chunk_avx2(x, 0, n);
        } else {
            const int64_t nslots = (n + kGrain - 1) / kGrain;
            std::vector<float> partials(nslots, 1.0f);
            tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
                partials[b / kGrain] = product_f32_chunk_avx2(x, b, e);
            });
            product = 1.0f;
            for (int64_t k = 0; k < nslots; ++k) product *= partials[k];
        }
        *out = static_cast<double>(product);
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        double product;
        if (n < kGrain) {
            product = product_f64_chunk_avx2(x, 0, n);
        } else {
            const int64_t nslots = (n + kGrain - 1) / kGrain;
            std::vector<double> partials(nslots, 1.0);
            tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
                partials[b / kGrain] = product_f64_chunk_avx2(x, b, e);
            });
            product = 1.0;
            for (int64_t k = 0; k < nslots; ++k) product *= partials[k];
        }
        *out = product;
        return true;
    }
    return false;
}

static bool try_product_real_avx512(const void* xv, int64_t n, DType dt,
                                    double* out) {
    if (!reduce_avx512_available() || n < 4096) return false;    constexpr int64_t kGrain = 32768;
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        const int64_t nslots = (n + kGrain - 1) / kGrain;
        std::vector<float> partials(nslots, 1.0f);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            partials[b / kGrain] = product_f32_chunk_avx512(x, b, e);
        });
        float product = 1.0f;
        for (int64_t k = 0; k < nslots; ++k) product *= partials[k];
        *out = static_cast<double>(product);
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        const int64_t nslots = (n + kGrain - 1) / kGrain;
        std::vector<double> partials(nslots, 1.0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            partials[b / kGrain] = product_f64_chunk_avx512(x, b, e);
        });
        double product = 1.0;
        for (int64_t k = 0; k < nslots; ++k) product *= partials[k];
        *out = product;
        return true;
    }
    return false;
}

static bool try_product_lastdim_real_avx512(
    const void* xv, void* outv, int64_t outer, int64_t d_size, DType dt) {
    if (!reduce_avx512_available() || d_size < 64 || outer <= 0) return false;
    const int64_t row_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (dt == DType::Float32) {
        const float* input = static_cast<const float*>(xv);
        float* output = static_cast<float*>(outv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    output[row] = product_f32_chunk_avx512(
                        input + row * d_size, 0, d_size);
                }
            });
        return true;
    }
    if (dt == DType::Float64) {
        const double* input = static_cast<const double*>(xv);
        double* output = static_cast<double*>(outv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    output[row] = product_f64_chunk_avx512(
                        input + row * d_size, 0, d_size);
                }
            });
        return true;
    }
    return false;
}

template <bool WantAll>
__attribute__((target("avx512f,avx512bw")))
static bool byte_reduce_avx512_range(const uint8_t* data, int64_t begin,
                                     int64_t end) {
    int64_t i = begin;
    const __m512i zero = _mm512_setzero_si512();
    constexpr uint64_t kAllBytes = 0xffffffffffffffffULL;
    for (; i + 64 <= end; i += 64) {
        const __m512i values = _mm512_loadu_si512(data + i);
        const uint64_t zero_mask = static_cast<uint64_t>(
            _mm512_cmpeq_epi8_mask(values, zero));
        if constexpr (WantAll) {
            if (zero_mask != 0) return false;
        } else if (zero_mask != kAllBytes) {
            return true;
        }
    }
    for (; i < end; ++i) {
        if constexpr (WantAll) {
            if (data[i] == 0) return false;
        } else if (data[i] != 0) {
            return true;
        }
    }
    return WantAll;
}

template <bool WantAll>
static bool byte_reduce_avx512(const uint8_t* data, int64_t n) {
    if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        return byte_reduce_avx512_range<WantAll>(data, 0, n);
    }

    const int num_threads = get_num_threads();
    std::vector<unsigned char> partials(
        static_cast<size_t>(num_threads), WantAll ? 1 : 0);
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        partials[get_thread_num()] = static_cast<unsigned char>(
            byte_reduce_avx512_range<WantAll>(data, begin, end));
    });
    for (unsigned char partial : partials) {
        if constexpr (WantAll) {
            if (partial == 0) return false;
        } else if (partial != 0) {
            return true;
        }
    }
    return WantAll;
}

template <bool WantAll>
__attribute__((target("avx512f")))
static bool logical_reduce_f32_range(
    const float* data, int64_t begin, int64_t end) {
    int64_t i = begin;
    constexpr uint16_t kAllLanes = 0xffff;
    const __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= end; i += 16) {
        const uint16_t nonzero = static_cast<uint16_t>(
            _mm512_cmp_ps_mask(_mm512_loadu_ps(data + i), zero, _CMP_NEQ_UQ));
        if constexpr (WantAll) {
            if (nonzero != kAllLanes) return false;
        } else if (nonzero != 0) {
            return true;
        }
    }
    for (; i < end; ++i) {
        if constexpr (WantAll) {
            if (data[i] == 0.0f) return false;
        } else if (data[i] != 0.0f) {
            return true;
        }
    }
    return WantAll;
}

template <bool WantAll>
__attribute__((target("avx512f")))
static bool logical_reduce_f64_range(
    const double* data, int64_t begin, int64_t end) {
    int64_t i = begin;
    constexpr uint8_t kAllLanes = 0xff;
    const __m512d zero = _mm512_setzero_pd();
    for (; i + 8 <= end; i += 8) {
        const uint8_t nonzero = static_cast<uint8_t>(
            _mm512_cmp_pd_mask(_mm512_loadu_pd(data + i), zero, _CMP_NEQ_UQ));
        if constexpr (WantAll) {
            if (nonzero != kAllLanes) return false;
        } else if (nonzero != 0) {
            return true;
        }
    }
    for (; i < end; ++i) {
        if constexpr (WantAll) {
            if (data[i] == 0.0) return false;
        } else if (data[i] != 0.0) {
            return true;
        }
    }
    return WantAll;
}

template <bool WantAll>
static bool logical_reduce_full_avx512(const void* data, int64_t n, DType dt) {
    if (!reduce_avx512_available()) return false;
    if (dt == DType::Float32) {
        const float* values = static_cast<const float*>(data);
        if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
            return logical_reduce_f32_range<WantAll>(values, 0, n);
        }
        const int num_threads = get_num_threads();
        std::vector<unsigned char> partials(
            static_cast<size_t>(num_threads), WantAll ? 1 : 0);
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            partials[get_thread_num()] = static_cast<unsigned char>(
                logical_reduce_f32_range<WantAll>(values, begin, end));
        });
        for (unsigned char partial : partials) {
            if constexpr (WantAll) {
                if (partial == 0) return false;
            } else if (partial != 0) {
                return true;
            }
        }
        return WantAll;
    }
    if (dt == DType::Float64) {
        const double* values = static_cast<const double*>(data);
        if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
            return logical_reduce_f64_range<WantAll>(values, 0, n);
        }
        const int num_threads = get_num_threads();
        std::vector<unsigned char> partials(
            static_cast<size_t>(num_threads), WantAll ? 1 : 0);
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            partials[get_thread_num()] = static_cast<unsigned char>(
                logical_reduce_f64_range<WantAll>(values, begin, end));
        });
        for (unsigned char partial : partials) {
            if constexpr (WantAll) {
                if (partial == 0) return false;
            } else if (partial != 0) {
                return true;
            }
        }
        return WantAll;
    }
    return false;
}

template <bool WantAll>
__attribute__((target("avx512f")))
static void logical_reduce_f32_dim_row(
    const float* input, uint8_t* output, int64_t d_size, int64_t inner,
    int64_t col_begin, int64_t col_end) {
    if (inner == 1 && col_begin == 0 && col_end != 0) {
        output[0] = static_cast<uint8_t>(
            logical_reduce_f32_range<WantAll>(input, 0, d_size));
        return;
    }
    constexpr uint16_t kAllLanes = 0xffff;
    const __m512 zero = _mm512_setzero_ps();
    int64_t col = col_begin;
    for (; col + 64 <= col_end; col += 64) {
        uint16_t result0 = WantAll ? kAllLanes : 0;
        uint16_t result1 = WantAll ? kAllLanes : 0;
        uint16_t result2 = WantAll ? kAllLanes : 0;
        uint16_t result3 = WantAll ? kAllLanes : 0;
        for (int64_t row = 0; row < d_size; ++row) {
            const float* row_input = input + row * inner + col;
            const uint16_t nonzero0 = static_cast<uint16_t>(
                _mm512_cmp_ps_mask(_mm512_loadu_ps(row_input), zero, _CMP_NEQ_UQ));
            const uint16_t nonzero1 = static_cast<uint16_t>(
                _mm512_cmp_ps_mask(_mm512_loadu_ps(row_input + 16), zero, _CMP_NEQ_UQ));
            const uint16_t nonzero2 = static_cast<uint16_t>(
                _mm512_cmp_ps_mask(_mm512_loadu_ps(row_input + 32), zero, _CMP_NEQ_UQ));
            const uint16_t nonzero3 = static_cast<uint16_t>(
                _mm512_cmp_ps_mask(_mm512_loadu_ps(row_input + 48), zero, _CMP_NEQ_UQ));
            if constexpr (WantAll) {
                result0 &= nonzero0;
                result1 &= nonzero1;
                result2 &= nonzero2;
                result3 &= nonzero3;
                if ((result0 | result1 | result2 | result3) == 0) break;
            } else {
                result0 |= nonzero0;
                result1 |= nonzero1;
                result2 |= nonzero2;
                result3 |= nonzero3;
                if ((result0 & result1 & result2 & result3) == kAllLanes) break;
            }
        }
        alignas(64) uint8_t reduced[64];
        for (int lane = 0; lane < 16; ++lane) {
            reduced[lane] = static_cast<uint8_t>((result0 >> lane) & 1);
            reduced[16 + lane] = static_cast<uint8_t>((result1 >> lane) & 1);
            reduced[32 + lane] = static_cast<uint8_t>((result2 >> lane) & 1);
            reduced[48 + lane] = static_cast<uint8_t>((result3 >> lane) & 1);
        }
        std::memcpy(output + col, reduced, sizeof(reduced));
    }
    for (; col + 16 <= col_end; col += 16) {
        uint16_t result = WantAll ? kAllLanes : 0;
        for (int64_t row = 0; row < d_size; ++row) {
            const uint16_t nonzero = static_cast<uint16_t>(
                _mm512_cmp_ps_mask(
                    _mm512_loadu_ps(input + row * inner + col),
                    zero, _CMP_NEQ_UQ));
            if constexpr (WantAll) {
                result &= nonzero;
                if (result == 0) break;
            } else {
                result |= nonzero;
                if (result == kAllLanes) break;
            }
        }
        uint8_t reduced[16];
        for (int lane = 0; lane < 16; ++lane) {
            reduced[lane] = static_cast<uint8_t>((result >> lane) & 1);
        }
        std::memcpy(output + col, reduced, sizeof(reduced));
    }
    for (; col < col_end; ++col) {
        bool value = WantAll;
        for (int64_t row = 0; row < d_size; ++row) {
            const bool nonzero = input[row * inner + col] != 0.0f;
            if constexpr (WantAll) {
                value = value && nonzero;
                if (!value) break;
            } else {
                value = value || nonzero;
                if (value) break;
            }
        }
        output[col] = static_cast<uint8_t>(value);
    }
}

template <bool WantAll>
__attribute__((target("avx512f")))
static void logical_reduce_f64_dim_row(
    const double* input, uint8_t* output, int64_t d_size, int64_t inner,
    int64_t col_begin, int64_t col_end) {
    if (inner == 1 && col_begin == 0 && col_end != 0) {
        output[0] = static_cast<uint8_t>(
            logical_reduce_f64_range<WantAll>(input, 0, d_size));
        return;
    }
    constexpr uint8_t kAllLanes = 0xff;
    const __m512d zero = _mm512_setzero_pd();
    int64_t col = col_begin;
    for (; col + 32 <= col_end; col += 32) {
        uint8_t result0 = WantAll ? kAllLanes : 0;
        uint8_t result1 = WantAll ? kAllLanes : 0;
        uint8_t result2 = WantAll ? kAllLanes : 0;
        uint8_t result3 = WantAll ? kAllLanes : 0;
        for (int64_t row = 0; row < d_size; ++row) {
            const double* row_input = input + row * inner + col;
            const uint8_t nonzero0 = static_cast<uint8_t>(
                _mm512_cmp_pd_mask(_mm512_loadu_pd(row_input), zero, _CMP_NEQ_UQ));
            const uint8_t nonzero1 = static_cast<uint8_t>(
                _mm512_cmp_pd_mask(_mm512_loadu_pd(row_input + 8), zero, _CMP_NEQ_UQ));
            const uint8_t nonzero2 = static_cast<uint8_t>(
                _mm512_cmp_pd_mask(_mm512_loadu_pd(row_input + 16), zero, _CMP_NEQ_UQ));
            const uint8_t nonzero3 = static_cast<uint8_t>(
                _mm512_cmp_pd_mask(_mm512_loadu_pd(row_input + 24), zero, _CMP_NEQ_UQ));
            if constexpr (WantAll) {
                result0 &= nonzero0;
                result1 &= nonzero1;
                result2 &= nonzero2;
                result3 &= nonzero3;
                if ((result0 | result1 | result2 | result3) == 0) break;
            } else {
                result0 |= nonzero0;
                result1 |= nonzero1;
                result2 |= nonzero2;
                result3 |= nonzero3;
                if ((result0 & result1 & result2 & result3) == kAllLanes) break;
            }
        }
        alignas(64) uint8_t reduced[32];
        for (int lane = 0; lane < 8; ++lane) {
            reduced[lane] = static_cast<uint8_t>((result0 >> lane) & 1);
            reduced[8 + lane] = static_cast<uint8_t>((result1 >> lane) & 1);
            reduced[16 + lane] = static_cast<uint8_t>((result2 >> lane) & 1);
            reduced[24 + lane] = static_cast<uint8_t>((result3 >> lane) & 1);
        }
        std::memcpy(output + col, reduced, sizeof(reduced));
    }
    for (; col + 8 <= col_end; col += 8) {
        uint8_t result = WantAll ? kAllLanes : 0;
        for (int64_t row = 0; row < d_size; ++row) {
            const uint8_t nonzero = static_cast<uint8_t>(
                _mm512_cmp_pd_mask(
                    _mm512_loadu_pd(input + row * inner + col),
                    zero, _CMP_NEQ_UQ));
            if constexpr (WantAll) {
                result &= nonzero;
                if (result == 0) break;
            } else {
                result |= nonzero;
                if (result == kAllLanes) break;
            }
        }
        uint8_t reduced[8];
        for (int lane = 0; lane < 8; ++lane) {
            reduced[lane] = static_cast<uint8_t>((result >> lane) & 1);
        }
        std::memcpy(output + col, reduced, sizeof(reduced));
    }
    for (; col < col_end; ++col) {
        bool value = WantAll;
        for (int64_t row = 0; row < d_size; ++row) {
            const bool nonzero = input[row * inner + col] != 0.0;
            if constexpr (WantAll) {
                value = value && nonzero;
                if (!value) break;
            } else {
                value = value || nonzero;
                if (value) break;
            }
        }
        output[col] = static_cast<uint8_t>(value);
    }
}

// A reduction along the leading axis visits every row once, so the plane
// is read along its rows and one byte per output column carries the answer
// accumulated so far.  Reading a column at a time instead would turn the
// walk into one strided access per element.  The rows are split across
// threads and their partial answers folded together afterwards.  The
// accumulator is inspected every few rows so a plane whose answer cannot
// change any more stops early.
template <typename ctype, bool WantAll>
static void logical_reduce_dim0_rows(
    const ctype* input, bool* output, int64_t d_size, int64_t inner) {
    const uint8_t init = WantAll ? 1 : 0;
    auto settled = [&](const uint8_t* lanes) {
        for (int64_t col = 0; col < inner; ++col) {
            if constexpr (WantAll) {
                if (lanes[col] != 0) return false;
            } else {
                if (lanes[col] == 0) return false;
            }
        }
        return true;
    };
    auto scan_rows = [&](int64_t row_begin, int64_t row_end,
                         std::vector<uint8_t>& acc) {
        acc.assign(static_cast<size_t>(inner), init);
        uint8_t* lanes = acc.data();
        for (int64_t row = row_begin; row < row_end; ++row) {
            const ctype* p = input + row * inner;
            for (int64_t col = 0; col < inner; ++col) {
                const uint8_t hit = static_cast<bool>(p[col]) ? 1 : 0;
                if constexpr (WantAll) {
                    lanes[col] &= hit;
                } else {
                    lanes[col] |= hit;
                }
            }
            if (((row - row_begin) & 7) == 7 && settled(lanes)) break;
        }
    };
    const int num_threads =
        (d_size * inner < GRAIN_SIZE || get_num_threads() == 1 ||
         in_parallel_region())
        ? 1
        : get_num_threads();
    std::vector<std::vector<uint8_t>> partials(static_cast<size_t>(num_threads));
    if (num_threads == 1) {
        scan_rows(0, d_size, partials[0]);
    } else {
        const int64_t rows_per = (d_size + num_threads - 1) / num_threads;
        parallel_for(0, num_threads, 1, [&](int64_t begin, int64_t end) {
            for (int64_t t = begin; t < end; ++t) {
                const int64_t row_begin = t * rows_per;
                const int64_t row_end = std::min(d_size, row_begin + rows_per);
                scan_rows(row_begin, row_end, partials[static_cast<size_t>(t)]);
            }
        });
    }
    uint8_t* acc = partials[0].data();
    for (int t = 1; t < num_threads; ++t) {
        const uint8_t* part = partials[static_cast<size_t>(t)].data();
        for (int64_t col = 0; col < inner; ++col) {
            if constexpr (WantAll) {
                acc[col] &= part[col];
            } else {
                acc[col] |= part[col];
            }
        }
    }
    for (int64_t col = 0; col < inner; ++col) {
        output[col] = acc[col] != 0;
    }
}

template <bool WantAll>
static bool try_logical_reduce_dim_avx512(
    const void* input, bool* output, int64_t outer, int64_t d_size,
    int64_t inner, DType dt) {
    if (!reduce_avx512_available() || outer <= 0 || d_size <= 0) return false;
    if (dt != DType::Float32 && dt != DType::Float64) return false;
    if (inner <= 0) return true;
    const int64_t work = outer * inner * d_size;
    const int64_t vector_width = dt == DType::Float32 ? 16 : 8;
    const int64_t columns_per_task = std::max<int64_t>(
        vector_width * 4,
        ((GRAIN_SIZE / d_size + vector_width * 4 - 1) / (vector_width * 4)) *
            (vector_width * 4));
    const int64_t chunks_per_row =
        (inner + columns_per_task - 1) / columns_per_task;
    const int64_t task_count = outer * chunks_per_row;
    auto reduce_tasks = [&](int64_t begin, int64_t end) {
        uint8_t* output_bytes = reinterpret_cast<uint8_t*>(output);
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row = task / chunks_per_row;
            const int64_t chunk = task - row * chunks_per_row;
            const int64_t col_begin = chunk * columns_per_task;
            const int64_t col_end = std::min(inner, col_begin + columns_per_task);
            if (dt == DType::Float32) {
                const float* values = static_cast<const float*>(input);
                logical_reduce_f32_dim_row<WantAll>(
                    values + row * d_size * inner, output_bytes + row * inner,
                    d_size, inner, col_begin, col_end);
            } else {
                const double* values = static_cast<const double*>(input);
                logical_reduce_f64_dim_row<WantAll>(
                    values + row * d_size * inner, output_bytes + row * inner,
                    d_size, inner, col_begin, col_end);
            }
        }
    };
    if (work < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        reduce_tasks(0, task_count);
    } else {
        parallel_for(0, task_count, 1, reduce_tasks);
    }
    return true;
}

template <bool WantAll>
__attribute__((target("avx512f,avx512bw")))
static void byte_reduce_dim_avx512_row(
    const uint8_t* input, uint8_t* output, int64_t d_size, int64_t inner,
    int64_t col_begin, int64_t col_end) {
    if (inner == 1 && col_begin == 0 && col_end != 0) {
        output[0] = static_cast<uint8_t>(
            byte_reduce_avx512_range<WantAll>(input, 0, d_size));
        return;
    }

    constexpr uint64_t kAllBytes = 0xffffffffffffffffULL;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi8(1);
    int64_t col = col_begin;
    for (; col + 256 <= col_end; col += 256) {
        uint64_t result0 = WantAll ? kAllBytes : 0;
        uint64_t result1 = WantAll ? kAllBytes : 0;
        uint64_t result2 = WantAll ? kAllBytes : 0;
        uint64_t result3 = WantAll ? kAllBytes : 0;
        for (int64_t row = 0; row < d_size; ++row) {
            const uint8_t* row_input = input + row * inner + col;
            const uint64_t zero_mask0 = static_cast<uint64_t>(
                _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512(row_input), zero));
            const uint64_t zero_mask1 = static_cast<uint64_t>(
                _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512(row_input + 64), zero));
            const uint64_t zero_mask2 = static_cast<uint64_t>(
                _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512(row_input + 128), zero));
            const uint64_t zero_mask3 = static_cast<uint64_t>(
                _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512(row_input + 192), zero));
            if constexpr (WantAll) {
                result0 &= ~zero_mask0;
                result1 &= ~zero_mask1;
                result2 &= ~zero_mask2;
                result3 &= ~zero_mask3;
                if ((result0 | result1 | result2 | result3) == 0) break;
            } else {
                result0 |= ~zero_mask0;
                result1 |= ~zero_mask1;
                result2 |= ~zero_mask2;
                result3 |= ~zero_mask3;
                if ((result0 & result1 & result2 & result3) == kAllBytes) break;
            }
        }
        _mm512_storeu_si512(
            output + col,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(result0), one));
        _mm512_storeu_si512(
            output + col + 64,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(result1), one));
        _mm512_storeu_si512(
            output + col + 128,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(result2), one));
        _mm512_storeu_si512(
            output + col + 192,
            _mm512_maskz_mov_epi8(static_cast<__mmask64>(result3), one));
    }
    for (; col + 64 <= col_end; col += 64) {
        uint64_t result = WantAll ? kAllBytes : 0;
        for (int64_t row = 0; row < d_size; ++row) {
            const __m512i values = _mm512_loadu_si512(
                input + row * inner + col);
            const uint64_t zero_mask = static_cast<uint64_t>(
                _mm512_cmpeq_epi8_mask(values, zero));
            if constexpr (WantAll) {
                result &= ~zero_mask;
                if (result == 0) break;
            } else {
                result |= ~zero_mask;
                if (result == kAllBytes) break;
            }
        }
        const __m512i reduced = _mm512_maskz_mov_epi8(
            static_cast<__mmask64>(result), one);
        _mm512_storeu_si512(output + col, reduced);
    }
    for (; col < col_end; ++col) {
        bool value = WantAll;
        for (int64_t row = 0; row < d_size; ++row) {
            const bool nonzero = input[row * inner + col] != 0;
            if constexpr (WantAll) {
                value = value && nonzero;
                if (!value) break;
            } else {
                value = value || nonzero;
                if (value) break;
            }
        }
        output[col] = static_cast<uint8_t>(value);
    }
}

template <bool WantAll>
static bool try_byte_reduce_dim_avx512(
    const uint8_t* input, bool* output, int64_t outer, int64_t d_size,
    int64_t inner) {
    if (!byte_reduce_avx512_available() || outer <= 0 || d_size <= 0) {
        return false;
    }
    if (inner <= 0) return true;
    const int64_t output_count = outer * inner;
    const int64_t work = output_count * d_size;
    constexpr int64_t vector_width = 64;
    const int64_t columns_per_task = std::max<int64_t>(
        vector_width * 4,
        ((GRAIN_SIZE / d_size + vector_width * 4 - 1) / (vector_width * 4)) *
            (vector_width * 4));
    const int64_t chunks_per_row =
        (inner + columns_per_task - 1) / columns_per_task;
    const int64_t task_count = outer * chunks_per_row;
    auto reduce_tasks = [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t row_index = task / chunks_per_row;
            const int64_t chunk = task - row_index * chunks_per_row;
            const int64_t col_begin = chunk * columns_per_task;
            const int64_t col_end = std::min(inner, col_begin + columns_per_task);
            byte_reduce_dim_avx512_row<WantAll>(
                input + row_index * d_size * inner,
                reinterpret_cast<uint8_t*>(output) + row_index * inner,
                d_size, inner, col_begin, col_end);
        }
    };
    if (work < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        reduce_tasks(0, task_count);
    } else {
        parallel_for(0, task_count, 1, reduce_tasks);
    }
    return true;
}

template <bool IsMax>
__attribute__((target("avx512f")))
static std::pair<int32_t, int64_t> extremum_i32_row_avx512(
    const int32_t* input, int64_t n) {
    constexpr int64_t width = 16;
    const __m512i lane_indices = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i best_values = _mm512_loadu_si512(input);
    __m512i best_indices = lane_indices;
    int64_t i = width;
    for (; i + width <= n; i += width) {
        const __m512i values = _mm512_loadu_si512(input + i);
        const __mmask16 better = IsMax
            ? _mm512_cmp_epi32_mask(values, best_values, _MM_CMPINT_GT)
            : _mm512_cmp_epi32_mask(values, best_values, _MM_CMPINT_LT);
        best_values = _mm512_mask_blend_epi32(better, best_values, values);
        best_indices = _mm512_mask_blend_epi32(
            better, best_indices,
            _mm512_add_epi32(lane_indices, _mm512_set1_epi32(static_cast<int>(i))));
    }
    alignas(64) int32_t values[width];
    alignas(64) int32_t indices[width];
    _mm512_storeu_si512(values, best_values);
    _mm512_storeu_si512(indices, best_indices);
    int32_t best = values[0];
    int64_t best_index = indices[0];
    for (int lane = 1; lane < width; ++lane) {
        const bool better = IsMax ? values[lane] > best : values[lane] < best;
        if (better || (values[lane] == best && indices[lane] < best_index)) {
            best = values[lane];
            best_index = indices[lane];
        }
    }
    for (; i < n; ++i) {
        const int32_t value = input[i];
        const bool better = IsMax ? value > best : value < best;
        if (better || (value == best && i < best_index)) {
            best = value;
            best_index = i;
        }
    }
    return {best, best_index};
}

template <bool IsMax>
__attribute__((target("avx512f")))
static std::pair<int64_t, int64_t> extremum_i64_row_avx512(
    const int64_t* input, int64_t n) {
    constexpr int64_t width = 8;
    const __m512i lane_indices = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i best_values = _mm512_loadu_si512(input);
    __m512i best_indices = lane_indices;
    int64_t i = width;
    for (; i + width <= n; i += width) {
        const __m512i values = _mm512_loadu_si512(input + i);
        const __mmask8 better = IsMax
            ? _mm512_cmp_epi64_mask(values, best_values, _MM_CMPINT_GT)
            : _mm512_cmp_epi64_mask(values, best_values, _MM_CMPINT_LT);
        best_values = _mm512_mask_blend_epi64(better, best_values, values);
        best_indices = _mm512_mask_blend_epi64(
            better, best_indices, _mm512_add_epi64(lane_indices, _mm512_set1_epi64(i)));
    }
    alignas(64) int64_t values[width];
    alignas(64) int64_t indices[width];
    _mm512_storeu_si512(values, best_values);
    _mm512_storeu_si512(indices, best_indices);
    int64_t best = values[0];
    int64_t best_index = indices[0];
    for (int lane = 1; lane < width; ++lane) {
        const bool better = IsMax ? values[lane] > best : values[lane] < best;
        if (better || (values[lane] == best && indices[lane] < best_index)) {
            best = values[lane];
            best_index = indices[lane];
        }
    }
    for (; i < n; ++i) {
        const int64_t value = input[i];
        const bool better = IsMax ? value > best : value < best;
        if (better || (value == best && i < best_index)) {
            best = value;
            best_index = i;
        }
    }
    return {best, best_index};
}

template <bool IsMax>
static bool try_extremum_lastdim_integral_avx512(
    const Tensor& input, Tensor& values, int64_t* indices, int64_t outer,
    int64_t d_size) {
    if (!reduce_avx512_available() || d_size < 64 || outer <= 0 ||
        !input.is_contiguous() ||
        (input.dtype() != DType::Int32 && input.dtype() != DType::Int64)) {
        return false;
    }
    const int64_t row_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (input.dtype() == DType::Int32) {
        const int32_t* input_data = input.data_ptr<int32_t>();
        int32_t* output_data = values.data_ptr<int32_t>();
        parallel_for(0, outer, row_grain, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                const auto result = extremum_i32_row_avx512<IsMax>(
                    input_data + row * d_size, d_size);
                output_data[row] = result.first;
                indices[row] = result.second;
            }
        });
    } else {
        const int64_t* input_data = input.data_ptr<int64_t>();
        int64_t* output_data = values.data_ptr<int64_t>();
        parallel_for(0, outer, row_grain, [&](int64_t begin, int64_t end) {
            for (int64_t row = begin; row < end; ++row) {
                const auto result = extremum_i64_row_avx512<IsMax>(
                    input_data + row * d_size, d_size);
                output_data[row] = result.first;
                indices[row] = result.second;
            }
        });
    }
    return true;
}

// One fold step for the running extremum: a NaN candidate always wins, a
// NaN running best is never displaced, and ties keep the earlier element.
// The index scan relies on this first-occurrence convention.
template <bool IsMax, typename T>
inline void extremum_fold_step(T& best, T x) {
    if (!std::isnan(best) &&
        (std::isnan(x) || (IsMax ? x > best : x < best))) {
        best = x;
    }
}

// Extremum over contiguous rows in two passes: a vector fold produces the
// value, then a scan finds the first position holding it. The fold yields
// a NaN exactly when the row contains one, so the scan locates either the
// first NaN or the first element equal to the folded extremum.
template <bool IsMax, typename T>
static void extremum_lastdim_rows_vec(const T* in, T* out_values,
                                      int64_t* indices, int64_t outer,
                                      int64_t d_size) {
    using vec_t = Vectorized<T>;
    constexpr int64_t width = static_cast<int64_t>(vec_t::size());
    const int64_t row_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size);
    parallel_for(0, outer, row_grain, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const T* line = in + row * d_size;
            vec_t acc = vec_t::loadu(line);
            int64_t i = width;
            for (; i + width <= d_size; i += width) {
                const vec_t x = vec_t::loadu(line + i);
                acc = IsMax ? maximum(acc, x) : minimum(acc, x);
            }
            alignas(64) T lanes[width];
            acc.store(lanes);
            T best = lanes[0];
            for (int64_t k = 1; k < width; ++k) {
                extremum_fold_step<IsMax>(best, lanes[k]);
            }
            for (; i < d_size; ++i) {
                extremum_fold_step<IsMax>(best, line[i]);
            }
            int64_t bi = 0;
            if (std::isnan(best)) {
                while (!std::isnan(line[bi])) ++bi;
            } else {
                while (line[bi] != best) ++bi;
            }
            out_values[row] = best;
            indices[row] = bi;
        }
    });
}

// Tier copies of the last-dimension extremum fold: the lane index travels
// with the lane value, so a row resolves in one pass instead of folding the
// value and then rescanning the row for its first occurrence.
template <bool IsMax>
static bool try_extremum_lastdim_real(
    const void* xv, void* vv, int64_t* indices, int64_t outer,
    int64_t d_size, DType dt) {
    if (d_size < 8 || outer <= 0) return false;
    const int64_t row_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (dt == DType::Float32) {
        const float* input = static_cast<const float*>(xv);
        float* values = static_cast<float*>(vv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    auto result = vec::row_extremum<IsMax>(
                        input + row * d_size, d_size);
                    values[row] = result.first;
                    indices[row] = result.second;
                }
            });
        return true;
    }
    if (dt == DType::Float64) {
        const double* input = static_cast<const double*>(xv);
        double* values = static_cast<double*>(vv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    auto result = vec::row_extremum<IsMax>(
                        input + row * d_size, d_size);
                    values[row] = result.first;
                    indices[row] = result.second;
                }
            });
        return true;
    }
    return false;
}

// The same fold for a reduced width: the lanes widen a step at a time, the
// extreme is folded there, and a second walk finds the first position that
// holds it.  A value the walk reads back from the layout keeps the exact bits
// the first holder carried, which is the element the flat fold would keep.
template <bool IsMax>
inline float reduced_extremum_step(float best, float v) {
    if (std::isnan(best)) return best;
    if (std::isnan(v)) return v;
    return IsMax ? (v > best ? v : best) : (v < best ? v : best);
}

template <bool IsMax, bool BFloat>
__attribute__((target("avx2,fma,f16c"), noinline))
static void reduced_row_extremum(const uint16_t* line, int64_t n,
                                 uint16_t* value_out, int64_t* index_out) {
    __m256 acc = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_8(line)
                        : tensorplay::cpu::vecreduced::widen8(line);
    int64_t i = 8;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_8(line + i)
                                : tensorplay::cpu::vecreduced::widen8(line + i);
        const __m256 folded = IsMax ? _mm256_max_ps(acc, v) : _mm256_min_ps(acc, v);
        // The plain fold answers with one operand, so a not-a-number in either
        // lane would be replaced; the unordered lanes are put back.
        acc = _mm256_or_ps(folded, _mm256_cmp_ps(acc, v, _CMP_UNORD_Q));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float best = lanes[0];
    for (int k = 1; k < 8; ++k) best = reduced_extremum_step<IsMax>(best, lanes[k]);
    for (; i < n; ++i) {
        const float v = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_one(line[i])
                               : tensorplay::cpu::vecreduced::widen_one(line[i]);
        best = reduced_extremum_step<IsMax>(best, v);
    }
    const __m256 vb = _mm256_set1_ps(best);
    const bool want_nan = std::isnan(best);
    int64_t found = -1;
    int64_t j = 0;
    for (; found < 0 && j + 8 <= n; j += 8) {
        const __m256 v = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_8(line + j)
                                : tensorplay::cpu::vecreduced::widen8(line + j);
        const int mask = want_nan
            ? _mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q))
            : _mm256_movemask_ps(_mm256_cmp_ps(v, vb, _CMP_EQ_OQ));
        if (mask) found = j + __builtin_ctz(static_cast<unsigned>(mask));
    }
    for (; found < 0 && j < n; ++j) {
        const float v = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_one(line[j])
                               : tensorplay::cpu::vecreduced::widen_one(line[j]);
        if (want_nan ? std::isnan(v) : (v == best)) found = j;
    }
    *index_out = found;
    *value_out = line[found];
}

// The same fold down a scanned axis that leaves packed columns per row: a
// block of eight columns per vector is folded down the axis, and a second walk
// down the same block records, per lane, the first row that holds the folded
// value.  A not-a-number lane is matched by its own rule, since such a value
// never equals anything.
template <bool IsMax, bool BFloat, int V>
__attribute__((target("avx2,fma,f16c"), noinline))
static void reduced_extremum_columns(const uint16_t* x, uint16_t* values,
                                     int64_t* indices, int64_t d_size,
                                     int64_t inner) {
    __m256 best[V];
    for (int k = 0; k < V; ++k) {
        best[k] = BFloat ? tensorplay::cpu::vecreduced::widen_bf16_8(x + 8 * k)
                         : tensorplay::cpu::vecreduced::widen8(x + 8 * k);
    }
    for (int64_t r = 1; r < d_size; ++r) {
        for (int k = 0; k < V; ++k) {
            const __m256 v = BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(x + r * inner + 8 * k)
                : tensorplay::cpu::vecreduced::widen8(x + r * inner + 8 * k);
            const __m256 folded = IsMax ? _mm256_max_ps(best[k], v)
                                        : _mm256_min_ps(best[k], v);
            best[k] = _mm256_or_ps(folded,
                                   _mm256_cmp_ps(best[k], v, _CMP_UNORD_Q));
        }
    }
    uint32_t pending = (8 * V >= 32) ? 0xFFFFFFFFu : ((1u << (8 * V)) - 1u);
    for (int64_t r = 0; r < d_size && pending != 0; ++r) {
        for (int k = 0; k < V; ++k) {
            const __m256 v = BFloat
                ? tensorplay::cpu::vecreduced::widen_bf16_8(x + r * inner + 8 * k)
                : tensorplay::cpu::vecreduced::widen8(x + r * inner + 8 * k);
            const __m256 equal = _mm256_cmp_ps(v, best[k], _CMP_EQ_OQ);
            const __m256 unnumbered = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            const __m256 nan_lane = _mm256_cmp_ps(best[k], best[k], _CMP_UNORD_Q);
            const uint32_t bits =
                static_cast<uint32_t>(_mm256_movemask_ps(
                    _mm256_blendv_ps(equal, unnumbered, nan_lane)))
                << (8 * k);
            uint32_t fresh = bits & pending;
            while (fresh != 0) {
                unsigned lane = static_cast<unsigned>(__builtin_ctz(fresh));
                indices[lane] = r;
                fresh &= fresh - 1;
            }
            pending &= ~bits;
        }
    }
    for (int lane = 0; lane < 8 * V; ++lane) {
        values[lane] = x[indices[lane] * inner + lane];
    }
}

template <bool IsMax>
static bool try_extremum_reduced_columns(const Tensor& input, Tensor& values,
                                         int64_t* indices, int64_t outer,
                                         int64_t d_size, int64_t inner) {
    const DType dt = input.dtype();
    const bool half_width = dt == DType::Float16;
    if (!half_width && dt != DType::BFloat16) return false;
    if (inner < 8 || d_size < 2) return false;
    if (!tensorplay::cpu::vecreduced::f16c_available()) return false;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    uint16_t* out = half_width
        ? reinterpret_cast<uint16_t*>(values.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(values.data_ptr<tensorplay::BFloat16>());
    const int v = inner >= 32 ? 4 : 1;
    const int64_t width = 8 * v;
    const int64_t full = inner / width;
    const int64_t block_grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    const auto block = [&](int64_t flat) {
        const int64_t o = flat / full;
        const int64_t c0 = (flat - o * full) * width;
        const int64_t src = o * d_size * inner + c0;
        const int64_t dst = o * inner + c0;
        if (half_width) {
            if (v == 4) {
                reduced_extremum_columns<IsMax, false, 4>(
                    in + src, out + dst, indices + dst, d_size, inner);
            } else {
                reduced_extremum_columns<IsMax, false, 1>(
                    in + src, out + dst, indices + dst, d_size, inner);
            }
        } else {
            if (v == 4) {
                reduced_extremum_columns<IsMax, true, 4>(
                    in + src, out + dst, indices + dst, d_size, inner);
            } else {
                reduced_extremum_columns<IsMax, true, 1>(
                    in + src, out + dst, indices + dst, d_size, inner);
            }
        }
    };
    parallel_for(0, outer * full, block_grain,
                 [&](int64_t b, int64_t e) {
                     for (int64_t flat = b; flat < e; ++flat) block(flat);
                 });
    if (full * width < inner) {
        // The last block covers the columns at the end of the row, which the
        // block before it also covered; both answer the same values there.
        const auto tail = [&](int64_t flat) {
            const int64_t o = flat;
            const int64_t src = o * d_size * inner + (inner - width);
            const int64_t dst = o * inner + (inner - width);
            if (half_width) {
                if (v == 4) {
                    reduced_extremum_columns<IsMax, false, 4>(
                        in + src, out + dst, indices + dst, d_size, inner);
                } else {
                    reduced_extremum_columns<IsMax, false, 1>(
                        in + src, out + dst, indices + dst, d_size, inner);
                }
            } else {
                if (v == 4) {
                    reduced_extremum_columns<IsMax, true, 4>(
                        in + src, out + dst, indices + dst, d_size, inner);
                } else {
                    reduced_extremum_columns<IsMax, true, 1>(
                        in + src, out + dst, indices + dst, d_size, inner);
                }
            }
        };
        parallel_for(0, outer, block_grain,
                     [&](int64_t b, int64_t e) {
                         for (int64_t flat = b; flat < e; ++flat) tail(flat);
                     });
    }
    return true;
}

template <bool IsMax>
static bool try_extremum_lastdim_reduced(const Tensor& input, Tensor& values,
                                         int64_t* indices, int64_t outer,
                                         int64_t d_size) {
    const DType dt = input.dtype();
    const bool half_width = dt == DType::Float16;
    if (!half_width && dt != DType::BFloat16) return false;
    if (!tensorplay::cpu::vecreduced::f16c_available()) return false;
    const uint16_t* in = half_width
        ? reinterpret_cast<const uint16_t*>(input.data_ptr<tensorplay::Half>())
        : reinterpret_cast<const uint16_t*>(
              input.data_ptr<tensorplay::BFloat16>());
    uint16_t* out = half_width
        ? reinterpret_cast<uint16_t*>(values.data_ptr<tensorplay::Half>())
        : reinterpret_cast<uint16_t*>(values.data_ptr<tensorplay::BFloat16>());
    const int64_t row_grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            if (half_width) {
                reduced_row_extremum<IsMax, false>(in + r * d_size, d_size,
                                                   out + r, indices + r);
            } else {
                reduced_row_extremum<IsMax, true>(in + r * d_size, d_size,
                                                  out + r, indices + r);
            }
        }
    });
    return true;
}

template <bool IsMax>
static bool try_extremum_lastdim_vec(const Tensor& input, Tensor& values,
                                     int64_t* indices, int64_t outer,
                                     int64_t d_size) {
    if (d_size < 32 || outer <= 0 || !input.is_contiguous()) return false;
    if (input.dtype() == DType::Float32) {
        extremum_lastdim_rows_vec<IsMax>(input.data_ptr<float>(),
                                         values.data_ptr<float>(), indices,
                                         outer, d_size);
        return true;
    }
    if (input.dtype() == DType::Float64) {
        extremum_lastdim_rows_vec<IsMax>(input.data_ptr<double>(),
                                         values.data_ptr<double>(), indices,
                                         outer, d_size);
        return true;
    }
    return try_extremum_lastdim_reduced<IsMax>(input, values, indices, outer,
                                               d_size);
}

// Column-wise fold over a leading dimension: the output span gathers the
// rows that share its columns, so the walk strides by the row length.  Four
// accumulators per pass keep any one chain short, and the tier compilation
// picks the lane width.
template <typename T>
static void sum_leading_range(const T* input, T* output, int64_t rows,
                              int64_t cols, int64_t begin, int64_t end) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = static_cast<int64_t>(Vec::size());
    int64_t col = begin;
    for (; col + 4 * W <= end; col += 4 * W) {
        Vec s0(T(0)), s1(T(0)), s2(T(0)), s3(T(0));
        for (int64_t row = 0; row < rows; ++row) {
            const T* p = input + row * cols + col;
            s0 = s0 + Vec::loadu(p);
            s1 = s1 + Vec::loadu(p + W);
            s2 = s2 + Vec::loadu(p + 2 * W);
            s3 = s3 + Vec::loadu(p + 3 * W);
        }
        s0.store(output + col);
        s1.store(output + col + W);
        s2.store(output + col + 2 * W);
        s3.store(output + col + 3 * W);
    }
    for (; col + W <= end; col += W) {
        Vec acc(T(0));
        for (int64_t row = 0; row < rows; ++row) {
            acc = acc + Vec::loadu(input + row * cols + col);
        }
        acc.store(output + col);
    }
    for (; col < end; ++col) {
        T sum = T(0);
        for (int64_t row = 0; row < rows; ++row) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

// Contiguous reduction over one non-trailing dimension; returns false when
// the caller should use the iterator path.
static bool try_sum_dim_real(const Tensor& input, Tensor& output, int64_t dim) {
    if (!input.is_contiguous() || input.numel() == 0 || input.dim() == 0) {
        return false;
    }
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        return false;
    }

    const int64_t ndim = input.dim();
    if (dim == ndim - 1) {
        const int64_t d_size = input.size(dim);
        constexpr int64_t kMinRow = 16;
        if (d_size < kMinRow) return false;
        const int64_t rows = input.numel() / d_size;
        // A handful of very long rows cannot be spread across the team by
        // row: there are fewer rows than workers.  Chunk the column range
        // instead, fold one partial per worker slot, and merge the slots in
        // a fixed order so a rerun reproduces the same quotient.
        if (d_size >= 65536 && rows <= 8) {
            const int64_t nslots = std::max<int64_t>(get_num_threads(), 1);
            // Keep at least one chunk per slot; chunks finer than the floor
            // would only add scheduling overhead.
            const int64_t grain =
                std::min<int64_t>(GRAIN_SIZE, std::max<int64_t>(8192, d_size / nslots));
            if (input.dtype() == DType::Float32) {
                const float* values = input.data_ptr<float>();
                float* results = output.data_ptr<float>();
                std::vector<double> part(rows * nslots, 0.0);
                parallel_for(0, d_size, grain,
                             [&](int64_t b, int64_t e) {
                    const int64_t tid = get_thread_num();
                    for (int64_t row = 0; row < rows; ++row) {
                        part[row * nslots + tid] +=
                            vec::sum(values + row * d_size, b, e);
                    }
                });
                for (int64_t row = 0; row < rows; ++row) {
                    double s = 0.0;
                    for (int64_t k = 0; k < nslots; ++k) {
                        s += part[row * nslots + k];
                    }
                    results[row] = static_cast<float>(s);
                }
            } else {
                const double* values = input.data_ptr<double>();
                double* results = output.data_ptr<double>();
                std::vector<double> part(rows * nslots, 0.0);
                parallel_for(0, d_size, grain,
                             [&](int64_t b, int64_t e) {
                    const int64_t tid = get_thread_num();
                    for (int64_t row = 0; row < rows; ++row) {
                        part[row * nslots + tid] +=
                            vec::sum(values + row * d_size, b, e);
                    }
                });
                for (int64_t row = 0; row < rows; ++row) {
                    double s = 0.0;
                    for (int64_t k = 0; k < nslots; ++k) {
                        s += part[row * nslots + k];
                    }
                    results[row] = s;
                }
            }
            return true;
        }
        const int64_t row_grain = std::max<int64_t>(
            1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
        if (input.dtype() == DType::Float32) {
            const float* values = input.data_ptr<float>();
            float* results = output.data_ptr<float>();
            parallel_for(0, rows, row_grain, [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    results[row] = vec::sum(values + row * d_size, 0, d_size);
                }
            });
        } else {
            const double* values = input.data_ptr<double>();
            double* results = output.data_ptr<double>();
            parallel_for(0, rows, row_grain, [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    results[row] = vec::sum(values + row * d_size, 0, d_size);
                }
            });
        }
        return true;
    }

    if (dim < 0 || dim >= ndim) return false;
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= input.size(i);
    const int64_t rows = input.size(dim);
    const int64_t cols = input.numel() / (outer * rows);
    const int64_t vector_columns =
        static_cast<int64_t>(Vectorized<float>::size()) * 4;
    if (cols < static_cast<int64_t>(Vectorized<float>::size())) return false;
    const int64_t approximate_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(rows, 1));
    const int64_t column_grain = std::max<int64_t>(
        vector_columns,
        ((approximate_grain + vector_columns - 1) / vector_columns) *
            vector_columns);
    if (input.dtype() == DType::Float32) {
        const float* values = input.data_ptr<float>();
        float* results = output.data_ptr<float>();
        const int64_t chunks_per_row =
            (cols + column_grain - 1) / column_grain;
        const int64_t task_count = outer * chunks_per_row;
        auto reduce_tasks = [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t row = task / chunks_per_row;
                const int64_t chunk = task - row * chunks_per_row;
                const int64_t col_begin = chunk * column_grain;
                const int64_t col_end = std::min(cols, col_begin + column_grain);
                sum_leading_range(
                    values + row * rows * cols, results + row * cols,
                    rows, cols, col_begin, col_end);
            }
        };
        if (input.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            reduce_tasks(0, task_count);
        } else {
            parallel_for(0, task_count, 1, reduce_tasks);
        }
    } else {
        const double* values = input.data_ptr<double>();
        double* results = output.data_ptr<double>();
        const int64_t chunks_per_row =
            (cols + column_grain - 1) / column_grain;
        const int64_t task_count = outer * chunks_per_row;
        auto reduce_tasks = [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t row = task / chunks_per_row;
                const int64_t chunk = task - row * chunks_per_row;
                const int64_t col_begin = chunk * column_grain;
                const int64_t col_end = std::min(cols, col_begin + column_grain);
                sum_leading_range(
                    values + row * rows * cols, results + row * cols,
                    rows, cols, col_begin, col_end);
            }
        };
        if (input.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            reduce_tasks(0, task_count);
        } else {
            parallel_for(0, task_count, 1, reduce_tasks);
        }
    }
    return true;
}
// Leading-dim reduction whose columns fit in one to four lane-vectors: each
// column vector keeps its accumulator in a register, and four rows fold into
// it at a time. The loads of the other three rows run while the accumulator's
// own add is still in flight, so the walk is limited by the input stream and
// not by the latency of one dependent add per row.
template <typename Scalar, int64_t NC>
inline void sum_leading_narrow_range(const Scalar* TP_RESTRICT in,
                                     Scalar* TP_RESTRICT dst, int64_t rows,
                                     int64_t cols, int64_t begin) {
    using Vec = Vectorized<Scalar>;
    constexpr int64_t vs = Vec::size();
    Vec acc[NC];
    for (int64_t u = 0; u < NC; ++u) {
        acc[u] = Vec::loadu(in + begin + u * vs);
    }
    int64_t r = 1;
    for (; r + 4 <= rows; r += 4) {
        const Scalar* TP_RESTRICT p0 = in + (r + 0) * cols + begin;
        const Scalar* TP_RESTRICT p1 = in + (r + 1) * cols + begin;
        const Scalar* TP_RESTRICT p2 = in + (r + 2) * cols + begin;
        const Scalar* TP_RESTRICT p3 = in + (r + 3) * cols + begin;
        for (int64_t u = 0; u < NC; ++u) {
            acc[u] = acc[u] + ((Vec::loadu(p0 + u * vs) + Vec::loadu(p1 + u * vs)) +
                               (Vec::loadu(p2 + u * vs) + Vec::loadu(p3 + u * vs)));
        }
    }
    for (; r < rows; ++r) {
        const Scalar* TP_RESTRICT p = in + r * cols + begin;
        for (int64_t u = 0; u < NC; ++u) {
            acc[u] = acc[u] + Vec::loadu(p + u * vs);
        }
    }
    for (int64_t u = 0; u < NC; ++u) {
        acc[u].store(dst + begin + u * vs);
    }
}

// The same walk over a column count that is not a whole number of
// lane-vectors: the last band folds with the lanes past the column count
// replaced by zero, and the tensor's own last row is read through a padded
// copy, so no read leaves the buffer and no store writes past the answer's own
// end.
template <typename Scalar, int64_t NC>
inline void sum_leading_narrow_tail_range(const Scalar* TP_RESTRICT in,
                                          Scalar* TP_RESTRICT dst, int64_t rows,
                                          int64_t cols, int64_t begin) {
    using Vec = Vectorized<Scalar>;
    constexpr int64_t vs = Vec::size();
    const int64_t span = cols - begin;
    const int64_t tail = span - (NC - 1) * vs;
    const Vec zero(Scalar(0));
    alignas(64) unsigned char bits[vs * sizeof(Scalar)];
    std::memset(bits, 0, sizeof(bits));
    std::memset(bits, 0xFF, static_cast<size_t>(tail) * sizeof(Scalar));
    const Vec keep = Vec::loadu(reinterpret_cast<const Scalar*>(bits));
    const int64_t last = begin + (NC - 1) * vs;

    Vec acc[NC];
    for (int64_t u = 0; u + 1 < NC; ++u) acc[u] = Vec::loadu(in + begin + u * vs);
    acc[NC - 1] = Vec::blendv(zero, Vec::loadu(in + last), keep);
    int64_t r = 1;
    for (; r + 4 < rows; r += 4) {
        const Scalar* TP_RESTRICT p0 = in + (r + 0) * cols;
        const Scalar* TP_RESTRICT p1 = in + (r + 1) * cols;
        const Scalar* TP_RESTRICT p2 = in + (r + 2) * cols;
        const Scalar* TP_RESTRICT p3 = in + (r + 3) * cols;
        for (int64_t u = 0; u + 1 < NC; ++u) {
            const int64_t at = begin + u * vs;
            acc[u] = acc[u] + ((Vec::loadu(p0 + at) + Vec::loadu(p1 + at)) +
                               (Vec::loadu(p2 + at) + Vec::loadu(p3 + at)));
        }
        acc[NC - 1] = acc[NC - 1] +
            ((Vec::blendv(zero, Vec::loadu(p0 + last), keep) +
              Vec::blendv(zero, Vec::loadu(p1 + last), keep)) +
             (Vec::blendv(zero, Vec::loadu(p2 + last), keep) +
              Vec::blendv(zero, Vec::loadu(p3 + last), keep)));
    }
    for (; r < rows - 1; ++r) {
        const Scalar* TP_RESTRICT p = in + r * cols;
        for (int64_t u = 0; u + 1 < NC; ++u) {
            acc[u] = acc[u] + Vec::loadu(p + begin + u * vs);
        }
        acc[NC - 1] = acc[NC - 1] + Vec::blendv(zero, Vec::loadu(p + last), keep);
    }
    {
        const Scalar* TP_RESTRICT p = in + (rows - 1) * cols;
        for (int64_t u = 0; u + 1 < NC; ++u) {
            acc[u] = acc[u] + Vec::loadu(p + begin + u * vs);
        }
        alignas(64) Scalar pad[vs];
        std::memcpy(pad, p + last, static_cast<size_t>(tail) * sizeof(Scalar));
        for (int64_t k = tail; k < vs; ++k) pad[k] = Scalar(0);
        acc[NC - 1] = acc[NC - 1] + Vec::loadu(pad);
    }
    for (int64_t u = 0; u + 1 < NC; ++u) acc[u].store(dst + begin + u * vs);
    alignas(64) Scalar lanes[vs];
    acc[NC - 1].store(lanes);
    std::memcpy(dst + last, lanes, static_cast<size_t>(tail) * sizeof(Scalar));
}

// Dispatch the ragged walk by the number of lane-vectors its column span
// needs; the span may start at a band boundary.
template <typename Scalar>
inline bool sum_leading_narrow_tail_dispatch(const Scalar* TP_RESTRICT in,
                                             Scalar* TP_RESTRICT out,
                                             int64_t rows, int64_t cols,
                                             int64_t begin) {
    using Vec = Vectorized<Scalar>;
    constexpr int64_t vs = Vec::size();
    const int64_t span = cols - begin;
    switch ((span + vs - 1) / vs) {
        case 1:
            sum_leading_narrow_tail_range<Scalar, 1>(in, out, rows, cols,
                                                       begin);
            return true;
        case 2:
            sum_leading_narrow_tail_range<Scalar, 2>(in, out, rows, cols,
                                                       begin);
            return true;
        case 3:
            sum_leading_narrow_tail_range<Scalar, 3>(in, out, rows, cols,
                                                       begin);
            return true;
        case 4:
            sum_leading_narrow_tail_range<Scalar, 4>(in, out, rows, cols,
                                                       begin);
            return true;
        case 5:
            sum_leading_narrow_tail_range<Scalar, 5>(in, out, rows, cols,
                                                       begin);
            return true;
        case 6:
            sum_leading_narrow_tail_range<Scalar, 6>(in, out, rows, cols,
                                                       begin);
            return true;
        case 7:
            sum_leading_narrow_tail_range<Scalar, 7>(in, out, rows, cols,
                                                       begin);
            return true;
        case 8:
            sum_leading_narrow_tail_range<Scalar, 8>(in, out, rows, cols,
                                                       begin);
            return true;
        case 9:
            sum_leading_narrow_tail_range<Scalar, 9>(in, out, rows, cols,
                                                       begin);
            return true;
        case 10:
            sum_leading_narrow_tail_range<Scalar, 10>(in, out, rows, cols,
                                                       begin);
            return true;
        case 11:
            sum_leading_narrow_tail_range<Scalar, 11>(in, out, rows, cols,
                                                       begin);
            return true;
        case 12:
            sum_leading_narrow_tail_range<Scalar, 12>(in, out, rows, cols,
                                                       begin);
            return true;
        case 13:
            sum_leading_narrow_tail_range<Scalar, 13>(in, out, rows, cols,
                                                       begin);
            return true;
        case 14:
            sum_leading_narrow_tail_range<Scalar, 14>(in, out, rows, cols,
                                                       begin);
            return true;
        case 15:
            sum_leading_narrow_tail_range<Scalar, 15>(in, out, rows, cols,
                                                       begin);
            return true;
        case 16:
            sum_leading_narrow_tail_range<Scalar, 16>(in, out, rows, cols,
                                                       begin);
            return true;
        default:
            return false;
    }
}

// Dispatch the narrow-column walk over the data types it serves; a row count
// too short to fill four folds is left to the other shapes.
template <typename Scalar>
inline bool sum_leading_narrow_impl(const Scalar* TP_RESTRICT in,
                                    Scalar* TP_RESTRICT out, int64_t rows,
                                    int64_t cols) {
    using Vec = Vectorized<Scalar>;
    constexpr int64_t vs = Vec::size();
    constexpr int64_t bands = 16 * vs;
    if (rows < 8 || cols == 0) {
        return false;
    }
    if (cols > bands) {
        // More columns than the register file holds at once: a slice that still
        // fits a level-one cache is walked in passes of sixteen lane-vectors,
        // so every total stays in a register across the rows it folds.
        if (cols * static_cast<int64_t>(sizeof(Scalar)) > 4096) return false;
        int64_t begin = 0;
        for (; begin + bands <= cols; begin += bands) {
            sum_leading_narrow_range<Scalar, 16>(in, out, rows, cols, begin);
        }
        if (begin < cols) {
            return sum_leading_narrow_tail_dispatch<Scalar>(in, out, rows, cols,
                                                            begin);
        }
        return true;
    }
    if (cols % vs != 0) {
        return sum_leading_narrow_tail_dispatch<Scalar>(in, out, rows, cols, 0);
    }
    switch (cols / vs) {
        case 1:
            sum_leading_narrow_range<Scalar, 1>(in, out, rows, cols, 0);
            return true;
        case 2:
            sum_leading_narrow_range<Scalar, 2>(in, out, rows, cols, 0);
            return true;
        case 3:
            sum_leading_narrow_range<Scalar, 3>(in, out, rows, cols, 0);
            return true;
        case 4:
            sum_leading_narrow_range<Scalar, 4>(in, out, rows, cols, 0);
            return true;
        case 5:
            sum_leading_narrow_range<Scalar, 5>(in, out, rows, cols, 0);
            return true;
        case 6:
            sum_leading_narrow_range<Scalar, 6>(in, out, rows, cols, 0);
            return true;
        case 7:
            sum_leading_narrow_range<Scalar, 7>(in, out, rows, cols, 0);
            return true;
        case 8:
            sum_leading_narrow_range<Scalar, 8>(in, out, rows, cols, 0);
            return true;
        case 9:
            sum_leading_narrow_range<Scalar, 9>(in, out, rows, cols, 0);
            return true;
        case 10:
            sum_leading_narrow_range<Scalar, 10>(in, out, rows, cols, 0);
            return true;
        case 11:
            sum_leading_narrow_range<Scalar, 11>(in, out, rows, cols, 0);
            return true;
        case 12:
            sum_leading_narrow_range<Scalar, 12>(in, out, rows, cols, 0);
            return true;
        case 13:
            sum_leading_narrow_range<Scalar, 13>(in, out, rows, cols, 0);
            return true;
        case 14:
            sum_leading_narrow_range<Scalar, 14>(in, out, rows, cols, 0);
            return true;
        case 15:
            sum_leading_narrow_range<Scalar, 15>(in, out, rows, cols, 0);
            return true;
        case 16:
            sum_leading_narrow_range<Scalar, 16>(in, out, rows, cols, 0);
            return true;
        default:
            return false;
    }
}

// Leading-dim contiguous reduction over a small row count. Each task owns a
// contiguous span of the output and streams one full row segment at a time
// with kUnroll independent vector accumulators. A round-robin walk over row
// groups instead interleaves one stream per row, and once the row stride
// exceeds the page size those streams are too far apart for the hardware
// prefetchers to track together. Accumulation is sequential over at most
// kMaxRows terms per output element, the same error class as the unbounded
// per-row walks used for trailing-dim reductions. Every output element is
// stored exactly once, so the result is written into freshly allocated
// storage without pre-zeroing it.
template <int64_t kUnroll, typename Scalar>
bool sum_leading_rows_stream(const Tensor& input, Tensor& output) {
    constexpr int64_t kMaxRows = 32;
    const int64_t rows = input.size(0);
    const int64_t cols = input.numel() / rows;
    if (rows < 2 || rows > kMaxRows || cols == 0) {
        return false;
    }

    using Vec = Vectorized<Scalar>;
    constexpr int64_t vec_size = Vec::size();
    const int64_t group = kUnroll * vec_size;
    const int64_t chunk_target =
        std::max<int64_t>(group, GRAIN_SIZE / rows);
    const int64_t chunk = ((chunk_target + group - 1) / group) * group;
    const Scalar* TP_RESTRICT in =
        static_cast<const Scalar*>(input.data_ptr());
    Scalar* TP_RESTRICT out = static_cast<Scalar*>(output.data_ptr());
    parallel_for(0, cols, chunk, [&](int64_t begin, int64_t end) {
        const int64_t n = end - begin;
        const Scalar* TP_RESTRICT base0 = in + begin;
        Scalar* TP_RESTRICT dst = out + begin;
        int64_t j = 0;
        for (; j + group <= n; j += group) {
            Vec acc[kUnroll];
            for (int64_t u = 0; u < kUnroll; ++u) {
                acc[u] = Vec::loadu(base0 + j + u * vec_size);
            }
            for (int64_t r = 1; r < rows; ++r) {
                const Scalar* TP_RESTRICT row = in + r * cols + begin + j;
                for (int64_t u = 0; u < kUnroll; ++u) {
                    acc[u] = acc[u] + Vec::loadu(row + u * vec_size);
                }
            }
            for (int64_t u = 0; u < kUnroll; ++u) {
                acc[u].store(dst + j + u * vec_size);
            }
        }
        for (; j + vec_size <= n; j += vec_size) {
            Vec a = Vec::loadu(base0 + j);
            for (int64_t r = 1; r < rows; ++r) {
                a = a + Vec::loadu(in + r * cols + begin + j);
            }
            a.store(dst + j);
        }
        for (; j < n; ++j) {
            Scalar s = base0[j];
            for (int64_t r = 1; r < rows; ++r) {
                s += in[r * cols + begin + j];
            }
            dst[j] = s;
        }
    });
    return true;
}

static std::optional<Tensor> try_sum_leading_rows(
    const Tensor& input, const std::vector<int64_t>& out_shape, int64_t dim) {
    if (dim != 0 || !input.is_contiguous() || input.dim() == 0 ||
        input.numel() == 0) {
        return std::nullopt;
    }
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        return std::nullopt;
    }
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    if (input.size(0) == 1) {
        // One row: the reduction is the row itself, copied verbatim.
        const size_t bytes = input.numel() * (input.dtype() == DType::Float32
                                                  ? sizeof(float)
                                                  : sizeof(double));
        std::memcpy(out.data_ptr(), input.data_ptr(), bytes);
        return out;
    }
    bool handled = false;
    const bool alone = tensorplay::parallel::get_num_threads() == 1;
    if (alone) {
        // A single walker keeps every column vector's total in a register; a
        // team would have each member re-read the whole input for its own
        // columns, so the row split serves that case.
        const int64_t rows = input.size(0);
        const int64_t cols = input.numel() / rows;
        handled = input.dtype() == DType::Float32
            ? sum_leading_narrow_impl<float>(
                  input.data_ptr<float>(), out.data_ptr<float>(), rows, cols)
            : sum_leading_narrow_impl<double>(
                  input.data_ptr<double>(), out.data_ptr<double>(), rows, cols);
    }
    if (!handled && input.dtype() == DType::Float32) {
        handled = sum_leading_rows_stream<8, float>(input, out);
    }
    if (!handled) {
        return std::nullopt;
    }
    return out;
}

// Row-split leading-dim reduction: the row range is tiled into one block per
// task, and each task folds its block into a private column slice with fully
// sequential reads before the slices combine into the output. A column split
// makes every task walk all rows once per column block, and past a few dozen
// rows those strided walks overwhelm the prefetchers; the row split reads one
// contiguous block per task and pays a buffer pass instead, which wins once
// the row count is a large multiple of the team size. Tasks own their slice
// by index and zero it themselves, so every slice is written exactly once and
// the fold needs no cross-task coordination.
//
// A running sum's rounding error grows with the depth of the accumulation
// chain, so a task folding thousands of rows into one slice compounds far more
// error than a short partial sum. Tasks past a few dozen rows per task fold
// through a carry cascade instead: level 0 absorbs a fixed group of rows and
// is carried into level 1, level 1 into level 2 every 16 groups, level 2 into
// level 3 every 256 groups, and the levels fold together at the end. Each add
// then lands on an accumulator holding only a bounded number of rows' worth of
// magnitude no matter how many rows the task owns, capping the effective chain
// depth near the group size times the level count.
template <typename Scalar>
std::optional<Tensor> sum_leading_rows_split_impl(
    const Tensor& input, const std::vector<int64_t>& out_shape) {
    const int64_t rows = input.size(0);
    const int64_t cols = input.numel() / rows;
    const int64_t nthreads =
        std::max<int64_t>(1, tensorplay::parallel::get_num_threads());
    // Each task's accumulator slice must stay cache-resident while its rows
    // fold in; past that the read-modify-write traffic rivals the input walk
    // and the column split is the better shape. The row count must also be a
    // wide multiple of the team size: below a few dozen rows per task the
    // column split's strided walks still prefetch well and the buffer pass is
    // pure overhead.
    if (rows < 32 * nthreads || cols <= 0 ||
        cols * static_cast<int64_t>(sizeof(Scalar)) > 16384) {
        return std::nullopt;
    }
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    const int64_t grain = (rows + nthreads - 1) / nthreads;
    const bool cascade = grain > 64;
    // The group stays at 16 rows unless the task's row count outgrows what
    // four carry levels cap, so the wide levels only ever hold whole groups.
    int64_t lg = 0;
    while ((int64_t(1) << lg) < grain) {
        ++lg;
    }
    const int64_t group = int64_t(1) << std::max<int64_t>(4, std::min<int64_t>(7, lg / 2));
    const int64_t tslice = cascade ? 4 * cols : cols;
    Tensor buffer =
        Tensor::empty({nthreads * tslice}, input.dtype(), input.device());

    const Scalar* TP_RESTRICT in =
        static_cast<const Scalar*>(input.data_ptr());
    Scalar* TP_RESTRICT buf = static_cast<Scalar*>(buffer.data_ptr());
    Scalar* TP_RESTRICT outp = static_cast<Scalar*>(out.data_ptr());
    using Vec = Vectorized<Scalar>;
    constexpr int64_t vs = Vec::size();
    const int64_t vec_cols = cols & ~(vs - 1);

    // Fold n rows starting at row index r0 into one accumulator slice. Four
    // rows land in each accumulator slot before it is read back, so the buffer
    // is touched a quarter as often: that is the difference between a pass
    // limited by L1 traffic and one limited by the input stream.
    auto fold_rows = [&](Scalar* TP_RESTRICT acc, int64_t r0, int64_t n) {
        int64_t s = 0;
        for (; s + 4 <= n; s += 4) {
            const Scalar* TP_RESTRICT p0 = in + (r0 + s) * cols;
            const Scalar* TP_RESTRICT p1 = p0 + cols;
            const Scalar* TP_RESTRICT p2 = p1 + cols;
            const Scalar* TP_RESTRICT p3 = p2 + cols;
            int64_t j = 0;
            for (; j + vs <= cols; j += vs) {
                (Vec::loadu(acc + j) +
                 ((Vec::loadu(p0 + j) + Vec::loadu(p1 + j)) +
                  (Vec::loadu(p2 + j) + Vec::loadu(p3 + j))))
                    .store(acc + j);
            }
            for (; j < cols; ++j) {
                acc[j] += ((p0[j] + p1[j]) + (p2[j] + p3[j]));
            }
        }
        for (; s < n; ++s) {
            const Scalar* TP_RESTRICT row = in + (r0 + s) * cols;
            int64_t j = 0;
            for (; j + vs <= cols; j += vs) {
                (Vec::loadu(acc + j) + Vec::loadu(row + j)).store(acc + j);
            }
            for (; j < cols; ++j) {
                acc[j] += row[j];
            }
        }
    };

    tensorplay::parallel::parallel_for(
        0, nthreads, 1, [&](int64_t b, int64_t be) {
            (void)be;
            Scalar* TP_RESTRICT a0 = buf + b * tslice;
            const int64_t rb = b * grain;
            const int64_t re = std::min(rows, rb + grain);
            if (!cascade) {
                std::memset(a0, 0, cols * sizeof(Scalar));
                fold_rows(a0, rb, re - rb);
                return;
            }
            Scalar* TP_RESTRICT a1 = a0 + cols;
            Scalar* TP_RESTRICT a2 = a1 + cols;
            Scalar* TP_RESTRICT a3 = a2 + cols;
            std::memset(a0, 0, 4 * cols * sizeof(Scalar));
            const Vec zero(Scalar(0));
            // Fold one group of rows into level 0, then carry: level 1 takes
            // every group total, level 2 takes level 1 every 16 groups, level
            // 3 takes level 2 every 256 groups. A carry zeroes its donor, so
            // each level only ever holds whole groups of rows.
            int64_t r = rb;
            int64_t groups = 0;
            for (; r + group <= re; r += group) {
                fold_rows(a0, r, group);
                ++groups;
                for (int64_t j = 0; j + vs <= cols; j += vs) {
                    (Vec::loadu(a1 + j) + Vec::loadu(a0 + j)).store(a1 + j);
                    zero.store(a0 + j);
                }
                for (int64_t j = vec_cols; j < cols; ++j) {
                    a1[j] += a0[j];
                    a0[j] = Scalar(0);
                }
                if ((groups & 15) == 0) {
                    for (int64_t j = 0; j + vs <= cols; j += vs) {
                        (Vec::loadu(a2 + j) + Vec::loadu(a1 + j)).store(a2 + j);
                        zero.store(a1 + j);
                    }
                    for (int64_t j = vec_cols; j < cols; ++j) {
                        a2[j] += a1[j];
                        a1[j] = Scalar(0);
                    }
                    if ((groups & 255) == 0) {
                        for (int64_t j = 0; j + vs <= cols; j += vs) {
                            (Vec::loadu(a3 + j) + Vec::loadu(a2 + j))
                                .store(a3 + j);
                            zero.store(a2 + j);
                        }
                        for (int64_t j = vec_cols; j < cols; ++j) {
                            a3[j] += a2[j];
                            a2[j] = Scalar(0);
                        }
                    }
                }
            }
            fold_rows(a0, r, re - r);
            for (int64_t j = 0; j + vs <= cols; j += vs) {
                (Vec::loadu(a0 + j) + Vec::loadu(a1 + j) + Vec::loadu(a2 + j) +
                 Vec::loadu(a3 + j))
                    .store(a0 + j);
            }
            for (int64_t j = vec_cols; j < cols; ++j) {
                a0[j] += a1[j] + a2[j] + a3[j];
            }
        });
    tensorplay::parallel::parallel_for(0, cols, 256, [&](int64_t cb, int64_t ce) {
        int64_t j = cb;
        for (; j + vs <= ce; j += vs) {
            Vec a = Vec::loadu(buf + j);
            for (int64_t t = 1; t < nthreads; ++t) {
                a = a + Vec::loadu(buf + t * tslice + j);
            }
            a.store(outp + j);
        }
        for (; j < ce; ++j) {
            Scalar s = buf[j];
            for (int64_t t = 1; t < nthreads; ++t) {
                s += buf[t * tslice + j];
            }
            outp[j] = s;
        }
    });
    return out;
}

static std::optional<Tensor> try_sum_leading_rows_split(
    const Tensor& input, const std::vector<int64_t>& out_shape) {
    if (!input.is_contiguous() || input.dim() < 2 || input.numel() == 0) {
        return std::nullopt;
    }
    if (input.dtype() == DType::Float32) {
        return sum_leading_rows_split_impl<float>(input, out_shape);
    }
    if (input.dtype() == DType::Float64) {
        return sum_leading_rows_split_impl<double>(input, out_shape);
    }
    return std::nullopt;
}

__attribute__((target("avx512f")))
static void product_f32_leading_range_avx512(
    const float* input, float* output, int64_t rows, int64_t cols,
    int64_t begin, int64_t end) {
    int64_t col = begin;
    for (; col + 64 <= end; col += 64) {
        __m512 product0 = _mm512_set1_ps(1.0f);
        __m512 product1 = _mm512_set1_ps(1.0f);
        __m512 product2 = _mm512_set1_ps(1.0f);
        __m512 product3 = _mm512_set1_ps(1.0f);
        for (int64_t row = 0; row < rows; ++row) {
            const float* row_input = input + row * cols + col;
            product0 = _mm512_mul_ps(product0, _mm512_loadu_ps(row_input));
            product1 = _mm512_mul_ps(product1, _mm512_loadu_ps(row_input + 16));
            product2 = _mm512_mul_ps(product2, _mm512_loadu_ps(row_input + 32));
            product3 = _mm512_mul_ps(product3, _mm512_loadu_ps(row_input + 48));
        }
        _mm512_storeu_ps(output + col, product0);
        _mm512_storeu_ps(output + col + 16, product1);
        _mm512_storeu_ps(output + col + 32, product2);
        _mm512_storeu_ps(output + col + 48, product3);
    }
    for (; col + 16 <= end; col += 16) {
        __m512 product = _mm512_set1_ps(1.0f);
        for (int64_t row = 0; row < rows; ++row) {
            product = _mm512_mul_ps(
                product, _mm512_loadu_ps(input + row * cols + col));
        }
        _mm512_storeu_ps(output + col, product);
    }
    for (; col < end; ++col) {
        float product = 1.0f;
        for (int64_t row = 0; row < rows; ++row) {
            product *= input[row * cols + col];
        }
        output[col] = product;
    }
}

__attribute__((target("avx512f")))
static void product_f64_leading_range_avx512(
    const double* input, double* output, int64_t rows, int64_t cols,
    int64_t begin, int64_t end) {
    int64_t col = begin;
    for (; col + 32 <= end; col += 32) {
        __m512d product0 = _mm512_set1_pd(1.0);
        __m512d product1 = _mm512_set1_pd(1.0);
        __m512d product2 = _mm512_set1_pd(1.0);
        __m512d product3 = _mm512_set1_pd(1.0);
        for (int64_t row = 0; row < rows; ++row) {
            const double* row_input = input + row * cols + col;
            product0 = _mm512_mul_pd(product0, _mm512_loadu_pd(row_input));
            product1 = _mm512_mul_pd(product1, _mm512_loadu_pd(row_input + 8));
            product2 = _mm512_mul_pd(product2, _mm512_loadu_pd(row_input + 16));
            product3 = _mm512_mul_pd(product3, _mm512_loadu_pd(row_input + 24));
        }
        _mm512_storeu_pd(output + col, product0);
        _mm512_storeu_pd(output + col + 8, product1);
        _mm512_storeu_pd(output + col + 16, product2);
        _mm512_storeu_pd(output + col + 24, product3);
    }
    for (; col + 8 <= end; col += 8) {
        __m512d product = _mm512_set1_pd(1.0);
        for (int64_t row = 0; row < rows; ++row) {
            product = _mm512_mul_pd(
                product, _mm512_loadu_pd(input + row * cols + col));
        }
        _mm512_storeu_pd(output + col, product);
    }
    for (; col < end; ++col) {
        double product = 1.0;
        for (int64_t row = 0; row < rows; ++row) {
            product *= input[row * cols + col];
        }
        output[col] = product;
    }
}

static bool try_product_dim_real_avx512(
    const Tensor& input, Tensor& output, int64_t outer, int64_t d_size,
    int64_t inner) {
    if (!reduce_avx512_available() || !input.is_contiguous() ||
        input.numel() == 0 || input.dim() == 0 || outer <= 0 ||
        d_size <= 0 || inner < 1 ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64)) {
        return false;
    }
    const int64_t vector_width = input.dtype() == DType::Float32 ? 16 : 8;
    const int64_t vector_columns = vector_width * 4;
    if (inner < vector_width) return false;
    const int64_t approximate_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    const int64_t column_grain = std::max<int64_t>(
        vector_columns,
        ((approximate_grain + vector_columns - 1) / vector_columns) *
            vector_columns);
    const int64_t chunks_per_row =
        (inner + column_grain - 1) / column_grain;
    const int64_t task_count = outer * chunks_per_row;
    if (input.dtype() == DType::Float32) {
        const float* values = input.data_ptr<float>();
        float* results = output.data_ptr<float>();
        auto reduce_tasks = [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t row = task / chunks_per_row;
                const int64_t chunk = task - row * chunks_per_row;
                const int64_t col_begin = chunk * column_grain;
                const int64_t col_end = std::min(inner, col_begin + column_grain);
                product_f32_leading_range_avx512(
                    values + row * d_size * inner, results + row * inner,
                    d_size, inner, col_begin, col_end);
            }
        };
        if (input.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            reduce_tasks(0, task_count);
        } else {
            parallel_for(0, task_count, 1, reduce_tasks);
        }
    } else {
        const double* values = input.data_ptr<double>();
        double* results = output.data_ptr<double>();
        auto reduce_tasks = [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t row = task / chunks_per_row;
                const int64_t chunk = task - row * chunks_per_row;
                const int64_t col_begin = chunk * column_grain;
                const int64_t col_end = std::min(inner, col_begin + column_grain);
                product_f64_leading_range_avx512(
                    values + row * d_size * inner, results + row * inner,
                    d_size, inner, col_begin, col_end);
            }
        };
        if (input.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            reduce_tasks(0, task_count);
        } else {
            parallel_for(0, task_count, 1, reduce_tasks);
        }
    }
    return true;
}

template <bool IsMax>
__attribute__((target("avx512f")))
static void extremum_i32_dim_row_avx512(
    const int32_t* input, int32_t* output, int64_t* indices,
    int64_t d_size, int64_t inner, int64_t col_begin, int64_t col_end) {
    const __m512i zero_indices = _mm512_setzero_si512();
    int64_t col = col_begin;
    for (; col + 16 <= col_end; col += 16) {
        __m512i best = _mm512_loadu_si512(input + col);
        __m512i best_indices = zero_indices;
        for (int64_t row = 1; row < d_size; ++row) {
            const __m512i values = _mm512_loadu_si512(
                input + row * inner + col);
            const __mmask16 better = IsMax
                ? _mm512_cmp_epi32_mask(values, best, _MM_CMPINT_GT)
                : _mm512_cmp_epi32_mask(values, best, _MM_CMPINT_LT);
            best = _mm512_mask_blend_epi32(better, best, values);
            best_indices = _mm512_mask_blend_epi32(
                better, best_indices, _mm512_set1_epi32(static_cast<int>(row)));
        }
        _mm512_storeu_si512(output + col, best);
        alignas(64) int32_t index_buffer[16];
        _mm512_storeu_si512(index_buffer, best_indices);
        for (int lane = 0; lane < 16; ++lane) {
            indices[col + lane] = index_buffer[lane];
        }
    }
    for (; col < col_end; ++col) {
        int32_t best = input[col];
        int64_t best_index = 0;
        for (int64_t row = 1; row < d_size; ++row) {
            const int32_t value = input[row * inner + col];
            const bool better = IsMax ? value > best : value < best;
            if (better) {
                best = value;
                best_index = row;
            }
        }
        output[col] = best;
        indices[col] = best_index;
    }
}

template <bool IsMax>
__attribute__((target("avx512f")))
static void extremum_i64_dim_row_avx512(
    const int64_t* input, int64_t* output, int64_t* indices,
    int64_t d_size, int64_t inner, int64_t col_begin, int64_t col_end) {
    const __m512i zero_indices = _mm512_setzero_si512();
    int64_t col = col_begin;
    for (; col + 8 <= col_end; col += 8) {
        __m512i best = _mm512_loadu_si512(input + col);
        __m512i best_indices = zero_indices;
        for (int64_t row = 1; row < d_size; ++row) {
            const __m512i values = _mm512_loadu_si512(
                input + row * inner + col);
            const __mmask8 better = IsMax
                ? _mm512_cmp_epi64_mask(values, best, _MM_CMPINT_GT)
                : _mm512_cmp_epi64_mask(values, best, _MM_CMPINT_LT);
            best = _mm512_mask_blend_epi64(better, best, values);
            best_indices = _mm512_mask_blend_epi64(
                better, best_indices, _mm512_set1_epi64(row));
        }
        _mm512_storeu_si512(output + col, best);
        alignas(64) int64_t index_buffer[8];
        _mm512_storeu_si512(index_buffer, best_indices);
        for (int lane = 0; lane < 8; ++lane) {
            indices[col + lane] = index_buffer[lane];
        }
    }
    for (; col < col_end; ++col) {
        int64_t best = input[col];
        int64_t best_index = 0;
        for (int64_t row = 1; row < d_size; ++row) {
            const int64_t value = input[row * inner + col];
            const bool better = IsMax ? value > best : value < best;
            if (better) {
                best = value;
                best_index = row;
            }
        }
        output[col] = best;
        indices[col] = best_index;
    }
}

template <bool IsMax>
static bool try_extremum_integral_dim_avx512(
    const Tensor& input, Tensor& output, int64_t* indices, int64_t outer,
    int64_t d_size, int64_t inner) {
    if (!reduce_avx512_available() || !input.is_contiguous() ||
        input.numel() == 0 || input.dim() == 0 || outer <= 0 ||
        d_size <= 0 || inner < 1 ||
        (input.dtype() != DType::Int32 && input.dtype() != DType::Int64)) {
        return false;
    }
    const int64_t vector_width = input.dtype() == DType::Int32 ? 16 : 8;
    if (inner < vector_width) return false;
    const int64_t column_block = vector_width * 4;
    const int64_t approximate_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    const int64_t column_grain = std::max<int64_t>(
        column_block,
        ((approximate_grain + column_block - 1) / column_block) * column_block);
    const int64_t chunks_per_row =
        (inner + column_grain - 1) / column_grain;
    const int64_t task_count = outer * chunks_per_row;
    if (input.dtype() == DType::Int32) {
        const int32_t* values = input.data_ptr<int32_t>();
        int32_t* results = output.data_ptr<int32_t>();
        auto reduce_tasks = [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t row = task / chunks_per_row;
                const int64_t chunk = task - row * chunks_per_row;
                const int64_t col_begin = chunk * column_grain;
                const int64_t col_end = std::min(inner, col_begin + column_grain);
                extremum_i32_dim_row_avx512<IsMax>(
                    values + row * d_size * inner,
                    results + row * inner,
                    indices + row * inner,
                    d_size, inner, col_begin, col_end);
            }
        };
        if (input.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            reduce_tasks(0, task_count);
        } else {
            parallel_for(0, task_count, 1, reduce_tasks);
        }
    } else {
        const int64_t* values = input.data_ptr<int64_t>();
        int64_t* results = output.data_ptr<int64_t>();
        auto reduce_tasks = [&](int64_t begin, int64_t end) {
            for (int64_t task = begin; task < end; ++task) {
                const int64_t row = task / chunks_per_row;
                const int64_t chunk = task - row * chunks_per_row;
                const int64_t col_begin = chunk * column_grain;
                const int64_t col_end = std::min(inner, col_begin + column_grain);
                extremum_i64_dim_row_avx512<IsMax>(
                    values + row * d_size * inner,
                    results + row * inner,
                    indices + row * inner,
                    d_size, inner, col_begin, col_end);
            }
        };
        if (input.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            reduce_tasks(0, task_count);
        } else {
            parallel_for(0, task_count, 1, reduce_tasks);
        }
    }
    return true;
}

}  // namespace

// Full-tensor contiguous sum; returns false -> caller uses the iterator path.
static bool try_sum_real(const void* xv, int64_t n, DType dt, double* out) {
    if (n < 4096) return false;
    constexpr int64_t kGrain = 32768;
    // One slot per worker, indexed by the worker id: a chunk boundary is the
    // team's even split, not a multiple of the grain, so the chunk start is
    // not a slot number.
    const int64_t nslots = std::max<int64_t>(get_num_threads(), 1);
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        // The chunk sums are float, but the cross-worker merge is double: it
        // keeps the wide reduction from paying for its parallelism in
        // rounding.
        std::vector<double> part(nslots, 0.0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[get_thread_num()] += vec::sum(x, b, e);
        });
        double s = 0.0;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = s;
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        std::vector<double> part(nslots, 0.0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[get_thread_num()] += vec::sum(x, b, e);
        });
        double s = 0.0;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = s;
        return true;
    }
    return false;
}

__attribute__((target("avx512f")))
float max_f32_chunk_avx512(const float* x, int64_t b, int64_t e,
                           bool* has_nan) {
    __m512 a0 = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    __m512 a1 = a0;
    __m512 a2 = a0;
    __m512 a3 = a0;
    __mmask16 nan_mask = 0;
    int64_t i = b;
    for (; i + 64 <= e; i += 64) {
        const __m512 v0 = _mm512_loadu_ps(x + i);
        const __m512 v1 = _mm512_loadu_ps(x + i + 16);
        const __m512 v2 = _mm512_loadu_ps(x + i + 32);
        const __m512 v3 = _mm512_loadu_ps(x + i + 48);
        nan_mask |= _mm512_cmp_ps_mask(v0, v0, _CMP_UNORD_Q);
        nan_mask |= _mm512_cmp_ps_mask(v1, v1, _CMP_UNORD_Q);
        nan_mask |= _mm512_cmp_ps_mask(v2, v2, _CMP_UNORD_Q);
        nan_mask |= _mm512_cmp_ps_mask(v3, v3, _CMP_UNORD_Q);
        a0 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v0, a0, _CMP_GT_OQ), a0, v0);
        a1 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v1, a1, _CMP_GT_OQ), a1, v1);
        a2 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v2, a2, _CMP_GT_OQ), a2, v2);
        a3 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v3, a3, _CMP_GT_OQ), a3, v3);
    }
    for (; i + 16 <= e; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        nan_mask |= _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
        a0 = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v, a0, _CMP_GT_OQ), a0, v);
    }
    alignas(64) float lanes[16];
    _mm512_storeu_ps(lanes, _mm512_max_ps(_mm512_max_ps(a0, a1),
                                          _mm512_max_ps(a2, a3)));
    float best = lanes[0];
    for (int j = 1; j < 16; ++j) {
        if (lanes[j] > best) best = lanes[j];
    }
    for (; i < e; ++i) {
        if (std::isnan(x[i])) {
            nan_mask = 1;
        } else if (x[i] > best) {
            best = x[i];
        }
    }
    *has_nan = nan_mask != 0;
    return best;
}

__attribute__((target("avx512f")))
double max_f64_chunk_avx512(const double* x, int64_t b, int64_t e,
                            bool* has_nan) {
    __m512d a0 = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    __m512d a1 = a0;
    __mmask8 nan_mask = 0;
    int64_t i = b;
    for (; i + 16 <= e; i += 16) {
        const __m512d v0 = _mm512_loadu_pd(x + i);
        const __m512d v1 = _mm512_loadu_pd(x + i + 8);
        nan_mask |= _mm512_cmp_pd_mask(v0, v0, _CMP_UNORD_Q);
        nan_mask |= _mm512_cmp_pd_mask(v1, v1, _CMP_UNORD_Q);
        a0 = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(v0, a0, _CMP_GT_OQ), a0, v0);
        a1 = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(v1, a1, _CMP_GT_OQ), a1, v1);
    }
    for (; i + 8 <= e; i += 8) {
        const __m512d v = _mm512_loadu_pd(x + i);
        nan_mask |= _mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q);
        a0 = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(v, a0, _CMP_GT_OQ), a0, v);
    }
    alignas(64) double lanes[8];
    _mm512_storeu_pd(lanes, _mm512_max_pd(a0, a1));
    double best = lanes[0];
    for (int j = 1; j < 8; ++j) {
        if (lanes[j] > best) best = lanes[j];
    }
    for (; i < e; ++i) {
        if (std::isnan(x[i])) {
            nan_mask = 1;
        } else if (x[i] > best) {
            best = x[i];
        }
    }
    *has_nan = nan_mask != 0;
    return best;
}

__attribute__((target("avx2")))
float max_f32_chunk_avx2(const float* x, int64_t b, int64_t e, bool* has_nan) {
    __m256 a0 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256 a1 = a0;
    int nan_mask = 0;
    int64_t i = b;
    for (; i + 16 <= e; i += 16) {
        const __m256 v0 = _mm256_loadu_ps(x + i);
        const __m256 v1 = _mm256_loadu_ps(x + i + 8);
        nan_mask |= _mm256_movemask_ps(_mm256_cmp_ps(v0, v0, _CMP_UNORD_Q));
        nan_mask |= _mm256_movemask_ps(_mm256_cmp_ps(v1, v1, _CMP_UNORD_Q));
        // NaN lanes keep the accumulator: an Inf can never beat a seen max.
        a0 = _mm256_max_ps(a0, _mm256_blendv_ps(
            _mm256_set1_ps(-std::numeric_limits<float>::infinity()), v0,
            _mm256_cmp_ps(v0, v0, _CMP_ORD_Q)));
        a1 = _mm256_max_ps(a1, _mm256_blendv_ps(
            _mm256_set1_ps(-std::numeric_limits<float>::infinity()), v1,
            _mm256_cmp_ps(v1, v1, _CMP_ORD_Q)));
    }
    alignas(32) float lanes[8];
    _mm256_storeu_ps(lanes, _mm256_max_ps(a0, a1));
    float best = lanes[0];
    for (int j = 1; j < 8; ++j) {
        if (lanes[j] > best) best = lanes[j];
    }
    for (; i < e; ++i) {
        if (std::isnan(x[i])) {
            nan_mask = 1;
        } else if (x[i] > best) {
            best = x[i];
        }
    }
    *has_nan = nan_mask != 0;
    return best;
}

__attribute__((target("avx2")))
double max_f64_chunk_avx2(const double* x, int64_t b, int64_t e, bool* has_nan) {
    __m256d a0 = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    __m256d a1 = a0;
    int64_t nan_mask = 0;
    int64_t i = b;
    for (; i + 8 <= e; i += 8) {
        const __m256d v0 = _mm256_loadu_pd(x + i);
        nan_mask |= _mm256_movemask_pd(_mm256_cmp_pd(v0, v0, _CMP_UNORD_Q));
        a0 = _mm256_max_pd(a0, _mm256_blendv_pd(
            _mm256_set1_pd(-std::numeric_limits<double>::infinity()), v0,
            _mm256_cmp_pd(v0, v0, _CMP_ORD_Q)));
    }
    alignas(32) double lanes[4];
    _mm256_storeu_pd(lanes, _mm256_max_pd(a0, a1));
    double best = lanes[0];
    for (int j = 1; j < 4; ++j) {
        if (lanes[j] > best) best = lanes[j];
    }
    for (; i < e; ++i) {
        if (std::isnan(x[i])) {
            nan_mask = 1;
        } else if (x[i] > best) {
            best = x[i];
        }
    }
    *has_nan = nan_mask != 0;
    return best;
}

// The AVX2 tier of the max reduction: same shape and NaN semantics as the
// AVX512 one, for hosts whose lanes stop at 256 bits.  Small sizes answer
// serially — the partials plumbing costs more than the walk itself.
static bool try_max_real_avx2(const void* xv, int64_t n, DType dt,
                              double* out, bool* has_nan) {
    if (!reduce_avx2_available() || n < 512) return false;
    constexpr int64_t kGrain = 32768;
    const int64_t nslots = (n + kGrain - 1) / kGrain;
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        std::vector<float> part(nslots, -std::numeric_limits<float>::infinity());
        std::vector<uint8_t> nan(nslots, 0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            bool local_nan = false;
            part[b / kGrain] = max_f32_chunk_avx2(x, b, e, &local_nan);
            nan[b / kGrain] = static_cast<uint8_t>(local_nan);
        });
        float best = part[0];
        bool any_nan = nan[0] != 0;
        for (int64_t k = 1; k < nslots; ++k) {
            any_nan = any_nan || nan[k] != 0;
            if (part[k] > best) best = part[k];
        }
        *out = static_cast<double>(best);
        *has_nan = any_nan;
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        std::vector<double> part(nslots, -std::numeric_limits<double>::infinity());
        std::vector<uint8_t> nan(nslots, 0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            bool local_nan = false;
            part[b / kGrain] = max_f64_chunk_avx2(x, b, e, &local_nan);
            nan[b / kGrain] = static_cast<uint8_t>(local_nan);
        });
        double best = part[0];
        bool any_nan = nan[0] != 0;
        for (int64_t k = 1; k < nslots; ++k) {
            any_nan = any_nan || nan[k] != 0;
            if (part[k] > best) best = part[k];
        }
        *out = best;
        *has_nan = any_nan;
        return true;
    }
    return false;
}

static bool try_max_real_avx512(const void* xv, int64_t n, DType dt,
                                double* out, bool* has_nan) {
    if (!reduce_avx512_available() || n < 4096) return false;    constexpr int64_t kGrain = 32768;
    const int64_t nslots = (n + kGrain - 1) / kGrain;
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        std::vector<float> part(nslots, -std::numeric_limits<float>::infinity());
        std::vector<uint8_t> nan(nslots, 0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            bool local_nan = false;
            part[b / kGrain] = max_f32_chunk_avx512(x, b, e, &local_nan);
            nan[b / kGrain] = static_cast<uint8_t>(local_nan);
        });
        float best = part[0];
        bool any_nan = nan[0] != 0;
        for (int64_t k = 1; k < nslots; ++k) {
            any_nan = any_nan || nan[k] != 0;
            if (part[k] > best) best = part[k];
        }
        *out = static_cast<double>(best);
        *has_nan = any_nan;
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        std::vector<double> part(nslots, -std::numeric_limits<double>::infinity());
        std::vector<uint8_t> nan(nslots, 0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            bool local_nan = false;
            part[b / kGrain] = max_f64_chunk_avx512(x, b, e, &local_nan);
            nan[b / kGrain] = static_cast<uint8_t>(local_nan);
        });
        double best = part[0];
        bool any_nan = nan[0] != 0;
        for (int64_t k = 1; k < nslots; ++k) {
            any_nan = any_nan || nan[k] != 0;
            if (part[k] > best) best = part[k];
        }
        *out = best;
        *has_nan = any_nan;
        return true;
    }
    return false;
}
#endif  // __x86_64__

// TensorIterator-based reduction kernel: adds one input element into the
// output elementwise (out = out + in), vectorized over 4 accumulators.
// (the input is pre-cast to out_dtype by the caller).
static void sum_kernel_iter(TensorIteratorBase& iter) {
#define OP_CASE(ctype, name) \
    case DType::name: { \
        binary_kernel_reduce_vec(iter, \
            [=](ctype a, ctype b) -> ctype { return a + b; }, \
            [=](Vectorized<ctype> a, Vectorized<ctype> b) { return a + b; }); \
        break; \
    }
    switch (iter.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        case DType::ComplexFloat:
            binary_kernel_reduce(iter, CxSumOps<complex<float>>{}, complex<float>(0));
            break;
        case DType::ComplexDouble:
            binary_kernel_reduce(iter, CxSumOps<complex<double>>{}, complex<double>(0));
            break;
        case DType::ComplexHalf:
        case DType::BComplex32:
            // Reduced complexes accumulate in complex64 (opmath rule); the
            // caller pre-casts the input to the acc dtype.
            binary_kernel_reduce(iter, CxSumOps<complex<float>>{}, complex<float>(0));
            break;
        default: TP_THROW(NotImplementedError, "sum not implemented for this dtype");
    }
    #undef OP_CASE
}

template <typename ctype>
static void product_kernel_vec(TensorIteratorBase& iter) {
    binary_kernel_reduce_vec(iter,
        [](ctype a, ctype b) -> ctype { return a * b; },
        [](Vectorized<ctype> a, Vectorized<ctype> b) { return a * b; },
        1);
}

static bool try_product_kernel_vec(TensorIteratorBase& iter) {
    switch (iter.dtype()) {
        case DType::Int32:
            product_kernel_vec<int32_t>(iter);
            return true;
        case DType::Int64:
            product_kernel_vec<int64_t>(iter);
            return true;
        case DType::Float32:
            product_kernel_vec<float>(iter);
            return true;
        case DType::Float64:
            product_kernel_vec<double>(iter);
            return true;
        case DType::Float16:
            // The reduced widths multiply through single precision and keep
            // the product in their own width, one vector at a time.
            product_kernel_vec<tensorplay::Half>(iter);
            return true;
        case DType::BFloat16:
            product_kernel_vec<tensorplay::BFloat16>(iter);
            return true;
        default:
            return false;
    }
}

static bool should_use_acc_buffer(const TensorIteratorBase& iter) {
    if (iter.noutputs() != 1 ||
        !isReducedFloatingType(iter.common_dtype()) ||
        iter.ndim() < 2) {
        return false;
    }
    const auto& output_strides = iter.strides(0);
    return output_strides[0] == 0 && output_strides[1] == 0;
}

Tensor sum_kernel_impl(const Tensor& self, DType dtype);
Tensor sum_dim_kernel_impl(const Tensor& self,
                           const std::vector<int64_t>& dims,
                           bool keepdim,
                           DType dtype);

static Tensor sum_to_float(const Tensor& self,
                           const std::vector<int64_t>& dims,
                           bool keepdim) {
    Tensor input = self.dtype() == DType::Float32
        ? self
        : self.to(DType::Float32);
    if (dims.empty()) {
        return sum_kernel_impl(input, DType::Float32);
    }
    return sum_dim_kernel_impl(input, dims, keepdim, DType::Float32);
}

Tensor sum_kernel_impl(const Tensor& self, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }

    // reduced complexes accumulate in complex64.
    DType acc_dtype = out_dtype;
    if (isReducedFloatingType(out_dtype)) {
        acc_dtype = DType::Float32;
    } else if (out_dtype == DType::ComplexHalf || out_dtype == DType::BComplex32) {
        acc_dtype = DType::ComplexFloat;
    }

    Tensor out = Tensor::zeros({}, out_dtype, self.device());

    Tensor input = self;
    if (self.dtype() != out_dtype) {
        input = self.to(out_dtype);
    }

    // try_sum_real is defined in the x86 section above; the other
    // architectures reach the iterator below.
#if defined(__x86_64__)
    if ((acc_dtype == DType::Float32 || acc_dtype == DType::Float64) &&
        input.dtype() == acc_dtype && input.is_contiguous() &&
        input.numel() > 0) {
        double s = 0.0;
        if (try_sum_real(input.data_ptr(), input.numel(), acc_dtype, &s)) {
            if (acc_dtype == DType::Float32) {
                *out.data_ptr<float>() = static_cast<float>(s);
            } else {
                *out.data_ptr<double>() = s;
            }
            return out;
        }
    }
#endif

    if (acc_dtype == DType::Float32 || acc_dtype == DType::Float64) {
        TensorIterator iter = TensorIterator::reduce_op(out, input);
        if (iter.numel() != 0 && should_use_acc_buffer(iter)) {
            Tensor tmp = sum_to_float(self, {}, false);
            out.copy_(tmp);
            return out;
        }
        if (out_dtype == DType::Float16) {
            sum_detail::cascade_sum<float, Half>(iter);
        } else if (out_dtype == DType::BFloat16) {
            sum_detail::cascade_sum<float, BFloat16>(iter);
        } else if (out_dtype == DType::Float32) {
            sum_detail::cascade_sum<float>(iter);
        } else {
            sum_detail::cascade_sum<double>(iter);
        }
        return out;
    }

    // AVX2 complex full-reduction fast path (cpu/VecComplex.h): two vector
    // accumulators + per-chunk partials; the iterator path below reduces
    // complex scalars one element at a time.
    if ((acc_dtype == DType::ComplexFloat || acc_dtype == DType::ComplexDouble) &&
        input.is_contiguous() && input.numel() > 0) {
        double re = 0.0, im = 0.0;
        if (veccomplex::try_sum(input.data_ptr(), input.numel(), acc_dtype,
                                &re, &im)) {
            Tensor out = Tensor::zeros({}, acc_dtype, self.device());
            if (acc_dtype == DType::ComplexFloat) {
                *out.data_ptr<complex<float>>() = complex<float>(
                    static_cast<float>(re), static_cast<float>(im));
            } else {
                *out.data_ptr<complex<double>>() = complex<double>(re, im);
            }
            return acc_dtype == out_dtype ? out : out.to(out_dtype);
        }
    }

    TensorIterator iter = TensorIterator::reduce_op(out, input);
    sum_kernel_iter(iter);

    return acc_dtype == out_dtype ? out : out.to(out_dtype);
}

// A tensor that lies dense in memory in an order other than the one its shape
// gives -- a transpose, a permuted view -- is the same buffer read in another
// order.  The axes are put in the order their strides give, which is the dense
// order, the reduced axes are named there, and the answer is taken back to the
// order that was asked for.  The walk then runs on a dense tensor, where every
// fold below asks to be, instead of one element at a time through the strides.
// The reduced axes are kept while the walk runs and dropped afterwards, so the
// shape bookkeeping is one permutation of the answer either way.
template <typename Run>
inline std::optional<Tensor> review_permuted_reduction(
        const Tensor& self, const std::vector<int64_t>& dims, bool keepdim,
        Run&& run) {
    const int64_t ndim = self.dim();
    if (ndim < 2 || self.is_contiguous()) return std::nullopt;
    if (self.numel() == 0) return std::nullopt;
    const std::vector<int64_t> stride =
        static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> order(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) { return stride[a] > stride[b]; });
    Tensor dense = self.permute(order);
    if (!dense.is_contiguous()) return std::nullopt;

    std::vector<int64_t> place(static_cast<size_t>(ndim), 0);
    for (int64_t k = 0; k < ndim; ++k) {
        place[static_cast<size_t>(order[static_cast<size_t>(k)])] = k;
    }
    std::vector<bool> reduced(static_cast<size_t>(ndim), false);
    std::vector<int64_t> dense_dims;
    for (int64_t d : dims) {
        int64_t rank = d < 0 ? d + ndim : d;
        if (rank < 0 || rank >= ndim || reduced[static_cast<size_t>(rank)]) {
            return std::nullopt;
        }
        reduced[static_cast<size_t>(rank)] = true;
        dense_dims.push_back(place[static_cast<size_t>(rank)]);
    }
    if (dense_dims.empty()) return std::nullopt;

    Tensor held = run(dense, dense_dims);
    // The kept dimensions in the order the dense layout holds them.  When they
    // already run in the original order the answer is the dense result re-cut
    // to the shape the caller asked for: no second walk of the memory to put
    // the axes back, and no copy to make the result contiguous.
    std::vector<int64_t> kept;
    kept.reserve(static_cast<size_t>(ndim));
    for (int64_t k = 0; k < ndim; ++k) {
        const int64_t rank = order[static_cast<size_t>(k)];
        if (!reduced[static_cast<size_t>(rank)]) {
            kept.push_back(rank);
        }
    }
    if (std::is_sorted(kept.begin(), kept.end())) {
        std::vector<int64_t> shape;
        shape.reserve(kept.size());
        for (int64_t i = 0; i < ndim; ++i) {
            const bool gone = reduced[static_cast<size_t>(i)];
            if (gone && !keepdim) continue;
            shape.push_back(gone ? 1 : self.size(i));
        }
        return held.reshape(shape);
    }
    std::vector<int64_t> back_order(static_cast<size_t>(ndim), 0);
    for (int64_t k = 0; k < ndim; ++k) {
        back_order[static_cast<size_t>(order[static_cast<size_t>(k)])] = k;
    }
    Tensor back = held.permute(back_order);
    if (!keepdim) {
        std::vector<int64_t> shape;
        for (int64_t i = 0; i < ndim; ++i) {
            if (!reduced[static_cast<size_t>(i)]) {
                shape.push_back(self.size(i));
            }
        }
        back = back.reshape(shape);
    }
    return back.contiguous();
}

#if defined(TP_VECREDUCED_X86)
// Four answers of a band summed into the wider integer the total is kept in.
__attribute__((target("avx2"), always_inline)) inline
int64_t sum_band_i64(__m256i v) {
    const __m128i pair = _mm_add_epi64(_mm256_castsi256_si128(v),
                                       _mm256_extracti128_si256(v, 1));
    return static_cast<int64_t>(_mm_cvtsi128_si64(pair) +
                                _mm_extract_epi64(pair, 1));
}

// A row of integers summed into the wider integer its answer is kept in.  A
// band of the row is widened once and kept in registers, so every element is
// read exactly once and the answer is formed without a converted copy of the
// row that is eight times its size.  A partial sum in the narrow width is
// folded into the wide one before it can overflow, so every answer is exact.
template <typename T>
__attribute__((target("avx2"), always_inline)) inline
int64_t sum_row_int(const T* x, int64_t n) {
    __m256i wide = _mm256_setzero_si256();
    int64_t i = 0;
    const __m256i one16 = _mm256_set1_epi16(1);
    if constexpr (sizeof(T) == 1) {
        if constexpr (std::is_unsigned_v<T>) {
            // A byte band folds eight bytes into each lane of the wide total
            // in one step.
            const __m256i zero = _mm256_setzero_si256();
            for (; i + 32 <= n; i += 32) {
                wide = _mm256_add_epi64(
                    wide,
                    _mm256_sad_epu8(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(x + i)),
                        zero));
            }
        } else {
            const __m256i one8 = _mm256_set1_epi8(1);
            __m256i part = _mm256_setzero_si256();
            for (; i + 32 <= n; i += 32) {
                const __m256i band = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(
                        one8, _mm256_loadu_si256(
                                  reinterpret_cast<const __m256i*>(x + i))),
                    one16);
                part = _mm256_add_epi32(part, band);
                if (((i + 32) & 131071) == 0) {
                    wide = _mm256_add_epi64(
                        wide, _mm256_cvtepi32_epi64(
                                  _mm256_castsi256_si128(part)));
                    wide = _mm256_add_epi64(
                        wide, _mm256_cvtepi32_epi64(
                                  _mm256_extracti128_si256(part, 1)));
                    part = _mm256_setzero_si256();
                }
            }
            wide = _mm256_add_epi64(
                wide, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(part)));
            wide = _mm256_add_epi64(
                wide,
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(part, 1)));
        }
    } else if constexpr (sizeof(T) == 2) {
        if constexpr (std::is_unsigned_v<T>) {
            __m256i part = _mm256_setzero_si256();
            for (; i + 16 <= n; i += 16) {
                const __m128i half0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(x + i));
                const __m128i half1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(x + i + 8));
                part = _mm256_add_epi32(
                    part, _mm256_add_epi32(_mm256_cvtepu16_epi32(half0),
                                           _mm256_cvtepu16_epi32(half1)));
                if (((i + 16) & 65535) == 0) {
                    wide = _mm256_add_epi64(
                        wide, _mm256_cvtepi32_epi64(
                                  _mm256_castsi256_si128(part)));
                    wide = _mm256_add_epi64(
                        wide, _mm256_cvtepi32_epi64(
                                  _mm256_extracti128_si256(part, 1)));
                    part = _mm256_setzero_si256();
                }
            }
            wide = _mm256_add_epi64(
                wide, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(part)));
            wide = _mm256_add_epi64(
                wide,
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(part, 1)));
        } else {
            __m256i part = _mm256_setzero_si256();
            for (; i + 16 <= n; i += 16) {
                const __m256i band = _mm256_madd_epi16(
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(x + i)),
                    one16);
                part = _mm256_add_epi32(part, band);
                if (((i + 16) & 262143) == 0) {
                    wide = _mm256_add_epi64(
                        wide, _mm256_cvtepi32_epi64(
                                  _mm256_castsi256_si128(part)));
                    wide = _mm256_add_epi64(
                        wide, _mm256_cvtepi32_epi64(
                                  _mm256_extracti128_si256(part, 1)));
                    part = _mm256_setzero_si256();
                }
            }
            wide = _mm256_add_epi64(
                wide, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(part)));
            wide = _mm256_add_epi64(
                wide,
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(part, 1)));
        }
    } else if constexpr (sizeof(T) == 4) {
        // A half-width band is sign or zero extended once and carried in the
        // wide total directly.
        for (; i + 8 <= n; i += 8) {
            const __m256i band = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(x + i));
            __m256i low;
            __m256i high;
            if constexpr (std::is_unsigned_v<T>) {
                low = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(band));
                high = _mm256_cvtepu32_epi64(
                    _mm256_extracti128_si256(band, 1));
            } else {
                low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(band));
                high = _mm256_cvtepi32_epi64(
                    _mm256_extracti128_si256(band, 1));
            }
            wide = _mm256_add_epi64(wide, _mm256_add_epi64(low, high));
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            wide = _mm256_add_epi64(
                wide,
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(x + i)));
        }
    }
    int64_t total = sum_band_i64(wide);
    for (; i < n; ++i) total += static_cast<int64_t>(x[i]);
    return total;
}

// A band of columns summed down the leading axis of an integral tensor.  The
// totals are kept in the wide integer, and a total in the narrow width is
// folded into them before it could overflow, so every answer is exact however
// many rows are summed.  A whole band stays in two registers across the rows,
// so each element is read exactly once and no converted copy is made.
template <typename T, int64_t Step>
__attribute__((target("avx2"), always_inline)) inline
void sum_cols_int_band(const T* in, int64_t rows, int64_t cols, int64_t c,
                       int64_t* out) {
    __m256i lo = _mm256_setzero_si256();
    __m256i hi = _mm256_setzero_si256();
    if constexpr (sizeof(T) <= 2) {
        // A band of eight columns is widened once per row; the narrow total is
        // folded into the wide pair every few thousand rows.
        constexpr int64_t kBlock = 2048;
        for (int64_t r0 = 0; r0 < rows; r0 += kBlock) {
            const int64_t r1 = std::min(rows, r0 + kBlock);
            __m256i acc = _mm256_setzero_si256();
            for (int64_t r = r0; r < r1; ++r) {
                const T* p = in + r * cols + c;
                __m256i band;
                if constexpr (sizeof(T) == 1) {
                    const __m128i raw = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(p));
                    if constexpr (std::is_unsigned_v<T>) {
                        band = _mm256_cvtepu8_epi32(raw);
                    } else {
                        band = _mm256_cvtepi8_epi32(raw);
                    }
                } else {
                    const __m128i raw = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(p));
                    if constexpr (std::is_unsigned_v<T>) {
                        band = _mm256_cvtepu16_epi32(raw);
                    } else {
                        band = _mm256_cvtepi16_epi32(raw);
                    }
                }
                acc = _mm256_add_epi32(acc, band);
            }
            lo = _mm256_add_epi64(
                lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc)));
            hi = _mm256_add_epi64(
                hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc, 1)));
        }
    } else if constexpr (sizeof(T) == 4) {
        for (int64_t r = 0; r < rows; ++r) {
            const __m256i band = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(in + r * cols + c));
            if constexpr (std::is_unsigned_v<T>) {
                lo = _mm256_add_epi64(
                    lo, _mm256_cvtepu32_epi64(
                            _mm256_castsi256_si128(band)));
                hi = _mm256_add_epi64(
                    hi, _mm256_cvtepu32_epi64(
                            _mm256_extracti128_si256(band, 1)));
            } else {
                lo = _mm256_add_epi64(
                    lo, _mm256_cvtepi32_epi64(
                            _mm256_castsi256_si128(band)));
                hi = _mm256_add_epi64(
                    hi, _mm256_cvtepi32_epi64(
                            _mm256_extracti128_si256(band, 1)));
            }
        }
    } else {
        // A width that already fills a lane is carried sixteen columns at a
        // time, four chains of four lanes: the band of a row is contiguous, so
        // a row is read in two whole cache lines rather than one fragment of
        // eight, and the four chains keep four adds in flight over the rows.
        static_assert(Step == 16);
        __m256i a0 = _mm256_setzero_si256();
        __m256i a1 = a0, a2 = a0, a3 = a0;
        for (int64_t r = 0; r < rows; ++r) {
            const __m256i* p = reinterpret_cast<const __m256i*>(
                in + r * cols + c);
            a0 = _mm256_add_epi64(a0, _mm256_loadu_si256(p));
            a1 = _mm256_add_epi64(a1, _mm256_loadu_si256(p + 1));
            a2 = _mm256_add_epi64(a2, _mm256_loadu_si256(p + 2));
            a3 = _mm256_add_epi64(a3, _mm256_loadu_si256(p + 3));
        }
        alignas(32) int64_t sixteen[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(sixteen), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sixteen + 4), a1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sixteen + 8), a2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(sixteen + 12), a3);
        for (int64_t k = 0; k < Step; ++k) out[c + k] = sixteen[k];
        return;
    }
    alignas(32) int64_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 4), hi);
    for (int64_t k = 0; k < Step; ++k) out[c + k] = lanes[k];
}

// One span of column bands, so a worker's share is a single call.
template <typename T>
__attribute__((target("avx2"), noinline))
void sum_cols_int_span(const T* in, int64_t rows, int64_t cols, int64_t begin,
                       int64_t end, int64_t* out) {
    int64_t c = begin;
    if constexpr (sizeof(T) == 8) {
        for (; c + 16 <= end; c += 16) {
            sum_cols_int_band<T, 16>(in, rows, cols, c, out);
        }
        // The columns past the last whole band are answered eight at a time
        // and then one at a time.
        for (; c + 4 <= end; c += 4) {
            __m256i lo = _mm256_setzero_si256();
            for (int64_t r = 0; r < rows; ++r) {
                lo = _mm256_add_epi64(
                    lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                            in + r * cols + c)));
            }
            alignas(32) int64_t lanes[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), lo);
            for (int64_t k = 0; k < 4; ++k) out[c + k] = lanes[k];
        }
    } else {
        for (; c + 8 <= end; c += 8) {
            sum_cols_int_band<T, 8>(in, rows, cols, c, out);
        }
    }
    for (; c < end; ++c) {
        int64_t total = 0;
        for (int64_t r = 0; r < rows; ++r) {
            total += static_cast<int64_t>(in[r * cols + c]);
        }
        out[c] = total;
    }
}

template <typename T>
static void sum_cols_int(const Tensor& input, int64_t* outp, int64_t rows,
                         int64_t cols, int64_t grain) {
    const T* in = reinterpret_cast<const T*>(input.data_ptr());
    const int64_t bands = sizeof(T) == 8 ? 16 : 8;
    const int64_t grain_cols = std::max<int64_t>(1, grain / std::max<int64_t>(rows, 1) / bands * bands);
    parallel_for(0, cols, grain_cols, [&](int64_t b, int64_t e) {
        sum_cols_int_span<T>(in, rows, cols, b, e, outp);
    });
}

// The leading axis of an integral tensor summed into the wide integer its
// answer is kept in.
inline std::optional<Tensor> try_sum_leading_int(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    const DType dt = input.dtype();
    if (!input.is_contiguous() || !isIntegralType(dt, true)) {
        return std::nullopt;
    }
    if (input.dim() < 1) return std::nullopt;
    const int64_t rows = input.size(0);
    const int64_t cols = rows > 0 ? input.numel() / rows : 0;
    if (rows < 2 || cols < 1) return std::nullopt;
    // A band of columns is carried across every row, so one pass over the rows
    // touches one cache line of every row.  While those lines stay in cache
    // every element is still read exactly once; once the rows no longer fit,
    // each pass reads a line to use a fraction of it, and the walk that streams
    // the input once answers faster.
    if (rows * 64 > 4LL * 1024 * 1024) return std::nullopt;
    Tensor out = Tensor::empty(out_shape, DType::Int64, input.device());
    int64_t* outp = out.data_ptr<int64_t>();
    const int64_t grain = GRAIN_SIZE;
    switch (dt) {
        case DType::UInt8:
        case DType::Bool:
            sum_cols_int<uint8_t>(input, outp, rows, cols, grain);
            break;
        case DType::Int8:
            sum_cols_int<int8_t>(input, outp, rows, cols, grain);
            break;
        case DType::Int16:
            sum_cols_int<int16_t>(input, outp, rows, cols, grain);
            break;
        case DType::UInt16:
            sum_cols_int<uint16_t>(input, outp, rows, cols, grain);
            break;
        case DType::Int32:
            sum_cols_int<int32_t>(input, outp, rows, cols, grain);
            break;
        case DType::UInt32:
            sum_cols_int<uint32_t>(input, outp, rows, cols, grain);
            break;
        case DType::Int64:
            sum_cols_int<int64_t>(input, outp, rows, cols, grain);
            break;
        case DType::UInt64:
            sum_cols_int<uint64_t>(input, outp, rows, cols, grain);
            break;
        default:
            return std::nullopt;
    }
    return out;
}

// Every row of an integral tensor summed the same way, split among the workers.
template <typename T>
__attribute__((target("avx2"), noinline))
static void sum_rows_int_span(const T* in, int64_t* outp, int64_t begin,
                              int64_t end, int64_t d_size) {
    for (int64_t r = begin; r < end; ++r) {
        outp[r] = sum_row_int<T>(in + r * d_size, d_size);
    }
}

template <typename T>
static void sum_rows_int(const Tensor& input, int64_t* outp, int64_t rows,
                         int64_t d_size, int64_t grain) {
    const T* in = reinterpret_cast<const T*>(input.data_ptr());
    parallel_for(0, rows, grain, [&](int64_t b, int64_t e) {
        sum_rows_int_span<T>(in, outp, b, e, d_size);
    });
}

// The last axis of an integral tensor summed into the wide integer its answer
// is kept in: every row is one contiguous pass, so it is summed where it lies.
inline std::optional<Tensor> try_sum_lastdim_int(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    const DType dt = input.dtype();
    if (!input.is_contiguous() || !isIntegralType(dt, true)) {
        return std::nullopt;
    }
    if (input.dim() < 1) return std::nullopt;
    const int64_t d_size = input.size(input.dim() - 1);
    if (d_size < 1 || input.numel() == 0) return std::nullopt;
    const int64_t rows = input.numel() / d_size;
    Tensor out = Tensor::empty(out_shape, DType::Int64, input.device());
    int64_t* outp = out.data_ptr<int64_t>();
    const int64_t row_grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    switch (dt) {
        case DType::UInt8:
        case DType::Bool:
            sum_rows_int<uint8_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::Int8:
            sum_rows_int<int8_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::Int16:
            sum_rows_int<int16_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::UInt16:
            sum_rows_int<uint16_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::Int32:
            sum_rows_int<int32_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::UInt32:
            sum_rows_int<uint32_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::Int64:
            sum_rows_int<int64_t>(input, outp, rows, d_size, row_grain);
            break;
        case DType::UInt64:
            sum_rows_int<uint64_t>(input, outp, rows, d_size, row_grain);
            break;
        default:
            return std::nullopt;
    }
    return out;
}

// A middle axis of an integral tensor summed into the wide integer its answer
// is kept in.  A block of outer indices is a contiguous plane of rows, so each
// plane is summed where it lies, one band of columns at a time, instead of
// through a converted copy of the whole input.
template <typename T>
__attribute__((target("avx2"), noinline))
static void sum_middle_int_span(const T* in, int64_t* outp, int64_t red,
                                int64_t inner, int64_t begin, int64_t end) {
    for (int64_t o = begin; o < end; ++o) {
        sum_cols_int_span<T>(in + o * red * inner, red, inner, 0, inner,
                             outp + o * inner);
    }
}

template <typename T>
static void sum_middle_int(const Tensor& input, int64_t* outp, int64_t outer,
                           int64_t red, int64_t inner) {
    const T* in = reinterpret_cast<const T*>(input.data_ptr());
    const int64_t grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(red * inner, 1));
    if (get_num_threads() == 1 || in_parallel_region() ||
        outer * red * inner < GRAIN_SIZE) {
        sum_middle_int_span<T>(in, outp, red, inner, 0, outer);
        return;
    }
    parallel_for(0, outer, grain, [&](int64_t b, int64_t e) {
        sum_middle_int_span<T>(in, outp, red, inner, b, e);
    });
}

// Guards: a band pass touches one cache line of every row of the plane, so a
// plane that stays in the first-level cache pays that traffic once.  Beyond it,
// only a plane whose rows already fill a cache line is summed this way, so every
// pass consumes whole lines instead of re-reading them for a fragment.
inline std::optional<Tensor> try_sum_middle_int(
        const Tensor& input, const std::vector<int64_t>& out_shape,
        int64_t axis) {
    const DType dt = input.dtype();
    if (!input.is_contiguous() || !isIntegralType(dt, true)) {
        return std::nullopt;
    }
    const int64_t rank = input.dim();
    if (rank < 3 || axis <= 0 || axis >= rank - 1) return std::nullopt;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= input.size(i);
    for (int64_t i = axis + 1; i < rank; ++i) inner *= input.size(i);
    const int64_t red = input.size(axis);
    if (outer < 1 || red < 1 || inner < 1) return std::nullopt;
    const int64_t itemsize = static_cast<int64_t>(elementSize(dt));
    if (red * inner * itemsize > 32768 && inner * itemsize < 64) {
        return std::nullopt;
    }
    Tensor out = Tensor::empty(out_shape, DType::Int64, input.device());
    int64_t* outp = out.data_ptr<int64_t>();
    switch (dt) {
        case DType::UInt8:
        case DType::Bool:
            sum_middle_int<uint8_t>(input, outp, outer, red, inner);
            break;
        case DType::Int8:
            sum_middle_int<int8_t>(input, outp, outer, red, inner);
            break;
        case DType::Int16:
            sum_middle_int<int16_t>(input, outp, outer, red, inner);
            break;
        case DType::UInt16:
            sum_middle_int<uint16_t>(input, outp, outer, red, inner);
            break;
        case DType::Int32:
            sum_middle_int<int32_t>(input, outp, outer, red, inner);
            break;
        case DType::UInt32:
            sum_middle_int<uint32_t>(input, outp, outer, red, inner);
            break;
        case DType::Int64:
            sum_middle_int<int64_t>(input, outp, outer, red, inner);
            break;
        case DType::UInt64:
            sum_middle_int<uint64_t>(input, outp, outer, red, inner);
            break;
        default:
            return std::nullopt;
    }
    return out;
}

#endif  // TP_VECREDUCED_X86

#if defined(TP_VECREDUCED_X86)
// Four factors of one row widened into the wide integer, so that a band of
// columns is multiplied in the width its answer is kept in.
template <typename T>
__attribute__((target("avx2"), always_inline)) inline
__m256i load_widen_i64(const T* p) {
    if constexpr (sizeof(T) == 8) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    } else if constexpr (sizeof(T) == 4) {
        const __m128i raw =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        if constexpr (std::is_unsigned_v<T>) {
            return _mm256_cvtepu32_epi64(raw);
        } else {
            return _mm256_cvtepi32_epi64(raw);
        }
    } else if constexpr (sizeof(T) == 2) {
        const __m128i raw =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        if constexpr (std::is_unsigned_v<T>) {
            return _mm256_cvtepu16_epi64(raw);
        } else {
            return _mm256_cvtepi16_epi64(raw);
        }
    } else {
        int32_t word;
        std::memcpy(&word, p, sizeof(word));
        const __m128i raw = _mm_cvtsi32_si128(word);
        if constexpr (std::is_unsigned_v<T>) {
            return _mm256_cvtepu8_epi64(raw);
        } else {
            return _mm256_cvtepi8_epi64(raw);
        }
    }
}

// Four consecutive elements of any integral width widened into one register.
template <typename T>
__attribute__((target("avx2"), always_inline)) inline
__m256i load_widen4_i64(const T* p) {
    if constexpr (sizeof(T) == 8) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    } else if constexpr (sizeof(T) == 4) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        if constexpr (std::is_unsigned_v<T>) {
            return _mm256_cvtepu32_epi64(raw);
        } else {
            return _mm256_cvtepi32_epi64(raw);
        }
    } else if constexpr (sizeof(T) == 2) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        if constexpr (std::is_unsigned_v<T>) {
            return _mm256_cvtepu16_epi64(raw);
        } else {
            return _mm256_cvtepi16_epi64(raw);
        }
    } else {
        int32_t word;
        std::memcpy(&word, p, sizeof(word));
        const __m128i raw = _mm_cvtsi32_si128(word);
        if constexpr (std::is_unsigned_v<T>) {
            return _mm256_cvtepu8_epi64(raw);
        } else {
            return _mm256_cvtepi8_epi64(raw);
        }
    }
}

// The low 64 bits of each lane's product: the low halves multiplied as 32-bit
// unsigned products plus the two cross terms shifted into place.
__attribute__((target("avx2"), always_inline)) inline
__m256i mul_i64_low(const __m256i& a, const __m256i& b) {
    const __m256i low = _mm256_mul_epu32(a, b);
    const __m256i cross =
        _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), b),
                         _mm256_mul_epu32(a, _mm256_srli_epi64(b, 32)));
    return _mm256_add_epi64(low, _mm256_slli_epi64(cross, 32));
}

template <typename T>
__attribute__((target("avx2"), noinline))
static void prod_lastdim_int_span(const T* in, int64_t* outp, int64_t begin,
                                  int64_t end, int64_t d_size) {
    int64_t r = begin;
    for (; d_size >= 4 && r + 4 <= end; r += 4) {
        const T* p0 = in + r * d_size;
        const T* p1 = p0 + d_size;
        const T* p2 = p1 + d_size;
        const T* p3 = p2 + d_size;
        __m256i a0 = load_widen4_i64(p0);
        __m256i a1 = load_widen4_i64(p1);
        __m256i a2 = load_widen4_i64(p2);
        __m256i a3 = load_widen4_i64(p3);
        int64_t k = 4;
        for (; k + 4 <= d_size; k += 4) {
            a0 = mul_i64_low(a0, load_widen4_i64(p0 + k));
            a1 = mul_i64_low(a1, load_widen4_i64(p1 + k));
            a2 = mul_i64_low(a2, load_widen4_i64(p2 + k));
            a3 = mul_i64_low(a3, load_widen4_i64(p3 + k));
        }
        alignas(32) int64_t l0[4], l1[4], l2[4], l3[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(l0), a0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(l1), a1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(l2), a2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(l3), a3);
        {
            uint64_t t0 = static_cast<uint64_t>(l0[0]) * static_cast<uint64_t>(l0[1]) *
                              static_cast<uint64_t>(l0[2]) * static_cast<uint64_t>(l0[3]);
            uint64_t t1 = static_cast<uint64_t>(l1[0]) * static_cast<uint64_t>(l1[1]) *
                              static_cast<uint64_t>(l1[2]) * static_cast<uint64_t>(l1[3]);
            uint64_t t2 = static_cast<uint64_t>(l2[0]) * static_cast<uint64_t>(l2[1]) *
                              static_cast<uint64_t>(l2[2]) * static_cast<uint64_t>(l2[3]);
            uint64_t t3 = static_cast<uint64_t>(l3[0]) * static_cast<uint64_t>(l3[1]) *
                              static_cast<uint64_t>(l3[2]) * static_cast<uint64_t>(l3[3]);
            for (; k < d_size; ++k) {
                t0 *= static_cast<uint64_t>(p0[k]);
                t1 *= static_cast<uint64_t>(p1[k]);
                t2 *= static_cast<uint64_t>(p2[k]);
                t3 *= static_cast<uint64_t>(p3[k]);
            }
            outp[r] = static_cast<int64_t>(t0);
            outp[r + 1] = static_cast<int64_t>(t1);
            outp[r + 2] = static_cast<int64_t>(t2);
            outp[r + 3] = static_cast<int64_t>(t3);
        }
    }
    for (; r < end; ++r) {
        const T* row = in + r * d_size;
        uint64_t acc = static_cast<uint64_t>(row[0]);
        for (int64_t k = 1; k < d_size; ++k) {
            acc *= static_cast<uint64_t>(row[k]);
        }
        outp[r] = static_cast<int64_t>(acc);
    }
}

// The last axis of an integral tensor multiplied where it lies, one row at a
// time, instead of through a converted copy of the whole input.
template <typename T>
static void prod_lastdim_int(const Tensor& input, int64_t* outp, int64_t rows,
                             int64_t d_size) {
    const T* in = reinterpret_cast<const T*>(input.data_ptr());
    const int64_t grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (get_num_threads() == 1 || in_parallel_region() ||
        rows * d_size < GRAIN_SIZE) {
        prod_lastdim_int_span<T>(in, outp, 0, rows, d_size);
        return;
    }
    parallel_for(0, rows, grain, [&](int64_t b, int64_t e) {
        prod_lastdim_int_span<T>(in, outp, b, e, d_size);
    });
}

// Guards: the row is read once, so any length is worth answering here, but a
// row so short that the call costs more than the multiply keeps the iterator.
inline std::optional<Tensor> try_prod_lastdim_int(
        const Tensor& input, const std::vector<int64_t>& out_shape) {
    const DType dt = input.dtype();
    if (!input.is_contiguous() || !isIntegralType(dt, true)) {
        return std::nullopt;
    }
    if (input.dim() < 1) return std::nullopt;
    const int64_t d_size = input.size(input.dim() - 1);
    if (d_size < 2 || input.numel() == 0) return std::nullopt;
    Tensor out = Tensor::empty(out_shape, DType::Int64, input.device());
    int64_t* outp = out.data_ptr<int64_t>();
    const int64_t rows = input.numel() / d_size;
    switch (dt) {
        case DType::UInt8:
        case DType::Bool:
            prod_lastdim_int<uint8_t>(input, outp, rows, d_size);
            break;
        case DType::Int8:
            prod_lastdim_int<int8_t>(input, outp, rows, d_size);
            break;
        case DType::Int16:
            prod_lastdim_int<int16_t>(input, outp, rows, d_size);
            break;
        case DType::UInt16:
            prod_lastdim_int<uint16_t>(input, outp, rows, d_size);
            break;
        case DType::Int32:
            prod_lastdim_int<int32_t>(input, outp, rows, d_size);
            break;
        case DType::UInt32:
            prod_lastdim_int<uint32_t>(input, outp, rows, d_size);
            break;
        case DType::Int64:
            prod_lastdim_int<int64_t>(input, outp, rows, d_size);
            break;
        case DType::UInt64:
            prod_lastdim_int<uint64_t>(input, outp, rows, d_size);
            break;
        default:
            return std::nullopt;
    }
    return out;
}

// A middle axis of an integral tensor multiplied into the wide integer its
// answer is kept in.  Each plane of rows is multiplied where it lies, a band of
// eight columns at a time, and the products wrap in that width, so the answer
// is the exact product modulo it.
template <typename T>
__attribute__((target("avx2"), noinline))
static void prod_middle_int_span(const T* in, int64_t* outp, int64_t red,
                                 int64_t inner, int64_t begin, int64_t end) {
    for (int64_t o = begin; o < end; ++o) {
        const T* plane = in + o * red * inner;
        int64_t* acc = outp + o * inner;
        int64_t c = 0;
        for (; c + 8 <= inner; c += 8) {
            __m256i lo = load_widen_i64(plane + c);
            __m256i hi = load_widen_i64(plane + c + 4);
            for (int64_t r = 1; r < red; ++r) {
                const T* row = plane + r * inner + c;
                lo = mul_i64_low(lo, load_widen_i64(row));
                hi = mul_i64_low(hi, load_widen_i64(row + 4));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c), lo);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 4), hi);
        }
        for (; c < inner; ++c) {
            uint64_t total = static_cast<uint64_t>(plane[c]);
            for (int64_t r = 1; r < red; ++r) {
                total *= static_cast<uint64_t>(plane[r * inner + c]);
            }
            acc[c] = static_cast<int64_t>(total);
        }
    }
}

template <typename T>
static void prod_middle_int(const Tensor& input, int64_t* outp, int64_t outer,
                            int64_t red, int64_t inner) {
    const T* in = reinterpret_cast<const T*>(input.data_ptr());
    const int64_t grain =
        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(red * inner, 1));
    if (get_num_threads() == 1 || in_parallel_region() ||
        outer * red * inner < GRAIN_SIZE) {
        prod_middle_int_span<T>(in, outp, red, inner, 0, outer);
        return;
    }
    parallel_for(0, outer, grain, [&](int64_t b, int64_t e) {
        prod_middle_int_span<T>(in, outp, red, inner, b, e);
    });
}

// The same guards as the middle-axis sum: a plane that stays in the
// first-level cache is walked band by band, and beyond it only a plane whose
// rows fill whole cache lines is, so no pass re-reads a line for a fragment.
inline std::optional<Tensor> try_prod_middle_int(
        const Tensor& input, const std::vector<int64_t>& out_shape,
        int64_t axis) {
    const DType dt = input.dtype();
    if (!input.is_contiguous() || !isIntegralType(dt, true)) {
        return std::nullopt;
    }
    const int64_t rank = input.dim();
    if (rank < 3 || axis <= 0 || axis >= rank - 1) return std::nullopt;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= input.size(i);
    for (int64_t i = axis + 1; i < rank; ++i) inner *= input.size(i);
    const int64_t red = input.size(axis);
    if (outer < 1 || red < 1 || inner < 1) return std::nullopt;
    const int64_t itemsize = static_cast<int64_t>(elementSize(dt));
    if (red * inner * itemsize > 32768 && inner * itemsize < 64) {
        return std::nullopt;
    }
    Tensor out = Tensor::empty(out_shape, DType::Int64, input.device());
    int64_t* outp = out.data_ptr<int64_t>();
    switch (dt) {
        case DType::UInt8:
        case DType::Bool:
            prod_middle_int<uint8_t>(input, outp, outer, red, inner);
            break;
        case DType::Int8:
            prod_middle_int<int8_t>(input, outp, outer, red, inner);
            break;
        case DType::Int16:
            prod_middle_int<int16_t>(input, outp, outer, red, inner);
            break;
        case DType::UInt16:
            prod_middle_int<uint16_t>(input, outp, outer, red, inner);
            break;
        case DType::Int32:
            prod_middle_int<int32_t>(input, outp, outer, red, inner);
            break;
        case DType::UInt32:
            prod_middle_int<uint32_t>(input, outp, outer, red, inner);
            break;
        case DType::Int64:
            prod_middle_int<int64_t>(input, outp, outer, red, inner);
            break;
        case DType::UInt64:
            prod_middle_int<uint64_t>(input, outp, outer, red, inner);
            break;
        default:
            return std::nullopt;
    }
    return out;
}

#endif  // TP_VECREDUCED_X86

Tensor sum_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }

    if (dims.empty()) {
        return sum_kernel_impl(self, dtype);
    }

    // Dimensions that name every axis still ask for one number: the flat walk
    // answers that without the per-element iterator, and keepdim only re-cuts
    // the answer to a shape of ones.
    if (names_every_axis(dims, self.dim())) {
        Tensor full = sum_kernel_impl(self, dtype);
        if (!keepdim) return full;
        return full.reshape(
            std::vector<int64_t>(static_cast<size_t>(self.dim()), 1));
    }

    if (std::optional<Tensor> permuted = review_permuted_reduction(
            self, dims, keepdim,
            [&](const Tensor& dense, const std::vector<int64_t>& dense_dims) {
                return sum_dim_kernel_impl(dense, dense_dims, true, dtype);
            })) {
        return std::move(*permuted);
    }

    if (self.dim() == 0) {
        // A 0-dim tensor has a single element; reducing over its only
        // (virtual) dimension returns the full reduction, optionally keeping
        // a size-1 dimension.
        Tensor out = sum_kernel_impl(self, dtype);
        return keepdim ? out.reshape({1}) : out;
    }

    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);

    DType acc_dtype = out_dtype;
    if (isReducedFloatingType(out_dtype)) {
        acc_dtype = DType::Float32;
    } else if (out_dtype == DType::ComplexHalf || out_dtype == DType::BComplex32) {
        acc_dtype = DType::ComplexFloat;
    }
    if (acc_dtype == DType::Float32 || acc_dtype == DType::Float64) {
        Tensor input = self;
        if (self.dtype() != out_dtype) {
            input = self.to(out_dtype);
        }

        const int64_t ndim = self.dim();
        // The leading-rows kernel stores every output element exactly once,
        // so it allocates its own result and runs before the zero-filled
        // buffer below (whose memset it would otherwise render waste).
        // try_sum_leading_rows is defined in the x86 section above, so only
        // the x86 tiers have a leading-rows route at all; the other
        // architectures reach the iterator below instead.
#if defined(__x86_64__)
        if (!reduce_avx512_available() && dims.size() == 1 &&
            input.dtype() == out_dtype) {
            const int64_t dim = dims[0] < 0 ? dims[0] + ndim : dims[0];
            if (std::optional<Tensor> leading =
                    try_sum_leading_rows(input, out_shape, dim)) {
                return std::move(*leading);
            }
            if (dim == 0) {
                if (std::optional<Tensor> split =
                        try_sum_leading_rows_split(input, out_shape)) {
                    return std::move(*split);
                }
            }
        }
#endif

        Tensor out = Tensor::zeros(out_shape, out_dtype, self.device());
        std::vector<bool> mask(ndim, false);
        for (int64_t d : dims) {
            if (d < 0) d += ndim;
            mask[d] = true;
        }
        if (dims.size() == 1) {
            const int64_t dim = dims[0] < 0 ? dims[0] + ndim : dims[0];
#if defined(__x86_64__)
            // A reduced width folds the last axis a row of lanes at a time,
            // and the leading axis a whole row of lanes at a time, without a
            // converted copy of the input.
            if (input.dtype() == out_dtype) {
                if (dim == ndim - 1) {
                    if (std::optional<Tensor> folded =
                            try_sum_lastdim_reduced(input, out_shape)) {
                        return std::move(*folded);
                    }
                } else if (dim == 0) {
                    if (std::optional<Tensor> folded =
                            try_sum_leading_reduced(input, out_shape)) {
                        return std::move(*folded);
                    }
                }
                if (std::optional<Tensor> folded =
                        try_sum_dim_reduced_half(input, out_shape, dim)) {
                    return std::move(*folded);
                }
            }
#endif
            // try_sum_dim_real is x86-only like try_sum_leading_rows above.
#if defined(__x86_64__)
            if (input.dtype() == out_dtype &&
                try_sum_dim_real(input, out, dim)) {
                return out;
            }
#endif
            bool handled = false;
            if (out_dtype == DType::Float16) {
                handled = sum_detail::contiguous_sum_dim<float, Half>(input, out, dim);
            } else if (out_dtype == DType::BFloat16) {
                handled = sum_detail::contiguous_sum_dim<float, BFloat16>(input, out, dim);
            } else if (out_dtype == DType::Float32) {
                handled = sum_detail::contiguous_sum_dim<float>(input, out, dim);
            } else {
                handled = sum_detail::contiguous_sum_dim<double>(input, out, dim);
            }
            if (handled) {
                return out;
            }
        }
        Tensor viewed = review_reduce_result(out, ndim, mask, keepdim);
        TensorIterator iter = TensorIterator::reduce_op(viewed, input);
        if (iter.numel() != 0 && should_use_acc_buffer(iter)) {
            Tensor tmp = sum_to_float(self, dims, keepdim);
            out.copy_(tmp);
            return out;
        }
        if (out_dtype == DType::Float16) {
            sum_detail::cascade_sum<float, Half>(iter);
        } else if (out_dtype == DType::BFloat16) {
            sum_detail::cascade_sum<float, BFloat16>(iter);
        } else if (out_dtype == DType::Float32) {
            sum_detail::cascade_sum<float>(iter);
        } else {
            sum_detail::cascade_sum<double>(iter);
        }
        return out;
    }

    // Fast path: reducing the leading dim of a contiguous real tensor.
    // Accumulate whole rows into per-thread column buffers (row-major reads
    // stay contiguous), then fold the buffers; per-thread partials avoid
    // cross-thread contention on the output.
    if (dims.size() == 1 && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        self.is_contiguous()) {
        const int64_t ndim0 = self.dim();
        int64_t d0 = dims[0] < 0 ? dims[0] + ndim0 : dims[0];
        if (d0 == 0 && ndim0 >= 1) {
            const int64_t rows = self.size(0);
            const int64_t cols = self.numel() / std::max<int64_t>(rows, 1);
            Tensor out = Tensor::zeros(out_shape, self.dtype(), self.device());
            if (self.numel() == 0) return out;
            const int64_t nthreads = std::max<int64_t>(1, tensorplay::parallel::get_num_threads());
            if (self.dtype() == DType::Float32) {
                const float* in = self.data_ptr<float>();
                float* outp = out.data_ptr<float>();
                std::vector<std::vector<float>> partials(
                    nthreads, std::vector<float>(cols, 0.f));
                const int64_t row_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(cols, 1));
                tensorplay::parallel::parallel_for(0, rows, row_grain, [&](int64_t rb, int64_t re) {
                    float* TP_RESTRICT acc = partials[tensorplay::parallel::get_thread_num()].data();
                    // Four rows land in each accumulator slot before it is read
                    // back, so the column buffer is touched a quarter as often:
                    // that is the difference between a pass limited by L1
                    // traffic and one limited by memory.
                    const int64_t re4 = rb + (re - rb) / 4 * 4;
                    for (int64_t r = rb; r < re4; r += 4) {
                        const float* TP_RESTRICT r0 = in + r * cols;
                        const float* TP_RESTRICT r1 = r0 + cols;
                        const float* TP_RESTRICT r2 = r1 + cols;
                        const float* TP_RESTRICT r3 = r2 + cols;
                        for (int64_t j = 0; j < cols; ++j)
                            acc[j] += ((r0[j] + r1[j]) + (r2[j] + r3[j]));
                    }
                    for (int64_t r = re4; r < re; ++r) {
                        const float* TP_RESTRICT rowp = in + r * cols;
                        for (int64_t j = 0; j < cols; ++j) acc[j] += rowp[j];
                    }
                });
                std::vector<float> total(cols, 0.f);
                for (int64_t t = 0; t < nthreads; ++t)
                    for (int64_t j = 0; j < cols; ++j) total[j] += partials[t][j];
                for (int64_t j = 0; j < cols; ++j) outp[j] = total[j];
            } else {
                const double* in = self.data_ptr<double>();
                double* outp = out.data_ptr<double>();
                std::vector<std::vector<double>> partials(
                    nthreads, std::vector<double>(cols, 0.0));
                const int64_t row_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(cols, 1));
                tensorplay::parallel::parallel_for(0, rows, row_grain, [&](int64_t rb, int64_t re) {
                    double* TP_RESTRICT acc = partials[tensorplay::parallel::get_thread_num()].data();
                    const int64_t re4 = rb + (re - rb) / 4 * 4;
                    for (int64_t r = rb; r < re4; r += 4) {
                        const double* TP_RESTRICT r0 = in + r * cols;
                        const double* TP_RESTRICT r1 = r0 + cols;
                        const double* TP_RESTRICT r2 = r1 + cols;
                        const double* TP_RESTRICT r3 = r2 + cols;
                        for (int64_t j = 0; j < cols; ++j)
                            acc[j] += ((r0[j] + r1[j]) + (r2[j] + r3[j]));
                    }
                    for (int64_t r = re4; r < re; ++r) {
                        const double* TP_RESTRICT rowp = in + r * cols;
                        for (int64_t j = 0; j < cols; ++j) acc[j] += rowp[j];
                    }
                });
                std::vector<double> total(cols, 0.0);
                for (int64_t t = 0; t < nthreads; ++t)
                    for (int64_t j = 0; j < cols; ++j) total[j] += partials[t][j];
                for (int64_t j = 0; j < cols; ++j) outp[j] = total[j];
            }
            return out;
        }
    }

    // Fast path: reducing the trailing dim of a contiguous real tensor.
    // Every output item owns one contiguous input row, so rows can be split
    // independently and each row can use the wide reduction loop.
    if (dims.size() == 1 && out_dtype == self.dtype() &&
        (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
        self.is_contiguous()) {
        const int64_t ndim0 = self.dim();
        const int64_t d0 = dims[0] < 0 ? dims[0] + ndim0 : dims[0];
        if (d0 == ndim0 - 1 && ndim0 >= 1) {
            const int64_t d_size = self.size(d0);
            Tensor out = Tensor::zeros(out_shape, self.dtype(), self.device());
            const int64_t rows = d_size > 0 ? self.numel() / d_size : out.numel();
            if (rows == 0 || d_size == 0) return out;
            const int64_t row_grain = std::max<int64_t>(
                1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
#if defined(__x86_64__)
            const bool use_avx512 = reduce_avx512_available() && d_size >= 16;
#else
            const bool use_avx512 = false;
#endif
            if (self.dtype() == DType::Float32) {
                const float* in = self.data_ptr<float>();
                float* outp = out.data_ptr<float>();
                parallel_for(0, rows, row_grain, [&](int64_t rb, int64_t re) {
                    for (int64_t r = rb; r < re; ++r) {
                        const float* row = in + r * d_size;
                        float total = 0.0f;
#if defined(__x86_64__)
                        if (use_avx512) {
                            total = sum_f32_chunk_avx512(row, 0, d_size);
                        } else
#endif
                        {
                            for (int64_t j = 0; j < d_size; ++j) total += row[j];
                        }
                        outp[r] = total;
                    }
                });
            } else {
                const double* in = self.data_ptr<double>();
                double* outp = out.data_ptr<double>();
                parallel_for(0, rows, row_grain, [&](int64_t rb, int64_t re) {
                    for (int64_t r = rb; r < re; ++r) {
                        const double* row = in + r * d_size;
                        double total = 0.0;
#if defined(__x86_64__)
                        if (use_avx512) {
                            total = sum_f64_chunk_avx512(row, 0, d_size);
                        } else
#endif
                        {
                            for (int64_t j = 0; j < d_size; ++j) total += row[j];
                        }
                        outp[r] = total;
                    }
                });
            }
            return out;
        }
    }

    // complexes in complex64.
    acc_dtype = out_dtype;
    if (isReducedFloatingType(out_dtype)) {
        acc_dtype = DType::Float32;
    } else if (out_dtype == DType::ComplexHalf || out_dtype == DType::BComplex32) {
        acc_dtype = DType::ComplexFloat;
    }

    // An integral tensor whose answer is a wide integer is summed where it
    // lies, one contiguous row at a time, instead of through a converted copy
    // of the whole input first.
#if defined(TP_VECREDUCED_X86)
    if (acc_dtype == DType::Int64 && dims.size() == 1) {
        const int64_t rank = self.dim();
        const int64_t axis = dims[0] < 0 ? dims[0] + rank : dims[0];
        if (rank >= 1 && axis == rank - 1 && self.is_contiguous()) {
            if (std::optional<Tensor> direct =
                    try_sum_lastdim_int(self, out_shape)) {
                return std::move(*direct);
            }
        } else if (axis == 0 && self.is_contiguous()) {
            if (std::optional<Tensor> direct =
                    try_sum_leading_int(self, out_shape)) {
                return std::move(*direct);
            }
        } else if (self.is_contiguous()) {
            if (std::optional<Tensor> direct =
                    try_sum_middle_int(self, out_shape, axis)) {
                return std::move(*direct);
            }
        }
    }
#endif  // TP_VECREDUCED_X86

    Tensor out = Tensor::zeros(out_shape, acc_dtype, self.device());
    
    Tensor input = self;
    if (self.dtype() != acc_dtype) {
        input = self.to(acc_dtype);
    }
    
    // As-strided view of the output with the reduced dims materialized as
    // size-1/stride-0 dims (see review_reduce_result), so the iterator knows
    // which input dims are reduced.
    int64_t ndim = self.dim();
    std::vector<bool> mask(ndim, false);
    for (int64_t d : dims) {
        if (d < 0) d += ndim;
        mask[d] = true;
    }
    Tensor viewed = review_reduce_result(out, ndim, mask, keepdim);
    
    TensorIterator iter = TensorIterator::reduce_op(viewed, input);
    sum_kernel_iter(iter);

    return acc_dtype == out_dtype ? out : out.to(out_dtype);
}

// NaN-masked wide reduction over contiguous float rows; answers through
// `handled` so the caller can fall back for the layouts this skips.  A fold
// that collapses to one value runs as a single masked pass; the multi-row
// shapes zero the NaN lanes into a scratch copy and let the plain
// reduction's row plans carry every layout they know.
Tensor nansum_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims,
                              bool keepdim, DType dtype, bool* handled) {
    *handled = false;
    if (!self.is_contiguous() || self.numel() == 0 ||
        (self.dtype() != DType::Float32 && self.dtype() != DType::Float64)) {
        return Tensor();
    }
    *handled = true;
    const int64_t nd = self.dim();
    std::vector<int64_t> dim = dims;
    if (dim.empty()) {
        for (int64_t i = 0; i < nd; ++i) dim.push_back(i);
    }
    for (int64_t& d : dim) {
        if (d < 0) d += nd;
    }
    std::vector<int64_t> distinct(dim);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    const bool covers_all = static_cast<int64_t>(distinct.size()) == nd;

    if (covers_all) {
        Tensor out = Tensor::zeros(
            keepdim ? std::vector<int64_t>(static_cast<size_t>(nd), 1)
                    : std::vector<int64_t>{},
            dtype, self.device());
        const int64_t n = self.numel();
        double s = 0.0;
        if (dtype == DType::Float32) {
            const float* x = self.data_ptr<float>();
            if (n < 4096 || get_num_threads() == 1 || in_parallel_region()) {
                s = vec::nansum(x, 0, n);
            } else {
                std::vector<double> part(
                    static_cast<size_t>(std::max<int64_t>(get_num_threads(), 1)), 0.0);
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                    part[static_cast<size_t>(get_thread_num())] +=
                        vec::nansum(x, b, e);
                });
                for (double p : part) s += p;
            }
        } else {
            const double* x = self.data_ptr<double>();
            if (n < 4096 || get_num_threads() == 1 || in_parallel_region()) {
                s = vec::nansum(x, 0, n);
            } else {
                std::vector<double> part(
                    static_cast<size_t>(std::max<int64_t>(get_num_threads(), 1)), 0.0);
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                    part[static_cast<size_t>(get_thread_num())] +=
                        vec::nansum(x, b, e);
                });
                for (double p : part) s += p;
            }
        }
        if (dtype == DType::Float32) {
            *out.data_ptr<float>() = static_cast<float>(s);
        } else {
            *out.data_ptr<double>() = s;
        }
        return out;
    }

    Tensor x = self;
    const int64_t n = self.numel();
    const int64_t nslots = static_cast<int64_t>(std::max<int64_t>(get_num_threads(), 1));
    std::vector<char> hits(static_cast<size_t>(nslots), 0);
    if (self.dtype() == DType::Float32) {
        const float* xp = x.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            hits[static_cast<size_t>(get_thread_num())] |= vec::any_nan(xp, b, e) ? 1 : 0;
        });
    } else {
        const double* xp = x.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            hits[static_cast<size_t>(get_thread_num())] |= vec::any_nan(xp, b, e) ? 1 : 0;
        });
    }
    bool dirty = false;
    for (char h : hits) dirty = dirty || h != 0;
    if (dirty) {
        Tensor clean = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        if (self.dtype() == DType::Float32) {
            const float* xp = x.data_ptr<float>();
            float* dp = clean.data_ptr<float>();
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                vec::store_without_nan(xp, b, e, dp);
            });
        } else {
            const double* xp = x.data_ptr<double>();
            double* dp = clean.data_ptr<double>();
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
                vec::store_without_nan(xp, b, e, dp);
            });
        }
        x = clean;
    }
    return sum_dim_kernel_impl(x, dims, keepdim, dtype);
}





template <typename T>
T get_lowest() {
    if constexpr (std::is_floating_point_v<T>) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <typename T>
T get_highest() {
    if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <bool WantAll>
bool byte_reduce_parallel(const uint8_t* data, int64_t n) {
#if defined(__x86_64__)
    if (byte_reduce_avx512_available() && n >= 64) {
        return byte_reduce_avx512<WantAll>(data, n);
    }
#endif

    auto serial_reduce = [&]() {
        int64_t i = 0;
        constexpr uint64_t kOnes = 0x0101010101010101ULL;
        constexpr uint64_t kHighBits = 0x8080808080808080ULL;
        for (; i + 8 <= n; i += 8) {
            uint64_t word;
            std::memcpy(&word, data + i, sizeof(word));
            if constexpr (WantAll) {
                if (((word - kOnes) & ~word & kHighBits) != 0) return false;
            } else if (word != 0) {
                return true;
            }
        }
        for (; i < n; ++i) {
            if constexpr (WantAll) {
                if (data[i] == 0) return false;
            } else if (data[i] != 0) {
                return true;
            }
        }
        return WantAll;
    };

    if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        return serial_reduce();
    }

    const int num_threads = get_num_threads();
    std::vector<unsigned char> partials(
        static_cast<size_t>(num_threads), WantAll ? 1 : 0);
    constexpr uint64_t kOnes = 0x0101010101010101ULL;
    constexpr uint64_t kHighBits = 0x8080808080808080ULL;
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t i = begin;
        for (; i + 8 <= end; i += 8) {
            uint64_t word;
            std::memcpy(&word, data + i, sizeof(word));
            if constexpr (WantAll) {
                if (((word - kOnes) & ~word & kHighBits) != 0) {
                    partials[get_thread_num()] = 0;
                    return;
                }
            } else if (word != 0) {
                partials[get_thread_num()] = 1;
                return;
            }
        }
        for (; i < end; ++i) {
            if constexpr (WantAll) {
                if (data[i] == 0) {
                    partials[get_thread_num()] = 0;
                    return;
                }
            } else if (data[i] != 0) {
                partials[get_thread_num()] = 1;
                return;
            }
        }
    });

    for (unsigned char partial : partials) {
        if constexpr (WantAll) {
            if (partial == 0) return false;
        } else if (partial != 0) {
            return true;
        }
    }
    return WantAll;
}

template <typename scalar_t>
scalar_t product_reduce_parallel(const scalar_t* data, int64_t n) {
    const scalar_t identity = scalar_t(1);
    auto serial_reduce = [&]() {
        scalar_t value = identity;
        for (int64_t i = 0; i < n; ++i) {
            Accumulator<scalar_t>::mul(value, data[i]);
        }
        return value;
    };

    if constexpr (std::is_same_v<scalar_t, bool>) {
        return serial_reduce();
    } else {
        if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
            return serial_reduce();
        }

        const int num_threads = get_num_threads();
        std::vector<scalar_t> partials(static_cast<size_t>(num_threads), identity);
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            scalar_t local0 = identity;
            scalar_t local1 = identity;
            scalar_t local2 = identity;
            scalar_t local3 = identity;
            int64_t i = begin;
            if constexpr (std::is_arithmetic_v<scalar_t>) {
                for (; i + 3 < end; i += 4) {
                    Accumulator<scalar_t>::mul(local0, data[i]);
                    Accumulator<scalar_t>::mul(local1, data[i + 1]);
                    Accumulator<scalar_t>::mul(local2, data[i + 2]);
                    Accumulator<scalar_t>::mul(local3, data[i + 3]);
                }
            }
            for (; i < end; ++i) {
                Accumulator<scalar_t>::mul(local0, data[i]);
            }
            Accumulator<scalar_t>::mul(local0, local1);
            Accumulator<scalar_t>::mul(local0, local2);
            Accumulator<scalar_t>::mul(local0, local3);
            Accumulator<scalar_t>::mul(partials[get_thread_num()], local0);
        });

        scalar_t value = identity;
        for (const scalar_t partial : partials) {
            Accumulator<scalar_t>::mul(value, partial);
        }
        return value;
    }
}

template <typename scalar_t>
bool all_reduce_parallel(const scalar_t* data, int64_t n) {
    auto serial_reduce = [&]() {
        for (int64_t i = 0; i < n; ++i) {
            if (!static_cast<bool>(data[i])) return false;
        }
        return true;
    };

    if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        return serial_reduce();
    }

    const int num_threads = get_num_threads();
    std::vector<unsigned char> partials(static_cast<size_t>(num_threads), 1);
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (!static_cast<bool>(data[i])) {
                partials[get_thread_num()] = 0;
                break;
            }
        }
    });
    for (unsigned char partial : partials) {
        if (partial == 0) return false;
    }
    return true;
}

template <typename scalar_t>
bool any_reduce_parallel(const scalar_t* data, int64_t n) {
    auto serial_reduce = [&]() {
        for (int64_t i = 0; i < n; ++i) {
            if (static_cast<bool>(data[i])) return true;
        }
        return false;
    };

    if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        return serial_reduce();
    }

    const int num_threads = get_num_threads();
    std::vector<unsigned char> partials(static_cast<size_t>(num_threads), 0);
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (static_cast<bool>(data[i])) {
                partials[get_thread_num()] = 1;
                break;
            }
        }
    });
    for (unsigned char partial : partials) {
        if (partial != 0) return true;
    }
    return false;
}

// A lane is unnumbered when the word it carries cannot be ordered against
// another lane, so no compare can hand it to an extremum: a walk has to ask
// the question directly. The reduced widths are classes that carry their
// payload through a float conversion, so the same test reaches them; a lane
// of a count is always orderable and answers false, folding the question out
// of the walks that count.
template <typename T>
inline bool lane_unnumbered(T value) {
    if constexpr (std::is_floating_point_v<T> ||
                  std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return std::isnan(value);
    }
    return false;
}

template <typename scalar_t>
int64_t argmin_reduce_parallel(const scalar_t* data, int64_t n) {
    auto serial_reduce = [&]() {
        scalar_t value = get_highest<scalar_t>();
        int64_t index = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (lane_unnumbered(data[i])) return i;
            if (data[i] < value) {
                value = data[i];
                index = i;
            }
        }
        return index;
    };

    if constexpr (std::is_same_v<scalar_t, bool>) {
        return serial_reduce();
    } else {
        if (n < GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
            return serial_reduce();
        }

        const int64_t chunk_count = std::min<int64_t>(32, n);
        const int64_t chunk_size = (n + chunk_count - 1) / chunk_count;
        std::vector<scalar_t> chunk_values(
            static_cast<size_t>(chunk_count), get_highest<scalar_t>());
        std::vector<int64_t> chunk_indices(static_cast<size_t>(chunk_count), 0);
        std::vector<int64_t> chunk_unnumbered(
            static_cast<size_t>(chunk_count), -1);
        parallel_for(0, chunk_count, 1, [&](int64_t begin, int64_t end) {
            for (int64_t chunk = begin; chunk < end; ++chunk) {
                const int64_t lo = chunk * chunk_size;
                const int64_t hi = std::min(n, lo + chunk_size);
                scalar_t value = get_highest<scalar_t>();
                int64_t index = lo;
                for (int64_t i = lo; i < hi; ++i) {
                    if (lane_unnumbered(data[i])) {
                        chunk_unnumbered[static_cast<size_t>(chunk)] = i;
                        break;
                    }
                    if (data[i] < value) {
                        value = data[i];
                        index = i;
                    }
                }
                chunk_values[static_cast<size_t>(chunk)] = value;
                chunk_indices[static_cast<size_t>(chunk)] = index;
            }
        });

        // Chunks combine in ascending order, so the first chunk that holds an
        // unnumbered lane names the earliest one and no compare after it can
        // displace that answer.
        for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int64_t held = chunk_unnumbered[static_cast<size_t>(chunk)];
            if (held >= 0) return held;
        }
        scalar_t value = get_highest<scalar_t>();
        int64_t index = 0;
        for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
            if (chunk_values[static_cast<size_t>(chunk)] < value) {
                value = chunk_values[static_cast<size_t>(chunk)];
                index = chunk_indices[static_cast<size_t>(chunk)];
            }
        }
        return index;
    }
}

// returns NaN when any element is NaN), plain compare otherwise.
// The reduced-float types carry NaN payloads through their float conversions,
// so the NaN rule covers them too; their vectorized maximum/minimum and the
// dim-reduction kernels already propagate NaN, and the whole-tensor fold must
// agree with both.
template <typename T>
inline T nan_max(T a, T b) {
    if constexpr (std::is_floating_point_v<T> ||
                  std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<T>::quiet_NaN();
    }
    return a < b ? b : a;
}

template <typename T>
inline T nan_min(T a, T b) {
    if constexpr (std::is_floating_point_v<T> ||
                  std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<T>::quiet_NaN();
    }
    return b < a ? b : a;
}

template <typename T>
struct ExtremaValuesOps {
    bool maximum_value;
    T combine(T a, T b) const { return maximum_value ? nan_max(a, b) : nan_min(a, b); }
    T reduce(T a, T b, int64_t) const { return combine(a, b); }
    T project(T value) const { return value; }
    T translate_idx(T value, int64_t) const { return value; }
};

template <typename T>
T extrema_identity(bool maximum_value) {
    if constexpr (std::is_floating_point_v<T> ||
                  std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return maximum_value ? -std::numeric_limits<T>::infinity()
                             : std::numeric_limits<T>::infinity();
    } else {
        return maximum_value ? std::numeric_limits<T>::lowest()
                             : std::numeric_limits<T>::max();
    }
}

// One contiguous run folded the way the rule is written: every combine tests
// for a NaN, so the walk costs a compare, a broadcast of its result and an or
// per vector on top of the max itself.  It is the answer of record, and the
// fast walk below hands a run that met a NaN back to it.
template <typename ctype, bool WantMax>
inline ctype fold_run_exact(const ctype* data, int64_t n) {
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    Vec acc[4];
    for (auto& a : acc) a = Vec(extrema_identity<ctype>(WantMax));
    int64_t i = 0;
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        for (int j = 0; j < 4; ++j) {
            const Vec value = Vec::loadu(data + i + j * lanes);
            acc[j] = WantMax ? maximum(acc[j], value) : minimum(acc[j], value);
        }
    }
    ctype values[lanes];
    ctype best = extrema_identity<ctype>(WantMax);
    for (auto& a : acc) {
        a.store(values);
        for (int64_t k = 0; k < lanes; ++k) {
            best = WantMax ? nan_max(best, values[k]) : nan_min(best, values[k]);
        }
    }
    for (; i < n; ++i) best = WantMax ? nan_max(best, data[i]) : nan_min(best, data[i]);
    return best;
}

// Lane-wise larger (WantMax) or smaller value, taken straight from the
// instruction that owns the compare.  What that instruction does with a NaN
// operand is its own business: the caller runs a second chain over the same
// loads, and a NaN anywhere in the run shows up in it, so the ordering walk
// never has to stop and check.  The branch follows the tier the vector type
// was built for, not the compiler's ISA macros: MSVC builds every tier with
// /arch:AVX2, where the baseline Vectorized is not a __m256.
template <bool WantMax>
inline Vectorized<float> extremum_vec(const Vectorized<float>& acc,
                                      const Vectorized<float>& value) {
#if defined(CPU_CAPABILITY_AVX512)
    return WantMax ? _mm512_max_ps(acc, value) : _mm512_min_ps(acc, value);
#elif defined(CPU_CAPABILITY_AVX2)
    return WantMax ? _mm256_max_ps(acc, value) : _mm256_min_ps(acc, value);
#else
    return WantMax ? maximum(acc, value) : minimum(acc, value);
#endif
}

template <bool WantMax>
inline Vectorized<double> extremum_vec(const Vectorized<double>& acc,
                                       const Vectorized<double>& value) {
#if defined(CPU_CAPABILITY_AVX512)
    return WantMax ? _mm512_max_pd(acc, value) : _mm512_min_pd(acc, value);
#elif defined(CPU_CAPABILITY_AVX2)
    return WantMax ? _mm256_max_pd(acc, value) : _mm256_min_pd(acc, value);
#else
    return WantMax ? maximum(acc, value) : minimum(acc, value);
#endif
}

// One contiguous run folded into four accumulators.  Grouping an extremum is
// free: the answer is the same in any order, NaN included, because a NaN
// operand puts a NaN in the result wherever it lands.  Four chains hide the
// latency of the compare the way the iterator's own inner loop does, and the
// lane chain and the ragged end are the only scalar work the row needs.
//
// The run is walked for its largest value with one instruction per vector, and
// the same vectors are accumulated into a sum on the side.  A NaN reaches the
// sum and stays there whichever lane it was in, while a finite run can only
// take the sum to another finite value or to an infinity, so a NaN in the sum
// means either that the run holds one or that it holds both infinities.  Both
// answers come from the exact walk, and the cost of that second pass falls only
// on a run that holds a NaN or an infinity pair.
template <typename ctype, bool WantMax>
inline ctype fold_run_extremum(const ctype* data, int64_t n) {
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    Vec acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    const Vec seed(extrema_identity<ctype>(WantMax));
    acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = acc7 = seed;
    Vec trace0{ctype(0)}, trace1{ctype(0)};
    Vec trace2{ctype(0)}, trace3{ctype(0)};
    int64_t i = 0;
    for (; i + 8 * lanes <= n; i += 8 * lanes) {
        const Vec v0 = Vec::loadu(data + i + 0 * lanes);
        const Vec v1 = Vec::loadu(data + i + 1 * lanes);
        const Vec v2 = Vec::loadu(data + i + 2 * lanes);
        const Vec v3 = Vec::loadu(data + i + 3 * lanes);
        const Vec v4 = Vec::loadu(data + i + 4 * lanes);
        const Vec v5 = Vec::loadu(data + i + 5 * lanes);
        const Vec v6 = Vec::loadu(data + i + 6 * lanes);
        const Vec v7 = Vec::loadu(data + i + 7 * lanes);
        acc0 = extremum_vec<WantMax>(acc0, v0);
        acc1 = extremum_vec<WantMax>(acc1, v1);
        acc2 = extremum_vec<WantMax>(acc2, v2);
        acc3 = extremum_vec<WantMax>(acc3, v3);
        acc4 = extremum_vec<WantMax>(acc4, v4);
        acc5 = extremum_vec<WantMax>(acc5, v5);
        acc6 = extremum_vec<WantMax>(acc6, v6);
        acc7 = extremum_vec<WantMax>(acc7, v7);
        trace0 = trace0 + (v0 + v1);
        trace1 = trace1 + (v2 + v3);
        trace2 = trace2 + (v4 + v5);
        trace3 = trace3 + (v6 + v7);
    }
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        const Vec v0 = Vec::loadu(data + i + 0 * lanes);
        const Vec v1 = Vec::loadu(data + i + 1 * lanes);
        const Vec v2 = Vec::loadu(data + i + 2 * lanes);
        const Vec v3 = Vec::loadu(data + i + 3 * lanes);
        acc0 = extremum_vec<WantMax>(acc0, v0);
        acc1 = extremum_vec<WantMax>(acc1, v1);
        acc2 = extremum_vec<WantMax>(acc2, v2);
        acc3 = extremum_vec<WantMax>(acc3, v3);
        trace0 = trace0 + (v0 + v1);
        trace1 = trace1 + (v2 + v3);
    }
    ctype marks[lanes];
    ((trace0 + trace1) + (trace2 + trace3)).store(marks);
    for (int64_t k = 0; k < lanes; ++k) {
        if (!(marks[k] == marks[k])) return fold_run_exact<ctype, WantMax>(data, n);
    }
    ctype values[lanes];
    ctype best = extrema_identity<ctype>(WantMax);
    const Vec pairs[4] = {
        extremum_vec<WantMax>(acc0, acc1), extremum_vec<WantMax>(acc2, acc3),
        extremum_vec<WantMax>(acc4, acc5), extremum_vec<WantMax>(acc6, acc7)};
    for (const Vec& a : pairs) {
        a.store(values);
        for (int64_t k = 0; k < lanes; ++k) {
            best = WantMax ? (best < values[k] ? values[k] : best)
                           : (values[k] < best ? values[k] : best);
        }
    }
    for (; i < n; ++i) {
        const ctype value = data[i];
        if (!(value == value)) return fold_run_exact<ctype, WantMax>(data, n);
        best = WantMax ? (best < value ? value : best)
                       : (value < best ? value : best);
    }
    return best;
}

// A short run folds the same way without the second chain: the compare that
// owns the pair also reports when either operand is a not-a-number, and that
// answer is put back into the lane, so a run holding one reaches the exact
// walk from the fold itself.
template <bool WantMax>
inline Vectorized<float> extremum_vec_nan(const Vectorized<float>& acc,
                                          const Vectorized<float>& value) {
#if defined(CPU_CAPABILITY_AVX512)
    return _mm512_or_ps(WantMax ? _mm512_max_ps(acc, value)
                                : _mm512_min_ps(acc, value),
                        _mm512_cmp_ps_mask(acc, value, _CMP_UNORD_Q) == 0
                            ? _mm512_setzero_ps()
                            : _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
#elif defined(CPU_CAPABILITY_AVX2)
    return _mm256_or_ps(WantMax ? _mm256_max_ps(acc, value)
                                : _mm256_min_ps(acc, value),
                        _mm256_cmp_ps(acc, value, _CMP_UNORD_Q));
#else
    return WantMax ? maximum(acc, value) : minimum(acc, value);
#endif
}

template <bool WantMax, typename ctype>
inline Vectorized<ctype> extremum_vec_nan(const Vectorized<ctype>& acc,
                                          const Vectorized<ctype>& value) {
    return WantMax ? maximum(acc, value) : minimum(acc, value);
}

template <bool WantMax>
inline Vectorized<double> extremum_vec_nan(const Vectorized<double>& acc,
                                           const Vectorized<double>& value) {
#if defined(CPU_CAPABILITY_AVX512)
    return _mm512_or_pd(WantMax ? _mm512_max_pd(acc, value)
                                : _mm512_min_pd(acc, value),
                        _mm512_cmp_pd_mask(acc, value, _CMP_UNORD_Q) == 0
                            ? _mm512_setzero_pd()
                            : _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN()));
#elif defined(CPU_CAPABILITY_AVX2)
    return _mm256_or_pd(WantMax ? _mm256_max_pd(acc, value)
                                : _mm256_min_pd(acc, value),
                        _mm256_cmp_pd(acc, value, _CMP_UNORD_Q));
#else
    return WantMax ? maximum(acc, value) : minimum(acc, value);
#endif
}

// A run no longer than a few hundred elements folds through four chains that
// each carry the not-a-number report along with the compare: the wide ladder
// above spends a second chain on every load, which a short run cannot pay for,
// while one chain alone would wait on the latency of its own compare.
template <typename ctype, bool WantMax>
inline ctype fold_run_extremum_short(const ctype* data, int64_t n) {
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    Vec acc[4];
    for (auto& a : acc) a = Vec(extrema_identity<ctype>(WantMax));
    int64_t i = 0;
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        for (int j = 0; j < 4; ++j) {
            acc[j] = extremum_vec_nan<WantMax>(acc[j],
                                               Vec::loadu(data + i + j * lanes));
        }
    }
    for (; i + lanes <= n; i += lanes) {
        acc[0] = extremum_vec_nan<WantMax>(acc[0], Vec::loadu(data + i));
    }
    Vec merged = acc[0];
    for (int j = 1; j < 4; ++j) merged = extremum_vec_nan<WantMax>(merged, acc[j]);
    ctype values[lanes];
    merged.store(values);
    ctype best = values[0];
    for (int64_t k = 1; k < lanes; ++k) {
        const ctype value = values[k];
        if (!(value == value)) return fold_run_exact<ctype, WantMax>(data, n);
        best = WantMax ? (best < value ? value : best)
                       : (value < best ? value : best);
    }
    for (; i < n; ++i) {
        const ctype value = data[i];
        if (!(value == value)) return fold_run_exact<ctype, WantMax>(data, n);
        best = WantMax ? (best < value ? value : best)
                       : (value < best ? value : best);
    }
    if (!(best == best)) return fold_run_exact<ctype, WantMax>(data, n);
    return best;
}

// Four rows folded at once: each row keeps its own chain over the columns, so
// a short row's chain is never the only work in flight.  The lanes of each
// chain carry the not-a-number report the same way the single-row fold does.
template <typename ctype, bool WantMax>
inline ctype short_row_horizontal(const Vectorized<ctype>& acc, const ctype* data,
                                  int64_t from, int64_t n) {
    constexpr int64_t lanes = Vectorized<ctype>::size();
    ctype values[lanes];
    acc.store(values);
    if (!(values[0] == values[0])) return fold_run_exact<ctype, WantMax>(data, n);
    ctype best = values[0];
    for (int64_t k = 1; k < lanes; ++k) {
        const ctype value = values[k];
        if (!(value == value)) return fold_run_exact<ctype, WantMax>(data, n);
        best = WantMax ? (best < value ? value : best)
                       : (value < best ? value : best);
    }
    for (int64_t i = from; i < n; ++i) {
        const ctype value = data[i];
        if (!(value == value)) return fold_run_exact<ctype, WantMax>(data, n);
        best = WantMax ? (best < value ? value : best)
                       : (value < best ? value : best);
    }
    if (!(best == best)) return fold_run_exact<ctype, WantMax>(data, n);
    return best;
}

// The lanes of a value that hold a not-a-number, one bit per lane.
#if defined(CPU_CAPABILITY_AVX2)
inline int nan_lanes(const Vectorized<float>& value) {
    return _mm256_movemask_ps(_mm256_cmp_ps(value, value, _CMP_UNORD_Q));
}

inline int nan_lanes(const Vectorized<double>& value) {
    return _mm256_movemask_pd(_mm256_cmp_pd(value, value, _CMP_UNORD_Q));
}
#else
template <typename ctype>
inline int nan_lanes(const Vectorized<ctype>& value) {
    constexpr int64_t lanes = Vectorized<ctype>::size();
    ctype held[lanes];
    value.store(held);
    int mask = 0;
    for (int64_t k = 0; k < lanes; ++k) {
        if (!(held[k] == held[k])) mask |= 1 << k;
    }
    return mask;
}
#endif

#if defined(TP_VECREDUCED_X86)
// A compare that reports a not-a-number from either operand.
template <bool WantMax>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
__m256 extremum_step8(const __m256& a, const __m256& b) {
    const __m256 pick = WantMax ? _mm256_max_ps(a, b) : _mm256_min_ps(a, b);
    return _mm256_or_ps(pick, _mm256_cmp_ps(a, b, _CMP_UNORD_Q));
}
#endif  // TP_VECREDUCED_X86

// A run shorter than one step still reads a whole step per row: the lanes past
// the run are replaced by the identity, so a row's compare cannot see the next
// row's values.  Only the tensor's own last row needs a padded copy to read,
// and a row holding a not-a-number takes the exact walk.
template <typename ctype, bool WantMax>
inline void fold_step_rows(const ctype* data, int64_t run, int64_t begin,
                           int64_t end, ctype* out, int64_t last_row) {
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    const Vec identity(extrema_identity<ctype>(WantMax));
    alignas(64) unsigned char bits[lanes * sizeof(ctype)];
    std::memset(bits, 0, sizeof(bits));
    std::memset(bits, 0xFF, static_cast<size_t>(run) * sizeof(ctype));
    const Vec keep = Vec::loadu(reinterpret_cast<const ctype*>(bits));
    for (int64_t r = begin; r < end; ++r) {
        const ctype* p = data + r * run;
        Vec value;
        if (r == last_row) {
            alignas(64) ctype pad[lanes];
            for (int64_t k = 0; k < run; ++k) pad[k] = p[k];
            for (int64_t k = run; k < lanes; ++k) {
                pad[k] = extrema_identity<ctype>(WantMax);
            }
            value = Vec::loadu(pad);
        } else {
            value = Vec::loadu(p);
        }
        if (nan_lanes(value) != 0) {
            out[r] = WantMax ? fold_run_extremum_short<ctype, true>(p, run)
                             : fold_run_extremum_short<ctype, false>(p, run);
            continue;
        }
        const Vec masked = Vec::blendv(identity, value, keep);
        out[r] = WantMax ? masked.reduce_max() : masked.reduce_min();
    }
}

#if defined(CPU_CAPABILITY_AVX2)
// Four rows of a short run answered together: each value is folded to its own
// halves, the four are transposed so that one row's lanes meet in one lane, and
// the four answers leave as one store.  A row's remaining lanes are read with
// the lanes past the run already replaced by the identity, so no scalar walk
// over the lanes is needed.
template <bool WantMax>
inline void store_four_float(const Vectorized<float>& u0,
                             const Vectorized<float>& u1,
                             const Vectorized<float>& u2,
                             const Vectorized<float>& u3, float* out) {
    auto pair = [](const __m256& a, const __m256& b) {
        return WantMax ? _mm256_max_ps(a, b) : _mm256_min_ps(a, b);
    };
    const __m256 long0 = pair(u0, _mm256_permute2f128_ps(u0, u0, 0x1));
    const __m256 long1 = pair(u1, _mm256_permute2f128_ps(u1, u1, 0x1));
    const __m256 long2 = pair(u2, _mm256_permute2f128_ps(u2, u2, 0x1));
    const __m256 long3 = pair(u3, _mm256_permute2f128_ps(u3, u3, 0x1));
    const __m128 h0 = _mm256_castps256_ps128(long0);
    const __m128 h1 = _mm256_castps256_ps128(long1);
    const __m128 h2 = _mm256_castps256_ps128(long2);
    const __m128 h3 = _mm256_castps256_ps128(long3);
    auto pair_half = [](const __m128& a, const __m128& b) {
        return WantMax ? _mm_max_ps(a, b) : _mm_min_ps(a, b);
    };
    const __m128 lo01 = _mm_unpacklo_ps(h0, h1);
    const __m128 hi01 = _mm_unpackhi_ps(h0, h1);
    const __m128 lo23 = _mm_unpacklo_ps(h2, h3);
    const __m128 hi23 = _mm_unpackhi_ps(h2, h3);
    const __m128 s0 = _mm_movelh_ps(lo01, lo23);
    const __m128 s1 = _mm_movehl_ps(lo23, lo01);
    const __m128 s2 = _mm_movelh_ps(hi01, hi23);
    const __m128 s3 = _mm_movehl_ps(hi23, hi01);
    _mm_storeu_ps(out, pair_half(pair_half(s0, s1), pair_half(s2, s3)));
}

template <bool WantMax>
inline void store_four_double(const Vectorized<double>& u0,
                              const Vectorized<double>& u1,
                              const Vectorized<double>& u2,
                              const Vectorized<double>& u3, double* out) {
    auto pair = [](const __m256d& a, const __m256d& b) {
        return WantMax ? _mm256_max_pd(a, b) : _mm256_min_pd(a, b);
    };
    const __m256d long0 = pair(u0, _mm256_permute2f128_pd(u0, u0, 0x1));
    const __m256d long1 = pair(u1, _mm256_permute2f128_pd(u1, u1, 0x1));
    const __m256d long2 = pair(u2, _mm256_permute2f128_pd(u2, u2, 0x1));
    const __m256d long3 = pair(u3, _mm256_permute2f128_pd(u3, u3, 0x1));
    const __m256d s0 = _mm256_permute2f128_pd(
        _mm256_unpacklo_pd(long0, long1), _mm256_unpacklo_pd(long2, long3),
        0x20);
    const __m256d s1 = _mm256_permute2f128_pd(
        _mm256_unpackhi_pd(long0, long1), _mm256_unpackhi_pd(long2, long3),
        0x20);
    _mm256_storeu_pd(out, pair(s0, s1));
}
#endif

template <typename ctype, bool WantMax>
inline void fold_short_rows(const ctype* data, int64_t run, int64_t begin,
                            int64_t end, ctype* out, int64_t last_row) {
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    if (run < lanes) {
        fold_step_rows<ctype, WantMax>(data, run, begin, end, out, last_row);
        return;
    }
    const Vec identity(extrema_identity<ctype>(WantMax));
    // The lanes of a row that its run does not fill are replaced by the
    // identity: only the tensor's own last row needs a copy to read, and every
    // other row would otherwise read the next row's values.
    const int64_t tail = run - (run / lanes) * lanes;
    alignas(64) unsigned char tail_bits[lanes * sizeof(ctype)];
    std::memset(tail_bits, 0, sizeof(tail_bits));
    std::memset(tail_bits, 0xFF, static_cast<size_t>(tail) * sizeof(ctype));
    const Vec tail_keep =
        Vec::loadu(reinterpret_cast<const ctype*>(tail_bits));
    const int64_t tail_at = run - tail;
    auto load_tail = [&](const ctype* p, bool pad) {
        if (!pad) return Vec::blendv(identity, Vec::loadu(p + tail_at),
                                     tail_keep);
        alignas(64) ctype copy[lanes];
        for (int64_t k = 0; k < tail; ++k) copy[k] = p[tail_at + k];
        for (int64_t k = tail; k < lanes; ++k) {
            copy[k] = static_cast<ctype>(extrema_identity<ctype>(WantMax));
        }
        return Vec::loadu(copy);
    };
    int64_t r = begin;
    for (; r + 4 <= end; r += 4) {
        const ctype* p0 = data + (r + 0) * run;
        const ctype* p1 = data + (r + 1) * run;
        const ctype* p2 = data + (r + 2) * run;
        const ctype* p3 = data + (r + 3) * run;
        // Two accumulators per row, named rather than indexed: an array of
        // vectors in this loop is spilled to the stack and read back.
        Vec a0 = identity, a1 = identity, a2 = identity, a3 = identity;
        Vec b0 = identity, b1 = identity, b2 = identity, b3 = identity;
        int64_t i = 0;
        for (; i + 2 * lanes <= run; i += 2 * lanes) {
            a0 = extremum_vec_nan<WantMax>(a0, Vec::loadu(p0 + i));
            a1 = extremum_vec_nan<WantMax>(a1, Vec::loadu(p1 + i));
            a2 = extremum_vec_nan<WantMax>(a2, Vec::loadu(p2 + i));
            a3 = extremum_vec_nan<WantMax>(a3, Vec::loadu(p3 + i));
            b0 = extremum_vec_nan<WantMax>(b0, Vec::loadu(p0 + i + lanes));
            b1 = extremum_vec_nan<WantMax>(b1, Vec::loadu(p1 + i + lanes));
            b2 = extremum_vec_nan<WantMax>(b2, Vec::loadu(p2 + i + lanes));
            b3 = extremum_vec_nan<WantMax>(b3, Vec::loadu(p3 + i + lanes));
        }
        if (i + lanes <= run) {
            a0 = extremum_vec_nan<WantMax>(a0, Vec::loadu(p0 + i));
            a1 = extremum_vec_nan<WantMax>(a1, Vec::loadu(p1 + i));
            a2 = extremum_vec_nan<WantMax>(a2, Vec::loadu(p2 + i));
            a3 = extremum_vec_nan<WantMax>(a3, Vec::loadu(p3 + i));
        }
        Vec v0 = extremum_vec_nan<WantMax>(a0, b0);
        Vec v1 = extremum_vec_nan<WantMax>(a1, b1);
        Vec v2 = extremum_vec_nan<WantMax>(a2, b2);
        Vec v3 = extremum_vec_nan<WantMax>(a3, b3);
        if (tail > 0) {
            v0 = extremum_vec_nan<WantMax>(v0, load_tail(p0, r + 0 == last_row));
            v1 = extremum_vec_nan<WantMax>(v1, load_tail(p1, r + 1 == last_row));
            v2 = extremum_vec_nan<WantMax>(v2, load_tail(p2, r + 2 == last_row));
            v3 = extremum_vec_nan<WantMax>(v3, load_tail(p3, r + 3 == last_row));
        }
#if defined(CPU_CAPABILITY_AVX2)
        if constexpr (std::is_same_v<ctype, float> ||
                      std::is_same_v<ctype, double>) {
            if ((nan_lanes(v0) | nan_lanes(v1) | nan_lanes(v2) | nan_lanes(v3)) ==
                0) {
                if constexpr (std::is_same_v<ctype, float>) {
                    store_four_float<WantMax>(v0, v1, v2, v3, out + r);
                } else {
                    store_four_double<WantMax>(v0, v1, v2, v3, out + r);
                }
                continue;
            }
        }
#endif
        out[r + 0] = short_row_horizontal<ctype, WantMax>(v0, p0, run, run);
        out[r + 1] = short_row_horizontal<ctype, WantMax>(v1, p1, run, run);
        out[r + 2] = short_row_horizontal<ctype, WantMax>(v2, p2, run, run);
        out[r + 3] = short_row_horizontal<ctype, WantMax>(v3, p3, run, run);
    }
    for (; r < end; ++r) {
        const ctype* p = data + r * run;
        Vec a = identity, b = identity;
        int64_t i = 0;
        for (; i + 2 * lanes <= run; i += 2 * lanes) {
            a = extremum_vec_nan<WantMax>(a, Vec::loadu(p + i));
            b = extremum_vec_nan<WantMax>(b, Vec::loadu(p + i + lanes));
        }
        if (i + lanes <= run) {
            a = extremum_vec_nan<WantMax>(a, Vec::loadu(p + i));
        }
        Vec v = extremum_vec_nan<WantMax>(a, b);
        if (tail > 0) v = extremum_vec_nan<WantMax>(v, load_tail(p, r == last_row));
        out[r] = short_row_horizontal<ctype, WantMax>(v, p, run, run);
    }
}

// A packed row shorter than one widened step: the row is widened from where it
// lies with the lanes past its run replaced by the identity, so the compare and
// its horizontal fold run on eight single-precision lanes rather than one
// element at a time.  A row that holds a not-a-number takes the exact walk, and
// the tensor's own last row reads through a padded copy.
#if defined(TP_VECREDUCED_X86)
template <bool BFloat, bool WantMax>
__attribute__((target("avx2,fma,f16c"), noinline))
void fold_step_rows_hw(const uint16_t* in, int64_t run, int64_t begin,
                       int64_t end, uint16_t* out, int64_t last_row) {
    alignas(32) int32_t keep_bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t k = 0; k < run; ++k) keep_bits[k] = -1;
    const __m256 keep =
        _mm256_castsi256_ps(_mm256_load_si256(
            reinterpret_cast<const __m256i*>(keep_bits)));
    const __m256 identity = _mm256_set1_ps(
        WantMax ? -std::numeric_limits<float>::infinity()
                : std::numeric_limits<float>::infinity());
    using Packed = std::conditional_t<BFloat, tensorplay::BFloat16,
                                      tensorplay::Half>;
    for (int64_t r = begin; r < end; ++r) {
        const uint16_t* p = in + r * run;
        __m256 value;
        if (r == last_row) {
            alignas(16) uint16_t pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int64_t k = 0; k < run; ++k) pad[k] = p[k];
            value = widen_row8<BFloat>(pad);
        } else {
            value = widen_row8<BFloat>(p);
        }
        if (_mm256_movemask_ps(_mm256_and_ps(
                _mm256_cmp_ps(value, value, _CMP_UNORD_Q), keep)) != 0) {
            const Packed* spelled = reinterpret_cast<const Packed*>(p);
            Packed best = spelled[0];
            for (int64_t k = 1; k < run; ++k) {
                best = WantMax ? nan_max(best, spelled[k])
                               : nan_min(best, spelled[k]);
            }
            reinterpret_cast<Packed*>(out)[r] = best;
            continue;
        }
        const __m256 held = _mm256_blendv_ps(identity, value, keep);
        __m256 folded = _mm256_permute2f128_ps(held, held, 0x1);
        folded = WantMax ? _mm256_max_ps(held, folded)
                         : _mm256_min_ps(held, folded);
        const __m256 swapped = _mm256_shuffle_ps(folded, folded, 0x4E);
        folded = WantMax ? _mm256_max_ps(folded, swapped)
                         : _mm256_min_ps(folded, swapped);
        const __m256 flipped = _mm256_shuffle_ps(folded, folded, 0xB1);
        folded = WantMax ? _mm256_max_ps(folded, flipped)
                         : _mm256_min_ps(folded, flipped);
        const float answer = _mm256_cvtss_f32(folded);
        out[r] = BFloat ? tensorplay::cpu::vecreduced::narrow_bf16_one(answer)
                        : tensorplay::cpu::vecreduced::narrow_one_hw(answer);
    }
}

// One band of a row whose run does not fill it: the lanes past the run belong to
// whatever follows in the buffer, so they are replaced by the identity before
// the band is compared.  The tensor's own last row instead reads through a copy
// whose remaining lanes repeat one of the row's own values.
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
__m256 fold_kept_band_hw(const uint16_t* p, int64_t keep, const __m256& ident,
                         const __m256& drop, bool pad) {
    if (pad) {
        alignas(16) uint16_t copy[8];
        std::memcpy(copy, p, static_cast<size_t>(keep) * sizeof(uint16_t));
        for (int64_t k = keep; k < 8; ++k) copy[k] = copy[0];
        return widen_row8<BFloat>(copy);
    }
    return _mm256_blendv_ps(widen_row8<BFloat>(p), ident, drop);
}

// One packed row of a few dozen columns, widened a band at a time and compared
// in single precision.  A run narrower than one band is that one kept band.
template <bool BFloat, bool WantMax>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
__m256 fold_row_bands_hw(const uint16_t* p, int64_t wide, int64_t rem,
                         const __m256& ident, const __m256& drop, bool pad) {
    if (wide == 0) {
        return fold_kept_band_hw<BFloat>(p, rem, ident, drop, pad);
    }
    __m256 a = widen_row8<BFloat>(p);
    for (int64_t c = 8; c < wide; c += 8) {
        a = extremum_step8<WantMax>(a, widen_row8<BFloat>(p + c));
    }
    if (rem > 0) {
        a = extremum_step8<WantMax>(
            a, fold_kept_band_hw<BFloat>(p + wide, rem, ident, drop, pad));
    }
    return a;
}

template <bool WantMax>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
float horizontal_extremum8(const __m256& v) {
    __m256 folded = WantMax
        ? _mm256_max_ps(v, _mm256_permute2f128_ps(v, v, 0x1))
        : _mm256_min_ps(v, _mm256_permute2f128_ps(v, v, 0x1));
    const __m256 swapped = _mm256_shuffle_ps(folded, folded, 0x4E);
    folded = WantMax ? _mm256_max_ps(folded, swapped)
                     : _mm256_min_ps(folded, swapped);
    const __m256 flipped = _mm256_shuffle_ps(folded, folded, 0xB1);
    folded = WantMax ? _mm256_max_ps(folded, flipped)
                     : _mm256_min_ps(folded, flipped);
    return _mm256_cvtss_f32(folded);
}

// Four packed rows of a few dozen columns at a time: each row's bands fold into
// one single-precision lane, and the answer narrows once.  A row that holds a
// not-a-number takes the exact walk, since a horizontal fold drops one from the
// lane it sits in when a later lane wins the compare.
template <bool BFloat, bool WantMax>
__attribute__((target("avx2,fma,f16c"), noinline))
void fold_short_rows_hw(const uint16_t* in, int64_t run, int64_t begin,
                        int64_t end, uint16_t* out, int64_t last_row) {
    using Packed = std::conditional_t<BFloat, tensorplay::BFloat16,
                                      tensorplay::Half>;
    const int64_t wide = run & ~static_cast<int64_t>(7);
    const int64_t rem = run - wide;
    // A band that a run leaves partly filled keeps its first `rem` lanes and
    // drops the rest; the dropped lanes take the identity of the wanted
    // direction so that they cannot win the compare.
    const int64_t keep = rem > 0 ? rem : 8;
    const __m256 ident = _mm256_set1_ps(
        WantMax ? -std::numeric_limits<float>::infinity()
                : std::numeric_limits<float>::infinity());
    alignas(32) int32_t dropbits[8];
    for (int64_t k = 0; k < 8; ++k) dropbits[k] = k < keep ? 0 : -1;
    const __m256 drop = _mm256_castsi256_ps(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(dropbits)));
    int64_t r = begin;
    for (; r + 4 <= end; r += 4) {
        const uint16_t* p0 = in + (r + 0) * run;
        const uint16_t* p1 = in + (r + 1) * run;
        const uint16_t* p2 = in + (r + 2) * run;
        const uint16_t* p3 = in + (r + 3) * run;
        __m256 a0 = fold_row_bands_hw<BFloat, WantMax>(p0, wide, rem, ident,
                                                      drop, false);
        __m256 a1 = fold_row_bands_hw<BFloat, WantMax>(p1, wide, rem, ident,
                                                      drop, false);
        __m256 a2 = fold_row_bands_hw<BFloat, WantMax>(p2, wide, rem, ident,
                                                      drop, false);
        __m256 a3 = fold_row_bands_hw<BFloat, WantMax>(p3, wide, rem, ident,
                                                      drop, r + 3 == last_row);
        const uint16_t* rows[4] = {p0, p1, p2, p3};
        __m256 held[4] = {a0, a1, a2, a3};
        for (int64_t k = 0; k < 4; ++k) {
            if (_mm256_movemask_ps(
                    _mm256_cmp_ps(held[k], held[k], _CMP_UNORD_Q)) != 0) {
                reinterpret_cast<Packed*>(out)[r + k] = WantMax
                    ? fold_run_extremum_short<Packed, true>(
                          reinterpret_cast<const Packed*>(rows[k]), run)
                    : fold_run_extremum_short<Packed, false>(
                          reinterpret_cast<const Packed*>(rows[k]), run);
            } else {
                const float answer = horizontal_extremum8<WantMax>(held[k]);
                out[r + k] = BFloat
                    ? tensorplay::cpu::vecreduced::narrow_bf16_one(answer)
                    : tensorplay::cpu::vecreduced::narrow_one_hw(answer);
            }
        }
    }
    for (; r < end; ++r) {
        const uint16_t* p = in + r * run;
        const __m256 a = fold_row_bands_hw<BFloat, WantMax>(
            p, wide, rem, ident, drop, r == last_row);
        if (_mm256_movemask_ps(_mm256_cmp_ps(a, a, _CMP_UNORD_Q)) != 0) {
            reinterpret_cast<Packed*>(out)[r] = WantMax
                ? fold_run_extremum_short<Packed, true>(
                      reinterpret_cast<const Packed*>(p), run)
                : fold_run_extremum_short<Packed, false>(
                      reinterpret_cast<const Packed*>(p), run);
        } else {
            const float answer = horizontal_extremum8<WantMax>(a);
            out[r] = BFloat
                ? tensorplay::cpu::vecreduced::narrow_bf16_one(answer)
                : tensorplay::cpu::vecreduced::narrow_one_hw(answer);
        }
    }
}

#endif  // TP_VECREDUCED_X86

// A single trailing dimension of a dense tensor is one contiguous run per row,
// so the rows can go straight to the pool and skip the reduction iterator with
// its identity fill and its per-element stride machinery.  Answers agree with
// the iterator because both fold the same run with the same compare.
// An integral run folded with a plain compare.  There is no not-a-number to
// watch for and the answer is carried in the run's own width, so the seed is
// the run's first value: no sentinel is needed, and a 64-bit run never passes
// through a wider type that could not carry it.  Four running bests keep the
// compare from being a single dependent chain.
template <typename ctype, bool WantMax>
inline ctype fold_run_integral(const ctype* data, int64_t n) {
    auto pick = [](ctype a, ctype b) {
        return WantMax ? (a < b ? b : a) : (b < a ? b : a);
    };
    if (n <= 0) return extrema_identity<ctype>(WantMax);
    ctype b0 = data[0];
    int64_t i = 1;
    if (n >= 4) {
        ctype b1 = data[1], b2 = data[2], b3 = data[3];
        i = 4;
        for (; i + 4 <= n; i += 4) {
            b0 = pick(b0, data[i + 0]);
            b1 = pick(b1, data[i + 1]);
            b2 = pick(b2, data[i + 2]);
            b3 = pick(b3, data[i + 3]);
        }
        b0 = pick(pick(b0, b1), pick(b2, b3));
    }
    for (; i < n; ++i) b0 = pick(b0, data[i]);
    return b0;
}

#if defined(CPU_CAPABILITY_AVX2)
// This copy is compiled with its ISA flags for the whole translation unit, so
// a per-function target only restates them; the two inlining forms are spelled
// for compilers without the GNU attribute.
#if defined(_MSC_VER)
#define TP_REDUCTION_AVX2_INLINE __forceinline
#define TP_REDUCTION_AVX2_NOINLINE __declspec(noinline)
#else
#define TP_REDUCTION_AVX2_INLINE \
    __attribute__((target("avx2,fma,f16c"), always_inline)) inline
#define TP_REDUCTION_AVX2_NOINLINE \
    __attribute__((target("avx2,fma,f16c"), noinline))
#endif

// One extremum step between two registers of any integral width, chosen at
// compile time so the sign of the element and the instruction exist.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_INLINE
__m256i extrema_reg_step(const __m256i& a, const __m256i& b) {
    if constexpr (sizeof(ctype) == 1) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm256_max_epi8(a, b) : _mm256_min_epi8(a, b);
        } else {
            return WantMax ? _mm256_max_epu8(a, b) : _mm256_min_epu8(a, b);
        }
    } else if constexpr (sizeof(ctype) == 2) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm256_max_epi16(a, b) : _mm256_min_epi16(a, b);
        } else {
            return WantMax ? _mm256_max_epu16(a, b) : _mm256_min_epu16(a, b);
        }
    } else if constexpr (sizeof(ctype) == 4) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm256_max_epi32(a, b) : _mm256_min_epi32(a, b);
        } else {
            return WantMax ? _mm256_max_epu32(a, b) : _mm256_min_epu32(a, b);
        }
    } else {
        __m256i x = a;
        __m256i y = b;
        if constexpr (std::is_unsigned_v<ctype>) {
            const __m256i sign =
                _mm256_set1_epi64x(std::numeric_limits<int64_t>::min());
            x = _mm256_xor_si256(x, sign);
            y = _mm256_xor_si256(y, sign);
        }
        const __m256i take =
            WantMax ? _mm256_cmpgt_epi64(y, x) : _mm256_cmpgt_epi64(x, y);
        return _mm256_blendv_epi8(a, b, take);
    }
}

// The identity of an integral extremum, broadcast in the width it is carried in.
template <typename ctype>
TP_REDUCTION_AVX2_INLINE
__m256i integral_seed(bool maximum_value) {
    const ctype value = extrema_identity<ctype>(maximum_value);
    if constexpr (sizeof(ctype) == 1) {
        return _mm256_set1_epi8(static_cast<char>(value));
    } else if constexpr (sizeof(ctype) == 2) {
        return _mm256_set1_epi16(static_cast<int16_t>(value));
    } else if constexpr (sizeof(ctype) == 4) {
        return _mm256_set1_epi32(static_cast<int32_t>(value));
    } else {
        return _mm256_set1_epi64x(static_cast<int64_t>(value));
    }
}

// One step of an integral extremum in the width's own registers: the width's
// own max or min instruction where the ISA has one, and a compare and a blend
// for the eight-byte widths, which the ISA orders only as signed values; an
// unsigned eight-byte value flips its sign bit before the compare and comes
// back unchanged.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_INLINE
__m256i integral_step(const __m256i& a, const __m256i& b) {
    if constexpr (sizeof(ctype) == 1) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm256_max_epi8(a, b) : _mm256_min_epi8(a, b);
        } else {
            return WantMax ? _mm256_max_epu8(a, b) : _mm256_min_epu8(a, b);
        }
    } else if constexpr (sizeof(ctype) == 2) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm256_max_epi16(a, b) : _mm256_min_epi16(a, b);
        } else {
            return WantMax ? _mm256_max_epu16(a, b) : _mm256_min_epu16(a, b);
        }
    } else if constexpr (sizeof(ctype) == 4) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm256_max_epi32(a, b) : _mm256_min_epi32(a, b);
        } else {
            return WantMax ? _mm256_max_epu32(a, b) : _mm256_min_epu32(a, b);
        }
    } else {
        __m256i x = a;
        __m256i y = b;
        if constexpr (std::is_unsigned_v<ctype>) {
            const __m256i sign =
                _mm256_set1_epi64x(std::numeric_limits<int64_t>::min());
            x = _mm256_xor_si256(x, sign);
            y = _mm256_xor_si256(y, sign);
        }
        const __m256i take_b =
            WantMax ? _mm256_cmpgt_epi64(y, x) : _mm256_cmpgt_epi64(x, y);
        return _mm256_blendv_epi8(a, b, take_b);
    }
}

// One step of an integral extremum over half a register.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_INLINE
__m128i integral_step128(const __m128i& a, const __m128i& b) {
    if constexpr (sizeof(ctype) == 1) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm_max_epi8(a, b) : _mm_min_epi8(a, b);
        } else {
            return WantMax ? _mm_max_epu8(a, b) : _mm_min_epu8(a, b);
        }
    } else if constexpr (sizeof(ctype) == 2) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm_max_epi16(a, b) : _mm_min_epi16(a, b);
        } else {
            return WantMax ? _mm_max_epu16(a, b) : _mm_min_epu16(a, b);
        }
    } else if constexpr (sizeof(ctype) == 4) {
        if constexpr (std::is_signed_v<ctype>) {
            return WantMax ? _mm_max_epi32(a, b) : _mm_min_epi32(a, b);
        } else {
            return WantMax ? _mm_max_epu32(a, b) : _mm_min_epu32(a, b);
        }
    } else {
        __m128i x = a;
        __m128i y = b;
        if constexpr (std::is_unsigned_v<ctype>) {
            const __m128i sign =
                _mm_set1_epi64x(std::numeric_limits<int64_t>::min());
            x = _mm_xor_si128(x, sign);
            y = _mm_xor_si128(y, sign);
        }
        const __m128i take_b =
            WantMax ? _mm_cmpgt_epi64(y, x) : _mm_cmpgt_epi64(x, y);
        return _mm_blendv_epi8(a, b, take_b);
    }
}

// The lanes of a register reduced to their one extremum without leaving the
// registers: the halves are compared against each other until a single lane
// holds the answer.  A walk that reads its answer back from memory pays a
// store-forwarding step for every row it answers.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_INLINE
ctype integral_horizontal(const __m256i& v) {
    __m128i m = integral_step128<ctype, WantMax>(
        _mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    if constexpr (sizeof(ctype) == 8) {
        // One lane already holds the answer: the two halves are four bytes.
        const __m128i hi = _mm_unpackhi_epi64(m, m);
        const ctype answer = static_cast<ctype>(_mm_cvtsi128_si64(
            integral_step128<ctype, WantMax>(m, hi)));
        return answer;
    }
    m = integral_step128<ctype, WantMax>(m, _mm_shuffle_epi32(m, 0x4E));
    m = integral_step128<ctype, WantMax>(m, _mm_shuffle_epi32(m, 0xB1));
    if constexpr (sizeof(ctype) == 4) {
        return static_cast<ctype>(_mm_cvtsi128_si32(m));
    } else if constexpr (sizeof(ctype) == 2) {
        m = integral_step128<ctype, WantMax>(m, _mm_srli_si128(m, 2));
        return static_cast<ctype>(_mm_extract_epi16(m, 0));
    } else {
        m = integral_step128<ctype, WantMax>(m, _mm_srli_si128(m, 2));
        m = integral_step128<ctype, WantMax>(m, _mm_srli_epi16(m, 8));
        return static_cast<ctype>(_mm_cvtsi128_si32(m) & 0xFF);
    }
}

// A trailing run of any integral width folded with four register chains, so a
// row keeps its compares in flight instead of waiting on one dependent chain,
// and the answer is reduced from the registers instead of being read back from
// a slice written once per row.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_NOINLINE
void fold_integral_rows_reg(const ctype* in, int64_t run, int64_t begin,
                            int64_t end, ctype* out) {
    constexpr int64_t W = 32 / sizeof(ctype);
    for (int64_t r = begin; r < end; ++r) {
        const ctype* p = in + r * run;
        int64_t i = 0;
        ctype best = extrema_identity<ctype>(WantMax);
        if (run >= W) {
            __m256i a0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(p));
            __m256i a1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(p + W));
            __m256i a2 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(p + 2 * W));
            __m256i a3 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(p + 3 * W));
            i = 4 * W;
            for (; i + 4 * W <= run; i += 4 * W) {
                a0 = extrema_reg_step<ctype, WantMax>(
                    a0, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(p + i)));
                a1 = extrema_reg_step<ctype, WantMax>(
                    a1, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(p + i + W)));
                a2 = extrema_reg_step<ctype, WantMax>(
                    a2, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(p + i + 2 * W)));
                a3 = extrema_reg_step<ctype, WantMax>(
                    a3, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(p + i + 3 * W)));
            }
            for (; i + W <= run; i += W) {
                a0 = extrema_reg_step<ctype, WantMax>(
                    a0, _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(p + i)));
            }
            const __m256i held = extrema_reg_step<ctype, WantMax>(
                extrema_reg_step<ctype, WantMax>(a0, a1),
                extrema_reg_step<ctype, WantMax>(a2, a3));
            best = integral_horizontal<ctype, WantMax>(held);
        }
        const int64_t tail = run - i;
        for (int64_t k = 0; k < tail; ++k) {
            best = WantMax ? (best < p[i + k] ? p[i + k] : best)
                           : (p[i + k] < best ? p[i + k] : best);
        }
        out[r] = best;
    }
}

// A trailing run shorter than four chains: one or two vectors hold the whole
// row, so the fold needs no loop over the row at all and no staging slice.
// The tail past the last whole vector is compared scalar-wise against the
// answer reduced from the registers, which keeps the result exact.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_NOINLINE
void fold_integral_rows_narrow(const ctype* in, int64_t run, int64_t begin,
                               int64_t end, ctype* out) {
    constexpr int64_t W = 32 / sizeof(ctype);
    for (int64_t r = begin; r < end; ++r) {
        const ctype* p = in + r * run;
        int64_t i = 0;
        __m256i acc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        i = W;
        int64_t vectors = 1;
        for (; vectors < 4 && i + W <= run; ++vectors, i += W) {
            const __m256i v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            acc = extrema_reg_step<ctype, WantMax>(acc, v);
        }
        __m256i a0 = acc;
        ctype best = integral_horizontal<ctype, WantMax>(a0);
        for (; i < run; ++i) {
            const ctype v = p[i];
            best = WantMax ? (best < v ? v : best) : (v < best ? v : best);
        }
        out[r] = best;
    }
}


// The rows of a run answered with four accumulators so that the compare is not
// a single dependent chain; the lanes are combined once per row and the
// columns past the last full band are compared one at a time.
template <typename ctype, bool WantMax>
TP_REDUCTION_AVX2_NOINLINE
void fold_integral_rows(const ctype* in, int64_t run, int64_t begin,
                        int64_t end, ctype* out) {
    constexpr int64_t W = 32 / sizeof(ctype);
    const __m256i seed = integral_seed<ctype>(WantMax);
    auto load = [](const ctype* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    };
    auto pick = [](ctype a, ctype b) {
        return WantMax ? (a < b ? b : a) : (b < a ? b : a);
    };
    for (int64_t r = begin; r < end; ++r) {
        const ctype* p = in + r * run;
        int64_t i = 0;
        ctype best = extrema_identity<ctype>(WantMax);
        // A run shorter than one band has nothing to fold wide, and reading one
        // anyway would compare a band of the identity against the handful of
        // values the row really holds.
        if (run >= W) {
            __m256i a0 = seed, a1 = seed, a2 = seed, a3 = seed;
            for (; i + 4 * W <= run; i += 4 * W) {
                a0 = integral_step<ctype, WantMax>(a0, load(p + i));
                a1 = integral_step<ctype, WantMax>(a1, load(p + i + W));
                a2 = integral_step<ctype, WantMax>(a2, load(p + i + 2 * W));
                a3 = integral_step<ctype, WantMax>(a3, load(p + i + 3 * W));
            }
            for (; i + W <= run; i += W) {
                a0 = integral_step<ctype, WantMax>(a0, load(p + i));
            }
            const __m256i held = integral_step<ctype, WantMax>(
                integral_step<ctype, WantMax>(a0, a1),
                integral_step<ctype, WantMax>(a2, a3));
            best = pick(best, integral_horizontal<ctype, WantMax>(held));
        }
        for (; i < run; ++i) best = pick(best, p[i]);
        out[r] = best;
    }
}
#undef TP_REDUCTION_AVX2_INLINE
#undef TP_REDUCTION_AVX2_NOINLINE
#endif

// One extremum step on a vector of any width: an integral width compares in its
// own registers, everything else follows the rule that carries a not-a-number.
template <typename T, bool WantMax>
inline Vectorized<T> extremum_vec_step(const Vectorized<T>& a,
                                       const Vectorized<T>& b) {
#if defined(CPU_CAPABILITY_AVX2)
    // Only the widths a vector register holds a whole number of times carry
    // their own comparison; the rest keep the scalar rule below.
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
                  std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
                  std::is_same_v<T, int64_t>) {
        return Vectorized<T>(integral_step<T, WantMax>(
            static_cast<__m256i>(a), static_cast<__m256i>(b)));
    }
#endif
    return WantMax ? maximum(a, b) : minimum(a, b);
}

template <typename ctype>
bool fold_trailing_extrema(Tensor& result, const Tensor& self, int64_t ndim,
                           const std::vector<int64_t>& resolved,
                           bool maximum_value) {
    if (resolved.size() != 1 || ndim < 1 || resolved[0] != ndim - 1) return false;
    if (!self.is_contiguous()) return false;
    const int64_t run = self.size(-1);
    if (run <= 0) return false;
    const int64_t rows = self.numel() / run;
    if (rows * run != self.numel() || result.numel() != rows) return false;

    const ctype* data = self.data_ptr<ctype>();
    ctype* out = result.data_ptr<ctype>();
    constexpr bool kReduced = std::is_same_v<ctype, tensorplay::Half> ||
                              std::is_same_v<ctype, tensorplay::BFloat16>;
    constexpr bool kReal = std::is_same_v<ctype, float> ||
                           std::is_same_v<ctype, double>;
    if constexpr (kReduced) {
#if defined(__x86_64__)
        // A reduced width folds the wide extremes row by row and reads the
        // answer back from the layout, which is the element the flat fold
        // would keep.
        // A short run pays for the per-row walk, so the fold there is done
        // wide and the holder read back; a long run streams better through
        // the flat fold the caller falls back to.
        if (!tensorplay::cpu::vecreduced::f16c_available()) return false;
        constexpr bool kBFloat = std::is_same_v<ctype, tensorplay::BFloat16>;
        const uint16_t* bits = reinterpret_cast<const uint16_t*>(data);
        uint16_t* obits = reinterpret_cast<uint16_t*>(out);
        if (run < 8) {
            auto fold_steps = [&](int64_t begin, int64_t end) {
                if (maximum_value) {
                    fold_step_rows_hw<kBFloat, true>(bits, run, begin, end,
                                                     obits, rows - 1);
                } else {
                    fold_step_rows_hw<kBFloat, false>(bits, run, begin, end,
                                                      obits, rows - 1);
                }
            };
            if (self.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
                in_parallel_region()) {
                fold_steps(0, rows);
            } else {
                parallel_for(0, rows, 1, fold_steps);
            }
            return true;
        }
        if (run > 512) return false;
        auto fold_bits = [&](int64_t begin, int64_t end) {
            if (maximum_value) {
                fold_short_rows_hw<kBFloat, true>(bits, run, begin, end, obits,
                                                  rows - 1);
            } else {
                fold_short_rows_hw<kBFloat, false>(bits, run, begin, end, obits,
                                                   rows - 1);
            }
        };
        if (self.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            fold_bits(0, rows);
        } else {
            parallel_for(0, rows, 1, fold_bits);
        }
        return true;
#else
        return false;
#endif
    } else if constexpr (kReal) {
        // A short run folds four rows at once so no row waits on its own
        // chain alone; a long run streams better through the wide ladder.
        const bool short_run = run <= 512;
        auto fold_rows = [&](int64_t begin, int64_t end) {
            if (short_run) {
                if (maximum_value) {
                    fold_short_rows<ctype, true>(data, run, begin, end, out,
                                                 rows - 1);
                } else {
                    fold_short_rows<ctype, false>(data, run, begin, end, out,
                                                  rows - 1);
                }
                return;
            }
            for (int64_t r = begin; r < end; ++r) {
                out[r] = maximum_value
                    ? fold_run_extremum<ctype, true>(data + r * run, run)
                    : fold_run_extremum<ctype, false>(data + r * run, run);
            }
        };
        if (self.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            fold_rows(0, rows);
        } else {
            parallel_for(0, rows, 1, fold_rows);
        }
        return true;
    } else if constexpr (std::is_integral_v<ctype>) {
        // An integral row has no not-a-number to look for, so it folds with a
        // plain compare and needs no identity at all; each row is read as one
        // run rather than element by element.
#if defined(CPU_CAPABILITY_AVX2)
        // Four register chains fold the row without a round trip through a
        // staging slice, so a run of wide elements streams at the speed of its
        // compares.  A run too short to fill four chains keeps the memory
        // fold, which then touches no stack at all.
        const int64_t W = 32 / sizeof(ctype);
        const bool narrow_run = run >= W && run < 8 * W;
        auto fold_rows = [&](int64_t begin, int64_t end) {
            if (run < W) {
                if (maximum_value) {
                    fold_integral_rows<ctype, true>(data, run, begin, end, out);
                } else {
                    fold_integral_rows<ctype, false>(data, run, begin, end, out);
                }
                return;
            }
            // Up to four whole vectors are folded without a loop over the
            // vectors; past that the four-chain walk pays for itself.
            if (narrow_run) {
                if (maximum_value) {
                    fold_integral_rows_narrow<ctype, true>(data, run, begin, end,
                                                           out);
                } else {
                    fold_integral_rows_narrow<ctype, false>(data, run, begin, end,
                                                            out);
                }
                return;
            }
            if (maximum_value) {
                fold_integral_rows_reg<ctype, true>(data, run, begin, end, out);
            } else {
                fold_integral_rows_reg<ctype, false>(data, run, begin, end, out);
            }
        };
#else
        auto fold_rows = [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                out[r] = maximum_value
                    ? fold_run_integral<ctype, true>(data + r * run, run)
                    : fold_run_integral<ctype, false>(data + r * run, run);
            }
        };
#endif
        if (self.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            fold_rows(0, rows);
        } else {
            parallel_for(0, rows, 1, fold_rows);
        }
        return true;
    } else {
        return false;
    }
}

// Every dimension of a dense tensor reduced away leaves one contiguous run over
// the whole buffer and one answer.  The run is cut into a chunk per thread, each
// chunk folded into its own slot, and the slots combined at the end: an
// extremum of chunk extrema is the extremum, and a chunk that met a NaN hands
// one to the combine, so no identity value enters anywhere.
template <typename ctype>
bool fold_whole_extrema(Tensor& result, const Tensor& self, int64_t ndim,
                        const std::vector<int64_t>& resolved,
                        bool maximum_value) {
    if (ndim < 1 || resolved.size() != static_cast<size_t>(ndim)) return false;
    if (!self.is_contiguous()) return false;
    const int64_t n = self.numel();
    if (n <= 0 || result.numel() != 1) return false;

    const ctype* data = self.data_ptr<ctype>();
    // A chunk below a grain is not worth a slot: the split would cost more than
    // the walk it saves.
    const int64_t chunks = std::min<int64_t>(
        get_num_threads(), (n + GRAIN_SIZE - 1) / GRAIN_SIZE);
    const int64_t per = (n + chunks - 1) / chunks;
    // A plain array rather than a std::vector: a one-bit element type would
    // hand back a proxy reference, and the answers here are read as values.
    std::unique_ptr<ctype[]> parts(new ctype[static_cast<size_t>(chunks)]);
    auto fold_chunks = [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            const int64_t lo = c * per;
            const int64_t hi = std::min(lo + per, n);
            if constexpr (std::is_integral_v<ctype> &&
                          !std::is_same_v<ctype, bool>) {
#if defined(CPU_CAPABILITY_AVX2)
                if (maximum_value) {
                    fold_integral_rows<ctype, true>(data + lo, hi - lo, 0, 1,
                                                    &parts[c]);
                } else {
                    fold_integral_rows<ctype, false>(data + lo, hi - lo, 0, 1,
                                                     &parts[c]);
                }
#else
                parts[c] = maximum_value
                    ? fold_run_integral<ctype, true>(data + lo, hi - lo)
                    : fold_run_integral<ctype, false>(data + lo, hi - lo);
#endif
            } else {
                parts[c] = maximum_value
                    ? fold_run_extremum<ctype, true>(data + lo, hi - lo)
                    : fold_run_extremum<ctype, false>(data + lo, hi - lo);
            }
        }
    };
    parallel_for(0, chunks, 1, fold_chunks);

    ctype* out = result.data_ptr<ctype>();
    ctype best = parts[0];
    for (int64_t c = 1; c < chunks; ++c) {
        best = maximum_value ? nan_max(best, parts[c]) : nan_min(best, parts[c]);
    }
    *out = best;
    return true;
}

// A band of eight single-precision answers left at the reduced width in one
// store: the crossing and the move that puts the answers where they lie are
// the two instructions the packed layout costs, instead of one crossing per
// answer through a call that cannot be inlined.
#if defined(TP_VECREDUCED_X86)
template <bool BFloat>
__attribute__((target("avx2,fma,f16c"), always_inline)) inline
void store_band_hw(uint16_t* acc, int64_t j, const __m256& a) {
    if constexpr (BFloat) {
        tensorplay::cpu::vecreduced::store_bf16_8(acc + j, a);
    } else {
        tensorplay::cpu::vecreduced::lay_f16_8<false>(acc + j, a);
    }
}

// The same block fold over packed columns: each band of eight columns is
// widened once per row and compared in single precision, so a compare covers
// eight columns and the packed value is formed only when the band is stored.
// A packed width compares to the same answer in single precision as in its own
// width -- the widening is exact -- so the fold reports what the narrow compare
// would.  Four bands run side by side so that the chain of compares on one band
// is not what the loop waits on.
template <bool BFloat, bool WantMax>
__attribute__((target("avx2,fma,f16c"), noinline))
void fold_leading_block_hw(const uint16_t* in, uint16_t* acc, int64_t cols,
                           int64_t row_begin, int64_t n) {
    const int64_t wide = cols & ~static_cast<int64_t>(7);
    const uint16_t* seed = in + row_begin * cols;
    int64_t j = 0;
    for (; j + 32 <= wide; j += 32) {
        __m256 a0 = widen_row8<BFloat>(seed + j);
        __m256 a1 = widen_row8<BFloat>(seed + j + 8);
        __m256 a2 = widen_row8<BFloat>(seed + j + 16);
        __m256 a3 = widen_row8<BFloat>(seed + j + 24);
        int64_t s = 1;
        for (; s + 4 <= n; s += 4) {
            const uint16_t* q0 = in + (row_begin + s + 0) * cols + j;
            const uint16_t* q1 = in + (row_begin + s + 1) * cols + j;
            const uint16_t* q2 = in + (row_begin + s + 2) * cols + j;
            const uint16_t* q3 = in + (row_begin + s + 3) * cols + j;
            const __m256 v00 = widen_row8<BFloat>(q0);
            const __m256 v01 = widen_row8<BFloat>(q0 + 8);
            const __m256 v02 = widen_row8<BFloat>(q0 + 16);
            const __m256 v03 = widen_row8<BFloat>(q0 + 24);
            const __m256 v10 = widen_row8<BFloat>(q1);
            const __m256 v11 = widen_row8<BFloat>(q1 + 8);
            const __m256 v12 = widen_row8<BFloat>(q1 + 16);
            const __m256 v13 = widen_row8<BFloat>(q1 + 24);
            const __m256 v20 = widen_row8<BFloat>(q2);
            const __m256 v21 = widen_row8<BFloat>(q2 + 8);
            const __m256 v22 = widen_row8<BFloat>(q2 + 16);
            const __m256 v23 = widen_row8<BFloat>(q2 + 24);
            const __m256 v30 = widen_row8<BFloat>(q3);
            const __m256 v31 = widen_row8<BFloat>(q3 + 8);
            const __m256 v32 = widen_row8<BFloat>(q3 + 16);
            const __m256 v33 = widen_row8<BFloat>(q3 + 24);
            a0 = extremum_step8<WantMax>(
                a0, extremum_step8<WantMax>(extremum_step8<WantMax>(v00, v10),
                                            extremum_step8<WantMax>(v20, v30)));
            a1 = extremum_step8<WantMax>(
                a1, extremum_step8<WantMax>(extremum_step8<WantMax>(v01, v11),
                                            extremum_step8<WantMax>(v21, v31)));
            a2 = extremum_step8<WantMax>(
                a2, extremum_step8<WantMax>(extremum_step8<WantMax>(v02, v12),
                                            extremum_step8<WantMax>(v22, v32)));
            a3 = extremum_step8<WantMax>(
                a3, extremum_step8<WantMax>(extremum_step8<WantMax>(v03, v13),
                                            extremum_step8<WantMax>(v23, v33)));
        }
        for (; s < n; ++s) {
            const uint16_t* q = in + (row_begin + s) * cols + j;
            a0 = extremum_step8<WantMax>(a0, widen_row8<BFloat>(q));
            a1 = extremum_step8<WantMax>(a1, widen_row8<BFloat>(q + 8));
            a2 = extremum_step8<WantMax>(a2, widen_row8<BFloat>(q + 16));
            a3 = extremum_step8<WantMax>(a3, widen_row8<BFloat>(q + 24));
        }
        store_band_hw<BFloat>(acc, j, a0);
        store_band_hw<BFloat>(acc, j + 8, a1);
        store_band_hw<BFloat>(acc, j + 16, a2);
        store_band_hw<BFloat>(acc, j + 24, a3);
    }
    for (; j < wide; j += 8) {
        __m256 a = widen_row8<BFloat>(seed + j);
        int64_t s = 1;
        for (; s + 4 <= n; s += 4) {
            const uint16_t* q0 = in + (row_begin + s + 0) * cols + j;
            const uint16_t* q1 = in + (row_begin + s + 1) * cols + j;
            const uint16_t* q2 = in + (row_begin + s + 2) * cols + j;
            const uint16_t* q3 = in + (row_begin + s + 3) * cols + j;
            a = extremum_step8<WantMax>(
                a, extremum_step8<WantMax>(
                       extremum_step8<WantMax>(widen_row8<BFloat>(q0),
                                               widen_row8<BFloat>(q1)),
                       extremum_step8<WantMax>(widen_row8<BFloat>(q2),
                                               widen_row8<BFloat>(q3))));
        }
        for (; s < n; ++s) {
            a = extremum_step8<WantMax>(
                a, widen_row8<BFloat>(in + (row_begin + s) * cols + j));
        }
        store_band_hw<BFloat>(acc, j, a);
    }
    if (wide < cols) {
        using Packed = std::conditional_t<BFloat, tensorplay::BFloat16,
                                          tensorplay::Half>;
        const Packed* spelled = reinterpret_cast<const Packed*>(in);
        for (int64_t c = wide; c < cols; ++c) {
            Packed best = spelled[row_begin * cols + c];
            for (int64_t s = 1; s < n; ++s) {
                const Packed value = spelled[(row_begin + s) * cols + c];
                best = WantMax ? nan_max(best, value) : nan_min(best, value);
            }
            reinterpret_cast<Packed*>(acc)[c] = best;
        }
    }
}

#endif  // TP_VECREDUCED_X86

// One block of rows folded into a column slice held by the task that owns it.
// The slice is read once and written once per group of four rows rather than per
// row, which is what keeps the pass behind the input stream instead of behind
// its own accumulator traffic.  The combine answers a tie with its second
// operand, so a group reports its own latest row and a later group replaces what
// an earlier one held; that is the rule the lane-wise walk in the general
// reduction already follows.  Only the sign of a zero is affected, since -0.0
// and 0.0 compare equal.
template <typename ctype, bool WantMax>
void fold_leading_block(const ctype* in, ctype* acc, int64_t cols,
                        int64_t row_begin, int64_t n) {
#if defined(__x86_64__)
    if constexpr (std::is_same_v<ctype, tensorplay::Half> ||
                  std::is_same_v<ctype, tensorplay::BFloat16>) {
        fold_leading_block_hw<std::is_same_v<ctype, tensorplay::BFloat16>,
                              WantMax>(
            reinterpret_cast<const uint16_t*>(in),
            reinterpret_cast<uint16_t*>(acc), cols, row_begin, n);
        return;
    }
#endif
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    auto pair_of = [](const Vec& a, const Vec& b) {
        return extremum_vec_step<ctype, WantMax>(a, b);
    };
    // Seeding from the block's first row keeps an identity value out of the
    // fold, so a block holding a NaN cannot report anything but a NaN.
    const ctype* seed = in + row_begin * cols;
    int64_t j = 0;
    for (; j + lanes <= cols; j += lanes) Vec::loadu(seed + j).store(acc + j);
    for (; j < cols; ++j) acc[j] = seed[j];

    int64_t s = 1;
    for (; s + 4 <= n; s += 4) {
        const ctype* p0 = in + (row_begin + s) * cols;
        const ctype* p1 = p0 + cols;
        const ctype* p2 = p1 + cols;
        const ctype* p3 = p2 + cols;
        int64_t k = 0;
        for (; k + lanes <= cols; k += lanes) {
            const Vec low = pair_of(Vec::loadu(p0 + k), Vec::loadu(p1 + k));
            const Vec high = pair_of(Vec::loadu(p2 + k), Vec::loadu(p3 + k));
            pair_of(Vec::loadu(acc + k), pair_of(low, high)).store(acc + k);
        }
        for (; k < cols; ++k) {
            ctype best = acc[k];
            best = WantMax ? nan_max(best, p0[k]) : nan_min(best, p0[k]);
            best = WantMax ? nan_max(best, p1[k]) : nan_min(best, p1[k]);
            best = WantMax ? nan_max(best, p2[k]) : nan_min(best, p2[k]);
            best = WantMax ? nan_max(best, p3[k]) : nan_min(best, p3[k]);
            acc[k] = best;
        }
    }
    for (; s < n; ++s) {
        const ctype* row = in + (row_begin + s) * cols;
        int64_t k = 0;
        for (; k + lanes <= cols; k += lanes) {
            pair_of(Vec::loadu(acc + k), Vec::loadu(row + k)).store(acc + k);
        }
        for (; k < cols; ++k) {
            acc[k] = WantMax ? nan_max(acc[k], row[k]) : nan_min(acc[k], row[k]);
        }
    }
}

// Few rows over a wide slice: four lane-vectors of columns keep their totals in
// registers, and four rows fold into them at a time with the compares of the
// other three rows still in flight.  Nothing but the input is touched, so a
// slice narrower than the level-one cache is not a precondition here.
template <typename ctype, bool WantMax>
inline void fold_leading_thin(const ctype* in, ctype* out, int64_t rows,
                              int64_t cols, int64_t begin, int64_t end) {
    using Vec = Vectorized<ctype>;
    constexpr int64_t lanes = Vec::size();
    auto pair_of = [](const Vec& a, const Vec& b) {
        return extremum_vec_step<ctype, WantMax>(a, b);
    };
    auto fold_rows = [&](const ctype* base, int64_t off, Vec& acc) {
        const ctype* q = base + off;
        acc = pair_of(acc, pair_of(pair_of(Vec::loadu(q), Vec::loadu(q + cols)),
                                   pair_of(Vec::loadu(q + 2 * cols),
                                           Vec::loadu(q + 3 * cols))));
    };
    int64_t j = begin;
    for (; j + 4 * lanes <= end; j += 4 * lanes) {
        Vec a0 = Vec::loadu(in + j);
        Vec a1 = Vec::loadu(in + j + lanes);
        Vec a2 = Vec::loadu(in + j + 2 * lanes);
        Vec a3 = Vec::loadu(in + j + 3 * lanes);
        int64_t s = 1;
        for (; s + 4 <= rows; s += 4) {
            const ctype* base = in + s * cols + j;
            fold_rows(base, 0, a0);
            fold_rows(base, lanes, a1);
            fold_rows(base, 2 * lanes, a2);
            fold_rows(base, 3 * lanes, a3);
        }
        for (; s < rows; ++s) {
            const ctype* row = in + s * cols + j;
            a0 = pair_of(a0, Vec::loadu(row));
            a1 = pair_of(a1, Vec::loadu(row + lanes));
            a2 = pair_of(a2, Vec::loadu(row + 2 * lanes));
            a3 = pair_of(a3, Vec::loadu(row + 3 * lanes));
        }
        a0.store(out + j);
        a1.store(out + j + lanes);
        a2.store(out + j + 2 * lanes);
        a3.store(out + j + 3 * lanes);
    }
    for (; j + lanes <= end; j += lanes) {
        Vec acc = Vec::loadu(in + j);
        for (int64_t s = 1; s < rows; ++s) {
            acc = pair_of(acc, Vec::loadu(in + s * cols + j));
        }
        acc.store(out + j);
    }
    for (; j < end; ++j) {
        ctype best = in[j];
        for (int64_t s = 1; s < rows; ++s) {
            const ctype value = in[s * cols + j];
            best = WantMax ? nan_max(best, value) : nan_min(best, value);
        }
        out[j] = best;
    }
}

// A leading dimension of a dense tensor is the one case where the rows can be
// handed out whole: each task sweeps a contiguous block and carries its partial
// answer as a private slice, so the walk never leaves the row it is reading.  The
// alternative folds a band of columns at a time, which walks every row of the
// plane once per band with a stride of the full row width; past a few cache lines
// that stride outruns the prefetchers, and the same bytes then take about twice
// as long.  The slices are folded together afterwards, and an extremum of slice
// extrema is the extremum, so no identity value enters anywhere.
template <typename ctype>
bool fold_leading_extrema(Tensor& result, const Tensor& self, int64_t ndim,
                          const std::vector<int64_t>& resolved,
                          bool maximum_value) {
    if (ndim < 2 || resolved.size() != 1 || resolved[0] != 0) return false;
    if (!self.is_contiguous()) return false;
    const int64_t rows = self.size(0);
    if (rows <= 0) return false;
    const int64_t cols = self.numel() / rows;
    if (cols <= 0 || rows * cols != self.numel() || result.numel() != cols) {
        return false;
    }
    // A slice wider than the level-one cache would be evicted between the rows
    // that fold into it, and a block shorter than a few dozen rows per task is
    // better served by the banded walk, which then wins the prefetcher back.
    const bool serial = self.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
                        in_parallel_region();
    const int64_t tasks = serial ? 1 : get_num_threads();
    const ctype* data = self.data_ptr<ctype>();
#if defined(__x86_64__)
    if constexpr (std::is_same_v<ctype, tensorplay::Half> ||
                  std::is_same_v<ctype, tensorplay::BFloat16>) {
        // One walker holds the packed columns widened in its registers over the
        // whole row range, so no row count needs a staging slice and no answer
        // is narrowed before it is final.
        if (serial && cols * static_cast<int64_t>(sizeof(ctype)) <= 16384) {
            ctype* out = result.data_ptr<ctype>();
            if (maximum_value) {
                fold_leading_block<ctype, true>(data, out, cols, 0, rows);
            } else {
                fold_leading_block<ctype, false>(data, out, cols, 0, rows);
            }
            return true;
        }
    }
#endif
    if (rows < 32 * tasks) {
        // With rows too few to share out, the columns keep their totals in
        // registers over the whole row range and the bands are what the team
        // splits instead.
        const int64_t block = std::max<int64_t>(
            8 * static_cast<int64_t>(Vectorized<ctype>::size()),
            GRAIN_SIZE / rows);
        ctype* out = result.data_ptr<ctype>();
        auto fold_band = [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; b += block) {
                const int64_t band_end = std::min(cols, b + block);
                if (maximum_value) {
                    fold_leading_thin<ctype, true>(data, out, rows, cols, b,
                                                   band_end);
                } else {
                    fold_leading_thin<ctype, false>(data, out, rows, cols, b,
                                                    band_end);
                }
            }
        };
        if (serial) {
            fold_band(0, cols);
        } else {
            parallel_for(0, cols, block, fold_band);
        }
        return true;
    }
    // A slice whose whole width fits the register file is held there over the
    // whole row range, however long that range is: the totals are never written
    // back between rows, so the walk pays one load and one compare per element
    // and nothing else.  Only a slice too wide for the registers has to be cut
    // into blocks, which is what the walk below does.
    if (serial && cols <= 8 * static_cast<int64_t>(Vectorized<ctype>::size())) {
        ctype* out = result.data_ptr<ctype>();
        if (maximum_value) {
            fold_leading_thin<ctype, true>(data, out, rows, cols, 0, cols);
        } else {
            fold_leading_thin<ctype, false>(data, out, rows, cols, 0, cols);
        }
        return true;
    }
    // A slice wider than the level-one cache cannot be held in registers over
    // the whole run.  An eight-byte integral width then folds a block of
    // columns at a time, each block walking the rows with its totals in
    // registers: the blocks partition every row, so each element is still read
    // exactly once, and no column is walked a second time to reach it.
    if (cols * static_cast<int64_t>(sizeof(ctype)) > 16384) {
        if constexpr (std::is_integral_v<ctype> &&
                      !std::is_same_v<ctype, bool> && sizeof(ctype) == 8) {
            ctype* out = result.data_ptr<ctype>();
            const int64_t block = std::max<int64_t>(
                8 * static_cast<int64_t>(Vectorized<ctype>::size()),
                GRAIN_SIZE / rows);
            auto fold_band = [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; b += block) {
                    const int64_t band_end = std::min(cols, b + block);
                    if (maximum_value) {
                        fold_leading_thin<ctype, true>(data, out, rows, cols, b,
                                                       band_end);
                    } else {
                        fold_leading_thin<ctype, false>(data, out, rows, cols,
                                                        b, band_end);
                    }
                }
            };
            if (serial) {
                fold_band(0, cols);
            } else {
                parallel_for(0, cols, block, fold_band);
            }
            return true;
        }
        return false;
    }

    ctype* out = result.data_ptr<ctype>();
    if (tasks == 1) {
        // One fold covers the whole row range, so its totals are the answer
        // and need neither a staging buffer nor a copy out of one.
        if (maximum_value) {
            fold_leading_block<ctype, true>(data, out, cols, 0, rows);
        } else {
            fold_leading_block<ctype, false>(data, out, cols, 0, rows);
        }
        return true;
    }

    const int64_t per = (rows + tasks - 1) / tasks;
    Tensor buffer = Tensor::empty({tasks * cols}, self.dtype(), self.device());
    ctype* slices = buffer.data_ptr<ctype>();
    auto fold_task = [&](int64_t t) {
        const int64_t begin = t * per;
        const int64_t end = std::min(rows, begin + per);
        ctype* acc = slices + t * cols;
        if (maximum_value) {
            fold_leading_block<ctype, true>(data, acc, cols, begin, end - begin);
        } else {
            fold_leading_block<ctype, false>(data, acc, cols, begin, end - begin);
        }
    };
    if (serial) {
        for (int64_t t = 0; t < tasks; ++t) fold_task(t);
    } else {
        parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
            for (int64_t t = begin; t < end; ++t) fold_task(t);
        });
    }

    parallel_for(0, cols, 256, [&](int64_t begin, int64_t end) {
        constexpr int64_t lanes = Vectorized<ctype>::size();
        int64_t j = begin;
        for (; j + lanes <= end; j += lanes) {
            Vectorized<ctype> acc = Vectorized<ctype>::loadu(slices + j);
            for (int64_t t = 1; t < tasks; ++t) {
                const Vectorized<ctype> part =
                    Vectorized<ctype>::loadu(slices + t * cols + j);
                acc = maximum_value ? maximum(acc, part) : minimum(acc, part);
            }
            acc.store(out + j);
        }
        for (; j < end; ++j) {
            ctype best = slices[j];
            for (int64_t t = 1; t < tasks; ++t) {
                best = maximum_value ? nan_max(best, slices[t * cols + j])
                                     : nan_min(best, slices[t * cols + j]);
            }
            out[j] = best;
        }
    });
    return true;
}

// ---- Fused dual-input reductions -------------------------------------------
// A single pass over both inputs accumulates sum((a-b)^2) or sum(|a-b|), so
// regression metrics never materialise the difference/power tensors and their
// separate kernels.

template <typename scalar_t>
void run_squared_error_sum(TensorIterator& iter) {
    binary_kernel_reduce2_vec(
        iter,
        [](scalar_t acc, scalar_t a, scalar_t b) -> scalar_t {
            const scalar_t d = a - b;
            return acc + d * d;
        },
        [](Vectorized<scalar_t> acc, Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
            const Vectorized<scalar_t> d = a - b;
            return acc + d * d;
        },
        [](scalar_t x, scalar_t y) -> scalar_t { return x + y; },
        [](Vectorized<scalar_t> x, Vectorized<scalar_t> y) -> Vectorized<scalar_t> { return x + y; },
        0);
}

template <typename scalar_t>
void run_abs_error_sum(TensorIterator& iter) {
    binary_kernel_reduce2_vec(
        iter,
        [](scalar_t acc, scalar_t a, scalar_t b) -> scalar_t {
            return acc + std::abs(a - b);
        },
        [](Vectorized<scalar_t> acc, Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
            return acc + (a - b).abs();
        },
        [](scalar_t x, scalar_t y) -> scalar_t { return x + y; },
        [](Vectorized<scalar_t> x, Vectorized<scalar_t> y) -> Vectorized<scalar_t> { return x + y; },
        0);
}

Tensor fused_dual_reduce_impl(const Tensor& self, const Tensor& other,
                              const std::vector<int64_t>& dims, bool keepdim,
                              bool squared) {
    const int64_t ndim = self.dim();
    bool same_shape = self.dim() == other.dim();
    if (same_shape) {
        for (int64_t i = 0; i < ndim; ++i) {
            if (self.size(i) != other.size(i)) {
                same_shape = false;
                break;
            }
        }
    }
    if (!same_shape) {
        TP_THROW(RuntimeError, "fused reduce: inputs must have the same shape");
    }
    const bool self_float = isFloatingType(self.dtype());
    const bool other_float = isFloatingType(other.dtype());
    DType out_dtype = (self_float || other_float)
        ? (self_float ? self.dtype() : other.dtype())
        : (isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype());
    if (out_dtype == DType::Float16 || out_dtype == DType::BFloat16) {
        // Narrow floats accumulate in single precision (opmath rule).
        out_dtype = DType::Float32;
    }

    if (ndim == 0) {
        Tensor a = self.dtype() == out_dtype ? self : self.to(out_dtype);
        Tensor b = other.dtype() == out_dtype ? other : other.to(out_dtype);
        Tensor diff = a - b;
        Tensor val = squared ? diff * diff : diff.abs();
        return keepdim ? val.reshape({1}) : val;
    }

    std::vector<int64_t> red_dims = dims;
    std::vector<bool> mask(static_cast<size_t>(ndim), false);
    if (red_dims.empty()) {
        red_dims.reserve(static_cast<size_t>(ndim));
        for (int64_t d = 0; d < ndim; ++d) {
            red_dims.push_back(d);
            mask[static_cast<size_t>(d)] = true;
        }
    } else {
        for (int64_t d : red_dims) {
            if (d < 0) d += ndim;
            if (d < 0 || d >= ndim) {
                TP_THROW(IndexError, "fused reduce: dimension out of range");
            }
            mask[static_cast<size_t>(d)] = true;
        }
    }

    std::vector<int64_t> out_shape = compute_reduction_shape(self, red_dims, keepdim);
    Tensor out = Tensor::zeros(out_shape, out_dtype, self.device());
    if (self.numel() == 0 || out.numel() == 0) return out;

    Tensor a = self.dtype() == out_dtype ? self : self.to(out_dtype);
    Tensor b = other.dtype() == out_dtype ? other : other.to(out_dtype);
    Tensor viewed = review_reduce_result(out, ndim, mask, keepdim);
    if (red_dims.size() == static_cast<size_t>(ndim)) {
        // Full reduction: flatten both inputs to one dimension so the fused
        // kernel sees exactly one reduce dimension (its fast path assumes a
        // single reduce dim plus an outer dim).
        a = a.reshape({-1});
        b = b.reshape({-1});
    }
    TensorIterator iter = TensorIterator::reduce_op2(viewed, a, b);

    switch (out_dtype) {
        case DType::Float32:
            if (squared) {
                run_squared_error_sum<float>(iter);
            } else {
                run_abs_error_sum<float>(iter);
            }
            break;
        case DType::Float64:
            if (squared) {
                run_squared_error_sum<double>(iter);
            } else {
                run_abs_error_sum<double>(iter);
            }
            break;
        case DType::Int64:
            if (squared) {
                run_squared_error_sum<int64_t>(iter);
            } else {
                run_abs_error_sum<int64_t>(iter);
            }
            break;
        default:
            TP_THROW(NotImplementedError, "fused reduce not implemented for this dtype");
    }
    return out;
}

Tensor sum_squared_error_kernel_impl(const Tensor& self, const Tensor& other,
                                     const std::vector<int64_t>& dims, bool keepdim) {
    return fused_dual_reduce_impl(self, other, dims, keepdim, /*squared=*/true);
}

Tensor sum_abs_error_kernel_impl(const Tensor& self, const Tensor& other,
                                const std::vector<int64_t>& dims, bool keepdim) {
    return fused_dual_reduce_impl(self, other, dims, keepdim, /*squared=*/false);
}

// A named axis other than the trailing block, with an inner run narrow enough
// to hold in registers: every band of the run carries its extremum across the
// reduced rows, so a row costs one compare per band instead of a conversion per
// element.  A band that runs past the end of the run reads what follows it in
// the buffer and writes only its own lanes back; the tensor's own last row
// reads through a copy that repeats a value, which cannot change the extremum
// of the lanes that are kept.
template <typename T, bool WantMax, int64_t NB>
inline void extrema_narrow_real(const T* in, T* out, int64_t outer, int64_t red,
                                int64_t inner) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const int64_t keep = inner - (NB - 1) * W;
    auto combine = [](const Vec& a, const Vec& b) {
        return extremum_vec_step<T, WantMax>(a, b);
    };
    auto load = [](const T* p, int64_t lanes, bool pad) {
        if (!pad) return Vec::loadu(p);
        alignas(64) T copy[W];
        std::memcpy(copy, p, static_cast<size_t>(lanes) * sizeof(T));
        for (int64_t k = lanes; k < W; ++k) copy[k] = copy[0];
        return Vec::loadu(copy);
    };
    for (int64_t o = 0; o < outer; ++o) {
        const T* base = in + o * red * inner;
        T* dst = out + o * inner;
        const bool last_block = o == outer - 1;
        Vec acc[NB];
        for (int64_t b = 0; b < NB; ++b) {
            const bool pad = last_block && red == 1 && b == NB - 1 && keep < W;
            acc[b] = load(base + b * W, keep, pad);
        }
        for (int64_t r = 1; r < red; ++r) {
            const T* row = base + r * inner;
            const bool last_row = last_block && r == red - 1;
            for (int64_t b = 0; b < NB; ++b) {
                const bool pad = last_row && b == NB - 1 && keep < W;
                acc[b] = combine(acc[b], load(row + b * W, keep, pad));
            }
        }
        for (int64_t b = 0; b < NB; ++b) {
            const int64_t lanes = b == NB - 1 ? keep : W;
            if (lanes >= W) {
                acc[b].store(dst + b * W);
            } else {
                alignas(64) T held[W];
                acc[b].store(held);
                std::memcpy(dst + b * W, held,
                            static_cast<size_t>(lanes) * sizeof(T));
            }
        }
    }
}


template <typename T, bool WantMax>
inline void extrema_narrow_real_dispatch(const T* in, T* out, int64_t outer,
                                         int64_t red, int64_t inner) {
    constexpr int64_t W = Vectorized<T>::size();
    switch ((inner + W - 1) / W) {
        case 1: extrema_narrow_real<T, WantMax, 1>(in, out, outer, red, inner); return;
        case 2: extrema_narrow_real<T, WantMax, 2>(in, out, outer, red, inner); return;
        case 3: extrema_narrow_real<T, WantMax, 3>(in, out, outer, red, inner); return;
        case 4: extrema_narrow_real<T, WantMax, 4>(in, out, outer, red, inner); return;
        default: return;
    }
}

#if defined(__x86_64__)
// A band of columns kept in registers over a whole run of rows: the walk
// streams the rows once and never writes a total back between them, which is
// what the staging slice of a general reduction cannot avoid.  The last band
// pads a short row from its own first lane so no identity value is compared.
template <typename T, bool WantMax, int64_t NB>
__attribute__((target("avx2"), noinline))
void extrema_leading_banded(const T* in, T* out, int64_t outer, int64_t red,
                            int64_t inner) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const int64_t keep = inner - (NB - 1) * W;
    for (int64_t o = 0; o < outer; ++o) {
        const T* base = in + o * red * inner;
        T* dst = out + o * inner;
        alignas(64) T copy[W];
        Vec acc[NB];
        for (int64_t b = 0; b < NB; ++b) {
            const T* p = base + b * W;
            if (b == NB - 1 && keep < W) {
                std::memcpy(copy, p, static_cast<size_t>(keep) * sizeof(T));
                for (int64_t k = keep; k < W; ++k) copy[k] = copy[0];
                p = copy;
            }
            acc[b] = Vec::loadu(p);
        }
        for (int64_t r = 1; r < red; ++r) {
            const T* row = base + r * inner;
            for (int64_t b = 0; b < NB; ++b) {
                const T* p = row + b * W;
                if (b == NB - 1 && keep < W) {
                    std::memcpy(copy, p, static_cast<size_t>(keep) * sizeof(T));
                    for (int64_t k = keep; k < W; ++k) copy[k] = copy[0];
                    p = copy;
                }
                acc[b] = extremum_vec_step<T, WantMax>(acc[b], Vec::loadu(p));
            }
        }
        for (int64_t b = 0; b < NB; ++b) {
            const int64_t lanes = b == NB - 1 ? keep : W;
            if (lanes >= W) {
                acc[b].store(dst + b * W);
            } else {
                acc[b].store(copy);
                std::memcpy(dst + b * W, copy,
                            static_cast<size_t>(lanes) * sizeof(T));
            }
        }
    }
}

template <typename T, bool WantMax>
inline void extrema_leading_banded_dispatch(const T* in, T* out, int64_t outer,
                                            int64_t red, int64_t inner) {
    constexpr int64_t W = Vectorized<T>::size();
    switch ((inner + W - 1) / W) {
        case 1: extrema_leading_banded<T, WantMax, 1>(in, out, outer, red, inner); return;
        case 2: extrema_leading_banded<T, WantMax, 2>(in, out, outer, red, inner); return;
        case 3: extrema_leading_banded<T, WantMax, 3>(in, out, outer, red, inner); return;
        default: extrema_leading_banded<T, WantMax, 4>(in, out, outer, red, inner); return;
    }
}

// The same walk over a packed width: a band of eight columns is widened once
// per row and compared in single precision, and the packed value is formed
// only when the band is written back.
template <bool BFloat, bool WantMax, int64_t NB>
__attribute__((target("avx2,fma,f16c"), noinline))
void extrema_narrow_packed_hw(const uint16_t* in, uint16_t* out, int64_t outer,
                              int64_t red, int64_t inner) {
    constexpr int64_t W = 8;
    const int64_t keep = inner - (NB - 1) * W;
    for (int64_t o = 0; o < outer; ++o) {
        const uint16_t* base = in + o * red * inner;
        uint16_t* dst = out + o * inner;
        const bool last_block = o == outer - 1;
        __m256 acc[NB];
        if (last_block && red == 1 && keep < W) {
            for (int64_t b = 0; b < NB - 1; ++b) {
                acc[b] = widen_row8<BFloat>(base + b * W);
            }
            alignas(16) uint16_t copy[W];
            std::memcpy(copy, base + (NB - 1) * W,
                        static_cast<size_t>(keep) * sizeof(uint16_t));
            for (int64_t k = keep; k < W; ++k) copy[k] = copy[0];
            acc[NB - 1] = widen_row8<BFloat>(copy);
        } else {
            for (int64_t b = 0; b < NB; ++b) {
                acc[b] = widen_row8<BFloat>(base + b * W);
            }
        }
        for (int64_t r = 1; r < red; ++r) {
            const uint16_t* row = base + r * inner;
            const bool last_row = last_block && r == red - 1;
            for (int64_t b = 0; b < NB - 1; ++b) {
                acc[b] = extremum_step8<WantMax>(
                    acc[b], widen_row8<BFloat>(row + b * W));
            }
            if (last_row && keep < W) {
                alignas(16) uint16_t copy[W];
                std::memcpy(copy, row + (NB - 1) * W,
                            static_cast<size_t>(keep) * sizeof(uint16_t));
                for (int64_t k = keep; k < W; ++k) copy[k] = copy[0];
                acc[NB - 1] = extremum_step8<WantMax>(
                    acc[NB - 1], widen_row8<BFloat>(copy));
            } else {
                acc[NB - 1] = extremum_step8<WantMax>(
                    acc[NB - 1], widen_row8<BFloat>(row + (NB - 1) * W));
            }
        }
        for (int64_t b = 0; b < NB - 1; ++b) {
            store_band_hw<BFloat>(dst, b * W, acc[b]);
        }
        const int64_t lanes = keep < W ? keep : W;
        alignas(32) float held[W];
        _mm256_store_ps(held, acc[NB - 1]);
        for (int64_t k = 0; k < lanes; ++k) {
            dst[(NB - 1) * W + k] = BFloat
                ? tensorplay::cpu::vecreduced::narrow_bf16_one(held[k])
                : tensorplay::cpu::vecreduced::narrow_one_hw(held[k]);
        }
    }
}

template <bool BFloat, bool WantMax>
inline void extrema_narrow_packed_dispatch(const uint16_t* in, uint16_t* out,
                                           int64_t outer, int64_t red,
                                           int64_t inner) {
    switch ((inner + 7) / 8) {
        case 1: extrema_narrow_packed_hw<BFloat, WantMax, 1>(in, out, outer, red, inner); return;
        case 2: extrema_narrow_packed_hw<BFloat, WantMax, 2>(in, out, outer, red, inner); return;
        case 3: extrema_narrow_packed_hw<BFloat, WantMax, 3>(in, out, outer, red, inner); return;
        case 4: extrema_narrow_packed_hw<BFloat, WantMax, 4>(in, out, outer, red, inner); return;
        default: return;
    }
}
#endif

// One named axis between two kept ones, answered without the reduction
// iterator: the inner run is short enough that its bands stay in registers for
// the whole reduced axis.  A register band of columns widened out of a wide
// element.
#if defined(__x86_64__)
constexpr int64_t kExtremaBand = 8 * static_cast<int64_t>(
    Vectorized<int64_t>::size());
#else
constexpr int64_t kExtremaBand = 8;
#endif
template <typename ctype>
bool fold_narrow_extrema(Tensor& result, const Tensor& self, int64_t ndim,
                         const std::vector<int64_t>& resolved,
                         bool maximum_value) {
    if (resolved.size() != 1 || ndim < 3) return false;
    const int64_t d = resolved[0];
    if (d <= 0 || d >= ndim - 1) return false;
    if (!self.is_contiguous() || self.numel() == 0) return false;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < d; ++i) outer *= self.size(i);
    for (int64_t i = d + 1; i < ndim; ++i) inner *= self.size(i);
    const int64_t red = self.size(d);
    if (red <= 0 || outer <= 0 || inner < 2) return false;
    if (result.numel() != outer * inner) return false;
    constexpr bool kReduced = std::is_same_v<ctype, tensorplay::Half> ||
                              std::is_same_v<ctype, tensorplay::BFloat16>;
    constexpr bool kReal = std::is_same_v<ctype, float> ||
                           std::is_same_v<ctype, double>;
    const ctype* data = self.data_ptr<ctype>();
    ctype* out = result.data_ptr<ctype>();
    if constexpr (kReal) {
        constexpr int64_t W = Vectorized<ctype>::size();
        if (inner <= 4 * W) {
            if (maximum_value) {
                extrema_narrow_real_dispatch<ctype, true>(data, out, outer, red,
                                                          inner);
            } else {
                extrema_narrow_real_dispatch<ctype, false>(data, out, outer, red,
                                                           inner);
            }
            return true;
        }
        // A slice wider than four registers keeps its column totals in one
        // accumulator slice, so the walk reads every element of the plane
        // exactly once.  That slice has to stay in the first-level cache while
        // the rows stream past it, which is what the bound below checks.
        if (inner * static_cast<int64_t>(sizeof(ctype)) > 32768) {
            return false;
        }
        for (int64_t o = 0; o < outer; ++o) {
            const ctype* block = data + o * red * inner;
            ctype* acc = out + o * inner;
            if (maximum_value) {
                fold_leading_block<ctype, true>(block, acc, inner, 0, red);
            } else {
                fold_leading_block<ctype, false>(block, acc, inner, 0, red);
            }
        }
        return true;
    } else if constexpr (kReduced) {
#if defined(__x86_64__)
        if (inner > 32) {
            // The same over a packed width: a slice that fits the nearest
            // cache is folded a whole block of outer indices at a time, one
            // band of widened columns at a time.
            if (!tensorplay::cpu::vecreduced::f16c_available()) return false;
            if (inner * 2 > 32768) return false;
            constexpr bool kBFloat = std::is_same_v<ctype, tensorplay::BFloat16>;
            const uint16_t* bits = reinterpret_cast<const uint16_t*>(data);
            uint16_t* obits = reinterpret_cast<uint16_t*>(out);
            for (int64_t o = 0; o < outer; ++o) {
                const uint16_t* block = bits + o * red * inner;
                uint16_t* acc = obits + o * inner;
                if (maximum_value) {
                    fold_leading_block_hw<kBFloat, true>(block, acc, inner, 0,
                                                         red);
                } else {
                    fold_leading_block_hw<kBFloat, false>(block, acc, inner, 0,
                                                          red);
                }
            }
            return true;
        }
        if (!tensorplay::cpu::vecreduced::f16c_available()) return false;
        constexpr bool kBFloat = std::is_same_v<ctype, tensorplay::BFloat16>;
        const uint16_t* bits = reinterpret_cast<const uint16_t*>(data);
        uint16_t* obits = reinterpret_cast<uint16_t*>(out);
        if (maximum_value) {
            extrema_narrow_packed_dispatch<kBFloat, true>(bits, obits, outer, red, inner);
        } else {
            extrema_narrow_packed_dispatch<kBFloat, false>(bits, obits, outer, red, inner);
        }
        return true;
#else
        return false;
#endif
    } else if constexpr (std::is_integral_v<ctype> &&
                         !std::is_same_v<ctype, bool>) {
        // A leading axis wide enough to hold four registers is folded with the
        // column totals in registers across the whole run, instead of through
        // a staging slice the iterator would have to build first.
#if defined(__x86_64__)
        if (inner == 1 && red > 4 * Vectorized<ctype>::size() &&
            red * static_cast<int64_t>(sizeof(ctype)) <= 32768) {
            if (maximum_value) {
                extrema_leading_banded_dispatch<ctype, true>(data, out, outer,
                                                             red, inner);
            } else {
                extrema_leading_banded_dispatch<ctype, false>(data, out, outer,
                                                              red, inner);
            }
            return true;
        }
#endif  // __x86_64__
        // Under one outer block the middle slice is a leading axis of its own:
        // its columns keep their totals over the whole run and are read once.
        // A slice that fits the first-level cache is walked row after row, which
        // is the stream the prefetcher follows; a wider one is walked a block of
        // columns at a time instead, since the totals could no longer stay near
        // the data.
        // A long run is walked with the totals in registers, since a total
        // read and written once per row costs more than the row itself.  A
        // short run keeps the totals in memory and streams the rows instead,
        // unless the totals no longer fit the cache nearest the data, which
        // only a block of columns at a time can keep near it.
        const bool row_major =
            inner * static_cast<int64_t>(sizeof(ctype)) <= 16384 &&
            (red < 512 ||
             inner > 4 * static_cast<int64_t>(Vectorized<ctype>::size()));
        auto fold_outer = [&](int64_t begin, int64_t end) {
            for (int64_t o = begin; o < end; ++o) {
                const ctype* block = data + o * red * inner;
                ctype* acc = out + o * inner;
                if (row_major) {
                    if (maximum_value) {
                        fold_leading_block<ctype, true>(block, acc, inner, 0,
                                                        red);
                    } else {
                        fold_leading_block<ctype, false>(block, acc, inner, 0,
                                                         red);
                    }
                } else if (maximum_value) {
                    fold_leading_thin<ctype, true>(block, acc, red, inner, 0,
                                                   inner);
                } else {
                    fold_leading_thin<ctype, false>(block, acc, red, inner, 0,
                                                    inner);
                }
            }
        };
        if (self.numel() < GRAIN_SIZE || get_num_threads() == 1 ||
            in_parallel_region()) {
            fold_outer(0, outer);
        } else {
            parallel_for(0, outer, 1, fold_outer);
        }
        return true;
    } else {
        return false;
    }
}

Tensor extrema_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims,
                              bool keepdim, bool maximum_value) {
    if (std::optional<Tensor> permuted = review_permuted_reduction(
            self, dims, keepdim,
            [&](const Tensor& dense, const std::vector<int64_t>& dense_dims) {
                return extrema_dim_kernel_impl(dense, dense_dims, true,
                                               maximum_value);
            })) {
        return std::move(*permuted);
    }

    const int64_t ndim = self.dim();
    std::vector<bool> mask(ndim, false);
    std::vector<int64_t> resolved;
    for (int64_t dim : dims) {
        const int64_t rank = std::max<int64_t>(1, ndim);
        TP_CHECK_INDEX(dim >= -rank && dim < rank,
                       "reduction dimension out of range");
        if (ndim == 0) continue;
        if (dim < 0) dim += ndim;
        TP_CHECK(!mask[dim], "reduction dimension appears more than once");
        mask[dim] = true;
        resolved.push_back(dim);
    }
    Tensor result = Tensor::empty(compute_reduction_shape(self, resolved, keepdim),
                                  self.dtype(), self.device());
    if (result.numel() == 0) return result;
// The walks below answer from the result the caller asked for and without the
// reduction iterator, so the layout the iterator needs is only built by the
// branches that fall through to it.
#define TP_EXTREMA_ITER() \
    Tensor viewed = review_reduce_result(result, ndim, mask, keepdim); \
    TensorIterator iter = TensorIterator::reduce_op(viewed, self)
#define TP_EXTREMA_CASE(ctype, name) \
    case DType::name: \
        if constexpr (std::is_same_v<ctype, tensorplay::Half> || \
                      std::is_same_v<ctype, tensorplay::BFloat16>) { \
            if (fold_trailing_extrema<ctype>(result, self, ndim, resolved, \
                                             maximum_value)) { \
                return result; \
            } \
            if (fold_leading_extrema<ctype>(result, self, ndim, resolved, \
                                            maximum_value)) { \
                return result; \
            } \
            if (fold_narrow_extrema<ctype>(result, self, ndim, resolved, \
                                           maximum_value)) { \
                return result; \
            } \
        } \
        if constexpr (std::is_same_v<ctype, float> || \
                      std::is_same_v<ctype, double>) { \
            if (fold_trailing_extrema<ctype>(result, self, ndim, resolved, \
                                             maximum_value)) { \
                return result; \
            } \
            if (fold_whole_extrema<ctype>(result, self, ndim, resolved, \
                                          maximum_value)) { \
                return result; \
            } \
            if (fold_leading_extrema<ctype>(result, self, ndim, resolved, \
                                            maximum_value)) { \
                return result; \
            } \
            if (fold_narrow_extrema<ctype>(result, self, ndim, resolved, \
                                           maximum_value)) { \
                return result; \
            } \
        } \
        if constexpr (std::is_integral_v<ctype> && \
                      !std::is_same_v<ctype, bool>) { \
            if (fold_trailing_extrema<ctype>(result, self, ndim, resolved, \
                                             maximum_value)) { \
                return result; \
            } \
            if (fold_whole_extrema<ctype>(result, self, ndim, resolved, \
                                          maximum_value)) { \
                return result; \
            } \
            if (fold_leading_extrema<ctype>(result, self, ndim, resolved, \
                                            maximum_value)) { \
                return result; \
            } \
            if (fold_narrow_extrema<ctype>(result, self, ndim, resolved, \
                                           maximum_value)) { \
                return result; \
            } \
        } \
        if constexpr (std::is_integral_v<ctype> && \
                      !std::is_same_v<ctype, bool> && sizeof(ctype) == 8) { \
            TP_EXTREMA_ITER(); \
            binary_kernel_reduce(iter, ExtremaValuesOps<ctype>{maximum_value}, \
                                 extrema_identity<ctype>(maximum_value)); \
        } else if (maximum_value) { \
            TP_EXTREMA_ITER(); \
            binary_kernel_reduce_vec(iter, \
                [](ctype a, ctype b) -> ctype { return nan_max(a, b); }, \
                [](Vectorized<ctype> a, Vectorized<ctype> b) { return maximum(a, b); }, \
                static_cast<double>(extrema_identity<ctype>(true))); \
        } else { \
            TP_EXTREMA_ITER(); \
            binary_kernel_reduce_vec(iter, \
                [](ctype a, ctype b) -> ctype { return nan_min(a, b); }, \
                [](Vectorized<ctype> a, Vectorized<ctype> b) { return minimum(a, b); }, \
                static_cast<double>(extrema_identity<ctype>(false))); \
        } \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_EXTREMA_CASE)
        default: TP_THROW(TypeError, "extrema reduction: unsupported dtype");
    }
#undef TP_EXTREMA_CASE
#undef TP_EXTREMA_ITER
    return result;
}

// Whole-tensor extremum folds over a contiguous buffer, seeded from the first
// element so no identity value is needed. Seeding also keeps the Int64/UInt64
// minima exact: the identity-only pair tracking exists because a double
// identity cannot round-trip 64-bit values, and a direct fold never touches
// one.
template <typename scalar_t>
inline scalar_t direct_fold_max(const scalar_t* data, int64_t n) {
    scalar_t best = data[0];
    for (int64_t i = 1; i < n; ++i) best = nan_max(best, data[i]);
    return best;
}

template <typename scalar_t>
inline scalar_t direct_fold_min(const scalar_t* data, int64_t n) {
    scalar_t best = data[0];
    for (int64_t i = 1; i < n; ++i) best = nan_min(best, data[i]);
    return best;
}

// At or below this element count, an all-reduce runs as the direct fold
// above: constructing the reduction iterator and filling its identity costs
// more host time than the whole scan.
constexpr int64_t kMaxDirectFoldElements = 128;

// Pair-tracking ops for reductions whose identity value cannot round-trip
// precision as doubles and would corrupt the identity fill.
template <typename scalar_t>
struct ExtremumValuePairOps {
    using arg_t = std::pair<scalar_t, int64_t>;
    arg_t reduce(arg_t acc, scalar_t data, int64_t idx) const {
        return cmp(data, acc.first) ? arg_t(data, idx) : acc;
    }
    arg_t combine(arg_t a, arg_t b) const {
        return cmp(b.first, a.first) ? b : a;
    }
    scalar_t project(arg_t a) const { return a.first; }
    // whole-tensor reduction: no index translation needed
    arg_t translate_idx(arg_t acc, int64_t) const { return acc; }
    bool (*cmp)(scalar_t, scalar_t);
};

// so the scan is parallelized with SIMD accumulators inside
// binary_kernel_reduce_vec instead of one serial scalar pass.
#if defined(_M_X64)
// Whole-tensor extrema at the vector width of the tier this TU is compiled for.
// The GCC build answers this from try_max_real_avx512; that body has per-function
// target attributes MSVC cannot spell, so the call fell through to
// binary_kernel_reduce_vec and its TensorIterator setup.
//
// Chunks are cut on grain boundaries, each answers a vector accumulator, and
// the chunk answers are folded scalar.  A NaN lane ORs into the accumulator the
// same way the scalar reference propagates it, and the fold repeats that rule.
template <typename T, bool Maximum>
T extrema_all_vec(const T* data, int64_t n) {
    using V = Vectorized<T>;
    constexpr int64_t W = V::size();
    const int64_t grain = tensorplay::parallel::GRAIN_SIZE;
    const int64_t chunks = (n + grain - 1) / grain;
    const T identity = Maximum ? std::numeric_limits<T>::lowest()
                               : std::numeric_limits<T>::max();
    const T nan = std::numeric_limits<T>::quiet_NaN();
    std::vector<T> partial(static_cast<size_t>(chunks), identity);
    // The split hands out ranges that need not start on a grain boundary, so
    // a range takes the next free slot instead of indexing by begin / grain.
    std::atomic<int64_t> next_slot{0};

    tensorplay::parallel::parallel_for(0, n, grain, [&](int64_t begin, int64_t end) {
        V acc(identity);
        V nan_acc(T(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const V v = V::loadu(data + i);
            nan_acc = nan_acc | v.isnan();
            acc = Maximum ? maximum(acc, v) : minimum(acc, v);
        }
        T best = Maximum ? acc.reduce_max() : acc.reduce_min();
        // A NaN lane is all-ones (i.e. a NaN bit pattern), so reduce_max on the
        // mask would drop it; read the mask lanes out instead.
        alignas(64) T lanes[W];
        nan_acc.store(lanes);
        for (int k = 0; k < static_cast<int>(W); ++k) {
            if (lanes[k] != T(0)) { best = nan; break; }
        }
        for (; i < end; ++i) {
            const T v = data[i];
            if (v != v || best != best) best = nan;
            else if (Maximum) best = (best < v) ? v : best;
            else best = (v < best) ? v : best;
        }
        const int64_t slot = next_slot.fetch_add(1, std::memory_order_relaxed);
        if (static_cast<size_t>(slot) < partial.size()) {
            partial[static_cast<size_t>(slot)] = best;
        }
    });

    T best = partial[0];
    for (size_t k = 1; k < partial.size(); ++k) {
        const T v = partial[k];
        if (v != v || best != best) { best = nan; continue; }
        if (Maximum) best = (best < v) ? v : best;
        else best = (v < best) ? v : best;
    }
    return best;
}
#endif

Tensor max_kernel_impl(const Tensor& self) {
    if (self.numel() == 0) {
        // over an empty tensor has no identity.
        TP_THROW(RuntimeError, "max(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    Tensor input = self.contiguous();
    Tensor out = Tensor::empty({}, self.dtype(), self.device());

#define TP_MAX_DIRECT_FOLD_CASE(ctype, name) \
    case DType::name: \
        *out.data_ptr<ctype>() = \
            direct_fold_max<ctype>(input.data_ptr<ctype>(), input.numel()); \
        return out;

    if (input.numel() <= kMaxDirectFoldElements) {
        switch (input.dtype()) {
            TP_MAX_DIRECT_FOLD_CASE(uint8_t, UInt8)
            TP_MAX_DIRECT_FOLD_CASE(int8_t, Int8)
            TP_MAX_DIRECT_FOLD_CASE(int16_t, Int16)
            TP_MAX_DIRECT_FOLD_CASE(int32_t, Int32)
            TP_MAX_DIRECT_FOLD_CASE(int64_t, Int64)
            TP_MAX_DIRECT_FOLD_CASE(uint16_t, UInt16)
            TP_MAX_DIRECT_FOLD_CASE(uint32_t, UInt32)
            TP_MAX_DIRECT_FOLD_CASE(uint64_t, UInt64)
            TP_MAX_DIRECT_FOLD_CASE(float, Float32)
            TP_MAX_DIRECT_FOLD_CASE(double, Float64)
            TP_MAX_DIRECT_FOLD_CASE(Half, Float16)
            TP_MAX_DIRECT_FOLD_CASE(BFloat16, BFloat16)
            TP_MAX_DIRECT_FOLD_CASE(bool, Bool)
            default: break;
        }
    }
#undef TP_MAX_DIRECT_FOLD_CASE

#if defined(_M_X64)
    if (input.numel() > kMaxDirectFoldElements) {
        if (input.dtype() == DType::Float32) {
            *out.data_ptr<float>() = extrema_all_vec<float, true>(
                input.data_ptr<float>(), input.numel());
            return out;
        }
        if (input.dtype() == DType::Float64) {
            *out.data_ptr<double>() = extrema_all_vec<double, true>(
                input.data_ptr<double>(), input.numel());
            return out;
        }
    }
#endif

#if defined(__x86_64__)
    if (input.is_contiguous() &&
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)) {
        double value = 0.0;
        bool has_nan = false;
        if (try_max_real_avx512(input.data_ptr(), input.numel(), input.dtype(),
                                &value, &has_nan) ||
            try_max_real_avx2(input.data_ptr(), input.numel(), input.dtype(),
                              &value, &has_nan)) {
            if (has_nan) {
                out.fill_(input.dtype() == DType::Float32
                    ? Scalar(std::numeric_limits<float>::quiet_NaN())
                    : Scalar(std::numeric_limits<double>::quiet_NaN()));
            } else if (input.dtype() == DType::Float32) {
                out.fill_(Scalar(static_cast<float>(value)));
            } else {
                out.fill_(Scalar(value));
            }
            return out;
        }
    }
#endif

    #define TP_MAX_VALUES_CASE(ctype, name) \
    case DType::name: \
        binary_kernel_reduce_vec(iter, \
            [](ctype a, ctype b) -> ctype { return nan_max(a, b); }, \
            [](Vectorized<ctype> a, Vectorized<ctype> b) { return maximum(a, b); }, \
            static_cast<double>(get_lowest<ctype>())); \
        break;

    TensorIterator iter = TensorIterator::reduce_op(out, input);
    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MAX_VALUES_CASE)
        default: TP_THROW(NotImplementedError, "max not implemented for this dtype");
    }
    #undef TP_MAX_VALUES_CASE
    return out;
}

std::tuple<Tensor, Tensor> max_dim_kernel_impl(const Tensor& self, int64_t dim0, bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "max(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(IndexError, "max(): Expected reduction dim ", dim, " to have non-zero size.");
    }

    Tensor sc = self.contiguous();
    std::vector<int64_t> in_shape = static_cast<std::vector<int64_t>>(sc.shape());
    const int64_t d_size = in_shape[dim];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= in_shape[i];
    for (int64_t i = dim + 1; i < nd; ++i) inner *= in_shape[i];

    std::vector<int64_t> out_shape = compute_reduction_shape(sc, {dim}, keepdim);
    Tensor vals = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(out_shape, DType::Int64, sc.device());

#if defined(__x86_64__)
    if (inner == 1 && try_extremum_lastdim_real<true>(
            sc.data_ptr(), vals.data_ptr(), idxs.data_ptr<int64_t>(), outer, d_size,
            sc.dtype())) {
        return {vals, idxs};
    }
    if (inner == 1 && try_extremum_lastdim_integral_avx512<true>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size)) {
        return {vals, idxs};
    }
    if (try_extremum_integral_dim_avx512<true>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size, inner)) {
        return {vals, idxs};
    }
    if (inner == 1 && try_extremum_lastdim_vec<true>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size)) {
        return {vals, idxs};
    }
    if (try_extremum_reduced_columns<true>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size, inner)) {
        return {vals, idxs};
    }
#endif

    // With the reduced dim removed (or sized 1 under keepdim), the output is
    // a contiguous [outer, inner] grid and line i lives at o*d_size*inner +
    // i*inner + in2 -- identical addressing for both keepdim modes. A NaN
    // candidate wins once, then a NaN running best is never displaced, so
    // the first NaN fixes both the value and the index; for integral
    // element types both added comparisons are tautologies the compiler
    // folds away.
#define TP_MAXMIN_DIM_CASE(ctype, name_, CMP_OP)                                        \
    case DType::name_: {                                                                \
        const ctype* sp = sc.data_ptr<ctype>();                                         \
        ctype* vp = vals.data_ptr<ctype>();                                             \
        int64_t* ip = idxs.data_ptr<int64_t>();                                         \
        /* A column walk reads one line per element and revisits the same lines    \
           once per column.  A block of columns that fills one line instead reads  \
           each line once and compares all of its lanes, so the rows stream past   \
           the cache a single time.  Lanes stay the innermost axis, which keeps    \
           each output's visits to the reduced axis in ascending order. */         \
        constexpr int64_t kLanes = std::max<int64_t>(1, 64 / static_cast<int64_t>(sizeof(ctype))); \
        if (inner >= kLanes) {                                                          \
            const int64_t col_blocks = (inner + kLanes - 1) / kLanes;                   \
            const int64_t block_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size);      \
            parallel_for(0, outer * col_blocks, block_grain, [&](int64_t b, int64_t e) { \
                for (int64_t flat = b; flat < e; ++flat) {                             \
                    const int64_t o = flat / col_blocks;                                \
                    const int64_t c0 = (flat - o * col_blocks) * kLanes;                \
                    const int64_t cols = std::min(kLanes, inner - c0);                  \
                    const ctype* base = sp + o * d_size * inner + c0;                   \
                    ctype* vrow = vp + o * inner + c0;                                  \
                    int64_t* irow = ip + o * inner + c0;                                \
                    if (cols == kLanes) {                                              \
                        ctype best[kLanes];                                            \
                        int64_t bi[kLanes];                                            \
                        for (int64_t c = 0; c < kLanes; ++c) {                          \
                            best[c] = base[c];                                         \
                            bi[c] = 0;                                                 \
                        }                                                               \
                        for (int64_t i = 1; i < d_size; ++i) {                          \
                            const ctype* row = base + i * inner;                        \
                            for (int64_t c = 0; c < kLanes; ++c) {                      \
                                const ctype x = row[c];                                 \
                                const bool take = (best[c] == best[c]) &&               \
                                                  (x != x || x CMP_OP best[c]);         \
                                if (take) { best[c] = x; bi[c] = i; }                   \
                            }                                                           \
                        }                                                               \
                        for (int64_t c = 0; c < kLanes; ++c) {                          \
                            vrow[c] = best[c];                                          \
                            irow[c] = bi[c];                                            \
                        }                                                               \
                    } else {                                                           \
                        for (int64_t c = 0; c < cols; ++c) {                            \
                            const ctype* lane = base + c;                               \
                            ctype best = lane[0];                                       \
                            int64_t bi = 0;                                             \
                            for (int64_t i = 1; i < d_size; ++i) {                      \
                                const ctype x = lane[i * inner];                        \
                                if (best == best && (x != x || x CMP_OP best)) {        \
                                    best = x;                                           \
                                    bi = i;                                             \
                                }                                                       \
                            }                                                           \
                            vrow[c] = best;                                             \
                            irow[c] = bi;                                               \
                        }                                                               \
                    }                                                                   \
                }                                                                       \
            });                                                                         \
            break;                                                                      \
        }                                                                               \
        const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size);           \
        parallel_for(0, outer * inner, line_grain, [&](int64_t b, int64_t e) {          \
            for (int64_t flat = b; flat < e; ++flat) {                                  \
                const int64_t o = flat / inner, in2 = flat % inner;                     \
                const ctype* line = sp + o * d_size * inner + in2;                      \
                ctype best = line[0];                                                   \
                int64_t bi = 0;                                                         \
                for (int64_t i = 1; i < d_size; ++i) {                                  \
                    const ctype x = line[i * inner];                                    \
                    if (best == best && (x != x || x CMP_OP best)) {                    \
                        best = x;                                                       \
                        bi = i;                                                         \
                    }                                                                   \
                }                                                                       \
                vp[flat] = best;                                                        \
                ip[flat] = bi;                                                          \
            }                                                                           \
        });                                                                             \
        break;                                                                          \
    }
#define TP_MAXMIN_MAX_DISPATCH()                       \
    switch (sc.dtype()) {                              \
        TP_MAXMIN_DIM_CASE(uint8_t, UInt8, >)          \
        TP_MAXMIN_DIM_CASE(int8_t, Int8, >)            \
        TP_MAXMIN_DIM_CASE(int16_t, Int16, >)          \
        TP_MAXMIN_DIM_CASE(int32_t, Int32, >)          \
        TP_MAXMIN_DIM_CASE(int64_t, Int64, >)          \
        TP_MAXMIN_DIM_CASE(uint16_t, UInt16, >)        \
        TP_MAXMIN_DIM_CASE(uint32_t, UInt32, >)        \
        TP_MAXMIN_DIM_CASE(uint64_t, UInt64, >)        \
        TP_MAXMIN_DIM_CASE(float, Float32, >)          \
        TP_MAXMIN_DIM_CASE(double, Float64, >)         \
        TP_MAXMIN_DIM_CASE(Half, Float16, >)           \
        TP_MAXMIN_DIM_CASE(BFloat16, BFloat16, >)      \
        TP_MAXMIN_DIM_CASE(bool, Bool, >)              \
        default:                                       \
            TP_THROW(NotImplementedError, "max_dim not implemented for this dtype"); \
    }
    TP_MAXMIN_MAX_DISPATCH()
#undef TP_MAXMIN_MAX_DISPATCH
#undef TP_MAXMIN_DIM_CASE
    return {vals, idxs};
}

Tensor min_kernel_impl(const Tensor& self) {
    if (self.numel() == 0) {
        TP_THROW(RuntimeError, "min(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    Tensor input = self.contiguous();
    Tensor out = Tensor::empty({}, self.dtype(), self.device());

#define TP_MIN_DIRECT_FOLD_CASE(ctype, name) \
    case DType::name: \
        *out.data_ptr<ctype>() = \
            direct_fold_min<ctype>(input.data_ptr<ctype>(), input.numel()); \
        return out;

    if (input.numel() <= kMaxDirectFoldElements) {
        switch (input.dtype()) {
            TP_MIN_DIRECT_FOLD_CASE(uint8_t, UInt8)
            TP_MIN_DIRECT_FOLD_CASE(int8_t, Int8)
            TP_MIN_DIRECT_FOLD_CASE(int16_t, Int16)
            TP_MIN_DIRECT_FOLD_CASE(int32_t, Int32)
            TP_MIN_DIRECT_FOLD_CASE(int64_t, Int64)
            TP_MIN_DIRECT_FOLD_CASE(uint16_t, UInt16)
            TP_MIN_DIRECT_FOLD_CASE(uint32_t, UInt32)
            TP_MIN_DIRECT_FOLD_CASE(uint64_t, UInt64)
            TP_MIN_DIRECT_FOLD_CASE(float, Float32)
            TP_MIN_DIRECT_FOLD_CASE(double, Float64)
            TP_MIN_DIRECT_FOLD_CASE(Half, Float16)
            TP_MIN_DIRECT_FOLD_CASE(BFloat16, BFloat16)
            TP_MIN_DIRECT_FOLD_CASE(bool, Bool)
            default: break;
        }
    }
#undef TP_MIN_DIRECT_FOLD_CASE

#if defined(_M_X64)
    if (input.numel() > kMaxDirectFoldElements) {
        if (input.dtype() == DType::Float32) {
            *out.data_ptr<float>() = extrema_all_vec<float, false>(
                input.data_ptr<float>(), input.numel());
            return out;
        }
        if (input.dtype() == DType::Float64) {
            *out.data_ptr<double>() = extrema_all_vec<double, false>(
                input.data_ptr<double>(), input.numel());
            return out;
        }
    }
#endif

    TensorIterator iter = TensorIterator::reduce_op(out, input);

    // the vec path's double ident, so it reduces with pair-tracking ops
    // unsigned 64-bit variant whose maximum also rounds up as a double.
    if (input.dtype() == DType::Int64 || input.dtype() == DType::UInt64) {
        #define TP_MIN_INT64_CASE(ctype, name) \
        case DType::name: { \
            binary_kernel_reduce(iter, \
                ExtremumValuePairOps<ctype>{[](ctype a, ctype b) { return a < b; }}, \
                std::pair<ctype, int64_t>(get_highest<ctype>(), -1)); \
            break; \
        }
        switch (input.dtype()) {
            TP_MIN_INT64_CASE(int64_t, Int64)
            TP_MIN_INT64_CASE(uint64_t, UInt64)
            default: break;
        }
        #undef TP_MIN_INT64_CASE
        return out;
    }

    #define TP_MIN_VALUES_CASE(ctype, name) \
    case DType::name: \
        binary_kernel_reduce_vec(iter, \
            [](ctype a, ctype b) -> ctype { return nan_min(a, b); }, \
            [](Vectorized<ctype> a, Vectorized<ctype> b) { return minimum(a, b); }, \
            static_cast<double>(get_highest<ctype>())); \
        break;

    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MIN_VALUES_CASE)
        default: TP_THROW(NotImplementedError, "min not implemented for this dtype");
    }
    #undef TP_MIN_VALUES_CASE
    return out;
}

std::tuple<Tensor, Tensor> min_dim_kernel_impl(const Tensor& self, int64_t dim0, bool keepdim) {
    // FIRST minimal index.
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "min(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(IndexError, "min(): Expected reduction dim ", dim, " to have non-zero size.");
    }

    Tensor sc = self.contiguous();
    std::vector<int64_t> in_shape = static_cast<std::vector<int64_t>>(sc.shape());
    const int64_t d_size = in_shape[dim];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= in_shape[i];
    for (int64_t i = dim + 1; i < nd; ++i) inner *= in_shape[i];

    std::vector<int64_t> out_shape = compute_reduction_shape(sc, {dim}, keepdim);
    Tensor vals = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(out_shape, DType::Int64, sc.device());

#if defined(__x86_64__)
    if (inner == 1 && try_extremum_lastdim_real<false>(
            sc.data_ptr(), vals.data_ptr(), idxs.data_ptr<int64_t>(), outer, d_size,
            sc.dtype())) {
        return {vals, idxs};
    }
    if (inner == 1 && try_extremum_lastdim_integral_avx512<false>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size)) {
        return {vals, idxs};
    }
    if (try_extremum_integral_dim_avx512<false>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size, inner)) {
        return {vals, idxs};
    }
    if (inner == 1 && try_extremum_lastdim_vec<false>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size)) {
        return {vals, idxs};
    }
    if (try_extremum_reduced_columns<false>(
            sc, vals, idxs.data_ptr<int64_t>(), outer, d_size, inner)) {
        return {vals, idxs};
    }
#endif

    // A NaN candidate wins once, then a NaN running best is never
    // displaced, so the first NaN fixes both the value and the index; for
    // integral element types both added comparisons are tautologies the
    // compiler folds away.
#define TP_MIN_DIM_CASE(ctype, name_)                                                   \
    case DType::name_: {                                                                \
        const ctype* sp = sc.data_ptr<ctype>();                                         \
        ctype* vp = vals.data_ptr<ctype>();                                             \
        int64_t* ip = idxs.data_ptr<int64_t>();                                         \
        const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size);           \
        parallel_for(0, outer * inner, line_grain, [&](int64_t b, int64_t e) {          \
            for (int64_t flat = b; flat < e; ++flat) {                                  \
                const int64_t o = flat / inner, in2 = flat % inner;                     \
                const ctype* line = sp + o * d_size * inner + in2;                      \
                ctype best = line[0];                                                   \
                int64_t bi = 0;                                                         \
                for (int64_t i = 1; i < d_size; ++i) {                                  \
                    const ctype x = line[i * inner];                                    \
                    if (best == best && (x != x || x < best)) {                         \
                        best = x;                                                       \
                        bi = i;                                                         \
                    }                                                                   \
                }                                                                       \
                vp[flat] = best;                                                        \
                ip[flat] = bi;                                                          \
            }                                                                           \
        });                                                                             \
        break;                                                                          \
    }
    switch (sc.dtype()) {
        TP_MIN_DIM_CASE(uint8_t, UInt8)
        TP_MIN_DIM_CASE(int8_t, Int8)
        TP_MIN_DIM_CASE(int16_t, Int16)
        TP_MIN_DIM_CASE(int32_t, Int32)
        TP_MIN_DIM_CASE(int64_t, Int64)
        TP_MIN_DIM_CASE(uint16_t, UInt16)
        TP_MIN_DIM_CASE(uint32_t, UInt32)
        TP_MIN_DIM_CASE(uint64_t, UInt64)
        TP_MIN_DIM_CASE(float, Float32)
        TP_MIN_DIM_CASE(double, Float64)
        TP_MIN_DIM_CASE(Half, Float16)
        TP_MIN_DIM_CASE(BFloat16, BFloat16)
        TP_MIN_DIM_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError, "min_dim not implemented for this dtype");
    }
#undef TP_MIN_DIM_CASE
    return {vals, idxs};
}



// Product
Tensor prod_kernel_impl(const Tensor& self, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }
    
    Tensor out = Tensor::empty({}, out_dtype, self.device());
    
    Tensor self_contig = self.contiguous();
    if (self_contig.dtype() != out_dtype) {
        self_contig = self_contig.to(out_dtype);
    }

#if defined(__x86_64__)
    if (self_contig.is_contiguous() && self_contig.numel() > 0) {
        double product = 1.0;
        if (try_product_real_avx512(self_contig.data_ptr(), self_contig.numel(),
                                    self_contig.dtype(), &product)) {
            if (self_contig.dtype() == DType::Float32) {
                out.fill_(Scalar(static_cast<float>(product)));
            } else {
                out.fill_(Scalar(product));
            }
            return out;
        }
        if (try_product_real_avx2(self_contig.data_ptr(), self_contig.numel(),
                                  self_contig.dtype(), &product)) {
            if (self_contig.dtype() == DType::Float32) {
                out.fill_(Scalar(static_cast<float>(product)));
            } else {
                out.fill_(Scalar(product));
            }
            return out;
        }
    }
#endif

    if (self_contig.numel() > 0) {
        TensorIterator iter = TensorIterator::reduce_op(out, self_contig);
        if (try_product_kernel_vec(iter)) return out;
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        /* direct-init works for both scalars (T(1)) and complex types
           (complex<T>(T(1))); plain `= 1` breaks reduced complexes */ \
        ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        ctype prod_val = product_reduce_parallel(data, n); \
        out.fill_(to_scalar(prod_val)); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(NotImplementedError, "prod not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

// One contiguous run multiplied through four independent lane accumulators, so
// the chain does not wait the multiply's latency on every step, then folded
// into the single answer the run carries.
template <typename T>
inline T product_row(const T* x, int64_t n) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = Vec::size();
    Vec a0(T(1)), a1(T(1)), a2(T(1)), a3(T(1));
    int64_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        a0 = a0 * Vec::loadu(x + i);
        a1 = a1 * Vec::loadu(x + i + W);
        a2 = a2 * Vec::loadu(x + i + 2 * W);
        a3 = a3 * Vec::loadu(x + i + 3 * W);
    }
    Vec acc = (a0 * a1) * (a2 * a3);
    for (; i + W <= n; i += W) acc = acc * Vec::loadu(x + i);
    alignas(64) T lanes[W];
    acc.store(lanes);
    T value = T(1);
    for (int64_t k = 0; k < W; ++k) value *= lanes[k];
    for (; i < n; ++i) value *= x[i];
    return value;
}

// A narrow middle slice of a real width multiplied a few bands of columns at a
// time: the totals stay in registers over the whole run and four rows are
// multiplied among themselves first, so the chain of multiplies on one band is
// never what the loop waits on, and a column of small factors crosses the range
// where a multiply costs far more than a normal one a group of rows at a time.
// Columns past the last whole band are answered one at a time, so a slice that
// is not a whole number of bands costs a handful of multiplies and no masking.
template <typename T, int64_t NB>
inline void product_band_tree_real(const T* in, T* out, int64_t outer,
                                  int64_t red, int64_t inner) {
    using Vec = Vectorized<T>;
    constexpr int64_t W = Vec::size();
    auto fold_rows = [](const T* p, int64_t cols) {
        return (Vec::loadu(p) * Vec::loadu(p + cols)) *
               (Vec::loadu(p + 2 * cols) * Vec::loadu(p + 3 * cols));
    };
    for (int64_t o = 0; o < outer; ++o) {
        const T* base = in + o * red * inner;
        T* dst = out + o * inner;
        int64_t j = 0;
        for (; j + NB * W <= inner; j += NB * W) {
            Vec acc[NB];
            for (int64_t b = 0; b < NB; ++b) {
                acc[b] = Vec::loadu(base + j + b * W);
            }
            int64_t r = 1;
            for (; r + 4 <= red; r += 4) {
                const T* rows = base + r * inner + j;
                for (int64_t b = 0; b < NB; ++b) {
                    acc[b] = acc[b] * fold_rows(rows + b * W, inner);
                }
            }
            for (; r < red; ++r) {
                const T* row = base + r * inner + j;
                for (int64_t b = 0; b < NB; ++b) {
                    acc[b] = acc[b] * Vec::loadu(row + b * W);
                }
            }
            for (int64_t b = 0; b < NB; ++b) {
                acc[b].store(dst + j + b * W);
            }
        }
        for (; j + W <= inner; j += W) {
            Vec acc = Vec::loadu(base + j);
            int64_t r = 1;
            for (; r + 4 <= red; r += 4) {
                acc = acc * fold_rows(base + r * inner + j, inner);
            }
            for (; r < red; ++r) {
                acc = acc * Vec::loadu(base + r * inner + j);
            }
            acc.store(dst + j);
        }
        for (; j < inner; ++j) {
            T value = base[j];
            for (int64_t r = 1; r < red; ++r) value *= base[r * inner + j];
            dst[j] = value;
        }
    }
}

// The same over every block of the outer axis, split among the workers.
template <typename T>
inline void product_band_tree_rows(const T* in, T* out, int64_t outer,
                                   int64_t red, int64_t inner) {
    constexpr int64_t NB = 4;
    const int64_t grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(red * inner, 1));
    parallel_for(0, outer, grain, [&](int64_t b, int64_t e) {
        product_band_tree_real<T, NB>(in + b * red * inner, out + b * inner,
                                      e - b, red, inner);
    });
}

#if defined(__x86_64__)
// The same walk over a packed width: a band of eight columns is widened into
// single precision for the whole run, so the packed value is formed once when
// the band is written back.
template <bool BFloat, int64_t NB>
__attribute__((target("avx2,fma,f16c"), noinline))
void product_narrow_packed_hw(const uint16_t* in, uint16_t* out, int64_t outer,
                              int64_t red, int64_t inner) {
    constexpr int64_t W = 8;
    const int64_t keep = inner - (NB - 1) * W;
    for (int64_t o = 0; o < outer; ++o) {
        const uint16_t* base = in + o * red * inner;
        uint16_t* dst = out + o * inner;
        const bool last_block = o == outer - 1;
        __m256 acc[NB];
        if (last_block && red == 1 && keep < W) {
            for (int64_t b = 0; b < NB - 1; ++b) {
                acc[b] = widen_row8<BFloat>(base + b * W);
            }
            alignas(16) uint16_t copy[W];
            std::memcpy(copy, base + (NB - 1) * W,
                        static_cast<size_t>(keep) * sizeof(uint16_t));
            for (int64_t k = keep; k < W; ++k) {
                copy[k] = BFloat ? 0x3F80u : 0x3C00u;
            }
            acc[NB - 1] = widen_row8<BFloat>(copy);
        } else {
            for (int64_t b = 0; b < NB; ++b) {
                acc[b] = widen_row8<BFloat>(base + b * W);
            }
        }
        for (int64_t r = 1; r < red; ++r) {
            const uint16_t* row = base + r * inner;
            const bool last_row = last_block && r == red - 1;
            for (int64_t b = 0; b < NB - 1; ++b) {
                acc[b] = _mm256_mul_ps(acc[b], widen_row8<BFloat>(row + b * W));
            }
            if (last_row && keep < W) {
                alignas(16) uint16_t copy[W];
                std::memcpy(copy, row + (NB - 1) * W,
                            static_cast<size_t>(keep) * sizeof(uint16_t));
                for (int64_t k = keep; k < W; ++k) {
                    copy[k] = BFloat ? 0x3F80u : 0x3C00u;
                }
                acc[NB - 1] =
                    _mm256_mul_ps(acc[NB - 1], widen_row8<BFloat>(copy));
            } else {
                acc[NB - 1] = _mm256_mul_ps(
                    acc[NB - 1], widen_row8<BFloat>(row + (NB - 1) * W));
            }
        }
        for (int64_t b = 0; b < NB - 1; ++b) {
            store_band_hw<BFloat>(dst, b * W, acc[b]);
        }
        const int64_t lanes = keep < W ? keep : W;
        alignas(32) float held[W];
        _mm256_store_ps(held, acc[NB - 1]);
        for (int64_t k = 0; k < lanes; ++k) {
            dst[(NB - 1) * W + k] = BFloat
                ? tensorplay::cpu::vecreduced::narrow_bf16_one(held[k])
                : tensorplay::cpu::vecreduced::narrow_one_hw(held[k]);
        }
    }
}

template <bool BFloat>
inline void product_narrow_packed(const uint16_t* in, uint16_t* out,
                                  int64_t outer, int64_t red, int64_t inner,
                                  int64_t bands) {
    switch (bands) {
        case 1:
            product_narrow_packed_hw<BFloat, 1>(in, out, outer, red, inner);
            return;
        case 2:
            product_narrow_packed_hw<BFloat, 2>(in, out, outer, red, inner);
            return;
        case 3:
            product_narrow_packed_hw<BFloat, 3>(in, out, outer, red, inner);
            return;
        case 4:
            product_narrow_packed_hw<BFloat, 4>(in, out, outer, red, inner);
            return;
        default:
            return;
    }
}
#endif

// Dispatch the narrow-inner product over the widths it serves.  A wider run is
// a stream and the walks that read it whole answer it better.
inline bool product_narrow_dim(const Tensor& input, Tensor& out, int64_t outer,
                               int64_t red, int64_t inner) {
    if (!input.is_contiguous() || out.numel() != outer * inner ||
        input.dim() == 0 || outer <= 0 || red < 1 || inner < 2) {
        return false;
    }
    const DType dt = input.dtype();
    if (dt == DType::Float32) {
        // Eight bands of a single-precision run stay in the register file; a
        // wider slice is walked a block of columns at a time by the caller.
        if (inner > 8 * static_cast<int64_t>(Vectorized<float>::size())) {
            return false;
        }
        product_band_tree_rows<float>(input.data_ptr<float>(),
                                      out.data_ptr<float>(), outer, red,
                                      inner);
        return true;
    }
    if (dt == DType::Float64) {
        if (inner > 8 * static_cast<int64_t>(Vectorized<double>::size())) {
            return false;
        }
        product_band_tree_rows<double>(input.data_ptr<double>(),
                                       out.data_ptr<double>(), outer, red,
                                       inner);
        return true;
    }
#if defined(__x86_64__)
    if (dt != DType::Float16 && dt != DType::BFloat16) return false;
    if (inner > 32) return false;
    if (dt == DType::Float16 &&
        !tensorplay::cpu::vecreduced::f16c_available()) {
        return false;
    }
    const int64_t bands = (inner + 7) / 8;
    if (dt == DType::Float16) {
        const uint16_t* in = reinterpret_cast<const uint16_t*>(
            input.data_ptr<tensorplay::Half>());
        uint16_t* op = reinterpret_cast<uint16_t*>(
            out.data_ptr<tensorplay::Half>());
        product_narrow_packed<false>(in, op, outer, red, inner, bands);
    } else {
        const uint16_t* in = reinterpret_cast<const uint16_t*>(
            input.data_ptr<tensorplay::BFloat16>());
        uint16_t* op = reinterpret_cast<uint16_t*>(
            out.data_ptr<tensorplay::BFloat16>());
        product_narrow_packed<true>(in, op, outer, red, inner, bands);
    }
    return true;
#else
    return false;
#endif
}

Tensor prod_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim, DType dtype) {
    if (std::optional<Tensor> permuted = review_permuted_reduction(
            self, dims, keepdim,
            [&](const Tensor& dense, const std::vector<int64_t>& dense_dims) {
                return prod_dim_kernel_impl(dense, dense_dims, true, dtype);
            })) {
        return std::move(*permuted);
    }

    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }
    
    if (dims.empty()) {
        return prod_kernel_impl(self, dtype);
    }

    // Every axis named is the whole tensor, which the flat product walk
    // answers without the per-element iterator.
    if (names_every_axis(dims, self.dim())) {
        Tensor full = prod_kernel_impl(self, dtype);
        if (!keepdim) return full;
        return full.reshape(
            std::vector<int64_t>(static_cast<size_t>(self.dim()), 1));
    }
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);

    // An integral tensor reduced over the last axis keeps each row contiguous,
    // so the rows are multiplied where they lie.
#if defined(TP_VECREDUCED_X86)
    if (dims.size() == 1 && out_dtype == DType::Int64) {
        const int64_t rank = self.dim();
        const int64_t axis = dims[0] < 0 ? dims[0] + rank : dims[0];
        if (rank >= 1 && axis == rank - 1 && self.is_contiguous()) {
            if (std::optional<Tensor> direct =
                    try_prod_lastdim_int(self, out_shape)) {
                return std::move(*direct);
            }
        }
    }

    // An integral tensor reduced over a middle axis keeps each plane of rows
    // contiguous, so the planes are multiplied where they lie instead of
    // through a converted copy of the whole input.
    if (dims.size() == 1 && out_dtype == DType::Int64) {
        const int64_t rank = self.dim();
        const int64_t axis = dims[0] < 0 ? dims[0] + rank : dims[0];
        if (rank >= 3 && axis > 0 && axis < rank - 1 &&
            self.is_contiguous()) {
            if (std::optional<Tensor> direct =
                    try_prod_middle_int(self, out_shape, axis)) {
                return std::move(*direct);
            }
        }
    }
#endif  // TP_VECREDUCED_X86

    Tensor out = Tensor::empty(out_shape, out_dtype, self.device());

    Tensor self_in = self;
    if (self.dtype() != out_dtype) {
        self_in = self.to(out_dtype);
    }

    if (dims.size() == 1) {
        int64_t dim = dims[0];
        if (dim < 0) dim += self_in.dim();
        Tensor input = self_in.contiguous();
        std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
        const int64_t d_size = shape[dim];
        if (d_size == 0) {
            out.fill_(Scalar(1));
            return out;
        }
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
        for (int64_t i = dim + 1; i < input.dim(); ++i) inner *= shape[i];

#if defined(__x86_64__)
        // A reduced width is multiplied where it lies: a row of lanes over the
        // last axis, whole rows into column totals over the leading one.
        if (input.dtype() == out_dtype && d_size >= 1) {
            if (dim == input.dim() - 1) {
                if (std::optional<Tensor> folded =
                        try_product_lastdim_reduced(input, out_shape)) {
                    return std::move(*folded);
                }
            } else if (dim == 0) {
                if (std::optional<Tensor> folded =
                        try_product_leading_reduced(input, out_shape)) {
                    return std::move(*folded);
                }
            }
        }
        // A leading axis of rows is multiplied where it lies: the rows stream
        // once into per-column totals, so no output element is walked a second
        // time to reach its own column.  An integral tensor is kept in the wide
        // integer its answer is stored in, where the product wraps modulo that
        // width -- which is the answer a product of any narrower width gives.
        if (dim == 0 && input.dtype() == out_dtype &&
            (out_dtype == DType::Float32 || out_dtype == DType::Float64 ||
             out_dtype == DType::Int64) &&
            d_size >= 2) {
            if (out_dtype == DType::Float32) {
                product_band_tree_rows<float>(input.data_ptr<float>(),
                                              out.data_ptr<float>(), outer,
                                              d_size, inner);
            } else if (out_dtype == DType::Float64) {
                product_band_tree_rows<double>(input.data_ptr<double>(),
                                               out.data_ptr<double>(), outer,
                                               d_size, inner);
            } else {
                product_band_tree_rows<int64_t>(
                    input.data_ptr<int64_t>(), out.data_ptr<int64_t>(), outer,
                    d_size, inner);
            }
            return out;
        }
        if (inner == 1 && try_product_lastdim_real_avx512(
                input.data_ptr(), out.data_ptr(), outer, d_size, input.dtype())) {
            return out;
        }
        if (input.dtype() == out_dtype &&
            try_product_dim_real_avx512(input, out, outer, d_size, inner)) {
            return out;
        }
        // An inner run that is narrower than a stream still holds several
        // bands: carrying them across the reduced rows answers a row in one
        // multiply per band, which the walks above leave to their tails.
        if (input.dtype() == out_dtype &&
            product_narrow_dim(input, out, outer, d_size, inner)) {
            return out;
        }
#endif
        // A reduced axis under an outer block, wider than a band: each block
        // multiplies its rows into its own column totals, where the iterator
        // would walk the reduced axis one output element at a time.
        if (input.dtype() == out_dtype && dim > 0 && dim + 1 < input.dim()) {
            const DType dt = input.dtype();
            if (dt == DType::Float32 &&
                inner > 8 * static_cast<int64_t>(Vectorized<float>::size())) {
                product_blocks_real<float>(input, out, outer, d_size, inner);
                return out;
            }
            if (dt == DType::Float64 &&
                inner > 8 * static_cast<int64_t>(Vectorized<double>::size())) {
                product_blocks_real<double>(input, out, outer, d_size, inner);
                return out;
            }
#if defined(__x86_64__)
            // Over a packed width the block walk pays a widening for every
            // factor it multiplies, so a run of rows long enough for the
            // reduction iterator is left to it, which widens the same way but
            // streams the blocks.  A run short enough to fall below the
            // iterator's threshold is answered one value at a time in the
            // packed width instead, where a widening per multiply costs more
            // than a whole band; the block walk carries the totals widened and
            // answers it in one multiply per factor.
            if ((dt == DType::Float16 || dt == DType::BFloat16) && inner > 32) {
                if (dt == DType::BFloat16) {
                    product_short_rows_packed<true>(input, out, outer, d_size,
                                                    inner);
                    return out;
                }
                if (tensorplay::cpu::vecreduced::f16c_available()) {
                    product_short_rows_packed<false>(input, out, outer, d_size,
                                                     inner);
                    return out;
                }
            }
#endif
        }
        // A trailing run is one contiguous stream per answer: it multiplies
        // where it lies instead of through the reduction iterator, which pays
        // its own setup for every row of a short answer.
        if (inner == 1 && input.dtype() == out_dtype && d_size >= 16 &&
            (out_dtype == DType::Float32 || out_dtype == DType::Float64)) {
            const int64_t row_grain =
                std::max<int64_t>(1, GRAIN_SIZE / d_size);
            if (out_dtype == DType::Float32) {
                const float* values = input.data_ptr<float>();
                float* outp = out.data_ptr<float>();
                parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                    for (int64_t r = b; r < e; ++r) {
                        outp[r] = product_row<float>(values + r * d_size, d_size);
                    }
                });
            } else {
                const double* values = input.data_ptr<double>();
                double* outp = out.data_ptr<double>();
                parallel_for(0, outer, row_grain, [&](int64_t b, int64_t e) {
                    for (int64_t r = b; r < e; ++r) {
                        outp[r] = product_row<double>(values + r * d_size, d_size);
                    }
                });
            }
            return out;
        }

        // Leading-dim product of a real matrix: walk whole rows into per-thread
        // column buffers, so reads stay contiguous, then fold the buffers. Four
        // rows are in flight before an accumulator slot is read back, which is
        // what separates a memory-bound pass from an L1-traffic-bound one. Only
        // the single-block case (outer == 1) is covered; wider reductions keep
        // the iterator path.
        if (dim == 0 && outer == 1 &&
            (out_dtype == DType::Float32 || out_dtype == DType::Float64)) {
            const int64_t rows = d_size;
            const int64_t cols = inner;
            auto column_product = [&](auto identity) {
                using CT = decltype(identity);
                const CT* in = input.data_ptr<CT>();
                CT* outp = out.data_ptr<CT>();
                const int64_t nthreads =
                    std::max<int64_t>(1, tensorplay::parallel::get_num_threads());
                std::vector<std::vector<CT>> partials(
                    nthreads, std::vector<CT>(cols, identity));
                const int64_t row_grain = std::max<int64_t>(
                    1, GRAIN_SIZE / std::max<int64_t>(cols, 1));
                tensorplay::parallel::parallel_for(0, rows, row_grain, [&](int64_t rb, int64_t re) {
                    CT* TP_RESTRICT acc =
                        partials[tensorplay::parallel::get_thread_num()].data();
                    const int64_t re4 = rb + (re - rb) / 4 * 4;
                    for (int64_t r = rb; r < re4; r += 4) {
                        const CT* TP_RESTRICT r0 = in + r * cols;
                        const CT* TP_RESTRICT r1 = r0 + cols;
                        const CT* TP_RESTRICT r2 = r1 + cols;
                        const CT* TP_RESTRICT r3 = r2 + cols;
                        for (int64_t j = 0; j < cols; ++j)
                            acc[j] *= ((r0[j] * r1[j]) * (r2[j] * r3[j]));
                    }
                    for (int64_t r = re4; r < re; ++r) {
                        const CT* TP_RESTRICT rowp = in + r * cols;
                        for (int64_t j = 0; j < cols; ++j) acc[j] *= rowp[j];
                    }
                });
                std::vector<CT> total(cols, identity);
                for (int64_t t = 0; t < nthreads; ++t)
                    for (int64_t j = 0; j < cols; ++j) total[j] *= partials[t][j];
                for (int64_t j = 0; j < cols; ++j) outp[j] = total[j];
            };
            if (out_dtype == DType::Float32) column_product(1.0f);
            else column_product(1.0);
            return out;
        }

        if (d_size >= 16) {
            std::vector<bool> mask(input.dim(), false);
            mask[dim] = true;
            Tensor viewed = review_reduce_result(out, input.dim(), mask, keepdim);
            TensorIterator iter = TensorIterator::reduce_op(viewed, input);
            if (try_product_kernel_vec(iter)) return out;
        }

#define TP_PROD_DIM_FAST_CASE(ctype, name) \
        case DType::name: { \
            const ctype* input_data = input.data_ptr<ctype>(); \
            ctype* output_data = out.data_ptr<ctype>(); \
            const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * inner, line_grain, [&](int64_t begin, int64_t end) { \
                for (int64_t flat = begin; flat < end; ++flat) { \
                    const int64_t outer_index = flat / inner; \
                    const int64_t inner_index = flat % inner; \
                    const ctype* line = input_data + \
                        outer_index * d_size * inner + inner_index; \
                    ctype value = ctype(1); \
                    for (int64_t i = 0; i < d_size; ++i) { \
                        Accumulator<ctype>::mul(value, line[i * inner]); \
                    } \
                    output_data[flat] = value; \
                } \
            }); \
            break; \
        }
        switch (out_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_PROD_DIM_FAST_CASE)
            default: TP_THROW(NotImplementedError, "prod_dim not implemented for this dtype");
        }
#undef TP_PROD_DIM_FAST_CASE
        return out;
    }
    
    out.fill_(Scalar(1));

    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self_in.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self_in.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self_in.data_ptr<ctype>(); \
        ctype* out_data = out.data_ptr<ctype>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                Accumulator<ctype>::mul(out_data[out_off], inp_data[inp_off]); \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: TP_THROW(NotImplementedError, "prod_dim not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}



// All/Any
Tensor all_kernel_impl(const Tensor& self) {
    Tensor out = Tensor::zeros({}, DType::Bool, self.device());
    Tensor self_contig = self.contiguous();

#if defined(__x86_64__)
    if (self_contig.dtype() == DType::Float32 ||
        self_contig.dtype() == DType::Float64) {
        const bool value = logical_reduce_full_avx512<true>(
            self_contig.data_ptr(), self_contig.numel(), self_contig.dtype());
        if (reduce_avx512_available()) {
            out.fill_(Scalar(value));
            return out;
        }
    }
#endif
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        bool val; \
        if constexpr (sizeof(ctype) == 1) { \
            val = byte_reduce_parallel<true>(reinterpret_cast<const uint8_t*>(data), n); \
        } else { \
            val = all_reduce_parallel(data, n); \
        } \
        out.fill_(Scalar(val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "all not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor any_kernel_impl(const Tensor& self) {
    Tensor out = Tensor::zeros({}, DType::Bool, self.device());
    Tensor self_contig = self.contiguous();

#if defined(__x86_64__)
    if (self_contig.dtype() == DType::Float32 ||
        self_contig.dtype() == DType::Float64) {
        const bool value = logical_reduce_full_avx512<false>(
            self_contig.data_ptr(), self_contig.numel(), self_contig.dtype());
        if (reduce_avx512_available()) {
            out.fill_(Scalar(value));
            return out;
        }
    }
#endif
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        bool val; \
        if constexpr (sizeof(ctype) == 1) { \
            val = byte_reduce_parallel<false>(reinterpret_cast<const uint8_t*>(data), n); \
        } else { \
            val = any_reduce_parallel(data, n); \
        } \
        out.fill_(Scalar(val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "any not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor all_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    if (dims.empty()) return all_kernel_impl(self);
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::ones(out_shape, DType::Bool, self.device()); // Init with True

    if (dims.size() == 1) {
        int64_t dim = dims[0];
        if (dim < 0) dim += self.dim();
        Tensor input = self.contiguous();
        std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
        const int64_t d_size = shape[dim];
        if (d_size == 0) return out;
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
        for (int64_t i = dim + 1; i < input.dim(); ++i) inner *= shape[i];

#if defined(__x86_64__)
        if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
            try_logical_reduce_dim_avx512<true>(
                input.data_ptr(), out.data_ptr<bool>(), outer, d_size, inner,
                self.dtype())) {
            return out;
        }
#endif

#if defined(__x86_64__)
        if ((self.dtype() == DType::UInt8 || self.dtype() == DType::Int8 ||
             self.dtype() == DType::Bool) &&
            try_byte_reduce_dim_avx512<true>(
                input.data_ptr<uint8_t>(), out.data_ptr<bool>(), outer, d_size,
                inner)) {
            return out;
        }
#endif

        // logical_reduce_dim0_rows is defined in the x86 section above.
#if defined(__x86_64__)
        if (outer == 1 && inner >= 16 && inner <= 262144) {
            switch (self.dtype()) {
#define TP_ALL_DIM_ROW_CASE(ctype, name) \
                case DType::name: \
                    logical_reduce_dim0_rows<ctype, true>( \
                        input.data_ptr<ctype>(), out.data_ptr<bool>(), d_size, \
                        inner); \
                    return out;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_ALL_DIM_ROW_CASE)
#undef TP_ALL_DIM_ROW_CASE
                default: break;
            }
        }
#endif

#define TP_ALL_DIM_FAST_CASE(ctype, name) \
        case DType::name: { \
            const ctype* input_data = input.data_ptr<ctype>(); \
            bool* output_data = out.data_ptr<bool>(); \
            const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * inner, line_grain, [&](int64_t begin, int64_t end) { \
                for (int64_t flat = begin; flat < end; ++flat) { \
                    const int64_t outer_index = flat / inner; \
                    const int64_t inner_index = flat % inner; \
                    const ctype* line = input_data + \
                        outer_index * d_size * inner + inner_index; \
                    bool value = true; \
                    for (int64_t i = 0; i < d_size; ++i) { \
                        if (!static_cast<bool>(line[i * inner])) { \
                            value = false; \
                            break; \
                        } \
                    } \
                    output_data[flat] = value; \
                } \
            }); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_ALL_DIM_FAST_CASE)
            default: TP_THROW(NotImplementedError, "all_dim not implemented for this dtype");
        }
#undef TP_ALL_DIM_FAST_CASE
        return out;
    }
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self.data_ptr<ctype>(); \
        bool* out_data = out.data_ptr<bool>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                if (!static_cast<bool>(inp_data[inp_off])) out_data[out_off] = false; \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "all_dim not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor any_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    if (dims.empty()) return any_kernel_impl(self);
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::zeros(out_shape, DType::Bool, self.device()); // Init with False

    if (dims.size() == 1) {
        int64_t dim = dims[0];
        if (dim < 0) dim += self.dim();
        Tensor input = self.contiguous();
        std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
        const int64_t d_size = shape[dim];
        if (d_size == 0) return out;
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
        for (int64_t i = dim + 1; i < input.dim(); ++i) inner *= shape[i];

#if defined(__x86_64__)
        if ((self.dtype() == DType::Float32 || self.dtype() == DType::Float64) &&
            try_logical_reduce_dim_avx512<false>(
                input.data_ptr(), out.data_ptr<bool>(), outer, d_size, inner,
                self.dtype())) {
            return out;
        }
#endif

#if defined(__x86_64__)
        if ((self.dtype() == DType::UInt8 || self.dtype() == DType::Int8 ||
             self.dtype() == DType::Bool) &&
            try_byte_reduce_dim_avx512<false>(
                input.data_ptr<uint8_t>(), out.data_ptr<bool>(), outer, d_size,
                inner)) {
            return out;
        }
#endif

        // logical_reduce_dim0_rows is defined in the x86 section above.
#if defined(__x86_64__)
        if (outer == 1 && inner >= 16 && inner <= 262144) {
            switch (self.dtype()) {
#define TP_ANY_DIM_ROW_CASE(ctype, name) \
                case DType::name: \
                    logical_reduce_dim0_rows<ctype, false>( \
                        input.data_ptr<ctype>(), out.data_ptr<bool>(), d_size, \
                        inner); \
                    return out;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_ANY_DIM_ROW_CASE)
#undef TP_ANY_DIM_ROW_CASE
                default: break;
            }
        }
#endif

#define TP_ANY_DIM_FAST_CASE(ctype, name) \
        case DType::name: { \
            const ctype* input_data = input.data_ptr<ctype>(); \
            bool* output_data = out.data_ptr<bool>(); \
            const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * inner, line_grain, [&](int64_t begin, int64_t end) { \
                for (int64_t flat = begin; flat < end; ++flat) { \
                    const int64_t outer_index = flat / inner; \
                    const int64_t inner_index = flat % inner; \
                    const ctype* line = input_data + \
                        outer_index * d_size * inner + inner_index; \
                    bool value = false; \
                    for (int64_t i = 0; i < d_size; ++i) { \
                        if (static_cast<bool>(line[i * inner])) { \
                            value = true; \
                            break; \
                        } \
                    } \
                    output_data[flat] = value; \
                } \
            }); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_ANY_DIM_FAST_CASE)
            default: TP_THROW(NotImplementedError, "any_dim not implemented for this dtype");
        }
#undef TP_ANY_DIM_FAST_CASE
        return out;
    }
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self.data_ptr<ctype>(); \
        bool* out_data = out.data_ptr<bool>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                if (static_cast<bool>(inp_data[inp_off])) out_data[out_off] = true; \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "any_dim not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}



// Argmax/Argmin
// Last-dimension argmax/argmin: the index travels with the value in the row
// fold, so a line resolves in one pass.
template <bool IsMax>
static bool try_argmax_lastdim_real(const void* xv, int64_t* indices,
                                    int64_t outer, int64_t d_size, DType dt) {
    if (d_size < 32 || outer <= 0) return false;
    const int64_t row_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (dt == DType::Float32) {
        const float* input = static_cast<const float*>(xv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    indices[row] = vec::row_extremum<IsMax>(
                        input + row * d_size, d_size).second;
                }
            });
        return true;
    }
    if (dt == DType::Float64) {
        const double* input = static_cast<const double*>(xv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    indices[row] = vec::row_extremum<IsMax>(
                        input + row * d_size, d_size).second;
                }
            });
        return true;
    }
    return false;
}

#if defined(__x86_64__)
// An extremum over a flattened real tensor: each chunk folds through the wide
// extremum walk, and the chunks meet in one pass that keeps the earliest
// index on a tie and the first not-a-number anywhere.
template <bool IsMax>
static bool try_extremum_flattened_real(const Tensor& sc, int64_t* answer) {
    const int64_t n = sc.numel();
    if (n <= 0) return false;
    const DType dt = sc.dtype();
    if (dt != DType::Float32 && dt != DType::Float64) return false;
    constexpr int64_t kChunks = 32;
    const int64_t chunk = std::max<int64_t>(1, (n + kChunks - 1) / kChunks);
    const int64_t count = (n + chunk - 1) / chunk;

    const auto finish = [&](auto* vals, const int64_t* poss) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(vals)>>;
        T best = IsMax ? get_lowest<T>() : get_highest<T>();
        int64_t best_pos = 0;
        for (int64_t c = 0; c < count; ++c) {
            if (std::isnan(vals[c])) {
                *answer = poss[c];
                return;
            }
            if (IsMax ? vals[c] > best : vals[c] < best) {
                best = vals[c];
                best_pos = poss[c];
            }
        }
        *answer = best_pos;
    };

    if (dt == DType::Float32) {
        const float* data = sc.data_ptr<float>();
        std::vector<float> vals(static_cast<size_t>(count), 0.f);
        std::vector<int64_t> poss(static_cast<size_t>(count), 0);
        parallel_for(0, count, 1, [&](int64_t b, int64_t e) {
            for (int64_t c = b; c < e; ++c) {
                const int64_t lo = c * chunk;
                const int64_t hi = std::min(n, lo + chunk);
                const auto found = vec::row_extremum<IsMax>(data + lo, hi - lo);
                vals[static_cast<size_t>(c)] = found.first;
                poss[static_cast<size_t>(c)] = lo + found.second;
            }
        });
        finish(vals.data(), poss.data());
        return true;
    }
    const double* data = sc.data_ptr<double>();
    std::vector<double> vals(static_cast<size_t>(count), 0.0);
    std::vector<int64_t> poss(static_cast<size_t>(count), 0);
    parallel_for(0, count, 1, [&](int64_t b, int64_t e) {
        for (int64_t c = b; c < e; ++c) {
            const int64_t lo = c * chunk;
            const int64_t hi = std::min(n, lo + chunk);
            const auto found = vec::row_extremum<IsMax>(data + lo, hi - lo);
            vals[static_cast<size_t>(c)] = found.first;
            poss[static_cast<size_t>(c)] = lo + found.second;
        }
    });
    finish(vals.data(), poss.data());
    return true;
}
#endif

Tensor argmax_kernel_impl(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        // empty flatten has no well-defined index.
        if (self.numel() == 0) {
            TP_THROW(IndexError,
                     "argmax(): Expected reduction dim to be specified for input.numel() == 0.");
        }
        // Flatten: chunked scan (chunk-local max + first-NaN marker), then a
        // serial combine that keeps the earliest index on ties/NaN.
        Tensor self_contig = self.contiguous();
        int64_t max_idx = 0;
#if defined(__x86_64__)
        if (try_extremum_flattened_real<true>(self_contig, &max_idx)) {
            Tensor out = Tensor::zeros({}, DType::Int64, self.device());
            out.fill_(Scalar(max_idx));
            return out;
        }
#endif

        #define OP_CASE(ctype, name) \
        case DType::name: { \
            const ctype* data = self_contig.data_ptr<ctype>(); \
            int64_t n = self_contig.numel(); \
            constexpr int64_t kChunks = 32; \
            const int64_t chunk = std::max<int64_t>(1, (n + kChunks - 1) / kChunks); \
            ctype best_val = get_lowest<ctype>(); \
            int64_t best_pos = 0; \
            int64_t nan_pos = -1; \
            std::vector<ctype> chunk_val(kChunks, get_lowest<ctype>()); \
            std::vector<int64_t> chunk_pos(kChunks, 0); \
            std::vector<int64_t> chunk_nan(kChunks, -1); \
            tensorplay::parallel::parallel_for(0, kChunks, 1, [&](int64_t cb, int64_t ce) { \
                for (int64_t c = cb; c < ce; ++c) { \
                    const int64_t lo = c * chunk; \
                    if (lo >= n) break; \
                    const int64_t hi = std::min(n, lo + chunk); \
                    ctype lmax = get_lowest<ctype>(); \
                    int64_t lpos = lo; \
                    int64_t lnan = -1; \
                    for (int64_t i = lo; i < hi; ++i) { \
                        if (lane_unnumbered(data[i])) { lnan = i; break; } \
                        if (data[i] > lmax) { lmax = data[i]; lpos = i; } \
                    } \
                    chunk_val[c] = lmax; chunk_pos[c] = lpos; chunk_nan[c] = lnan; \
                } \
            }); \
            for (int64_t c = 0; c < kChunks; ++c) { \
                if (chunk_nan[c] >= 0) { nan_pos = chunk_nan[c]; break; } \
                if (chunk_val[c] > best_val) { best_val = chunk_val[c]; best_pos = chunk_pos[c]; } \
            } \
            max_idx = (nan_pos >= 0) ? nan_pos : best_pos; \
            break; \
        }

        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(NotImplementedError, "argmax not implemented for this dtype");
        }
        #undef OP_CASE
        
        Tensor out = Tensor::zeros({}, DType::Int64, self.device());
        out.fill_(Scalar(max_idx));
        return out;
    }
    
    int64_t d = dim.value();
    if (d < 0) d += self.dim();
    TP_CHECK(d >= 0 && d < self.dim(),
             "Dimension out of range (expected to be in range of [-", self.dim(), ", ",
             self.dim() - 1, "], but got ", dim.value(), ")");
    if (self.size(d) == 0) {
        TP_THROW(IndexError, "argmax(): Expected reduction dim ", d, " to have non-zero size.");
    }
    
    // Direct strided-line scan over the contiguous input: line for (o, in2)
    // starts at o*d_size*inner + in2, elements at stride `inner`.  Rows are
    // independent, so the scan parallelizes at line granularity.
    Tensor sc = self.contiguous();
    std::vector<int64_t> in_shape = static_cast<std::vector<int64_t>>(sc.shape());
    const int64_t d_size = in_shape[d];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < d; ++i) outer *= in_shape[i];
    for (int64_t i = d + 1; i < sc.dim(); ++i) inner *= in_shape[i];

    std::vector<int64_t> out_shape = compute_reduction_shape(self, {d}, keepdim);
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    int64_t* out_data = out.data_ptr<int64_t>();
#if defined(__x86_64__)
    if (inner == 1 && try_argmax_lastdim_real<true>(
            sc.data_ptr(), out_data, outer, d_size, sc.dtype())) {
        return out;
    }
#endif
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = sc.data_ptr<ctype>(); \
        /* Columns that fill a line are scanned together, so each line of the  \
           reduced axis is read once instead of once per column. */ \
        constexpr int64_t kLanes = std::max<int64_t>(1, 64 / static_cast<int64_t>(sizeof(ctype))); \
        if (inner >= kLanes) { \
            const int64_t col_blocks = (inner + kLanes - 1) / kLanes; \
            const int64_t block_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * col_blocks, block_grain, [&](int64_t b, int64_t e) { \
                for (int64_t flat = b; flat < e; ++flat) { \
                    const int64_t o = flat / col_blocks; \
                    const int64_t c0 = (flat - o * col_blocks) * kLanes; \
                    const int64_t cols = std::min(kLanes, inner - c0); \
                    const ctype* base = data + o * d_size * inner + c0; \
                    int64_t* orow = out_data + o * inner + c0; \
                    if (cols == kLanes) { \
                        ctype best[kLanes]; \
                        int64_t bidx[kLanes]; \
                        bool bnan[kLanes]; \
                        for (int64_t c = 0; c < kLanes; ++c) { \
                            best[c] = base[c]; \
                            bidx[c] = 0; \
                            bnan[c] = lane_unnumbered(base[c]); \
                        } \
                        for (int64_t i = 1; i < d_size; ++i) { \
                            const ctype* row = base + i * inner; \
                            for (int64_t c = 0; c < kLanes; ++c) { \
                                if (bnan[c]) continue; \
                                const ctype x = row[c]; \
                                if (lane_unnumbered(x)) { \
                                    bnan[c] = true; \
                                    bidx[c] = i; \
                                } else if (x > best[c]) { \
                                    best[c] = x; \
                                    bidx[c] = i; \
                                } \
                            } \
                        } \
                        for (int64_t c = 0; c < kLanes; ++c) orow[c] = bidx[c]; \
                    } else { \
                        for (int64_t c = 0; c < cols; ++c) { \
                            const ctype* line = base + c; \
                            ctype max_val = line[0]; \
                            int64_t max_idx = 0; \
                            for (int64_t j = 0; j < d_size; ++j) { \
                                const ctype val = line[j * inner]; \
                                if (lane_unnumbered(val)) { max_idx = j; break; } \
                                if (val > max_val) { max_val = val; max_idx = j; } \
                            } \
                            orow[c] = max_idx; \
                        } \
                    } \
                } \
            }); \
            break; \
        } \
        const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
        parallel_for(0, outer * inner, line_grain, [&](int64_t b, int64_t e) { \
            for (int64_t flat = b; flat < e; ++flat) { \
                const int64_t o = flat / inner, in2 = flat % inner; \
                const ctype* line = data + o * d_size * inner + in2; \
                ctype max_val = line[0]; \
                int64_t max_idx = 0; \
                for (int64_t j = 0; j < d_size; ++j) { \
                    const ctype val = line[j * inner]; \
                    if (lane_unnumbered(val)) { max_idx = j; break; } \
                    if (val > max_val) { max_val = val; max_idx = j; } \
                } \
                out_data[flat] = max_idx; \
            } \
        }); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "argmax not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

Tensor argmin_kernel_impl(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        if (self.numel() == 0) {
            TP_THROW(IndexError,
                     "argmin(): Expected reduction dim to be specified for input.numel() == 0.");
        }
        // Flatten
        Tensor self_contig = self.contiguous();
        int64_t min_idx = 0;
#if defined(__x86_64__)
        if (try_extremum_flattened_real<false>(self_contig, &min_idx)) {
            Tensor out = Tensor::zeros({}, DType::Int64, self.device());
            out.fill_(Scalar(min_idx));
            return out;
        }
#endif
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            const ctype* data = self_contig.data_ptr<ctype>(); \
            int64_t n = self_contig.numel(); \
            min_idx = argmin_reduce_parallel(data, n); \
            break; \
        }
        
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(NotImplementedError, "argmin not implemented for this dtype");
        }
        #undef OP_CASE
        
        Tensor out = Tensor::zeros({}, DType::Int64, self.device());
        out.fill_(Scalar(min_idx));
        return out;
    }
    
    int64_t d = dim.value();
    if (d < 0) d += self.dim();
    TP_CHECK(d >= 0 && d < self.dim(),
             "Dimension out of range (expected to be in range of [-", self.dim(), ", ",
             self.dim() - 1, "], but got ", dim.value(), ")");
    if (self.size(d) == 0) {
        TP_THROW(IndexError, "argmin(): Expected reduction dim ", d, " to have non-zero size.");
    }

    Tensor sc = self.contiguous();
    std::vector<int64_t> in_shape = static_cast<std::vector<int64_t>>(sc.shape());
    const int64_t d_size = in_shape[d];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < d; ++i) outer *= in_shape[i];
    for (int64_t i = d + 1; i < sc.dim(); ++i) inner *= in_shape[i];
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, {d}, keepdim);
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    int64_t* out_data = out.data_ptr<int64_t>();
#if defined(__x86_64__)
    if (inner == 1 && try_argmax_lastdim_real<false>(
            sc.data_ptr(), out_data, outer, d_size, sc.dtype())) {
        return out;
    }
#endif
    
#define TP_ARGMIN_DIM_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = sc.data_ptr<ctype>(); \
        /* Columns that fill a line are scanned together, so each line of the  \
           reduced axis is read once instead of once per column. */ \
        constexpr int64_t kLanes = std::max<int64_t>(1, 64 / static_cast<int64_t>(sizeof(ctype))); \
        if (inner >= kLanes) { \
            const int64_t col_blocks = (inner + kLanes - 1) / kLanes; \
            const int64_t block_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * col_blocks, block_grain, [&](int64_t begin, int64_t end) { \
                for (int64_t flat = begin; flat < end; ++flat) { \
                    const int64_t outer_index = flat / col_blocks; \
                    const int64_t c0 = (flat - outer_index * col_blocks) * kLanes; \
                    const int64_t cols = std::min(kLanes, inner - c0); \
                    const ctype* base = data + outer_index * d_size * inner + c0; \
                    int64_t* orow = out_data + outer_index * inner + c0; \
                    if (cols == kLanes) { \
                        ctype value[kLanes]; \
                        int64_t index[kLanes]; \
                        bool unnumbered[kLanes]; \
                        for (int64_t c = 0; c < kLanes; ++c) { \
                            value[c] = get_highest<ctype>(); \
                            index[c] = 0; \
                            unnumbered[c] = false; \
                        } \
                        for (int64_t i = 0; i < d_size; ++i) { \
                            const ctype* row = base + i * inner; \
                            for (int64_t c = 0; c < kLanes; ++c) { \
                                if (unnumbered[c]) continue; \
                                const ctype candidate = row[c]; \
                                if (lane_unnumbered(candidate)) { \
                                    unnumbered[c] = true; \
                                    index[c] = i; \
                                } else if (candidate < value[c]) { \
                                    value[c] = candidate; \
                                    index[c] = i; \
                                } \
                            } \
                        } \
                        for (int64_t c = 0; c < kLanes; ++c) orow[c] = index[c]; \
                    } else { \
                        for (int64_t c = 0; c < cols; ++c) { \
                            const ctype* line = base + c; \
                            ctype value = get_highest<ctype>(); \
                            int64_t index = 0; \
                            for (int64_t i = 0; i < d_size; ++i) { \
                                const ctype candidate = line[i * inner]; \
                                if (lane_unnumbered(candidate)) { index = i; break; } \
                                if (candidate < value) { \
                                    value = candidate; \
                                    index = i; \
                                } \
                            } \
                            orow[c] = index; \
                        } \
                    } \
                } \
            }); \
            break; \
        } \
        const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
        parallel_for(0, outer * inner, line_grain, [&](int64_t begin, int64_t end) { \
            for (int64_t flat = begin; flat < end; ++flat) { \
                const int64_t outer_index = flat / inner; \
                const int64_t inner_index = flat % inner; \
                const ctype* line = data + \
                    outer_index * d_size * inner + inner_index; \
                ctype value = get_highest<ctype>(); \
                int64_t index = 0; \
                for (int64_t i = 0; i < d_size; ++i) { \
                    const ctype candidate = line[i * inner]; \
                    if (lane_unnumbered(candidate)) { index = i; break; } \
                    if (candidate < value) { \
                        value = candidate; \
                        index = i; \
                    } \
                } \
                out_data[flat] = index; \
            } \
        }); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ARGMIN_DIM_CASE)
        default: TP_THROW(NotImplementedError, "argmin not implemented for this dtype");
    }
#undef TP_ARGMIN_DIM_CASE
    
    return out;
}



// Var/Std








// Norm




Tensor median_kernel_impl(const Tensor& self) {
    Tensor t = detail::contiguous_clone(self).view({-1});
    int64_t n = t.numel();
    if (n == 0) {
        // NaN for float dtypes, converts to true for bool, lowest() for
        // signed ints and 0 for unsigned ints.
        Tensor out = Tensor::empty({}, self.dtype(), t.device());
#define TP_MED_EMPTY(ctype, name) \
    case DType::name: \
        *out.data_ptr<ctype>() = [] { \
            if constexpr (std::is_same_v<ctype, bool>) return ctype(true); \
            else if constexpr (std::is_integral_v<ctype>) \
                return std::is_signed_v<ctype> ? std::numeric_limits<ctype>::lowest() : ctype(0); \
            else return ctype(std::numeric_limits<double>::quiet_NaN()); \
        }(); \
        break;
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_MED_EMPTY)
            default: TP_THROW(NotImplementedError, "median not implemented for this dtype");
        }
#undef TP_MED_EMPTY
        return out;
    }

    // nth_element finds the n-th smallest element.
    // (n-1)/2 gives the lower index.
    int64_t mid = (n - 1) / 2;
    
    Tensor out = Tensor::zeros({}, self.dtype(), self.device());

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = t.data_ptr<ctype>(); \
        std::nth_element(data, data + mid, data + n); \
        out.fill_(to_scalar(data[mid])); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "median not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

} // anonymous namespace

#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_SUM_DISPATCH(name, fn) ALSO_REGISTER_AVX512_DISPATCH(name, fn)
#else
#define REGISTER_SUM_DISPATCH(name, fn) REGISTER_DISPATCH(name, fn)
#endif
REGISTER_SUM_DISPATCH(sum_stub, &sum_kernel_impl);
REGISTER_SUM_DISPATCH(sum_dim_stub, &sum_dim_kernel_impl);
REGISTER_DISPATCH(nansum_dim_stub, &nansum_dim_kernel_impl);
REGISTER_DISPATCH(sum_squared_error_stub, &sum_squared_error_kernel_impl);
REGISTER_DISPATCH(sum_abs_error_stub, &sum_abs_error_kernel_impl);
REGISTER_SUM_DISPATCH(extrema_dim_stub, &extrema_dim_kernel_impl);
#undef REGISTER_SUM_DISPATCH
REGISTER_DISPATCH(max_stub, &max_kernel_impl);
REGISTER_DISPATCH(max_dim_stub, &max_dim_kernel_impl);
REGISTER_DISPATCH(min_stub, &min_kernel_impl);
REGISTER_DISPATCH(min_dim_stub, &min_dim_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(prod_dim_stub, &prod_dim_kernel_impl);
REGISTER_DISPATCH(all_stub, &all_kernel_impl);
REGISTER_DISPATCH(all_dim_stub, &all_dim_kernel_impl);
REGISTER_DISPATCH(any_stub, &any_kernel_impl);
REGISTER_DISPATCH(any_dim_stub, &any_dim_kernel_impl);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl);
REGISTER_DISPATCH(median_stub, &median_kernel_impl);
REGISTER_DISPATCH(norm_stub, &norm_kernel_impl);
REGISTER_DISPATCH(norm_dim_stub, &norm_dim_kernel_impl);

} // namespace cpu
} // namespace tensorplay