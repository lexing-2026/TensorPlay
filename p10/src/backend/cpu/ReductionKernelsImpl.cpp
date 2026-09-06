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
#include "cpu/VecComplex.h"
#include <iostream>
#include <numeric>
#include <vector>
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
        if (dim < 0) dim += shape.size();
        if (dim < 0 || dim >= (int64_t)shape.size()) {
             TP_THROW(IndexError, format_dim_range(shape.size(), d));
        }
        is_reduced[dim] = true;
    }
    
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (is_reduced[i]) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    return out_shape;
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

// Scalar accumulator used for reduced-precision norm paths. TensorIterator
// promotes Half/BFloat16 input to float when the output is float, matching
template <typename scalar_t, typename acc_t, typename out_t = acc_t>
struct NormTwoOps {
    acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
        const acc_t value = static_cast<acc_t>(data);
        return acc + value * value;
    }
    acc_t combine(acc_t a, acc_t b) const { return a + b; }
    out_t project(acc_t value) const {
        return static_cast<out_t>(std::sqrt(value));
    }
    acc_t translate_idx(acc_t acc, int64_t) const { return acc; }
};

// L2 norm fast path. The previous TensorPlay implementation composed
// native path reduces squares directly; this is particularly important for
// Muon's bfloat16 normalization step.

// --- AVX-512 runtime-dispatched sum-of-squares (full-tensor contiguous) ----
#if defined(__x86_64__)
namespace {

inline bool normsq_avx512_available() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

__attribute__((target("avx512f")))
float normsq_f32_chunk_avx512(const float* x, int64_t b, int64_t e) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int64_t i = b;
    for (; i + 64 <= e; i += 64) {
        __m512 v;
        v = _mm512_loadu_ps(x + i);          a0 = _mm512_fmadd_ps(v, v, a0);
        v = _mm512_loadu_ps(x + i + 16);     a1 = _mm512_fmadd_ps(v, v, a1);
        v = _mm512_loadu_ps(x + i + 32);     a2 = _mm512_fmadd_ps(v, v, a2);
        v = _mm512_loadu_ps(x + i + 48);     a3 = _mm512_fmadd_ps(v, v, a3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    for (; i + 16 <= e; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    alignas(64) float buf[16];
    _mm512_storeu_ps(buf, acc);
    float s = ((buf[0] + buf[1]) + (buf[2] + buf[3])) +
              ((buf[4] + buf[5]) + (buf[6] + buf[7])) +
              ((buf[8] + buf[9]) + (buf[10] + buf[11])) +
              ((buf[12] + buf[13]) + (buf[14] + buf[15]));
    for (; i < e; ++i) { float v = x[i]; s += v * v; }
    return s;
}

__attribute__((target("avx512f")))
double normsq_f64_chunk_avx512(const double* x, int64_t b, int64_t e) {
    __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
    int64_t i = b;
    for (; i + 16 <= e; i += 16) {
        __m512d v;
        v = _mm512_loadu_pd(x + i);          a0 = _mm512_fmadd_pd(v, v, a0);
        v = _mm512_loadu_pd(x + i + 8);      a1 = _mm512_fmadd_pd(v, v, a1);
    }
    __m512d acc = _mm512_add_pd(a0, a1);
    alignas(64) double buf[8];
    _mm512_storeu_pd(buf, acc);
    double s = (buf[0] + buf[1]) + (buf[2] + buf[3]) +
               (buf[4] + buf[5]) + (buf[6] + buf[7]);
    for (; i < e; ++i) { double v = x[i]; s += v * v; }
    return s;
}

// Returns false when the caller should use the iterator path.  Emits the raw
// sum of squares; the caller applies the square root in the element dtype.
static bool try_normsq_real_avx512(const void* xv, int64_t n, DType dt,
                                   double* out) {
    if (!normsq_avx512_available() || n < 4096) return false;
    constexpr int64_t kGrain = 32768;
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        const int64_t nslots = (n + kGrain - 1) / kGrain;
        std::vector<float> part(nslots, 0.f);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[b / kGrain] = normsq_f32_chunk_avx512(x, b, e);
        });
        float s = 0.f;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = static_cast<double>(s);
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        const int64_t nslots = (n + kGrain - 1) / kGrain;
        std::vector<double> part(nslots, 0.0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[b / kGrain] = normsq_f64_chunk_avx512(x, b, e);
        });
        double s = 0.0;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = s;
        return true;
    }
    return false;
}

}  // namespace
#endif  // __x86_64__

Tensor norm_kernel_impl(const Tensor& self, double p) {
    TP_CHECK(p == 2.0, "norm: only p=2 supported by the native CPU path");
    if (self.numel() == 0) {
        return Tensor::zeros({}, self.dtype(), self.device());
    }

#if defined(__x86_64__)
    if (self.is_contiguous()) {
        double r = 0.0;
        if (try_normsq_real_avx512(self.data_ptr(), self.numel(), self.dtype(), &r)) {
            Tensor out = Tensor::empty({}, self.dtype(), self.device());
            if (self.dtype() == DType::Float32) out.fill_(Scalar(std::sqrt(static_cast<float>(r))));
            else out.fill_(Scalar(std::sqrt(r)));
            return out;
        }
    }
#endif
    if (self.dtype() == DType::Float32) {
        Tensor out = Tensor::zeros({}, DType::Float32, self.device());
        TensorIterator iter = TensorIterator::reduce_op(out, self);
        binary_kernel_reduce(iter, NormTwoOps<float, float>{}, 0.0f);
        return out;
    }
    if (self.dtype() == DType::Float64) {
        Tensor out = Tensor::zeros({}, DType::Float64, self.device());
        TensorIterator iter = TensorIterator::reduce_op(out, self);
        binary_kernel_reduce(iter, NormTwoOps<double, double>{}, 0.0);
        return out;
    }
    if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        // Keep the input type in the reduction op and accumulate in float.
        // TensorIterator's reduced-precision path performs the same
        Tensor out = Tensor::zeros({}, DType::Float32, self.device());
        TensorIterator iter = TensorIterator::reduce_op(out, self);
        if (self.dtype() == DType::Float16) {
            binary_kernel_reduce(iter, NormTwoOps<Half, float>{}, 0.0f);
        } else {
            binary_kernel_reduce(iter, NormTwoOps<BFloat16, float>{}, 0.0f);
        }
        return out.to(self.dtype());
    }
    TP_THROW(NotImplementedError, "norm: unsupported floating dtype on CPU");
}

Tensor norm_dim_kernel_impl(const Tensor& self,
                            const std::vector<int64_t>& dims,
                            double p, bool keepdim) {
    TP_CHECK(p == 2.0, "norm: only p=2 supported by the native CPU path");
    if (dims.empty()) return norm_kernel_impl(self, p);

    const std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    const bool reduced_precision =
        self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16;
    const DType acc_dtype = reduced_precision ? DType::Float32 : self.dtype();
    Tensor out = Tensor::zeros(out_shape, acc_dtype, self.device());

    std::vector<bool> mask(self.dim(), false);
    for (int64_t d : dims) {
        if (d < 0) d += self.dim();
        TP_CHECK(d >= 0 && d < self.dim(), "norm: dimension out of range");
        TP_CHECK(!mask[static_cast<size_t>(d)], "norm: duplicate dimension");
        mask[static_cast<size_t>(d)] = true;
    }
    Tensor viewed = review_reduce_result(out, self.dim(), mask, keepdim);
    TensorIterator iter = TensorIterator::reduce_op(viewed, self);

    if (self.dtype() == DType::Float32) {
        binary_kernel_reduce(iter, NormTwoOps<float, float>{}, 0.0f);
    } else if (self.dtype() == DType::Float64) {
        binary_kernel_reduce(iter, NormTwoOps<double, double>{}, 0.0);
    } else if (reduced_precision) {
        if (self.dtype() == DType::Float16) {
            binary_kernel_reduce(iter, NormTwoOps<Half, float>{}, 0.0f);
        } else {
            binary_kernel_reduce(iter, NormTwoOps<BFloat16, float>{}, 0.0f);
        }
    } else {
        TP_THROW(NotImplementedError, "norm: unsupported floating dtype on CPU");
    }
    return reduced_precision ? out.to(self.dtype()) : out;
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

static bool try_product_real_avx512(const void* xv, int64_t n, DType dt,
                                    double* out) {
    if (!reduce_avx512_available() || n < 4096) return false;
    constexpr int64_t kGrain = 32768;
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
static std::pair<float, int64_t> extremum_f32_row_avx512(
    const float* input, int64_t n) {
    constexpr int64_t width = 16;
    const __m512 lane_values = _mm512_loadu_ps(input);
    const __mmask16 initial_nan =
        _mm512_cmp_ps_mask(lane_values, lane_values, _CMP_UNORD_Q);
    if (initial_nan != 0) {
        return {std::numeric_limits<float>::quiet_NaN(),
                static_cast<int64_t>(__builtin_ctz(static_cast<unsigned>(initial_nan)))};
    }
    const __m512i lane_indices = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512 best_values = lane_values;
    __m512i best_indices = lane_indices;
    int64_t i = width;
    for (; i + width <= n; i += width) {
        const __m512 values = _mm512_loadu_ps(input + i);
        const __mmask16 nan_mask = _mm512_cmp_ps_mask(values, values, _CMP_UNORD_Q);
        if (nan_mask != 0) {
            return {std::numeric_limits<float>::quiet_NaN(),
                    i + static_cast<int64_t>(__builtin_ctz(static_cast<unsigned>(nan_mask)))};
        }
        const __m512i indices = _mm512_add_epi32(
            lane_indices, _mm512_set1_epi32(static_cast<int>(i)));
        const __mmask16 better = IsMax
            ? _mm512_cmp_ps_mask(values, best_values, _CMP_GT_OQ)
            : _mm512_cmp_ps_mask(values, best_values, _CMP_LT_OQ);
        best_values = _mm512_mask_blend_ps(better, best_values, values);
        best_indices = _mm512_mask_blend_epi32(better, best_indices, indices);
    }

    alignas(64) float values[width];
    alignas(64) int32_t indices[width];
    _mm512_storeu_ps(values, best_values);
    _mm512_storeu_si512(indices, best_indices);
    float best = values[0];
    int64_t best_index = indices[0];
    for (int lane = 1; lane < width; ++lane) {
        const bool better = IsMax ? values[lane] > best : values[lane] < best;
        if (better || (values[lane] == best && indices[lane] < best_index)) {
            best = values[lane];
            best_index = indices[lane];
        }
    }
    for (; i < n; ++i) {
        const float value = input[i];
        if (std::isnan(value)) {
            return {std::numeric_limits<float>::quiet_NaN(), i};
        }
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
static std::pair<double, int64_t> extremum_f64_row_avx512(
    const double* input, int64_t n) {
    constexpr int64_t width = 8;
    const __m512d lane_values = _mm512_loadu_pd(input);
    const __mmask8 initial_nan =
        _mm512_cmp_pd_mask(lane_values, lane_values, _CMP_UNORD_Q);
    if (initial_nan != 0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                static_cast<int64_t>(__builtin_ctz(static_cast<unsigned>(initial_nan)))};
    }
    const __m512i lane_indices = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512d best_values = lane_values;
    __m512i best_indices = lane_indices;
    int64_t i = width;
    for (; i + width <= n; i += width) {
        const __m512d values = _mm512_loadu_pd(input + i);
        const __mmask8 nan_mask = _mm512_cmp_pd_mask(values, values, _CMP_UNORD_Q);
        if (nan_mask != 0) {
            return {std::numeric_limits<double>::quiet_NaN(),
                    i + static_cast<int64_t>(__builtin_ctz(static_cast<unsigned>(nan_mask)))};
        }
        const __m512i indices = _mm512_add_epi64(
            lane_indices, _mm512_set1_epi64(i));
        const __mmask8 better = IsMax
            ? _mm512_cmp_pd_mask(values, best_values, _CMP_GT_OQ)
            : _mm512_cmp_pd_mask(values, best_values, _CMP_LT_OQ);
        best_values = _mm512_mask_blend_pd(better, best_values, values);
        best_indices = _mm512_mask_blend_epi64(better, best_indices, indices);
    }

    alignas(64) double values[width];
    alignas(64) int64_t indices[width];
    _mm512_storeu_pd(values, best_values);
    _mm512_storeu_si512(indices, best_indices);
    double best = values[0];
    int64_t best_index = indices[0];
    for (int lane = 1; lane < width; ++lane) {
        const bool better = IsMax ? values[lane] > best : values[lane] < best;
        if (better || (values[lane] == best && indices[lane] < best_index)) {
            best = values[lane];
            best_index = indices[lane];
        }
    }
    for (; i < n; ++i) {
        const double value = input[i];
        if (std::isnan(value)) {
            return {std::numeric_limits<double>::quiet_NaN(), i};
        }
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
static bool try_extremum_lastdim_real_avx512(
    const void* xv, void* vv, int64_t* indices, int64_t outer,
    int64_t d_size, DType dt) {
    if (!reduce_avx512_available() || d_size < 64 || outer <= 0) return false;
    const int64_t row_grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (dt == DType::Float32) {
        const float* input = static_cast<const float*>(xv);
        float* values = static_cast<float*>(vv);
        tensorplay::parallel::parallel_for(0, outer, row_grain,
            [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    auto result = extremum_f32_row_avx512<IsMax>(
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
                    auto result = extremum_f64_row_avx512<IsMax>(
                        input + row * d_size, d_size);
                    values[row] = result.first;
                    indices[row] = result.second;
                }
            });
        return true;
    }
    return false;
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

__attribute__((target("avx512f")))
static void sum_f32_leading_range_avx512(
    const float* input, float* output, int64_t rows, int64_t cols,
    int64_t begin, int64_t end) {
    int64_t col = begin;
    for (; col + 64 <= end; col += 64) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();
        for (int64_t row = 0; row < rows; ++row) {
            const float* row_input = input + row * cols + col;
            sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(row_input));
            sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(row_input + 16));
            sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(row_input + 32));
            sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(row_input + 48));
        }
        _mm512_storeu_ps(output + col, sum0);
        _mm512_storeu_ps(output + col + 16, sum1);
        _mm512_storeu_ps(output + col + 32, sum2);
        _mm512_storeu_ps(output + col + 48, sum3);
    }
    for (; col + 16 <= end; col += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (int64_t row = 0; row < rows; ++row) {
            sum = _mm512_add_ps(
                sum, _mm512_loadu_ps(input + row * cols + col));
        }
        _mm512_storeu_ps(output + col, sum);
    }
    for (; col < end; ++col) {
        float sum = 0.0f;
        for (int64_t row = 0; row < rows; ++row) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

__attribute__((target("avx512f")))
static void sum_f64_leading_range_avx512(
    const double* input, double* output, int64_t rows, int64_t cols,
    int64_t begin, int64_t end) {
    int64_t col = begin;
    for (; col + 32 <= end; col += 32) {
        __m512d sum0 = _mm512_setzero_pd();
        __m512d sum1 = _mm512_setzero_pd();
        __m512d sum2 = _mm512_setzero_pd();
        __m512d sum3 = _mm512_setzero_pd();
        for (int64_t row = 0; row < rows; ++row) {
            const double* row_input = input + row * cols + col;
            sum0 = _mm512_add_pd(sum0, _mm512_loadu_pd(row_input));
            sum1 = _mm512_add_pd(sum1, _mm512_loadu_pd(row_input + 8));
            sum2 = _mm512_add_pd(sum2, _mm512_loadu_pd(row_input + 16));
            sum3 = _mm512_add_pd(sum3, _mm512_loadu_pd(row_input + 24));
        }
        _mm512_storeu_pd(output + col, sum0);
        _mm512_storeu_pd(output + col + 8, sum1);
        _mm512_storeu_pd(output + col + 16, sum2);
        _mm512_storeu_pd(output + col + 24, sum3);
    }
    for (; col + 8 <= end; col += 8) {
        __m512d sum = _mm512_setzero_pd();
        for (int64_t row = 0; row < rows; ++row) {
            sum = _mm512_add_pd(
                sum, _mm512_loadu_pd(input + row * cols + col));
        }
        _mm512_storeu_pd(output + col, sum);
    }
    for (; col < end; ++col) {
        double sum = 0.0;
        for (int64_t row = 0; row < rows; ++row) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

static bool try_sum_dim_real_avx512(
    const Tensor& input, Tensor& output, int64_t dim) {
    if (!reduce_avx512_available() || !input.is_contiguous() ||
        input.numel() == 0 || input.dim() == 0) {
        return false;
    }
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        return false;
    }

    const int64_t ndim = input.dim();
    if (dim == ndim - 1) {
        const int64_t d_size = input.size(dim);
        const int64_t vector_width = input.dtype() == DType::Float32 ? 16 : 8;
        if (d_size < vector_width) return false;
        const int64_t rows = input.numel() / d_size;
        const int64_t row_grain = std::max<int64_t>(
            1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
        if (input.dtype() == DType::Float32) {
            const float* values = input.data_ptr<float>();
            float* results = output.data_ptr<float>();
            parallel_for(0, rows, row_grain, [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    results[row] = sum_f32_chunk_avx512(
                        values + row * d_size, 0, d_size);
                }
            });
        } else {
            const double* values = input.data_ptr<double>();
            double* results = output.data_ptr<double>();
            parallel_for(0, rows, row_grain, [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    results[row] = sum_f64_chunk_avx512(
                        values + row * d_size, 0, d_size);
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
    const int64_t vector_width = input.dtype() == DType::Float32 ? 16 : 8;
    const int64_t vector_columns = vector_width * 4;
    if (cols < vector_width) return false;
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
                sum_f32_leading_range_avx512(
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
                sum_f64_leading_range_avx512(
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
static bool try_sum_real_avx512(const void* xv, int64_t n, DType dt,
                                double* out) {
    if (!reduce_avx512_available() || n < 4096) return false;
    constexpr int64_t kGrain = 32768;
    if (dt == DType::Float32) {
        const float* x = static_cast<const float*>(xv);
        const int64_t nslots = (n + kGrain - 1) / kGrain;
        std::vector<float> part(nslots, 0.f);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[b / kGrain] = sum_f32_chunk_avx512(x, b, e);
        });
        float s = 0.f;
        for (int64_t k = 0; k < nslots; ++k) s += part[k];
        *out = s;
        return true;
    }
    if (dt == DType::Float64) {
        const double* x = static_cast<const double*>(xv);
        const int64_t nslots = (n + kGrain - 1) / kGrain;
        std::vector<double> part(nslots, 0.0);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            part[b / kGrain] = sum_f64_chunk_avx512(x, b, e);
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

static bool try_max_real_avx512(const void* xv, int64_t n, DType dt,
                                double* out, bool* has_nan) {
    if (!reduce_avx512_available() || n < 4096) return false;
    constexpr int64_t kGrain = 32768;
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

#if defined(__x86_64__)
    if ((acc_dtype == DType::Float32 || acc_dtype == DType::Float64) &&
        input.dtype() == acc_dtype && input.is_contiguous() &&
        input.numel() > 0) {
        double s = 0.0;
        if (try_sum_real_avx512(input.data_ptr(), input.numel(), acc_dtype, &s)) {
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

    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);

    DType acc_dtype = out_dtype;
    if (isReducedFloatingType(out_dtype)) {
        acc_dtype = DType::Float32;
    } else if (out_dtype == DType::ComplexHalf || out_dtype == DType::BComplex32) {
        acc_dtype = DType::ComplexFloat;
    }
    if (acc_dtype == DType::Float32 || acc_dtype == DType::Float64) {
        Tensor out = Tensor::zeros(out_shape, out_dtype, self.device());
        Tensor input = self;
        if (self.dtype() != out_dtype) {
            input = self.to(out_dtype);
        }

        const int64_t ndim = self.dim();
        std::vector<bool> mask(ndim, false);
        for (int64_t d : dims) {
            if (d < 0) d += ndim;
            mask[d] = true;
        }
        if (dims.size() == 1) {
            const int64_t dim = dims[0] < 0 ? dims[0] + ndim : dims[0];
#if defined(__x86_64__)
            if (input.dtype() == out_dtype &&
                try_sum_dim_real_avx512(input, out, dim)) {
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
                    const float* TP_RESTRICT rp = in + rb * cols;
                    for (int64_t r = rb; r < re; ++r, rp += cols) {
                        const float* TP_RESTRICT rowp = rp;
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
                    const double* TP_RESTRICT rp = in + rb * cols;
                    for (int64_t r = rb; r < re; ++r, rp += cols) {
                        const double* TP_RESTRICT rowp = rp;
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

template <typename scalar_t>
int64_t argmin_reduce_parallel(const scalar_t* data, int64_t n) {
    auto serial_reduce = [&]() {
        scalar_t value = get_highest<scalar_t>();
        int64_t index = 0;
        for (int64_t i = 0; i < n; ++i) {
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
        parallel_for(0, chunk_count, 1, [&](int64_t begin, int64_t end) {
            for (int64_t chunk = begin; chunk < end; ++chunk) {
                const int64_t lo = chunk * chunk_size;
                const int64_t hi = std::min(n, lo + chunk_size);
                scalar_t value = get_highest<scalar_t>();
                int64_t index = lo;
                for (int64_t i = lo; i < hi; ++i) {
                    if (data[i] < value) {
                        value = data[i];
                        index = i;
                    }
                }
                chunk_values[static_cast<size_t>(chunk)] = value;
                chunk_indices[static_cast<size_t>(chunk)] = index;
            }
        });

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
template <typename T>
inline T nan_max(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<T>::quiet_NaN();
    }
    return a < b ? b : a;
}

template <typename T>
inline T nan_min(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<T>::quiet_NaN();
    }
    return b < a ? b : a;
}

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
Tensor max_kernel_impl(const Tensor& self) {
    if (self.numel() == 0) {
        // over an empty tensor has no identity.
        TP_THROW(RuntimeError, "max(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    Tensor input = self.contiguous();
    Tensor out = Tensor::empty({}, self.dtype(), self.device());

#if defined(__x86_64__)
    if (input.is_contiguous() &&
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float64)) {
        double value = 0.0;
        bool has_nan = false;
        if (try_max_real_avx512(input.data_ptr(), input.numel(), input.dtype(),
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
    if (inner == 1 && try_extremum_lastdim_real_avx512<true>(
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
#endif

    // With the reduced dim removed (or sized 1 under keepdim), the output is
    // a contiguous [outer, inner] grid and line i lives at o*d_size*inner +
    // i*inner + in2 -- identical addressing for both keepdim modes.
#define TP_MAXMIN_DIM_CASE(ctype, name_, CMP_OP)                                        \
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
                    if (line[i * inner] CMP_OP best) {                                  \
                        best = line[i * inner];                                         \
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
    if (inner == 1 && try_extremum_lastdim_real_avx512<false>(
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
#endif

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
                    if (line[i * inner] < best) {                                       \
                        best = line[i * inner];                                         \
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

Tensor prod_dim_kernel_impl(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim, DType dtype) {
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
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
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
        if (inner == 1 && try_product_lastdim_real_avx512(
                input.data_ptr(), out.data_ptr(), outer, d_size, input.dtype())) {
            return out;
        }
        if (input.dtype() == out_dtype &&
            try_product_dim_real_avx512(input, out, outer, d_size, inner)) {
            return out;
        }
#endif

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

        #define OP_CASE(ctype, name) \
        case DType::name: { \
            const ctype* data = self_contig.data_ptr<ctype>(); \
            int64_t n = self_contig.numel(); \
            constexpr bool is_float = std::is_floating_point_v<ctype>; \
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
                        if constexpr (is_float) { \
                            if (lnan < 0 && std::isnan(data[i])) { lnan = i; break; } \
                        } \
                        if (lnan < 0 && data[i] > lmax) { lmax = data[i]; lpos = i; } \
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
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = sc.data_ptr<ctype>(); \
        const int64_t line_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
        parallel_for(0, outer * inner, line_grain, [&](int64_t b, int64_t e) { \
            for (int64_t flat = b; flat < e; ++flat) { \
                const int64_t o = flat / inner, in2 = flat % inner; \
                const ctype* line = data + o * d_size * inner + in2; \
                ctype max_val = line[0]; \
                int64_t max_idx = 0; \
                bool has_nan = false; \
                for (int64_t j = 0; j < d_size; ++j) { \
                    ctype val = line[j * inner]; \
                    if constexpr (std::is_floating_point_v<ctype>) { \
                        if (!has_nan && std::isnan(val)) { has_nan = true; max_idx = j; break; } \
                    } \
                    if (!has_nan && val > max_val) { max_val = val; max_idx = j; } \
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
    
#define TP_ARGMIN_DIM_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = sc.data_ptr<ctype>(); \
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
