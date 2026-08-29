#include "Tensor.h"
#include "Dispatcher.h"
#include "ErrorReporting.h"
#include "Exception.h"
#include "Utils.h"
#include "Parallel.h"
#include "ReductionKernels.h"
#include "TensorIterator.h"
#include "cpu/Reduce.h"
#include "cpu/vec/vec.h"
#include "cpu/VecComplex.h"
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

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
            return Scalar(std::complex<float>(
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
  return result.as_strided(shape, stride);
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
Tensor norm_kernel_impl(const Tensor& self, double p) {
    TP_CHECK(p == 2.0, "norm: only p=2 supported by the native CPU path");
    if (self.numel() == 0) {
        return Tensor::zeros({}, self.dtype(), self.device());
    }

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
            binary_kernel_reduce(iter, CxSumOps<std::complex<float>>{}, std::complex<float>(0));
            break;
        case DType::ComplexDouble:
            binary_kernel_reduce(iter, CxSumOps<std::complex<double>>{}, std::complex<double>(0));
            break;
        case DType::ComplexHalf:
        case DType::BComplex32:
            // Reduced complexes accumulate in complex64 (opmath rule); the
            // caller pre-casts the input to the acc dtype.
            binary_kernel_reduce(iter, CxSumOps<std::complex<float>>{}, std::complex<float>(0));
            break;
        default: TP_THROW(NotImplementedError, "sum not implemented for this dtype");
    }
    #undef OP_CASE
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

    Tensor out = Tensor::zeros({}, acc_dtype, self.device());

    // iterator's common dtype matches out_dtype.
    Tensor input = self;
    if (self.dtype() != acc_dtype) {
        input = self.to(acc_dtype);
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
                *out.data_ptr<std::complex<float>>() = std::complex<float>(
                    static_cast<float>(re), static_cast<float>(im));
            } else {
                *out.data_ptr<std::complex<double>>() =
                    std::complex<double>(re, im);
            }
            return acc_dtype == out_dtype ? out : out.to(out_dtype);
        }
    }

#if defined(__x86_64__)
    // AVX-512 full-reduction fast path for contiguous real sums.
    if ((acc_dtype == DType::Float32 || acc_dtype == DType::Float64) &&
        input.is_contiguous() && input.numel() > 0) {
        double s = 0.0;
        if (try_sum_real_avx512(input.data_ptr(), input.numel(), acc_dtype, &s)) {
            Tensor o = Tensor::zeros({}, acc_dtype, self.device());
            if (acc_dtype == DType::Float32)
                *o.data_ptr<float>() = static_cast<float>(s);
            else
                *o.data_ptr<double>() = s;
            return acc_dtype == out_dtype ? o : o.to(out_dtype);
        }
    }
#endif

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
    // complexes in complex64.
    DType acc_dtype = out_dtype;
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

    // With the reduced dim removed (or sized 1 under keepdim), the output is
    // a contiguous [outer, inner] grid and line i lives at o*d_size*inner +
    // i*inner + in2 -- identical addressing for both keepdim modes.
#define TP_MAXMIN_DIM_CASE(ctype, name_, CMP_OP)                                        \
    case DType::name_: {                                                                \
        const ctype* sp = sc.data_ptr<ctype>();                                         \
        ctype* vp = vals.data_ptr<ctype>();                                             \
        int64_t* ip = idxs.data_ptr<int64_t>();                                         \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) {          \
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

#define TP_MIN_DIM_CASE(ctype, name_)                                                   \
    case DType::name_: {                                                                \
        const ctype* sp = sc.data_ptr<ctype>();                                         \
        ctype* vp = vals.data_ptr<ctype>();                                             \
        int64_t* ip = idxs.data_ptr<int64_t>();                                         \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) {          \
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
    
    Tensor out = Tensor::zeros({}, out_dtype, self.device());
    
    Tensor self_contig = self.contiguous();
    if (self_contig.dtype() != out_dtype) {
        self_contig = self_contig.to(out_dtype);
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        /* direct-init works for both scalars (T(1)) and complex types
           (complex<T>(T(1))); plain `= 1` breaks reduced complexes */ \
        ctype prod_val = ctype(1); \
        ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) Accumulator<ctype>::mul(prod_val, data[i]); \
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
    Tensor out = Tensor::ones(out_shape, out_dtype, self.device());
    
    Tensor self_in = self;
    if (self.dtype() != out_dtype) {
        self_in = self.to(out_dtype);
    }
    
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
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        bool val = true; \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) { \
            if (!static_cast<bool>(data[i])) { val = false; break; } \
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
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        bool val = false; \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) { \
            if (static_cast<bool>(data[i])) { val = true; break; } \
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
        // Flatten
        Tensor self_contig = self.contiguous();
        int64_t max_idx = 0;
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            const ctype* data = self_contig.data_ptr<ctype>(); \
            int64_t n = self_contig.numel(); \
            ctype max_val = get_lowest<ctype>(); \
            bool has_nan = false; \
            for(int64_t i=0; i<n; ++i) { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    if (!has_nan && std::isnan(data[i])) { has_nan = true; max_idx = i; continue; } \
                } \
                if (!has_nan && data[i] > max_val) { max_val = data[i]; max_idx = i; } \
            } \
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
    
    // Transpose d to end, reshape to (-1, size), find max idx per row
    Tensor t = self.transpose(d, -1);
    t = t.contiguous(); // Force copy/compact
    
    int64_t size = t.size(-1);
    int64_t n_rows = t.numel() / size;
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, {d}, keepdim);
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    int64_t* out_data = out.data_ptr<int64_t>();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = t.data_ptr<ctype>(); \
        for(int64_t i=0; i<n_rows; ++i) { \
            ctype max_val = get_lowest<ctype>(); \
            int64_t max_idx = 0; \
            bool has_nan = false; \
            for(int64_t j=0; j<size; ++j) { \
                ctype val = data[i*size + j]; \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    if (!has_nan && std::isnan(val)) { has_nan = true; max_idx = j; break; } \
                } \
                if (!has_nan && val > max_val) { max_val = val; max_idx = j; } \
            } \
            out_data[i] = max_idx; \
        } \
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
            ctype min_val = get_highest<ctype>(); \
            for(int64_t i=0; i<n; ++i) { \
                if (data[i] < min_val) { min_val = data[i]; min_idx = i; } \
            } \
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
    if (self.size(d) == 0) {
        TP_THROW(IndexError, "argmin(): Expected reduction dim ", d, " to have non-zero size.");
    }
    
    // Transpose d to end, reshape to (-1, size), find min idx per row
    Tensor t = self.transpose(d, -1);
    t = t.contiguous(); 
    
    int64_t size = t.size(-1);
    int64_t n_rows = t.numel() / size;
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, {d}, keepdim);
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    int64_t* out_data = out.data_ptr<int64_t>();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = t.data_ptr<ctype>(); \
        for(int64_t i=0; i<n_rows; ++i) { \
            ctype min_val = get_highest<ctype>(); \
            int64_t min_idx = 0; \
            for(int64_t j=0; j<size; ++j) { \
                ctype val = data[i*size + j]; \
                if (val < min_val) { min_val = val; min_idx = j; } \
            } \
            out_data[i] = min_idx; \
        } \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "argmin not implemented for this dtype");
    }
    #undef OP_CASE
    
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

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);
REGISTER_DISPATCH(sum_dim_stub, &sum_dim_kernel_impl);
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
