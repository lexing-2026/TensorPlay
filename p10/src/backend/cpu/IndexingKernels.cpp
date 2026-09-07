// High-throughput indexing, masking, scan, and sorting operators.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"
#include "Parallel.h"
#include "Bucketization.h"
#include "cpu/ComplexUnary.h"

#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

inline int64_t wrap_scan_dim(int64_t dim, int64_t ndim) {
    if (ndim == 0) {
        if (dim == -1 || dim == 0) return 0;
        TP_THROW(IndexError,
                 "Dimension out of range for a scalar tensor (expected -1 or 0, but got ",
                 dim, ")");
    }
    return wrap_dim(dim, ndim);
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

Tensor normalize_index_cpu(const Tensor& index, int64_t upper_bound,
                           const char* op) {
    Tensor index_c = (index.dtype() == DType::Int64)
        ? index.contiguous()
        : index.to(DType::Int64).contiguous();
    const int64_t* input = index_c.data_ptr<int64_t>();
    parallel_for(0, index_c.numel(), GRAIN_SIZE,
                 [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t value = input[i];
            if (value < 0 || value >= upper_bound) {
                TP_THROW(IndexError, op, ": index out of range");
            }
        }
    });
    return index_c;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// masked_fill / masked_fill_
//
// Masks are broadcast to the input shape. The out-of-place operation copies
// the input before applying the selected value to matching elements.
// ---------------------------------------------------------------------------

Tensor masked_fill_cpu(const Tensor& self, const Tensor& mask, Scalar value) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(mask.shape()));
    Tensor self_b = self.expand(out_shape).contiguous();
    Tensor mask_b = mask.expand(out_shape).contiguous();
    Tensor result = self_b.clone();
    int64_t n = result.numel();
#define TP_MF_CASE(ctype, name) \
    case DType::name: { \
        const ctype* s = self_b.data_ptr<ctype>(); \
        const bool* m = mask_b.data_ptr<bool>(); \
        ctype v = value.to<ctype>(); \
        ctype* d = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) d[i] = m[i] ? v : s[i]; \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MF_CASE)
        default: TP_THROW(TypeError, "masked_fill: unsupported dtype");
    }
#undef TP_MF_CASE
    return result;
}

Tensor& masked_fill__cpu(Tensor& self, const Tensor& mask, Scalar value) {
    Tensor r = masked_fill_cpu(self, mask, value);
    self.copy_(r);
    return self;
}

Tensor& masked_fill_tensor__cpu(Tensor& self, const Tensor& mask, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return masked_fill__cpu(self, mask, value.item());
}

Tensor masked_fill_tensor_cpu(const Tensor& self, const Tensor& mask, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return masked_fill_cpu(self, mask, value.item());
}

// ---------------------------------------------------------------------------
// tril / triu
//
// (triu_cpu): keep element (r, c) iff c <= r + k (lower) / c >= r + k
// (upper), zero elsewhere.
// ---------------------------------------------------------------------------

template <bool Lower>
Tensor triangular_mask_kernel(const Tensor& self, int64_t diagonal) {
    int64_t ndim = self.dim();
    if (ndim < 2) TP_THROW(RuntimeError, "tril/triu requires tensor with at least 2 dimensions");
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t rows = self.size(ndim - 2);
    int64_t cols = self.size(ndim - 1);
    int64_t batch = self.numel() / (rows * cols);
    int64_t stride_rc = rows * cols;
#define TP_TRI_CASE(ctype, name) \
    case DType::name: { \
        const ctype* s = self_c.data_ptr<ctype>(); \
        ctype* d = result.data_ptr<ctype>(); \
        parallel_for(0, batch * rows, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t bi = t / rows, r = t % rows; \
                const ctype* sp = s + bi * stride_rc + r * cols; \
                ctype* dp = d + bi * stride_rc + r * cols; \
                for (int64_t c = 0; c < cols; ++c) { \
                    bool keep = Lower ? (c <= r + diagonal) : (c >= r + diagonal); \
                    dp[c] = keep ? sp[c] : static_cast<ctype>(0); \
                } \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TRI_CASE)
        default: TP_THROW(TypeError, "tril/triu: unsupported dtype");
    }
#undef TP_TRI_CASE
    return result;
}

Tensor tril_cpu(const Tensor& self, int64_t diagonal) {
    return triangular_mask_kernel<true>(self, diagonal);
}
Tensor triu_cpu(const Tensor& self, int64_t diagonal) {
    return triangular_mask_kernel<false>(self, diagonal);
}

// ---------------------------------------------------------------------------
// cumsum / cumprod / logcumsumexp
//
// Each operation scans independent outer*inner slices sequentially along
// `dim`, using an operation-specific accumulator type.
// ---------------------------------------------------------------------------

template <typename ctype, typename acc_t, typename Op>
inline void cum_base(ctype* d, const ctype* s, int64_t d_size, int64_t outer,
                     int64_t inner, ctype init_val, Op op) {
    // Independent outer*inner slices, each scanned sequentially along `dim`.
    // The grain is expressed in slices scaled by per-slice work so short
    // slices still fill the thread pool.
    const int64_t slice_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    if (inner == 1) {
        parallel_for(0, outer, slice_grain, [&](int64_t b, int64_t e) {
            for (int64_t o = b; o < e; ++o) {
                const ctype* sp = s + o * d_size;
                ctype* dp = d + o * d_size;
                acc_t acc = static_cast<acc_t>(init_val);
                for (int64_t j = 0; j < d_size; ++j) {
                    acc = op(acc, static_cast<acc_t>(sp[j]));
                    dp[j] = static_cast<ctype>(acc);
                }
            }
        });
        return;
    }
    parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) {
        for (int64_t si = b; si < e; ++si) {
            int64_t o = si / inner, in2 = si % inner;
            acc_t acc = static_cast<acc_t>(init_val);
            const ctype* sp = s + o * d_size * inner + in2;
            ctype* dp = d + o * d_size * inner + in2;
            for (int64_t j = 0; j < d_size; ++j) {
                acc = op(acc, static_cast<acc_t>(sp[j * inner]));
                dp[j * inner] = static_cast<ctype>(acc);
            }
        }
    });
}

template <typename ComplexT, bool Product>
Tensor complex_scan_cpu(const Tensor& src, int64_t dim) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(src.shape()),
        src.dtype(), src.device());
    const int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
    if constexpr (Product) {
        cum_base<ComplexT, ComplexT>(
            result.data_ptr<ComplexT>(), src.data_ptr<ComplexT>(),
            d_size, outer, inner, ComplexT(1, 0),
            [](ComplexT a, ComplexT x) { return a * x; });
    } else {
        cum_base<ComplexT, ComplexT>(
            result.data_ptr<ComplexT>(), src.data_ptr<ComplexT>(),
            d_size, outer, inner, ComplexT(0, 0),
            [](ComplexT a, ComplexT x) { return a + x; });
    }
    return result;
}

template <typename T>
inline tensorplay::complex<T> logcumsumexp_complex_pair(
        const tensorplay::complex<T>& x, const tensorplay::complex<T>& y) {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    if (std::isnan(x.real()) || std::isnan(x.imag()) ||
        std::isnan(y.real()) || std::isnan(y.imag())) {
        return {nan, nan};
    }
    const tensorplay::complex<T> min = x.real() < y.real() ? x : y;
    const tensorplay::complex<T> max = x.real() >= y.real() ? x : y;
    const T min_real = min.real();
    const T max_real = max.real();
    if (!std::isfinite(min_real) && min_real == max_real) {
        if (min_real < 0) return min;
        return tensorplay::log(tensorplay::exp(min) + tensorplay::exp(max));
    }
    return cx_log1p(tensorplay::exp(min - max)) + max;
}

template <typename ComplexT>
Tensor complex_logcumsumexp_cpu(const Tensor& src, int64_t dim) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(src.shape()),
        src.dtype(), src.device());
    const int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
    using value_t = typename ComplexT::value_type;
    const ComplexT init(-std::numeric_limits<value_t>::infinity(), value_t(0));
    const int64_t slice_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
    parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) {
        for (int64_t si = b; si < e; ++si) {
            const int64_t o = si / inner;
            const int64_t in2 = si % inner;
            const ComplexT* sp = src.data_ptr<ComplexT>() + o * d_size * inner + in2;
            ComplexT* dp = result.data_ptr<ComplexT>() + o * d_size * inner + in2;
            ComplexT acc = init;
            for (int64_t j = 0; j < d_size; ++j) {
                acc = logcumsumexp_complex_pair(acc, sp[j * inner]);
                dp[j * inner] = acc;
            }
        }
    });
    return result;
}

Tensor cumsum_cpu(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    DType out_dtype = dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                                         : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
    if (nd == 0) {
        Tensor result = Tensor::empty({}, out_dtype, src.device());
        result.copy_(src);
        return result;
    }
    if (isComplexType(out_dtype)) {
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return complex_scan_cpu<tensorplay::complex<double>, false>(compute_src, dim)
                .to(out_dtype);
        }
        return complex_scan_cpu<tensorplay::complex<float>, false>(compute_src, dim)
            .to(out_dtype);
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()), out_dtype, src.device());
    int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);

#define TP_CUMSUM_FLOAT(ctype, acc_t, name) \
    case DType::name: \
        cum_base<ctype, acc_t>(result.data_ptr<ctype>(), src.data_ptr<ctype>(), d_size, outer, inner, \
                               static_cast<ctype>(0), [](acc_t a, acc_t x) { return a + x; }); \
        break;
#define TP_CUMSUM_INT(ctype, name) \
    case DType::name: \
        cum_base<ctype, ctype>(result.data_ptr<ctype>(), src.data_ptr<ctype>(), d_size, outer, inner, \
                               static_cast<ctype>(0), [](ctype a, ctype x) { return static_cast<ctype>(a + x); }); \
        break;
    switch (out_dtype) {
        TP_CUMSUM_INT(uint8_t, UInt8)
        TP_CUMSUM_INT(int8_t, Int8)
        TP_CUMSUM_INT(int16_t, Int16)
        TP_CUMSUM_INT(int32_t, Int32)
        TP_CUMSUM_INT(int64_t, Int64)
        TP_CUMSUM_INT(uint16_t, UInt16)
        TP_CUMSUM_INT(uint32_t, UInt32)
        TP_CUMSUM_INT(uint64_t, UInt64)
        TP_CUMSUM_INT(bool, Bool)
        TP_CUMSUM_FLOAT(float, double, Float32)
        TP_CUMSUM_FLOAT(double, double, Float64)
        TP_CUMSUM_FLOAT(Half, float, Float16)
        TP_CUMSUM_FLOAT(BFloat16, float, BFloat16)
        default: TP_THROW(TypeError, "cumsum: unsupported dtype");
    }
#undef TP_CUMSUM_FLOAT
#undef TP_CUMSUM_INT
    return result;
}

namespace {

template <typename index_t>
Tensor repeat_interleave_indices_cpu_impl(
        const Tensor& repeats, std::optional<int64_t> output_size) {
    Tensor rep = repeats.contiguous();
    const int64_t size = rep.numel();
    if (size == 0) {
        return Tensor::empty({0}, repeats.dtype(), repeats.device());
    }

    const index_t* repeat_ptr = rep.data_ptr<index_t>();
    if (!output_size.has_value()) {
        for (int64_t i = 0; i < size; ++i) {
            if (repeat_ptr[i] < 0) {
                TP_THROW(RuntimeError, "repeats can not be negative");
            }
        }
    }

    Tensor cumsum = cumsum_cpu(rep, 0, DType::Int64);
    const int64_t* cumsum_ptr = cumsum.data_ptr<int64_t>();
    const int64_t required_size = cumsum_ptr[size - 1];
    const int64_t result_size = output_size.value_or(required_size);
    if (result_size != required_size) {
        TP_THROW(RuntimeError, "allocated size does not match required size");
    }

    Tensor result = Tensor::empty(
        {result_size}, repeats.dtype(), repeats.device());
    index_t* result_ptr = result.data_ptr<index_t>();
    parallel_for(0, size, 1, [&](int64_t begin, int64_t end_index) {
        for (int64_t i = begin; i < end_index; ++i) {
            const int64_t end = cumsum_ptr[i];
            const int64_t count = static_cast<int64_t>(repeat_ptr[i]);
            const int64_t start = end - count;
            if (count < 0 || start < 0 || end > result_size) {
                TP_THROW(RuntimeError, "repeats can not be negative");
            }
            for (int64_t j = start; j < end; ++j) {
                result_ptr[j] = static_cast<index_t>(i);
            }
        }
    });
    return result;
}

} // anonymous namespace

Tensor repeat_interleave_indices_cpu(
        const Tensor& repeats, std::optional<int64_t> output_size) {
    if (repeats.dim() != 1) {
        TP_THROW(RuntimeError,
                 "repeat_interleave only accepts a 1D vector as repeats");
    }
    switch (repeats.dtype()) {
        case DType::UInt8:
            return repeat_interleave_indices_cpu_impl<uint8_t>(
                repeats, output_size);
        case DType::Int8:
            return repeat_interleave_indices_cpu_impl<int8_t>(
                repeats, output_size);
        case DType::Int16:
            return repeat_interleave_indices_cpu_impl<int16_t>(
                repeats, output_size);
        case DType::Int32:
            return repeat_interleave_indices_cpu_impl<int32_t>(
                repeats, output_size);
        case DType::Int64:
            return repeat_interleave_indices_cpu_impl<int64_t>(
                repeats, output_size);
        default:
            TP_THROW(RuntimeError,
                     "repeats must have an integer index dtype");
    }
}

Tensor cumprod_cpu(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    DType out_dtype = dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                                         : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
    if (nd == 0) {
        Tensor result = Tensor::empty({}, out_dtype, src.device());
        result.copy_(src);
        return result;
    }
    if (isComplexType(out_dtype)) {
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return complex_scan_cpu<tensorplay::complex<double>, true>(compute_src, dim)
                .to(out_dtype);
        }
        return complex_scan_cpu<tensorplay::complex<float>, true>(compute_src, dim)
            .to(out_dtype);
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()), out_dtype, src.device());
    int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
#define TP_CUMPROD_FLOAT(ctype, acc_t, name) \
    case DType::name: \
        cum_base<ctype, acc_t>(result.data_ptr<ctype>(), src.data_ptr<ctype>(), d_size, outer, inner, \
                               static_cast<ctype>(1), [](acc_t a, acc_t x) { return a * x; }); \
        break;
#define TP_CUMPROD_INT(ctype, name) \
    case DType::name: \
        cum_base<ctype, ctype>(result.data_ptr<ctype>(), src.data_ptr<ctype>(), d_size, outer, inner, \
                               static_cast<ctype>(1), [](ctype a, ctype x) { return static_cast<ctype>(a * x); }); \
        break;
    switch (out_dtype) {
        TP_CUMPROD_INT(uint8_t, UInt8)
        TP_CUMPROD_INT(int8_t, Int8)
        TP_CUMPROD_INT(int16_t, Int16)
        TP_CUMPROD_INT(int32_t, Int32)
        TP_CUMPROD_INT(int64_t, Int64)
        TP_CUMPROD_INT(uint16_t, UInt16)
        TP_CUMPROD_INT(uint32_t, UInt32)
        TP_CUMPROD_INT(uint64_t, UInt64)
        TP_CUMPROD_INT(bool, Bool)
        TP_CUMPROD_FLOAT(float, double, Float32)
        TP_CUMPROD_FLOAT(double, double, Float64)
        TP_CUMPROD_FLOAT(Half, float, Float16)
        TP_CUMPROD_FLOAT(BFloat16, float, BFloat16)
        default: TP_THROW(TypeError, "cumprod: unsupported dtype");
    }
#undef TP_CUMPROD_FLOAT
#undef TP_CUMPROD_INT
    return result;
}

Tensor logcumsumexp_cpu(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    // Stable recurrence: m = max(x, acc), then
    // result = m + log1p(exp(-|x - acc|)).
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    DType out_dtype = dtype.value_or(self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
    if (isComplexType(out_dtype)) {
        if (nd == 0) {
            Tensor result = Tensor::empty({}, out_dtype, src.device());
            result.copy_(src);
            return result;
        }
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return complex_logcumsumexp_cpu<tensorplay::complex<double>>(compute_src, dim)
                .to(out_dtype);
        }
        return complex_logcumsumexp_cpu<tensorplay::complex<float>>(compute_src, dim)
            .to(out_dtype);
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()), out_dtype, src.device());
    if (nd == 0) {
        switch (out_dtype) {
            case DType::Float32:
            case DType::Float64:
            case DType::Float16:
            case DType::BFloat16:
                result.copy_(src);
                return result;
            default:
                TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
        }
    }
    int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
#define TP_LCSE_CASE(ctype, acc_t, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        const ctype* s = src.data_ptr<ctype>(); \
        constexpr acc_t neg_inf = -std::numeric_limits<acc_t>::infinity(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                acc_t acc = neg_inf; \
                const ctype* sp = s + o * d_size * inner + in2; \
                ctype* dp = d + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) { \
                    acc_t x = static_cast<acc_t>(sp[j * inner]); \
                    acc_t m = std::max(x, acc); \
                    acc = (m == neg_inf) ? m : (m + std::log1p(std::exp(-std::fabs(x - acc)))); \
                    dp[j * inner] = static_cast<ctype>(acc); \
                } \
            } \
        }); \
        break; \
    }
    switch (out_dtype) {
        TP_LCSE_CASE(float, float, Float32)
        TP_LCSE_CASE(double, double, Float64)
        TP_LCSE_CASE(Half, float, Float16)
        TP_LCSE_CASE(BFloat16, float, BFloat16)
        default: TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
    }
#undef TP_LCSE_CASE
    return result;
}

// ---------------------------------------------------------------------------
// gather
//
// The backward result is initialized to zero and accumulates each gradient
// into the indexed locations.
// ---------------------------------------------------------------------------

Tensor gather_cpu(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as input tensor");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i != dim && index.size(i) > self.size(i)) {
            TP_THROW(IndexError, "Size does not match at dimension ", i,
                     " (input: ", self.size(i), ", index: ", index.size(i), ")");
        }
    }
    Tensor idx_c = normalize_index_cpu(index, self.size(dim), "gather");
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(idx_c.shape()), self.dtype(), self.device());
    // Decomposition runs over the RESULT (=index) shape; the source read
    // for i != dim, so the two extents may differ).
    int64_t idx_outer = 1;
    for (int64_t i = 0; i < dim; ++i) idx_outer *= idx_c.size(i);
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t n = result.numel();
    int64_t self_dim_size = self.size(dim);

#define TP_GATHER_CASE(ctype, name) \
    case DType::name: { \
        const ctype* s = self_c.data_ptr<ctype>(); \
        const int64_t* ip = idx_c.data_ptr<int64_t>(); \
        ctype* d = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t flat = begin; flat < end; ++flat) { \
                int64_t rem = flat; \
                int64_t outer_off = rem / (idx_dim_size * idx_inner); rem -= outer_off * idx_dim_size * idx_inner; \
                int64_t t = rem % idx_inner; \
                int64_t idx = ip[flat]; \
                d[flat] = s[(outer_off * self_dim_size + idx) * self_inner + t]; \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_GATHER_CASE)
        default: TP_THROW(TypeError, "gather: unsupported dtype");
    }
#undef TP_GATHER_CASE
    return result;
}

// ---------------------------------------------------------------------------
// scatter / scatter_add
//
// The CPU implementation updates one indexed slice at a time. The optional
// accumulation mode is intentionally serialized because duplicate indices
// must have deterministic update order.
// ---------------------------------------------------------------------------

enum class ScatterMode { Assign, Add };

Tensor scatter_base_cpu(const Tensor& self, int64_t dim, const Tensor& index,
                        const Tensor& src, ScatterMode mode) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = normalize_index_cpu(index, self.size(dim), "scatter");
    std::vector<int64_t> idx_shape(static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src.numel() == 1) {
        src_b = src.expand(idx_shape).contiguous();
    } else {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError, "scatter: src shape must broadcast to the index shape");
        }
        src_b = src.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) {
        src_b = src_b.to(self.dtype());
    }
    Tensor result = detail::contiguous_clone(self);
    // Layout of the index tensor: [outer][dim][idx_inner]; the destination
    // row stride inside `self` is its own inner extent (which may be larger
    // than the index's when the index is broadcast-thin along trailing dims).
    int64_t idx_outer = 1;
    for (int64_t i = 0; i < dim; ++i) idx_outer *= idx_c.size(i);
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t total_idx = idx_c.numel();
    int64_t self_dim_size = self.size(dim);
    // One destination element is produced for every index element:
    // out[oo][idx_value][t] <- src[oo][j][t].
#define TP_SCATTER_CASE(ctype, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        const int64_t* ip = idx_c.data_ptr<int64_t>(); \
        const ctype* vp = src_b.data_ptr<ctype>(); \
        parallel_for(0, total_idx, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t flat = begin; flat < end; ++flat) { \
                int64_t rem = flat; \
                int64_t outer_off = rem / (idx_dim_size * idx_inner); \
                rem -= outer_off * idx_dim_size * idx_inner; \
                int64_t t = rem % idx_inner; \
                int64_t idx = ip[flat]; \
                int64_t dst = (outer_off * self_dim_size + idx) * self_inner + t; \
                ctype v = vp[flat]; \
                if (mode == ScatterMode::Assign) { \
                    d[dst] = v; \
                } else { \
                    d[dst] += v; \
                } \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SCATTER_CASE)
        default: TP_THROW(TypeError, "scatter: unsupported dtype");
    }
#undef TP_SCATTER_CASE
    return result;
}

Tensor scatter_add_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_cpu(self, dim, index, src, ScatterMode::Add);
}

Tensor scatter_src_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_cpu(self, dim, index, src, ScatterMode::Assign);
}

Tensor scatter_value_cpu(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_cpu(self, dim, index, full, ScatterMode::Assign);
}

// The in-place variant writes directly into self instead of using a clone.
// It remains a sibling of the out-of-place path so their dispatch behavior is
// independent.
static Tensor& scatter_base_inplace_cpu(Tensor& self, int64_t dim, const Tensor& index,
                                        const Tensor& src, ScatterMode mode) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = normalize_index_cpu(index, self.size(dim), "scatter_");
    std::vector<int64_t> idx_shape(static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src.numel() == 1) {
        src_b = src.expand(idx_shape).contiguous();
    } else {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError, "scatter_: src shape must broadcast to the index shape");
        }
        src_b = src.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) {
        src_b = src_b.to(self.dtype());
    }
    if (!self.is_contiguous()) {
        // The raw-pointer loop below needs a contiguous destination; scatter
        // produces for a strided self).
        Tensor out = scatter_base_cpu(self, dim, index, src, mode);
        self.copy_(out);
        return self;
    }
    Tensor& result = self;
    int64_t idx_outer = 1;
    for (int64_t i = 0; i < dim; ++i) idx_outer *= idx_c.size(i);
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t total_idx = idx_c.numel();
    int64_t self_dim_size = self.size(dim);

#define TP_SCATTER_INPLACE_CASE(ctype, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        const int64_t* ip = idx_c.data_ptr<int64_t>(); \
        const ctype* vp = src_b.data_ptr<ctype>(); \
        parallel_for(0, total_idx, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t flat = begin; flat < end; ++flat) { \
                int64_t rem = flat; \
                int64_t outer_off = rem / (idx_dim_size * idx_inner); \
                rem -= outer_off * idx_dim_size * idx_inner; \
                int64_t t = rem % idx_inner; \
                int64_t idx = ip[flat]; \
                int64_t dst = (outer_off * self_dim_size + idx) * inner + t; \
                ctype v = vp[flat]; \
                if (mode == ScatterMode::Assign) { \
                    d[dst] = v; \
                } else { \
                    d[dst] += v; \
                } \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SCATTER_INPLACE_CASE)
        default: TP_THROW(TypeError, "scatter_: unsupported dtype");
    }
#undef TP_SCATTER_INPLACE_CASE
    return result;
}

Tensor& scatter_inplace_src_cpu(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cpu(self, dim, index, src, ScatterMode::Assign);
}

Tensor& scatter_inplace_value_cpu(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_inplace_cpu(self, dim, index, full, ScatterMode::Assign);
}

Tensor& scatter_add_inplace_cpu(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cpu(self, dim, index, src, ScatterMode::Add);
}

// ---------------------------------------------------------------------------
// index_select
//
// index_select_cpu_: memcpy per selected slice along dim.
// ---------------------------------------------------------------------------

Tensor index_select_cpu(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != 1) TP_THROW(IndexError, "index_select(): index should be a vector");
    Tensor idx = normalize_index_cpu(index, self.size(dim), "index_select");
    int64_t n_idx = idx.numel();
    int64_t row = self.size(dim);
    const int64_t* ip = idx.data_ptr<int64_t>();
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape(static_cast<std::vector<int64_t>>(self.shape()));
    out_shape[dim] = n_idx;
    Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
    Tensor self_c = self.contiguous();
#define TP_ISEL_CASE(ctype, name) \
    case DType::name: { \
        const ctype* s = self_c.data_ptr<ctype>(); \
        ctype* d = result.data_ptr<ctype>(); \
        if (inner == 1) { \
            const int64_t batch_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(n_idx, 1)); \
            parallel_for(0, outer, batch_grain, [&](int64_t ob, int64_t oe) { \
                for (int64_t o = ob; o < oe; ++o) { \
                    const ctype* src_row = s + o * row; \
                    ctype* dst_row = d + o * n_idx; \
                    for (int64_t k = 0; k < n_idx; ++k) { \
                        dst_row[k] = src_row[ip[k]]; \
                    } \
                } \
            }); \
        } else { \
            const int64_t slice_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(inner, 1)); \
            parallel_for(0, outer * n_idx, slice_grain, [&](int64_t b, int64_t e) { \
                for (int64_t t = b; t < e; ++t) { \
                    int64_t o = t / n_idx, k = t % n_idx; \
                    std::memcpy(d + static_cast<int64_t>(t) * inner, s + (o * row + ip[k]) * inner, inner * sizeof(ctype)); \
                } \
            }); \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ISEL_CASE)
        default: TP_THROW(TypeError, "index_select: unsupported dtype");
    }
#undef TP_ISEL_CASE
    return result;
}

// ---------------------------------------------------------------------------
// index_add
//
// Adds each source slice into self along the selected dimension.
// ---------------------------------------------------------------------------

Tensor index_add_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = normalize_index_cpu(index, self.size(dim), "index_add");
    Tensor result = detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    const int64_t* ip = idx.data_ptr<int64_t>();
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    Tensor source_c = source.contiguous();
    bool source_is_scalar = source_c.dim() == 0;
    if (!source_is_scalar && source_c.dim() != nd) {
        TP_THROW(RuntimeError, "index_add: source must have same number of dims as input");
    }
    if (!source_is_scalar && source_c.size(dim) != n_idx) {
        TP_THROW(RuntimeError, "index_add: source size along dim must equal index length");
    }
#define TP_IADD_CASE(ctype, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        const ctype* sp = source_c.data_ptr<ctype>(); \
        parallel_for(0, outer * n_idx, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t o = t / n_idx, k = t % n_idx; \
                int64_t iv = ip[k]; \
                const ctype* sv = source_is_scalar ? sp : sp + (o * n_idx + k) * inner; \
                ctype* dv = d + (o * row + iv) * inner; \
                for (int64_t c = 0; c < inner; ++c) dv[c] += sv[c]; \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IADD_CASE)
        default: TP_THROW(TypeError, "index_add: unsupported dtype");
    }
#undef TP_IADD_CASE
    return result;
}

// ---------------------------------------------------------------------------
// index_copy / index_fill
//
// Each selected slice is written into the output.
// ---------------------------------------------------------------------------

Tensor index_copy_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = normalize_index_cpu(index, self.size(dim), "index_copy");
    Tensor result = detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    const int64_t* ip = idx.data_ptr<int64_t>();
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    Tensor source_c = source.contiguous();
#define TP_ICOPY_CASE(ctype, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        const ctype* sp = source_c.data_ptr<ctype>(); \
        for (int64_t o = 0; o < outer; ++o) { \
            for (int64_t k = 0; k < n_idx; ++k) { \
                int64_t iv = ip[k]; \
                std::memcpy(d + (o * row + iv) * inner, sp + (o * n_idx + k) * inner, inner * sizeof(ctype)); \
            } \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ICOPY_CASE)
        default: TP_THROW(TypeError, "index_copy: unsupported dtype");
    }
#undef TP_ICOPY_CASE
    return result;
}

Tensor index_fill_scalar_cpu(const Tensor& self, int64_t dim, const Tensor& index, Scalar value);

Tensor index_fill_tensor_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    Scalar v = value.item();
    return index_fill_scalar_cpu(self, dim, index, v);
}

Tensor index_fill_scalar_cpu(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = normalize_index_cpu(index, self.size(dim), "index_fill");
    Tensor result = detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    const int64_t* ip = idx.data_ptr<int64_t>();
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
#define TP_IFILL_CASE(ctype, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        ctype v = value.to<ctype>(); \
        for (int64_t o = 0; o < outer; ++o) { \
            for (int64_t k = 0; k < n_idx; ++k) { \
                int64_t iv = ip[k]; \
                ctype* dp = d + (o * row + iv) * inner; \
                for (int64_t c = 0; c < inner; ++c) dp[c] = v; \
            } \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IFILL_CASE)
        default: TP_THROW(TypeError, "index_fill: unsupported dtype");
    }
#undef TP_IFILL_CASE
    return result;
}

Tensor& index_fill_scalar__cpu(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    // slice loop; tp composes it as fill-then-copy_ like the other
    // in-place index ops.
    self.copy_(index_fill_scalar_cpu(self, dim, index, value));
    return self;
}

Tensor& index_fill_tensor__cpu(Tensor& self, int64_t dim, const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return index_fill_scalar__cpu(self, dim, index, value.item());
}

// ---------------------------------------------------------------------------
// index_put / index_put_
//
// Native advanced indexing preserves destination strides and broadcasts
// values over the indexed result shape.
// ---------------------------------------------------------------------------


Tensor index_put_cpu(const Tensor& self, const std::vector<Tensor>& indices,
                     const Tensor& values, bool accumulate) {
    extern Tensor& index_put_native_cpu(Tensor&, const std::vector<Tensor>&,
                                         const Tensor&, bool);
    Tensor result = self.clone();
    return index_put_native_cpu(result, indices, values, accumulate);
}

Tensor& index_put__cpu(Tensor& self, const std::vector<Tensor>& indices,
                       const Tensor& values, bool accumulate) {
    extern Tensor& index_put_native_cpu(Tensor&, const std::vector<Tensor>&,
                                         const Tensor&, bool);
    return index_put_native_cpu(self, indices, values, accumulate);
}

// ---------------------------------------------------------------------------
// nonzero
//
// first count matching elements, then fill an (n, dim) Long tensor with
// coordinates in row-major order.
// ---------------------------------------------------------------------------

template <typename scalar_t>
Tensor nonzero_cpu_impl(const Tensor& self) {
    Tensor input = self.contiguous();
    const int64_t nd = self.dim();
    const int64_t n = input.numel();
    if (n == 0) {
        return Tensor::empty({0, nd}, DType::Int64, self.device());
    }

    const int64_t thread_count = get_num_threads();
    const bool use_parallel = n > GRAIN_SIZE && thread_count > 1 &&
        !in_parallel_region();
    const int64_t chunk_size = use_parallel
        ? std::max<int64_t>(GRAIN_SIZE, (n + thread_count - 1) / thread_count)
        : n;
    const int64_t chunk_count = (n + chunk_size - 1) / chunk_size;
    std::vector<int64_t> chunk_offsets(static_cast<size_t>(chunk_count + 1), 0);
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t count = 0;
        for (int64_t i = begin; i < end; ++i) {
            count += (input_data[i] != scalar_t(0)) ? 1 : 0;
        }
        chunk_offsets[static_cast<size_t>(begin / chunk_size + 1)] = count;
    });
    for (int64_t i = 1; i <= chunk_count; ++i) {
        chunk_offsets[static_cast<size_t>(i)] +=
            chunk_offsets[static_cast<size_t>(i - 1)];
    }

    const int64_t count = chunk_offsets.back();
    Tensor result = Tensor::empty({count, nd}, DType::Int64, self.device());
    if (count == 0) return result;
    int64_t* result_data = result.data_ptr<int64_t>();
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(input.shape());
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t output_index =
            chunk_offsets[static_cast<size_t>(begin / chunk_size)];
        std::vector<int64_t> coordinates(static_cast<size_t>(nd), 0);
        int64_t remaining = begin;
        for (int64_t d = nd - 1; d >= 0; --d) {
            coordinates[static_cast<size_t>(d)] = remaining % sizes[static_cast<size_t>(d)];
            remaining /= sizes[static_cast<size_t>(d)];
        }
        for (int64_t i = begin; i < end; ++i) {
            if (input_data[i] != scalar_t(0)) {
                int64_t* row = result_data + output_index * nd;
                for (int64_t d = 0; d < nd; ++d) {
                    row[d] = coordinates[static_cast<size_t>(d)];
                }
                ++output_index;
            }
            for (int64_t d = nd - 1; d >= 0; --d) {
                if (++coordinates[static_cast<size_t>(d)] <
                    sizes[static_cast<size_t>(d)]) {
                    break;
                }
                coordinates[static_cast<size_t>(d)] = 0;
            }
        }
    });
    return result;
}

Tensor nonzero_cpu(const Tensor& self) {
#define TP_NZ_CASE(ctype, name) \
    case DType::name: return nonzero_cpu_impl<ctype>(self);
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_NZ_CASE)
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZ_CASE
}

// ---------------------------------------------------------------------------
// sort / argsort
//
// Sort along dim while carrying original positions.  NaN ordering follows
// the project convention: NaN sorts after every non-NaN value in
// ascending order (and before them in descending order); ties keep their
// original relative order.  The (value, index) lexicographic comparator is a
// strict weak ordering even with NaN lanes, so plain std::sort is both
// well-defined and deterministic.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
inline bool sort_is_nan(T value) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return std::isnan(value);
    } else if constexpr (std::is_same_v<T, Half> ||
                         std::is_same_v<T, BFloat16>) {
        return std::isnan(static_cast<float>(value));
    } else {
        return false;
    }
}

template <typename ctype>
struct SortPairLess {
    bool descending;
    explicit SortPairLess(bool desc) : descending(desc) {}
    // returns true when pair a must come before pair b
    bool operator()(const std::pair<ctype, int64_t>& a, const std::pair<ctype, int64_t>& b) const {
        if constexpr (std::is_floating_point_v<ctype> ||
                      std::is_same_v<ctype, Half> ||
                      std::is_same_v<ctype, BFloat16>) {
            const bool na = sort_is_nan(a.first), nb = sort_is_nan(b.first);
            if (na != nb) return descending ? na : nb;  // NaN sinks in ascending
            if (na) return a.second < b.second;         // stable among NaNs
        }
        if (a.first != b.first) return descending ? (b.first < a.first) : (a.first < b.first);
        return a.second < b.second;  // stable among equal values
    }
};

}  // namespace

std::tuple<Tensor, Tensor> sort_cpu(const Tensor& self, int64_t dim, bool descending) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "sort: expects at least 1 dimension");
    dim = wrap_dim(dim, nd);
    Tensor self_c = self.contiguous();
    int64_t d_size = self_c.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer, inner);
    Tensor values = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), self_c.dtype(), self_c.device());
    Tensor indices = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), DType::Int64, self_c.device());

#define TP_SORT_CASE(ctype, name) \
    case DType::name: { \
        const ctype* s = self_c.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        const SortPairLess<ctype> less(descending); \
        const int64_t slice_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1)); \
        parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) { \
            using pair_t = std::pair<ctype, int64_t>; \
            constexpr int64_t kMaxCachedElems = 1 << 12; \
            static thread_local std::vector<pair_t> cached_buf; \
            std::vector<pair_t> transient_buf; \
            pair_t* buf = nullptr; \
            if (d_size <= kMaxCachedElems) { \
                if (static_cast<int64_t>(cached_buf.size()) < d_size) cached_buf.resize(static_cast<size_t>(d_size)); \
                buf = cached_buf.data(); \
            } else { \
                transient_buf.resize(static_cast<size_t>(d_size)); \
                buf = transient_buf.data(); \
            } \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                const ctype* base = s + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) buf[j] = {base[j * inner], j}; \
                std::sort(buf, buf + d_size, less); \
                ctype* vbase = vp + o * d_size * inner + in2; \
                int64_t* ibase = ip + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) { vbase[j * inner] = buf[j].first; ibase[j * inner] = buf[j].second; } \
            } \
        }); \
        break; \
    }
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SORT_CASE)
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
#undef TP_SORT_CASE
    return {values, indices};
}

Tensor argsort_cpu(const Tensor& self, int64_t dim, bool descending) {
    // Indices-only variant of sort; values are materialized as well and the
    // values tensor is dropped.
    return std::get<1>(sort_cpu(self, dim, descending));
}

// ---------------------------------------------------------------------------
// searchsorted / bucketize
//
// Per query value the kernel binary-searches the innermost dimension of the
// boundaries and returns an insertion position: right=false yields the lower
// bound (first boundary >= v) and right=true the upper bound (first boundary
// > v).  The bound comparators are written as `!(bd >= v)` / `!(bd > v)` so a
// NaN query compares greater than every boundary entry and lands at the end
// of the searched range instead of folding to position 0.
//
// Boundaries may be 1-D (shared lookup table for every query) or N-D matching
// all leading dimensions of the input (one lookup table per row, shared along
// the innermost axis).  A sorter tensor carries the permutation that orders an
// unsorted boundary tensor; boundary element access then goes through
// `bd[sorter[mid] + row_offset]`.
// ---------------------------------------------------------------------------

namespace {

// Minimal size for the contiguous searchsorted kernel to run in parallel.
constexpr int64_t kSearchSortedGrainSize = 200;

// The contiguous hot loop.  `boundaries` must be contiguous with the same
// dtype as `input`; `sorter` is either undefined or a contiguous permutation
// with the same shape as `boundaries`.
template <typename input_t, typename output_t>
void searchsorted_cpu_contiguous(Tensor& result, const Tensor& input,
                                 const Tensor& boundaries, bool right,
                                 bool is_1d_boundaries,
                                 const Tensor& sorter) {
    const int64_t numel_in = input.numel();
    const bool is_scalar_input = input.dim() == 0 && numel_in == 1;
    // Innermost dimension size of the input and of the lookup tables.
    const int64_t idim_in = is_scalar_input ? 1 : input.size(-1);
    const int64_t idim_bd = boundaries.size(-1);

    const input_t* data_in = input.data_ptr<input_t>();
    const input_t* data_bd = boundaries.data_ptr<input_t>();
    const int64_t* data_sorter =
        sorter.defined() ? sorter.data_ptr<int64_t>() : nullptr;
    output_t* data_out = result.data_ptr<output_t>();

    parallel_for(0, numel_in, kSearchSortedGrainSize, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            // A 1-D boundary table is shared by every query; a row-wise table
            // starts at (query row / input innermost) * table innermost.
            const int64_t row_offset =
                is_1d_boundaries ? 0 : i / idim_in * idim_bd;
            int64_t start_bd = row_offset;
            int64_t end_bd = start_bd + idim_bd;
            const input_t val = data_in[i];
            if (!right) {
                // lower bound: first position with bd >= val
                while (start_bd < end_bd) {
                    const int64_t mid = start_bd + ((end_bd - start_bd) >> 1);
                    const input_t mid_value = data_sorter
                        ? data_bd[data_sorter[mid] + row_offset]
                        : data_bd[mid];
                    if (!(mid_value >= val)) start_bd = mid + 1; else end_bd = mid;
                }
            } else {
                // upper bound: first position with bd > val
                while (start_bd < end_bd) {
                    const int64_t mid = start_bd + ((end_bd - start_bd) >> 1);
                    const input_t mid_value = data_sorter
                        ? data_bd[data_sorter[mid] + row_offset]
                        : data_bd[mid];
                    if (!(mid_value > val)) start_bd = mid + 1; else end_bd = mid;
                }
            }
            data_out[i] = static_cast<output_t>(start_bd - row_offset);
        }
    });
}

void searchsorted_dispatch(Tensor& result, const Tensor& input,
                           const Tensor& boundaries, bool right,
                           const Tensor& sorter) {
#define TP_SS_RUN(input_t, output_t)                                       \
    searchsorted_cpu_contiguous<input_t, output_t>(                        \
        result, input, boundaries, right, boundaries.dim() == 1, sorter)

    if (result.dtype() == DType::Int64) {
        switch (input.dtype()) {
            case DType::Float64: TP_SS_RUN(double, int64_t); return;
            case DType::Float32: TP_SS_RUN(float, int64_t); return;
            case DType::Float16: TP_SS_RUN(Half, int64_t); return;
            case DType::BFloat16: TP_SS_RUN(BFloat16, int64_t); return;
            case DType::Int64: TP_SS_RUN(int64_t, int64_t); return;
            case DType::Int32: TP_SS_RUN(int32_t, int64_t); return;
            case DType::Int16: TP_SS_RUN(int16_t, int64_t); return;
            case DType::Int8: TP_SS_RUN(int8_t, int64_t); return;
            case DType::UInt8: TP_SS_RUN(uint8_t, int64_t); return;
            case DType::UInt16: TP_SS_RUN(uint16_t, int64_t); return;
            case DType::UInt32: TP_SS_RUN(uint32_t, int64_t); return;
            case DType::UInt64: TP_SS_RUN(uint64_t, int64_t); return;
            case DType::Bool: TP_SS_RUN(bool, int64_t); return;
            default: break;
        }
    } else {
        switch (input.dtype()) {
            case DType::Float64: TP_SS_RUN(double, int32_t); return;
            case DType::Float32: TP_SS_RUN(float, int32_t); return;
            case DType::Float16: TP_SS_RUN(Half, int32_t); return;
            case DType::BFloat16: TP_SS_RUN(BFloat16, int32_t); return;
            case DType::Int64: TP_SS_RUN(int64_t, int32_t); return;
            case DType::Int32: TP_SS_RUN(int32_t, int32_t); return;
            case DType::Int16: TP_SS_RUN(int16_t, int32_t); return;
            case DType::Int8: TP_SS_RUN(int8_t, int32_t); return;
            case DType::UInt8: TP_SS_RUN(uint8_t, int32_t); return;
            case DType::UInt16: TP_SS_RUN(uint16_t, int32_t); return;
            case DType::UInt32: TP_SS_RUN(uint32_t, int32_t); return;
            case DType::UInt64: TP_SS_RUN(uint64_t, int32_t); return;
            case DType::Bool: TP_SS_RUN(bool, int32_t); return;
            default: break;
        }
    }
#undef TP_SS_RUN
    TP_THROW(TypeError, "searchsorted(): unsupported dtype ",
             toString(input.dtype()));
}

Tensor& searchsorted_out_cpu_impl(const Tensor& sorted_sequence,
                                  const Tensor& self, bool out_int32,
                                  bool right,
                                  const std::optional<std::string>& side_opt,
                                  const Tensor& sorter_opt, Tensor& result) {
    const Tensor& sorter = sorter_opt;
    bucketization::pre_check(sorted_sequence, self, result, out_int32, right,
                             side_opt, sorter);
    result.resize_(static_cast<std::vector<int64_t>>(self.shape()));

    // Two inputs control the bound direction; pre_check rejects conflicts.
    const bool is_right = side_opt.has_value() ? *side_opt == "right" : right;

    if (self.numel() == 0) {
        return result;
    }

    // Non-contiguous outputs are written through a contiguous copy and copied
    // back afterwards so the strided result keeps its layout.
    Tensor out = result;
    const bool out_is_contiguous = result.is_contiguous();
    if (!out_is_contiguous) out = result.contiguous();

    Tensor trimmed_input, trimmed_boundaries;
    Tensor sorter_work;
    if (sorter.defined()) {
        sorter_work = sorter.contiguous();
    }
    const Tensor& seq = sorted_sequence;
    bucketization::maybe_trim_input_tensors(trimmed_input, trimmed_boundaries,
                                            self, seq);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries =
        trimmed_boundaries.defined() ? trimmed_boundaries : seq;
    searchsorted_dispatch(out, final_input, final_boundaries, is_right,
                          sorter_work);

    if (!out_is_contiguous) result.copy_(out);
    return result;
}

Tensor empty_searchsorted_output(const Tensor& like, bool out_int32) {
    return Tensor::empty({}, out_int32 ? DType::Int32 : DType::Int64,
                         like.device());
}

} // anonymous namespace

Tensor& searchsorted_out_cpu(const Tensor& sorted_sequence, const Tensor& self,
                             bool out_int32, bool right,
                             const std::optional<std::string>& side_opt,
                             const std::optional<Tensor>& sorter_opt,
                             Tensor& result) {
    return searchsorted_out_cpu_impl(
        sorted_sequence, self, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
}

Tensor searchsorted_cpu(const Tensor& sorted_sequence, const Tensor& self,
                        bool out_int32, bool right,
                        const std::optional<std::string>& side_opt,
                        const std::optional<Tensor>& sorter_opt) {
    Tensor result = empty_searchsorted_output(self, out_int32);
    searchsorted_out_cpu_impl(
        sorted_sequence, self, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
    return result;
}

Tensor& searchsorted_scalar_out_cpu(const Tensor& sorted_sequence,
                                    const Scalar& self, bool out_int32,
                                    bool right,
                                    const std::optional<std::string>& side_opt,
                                    const std::optional<Tensor>& sorter_opt,
                                    Tensor& result) {
    Tensor scalar_query =
        bucketization::scalar_tensor(self, sorted_sequence.device());
    return searchsorted_out_cpu_impl(
        sorted_sequence, scalar_query, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
}

Tensor searchsorted_scalar_cpu(const Tensor& sorted_sequence, const Scalar& self,
                               bool out_int32, bool right,
                               const std::optional<std::string>& side_opt,
                               const std::optional<Tensor>& sorter_opt) {
    Tensor result = empty_searchsorted_output(sorted_sequence, out_int32);
    searchsorted_scalar_out_cpu(sorted_sequence, self, out_int32, right,
                                side_opt, sorter_opt, result);
    return result;
}

Tensor& bucketize_out_cpu(const Tensor& self, const Tensor& boundaries,
                          bool out_int32, bool right, Tensor& result) {
    TP_CHECK(boundaries.dim() == 1,
             "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
             boundaries.dim(), ")");
    return searchsorted_out_cpu_impl(boundaries, self, out_int32, right,
                                     std::nullopt, Tensor(), result);
}

Tensor bucketize_cpu(const Tensor& self, const Tensor& boundaries,
                     bool out_int32, bool right) {
    Tensor result = empty_searchsorted_output(self, out_int32);
    bucketize_out_cpu(self, boundaries, out_int32, right, result);
    return result;
}

Tensor& bucketize_scalar_out_cpu(const Scalar& self, const Tensor& boundaries,
                                 bool out_int32, bool right, Tensor& result) {
    Tensor scalar_query =
        bucketization::scalar_tensor(self, boundaries.device());
    return bucketize_out_cpu(scalar_query, boundaries, out_int32, right,
                             result);
}

Tensor bucketize_scalar_cpu(const Scalar& self, const Tensor& boundaries,
                            bool out_int32, bool right) {
    Tensor result = empty_searchsorted_output(boundaries, out_int32);
    bucketize_scalar_out_cpu(self, boundaries, out_int32, right, result);
    return result;
}

// ---------------------------------------------------------------------------
// bincount
//
// Inputs are one-dimensional non-negative integers. The number of bins is the
// larger of minlength and one more than the largest input value. Weighted
// accumulation keeps float32 weights in float32 and widens other weights.
// ---------------------------------------------------------------------------

template <typename input_t>
Tensor bincount_cpu_impl(const Tensor& self, const Tensor& weights,
                         int64_t minlength) {
    if (minlength < 0) {
        TP_THROW(RuntimeError, "minlength should be >= 0");
    }
    if (self.dim() == 1 && self.numel() == 0) {
        return Tensor::zeros({minlength}, DType::Int64, self.device());
    }
    if (self.dim() != 1) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    Tensor inp = self.contiguous();
    const input_t* ip = inp.data_ptr<input_t>();
    input_t min_v = std::numeric_limits<input_t>::max();
    input_t max_v = std::numeric_limits<input_t>::lowest();
    for (int64_t i = 0; i < inp.numel(); ++i) {
        if (ip[i] < min_v) min_v = ip[i];
        if (ip[i] > max_v) max_v = ip[i];
    }
    if constexpr (std::is_signed_v<input_t>) {
        if (min_v < 0) {
            TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
        }
    }
    const uint64_t max_unsigned = static_cast<uint64_t>(max_v);
    if (max_unsigned >= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        TP_THROW(RuntimeError, "maximum value of input overflowed");
    }
    const int64_t self_size = inp.size(0);
    const bool has_weights = weights.defined();
    if (has_weights && (weights.dim() != 1 || weights.size(0) != self_size)) {
        TP_THROW(RuntimeError, "weights should be 1-d and have the same length as input");
    }
    const int64_t nbins = std::max(static_cast<int64_t>(max_v) + 1, minlength);
    if (has_weights) {
        if (weights.dtype() == DType::Float32) {
            Tensor w = weights.contiguous();
            Tensor result = Tensor::zeros({nbins}, DType::Float32, self.device());
            const float* wp = w.data_ptr<float>();
            float* rp = result.data_ptr<float>();
            for (int64_t i = 0; i < self_size; ++i) {
                rp[static_cast<int64_t>(ip[i])] += wp[i];
            }
            return result;
        }
        Tensor w = weights.to(DType::Float64).contiguous();
        Tensor result = Tensor::zeros({nbins}, DType::Float64, self.device());
        const double* wp = w.data_ptr<double>();
        double* rp = result.data_ptr<double>();
        for (int64_t i = 0; i < self_size; ++i) {
            rp[static_cast<int64_t>(ip[i])] += wp[i];
        }
        return result;
    }
    Tensor result = Tensor::zeros({nbins}, DType::Int64, self.device());
    int64_t* rp = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < self_size; ++i) {
        rp[static_cast<int64_t>(ip[i])] += 1;
    }
    return result;
}

Tensor bincount_cpu(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
    const Tensor weights = weights_opt.value_or(Tensor());
#define TP_BINCOUNT_CASE(ctype, name) \
    case DType::name: \
        return bincount_cpu_impl<ctype>(self, weights, minlength);
    switch (self.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BINCOUNT_CASE)
        default:
            TP_THROW(RuntimeError,
                     "bincount only supports 1-d non-negative integral inputs.");
    }
#undef TP_BINCOUNT_CASE
}

// ---------------------------------------------------------------------------
// take
//
// flatten self, index_select along dim 0, reshape to index.sizes().
// ---------------------------------------------------------------------------

Tensor take_cpu(const Tensor& self, const Tensor& index) {
    Tensor flat = self.reshape({self.numel()});
    return index_select_cpu(flat, 0, index.reshape({index.numel()}))
        .reshape(static_cast<std::vector<int64_t>>(index.shape()));
}

// ---------------------------------------------------------------------------
// masked_scatter
//
// launch_masked_scatter_kernel (and its CPU counterpart) consume `source`
// sequentially in mask order.
// ---------------------------------------------------------------------------

Tensor masked_scatter_cpu(const Tensor& self, const Tensor& mask, const Tensor& source) {
    if (source.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "masked_scatter: self and source must have the same dtype");
    }
    if (mask.device() != self.device() || source.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "masked_scatter: self, mask, and source must be on the same device");
    }
    Tensor m_full = mask.to(DType::Bool).expand(static_cast<std::vector<int64_t>>(self.shape())).contiguous();
    Tensor src = source.contiguous();
    Tensor result = detail::contiguous_clone(self);
    int64_t n = result.numel();
    const bool* mp = m_full.data_ptr<bool>();
    int64_t src_i = 0;
    int64_t src_n = src.numel();
#define TP_MS_CASE(ctype, name) \
    case DType::name: { \
        const ctype* sp = src.data_ptr<ctype>(); \
        ctype* d = result.data_ptr<ctype>(); \
        for (int64_t i = 0; i < n; ++i) { \
            if (!mp[i]) continue; \
            if (src_i >= src_n) { \
                TP_THROW(RuntimeError, \
                         "masked_scatter: source has fewer elements than the mask selects"); \
            } \
            d[i] = sp[src_i++]; \
        } \
        break; \
    }
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_MS_CASE)
        default: TP_THROW(TypeError, "masked_scatter: unsupported dtype");
    }
#undef TP_MS_CASE
    return result;
}

// ---------------------------------------------------------------------------
// cumsum_backward
//
// implemented as grad.flip(dim).cumsum(dim).flip(dim). Equivalent reverse
// walk: R[i] = sum_{j>=i} g[j].
// ---------------------------------------------------------------------------

Tensor cumsum_backward_cpu(const Tensor& grad, int64_t dim) {
    int64_t nd = grad.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor g = grad.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), g.dtype(), g.device());
    if (nd == 0) {
        result.copy_(g);
        return result;
    }
    int64_t d_size = g.size(dim);
    if (d_size == 0 || g.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(g.shape()), dim, outer, inner);
#define TP_CSB_CASE(ctype, acc_t, name) \
    case DType::name: { \
        const ctype* s = g.data_ptr<ctype>(); \
        ctype* d = result.data_ptr<ctype>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                acc_t acc = static_cast<acc_t>(0); \
                const ctype* sp = s + o * d_size * inner + in2; \
                ctype* dp = d + o * d_size * inner + in2; \
                for (int64_t j = d_size - 1; j >= 0; --j) { \
                    acc += static_cast<acc_t>(sp[j * inner]); \
                    dp[j * inner] = static_cast<ctype>(acc); \
                } \
            } \
        }); \
        break; \
    }
    switch (g.dtype()) {
        TP_CSB_CASE(uint8_t, uint8_t, UInt8)
        TP_CSB_CASE(int8_t, int8_t, Int8)
        TP_CSB_CASE(int16_t, int16_t, Int16)
        TP_CSB_CASE(int32_t, int32_t, Int32)
        TP_CSB_CASE(int64_t, int64_t, Int64)
        TP_CSB_CASE(uint16_t, uint16_t, UInt16)
        TP_CSB_CASE(uint32_t, uint32_t, UInt32)
        TP_CSB_CASE(uint64_t, uint64_t, UInt64)
        TP_CSB_CASE(float, double, Float32)
        TP_CSB_CASE(double, double, Float64)
        TP_CSB_CASE(Half, float, Float16)
        TP_CSB_CASE(BFloat16, float, BFloat16)
#define TP_CSB_COMPLEX_CASE(ctype, name) \
        TP_CSB_CASE(ctype, ctype, name)
        TENSORPLAY_FORALL_COMPLEX_TYPES(TP_CSB_COMPLEX_CASE)
#undef TP_CSB_COMPLEX_CASE
        default: TP_THROW(TypeError, "cumsum_backward: unsupported dtype");
    }
#undef TP_CSB_CASE
    return result;
}

// unique family.
//
// The flat form sorts the values and groups adjacent equal elements; the
// dim form sorts whole rows lexicographically and groups equal rows.
// Equality uses the original dtype's `!=`, so NaN entries never compare equal
// and each NaN survives as its own group (matching sorted order, where the
// sort kernel sinks NaNs to the rear).  `sorted=false` is accepted for API
// compatibility; grouping is inherently order-based, so the values output is
// always in ascending order.
//
// Returns (values, inverse, counts); inverse/counts are empty when the
// corresponding flag is false.
namespace {

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_flat_cpu_template(
        const Tensor& self, bool return_inverse, bool return_counts) {
    // A 0-dim input sorts as a single-element row; the inverse keeps the
    // original (scalar) shape.
    Tensor input = self.dim() == 0 ? self.reshape({1}).contiguous()
                                   : self.contiguous();
    const int64_t numel = input.numel();
    Tensor values = Tensor::empty({0}, self.dtype(), self.device());
    Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
    Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
    if (numel == 0) {
        if (return_inverse) {
            inverse.resize_(static_cast<std::vector<int64_t>>(self.shape()));
        }
        return {values, inverse, counts};
    }

    Tensor sorted_vals, order;
    std::tie(sorted_vals, order) = sort_cpu(input, 0, false);
    const scalar_t* sv = sorted_vals.data_ptr<scalar_t>();
    const int64_t* idx = order.data_ptr<int64_t>();

    // First sweep: number of groups (adjacent elements that differ).
    int64_t n_groups = 1;
    for (int64_t i = 1; i < numel; ++i) {
        if (sv[i] != sv[i - 1]) ++n_groups;
    }

    values = Tensor::empty({n_groups}, self.dtype(), self.device());
    scalar_t* vp = values.data_ptr<scalar_t>();
    int64_t* ip = return_inverse
        ? (inverse.resize_(static_cast<std::vector<int64_t>>(self.shape())),
           inverse.data_ptr<int64_t>())
        : nullptr;
    int64_t* cp = return_counts
        ? (counts.resize_({n_groups}), counts.data_ptr<int64_t>())
        : nullptr;

    // Second sweep: fill the group values, counts, and the per-element
    // inverse through the sort permutation.
    int64_t g = 0;
    int64_t group_start = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (i > 0 && sv[i] != sv[i - 1]) {
            if (return_counts) cp[g] = i - group_start;
            ++g;
            group_start = i;
        }
        vp[g] = sv[i];
        if (return_inverse) ip[idx[i]] = g;
    }
    if (return_counts) cp[g] = numel - group_start;
    return {values, inverse, counts};
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_consecutive_flat_cpu_template(
        const Tensor& self, bool return_inverse, bool return_counts) {
    Tensor input = self.dim() == 0 ? self.reshape({1}).contiguous()
                                   : self.contiguous();
    const int64_t numel = input.numel();
    Tensor values = Tensor::empty({0}, self.dtype(), self.device());
    Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
    Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
    if (numel == 0) {
        if (return_inverse) {
            inverse.resize_(static_cast<std::vector<int64_t>>(self.shape()));
        }
        return {values, inverse, counts};
    }

    values = Tensor::empty({numel}, self.dtype(), self.device());
    scalar_t* value_data = values.data_ptr<scalar_t>();
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    int64_t* inverse_data = nullptr;
    if (return_inverse) {
        inverse.resize_(static_cast<std::vector<int64_t>>(self.shape()));
        inverse_data = inverse.data_ptr<int64_t>();
    }
    int64_t* counts_data = nullptr;
    if (return_counts) {
        counts.resize_({numel});
        counts_data = counts.data_ptr<int64_t>();
    }

    scalar_t last_value = input_data[0];
    value_data[0] = last_value;
    int64_t group = 0;
    int64_t group_start = 0;
    if (inverse_data) inverse_data[0] = 0;
    for (int64_t i = 1; i < numel; ++i) {
        const scalar_t value = input_data[i];
        if (value != last_value) {
            if (counts_data) counts_data[group] = i - group_start;
            ++group;
            group_start = i;
            value_data[group] = value;
            last_value = value;
        }
        if (inverse_data) inverse_data[i] = group;
    }
    if (counts_data) counts_data[group] = numel - group_start;
    const int64_t group_count = group + 1;
    values.resize_({group_count});
    if (counts_data) counts.resize_({group_count});
    return {values, inverse, counts};
}

std::tuple<Tensor, Tensor, Tensor> unique_bool_cpu_template(
        const Tensor& self, bool return_inverse, bool return_counts) {
    Tensor input = self.dim() == 0 ? self.reshape({1}).contiguous()
                                   : self.contiguous();
    const int64_t numel = input.numel();
    Tensor values = Tensor::empty({0}, self.dtype(), self.device());
    Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
    Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
    if (numel == 0) {
        if (return_inverse) {
            inverse.resize_(static_cast<std::vector<int64_t>>(self.shape()));
        }
        return {values, inverse, counts};
    }

    const bool* input_data = input.data_ptr<bool>();
    const int thread_count = get_num_threads();
    std::vector<int64_t> true_counts(static_cast<size_t>(thread_count), 0);
    parallel_for(0, numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t& count = true_counts[static_cast<size_t>(get_thread_num())];
        for (int64_t i = begin; i < end; ++i) {
            count += input_data[i] ? 1 : 0;
        }
    });

    int64_t num_true = 0;
    for (int64_t count : true_counts) num_true += count;
    const int64_t num_false = numel - num_true;
    const int64_t num_values = (num_false > 0) + (num_true > 0);
    values.resize_({num_values});
    bool* value_data = values.data_ptr<bool>();
    const int64_t false_index = 0;
    const int64_t true_index = num_false > 0 ? 1 : 0;
    if (num_false > 0) value_data[false_index] = false;
    if (num_true > 0) value_data[true_index] = true;

    if (return_counts) {
        counts.resize_({num_values});
        int64_t* count_data = counts.data_ptr<int64_t>();
        if (num_false > 0) count_data[false_index] = num_false;
        if (num_true > 0) count_data[true_index] = num_true;
    }
    if (return_inverse) {
        inverse.resize_(static_cast<std::vector<int64_t>>(self.shape()));
        int64_t* inverse_data = inverse.data_ptr<int64_t>();
        parallel_for(0, numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                inverse_data[i] = input_data[i] ? true_index : false_index;
            }
        });
    }
    return {values, inverse, counts};
}

// Row-wise grouping over `self.moveaxis(dim, 0).view({n, -1})`: rows are
// sorted lexicographically (unless `consecutive`, which keeps the original
// order) and adjacent equal rows collapse into one output row.
template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_dim_cpu_template(
        const Tensor& self, int64_t dim, bool consecutive,
        bool return_inverse, bool return_counts) {
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(self.shape());
    const int64_t zero_dims = std::count(sizes.begin(), sizes.end(), 0);
    if (self.size(dim) == 0) {
        TP_CHECK(zero_dims == 1,
                 "Number of zero sized dimensions is more than one, so unique "
                 "cannot be applied");
        Tensor values = Tensor::empty(sizes, self.dtype(), self.device());
        Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
        Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
        return {values, inverse, counts};
    }
    TP_CHECK(zero_dims == 0,
             "There are 0 sized dimensions, and they aren't selected, so "
             "unique cannot be applied");

    Tensor input_flat = self.moveaxis(dim, 0).contiguous();
    std::vector<int64_t> front_sizes =
        static_cast<std::vector<int64_t>>(input_flat.shape());
    const int64_t n = front_sizes[0];
    input_flat = input_flat.reshape({n, -1});
    const int64_t row_len = input_flat.size(1);
    const scalar_t* rows = input_flat.data_ptr<scalar_t>();

    // Row ordering: identity for the consecutive form, lexicographic sort of
    // element columns otherwise (stable, so ties keep first-occurrence order).
    std::vector<int64_t> order(n);
    for (int64_t i = 0; i < n; ++i) order[i] = i;
    if (!consecutive) {
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            const scalar_t* ra = rows + a * row_len;
            const scalar_t* rb = rows + b * row_len;
            for (int64_t c = 0; c < row_len; ++c) {
                if (ra[c] < rb[c]) return true;
                if (rb[c] < ra[c]) return false;
            }
            return false;
        });
    }

    // Walk the ordered rows, collapsing equal adjacent rows.
    auto row_equal = [&](int64_t a, int64_t b) {
        const scalar_t* ra = rows + a * row_len;
        const scalar_t* rb = rows + b * row_len;
        for (int64_t c = 0; c < row_len; ++c) {
            if (ra[c] != rb[c]) return false;
        }
        return true;
    };

    Tensor kept_rows = Tensor::empty({n, row_len}, input_flat.dtype(),
                                     input_flat.device());
    scalar_t* kept = kept_rows.data_ptr<scalar_t>();
    Tensor inverse = Tensor::empty({n}, DType::Int64, self.device());
    int64_t* ip = inverse.data_ptr<int64_t>();
    Tensor counts_buf = Tensor::empty({0}, DType::Int64, self.device());
    int64_t* cp = return_counts
        ? (counts_buf.resize_({n}),
           counts_buf.data_ptr<int64_t>())
        : nullptr;

    int64_t n_groups = 0;
    for (int64_t k = 0; k < n; ++k) {
        const int64_t row = order[k];
        if (k == 0 || !row_equal(row, order[k - 1])) {
            std::memcpy(kept + n_groups * row_len, rows + row * row_len,
                        static_cast<size_t>(row_len) * sizeof(scalar_t));
            if (return_counts) cp[n_groups] = 1;
            ++n_groups;
        } else if (return_counts) {
            ++cp[n_groups - 1];
        }
        ip[row] = n_groups - 1;
    }

    // Rebuild the output with the selected dim resized to the group count.
    front_sizes[0] = n_groups;
    Tensor values = kept_rows.slice(0, 0, n_groups)
                        .reshape(front_sizes).moveaxis(0, dim);
    Tensor counts_t;
    if (return_counts) {
        counts_t = counts_buf.slice(0, 0, n_groups);
    }
    return {values, inverse, counts_t};
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_dispatch(const Tensor& self,
                                                   int64_t dim, bool consecutive,
                                                   bool return_inverse,
                                                   bool return_counts) {
    if (dim < 0) {
        return unique_flat_cpu_template<scalar_t>(self, return_inverse,
                                                  return_counts);
    }
    return unique_dim_cpu_template<scalar_t>(self, dim, consecutive,
                                             return_inverse, return_counts);
}

#define TP_UNIQUE_CASE(ctype, name)                                            \
    case DType::name:                                                          \
        return unique_dispatch<ctype>(self, dim, consecutive, return_inverse,  \
                                      return_counts);

std::tuple<Tensor, Tensor, Tensor> unique_any_cpu(const Tensor& self,
                                                  int64_t dim, bool consecutive,
                                                  bool return_inverse,
                                                  bool return_counts) {
    if (dim < 0 && self.dtype() == DType::Bool) {
        return unique_bool_cpu_template(self, return_inverse, return_counts);
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_UNIQUE_CASE)
        default: TP_THROW(TypeError, "unique: unsupported dtype ",
                          toString(self.dtype()));
    }
}
#undef TP_UNIQUE_CASE

} // anonymous namespace

// Flat unique with counts (the public `unique` entry point).
std::tuple<Tensor, Tensor, Tensor> unique_cpu(const Tensor& self, bool sorted,
                                              bool return_inverse,
                                              bool return_counts) {
    (void)sorted;
    return unique_any_cpu(self, /*dim=*/-1, /*consecutive=*/false,
                          return_inverse, return_counts);
}

// Flat unique without counts (legacy two-output alias).
std::tuple<Tensor, Tensor> _unique_cpu(const Tensor& self, bool sorted,
                                       bool return_inverse) {
    (void)sorted;
    auto result = unique_any_cpu(self, /*dim=*/-1, /*consecutive=*/false,
                                 return_inverse, /*return_counts=*/false);
    return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

// Full flat unique (three outputs); `unique` currently forwards here.
std::tuple<Tensor, Tensor, Tensor> _unique2_cpu(const Tensor& self, bool sorted,
                                                bool return_inverse,
                                                bool return_counts) {
    (void)sorted;
    return unique_any_cpu(self, /*dim=*/-1, /*consecutive=*/false,
                          return_inverse, return_counts);
}

// Dim-wise unique; the values output is always sorted (order-based grouping).
std::tuple<Tensor, Tensor, Tensor> unique_dim_cpu(const Tensor& self,
                                                  int64_t dim, bool sorted,
                                                  bool return_inverse,
                                                  bool return_counts) {
    (void)sorted;
    return unique_any_cpu(self, wrap_dim(dim, self.dim()),
                          /*consecutive=*/false, return_inverse, return_counts);
}

// Dim-wise unique without reordering between equal-adjacent rows.
std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_cpu(
        const Tensor& self, int64_t dim, bool return_inverse,
        bool return_counts) {
    return unique_any_cpu(self, wrap_dim(dim, self.dim()), /*consecutive=*/true,
                          return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_cpu(
        const Tensor& self, bool return_inverse, bool return_counts,
        std::optional<int64_t> dim) {
    if (!dim.has_value() || (dim.value() == 0 && self.dim() == 1)) {
#define TP_UNIQUE_CONSECUTIVE_CASE(ctype, name)                               \
        case DType::name:                                                      \
            return unique_consecutive_flat_cpu_template<ctype>(                \
                self, return_inverse, return_counts);
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_UNIQUE_CONSECUTIVE_CASE)
            default: TP_THROW(TypeError, "unique_consecutive: unsupported dtype ",
                              toString(self.dtype()));
        }
#undef TP_UNIQUE_CONSECUTIVE_CASE
    }
    return unique_dim_consecutive_cpu(self, dim.value(), return_inverse,
                                      return_counts);
}

Tensor scatter_reduce_cpu(const Tensor& self, int64_t dim, const Tensor& index,
                          const Tensor& src, const std::string& reduce,
                          bool include_self);
Tensor index_reduce_cpu(const Tensor& self, int64_t dim, const Tensor& index,
                        const Tensor& source, const std::string& reduce,
                        bool include_self);
Tensor scatter_reduce_backward_self_cpu(const Tensor& grad, const Tensor& self,
                                        int64_t dim, const Tensor& index,
                                        const Tensor& src,
                                        const std::string& reduce,
                                        bool include_self);
Tensor scatter_reduce_backward_src_cpu(const Tensor& grad, const Tensor& self,
                                       int64_t dim, const Tensor& index,
                                       const Tensor& src,
                                       const std::string& reduce,
                                       bool include_self);
Tensor index_reduce_backward_self_cpu(const Tensor& grad, const Tensor& self,
                                      int64_t dim, const Tensor& index,
                                      const Tensor& source,
                                      const std::string& reduce,
                                      bool include_self);
Tensor index_reduce_backward_src_cpu(const Tensor& grad, const Tensor& self,
                                     int64_t dim, const Tensor& index,
                                     const Tensor& source,
                                     const std::string& reduce,
                                     bool include_self);

namespace {

template <typename input_t, typename output_t>
void fill_coo_to_csr_cpu(Tensor& result, const Tensor& input, int64_t size) {
    const Tensor input_c = input.contiguous();
    const int64_t numel = input_c.numel();
    const input_t* data_in = input_c.data_ptr<input_t>();
    output_t* data_out = result.data_ptr<output_t>();

    if (numel == 0) {
        if (result.numel() > 0) {
            std::fill_n(data_out, result.numel(), static_cast<output_t>(0));
        }
        return;
    }

    const int64_t first = static_cast<int64_t>(data_in[0]);
    for (int64_t i = 0; i <= first; ++i) {
        data_out[i] = static_cast<output_t>(0);
    }

    parallel_for(0, numel - 1, GRAIN_SIZE,
                 [&](int64_t begin, int64_t end) {
        int64_t current = static_cast<int64_t>(data_in[begin]);
        for (int64_t i = begin; i < end; ++i) {
            const int64_t next = static_cast<int64_t>(data_in[i + 1]);
            for (; current < next; ++current) {
                data_out[current + 1] = static_cast<output_t>(i + 1);
            }
        }
    });

    const int64_t last = static_cast<int64_t>(data_in[numel - 1]);
    for (int64_t i = last + 1; i < size + 1; ++i) {
        data_out[i] = static_cast<output_t>(numel);
    }
}

template <typename output_t>
void dispatch_coo_to_csr_input(Tensor& result, const Tensor& input,
                               int64_t size) {
#define TP_COO_TO_CSR_CASE(ctype, name)                                      \
    case DType::name:                                                         \
        fill_coo_to_csr_cpu<ctype, output_t>(result, input, size);            \
        return;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_COO_TO_CSR_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_coo_to_csr: input must be integral");
    }
#undef TP_COO_TO_CSR_CASE
}

void check_coo_to_csr_cpu(const Tensor& input, int64_t size) {
    TP_CHECK(input.dim() <= 1,
             "_convert_indices_from_coo_to_csr: input must be a vector, got ",
             input.dim(), " dimensions");
    TP_CHECK(size >= 0,
             "_convert_indices_from_coo_to_csr: size must be non-negative, got ",
             size);
    TP_CHECK(isIntegralType(input.dtype(), false),
             "_convert_indices_from_coo_to_csr: input must be integral");
}

template <typename crow_t, typename col_t, typename output_t>
void fill_csr_to_coo_cpu(Tensor& result, const Tensor& crow_indices,
                         const Tensor& col_indices, bool transpose,
                         const std::vector<int64_t>& batch_shape) {
    const Tensor crow_c = crow_indices.contiguous();
    const Tensor col_c = col_indices.contiguous();
    const int64_t nrows = crow_c.size(-1) - 1;
    const int64_t nnz = col_c.size(-1);
    const int64_t total_nnz = col_c.numel();
    const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
    const int64_t batch_count =
        batch_ndim == 0 ? 1 : (nnz == 0 ? 0 : total_nnz / nnz);
    const int64_t row_count = batch_count * nrows;

    output_t* result_data = result.data_ptr<output_t>();
    if (nrows == 0 || nnz == 0) {
        if (result.numel() > 0) {
            std::fill_n(result_data, result.numel(), static_cast<output_t>(0));
        }
        return;
    }

    output_t* row0 = result.select(0, transpose ? batch_ndim + 1 : batch_ndim)
                          .data_ptr<output_t>();
    output_t* row1 = result.select(0, transpose ? batch_ndim : batch_ndim + 1)
                          .data_ptr<output_t>();
    const crow_t* crow_data = crow_c.data_ptr<crow_t>();
    const col_t* col_data = col_c.data_ptr<col_t>();

    parallel_for(0, total_nnz, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t batch = index / nnz;
            int64_t remainder = batch;
            for (int64_t dim = batch_ndim - 1; dim >= 0; --dim) {
                const int64_t extent = batch_shape[static_cast<size_t>(dim)];
                result_data[dim * total_nnz + index] =
                    static_cast<output_t>(remainder % extent);
                remainder /= extent;
            }
            if (transpose) {
                row0[index] = static_cast<output_t>(col_data[index]);
            } else {
                row1[index] = static_cast<output_t>(col_data[index]);
            }
        }
    });

    parallel_for(0, row_count, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t linear_row = begin; linear_row < end; ++linear_row) {
            const int64_t batch = linear_row / nrows;
            const int64_t row = linear_row % nrows;
            const int64_t base = batch * (nrows + 1);
            const int64_t start = static_cast<int64_t>(crow_data[base + row]);
            const int64_t finish =
                static_cast<int64_t>(crow_data[base + row + 1]);
            for (int64_t index = start; index < finish; ++index) {
                const int64_t flat = batch * nnz + index;
                if (transpose) {
                    row1[flat] = static_cast<output_t>(row);
                } else {
                    row0[flat] = static_cast<output_t>(row);
                }
            }
        }
    });
}

template <typename crow_t, typename output_t>
void dispatch_csr_to_coo_columns(Tensor& result, const Tensor& crow_indices,
                                const Tensor& col_indices, bool transpose,
                                const std::vector<int64_t>& batch_shape) {
#define TP_CSR_TO_COO_CASE(ctype, name)                                      \
    case DType::name:                                                         \
        fill_csr_to_coo_cpu<crow_t, ctype, output_t>(                         \
            result, crow_indices, col_indices, transpose, batch_shape);       \
        return;
    switch (col_indices.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CSR_TO_COO_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_csr_to_coo: columns must be integral");
    }
#undef TP_CSR_TO_COO_CASE
}

template <typename output_t>
void dispatch_csr_to_coo_rows(Tensor& result, const Tensor& crow_indices,
                              const Tensor& col_indices, bool transpose,
                              const std::vector<int64_t>& batch_shape) {
#define TP_CSR_TO_COO_ROW_CASE(ctype, name)                                  \
    case DType::name:                                                         \
        dispatch_csr_to_coo_columns<ctype, output_t>(                         \
            result, crow_indices, col_indices, transpose, batch_shape);       \
        return;
    switch (crow_indices.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CSR_TO_COO_ROW_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_csr_to_coo: row pointers must be integral");
    }
#undef TP_CSR_TO_COO_ROW_CASE
}

std::vector<int64_t> check_csr_to_coo_cpu(const Tensor& crow_indices,
                                          const Tensor& col_indices) {
    TP_CHECK(crow_indices.dim() >= 1 && col_indices.dim() >= 1,
             "_convert_indices_from_csr_to_coo: inputs must have at least one dimension");
    TP_CHECK(crow_indices.dim() == col_indices.dim(),
             "_convert_indices_from_csr_to_coo: inputs must have the same dimensionality");
    TP_CHECK(crow_indices.size(-1) >= 1,
             "_convert_indices_from_csr_to_coo: row pointer dimension must be non-empty");
    for (int64_t dim = 0; dim < crow_indices.dim() - 1; ++dim) {
        TP_CHECK(crow_indices.size(dim) == col_indices.size(dim),
                 "_convert_indices_from_csr_to_coo: batch dimensions must match");
    }
    TP_CHECK(isIntegralType(crow_indices.dtype(), false),
             "_convert_indices_from_csr_to_coo: row pointers must be integral");
    TP_CHECK(isIntegralType(col_indices.dtype(), false),
             "_convert_indices_from_csr_to_coo: columns must be integral");

    std::vector<int64_t> batch_shape;
    batch_shape.reserve(static_cast<size_t>(crow_indices.dim() - 1));
    int64_t batch_count = 1;
    for (int64_t dim = 0; dim < crow_indices.dim() - 1; ++dim) {
        const int64_t extent = crow_indices.size(dim);
        batch_shape.push_back(extent);
        batch_count *= extent;
    }
    const int64_t nrows = crow_indices.size(-1) - 1;
    const int64_t nnz = col_indices.size(-1);
    TP_CHECK(col_indices.numel() == batch_count * nnz,
             "_convert_indices_from_csr_to_coo: invalid batch layout");
    TP_CHECK(crow_indices.numel() == batch_count * (nrows + 1),
             "_convert_indices_from_csr_to_coo: invalid row pointer layout");
    return batch_shape;
}

Tensor convert_indices_from_coo_to_csr_cpu(const Tensor& input,
                                           int64_t size, bool out_int32) {
    check_coo_to_csr_cpu(input, size);
    Tensor result = Tensor::empty(
        {size + 1}, out_int32 ? DType::Int32 : DType::Int64, input.device());
    if (out_int32) {
        dispatch_coo_to_csr_input<int32_t>(result, input, size);
    } else {
        dispatch_coo_to_csr_input<int64_t>(result, input, size);
    }
    return result;
}

Tensor& _convert_indices_from_coo_to_csr_structured_cpu(
    const Tensor& input, int64_t size, bool out_int32, Tensor& out) {
    check_coo_to_csr_cpu(input, size);
    const DType dtype = out_int32 ? DType::Int32 : DType::Int64;
    TP_CHECK(out.defined() && out.dtype() == dtype,
             "_convert_indices_from_coo_to_csr: output dtype is incorrect");
    TP_CHECK(out.device() == input.device(),
             "_convert_indices_from_coo_to_csr: output device must match input");
    out.resize_({size + 1});
    const bool copy_back = !out.is_contiguous();
    Tensor target = copy_back
        ? Tensor::empty({size + 1}, dtype, input.device())
        : out;
    if (out_int32) {
        dispatch_coo_to_csr_input<int32_t>(target, input, size);
    } else {
        dispatch_coo_to_csr_input<int64_t>(target, input, size);
    }
    if (copy_back) out.copy_(target);
    return out;
}

Tensor convert_indices_from_csr_to_coo_cpu(const Tensor& crow_indices,
                                           const Tensor& col_indices,
                                           bool out_int32, bool transpose) {
    const std::vector<int64_t> batch_shape =
        check_csr_to_coo_cpu(crow_indices, col_indices);
    Tensor result = Tensor::empty(
        {col_indices.dim() + 1, col_indices.numel()},
        out_int32 ? DType::Int32 : DType::Int64, crow_indices.device());
    if (out_int32) {
        dispatch_csr_to_coo_rows<int32_t>(
            result, crow_indices, col_indices, transpose, batch_shape);
    } else {
        dispatch_csr_to_coo_rows<int64_t>(
            result, crow_indices, col_indices, transpose, batch_shape);
    }
    return result;
}

Tensor& _convert_indices_from_csr_to_coo_structured_cpu(
    const Tensor& crow_indices, const Tensor& col_indices, bool out_int32,
    bool transpose, Tensor& out) {
    const std::vector<int64_t> batch_shape =
        check_csr_to_coo_cpu(crow_indices, col_indices);
    const DType dtype = out_int32 ? DType::Int32 : DType::Int64;
    TP_CHECK(out.defined() && out.dtype() == dtype,
             "_convert_indices_from_csr_to_coo: output dtype is incorrect");
    TP_CHECK(out.device() == crow_indices.device(),
             "_convert_indices_from_csr_to_coo: output device must match input");
    const std::vector<int64_t> shape =
        {col_indices.dim() + 1, col_indices.numel()};
    out.resize_(shape);
    const bool copy_back = !out.is_contiguous();
    Tensor target = copy_back
        ? Tensor::empty(shape, dtype, crow_indices.device())
        : out;
    if (out_int32) {
        dispatch_csr_to_coo_rows<int32_t>(
            target, crow_indices, col_indices, transpose, batch_shape);
    } else {
        dispatch_csr_to_coo_rows<int64_t>(
            target, crow_indices, col_indices, transpose, batch_shape);
    }
    if (copy_back) out.copy_(target);
    return out;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CPU, IndexingKernels) {
    m.impl("masked_fill", masked_fill_cpu);
    m.impl("masked_fill_", masked_fill__cpu);
    m.impl("masked_fill.Tensor", masked_fill_tensor_cpu);
    m.impl("masked_fill_.Tensor", masked_fill_tensor__cpu);
    m.impl("tril", tril_cpu);
    m.impl("triu", triu_cpu);
    m.impl("cumsum", cumsum_cpu);
    m.impl("cumsum_backward", cumsum_backward_cpu);
    m.impl("cumprod", cumprod_cpu);
    m.impl("logcumsumexp", logcumsumexp_cpu);
    m.impl("repeat_interleave.Tensor", repeat_interleave_indices_cpu);
    m.impl("gather", gather_cpu);
    m.impl("scatter_add", scatter_add_cpu);
    m.impl("scatter_reduce", scatter_reduce_cpu);
    m.impl("index_reduce", index_reduce_cpu);
    m.impl("_scatter_reduce_backward_self", scatter_reduce_backward_self_cpu);
    m.impl("_scatter_reduce_backward_src", scatter_reduce_backward_src_cpu);
    m.impl("_index_reduce_backward_self", index_reduce_backward_self_cpu);
    m.impl("_index_reduce_backward_src", index_reduce_backward_src_cpu);
    m.impl("scatter.src", scatter_src_cpu);
    m.impl("scatter.value", scatter_value_cpu);
    m.impl("scatter_.src", scatter_inplace_src_cpu);
    m.impl("scatter_.value", scatter_inplace_value_cpu);
    m.impl("scatter_add_", scatter_add_inplace_cpu);
    m.impl("index_select", index_select_cpu);
    m.impl("index_add", index_add_cpu);
    m.impl("index_copy", index_copy_cpu);
    m.impl("index_fill.Tensor", index_fill_tensor_cpu);
    m.impl("index_fill.Scalar", index_fill_scalar_cpu);
    m.impl("index_fill_.Tensor", index_fill_tensor__cpu);
    m.impl("index_fill_.Scalar", index_fill_scalar__cpu);
    m.impl("index_put", index_put_cpu);
    m.impl("index_put_", index_put__cpu);
    m.impl("nonzero", nonzero_cpu);
    m.impl("unique", unique_cpu);
    m.impl("_unique", _unique_cpu);
    m.impl("_unique2", _unique2_cpu);
    m.impl("unique_dim", unique_dim_cpu);
    m.impl("unique_dim_consecutive", unique_dim_consecutive_cpu);
    m.impl("unique_consecutive", unique_consecutive_cpu);
    m.impl("sort", sort_cpu);
    m.impl("argsort", argsort_cpu);
    m.impl("searchsorted.Tensor", searchsorted_cpu);
    m.impl("searchsorted.Tensor_out", searchsorted_out_cpu);
    m.impl("searchsorted.Scalar", searchsorted_scalar_cpu);
    m.impl("searchsorted.Scalar_out", searchsorted_scalar_out_cpu);
    m.impl("bucketize.Tensor", bucketize_cpu);
    m.impl("bucketize.Tensor_out", bucketize_out_cpu);
    m.impl("bucketize.Scalar", bucketize_scalar_cpu);
    m.impl("bucketize.Scalar_out", bucketize_scalar_out_cpu);
    m.impl("_convert_indices_from_coo_to_csr", convert_indices_from_coo_to_csr_cpu);
    m.impl("_convert_indices_from_coo_to_csr.out",
           _convert_indices_from_coo_to_csr_structured_cpu);
    m.impl("_convert_indices_from_csr_to_coo", convert_indices_from_csr_to_coo_cpu);
    m.impl("_convert_indices_from_csr_to_coo.out",
           _convert_indices_from_csr_to_coo_structured_cpu);
    m.impl("bincount", bincount_cpu);
    m.impl("take", take_cpu);
    m.impl("masked_scatter", masked_scatter_cpu);
}


// ---------------------------------------------------------------------------
// Reduction modes are sum, product, mean, minimum, and maximum. With
// include_self=false only indexed slices are reset to the reduction identity;
// untouched positions keep their original values.
//
// Accumulation is deliberately serial: duplicate indices are the point of
// this op, so a data-parallel RMW over flats would race on collisions (the
// same reason bincount above is serial).
// ---------------------------------------------------------------------------

namespace {

enum class SrReduce { Sum, Prod, Mean, AMin, AMax };

SrReduce parse_sr_reduce(const std::string& r) {
    if (r == "sum") return SrReduce::Sum;
    if (r == "prod") return SrReduce::Prod;
    if (r == "mean") return SrReduce::Mean;
    if (r == "amin") return SrReduce::AMin;
    if (r == "amax") return SrReduce::AMax;
    TP_THROW(ValueError,
             "reduce argument must be one of 'sum', 'prod', 'mean', 'amin', "
             "'amax' but got: " + r);
}

template <typename T>
inline T sr_identity(SrReduce op) {
    switch (op) {
        case SrReduce::Sum:
        case SrReduce::Mean: return static_cast<T>(0);
        case SrReduce::Prod: return static_cast<T>(1);
        case SrReduce::AMin: return std::numeric_limits<T>::has_infinity
                                    ? std::numeric_limits<T>::infinity()
                                    : std::numeric_limits<T>::max();
        case SrReduce::AMax: return std::numeric_limits<T>::has_infinity
                                    ? -std::numeric_limits<T>::infinity()
                                    : std::numeric_limits<T>::lowest();
    }
    return static_cast<T>(0);  // unreachable
}

inline void sr_decode(int64_t flat, int64_t idx_dim_size, int64_t idx_inner,
                      int64_t& outer_off, int64_t& j, int64_t& idx,
                      const int64_t* ip, int64_t self_dim_size) {
    // flat layout: [outer][dim][idx_inner]
    int64_t rem = flat;
    outer_off = rem / (idx_dim_size * idx_inner);
    rem -= outer_off * idx_dim_size * idx_inner;
    j = rem % idx_inner;
    idx = ip[flat];
    if (idx < 0) idx += self_dim_size;
}

template <typename T>
inline T sr_floor_div(T v, T c) {
    T q = v / c;
    if ((v % c) != 0 && ((v < 0) != (c < 0))) --q;
    return q;
}

template <typename T>
inline void sr_mean_divide(T* d, const int64_t* cp, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t c = cp[i] == 0 ? 1 : cp[i];
        if constexpr (std::is_floating_point_v<T>) {
            d[i] /= static_cast<T>(c);
        } else if constexpr (std::is_integral_v<T> &&
                             !std::is_same_v<T, bool>) {
            d[i] = sr_floor_div(d[i], static_cast<T>(c));
        } else {
            // Half / BFloat16
            d[i] = static_cast<T>(static_cast<float>(d[i]) /
                                  static_cast<float>(c));
        }
    }
}

inline Tensor sr_where(const Tensor& cond, const Tensor& a, const Tensor& b) {
    return Tensor::where(cond, a, b);
}

} // anonymous namespace

// Shared forward body for scatter_reduce/index_reduce (identical indexing).
Tensor sr_reduce_forward(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& src_in, const std::string& reduce,
                         bool include_self) {
    const SrReduce op = parse_sr_reduce(reduce);
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "index must have the same number of dimensions as self");
    }
    if (index.numel() != 0 && index.dtype() != DType::Int32 &&
        index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected dtype int32/int64 for index");
    }
    if (src_in.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected self.dtype to be equal to src.dtype");
    }
    if (index.device() != self.device() || src_in.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "scatter_reduce: self, index, and src must be on the same device");
    }
    if (index.numel() == 0) return detail::contiguous_clone(self);
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(
        static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src_in.dim() == 0) {
        if (nd != 1 || idx_c.size(0) != 1) {
            TP_THROW(RuntimeError,
                     "src/source shape must match the index shape");
        }
        src_b = src_in.expand(idx_shape).contiguous();
    } else {
        if (src_in.dim() != nd) {
            TP_THROW(IndexError,
                     "src/source must have the same number of dimensions as index");
        }
        for (int64_t i = 0; i < nd; ++i) {
            if (i != dim && idx_c.size(i) > self.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed self shape outside the reduced dimension");
            }
            if (idx_c.size(i) > src_in.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed source shape");
            }
        }
        Tensor src_view = src_in;
        for (int64_t i = 0; i < nd; ++i) {
            if (src_view.size(i) > idx_shape[static_cast<size_t>(i)]) {
                src_view = src_view.narrow(
                    i, 0, idx_shape[static_cast<size_t>(i)]);
            }
        }
        src_b = src_view.contiguous();
    }

    const int64_t idx_inner = [&] {
        int64_t v = 1;
        for (int64_t i = dim + 1; i < nd; ++i) v *= idx_c.size(i);
        return v;
    }();
    const int64_t idx_dim_size = idx_c.size(dim);
    const int64_t total_idx = idx_c.numel();
    const int64_t self_dim_size = self.size(dim);

    {
        const int64_t* ip0 = idx_c.data_ptr<int64_t>();
        for (int64_t i = 0; i < total_idx; ++i) {
            // ("index -1 is out of bounds for dimension D with size N").
            const int64_t v = ip0[i];
            if (v < 0 || v >= self_dim_size) {
                TP_THROW(IndexError, "index ", v,
                         " is out of bounds for dimension ", dim,
                         " with size ", self_dim_size);
            }
        }
    }

    Tensor result = detail::contiguous_clone(self);
    if (!include_self && total_idx > 0 && result.numel() > 0) {
        // Reset indexed slices to the operation identity. These writes are
        // idempotent, so their flat traversal order does not matter.
        const int64_t self_inner = [&] {
            int64_t v = 1;
            for (int64_t i = dim + 1; i < nd; ++i) v *= self.size(i);
            return v;
        }();
#define TP_SR_INIT_CASE(ctype, name)                                         \
    case DType::name: {                                                      \
        ctype* d = result.data_ptr<ctype>();                                 \
        const int64_t* ip = idx_c.data_ptr<int64_t>();                       \
        const ctype init_v = sr_identity<ctype>(op);                         \
        for (int64_t flat = 0; flat < total_idx; ++flat) {                   \
            int64_t oo, j, idx;                                              \
            sr_decode(flat, idx_dim_size, idx_inner, oo, j, idx, ip,         \
                      self_dim_size);                                        \
            d[(oo * self_dim_size + idx) * self_inner + j] = init_v;         \
        }                                                                    \
        break;                                                               \
    }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_SR_INIT_CASE)
            default:
                TP_THROW(TypeError, "scatter_reduce: unsupported dtype");
        }
#undef TP_SR_INIT_CASE
    }

    // count.scatter_add_(dim, index, ones_like(src)).
    Tensor count;
    int64_t* cp = nullptr;
    if (op == SrReduce::Mean) {
        count = Tensor::full(static_cast<std::vector<int64_t>>(self.shape()),
                             include_self ? 1 : 0, DType::Int64,
                             self.device());
        cp = count.data_ptr<int64_t>();
    }

    const int64_t self_inner = [&] {
        int64_t v = 1;
        for (int64_t i = dim + 1; i < nd; ++i) v *= self.size(i);
        return v;
    }();

#define TP_SR_CASE(ctype, name)                                                \
    case DType::name: {                                                        \
        ctype* d = result.data_ptr<ctype>();                                   \
        const int64_t* ip = idx_c.data_ptr<int64_t>();                         \
        const ctype* vp = src_b.data_ptr<ctype>();                             \
        for (int64_t flat = 0; flat < total_idx; ++flat) {                     \
            int64_t oo, j, idx;                                                \
            sr_decode(flat, idx_dim_size, idx_inner, oo, j, idx, ip,           \
                      self_dim_size);                                          \
            const int64_t dst =                                                \
                (oo * self_dim_size + idx) * self_inner + j;                   \
            const ctype v = vp[flat];                                          \
            const ctype cur = d[dst];                                          \
            switch (op) {                                                      \
                case SrReduce::Sum:                                            \
                    d[dst] = cur + v;                                          \
                    break;                                                     \
                case SrReduce::Prod:                                           \
                    d[dst] = cur * v;                                          \
                    break;                                                     \
                case SrReduce::AMin:                                           \
                              \
                    d[dst] =                                                   \
                        (std::isnan(cur) || cur < v) ? cur : v;                 \
                    break;                                                     \
                case SrReduce::AMax:                                           \
                              \
                    d[dst] =                                                   \
                        (std::isnan(cur) || cur > v) ? cur : v;                 \
                    break;                                                     \
                case SrReduce::Mean:                                           \
                    d[dst] = cur + v;                                          \
                    cp[dst] += 1;                                              \
                    break;                                                     \
            }                                                                  \
        }                                                                      \
        if (op == SrReduce::Mean) {                                            \
            sr_mean_divide<ctype>(d, cp, result.numel());                       \
        }                                                                      \
        break;                                                                 \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SR_CASE)
        default:
            TP_THROW(TypeError, "scatter_reduce: unsupported dtype");
    }
#undef TP_SR_CASE

    return result;
}

Tensor scatter_reduce_cpu(const Tensor& self, int64_t dim, const Tensor& index,
                          const Tensor& src, const std::string& reduce,
                          bool include_self) {
    return sr_reduce_forward(self, dim, index, src, reduce, include_self);
}

Tensor index_reduce_cpu(const Tensor& self, int64_t dim, const Tensor& index,
                        const Tensor& source, const std::string& reduce,
                        bool include_self) {
    // unlike scatter_reduce this variant takes a 1-D index, a source of
    // self's rank (equal sizes except dim == index.numel()), and rejects
    // 'sum'.
    if (reduce != "prod" && reduce != "mean" && reduce != "amax" &&
        reduce != "amin") {
        TP_THROW(ValueError,
                 "index_reduce(): Expected reduce to be one of prod, mean, "
                 "amax or amin but got ",
                 reduce);
    }
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (nd == 0) {
        TP_THROW(RuntimeError,
                 "index_reduce(): dimension not supported for scalar tensors");
    }
    if (source.dim() != nd) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector");
    }
    if (index.dim() > 1) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector, but got dim: ",
                 index.dim());
    }
    if (index.dtype() != DType::Int32 && index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected dtype int32/int64 for index");
    }
    if (source.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected self.dtype to be equal to source.dtype");
    }
    if (index.device() != self.device() || source.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "index_reduce: self, index, and source must be on the same device");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i == dim) continue;
        if (source.size(i) != self.size(i)) {
            TP_THROW(IndexError,
                     "index_reduce(): Expected source and self to have the "
                     "same size at dimension ", i);
        }
    }
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    Tensor src_c = source.contiguous();
    const int64_t K = idx_c.numel();
    if (src_c.size(dim) != K) {
        TP_THROW(IndexError,
                 "index_reduce(): Number of indices (", K,
                 ") should be equal to source.size(dim): (", src_c.size(dim),
                 "),");
    }
    const int64_t* ip = idx_c.data_ptr<int64_t>();
    const int64_t self_dim_size = self.size(dim);
    for (int64_t j = 0; j < K; ++j) {
        if (ip[j] < 0 || ip[j] >= self_dim_size) {
            TP_THROW(IndexError, "index ", ip[j],
                     " is out of bounds for dimension ", dim,
                     " with size ", self_dim_size);
        }
    }

    const SrReduce op = parse_sr_reduce(reduce);
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= self.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);

    Tensor result = detail::contiguous_clone(self);
    Tensor count;
    int64_t* cp = nullptr;
    if (reduce == "mean") {
        count = Tensor::full(
            static_cast<std::vector<int64_t>>(self.shape()),
            include_self ? 1 : 0, DType::Int64, self.device());
        cp = count.data_ptr<int64_t>();
    }
#define TP_IR_CASE(ctype, name)                                                \
    case DType::name: {                                                        \
        ctype* d = result.data_ptr<ctype>();                                   \
        const ctype* sp = src_c.data_ptr<ctype>();                             \
        const ctype init_v = sr_identity<ctype>(op);                           \
        if (!include_self && K > 0 && result.numel() > 0) {                    \
            /* index_fill_(dim, index, identity): whole rows reset */          \
            for (int64_t oo = 0; oo < outer; ++oo) {                           \
                for (int64_t j = 0; j < K; ++j) {                              \
                    ctype* row = d + (oo * self_dim_size + ip[j]) * self_inner;\
                    for (int64_t t = 0; t < self_inner; ++t) row[t] = init_v;  \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* serial: duplicate destinations are the point of this op */           \
        for (int64_t oo = 0; oo < outer; ++oo) {                               \
            for (int64_t j = 0; j < K; ++j) {                                  \
                const int64_t dst_row = (oo * self_dim_size + ip[j]) * self_inner; \
                const int64_t src_row = (oo * K + j) * self_inner;             \
                if (cp != nullptr) {                                           \
                    int64_t* crow = cp + dst_row;                              \
                    for (int64_t t = 0; t < self_inner; ++t) crow[t] += 1;      \
                }                                                              \
                for (int64_t t = 0; t < self_inner; ++t) {                      \
                    const ctype v = sp[src_row + t];                            \
                    const ctype cur = d[dst_row + t];                           \
                    switch (op) {                                               \
                        case SrReduce::Prod:                                    \
                            d[dst_row + t] = cur * v;                           \
                            break;                                              \
                        case SrReduce::AMin:                                    \
                            d[dst_row + t] =                                    \
                                (std::isnan(cur) || cur < v) ? cur : v;         \
                            break;                                              \
                        case SrReduce::AMax:                                    \
                            d[dst_row + t] =                                    \
                                (std::isnan(cur) || cur > v) ? cur : v;         \
                            break;                                              \
                        default:                                                \
                            d[dst_row + t] = cur + v;                           \
                            break;                                              \
                    }                                                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
        if (cp != nullptr) {                                                    \
            sr_mean_divide<ctype>(d, cp, result.numel());                        \
        }                                                                       \
        break;                                                                  \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IR_CASE)
        default:
            TP_THROW(TypeError, "index_reduce: unsupported dtype");
    }
#undef TP_IR_CASE
    return result;
}


namespace {

// Re-run the forward to obtain `result` inside backward helpers.
Tensor sr_result_for_backward(const Tensor& self, int64_t dim,
                              const Tensor& index, const Tensor& src,
                              const std::string& reduce, bool include_self) {
    return sr_reduce_forward(self, dim, index, src, reduce, include_self);
}

} // anonymous namespace

Tensor scatter_reduce_backward_self_cpu(const Tensor& grad, const Tensor& self,
                                        int64_t dim, const Tensor& index,
                                        const Tensor& src,
                                        const std::string& reduce,
                                        bool include_self) {
    const SrReduce op = parse_sr_reduce(reduce);
    if (op == SrReduce::Sum) {
        // The input gradient starts as a copy of the incoming gradient.
        if (!include_self) return grad.scatter(dim, index, Scalar(0));
        return grad;
    }
    if (op == SrReduce::Mean) {
        Tensor N = include_self ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                                : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = N.scatter_add(dim, index,
                          Tensor::ones_like(src, src.dtype(), src.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        Tensor gself = grad.div(N);
        if (!include_self) gself = gself.scatter(dim, index, 0.0);
        return gself;
    }
    if (op == SrReduce::AMin || op == SrReduce::AMax) {
        Tensor result = sr_result_for_backward(self, dim, index, src, reduce,
                                               include_self);
        Tensor value = result.gather(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor src_is_result = src.eq(value).to(self.dtype());
        Tensor n_dist = self_is_result.scatter_add(dim, index, src_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = self_is_result.mul(distributed);
        if (!include_self) out = out.scatter(dim, index, Scalar(0.0));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_result = sr_reduce_forward(masked_self, dim, index, src,
                                             reduce, include_self);
    Tensor gself = grad.mul(masked_result).div(masked_self);
    if (!include_self) gself = gself.scatter(dim, index, 0.0);
    return gself;
}

Tensor scatter_reduce_backward_src_cpu(const Tensor& grad, const Tensor& self,
                                       int64_t dim, const Tensor& index,
                                       const Tensor& src,
                                       const std::string& reduce,
                                       bool include_self) {
    const SrReduce op = parse_sr_reduce(reduce);
    if (op == SrReduce::Sum) return grad.gather(dim, index);
    if (op == SrReduce::Mean) {
        Tensor N = include_self ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                                : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = N.scatter_add(dim, index,
                          Tensor::ones_like(src, src.dtype(), src.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        return grad.gather(dim, index).div(N.gather(dim, index));
    }
    if (op == SrReduce::AMin || op == SrReduce::AMax) {
        Tensor result = sr_result_for_backward(self, dim, index, src, reduce,
                                               include_self);
        Tensor value = result.gather(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor src_is_result = src.eq(value).to(self.dtype());
        Tensor n_dist = self_is_result.scatter_add(dim, index, src_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = src_is_result.mul(distributed.gather(dim, index));
        // Excluding self only changes the input gradient; source gradients
        // still receive the accumulated contribution.
        return out;
    }
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_self_result = sr_reduce_forward(masked_self, dim, index, src,
                                                  reduce, include_self);
    Tensor src_zero = src.eq(0);
    Tensor num_zeros = Tensor::zeros_like(self, self.dtype(), self.device())
                           .scatter_add(dim, index,
                                        src_zero.to(self.dtype()))
                           .gather(dim, index);
    Tensor single_zero = src_zero.bitwise_and(num_zeros.eq(1));
    Tensor masked_src = src.masked_fill(single_zero, 1.0);
    Tensor masked_src_result = sr_reduce_forward(self, dim, index, masked_src,
                                                 reduce, include_self);
    Tensor result = sr_reduce_forward(self, dim, index, src, reduce,
                                      include_self);
    Tensor gsrc = sr_where(
        single_zero,
        grad.mul(masked_src_result).gather(dim, index),
        grad.mul(result).gather(dim, index).div(src.masked_fill(src_zero, 1.0)));
    return gsrc;
}

Tensor index_reduce_backward_self_cpu(const Tensor& grad, const Tensor& self,
                                      int64_t dim, const Tensor& index,
                                      const Tensor& source,
                                      const std::string& reduce,
                                      bool include_self) {
    // This backward path uses indexed reads and writes for each reduced
    // slice.
    const SrReduce op = parse_sr_reduce(reduce);
    if (op == SrReduce::Sum) {
        if (!include_self) return grad.index_fill(dim, index, Scalar(0));
        return grad;
    }
    if (op == SrReduce::Mean) {
        Tensor N = include_self ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                                : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = Tensor::index_add(N, dim, index,
                          Tensor::ones_like(source, source.dtype(),
                                            source.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        Tensor gself = grad.div(N);
        if (!include_self) gself = gself.index_fill(dim, index, 0.0);
        return gself;
    }
    Tensor result = index_reduce_cpu(self, dim, index, source, reduce,
                                           include_self);
    if (op == SrReduce::AMin || op == SrReduce::AMax) {
        Tensor value = result.index_select(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor source_is_result = source.eq(value).to(self.dtype());
        Tensor n_dist = Tensor::index_add(self_is_result, dim, index,
                                          source_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = self_is_result.mul(distributed);
        if (!include_self) out = out.index_fill(dim, index, Scalar(0.0));
        return out;
    }
    // product
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_result = index_reduce_cpu(masked_self, dim, index, source,
                                            reduce, include_self);
    Tensor gself = grad.mul(masked_result).div(masked_self);
    if (!include_self) gself = gself.index_fill(dim, index, 0.0);
    return gself;
}

Tensor index_reduce_backward_src_cpu(const Tensor& grad, const Tensor& self,
                                     int64_t dim, const Tensor& index,
                                     const Tensor& source,
                                     const std::string& reduce,
                                     bool include_self) {
    const SrReduce op = parse_sr_reduce(reduce);
    if (op == SrReduce::Sum) return grad.index_select(dim, index);
    if (op == SrReduce::Mean) {
        Tensor N = include_self ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                                : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = Tensor::index_add(N, dim, index,
                          Tensor::ones_like(source, source.dtype(),
                                            source.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        return grad.index_select(dim, index).div(
            N.index_select(dim, index));
    }
    Tensor result = index_reduce_cpu(self, dim, index, source, reduce,
                                           include_self);
    if (op == SrReduce::AMin || op == SrReduce::AMax) {
        Tensor value = result.index_select(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor source_is_result = source.eq(value).to(self.dtype());
        Tensor n_dist = Tensor::index_add(self_is_result, dim, index,
                                          source_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = source_is_result.mul(distributed.index_select(dim, index));
        // grad_src never receives the !include_self zeroing (see above).
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_self_result = index_reduce_cpu(masked_self, dim, index,
                                                 source, reduce,
                                                 include_self);
    Tensor src_zero = source.eq(0);
    Tensor num_zeros = Tensor::index_add(
              Tensor::zeros_like(self, self.dtype(), self.device()), dim,
              index, src_zero.to(self.dtype())).index_select(dim, index);
    Tensor single_zero = src_zero.bitwise_and(num_zeros.eq(1));
    Tensor masked_source = source.masked_fill(single_zero, 1.0);
    Tensor masked_result = index_reduce_cpu(self, dim, index, masked_source,
                                            reduce, include_self);
    Tensor gsrc = sr_where(
        single_zero,
        grad.mul(masked_result).index_select(dim, index),
        grad.mul(result).index_select(dim, index).div(
            source.masked_fill(src_zero, 1.0)));
    return gsrc;
}

} // namespace cpu

} // namespace tensorplay
