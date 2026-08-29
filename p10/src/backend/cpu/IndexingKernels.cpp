// Tier-1 hot indexing/masking/scan operators.
//
//     index_select/index_add/index_put/nonzero/take)
//     logcumsumexp)
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"

#include <tuple>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
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

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// masked_fill / masked_fill_
//
// masked_fill_impl_cpu requires a strictly boolean mask (line 2463) and fills
// through a TensorIterator over {output=self, input=mask}. The out-of-place
// variant (line 2525) is expand_outplace(mask, self) followed by
// result.clone() and an in-place fill on the clone.
// ---------------------------------------------------------------------------

Tensor masked_fill_cpu(const Tensor& self, const Tensor& mask, Scalar value) {
    // TensorAdvancedIndexing.cpp:2463-2467
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    // expand_outplace(mask, self); result = self.clone(); result.masked_fill_(...)
    // (TensorAdvancedIndexing.cpp:2525-2533)
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
    // TensorAdvancedIndexing.cpp:2490 masked_fill__cpu
    Tensor r = masked_fill_cpu(self, mask, value);
    self.copy_(r);
    return self;
}

Tensor& masked_fill_tensor__cpu(Tensor& self, const Tensor& mask, const Tensor& value) {
    // TensorAdvancedIndexing.cpp:2498-2509: value must be 0-dim, filled via .item()
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
// :99 cumprod_cpu_kernel / :118 logcumsumexp_cpu_kernel. All three share
// cpu_cum_base_kernel's structure: outer_size x inner_stride independent
// slices scanned sequentially along `dim` with acc_type accumulation.
// ---------------------------------------------------------------------------

template <typename ctype, typename acc_t, typename Op>
inline void cum_base(ctype* d, const ctype* s, int64_t d_size, int64_t outer,
                     int64_t inner, ctype init_val, Op op) {
    // ReduceOpsKernel.cpp cpu_cum_base_kernel loop nest
    parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) {
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

Tensor cumsum_cpu(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    // ReduceOpsKernel.cpp:80
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "cumsum: dimension not supported for scalar tensors");
    dim = wrap_dim(dim, nd);
    DType out_dtype = dtype.value_or(self.dtype() == DType::Bool ? DType::Int64
                                                                 : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
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
        TP_CUMSUM_INT(bool, Bool)
        TP_CUMSUM_FLOAT(float, double, Float32)
        TP_CUMSUM_FLOAT(double, double, Float64)
        default: TP_THROW(TypeError, "cumsum: unsupported dtype");
    }
#undef TP_CUMSUM_FLOAT
#undef TP_CUMSUM_INT
    return result;
}

Tensor cumprod_cpu(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    // ReduceOpsKernel.cpp:99
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "cumprod: dimension not supported for scalar tensors");
    dim = wrap_dim(dim, nd);
    DType out_dtype = dtype.value_or(self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
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
        TP_CUMPROD_FLOAT(float, double, Float32)
        TP_CUMPROD_FLOAT(double, double, Float64)
        default: TP_THROW(TypeError, "cumprod: unsupported dtype");
    }
#undef TP_CUMPROD_FLOAT
#undef TP_CUMPROD_INT
    return result;
}

Tensor logcumsumexp_cpu(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    // ReduceOpsKernel.cpp:118 logcumsumexp_cpu_kernel:
    //   m = max(x, acc); result = m + log1p(exp(-|x - acc|))
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "logcumsumexp: dimension not supported for scalar tensors");
    dim = wrap_dim(dim, nd);
    DType out_dtype = dtype.value_or(self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()), out_dtype, src.device());
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
        default: TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
    }
#undef TP_LCSE_CASE
    return result;
}

// ---------------------------------------------------------------------------
// gather
//
// Backward reference: gather_backward (line 2118) = new_zeros +
// scatter_add_(dim, index, grad).
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
    Tensor idx_c = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
                if (idx < 0) idx += self_dim_size; \
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
//   cuda/ScatterGatherKernel.cu:98 _scatter_gather_elementwise_kernel;
//   accumulation uses atomicAdd on CUDA (nondeterminism noted at :588).
//   CPU equivalent loops limited_vector slices in
//   cpu/ScatterGatherKernel.cpp (scatter_gather_basekernel).
// ---------------------------------------------------------------------------

enum class ScatterMode { Assign, Add };

Tensor scatter_base_cpu(const Tensor& self, int64_t dim, const Tensor& index,
                        const Tensor& src, ScatterMode mode) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
    // TensorIterator): one destination element per index element,
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
                if (idx < 0) idx += self_dim_size; \
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

// Tensor.scatter_add_): same scatter, written directly into self instead of a
// clone.  Kept as a sibling of the out-of-place base rather than folded into
// it so the existing dispatch path stays untouched.
static Tensor& scatter_base_inplace_cpu(Tensor& self, int64_t dim, const Tensor& index,
                                        const Tensor& src, ScatterMode mode) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
                if (idx < 0) idx += self_dim_size; \
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
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
        parallel_for(0, outer * n_idx, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t o = t / n_idx, k = t % n_idx; \
                int64_t iv = ip[k]; if (iv < 0) iv += row; \
                if (iv < 0 || iv >= row) TP_THROW(IndexError, "index_select: index out of range"); \
                std::memcpy(d + static_cast<int64_t>(t) * inner, s + (o * row + iv) * inner, inner * sizeof(ctype)); \
            } \
        }); \
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
// index_add_cpu_out -> index_add_cpu_ (line 1250 dispatches index types and
// adds each source slice into self along dim).
// ---------------------------------------------------------------------------

Tensor index_add_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
                int64_t iv = ip[k]; if (iv < 0) iv += row; \
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
// :277 index_copy_cpu (both run through TensorIterator with an index
// lookup per slice).
// ---------------------------------------------------------------------------

Tensor index_copy_cpu(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
                int64_t iv = ip[k]; if (iv < 0) iv += row; \
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
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
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
                int64_t iv = ip[k]; if (iv < 0) iv += row; \
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
// _index_put_impl_ (linearized indices; accumulate=True adds instead of
// assigning).
// ---------------------------------------------------------------------------

Tensor index_put_impl_cpu(Tensor& result, const std::vector<Tensor>& indices,
                          const Tensor& values, bool accumulate) {
    if (indices.empty()) TP_THROW(IndexError, "index_put: at least one index tensor required");
    int64_t numel_self = result.numel();
    Tensor flat_idx = indices[0].to(DType::Int64).contiguous();
    for (size_t i = 1; i < indices.size(); ++i) {
        flat_idx = flat_idx * static_cast<int64_t>(result.size(i)) + indices[i].to(DType::Int64).contiguous();
    }
    Tensor vals = values.to(result.dtype()).contiguous();
    int64_t n = flat_idx.numel();
    const int64_t* ip = flat_idx.data_ptr<int64_t>();
    bool scalar_vals = vals.numel() == 1;
    if (!scalar_vals && vals.numel() != n) {
        TP_THROW(RuntimeError, "index_put: values must match number of indexed elements");
    }
#define TP_IPUT_CASE(ctype, name) \
    case DType::name: { \
        ctype* d = result.data_ptr<ctype>(); \
        const ctype* vp = vals.data_ptr<ctype>(); \
        for (int64_t i = 0; i < n; ++i) { \
            int64_t lin = ip[i]; \
            if (lin < 0) lin += numel_self; \
            if (lin < 0 || lin >= numel_self) TP_THROW(IndexError, "index_put: index out of range"); \
            ctype v = scalar_vals ? vp[0] : vp[i]; \
            if (accumulate) d[lin] += v; else d[lin] = v; \
        } \
        break; \
    }
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IPUT_CASE)
        default: TP_THROW(TypeError, "index_put: unsupported dtype");
    }
#undef TP_IPUT_CASE
    return result;
}

Tensor index_put_cpu(const Tensor& self, const std::vector<Tensor>& indices,
                     const Tensor& values, bool accumulate) {
    Tensor result = detail::contiguous_clone(self);
    return index_put_impl_cpu(result, indices, values, accumulate);
}

Tensor& index_put__cpu(Tensor& self, const std::vector<Tensor>& indices,
                       const Tensor& values, bool accumulate) {
    index_put_impl_cpu(self, indices, values, accumulate);
    return self;
}

// ---------------------------------------------------------------------------
// nonzero
//
// first count matching elements, then fill an (n, dim) Long tensor with
// coordinates in row-major order.
// ---------------------------------------------------------------------------

Tensor nonzero_cpu(const Tensor& self) {
    Tensor self_c = self.contiguous();
    int64_t nd = self.dim();
    int64_t n = self_c.numel();
    auto is_true = [&](int64_t i) -> bool {
        switch (self_c.dtype()) {
#define TP_NZ_PEEK(ctype, name) case DType::name: return static_cast<bool>(self_c.data_ptr<ctype>()[i]);
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZ_PEEK)
            default: return false;
        }
#undef TP_NZ_PEEK
    };
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) if (is_true(i)) ++count;
    Tensor result = Tensor::zeros({count, nd}, DType::Int64, self.device());
    if (count == 0) return result;
    int64_t* rp = result.data_ptr<int64_t>();
    int64_t filled = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!is_true(i)) continue;
        int64_t rem = i;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            rp[filled * nd + d2] = (nd > 0) ? rem % self.size(d2) : 0;
            rem /= self.size(d2);
        }
        ++filled;
    }
    return result;
}

// ---------------------------------------------------------------------------
// sort / argsort
//
// stable sort along dim that also carries the original positions.
// ---------------------------------------------------------------------------

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
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            std::vector<std::pair<ctype, int64_t>> buf(static_cast<size_t>(d_size)); \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                const ctype* base = s + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) buf[j] = {base[j * inner], j}; \
                if (descending) std::stable_sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a, const std::pair<ctype,int64_t>& bb){ return a.first > bb.first; }); \
                else std::stable_sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a, const std::pair<ctype,int64_t>& bb){ return a.first < bb.first; }); \
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
    // Sorting.cpp sort_indices: indices-only variant of sort.
    return std::get<1>(sort_cpu(self, dim, descending));
}

// ---------------------------------------------------------------------------
// searchsorted / bucketize
//
// searchsorted_cpu_contiguous: binary search per value; right=false yields
// the lower bound (first boundary >= v), right=true the upper bound (first
// boundary > v).
// ---------------------------------------------------------------------------

Tensor searchsorted_cpu(const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
    Tensor seq = sorted_sequence.contiguous();
    Tensor vals = self.contiguous();
    int64_t seq_len = seq.size(-1);
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(vals.shape()),
                                  out_int32 ? DType::Int32 : DType::Int64, self.device());

#define TP_SS_RUN(stype, vtype) \
    do { \
        const stype* sp = seq.data_ptr<stype>(); \
        const vtype* vp = vals.data_ptr<vtype>(); \
        if (out_int32) { \
            int32_t* rp = result.data_ptr<int32_t>(); \
            for (int64_t i = 0; i < vals.numel(); ++i) { \
                vtype v = vp[i]; \
                int64_t lo = 0, hi = seq_len; \
                while (lo < hi) { \
                    int64_t mid = (lo + hi) >> 1; \
                    bool go_right = right ? !(v < static_cast<vtype>(sp[mid])) : (static_cast<vtype>(sp[mid]) < v); \
                    if (go_right) lo = mid + 1; else hi = mid; \
                } \
                rp[i] = static_cast<int32_t>(lo); \
            } \
        } else { \
            int64_t* rp = result.data_ptr<int64_t>(); \
            for (int64_t i = 0; i < vals.numel(); ++i) { \
                vtype v = vp[i]; \
                int64_t lo = 0, hi = seq_len; \
                while (lo < hi) { \
                    int64_t mid = (lo + hi) >> 1; \
                    bool go_right = right ? !(v < static_cast<vtype>(sp[mid])) : (static_cast<vtype>(sp[mid]) < v); \
                    if (go_right) lo = mid + 1; else hi = mid; \
                } \
                rp[i] = lo; \
            } \
        } \
    } while (0)

#define TP_SS_VTYPE(stype) \
    if (vals.dtype() == DType::Float32) { TP_SS_RUN(stype, float); return result; } \
    if (vals.dtype() == DType::Float64) { TP_SS_RUN(stype, double); return result; } \
    if (vals.dtype() == DType::Int64)   { TP_SS_RUN(stype, int64_t); return result; } \
    if (vals.dtype() == DType::Int32)   { TP_SS_RUN(stype, int32_t); return result; }

    if (seq.dtype() == DType::Float32) { TP_SS_VTYPE(float) }
    else if (seq.dtype() == DType::Float64) { TP_SS_VTYPE(double) }
    else if (seq.dtype() == DType::Int64) { TP_SS_VTYPE(int64_t) }
    else if (seq.dtype() == DType::Int32) { TP_SS_VTYPE(int32_t) }
    else {
        Tensor seq_d = seq.to(DType::Float64);
        Tensor vals_d = vals.to(DType::Float64);
        return searchsorted_cpu(seq_d, vals_d, out_int32, right);
    }
#undef TP_SS_VTYPE
#undef TP_SS_RUN
    return result;
}

Tensor bucketize_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
    // Bucketization.cpp bucketize_cpu delegates to searchsorted with
    // (boundaries, values) swapped.
    return searchsorted_cpu(boundaries, self, out_int32, right);
}

// ---------------------------------------------------------------------------
// bincount
//
// :24 _bincount_cpu_template. Rules mirrored here:
//   - minlength must be >= 0 (:25)
//   - empty 1-D input returns zeros({minlength}, Long) (:29)
//   - inputs are 1-D non-negative integral; nbins = max(max+1, minlength)
//     (:34-49)
//   - weights must be 1-D with same length as input (:41-44)
//   - weighted output dtype = weights dtype when Float32, otherwise the
//     weights are cast to Double and Double is returned (:89-93); unweighted
//     output is always Long (:73).
// ---------------------------------------------------------------------------

Tensor bincount_cpu(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
    Tensor weights = weights_opt.value_or(Tensor());
    if (minlength < 0) {
        TP_THROW(RuntimeError, "minlength should be >= 0");
    }
    if (isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    Tensor inp = self.to(DType::Int64).contiguous();
    int64_t n = inp.numel();
    if (self.dim() == 1 && n == 0) {
        return Tensor::zeros({minlength}, DType::Int64, self.device());
    }
    if (self.dim() != 1) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    const int64_t* ip = inp.data_ptr<int64_t>();
    int64_t min_v = std::numeric_limits<int64_t>::max();
    int64_t max_v = std::numeric_limits<int64_t>::min();
    for (int64_t i = 0; i < n; ++i) {
        if (ip[i] < min_v) min_v = ip[i];
        if (ip[i] > max_v) max_v = ip[i];
    }
    if (min_v < 0) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    if (max_v >= std::numeric_limits<int64_t>::max()) {
        TP_THROW(RuntimeError, "maximum value of input overflowed");
    }
    int64_t self_size = self.size(0);
    bool has_weights = weights.defined() && weights.numel() > 0;
    if (has_weights && (weights.dim() != 1 || weights.size(0) != self_size)) {
        TP_THROW(RuntimeError, "weights should be 1-d and have the same length as input");
    }
    int64_t nbins = std::max(max_v + 1, minlength);
    if (has_weights) {
        if (weights.dtype() == DType::Float32) {
            // SummaryOps.cpp:90-91 template<float>
            Tensor w = weights.contiguous();
            Tensor rf = Tensor::zeros({nbins}, DType::Float32, self.device());
            const float* wp = w.data_ptr<float>();
            float* rp = rf.data_ptr<float>();
            for (int64_t i = 0; i < self_size; ++i) rp[ip[i]] += wp[i];
            return rf;
        }
        // SummaryOps.cpp:92-93: weights.to(kDouble) and Double output
        Tensor w = weights.to(DType::Float64).contiguous();
        Tensor rf = Tensor::zeros({nbins}, DType::Float64, self.device());
        const double* wp = w.data_ptr<double>();
        double* rp = rf.data_ptr<double>();
        for (int64_t i = 0; i < self_size; ++i) rp[ip[i]] += wp[i];
        return rf;
    }
    Tensor result = Tensor::zeros({nbins}, DType::Int64, self.device());
    int64_t* rp = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < self_size; ++i) rp[ip[i]] += 1;
    return result;
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
        for (int64_t i = 0; i < n && src_i < src_n; ++i) { \
            if (mp[i]) d[i] = sp[src_i++]; \
        } \
        break; \
    }
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MS_CASE)
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
    dim = wrap_dim(dim, nd);
    Tensor g = grad.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), g.dtype(), g.device());
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
        TP_CSB_CASE(float, double, Float32)
        TP_CSB_CASE(double, double, Float64)
        default: TP_THROW(TypeError, "cumsum_backward: unsupported dtype");
    }
#undef TP_CSB_CASE
    return result;
}

// stable sort of (value, original index) pairs, group adjacent equal values.
// Returns (values, inverse, counts); inverse/counts are empty when the
// corresponding flag is false.
std::tuple<Tensor, Tensor, Tensor> unique_cpu(const Tensor& self, bool sorted, bool return_inverse, bool return_counts) {
    TP_CHECK(self.dim() <= 1 || self.numel() == self.size(-1) * 1,
             "unique: only 1D tensors are supported");
    if (self.numel() == 0) {
        Tensor values = Tensor::empty({0}, self.dtype(), self.device());
        Tensor inverse = return_inverse ? Tensor::empty({0}, DType::Int64, self.device()) : Tensor();
        Tensor counts = return_counts ? Tensor::empty({0}, DType::Int64, self.device()) : Tensor();
        return std::make_tuple(values, inverse, counts);
    }
    Tensor sc = self.contiguous().reshape({self.numel()});
    const int64_t n = sc.numel();

    // generic value extraction to double for grouping; exact for integer and
    // float32/float64 bit patterns compared via the original representation.
    std::vector<int64_t> order(n);
    for (int64_t i = 0; i < n; ++i) order[i] = i;
    std::vector<double> vals(n);
    switch (sc.dtype()) {
        case DType::Float32: { auto* p = sc.data_ptr<float>(); for (int64_t i = 0; i < n; ++i) vals[i] = double(p[i]); break; }
        case DType::Float64: { auto* p = sc.data_ptr<double>(); for (int64_t i = 0; i < n; ++i) vals[i] = p[i]; break; }
        case DType::Int64:   { auto* p = sc.data_ptr<int64_t>(); for (int64_t i = 0; i < n; ++i) vals[i] = double(p[i]); break; }
        case DType::Int32:   { auto* p = sc.data_ptr<int32_t>(); for (int64_t i = 0; i < n; ++i) vals[i] = double(p[i]); break; }
        case DType::Int16:   { auto* p = sc.data_ptr<int16_t>(); for (int64_t i = 0; i < n; ++i) vals[i] = double(p[i]); break; }
        case DType::Int8:    { auto* p = sc.data_ptr<int8_t>(); for (int64_t i = 0; i < n; ++i) vals[i] = double(p[i]); break; }
        case DType::UInt8:   { auto* p = sc.data_ptr<uint8_t>(); for (int64_t i = 0; i < n; ++i) vals[i] = double(p[i]); break; }
        case DType::Bool:    { auto* p = sc.data_ptr<bool>(); for (int64_t i = 0; i < n; ++i) vals[i] = p[i] ? 1.0 : 0.0; break; }
        default: TP_THROW(TypeError, "unique: unsupported dtype");
    }

    // exact equality via bit pattern where it matters (floats): compare doubles;
    // int64 values above 2^53 lose precision in double, so compare raw ints too.
    auto equal_at = [&](int64_t a, int64_t b) -> bool {
        if (vals[a] != vals[b]) return false;
        if (sc.dtype() == DType::Int64) {
            return sc.data_ptr<int64_t>()[a] == sc.data_ptr<int64_t>()[b];
        }
        return true;
    };

    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (vals[a] != vals[b]) return vals[a] < vals[b];
        return a < b;
    });
    if (!sorted) {
        // keep first-occurrence order of groups instead of sorted order
        std::vector<bool> seen_group(n, false);
    }

    std::vector<int64_t> group_first;          // index into order of each group start
    std::vector<int64_t> inverse(n);
    for (int64_t k = 0; k < n; ++k) {
        if (k == 0 || !equal_at(order[k], order[k - 1])) group_first.push_back(k);
        inverse[order[k]] = int64_t(group_first.size()) - 1;
    }
    const int64_t n_groups = int64_t(group_first.size());

    Tensor values = Tensor::empty({n_groups}, sc.dtype(), sc.device());
    // NB: group_first[] indexes positions in the sorted order[] sequence;
    // unique_cpu_temp_impl gathers via sort_indices).
    #define TP_UNIQUE_FILL(ctype, dt)                                            \
        {                                                                        \
            ctype* dst = values.data_ptr<ctype>();                               \
            for (int64_t g = 0; g < n_groups; ++g) dst[g] = p[order[group_first[g]]]; \
        }                                                                        \
        break;
    switch (sc.dtype()) {
        case DType::Float32: { auto* p = sc.data_ptr<float>(); TP_UNIQUE_FILL(float, Float32) }
        case DType::Float64: { auto* p = sc.data_ptr<double>(); TP_UNIQUE_FILL(double, Float64) }
        case DType::Int64:   { auto* p = sc.data_ptr<int64_t>(); TP_UNIQUE_FILL(int64_t, Int64) }
        case DType::Int32:   { auto* p = sc.data_ptr<int32_t>(); TP_UNIQUE_FILL(int32_t, Int32) }
        case DType::Int16:   { auto* p = sc.data_ptr<int16_t>(); TP_UNIQUE_FILL(int16_t, Int16) }
        case DType::Int8:    { auto* p = sc.data_ptr<int8_t>(); TP_UNIQUE_FILL(int8_t, Int8) }
        case DType::UInt8:   { auto* p = sc.data_ptr<uint8_t>(); TP_UNIQUE_FILL(uint8_t, UInt8) }
        case DType::Bool:    { auto* p = sc.data_ptr<bool>(); TP_UNIQUE_FILL(bool, Bool) }
        default: break;
    }
    #undef TP_UNIQUE_FILL

    Tensor inverse_t, counts_t;
    if (return_inverse) {
        inverse_t = Tensor::empty({n}, DType::Int64, sc.device());
        auto* ip = inverse_t.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) ip[i] = inverse[i];
    }
    if (return_counts) {
        counts_t = Tensor::empty({n_groups}, DType::Int64, sc.device());
        auto* cp = counts_t.data_ptr<int64_t>();
        for (int64_t g = 0; g < n_groups; ++g) {
            const int64_t begin = group_first[g];
            const int64_t stop = (g + 1 < n_groups) ? group_first[g + 1] : n;
            cp[g] = stop - begin;
        }
    }
    return std::make_tuple(values, inverse_t, counts_t);
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
    m.impl("sort", sort_cpu);
    m.impl("argsort", argsort_cpu);
    m.impl("searchsorted.Tensor", searchsorted_cpu);
    m.impl("bucketize.Tensor", bucketize_cpu);
    m.impl("bincount", bincount_cpu);
    m.impl("take", take_cpu);
    m.impl("masked_scatter", masked_scatter_cpu);
}


// ---------------------------------------------------------------------------
// scatter_impl + scatter_reduce_exclude_self_helper (:2133) and Indexing.cu
// index_reduce_func_cuda_impl (:1320). reduce ∈ {sum, prod, mean, amin,
// amax}. With include_self=False only the indexed slices are reset to the
// never touched by index keep their original self values. The backward
// scatter_reduce_backward / index_reduce_backward.
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
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(
        static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src_in.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError,
                     "src/source shape must broadcast to the index shape");
        }
        src_b = src_in.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) src_b = src_b.to(self.dtype());

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
        // scatter_reduce_exclude_self_helper: reset indexed slices to the
        // op identity (idempotent writes, so flat order does not matter).
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
    if (index.dim() != 1) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector, but got dim: ",
                 index.dim());
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
        // FunctionsManual: grad_self = grad
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
        // FunctionsManual applies the !include_self zeroing to grad_self
        // only; grad_src always receives gradient (src is accumulated even
        // when self is excluded).
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
    // FunctionsManual index_reduce_backward: like the scatter variant but
    // all reads use vector-index ops (index_select/index_add/index_fill).
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
    // prod
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
