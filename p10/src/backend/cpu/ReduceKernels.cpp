#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "Exception.h"
#include "Parallel.h"
#include "ReductionKernels.h"
#include "TypePromotion.h"
#include "Context.h"
#include "DTypeNames.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <complex>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>
#include "OutWrite.h"

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace ops = tensorplay::tpx::ops;

Tensor isnan_cpu(const Tensor& self);

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
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
    outer = 1;
    inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i)
        inner *= shape[i];
}

template <class AccT, class Step, class Done>
Tensor reduce_dims_impl(const Tensor& self, std::vector<int64_t> dims_in,
                        bool keepdim, DType out_dtype, AccT init, Step step,
                        Done done) {
    const int64_t nd = self.dim();
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    for (auto& d : dims_in) {
        d = wrap_dim(d, nd);
        reduced[static_cast<size_t>(d)] = true;
    }
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) {
        if (!reduced[static_cast<size_t>(i)]) out_shape.push_back(self.size(i));
        else if (keepdim) out_shape.push_back(1);
    }
    std::vector<int64_t> strides(static_cast<size_t>(nd), 0);
    int64_t stride = 1;
    for (int64_t i = nd - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= self.size(i);
    }

    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(out_shape, out_dtype, self.device());
    const int64_t out_numel = out.numel();
    if (out_numel == 0) return out;

    std::vector<int64_t> red_dims;
    std::vector<int64_t> red_strides;
    std::vector<int64_t> nonred_sizes;
    std::vector<int64_t> nonred_strides;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            red_dims.push_back(i);
            red_strides.push_back(strides[static_cast<size_t>(i)]);
        } else {
            nonred_sizes.push_back(self.size(i));
            nonred_strides.push_back(strides[static_cast<size_t>(i)]);
        }
    }
    std::vector<int64_t> nonred_output_strides(nonred_sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(nonred_sizes.size()) - 2; i >= 0; --i) {
        nonred_output_strides[static_cast<size_t>(i)] =
            nonred_output_strides[static_cast<size_t>(i + 1)] *
            nonred_sizes[static_cast<size_t>(i + 1)];
    }
    int64_t total_red = 1;
    for (const int64_t dim : red_dims) total_red *= self.size(dim);

#define TP_REDUCE_FORALL_SCALAR_TYPES(_) \
    _(uint8_t, UInt8) \
    _(int8_t, Int8) \
    _(int16_t, Int16) \
    _(int32_t, Int32) \
    _(int64_t, Int64) \
    _(uint16_t, UInt16) \
    _(uint32_t, UInt32) \
    _(uint64_t, UInt64) \
    _(float, Float32) \
    _(double, Float64) \
    _(tensorplay::Half, Float16) \
    _(tensorplay::BFloat16, BFloat16) \
    _(bool, Bool)

#define TP_REDUCE_RUN_OUTPUT(output_ctype, output_name) \
    case DType::output_name: { \
        output_ctype* output_ptr = out.data_ptr<output_ctype>(); \
        parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            std::vector<int64_t> coords(red_dims.size(), 0); \
            for (int64_t oi = begin; oi < end; ++oi) { \
                int64_t base = 0; \
                int64_t rest = oi; \
                for (size_t i = 0; i < nonred_sizes.size(); ++i) { \
                    const int64_t coord = rest / nonred_output_strides[i]; \
                    rest %= nonred_output_strides[i]; \
                    base += coord * nonred_strides[i]; \
                } \
                AccT acc = init; \
                std::fill(coords.begin(), coords.end(), 0); \
                for (int64_t c = 0; c < total_red; ++c) { \
                    int64_t offset = base; \
                    for (size_t r = 0; r < red_dims.size(); ++r) \
                        offset += coords[r] * red_strides[r]; \
                    acc = step(acc, static_cast<double>(input_ptr[offset])); \
                    for (int64_t r = static_cast<int64_t>(red_dims.size()) - 1; \
                         r >= 0; --r) { \
                        if (++coords[static_cast<size_t>(r)] < \
                            self.size(red_dims[static_cast<size_t>(r)])) \
                            break; \
                        coords[static_cast<size_t>(r)] = 0; \
                    } \
                } \
                output_ptr[oi] = static_cast<output_ctype>(done(acc)); \
            } \
        }); \
        break; \
    }

#define TP_REDUCE_RUN_INPUT(input_ctype, input_name) \
    case DType::input_name: { \
        const input_ctype* input_ptr = sc.data_ptr<input_ctype>(); \
        switch (out_dtype) { \
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_REDUCE_RUN_OUTPUT) \
            default: TP_THROW(TypeError, "reduce: unsupported output dtype"); \
        } \
        break; \
    }
    switch (sc.dtype()) {
        TP_REDUCE_FORALL_SCALAR_TYPES(TP_REDUCE_RUN_INPUT)
        default: TP_THROW(TypeError, "reduce: unsupported dtype");
    }
#undef TP_REDUCE_RUN_INPUT
#undef TP_REDUCE_RUN_OUTPUT
#undef TP_REDUCE_FORALL_SCALAR_TYPES
    return out;
}

}  // namespace

static void zero_numel_check_dims(const Tensor& self,
                                  const std::vector<int64_t>& dims,
                                  const char* fn_name) {
    if (dims.empty()) {
        TP_THROW(RuntimeError, fn_name,
                 ": Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    const int64_t nd = self.dim();
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK_INDEX(self.size(d) != 0, fn_name,
                       ": Expected reduction dim ", d,
                       " to have non-zero size.");
    }
}

Tensor amax_cpu(const Tensor& self, const std::vector<int64_t>& dim_in,
                bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim_in, "amax()");
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    return extrema_dim_stub(DeviceType::CPU, self, resolved, keepdim, true);
}

Tensor amin_cpu(const Tensor& self, const std::vector<int64_t>& dim_in,
                bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim_in, "amin()");
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    return extrema_dim_stub(DeviceType::CPU, self, resolved, keepdim, false);
}

std::tuple<Tensor, Tensor> aminmax_cpu(
    const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    if (self.numel() == 0) {
        if (dim_in.empty()) {
            TP_THROW(RuntimeError,
                     "aminmax(): cannot compute aminmax over an empty dimension as "
                     "the operation has no identity.");
        }
        zero_numel_check_dims(self, dim_in, "aminmax");
    }
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    return {amin_cpu(self, resolved, keepdim),
            amax_cpu(self, resolved, keepdim)};
}

std::tuple<Tensor, Tensor> aminmax_all_cpu(const Tensor& self) {
    return aminmax_cpu(self, {}, false);
}

std::tuple<Tensor, Tensor> aminmax_dim_cpu(const Tensor& self, int64_t dim,
                                           bool keepdim) {
    return aminmax_cpu(self, {dim}, keepdim);
}

// Every listed dimension is reduced in one pass; an empty list reduces them
// all.  Integer and boolean inputs answer in the default floating type.
Tensor logsumexp_cpu(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const bool integral = isIntegralType(self.dtype(), true);
    if (!isFloatingType(self.dtype()) && !integral)
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type, got ",
                 scalarTypeName(self.dtype()));
    const DType out_dtype = integral ? globalContext().defaultDType() : self.dtype();
    const int64_t nd = self.dim();
    std::vector<int64_t> dims;
    std::vector<bool> seen(static_cast<size_t>(std::max<int64_t>(nd, 1)), false);
    for (int64_t d : dim) {
        const int64_t wrapped = wrap_dim(d, std::max<int64_t>(nd, 1));
        TP_CHECK(!seen[static_cast<size_t>(wrapped)], "logsumexp(): dim ", wrapped,
                 " appears multiple times in the list of dims");
        seen[static_cast<size_t>(wrapped)] = true;
        dims.push_back(wrapped);
    }
    if (nd == 0) {
        // A zero-dim tensor reduces along dim 0 or -1 as one value along one
        // axis, and one value is its own log-sum-exp.
        return self.to(out_dtype).clone();
    }
    if (dims.empty()) {
        for (int64_t i = 0; i < nd; ++i) dims.push_back(i);
    }
    // The answer is built the way the reference runtime builds it: the
    // maximum over the reduced axes, the exponentials of the difference to
    // it, their sum, and the logarithm of that sum.  Every step is one of
    // the vectorized kernels, where a single walk over the values with a
    // running state would have to touch them one at a time.
    Tensor x = self;
    if (isReducedFloatingType(x.dtype())) {
        // A narrow input is walked at single precision.
        x = x.to(DType::Float32);
    } else if (integral) {
        x = x.to(out_dtype);
    }
    Tensor maxes = amax_cpu(x, dims, /*keepdim=*/true);
    // An infinite maximum would make every exponential overflow, so it is
    // set to zero; the sum then carries the infinity on its own.
    const double infinity = std::numeric_limits<double>::infinity();
    Tensor safe = maxes.masked_fill(maxes.abs().eq(Scalar(infinity)),
                                    Scalar(0.0));
    Tensor shifted = x.sub(safe);
    shifted.exp_();
    Tensor sums = sum_dim_stub(DeviceType::CPU, shifted, dims, keepdim, out_dtype);
    Tensor result = sums.log_();
    if (keepdim) {
        result = result.add(safe);
    } else {
        // The maximum is squeezed axis by axis, from the last one back, so
        // that the axis numbers of the ones before it stay valid.
        std::vector<int64_t> descending(dims);
        std::sort(descending.rbegin(), descending.rend());
        Tensor flat = safe;
        for (int64_t d : descending) flat = flat.squeeze(d);
        result = result.add(flat);
    }
    return result.to(out_dtype);
}

Tensor nansum_cpu(const Tensor& self, const std::vector<int64_t>& dim_in,
                  bool keepdim) {
    if (isComplexType(self.dtype())) {
        TP_THROW(RuntimeError, "nansum on CPU does not support complex inputs");
    }
    const DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    std::vector<int64_t> dim = dim_in;
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, dim, keepdim, out_dt, 0.0,
        [](double acc, double value) {
            return (value != value) ? acc : acc + value;
        },
        [](double acc) { return acc; });
}

std::tuple<Tensor, Tensor> cummax_cpu(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                DType::Int64, sc.device());
    if (nd == 0) {
        vals.copy_(sc);
        idxs.fill_(Scalar(0));
        return {vals, idxs};
    }
    const int64_t d_size = sc.size(dim);
    if (d_size == 0 || sc.numel() == 0) return {vals, idxs};
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
#define TP_CUMMAX_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        /* A column at a time reads and writes one line per element and    \
           revisits those lines once per column.  A block of columns that   \
           fills a line advances every lane of it together, so each line of \
           the scanned axis is read and written once. */ \
        constexpr int64_t kLanes = std::max<int64_t>(1, 64 / static_cast<int64_t>(sizeof(ctype))); \
        if (inner >= kLanes) { \
            const int64_t col_blocks = (inner + kLanes - 1) / kLanes; \
            const int64_t block_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * col_blocks, block_grain, [&](int64_t b, int64_t e) { \
                for (int64_t flat = b; flat < e; ++flat) { \
                    const int64_t o = flat / col_blocks; \
                    const int64_t c0 = (flat - o * col_blocks) * kLanes; \
                    const int64_t cols = std::min(kLanes, inner - c0); \
                    const ctype* base = sp + o * d_size * inner + c0; \
                    ctype* vbase = vp + o * d_size * inner + c0; \
                    int64_t* ibase = ip + o * d_size * inner + c0; \
                    if (cols == kLanes) { \
                        ctype best[kLanes]; \
                        int64_t best_i[kLanes]; \
                        for (int64_t c = 0; c < kLanes; ++c) { \
                            best[c] = base[c]; \
                            best_i[c] = 0; \
                            vbase[c] = best[c]; \
                            ibase[c] = 0; \
                        } \
                        for (int64_t j = 1; j < d_size; ++j) { \
                            const ctype* row = base + j * inner; \
                            ctype* vrow = vbase + j * inner; \
                            int64_t* irow = ibase + j * inner; \
                            for (int64_t c = 0; c < kLanes; ++c) { \
                                const ctype current = row[c]; \
                                if (current != current || \
                                    (best[c] == best[c] && current >= best[c])) { \
                                    best[c] = current; \
                                    best_i[c] = j; \
                                } \
                                vrow[c] = best[c]; \
                                irow[c] = best_i[c]; \
                            } \
                        } \
                    } else { \
                        for (int64_t c = 0; c < cols; ++c) { \
                            const ctype* src = base + c; \
                            ctype* dst = vbase + c; \
                            int64_t* out_i = ibase + c; \
                            ctype best = src[0]; \
                            int64_t best_i = 0; \
                            dst[0] = best; out_i[0] = 0; \
                            for (int64_t j = 1; j < d_size; ++j) { \
                                const ctype current = src[j * inner]; \
                                if (current != current || \
                                    (best == best && current >= best)) { \
                                    best = current; best_i = j; \
                                } \
                                dst[j * inner] = best; out_i[j * inner] = best_i; \
                            } \
                        } \
                    } \
                } \
            }); \
            break; \
        } \
        const int64_t slice_grain = \
            std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1)); \
        parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                ctype* dst = vp + o * d_size * inner + in2; \
                int64_t* out_i = ip + o * d_size * inner + in2; \
                ctype best = src[0]; \
                int64_t best_i = 0; \
                dst[0] = best; out_i[0] = 0; \
                for (int64_t j = 1; j < d_size; ++j) { \
                    const ctype current = src[j * inner]; \
                    if (current != current || (best == best && current >= best)) { \
                        best = current; best_i = j; \
                    } \
                    dst[j * inner] = best; out_i[j * inner] = best_i; \
                } \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CUMMAX_CASE)
        default: TP_THROW(TypeError, "cummax: unsupported dtype");
    }
#undef TP_CUMMAX_CASE
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> cummin_cpu(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()),
                                DType::Int64, sc.device());
    if (nd == 0) {
        vals.copy_(sc);
        idxs.fill_(Scalar(0));
        return {vals, idxs};
    }
    const int64_t d_size = sc.size(dim);
    if (d_size == 0 || sc.numel() == 0) return {vals, idxs};
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
#define TP_CUMMIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        /* A column at a time reads and writes one line per element and    \
           revisits those lines once per column.  A block of columns that   \
           fills a line advances every lane of it together, so each line of \
           the scanned axis is read and written once. */ \
        constexpr int64_t kLanes = std::max<int64_t>(1, 64 / static_cast<int64_t>(sizeof(ctype))); \
        if (inner >= kLanes) { \
            const int64_t col_blocks = (inner + kLanes - 1) / kLanes; \
            const int64_t block_grain = std::max<int64_t>(1, GRAIN_SIZE / d_size); \
            parallel_for(0, outer * col_blocks, block_grain, [&](int64_t b, int64_t e) { \
                for (int64_t flat = b; flat < e; ++flat) { \
                    const int64_t o = flat / col_blocks; \
                    const int64_t c0 = (flat - o * col_blocks) * kLanes; \
                    const int64_t cols = std::min(kLanes, inner - c0); \
                    const ctype* base = sp + o * d_size * inner + c0; \
                    ctype* vbase = vp + o * d_size * inner + c0; \
                    int64_t* ibase = ip + o * d_size * inner + c0; \
                    if (cols == kLanes) { \
                        ctype best[kLanes]; \
                        int64_t best_i[kLanes]; \
                        for (int64_t c = 0; c < kLanes; ++c) { \
                            best[c] = base[c]; \
                            best_i[c] = 0; \
                            vbase[c] = best[c]; \
                            ibase[c] = 0; \
                        } \
                        for (int64_t j = 1; j < d_size; ++j) { \
                            const ctype* row = base + j * inner; \
                            ctype* vrow = vbase + j * inner; \
                            int64_t* irow = ibase + j * inner; \
                            for (int64_t c = 0; c < kLanes; ++c) { \
                                const ctype current = row[c]; \
                                if (current != current || \
                                    (best[c] == best[c] && current <= best[c])) { \
                                    best[c] = current; \
                                    best_i[c] = j; \
                                } \
                                vrow[c] = best[c]; \
                                irow[c] = best_i[c]; \
                            } \
                        } \
                    } else { \
                        for (int64_t c = 0; c < cols; ++c) { \
                            const ctype* src = base + c; \
                            ctype* dst = vbase + c; \
                            int64_t* out_i = ibase + c; \
                            ctype best = src[0]; \
                            int64_t best_i = 0; \
                            dst[0] = best; out_i[0] = 0; \
                            for (int64_t j = 1; j < d_size; ++j) { \
                                const ctype current = src[j * inner]; \
                                if (current != current || \
                                    (best == best && current <= best)) { \
                                    best = current; best_i = j; \
                                } \
                                dst[j * inner] = best; out_i[j * inner] = best_i; \
                            } \
                        } \
                    } \
                } \
            }); \
            break; \
        } \
        const int64_t slice_grain = \
            std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1)); \
        parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                ctype* dst = vp + o * d_size * inner + in2; \
                int64_t* out_i = ip + o * d_size * inner + in2; \
                ctype best = src[0]; \
                int64_t best_i = 0; \
                dst[0] = best; out_i[0] = 0; \
                for (int64_t j = 1; j < d_size; ++j) { \
                    const ctype current = src[j * inner]; \
                    if (current != current || (best == best && current <= best)) { \
                        best = current; best_i = j; \
                    } \
                    dst[j * inner] = best; out_i[j * inner] = best_i; \
                } \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CUMMIN_CASE)
        default: TP_THROW(TypeError, "cummin: unsupported dtype");
    }
#undef TP_CUMMIN_CASE
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> std_mean_cpu(const Tensor& self,
                                        const std::vector<int64_t>& dim,
                                        bool unbiased, bool keepdim) {
    auto vr_mean = var_mean_stub(
        DeviceType::CPU, self, dim, unbiased ? int64_t(1) : int64_t(0),
        keepdim);
    return {vr_mean.first.sqrt(), vr_mean.second};
}

std::tuple<Tensor, Tensor> var_mean_cpu(const Tensor& self,
                                        const std::vector<int64_t>& dim,
                                        bool unbiased, bool keepdim) {
    return var_mean_stub(
        DeviceType::CPU, self, dim, unbiased ? int64_t(1) : int64_t(0),
        keepdim);
}

Tensor nanmedian_cpu(const Tensor& self) {
    Tensor flat = self.to(isFloatingType(self.dtype()) ? DType::Float64
                                                        : DType::Int64)
                      .reshape({self.numel()});
    std::vector<double> values;
    values.reserve(flat.numel());
    const double* data = flat.data_ptr<double>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        if (!(data[i] != data[i])) values.push_back(data[i]);
    }
    const DType out_dtype = isFloatingType(self.dtype()) ? self.dtype()
                                                          : DType::Int64;
    if (values.empty()) {
        Tensor out = Tensor::zeros({}, out_dtype, self.device());
        if (isFloatingType(out_dtype))
            out.fill_(Scalar(std::numeric_limits<double>::quiet_NaN()));
        else
            out.fill_(Scalar(std::numeric_limits<int64_t>::min()));
        return out;
    }
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>((values.size() - 1) / 2),
                     values.end());
    return Tensor::zeros({}, out_dtype, self.device())
        .fill_(Scalar(values[(values.size() - 1) / 2]));
}

std::tuple<Tensor, Tensor> nanmedian_dim_cpu(const Tensor& self, int64_t dim,
                                             bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0,
             "nanmedian(): expects a tensor with at least one dimension");
    dim = wrap_dim(dim, nd);
    TP_CHECK(isFloatingType(self.dtype()),
             "nanmedian(): only floating point dtypes are supported");
    Tensor sc = self.contiguous();
    const int64_t d_size = sc.size(dim);
    TP_CHECK(d_size > 0, "nanmedian(): Expected reduction dim ", dim,
             " to have non-zero size");
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i)
        out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, sc.device());

#define TP_NANMEDIAN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        const int64_t slice_grain = \
            std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1)); \
        parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) { \
            std::vector<std::pair<ctype, int64_t>> buf( \
                static_cast<size_t>(std::max<int64_t>(d_size, 1))); \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                int64_t valid = 0; \
                for (int64_t j = 0; j < d_size; ++j) { \
                    const ctype value = src[j * inner]; \
                    if (value != value) continue; \
                    buf[static_cast<size_t>(valid++)] = {value, j}; \
                } \
                const int64_t oi = keepdim ? si : (o * inner + in2); \
                if (valid == 0) { \
                    vp[oi] = std::numeric_limits<ctype>::quiet_NaN(); \
                    ip[oi] = 0; \
                    continue; \
                } \
                const int64_t middle = (valid - 1) / 2; \
                std::nth_element(buf.begin(), buf.begin() + middle, \
                                 buf.begin() + valid, \
                                 [](const auto& a, const auto& b) { \
                    if (a.first != b.first) return a.first < b.first; \
                    return a.second < b.second; \
                }); \
                vp[oi] = buf[static_cast<size_t>(middle)].first; \
                ip[oi] = buf[static_cast<size_t>(middle)].second; \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TP_NANMEDIAN_CASE(float, Float32)
        TP_NANMEDIAN_CASE(double, Float64)
        default: TP_THROW(TypeError, "nanmedian: unsupported dtype ",
                          toString(sc.dtype()));
    }
#undef TP_NANMEDIAN_CASE
    return {values, indices};
}

std::tuple<Tensor, Tensor> nanmedian_dim_values_cpu(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
    Tensor& indices) {
    auto result = nanmedian_dim_cpu(self, dim, keepdim);
    write_out(values, std::get<0>(result));
    write_out(indices, std::get<1>(result));
    return {values, indices};
}

std::tuple<Tensor, Tensor> mode_cpu(const Tensor& self, int64_t dim,
                                    bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar mode input: ", dim);
        }
        Tensor values = Tensor::empty({}, self.dtype(), self.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, self.device());
        values.copy_(self);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    const int64_t d_size = sc.size(dim);
    if (d_size == 0) {
        TP_THROW(RuntimeError,
                 "mode(): Expected reduction dim ", dim,
                 " to have non-zero size");
    }
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i)
        out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, sc.device());
#define TP_MODE_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        const int64_t slice_grain = std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1)); \
        parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) { \
            using pair_t = std::pair<ctype, int64_t>; \
            static thread_local std::vector<pair_t> buf; \
            if (static_cast<int64_t>(buf.size()) < d_size) buf.resize(static_cast<size_t>(d_size)); \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) buf[j] = {src[j * inner], j}; \
                std::sort(buf.begin(), buf.begin() + d_size, [](const auto& a, const auto& b) { \
                    if (!(a.first < b.first) && !(b.first < a.first)) return a.second < b.second; \
                    return a.first < b.first; \
                }); \
                ctype best_value = buf[0].first; \
                int64_t best_count = 0, best_index = buf[0].second, run = 0; \
                for (int64_t j = 0; j < d_size; ++j) { \
                    const bool same = j > 0 && !(buf[j].first < buf[j - 1].first) && \
                                      !(buf[j - 1].first < buf[j].first); \
                    run = same ? run + 1 : 1; \
                    if (run > best_count) { \
                        best_count = run; best_value = buf[j].first; best_index = buf[j].second; \
                    } \
                } \
                vp[si] = best_value; ip[si] = best_index; \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_CASE)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODE_CASE
    return {values, indices};
}

template <typename T>
inline bool kthvalue_is_nan(T value) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return std::isnan(value);
    } else if constexpr (std::is_same_v<T, Half> ||
                         std::is_same_v<T, BFloat16>) {
        return std::isnan(static_cast<float>(value));
    } else {
        return false;
    }
}

template <typename T>
struct KthValueLess {
    bool operator()(const std::pair<T, int64_t>& lhs,
                    const std::pair<T, int64_t>& rhs) const {
        const bool lhs_nan = kthvalue_is_nan(lhs.first);
        const bool rhs_nan = kthvalue_is_nan(rhs.first);
        if (lhs_nan != rhs_nan) return !lhs_nan;
        if (lhs_nan) return lhs.second < rhs.second;
        if (lhs.first != rhs.first) return lhs.first < rhs.first;
        return lhs.second < rhs.second;
    }
};

std::tuple<Tensor, Tensor> kthvalue_cpu(const Tensor& self, int64_t k,
                                        int64_t dim, bool keepdim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar kthvalue input: ", dim);
        }
        if (k != 1) {
            TP_THROW(RuntimeError,
                     "kthvalue(): selected number k out of range for dim 0");
        }
        Tensor values = Tensor::empty({}, self.dtype(), self.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, self.device());
        values.copy_(self);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    const int64_t d_size = sc.size(dim);
    if (k < 1 || k > d_size)
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i)
        out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, sc.device());
#define TP_KTH_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = values.data_ptr<ctype>(); \
        int64_t* ip = indices.data_ptr<int64_t>(); \
        const int64_t slice_grain = \
            std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1)); \
        parallel_for(0, outer * inner, slice_grain, [&](int64_t b, int64_t e) { \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            std::vector<ctype> vals; \
            if constexpr (std::is_same_v<ctype, float> || std::is_same_v<ctype, double>) { \
                vals.resize(static_cast<size_t>(d_size)); \
            } \
            for (int64_t si = b; si < e; ++si) { \
                const int64_t o = si / inner, in2 = si % inner; \
                const ctype* src = sp + o * d_size * inner + in2; \
                if constexpr (std::is_same_v<ctype, float> || std::is_same_v<ctype, double>) { \
                    /* Selecting over a dense copy of the slice keeps every probe in \
                       one short contiguous run and swaps a single element instead \
                       of a value/index pair; the index follows from the original \
                       slice under the same tie rule (value, then position).  Rows \
                       holding a NaN keep the pair path, where the NaN ordering and \
                       the index tie-break live. */ \
                    bool has_nan = false; \
                    for (int64_t j = 0; j < d_size; ++j) { \
                        const ctype v = src[j * inner]; \
                        vals[static_cast<size_t>(j)] = v; \
                        has_nan |= std::isnan(v); \
                    } \
                    if (!has_nan) { \
                        std::nth_element(vals.begin(), vals.begin() + (k - 1), vals.end()); \
                        const ctype v = vals[static_cast<size_t>(k - 1)]; \
                        int64_t less = 0; \
                        for (int64_t j = 0; j < d_size; ++j) \
                            less += (src[j * inner] < v) ? 1 : 0; \
                        int64_t idx = 0; \
                        while (src[idx * inner] != v) ++idx; \
                        int64_t rank = (k - 1) - less; \
                        for (int64_t j = idx + 1; rank > 0 && j < d_size; ++j) { \
                            if (src[j * inner] == v && --rank == 0) idx = j; \
                        } \
                        const int64_t oi = keepdim ? si : (o * inner + in2); \
                        vp[oi] = v; \
                        ip[oi] = idx; \
                        continue; \
                    } \
                } \
                for (int64_t j = 0; j < d_size; ++j) buf[j] = {src[j * inner], j}; \
                std::nth_element(buf.begin(), buf.begin() + (k - 1), buf.end(), KthValueLess<ctype>{}); \
                const int64_t oi = keepdim ? si : (o * inner + in2); \
                vp[oi] = buf[static_cast<size_t>(k - 1)].first; \
                ip[oi] = buf[static_cast<size_t>(k - 1)].second; \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_KTH_CASE)
        default: TP_THROW(TypeError, "kthvalue: unsupported dtype");
    }
#undef TP_KTH_CASE
    return {values, indices};
}

template <typename scalar_t>
inline bool count_nonzero_value(const scalar_t& value) {
    if constexpr (is_complex_type_v<scalar_t>) {
        using component_t = typename is_complex_type<scalar_t>::value_type;
        const component_t zero(0);
        return value.real() != zero || value.imag() != zero;
    } else {
        return static_cast<bool>(value);
    }
}

template <typename scalar_t>
int64_t count_nonzero_flat_cpu_impl(const Tensor& self) {
    Tensor input = self.contiguous();
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    std::vector<int64_t> thread_counts(
        static_cast<size_t>(get_num_threads()), 0);
    parallel_for(0, input.numel(), GRAIN_SIZE,
                 [&](int64_t begin, int64_t end) {
        int64_t local_count = 0;
        for (int64_t i = begin; i < end; ++i) {
            local_count += count_nonzero_value(input_data[i]) ? 1 : 0;
        }
        thread_counts[static_cast<size_t>(get_thread_num())] += local_count;
    });
    int64_t count = 0;
    for (const int64_t thread_count : thread_counts) count += thread_count;
    return count;
}

// Counts the elements that are not zero along one reduced dimension.  The
// buffer splits into (outer, dim_size, inner), so each count reads the
// elements of one row: a packed sequence when the dimension is the last one
// and a strided one otherwise.  The count of an outer slot and an inner
// position lands at the same place in the answer as the pair they name.
template <typename scalar_t>
Tensor count_nonzero_one_dim_cpu_impl(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) {
        if (i != dim) out_shape.push_back(self.size(i));
    }
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= self.size(i);
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    const int64_t dim_size = self.size(dim);

    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    const int64_t tasks = outer * inner;
    if (tasks == 0) return out;
    if (dim_size == 0) {
        // A row with no element holds no lane that stands, so every answer the
        // walk would have written is a zero rather than whatever the allocator
        // left behind.
        return out.zero_();
    }

    const scalar_t* in = sc.data_ptr<scalar_t>();
    int64_t* op = out.data_ptr<int64_t>();
    // One task counts one row, so the grain counts the rows a chunk covers.
    const int64_t grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(dim_size * inner, 1));
    parallel_for(0, tasks, grain, [&](int64_t begin, int64_t end) {
        for (int64_t t = begin; t < end; ++t) {
            const int64_t o = inner == 1 ? t : t / inner;
            const int64_t in_off = inner == 1 ? 0 : t % inner;
            const int64_t base = o * dim_size * inner + in_off;
            int64_t count = 0;
            for (int64_t k = 0; k < dim_size; ++k) {
                count += count_nonzero_value(in[base + k * inner]) ? 1 : 0;
            }
            op[t] = count;
        }
    });
    return out;
}

Tensor count_nonzero_cpu(const Tensor& self, const std::vector<int64_t>& dim) {
    if (dim.empty()) {
        int64_t count = 0;
#define TP_COUNT_CASE(ctype, name_) \
    case DType::name_: count = count_nonzero_flat_cpu_impl<ctype>(self); break;
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_COUNT_CASE)
            default: TP_THROW(TypeError, "count_nonzero: unsupported dtype");
        }
#undef TP_COUNT_CASE
        return Tensor::zeros({}, DType::Int64, self.device()).fill_(Scalar(count));
    }
    // One reduced dimension has a walk of its own; more than one falls back
    // to the composed reduction.
    const int64_t nd = self.dim();
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    std::vector<int64_t> dims;
    for (const int64_t d : dim) {
        const int64_t w = wrap_dim(d, nd);
        if (!reduced[static_cast<size_t>(w)]) {
            reduced[static_cast<size_t>(w)] = true;
            dims.push_back(w);
        }
    }
    if (dims.size() == 1) {
#define TP_COUNT_DIM_CASE(ctype, name_) \
    case DType::name_: \
        return count_nonzero_one_dim_cpu_impl<ctype>(self, dims[0]);
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_COUNT_DIM_CASE)
            default: TP_THROW(TypeError, "count_nonzero: unsupported dtype");
        }
#undef TP_COUNT_DIM_CASE
    }
    return reduce_dims_impl<double>(
        self, dim, false, DType::Int64, 0.0,
        [](double acc, double value) { return value != 0.0 ? acc + 1.0 : acc; },
        [](double acc) { return acc; });
}

Tensor dist_cpu(const Tensor& self, const Tensor& other, const Scalar& p) {
    return (self - other).norm(p.toDouble());
}

Tensor renorm_cpu(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
    if (p.isComplex()) {
        TP_THROW(TypeError, "renorm: p must be real-valued");
    }
    const double pd = p.toDouble();
    if (!(pd > 0.0)) {
        TP_THROW(ValueError, "renorm: norm order must be positive");
    }
    if (maxnorm.isComplex()) {
        TP_THROW(TypeError, "renorm: maxnorm must be real-valued");
    }
    const double max_norm = maxnorm.toDouble();
    if (!(max_norm >= 0.0)) {
        TP_THROW(ValueError, "renorm: maxnorm must be non-negative");
    }
    if (!isFloatingOrComplexType(self.dtype())) {
        TP_THROW(TypeError, "renorm: input must have a floating or complex dtype");
    }
    const int64_t nd = self.dim();
    if (nd <= 1) {
        TP_THROW(RuntimeError, "renorm: input must have at least 2 dimensions");
    }
    dim = wrap_dim(dim, nd);

    std::vector<int64_t> reduce_dims;
    reduce_dims.reserve(static_cast<size_t>(nd - 1));
    for (int64_t axis = 0; axis < nd; ++axis) {
        if (axis != dim) reduce_dims.push_back(axis);
    }

    Tensor norms = ops::norm(self, reduce_dims, pd, true);
    Tensor factors = ops::where(
        ops::gt(norms, Scalar(max_norm)),
        ops::div(
            Tensor::full_like(norms, Scalar(max_norm), norms.dtype(), norms.device()),
            ops::add(norms, Scalar(1e-7))),
        Scalar(1.0));
    Tensor result = ops::mul(self, factors);
    return result.dtype() == self.dtype() ? result : result.to(self.dtype());
}

Tensor nanmean_cpu(const Tensor& self, std::optional<int64_t> dim_opt,
                  bool keepdim, std::optional<DType> dtype) {
    DType acc_dt = dtype.value_or(DType::Undefined);
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "nanmean(): expected input to have floating point or complex dtype but got ",
                 toString(self.dtype()));
    }
    if (acc_dt != DType::Undefined && !isFloatingType(acc_dt) &&
        !isComplexType(acc_dt)) {
        TP_THROW(TypeError,
                 "nanmean(): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: ",
                 toString(acc_dt));
    }
    Tensor x = self;
    const bool complex_output_from_real =
        acc_dt != DType::Undefined && isComplexType(acc_dt) &&
        !isComplexType(x.dtype());
    if (acc_dt != DType::Undefined && x.dtype() != acc_dt &&
        !complex_output_from_real) {
        x = x.to(acc_dt);
    } else if (isReducedFloatingType(x.dtype()) && acc_dt == DType::Undefined) {
        x = x.to(DType::Float32);
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) {
        dims.push_back(*dim_opt);
    } else {
        for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    Tensor total = nansum_cpu(x, dims, keepdim);
    Tensor valid = isnan_cpu(x).logical_not();
    Tensor count = reduce_dims_impl<double>(
        valid, dims, keepdim, DType::Float32, 0.0,
        [](double acc, double value) { return acc + value; },
        [](double acc) { return acc; });
    Tensor zero = count.eq(Scalar(0.0));
    Tensor quot = total.div(count);
    return quot.masked_fill(
        zero, Scalar(std::numeric_limits<double>::quiet_NaN()))
        .to(acc_dt != DType::Undefined
                ? acc_dt
                : (isComplexType(self.dtype()) ? self.dtype() : total.dtype()));
}

TENSORPLAY_LIBRARY_IMPL(CPU, ReduceKernels) {
    m.impl("amax", amax_cpu);
    m.impl("amin", amin_cpu);
    m.impl("aminmax", aminmax_cpu);
    m.impl("_aminmax", aminmax_all_cpu);
    m.impl("_aminmax.dim", aminmax_dim_cpu);
    m.impl("logsumexp", logsumexp_cpu);
    m.impl("nansum", nansum_cpu);
    m.impl("nanmedian", nanmedian_cpu);
    m.impl("nanmedian.dim", nanmedian_dim_cpu);
    m.impl("nanmedian.dim_values", nanmedian_dim_values_cpu);
    m.impl("cummax", cummax_cpu);
    m.impl("cummin", cummin_cpu);
    m.impl("std_mean", std_mean_cpu);
    m.impl("var_mean", var_mean_cpu);
    m.impl("mode", mode_cpu);
    m.impl("kthvalue", kthvalue_cpu);
    m.impl("count_nonzero", count_nonzero_cpu);
    m.impl("dist", dist_cpu);
    m.impl("renorm", renorm_cpu);
    m.impl("nanmean", nanmean_cpu);
}
}  // namespace cpu
}  // namespace tensorplay
