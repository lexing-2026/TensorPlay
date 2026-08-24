#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Parallel.h"
#include "ReductionKernels.h"
#include "TensorIterator.h"
#include "cpu/Reduce.h"
#include "cpu/vec/vec.h"
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
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
             TP_THROW(RuntimeError, "Dimension out of range");
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

// Port of torch's review_reduce_result: as_strided the output to the input's
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

// TensorIterator-based reduction kernel: adds one input element into the
// output elementwise (out = out + in), vectorized over 4 accumulators.
// Accumulation happens in the output dtype, matching torch's CPU semantics
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

    // ATen alignment: Half/BFloat16 sums accumulate in float32 (acc type)
    DType acc_dtype = isReducedFloatingType(out_dtype) ? DType::Float32 : out_dtype;

    Tensor out = Tensor::zeros({}, acc_dtype, self.device());

    // torch's make_reduction: pre-cast the input to the output dtype so the
    // iterator's common dtype matches out_dtype.
    Tensor input = self;
    if (self.dtype() != acc_dtype) {
        input = self.to(acc_dtype);
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
    // ATen alignment: accumulate reduced floats in float32
    DType acc_dtype = isReducedFloatingType(out_dtype) ? DType::Float32 : out_dtype;
    Tensor out = Tensor::zeros(out_shape, acc_dtype, self.device());
    
    // torch's make_reduction: pre-cast the input to the output dtype.
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

Tensor max_kernel_impl(const Tensor& self) {
    Tensor out = Tensor::zeros({}, self.dtype(), self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype max_val = get_lowest<ctype>(); \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        if (n == 0) TP_THROW(RuntimeError, "max(): Expected reduction dim to be non-empty"); \
        for(int64_t i=0; i<n; ++i) { \
            if (data[i] > max_val) max_val = data[i]; \
        } \
        out.fill_(to_scalar(max_val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "max not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

std::tuple<Tensor, Tensor> max_dim_kernel_impl(const Tensor& self, int64_t dim0, bool keepdim) {
    // torch.max(input, dim) -> (values, indices); strict compare keeps the
    // FIRST maximal index, matching ATen's argmax pairing.
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "max(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(RuntimeError, "max(): Expected reduction dim ", dim, " to have non-zero size");
    }

    Tensor sc = self.is_contiguous() ? self : self.clone();
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
    Tensor out = Tensor::zeros({}, self.dtype(), self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype min_val = get_highest<ctype>(); \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        if (n == 0) TP_THROW(RuntimeError, "min(): Expected reduction dim to be non-empty"); \
        for(int64_t i=0; i<n; ++i) { \
            if (data[i] < min_val) min_val = data[i]; \
        } \
        out.fill_(to_scalar(min_val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "min not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

std::tuple<Tensor, Tensor> min_dim_kernel_impl(const Tensor& self, int64_t dim0, bool keepdim) {
    // torch.min(input, dim) -> (values, indices); strict compare keeps the
    // FIRST minimal index.
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "min(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(RuntimeError, "min(): Expected reduction dim ", dim, " to have non-zero size");
    }

    Tensor sc = self.is_contiguous() ? self : self.clone();
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
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    if (self_contig.dtype() != out_dtype) {
        self_contig = self_contig.to(out_dtype);
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype prod_val = 1; \
        ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) Accumulator<ctype>::mul(prod_val, data[i]); \
        out.fill_(to_scalar(prod_val)); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
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
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "prod_dim not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}



// All/Any
Tensor all_kernel_impl(const Tensor& self) {
    Tensor out = Tensor::zeros({}, DType::Bool, self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
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
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
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
        // Flatten
        Tensor self_contig = self.is_contiguous() ? self : self.clone();
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
    
    // Transpose d to end, reshape to (-1, size), find max idx per row
    Tensor t = self.transpose(d, -1);
    t = t.is_contiguous() ? t : t.clone(); // Force copy/compact
    
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
        // Flatten
        Tensor self_contig = self.is_contiguous() ? self : self.clone();
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
    
    // Transpose d to end, reshape to (-1, size), find min idx per row
    Tensor t = self.transpose(d, -1);
    t = t.is_contiguous() ? t : t.clone(); 
    
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
    Tensor t = self.clone().view({-1});
    int64_t n = t.numel();
    if (n == 0) return Tensor::tensor({std::numeric_limits<float>::quiet_NaN()}, DType::Float32, t.device());

    // nth_element finds the n-th smallest element.
    // For even n, PyTorch returns the smaller of the two middle elements.
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

} // namespace cpu
} // namespace tensorplay
