#pragma once

// Shared reduction-engine glue: reduced-dimension bookkeeping, the typed
// launch wrapper and the accumulator selection every reduction family unit
// builds on.  Each family unit includes this header and instantiates only
// the operations it registers.

#include "ReduceKernels.cuh"
#include "CUDAReduce.cuh"
#include "TensorIterator.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace reduce_family {

using reduction::ArgOps;
using reduction::ArgPair;
using reduction::AllOps;
using reduction::AnyOps;
using reduction::AbsMaxOps;
using reduction::AbsMinOps;
using reduction::ExtremumOps;
using reduction::MeanOps;
using reduction::MinMaxOps;
using reduction::MinMaxPair;
using reduction::MinMaxPairOps;
using reduction::NanSumOps;
using reduction::NormOps;
using reduction::NormOneOps;
using reduction::NormTwoOps;
using reduction::NormZeroOps;
using reduction::ProdOps;
using reduction::SumOps;
using reduction::WelfordData;
using reduction::WelfordOps;
using reduction::default_accumulation_t;
using reduction::is_half_like_v;

struct ReductionSpec {
    std::vector<int64_t> dims;
    std::vector<bool> mask;
    int64_t reduced_numel = 1;
};

inline ReductionSpec make_reduction_spec(
        const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t ndim = self.dim();
    ReductionSpec spec;
    spec.mask.assign(static_cast<size_t>(ndim), false);

    if (dims.empty()) {
        spec.dims.reserve(static_cast<size_t>(ndim));
        for (int64_t dim = 0; dim < ndim; ++dim) {
            spec.dims.push_back(dim);
            spec.mask[static_cast<size_t>(dim)] = true;
        }
    } else {
        spec.dims.reserve(dims.size());
        for (int64_t dim : dims) {
            if (dim < 0) dim += ndim;
            if (dim < 0 || dim >= ndim) {
                TP_THROW(IndexError, "CUDA reduction: dimension out of range");
            }
            if (spec.mask[static_cast<size_t>(dim)]) {
                TP_THROW(RuntimeError, "CUDA reduction: duplicate dimension");
            }
            spec.mask[static_cast<size_t>(dim)] = true;
            spec.dims.push_back(dim);
        }
        std::sort(spec.dims.begin(), spec.dims.end());
    }

    spec.reduced_numel = 1;
    for (int64_t dim : spec.dims) spec.reduced_numel *= self.size(dim);
    return spec;
}

inline std::vector<int64_t> reduction_output_shape(
        const Tensor& self, const ReductionSpec& spec, bool keepdim) {
    std::vector<int64_t> shape;
    const auto input_shape = static_cast<std::vector<int64_t>>(self.shape());
    shape.reserve(input_shape.size());
    for (size_t dim = 0; dim < input_shape.size(); ++dim) {
        if (spec.mask[dim]) {
            if (keepdim) shape.push_back(1);
        } else {
            shape.push_back(input_shape[dim]);
        }
    }
    return shape;
}

inline Tensor reduction_view(
        const Tensor& result, const Tensor& input, const ReductionSpec& spec,
        bool keepdim) {
    if (input.dim() == 0) return result;

    const auto input_shape = static_cast<std::vector<int64_t>>(input.shape());
    const auto result_strides = static_cast<std::vector<int64_t>>(result.strides());
    std::vector<int64_t> strides(input_shape.size(), 0);
    size_t result_dim = 0;
    for (size_t dim = 0; dim < input_shape.size(); ++dim) {
        if (!spec.mask[dim]) {
            // keepdim retains the reduced dimensions in the result shape;
            // without it, result strides are packed over non-reduced dims.
            strides[dim] = keepdim ? result_strides[dim] : result_strides[result_dim++];
        }
    }
    // TensorIterator identifies reduced dimensions by a zero output stride.
    // This must also be materialized for keepdim=true: a regular size-1
    // tensor has a non-zero stride, which would make the iterator mistake the
    // reduction for an elementwise operation.
    return result.as_strided(input_shape, strides);
}

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread = reduction::kDefaultValuesPerThread,
          typename SecondOutputT = OutputT>
Tensor run_reduction_typed(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        DType output_dtype, Ops ops, AccT identity, Tensor* second_output = nullptr) {
    static_assert(reduction::kReductionEngineRevision == 15);
    Tensor result = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), output_dtype, input.device());
    if (input.numel() == 0 || result.numel() == 0) return result;

    Tensor viewed = reduction_view(result, input, spec, keepdim);
    TensorIterator iter = TensorIterator::reduce_op(viewed, input);
    SecondOutputT* second_ptr =
        second_output == nullptr ? nullptr : second_output->data_ptr<SecondOutputT>();
    reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread,
                             reduction::input_vec_width_v<InputT>, SecondOutputT>(
        iter, ops, identity, second_ptr);
    return result;
}

template <typename T>
using same_dtype_acc_t = std::conditional_t<
    is_half_like_v<T>, float, default_accumulation_t<T>>;

template <typename T, bool MaxMode>
Tensor minmax_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    // Empty inputs are legal here: callers (max_kernel/min_kernel for full
    // reductions, max_dim_kernel/min_dim_kernel for dim reductions) already
    // reduction over non-empty dims of a zero-element tensor, which yields
    // an empty result (run_reduction_typed returns it untouched).
    using AccT = same_dtype_acc_t<T>;
    using Ops = MinMaxOps<T, AccT, T, MaxMode>;
    const AccT identity = MaxMode
        ? reduction::reduction_lower_bound<AccT>()
        : reduction::reduction_upper_bound<AccT>();
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, input.dtype(), Ops{}, identity);
}

template <typename T>
Tensor max_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    return minmax_same_dtype<T, true>(input, spec, keepdim);
}

template <typename T>
Tensor min_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    return minmax_same_dtype<T, false>(input, spec, keepdim);
}

// max(dim)/min(dim): the value and its index come out of one pass.  The
// float family with an int32-fitting logical extent runs the packed-u64
// form (one shuffle chain carries both outputs); every other dtype runs a
// pair accumulator through the same single pass with identical NaN/tie rules.
template <typename T, bool kIsMax>
std::tuple<Tensor, Tensor> extremum_with_indices_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    using ValueT = same_dtype_acc_t<T>;
    Tensor indices = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), DType::Int64,
        input.device());
    if constexpr (std::is_same_v<ValueT, float>) {
        if (spec.reduced_numel <= ((int64_t{1} << 31) - 1)) {
            Tensor values = run_reduction_typed<T, unsigned long long, T,
                reduction::PackedExtremumOps<T, kIsMax>,
                reduction::kDefaultValuesPerThread, int64_t>(
                input, spec, keepdim, input.dtype(),
                reduction::PackedExtremumOps<T, kIsMax>{},
                kIsMax ? 0ull : 0xFFFFFFFFFFFFFFFFull, &indices);
            return {values, indices};
        }
    }
    using StateT = reduction::ArgPair<ValueT>;
    using Ops = reduction::ExtremumOps<T, ValueT, T, kIsMax>;
    const ValueT identity = kIsMax
        ? reduction::reduction_lower_bound<ValueT>()
        : reduction::reduction_upper_bound<ValueT>();
    Tensor values = run_reduction_typed<T, StateT, T, Ops,
        reduction::kDefaultValuesPerThread, int64_t>(
        input, spec, keepdim, input.dtype(), Ops{}, StateT{identity, 0}, &indices);
    return {values, indices};
}

// Dtype-dispatch-friendly entry points (the macro fixes only <T>).
template <typename T>
std::tuple<Tensor, Tensor> max_with_indices_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    return extremum_with_indices_same_dtype<T, true>(input, spec, keepdim);
}

template <typename T>
std::tuple<Tensor, Tensor> min_with_indices_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    return extremum_with_indices_same_dtype<T, false>(input, spec, keepdim);
}

#define TP_DISPATCH_REDUCTION(FN, DTYPE, ...) \
    switch (DTYPE) { \
        case DType::UInt8: return FN<uint8_t>(__VA_ARGS__); \
        case DType::Int8: return FN<int8_t>(__VA_ARGS__); \
        case DType::Int16: return FN<int16_t>(__VA_ARGS__); \
        case DType::Int32: return FN<int32_t>(__VA_ARGS__); \
        case DType::Int64: return FN<int64_t>(__VA_ARGS__); \
        case DType::UInt16: return FN<uint16_t>(__VA_ARGS__); \
        case DType::UInt32: return FN<uint32_t>(__VA_ARGS__); \
        case DType::UInt64: return FN<uint64_t>(__VA_ARGS__); \
        case DType::Float32: return FN<float>(__VA_ARGS__); \
        case DType::Float64: return FN<double>(__VA_ARGS__); \
        case DType::Float16: return FN<Half>(__VA_ARGS__); \
        case DType::BFloat16: return FN<BFloat16>(__VA_ARGS__); \
        case DType::Bool: return FN<bool>(__VA_ARGS__); \
        default: TP_THROW(NotImplementedError, "CUDA reduction: unsupported dtype"); \
    }

// the entire bool instantiation tree (major ptxas time sink).
#define TP_DISPATCH_REDUCTION_NO_BOOL(FN, DTYPE, ...) \
    switch (DTYPE) { \
        case DType::UInt8: return FN<uint8_t>(__VA_ARGS__); \
        case DType::Int8: return FN<int8_t>(__VA_ARGS__); \
        case DType::Int16: return FN<int16_t>(__VA_ARGS__); \
        case DType::Int32: return FN<int32_t>(__VA_ARGS__); \
        case DType::Int64: return FN<int64_t>(__VA_ARGS__); \
        case DType::UInt16: return FN<uint16_t>(__VA_ARGS__); \
        case DType::UInt32: return FN<uint32_t>(__VA_ARGS__); \
        case DType::UInt64: return FN<uint64_t>(__VA_ARGS__); \
        case DType::Float32: return FN<float>(__VA_ARGS__); \
        case DType::Float64: return FN<double>(__VA_ARGS__); \
        case DType::Float16: return FN<Half>(__VA_ARGS__); \
        case DType::BFloat16: return FN<BFloat16>(__VA_ARGS__); \
        case DType::Bool: TP_THROW(NotImplementedError, "argmax/argmin not implemented for Bool"); \
        default: TP_THROW(NotImplementedError, "CUDA reduction: unsupported dtype"); \
    }

#define TP_DISPATCH_FLOAT_REDUCTION(FN, DTYPE, ...) \
    switch (DTYPE) { \
        case DType::Float32: return FN<float>(__VA_ARGS__); \
        case DType::Float64: return FN<double>(__VA_ARGS__); \
        case DType::Float16: return FN<Half>(__VA_ARGS__); \
        case DType::BFloat16: return FN<BFloat16>(__VA_ARGS__); \
        default: TP_THROW(NotImplementedError, "CUDA reduction: floating dtype required"); \
    }

} // namespace reduce_family
} // namespace cuda
} // namespace tensorplay
