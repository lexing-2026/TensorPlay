#include <iostream>
#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include "CUDAReduce.cuh"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include "TensorIterator.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>
#include <numeric>
#include <type_traits>

#ifdef USE_CUDNN
#include <cudnn.h>
#include "Atomic.cuh"
#endif

namespace tensorplay {
namespace cuda {

// --- Utils ---
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// Mean over complex dtypes: scale the accumulated sum by 1/count in place
// (grid-stride; src and dst may alias).
template <typename T>
__global__ void scale_complex_kernel(
        int64_t n, const tensorplay::complex<T>* src,
        tensorplay::complex<T> scale, tensorplay::complex<T>* dst) {
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = src[i] * scale;
}

// --- cuDNN Reduction Helper ---

#ifdef USE_CUDNN

Tensor cudnn_reduce_wrapper(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim, 
                           cudnnReduceTensorOp_t op, DType out_dtype) {
    // Handle empty dims -> reduce all
    std::vector<int64_t> actual_dims = dims;
    if (dims.empty()) {
        for(int i=0; i<self.dim(); ++i) actual_dims.push_back(i);
    }

    Tensor self_contig = self.contiguous();
    
    // Determine output shape
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(self.shape());
    for (auto d : actual_dims) {
        int64_t dd = d < 0 ? d + self.dim() : d;
        out_shape[dd] = 1;
    }
    
    Tensor result = Tensor::empty(out_shape, out_dtype, self.device());
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnReduceTensorDescriptor_t reduceDesc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduceDesc));
    
    cudnnDataType_t compType = (self.dtype() == DType::Float64) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    
    // indices type: NO_INDICES
    cudnnStatus_t status = cudnnSetReduceTensorDescriptor(reduceDesc, 
        op, 
        compType, 
        CUDNN_PROPAGATE_NAN, 
        CUDNN_REDUCE_TENSOR_NO_INDICES, 
        CUDNN_32BIT_INDICES);
        
    if (status != CUDNN_STATUS_SUCCESS) {
        TP_THROW(RuntimeError, "cudnnSetReduceTensorDescriptor failed");
    }
    
    // std::cout << "Creating aDesc..." << std::endl;
    cudnnTensorDescriptor_t aDesc = createTensorDescriptor(self_contig, true);
    // std::cout << "Creating cDesc..." << std::endl;
    cudnnTensorDescriptor_t cDesc = createTensorDescriptor(result, true);
    
    double alpha_d = 1.0, beta_d = 0.0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    void *alpha, *beta;
    
    if (compType == CUDNN_DATA_DOUBLE) {
        alpha = &alpha_d; beta = &beta_d;
    } else {
        alpha = &alpha_f; beta = &beta_f;
    }
    
    size_t wsSize = 0;
    // std::cout << "Getting Workspace Size..." << std::endl;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(handle, reduceDesc, aDesc, cDesc, &wsSize));
    
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(wsSize, self.device());
    
    // std::cout << "Running cudnnReduceTensor..." << std::endl;
    CUDNN_CHECK(cudnnReduceTensor(handle, reduceDesc, 
        nullptr, 0, 
        workspace.get(), wsSize,
        alpha, aDesc, self_contig.data_ptr(), 
        beta, cDesc, result.data_ptr()));
        
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduceDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(aDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cDesc));
    
    if (!keepdim) {
        std::vector<int64_t> final_shape;
        for (int i=0; i<self.dim(); ++i) {
            bool is_reduced = false;
            for(auto d : actual_dims) if((d < 0 ? d + self.dim() : d) == i) is_reduced = true;
            if(!is_reduced) final_shape.push_back(self.shape()[i]);
        }
        return result.reshape(final_shape);
    }
    
    return result;
}

#endif

namespace {

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

ReductionSpec make_reduction_spec(const Tensor& self, const std::vector<int64_t>& dims) {
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

std::vector<int64_t> reduction_output_shape(
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

Tensor reduction_view(
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
    static_assert(reduction::kReductionEngineRevision == 14);
    Tensor result = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), output_dtype, input.device());
    if (input.numel() == 0 || result.numel() == 0) return result;

    Tensor viewed = reduction_view(result, input, spec, keepdim);
    TensorIterator iter = TensorIterator::reduce_op(viewed, input);
    const auto config = reduction::make_reduce_config<InputT, AccT, OutputT>(iter);
    SecondOutputT* second_ptr =
        second_output == nullptr ? nullptr : second_output->data_ptr<SecondOutputT>();
    if (config.input_vec_size == 8) {
        reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 8, SecondOutputT>(
            iter, ops, identity, second_ptr);
    } else if (config.input_vec_size == 4) {
        reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 4, SecondOutputT>(
            iter, ops, identity, second_ptr);
    } else {
        reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 1, SecondOutputT>(
            iter, ops, identity, second_ptr);
    }
    return result;
}

template <typename T>
using same_dtype_acc_t = std::conditional_t<
    is_half_like_v<T>, float, default_accumulation_t<T>>;

template <typename T, typename OutT = T>
Tensor sum_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, SumOps<T, AccT, OutT>{}, AccT(0));
}

template <typename T, typename OutT = T>
Tensor nansum_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, NanSumOps<T, AccT, OutT>{}, AccT(0));
}

template <typename T, typename OutT = T>
Tensor prod_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::ones(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, ProdOps<T, AccT, OutT>{}, AccT(1));
}

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

template <typename T>
Tensor all_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        return Tensor::ones(reduction_output_shape(input, spec, keepdim), DType::Bool, input.device());
    }
    return run_reduction_typed<T, int, bool>(
        input, spec, keepdim, DType::Bool, AllOps<T, int, bool>{}, 1);
}

template <typename T>
Tensor any_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), DType::Bool, input.device());
    }
    return run_reduction_typed<T, int, bool>(
        input, spec, keepdim, DType::Bool, AnyOps<T, int, bool>{}, 0);
}

template <typename T, typename OutT = T>
Tensor mean_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), dtype, input.device());
    }
    using FactorT = typename tensorplay::scalar_value_type<AccT>::type;
    const FactorT factor = FactorT(1) / static_cast<FactorT>(spec.reduced_numel);
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, MeanOps<T, AccT, OutT>{factor}, AccT(0));
}

template <typename T>
Tensor norm_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, double p) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        if (spec.reduced_numel == 0 &&
            (p < 0.0 || p == std::numeric_limits<double>::infinity())) {
            TP_THROW(RuntimeError,
                     "norm cannot reduce an empty dimension for this order");
        }
        return Tensor::zeros(
            reduction_output_shape(input, spec, keepdim), input.dtype(), input.device());
    }
    if (p == 0.0) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), NormZeroOps<T, AccT, T>{}, AccT(0));
    }
    if (p == 1.0) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), NormOneOps<T, AccT, T>{}, AccT(0));
    }
    if (p == 2.0) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), NormTwoOps<AccT, T>{}, AccT(0));
    }
    if (p == std::numeric_limits<double>::infinity()) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), AbsMaxOps<T, AccT, T>{}, AccT(0));
    }
    if (p == -std::numeric_limits<double>::infinity()) {
        return run_reduction_typed<T, AccT, T>(
            input, spec, keepdim, input.dtype(), AbsMinOps<T, AccT, T>{},
            std::numeric_limits<AccT>::infinity());
    }
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, input.dtype(), NormOps<T, AccT, T>{static_cast<AccT>(p)},
        AccT(0));
}

template <typename T, typename IndexT>
Tensor welford_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    using AccT = same_dtype_acc_t<T>;
    using StateT = WelfordData<AccT, IndexT>;
    using Ops = WelfordOps<AccT, T, IndexT>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), input.dtype(), input.device());
    }
    return run_reduction_typed<T, StateT, T, Ops, 2>(
        input, spec, keepdim, input.dtype(),
        Ops{static_cast<AccT>(correction), take_sqrt}, StateT{AccT(0), AccT(0), 0, AccT(0)});
}

// var and mean out of one Welford pass: the accumulator already carries both,
// so the mean rides along instead of re-reading the whole input.
template <typename T, typename IndexT>
std::tuple<Tensor, Tensor> welford_var_mean_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction) {
    using AccT = same_dtype_acc_t<T>;
    using StateT = WelfordData<AccT, IndexT>;
    using Ops = WelfordOps<AccT, T, IndexT>;
    const auto shape = reduction_output_shape(input, spec, keepdim);
    if (input.numel() == 0) {
        const Scalar nan(Scalar(std::numeric_limits<float>::quiet_NaN()));
        return {Tensor::full(shape, nan, input.dtype(), input.device()),
                Tensor::full(shape, nan, input.dtype(), input.device())};
    }
    Tensor mean = Tensor::empty(shape, input.dtype(), input.device());
    Tensor var = run_reduction_typed<T, StateT, T>(
        input, spec, keepdim, input.dtype(),
        Ops{static_cast<AccT>(correction), false},
        StateT{AccT(0), AccT(0), 0, AccT(0)}, &mean);
    return {var, mean};
}

// The dtype dispatch macro fixes the first template argument, so the count
// width is chosen here.
template <typename T>
std::tuple<Tensor, Tensor> welford_var_mean_dispatch(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction) {
    if (spec.reduced_numel <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return welford_var_mean_same_dtype<T, int32_t>(
            input, spec, keepdim, correction);
    }
    return welford_var_mean_same_dtype<T, int64_t>(
        input, spec, keepdim, correction);
}

template <typename T>
Tensor argmax_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    // here kills the whole ArgPair<int> instantiation tree.
    static_assert(!std::is_same_v<T, bool>, "argmax is not implemented for bool");
    // Zero-element inputs with a non-empty reduction dim produce empty
    using ValueT = same_dtype_acc_t<T>;
    // Warp-shuffle fast path: float-family reductions whose logical index
    // fits int32 run the packed-u64 max form (identical winners — value
    // desc / first occurrence); everything else keeps the ArgPair tree.
    if constexpr (std::is_same_v<ValueT, float>) {
        if (spec.reduced_numel <= ((int64_t{1} << 31) - 1)) {
            return run_reduction_typed<T, unsigned long long, int64_t>(
                input, spec, keepdim, DType::Int64,
                reduction::PackedArgMaxOps{}, 0ull);
        }
    }
    using StateT = ArgPair<ValueT>;
    using Ops = ArgOps<ValueT, true>;
    return run_reduction_typed<T, StateT, int64_t>(
        input, spec, keepdim, DType::Int64, Ops{},
        StateT{reduction::reduction_lower_bound<ValueT>(), 0});
}

template <typename T>
Tensor argmin_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    static_assert(!std::is_same_v<T, bool>, "argmin is not implemented for bool");
    using ValueT = same_dtype_acc_t<T>;
    // Same packed-u64 fast path as argmax, with the key inverted for the
    // min direction (identical winners — value asc / first occurrence).
    if constexpr (std::is_same_v<ValueT, float>) {
        if (spec.reduced_numel <= ((int64_t{1} << 31) - 1)) {
            return run_reduction_typed<T, unsigned long long, int64_t>(
                input, spec, keepdim, DType::Int64,
                reduction::PackedArgMinOps{}, ~0ull);
        }
    }
    using StateT = ArgPair<ValueT>;
    using Ops = ArgOps<ValueT, false>;
    return run_reduction_typed<T, StateT, int64_t>(
        input, spec, keepdim, DType::Int64, Ops{},
        StateT{reduction::reduction_upper_bound<ValueT>(), 0});
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

// aminmax: both extreme values out of one pass.  The float family with an
// int32-fitting logical extent runs the packed two-keys-per-word form (one
// shuffle chain carries both outputs); every other dtype carries a
// MinMaxPair accumulator through the same single read.
template <typename T>
std::tuple<Tensor, Tensor> aminmax_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    using ValueT = same_dtype_acc_t<T>;
    if constexpr (std::is_same_v<ValueT, float>) {
        if (spec.reduced_numel <= ((int64_t{1} << 31) - 1)) {
            Tensor min_values = Tensor::empty(
                reduction_output_shape(input, spec, keepdim), input.dtype(),
                input.device());
            Tensor max_values = run_reduction_typed<T, unsigned long long, T>(
                input, spec, keepdim, input.dtype(),
                reduction::PackedAminMaxOps<T>{}, 0ull, &min_values);
            return {min_values, max_values};
        }
    }
    Tensor min_values = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), input.dtype(),
        input.device());
    using StateT = reduction::MinMaxPair<ValueT>;
    using Ops = reduction::MinMaxPairOps<T, ValueT, T>;
    Tensor max_values = run_reduction_typed<T, StateT, T>(
        input, spec, keepdim, input.dtype(), Ops{},
        StateT{reduction::reduction_upper_bound<ValueT>(),
               reduction::reduction_lower_bound<ValueT>()},
        &min_values);
    return {min_values, max_values};
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

} // namespace

// --- Implementations ---

// Sum
Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    // Cast-and-reduce in a single kernel for the promoted dtype pairs: the
    // generic engine reads the narrow input and accumulates in the wider
    // type, so no materialized copy is needed.
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return sum_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return sum_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
        // Accumulate in complex<float> without promoting the input first.
        if (out_dtype == DType::ComplexFloat) {
            return sum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                self, spec, keepdim, out_dtype);
        }
        return sum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
            self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    // Complex accumulates in its own width via the generic ops (+ only).
    if (input.dtype() == DType::ComplexFloat) {
        return sum_same_dtype<tensorplay::complex<float>>(input, input_spec, keepdim, out_dtype);
    }
    if (input.dtype() == DType::ComplexDouble) {
        return sum_same_dtype<tensorplay::complex<double>>(input, input_spec, keepdim, out_dtype);
    }
    TP_DISPATCH_REDUCTION(sum_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return sum_dim_kernel(self, {}, false, dtype);
}

Tensor amax_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(max_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor amin_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(min_same_dtype, self.dtype(), self, spec, keepdim);
}

std::tuple<Tensor, Tensor> aminmax_dim_kernel(const Tensor& self,
                                              const std::vector<int64_t>& dim,
                                              bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(aminmax_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor nansum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                         bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isFloatingOrComplexType(self.dtype()) ? self.dtype() : DType::Int64;
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return nansum_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return nansum_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
        if (out_dtype == DType::ComplexFloat) {
            return nansum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                self, spec, keepdim, out_dtype);
        }
        return nansum_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
            self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    if (input.dtype() == DType::ComplexFloat) {
        return nansum_same_dtype<tensorplay::complex<float>>(
            input, input_spec, keepdim, out_dtype);
    }
    if (input.dtype() == DType::ComplexDouble) {
        return nansum_same_dtype<tensorplay::complex<double>>(
            input, input_spec, keepdim, out_dtype);
    }
    TP_DISPATCH_REDUCTION(nansum_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

// Mean
Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isFloatingOrComplexType(self.dtype()) ? self.dtype() : DType::Float32;
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return mean_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return mean_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (isComplexType(self.dtype())) {
        // Single-pass complex mean: MeanOps accumulates the complex sum and
        // scales it by a real factor (1/count) on projection.
        if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
            if (out_dtype == DType::ComplexFloat) {
                return mean_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                    self, spec, keepdim, out_dtype);
            }
            return mean_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
                self, spec, keepdim, out_dtype);
        }
        if (self.dtype() == DType::ComplexFloat) {
            return mean_same_dtype<tensorplay::complex<float>>(self, spec, keepdim, out_dtype);
        }
        return mean_same_dtype<tensorplay::complex<double>>(self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    TP_DISPATCH_FLOAT_REDUCTION(mean_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

Tensor mean_dim_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self,
                                     const std::vector<int64_t>& dims, bool keepdim) {
    std::vector<int64_t> normalized;
    normalized.reserve(dims.size());
    for (int64_t d : dims) {
        if (d < 0) d += self.dim();
        if (d < 0 || d >= self.dim()) {
            TP_THROW(IndexError, "mean.dim backward: dimension out of range");
        }
        normalized.push_back(d);
    }
    std::sort(normalized.begin(), normalized.end());
    if (std::adjacent_find(normalized.begin(), normalized.end()) != normalized.end()) {
        TP_THROW(RuntimeError, "mean.dim backward: duplicate dimensions");
    }
    Tensor expanded = grad_output;
    if (!keepdim) {
        for (int64_t d : normalized) expanded = expanded.unsqueeze(d);
    }
    int64_t count = 1;
    for (int64_t d : normalized) count *= self.size(d);
    Tensor grad = expanded.expand(static_cast<std::vector<int64_t>>(self.shape())) /
                  Scalar(static_cast<float>(count));
    return grad.dtype() == self.dtype() ? grad : grad.to(self.dtype());
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    return mean_dim_kernel(self, {}, false, dtype);
}

Tensor sum_dim_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self,
                                    const std::vector<int64_t>& dims, bool keepdim) {
    std::vector<int64_t> normalized;
    normalized.reserve(dims.size());
    for (int64_t d : dims) {
        if (d < 0) d += self.dim();
        if (d < 0 || d >= self.dim()) {
            TP_THROW(IndexError, "sum.dim backward: dimension out of range");
        }
        normalized.push_back(d);
    }
    std::sort(normalized.begin(), normalized.end());
    Tensor expanded = grad_output;
    if (!keepdim) {
        // Insert unit dims at the ORIGINAL (ascending) positions: after the
        // reduction the surviving dims are left-packed, so ascending order
        // restores the exact source layout (reverse order misaligns).
        for (int64_t d : normalized) {
            expanded = expanded.unsqueeze(d);
        }
    }
    return expanded.expand(static_cast<std::vector<int64_t>>(self.shape()));
}

// Prod
Tensor prod_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return prod_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return prod_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
        if (out_dtype == DType::ComplexFloat) {
            return prod_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                self, spec, keepdim, out_dtype);
        }
        return prod_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
            self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    if (input.dtype() == DType::ComplexFloat) {
        return prod_same_dtype<tensorplay::complex<float>>(input, input_spec, keepdim, out_dtype);
    }
    if (input.dtype() == DType::ComplexDouble) {
        return prod_same_dtype<tensorplay::complex<double>>(input, input_spec, keepdim, out_dtype);
    }
    TP_DISPATCH_REDUCTION(prod_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return prod_dim_kernel(self, {}, false, dtype);
}

// Max
std::tuple<Tensor, Tensor> max_dim_kernel(const Tensor& self, int64_t dim0, bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "max(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(IndexError, "max(): Expected reduction dim ", dim, " to have non-zero size.");
    }
    const ReductionSpec spec = make_reduction_spec(self, {dim});
    // The dispatch macros expand to a returning switch; route them through an
    // immediately-invoked lambda to capture both outputs.
    return [&]() -> std::tuple<Tensor, Tensor> {
        TP_DISPATCH_REDUCTION(
            max_with_indices_same_dtype, self.dtype(), self, spec, keepdim);
        TP_THROW(NotImplementedError, "max(dim): unsupported dtype");
    }();
}

Tensor max_kernel(const Tensor& self) {
    if (self.numel() == 0) {
        TP_THROW(RuntimeError, "max(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    auto spec = make_reduction_spec(self, {});
    Tensor values = [&]() -> Tensor {
        TP_DISPATCH_REDUCTION(max_same_dtype, self.dtype(), self, spec, false);
        return Tensor();
    }();
    return values;
}

// Min
std::tuple<Tensor, Tensor> min_dim_kernel(const Tensor& self, int64_t dim0, bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "min(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(IndexError, "min(): Expected reduction dim ", dim, " to have non-zero size.");
    }
    const ReductionSpec spec = make_reduction_spec(self, {dim});
    return [&]() -> std::tuple<Tensor, Tensor> {
        TP_DISPATCH_REDUCTION(
            min_with_indices_same_dtype, self.dtype(), self, spec, keepdim);
        TP_THROW(NotImplementedError, "min(dim): unsupported dtype");
    }();
}

Tensor min_kernel(const Tensor& self) {
    if (self.numel() == 0) {
        TP_THROW(RuntimeError, "min(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    auto spec = make_reduction_spec(self, {});
    Tensor values = [&]() -> Tensor {
        TP_DISPATCH_REDUCTION(min_same_dtype, self.dtype(), self, spec, false);
        return Tensor();
    }();
    return values;
}

// Norm (L2)
//
// The generic reduction engine is deliberately flexible, but its global
// reduction configuration can split a single scalar across many CTAs and
// then allocate/finalize a partial buffer.  That overhead is visible in the
// hot Muon path, which normalizes one matrix per optimizer step.  Keep a
// native norm kernel: one coalesced grid pass and one small final reduction.
template <typename InputT, typename AccT>
__global__ void norm2_partial_kernel(
        int64_t n, const InputT* input, AccT* partials) {
    AccT value = AccT(0);
    const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = first; i < n; i += stride) {
        const AccT v = static_cast<AccT>(input[i]);
        value += v * v;
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) partials[blockIdx.x] = value;
    }
}

template <typename AccT, typename OutputT>
__global__ void norm2_finalize_kernel(
        int64_t count, const AccT* partials, OutputT* output) {
    AccT value = AccT(0);
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        value += partials[i];
    }
    __shared__ AccT warp_values[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) {
            if constexpr (std::is_same_v<AccT, float>) {
                output[0] = static_cast<OutputT>(sqrtf(value));
            } else {
                output[0] = static_cast<OutputT>(::sqrt(value));
            }
        }
    }
}

template <typename InputT, typename AccT, typename OutputT>
__global__ void norm2_single_block_kernel(
        int64_t n, const InputT* input, OutputT* output) {
    AccT value = AccT(0);
    for (int64_t base = static_cast<int64_t>(threadIdx.x) * 4;
         base < n;
         base += static_cast<int64_t>(blockDim.x) * 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t i = base + j;
            if (i < n) {
                const AccT v = static_cast<AccT>(input[i]);
                value += v * v;
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) {
            if constexpr (std::is_same_v<AccT, float>) {
                output[0] = static_cast<OutputT>(sqrtf(value));
            } else {
                output[0] = static_cast<OutputT>(::sqrt(value));
            }
        }
    }
}

template <typename InputT, typename AccT>
__global__ void norm2_atomic_kernel(
        int64_t n, const InputT* input, AccT* accumulator) {
    AccT value = AccT(0);
    for (int64_t base =
             static_cast<int64_t>(blockIdx.x) * blockDim.x * 4 +
             static_cast<int64_t>(threadIdx.x) * 4;
         base < n;
         base += static_cast<int64_t>(gridDim.x) * blockDim.x * 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t i = base + j;
            if (i < n) {
                const AccT v = static_cast<AccT>(input[i]);
                value += v * v;
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ AccT warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : AccT(0);
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) atomicAdd(accumulator, value);
    }
}

template <typename AccT, typename OutputT>
__global__ void norm2_finalize_scalar_kernel(
        const AccT* accumulator, OutputT* output) {
    if (threadIdx.x == 0) {
        const AccT value = *accumulator;
        if constexpr (std::is_same_v<AccT, float>) {
            output[0] = static_cast<OutputT>(sqrtf(value));
        } else {
            output[0] = static_cast<OutputT>(::sqrt(value));
        }
    }
}

template <typename InputT, typename AccT>
Tensor norm2_global_fast_typed(const Tensor& self) {
    Tensor result = Tensor::empty({}, self.dtype(), self.device());
    const int64_t n = self.numel();
    if (n == 0) {
        result.zero_();
        return result;
    }

    constexpr int block_size = 256;
    const auto stream = getCurrentCUDAStream().stream();

    // Muon commonly normalizes matrices up to a few million elements.  A
    // single coalesced block avoids both the temporary partial allocation and
    // the second launch while retaining a grid-stride loop over the complete
    // tensor.  Use the two-stage path for very large reductions so bandwidth
    // remains the priority there.
    constexpr int64_t single_block_limit = 64 * 1024;
    if (n <= single_block_limit) {
        const int single_block_threads = n > 64 * 1024 ? 512 : block_size;
        norm2_single_block_kernel<InputT, AccT, InputT><<<1, single_block_threads, 0, stream>>>(
            n, static_cast<const InputT*>(self.data_ptr()),
            static_cast<InputT*>(result.data_ptr()));
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    constexpr int elements_per_thread = 8;
    const int64_t needed =
        (n + static_cast<int64_t>(block_size * elements_per_thread) - 1) /
        static_cast<int64_t>(block_size * elements_per_thread);
    // Keep the launch geometry out of the hot host path.  Querying
    // cudaGetDeviceProperties for every Muon step costs roughly 0.9 ms on
    // the target GPU, dwarfing the reduction itself.  256 CTAs is a safe
    // upper bound for this one-output reduction across the supported CUDA
    // devices; the grid-stride loop handles smaller tensors naturally.
    constexpr int64_t target = 256;
    const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, target)));

    const DType accumulator_dtype =
        std::is_same_v<AccT, float> ? DType::Float32 : DType::Float64;
    Tensor accumulator = Tensor::empty({}, accumulator_dtype, self.device());
    CUDA_CHECK(cudaMemsetAsync(accumulator.data_ptr(), 0, sizeof(AccT), stream));
    norm2_atomic_kernel<InputT, AccT><<<blocks, block_size, 0, stream>>>(
        n, static_cast<const InputT*>(self.data_ptr()),
        static_cast<AccT*>(accumulator.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
    norm2_finalize_scalar_kernel<AccT, InputT><<<1, 32, 0, stream>>>(
        static_cast<const AccT*>(accumulator.data_ptr()),
        static_cast<InputT*>(result.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor norm_global_kernel(const Tensor& self, double p) {
    if (isComplexType(self.dtype())) {
        Tensor areal = self.abs();
        return norm_global_kernel(areal, p);
    }
    // A transpose of a contiguous 2-D tensor is logically non-contiguous but
    // still occupies one dense storage span.  Since a global norm is
    // independent of element order, scan that span directly instead of
    // falling through to the generic strided reduction (or making a copy).
    const bool dense_storage =
        self.is_contiguous() ||
        (self.dim() == 2 && self.stride(0) == 1 &&
         self.stride(1) == self.size(0));
    if (p == 2.0 && dense_storage) {
        switch (self.dtype()) {
            case DType::Float32:
                return norm2_global_fast_typed<float, float>(self);
            case DType::Float64:
                return norm2_global_fast_typed<double, double>(self);
            case DType::Float16:
                return norm2_global_fast_typed<Half, float>(self);
            case DType::BFloat16:
                return norm2_global_fast_typed<BFloat16, float>(self);
            default:
                break;
        }
    }
    const ReductionSpec spec = make_reduction_spec(self, {});
    TP_DISPATCH_FLOAT_REDUCTION(norm_same_dtype, self.dtype(), self, spec, false, p);
}

Tensor norm_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, double p, bool keepdim) {
    if (isComplexType(self.dtype())) {
        Tensor areal = self.abs();
        return norm_dim_kernel(areal, dim, p, keepdim);
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(norm_same_dtype, self.dtype(), self, spec, keepdim, p);
}

// All / Any
Tensor all_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(all_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor all_dim_int_kernel(const Tensor& self, int64_t dim, bool keepdim) {
    return all_dim_kernel(self, std::vector<int64_t>{dim}, keepdim);
}

Tensor all_kernel(const Tensor& self) {
    return all_dim_kernel(self, {}, false);
}

Tensor any_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(any_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor any_dim_int_kernel(const Tensor& self, int64_t dim, bool keepdim) {
    return any_dim_kernel(self, std::vector<int64_t>{dim}, keepdim);
}

Tensor any_kernel(const Tensor& self) {
    return any_dim_kernel(self, {}, false);
}

// Var / Std
// 32-bit counts whenever the reduction extent allows: the count rides along
// every shuffle and staging step, and a narrower count shrinks both.
template <typename T>
Tensor welford_dispatch(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    if (spec.reduced_numel <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return welford_same_dtype<T, int32_t>(input, spec, keepdim, correction, take_sqrt);
    }
    return welford_same_dtype<T, int64_t>(input, spec, keepdim, correction, take_sqrt);
}

Tensor var_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        Tensor real = tensorplay::tpx::ops::real(self);
        Tensor imag = tensorplay::tpx::ops::imag(self);
        return var_dim_kernel(real, dim, correction, keepdim) +
               var_dim_kernel(imag, dim, correction, keepdim);
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_dispatch, self.dtype(), self, spec,
                                keepdim, correction, false);
}

std::tuple<Tensor, Tensor> var_mean_dim_kernel(
        const Tensor& self, const std::vector<int64_t>& dim,
        int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        // Complex keeps the two-pass form: the variance adds over the
        // components while the mean has to be rebuilt as a complex value.
        return {var_dim_kernel(self, dim, correction, keepdim),
                mean_dim_kernel(self, dim, keepdim, DType::Undefined)};
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_var_mean_dispatch, self.dtype(), self, spec,
                                keepdim, correction);
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
    return var_dim_kernel(self, {}, correction, false);
}

Tensor std_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        return var_dim_kernel(self, dim, correction, keepdim).sqrt();
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_dispatch, self.dtype(), self, spec,
                                keepdim, correction, true);
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
    if (isComplexType(self.dtype())) {
        return var_kernel(self, correction).sqrt();
    }
    return std_dim_kernel(self, {}, correction, false);
}

Tensor argmax_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        TP_CHECK_INDEX(self.numel() != 0,
                       "argmax(): Expected reduction dim to be specified for input.numel() == 0.");
    } else {
        const int64_t d = *dim < 0 ? *dim + self.dim() : *dim;
        TP_CHECK_INDEX(self.size(d) != 0,
                       "argmax(): Expected reduction dim ", d, " to have non-zero size.");
    }
    Tensor input = self;
    if (!dim.has_value() && !input.is_contiguous()) input = input.contiguous();
    const ReductionSpec spec = make_reduction_spec(
        input, dim.has_value() ? std::vector<int64_t>{*dim} : std::vector<int64_t>{});
    TP_DISPATCH_REDUCTION_NO_BOOL(argmax_same_dtype, input.dtype(), input, spec, keepdim);
}

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        TP_CHECK_INDEX(self.numel() != 0,
                       "argmin(): Expected reduction dim to be specified for input.numel() == 0.");
    } else {
        const int64_t d = *dim < 0 ? *dim + self.dim() : *dim;
        TP_CHECK_INDEX(self.size(d) != 0,
                       "argmin(): Expected reduction dim ", d, " to have non-zero size.");
    }
    Tensor input = self;
    if (!dim.has_value() && !input.is_contiguous()) input = input.contiguous();
    const ReductionSpec spec = make_reduction_spec(
        input, dim.has_value() ? std::vector<int64_t>{*dim} : std::vector<int64_t>{});
    TP_DISPATCH_REDUCTION_NO_BOOL(argmin_same_dtype, input.dtype(), input, spec, keepdim);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReductionKernels) {
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    m.impl("mean_dim_backward", mean_dim_backward_kernel_cuda);
    m.impl("_sum_dim_backward", sum_dim_backward_kernel_cuda);
    
    m.impl("prod", prod_kernel);
    m.impl("prod.dim_IntList", prod_dim_kernel);
    
    m.impl("max", max_kernel);
    m.impl("max.dim", max_dim_kernel);
    
    m.impl("min", min_kernel);
    m.impl("min.dim", min_dim_kernel);
    
    m.impl("norm", norm_global_kernel);
    m.impl("norm.dim", norm_dim_kernel);
    
    m.impl("all", all_kernel);
    m.impl("all.dim", all_dim_int_kernel);
    
    m.impl("any", any_kernel);
    m.impl("any.dim", any_dim_int_kernel);
    
    m.impl("var", var_kernel);
    m.impl("var.dim", var_dim_kernel);
    
    m.impl("std", std_kernel);
    m.impl("std.dim", std_dim_kernel);

    m.impl("argmax", argmax_kernel);
    m.impl("argmin", argmin_kernel);
}

} // namespace cuda
} // namespace tensorplay
