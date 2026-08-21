#include <iostream>
#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include "CUDAReduce.cuh"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include "TensorIterator.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>
#include <numeric>

#ifdef USE_CUDNN
#include <cudnn.h>
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
using reduction::MeanOps;
using reduction::MinMaxOps;
using reduction::NormTwoOps;
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
          int ValuesPerThread = reduction::kDefaultValuesPerThread>
Tensor run_reduction_typed(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        DType output_dtype, Ops ops, AccT identity) {
    static_assert(reduction::kReductionEngineRevision == 2);
    Tensor result = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), output_dtype, input.device());
    if (input.numel() == 0 || result.numel() == 0) return result;

    Tensor viewed = reduction_view(result, input, spec, keepdim);
    TensorIterator iter = TensorIterator::reduce_op(viewed, input);
    const auto config = reduction::make_reduce_config<InputT, AccT, OutputT>(iter);
    if (config.input_vec_size == 8) {
        reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 8>(
            iter, ops, identity);
    } else if (config.input_vec_size == 4) {
        reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 4>(
            iter, ops, identity);
    } else {
        reduction::launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 1>(
            iter, ops, identity);
    }
    return result;
}

template <typename T>
using same_dtype_acc_t = std::conditional_t<
    is_half_like_v<T>, float, default_accumulation_t<T>>;

template <typename T>
Tensor sum_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, dtype, SumOps<T, AccT, T>{}, AccT(0));
}

template <typename T>
Tensor prod_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::ones(reduction_output_shape(input, spec, keepdim), dtype, input.device());
    }
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, dtype, ProdOps<T, AccT, T>{}, AccT(1));
}

template <typename T, bool MaxMode>
Tensor minmax_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        TP_THROW(RuntimeError, MaxMode
            ? "max(): Expected reduction dim to be non-empty"
            : "min(): Expected reduction dim to be non-empty");
    }
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

template <typename T>
Tensor mean_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), dtype, input.device());
    }
    const AccT factor = AccT(1) / static_cast<AccT>(spec.reduced_numel);
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, dtype, MeanOps<T, AccT, T>{factor}, AccT(0));
}

template <typename T>
Tensor norm_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, double p) {
    if (p != 2.0) {
        TP_THROW(NotImplementedError, "norm: only p=2 supported on CUDA");
    }
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), input.dtype(), input.device());
    }
    return run_reduction_typed<T, AccT, T>(
        input, spec, keepdim, input.dtype(), NormTwoOps<AccT, T>{}, AccT(0));
}

template <typename T>
Tensor welford_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    using AccT = same_dtype_acc_t<T>;
    using StateT = WelfordData<AccT>;
    using Ops = WelfordOps<AccT, T>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), input.dtype(), input.device());
    }
    return run_reduction_typed<T, StateT, T, Ops, 2>(
        input, spec, keepdim, input.dtype(),
        Ops{static_cast<AccT>(correction), take_sqrt}, StateT{AccT(0), AccT(0), 0, AccT(0)});
}

template <typename T>
Tensor argmax_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        TP_THROW(RuntimeError, "argmax(): Expected reduction dim to be non-empty");
    }
    using ValueT = same_dtype_acc_t<T>;
    using StateT = ArgPair<ValueT>;
    using Ops = ArgOps<ValueT, true>;
    return run_reduction_typed<T, StateT, int64_t>(
        input, spec, keepdim, DType::Int64, Ops{},
        StateT{reduction::reduction_lower_bound<ValueT>(), 0});
}

template <typename T>
Tensor argmin_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        TP_THROW(RuntimeError, "argmin(): Expected reduction dim to be non-empty");
    }
    using ValueT = same_dtype_acc_t<T>;
    using StateT = ArgPair<ValueT>;
    using Ops = ArgOps<ValueT, false>;
    return run_reduction_typed<T, StateT, int64_t>(
        input, spec, keepdim, DType::Int64, Ops{},
        StateT{reduction::reduction_upper_bound<ValueT>(), 0});
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
Tensor sum_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec spec = make_reduction_spec(input, dim);
    TP_DISPATCH_REDUCTION(sum_same_dtype, input.dtype(), input, spec, keepdim, out_dtype);
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return sum_dim_kernel(self, {}, false, dtype);
}

// Mean
Tensor mean_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isFloatingType(self.dtype()) ? self.dtype() : DType::Float32;
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec spec = make_reduction_spec(input, dim);
    TP_DISPATCH_FLOAT_REDUCTION(mean_same_dtype, input.dtype(), input, spec, keepdim, out_dtype);
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

// Prod
Tensor prod_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isIntegralType(self.dtype(), true) ? DType::Int64 : self.dtype();
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec spec = make_reduction_spec(input, dim);
    TP_DISPATCH_REDUCTION(prod_same_dtype, input.dtype(), input, spec, keepdim, out_dtype);
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return prod_dim_kernel(self, {}, false, dtype);
}

// Max
Tensor max_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(max_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor max_kernel(const Tensor& self) {
    return max_dim_kernel(self, {}, false);
}

// Min
Tensor min_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(min_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor min_kernel(const Tensor& self) {
    return min_dim_kernel(self, {}, false);
}

// Norm (L2)
Tensor norm_global_kernel(const Tensor& self, double p) {
    const ReductionSpec spec = make_reduction_spec(self, {});
    TP_DISPATCH_FLOAT_REDUCTION(norm_same_dtype, self.dtype(), self, spec, false, p);
}

Tensor norm_dim_kernel(const Tensor& self, std::vector<int64_t> dim, double p, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(norm_same_dtype, self.dtype(), self, spec, keepdim, p);
}

// All / Any
Tensor all_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(all_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor all_kernel(const Tensor& self) {
    return all_dim_kernel(self, {}, false);
}

Tensor any_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(any_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor any_kernel(const Tensor& self) {
    return any_dim_kernel(self, {}, false);
}

// Var / Std
Tensor var_dim_kernel(const Tensor& self, std::vector<int64_t> dim, int64_t correction, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_same_dtype, self.dtype(), self, spec,
                                keepdim, correction, false);
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
    return var_dim_kernel(self, {}, correction, false);
}

Tensor std_dim_kernel(const Tensor& self, std::vector<int64_t> dim, int64_t correction, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_same_dtype, self.dtype(), self, spec,
                                keepdim, correction, true);
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
    return std_dim_kernel(self, {}, correction, false);
}

Tensor argmax_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    Tensor input = self;
    if (!dim.has_value() && !input.is_contiguous()) input = input.contiguous();
    const ReductionSpec spec = make_reduction_spec(
        input, dim.has_value() ? std::vector<int64_t>{*dim} : std::vector<int64_t>{});
    TP_DISPATCH_REDUCTION(argmax_same_dtype, input.dtype(), input, spec, keepdim);
}

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    Tensor input = self;
    if (!dim.has_value() && !input.is_contiguous()) input = input.contiguous();
    const ReductionSpec spec = make_reduction_spec(
        input, dim.has_value() ? std::vector<int64_t>{*dim} : std::vector<int64_t>{});
    TP_DISPATCH_REDUCTION(argmin_same_dtype, input.dtype(), input, spec, keepdim);
}


TENSORPLAY_LIBRARY_IMPL(CUDA, ReductionKernels) {
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    m.impl("mean_dim_backward", mean_dim_backward_kernel_cuda);
    
    m.impl("prod", prod_kernel);
    m.impl("prod.dim_IntList", prod_dim_kernel);
    
    m.impl("max", max_kernel);
    m.impl("max.dim", max_dim_kernel);
    
    m.impl("min", min_kernel);
    m.impl("min.dim", min_dim_kernel);
    
    m.impl("norm", norm_global_kernel);
    m.impl("norm.dim", norm_dim_kernel);
    
    m.impl("all", all_kernel);
    m.impl("all.dim", all_dim_kernel);
    
    m.impl("any", any_kernel);
    m.impl("any.dim", any_dim_kernel);
    
    m.impl("var", var_kernel);
    m.impl("var.dim", var_dim_kernel);
    
    m.impl("std", std_kernel);
    m.impl("std.dim", std_dim_kernel);

    m.impl("argmax", argmax_kernel);
    m.impl("argmin", argmin_kernel);
}

} // namespace cuda
} // namespace tensorplay
