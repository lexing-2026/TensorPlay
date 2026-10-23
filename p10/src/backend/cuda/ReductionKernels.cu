#include "ReduceEngine.cuh"

#include <iostream>
#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <type_traits>

#ifdef USE_CUDNN
#include <cudnn.h>
#include "Atomic.cuh"
#endif

namespace tensorplay {
namespace cuda {

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

using namespace reduce_family;

// aminmax: both extreme values out of one pass.  A MinMaxPair accumulator
// carries both ends through the same single read.
template <typename T>
std::tuple<Tensor, Tensor> aminmax_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    using ValueT = same_dtype_acc_t<T>;
    Tensor min_values = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), input.dtype(),
        input.device());
    // An extreme never overflows, so a run is bounded at the width of the
    // values it holds: widening a pair doubles the state every lane folds,
    // shuffles and stages through shared memory, and buys a range no element
    // of the run can ever need.  Bool is the exception -- a two-bit pair
    // aligns worse than it saves, so it keeps the wider accumulator.
    using AccScalarT = std::conditional_t<std::is_same_v<T, bool>, ValueT, T>;
    using StateT = reduction::MinMaxPair<AccScalarT>;
    using Ops = reduction::MinMaxPairOps<T, AccScalarT, T>;
    Tensor max_values = run_reduction_typed<T, StateT, T>(
        input, spec, keepdim, input.dtype(), Ops{},
        StateT{reduction::reduction_upper_bound<AccScalarT>(),
               reduction::reduction_lower_bound<AccScalarT>()},
        &min_values);
    return {min_values, max_values};
}

} // namespace

std::tuple<Tensor, Tensor> aminmax_dim_kernel(const Tensor& self,
                                              const std::vector<int64_t>& dim,
                                              bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(aminmax_same_dtype, self.dtype(), self, spec, keepdim);
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

TENSORPLAY_LIBRARY_IMPL(CUDA, ReductionKernels) {
    m.impl("mean_dim_backward", mean_dim_backward_kernel_cuda);
    m.impl("_sum_dim_backward", sum_dim_backward_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
