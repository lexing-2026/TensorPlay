#include <iostream>
#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include <cuda_runtime.h>
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

// --- ArgMax / ArgMin Custom Kernels ---

template <typename T, bool MAX_MODE>
__global__ void arg_reduce_lastdim_kernel(
    int64_t outer_size, int64_t inner_size,
    const T* input, int64_t* output) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outer_size) {
        const T* row = input + i * inner_size;
        T best_val = row[0];
        int64_t best_idx = 0;
        
        for (int64_t j = 1; j < inner_size; ++j) {
            T val = row[j];
            if (MAX_MODE) {
                if (val > best_val) { best_val = val; best_idx = j; }
            } else {
                if (val < best_val) { best_val = val; best_idx = j; }
            }
        }
        output[i] = best_idx;
    }
}

Tensor arg_reduce_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim, bool max_mode) {
    // Basic implementation for contiguous last dim
    Tensor in = self;
    bool flattened = false;
    
    if (!dim.has_value()) {
        in = self.reshape({-1});
        flattened = true;
    }
    
    int64_t d = flattened ? 0 : *dim;
    if (d < 0) d += in.dim();
    
    // Permute target dim to last
    if (d != in.dim() - 1) {
        std::vector<int64_t> perm;
        for (int i=0; i<in.dim(); ++i) if (i != d) perm.push_back(i);
        perm.push_back(d);
        in = in.permute(perm);
    }
    
    in = in.contiguous();
    
    int64_t inner_size = in.size(in.dim() - 1);
    int64_t outer_size = in.numel() / inner_size;
    
    Tensor result = Tensor::empty({outer_size}, DType::Int64, self.device());
    
    int threads = 256;
    int blocks = (outer_size + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // Cap grid size just in case
    
    if (in.dtype() == DType::Float32) {
        if (max_mode)
             arg_reduce_lastdim_kernel<float, true><<<blocks, threads>>>(outer_size, inner_size, in.data_ptr<float>(), result.data_ptr<int64_t>());
        else
             arg_reduce_lastdim_kernel<float, false><<<blocks, threads>>>(outer_size, inner_size, in.data_ptr<float>(), result.data_ptr<int64_t>());
    } else {
        TP_THROW(NotImplementedError, "argmax/argmin CUDA: only float32 supported");
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    if (flattened) {
         if (keepdim) return result.reshape(std::vector<int64_t>(self.dim(), 1));
         return result.reshape({}); // Scalar
    }
    
    if (keepdim) {
         std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
         shape[d] = 1;
         return result.reshape(shape);
    } else {
         std::vector<int64_t> shape;
         for (int i=0; i<self.dim(); ++i) {
             if (i != d) shape.push_back(self.size(i));
         }
         return result.reshape(shape);
    }
}

Tensor argmax_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return arg_reduce_kernel(self, dim, keepdim, true);
}

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return arg_reduce_kernel(self, dim, keepdim, false);
}


// --- cuDNN Reduction Helper ---

#ifdef USE_CUDNN

Tensor cudnn_reduce_wrapper(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim, 
                           cudnnReduceTensorOp_t op, DType out_dtype) {
    // DEBUG: Print input shape
    
    std::cout << "cudnn_reduce_wrapper input: shape=(";
    for(auto s : self.shape()) std::cout << s << ",";
    std::cout << ")" << std::endl;
    

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
    
    void* workspace = nullptr;
    if (wsSize > 0) {
        cudaMalloc(&workspace, wsSize);
    }
    
    // std::cout << "Running cudnnReduceTensor..." << std::endl;
    CUDNN_CHECK(cudnnReduceTensor(handle, reduceDesc, 
        nullptr, 0, 
        workspace, wsSize, 
        alpha, aDesc, self_contig.data_ptr(), 
        beta, cDesc, result.data_ptr()));
        
    if (workspace) cudaFree(workspace);
    
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

// --- Implementations ---

// Sum
Tensor sum_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim, DType dtype) {
    if (dtype == DType::Undefined) dtype = self.dtype();
#ifdef USE_CUDNN
    if (dtype == DType::Float32 || dtype == DType::Float64)
        return cudnn_reduce_wrapper(self, dim, keepdim, CUDNN_REDUCE_TENSOR_ADD, dtype);
#endif
    TP_THROW(NotImplementedError, "sum: only float32/64 supported via cuDNN");
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    return sum_dim_kernel(self, {}, false, dtype);
}

// Mean
Tensor mean_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim, DType dtype) {
    if (dtype == DType::Undefined) dtype = self.dtype();
#ifdef USE_CUDNN
    if (dtype == DType::Float32 || dtype == DType::Float64) {
        return cudnn_reduce_wrapper(self, dim, keepdim, CUDNN_REDUCE_TENSOR_AVG, dtype);
    }
#endif
    TP_THROW(NotImplementedError, "mean: only float32/64 supported via cuDNN");
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    return mean_dim_kernel(self, {}, false, dtype);
}

// Prod
Tensor prod_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim, DType dtype) {
    if (dtype == DType::Undefined) dtype = self.dtype();
#ifdef USE_CUDNN
    if (dtype == DType::Float32 || dtype == DType::Float64)
        return cudnn_reduce_wrapper(self, dim, keepdim, CUDNN_REDUCE_TENSOR_MUL, dtype);
#endif
    TP_THROW(NotImplementedError, "prod: only float32/64 supported via cuDNN");
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
    return prod_dim_kernel(self, {}, false, dtype);
}

// Max
Tensor max_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
#ifdef USE_CUDNN
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64)
        return cudnn_reduce_wrapper(self, dim, keepdim, CUDNN_REDUCE_TENSOR_MAX, self.dtype());
#endif
    TP_THROW(NotImplementedError, "max: only float32/64 supported via cuDNN");
}

Tensor max_kernel(const Tensor& self) {
    return max_dim_kernel(self, {}, false);
}

// Min
Tensor min_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
#ifdef USE_CUDNN
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64)
        return cudnn_reduce_wrapper(self, dim, keepdim, CUDNN_REDUCE_TENSOR_MIN, self.dtype());
#endif
    TP_THROW(NotImplementedError, "min: only float32/64 supported via cuDNN");
}

Tensor min_kernel(const Tensor& self) {
    return min_dim_kernel(self, {}, false);
}

// Norm (L2)
Tensor norm_global_kernel(const Tensor& self, double p) {
    if (p != 2.0) {
         TP_THROW(NotImplementedError, "norm: only p=2 supported on CUDA");
    }
#ifdef USE_CUDNN
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        // Global norm
        std::vector<int64_t> all_dims(self.dim());
        std::iota(all_dims.begin(), all_dims.end(), 0);
        return cudnn_reduce_wrapper(self, all_dims, false, CUDNN_REDUCE_TENSOR_NORM2, self.dtype());
    }
#endif
    TP_THROW(NotImplementedError, "norm: only float32/64 supported via cuDNN");
}

Tensor norm_dim_kernel(const Tensor& self, std::vector<int64_t> dim, double p, bool keepdim) {
    if (p != 2.0) {
         TP_THROW(NotImplementedError, "norm: only p=2 supported on CUDA");
    }
#ifdef USE_CUDNN
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64)
        return cudnn_reduce_wrapper(self, dim, keepdim, CUDNN_REDUCE_TENSOR_NORM2, self.dtype());
#endif
    TP_THROW(NotImplementedError, "norm: only float32/64 supported via cuDNN");
}

// All / Any (via Float Min/Max)
Tensor all_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    Tensor f = self.to(DType::Float32);
    Tensor m = min_dim_kernel(f, dim, keepdim);
    return m.to(DType::Bool);
}

Tensor all_kernel(const Tensor& self) {
    Tensor f = self.to(DType::Float32);
    Tensor m = min_kernel(f);
    return m.to(DType::Bool);
}

Tensor any_dim_kernel(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    Tensor f = self.to(DType::Float32);
    Tensor m = max_dim_kernel(f, dim, keepdim);
    return m.to(DType::Bool);
}

Tensor any_kernel(const Tensor& self) {
    Tensor f = self.to(DType::Float32);
    Tensor m = max_kernel(f);
    return m.to(DType::Bool);
}

// Var / Std
int64_t get_reduced_size(const Tensor& self, const std::vector<int64_t>& dim) {
    if (dim.empty()) return self.numel();
    int64_t size = 1;
    for (auto d : dim) {
        int64_t dd = d < 0 ? d + self.dim() : d;
        size *= self.size(dd);
    }
    return size;
}

Tensor var_dim_kernel(const Tensor& self, std::vector<int64_t> dim, int64_t correction, bool keepdim) {
    Tensor m = self.mean(dim, true); 
    Tensor diff = self - m; 
    Tensor sq = diff.pow(2.0); 
    Tensor s = sq.sum(dim, keepdim);
    
    int64_t n = get_reduced_size(self, dim);
    double div_val = std::max<double>(0.0, static_cast<double>(n - correction));
    
    // Use Tensor division to avoid Scalar issues
    Tensor div_t = Tensor::full({}, Scalar(div_val), s.dtype(), s.device());
    return s / div_t;
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
    return var_dim_kernel(self, {}, correction, false);
}

Tensor std_dim_kernel(const Tensor& self, std::vector<int64_t> dim, int64_t correction, bool keepdim) {
    return var_dim_kernel(self, dim, correction, keepdim).sqrt();
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
    return var_kernel(self, correction).sqrt();
}


TENSORPLAY_LIBRARY_IMPL(CUDA, ReductionKernels) {
    m.impl("sum", sum_kernel);
    m.impl("sum.dim_IntList", sum_dim_kernel);
    
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    
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
