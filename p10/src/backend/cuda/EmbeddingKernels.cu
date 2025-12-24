#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Allocator.h"
#include <cuda_runtime.h>
#include <vector>

namespace tensorplay {
namespace cuda {

// Helper to check CUDA errors
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

template <typename T>
__global__ void embedding_kernel_cuda_impl(
    int64_t n_indices,
    int64_t embedding_dim,
    int64_t num_weights,
    int64_t padding_idx,
    const T* weight,
    const int64_t* indices,
    T* output) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_indices) return;
    
    int64_t idx = indices[i];
    
    T* dst = output + i * embedding_dim;
    
    if (padding_idx != -1 && idx == padding_idx) {
        // Zero out
        for (int64_t j = 0; j < embedding_dim; ++j) {
            dst[j] = 0;
        }
        return;
    }
    
    if (idx < 0 || idx >= num_weights) {
        // We can't throw exception from kernel safely without asserting
        // For robustness in this demo, we zero out out-of-bound access or ignore
        // PyTorch throws error. Since we can't throw, let's just ignore or zero.
        // Let's zero it to be safe.
        for (int64_t j = 0; j < embedding_dim; ++j) {
             dst[j] = 0;
        }
        return; 
    }
    
    // Copy embedding vector
    const T* src = weight + idx * embedding_dim;
    for (int64_t j = 0; j < embedding_dim; ++j) {
        dst[j] = src[j];
    }
}

Tensor embedding_cuda(const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    if (scale_grad_by_freq) TP_THROW(NotImplementedError, "embedding_cuda: scale_grad_by_freq not supported");
    if (sparse) TP_THROW(NotImplementedError, "embedding_cuda: sparse not supported");
    
    // Ensure weight is contiguous
    Tensor weight_contig = weight.contiguous();
    
    // Ensure indices are Long (int64) and contiguous
    Tensor indices_contig = indices.contiguous();
    if (indices_contig.dtype() != DType::Int64) {
        indices_contig = indices_contig.to(DType::Int64);
    }
    
    int64_t num_weights = weight_contig.size(0);
    int64_t embedding_dim = weight_contig.size(1);
    int64_t n_indices = indices_contig.numel();
    
    // Output shape: indices.shape + (embedding_dim,)
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(indices.shape());
    out_shape.push_back(embedding_dim);
    
    Tensor output = Tensor::empty(out_shape, weight.dtype(), weight.device());
    
    dim3 block(256);
    dim3 grid((n_indices + 255) / 256);
    
    if (weight.dtype() == DType::Float32) {
        embedding_kernel_cuda_impl<float><<<grid, block>>>(
            n_indices, 
            embedding_dim, 
            num_weights, 
            padding_idx,
            weight_contig.data_ptr<float>(),
            indices_contig.data_ptr<int64_t>(),
            output.data_ptr<float>()
        );
    } else {
        TP_THROW(NotImplementedError, "embedding_cuda: only float32 supported for now");
    }
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, EmbeddingKernels) {
    m.impl("embedding", embedding_cuda);
}

} // namespace cuda
} // namespace tensorplay
