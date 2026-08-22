#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "SparseKernels.h"
#include <cstring>
#include <vector>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#endif

namespace tensorplay {
namespace cpu {

Tensor to_kernel(Tensor& self, DType dtype, bool non_blocking, bool copy) {
    if (self.dtype() == dtype) {
        return copy ? self.clone() : self; // clone not impl yet, use copy_
    }
    // Create new tensor
    Tensor result(static_cast<std::vector<int64_t>>(self.shape()), dtype, self.device());
    result.copy_(self);
    return result;
}

// Helper for recursive copy
template <typename T_SELF, typename T_SRC>
void copy_recursive(
    T_SELF* self_data, const std::vector<int64_t>& self_strides,
    const T_SRC* src_data, const std::vector<int64_t>& src_strides,
    const std::vector<int64_t>& sizes,
    int64_t dim,
    int64_t self_offset, int64_t src_offset) {
    
    if (sizes.empty()) { // Scalar case
        self_data[self_offset] = cast_value<T_SELF>(src_data[src_offset]);
        return;
    }

    if (dim == sizes.size() - 1) {
        int64_t n = sizes[dim];
        int64_t self_stride = self_strides[dim];
        int64_t src_stride = src_strides[dim];
        for (int64_t i = 0; i < n; ++i) {
            self_data[self_offset + i * self_stride] = cast_value<T_SELF>(src_data[src_offset + i * src_stride]);
        }
    } else {
        int64_t n = sizes[dim];
        int64_t self_stride = self_strides[dim];
        int64_t src_stride = src_strides[dim];
        for (int64_t i = 0; i < n; ++i) {
            copy_recursive(self_data, self_strides, src_data, src_strides, sizes, dim + 1, 
                           self_offset + i * self_stride, src_offset + i * src_stride);
        }
    }
}

// Helper for dynamic dispatch
template <typename T>
struct TypeTag { using type = T; };

template <typename F>
void dispatch_dtype(DType dtype, F&& callback) {
    #define DISPATCH_CASE(ctype, name) \
    case DType::name: { \
        callback(TypeTag<ctype>{}); \
        return; \
    }

    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(DISPATCH_CASE)
        default:
            throw std::runtime_error("Unsupported dtype in dispatch");
    }
    #undef DISPATCH_CASE
}

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    if (!self.device().is_cpu()) {
        throw std::runtime_error("copy_kernel (CPU) called with non-CPU destination");
    }

    if (src.device().is_cuda()) {
#ifdef USE_CUDA
        cuda::CUDAGuard device_guard(src.device());
        auto stream = cuda::getCurrentCUDAStream(static_cast<int>(src.device().index()));

        Tensor src_ready = src;
        if (!src.is_contiguous() || src.dtype() != self.dtype()) {
            src_ready = Tensor(static_cast<std::vector<int64_t>>(src.shape()), self.dtype(), src.device());
            src_ready.copy_(src);
        }

        const size_t nbytes = self.numel() * self.itemsize();
        if (self.is_contiguous()) {
            cuda::checkCuda(cudaMemcpyAsync(self.data_ptr(), src_ready.data_ptr(), nbytes,
                                            cudaMemcpyDeviceToHost, stream.stream()),
                            "cudaMemcpyAsync (D2H)");
            if (non_blocking && self.is_pinned()) {
                cuda::recordPinnedStream(
                    self.unsafeGetTensorImpl()->storage().data(), stream);
            } else {
                // Ordinary CPU storage may be consumed immediately after
                // copy_ returns, so the host-visible result must be complete.
                stream.synchronize();
            }
        } else {
            Tensor host_contiguous(static_cast<std::vector<int64_t>>(self.shape()),
                                   self.dtype(), Device(DeviceType::CPU));
            cuda::checkCuda(cudaMemcpyAsync(host_contiguous.data_ptr(), src_ready.data_ptr(), nbytes,
                                            cudaMemcpyDeviceToHost, stream.stream()),
                            "cudaMemcpyAsync (D2H staging)");
            stream.synchronize();
            self.copy_(host_contiguous);
        }
        return self;
#else
        throw std::runtime_error("CUDA source but USE_CUDA not enabled");
#endif
    }
    
    if (!src.device().is_cpu()) {
        throw std::runtime_error("copy_kernel only supports CPU or CUDA source");
    }
    
    dispatch_dtype(self.dtype(), [&](auto self_tag) {
        using self_t = typename decltype(self_tag)::type;
        
        dispatch_dtype(src.dtype(), [&](auto src_tag) {
            using src_t = typename decltype(src_tag)::type;
            
            // Optimization for contiguous same-dtype copy
            if (std::is_same_v<self_t, src_t> && self.is_contiguous() && src.is_contiguous()) {
                 size_t nbytes = self.numel() * self.itemsize();
                 std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
                 return;
            }
            
            copy_recursive(self.data_ptr<self_t>(), self.strides(), 
                           src.data_ptr<src_t>(), src.strides(), 
                           static_cast<std::vector<int64_t>>(self.shape()), 0, 0, 0);
        });
    });
    
    return self;
}



Tensor masked_select_cpu(const Tensor& self, const Tensor& mask) {
    // 1. Broadcast shapes
    std::vector<int64_t> broadcast_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(mask.shape()));
    
    Tensor self_expanded = self.expand(broadcast_shape);
    Tensor mask_expanded = mask.expand(broadcast_shape);
    
    // 2. Make contiguous for simple iteration
    Tensor self_contig = self_expanded.is_contiguous() ? self_expanded : self_expanded.clone();
    Tensor mask_contig = mask_expanded.is_contiguous() ? mask_expanded : mask_expanded.clone();
    
    int64_t numel = self_contig.numel();
    const uint8_t* mask_ptr = nullptr;
    
    // Handle mask dtype
    if (mask_contig.dtype() == DType::Bool || mask_contig.dtype() == DType::UInt8) {
        mask_ptr = mask_contig.data_ptr<uint8_t>();
    } else {
        TP_THROW(TypeError, "masked_select: mask must be Bool or Byte");
    }
    
    // 3. Count true elements
    int64_t true_count = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (mask_ptr[i]) true_count++;
    }
    
    // 4. Allocate result
    Tensor result = Tensor::empty({true_count}, self.dtype(), self.device());
    
    // 5. Fill result
    dispatch_dtype(self.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* src = self_contig.data_ptr<T>();
        T* dst = result.data_ptr<T>();
        
        int64_t idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_ptr[i]) {
                dst[idx++] = src[i];
            }
        }
    });
    
    return result;
}

Tensor embedding_cpu(const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    // 1. Check inputs
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding: indices must be Int64 or Int32");
    }
    
    // 2. Calculate output shape
    // Output shape = indices.shape + weight.shape[1:]
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(indices.shape());
    std::vector<int64_t> weight_shape = static_cast<std::vector<int64_t>>(weight.shape());
    
    if (weight.dim() == 0) {
        TP_THROW(RuntimeError, "embedding: weight must be at least 1-dim");
    }
    
    for (size_t i = 1; i < weight_shape.size(); ++i) {
        out_shape.push_back(weight_shape[i]);
    }
    
    // 3. Allocate output
    Tensor output = Tensor::empty(out_shape, weight.dtype(), weight.device());
    
    // 4. Copy data
    // Flatten indices for iteration
    int64_t num_indices = indices.numel();
    int64_t row_size = 1;
    for (size_t i = 1; i < weight_shape.size(); ++i) row_size *= weight_shape[i];
    int64_t weight_size_0 = weight.size(0);
    
    // Contiguous access optimization
    Tensor indices_contig = indices.is_contiguous() ? indices : indices.clone();
    Tensor weight_contig = weight.is_contiguous() ? weight : weight.clone();
    
    dispatch_dtype(weight.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* weight_data = weight_contig.data_ptr<T>();
        T* out_data = output.data_ptr<T>();
        
        // Handle indices type
        if (indices.dtype() == DType::Int64) {
            const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = idx_data[i];
                if (idx < 0) idx += weight_size_0;
                if (idx < 0 || idx >= weight_size_0) {
                    TP_THROW(IndexError, "embedding: index out of range");
                }
                
                // Copy row
                std::memcpy(out_data + i * row_size, weight_data + idx * row_size, row_size * sizeof(T));
            }
        } else { // Int32
            const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = static_cast<int64_t>(idx_data[i]);
                if (idx < 0) idx += weight_size_0;
                if (idx < 0 || idx >= weight_size_0) {
                    TP_THROW(IndexError, "embedding: index out of range");
                }
                
                std::memcpy(out_data + i * row_size, weight_data + idx * row_size, row_size * sizeof(T));
            }
        }
    });
    
    return output;
}

Tensor embedding_dense_backward_cpu(const Tensor& grad_output, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    // 1. Check inputs
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_dense_backward: indices must be Int64 or Int32");
    }

    // 2. Allocate grad_weight
    std::vector<int64_t> grad_weight_shape;
    grad_weight_shape.push_back(num_weights);
    int64_t weight_dims = grad_output.dim() - indices.dim();
    for (int i = 0; i < weight_dims; ++i) {
        grad_weight_shape.push_back(grad_output.size(indices.dim() + i));
    }
    
    Tensor grad_weight = Tensor::zeros(grad_weight_shape, grad_output.dtype(), grad_output.device());
    
    // 3. Accumulate gradients
    int64_t num_indices = indices.numel();
    int64_t grad_numel = grad_output.numel();
    int64_t row_size = num_indices > 0 ? grad_numel / num_indices : 0;
    
    if (num_indices == 0) return grad_weight;

    Tensor indices_contig = indices.is_contiguous() ? indices : indices.clone();
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
    
    dispatch_dtype(grad_output.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, bool>) {
             TP_THROW(RuntimeError, "embedding_dense_backward: grad_output cannot be Bool");
        } else {
            const T* grad_data = grad_output_contig.data_ptr<T>();
            T* weight_grad_data = grad_weight.data_ptr<T>();
            
            // Handle indices type
            if (indices.dtype() == DType::Int64) {
                const int64_t* idx_data = indices_contig.data_ptr<int64_t>();
                for (int64_t i = 0; i < num_indices; ++i) {
                    int64_t idx = idx_data[i];
                    if (idx == padding_idx) continue;
                    if (idx < 0) idx += num_weights;
                    if (idx < 0 || idx >= num_weights) {
                         TP_THROW(IndexError, "embedding_dense_backward: index out of range");
                    }
                    
                    // Add row: weight_grad[idx] += grad[i]
                    T* dst_row = weight_grad_data + idx * row_size;
                    const T* src_row = grad_data + i * row_size;
                    for (int64_t j = 0; j < row_size; ++j) {
                        dst_row[j] += src_row[j];
                    }
                }
            } else { // Int32
                 const int32_t* idx_data = indices_contig.data_ptr<int32_t>();
                for (int64_t i = 0; i < num_indices; ++i) {
                    int64_t idx = static_cast<int64_t>(idx_data[i]);
                    if (idx == padding_idx) continue;
                    if (idx < 0) idx += num_weights;
                    if (idx < 0 || idx >= num_weights) {
                         TP_THROW(IndexError, "embedding_dense_backward: index out of range");
                    }
                    
                    // Add row
                    T* dst_row = weight_grad_data + idx * row_size;
                    const T* src_row = grad_data + i * row_size;
                    for (int64_t j = 0; j < row_size; ++j) {
                        dst_row[j] += src_row[j];
                    }
                }
            }
        }
    });
    
    return grad_weight;
}

Tensor embedding_backward_cpu(const Tensor& grad_output, const Tensor& indices,
                              int64_t num_weights, int64_t padding_idx,
                              bool scale_grad_by_freq, bool sparse) {
    if (sparse) {
        return embedding_sparse_backward_cpu(grad_output, indices, num_weights,
                                             padding_idx, scale_grad_by_freq);
    }
    return embedding_dense_backward_cpu(grad_output, indices, num_weights,
                                       padding_idx, scale_grad_by_freq);
}

TENSORPLAY_LIBRARY_IMPL(CPU, CopyKernels) {
    m.impl("to", to_kernel);
    m.impl("masked_select", masked_select_cpu);
    m.impl("copy_", copy_kernel);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cpu);
    m.impl("sparse_mask", sparse_mask_cpu);
    m.impl("embedding", embedding_cpu);
    m.impl("embedding_dense_backward", embedding_dense_backward_cpu);
    m.impl("embedding_sparse_backward", embedding_sparse_backward_cpu);
    m.impl("embedding_backward", embedding_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
