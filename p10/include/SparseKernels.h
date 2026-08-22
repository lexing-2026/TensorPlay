#pragma once

#include "Tensor.h"

namespace tensorplay {
namespace cpu {

Tensor sparse_coo_tensor_cpu(const Tensor& indices, const Tensor& values,
                             const std::vector<int64_t>& size, bool is_coalesced);
Tensor coalesce_sparse_cpu(const Tensor& self);
Tensor sparse_mask_cpu(const Tensor& dense, const Tensor& mask);
Tensor& add_sparse_to_dense_cpu(Tensor& dense, const Tensor& sparse, Scalar alpha);
Tensor embedding_sparse_backward_cpu(const Tensor& grad,
                                     const Tensor& indices,
                                     int64_t num_weights,
                                     int64_t padding_idx,
                                     bool scale_grad_by_freq);

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

Tensor sparse_coo_tensor_cuda(const Tensor& indices, const Tensor& values,
                              const std::vector<int64_t>& size, bool is_coalesced);
Tensor coalesce_sparse_cuda(const Tensor& self);
Tensor sparse_mask_cuda(const Tensor& dense, const Tensor& mask);
Tensor& add_sparse_to_dense_cuda(Tensor& dense, const Tensor& sparse, Scalar alpha);
Tensor embedding_sparse_backward_cuda(const Tensor& grad,
                                      const Tensor& indices,
                                      int64_t num_weights,
                                      int64_t padding_idx,
                                      bool scale_grad_by_freq);

} // namespace cuda
#endif
} // namespace tensorplay
