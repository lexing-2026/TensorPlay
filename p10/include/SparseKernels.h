#pragma once

#include "Tensor.h"

#include <optional>

namespace tensorplay {
namespace cpu {

Tensor sparse_coo_tensor_cpu(const Tensor& indices, const Tensor& values,
                             std::optional<std::vector<int64_t>> size,
                             bool is_coalesced);
Tensor coalesce_sparse_cpu(const Tensor& self);
Tensor sparse_mask_cpu(const Tensor& dense, const Tensor& mask);
Tensor& add_sparse_to_dense_cpu(Tensor& dense, const Tensor& sparse, Scalar alpha);
Tensor embedding_sparse_backward_cpu(const Tensor& grad,
                                     const Tensor& indices,
                                     int64_t num_weights,
                                     int64_t padding_idx,
                                     bool scale_grad_by_freq);
// Materializes a sparse COO/CSR tensor into a freshly allocated dense tensor.
Tensor to_dense_sparse_cpu(const Tensor& self);
// Number of stored elements (values.size(0)); valid for COO and CSR.
int64_t sparse_nnz_cpu(const Tensor& self);
// Dense -> sparse conversions.  to_sparse produces a coalesced COO covering
// every nonzero coordinate; to_sparse_csr accepts exactly-2-D input and
// produces the compressed-row form.
Tensor to_sparse_coo_cpu(const Tensor& self);
Tensor to_sparse_csr_cpu(const Tensor& self);
// sparse @ dense for 2-D COO/CSR `self` (SpMM / SpMM-dense).
Tensor sparse_mm_cpu(const Tensor& self, const Tensor& dense);
// Full reduction of a sparse tensor to a 0-dim dense tensor.
Tensor sparse_sum_cpu(const Tensor& self);

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

Tensor sparse_coo_tensor_cuda(const Tensor& indices, const Tensor& values,
                              std::optional<std::vector<int64_t>> size,
                              bool is_coalesced);
Tensor coalesce_sparse_cuda(const Tensor& self);
Tensor sparse_mask_cuda(const Tensor& dense, const Tensor& mask);
Tensor& add_sparse_to_dense_cuda(Tensor& dense, const Tensor& sparse, Scalar alpha);
Tensor embedding_sparse_backward_cuda(const Tensor& grad,
                                      const Tensor& indices,
                                      int64_t num_weights,
                                      int64_t padding_idx,
                                      bool scale_grad_by_freq);
Tensor to_dense_sparse_cuda(const Tensor& self);
int64_t sparse_nnz_cuda(const Tensor& self);
Tensor to_sparse_coo_cuda(const Tensor& self);
Tensor to_sparse_csr_cuda(const Tensor& self);
Tensor sparse_mm_cuda(const Tensor& self, const Tensor& dense);
Tensor sparse_sum_cuda(const Tensor& self);

} // namespace cuda
#endif
} // namespace tensorplay
