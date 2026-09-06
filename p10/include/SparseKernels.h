#pragma once

#include "Tensor.h"

#include <array>
#include <optional>
#include <unordered_set>
#include <vector>

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
Tensor to_sparse_coo_cpu_sparse_dim(const Tensor& self, int64_t sparse_dim);
Tensor to_sparse_csr_cpu(const Tensor& self);
// sparse @ dense for 2-D COO/CSR `self` (SpMM / SpMM-dense).
Tensor sparse_mm_cpu(const Tensor& self, const Tensor& dense);
// COO @ COO elementwise product on matching coordinates and coordinate-union
// addition; both require same dtype/shape and non-hybrid values.
Tensor sparse_mul_cpu(const Tensor& self, const Tensor& other);
Tensor sparse_add_cpu(const Tensor& self, const Tensor& other);
// coalesced COO over the kept dims (duplicates folded); all sparse dims ->
// dense tensor.  ``dtype`` converts the input first (accumulation dtype).
Tensor sparse_sum_cpu(const Tensor& self, std::optional<std::vector<int64_t>> dim,
                      std::optional<DType> dtype);
// ``layout`` selects the output: 0 = sparse COO, 1 = sparse CSR.
Tensor spdiags_cpu(const Tensor& diagonals, const Tensor& offsets,
                   std::vector<int64_t> shape,
                   std::optional<int64_t> layout);
// r = beta * t + alpha * (sparse COO @ dense), materialized as a sparse COO
// over the union of the accumulator's coordinates and every column of each
// non-empty product row.  smm() routes through this with an empty
// accumulator.
Tensor sparse_sspaddmm_cpu(const Tensor& t, const Tensor& sparse,
                           const Tensor& dense, Scalar beta, Scalar alpha);
// sparse @ dense -> sparse; per non-empty row the full product row (zero
// entries included) is emitted.
Tensor smm_cpu(const Tensor& self, const Tensor& mat2);
// Dense -> compressed sparse (CSR/CSC/BSR/BSC).  The N-D input decomposes
// into (*batch, row, col, *dense); blocked layouts tile row/col into blocks
// and require sizes divisible by the block sizes.  Batched inputs join along
// the compressed axis and require uniform stored-element counts.
Tensor to_sparse_compressed_cpu(const Tensor& self, int layout,
                                std::array<int64_t, 2> blocksize,
                                std::optional<int64_t> dense_dim_opt,
                                const char* name);
// _sparse_sum family: the value-only overloads and the dim/keepdim/dtype
// overloads (see the reference sparse sum semantics).
Tensor _sparse_sum_cpu(const Tensor& input);
Tensor _sparse_sum_dtype_cpu(const Tensor& input, DType dtype);
Tensor _sparse_sum_dim_cpu(const Tensor& input, std::vector<int64_t> dims_to_sum,
                           std::optional<DType> dtype);
Tensor _sparse_sum_dim_dtype_cpu(const Tensor& input,
                                 std::vector<int64_t> dims_to_sum,
                                 DType dtype);
Tensor _sparse_sum_dim_cpu_2(const Tensor& input,
                             std::vector<int64_t> dims_to_sum);
Tensor _sparse_sum_backward_cpu(const Tensor& grad, const Tensor& input,
                                std::vector<int64_t> dims_to_sum);
// Sparse norm: full reductions only, no keepdim/dtype support.
Tensor native_norm_cpu(const Tensor& self, Scalar p);
Tensor native_norm_dim_cpu(const Tensor& self, std::optional<Scalar> p,
                           std::vector<int64_t> dims, bool keepdim,
                           std::optional<DType> dtype);

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
Tensor to_sparse_coo_cuda_sparse_dim(const Tensor& self, int64_t sparse_dim);
Tensor to_sparse_csr_cuda(const Tensor& self);
Tensor sparse_mm_cuda(const Tensor& self, const Tensor& dense);
Tensor sparse_sum_cuda(const Tensor& self, std::optional<std::vector<int64_t>> dim,
                       std::optional<DType> dtype);
Tensor spdiags_cuda(const Tensor& diagonals, const Tensor& offsets,
                    std::vector<int64_t> shape,
                    std::optional<int64_t> layout);

Tensor sparse_mul_cuda(const Tensor& self, const Tensor& other);
Tensor sparse_add_cuda(const Tensor& self, const Tensor& other);

} // namespace cuda
#endif
} // namespace tensorplay
