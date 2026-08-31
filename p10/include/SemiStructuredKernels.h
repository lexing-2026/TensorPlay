#pragma once

#include "Tensor.h"

#include <optional>
#include <tuple>

namespace tensorplay {

namespace cpu {

std::tuple<Tensor, Tensor> sparse_semi_structured_compress_cpu(
    const Tensor& dense);
Tensor sparse_semi_structured_to_dense_cpu(const Tensor& packed,
                                           const Tensor& meta);
Tensor sparse_semi_structured_mask_grad_cpu(
    const Tensor& grad, const Tensor& packed, const Tensor& meta);
Tensor sparse_semi_structured_gather_grad_cpu(
    const Tensor& grad, const Tensor& packed, const Tensor& meta);
Tensor sparse_semi_structured_mm_cpu(
    const Tensor& packed, const Tensor& meta, const Tensor& dense,
    std::optional<DType> out_dtype);
Tensor sparse_semi_structured_mm_right_cpu(
    const Tensor& dense, const Tensor& packed, const Tensor& meta,
    std::optional<DType> out_dtype);
Tensor sparse_semi_structured_addmm_cpu(
    const Tensor& input, const Tensor& packed, const Tensor& meta,
    const Tensor& dense, Scalar alpha, Scalar beta,
    std::optional<DType> out_dtype);

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

std::tuple<Tensor, Tensor> sparse_semi_structured_compress_cuda(
    const Tensor& dense);
Tensor sparse_semi_structured_to_dense_cuda(const Tensor& packed,
                                            const Tensor& meta);
Tensor sparse_semi_structured_mask_grad_cuda(
    const Tensor& grad, const Tensor& packed, const Tensor& meta);
Tensor sparse_semi_structured_gather_grad_cuda(
    const Tensor& grad, const Tensor& packed, const Tensor& meta);
Tensor sparse_semi_structured_mm_cuda(
    const Tensor& packed, const Tensor& meta, const Tensor& dense,
    std::optional<DType> out_dtype);
Tensor sparse_semi_structured_mm_right_cuda(
    const Tensor& dense, const Tensor& packed, const Tensor& meta,
    std::optional<DType> out_dtype);
Tensor sparse_semi_structured_addmm_cuda(
    const Tensor& input, const Tensor& packed, const Tensor& meta,
    const Tensor& dense, Scalar alpha, Scalar beta,
    std::optional<DType> out_dtype);

} // namespace cuda
#endif

} // namespace tensorplay
