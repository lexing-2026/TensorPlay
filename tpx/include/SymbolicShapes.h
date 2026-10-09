#pragma once
#include "Tensor.h"

namespace tensorplay::tpx {

TENSORPLAY_API std::vector<SymInt> symbolic_sizes(const Tensor& value);
TENSORPLAY_API Tensor expand_symint(const Tensor& value, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor reshape_symint(const Tensor& value, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor div_symint(const Tensor& value, const SymInt& divisor);
TENSORPLAY_API Tensor new_zeros_symint(const Tensor& value, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor unsqueeze_to_symint(const Tensor& grad, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor unsqueeze_to_symint(
    const Tensor& grad, const std::vector<int64_t>& dims, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor repeat_backward_symint(
    Tensor grad, const std::vector<int64_t>& repeats, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor slice_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes, int64_t dim,
    std::optional<int64_t> start, std::optional<int64_t> end, int64_t step);
TENSORPLAY_API Tensor select_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes, int64_t dim, int64_t index);
TENSORPLAY_API Tensor diagonal_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes,
    int64_t offset, int64_t dim1, int64_t dim2);
TENSORPLAY_API Tensor sum_to_symint(const Tensor& grad, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor sum_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes,
    std::vector<int64_t> dims, bool keepdim);
TENSORPLAY_API Tensor mean_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes,
    std::vector<int64_t> dims, bool keepdim);

} // namespace tensorplay::tpx
