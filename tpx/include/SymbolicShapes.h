#pragma once
#include "Tensor.h"

namespace tensorplay::tpx {

TENSORPLAY_API std::vector<SymInt> symbolic_sizes(const Tensor& value);
TENSORPLAY_API Tensor expand_symint(const Tensor& value, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor reshape_symint(const Tensor& value, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor div_symint(const Tensor& value, const SymInt& divisor);
TENSORPLAY_API Tensor sum_to_symint(const Tensor& grad, const std::vector<SymInt>& sizes);
TENSORPLAY_API Tensor sum_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes,
    std::vector<int64_t> dims, bool keepdim);
TENSORPLAY_API Tensor mean_backward_symint(
    const Tensor& grad, const std::vector<SymInt>& sizes,
    std::vector<int64_t> dims, bool keepdim);

} // namespace tensorplay::tpx
