// 1-D pooling wrappers delegate to concrete 2-D pooling kernels after
// inserting a singleton spatial axis.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstddef>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay::cpu {

namespace ops = tensorplay::tpx::ops;

namespace {

void check_dim_range(const Tensor& self, const char* function) {
    if (self.dim() < 2 || self.dim() > 3) {
        TP_THROW(RuntimeError, function,
                 ": Expected 2D or 3D input, but got ", self.dim(), "D");
    }
}

int64_t check1d(const std::vector<int64_t>& value, const char* function,
                const char* argument, bool allow_empty, int64_t empty_value) {
    if (allow_empty && value.empty()) return empty_value;
    if (value.size() != 1) {
        TP_THROW(RuntimeError, function, "() argument '", argument,
                 "' should contain one int (got ", value.size(), ")");
    }
    return value[0];
}

} // namespace

Tensor adaptive_avg_pool1d_native_cpu(
    const Tensor& self, const std::vector<int64_t>& output_size) {
    check_dim_range(self, "adaptive_avg_pool1d");
    const int64_t output = check1d(output_size, "adaptive_avg_pool1d",
                                   "output_size", false, 0);
    return ops::squeeze(
        ops::adaptive_avg_pool2d(ops::unsqueeze(self, -2), {1, output}), -2);
}

std::tuple<Tensor, Tensor> adaptive_max_pool1d_native_cpu(
    const Tensor& self, const std::vector<int64_t>& output_size) {
    check_dim_range(self, "adaptive_max_pool1d");
    const int64_t output = check1d(output_size, "adaptive_max_pool1d",
                                   "output_size", false, 0);
    for (int64_t dim = 1; dim < self.dim(); ++dim) {
        if (self.size(dim) == 0) {
            TP_THROW(RuntimeError,
                     "adaptive_max_pool1d(): Expected input to have non-zero "
                     "size for non-batch dimensions");
        }
    }
    auto pooled = ops::adaptive_max_pool2d_with_indices(
        ops::unsqueeze(self, -2), {1, output});
    return {ops::squeeze(std::get<0>(pooled), -2),
            ops::squeeze(std::get<1>(pooled), -2)};
}

std::tuple<Tensor, Tensor> max_pool1d_with_indices_native_cpu(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    check_dim_range(self, "max_pool1d");
    const int64_t kernel = check1d(kernel_size, "max_pool1d", "kernel_size",
                                   false, 0);
    const int64_t step = check1d(stride, "max_pool1d", "stride", true, kernel);
    const int64_t pad = check1d(padding, "max_pool1d", "padding", true, 0);
    const int64_t dil = check1d(dilation, "max_pool1d", "dilation", true, 1);
    auto pooled = ops::max_pool2d_with_indices(
        ops::unsqueeze(self, -2), {1, kernel}, {1, step}, {0, pad}, {1, dil},
        ceil_mode);
    return {ops::squeeze(std::get<0>(pooled), -2),
            ops::squeeze(std::get<1>(pooled), -2)};
}

Tensor avg_pool1d_native_cpu(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    bool ceil_mode, bool count_include_pad) {
    check_dim_range(self, "avg_pool1d");
    const int64_t kernel = check1d(kernel_size, "avg_pool1d", "kernel_size",
                                   false, 0);
    const int64_t step = check1d(stride, "avg_pool1d", "stride", true, kernel);
    const int64_t pad = check1d(padding, "avg_pool1d", "padding", true, 0);
    Tensor output = ops::avg_pool2d(
        ops::unsqueeze(self, -2), {1, kernel}, {1, step}, {0, pad},
        ceil_mode, count_include_pad, std::nullopt);
    return ops::squeeze(output, -2);
}

TENSORPLAY_LIBRARY_IMPL(CPU, NativePooling) {
    m.impl("adaptive_avg_pool1d", adaptive_avg_pool1d_native_cpu);
    m.impl("adaptive_max_pool1d", adaptive_max_pool1d_native_cpu);
    m.impl("max_pool1d_with_indices", max_pool1d_with_indices_native_cpu);
    m.impl("avg_pool1d", avg_pool1d_native_cpu);
}

} // namespace tensorplay::cpu
