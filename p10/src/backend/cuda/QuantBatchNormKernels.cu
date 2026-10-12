#include "QuantNative.h"
#include "Dispatcher.h"

namespace tensorplay::cuda {
Tensor quantized_batch_norm_cuda(
    const Tensor& input, const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, const Tensor& mean,
    const Tensor& var, double eps, double output_scale,
    int64_t output_zero_point) {
    return quant::batch_norm_cuda(input, weight_opt, bias_opt, mean, var,
        eps, output_scale, output_zero_point, false);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantBatchNormKernels) {
    m.impl("quantized_batch_norm", quantized_batch_norm_cuda);
}

} // namespace tensorplay::cuda