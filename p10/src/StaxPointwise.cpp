#include "StaxPointwise.h"

namespace tensorplay {
namespace cpu {

DEFINE_DISPATCH(stax_pointwise_stub);
DEFINE_DISPATCH(stax_pointwise_multi_stub);

Tensor stax_fused_pointwise_cpu(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants) {
    return stax_pointwise_stub(
        DeviceType::CPU,
        inputs,
        program,
        constants);
}

std::vector<Tensor> stax_fused_pointwise_cpu_multi(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs) {
    return stax_pointwise_multi_stub(
        DeviceType::CPU,
        inputs,
        program,
        constants,
        output_refs);
}

} // namespace cpu
} // namespace tensorplay
