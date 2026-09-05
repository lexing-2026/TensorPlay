#pragma once

#ifdef USE_VULKAN

#include "Common.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor slice_kernel(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step);

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
