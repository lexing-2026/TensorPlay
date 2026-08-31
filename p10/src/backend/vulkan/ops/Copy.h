#pragma once

#ifdef USE_VULKAN

#include "Common.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

// Exposed for the CPU copy kernel: pulls a Vulkan source through the
// staging path into a CPU destination.
void transfer_vulkan_to_cpu_impl(api::vTensor& v_src, Tensor& dst);

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
