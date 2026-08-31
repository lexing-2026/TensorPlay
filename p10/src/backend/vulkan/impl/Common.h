#pragma once

#ifdef USE_VULKAN

#include "../api/Tensor.h"

namespace tensorplay {
namespace vulkan {

/*
 * Selects a local work group size adapted to the global work group shape.
 * Full 4x4x4 tiles for volumetric problems; flattened 16x4 / 8x8 tiles when
 * the depth axis degenerates to one, matching the occupancy behavior of the
 * image shaders.
 */
api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
