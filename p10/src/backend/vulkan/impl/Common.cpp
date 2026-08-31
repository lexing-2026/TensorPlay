#ifdef USE_VULKAN

#include "Common.h"

namespace tensorplay {
namespace vulkan {

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group) {
  api::utils::uvec3 local_group_size = {4, 4, 4};
  if (global_work_group[2u] == 1) {
    if (global_work_group[1u] < 8) {
      local_group_size[0u] = 16;
      local_group_size[1u] = 4;
      local_group_size[2u] = 1;
    } else {
      local_group_size[0u] = 8;
      local_group_size[1u] = 8;
      local_group_size[2u] = 1;
    }
  }
  return local_group_size;
}

} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
