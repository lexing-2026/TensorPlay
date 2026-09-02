#ifdef USE_VULKAN

#include "QueryPool.h"
#include "Exception.h"

#include <algorithm>

namespace tensorplay {
namespace vulkan {
namespace api {

QueryPool::QueryPool(VkDevice device, const QueryPoolConfig& config)
    : device_(device),
      config_(config),
      capacity_{0u},
      in_use_{0u} {
  TP_CHECK(
      config_.queryPoolInitialSize > 0,
      "Vulkan QueryPool initial capacity must be greater than 0!");
  TP_CHECK(
      config_.queryPoolBatchSize > 0,
      "Vulkan QueryPool batch size must be greater than 0!");

  allocate_new_batch(config_.queryPoolInitialSize);
}

QueryPool::~QueryPool() {
  for (VkQueryPool pool : pools_) {
    if (VK_NULL_HANDLE != pool) {
      vkDestroyQueryPool(device_, pool, nullptr);
    }
  }
}

void QueryPool::allocate_new_batch(const uint32_t queries) {
  const VkQueryPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_QUERY_TYPE_TIMESTAMP, // queryType
      queries, // queryCount
      0u, // pipelineStatistics
  };

  VkQueryPool handle = VK_NULL_HANDLE;
  VK_CHECK(vkCreateQueryPool(device_, &create_info, nullptr, &handle));
  pools_.push_back(handle);
  capacity_ += queries;
}

std::pair<VkQueryPool, uint32_t> QueryPool::get_new_query() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (in_use_ >= capacity_) {
    allocate_new_batch(config_.queryPoolBatchSize);
  }

  const uint32_t global_index = in_use_++;
  const uint32_t pool_index = global_index / config_.queryPoolBatchSize;
  const uint32_t query_index = global_index % config_.queryPoolBatchSize;

  return {pools_[pool_index], query_index};
}

void QueryPool::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  // vkResetQueryPool entered core in Vulkan 1.2; older loaders expose it
  // through the VK_EXT_host_query_reset entry point or not at all.  Resolve
  // the pointer once per process: a null implementation means recycling is
  // unsupported and the handed-out slots simply keep growing.
  static PFN_vkResetQueryPool reset_fn = reinterpret_cast<PFN_vkResetQueryPool>(
      vkGetDeviceProcAddr(device_, "vkResetQueryPool"));

  if (reset_fn == nullptr) {
    return;
  }

  for (size_t i = 0; i < pools_.size(); ++i) {
    const uint32_t queries_this_pool =
        std::min<uint32_t>(
                 capacity_ - static_cast<uint32_t>(i) * config_.queryPoolBatchSize,
                 config_.queryPoolBatchSize);
    reset_fn(device_, pools_[i], 0u, queries_this_pool);
  }

  in_use_ = 0u;
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
