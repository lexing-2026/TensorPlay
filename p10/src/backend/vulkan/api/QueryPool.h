#pragma once

#ifdef USE_VULKAN

#include "Utils.h"
#include "vk_api.h"

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace api {

struct QueryPoolConfig final {
  uint32_t queryPoolInitialSize;
  uint32_t queryPoolBatchSize;
};

//
// Grows a pool of timestamp queries in batches and hands out
// (VkQueryPool, query index) pairs.  Every handed-out query is written with
// VK_QUERY_TYPE_TIMESTAMP values; a reset returns the full capacity of all
// allocated pools for reuse, which the caller does after GPU work has
// completed (typically at flush time, since the device is idle then).
//
class QueryPool final {
 public:
  explicit QueryPool(VkDevice, const QueryPoolConfig&);

  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;

  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;

  ~QueryPool();

 private:
  VkDevice device_;
  QueryPoolConfig config_;
  std::mutex mutex_;
  // Owned VkQueryPool handles; each pool backs `queryPoolBatchSize`
  // consecutive query slots.
  std::vector<VkQueryPool> pools_;
  uint32_t capacity_;
  uint32_t in_use_;

 public:
  // Reserves the next free query slot.  The returned index is relative to
  // the returned pool (always smaller than queryPoolBatchSize).
  std::pair<VkQueryPool, uint32_t> get_new_query();

  // Total query slots handed out since the last reset.
  inline uint32_t size() const {
    return in_use_;
  }

  // Returns all query slots for reuse.  Only legal once the GPU is idle;
  // resetting while queries are in flight invalidates results that are
  // still pending.
  void reset();

 private:
  void allocate_new_batch(const uint32_t);
};

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
