#pragma once

//
// Do NOT include vk_mem_alloc.h directly.
// Always include this file (Allocator.h) instead.
//

#include "vk_api.h"

#ifdef USE_VULKAN

#define VMA_VULKAN_VERSION 1000000
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (32ull * 1024 * 1024)
#define VMA_SMALL_HEAP_MAX_SIZE (256ull * 1024 * 1024)

#define VMA_STATS_STRING_ENABLED 0

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Winconsistent-missing-destructor-override"
#endif /* __clang__ */

#include <vk_mem_alloc.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif /* __clang__ */

#endif /* USE_VULKAN */
