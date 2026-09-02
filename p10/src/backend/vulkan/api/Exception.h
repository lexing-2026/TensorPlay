#pragma once

#ifdef USE_VULKAN

#include "vk_api.h"
#include "../../../../include/Exception.h"

#include <string>
#include <vector>

#define VK_CHECK(f)                                                             \
  do {                                                                          \
    const VkResult _check_result = (f);                                         \
    TP_CHECK(                                                                   \
        _check_result == VK_SUCCESS,                                            \
        "Vulkan error: ",                                                       \
        #f,                                                                     \
        " returned VkResult::",                                                 \
        tensorplay::vulkan::api::to_string(_check_result));                     \
  } while (0)

#define VK_CHECK_MSG(f, ...)                                                    \
  do {                                                                          \
    const VkResult _check_result = (f);                                         \
    TP_CHECK(                                                                   \
        _check_result == VK_SUCCESS,                                            \
        tensorplay::vulkan::api::make_error_message(                            \
            #f,                                                                 \
            tensorplay::vulkan::api::to_string(_check_result),                  \
            __VA_ARGS__));                                                      \
  } while (0)

#define VK_CHECK_COND(exp, ...)                                                 \
  do {                                                                          \
    if (!(exp)) {                                                               \
      TP_THROW(RuntimeError, __VA_ARGS__);                                      \
    }                                                                           \
  } while (0)

#define VK_THROW(...) TP_THROW(RuntimeError, __VA_ARGS__)

namespace tensorplay {
namespace vulkan {
namespace api {

const char* to_string(const VkResult result);

// Assembles the failure text used by VK_CHECK_MSG: the rejected call plus
// the returned result code, followed by any caller-supplied context.
std::string make_error_message(
    const std::string& function,
    const std::string& result,
    const std::string& message);

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
