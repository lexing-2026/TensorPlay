#pragma once

#include <atomic>

#include "Tensor.h"

namespace tensorplay {
namespace vulkan {

//
// Registry through which the (optionally compiled) Vulkan backend exposes
// its availability probe and cross-device copy entry point to code that
// must link regardless of backend support.  When no backend registered,
// availability is false and the copy entry point raises.
//
struct ImplInterface {
  virtual ~ImplInterface() = default;
  virtual bool is_vulkan_available() const = 0;
  virtual Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const = 0;
};

extern std::atomic<const ImplInterface*> g_vulkan_impl_registry;

class ImplRegistrar {
 public:
  explicit ImplRegistrar(ImplInterface* impl);
};

Tensor& vulkan_copy_(Tensor& self, const Tensor& src);
} // namespace vulkan

namespace ops {
bool is_vulkan_available();
} // namespace ops

} // namespace tensorplay
