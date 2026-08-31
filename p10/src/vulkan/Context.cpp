#include "vulkan/Context.h"

#include "Dispatcher.h"
#include "Exception.h"

#include <atomic>

namespace tensorplay {
namespace vulkan {

std::atomic<const ImplInterface*> g_vulkan_impl_registry;

ImplRegistrar::ImplRegistrar(ImplInterface* impl) {
  g_vulkan_impl_registry.store(impl);
}

Tensor& vulkan_copy_(Tensor& self, const Tensor& src) {
  auto p = g_vulkan_impl_registry.load();
  if (p) {
    return p->vulkan_copy_(self, src);
  }
  TP_THROW(RuntimeError, "Vulkan backend was not linked to the build");
}
} // namespace vulkan

namespace ops {
bool is_vulkan_available() {
  auto p = vulkan::g_vulkan_impl_registry.load();
  return p ? p->is_vulkan_available() : false;
}

TENSORPLAY_LIBRARY_IMPL(Composite, Vulkan) {
  m.impl("is_vulkan_available", &ops::is_vulkan_available);
}
} // namespace ops

} // namespace tensorplay
