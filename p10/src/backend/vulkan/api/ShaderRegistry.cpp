#ifdef USE_VULKAN

#include "ShaderRegistry.h"
#include "Exception.h"

namespace tensorplay {
namespace vulkan {
namespace api {

namespace {

ShaderRegistry& get_registry() {
  static ShaderRegistry registry;
  return registry;
}

} // namespace

bool ShaderRegistry::has_shader(const std::string& shader_name) {
  return listings_.count(shader_name) > 0;
}

void ShaderRegistry::register_shader(ShaderInfo&& shader_info) {
  const std::string name = shader_info.kernel_name;
  listings_.emplace(name, std::move(shader_info));
}

const ShaderInfo& ShaderRegistry::get_shader_info(
    const std::string& shader_name) {
  const auto it = listings_.find(shader_name);
  VK_CHECK_COND(
      it != listings_.end(),
      "Vulkan shader not registered: ",
      shader_name);
  return it->second;
}

ShaderRegistry& shader_registry() {
  return get_registry();
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
