#ifdef USE_VULKAN

#include "Runtime.h"
#include "Exception.h"

#include <cstring>
#include <iostream>
#include <sstream>

namespace tensorplay {
namespace vulkan {
namespace api {

namespace {

void find_requested_layers_and_extensions(
    std::vector<const char*>& enabled_layers,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_layers,
    const std::vector<const char*>& requested_extensions) {
  // Get supported instance layers
  uint32_t layer_count = 0;
  VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));

  std::vector<VkLayerProperties> layer_properties(layer_count);
  VK_CHECK(vkEnumerateInstanceLayerProperties(
      &layer_count, layer_properties.data()));

  // Search for requested layers
  for (const auto& requested_layer : requested_layers) {
    for (const auto& layer : layer_properties) {
      if (strcmp(requested_layer, layer.layerName) == 0) {
        enabled_layers.push_back(requested_layer);
        break;
      }
    }
  }

  // Get supported instance extensions
  uint32_t extension_count = 0;
  VK_CHECK(vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_count, nullptr));

  std::vector<VkExtensionProperties> extension_properties(extension_count);
  VK_CHECK(vkEnumerateInstanceExtensionProperties(
      nullptr, &extension_count, extension_properties.data()));

  // Search for requested extensions
  for (const auto& requested_extension : requested_extensions) {
    for (const auto& extension : extension_properties) {
      if (strcmp(requested_extension, extension.extensionName) == 0) {
        enabled_extensions.push_back(requested_extension);
        break;
      }
    }
  }
}

VkInstance create_instance(const RuntimeConfiguration& config) {
  const VkApplicationInfo application_info{
      VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
      nullptr, // pNext
      "TensorPlay Vulkan Backend", // pApplicationName
      0, // applicationVersion
      nullptr, // pEngineName
      0, // engineVersion
      VK_API_VERSION_1_0, // apiVersion
  };

  std::vector<const char*> enabled_layers;
  std::vector<const char*> enabled_extensions;

  if (config.enableValidationMessages) {
    std::vector<const char*> requested_layers{
        "VK_LAYER_KHRONOS_validation",
    };
    std::vector<const char*> requested_extensions{};

    find_requested_layers_and_extensions(
        enabled_layers,
        enabled_extensions,
        requested_layers,
        requested_extensions);
  }

  const VkInstanceCreateInfo instance_create_info{
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      &application_info, // pApplicationInfo
      static_cast<uint32_t>(enabled_layers.size()), // enabledLayerCount
      enabled_layers.data(), // ppEnabledLayerNames
      static_cast<uint32_t>(enabled_extensions.size()), // enabledExtensionCount
      enabled_extensions.data(), // ppEnabledExtensionNames
  };

  VkInstance instance{};
  VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &instance));
  VK_CHECK_COND(instance, "Invalid Vulkan instance!");

  return instance;
}

std::vector<Runtime::DeviceMapping> create_physical_devices(
    VkInstance instance) {
  if (VK_NULL_HANDLE == instance) {
    return std::vector<Runtime::DeviceMapping>();
  }

  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));

  std::vector<VkPhysicalDevice> devices(device_count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));

  std::vector<Runtime::DeviceMapping> device_mappings;
  device_mappings.reserve(device_count);
  for (VkPhysicalDevice physical_device : devices) {
    device_mappings.emplace_back(PhysicalDevice(physical_device), -1);
  }

  return device_mappings;
}

//
// Adapter selection methods
//

uint32_t select_first(const std::vector<Runtime::DeviceMapping>& devices) {
  if (devices.empty()) {
    return devices.size() + 1; // return out of range to signal invalidity
  }

  // Select the first adapter that has compute capability
  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i].first.num_compute_queues > 0) {
      return i;
    }
  }

  return devices.size() + 1;
}

//
// Global runtime initialization
//

std::unique_ptr<Runtime> init_global_vulkan_runtime() {
  const bool enableValidationMessages =
#if defined(VULKAN_DEBUG)
      true;
#else
      false;
#endif /* VULKAN_DEBUG */
  const bool initDefaultDevice = true;
  const uint32_t numRequestedQueues = 1; // TODO: raise this value

  const RuntimeConfiguration default_config{
      enableValidationMessages,
      initDefaultDevice,
      AdapterSelector::First,
      numRequestedQueues,
  };

  try {
    return std::make_unique<Runtime>(Runtime(default_config));
  } catch (...) {
  }

  return std::unique_ptr<Runtime>(nullptr);
}

} // namespace

Runtime::Runtime(const RuntimeConfiguration config)
    : config_(config),
      instance_(create_instance(config_)),
      device_mappings_(create_physical_devices(instance_)),
      adapters_{},
      default_adapter_i_(UINT32_MAX) {
  // List of adapters will never exceed the number of physical devices
  adapters_.reserve(device_mappings_.size());

  if (config.initDefaultDevice) {
    try {
      switch (config.defaultSelector) {
        case AdapterSelector::First:
          default_adapter_i_ = create_adapter(select_first);
      }
    } catch (...) {
    }
  }
}

Runtime::~Runtime() {
  if (VK_NULL_HANDLE == instance_) {
    return;
  }

  // Clear adapters list to trigger device destruction before destroying
  // VkInstance
  adapters_.clear();

  vkDestroyInstance(instance_, nullptr);
  instance_ = VK_NULL_HANDLE;
}

Runtime::Runtime(Runtime&& other) noexcept
    : config_(other.config_),
      instance_(other.instance_),
      device_mappings_(std::move(other.device_mappings_)),
      adapters_(std::move(other.adapters_)),
      default_adapter_i_(other.default_adapter_i_) {
  other.instance_ = VK_NULL_HANDLE;
  other.device_mappings_.clear();
  other.adapters_.clear();
  other.default_adapter_i_ = UINT32_MAX;
}

uint32_t Runtime::create_adapter(const Selector& selector) {
  VK_CHECK_COND(
      !device_mappings_.empty(),
      "TensorPlay Vulkan Runtime: Could not initialize adapter because no "
      "devices were found by the Vulkan instance.");

  uint32_t physical_device_i = selector(device_mappings_);
  VK_CHECK_COND(
      physical_device_i < device_mappings_.size(),
      "TensorPlay Vulkan Runtime: no suitable device adapter was selected! "
      "Device could not be initialized");

  Runtime::DeviceMapping& device_mapping = device_mappings_[physical_device_i];
  // If an Adapter has already been created, return that
  int32_t adapter_i = device_mapping.second;
  if (adapter_i >= 0) {
    return adapter_i;
  }
  // Otherwise, create an adapter for the selected physical device
  adapter_i = static_cast<int32_t>(adapters_.size());
  adapters_.emplace_back(
      new Adapter(instance_, device_mapping.first, config_.numRequestedQueues));
  device_mapping.second = adapter_i;

  return adapter_i;
}

Runtime* try_runtime() {
  // The global vulkan runtime is declared as a static local variable within a
  // non-static function to ensure it has external linkage. If it were a global
  // static variable there would be one copy per translation unit that includes
  // Runtime.h as it would have internal linkage.
  static const std::unique_ptr<Runtime> p_runtime =
      init_global_vulkan_runtime();

  if (!p_runtime || !p_runtime->is_initialized()) {
    return nullptr;
  }

  return p_runtime.get();
}

Runtime* runtime() {
  Runtime* p_runtime = try_runtime();
  VK_CHECK_COND(
      p_runtime,
      "TensorPlay Vulkan Runtime: The global runtime could not be retrieved "
      "because it failed to initialize.");

  return p_runtime;
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
