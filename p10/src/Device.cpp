#include "Device.h"
#include "Exception.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>

namespace tensorplay {

namespace {

constexpr const char* kPrivateUse1DefaultName = "privateuseone";

// Written once under the lock, then published by the flag: a reader that
// sees the flag set reads a name that never changes again.
std::atomic<bool> privateuse1_name_set{false};
std::string privateuse1_name;
std::mutex privateuse1_lock;

std::string lowered(const std::string& text) {
    std::string out = text;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

// Spellings of the in-tree device types.  "vk" is the short spelling of
// Vulkan; "privateuseone" stays accepted after the slot is renamed.
DeviceType in_tree_device_type(const std::string& name, bool* found) {
    *found = true;
    if (name == "cpu") return DeviceType::CPU;
    if (name == "cuda") return DeviceType::CUDA;
    if (name == "vulkan" || name == "vk") return DeviceType::Vulkan;
    if (name == "meta") return DeviceType::Meta;
    if (name == "mps") return DeviceType::MPS;
    if (name == "hip") return DeviceType::HIP;
    if (name == "mtia") return DeviceType::MTIA;
    if (name == "xpu") return DeviceType::XPU;
    if (name == "hpu") return DeviceType::HPU;
    if (name == kPrivateUse1DefaultName) return DeviceType::PrivateUse1;
    *found = false;
    return DeviceType::Unknown;
}

// Device types whose spelling defaults to index 0 when none is given.
bool defaults_to_first_index(DeviceType type) {
    return type == DeviceType::CUDA || type == DeviceType::Vulkan ||
           type == DeviceType::MPS || type == DeviceType::PrivateUse1;
}

// Device types that print their index.
bool prints_index(DeviceType type) {
    return type == DeviceType::CPU || defaults_to_first_index(type);
}

} // namespace

void register_privateuse1_backend(const std::string& backend_name) {
    const std::string name = lowered(backend_name);
    TP_CHECK_VALUE(!name.empty(), "the out-of-tree backend name must not be empty");
    TP_CHECK_VALUE(std::all_of(name.begin(), name.end(),
                         [](unsigned char c) {
                             return std::isalnum(c) || c == '_';
                         }) &&
                 !std::isdigit(static_cast<unsigned char>(name[0])),
             "the out-of-tree backend name must be an identifier "
             "(letters, digits and underscores, not starting with a digit), "
             "got '", backend_name, "'");
    bool in_tree = false;
    (void)in_tree_device_type(name, &in_tree);
    TP_CHECK_VALUE(!in_tree || name == kPrivateUse1DefaultName,
             "cannot name the out-of-tree backend after the in-tree device "
             "type '", name, "'");
    std::lock_guard<std::mutex> guard(privateuse1_lock);
    if (privateuse1_name_set.load(std::memory_order_acquire)) {
        TP_CHECK_VALUE(privateuse1_name == name,
                 "the out-of-tree backend is already registered as '",
                 privateuse1_name, "'; it cannot be renamed to '", name, "'");
        return;
    }
    privateuse1_name = name;
    privateuse1_name_set.store(true, std::memory_order_release);
}

bool is_privateuse1_backend_registered() {
    return privateuse1_name_set.load(std::memory_order_acquire);
}

std::string get_privateuse1_backend(bool lower_case) {
    std::string name = is_privateuse1_backend_registered()
        ? privateuse1_name
        : std::string(kPrivateUse1DefaultName);
    if (!lower_case) {
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return std::toupper(c); });
    }
    return name;
}

std::string device_type_name(DeviceType type) {
    switch (type) {
        case DeviceType::PrivateUse1:
            return get_privateuse1_backend(true);
        default:
            break;
    }
    switch (type) {
        #define CASE_DEVICE_TYPE(name)          \
            case DeviceType::name:              \
                return lowered(#name);
        TENSORPLAY_FORALL_DEVICE_TYPES(CASE_DEVICE_TYPE)
        #undef CASE_DEVICE_TYPE
    }
    return "unknown";
}

DeviceType parse_device_type(const std::string& name) {
    const std::string s = lowered(name);
    bool found = false;
    const DeviceType type = in_tree_device_type(s, &found);
    if (found) return type;
    if (is_privateuse1_backend_registered() && s == privateuse1_name) {
        return DeviceType::PrivateUse1;
    }
    TP_THROW(ValueError, "Invalid device type: " + name);
}

std::string Device::toString() const {
    std::string result = device_type_name(type_);
    if (prints_index(type_) && index_ >= 0) {
        result += ":" + std::to_string(index_);
    }
    return result;
}

Device::Device(const std::string& device_str) : type_(DeviceType::CPU), index_(-1) {
    const std::string s = lowered(device_str);
    const size_t colon = s.find(':');
    const std::string type_part = s.substr(0, colon);
    try {
        type_ = parse_device_type(type_part);
    } catch (const ValueError&) {
        TP_THROW(ValueError, "Invalid device string: " + device_str);
    }
    if (type_ == DeviceType::Meta && colon != std::string::npos) {
        // The meta device is unindexed: it holds shapes only, so there is no
        // physical slot a ":n" suffix could name.
        TP_THROW(ValueError, "Invalid device string: " + device_str);
    }
    if (colon != std::string::npos) {
        const std::string index_part = s.substr(colon + 1);
        size_t consumed = 0;
        try {
            index_ = std::stoi(index_part, &consumed);
        } catch (...) {
            TP_THROW(ValueError, "Invalid device string: " + device_str);
        }
        if (consumed != index_part.size() || index_ < 0) {
            TP_THROW(ValueError, "Invalid device string: " + device_str);
        }
    } else if (defaults_to_first_index(type_)) {
        // An accelerator spelled without an index names its first device,
        // so "cuda" and "cuda:0" compare equal.
        index_ = 0;
    }
}

Device::Device(const std::string& type_str, int64_t index) : index_(index) {
    type_ = parse_device_type(type_str);
}

} // namespace tensorplay
