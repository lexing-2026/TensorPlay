#include "Context.h"
#include "Exception.h"

#include <algorithm>
#include <cctype>

namespace tensorplay {

std::string Float32MatmulPrecisionToString(Float32MatmulPrecision precision) {
    switch (precision) {
        case Float32MatmulPrecision::HIGHEST: return "highest";
        case Float32MatmulPrecision::HIGH: return "high";
        case Float32MatmulPrecision::MEDIUM: return "medium";
    }
    TP_THROW(RuntimeError, "unknown float32 matmul precision");
}

Float32MatmulPrecision Float32MatmulPrecisionFromString(const std::string& s) {
    if (s == "highest") return Float32MatmulPrecision::HIGHEST;
    if (s == "high") return Float32MatmulPrecision::HIGH;
    if (s == "medium") return Float32MatmulPrecision::MEDIUM;
    TP_THROW(RuntimeError,
             "float32_matmul_precision must be \"highest\", \"high\", or \"medium\"",
             " but got ", s);
}

bool isFloatingPoint(DType dtype) {
    switch (dtype) {
        case DType::Float16:
        case DType::BFloat16:
        case DType::Float32:
        case DType::Float64:
            return true;
        default:
            return false;
    }
}

void Context::setDefaultDType(DType dtype) {
    if (!isFloatingPoint(dtype)) {
        TP_THROW(TypeError, "invalid dtype object: only floating-point types are supported as the default type");
    }
    default_dtype_ = dtype;
}

void Context::alertNotDeterministic(const std::string& caller) const {
    static const std::string warn_msg =
        " does not have a deterministic implementation, but you set "
        "'use_deterministic_algorithms(True)'...";
    static const std::string error_msg =
        " does not have a deterministic implementation, but you set "
        "'use_deterministic_algorithms(True)'. You can turn off determinism "
        "by calling 'use_deterministic_algorithms(False)', or call "
        "'use_deterministic_algorithms(True, warn_only=True)' to only receive warnings.";
    if (deterministicAlgorithms()) {
        if (deterministicAlgorithmsWarnOnly()) {
            TP_WARN(caller + warn_msg);
        } else {
            TP_THROW(RuntimeError, caller + error_msg);
        }
    }
}

void Context::setAllowTF32CuBLAS(bool b) {
    setFloat32MatmulPrecision(b ? Float32MatmulPrecision::HIGH
                                : Float32MatmulPrecision::HIGHEST);
}

// The device-override slots live translation-unit local: thread-storage
// objects cannot carry a dll interface on Windows, so callers go through
// the exported Context accessors instead of touching the variables here.
namespace {
thread_local std::optional<Device> tp_default_device;
thread_local std::vector<std::optional<Device>> tp_device_stack;
} // namespace

void Context::setDefaultDevice(std::optional<Device> device) {
    tp_default_device = device;
}

void Context::clearDefaultDevice() { tp_default_device.reset(); }

std::optional<Device> Context::getDefaultDeviceOverride() const {
    return tp_default_device;
}

void Context::pushDefaultDevice(Device device) {
    tp_device_stack.push_back(tp_default_device);
    tp_default_device = device;
}

void Context::popDefaultDevice() {
    if (!tp_device_stack.empty()) {
        tp_default_device = tp_device_stack.back();
        tp_device_stack.pop_back();
    } else {
        tp_default_device.reset();
    }
}

Context& globalContext() {
    static Context global_context;
    return global_context;
}

} // namespace tensorplay
