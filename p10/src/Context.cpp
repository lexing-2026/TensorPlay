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
    // Mirrors torch._C._set_default_dtype's check in
    // torch/csrc/tensor/python_tensor.cpp.
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
        "by calling 'torch.use_deterministic_algorithms(False)', or call "
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

void Context::pushDefaultDevice(Device device) {
    device_stack_.push_back(default_device_);
    default_device_ = device;
}

void Context::popDefaultDevice() {
    if (!device_stack_.empty()) {
        default_device_ = device_stack_.back();
        device_stack_.pop_back();
    } else {
        default_device_.reset();
    }
}

Context& globalContext() {
    static Context global_context;
    return global_context;
}

} // namespace tensorplay
