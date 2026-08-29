#pragma once

#include <optional>
#include <vector>

#include "Autograd.h"
#include "GradMode.h"
#include "Tensor.h"
#include "autocast_mode.h"

namespace tensorplay {
namespace autocast {

// state lives in p10's autocast_mode.h; the casting helpers live in tpx
// because differentiability comes from the differentiable `to`

// Policies correspond to op categories that need code-divergent handling.
enum class CastPolicy : uint8_t {
    lower_precision_fp = 0, // Cast all inputs to lower_precision_fp before
                            // running the op. Currently, lower_precision_fp
                            // is fp16 for AutocastCUDA, and is defined by user
                            // (default bf16) for AutocastCPU.
    fp32,               // Cast all inputs to DType::Float32 before running the op.
    fp32_set_opt_dtype, // Treats functions (like softmax) that
                        //  1. we'd like to run in fp32 and
                        //  2. have a dtype arg that controls the output type.
                        // Wrappers' policy is: if the output type is already
                        // set, don't touch it, otherwise set it to Float32.
    promote,            // Run in the widest dtype among several args.
};

// ------------------------------------------------------------------
// ------------------------------------------------------------------

inline bool is_autocast_eligible(const Tensor& tensor, DeviceType device_type) {
    // TP alignment: Tensor exposes device()/dtype() rather than is_cuda()/is_floating_point()
    const Device dev = tensor.device();
    const DType dt = tensor.dtype();
    switch (device_type) {
        case DeviceType::CUDA:
            return dev.is_cuda() && isFloatingType(dt);
        case DeviceType::CPU:
            return dev.is_cpu() && isFloatingType(dt);
        default:
            return false;
    }
}

inline DType get_lower_precision_fp_from_device_type(DeviceType device_type) {
    TP_CHECK(
        is_autocast_available(device_type),
        "unknown device type for autocast in get_lower_precision_fp_from_device_type");
    return get_autocast_dtype(device_type);
}

inline bool is_eligible(const Tensor& arg, DeviceType device_type) {
    return (
        arg.defined() && is_autocast_eligible(arg, device_type) &&
        (arg.dtype() != DType::Float64));
}

// ------------------------------------------------------------------
// Logic to extract the promote type from any Tensor or TensorList args
// ------------------------------------------------------------------

inline DType prioritize(DType current, const Tensor& nextArg, DeviceType device_type) {
    if (current == DType::Float64) {
        TP_CHECK(false, "promote type is double in promote_type");
        return current;
    }
    DType lower_precision_fp = get_lower_precision_fp_from_device_type(device_type);
    if (is_autocast_eligible(nextArg, device_type)) {
        auto next = nextArg.dtype();
        if (next == DType::Float64) {
            return current; // ignores double tensors
        } else if (current == DType::Float32 || next == DType::Float32) {
            return DType::Float32; // prioritizes float over lower_precision_fp
        } else if (current == lower_precision_fp && next == lower_precision_fp) {
            return lower_precision_fp;
        } else if ((next == DType::Float16 || next == DType::BFloat16) &&
                   current == lower_precision_fp) {
            // Mixed low-precision pair (e.g. fp16 inputs under cpu autocast):
            // fold into the device's lower-precision family instead of
            // rejecting -- matches the repo's amp contract for promote ops.
            return lower_precision_fp;
        } else {
            TP_CHECK(false, "Unexpected floating ScalarType in promote_type");
            return current;
        }
    } else {
        return current;
    }
}

inline DType prioritize(DType current, const std::vector<Tensor>& list, DeviceType device_type) {
    for (const auto& tensor : list) {
        current = prioritize(current, tensor, device_type);
    }
    return current;
}

template <typename T>
inline DType prioritize(DType current, T /*nextArg*/, DeviceType /*device_type*/) {
    return current;
}

// Overload for the tail case.
inline DType promote_type(DType current, DeviceType /*device_type*/) {
    return current;
}

// Unpack args and determine if incoming lower_precision_fp tensors need to be
// promoted to float32. Non-Tensor arguments are ignored.
template <typename Arg0, typename... Args>
inline DType promote_type(DType current, DeviceType device_type, Arg0 arg0, Args... args) {
    auto new_current = prioritize(current, arg0, device_type);
    return promote_type(new_current, device_type, args...);
}

// ------------------------------------------------------------------
// differentiable through `to` (ToCopyBackward), so no custom node is needed.
// ------------------------------------------------------------------

P10_API Tensor cached_cast(DType to_type, const Tensor& arg, DeviceType device_type);

inline std::optional<Tensor> cached_cast(
    DType to_type,
    const std::optional<Tensor>& arg,
    DeviceType device_type) {
    if (arg.has_value()) {
        return cached_cast(to_type, *arg, device_type);
    } else {
        return std::nullopt;
    }
}

inline std::vector<Tensor> cached_cast(
    DType to_type,
    const std::vector<Tensor>& arg,
    DeviceType device_type) {
    std::vector<Tensor> vec;
    vec.reserve(arg.size());
    for (const auto& t : arg) {
        vec.emplace_back(cached_cast(to_type, t, device_type));
    }
    return vec;
}

// Template to catch non-Tensor args.
template <typename T>
inline T cached_cast(DType /*to_type*/, T arg, DeviceType /*device_type*/) {
    return arg;
}

// ------------------------------------------------------------------
// If the user has explicitly specified a dtype, respect it. Otherwise, set it
// to the requested type.
// ------------------------------------------------------------------

inline DType set_opt_dtype(DType to_type, DType dtype) {
    return dtype == DType::Undefined ? to_type : dtype;
}

template <typename T>
inline T set_opt_dtype(DType /*to_type*/, T arg) {
    return arg;
}

template <typename... Args>
inline bool firstarg_is_eligible(DeviceType device_type, const Tensor& arg, Args... /*args*/) {
    return is_eligible(arg, device_type);
}

template <typename... Args>
inline DType type_from_firstarg(DeviceType device_type, DType to_type, const Tensor& arg, Args... /*args*/) {
    return (is_eligible(arg, device_type) ? to_type : arg.dtype());
}

} // namespace autocast
} // namespace tensorplay
