#pragma once

// Shared dtype-dispatch and bounds-checking helpers for the sampling and
// factory kernels.

#include <cstdint>
#include <limits>
#include <type_traits>

#include "DType.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"

namespace tensorplay {
namespace distribution {

// Dispatch over all numeric storage dtypes (bool included).
template <typename Func>
inline void dispatch_dtype(DType dtype, Func&& fn) {
    switch (dtype) {
        case DType::UInt8: fn(uint8_t{}); break;
        case DType::Int8: fn(int8_t{}); break;
        case DType::Int16: fn(int16_t{}); break;
        case DType::Int32: fn(int32_t{}); break;
        case DType::Int64: fn(int64_t{}); break;
        case DType::UInt16: fn(uint16_t{}); break;
        case DType::UInt32: fn(uint32_t{}); break;
        case DType::UInt64: fn(uint64_t{}); break;
        case DType::Float32: fn(float{}); break;
        case DType::Float64: fn(double{}); break;
        case DType::Float16: fn(Half{}); break;
        case DType::BFloat16: fn(BFloat16{}); break;
        case DType::Bool: fn(bool{}); break;
        default:
            TP_THROW(NotImplementedError, "distribution does not support this dtype");
    }
}

// C-style spelling of each dtype, as surfaced by the
// "<name> is out of bounds for <dtype>" family of messages.
inline const char* bounds_dtype_name(DType dtype) {
    switch (dtype) {
        case DType::UInt8: return "unsigned char";
        case DType::Int8: return "signed char";
        case DType::Int16: return "short";
        case DType::Int32: return "int";
        case DType::Int64: return "long";
        case DType::UInt16: return "unsigned short";
        case DType::UInt32: return "unsigned int";
        case DType::UInt64: return "unsigned long";
        case DType::Float32: return "float";
        case DType::Float64: return "double";
        case DType::Float16: return "c10::Half";
        case DType::BFloat16: return "c10::BFloat16";
        case DType::Bool: return "bool";
        default: return "UNKNOWN_SCALAR";
    }
}

// Extremes of a floating-point storage dtype in double precision.
// std::numeric_limits is not specialized for the Half/BFloat16 wrappers.
template <typename T>
inline double fp_dtype_max() { return std::numeric_limits<double>::max(); }
template <> inline double fp_dtype_max<float>() { return std::numeric_limits<float>::max(); }
template <> inline double fp_dtype_max<double>() { return std::numeric_limits<double>::max(); }
// IEEE 754 binary16: (2 - 2^-10) * 2^15; bfloat16: (2 - 2^-7) * 2^127.
template <> inline double fp_dtype_max<Half>() { return 65504.0; }
template <> inline double fp_dtype_max<BFloat16>() {
    return std::ldexp(2.0 - std::ldexp(1.0, -7), 127);
}

template <typename T>
inline double fp_dtype_lowest() { return -fp_dtype_max<T>(); }

// 'from' and 'to - 1' of a discrete uniform draw over [from, to) must be
// representable in the destination dtype.
inline void check_random_from_to_bounds(int64_t low, int64_t high, DType dtype) {
    dispatch_dtype(dtype, [&](auto tag) {
        using scalar_t = decltype(tag);
        int64_t min;
        int64_t max;
        if constexpr (std::is_same_v<scalar_t, bool>) {
            min = 0;
            max = 1;
        } else if constexpr (std::is_same_v<scalar_t, uint64_t>) {
            min = 0;
            max = std::numeric_limits<int64_t>::max();
        } else if constexpr (std::is_unsigned_v<scalar_t>) {
            min = 0;
            max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        } else if constexpr (std::is_floating_point_v<scalar_t> ||
                             std::is_same_v<scalar_t, Half> ||
                             std::is_same_v<scalar_t, BFloat16>) {
            min = static_cast<int64_t>(-fp_dtype_max<scalar_t>());
            max = static_cast<int64_t>(fp_dtype_max<scalar_t>());
        } else {
            min = std::numeric_limits<scalar_t>::lowest();
            max = std::numeric_limits<scalar_t>::max();
        }
        TP_THROW_IF(low < min || low > max, RuntimeError,
                    "from is out of bounds for ", bounds_dtype_name(dtype));
        TP_THROW_IF(high - 1 < min || high - 1 > max, RuntimeError,
                    "to - 1 is out of bounds for ", bounds_dtype_name(dtype));
    });
}

} // namespace distribution
} // namespace tensorplay
