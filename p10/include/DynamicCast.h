#pragma once

// Dynamic casting for elementwise kernels: convert between the runtime
// dtype carried by a ScalarType tag and the functor's static C++ argument
// types.  One kernel instantiation then serves every dtype combination a
// promotion rule can produce.  The dtype switch is uniform within a warp
// (no lane divergence), so the extra comparisons ride under memory latency.

#include "DType.h"
#include "Half.h"
#include "BFloat16.h"
#include "Float8_e4m3fn.h"
#include "Float8_e5m2.h"
#include "Float8_e4m3fnuz.h"
#include "Float8_e5m2fnuz.h"
#include "Float8_e8m0fnu.h"
#include "Exception.h"

#include <cassert>
#include <complex>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
// The whole translation unit mixes host and device code, so these helpers
// must be callable from either side in every compilation pass.
#define TP_DYNAMIC_CAST_HOST_DEVICE __host__ __device__
#define TP_DYNAMIC_CAST_ERROR(msg) assert(false && (msg));
#else
#define TP_DYNAMIC_CAST_HOST_DEVICE
#define TP_DYNAMIC_CAST_ERROR(msg) TP_THROW(TypeError, msg);
#endif

namespace tensorplay {

namespace detail {

// Element-aligned accesses only: TensorIterator operand data pointers and
// offsets are always a multiple of the element size.
template <typename T>
TP_DYNAMIC_CAST_HOST_DEVICE inline T dynamic_load(const void* ptr) {
    return *reinterpret_cast<const T*>(ptr);
}

template <typename dest_t, typename src_t>
TP_DYNAMIC_CAST_HOST_DEVICE inline dest_t scalar_convert(src_t v) {
    if constexpr (is_complex_type_v<dest_t> && is_complex_type_v<src_t>) {
        using dv_t = typename is_complex_type<dest_t>::value_type;
        return dest_t(detail::scalar_cast<dv_t>(v.real()),
                      detail::scalar_cast<dv_t>(v.imag()));
    } else if constexpr (is_complex_type_v<dest_t>) {
        using dv_t = typename is_complex_type<dest_t>::value_type;
        return dest_t(detail::scalar_cast<dv_t>(v), dv_t(0));
    } else if constexpr (is_complex_type_v<src_t>) {
        return detail::scalar_cast<dest_t>(v);
    } else {
        return detail::scalar_cast<dest_t>(v);
    }
}

}  // namespace detail

#define TP_FETCH_AND_CAST_CASE(ctype, name)                        \
    case ScalarType::name:                                         \
        return detail::scalar_convert<dest_t>(                     \
            detail::dynamic_load<ctype>(ptr));

// Fetch a value with dynamic type src_type from ptr and convert it to the
// static type dest_t.
template <typename dest_t>
TP_DYNAMIC_CAST_HOST_DEVICE dest_t fetch_and_cast(const ScalarType src_type,
                                                  const void* ptr) {
    switch (src_type) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_FETCH_AND_CAST_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_FETCH_AND_CAST_CASE)
        default:
            break;
    }
    TP_DYNAMIC_CAST_ERROR("Unexpected scalar type in fetch_and_cast")
    return dest_t(0);
}

#define TP_CAST_AND_STORE_CASE(ctype, name)                        \
    case ScalarType::name:                                         \
        *reinterpret_cast<ctype*>(ptr) =                           \
            detail::scalar_convert<ctype>(value);                  \
        return;

// Convert a statically typed value into the dynamic type dest_type and
// store it to ptr.
template <typename src_t>
TP_DYNAMIC_CAST_HOST_DEVICE void cast_and_store(const ScalarType dest_type,
                                                void* ptr, src_t value) {
    switch (dest_type) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_CAST_AND_STORE_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_CAST_AND_STORE_CASE)
        default:
            break;
    }
    TP_DYNAMIC_CAST_ERROR("Unexpected scalar type in cast_and_store")
}

}  // namespace tensorplay
