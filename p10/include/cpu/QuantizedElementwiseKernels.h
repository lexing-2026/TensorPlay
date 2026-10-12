#pragma once

#include "DispatchStub.h"
#include "DType.h"
#include "Half.h"
#include <cstdint>

namespace tensorplay::cpu {
using quantized_bilinear_fn = void (*)(const void*, const void*, const void*, const void*,
    void*, int64_t, DType, float, float, int64_t);
DECLARE_DISPATCH(quantized_bilinear_fn, quantized_bilinear_stub)
using quantized_clip_fn = void (*)(const void*, void*, int64_t, DType, int64_t, int64_t);
using quantized_lookup_fn = void (*)(const uint8_t*, uint8_t*, int64_t, const float*);
using quantized_binary_fn = void (*)(const void*, const void*, void*, int64_t, DType,
    double, int64_t, double, int64_t, double, int64_t, int, bool);
using quantized_fp16_linear_fn = void (*)(const float*, const Half*, const float*, float*,
    int64_t, int64_t, int64_t, int64_t, bool);
using quantized_embedding_fn = void (*)(const uint8_t*, const int64_t*, const int64_t*,
    const int64_t*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool);
DECLARE_DISPATCH(quantized_embedding_fn, quantized_embedding_stub)
DECLARE_DISPATCH(quantized_binary_fn, quantized_binary_stub)
DECLARE_DISPATCH(quantized_fp16_linear_fn, quantized_fp16_linear_stub)
DECLARE_DISPATCH(quantized_clip_fn, quantized_clip_stub)
DECLARE_DISPATCH(quantized_lookup_fn, quantized_lookup_stub)
}
