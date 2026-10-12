#pragma once

#include "DispatchStub.h"
#include "DType.h"
#include <cstdint>

namespace tensorplay::cpu {
using quantized_clip_fn = void (*)(const void*, void*, int64_t, DType, int64_t, int64_t);
using quantized_lookup_fn = void (*)(const uint8_t*, uint8_t*, int64_t, const float*);
DECLARE_DISPATCH(quantized_clip_fn, quantized_clip_stub)
DECLARE_DISPATCH(quantized_lookup_fn, quantized_lookup_stub)
}
