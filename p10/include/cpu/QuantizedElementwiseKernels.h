#pragma once

#include "DispatchStub.h"
#include "DType.h"
#include "Half.h"
#include <cstdint>

namespace tensorplay::cpu {
using quantized_dynamic_dot_fn = int64_t (*)(const uint8_t*,const int8_t*,int64_t);
DECLARE_DISPATCH(quantized_dynamic_dot_fn, quantized_dynamic_dot_stub)
using quantized_dynamic_block_fn = void (*)(const uint8_t*,const int8_t*,int64_t*,
    int64_t,int64_t,int64_t,int64_t);
DECLARE_DISPATCH(quantized_dynamic_block_fn, quantized_dynamic_block_stub)
using quantized_recurrent_step_fn = void (*)(const float*,const float*,const float*,
    const float*,float*,float*,int64_t,int64_t,int);
DECLARE_DISPATCH(quantized_recurrent_step_fn, quantized_recurrent_step_stub)
using quantized_int4_pack_fn = void (*)(const int32_t*, uint8_t*, int64_t, int64_t);
using quantized_int4_mm_fn = void (*)(const void*, const uint8_t*, const void*, void*,
    DType, int64_t, int64_t, int64_t, int64_t);
DECLARE_DISPATCH(quantized_int4_pack_fn, quantized_int4_pack_stub)
DECLARE_DISPATCH(quantized_int4_mm_fn, quantized_int4_mm_stub)
using quantized_batch_norm_fn = void (*)(const void*, void*, int64_t, DType,
    const float*, const float*, int64_t, int64_t, bool, int64_t);
DECLARE_DISPATCH(quantized_batch_norm_fn, quantized_batch_norm_stub)
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
