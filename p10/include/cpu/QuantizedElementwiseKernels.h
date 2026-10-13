#pragma once

#include "DispatchStub.h"
#include "DType.h"
#include "Half.h"
#include <cstdint>

namespace tensorplay::quant { struct PoolGeometry; struct PoolWindow; }

namespace tensorplay::cpu {
using quantized_subbyte_pack_fn = void (*)(const float*,uint8_t*,int64_t,float,float,int);
using quantized_subbyte_unpack_fn = void (*)(const uint8_t*,float*,int64_t,float,float,int);
DECLARE_DISPATCH(quantized_subbyte_pack_fn, quantized_subbyte_pack_stub)
DECLARE_DISPATCH(quantized_subbyte_unpack_fn, quantized_subbyte_unpack_stub)
using quantized_unsigned_pack_fn = void (*)(const float*, uint8_t*, int64_t, float, int64_t);
DECLARE_DISPATCH(quantized_unsigned_pack_fn, quantized_unsigned_pack_stub)
using quantized_dynamic_dot_fn = int64_t (*)(const uint8_t*,const int8_t*,int64_t);
DECLARE_DISPATCH(quantized_dynamic_dot_fn, quantized_dynamic_dot_stub)
using quantized_dynamic_block_fn = void (*)(const uint8_t*,const int8_t*,int64_t*,
    int64_t,int64_t,int64_t,int64_t);
DECLARE_DISPATCH(quantized_dynamic_block_fn, quantized_dynamic_block_stub)
using quantized_linear_epilogue_fn = void (*)(const int32_t*, const int64_t*,
    const float*, const int64_t*, const float*, void*, DType, int64_t, int64_t,
    int64_t, int64_t, double, double, int64_t, bool, bool, int64_t, double);
DECLARE_DISPATCH(quantized_linear_epilogue_fn, quantized_linear_epilogue_stub)
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
using quantized_moment_fn = void (*)(const void*, void*, int64_t, int64_t, DType,
    double, int64_t, bool, double);
DECLARE_DISPATCH(quantized_moment_fn, quantized_moment_stub)
using quantized_spatial_mean_fn = void (*)(const void*, void*, int64_t, int64_t,
    int64_t, int64_t, DType, float, int64_t);
DECLARE_DISPATCH(quantized_spatial_mean_fn, quantized_spatial_mean_stub)
using quantized_pool_fn = void (*)(const void*, void*, int64_t, int64_t, DType,
    const quant::PoolGeometry&, const quant::PoolWindow*);
DECLARE_DISPATCH(quantized_pool_fn, quantized_pool_stub)
using quantized_prelu_fn = void (*)(const void*, const void*, void*, int64_t, bool, DType,
    int64_t, int64_t, int64_t, int64_t, float);
DECLARE_DISPATCH(quantized_prelu_fn, quantized_prelu_stub)
using quantized_float_prelu_fn = void (*)(const void*, const float*, uint8_t*, int64_t,
    bool, DType, int64_t, float, float, int64_t);
DECLARE_DISPATCH(quantized_float_prelu_fn, quantized_float_prelu_stub)
using quantized_word_activation_fn = void (*)(const int32_t*, int32_t*, int64_t, float, int64_t, double, int64_t, bool);
DECLARE_DISPATCH(quantized_word_activation_fn, quantized_sigmoid_stub)
DECLARE_DISPATCH(quantized_word_activation_fn, quantized_tanh_stub)
using quantized_clip_fn = void (*)(const void*, void*, int64_t, DType, int64_t, int64_t);
using quantized_lookup_fn = void (*)(const uint8_t*, uint8_t*, int64_t, const float*);
using quantized_softmax_fn = void (*)(const void*, void*, DType, int64_t, int64_t,
    int64_t, int64_t, float, float);
DECLARE_DISPATCH(quantized_softmax_fn, quantized_softmax_stub)
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
