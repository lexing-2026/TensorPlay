#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

/*
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 */

namespace tensorplay::cuda::activation_math {

__device__ inline float expm1_scalar(float x) {
    const unsigned bits = __float_as_uint(x), magnitude = bits & 0x7fffffffU;
    const bool negative = bits >> 31;
    if (magnitude >= 0x4195b844U) {
        if (magnitude > 0x7f800000U) return x;
        if (magnitude == 0x7f800000U) return negative ? -1.f : x;
        if (x > 88.721679688f) return CUDART_INF_F;
        if (negative) return -1.f;
    }
    float correction = 0.f;
    int exponent = 0;
    if (magnitude > 0x3eb17218U) {
        float high, low;
        if (magnitude < 0x3f851592U) {
            high = negative ? __fadd_rn(x, .69313812256f) : __fsub_rn(x, .69313812256f);
            low = negative ? -9.0580006145e-6f : 9.0580006145e-6f;
            exponent = negative ? -1 : 1;
        } else {
            exponent = __float2int_rz(__fadd_rn(__fmul_rn(1.4426950216f, x), negative ? -.5f : .5f));
            high = __fsub_rn(x, __fmul_rn(static_cast<float>(exponent), .69313812256f));
            low = __fmul_rn(static_cast<float>(exponent), 9.0580006145e-6f);
        }
        x = __fsub_rn(high, low);
        correction = __fsub_rn(__fsub_rn(high, x), low);
    } else if (magnitude < 0x33000000U) return x;
    const float half = __fmul_rn(.5f, x), square_half = __fmul_rn(x, half);
    float polynomial = -2.0109921195e-7f;
    polynomial = __fadd_rn(4.0082177293e-6f, __fmul_rn(square_half, polynomial));
    polynomial = __fadd_rn(-7.9365076090e-5f, __fmul_rn(square_half, polynomial));
    polynomial = __fadd_rn(1.5873016091e-3f, __fmul_rn(square_half, polynomial));
    polynomial = __fadd_rn(-3.3333335072e-2f, __fmul_rn(square_half, polynomial));
    const float r = __fadd_rn(1.f, __fmul_rn(square_half, polynomial));
    const float t = __fsub_rn(3.f, __fmul_rn(r, half));
    float error = __fmul_rn(square_half, __fdiv_rn(__fsub_rn(r, t), __fsub_rn(6.f, __fmul_rn(x, t))));
    if (!exponent) return __fsub_rn(x, __fsub_rn(__fmul_rn(x, error), square_half));
    error = __fsub_rn(__fmul_rn(x, __fsub_rn(error, correction)), correction);
    error = __fsub_rn(error, square_half);
    if (exponent == -1) return __fsub_rn(__fmul_rn(.5f, __fsub_rn(x, error)), .5f);
    if (exponent == 1) {
        if (x < -.25f) return __fmul_rn(-2.f, __fsub_rn(error, __fadd_rn(x, .5f)));
        return __fadd_rn(1.f, __fmul_rn(2.f, __fsub_rn(x, error)));
    }
    float value;
    if (exponent <= -2 || exponent > 56) {
        value = __fsub_rn(1.f, __fsub_rn(error, x));
        value = __uint_as_float(__float_as_uint(value) + static_cast<unsigned>(exponent * 0x800000));
        return __fsub_rn(value, 1.f);
    }
    if (exponent < 23) {
        const float offset = __uint_as_float(0x3f800000U - (0x1000000U >> exponent));
        value = __fsub_rn(offset, __fsub_rn(error, x));
    } else {
        const float offset = __uint_as_float(static_cast<unsigned>(127 - exponent) << 23);
        value = __fadd_rn(__fsub_rn(x, __fadd_rn(error, offset)), 1.f);
    }
    return __uint_as_float(__float_as_uint(value) + static_cast<unsigned>(exponent * 0x800000));
}

__device__ inline float tanh_scalar(float x) {
    const unsigned bits = __float_as_uint(x), magnitude = bits & 0x7fffffffU;
    if (magnitude > 0x7f800000U) return x;
    if (magnitude >= 0x41b00000U) return ::copysignf(1.f, x);
    if (!magnitude) return x;
    if (magnitude < 0x24000000U) return __fmul_rn(x, __fadd_rn(1.f, x));
    const float absolute = ::fabsf(x);
    float value;
    if (magnitude >= 0x3f800000U) {
        const float exponential = expm1_scalar(__fmul_rn(2.f, absolute));
        value = __fsub_rn(1.f, __fdiv_rn(2.f, __fadd_rn(exponential, 2.f)));
    } else {
        const float exponential = expm1_scalar(__fmul_rn(-2.f, absolute));
        value = __fdiv_rn(-exponential, __fadd_rn(exponential, 2.f));
    }
    return ::copysignf(value, x);
}

} // namespace tensorplay::cuda::activation_math
