#pragma once

// The two flat loops that answer a float-math map -- one over a single tensor,
// one over a pair broadcast to a common width -- together with the judgement
// each makes about a destination the caller already holds.
//
// Both loops evaluate in Float32 (or Float64 for a double pair) and narrow once
// at the end, so a destination only takes the pass when it already carries the
// answer at the evaluated width; everything else -- a reduced width, a cast, a
// resize, an alias the loop would read after storing -- is left to the handover
// in OutWrite.h.  The loops answer the same question either way; the destination
// only decides whether the answer is written once or twice.

#include <vector>

#include "DType.h"
#include "Exception.h"
#include "OutWrite.h"
#include "Parallel.h"
#include "Tensor.h"
#include "cpu/VecUnary.h"
#include "TypePromotion.h"
#include "Utils.h"

namespace tensorplay {
namespace cpu {

template <typename F>  // F: double -> double
inline Tensor float_math_kernel(const Tensor& self, F f, const char* name,
                                const Tensor* destination = nullptr) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = (self.dtype() == compute_dt) ? self.contiguous()
                                            : self.to(compute_dt).contiguous();
    Tensor t = destination != nullptr
                   ? *destination
                   : Tensor::empty(static_cast<std::vector<int64_t>>(w.shape()), compute_dt,
                                   w.device());
    int64_t n = w.numel();
    if (compute_dt == DType::Float64) {
        const double* sp = w.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = f(sp[i]);
        });
    } else {
        const float* sp = w.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<float>(f(sp[i]));
        });
    }
    (void)name;
    return (out_dt == compute_dt) ? t : t.to(out_dt);
}

// The loop above widens every element to double before calling `f`, so a
// single-precision input pays the double routine and lands at about half the
// rate.  Ops that have a vector walk take that instead, at their own width.
// Every other dtype keeps the generic loop and its promotion rules, and so does
// a caller that supplies a destination.
template <typename F>
inline Tensor float_math_vec_kernel(const Tensor& self, F f, const char* name,
                                    vecunary::VOp op, vecunary::VParams prm = {},
                                    const Tensor* destination = nullptr) {
    const DType in = self.dtype();
    if ((in != DType::Float32 && in != DType::Float64) || destination != nullptr)
        return float_math_kernel(self, f, name, destination);
    Tensor w = self.contiguous();
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(w.shape()), in,
                             w.device());
    const int64_t n = w.numel();
    if (in == DType::Float64) {
        const double* sp = w.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
            vecunary::run_f64(op, prm, sp, dp, b, e);
        });
    } else {
        const float* sp = w.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t b, int64_t e) {
            vecunary::run_f32(op, prm, sp, dp, b, e);
        });
    }
    return t;
}

// Whether the flat map over `self` can fill `out` itself instead of making a
// result that is copied over afterwards.  The loop pairs element i of a packed
// source with element i of the destination, so the destination has to be packed
// too and has to carry the answer in the width the loop computes at -- a
// Half or BFloat16 answer is still narrowed by a conversion of its own, and a
// complex or quantized operand goes by some other route entirely.
inline bool float_math_takes(const Tensor& self, const Tensor& out) {
    const DType in = self.dtype();
    if (isComplexType(in) || isQuantizedType(in)) {
        return false;
    }
    const DType out_dt = isFloatingType(in) ? in : DType::Float32;
    const DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    return out_dt == compute_dt &&
           destination_takes_flat(out, self.device(), out_dt,
                                  static_cast<std::vector<int64_t>>(self.shape()),
                                  self);
}

// One map, one pass, into the caller's tensor wherever that holds; everywhere
// else the value comes from the same kernel that answers without a destination.
template <typename F, typename Alloc>
inline Tensor& float_math_out_kernel(const Tensor& self, Tensor& out,
                                     Alloc allocating, F f, const char* name) {
    if (float_math_takes(self, out)) {
        float_math_kernel(self, f, name, &out);
        return out;
    }
    return write_out(out, allocating(self));
}

template <typename F>  // F: (double,double) -> double
inline Tensor binary_float_kernel(const Tensor& a_in, const Tensor& b_in, F f,
                                  const char* name,
                                  const Tensor* destination = nullptr) {
    if (a_in.device() != b_in.device()) {
        TP_THROW(DeviceMismatchError, name, ": inputs must be on the same device");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    // Reduced-width inputs are evaluated in Float32 and narrowed once at the
    // end; the loops below only ever address float or double buffers.
    DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor ac = a_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor out = destination != nullptr
                     ? *destination
                     : Tensor::empty(out_shape, compute_dt, a_in.device());
    int64_t n = out.numel();
    if (compute_dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = out.data_ptr<double>();
        parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = f(ap[i], bp[i]);
        });
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = out.data_ptr<float>();
        parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<float>(f(ap[i], bp[i]));
        });
    }
    return (dt == compute_dt) ? out : out.to(dt);
}

// Whether the flat pairwise map can fill the caller's tensor itself instead of
// making a result that is copied over afterwards.  The loop pairs element i of
// one packed operand with element i of the other and stores at the same index,
// so the destination has to be packed and hold the answer in the width the loop
// computes at -- a Half or BFloat16 answer still needs a narrowing pass of its
// own -- and neither operand may be a second window over those bytes.
inline bool binary_float_takes(const Tensor& a_in, const Tensor& b_in,
                               const Tensor& out) {
    const DType lhs = a_in.dtype(), rhs = b_in.dtype();
    if (isComplexType(lhs) || isComplexType(rhs) ||
        isQuantizedType(lhs) || isQuantizedType(rhs)) {
        return false;
    }
    DType dt = promoteTypes(lhs, rhs);
    if (isComplexType(dt) || isQuantizedType(dt)) {
        return false;
    }
    if (!isFloatingType(dt)) dt = DType::Float32;
    const DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    if (dt != compute_dt) {
        return false;
    }
    const std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    return destination_takes_flat(out, a_in.device(), compute_dt, shape, a_in, b_in);
}

// One map, one pass, into the caller's tensor wherever that holds; everywhere
// else the value comes from the same kernel that answers without a destination.
template <typename F, typename Alloc>
inline Tensor& binary_float_out_kernel(const Tensor& a_in, const Tensor& b_in,
                                       Tensor& out, Alloc allocating, F f,
                                       const char* name) {
    if (binary_float_takes(a_in, b_in, out)) {
        binary_float_kernel(a_in, b_in, f, name, &out);
        return out;
    }
    return write_out(out, allocating(a_in, b_in));
}

}  // namespace cpu
}  // namespace tensorplay
