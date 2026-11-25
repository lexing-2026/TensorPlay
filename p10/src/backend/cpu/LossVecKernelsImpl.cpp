// Tier-compiled forward and backward walks for the losses: binary cross
// entropy, binary cross entropy over logits, huber, smooth l1, soft margin,
// poisson, mse, l1, kl divergence, multilabel soft margin and triplet margin.
// Each CPU capability tier compiles its own copy of this file (see
// TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt) and registers the slots declared in
// cpu/LossVecKernels.h; DispatchStub picks the widest copy at run time.
//
// The folds read the operand buffers as they are, so no widened copy of an
// input is materialized, and the per-lane arithmetic stays in the operand
// width (float for float inputs) while the range answers are folded in double.

#include "cpu/LossVecKernels.h"
#include "cpu/vec/vec.h"

#include "DType.h"
#include "Exception.h"

#include "LossReduce.h"

#include <algorithm>
#include <atomic>
#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace tensorplay {
namespace cpu {

namespace {

// Work per unit of the elementwise walks and of the row-wise walk.
constexpr int64_t kElementwiseUnit = 1;

template <typename T>
double huber_sum_typed(const T* input, const T* target, int64_t n,
                       double delta) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T d = static_cast<T>(delta);
    const Vec dv(d);
    const Vec half(static_cast<T>(0.5));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec z = Vec::loadu(input + i) - Vec::loadu(target + i);
            const Vec az = z.abs();
            const Vec quad = z * z * half;
            const Vec linear = dv * (az - dv * half);
            acc = acc + Vec::blendv(linear, quad, az <= dv);
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const T z = input[i] - target[i];
            const T az = z < static_cast<T>(0) ? -z : z;
            total += static_cast<double>(az <= d
                                             ? z * z * static_cast<T>(0.5)
                                             : d * (az - d * static_cast<T>(0.5)));
        }
        return total;
    });
}


template <typename T>
void huber_into_typed(const T* input, const T* target, T* out, int64_t n,
                      double delta) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T d = static_cast<T>(delta);
    const Vec dv(d);
    const Vec half(static_cast<T>(0.5));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z =
                                Vec::loadu(input + i) - Vec::loadu(target + i);
                            const Vec az = z.abs();
                            const Vec quad = z * z * half;
                            const Vec linear = dv * (az - dv * half);
                            Vec::blendv(linear, quad, az <= dv).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            const T az = z < static_cast<T>(0) ? -z : z;
                            out[i] = az <= d
                                         ? z * z * static_cast<T>(0.5)
                                         : d * (az - d * static_cast<T>(0.5));
                        }
                        return 0.0;
                    });
}

// z = x - t; the quadratic branch holds only the lanes strictly inside the
// width, so a zero width answers the deviation itself and never divides.
template <typename T>
double smooth_l1_sum_typed(const T* input, const T* target, int64_t n,
                           double beta) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T b = static_cast<T>(beta);
    const Vec bv(b);
    const Vec half(static_cast<T>(0.5));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec z = Vec::loadu(input + i) - Vec::loadu(target + i);
            const Vec az = z.abs();
            const Vec quad = z * z * (half / bv);
            const Vec linear = az - bv * half;
            acc = acc + Vec::blendv(linear, quad, az < bv);
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const T z = input[i] - target[i];
            const T az = z < static_cast<T>(0) ? -z : z;
            total += static_cast<double>(
                az < b ? z * z * static_cast<T>(0.5) / b
                       : az - b * static_cast<T>(0.5));
        }
        return total;
    });
}

template <typename T>
void smooth_l1_into_typed(const T* input, const T* target, T* out, int64_t n,
                          double beta) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T b = static_cast<T>(beta);
    const Vec bv(b);
    const Vec half(static_cast<T>(0.5));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec z = Vec::loadu(input + i) - Vec::loadu(target + i);
            const Vec az = z.abs();
            const Vec quad = z * z * (half / bv);
            const Vec linear = az - bv * half;
            Vec::blendv(linear, quad, az < bv).store(out + i);
        }
        for (; i < end; ++i) {
            const T z = input[i] - target[i];
            const T az = z < static_cast<T>(0) ? -z : z;
            out[i] = az < b ? z * z * static_cast<T>(0.5) / b
                            : az - b * static_cast<T>(0.5);
        }
        return 0.0;
    });
}

// The slope is a three-way test on the signed deviation; the two outer tests
// are reached before the division, so a lane sitting on the width answers the
// full slope and a zero width never divides.
template <typename T>
void smooth_l1_grad_typed(const T* grad_output, const T* input, const T* target,
                          T* out, int64_t n, double beta, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T b = static_cast<T>(beta);
    const Vec bv(b);
    const Vec one(static_cast<T>(1));
    const Vec neg_one(static_cast<T>(-1));
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec z = Vec::loadu(input + i) - Vec::loadu(target + i);
            const Vec inner = Vec::blendv(z / bv, one, z >= bv);
            const Vec side = Vec::blendv(inner, neg_one, z <= -bv);
            (slope * side * Vec::loadu(grad_output + i)).store(out + i);
        }
        for (; i < end; ++i) {
            const T z = input[i] - target[i];
            const T side = z <= -b ? static_cast<T>(-1)
                                   : (z >= b ? static_cast<T>(1)
                                             : z / b);
            out[i] = side * static_cast<T>(scale) * grad_output[i];
        }
        return 0.0;
    });
}

// log(1 + exp(-y)) with y = t x, built as max(-y, 0) + log1p(exp(-|y|)) so a
// large negative product cannot overflow the exponential.
template <typename T>
double softmargin_sum_typed(const T* input, const T* target, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec y = Vec::loadu(input + i) * Vec::loadu(target + i);
            const Vec tail = (-y.abs()).exp().log1p();
            acc = acc + vec::maximum(-y, zero) + tail;
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const double y = static_cast<double>(input[i]) *
                             static_cast<double>(target[i]);
            total += std::max(-y, 0.0) + std::log1p(std::exp(-std::fabs(y)));
        }
        return total;
    });
}

template <typename T>
void softmargin_into_typed(const T* input, const T* target, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec y = Vec::loadu(input + i) * Vec::loadu(target + i);
            const Vec tail = (-y.abs()).exp().log1p();
            (vec::maximum(-y, zero) + tail).store(out + i);
        }
        for (; i < end; ++i) {
            const double y = static_cast<double>(input[i]) *
                             static_cast<double>(target[i]);
            out[i] = static_cast<T>(std::max(-y, 0.0) +
                                    std::log1p(std::exp(-std::fabs(y))));
        }
        return 0.0;
    });
}

// d/dx = -t * sigmoid(-t x) = -t / (1 + exp(t x)); the denominator grows with
// the product, so the exponential cannot overflow the reciprocal.
template <typename T>
void softmargin_grad_typed(const T* grad_output, const T* input, const T* target,
                           T* out, int64_t n, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec one(static_cast<T>(1));
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec t = Vec::loadu(target + i);
            const Vec y = Vec::loadu(input + i) * t;
            const Vec sig = one / (one + y.exp());
            (slope * (-t * sig) * Vec::loadu(grad_output + i)).store(out + i);
        }
        for (; i < end; ++i) {
            const double t = static_cast<double>(target[i]);
            const double y = static_cast<double>(input[i]) * t;
            const double sig = 1.0 / (1.0 + std::exp(y));
            out[i] = static_cast<T>(scale) * (-t * sig) * grad_output[i];
        }
        return 0.0;
    });
}

// exp(x) - t x (log_input) or x - t log(x + eps), plus the Stirling term
// t log t - t + 0.5 log(2 pi t) that only exists for targets above one.
template <typename T>
T poisson_lane(const T x, const T t, bool log_input, bool full, double eps) {
    T loss = log_input ? static_cast<T>(std::exp(x)) - t * x
                       : x - t * static_cast<T>(std::log(
                             x + static_cast<T>(eps)));
    if (full && t > static_cast<T>(1)) {
        loss += t * static_cast<T>(std::log(t)) - t +
                static_cast<T>(0.5) * static_cast<T>(
                    std::log(static_cast<T>(2.0 * M_PI) * t));
    }
    return loss;
}

template <typename T>
double poisson_sum_typed(const T* input, const T* target, int64_t n,
                         bool log_input, bool full, double eps) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec half(static_cast<T>(0.5));
    const Vec two_pi(static_cast<T>(2.0 * M_PI));
    const Vec epsv(static_cast<T>(eps));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec x = Vec::loadu(input + i);
            const Vec t = Vec::loadu(target + i);
            Vec loss = log_input ? x.exp() - t * x
                                 : x - t * (x + epsv).log();
            if (full) {
                const Vec ts = Vec::blendv(one, t, t > one);
                const Vec stirling =
                    t * ts.log() - t + (ts * two_pi).log() * half;
                loss = loss + Vec::blendv(zero, stirling, t > one);
            }
            acc = acc + loss;
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            total += static_cast<double>(
                poisson_lane(input[i], target[i], log_input, full, eps));
        }
        return total;
    });
}

template <typename T>
void poisson_into_typed(const T* input, const T* target, T* out, int64_t n,
                        bool log_input, bool full, double eps) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec half(static_cast<T>(0.5));
    const Vec two_pi(static_cast<T>(2.0 * M_PI));
    const Vec epsv(static_cast<T>(eps));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec x = Vec::loadu(input + i);
                            const Vec t = Vec::loadu(target + i);
                            Vec loss = log_input ? x.exp() - t * x
                                                 : x - t * (x + epsv).log();
                            if (full) {
                                const Vec ts = Vec::blendv(one, t, t > one);
                                const Vec stirling =
                                    t * ts.log() - t +
                                    (ts * two_pi).log() * half;
                                loss = loss +
                                       Vec::blendv(zero, stirling, t > one);
                            }
                            loss.store(out + i);
                        }
                        for (; i < end; ++i) {
                            out[i] = poisson_lane(input[i], target[i],
                                                  log_input, full, eps);
                        }
                        return 0.0;
                    });
}

// d/dx is exp(x) - t (log_input) or 1 - t / (x + eps); the Stirling term is
// constant in x, so `full` does not reach the gradient.
template <typename T>
void poisson_grad_typed(const T* grad_output, const T* input, const T* target,
                        T* out, int64_t n, bool log_input, double eps,
                        double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec one(static_cast<T>(1));
    const Vec slope(static_cast<T>(scale));
    const Vec epsv(static_cast<T>(eps));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec x = Vec::loadu(input + i);
                            const Vec t = Vec::loadu(target + i);
                            const Vec lane = log_input ? x.exp() - t
                                                       : one - t / (x + epsv);
                            (slope * lane * Vec::loadu(grad_output + i))
                                .store(out + i);
                        }
                        for (; i < end; ++i) {
                            const double x = static_cast<double>(input[i]);
                            const double t = static_cast<double>(target[i]);
                            const double lane = log_input ? std::exp(x) - t
                                                          : 1.0 - t / (x + eps);
                            out[i] = static_cast<T>(scale) * lane *
                                     grad_output[i];
                        }
                        return 0.0;
                    });
}

// x when the target lane is 1, max(0, margin - x) when it is -1, and their
// sum for any other lane.
template <typename T>
double hinge_sum_typed(const T* input, const T* target, int64_t n,
                       double margin) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T m = static_cast<T>(margin);
    const Vec mv(m);
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec neg_one(static_cast<T>(-1));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec x = Vec::loadu(input + i);
            const Vec t = Vec::loadu(target + i);
            const Vec pos = vec::maximum(mv - x, zero);
            const Vec margin_part = Vec::blendv(zero, pos, t != one);
            const Vec self_part = Vec::blendv(zero, x, t != neg_one);
            acc = acc + margin_part + self_part;
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const T x = input[i];
            const T t = target[i];
            const T pos = m - x > static_cast<T>(0) ? m - x : static_cast<T>(0);
            const T margin_part =
                t != static_cast<T>(1) ? pos : static_cast<T>(0);
            const T self_part =
                t != static_cast<T>(-1) ? x : static_cast<T>(0);
            total += static_cast<double>(margin_part + self_part);
        }
        return total;
    });
}

template <typename T>
void hinge_into_typed(const T* input, const T* target, T* out, int64_t n,
                      double margin) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T m = static_cast<T>(margin);
    const Vec mv(m);
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec neg_one(static_cast<T>(-1));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec x = Vec::loadu(input + i);
                            const Vec t = Vec::loadu(target + i);
                            const Vec pos = vec::maximum(mv - x, zero);
                            const Vec margin_part =
                                Vec::blendv(zero, pos, t != one);
                            const Vec self_part =
                                Vec::blendv(zero, x, t != neg_one);
                            (margin_part + self_part).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T x = input[i];
                            const T t = target[i];
                            const T pos = m - x > static_cast<T>(0)
                                              ? m - x
                                              : static_cast<T>(0);
                            out[i] = (t != static_cast<T>(1)
                                          ? pos
                                          : static_cast<T>(0)) +
                                     (t != static_cast<T>(-1) ? x
                                                              : static_cast<T>(0));
                        }
                        return 0.0;
                    });
}

// The slope is 1 for target 1, -(margin - x > 0) for target -1, and
// 1 - (margin - x > 0) otherwise.
template <typename T>
void hinge_grad_typed(const T* grad_output, const T* input, const T* target,
                      T* out, int64_t n, double margin, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T m = static_cast<T>(margin);
    const Vec mv(m);
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec neg_one(static_cast<T>(-1));
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec x = Vec::loadu(input + i);
                            const Vec t = Vec::loadu(target + i);
                            const Vec active =
                                Vec::blendv(zero, one, (mv - x) > zero);
                            const Vec inner =
                                Vec::blendv(one - active, -active, t == neg_one);
                            const Vec side = Vec::blendv(inner, one, t == one);
                            (slope * side * Vec::loadu(grad_output + i))
                                .store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T x = input[i];
                            const T t = target[i];
                            const T active = m - x > static_cast<T>(0)
                                                 ? static_cast<T>(1)
                                                 : static_cast<T>(0);
                            const T side =
                                t == static_cast<T>(1)
                                    ? static_cast<T>(1)
                                    : (t == static_cast<T>(-1)
                                           ? -active
                                           : static_cast<T>(1) - active);
                            out[i] = side * static_cast<T>(scale) *
                                     grad_output[i];
                        }
                        return 0.0;
                    });
}

// max(0, margin - (x1 - x2) t), read once per operand and folded in the same
// walk.
template <typename T>
double margin_rank_sum_typed(const T* input1, const T* input2,
                             const T* target, int64_t n, double margin) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T m = static_cast<T>(margin);
    const Vec mv(m);
    const Vec zero(static_cast<T>(0));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec d = Vec::loadu(input1 + i) - Vec::loadu(input2 + i);
            const Vec t = Vec::loadu(target + i);
            acc = acc + vec::maximum(mv - d * t, zero);
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const T d = input1[i] - input2[i];
            const T t = target[i];
            const T lane = m - d * t > static_cast<T>(0) ? m - d * t
                                                         : static_cast<T>(0);
            total += static_cast<double>(lane);
        }
        return total;
    });
}

template <typename T>
void margin_rank_into_typed(const T* input1, const T* input2, const T* target,
                            T* out, int64_t n, double margin) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T m = static_cast<T>(margin);
    const Vec mv(m);
    const Vec zero(static_cast<T>(0));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec d =
                                Vec::loadu(input1 + i) - Vec::loadu(input2 + i);
                            const Vec t = Vec::loadu(target + i);
                            (vec::maximum(mv - d * t, zero)).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T d = input1[i] - input2[i];
                            const T t = target[i];
                            out[i] = m - d * t > static_cast<T>(0)
                                         ? m - d * t
                                         : static_cast<T>(0);
                        }
                        return 0.0;
                    });
}

// The active mask is whether the margin expression stays positive; the first
// operand receives -(active t) and the second its negation.
template <typename T>
void margin_rank_grad_typed(const T* grad_output, const T* input1,
                            const T* input2, const T* target, T* out1, T* out2,
                            int64_t n, double margin, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T m = static_cast<T>(margin);
    const Vec mv(m);
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec d = Vec::loadu(input1 + i) -
                                          Vec::loadu(input2 + i);
                            const Vec t = Vec::loadu(target + i);
                            const Vec active =
                                Vec::blendv(zero, one, (mv - d * t) > zero);
                            const Vec g = slope * active * t *
                                          Vec::loadu(grad_output + i);
                            (-g).store(out1 + i);
                            g.store(out2 + i);
                        }
                        for (; i < end; ++i) {
                            const T d = input1[i] - input2[i];
                            const T t = target[i];
                            const T active =
                                m - d * t > static_cast<T>(0)
                                    ? static_cast<T>(1)
                                    : static_cast<T>(0);
                            const T g = active * static_cast<T>(scale) * t *
                                        grad_output[i];
                            out1[i] = -g;
                            out2[i] = g;
                        }
                        return 0.0;
                    });
}

// One row's cosine: the two squared norms and the dot product are accumulated
// in vector width, then the row answer is the target's branch on the cosine.
// A row of nothing divides zero by zero, as the composed form does.
template <typename T>
double cosine_rows_typed(const T* x1, const T* x2, const T* target, T* out,
                         int64_t rows, int64_t width, double margin) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    return parallel_ranges(rows, width,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const T* a = x1 + r * width;
            const T* b = x2 + r * width;
            Vec acc_a2(static_cast<T>(0));
            Vec acc_b2(static_cast<T>(0));
            Vec acc_ab(static_cast<T>(0));
            int64_t i = 0;
            for (; i + W <= width; i += W) {
                const Vec av = Vec::loadu(a + i);
                const Vec bv = Vec::loadu(b + i);
                acc_a2 = acc_a2 + av * av;
                acc_b2 = acc_b2 + bv * bv;
                acc_ab = acc_ab + av * bv;
            }
            double n1 = static_cast<double>(acc_a2.reduce_add());
            double n2 = static_cast<double>(acc_b2.reduce_add());
            double dot = static_cast<double>(acc_ab.reduce_add());
            for (; i < width; ++i) {
                const double av = static_cast<double>(a[i]);
                const double bv = static_cast<double>(b[i]);
                n1 += av * av;
                n2 += bv * bv;
                dot += av * bv;
            }
            const double d = std::sqrt(n1 * n2);
            const double cos = dot / d;
            const double t = static_cast<double>(target[r]);
            const double row = t == 1.0
                                   ? 1.0 - cos
                                   : (t == -1.0
                                          ? std::max(0.0, cos - margin)
                                          : 0.0);
            total += row;
            if (out) out[r] = static_cast<T>(row);
        }
        return total;
    });
}

// The row's loss slope times the incoming gradient scales both input planes;
// the denominator d is floored at 1e-12 while the squared norms divide as they
// are, matching the composed backward.
template <typename T>
void cosine_grad_typed(const T* grad_output, const T* x1, const T* x2,
                       const T* target, T* out1, T* out2, int64_t rows,
                       int64_t width, double margin, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    parallel_ranges(rows, width,
                    [&](int64_t begin, int64_t end) -> double {
        for (int64_t r = begin; r < end; ++r) {
            const T* a = x1 + r * width;
            const T* b = x2 + r * width;
            Vec acc_a2(static_cast<T>(0));
            Vec acc_b2(static_cast<T>(0));
            Vec acc_ab(static_cast<T>(0));
            int64_t i = 0;
            for (; i + W <= width; i += W) {
                const Vec av = Vec::loadu(a + i);
                const Vec bv = Vec::loadu(b + i);
                acc_a2 = acc_a2 + av * av;
                acc_b2 = acc_b2 + bv * bv;
                acc_ab = acc_ab + av * bv;
            }
            double n1 = static_cast<double>(acc_a2.reduce_add());
            double n2 = static_cast<double>(acc_b2.reduce_add());
            double dot = static_cast<double>(acc_ab.reduce_add());
            for (; i < width; ++i) {
                const double av = static_cast<double>(a[i]);
                const double bv = static_cast<double>(b[i]);
                n1 += av * av;
                n2 += bv * bv;
                dot += av * bv;
            }
            const double d_raw = std::sqrt(n1 * n2);
            const double d = std::max(d_raw, 1e-12);
            const double cos = dot / d_raw;
            const double t = static_cast<double>(target[r]);
            const double dl_dcos =
                t == 1.0 ? -1.0
                         : (t == -1.0
                                ? ((cos - margin) > 0.0 ? 1.0 : 0.0)
                                : 0.0);
            const double row_scale =
                dl_dcos * static_cast<double>(grad_output[r]) * scale;
            const Vec dd(static_cast<T>(d));
            const Vec n1v(static_cast<T>(n1));
            const Vec n2v(static_cast<T>(n2));
            const Vec cv(static_cast<T>(cos));
            const Vec rs(static_cast<T>(row_scale));
            int64_t j = 0;
            for (; j + W <= width; j += W) {
                const Vec av = Vec::loadu(a + j);
                const Vec bv = Vec::loadu(b + j);
                ((bv / dd - cv * (av / n1v)) * rs).store(out1 + r * width + j);
                ((av / dd - cv * (bv / n2v)) * rs).store(out2 + r * width + j);
            }
            for (; j < width; ++j) {
                const double av = static_cast<double>(a[j]);
                const double bv = static_cast<double>(b[j]);
                out1[r * width + j] =
                    static_cast<T>((bv / d - cos * av / n1) * row_scale);
                out2[r * width + j] =
                    static_cast<T>((av / d - cos * bv / n2) * row_scale);
            }
        }
        return 0.0;
    });
}

// z = x - t, squared and folded.  One load of each operand answers a whole
// lane of the sum, so the deviation plane the composed form would allocate
// never exists.
template <typename T>
double mse_sum_typed(const T* input, const T* target, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
                               Vec acc(static_cast<T>(0));
                               int64_t i = begin;
                               for (; i + W <= end; i += W) {
                                   const Vec z = Vec::loadu(input + i) -
                                                 Vec::loadu(target + i);
                                   acc = acc + z * z;
                               }
                               double total =
                                   static_cast<double>(acc.reduce_add());
                               for (; i < end; ++i) {
                                   const T z = input[i] - target[i];
                                   total += static_cast<double>(z * z);
                               }
                               return total;
                           });
}

template <typename T>
void mse_into_typed(const T* input, const T* target, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            (z * z).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            out[i] = z * z;
                        }
                        return 0.0;
                    });
}

// The squared walk answers twice the deviation, moved by the incoming
// gradient lane by lane.
template <typename T>
void mse_grad_typed(const T* grad_output, const T* input, const T* target,
                    T* out, int64_t n, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            (slope * z * Vec::loadu(grad_output + i))
                                .store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            out[i] = static_cast<T>(scale) * z *
                                     grad_output[i];
                        }
                        return 0.0;
                    });
}

// The same deviation read without its sign.
template <typename T>
double l1_sum_typed(const T* input, const T* target, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
                               Vec acc(static_cast<T>(0));
                               int64_t i = begin;
                               for (; i + W <= end; i += W) {
                                   const Vec z = Vec::loadu(input + i) -
                                                 Vec::loadu(target + i);
                                   acc = acc + z.abs();
                               }
                               double total =
                                   static_cast<double>(acc.reduce_add());
                               for (; i < end; ++i) {
                                   const T z = input[i] - target[i];
                                   total += static_cast<double>(
                                       z < static_cast<T>(0) ? -z : z);
                               }
                               return total;
                           });
}

template <typename T>
void l1_into_typed(const T* input, const T* target, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            z.abs().store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            out[i] = z < static_cast<T>(0) ? -z : z;
                        }
                        return 0.0;
                    });
}

// The absolute walk answers the side the deviation sits on: both sides carry
// one, and the width itself answers nothing.
template <typename T>
void l1_grad_typed(const T* grad_output, const T* input, const T* target,
                   T* out, int64_t n, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            const Vec side = Vec::blendv(
                                zero, Vec::blendv(-one, one, z > zero),
                                z != zero);
                            (side * Vec::loadu(grad_output + i))
                                .store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            const T side = z > static_cast<T>(0)
                                               ? static_cast<T>(1)
                                               : (z < static_cast<T>(0)
                                                      ? static_cast<T>(-1)
                                                      : static_cast<T>(0));
                            out[i] = side * static_cast<T>(scale) *
                                     grad_output[i];
                        }
                        return 0.0;
                    });
}

// t log t - t x, with a zero target contributing nothing; or exp(t) (t - x)
// when the target already sits in log space.
template <typename T>
T kl_lane(const T& x, const T& t, bool log_target) {
    if (log_target) {
        return static_cast<T>(std::exp(t) * (t - x));
    }
    if (t == static_cast<T>(0)) return static_cast<T>(0);
    return t * static_cast<T>(std::log(t)) - t * x;
}

template <typename T>
double kl_sum_typed(const T* input, const T* target, int64_t n,
                    bool log_target) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
                               Vec acc(static_cast<T>(0));
                               int64_t i = begin;
                               for (; i + W <= end; i += W) {
                                   const Vec x = Vec::loadu(input + i);
                                   const Vec t = Vec::loadu(target + i);
                                   if (log_target) {
                                       acc = acc + t.exp() * (t - x);
                                   } else {
                                       const Vec lane = t * t.log() - t * x;
                                       acc = acc + Vec::blendv(zero, lane,
                                                               t != zero);
                                   }
                               }
                               double total =
                                   static_cast<double>(acc.reduce_add());
                               for (; i < end; ++i) {
                                   total += static_cast<double>(kl_lane(
                                       input[i], target[i], log_target));
                               }
                               return total;
                           });
}

template <typename T>
void kl_into_typed(const T* input, const T* target, T* out, int64_t n,
                   bool log_target) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec x = Vec::loadu(input + i);
                            const Vec t = Vec::loadu(target + i);
                            if (log_target) {
                                (t.exp() * (t - x)).store(out + i);
                            } else {
                                const Vec lane = t * t.log() - t * x;
                                Vec::blendv(zero, lane, t != zero)
                                    .store(out + i);
                            }
                        }
                        for (; i < end; ++i) {
                            out[i] = kl_lane(input[i], target[i], log_target);
                        }
                        return 0.0;
                    });
}

// d/dinput is the negated target (probability space, zero lanes contributing
// nothing) or the negated exponent (log space), moved by the incoming gradient
// and by the slope the reduction divides with.
template <typename T>
void kl_grad_typed(const T* grad_output, const T* input, const T* target,
                   T* out, int64_t n, bool log_target, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec t = Vec::loadu(target + i);
                            const Vec g = Vec::loadu(grad_output + i);
                            Vec lane;
                            if (log_target) {
                                lane = -t.exp() * g;
                            } else {
                                lane = -t * g;
                                lane = Vec::blendv(zero, lane, t != zero);
                            }
                            (slope * lane).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T t = target[i];
                            const T lane = log_target
                                               ? static_cast<T>(-std::exp(t) *
                                                                grad_output[i])
                                               : (t == static_cast<T>(0)
                                                      ? static_cast<T>(0)
                                                      : -t * grad_output[i]);
                            out[i] = static_cast<T>(scale) * lane;
                        }
                        return 0.0;
                    });
}

// log x and log(1 - x), both floored at -100, in the operand width.
template <typename T>
double bce_sum_typed(const T* input, const T* target, int64_t n,
                     int* bad_input, int* bad_target) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec floor_value(static_cast<T>(-100));
    std::atomic<int> in_bad{0};
    std::atomic<int> tg_bad{0};
    const double total =
        parallel_ranges(n, kElementwiseUnit,
                        [&](int64_t begin, int64_t end) -> double {
                            Vec acc(static_cast<T>(0));
                            bool chunk_in_bad = false;
                            bool chunk_tg_bad = false;
                            int64_t i = begin;
                            for (; i + W <= end; i += W) {
                                const Vec iv = Vec::loadu(input + i);
                                const Vec tv = Vec::loadu(target + i);
                                const Vec in_ok = Vec::blendv(
                                    zero, one, (iv >= zero) & (iv <= one));
                                const Vec tg_ok = Vec::blendv(
                                    zero, one, (tv >= zero) & (tv <= one));
                                if (in_ok.reduce_min() == static_cast<T>(0)) chunk_in_bad = true;
                                if (tg_ok.reduce_min() == static_cast<T>(0)) chunk_tg_bad = true;
                                const Vec log_x =
                                    vec::maximum(iv.log(), floor_value);
                                const Vec log_1mx =
                                    vec::maximum((one - iv).log(), floor_value);
                                acc = acc + (tv - one) * log_1mx - tv * log_x;
                            }
                            double local = static_cast<double>(acc.reduce_add());
                            for (; i < end; ++i) {
                                const T iv = input[i];
                                const T tv = target[i];
                                if (!(iv >= static_cast<T>(0) &&
                                      iv <= static_cast<T>(1)))
                                    chunk_in_bad = true;
                                if (!(tv >= static_cast<T>(0) &&
                                      tv <= static_cast<T>(1)))
                                    chunk_tg_bad = true;
                                const T log_x = std::max(
                                    static_cast<T>(std::log(iv)),
                                    static_cast<T>(-100));
                                const T log_1mx = std::max(
                                    static_cast<T>(std::log1p(-iv)),
                                    static_cast<T>(-100));
                                local += static_cast<double>(
                                    (tv - static_cast<T>(1)) * log_1mx -
                                    tv * log_x);
                            }
                            if (chunk_in_bad)
                                in_bad.store(1, std::memory_order_relaxed);
                            if (chunk_tg_bad)
                                tg_bad.store(1, std::memory_order_relaxed);
                            return local;
                        });
    *bad_input = in_bad.load(std::memory_order_relaxed);
    *bad_target = tg_bad.load(std::memory_order_relaxed);
    return total;
}


template <typename T>
void bce_into_typed(const T* input, const T* target, T* out, int64_t n,
                    int* bad_input, int* bad_target) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec floor_value(static_cast<T>(-100));
    std::atomic<int> in_bad{0};
    std::atomic<int> tg_bad{0};
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        bool chunk_in_bad = false;
                        bool chunk_tg_bad = false;
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec iv = Vec::loadu(input + i);
                            const Vec tv = Vec::loadu(target + i);
                            const Vec in_ok =
                                Vec::blendv(zero, one, (iv >= zero) & (iv <= one));
                            const Vec tg_ok =
                                Vec::blendv(zero, one, (tv >= zero) & (tv <= one));
                            if (in_ok.reduce_min() == static_cast<T>(0))
                                chunk_in_bad = true;
                            if (tg_ok.reduce_min() == static_cast<T>(0))
                                chunk_tg_bad = true;
                            const Vec log_x = vec::maximum(iv.log(), floor_value);
                            const Vec log_1mx =
                                vec::maximum((one - iv).log(), floor_value);
                            ((tv - one) * log_1mx - tv * log_x).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T iv = input[i];
                            const T tv = target[i];
                            if (!(iv >= static_cast<T>(0) && iv <= static_cast<T>(1)))
                                chunk_in_bad = true;
                            if (!(tv >= static_cast<T>(0) && tv <= static_cast<T>(1)))
                                chunk_tg_bad = true;
                            const T log_x = std::max(static_cast<T>(std::log(iv)),
                                                     static_cast<T>(-100));
                            const T log_1mx = std::max(
                                static_cast<T>(std::log1p(-iv)),
                                static_cast<T>(-100));
                            out[i] = (tv - static_cast<T>(1)) * log_1mx - tv * log_x;
                        }
                        if (chunk_in_bad)
                            in_bad.store(1, std::memory_order_relaxed);
                        if (chunk_tg_bad)
                            tg_bad.store(1, std::memory_order_relaxed);
                        return 0.0;
                    });
    *bad_input = in_bad.load(std::memory_order_relaxed);
    *bad_target = tg_bad.load(std::memory_order_relaxed);
}

// The loss one lane contributes, negated: t log sigmoid(x) + (1 - t)
// log sigmoid(-x) = t max(-x, 0) + (1 - t) max(x, 0) + log1p(exp(-|x|)), and
// the row answer is its negated sum divided by the row length.
template <typename T>
T multilabel_lane(const T x, const T t) {
    const double xv = static_cast<double>(x);
    const double tv = static_cast<double>(t);
    const double tail = std::log1p(std::exp(-std::fabs(xv)));
    return static_cast<T>(tv * std::max(-xv, 0.0) +
                          (1.0 - tv) * std::max(xv, 0.0) + tail);
}

// The row answer is the walk over the row divided by the row length, and the
// rows are cut off the flat operands, so a row walk needs no second pass.
template <typename T>
double multilabel_rows_typed(const T* input, const T* target, T* out,
                             int64_t rows, int64_t width) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    if (width <= 0) return 0.0;
    const T scale = static_cast<T>(1) / static_cast<T>(width);
    return parallel_ranges(rows, width,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const T* row_in = input + r * width;
            const T* row_tg = target + r * width;
            Vec acc(static_cast<T>(0));
            int64_t i = 0;
            for (; i + W <= width; i += W) {
                const Vec xv = Vec::loadu(row_in + i);
                const Vec tv = Vec::loadu(row_tg + i);
                const Vec tail = (-xv.abs()).exp().log1p();
                const Vec positive = vec::maximum(xv, zero);
                const Vec negative = vec::maximum(-xv, zero);
                acc = acc + (tv * negative + (one - tv) * positive + tail);
            }
            T sum = acc.reduce_add();
            for (; i < width; ++i) sum += multilabel_lane(row_in[i], row_tg[i]);
            const T answer = sum * scale;
            total += static_cast<double>(answer);
            if (out) out[r] = answer;
        }
        return total;
    });
}

// sp(y) = max(y, 0) + log1p(exp(-|y|)); the two softplus terms of one lane
// share their tail, so the walk builds it once and keeps the positive parts.
template <typename T>
double bce_logits_rows_typed(const T* input, const T* target, const T* weight,
                             const T* pos_weight, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec xv = Vec::loadu(input + i);
            const Vec tv = Vec::loadu(target + i);
            const Vec tail = (-xv.abs()).exp().log1p();
            const Vec sp_x = vec::maximum(xv, zero) + tail;
            const Vec sp_neg_x = vec::maximum(-xv, zero) + tail;
            const Vec positive = pos_weight ? Vec::loadu(pos_weight + i) : one;
            Vec value = positive * tv * sp_neg_x + (one - tv) * sp_x;
            if (weight) value = value * Vec::loadu(weight + i);
            acc = acc + value;
            if (out) value.store(out + i);
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const double xv = static_cast<double>(input[i]);
            const double tv = static_cast<double>(target[i]);
            const double tail = std::log1p(std::exp(-std::fabs(xv)));
            const double sp_x = std::max(xv, 0.0) + tail;
            const double sp_neg_x = std::max(-xv, 0.0) + tail;
            const double positive =
                pos_weight ? static_cast<double>(pos_weight[i]) : 1.0;
            double value = positive * tv * sp_neg_x + (1.0 - tv) * sp_x;
            if (weight) value *= static_cast<double>(weight[i]);
            total += value;
            if (out) out[i] = static_cast<T>(value);
        }
        return total;
    });
}

// Order-p norm of (u - v + eps) over one row: the row is shifted by eps
// elementwise before the magnitude is accumulated, so two rows that agree
// answer eps rather than zero.  Orders 1, 2 and the two infinite orders walk
// the row in vector width; the rest (including the zero-order count of unequal
// entries and negative finite orders) take the scalar recurrence.
template <typename T>
double row_norm(const T* u, const T* v, int64_t width, double p, double eps) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec shift(static_cast<T>(eps));
    int64_t i = 0;
    if (std::isinf(p)) {
        const bool largest = p > 0;
        Vec best(static_cast<T>(largest ? 0
                                        : std::numeric_limits<T>::max()));
        for (; i + W <= width; i += W) {
            const Vec d = ((Vec::loadu(u + i) - Vec::loadu(v + i)) + shift).abs();
            best = largest ? vec::maximum(best, d) : vec::minimum(best, d);
        }
        double answer = static_cast<double>(
            largest ? best.reduce_max() : best.reduce_min());
        for (; i < width; ++i) {
            const double d =
                std::fabs(static_cast<double>(u[i]) - static_cast<double>(v[i]) + eps);
            answer = largest ? std::max(answer, d) : std::min(answer, d);
        }
        return answer;
    }
    if (p == 2.0) {
        Vec acc(static_cast<T>(0));
        for (; i + W <= width; i += W) {
            const Vec d = (Vec::loadu(u + i) - Vec::loadu(v + i)) + shift;
            acc = acc + d * d;
        }
        double answer = static_cast<double>(acc.reduce_add());
        for (; i < width; ++i) {
            const double d =
                static_cast<double>(u[i]) - static_cast<double>(v[i]) + eps;
            answer += d * d;
        }
        return std::sqrt(answer);
    }
    if (p == 1.0) {
        Vec acc(static_cast<T>(0));
        for (; i + W <= width; i += W) {
            acc = acc + ((Vec::loadu(u + i) - Vec::loadu(v + i)) + shift).abs();
        }
        double answer = static_cast<double>(acc.reduce_add());
        for (; i < width; ++i)
            answer += std::fabs(static_cast<double>(u[i]) -
                                static_cast<double>(v[i]) + eps);
        return answer;
    }
    if (p == 0.0) {
        double count = 0.0;
        for (; i < width; ++i) {
            if (static_cast<double>(u[i]) - static_cast<double>(v[i]) + eps != 0.0)
                count += 1.0;
        }
        return count;
    }
    double answer = 0.0;
    for (; i < width; ++i) {
        answer += std::pow(std::fabs(static_cast<double>(u[i]) -
                                     static_cast<double>(v[i]) + eps), p);
    }
    return std::pow(answer, 1.0 / p);
}

template <typename T>
double triplet_rows_typed(const T* anchor, const T* positive, const T* negative,
                          T* out, int64_t rows, int64_t width, double margin,
                          double p, double eps, bool swap) {
    return parallel_ranges(rows, width,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const T* a = anchor + r * width;
            const T* pos = positive + r * width;
            const T* neg = negative + r * width;
            double dn = row_norm(a, neg, width, p, eps);
            if (swap) {
                dn = std::min(dn, row_norm(pos, neg, width, p, eps));
            }
            const double row =
                std::max(0.0, row_norm(a, pos, width, p, eps) - dn + margin);
            if (out != nullptr) out[r] = static_cast<T>(row);
            total += row;
        }
        return total;
    });
}

// One row of nll_loss reads only the gathered class score, so the rows are the
// natural parallel unit; the loss sum and the contributing weight are folded
// in two light walks over the same rows.
template <typename T>
double nll_weight_total_typed(const int64_t* target, const T* weight,
                              int64_t rows, int64_t ignore_index,
                              bool has_weight) {
    return parallel_ranges(rows, 256, [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            if (target[r] != ignore_index) {
                total += has_weight ? static_cast<double>(weight[target[r]])
                                    : 1.0;
            }
        }
        return total;
    });
}

template <typename T>
std::tuple<double, double> nll_rows_typed(const T* input,
                                          const int64_t* target,
                                          const T* weight, T* out,
                                          int64_t rows, int64_t cols,
                                          int64_t ignore_index,
                                          bool has_weight) {
    const double loss = parallel_ranges(
        rows, 512, [&](int64_t begin, int64_t end) -> double {
            double total = 0.0;
            // One element per row, and the rows are 2 KB apart, so the walk is
            // a chain of cold lines.  ATen's NLL loop does exactly this with no
            // software prefetch: the out-of-order window already keeps several
            // rows in flight, and a T0 hint 16 rows ahead only evicts the line
            // before it is wanted.
            for (int64_t r = begin; r < end; ++r) {
                const int64_t t = target[r];
                double row_loss = 0.0;
                if (t != ignore_index) {
                    const double w =
                        has_weight ? static_cast<double>(weight[t]) : 1.0;
                    row_loss = -static_cast<double>(input[r * cols + t]) * w;
                }
                total += row_loss;
                if (out) out[r] = static_cast<T>(row_loss);
            }
            return total;
        });
    const double weight_total =
        nll_weight_total_typed(target, weight, rows, ignore_index, has_weight);
    return std::make_tuple(loss, weight_total);
}

// The backward scatters into the target column of every row: the incoming
// gradient is one value per row under reduction none, otherwise the single
// value a reduced call leaves, divided by the total weight under mean.
template <typename T>
void nll_grad_typed(const T* grad_output, const int64_t* target,
                    const T* weight, T* grad_input, int64_t rows, int64_t cols,
                    int64_t ignore_index, int reduction, double total_weight) {
    const bool per_row = reduction == 0;
    const double inv_tw =
        (reduction == 1 && total_weight > 0.0) ? 1.0 / total_weight : 1.0;
    parallel_ranges(rows, 256, [&](int64_t begin, int64_t end) -> double {
        for (int64_t r = begin; r < end; ++r) {
            const int64_t t = target[r];
            if (t == ignore_index) continue;
            const double w = weight ? static_cast<double>(weight[t]) : 1.0;
            const double g =
                (per_row ? static_cast<double>(grad_output[r])
                         : static_cast<double>(grad_output[0])) *
                inv_tw;
            grad_input[r * cols + t] = static_cast<T>(-w * g);
        }
        return 0.0;
    });
}

// The spatial plane of nll_loss2d keeps a sample's height*width positions
// adjacent, so the walk takes the batch axis as its parallel unit and sweeps
// the positions inside it: the row index needs no division, and one sample's
// sampled class column is reached with a class stride equal to the spatial
// length.
template <typename T>
double nll2d_rows_typed(const T* input, const int64_t* target, const T* weight,
                        T* out, int64_t batch, int64_t classes, int64_t hw,
                        int64_t ignore_index, bool has_weight) {
    return parallel_ranges(batch, hw, [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t b = begin; b < end; ++b) {
            const T* base = input + b * classes * hw;
            const int64_t* trow = target + b * hw;
            T* orow = out ? out + b * hw : nullptr;
            for (int64_t pos = 0; pos < hw; ++pos) {
                const int64_t t = trow[pos];
                double row_loss = 0.0;
                if (t != ignore_index) {
                    const double w =
                        has_weight ? static_cast<double>(weight[t]) : 1.0;
                    row_loss = -static_cast<double>(base[t * hw + pos]) * w;
                }
                total += row_loss;
                if (orow) orow[pos] = static_cast<T>(row_loss);
            }
        }
        return total;
    });
}

// The backward writes only the target class of every spatial position, one
// scattered value per sampled score.
template <typename T>
void nll2d_grad_typed(const T* grad_output, const int64_t* target,
                      const T* weight, T* grad_input, int64_t batch,
                      int64_t classes, int64_t hw, int64_t ignore_index,
                      int reduction, double total_weight) {
    const bool per_row = reduction == 0;
    const double inv_tw =
        (reduction == 1 && total_weight > 0.0) ? 1.0 / total_weight : 1.0;
    parallel_ranges(batch, hw, [&](int64_t begin, int64_t end) -> double {
        for (int64_t b = begin; b < end; ++b) {
            const int64_t* trow = target + b * hw;
            T* grow = grad_input + b * classes * hw;
            for (int64_t pos = 0; pos < hw; ++pos) {
                const int64_t t = trow[pos];
                if (t == ignore_index) continue;
                const double w = weight ? static_cast<double>(weight[t]) : 1.0;
                const double g =
                    (per_row
                         ? static_cast<double>(grad_output[b * hw + pos])
                         : static_cast<double>(grad_output[0])) *
                    inv_tw;
                grow[t * hw + pos] = static_cast<T>(-w * g);
            }
        }
        return 0.0;
    });
}

// The hinge sum runs over every column except the target one.  The walk splits
// the row around that column, so the target lane never enters the accumulator
// and no correction term is needed afterwards.
template <typename T>
double multi_margin_rows_typed(const T* input, const int64_t* target,
                               const T* weight, T* out, int64_t rows,
                               int64_t cols, int p, double margin) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const T mg = static_cast<T>(margin);
    const Vec mgv(mg);
    const double scale = 1.0 / static_cast<double>(cols);
    return parallel_ranges(rows, cols,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const int64_t y = target[r];
            const T* row = input + r * cols;
            const T xy = row[y];
            const Vec xyv(xy);
            Vec acc(static_cast<T>(0));
            int64_t i = 0;
            for (; i + W <= y; i += W) {
                const Vec z = mgv - xyv + Vec::loadu(row + i);
                if (p == 1) {
                    acc = acc + vec::maximum(z, zero);
                } else {
                    const Vec pos = vec::maximum(z, zero);
                    acc = acc + pos * pos;
                }
            }
            double sum = 0.0;
            for (; i < y; ++i) {
                const double z =
                    margin - static_cast<double>(xy) +
                    static_cast<double>(row[i]);
                if (z > 0) sum += (p == 1) ? z : z * z;
            }
            i = y + 1;
            for (; i + W <= cols; i += W) {
                const Vec z = mgv - xyv + Vec::loadu(row + i);
                if (p == 1) {
                    acc = acc + vec::maximum(z, zero);
                } else {
                    const Vec pos = vec::maximum(z, zero);
                    acc = acc + pos * pos;
                }
            }
            for (; i < cols; ++i) {
                const double z =
                    margin - static_cast<double>(xy) +
                    static_cast<double>(row[i]);
                if (z > 0) sum += (p == 1) ? z : z * z;
            }
            sum += static_cast<double>(acc.reduce_add());
            const double w = weight ? static_cast<double>(weight[y]) : 1.0;
            const double answer = sum * w * scale;
            total += answer;
            if (out) out[r] = static_cast<T>(answer);
        }
        return total;
    });
}

// The derivative is p z^(p-1) times the row gradient, the target-class weight
// and the reduction scale; the target column receives the negated sum over the
// other columns.  Like the forward, the row splits around the target column so
// the target lane never contributes to the sum it negates.
template <typename T>
void multi_margin_grad_typed(const T* grad_output, const T* input,
                             const int64_t* target, const T* weight,
                             T* grad_input, int64_t rows, int64_t cols, int p,
                             double margin, double scale, bool per_row) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const T mg = static_cast<T>(margin);
    const Vec mgv(mg);
    parallel_ranges(rows, cols, [&](int64_t begin, int64_t end) -> double {
        for (int64_t r = begin; r < end; ++r) {
            const int64_t y = target[r];
            const T* row = input + r * cols;
            T* gout = grad_input + r * cols;
            const T xy = row[y];
            const Vec xyv(xy);
            const double row_g =
                per_row ? static_cast<double>(grad_output[r])
                        : static_cast<double>(grad_output[0]);
            const double w = weight ? static_cast<double>(weight[y]) : 1.0;
            const double rs = scale * w * row_g;
            Vec acc(static_cast<T>(0));
            int64_t i = 0;
            for (; i + W <= y; i += W) {
                const Vec z = mgv - xyv + Vec::loadu(row + i);
                const Vec pos = vec::maximum(z, zero);
                const Vec h =
                    (p == 1) ? Vec::blendv(zero, Vec(static_cast<T>(rs)),
                                           z > zero)
                             : pos * Vec(static_cast<T>(2.0 * rs));
                acc = acc + h;
                h.store(gout + i);
            }
            double sum = 0.0;
            for (; i < y; ++i) {
                const double z =
                    margin - static_cast<double>(xy) +
                    static_cast<double>(row[i]);
                T h = static_cast<T>(0);
                if (z > 0) {
                    h = static_cast<T>((p == 1 ? 1.0 : 2.0 * z) * rs);
                    sum += static_cast<double>(h);
                }
                gout[i] = h;
            }
            gout[y] = static_cast<T>(0);
            i = y + 1;
            for (; i + W <= cols; i += W) {
                const Vec z = mgv - xyv + Vec::loadu(row + i);
                const Vec pos = vec::maximum(z, zero);
                const Vec h =
                    (p == 1) ? Vec::blendv(zero, Vec(static_cast<T>(rs)),
                                           z > zero)
                             : pos * Vec(static_cast<T>(2.0 * rs));
                acc = acc + h;
                h.store(gout + i);
            }
            for (; i < cols; ++i) {
                const double z =
                    margin - static_cast<double>(xy) +
                    static_cast<double>(row[i]);
                T h = static_cast<T>(0);
                if (z > 0) {
                    h = static_cast<T>((p == 1 ? 1.0 : 2.0 * z) * rs);
                    sum += static_cast<double>(h);
                }
                gout[i] = h;
            }
            sum += static_cast<double>(acc.reduce_add());
            gout[y] = static_cast<T>(-sum);
        }
        return 0.0;
    });
}

// One row first scatters 1.0 into the is_target mask at every listed class,
// then walks each target's margin expression against the non-target columns;
// the mask zeroes the target lanes inside the vector walk.
template <typename T>
double multilabel_margin_rows_typed(const T* input, const int64_t* target,
                                    T* out, T* is_target, int64_t rows,
                                    int64_t cols) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const double scale = 1.0 / static_cast<double>(cols);
    return parallel_ranges(rows, cols,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const T* row = input + r * cols;
            const int64_t* tg = target + r * cols;
            T* is = is_target + r * cols;
            for (int64_t dt = 0; dt < cols; ++dt) {
                const int64_t idx = tg[dt];
                if (idx < 0) break;
                is[idx] = static_cast<T>(1);
            }
            double sum = 0.0;
            for (int64_t dt = 0; dt < cols; ++dt) {
                const int64_t idx = tg[dt];
                if (idx < 0) break;
                const T xy = row[idx];
                const Vec xyv(xy);
                const Vec base = one - Vec(xy);
                Vec acc(static_cast<T>(0));
                int64_t i = 0;
                for (; i + W <= cols; i += W) {
                    const Vec z = base + Vec::loadu(row + i);
                    const Vec not_target = one - Vec::loadu(is + i);
                    acc = acc + vec::maximum(z, zero) * not_target;
                }
                double sub = static_cast<double>(acc.reduce_add());
                for (; i < cols; ++i) {
                    if (is[i] == static_cast<T>(0)) {
                        const double z =
                            1.0 - static_cast<double>(xy) +
                            static_cast<double>(row[i]);
                        if (z > 0) sub += z;
                    }
                }
                sum += sub;
            }
            const double answer = sum * scale;
            total += answer;
            if (out) out[r] = static_cast<T>(answer);
        }
        return total;
    });
}

// For every (target, non-target) pair with positive margin the walk adds the
// shared gradient to the non-target column and removes it from the target
// column; the vector loop accumulates the non-target columns in place.
template <typename T>
void multilabel_margin_grad_typed(const T* grad_output, const T* input,
                                  const int64_t* target, const T* is_target,
                                  T* grad_input, int64_t rows, int64_t cols,
                                  double scale, bool per_row) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    parallel_ranges(rows, cols, [&](int64_t begin, int64_t end) -> double {
        for (int64_t r = begin; r < end; ++r) {
            const T* row = input + r * cols;
            const int64_t* tg = target + r * cols;
            const T* is = is_target + r * cols;
            T* gout = grad_input + r * cols;
            const double row_g =
                per_row ? static_cast<double>(grad_output[r])
                        : static_cast<double>(grad_output[0]);
            const double g = scale * row_g;
            const T gv = static_cast<T>(g);
            for (int64_t dt = 0; dt < cols; ++dt) {
                const int64_t idx = tg[dt];
                if (idx < 0) break;
                const T xy = row[idx];
                const Vec xyv(xy);
                const Vec base = one - Vec(xy);
                Vec acc(static_cast<T>(0));
                int64_t i = 0;
                for (; i + W <= cols; i += W) {
                    const Vec z = base + Vec::loadu(row + i);
                    const Vec not_target = one - Vec::loadu(is + i);
                    const Vec h =
                        Vec::blendv(zero, Vec(gv), z > zero) * not_target;
                    acc = acc + h;
                    (Vec::loadu(gout + i) + h).store(gout + i);
                }
                double sub = static_cast<double>(acc.reduce_add());
                for (; i < cols; ++i) {
                    if (is[i] == static_cast<T>(0)) {
                        const double z =
                            1.0 - static_cast<double>(xy) +
                            static_cast<double>(row[i]);
                        if (z > 0) {
                            gout[i] += gv;
                            sub += static_cast<double>(gv);
                        }
                    }
                }
                gout[idx] = static_cast<T>(
                    static_cast<double>(gout[idx]) - sub);
            }
        }
        return 0.0;
    });
}


void bce_into_impl(const void* input, const void* target, void* out, int64_t n,
                   int* bad_input, int* bad_target, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            bce_into_typed(static_cast<const double*>(input),
                           static_cast<const double*>(target),
                           static_cast<double*>(out), n, bad_input, bad_target);
            break;
        case DType::Float32:
            bce_into_typed(static_cast<const float*>(input),
                           static_cast<const float*>(target),
                           static_cast<float*>(out), n, bad_input, bad_target);
            break;
        default:
            TP_THROW(TypeError, "binary_cross_entropy: unsupported dtype");
    }
}

void huber_into_impl(const void* input, const void* target, void* out, int64_t n,
                     double delta, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            huber_into_typed(static_cast<const double*>(input),
                             static_cast<const double*>(target),
                             static_cast<double*>(out), n, delta);
            break;
        case DType::Float32:
            huber_into_typed(static_cast<const float*>(input),
                             static_cast<const float*>(target),
                             static_cast<float*>(out), n, delta);
            break;
        default:
            TP_THROW(TypeError, "huber_loss: unsupported dtype");
    }
}

double bce_sum_impl(const void* input, const void* target, int64_t n,
                    int* bad_input, int* bad_target, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return bce_sum_typed(static_cast<const double*>(input),
                                 static_cast<const double*>(target), n,
                                 bad_input, bad_target);
        case DType::Float32:
            return bce_sum_typed(static_cast<const float*>(input),
                                 static_cast<const float*>(target), n,
                                 bad_input, bad_target);
        default:
            TP_THROW(TypeError, "binary_cross_entropy: unsupported dtype");
    }
}

double huber_sum_impl(const void* input, const void* target, int64_t n,
                      double delta, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return huber_sum_typed(static_cast<const double*>(input),
                                   static_cast<const double*>(target), n, delta);
        case DType::Float32:
            return huber_sum_typed(static_cast<const float*>(input),
                                   static_cast<const float*>(target), n, delta);
        default:
            TP_THROW(TypeError, "huber_loss: unsupported dtype");
    }
}

double smooth_l1_sum_impl(const void* input, const void* target, int64_t n,
                          double beta, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return smooth_l1_sum_typed(static_cast<const double*>(input),
                                       static_cast<const double*>(target),
                                       n, beta);
        case DType::Float32:
            return smooth_l1_sum_typed(static_cast<const float*>(input),
                                       static_cast<const float*>(target),
                                       n, beta);
        default:
            TP_THROW(TypeError, "smooth_l1_loss: unsupported dtype");
    }
}

void smooth_l1_into_impl(const void* input, const void* target, void* out,
                         int64_t n, double beta, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            smooth_l1_into_typed(static_cast<const double*>(input),
                                 static_cast<const double*>(target),
                                 static_cast<double*>(out), n, beta);
            break;
        case DType::Float32:
            smooth_l1_into_typed(static_cast<const float*>(input),
                                 static_cast<const float*>(target),
                                 static_cast<float*>(out), n, beta);
            break;
        default:
            TP_THROW(TypeError, "smooth_l1_loss: unsupported dtype");
    }
}

void smooth_l1_grad_impl(const void* grad_output, const void* input,
                         const void* target, void* out, int64_t n,
                         double beta, double scale, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            smooth_l1_grad_typed(static_cast<const double*>(grad_output),
                                 static_cast<const double*>(input),
                                 static_cast<const double*>(target),
                                 static_cast<double*>(out), n, beta, scale);
            break;
        case DType::Float32:
            smooth_l1_grad_typed(static_cast<const float*>(grad_output),
                                 static_cast<const float*>(input),
                                 static_cast<const float*>(target),
                                 static_cast<float*>(out), n, beta, scale);
            break;
        default:
            TP_THROW(TypeError, "smooth_l1_loss backward: unsupported dtype");
    }
}

double softmargin_sum_impl(const void* input, const void* target, int64_t n,
                           int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return softmargin_sum_typed(static_cast<const double*>(input),
                                        static_cast<const double*>(target), n);
        case DType::Float32:
            return softmargin_sum_typed(static_cast<const float*>(input),
                                        static_cast<const float*>(target), n);
        default:
            TP_THROW(TypeError, "soft_margin_loss: unsupported dtype");
    }
}

void softmargin_into_impl(const void* input, const void* target, void* out,
                          int64_t n, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            softmargin_into_typed(static_cast<const double*>(input),
                                  static_cast<const double*>(target),
                                  static_cast<double*>(out), n);
            break;
        case DType::Float32:
            softmargin_into_typed(static_cast<const float*>(input),
                                  static_cast<const float*>(target),
                                  static_cast<float*>(out), n);
            break;
        default:
            TP_THROW(TypeError, "soft_margin_loss: unsupported dtype");
    }
}

void softmargin_grad_impl(const void* grad_output, const void* input,
                          const void* target, void* out, int64_t n,
                          double scale, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            softmargin_grad_typed(static_cast<const double*>(grad_output),
                                  static_cast<const double*>(input),
                                  static_cast<const double*>(target),
                                  static_cast<double*>(out), n, scale);
            break;
        case DType::Float32:
            softmargin_grad_typed(static_cast<const float*>(grad_output),
                                  static_cast<const float*>(input),
                                  static_cast<const float*>(target),
                                  static_cast<float*>(out), n, scale);
            break;
        default:
            TP_THROW(TypeError, "soft_margin_loss backward: unsupported dtype");
    }
}

double poisson_sum_impl(const void* input, const void* target, int64_t n,
                        int log_input, int full, double eps, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return poisson_sum_typed(static_cast<const double*>(input),
                                     static_cast<const double*>(target), n,
                                     log_input != 0, full != 0, eps);
        case DType::Float32:
            return poisson_sum_typed(static_cast<const float*>(input),
                                     static_cast<const float*>(target), n,
                                     log_input != 0, full != 0, eps);
        default:
            TP_THROW(TypeError, "poisson_nll_loss: unsupported dtype");
    }
}

void poisson_into_impl(const void* input, const void* target, void* out,
                       int64_t n, int log_input, int full, double eps,
                       int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            poisson_into_typed(static_cast<const double*>(input),
                               static_cast<const double*>(target),
                               static_cast<double*>(out), n, log_input != 0,
                               full != 0, eps);
            break;
        case DType::Float32:
            poisson_into_typed(static_cast<const float*>(input),
                               static_cast<const float*>(target),
                               static_cast<float*>(out), n, log_input != 0,
                               full != 0, eps);
            break;
        default:
            TP_THROW(TypeError, "poisson_nll_loss: unsupported dtype");
    }
}

void poisson_grad_impl(const void* grad_output, const void* input,
                       const void* target, void* out, int64_t n,
                       int log_input, double eps, double scale, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            poisson_grad_typed(static_cast<const double*>(grad_output),
                               static_cast<const double*>(input),
                               static_cast<const double*>(target),
                               static_cast<double*>(out), n, log_input != 0,
                               eps, scale);
            break;
        case DType::Float32:
            poisson_grad_typed(static_cast<const float*>(grad_output),
                               static_cast<const float*>(input),
                               static_cast<const float*>(target),
                               static_cast<float*>(out), n, log_input != 0,
                               eps, scale);
            break;
        default:
            TP_THROW(TypeError, "poisson_nll_loss backward: unsupported dtype");
    }
}

double hinge_sum_impl(const void* input, const void* target, int64_t n,
                      double margin, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return hinge_sum_typed(static_cast<const double*>(input),
                                   static_cast<const double*>(target), n,
                                   margin);
        case DType::Float32:
            return hinge_sum_typed(static_cast<const float*>(input),
                                   static_cast<const float*>(target), n,
                                   margin);
        default:
            TP_THROW(TypeError, "hinge_embedding_loss: unsupported dtype");
    }
}

void hinge_into_impl(const void* input, const void* target, void* out,
                     int64_t n, double margin, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            hinge_into_typed(static_cast<const double*>(input),
                             static_cast<const double*>(target),
                             static_cast<double*>(out), n, margin);
            break;
        case DType::Float32:
            hinge_into_typed(static_cast<const float*>(input),
                             static_cast<const float*>(target),
                             static_cast<float*>(out), n, margin);
            break;
        default:
            TP_THROW(TypeError, "hinge_embedding_loss: unsupported dtype");
    }
}

void hinge_grad_impl(const void* grad_output, const void* input,
                     const void* target, void* out, int64_t n, double margin,
                     double scale, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            hinge_grad_typed(static_cast<const double*>(grad_output),
                             static_cast<const double*>(input),
                             static_cast<const double*>(target),
                             static_cast<double*>(out), n, margin, scale);
            break;
        case DType::Float32:
            hinge_grad_typed(static_cast<const float*>(grad_output),
                             static_cast<const float*>(input),
                             static_cast<const float*>(target),
                             static_cast<float*>(out), n, margin, scale);
            break;
        default:
            TP_THROW(TypeError, "hinge_embedding_loss backward: unsupported dtype");
    }
}

double margin_rank_sum_impl(const void* input1, const void* input2,
                            const void* target, int64_t n, double margin,
                            int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return margin_rank_sum_typed(static_cast<const double*>(input1),
                                         static_cast<const double*>(input2),
                                         static_cast<const double*>(target), n,
                                         margin);
        case DType::Float32:
            return margin_rank_sum_typed(static_cast<const float*>(input1),
                                         static_cast<const float*>(input2),
                                         static_cast<const float*>(target), n,
                                         margin);
        default:
            TP_THROW(TypeError, "margin_ranking_loss: unsupported dtype");
    }
}

void margin_rank_into_impl(const void* input1, const void* input2,
                           const void* target, void* out, int64_t n,
                           double margin, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            margin_rank_into_typed(static_cast<const double*>(input1),
                                   static_cast<const double*>(input2),
                                   static_cast<const double*>(target),
                                   static_cast<double*>(out), n, margin);
            break;
        case DType::Float32:
            margin_rank_into_typed(static_cast<const float*>(input1),
                                   static_cast<const float*>(input2),
                                   static_cast<const float*>(target),
                                   static_cast<float*>(out), n, margin);
            break;
        default:
            TP_THROW(TypeError, "margin_ranking_loss: unsupported dtype");
    }
}

void margin_rank_grad_impl(const void* grad_output, const void* input1,
                           const void* input2, const void* target, void* out1,
                           void* out2, int64_t n, double margin, double scale,
                           int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            margin_rank_grad_typed(static_cast<const double*>(grad_output),
                                   static_cast<const double*>(input1),
                                   static_cast<const double*>(input2),
                                   static_cast<const double*>(target),
                                   static_cast<double*>(out1),
                                   static_cast<double*>(out2), n, margin,
                                   scale);
            break;
        case DType::Float32:
            margin_rank_grad_typed(static_cast<const float*>(grad_output),
                                   static_cast<const float*>(input1),
                                   static_cast<const float*>(input2),
                                   static_cast<const float*>(target),
                                   static_cast<float*>(out1),
                                   static_cast<float*>(out2), n, margin,
                                   scale);
            break;
        default:
            TP_THROW(TypeError, "margin_ranking_loss backward: unsupported dtype");
    }
}

double cosine_rows_impl(const void* x1, const void* x2, const void* target,
                        void* out, int64_t rows, int64_t width, double margin,
                        int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return cosine_rows_typed(static_cast<const double*>(x1),
                                     static_cast<const double*>(x2),
                                     static_cast<const double*>(target),
                                     static_cast<double*>(out), rows, width,
                                     margin);
        case DType::Float32:
            return cosine_rows_typed(static_cast<const float*>(x1),
                                     static_cast<const float*>(x2),
                                     static_cast<const float*>(target),
                                     static_cast<float*>(out), rows, width,
                                     margin);
        default:
            TP_THROW(TypeError, "cosine_embedding_loss: unsupported dtype");
    }
}

void cosine_grad_impl(const void* grad_output, const void* x1, const void* x2,
                      const void* target, void* out1, void* out2, int64_t rows,
                      int64_t width, double margin, double scale, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            cosine_grad_typed(static_cast<const double*>(grad_output),
                              static_cast<const double*>(x1),
                              static_cast<const double*>(x2),
                              static_cast<const double*>(target),
                              static_cast<double*>(out1),
                              static_cast<double*>(out2), rows, width, margin,
                              scale);
            break;
        case DType::Float32:
            cosine_grad_typed(static_cast<const float*>(grad_output),
                              static_cast<const float*>(x1),
                              static_cast<const float*>(x2),
                              static_cast<const float*>(target),
                              static_cast<float*>(out1),
                              static_cast<float*>(out2), rows, width, margin,
                              scale);
            break;
        default:
            TP_THROW(TypeError, "cosine_embedding_loss backward: unsupported dtype");
    }
}

double kl_sum_impl(const void* input, const void* target, int64_t n,
                  int log_target, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return kl_sum_typed(static_cast<const double*>(input),
                                static_cast<const double*>(target), n,
                                log_target != 0);
        case DType::Float32:
            return kl_sum_typed(static_cast<const float*>(input),
                                static_cast<const float*>(target), n,
                                log_target != 0);
        default:
            TP_THROW(TypeError, "kl_div: unsupported dtype");
    }
}

void kl_into_impl(const void* input, const void* target, void* out, int64_t n,
                  int log_target, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            kl_into_typed(static_cast<const double*>(input),
                          static_cast<const double*>(target),
                          static_cast<double*>(out), n, log_target != 0);
            break;
        case DType::Float32:
            kl_into_typed(static_cast<const float*>(input),
                          static_cast<const float*>(target),
                          static_cast<float*>(out), n, log_target != 0);
            break;
        default:
            TP_THROW(TypeError, "kl_div: unsupported dtype");
    }
}

void kl_grad_impl(const void* grad_output, const void* input,
                  const void* target, void* out, int64_t n, int log_target,
                  double scale, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            kl_grad_typed(static_cast<const double*>(grad_output),
                          static_cast<const double*>(input),
                          static_cast<const double*>(target),
                          static_cast<double*>(out), n, log_target != 0,
                          scale);
            break;
        case DType::Float32:
            kl_grad_typed(static_cast<const float*>(grad_output),
                          static_cast<const float*>(input),
                          static_cast<const float*>(target),
                          static_cast<float*>(out), n, log_target != 0,
                          scale);
            break;
        default:
            TP_THROW(TypeError, "kl_div backward: unsupported dtype");
    }
}

double mse_sum_impl(const void* input, const void* target, int64_t n,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return mse_sum_typed(static_cast<const double*>(input),
                                 static_cast<const double*>(target), n);
        case DType::Float32:
            return mse_sum_typed(static_cast<const float*>(input),
                                 static_cast<const float*>(target), n);
        default:
            TP_THROW(TypeError, "mse_loss: unsupported dtype");
    }
}

void mse_into_impl(const void* input, const void* target, void* out, int64_t n,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            mse_into_typed(static_cast<const double*>(input),
                           static_cast<const double*>(target),
                           static_cast<double*>(out), n);
            break;
        case DType::Float32:
            mse_into_typed(static_cast<const float*>(input),
                           static_cast<const float*>(target),
                           static_cast<float*>(out), n);
            break;
        default:
            TP_THROW(TypeError, "mse_loss: unsupported dtype");
    }
}

void mse_grad_impl(const void* grad_output, const void* input,
                   const void* target, void* out, int64_t n, double scale,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            mse_grad_typed(static_cast<const double*>(grad_output),
                           static_cast<const double*>(input),
                           static_cast<const double*>(target),
                           static_cast<double*>(out), n, scale);
            break;
        case DType::Float32:
            mse_grad_typed(static_cast<const float*>(grad_output),
                           static_cast<const float*>(input),
                           static_cast<const float*>(target),
                           static_cast<float*>(out), n, scale);
            break;
        default:
            TP_THROW(TypeError, "mse_loss backward: unsupported dtype");
    }
}

double l1_sum_impl(const void* input, const void* target, int64_t n,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return l1_sum_typed(static_cast<const double*>(input),
                                static_cast<const double*>(target), n);
        case DType::Float32:
            return l1_sum_typed(static_cast<const float*>(input),
                                static_cast<const float*>(target), n);
        default:
            TP_THROW(TypeError, "l1_loss: unsupported dtype");
    }
}

void l1_into_impl(const void* input, const void* target, void* out, int64_t n,
                  int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            l1_into_typed(static_cast<const double*>(input),
                          static_cast<const double*>(target),
                          static_cast<double*>(out), n);
            break;
        case DType::Float32:
            l1_into_typed(static_cast<const float*>(input),
                          static_cast<const float*>(target),
                          static_cast<float*>(out), n);
            break;
        default:
            TP_THROW(TypeError, "l1_loss: unsupported dtype");
    }
}

void l1_grad_impl(const void* grad_output, const void* input,
                  const void* target, void* out, int64_t n, double scale,
                  int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            l1_grad_typed(static_cast<const double*>(grad_output),
                          static_cast<const double*>(input),
                          static_cast<const double*>(target),
                          static_cast<double*>(out), n, scale);
            break;
        case DType::Float32:
            l1_grad_typed(static_cast<const float*>(grad_output),
                          static_cast<const float*>(input),
                          static_cast<const float*>(target),
                          static_cast<float*>(out), n, scale);
            break;
        default:
            TP_THROW(TypeError, "l1_loss backward: unsupported dtype");
    }
}

double multilabel_rows_impl(const void* input, const void* target, void* out,
                            int64_t rows, int64_t width, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return multilabel_rows_typed(static_cast<const double*>(input),
                                         static_cast<const double*>(target),
                                         static_cast<double*>(out), rows, width);
        case DType::Float32:
            return multilabel_rows_typed(static_cast<const float*>(input),
                                         static_cast<const float*>(target),
                                         static_cast<float*>(out), rows, width);
        default:
            TP_THROW(TypeError, "multilabel_soft_margin_loss: unsupported dtype");
    }
}

double bce_logits_rows_impl(const void* input, const void* target,
                            const void* weight, const void* pos_weight,
                            void* out, int64_t n, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return bce_logits_rows_typed(
                static_cast<const double*>(input),
                static_cast<const double*>(target),
                static_cast<const double*>(weight),
                static_cast<const double*>(pos_weight),
                static_cast<double*>(out), n);
        case DType::Float32:
            return bce_logits_rows_typed(
                static_cast<const float*>(input),
                static_cast<const float*>(target),
                static_cast<const float*>(weight),
                static_cast<const float*>(pos_weight),
                static_cast<float*>(out), n);
        default:
            TP_THROW(TypeError,
                     "binary_cross_entropy_with_logits: unsupported dtype");
    }
}

#define TP_STR2(x) #x
#define TP_STR(x) TP_STR2(x)
double triplet_rows_impl(const void* anchor, const void* positive,
                         const void* negative, void* out, int64_t rows,
                         int64_t width, double margin, double p, double eps,
                         bool swap, int dtype) {
    if (std::getenv("TP_TRIPLET_DEBUG")) {
        std::fprintf(stderr, "[triplet] cap=%s dtype=%d rows=%lld width=%lld threads=%lld\n",
                     TP_STR(CPU_CAPABILITY), dtype, (long long)rows, (long long)width,
                     (long long)parallel::get_num_threads());
    }    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return triplet_rows_typed(static_cast<const double*>(anchor),
                                      static_cast<const double*>(positive),
                                      static_cast<const double*>(negative),
                                      static_cast<double*>(out), rows, width,
                                      margin, p, eps, swap);
        case DType::Float32:
            return triplet_rows_typed(static_cast<const float*>(anchor),
                                      static_cast<const float*>(positive),
                                      static_cast<const float*>(negative),
                                      static_cast<float*>(out), rows, width,
                                      margin, p, eps, swap);
        default:
            TP_THROW(TypeError, "triplet_margin_loss: unsupported dtype");
    }
}

std::tuple<double, double> nll_rows_impl(const void* input, const void* target,
                                         const void* weight, void* out,
                                         int64_t rows, int64_t cols,
                                         int64_t ignore_index, int has_weight,
                                         int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return nll_rows_typed(static_cast<const double*>(input),
                                  static_cast<const int64_t*>(target),
                                  static_cast<const double*>(weight),
                                  static_cast<double*>(out), rows, cols,
                                  ignore_index, has_weight != 0);
        case DType::Float32:
            return nll_rows_typed(static_cast<const float*>(input),
                                  static_cast<const int64_t*>(target),
                                  static_cast<const float*>(weight),
                                  static_cast<float*>(out), rows, cols,
                                  ignore_index, has_weight != 0);
        default:
            TP_THROW(TypeError, "nll_loss: unsupported dtype");
    }
}

double nll_weight_total_impl(const void* target, const void* weight,
                             int64_t rows, int64_t ignore_index,
                             int has_weight, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return nll_weight_total_typed(static_cast<const int64_t*>(target),
                                          static_cast<const double*>(weight),
                                          rows, ignore_index,
                                          has_weight != 0);
        case DType::Float32:
            return nll_weight_total_typed(static_cast<const int64_t*>(target),
                                          static_cast<const float*>(weight),
                                          rows, ignore_index,
                                          has_weight != 0);
        default:
            TP_THROW(TypeError, "nll_loss: unsupported dtype");
    }
}

void nll_grad_impl(const void* grad_output, const void* target,
                   const void* weight, void* grad_input, int64_t rows,
                   int64_t cols, int64_t ignore_index, int reduction,
                   double total_weight, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            nll_grad_typed(static_cast<const double*>(grad_output),
                           static_cast<const int64_t*>(target),
                           static_cast<const double*>(weight),
                           static_cast<double*>(grad_input), rows, cols,
                           ignore_index, reduction, total_weight);
            break;
        case DType::Float32:
            nll_grad_typed(static_cast<const float*>(grad_output),
                           static_cast<const int64_t*>(target),
                           static_cast<const float*>(weight),
                           static_cast<float*>(grad_input), rows, cols,
                           ignore_index, reduction, total_weight);
            break;
        default:
            TP_THROW(TypeError, "nll_loss backward: unsupported dtype");
    }
}

double nll2d_rows_impl(const void* input, const void* target,
                       const void* weight, void* out, int64_t batch,
                       int64_t classes, int64_t hw, int64_t ignore_index,
                       int has_weight, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return nll2d_rows_typed(static_cast<const double*>(input),
                                    static_cast<const int64_t*>(target),
                                    static_cast<const double*>(weight),
                                    static_cast<double*>(out), batch, classes,
                                    hw, ignore_index, has_weight != 0);
        case DType::Float32:
            return nll2d_rows_typed(static_cast<const float*>(input),
                                    static_cast<const int64_t*>(target),
                                    static_cast<const float*>(weight),
                                    static_cast<float*>(out), batch, classes,
                                    hw, ignore_index, has_weight != 0);
        default:
            TP_THROW(TypeError, "nll_loss2d: unsupported dtype");
    }
}

void nll2d_grad_impl(const void* grad_output, const void* target,
                     const void* weight, void* grad_input, int64_t batch,
                     int64_t classes, int64_t hw, int64_t ignore_index,
                     int reduction, double total_weight, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            nll2d_grad_typed(static_cast<const double*>(grad_output),
                             static_cast<const int64_t*>(target),
                             static_cast<const double*>(weight),
                             static_cast<double*>(grad_input), batch, classes,
                             hw, ignore_index, reduction, total_weight);
            break;
        case DType::Float32:
            nll2d_grad_typed(static_cast<const float*>(grad_output),
                             static_cast<const int64_t*>(target),
                             static_cast<const float*>(weight),
                             static_cast<float*>(grad_input), batch, classes,
                             hw, ignore_index, reduction, total_weight);
            break;
        default:
            TP_THROW(TypeError, "nll_loss2d backward: unsupported dtype");
    }
}

double multi_margin_rows_impl(const void* input, const void* target,
                              const void* weight, void* out, int64_t rows,
                              int64_t cols, int p, double margin, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return multi_margin_rows_typed(
                static_cast<const double*>(input),
                static_cast<const int64_t*>(target),
                static_cast<const double*>(weight),
                static_cast<double*>(out), rows, cols, p, margin);
        case DType::Float32:
            return multi_margin_rows_typed(
                static_cast<const float*>(input),
                static_cast<const int64_t*>(target),
                static_cast<const float*>(weight),
                static_cast<float*>(out), rows, cols, p, margin);
        default:
            TP_THROW(TypeError, "multi_margin_loss: unsupported dtype");
    }
}

void multi_margin_grad_impl(const void* grad_output, const void* input,
                            const void* target, const void* weight,
                            void* grad_input, int64_t rows, int64_t cols,
                            int p, double margin, double scale, int per_row,
                            int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            multi_margin_grad_typed(
                static_cast<const double*>(grad_output),
                static_cast<const double*>(input),
                static_cast<const int64_t*>(target),
                static_cast<const double*>(weight),
                static_cast<double*>(grad_input), rows, cols, p, margin,
                scale, per_row != 0);
            break;
        case DType::Float32:
            multi_margin_grad_typed(
                static_cast<const float*>(grad_output),
                static_cast<const float*>(input),
                static_cast<const int64_t*>(target),
                static_cast<const float*>(weight),
                static_cast<float*>(grad_input), rows, cols, p, margin,
                scale, per_row != 0);
            break;
        default:
            TP_THROW(TypeError, "multi_margin_loss backward: unsupported dtype");
    }
}

double multilabel_margin_rows_impl(const void* input, const void* target,
                                   void* out, void* is_target, int64_t rows,
                                   int64_t cols, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return multilabel_margin_rows_typed(
                static_cast<const double*>(input),
                static_cast<const int64_t*>(target),
                static_cast<double*>(out),
                static_cast<double*>(is_target), rows, cols);
        case DType::Float32:
            return multilabel_margin_rows_typed(
                static_cast<const float*>(input),
                static_cast<const int64_t*>(target),
                static_cast<float*>(out),
                static_cast<float*>(is_target), rows, cols);
        default:
            TP_THROW(TypeError,
                     "multilabel_margin_loss forward: unsupported dtype");
    }
}

void multilabel_margin_grad_impl(const void* grad_output, const void* input,
                                 const void* target, const void* is_target,
                                 void* grad_input, int64_t rows, int64_t cols,
                                 double scale, int per_row, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            multilabel_margin_grad_typed(
                static_cast<const double*>(grad_output),
                static_cast<const double*>(input),
                static_cast<const int64_t*>(target),
                static_cast<const double*>(is_target),
                static_cast<double*>(grad_input), rows, cols, scale,
                per_row != 0);
            break;
        case DType::Float32:
            multilabel_margin_grad_typed(
                static_cast<const float*>(grad_output),
                static_cast<const float*>(input),
                static_cast<const int64_t*>(target),
                static_cast<const float*>(is_target),
                static_cast<float*>(grad_input), rows, cols, scale,
                per_row != 0);
            break;
        default:
            TP_THROW(TypeError,
                     "multilabel_margin_loss backward: unsupported dtype");
    }
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(bce_sum_stub, &bce_sum_impl);
REGISTER_DISPATCH(bce_into_stub, &bce_into_impl);
REGISTER_DISPATCH(kl_sum_stub, &kl_sum_impl);
REGISTER_DISPATCH(kl_into_stub, &kl_into_impl);
REGISTER_DISPATCH(kl_grad_stub, &kl_grad_impl);
REGISTER_DISPATCH(mse_sum_stub, &mse_sum_impl);
REGISTER_DISPATCH(mse_into_stub, &mse_into_impl);
REGISTER_DISPATCH(mse_grad_stub, &mse_grad_impl);
REGISTER_DISPATCH(l1_sum_stub, &l1_sum_impl);
REGISTER_DISPATCH(l1_into_stub, &l1_into_impl);
REGISTER_DISPATCH(l1_grad_stub, &l1_grad_impl);
REGISTER_DISPATCH(huber_into_stub, &huber_into_impl);
REGISTER_DISPATCH(huber_sum_stub, &huber_sum_impl);
REGISTER_DISPATCH(smooth_l1_sum_stub, &smooth_l1_sum_impl);
REGISTER_DISPATCH(smooth_l1_into_stub, &smooth_l1_into_impl);
REGISTER_DISPATCH(smooth_l1_grad_stub, &smooth_l1_grad_impl);
REGISTER_DISPATCH(softmargin_sum_stub, &softmargin_sum_impl);
REGISTER_DISPATCH(softmargin_into_stub, &softmargin_into_impl);
REGISTER_DISPATCH(softmargin_grad_stub, &softmargin_grad_impl);
REGISTER_DISPATCH(poisson_sum_stub, &poisson_sum_impl);
REGISTER_DISPATCH(poisson_into_stub, &poisson_into_impl);
REGISTER_DISPATCH(poisson_grad_stub, &poisson_grad_impl);
REGISTER_DISPATCH(hinge_sum_stub, &hinge_sum_impl);
REGISTER_DISPATCH(hinge_into_stub, &hinge_into_impl);
REGISTER_DISPATCH(hinge_grad_stub, &hinge_grad_impl);
REGISTER_DISPATCH(margin_rank_sum_stub, &margin_rank_sum_impl);
REGISTER_DISPATCH(margin_rank_into_stub, &margin_rank_into_impl);
REGISTER_DISPATCH(margin_rank_grad_stub, &margin_rank_grad_impl);
REGISTER_DISPATCH(cosine_rows_stub, &cosine_rows_impl);
REGISTER_DISPATCH(cosine_grad_stub, &cosine_grad_impl);
REGISTER_DISPATCH(bce_logits_rows_stub, &bce_logits_rows_impl);
REGISTER_DISPATCH(multilabel_rows_stub, &multilabel_rows_impl);
REGISTER_DISPATCH(triplet_rows_stub, &triplet_rows_impl);
REGISTER_DISPATCH(nll_rows_stub, &nll_rows_impl);
REGISTER_DISPATCH(nll_weight_total_stub, &nll_weight_total_impl);
REGISTER_DISPATCH(nll_grad_stub, &nll_grad_impl);
REGISTER_DISPATCH(nll2d_rows_stub, &nll2d_rows_impl);
REGISTER_DISPATCH(nll2d_grad_stub, &nll2d_grad_impl);
REGISTER_DISPATCH(multi_margin_rows_stub, &multi_margin_rows_impl);
REGISTER_DISPATCH(multi_margin_grad_stub, &multi_margin_grad_impl);
REGISTER_DISPATCH(multilabel_margin_rows_stub, &multilabel_margin_rows_impl);
REGISTER_DISPATCH(multilabel_margin_grad_stub, &multilabel_margin_grad_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(bce_sum_stub, &bce_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(bce_into_stub, &bce_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(kl_sum_stub, &kl_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(kl_into_stub, &kl_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(kl_grad_stub, &kl_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(mse_sum_stub, &mse_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(mse_into_stub, &mse_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(mse_grad_stub, &mse_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(l1_sum_stub, &l1_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(l1_into_stub, &l1_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(l1_grad_stub, &l1_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(huber_into_stub, &huber_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(huber_sum_stub, &huber_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(smooth_l1_sum_stub, &smooth_l1_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(smooth_l1_into_stub, &smooth_l1_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(smooth_l1_grad_stub, &smooth_l1_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmargin_sum_stub, &softmargin_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmargin_into_stub, &softmargin_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmargin_grad_stub, &softmargin_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(poisson_sum_stub, &poisson_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(poisson_into_stub, &poisson_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(poisson_grad_stub, &poisson_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(hinge_sum_stub, &hinge_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(hinge_into_stub, &hinge_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(hinge_grad_stub, &hinge_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(margin_rank_sum_stub, &margin_rank_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(margin_rank_into_stub, &margin_rank_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(margin_rank_grad_stub, &margin_rank_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(cosine_rows_stub, &cosine_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(cosine_grad_stub, &cosine_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(bce_logits_rows_stub, &bce_logits_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(multilabel_rows_stub, &multilabel_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(triplet_rows_stub, &triplet_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(nll_rows_stub, &nll_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(nll_weight_total_stub, &nll_weight_total_impl);
ALSO_REGISTER_AVX512_DISPATCH(nll_grad_stub, &nll_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(nll2d_rows_stub, &nll2d_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(nll2d_grad_stub, &nll2d_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(multi_margin_rows_stub, &multi_margin_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(multi_margin_grad_stub, &multi_margin_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(multilabel_margin_rows_stub, &multilabel_margin_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(multilabel_margin_grad_stub, &multilabel_margin_grad_impl);
#endif

} // namespace cpu
} // namespace tensorplay
