#pragma once

#include "Tensor.h"

#include <tuple>

namespace tensorplay {
namespace cpu {

// Native forward-mode AD (JVP) kernels.  Each op takes the primal input(s)
// followed by the matching tangent(s) and returns {primal_out, tangent_out},
// evaluating both in one pass.  Supported dtypes: Float32/Float64; binary
// ops require matching operand shapes (broadcast tangents must be expanded
// by the caller).
//
//   forward_neg(a, da)                -> (-a, -da)
//   forward_exp(a, da)                -> (e^a, e^a * da)
//   forward_log(a, da)                -> (ln a, da / a)
//   forward_sin(a, da)                -> (sin a, cos a * da)
//   forward_cos(a, da)                -> (cos a, -sin a * da)
//   forward_sqrt(a, da)               -> (sqrt a, da / (2 sqrt a))
//   forward_tanh(a, da)               -> (tanh a, (1 - tanh^2 a) * da)
//   forward_sigmoid(a, da)            -> (s, s (1-s) * da), s = sigmoid(a)
//   forward_relu(a, da)               -> (relu a, a > 0 ? da : 0)
//   forward_add/sub/mul/div(a,da,b,db)-> elementwise linear combination
//   forward_pow(a, da, b, db)         -> (a^b, r (db ln a + b da / a))
//   forward_mm(a, da, b, db)          -> (a@b, da@b + a@db)

std::tuple<Tensor, Tensor> forward_neg_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_exp_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_log_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_sin_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_cos_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_sqrt_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_tanh_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_sigmoid_cpu(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_relu_cpu(const Tensor& a, const Tensor& da);

std::tuple<Tensor, Tensor> forward_add_cpu(const Tensor& a, const Tensor& da,
                                           const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_sub_cpu(const Tensor& a, const Tensor& da,
                                           const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_mul_cpu(const Tensor& a, const Tensor& da,
                                           const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_div_cpu(const Tensor& a, const Tensor& da,
                                           const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_pow_cpu(const Tensor& a, const Tensor& da,
                                           const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_mm_cpu(const Tensor& a, const Tensor& da,
                                          const Tensor& b, const Tensor& db);

} // namespace cpu

#ifdef USE_CUDA
namespace cuda {

std::tuple<Tensor, Tensor> forward_neg_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_exp_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_log_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_sin_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_cos_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_sqrt_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_tanh_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_sigmoid_cuda(const Tensor& a, const Tensor& da);
std::tuple<Tensor, Tensor> forward_relu_cuda(const Tensor& a, const Tensor& da);

std::tuple<Tensor, Tensor> forward_add_cuda(const Tensor& a, const Tensor& da,
                                            const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_sub_cuda(const Tensor& a, const Tensor& da,
                                            const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_mul_cuda(const Tensor& a, const Tensor& da,
                                            const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_div_cuda(const Tensor& a, const Tensor& da,
                                            const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_pow_cuda(const Tensor& a, const Tensor& da,
                                            const Tensor& b, const Tensor& db);
std::tuple<Tensor, Tensor> forward_mm_cuda(const Tensor& a, const Tensor& da,
                                           const Tensor& b, const Tensor& db);

} // namespace cuda
#endif
} // namespace tensorplay
